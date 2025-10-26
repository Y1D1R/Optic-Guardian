# src/RetinoNet/components/model_training.py
from pathlib import Path
import math
import json
import os
from collections import Counter
from typing import Optional, Dict, Any
import numpy as np



import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks, optimizers, losses, metrics  # type: ignore

from RetinoNet.entity.config_entity import ModelTrainingConfig
from RetinoNet import logger

log = logger.getChild(__name__)


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = self._select_device()
        log.info(f"Using device: {self.device}")
        self._maybe_enable_mixed_precision()

    def _select_device(self) -> str:
        """
        Pick GPU (CUDA) then DML (DirectML) else CPU. 
        Enable memory growth on CUDA GPUs.
        """
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        log.debug("set_memory_growth failed for GPU device", exc_info=True)
                return "/GPU:0"
            dml = tf.config.list_physical_devices("DML")
            if dml:
                return "/device:DML:0"
        except Exception:
            log.exception("Device detection failed, falling back to CPU.")
        return "/CPU:0"

    def _maybe_enable_mixed_precision(self) -> None:
        """Enable mixed precision if requested in config and supported by TF/device."""
        try:
            if getattr(self.config, "enable_mixed_precision", False):
                # Only enable when GPU present
                if tf.config.list_physical_devices("GPU"):
                    from tensorflow.keras.mixed_precision import experimental as mixed_precision  # type: ignore
                    policy = "mixed_float16"
                    mixed_precision.set_policy(policy)
                    log.info("Mixed precision enabled (policy=%s).", policy)
                else:
                    log.info("Mixed precision requested but no GPU found; skipping.")
        except Exception:
            log.exception("Failed to enable mixed precision; continuing without it.")

    def get_base_model(self):
        """Load and compile the model (keeps compile False on load, then compiles)."""
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(
                self.config.updated_base_model_path,
                compile=False
            )

        # If mixed precision is active, ensure final output dtype is float32
        try:
            # If model's output dtype is float16 (due to mixed precision), cast outputs to float32
            if getattr(self.model, "dtype", None) == "float16":
                log.info("Model dtype is float16 after load (mixed precision).")
        except Exception:
            pass

        # Configure optimizer with gradient clipping
        adam_kwargs = dict(learning_rate=self.config.params_learning_rate)
        if getattr(self.config, "gradient_clip_norm", None):
            adam_kwargs["clipnorm"] = float(self.config.gradient_clip_norm)
        if getattr(self.config, "gradient_clip_value", None):
            adam_kwargs["clipvalue"] = float(self.config.gradient_clip_value)

        optimizer = optimizers.Adam(**adam_kwargs)

        # Loss & metrics
        loss_fn = losses.CategoricalCrossentropy()
        metric_list = [metrics.CategoricalAccuracy(name="accuracy")]

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_list,
        )
        log.info("Model compiled with optimizer=%s, loss=%s", type(optimizer).__name__, type(loss_fn).__name__)

    def train_valid_generator(self):
        """Prepare ImageDataGenerators for training and validation."""
        datagenerator_kwargs = dict(rescale=1.0 / 255.0, validation_split=0.20)
        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        self._log_split_stats()

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save a model safely, creating parent dirs if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        log.info("Model saved to %s", str(path))

    def _compute_class_weights(self) -> Optional[Dict[int, float]]:
        """Compute class weights from train_generator to help with imbalanced classes."""
        try:
            counts = Counter(self.train_generator.classes)
            n_classes = len(self.train_generator.class_indices)
            total = float(sum(counts.values()))
            # class_weight[class_index] = total / (n_classes * count)
            class_weight = {int(cls): total / (n_classes * count) for cls, count in counts.items()}
            log.info("Computed class weights: %s", class_weight)
            return class_weight
        except Exception:
            log.exception("Failed to compute class weights.")
            return None

    def _build_callbacks(self) -> list:
        """Construct a list of Keras callbacks for robust training."""
        cb = []

        # Ensure artifact dir exists
        artifacts_dir = Path(self.config.artifacts_dir) if getattr(self.config, "artifacts_dir", None) else Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ModelCheckpoint: save best model by monitored metric (val_loss by default)
        monitor_metric = getattr(self.config, "monitor_metric", "val_loss")
        mode = "min" if "loss" in monitor_metric else "max"
        checkpoint_path = Path(self.config.trained_model_path).with_suffix(".best.h5")
        cb.append(callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode=mode
        ))

        # EarlyStopping
        if getattr(self.config, "use_early_stopping", True):
            patience = int(getattr(self.config, "early_stopping_patience", 10))
            cb.append(callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
                verbose=1,
                restore_best_weights=True,
                mode=mode
            ))

        # ReduceLROnPlateau
        if getattr(self.config, "use_reduce_lr_on_plateau", True):
            cb.append(callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=float(getattr(self.config, "reduce_lr_factor", 0.5)),
                patience=int(getattr(self.config, "reduce_lr_patience", 5)),
                min_lr=float(getattr(self.config, "min_lr", 1e-7)),
                verbose=1,
                mode=mode
            ))

        # CSV logger (save history)
        csv_log_path = artifacts_dir / "training_history.csv"
        cb.append(callbacks.CSVLogger(str(csv_log_path), append=True))

        # TensorBoard (optional)
        if getattr(self.config, "use_tensorboard", False):
            tb_logdir = artifacts_dir / "tensorboard"
            tb_logdir.mkdir(parents=True, exist_ok=True)
            cb.append(callbacks.TensorBoard(log_dir=str(tb_logdir), histogram_freq=1))

        return cb

    def train(self) -> Dict[str, Any]:
        """
        Main training orchestration.
        - support resume_from_checkpoint: if True and best checkpoint exists, load it and continue training
        - returns history dict (merged with previous history if resuming)
        """
        # steps
        steps_per_epoch = max(1, math.ceil(self.train_generator.samples / float(self.train_generator.batch_size)))
        validation_steps = max(1, math.ceil(self.valid_generator.samples / float(self.valid_generator.batch_size)))

        # artifacts dir robust
        artifacts_dir = Path(self.config.artifacts_dir) if getattr(self.config, "artifacts_dir", None) else Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # class weight
        class_weight = None
        if getattr(self.config, "use_class_weight", True):
            class_weight = self._compute_class_weights()

        cb_list = self._build_callbacks()

        # checkpoint path (where ModelCheckpoint wrote best model)
        best_checkpoint = Path(self.config.trained_model_path).with_suffix(".best.h5")

        # determine whether to resume from checkpoint
        resume_flag = self.config.resume_from_checkpoint
        initial_epoch = 0
        prev_history: Dict[str, Any] = {}

        # if resuming and checkpoint exists: load model and compute initial_epoch from previous history
        if resume_flag and best_checkpoint.exists():
            try:
                log.info("Resume requested and checkpoint found at %s. Loading checkpoint to continue training.", str(best_checkpoint))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # load full model (weights + architecture)
                # load with compile=False and recompile to ensure same optimizer/loss/metrics
                self.model = tf.keras.models.load_model(str(best_checkpoint), compile=False)

                # Recompile with same optimizer settings as in get_base_model()
                adam_kwargs = dict(learning_rate=self.config.params_learning_rate)
                if getattr(self.config, "gradient_clip_norm", None):
                    adam_kwargs["clipnorm"] = float(self.config.gradient_clip_norm)
                if getattr(self.config, "gradient_clip_value", None):
                    adam_kwargs["clipvalue"] = float(self.config.gradient_clip_value)
                optimizer = tf.keras.optimizers.Adam(**adam_kwargs)
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
                metric_list = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]

                self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric_list)
                log.info("Checkpoint model loaded and compiled.")

                # try to read previous history to determine where to resume
                hist_path = artifacts_dir / "history.json"
                if hist_path.exists():
                    try:
                        with open(hist_path, "r") as f:
                            prev_history = json.load(f)
                        # determine initial_epoch from length of an available list
                        initial_epoch = len(prev_history.get("loss", []) or prev_history.get("accuracy", []) or [])
                        log.info("Previous history loaded, will resume from initial_epoch=%d", initial_epoch)
                    except Exception:
                        log.warning("Could not read previous history.json; starting from initial_epoch=0")
                        initial_epoch = 0
                else:
                    # fallback: check CSV logger (training_history.csv)
                    csv_path = artifacts_dir / "training_history.csv"
                    if csv_path.exists():
                        try:
                            # count lines minus header
                            with open(csv_path, "r") as f:
                                n_lines = sum(1 for _ in f)
                            initial_epoch = max(0, n_lines - 1)
                            log.info("Found CSV history, resume initial_epoch=%d", initial_epoch)
                        except Exception:
                            initial_epoch = 0

            except Exception:
                log.exception("Failed to load checkpoint for resume; will train from current model instance.")
                initial_epoch = 0
        else:
            if resume_flag:
                log.warning("Resume requested but no checkpoint found at %s. Training will start from current model (base).", str(best_checkpoint))
            else:
                log.info("Resume flag is False; training will start from current model (probably base model).")

        total_epochs = int(self.config.params_epochs)

        # If the previous run already reached or exceeded total_epochs ==> nothing to do
        if initial_epoch >= total_epochs:
            log.info("Initial epoch (%d) >= total_epochs (%d). Nothing to train.", initial_epoch, total_epochs)
            # still ensure model saved to final path
            try:
                self.save_model(path=Path(self.config.trained_model_path), model=self.model)
            except Exception:
                log.exception("Failed to save model at end of train() when skipping training.")
            return prev_history or {}

        history = {}
        try:
            with tf.device(self.device):
                # Fit with initial_epoch so that training continues where left off
                history_obj = self.model.fit(
                    self.train_generator,
                    epochs=total_epochs,
                    initial_epoch=initial_epoch,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=self.valid_generator,
                    validation_steps=validation_steps,
                    callbacks=cb_list,
                    class_weight=class_weight,
                    verbose=1
                )
                new_history = history_obj.history

                # Save final model
                self.save_model(path=Path(self.config.trained_model_path), model=self.model)

                # Merge previous history + new_history if resuming, else new_history only
                if prev_history:
                    merged = {}
                    for k in set(list(prev_history.keys()) + list(new_history.keys())):
                        prev_list = prev_history.get(k, [])
                        new_list = new_history.get(k, [])
                        merged[k] = prev_list + new_list
                    history = merged
                else:
                    history = new_history

                # JSON
                hist_path = artifacts_dir / "history.json"
                try:
                    serializable_history = self._make_serializable(history)
                    with open(hist_path, "w") as f:
                        json.dump(serializable_history, f, indent=2)
                    log.info("Training history saved to %s", str(hist_path))
                except Exception:
                    log.exception("Failed to write training history to %s", str(hist_path))

        except tf.errors.ResourceExhaustedError as e:
            log.exception("ResourceExhaustedError during training (maybe GPU OOM).")
            raise e
        except Exception:
            log.exception("Unexpected exception during training.")
            raise

        return history or {}


    def _log_split_stats(self) -> None:
        """
        Log number of images per class for train and validation splits.
        """
        idx_to_label = {v: k for k, v in self.train_generator.class_indices.items()}

        if getattr(self.valid_generator, "class_indices", {}) != self.train_generator.class_indices:
            log.warning("class_indices differ between train and val; using train mapping for logging.")

        train_counts = Counter(self.train_generator.classes)
        val_counts = Counter(self.valid_generator.classes)

        # total
        n_train = self.train_generator.samples
        n_val = self.valid_generator.samples
        n_classes = len(idx_to_label)

        log.info(f"dataset summary -> train: {n_train} | val: {n_val} | classes: {n_classes}")

        for cls_idx, cls_name in sorted(idx_to_label.items(), key=lambda x: x[1].lower()):
            t = train_counts.get(cls_idx, 0)
            v = val_counts.get(cls_idx, 0)
            log.info(f"class '{cls_name}': train={t} | val={v}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert numpy / tensorflow scalar/arrays to Python native types
        so that json.dump can serialize them.
        """
        # handle dict
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}

        # handle list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]

        # numpy scalar
        if isinstance(obj, np.generic):
            return obj.item()

        # numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # tensorflow tensor (best effort)
        try:
            import tensorflow as tf  # type: ignore
            if isinstance(obj, tf.Tensor):
                arr = obj.numpy()
                if isinstance(arr, np.ndarray):
                    return arr.tolist()
                else:
                    return self._make_serializable(arr)
        except Exception:
            pass

        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj

        # fallback : convert to string
        return str(obj)
