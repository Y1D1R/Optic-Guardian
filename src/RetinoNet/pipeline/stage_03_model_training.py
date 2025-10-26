# src/RetinoNet/pipeline/stage_03_model_training.py
import os
from pathlib import Path
import tensorflow as tf  # type: ignore
from RetinoNet.config.configuration import ConfigurationManager
from RetinoNet.components.model_training import ModelTraining
from RetinoNet import get_logger

STAGE_NAME = "03 - Model Training stage"
log = get_logger(__name__)


class ModelTrainingPipeline:
    def __init__(self) -> None:
        # Optionnel : éviter que TF alloue toute la mémoire GPU au démarrage
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    def main(self) -> None:
        # Charger la configuration
        config = ConfigurationManager()
        model_training_config = config.model_training_config()

        # create artifacts dir if missing (used by ModelTraining callbacks)
        artifacts_dir = Path(getattr(model_training_config, "artifacts_dir", "artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser l'entraîneur
        model_trainer = ModelTraining(config=model_training_config)

        # Charger / compiler le modèle de base et préparer les générateurs
        model_trainer.get_base_model()
        model_trainer.train_valid_generator()

        # Lancer l'entraînement (retourne l'history dict)
        history = model_trainer.train()

        # Après l'entraînement, s'il existe un checkpoint "best", on le charge et on l'enregistre
        # sous le chemin final attendu (trained_model_path). Cela évite d'utiliser un modèle non-optimal.
        trained_model_path = Path(model_training_config.trained_model_path)
        best_checkpoint = trained_model_path.with_suffix(".best.h5")
        try:
            if best_checkpoint.exists():
                log.info("Found best checkpoint at %s — loading and saving to final path %s",
                         str(best_checkpoint), str(trained_model_path))
                best_model = tf.keras.models.load_model(str(best_checkpoint), compile=False)
                # Compiler si besoin pour l'export (optionnel, adapte selon ton workflow)
                best_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=model_training_config.params_learning_rate),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"]
                )
                # Ensure parent dirs
                trained_model_path.parent.mkdir(parents=True, exist_ok=True)
                best_model.save(str(trained_model_path))
                log.info("Best model saved to %s", str(trained_model_path))
            else:
                # Si pas de checkpoint best (rare), sauvegarder le modèle courant (train() l'a normalement déjà fait)
                if getattr(model_trainer, "model", None) is not None:
                    log.info("No best checkpoint found; saving current model to %s", str(trained_model_path))
                    model_trainer.save_model(path=trained_model_path, model=model_trainer.model)
        except Exception:
            log.exception("Failed to save best model to final path.")

        # Log synthèse : dernière métrique disponible (si history présent)
        try:
            if history:
                # history est un dict: metric -> list(epoch_values)
                # chercher val_loss / val_accuracy ou fallback sur loss/accuracy
                def last(metric_name):
                    vals = history.get(metric_name)
                    return vals[-1] if vals else None

                vl = last("val_loss") or last("loss")
                va = last("val_accuracy") or last("accuracy")
                log.info("Training finished. Last metrics -> val_loss: %s | val_accuracy: %s", vl, va)
        except Exception:
            log.exception("Could not extract training metrics from history.")

        log.info("Model training pipeline finished.")


if __name__ == "__main__":
    try:
        log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        ModelTrainingPipeline().main()
        log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception:
        log.exception("stage failed")
        raise
