# src/RetinoNet/components/prepare_base_model.py
from pathlib import Path
import tensorflow as tf # type: ignore

from RetinoNet.entity.config_entity import PrepareBaseModelConfig
from RetinoNet import logger

log = logger.getChild(__name__)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config

    def get_base_model(self):
        if self.config.model_name == "MobileNetV2":
            self.model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights
            )
        elif self.config.model_name == "VGG16":
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights
            )
        else:
            raise ValueError(f"Model name not supported: {self.config.model_name}")

        self.model.save(self.config.base_model_path)
        log.info(f"base model is saved at: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, 
                            kernel_l2=1e-4, mlp_units=(128,), dropout_rate=0.5,
                            use_augmentation_layer=False):
        """
        Build the full model with:
         - smaller L2 (kernel_l2),
         - smaller MLP (mlp_units),
         - stronger dropout (dropout_rate),
         - optional augmentation layer inside the model (use_augmentation_layer).
        """

        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # freeze all but last `freeze_till` layers
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Build new input so we can optionally insert augmentation layer
        inp = tf.keras.layers.Input(shape=tuple(model.input_shape[1:]), name="input_1")

        x = inp
        if use_augmentation_layer:
            # mild augmentation as regularizer inside model (applied only during training)
            data_aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.08),
                tf.keras.layers.RandomTranslation(0.05, 0.05)
            ], name="data_augmentation")
            x = data_aug(x)

        x = model(x, training=False)

        # pooling
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

        # smaller L2 regularizer
        kernel_reg = tf.keras.regularizers.l2(l2=kernel_l2)

        use_batchnorm = True
        activation_hidden: str = "relu"
        final_activation: str = "softmax"

        for i, units in enumerate(mlp_units, start=1):
            x = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=kernel_reg, name=f"dense_{i}")(x)
            if use_batchnorm:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = tf.keras.layers.Activation(activation_hidden, name=f"act_{i}")(x)
            if dropout_rate and dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

        outputs = tf.keras.layers.Dense(classes, activation=final_activation, name="predictions")(x)

        full_model = tf.keras.models.Model(inputs=inp, outputs=outputs, name="optic_guardian_classifier")

        # compile with label smoothing to reduce overconfidence
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        kernel_l2 = float(getattr(self.config, "kernel_l2", 1e-4))
        mlp_units = tuple(getattr(self.config, "mlp_units", [128]))
        dropout_rate = float(getattr(self.config, "dropout_rate", 0.5))
        use_aug_layer = bool(getattr(self.config, "use_augmentation_in_model", False))

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            kernel_l2=kernel_l2,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate,
            use_augmentation_layer=use_aug_layer
        )

        self.save_updated_base_model(
            updated_base_model_path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_updated_base_model(updated_base_model_path: Path, model: tf.keras.Model):
        model.save(updated_base_model_path)
        log.info(f"updated base model is saved at: {updated_base_model_path}")
