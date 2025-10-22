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
                input_shape = self.config.params_image_size,
                include_top = self.config.params_include_top,
                weights = self.config.params_weights
            )
        elif self.config.model_name == "VGG16":
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape = self.config.params_image_size,
                include_top = self.config.params_include_top,
                weights = self.config.params_weights
            )
        self.model.save(self.config.base_model_path)
        log.info(f"base model is saved at: {self.config.base_model_path}")
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        #flatten_in = tf.keras.layers.Flatten()()
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(model.output)
        kernel_reg = tf.keras.regularizers.l2(l2=0.01)
        
        mlp_units = [512, 256]
        use_batchnorm = True
        activation_hidden: str = "relu"
        final_activation: str = "softmax"
        dropout_rate: float = 0.3
        
        for i, units in enumerate(mlp_units, start=1):
            x = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=kernel_reg, name=f"dense_{i}")(x)
            if use_batchnorm:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = tf.keras.layers.Activation(activation_hidden, name=f"act_{i}")(x)
            if dropout_rate and dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

        outputs = tf.keras.layers.Dense(classes, activation=final_activation, name="predictions")(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=outputs, name="optic_guardian_classifier")

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_updated_base_model(
            updated_base_model_path=self.config.updated_base_model_path,
            model=self.full_model
        )    
        
    @staticmethod
    def save_updated_base_model(updated_base_model_path: Path, model: tf.keras.Model):
        model.save(updated_base_model_path)
        log.info(f"updated base model is saved at: {updated_base_model_path}")
