# src/RetinoNet/config/configuration.py
from pathlib import Path
import os

from RetinoNet.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from RetinoNet.utils.common import read_yaml, create_directories
from RetinoNet.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, ModelTrainingConfig
from RetinoNet import logger
from dotenv import load_dotenv

log = logger.getChild(__name__)
load_dotenv() 

class ConfigurationManager:
    """
    Load YAML configs and build typed config objects for each pipeline stage.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH  ) -> None:
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # ensure artifacts root
        artifacts_root = Path(self.config.artifacts_root)
        create_directories([artifacts_root])
        log.info(f"artifacts root ready: {artifacts_root}")

    # ---------------- Data Ingestion ----------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion

        root_dir = Path(cfg.root_dir)
        local_data_file = Path(cfg.local_data_file)
        unzip_dir = Path(cfg.unzip_dir)

        data = DataIngestionConfig(
            root_dir=root_dir,
            source_URL=str(cfg.source_URL),
            local_data_file=local_data_file,
            unzip_dir=unzip_dir,
        )
        log.info("data_ingestion config built")
        return data
    
    # ---------------- Prepare Base Model ----------------
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([Path(config.root_dir)])

        base_model = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_classes = self.params.CLASSES,
            model_name = self.params.MODEL_NAME
        )
        log.info("Base model config built")
        return base_model

    # ---------------- Model Training ----------------
    def model_training_config(self) -> ModelTrainingConfig:
        model_training = self.config.model_training
        prepare_base_model_config = self.config.prepare_base_model
        params = self.params
        training_data = Path(self.config.data_ingestion.unzip_dir)

    
        create_directories([
            Path(model_training.root_dir),
            Path(model_training.trained_model_path).parent,
        ])
        
        model_training_config = ModelTrainingConfig(
            root_dir=Path(model_training.root_dir),
            trained_model_path=Path(model_training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            model_name=params.MODEL_NAME,
            artifacts_dir=params.ARTIFACTS_DIR,
            resume_from_checkpoint = bool(params.RESUME_TRAINING)
        )

        return model_training_config