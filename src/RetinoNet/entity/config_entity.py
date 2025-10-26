# src/RetinoNet/entity/config_entity.py
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True )
class PrepareBaseModelConfig:
    root_dir: Path
    model_name: str
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True )
class ModelTrainingConfig:
    root_dir: Path
    model_name: str
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    artifacts_dir: str
    resume_from_checkpoint: bool
    