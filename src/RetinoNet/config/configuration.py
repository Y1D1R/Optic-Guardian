# src/RetinoNet/config/configuration.py
from pathlib import Path
import os

from RetinoNet.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from RetinoNet.utils.common import read_yaml, create_directories
from RetinoNet.entity.config_entity import DataIngestionConfig
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

        di = DataIngestionConfig(
            root_dir=root_dir,
            source_URL=str(cfg.source_URL),
            local_data_file=local_data_file,
            unzip_dir=unzip_dir,
        )
        log.info("data_ingestion config built")
        return di