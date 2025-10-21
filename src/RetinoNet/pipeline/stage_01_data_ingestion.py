# src/RetinoNet/pipeline/stage_01_data_ingestion.py
from RetinoNet.config.configuration import ConfigurationManager
from RetinoNet.components.data_ingestion import DataIngestion
from RetinoNet import get_logger

STAGE_NAME = "01 - Data Ingestion stage"
log = get_logger(__name__)


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        DataIngestionTrainingPipeline().main()
        log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception:
        log.exception("stage failed")
        raise