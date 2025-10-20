from RetinoNet import logger # type: ignore
from RetinoNet import get_logger
from RetinoNet.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "01 - Data Ingestion stage"
log = get_logger(__name__)
try:
    log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    DataIngestionTrainingPipeline().main()
    log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception:
    log.exception("stage failed")
    raise