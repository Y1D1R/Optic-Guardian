from RetinoNet import logger # type: ignore
from RetinoNet import get_logger
from RetinoNet.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from RetinoNet.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "01 - Data Ingestion stage"
log = get_logger(__name__)
try:
    log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    DataIngestionTrainingPipeline().main()
    log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception:
    log.exception("stage failed")
    raise

STAGE_NAME = "02 - Prepare Base Model stage"
log = get_logger(__name__)
try:
    log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    PrepareBaseModelTrainingPipeline().main()
    log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception:
    log.exception("stage failed")
    raise