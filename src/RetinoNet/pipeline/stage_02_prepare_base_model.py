# src/RetinoNet/pipeline/stage_02_prepare_base_model.py
from RetinoNet.config.configuration import ConfigurationManager
from RetinoNet.components.prepare_base_model import PrepareBaseModel
from RetinoNet import get_logger

STAGE_NAME = "02 - Prepare Base Model stage"
log = get_logger(__name__)

class PrepareBaseModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        log.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        PrepareBaseModelTrainingPipeline().main()
        log.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception:
        log.exception("stage failed")
        raise