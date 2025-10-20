# src/RetinoNet/components/data_ingestion.py
import zipfile
import gdown
from pathlib import Path

from RetinoNet import logger
from RetinoNet.entity.config_entity import DataIngestionConfig
from RetinoNet.utils.common import get_size

log = logger.getChild(__name__)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        """
        Download the dataset archive into local_data_file
        """
        dataset_url = str(self.config.source_URL)
        zip_download_path = Path(self.config.local_data_file)

        # ensure target folder
        zip_download_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"downloading from {dataset_url} -> {zip_download_path}")

        try:
            # If URL is a Google Drive "view" link, extract the file id
            if "drive.google.com" in dataset_url:
                file_id = dataset_url.split("/")[-2]
                prefix = "https://drive.google.com/uc?export=download&id="
                gdown.download(url=f"{prefix}{file_id}", output=str(zip_download_path), quiet=False)
            else:
                gdown.download(url=dataset_url, output=str(zip_download_path), quiet=False)

            if not zip_download_path.exists() or zip_download_path.stat().st_size == 0:
                raise RuntimeError("download failed or empty file")

            log.info(f"downloaded -> {zip_download_path} ({get_size(zip_download_path)})")
        except Exception:
            log.exception("download failed")
            raise

    def extract_zip_file(self) -> None:
        """
        Extract the zip archive into unzip_dir
        """
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)
        zip_path = Path(self.config.local_data_file)
        log.info(f"extracting {self.config.local_data_file} -> {unzip_path}")

        try:
            with zipfile.ZipFile(self.config.local_data_file, "r") as zf:
                zf.extractall(unzip_path)
            log.info("extraction done")
            # remove archive to free space (only after success)
            try:
                freed = get_size(zip_path)  # human-readable size
            except Exception:
                freed = "unknown size"
            
            try:
                if zip_path.exists():
                    zip_path.unlink()
                    log.info(f"removed archive: {zip_path} (freed {freed})")
                else:
                    log.warning(f"archive not found (already removed?): {zip_path}")
            except Exception:
                log.exception(f"failed to remove archive: {zip_path}")
        except Exception:
            log.exception("extraction failed")
            raise