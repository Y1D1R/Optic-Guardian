 # src/RetinoNet/utils/common.py
#from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Sequence, Union

from box import ConfigBox # type: ignore
from box.exceptions import BoxValueError # type: ignore
from ensure import ensure_annotations # type: ignore
import joblib # type: ignore
import yaml # type: ignore

from RetinoNet import logger
log = logger.getChild(__name__)

PathLike = Union[str, Path]


# ---------------- internal helpers ----------------

def _to_path(p: PathLike) -> Path:
    """
    Accept str or Path and return Path
    """
    return p if isinstance(p, Path) else Path(p)


def _atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Write text atomically: write to .tmp then replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="\n") as f:
        f.write(data)
    os.replace(tmp, path)


# ---------------- YAML ----------------

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return a ConfigBox
    Raises ValueError if file is empty
    """
    p = _to_path(path_to_yaml)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if content is None:
            raise ValueError("YAML file is empty")
        if not isinstance(content, dict):
            raise TypeError("YAML root must be a dict")
        log.info(f"yaml loaded: {p}")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception:
        log.exception(f"failed to load yaml: {p}")
        raise


# ---------------- directories ----------------

@ensure_annotations
def create_directories(paths: list, verbose: bool = True) -> type(None): # type: ignore
    """
    Create a list of directories (parents included)
    """
    for path in paths:
        p = _to_path(path)
        p.mkdir(parents=True, exist_ok=True)
        if verbose:
            log.info(f"dir ready: {p}")


# ---------------- JSON ----------------

@ensure_annotations
def save_json(path: Path, data: dict) -> type(None): # type: ignore
    """
    Save dict to JSON using atomic write
    """
    p = _to_path(path)
    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        _atomic_write_text(p, payload, encoding="utf-8")
        log.info(f"json saved: {p}")
    except Exception:
        log.exception(f"failed to save json: {p}")
        raise


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON and return a ConfigBox to access keys as attributes
    """
    p = _to_path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, dict):
            out = ConfigBox(content)
        else:
            # fallback: wrap non-dict values
            out = ConfigBox({"value": content})
        log.info(f"json loaded: {p}")
        return out
    except Exception:
        log.exception(f"failed to load json: {p}")
        raise


# ---------------- binary (joblib) ----------------

@ensure_annotations
def save_bin(data: Any, path: Path) -> type(None): # type: ignore
    """
    Save an object with joblib (with light compression)
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(value=data, filename=str(p), compress=3)
        log.info(f"bin saved: {p}")
    except Exception:
        log.exception(f"failed to save bin: {p}")
        raise


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load an object saved with joblib
    """
    p = _to_path(path)
    if not p.exists():
        raise FileNotFoundError(f"bin not found: {p}")
    try:
        obj = joblib.load(str(p))
        log.info(f"bin loaded: {p}")
        return obj
    except Exception:
        log.exception(f"failed to load bin: {p}")
        raise


# ---------------- file size ----------------

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Return size (B/KB/MB/GB)
    """
    p = _to_path(path)
    size = p.stat().st_size  # bytes
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


# ---------------- base64 (images) ----------------

@ensure_annotations
def decodeImage(imgstring: Union[str, bytes], fileName: Path) -> Path:
    """
    Decode a base64 image (also supports data URL) and write it to disk
    Returns the written Path
    """
    p = _to_path(fileName)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(imgstring, str):
        # remove data URL prefix if present
        if ";base64," in imgstring:
            imgstring = imgstring.split(";base64,", 1)[1]
        img_bytes = base64.b64decode(imgstring, validate=False)
    else:
        img_bytes = base64.b64decode(imgstring, validate=False)

    with p.open("wb") as f:
        f.write(img_bytes)
    log.info(f"image decoded to: {p}")
    return p


@ensure_annotations
def encodeImageIntoBase64(imagePath: Path) -> bytes:
    """
    Read an image from disk and return base64 bytes
    Use .decode('utf-8') if you need a string
    """
    p = _to_path(imagePath)
    with p.open("rb") as f:
        encoded = base64.b64encode(f.read())
    log.info(f"image encoded from: {p}")
    return encoded


__all__ = [
    "read_yaml",
    "create_directories",
    "save_json",
    "load_json",
    "save_bin",
    "load_bin",
    "get_size",
    "decodeImage",
    "encodeImageIntoBase64",
]