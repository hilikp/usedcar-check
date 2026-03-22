import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings


def _local_dir() -> Path:
    p = Path(settings.local_storage_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_upload(kind: str, file: UploadFile) -> tuple[str, int]:
    """
    Returns (relative_filename, bytes_written).
    Stored under: <LOCAL_STORAGE_DIR>/<kind>/<uuid>_<original_name>
    """
    safe_name = (file.filename or f"{kind}.bin").replace("\\", "_").replace("/", "_")
    rel = f"{kind}/{uuid4()}_{safe_name}"
    full = _local_dir() / rel
    full.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with full.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)

    return rel, size


def local_path(rel: str) -> str:
    return str((_local_dir() / rel).resolve())


def public_url(rel: str) -> str:
    base = settings.public_base_url.rstrip("/")
    return f"{base}/v1/media/{rel}"

