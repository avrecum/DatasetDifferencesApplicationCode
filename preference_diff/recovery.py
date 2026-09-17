"""Identity checks for faithful retrieval of original images from pinned archives."""

from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID
from .adapters import normalize_prompt


def source_uid(value):
    if not isinstance(value, str):
        return None
    candidate = Path(urlparse(value).path).stem if "://" in value else value
    try:
        return str(UUID(candidate))
    except ValueError:
        return None


def verify_identity(row, original):
    uid = original["image_id"].removeprefix("pickapic:")
    if source_uid(row.get("url")) != uid:
        raise ValueError("Archive URL UID differs from original image identity")
    if normalize_prompt(row.get("prompt")) != normalize_prompt(
        original["original_prompt"]
    ):
        raise ValueError("Archive generation prompt differs from original metadata")
    metadata = original["generation_metadata"]
    for key in ("negative_prompt", "model_id", "image_id", "seed", "idx"):
        if key in row and metadata.get(key) is not None and row[key] != metadata[key]:
            raise ValueError(f"Archive {key} differs from original metadata")
