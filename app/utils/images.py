from pathlib import Path
from typing import Iterable

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if not extensions:
        return DEFAULT_EXTENSIONS
    normalized = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized or DEFAULT_EXTENSIONS


def list_image_paths(
    input_dir: Path,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
) -> list[Path]:
    normalized = normalize_extensions(extensions)
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in normalized]
    return sorted(files)
