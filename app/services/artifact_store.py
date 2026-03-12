import json
import mimetypes
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.path_safety import is_path_allowed, resolve_safe_path


class ArtifactRecord(TypedDict):
    artifact_id: str
    kind: Literal["file", "directory"]
    source: str
    file_name: str
    path: str
    size_bytes: int | None
    content_type: str | None
    created_at: str
    task_id: str | None
    task_type: str | None


_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _storage_root() -> Path:
    root = resolve_safe_path(
        get_settings().resolved_storage_root,
        field_name="storage_root",
        must_exist=False,
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _metadata_dir() -> Path:
    path = _storage_root() / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_cache_dir() -> Path:
    path = _storage_root() / "download_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(filename: str) -> str:
    text = Path(filename or "upload.bin").name.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or "upload.bin"


def _artifact_metadata_path(artifact_id: str) -> Path:
    return _metadata_dir() / f"{artifact_id}.json"


def _write_record(record: ArtifactRecord) -> None:
    with _artifact_metadata_path(record["artifact_id"]).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)


def _read_record(path: Path) -> ArtifactRecord:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_record(
    *,
    artifact_id: str,
    path: Path,
    source: str,
    content_type: str | None = None,
    task_id: str | None = None,
    task_type: str | None = None,
    file_name: str | None = None,
) -> ArtifactRecord:
    resolved = resolve_safe_path(path, field_name="artifact_path", must_exist=True)
    stat = resolved.stat()
    guessed_type, _ = mimetypes.guess_type(resolved.name)
    return {
        "artifact_id": artifact_id,
        "kind": "directory" if resolved.is_dir() else "file",
        "source": source,
        "file_name": file_name or resolved.name,
        "path": str(resolved),
        "size_bytes": None if resolved.is_dir() else int(stat.st_size),
        "content_type": content_type or guessed_type,
        "created_at": _now_iso(),
        "task_id": task_id,
        "task_type": task_type,
    }


async def save_upload(
    upload: UploadFile,
    *,
    source: str = "upload",
    task_id: str | None = None,
    task_type: str | None = None,
) -> ArtifactRecord:
    settings = get_settings()
    artifact_id = uuid.uuid4().hex
    safe_name = _safe_filename(upload.filename or "upload.bin")
    upload_dir = _storage_root() / "uploads" / datetime.now(timezone.utc).strftime("%Y/%m/%d")
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_path = upload_dir / f"{artifact_id}__{safe_name}"

    size_bytes = 0
    with output_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_size_bytes:
                handle.close()
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    f"upload exceeds max size of {settings.max_upload_size_mb} MB"
                )
            handle.write(chunk)

    record = _build_record(
        artifact_id=artifact_id,
        path=output_path,
        source=source,
        content_type=upload.content_type,
        task_id=task_id,
        task_type=task_type,
        file_name=safe_name,
    )
    with _LOCK:
        _write_record(record)
    await upload.close()
    return record


def register_path(
    path: str | Path,
    *,
    source: str,
    task_id: str | None = None,
    task_type: str | None = None,
    file_name: str | None = None,
) -> ArtifactRecord:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise ValueError(f"artifact path does not exist: {resolved}")
    if not is_path_allowed(resolved):
        raise ValueError(f"artifact path is outside allowed roots: {resolved}")

    record = _build_record(
        artifact_id=uuid.uuid4().hex,
        path=resolved,
        source=source,
        task_id=task_id,
        task_type=task_type,
        file_name=file_name,
    )
    with _LOCK:
        _write_record(record)
    return record


def list_artifacts(
    *,
    source: str | None = None,
    task_id: str | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for path in sorted(_metadata_dir().glob("*.json"), reverse=True):
        record = _read_record(path)
        if source and record["source"] != source:
            continue
        if task_id and record["task_id"] != task_id:
            continue
        if kind and record["kind"] != kind:
            continue
        records.append(record)
        if len(records) >= limit:
            break
    return records


def get_artifact(artifact_id: str) -> ArtifactRecord | None:
    path = _artifact_metadata_path(artifact_id)
    if not path.exists():
        return None
    return _read_record(path)


def prepare_download(artifact_id: str) -> tuple[Path, str, str | None]:
    record = get_artifact(artifact_id)
    if record is None:
        raise ValueError(f"artifact not found: {artifact_id}")

    artifact_path = Path(record["path"]).expanduser().resolve(strict=False)
    if not artifact_path.exists():
        raise ValueError(f"artifact path no longer exists: {artifact_path}")

    if artifact_path.is_file():
        return artifact_path, record["file_name"], record["content_type"]

    archive_base = _download_cache_dir() / f"{artifact_id}_{artifact_path.name}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=artifact_path))
    return archive_path, f"{artifact_path.name}.zip", "application/zip"


def summarize_record(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": record["artifact_id"],
        "kind": record["kind"],
        "source": record["source"],
        "file_name": record["file_name"],
        "size_bytes": record["size_bytes"],
        "content_type": record["content_type"],
        "created_at": record["created_at"],
        "task_id": record["task_id"],
        "task_type": record["task_type"],
    }
