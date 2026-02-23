import shutil
import zipfile
from pathlib import Path

from app.schemas.preprocess import (
    CopyPathRequest,
    CopyPathResponse,
    MovePathRequest,
    MovePathResponse,
    UnzipArchiveRequest,
    UnzipArchiveResponse,
    ZipFolderRequest,
    ZipFolderResponse,
)


def run_zip_folder(request: ZipFolderRequest) -> ZipFolderResponse:
    input_dir = Path(request.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")

    if request.output_zip_path:
        output_zip_path = Path(request.output_zip_path).expanduser().resolve()
    else:
        output_zip_path = input_dir.parent / f"{input_dir.name}.zip"

    if output_zip_path.suffix.lower() != ".zip":
        output_zip_path = output_zip_path.with_suffix(".zip")

    if output_zip_path.exists() and not request.overwrite:
        raise ValueError(f"output zip already exists: {output_zip_path}")

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    packed_files = 0
    total_bytes = 0
    mode = "w" if request.overwrite else "x"
    with zipfile.ZipFile(output_zip_path, mode=mode, compression=zipfile.ZIP_DEFLATED) as zipf:
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue
            if request.include_root_dir:
                arcname = path.relative_to(input_dir.parent)
            else:
                arcname = path.relative_to(input_dir)
            zipf.write(path, arcname=arcname)
            packed_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    return ZipFolderResponse(
        input_dir=str(input_dir),
        output_zip_path=str(output_zip_path),
        packed_files=packed_files,
        total_bytes=total_bytes,
    )


def _ensure_safe_extract_path(output_dir: Path, member_name: str) -> Path:
    target_path = (output_dir / member_name).resolve()
    base = output_dir.resolve()
    if target_path != base and base not in target_path.parents:
        raise ValueError(f"unsafe archive path detected: {member_name}")
    return target_path


def run_unzip_archive(request: UnzipArchiveRequest) -> UnzipArchiveResponse:
    archive_path = Path(request.archive_path).expanduser().resolve()
    if not archive_path.exists() or not archive_path.is_file():
        raise ValueError(f"archive_path does not exist or is not a file: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"archive_path is not a valid zip file: {archive_path}")

    if request.output_dir:
        output_dir = Path(request.output_dir).expanduser().resolve()
    else:
        output_dir = (archive_path.parent / archive_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = 0
    skipped_files = 0

    with zipfile.ZipFile(archive_path, mode="r") as zipf:
        for member in zipf.infolist():
            if member.is_dir():
                continue

            target_path = _ensure_safe_extract_path(output_dir, member.filename)
            if target_path.exists() and not request.overwrite:
                skipped_files += 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zipf.open(member, mode="r") as src:
                with target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            extracted_files += 1

    return UnzipArchiveResponse(
        archive_path=str(archive_path),
        output_dir=str(output_dir),
        extracted_files=extracted_files,
        skipped_files=skipped_files,
    )


def run_move_path(request: MovePathRequest) -> MovePathResponse:
    source_path = Path(request.source_path).expanduser().resolve()
    target_dir = Path(request.target_dir).expanduser().resolve()

    if not source_path.exists():
        raise ValueError(f"source_path does not exist: {source_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_dir.is_dir():
        raise ValueError(f"target_dir is not a directory: {target_dir}")

    moved_type = "directory" if source_path.is_dir() else "file"
    target_path = target_dir / source_path.name

    if source_path == target_path:
        raise ValueError("source_path and target_path are identical")

    if moved_type == "directory" and (source_path in target_dir.parents or source_path == target_dir):
        raise ValueError("cannot move directory into itself or its subdirectory")

    if target_path.exists():
        if not request.overwrite:
            raise ValueError(f"target path already exists: {target_path}")

        if source_path.is_file() and target_path.is_dir():
            raise ValueError("cannot overwrite directory with file")
        if source_path.is_dir() and target_path.is_file():
            raise ValueError("cannot overwrite file with directory")

        if target_path.is_file():
            target_path.unlink()
        else:
            shutil.rmtree(target_path)

    try:
        shutil.move(str(source_path), str(target_path))
    except OSError as exc:
        raise ValueError(f"failed to move path: {exc}") from exc

    return MovePathResponse(
        source_path=str(source_path),
        target_path=str(target_path),
        moved_type="directory" if moved_type == "directory" else "file",
    )


def run_copy_path(request: CopyPathRequest) -> CopyPathResponse:
    source_path = Path(request.source_path).expanduser().resolve()
    target_dir = Path(request.target_dir).expanduser().resolve()

    if not source_path.exists():
        raise ValueError(f"source_path does not exist: {source_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_dir.is_dir():
        raise ValueError(f"target_dir is not a directory: {target_dir}")

    copied_type = "directory" if source_path.is_dir() else "file"
    target_path = target_dir / source_path.name

    if source_path == target_path:
        raise ValueError("source_path and target_path are identical")

    if copied_type == "directory" and (source_path in target_dir.parents or source_path == target_dir):
        raise ValueError("cannot copy directory into itself or its subdirectory")

    if target_path.exists():
        if not request.overwrite:
            raise ValueError(f"target path already exists: {target_path}")

        if source_path.is_file() and target_path.is_dir():
            raise ValueError("cannot overwrite directory with file")
        if source_path.is_dir() and target_path.is_file():
            raise ValueError("cannot overwrite file with directory")

        if target_path.is_file():
            target_path.unlink()
        else:
            shutil.rmtree(target_path)

    try:
        if source_path.is_file():
            shutil.copy2(str(source_path), str(target_path))
        else:
            shutil.copytree(str(source_path), str(target_path))
    except OSError as exc:
        raise ValueError(f"failed to copy path: {exc}") from exc

    return CopyPathResponse(
        source_path=str(source_path),
        target_path=str(target_path),
        copied_type="directory" if copied_type == "directory" else "file",
    )
