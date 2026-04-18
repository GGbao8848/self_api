"""Smoke tests for preprocess request models that previously had aliases (now removed)."""

from app.schemas.preprocess import (
    MovePathRequest,
    UnzipArchiveRequest,
    ZipFolderRequest,
)


def test_zip_folder_request_fields() -> None:
    m = ZipFolderRequest.model_validate(
        {"input_dir": "/src", "output_zip_path": "/dst/a.zip"},
    )
    assert m.output_zip_path == "/dst/a.zip"


def test_unzip_archive_request_fields() -> None:
    m = UnzipArchiveRequest.model_validate(
        {"archive_path": "/a.zip", "output_dir": "/out"},
    )
    assert m.archive_path == "/a.zip"


def test_move_path_request_fields() -> None:
    m = MovePathRequest.model_validate(
        {"source_path": "/from", "target_dir": "/to"},
    )
    assert m.source_path == "/from"
    assert m.target_dir == "/to"
