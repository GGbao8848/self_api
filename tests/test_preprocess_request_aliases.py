"""Legacy JSON keys (aliases) still validate on preprocess request models."""

from app.schemas.preprocess import (
    MovePathRequest,
    SplitYoloDatasetRequest,
    UnzipArchiveRequest,
    ZipFolderRequest,
)


def test_split_yolo_accepts_input_dir_alias() -> None:
    m = SplitYoloDatasetRequest.model_validate(
        {"input_dir": "/data/yolo", "output_dir": "/out/split"},
    )
    assert m.dataset_dir == "/data/yolo"
    assert m.output_dir == "/out/split"


def test_zip_folder_accepts_output_dir_alias() -> None:
    m = ZipFolderRequest.model_validate(
        {"input_dir": "/src", "output_dir": "/dst/a.zip"},
    )
    assert m.output_zip_path == "/dst/a.zip"


def test_unzip_accepts_input_dir_alias() -> None:
    m = UnzipArchiveRequest.model_validate(
        {"input_dir": "/a.zip", "output_dir": "/out"},
    )
    assert m.archive_path == "/a.zip"


def test_move_path_accepts_input_output_dir_aliases() -> None:
    m = MovePathRequest.model_validate(
        {"input_dir": "/from", "output_dir": "/to"},
    )
    assert m.source_path == "/from"
    assert m.target_dir == "/to"
