from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_text_file


def test_move_path_endpoint_file(client: TestClient, case_dir: Path) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    response = client.post(
        "/api/v1/preprocess/move-path",
        json={"source_path": str(source_file), "target_dir": str(target_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["moved_type"] == "file"
    moved_path = target_dir / "one.txt"
    assert moved_path.exists()
    assert moved_path.read_text(encoding="utf-8") == "payload"
    assert not source_file.exists()


def test_move_path_endpoint_directory(client: TestClient, case_dir: Path) -> None:
    source_dir = case_dir / "folder_a"
    create_text_file(source_dir / "nested" / "k.txt", "v")
    target_dir = case_dir / "target_root"

    response = client.post(
        "/api/v1/preprocess/move-path",
        json={"source_path": str(source_dir), "target_dir": str(target_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["moved_type"] == "directory"
    moved_dir = target_dir / "folder_a"
    assert moved_dir.exists()
    assert (moved_dir / "nested" / "k.txt").read_text(encoding="utf-8") == "v"
    assert not source_dir.exists()
