from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_text_file


def test_copy_path_endpoint_file(client: TestClient, case_dir: Path) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    response = client.post(
        "/api/v1/preprocess/copy-path",
        json={"source_path": str(source_file), "target_dir": str(target_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["copied_type"] == "file"
    copied_path = target_dir / "one.txt"
    assert copied_path.exists()
    assert copied_path.read_text(encoding="utf-8") == "payload"
    assert source_file.exists()
    assert source_file.read_text(encoding="utf-8") == "payload"


def test_copy_path_endpoint_directory(client: TestClient, case_dir: Path) -> None:
    source_dir = case_dir / "folder_a"
    create_text_file(source_dir / "nested" / "k.txt", "v")
    target_dir = case_dir / "target_root"

    response = client.post(
        "/api/v1/preprocess/copy-path",
        json={"source_path": str(source_dir), "target_dir": str(target_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["copied_type"] == "directory"
    copied_dir = target_dir / "folder_a"
    assert copied_dir.exists()
    assert (copied_dir / "nested" / "k.txt").read_text(encoding="utf-8") == "v"
    assert source_dir.exists()
    assert (source_dir / "nested" / "k.txt").read_text(encoding="utf-8") == "v"
