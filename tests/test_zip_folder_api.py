import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_text_file


def test_zip_folder_endpoint(client: TestClient, case_dir: Path) -> None:
    source_dir = case_dir / "source_pack"
    create_text_file(source_dir / "a.txt", "hello")
    create_text_file(source_dir / "nested" / "b.txt", "world")

    response = client.post(
        "/api/v1/preprocess/zip-folder",
        json={"input_dir": str(source_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["packed_files"] == 2
    zip_path = Path(data["output_zip_path"])
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, mode="r") as zipf:
        names = sorted(zipf.namelist())
    assert f"{source_dir.name}/a.txt" in names
    assert f"{source_dir.name}/nested/b.txt" in names
