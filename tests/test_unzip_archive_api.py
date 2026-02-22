import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


def test_unzip_archive_endpoint(client: TestClient, case_dir: Path) -> None:
    archive_path = case_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("sample/x.txt", "x")
        zipf.writestr("sample/y.txt", "y")

    output_dir = case_dir / "unzipped"
    response = client.post(
        "/api/v1/preprocess/unzip-archive",
        json={"archive_path": str(archive_path), "output_dir": str(output_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["extracted_files"] == 2
    assert data["skipped_files"] == 0
    assert (output_dir / "sample" / "x.txt").read_text(encoding="utf-8") == "x"
    assert (output_dir / "sample" / "y.txt").read_text(encoding="utf-8") == "y"
