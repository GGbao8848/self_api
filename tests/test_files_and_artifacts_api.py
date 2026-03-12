from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_upload_list_and_download_artifact(isolated_runtime) -> None:
    get_settings.cache_clear()

    with TestClient(app) as client:
        upload_resp = client.post(
            "/api/v1/files/upload",
            files={"file": ("sample.txt", b"hello artifact", "text/plain")},
        )
        assert upload_resp.status_code == 201
        artifact = upload_resp.json()["artifact"]
        artifact_id = artifact["artifact_id"]
        assert artifact["file_name"] == "sample.txt"
        assert artifact["source"] == "upload"

        list_resp = client.get("/api/v1/artifacts")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] >= 1

        detail_resp = client.get(f"/api/v1/artifacts/{artifact_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["artifact_id"] == artifact_id

        download_resp = client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert download_resp.status_code == 200
        assert download_resp.content == b"hello artifact"
