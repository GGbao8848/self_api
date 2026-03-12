from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_rejects_paths_outside_allowed_roots(tmp_path, case_dir, isolated_runtime) -> None:
    get_settings.cache_clear()

    external_file = tmp_path / "outside.txt"
    external_file.write_text("outside", encoding="utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/preprocess/copy-path",
            json={
                "source_path": str(external_file),
                "target_dir": str(case_dir),
                "overwrite": False,
            },
        )
        assert response.status_code == 400
        assert "outside allowed data roots" in response.json()["detail"]
