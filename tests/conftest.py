import re
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import task_manager

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def default_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_API_APP_ENV", "dev")
    monkeypatch.setenv("SELF_API_PUBLIC_BASE_URL", "")
    monkeypatch.setenv("SELF_API_FILE_ACCESS_ROOTS", "")
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "false")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "change-me-in-production")
    monkeypatch.setenv("SELF_API_SESSION_COOKIE_SECURE", "false")
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def tmp_datasets_root() -> Path:
    data_root = ROOT_DIR / "tmp_datasets" / "pytest_data"
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


@pytest.fixture
def case_dir(tmp_datasets_root: Path, request: pytest.FixtureRequest) -> Path:
    case_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", request.node.name).strip("_")
    case_path = tmp_datasets_root / case_name
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True, exist_ok=True)
    return case_path


@pytest.fixture
def isolated_runtime(monkeypatch: pytest.MonkeyPatch) -> Path:
    storage_dir = ROOT_DIR / "tmp_datasets" / "test_storage"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    monkeypatch.setenv("SELF_API_STORAGE_ROOT", "./tmp_datasets/test_storage")
    get_settings.cache_clear()
    task_manager._TASKS.clear()
    task_manager._TASK_QUEUES.clear()
    task_manager._ACTIVE_QUEUED_TASKS.clear()
    task_manager._CALLBACK_URL_LOCKS.clear()
    task_manager._PIPELINE_PROGRESS.clear()

    yield storage_dir

    task_manager._TASKS.clear()
    task_manager._TASK_QUEUES.clear()
    task_manager._ACTIVE_QUEUED_TASKS.clear()
    task_manager._CALLBACK_URL_LOCKS.clear()
    task_manager._PIPELINE_PROGRESS.clear()
    get_settings.cache_clear()
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
