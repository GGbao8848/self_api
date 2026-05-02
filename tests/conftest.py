import re
import shutil
import sys
import asyncio
from pathlib import Path

import anyio.to_thread
import httpx
import pytest
import fastapi.testclient
import starlette.testclient

from app.core.config import get_settings
from app.main import app
from app.services import task_manager
from app.agent.sessions import agent_session_store

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestClient:
    __test__ = False

    def __init__(self, app, base_url: str = "http://testserver") -> None:
        self._app = app
        self._base_url = base_url
        self._cookies = httpx.Cookies()
        self._headers = {"user-agent": "testclient"}

    async def _request_async(self, method: str, url: str, **kwargs) -> httpx.Response:
        headers = dict(self._headers)
        headers.update(kwargs.pop("headers", {}) or {})
        transport = httpx.ASGITransport(app=self._app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=self._base_url,
            follow_redirects=True,
            cookies=self._cookies,
            headers=headers,
        ) as client:
            response = await client.request(method, url, **kwargs)
            self._cookies = client.cookies
        return response

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request_async(method, url, **kwargs))

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        return None

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


fastapi.testclient.TestClient = TestClient
starlette.testclient.TestClient = TestClient


@pytest.fixture(scope="session", autouse=True)
def inline_sync_routes_for_tests():
    original_run_sync = anyio.to_thread.run_sync

    async def inline_run_sync(func, *args, **kwargs):
        kwargs.pop("abandon_on_cancel", None)
        kwargs.pop("cancellable", None)
        kwargs.pop("limiter", None)
        return func(*args)

    anyio.to_thread.run_sync = inline_run_sync
    yield
    anyio.to_thread.run_sync = original_run_sync


@pytest.fixture(scope="session")
def client(inline_sync_routes_for_tests) -> TestClient:
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
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_HOST", "")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_USERNAME", "")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH", "")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PORT", "22")
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
    task_manager.reset_runtime_state(clear_persistent_store=True)
    agent_session_store.clear()

    yield storage_dir

    task_manager.reset_runtime_state(clear_persistent_store=True)
    agent_session_store.clear()
    get_settings.cache_clear()
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
