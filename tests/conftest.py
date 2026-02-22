import re
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


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
