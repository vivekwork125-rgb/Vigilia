import os
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = tempfile.mkdtemp(prefix="vigilia-tests-")
os.environ["DATA_DIR"] = TEST_DATA
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DATA}/test.db"
os.environ["DEMO_MODE"] = "true"
os.environ["PERCEPTION_BACKEND"] = "hog"
sys.path.insert(0, str(ROOT / "backend"))
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
