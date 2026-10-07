"""Explicit, environment-driven configuration. No network/model loads at import."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
DATA.mkdir(parents=True, exist_ok=True)
MEDIA = DATA / "media"
MEDIA.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA / 'vigilia.db'}")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
TOKEN = os.getenv("API_TOKEN", "")
SAMPLE_FPS = max(0.2, min(10, float(os.getenv("SAMPLE_FPS", "2"))))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
WEIGHTS = {
    "semantic": 0.35,
    "attributes": 0.2,
    "temporal": 0.15,
    "spatial": 0.1,
    "event": 0.15,
    "graph": 0.05,
}
for key in WEIGHTS:
    WEIGHTS[key] = max(0, float(os.getenv(f"WEIGHT_{key.upper()}", str(WEIGHTS[key]))))
