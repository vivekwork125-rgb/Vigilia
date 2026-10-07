"""Run the checked-in query fixture without requiring a running HTTP server."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.db import initialize, session
from app.demo import seed
from app.evaluation import run_evaluation

initialize()
seed()
with session() as s:
    result = run_evaluation(s)
print(json.dumps(result, indent=2))
