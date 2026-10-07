# VIGILIA

**Evidence-Grounded Surveillance Intelligence**

A standalone hackathon prototype that connects natural-language event search to source footage, exact frames, timelines, relationships, and evidence-backed reports. Observed facts, correlations, and inferences are kept visibly separate.

![VIGILIA investigation workspace](docs/preview.jpg)

## Run locally

Requirements: Python 3.12 or 3.13, Node 20.9+ (22 recommended), and npm.

```bash
./scripts/dev.sh
```

Open **http://localhost:3100**. API documentation: **http://127.0.0.1:8100/docs**.

The script creates an isolated Python environment, installs dependencies, starts both services, and creates three deterministic synthetic videos on first launch. No cloud credentials or GPU are needed. Runtime data stays in `data/` and is excluded from Git. The app can ingest real MP4/MOV/AVI/MKV footage separately from its demo.

For manual startup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm ci && cd ..
PYTHONPATH=backend .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
# In another terminal:
cd frontend && npm run dev
```

## What works

- Seven responsive screens: command center, ingestion/library, search/results, investigation, evidence graph, and evaluation.
- Validated streaming uploads, source hashing, persisted async jobs, progress/error reporting, retry, camera/time metadata, and browser-playable video derivatives.
- Modular YOLO + ByteTrack integration; lightweight pretrained HOG + IoU fallback for CPU-only people detection. Track crops, boxes, observations, events, and evidence are stored transactionally.
- Event/attribute/camera/location/duration/clock filters, temporal joins, lexical semantic baseline, optional OpenCLIP retrieval, and measured match explanations.
- Exact video seek, bounding-box overlay, downloadable clips, entity timeline, conservative cross-camera candidates, uncertainty labels, and negative-search coverage.
- Coarse reference-image search, interactive graph, persistent investigation collections, and Markdown reports.
- Forty annotated benchmark queries, calculated retrieval metrics, exportable runs, and paired investigation-time measurement.
- SQLite quickstart; PostgreSQL/pgvector Docker configuration; same-origin API proxy and optional shared-token access control.

## Model modes

The default one-command demo uses the small CPU baseline. Enable the stronger perception adapter for real footage:

```bash
.venv/bin/pip install -r backend/requirements-models.txt
# Set these in .env before running scripts/dev.sh:
# PERCEPTION_BACKEND=yolo
# YOLO_MODEL=yolo11n.pt
# ENABLE_CLIP=true      # optional, downloads a vision-language model
# ENABLE_OCR=true       # optional, downloads OCR models
```

Copy `.env.example` to `.env` to configure FPS, detection size, database, model adapters, and upload limits. Set `API_TOKEN` in `.env` for protected access. `DEMO_MODE=false` requires at least 24 token characters. Both services must share the token. Outside `scripts/dev.sh`, load the variables into each process explicitly; Python does not automatically read `.env`.

See [model notes](models/README.md) for checkpoint, licensing, capabilities, and limitations. Missing enabled model dependencies fail explicitly instead of fabricating detections. No foundation-model training is required.

## Docker / PostgreSQL

```bash
docker compose up --build
```

The database is internal; API and web ports bind to localhost. PostgreSQL starts with pgvector enabled. The API creates the relational schema automatically. Use **one API worker**: this prototype's durable queue is single-process. Docker's default image includes CPU perception; install model extras in a custom image for YOLO/OpenCLIP/OCR. Docker execution should be separately validated on the target host.

## Verify

```bash
.venv/bin/pytest -q
cd frontend && npm run typecheck && npm run build
cd .. && .venv/bin/python scripts/benchmark.py > docs/benchmark-results.json
```

Tests exercise actual upload/CPU processing, corrupted input, source linking, retrieval, temporal constraints, empty results, ambiguity, image search, clips, report export, metric calculations, and deterministic detector integration. See [verification record](docs/verification.md) for what was actually run.

## Evidence honesty

The included footage is a clearly labeled synthetic reconstruction. Demo events and trajectories are authored annotations, not YOLO output. Baseline text search is TF-IDF with normalized vocabulary; image search is HSV similarity. Cross-camera candidates are appearance hypotheses, never confirmed identities. No face recognition is implemented.

Real-footage placement/pickup/ownership and complex interaction understanding are not production-validated. The CPU fallback is people-only and weak under occlusion. Temporal localization and identity metrics remain unmeasured. The 40-query fixture is a development benchmark, not a held-out dataset; its results must not be advertised as real-world accuracy. See the [technical report and limitations](docs/architecture.md).

## Repository map

```text
backend/app/     FastAPI, schema, perception, queue, events, retrieval, evidence, evaluation
frontend/        Next.js / React / TypeScript / Tailwind investigation UI
models/          Model adapter documentation; optional local weights (ignored)
scripts/         One-command startup, benchmark runner, schema exporter
tests/           Backend and integration tests
data/            Generated media, uploads, derivatives, SQLite (ignored)
db/              PostgreSQL setup and generated schema
docs/            Technical report, demo guide, measured benchmark, verification
```

[Demo guide](docs/demo.md) · [Architecture and evaluation methodology](docs/architecture.md) · [Database schema](db/schema.sql)

**Search retrieves evidence. Reasoning connects evidence. The interface exposes evidence.**
