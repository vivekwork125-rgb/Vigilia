# VIGILIA implementation audit

Baseline audited 2026-10-07 at commit `6f185a8`. This table records the state **before** the completion changes in this task. “Works” means the path was exercised, not merely present in source. The local database contains three synthetic demo videos and no uploaded user video. Baseline backend tests: `20 passed` (`.venv/bin/pytest -q`). Earlier model smoke indexed a repeated still image, which proves the YOLO/ByteTrack adapter runs but does not validate action recognition.

| Requirement | Existing implementation | Actually works? | Evidence | Missing work |
|---|---|---|---|---|
| Architecture and schema | FastAPI, SQLAlchemy, SQLite default/PostgreSQL option; Next.js frontend | YES, locally | `backend/app/main.py`, `db.py`, `frontend/src/components/`; 20 tests pass | PostgreSQL and Docker end-to-end unverified |
| Upload and indexing | Upload, durable queued worker, FFmpeg/OpenCV decode, progress | YES, tested fixture | `tests/test_evidence_flow.py` upload/index tests | Repeat with realistic moving footage and deployment stack |
| Detection and tracking | HOG CPU default; optional YOLO + ByteTrack | PARTIAL | `vision.py`, `docs/model-smoke-results.json` | HOG detects people only; no verified object/vehicle action scenario through YOLO |
| Event intelligence | Appeared, disappeared, stopped; one zone and vehicle-approach rule | PARTIAL | `processing.py:extract_events`, `process` | Starts moving, exits zone, remains zone, pairwise approach/move-away, placement, pickup, unattended state machine |
| Synthetic demo | Three encoded videos, but database tracks and rich events seeded from `actors()` and `EVENTS` | NO for pipeline-derived claims | `demo.py:seed` directly constructs Observation and authored events | Index demo through perception pipeline; keep independent scenario ground truth separate |
| Temporal queries | Deterministic parser and `before`/`after`/`during`/`near` search joins | PARTIAL | `retrieval.py`; baseline tests | Reusable explicit temporal operators, overlap/followed-by/window tests |
| Evidence provenance | Evidence points to video/frame range and often observation | PARTIAL | `evidence.py:create_event`; API returns source URLs | Verify every new event has primary observation/track/box and browser jumps to exact source |
| Natural language retrieval | Hard filters, TF-IDF and weighted scoring, optional CLIP | YES for seeded data | `retrieval.py`; search tests | Validate ranking on independently labeled pipeline-generated events |
| Image search | HSV image embedding and source result | PARTIAL | `main.py`, image-search test | Evaluate on actual uploaded footage; optional CLIP execution unverified |
| OCR | Optional EasyOCR import and event write | NO, unverified | `processing.py`; package not installed | Execute actual OCR or mark unavailable; test text retrieval |
| Cross-camera association | Appearance/color/time/connectivity candidate scoring with ambiguity | PARTIAL | `retrieval.py`, ambiguity test | Validate real track-derived candidates, gap-aware possible transition and rejection thresholds |
| Timeline and video playback | Event list and source-linked media/clip, timeline seeks on selection | PARTIAL | `evidence-workspace.tsx`, API tests | Browser verify source jump, zoom/filter/chronological navigation |
| Graph | Event/entity graph displayed | PARTIAL | `graph.tsx`, graph API | Use stored relationships, camera/location nodes and evidence-backed edge interaction |
| Reports | Stored findings to report | YES for seeded data | `main.py`, report test | Verify with track-derived findings and preserve uncertainty |
| Evaluation | 40 authored queries against authored events; ranking formulas | YES only as synthetic retrieval exercise | `demo.py:QUERIES`, `evaluation.py` | Independent interval/entity labels on pipeline output, temporal localization; no real-world accuracy claim |
| Investigation time | Paired timing records and reduction formula | PARTIAL | `evaluation.py` | Actual human paired timings; protocol for first relevant/correct/total time |
| Negative evidence | Empty search and candidate/coverage summary | PARTIAL | `retrieval.py`; existing tests | Check and show camera/time coverage and thresholds in UI |
| Frontend | Next.js workspace, ingestion, evidence, graph, evaluation | PARTIAL | `frontend/src/components`; prior typecheck/build passed | Browser interaction verification and missing state controls |
| Automated tests | 20 backend tests, 3 frontend Playwright specs | PARTIAL | `.venv/bin/pytest -q`: 20 passed | Event state, temporal, full chain, browser tests |
| Docker/PostgreSQL | Compose file, pgvector image, backend/frontend Dockerfiles | NO, unverified | `docker-compose.yml` inspected | Start stack and exercise ingestion/search or explicitly report environment limitation |

This audit is a baseline; [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md) will record the tested final state. Code presence alone is not a YES. A synthetic test does not establish field accuracy.

## Completion audit delta

The baseline was preserved above. Changes after it: `temporal_events.py` now consumes detector/tracker samples for motion, zone changes, pairwise approach/departure, placement, pickup, and unattended hypotheses with a configurable duration; `temporal.py` centralizes interval joins; inferred events retain null action confidence and source-linked observations. `/evidence/{id}` now returns the track and relevant boxes. Processing stores relationship rows for pairwise events, and the graph exposes those rows plus camera/location context. The timeline adds status and time-window filters. A separate encoded synthetic fixture is pixel-indexed through `process()`; its event labels are not injected from scenario code. Its UI label distinguishes it from the old authored fixture. Independent-label and human-timing recorders were added, but **no representative labels or human measurements exist**. Docker/PostgreSQL, OCR, and OpenCLIP remain unverified. See [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md) for executed evidence and open acceptance items.
