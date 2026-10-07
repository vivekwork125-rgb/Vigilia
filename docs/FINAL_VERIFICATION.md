# Final verification, 2026-10-07

“Tested YES” below means the named path was executed in this repository. Synthetic-video tests are identified as such; they do not establish field performance. This project does **not** meet every acceptance criterion in the master prompt.

| Capability | Implemented | Tested | Evidence |
|---|---|---|---|
| Video ingestion | YES | YES | Upload/metadata/hash and index tests in `tests/test_evidence_flow.py` and `tests/test_temporal_intelligence.py` |
| Object detection | YES | YES | HOG upload test; synthetic decoded-pixel contours; earlier YOLO/ByteTrack still-image smoke in `model-smoke-results.json`; moving natural-video object accuracy unmeasured |
| Tracking | YES | YES | Persisted observations/tracks/boxes asserted in pixel-video integration test; YOLO/ByteTrack adapter integration smoke only |
| Event extraction | YES | YES | `temporal_events.py` track transitions; 29 backend tests; encoded-video fixture |
| Placement | YES | YES | Moving object → stationary → person departs; direct state tests and uploaded pixel-video source/search test; synthetic footage only |
| Pickup | YES | YES | Stationary object → nearby coupled motion; direct and seeded decoded-video tests; synthetic footage only |
| Unattended object | YES | YES | Configurable duration after person leaves, object stays, and no other tracked person remains near; source-linked fixture and state tests; synthetic footage only |
| Temporal queries | YES | YES | `temporal.py` operator tests and source-dated before/after query test |
| Semantic search | YES | YES | TF-IDF hybrid retrieval and query tests; broad language generalization unmeasured |
| Image search | YES | YES | HSV reference-image search and source/clip test; OpenCLIP unavailable |
| OCR | NO | NO | `easyocr` not installed; optional code path unexecuted |
| OpenCLIP text/image retrieval | NO | NO | `open_clip` not installed; no embeddings/search run |
| Cross-camera association | YES | YES | Candidate/ambiguity test on authored synthetic tracks; no real-video identity validation |
| Evidence graph | YES | YES | Stored relationship edges in API; browser graph edge opened pipeline unattended evidence; old authored relation remains labeled |
| Evidence provenance | YES | YES | Event → evidence → observation → track → sampled box → video in API/integration test; source SHA-256 |
| Timeline | YES | YES | Browser viewed pipeline timeline and selected events; category/time-window controls compile, but user-study usability unmeasured |
| Video evidence jump | YES | YES | Browser showed 12.0-second seek for unattended event, frames 120–185; media Range and report tests |
| Fact/inference separation | YES | YES | Event categories, viewer badge, report assertions; pixel-derived behavior remains INFERRED |
| Negative evidence | YES | YES | Camera/time coverage and no-absolute-absence test; camera operational status unknown |
| Reports | YES | YES | Evidence ID and INFERRED category asserted in uploaded-video report test |
| Retrieval evaluation on independently labeled footage | NO | NO | 40-query authored fixture runs; independent 30–50-query field dataset absent; `scripts/evaluate_independent.py` is ready but no results exist |
| Investigation-time evaluation | NO | NO | Recorder and pairing logic tested with unit values; zero measured human pairs; no time-reduction claim |
| Docker/PostgreSQL path | NO | NO | `docker version` and `docker compose ps` failed: Docker API socket missing; Compose stack not run |

Validation completed: full backend test suite passed, TypeScript typecheck passed, and the Next.js production build passed. The live browser opened a pipeline-generated unattended event and its stored graph relationship, showing the source seek and uncertainty label. The earlier three-camera benchmark is intentionally isolated from the newer pipeline fixture; it remains an authored-annotation regression metric only.

Acceptance remains open for independently reviewed natural footage (especially people, portable objects, vehicles, occlusion, and low light), a representative 30–50-query retrieval benchmark, actual paired investigator timings, optional OCR/OpenCLIP if claimed, and Docker/PostgreSQL deployment verification. No significant investigation-time improvement or real-world action accuracy is established.
