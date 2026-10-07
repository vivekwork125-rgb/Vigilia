# Verification record (historical baseline)

For the current completion audit and expanded test suite, see [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md).

## User story

An investigator searches for an object-placement incident, opens source evidence, examines the entity timeline and ambiguous camera transitions, collects findings, exports a report, and runs the controlled retrieval benchmark.

## Verified locally

- Production frontend compilation and TypeScript validation succeeded with Next.js 16.3.8 / React 19.3.0. Webpack is selected because the host sandbox prevented Turbopack's internal port binding.
- The backend suite contains 20 tests and tests source upload, real CPU processing, corrupt video rejection, source hashing, exact frame provenance, filters, temporal queries, negative coverage, ambiguous association, reference-image retrieval, clip encoding, reports, metric formulas, measured timing aggregation, and token protection.
- The actual YOLO11n + ByteTrack adapter ran on a five-second replay of Ultralytics' bundled bus image. It produced five observations, ten events, and four person-stopped retrieval results. This is a model integration smoke test, not a motion/identity accuracy benchmark. See `model-smoke-results.json`.
- Browser verification covers the live Next.js → FastAPI → SQLite → media flow. The sample query retrieved `EVT-E03`, and the video loaded at the evidence interval (12–15 seconds) with 768×432 source dimensions and its frame range 120–150.
- The 40-query fixture was actually executed and the raw expected/retrieved IDs saved in `benchmark-results.json`. Independent detector accuracy, temporal localization, and investigation-time reduction are not inferred from this run.

## Issues found and fixed

- The latest optional OpenCV major release removed the HOG API. A shared `<5` version constraint now prevents that conflict when model extras are installed.
- Multi-entity events initially selected an alphabetically first object. Event serialization now prioritizes the source observation's entity, making the person timeline and bounding-box label consistent.
- Baseline event scores and annotation labels remain separate: authored demo annotations do not receive made-up detector probabilities.

## Remaining verification boundaries

- PostgreSQL/pgvector Docker configuration is supplied; only SQLite was exercised in the local backend suite unless noted below.
- Optional OpenCLIP and EasyOCR adapters are implemented but were not executed in this environment.
- Synthetic retrieval labels are authored for the supported vocabulary and do not establish generalization. There is no held-out real surveillance benchmark.
- Deployment to a remote application host is not part of this local run. GitHub stores the reproducible source; uploaded/generated media and local credentials are excluded.
