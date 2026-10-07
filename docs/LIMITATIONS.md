# Limitations and unverified paths

- **Behavior:** Placement, pickup, and unattended rules use box motion, proximity, and sustained state changes. The produced records are INFERRED hypotheses, not proof of carrying, ownership, intent, or criminal conduct. An independently labeled field-video set has not been run. Vehicle entry/exit is not inferred; disappearance near a car could be occlusion.
- **Perception:** HOG is people-only and weak under low light/occlusion. YOLO + ByteTrack ran previously on a repeated still-image smoke clip, which checks integration only. It has not been evaluated here for moving objects, detector misses, track swaps, or action accuracy.
- **Cross-camera:** The association route compares coarse HSV crop appearance, color, camera connectivity, and travel-time compatibility. Similar people can remain ambiguous; there is no continuous tracking or verified identity. The configured travel bounds are illustrative.
- **Language and image retrieval:** TF-IDF with a small vocabulary is the verified text baseline. HSV reference-image search is sensitive to crop/background. OpenCLIP package/model loading and text-to-image retrieval were not executed; `open_clip` is absent in this environment.
- **OCR:** The optional EasyOCR path exists but `easyocr` is absent here, so OCR ingestion and text search are unverified. No OCR capability is claimed for this run.
- **Evaluation:** Forty development queries point to authored synthetic event IDs. This checks parser/ranking regressions, not natural-video action accuracy. Independent footage labels and 30–50 representative queries have not been supplied. The independent evaluator and paired time-study recorder are ready, but no actual human timings exist; time reduction is unmeasured.
- **Deployment:** SQLite and localhost were exercised. Docker/pgvector could not be started because the local Docker API socket does not exist. Production operation, resilience, privacy controls, user roles, migrations, audit logs, and retention remain outside verified scope.
- **Negative search:** A missing match means no indexed candidate satisfied current filters/thresholds over known footage. Camera operational status is unknown, and gaps or missed detections prevent an absolute absence claim.

See [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md) for per-capability status.
