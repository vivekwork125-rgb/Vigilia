# Limitations and unverified paths

- **Behavior:** Placement, pickup, and unattended rules use box motion, proximity, and sustained state changes. The records are INFERRED hypotheses, not proof of carrying, ownership, intent, or criminal conduct. Eight real MEVA videos have now been run against independent labels; action detection is weak and pickup/placement examples are missed. Vehicle entry/exit is not inferred; disappearance near a car could be occlusion.
- **Perception:** HOG is people-only and weak under low light/occlusion. The MEVA benchmark uses YOLO11n + ByteTrack at two samples/second and size 640. Small objects, occlusion, detection gaps, fragmented tracks and source-content anomalies limit performance. No verified tracking-identity accuracy is claimed.
- **Cross-camera:** The association route compares coarse HSV crop appearance, color, camera connectivity, and travel-time compatibility. Similar people can remain ambiguous; there is no continuous tracking or verified identity. The configured travel bounds are illustrative.
- **Language and image retrieval:** TF-IDF with a small vocabulary is the verified text baseline. HSV reference-image search is sensitive to crop/background. OpenCLIP package/model loading and text-to-image retrieval were not executed; `open_clip` is absent in this environment.
- **OCR:** The optional EasyOCR path exists but `easyocr` is absent here, so OCR ingestion and text search are unverified. No OCR capability is claimed for this run.
- **Evaluation:** Forty development queries point to authored synthetic event IDs; the UI lab remains a development fixture. The separate real MEVA evaluator compares independent camera/time/type/class/spatial labels and reports errors. Approximate appearance/disappearance proxies have separate scores. Unsupported actions receive no accuracy credit; approached-object/unattended relevance is unmeasured. No actual human timings exist; time reduction is unmeasured.
- **Deployment:** SQLite and localhost were exercised. Docker/pgvector could not be started because the local Docker API socket does not exist. Production operation, resilience, privacy controls, user roles, migrations, audit logs, and retention remain outside verified scope.
- **Negative search:** A missing match means no indexed candidate satisfied current filters/thresholds over known footage. Camera operational status is unknown, and gaps or missed detections prevent an absolute absence claim.

See [FINAL_VERIFICATION.md](FINAL_VERIFICATION.md) for per-capability status.

That earlier report predates the MEVA integration. See [MEVA_REPORT.md](MEVA_REPORT.md) for current real-data results and remaining failures.
