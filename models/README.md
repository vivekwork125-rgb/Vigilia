# Model adapters

- **CPU default:** OpenCV's pretrained HOG person detector plus a lightweight IoU tracker. No network download. People only; sparse sampling and occlusion may fragment tracks. SVM margins are not probabilities.
- **Recommended perception:** `PERCEPTION_BACKEND=yolo`, using Ultralytics YOLO11n + ByteTrack. Install `backend/requirements-models.txt`. Weights download on the first real indexing job; set `YOLO_MODEL` to a local path for offline operation. Review Ultralytics AGPL/commercial licensing before distribution.
- **Vision-language embeddings:** `ENABLE_CLIP=true` enables OpenCLIP ViT-B/32 (LAION-2B checkpoint). One largest visible crop per track is embedded and persisted; text/crop cosine participates in retrieval for indexed uploads. Demo annotations use the lightweight retrieval baseline.
- **Appearance baseline:** actual normalized HSV histograms (48 values), never an identity embedding. This drives reference-image search and conservative cross-camera candidates.
- **OCR:** `ENABLE_OCR=true` enables EasyOCR (English). OCR is labeled inferred, retains source frame, and requires manual verification. No dedicated plate detector or plate normalization is provided.

Full models are optional and are not silently loaded in CPU/demo mode. Failures are shown in the ingestion UI. No face recognition or real-person identification is implemented.
