"""Local manual MEVA object-identity review UI; source frames only, no models."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.dataset import resolve
from meva.object_identity_benchmark import (
    clip_index,
    frame_record,
    load_annotations,
    save_annotations,
    validate_record,
)


def frame_jpeg(clip, frame):
    path = resolve(clip["source_video"])
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source video {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode source frame {frame}")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Cannot encode source frame")
        return encoded.tobytes()
    finally:
        cap.release()


def make_handler(clips, annotations_path):
    html = (ROOT / "scripts/object_identity_review.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, kind="application/json"):
            if isinstance(data, (dict, list)):
                data = json.dumps(data).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def requested_frame(self):
            query = parse_qs(urlparse(self.path).query)
            clip_id = query.get("clip", [None])[0]
            value = query.get("frame", [None])[0]
            if clip_id not in clips or value is None:
                raise ValueError("Unknown clip/frame")
            try:
                frame = int(value)
            except ValueError as exc:
                raise ValueError("Frame must be an integer") from exc
            clip = clips[clip_id]
            if frame not in clip["frame_indices"]:
                raise ValueError("Frame is outside clip")
            return clip, frame

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/":
                    self.reply(200, html, "text/html; charset=utf-8")
                elif route == "/api/clips":
                    records = load_annotations(annotations_path)
                    statuses = {clip_id: {
                        "saved": sum(key[0] == clip_id for key in records),
                        "complete": sum(key[0] == clip_id and row["review_state"] == "complete"
                                        for key, row in records.items()),
                    } for clip_id in clips}
                    identities = {clip_id: sorted({ann["object_id"] for (cid, _), row in records.items()
                                                   if cid == clip_id for ann in row["annotations"]})
                                  for clip_id in clips}
                    self.reply(200, {"clips": list(clips.values()), "statuses": statuses,
                                     "identities": identities})
                elif route == "/api/frame":
                    clip, frame = self.requested_frame()
                    records = load_annotations(annotations_path)
                    self.reply(200, records.get((clip["clip_id"], frame), frame_record(clip, frame)))
                elif route == "/frame.jpg":
                    clip, frame = self.requested_frame()
                    self.reply(200, frame_jpeg(clip, frame), "image/jpeg")
                else:
                    self.reply(404, {"error": "Not found"})
            except (ValueError, RuntimeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

        def do_PUT(self):
            if urlparse(self.path).path != "/api/frame":
                self.reply(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Invalid request length")
                row = json.loads(self.rfile.read(length))
                validate_record(row, clips)
                records = load_annotations(annotations_path)
                records[(row["clip_id"], row["frame_index"])] = row
                save_annotations(annotations_path, records)
                self.reply(200, {"saved": True, "clip_id": row["clip_id"],
                                 "frame_index": row["frame_index"]})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path,
                        default=ROOT / "datasets/meva/object-identity/clips.json")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "datasets/meva/object-identity/annotations.jsonl")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    _, clips = clip_index(args.clips)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(clips, args.annotations))
    print(f"Manual review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
