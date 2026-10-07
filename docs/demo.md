# Six-minute demonstration

1. Start with `./scripts/dev.sh`, then open http://localhost:3100. First launch creates three 40-second synthetic MP4s; no GPU or external service is required.
2. In Command Center, choose the object-left-at-entrance suggestion.
3. Open the strongest result: `EVT-E03`. The player seeks to 12 seconds in `CAM-01`, recording time 18:04:12, frames 120–150.
4. Inspect the source hash, annotation method, and OBSERVED label. The scene and observations are explicitly synthetic.
5. Select `P-E01` in the timeline, click the prior carried-backpack event, then return to the placement event. Use “Immediately before” for temporal context.
6. Inspect cross-camera candidates. Similar red-clothed entities in the west plaza are ambiguous. The time gap is visible; no continuous identity is claimed.
7. Collect relevant events into an investigation. Add the unattended-bag hypothesis to contrast OBSERVED with INFERRED. Export the Markdown report and inspect evidence IDs, frame ranges, source hashes, and unresolved ambiguity.
8. Open the graph and click an event node, then open its evidence.
9. Open Evaluation Lab and run the 40-query fixture. Explain that this tests retrieval from annotations, not detector accuracy. Precision@5 can be lower because it divides by five even when there is only one relevant event.
10. Upload a short consented clip via Footage Library with the correct recording start. Observe progress and inspect the resulting detections/events. CPU HOG detects only people; use YOLO + ByteTrack for the full object vocabulary.

For an honest timing comparison, measure the same task manually and using the application, save both actual elapsed times under the same task name, then rerun evaluation. Do not use synthetic timings in a judging claim.
