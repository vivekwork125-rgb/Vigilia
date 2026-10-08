# Evaluation protocol

The in-app lab executes 40 authored queries against authored events in the original three-camera synthetic fixture. It reports P@1, P@5, R@5, MRR, mAP@5, negative false-positive rate, and latency for **retrieval from those annotations**. It does not measure model perception, temporal localization, cross-camera identity, or field accuracy. The pipeline-generated demo is excluded from that legacy fixture benchmark.

## Independent retrieval study

1. Obtain consented recordings from multiple cameras with varied lighting, occlusion, lookalikes, camera gaps, and true negative cases. Keep originals and source hashes; record camera start times and connectivity separately. Do not use synthetic event rows as ground truth.
2. Have a reviewer watch the source video and write **30–50 or more** investigation queries and labels before inspecting VIGILIA results. Include positive and negative queries. A second reviewer should adjudicate ambiguous intervals. Record the annotation protocol and disagreements.
3. Store labels in CSV with this exact header: `query_id,query,camera_id,video_id,event_type,start_seconds,end_seconds,tolerance_seconds,entity_id,negative,case`. Repeat `query_id` for multiple relevant intervals. For a negative row, set `negative=true` and leave event fields empty. `start_seconds`/`end_seconds` are offsets in the source video; `tolerance_seconds` is an acceptable annotation error chosen **before** evaluation. Use the same query text on each repeated row. `case` records hard-case strata such as `occlusion`, `poor-light`, `similar-people`, `camera-gap`, or `missed-detection`.
4. Index the videos with the documented detector and sampling settings. Run `./scripts/evaluate_independent.py labels.csv --output results.json`. It refuses fewer than 30 distinct queries. `--allow-small-fixture` exists only for a pipeline smoke check and invalidates a representative benchmark claim.
5. Inspect each error and report dataset size, camera distribution, source hashes, model/settings, annotation tolerance, missing detections, and uncertainty. The tool matches **video, event type, optional entity, and independently labeled interval**, never generated event IDs. It calculates P@1, P@5, R@5, MRR, negative false-positive rate, and temporal IoU/start/end error for matched hits. A null metric means there were no eligible measured cases; do not fill it with an estimate.

## Paired investigation-time study

Define a task with a known answer and source interval. For each participant/task/trial, counterbalance manual-first and assisted-first ordering to reduce practice effects. Start a timer at task presentation. Record elapsed seconds to first relevant evidence, correct evidence, and completion. The participant should not know the answer in advance. Preserve mistakes and censored/not-found trials in study notes; do not silently discard them.

Record actual measurements after each run:

```bash
./scripts/timing_study.py record --task case-01 --participant P01 --trial 1 --mode manual --first-relevant 120 --correct 180 --total 240
./scripts/timing_study.py record --task case-01 --participant P01 --trial 1 --mode assisted --first-relevant 80 --correct 130 --total 190
./scripts/timing_study.py summarize
```

The numbers above illustrate command syntax only; they are **not** study data and are not stored in the repository. The recorder writes ignored `data/timing-study.csv`. Summary metrics remain null until a matching manual/assisted pair exists. Report the number of participants/tasks, per-condition times, paired reduction, distribution/uncertainty, and task failures. The in-app paired timing field currently records a single total per mode; the CLI captures the three required milestones.

Current state: **zero independent real-footage labels and zero human timing pairs have been collected for this completion audit**. No evidence supports a “significantly faster” claim yet.
# Real MEVA evaluation

The separate [MEVA protocol](MEVA.md) evaluates eight real sources against external KPF annotations without injecting ground truth into runtime. It records the original baseline, one-to-one camera/time/type/class/spatial matches, direct and approximate metrics separately, unsupported activity counts, source evidence validation, query checks, performance and every FP/FN. See [measured results](MEVA_REPORT.md). The development-fixture results below retain their original scope.
