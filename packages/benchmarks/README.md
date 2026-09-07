# InstaDescribe Benchmarks

Tiny, readable evaluation benchmarks for InstaDescribe's video-intelligence
direction. One track exists today: **ID Event Light Bench v0**.

## Purpose

ID Event Light Bench v0 evaluates whether a system can detect and temporally
localize natural-language-specified events in short videos. Each benchmark item
is conceptually:

```
f(V, q) -> (y, t_s, t_e, c)
```

where `V` is a video, `q` a natural-language event query ("Does a person enter
the vehicle?"), `y` whether the event occurred, `t_s`/`t_e` the event interval
when applicable, and `c` the system's confidence.

## Why it exists

The long-term InstaDescribe system should not be evaluated only by subjective
natural-language output. Individual video-intelligence capabilities need
reproducible, inspectable measurements — this package is the first such
evaluation surface, small enough to read end to end.

## What this benchmark is NOT

It is not yet:

- a large public benchmark
- a scientific claim of general video understanding
- a substitute for ActEV/VIRAT/MEVA
- a geolocation benchmark
- a retrieval benchmark

The shipped `sample_manifest.json` contains **placeholder records only** (no
real videos). No accuracy, calibration, latency or memory result derived from
placeholder data may be reported anywhere.

## Layout

```
src/instadescribe_benchmarks/id_event_light_v0/
├── schema.py             # manifest + prediction contracts, fail-closed validation
├── metrics.py            # accuracy/precision/recall/F1, temporal IoU — formulas visible
├── evaluator.py          # scoring rules + CLI
├── validate_manifest.py  # manifest validation CLI
└── data/                 # placeholder sample manifest + predictions
```

## How to run

From this directory (or via the repo-root `make benchmarks-*` targets):

```bash
# validate the sample manifest (placeholder video paths => skip existence check)
uv run --locked --extra dev python -m instadescribe_benchmarks.id_event_light_v0.validate_manifest \
    --manifest src/instadescribe_benchmarks/id_event_light_v0/data/sample_manifest.json \
    --skip-video-check

# score the sample predictions
uv run --locked --extra dev python -m instadescribe_benchmarks.id_event_light_v0.evaluator \
    --manifest src/instadescribe_benchmarks/id_event_light_v0/data/sample_manifest.json \
    --predictions src/instadescribe_benchmarks/id_event_light_v0/data/sample_predictions.json

# tests
uv run --locked --extra dev pytest tests -q
```

Both CLIs accept `--json` for machine-readable output.

## Baselines

`baselines/naive_visual.py` is the first complete solver: a deliberately
primitive, query-agnostic appearance-change detector (ffmpeg grayscale
sampling → hand-written Sobel edges → grid features → cosine change →
sigmoid → threshold). It establishes the floor every later system must beat.
Its thresholds are uncalibrated placeholders and no result produced with them
is reported anywhere.

```bash
uv run --locked --extra dev python -m instadescribe_benchmarks.id_event_light_v0.baselines.naive_visual \
    --manifest <manifest.json> --out predictions.json
```

The manifest's video files must exist (the solver reads pixels); the output is
scored with the evaluator above.

## Prediction format

The evaluator is model-agnostic: it scores a JSON file, never runs a model. A
hand-written heuristic, a CNN, a tracker, a VLM or the full InstaDescribe
pipeline are all scored identically:

```json
{
  "schema_version": "id-event-light-bench-predictions/0",
  "system": {"name": "optional free-form metadata, reported but never scored"},
  "predictions": [
    {"id": "pos-001", "event_detected": true, "start_time": 12.8, "end_time": 15.7, "confidence": 0.87}
  ]
}
```

Every manifest item needs exactly one prediction; missing or unknown ids fail
the run. `confidence` is validated but unused in v0.

## Scoring rules in one paragraph

Classification (accuracy, precision, recall, F1) is driven by `event_detected`
vs `label` alone. Temporal metrics cover true positives only: a prediction
with both timestamps present and `start <= end` is "localized" and gets a real
IoU plus absolute start/end errors; a missing or reversed interval scores IoU
0.0 and is excluded from the error means. Mean IoU averages over **all** true
positives, so omitting timestamps never helps. Zero denominators yield `n/a`,
never a fake 0.0. Per-difficulty buckets (positive / negative / hard_negative)
report counts and, for the negative buckets, the false-positive rate. The full
rationale lives in `docs/benchmark-decisions.md` at the repo root and in the
module docstrings.

## Long-term direction

Future tracks may test geolocation (ID-GEO), retrieval (ID-RETRIEVAL), entity
recognition (ID-ENTITY), tracking (ID-TRACK) and change detection (ID-CHANGE).
Each would be an independently scored sibling submodule with its own manifest,
metrics and CLI — never folded into one composite score.

## Honest limits

- Placeholder data; the target of ~24 real, rights-cleared clips (8 positive /
  8 negative / 8 hard-negative) is not yet collected.
- Single-event-per-video assumption; no multi-instance events.
- No confidence-based scoring (thresholds, calibration) in v0.
- System metadata (`system` block) is submitter-reported and unverified.
