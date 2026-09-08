# InstaDescribe Benchmarks

Tiny, readable evaluation benchmarks for InstaDescribe's video-intelligence
direction. One track exists today: **ID Event Light Bench v0**.

## Overview

InstaDescribe is being built toward answering practical questions about video
footage: did a described event happen? When exactly, and for how long? Later:
which entities appeared, what changed over time, and where the footage most
likely came from. Answering such questions credibly starts with measurement,
not features — before any capability is claimed, there has to be a way to
check it.

That is what this package is. So far it contains the measuring instrument (a
small benchmark schema and evaluator anyone can read end to end), two
reference systems to measure — a deliberately simple visual-change detector
and a CLIP-based solver that actually reads the query — and the tooling to
build a carefully controlled test dataset. The current stage is the
unglamorous one: filming, annotating, and reviewing 48 real video clips,
deliberately without letting any model near them, so that the first
comparison between the two systems means something.

From there the path is one capability at a time, each earning its place with
numbers on frozen data: event detection first; a semantic verification stage
next if the evidence shows it is needed; then, as separate benchmark tracks,
geolocation, retrieval, entity recognition, tracking and change detection.
Nothing is claimed until it is measured.

## Status: Data collection in progress

The benchmark/evaluator, the E001 naive visual baseline, the E002
retrieve-only CLIP solver, live image↔text inference verification, and the
dataset construction/freeze tooling are implemented. The rights-cleared
evaluation dataset is currently being collected.

Benchmark-quality results are intentionally not reported yet: the dataset
will be fully annotated, reviewed, split, and frozen before E001 or E002 is
run on candidate benchmark clips. The dataset is constructed before any
solver inspection to avoid benchmark leakage and post-hoc dataset selection;
E001/E002 evaluation begins only after `DATASET_FREEZE_V1`.

- [x] Benchmark schema + evaluator
- [x] E001 naive visual baseline
- [x] E002 retrieve-only CLIP solver
- [x] Live image↔text inference verification
- [x] Dataset collection/provenance/freeze tooling
- [ ] Rights-cleared 48-item dataset
- [ ] DATASET_FREEZE_V1
- [ ] Frozen E001 vs E002 evaluation
- [ ] E003 semantic verification

Next milestone: collect, annotate, review and freeze the 48-item dataset
(see `DATASET_COLLECTION.md`). After `DATASET_FREEZE_V1`: dev-only
configuration selection, one frozen E001/E002 test comparison, failure
analysis, and a GO/NO-GO decision for E003.

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
├── data/                 # placeholder sample manifest + predictions
├── baselines/            # E001 naive visual baseline (query-agnostic floor)
├── retrieve/             # E002 retrieve-only CLIP solver
└── dataset/              # model-blind dataset construction/freeze tooling
```

## Model artifacts

The E002 solver supports local CLIP inference via **user-supplied,
digest-pinned model artifacts** (an optional `clip` extra provides the
runtimes; the package core stays dependency-free). The repository ships no
model weights. The currently tested Xenova CLIP export is used for local
development/evaluation only; its redistribution/production licence status
remains unresolved in this project.

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

The E002 retrieve-only solver runs the same way via
`python -m instadescribe_benchmarks.id_event_light_v0.retrieve.solve` with a
solver TOML naming the digest-pinned model artifacts — setup and provenance
details in `experiments/E002_retrieve_solver.md`.

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

- Placeholder data; the target of 48 real, rights-cleared clips (16 positive /
  16 negative / 16 hard-negative) is being collected and is not yet frozen.
- Single-event-per-video assumption; no multi-instance events.
- No confidence-based scoring (thresholds, calibration) in v0.
- System metadata (`system` block) is submitter-reported and unverified.
