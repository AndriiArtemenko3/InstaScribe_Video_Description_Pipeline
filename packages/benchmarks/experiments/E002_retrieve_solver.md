# E002 — retrieve-only solver

Status: implemented — see `src/instadescribe_benchmarks/id_event_light_v0/retrieve/`.
No benchmark-quality claim exists for it yet, and the verification rung (E003)
has NOT started.

## Algorithm

For each manifest item, `f(V, q) -> (event_detected, start_time, end_time,
confidence)`:

- **S0 decode/sample** (`retrieve/sampling.py`): one hardened ffmpeg pass
  writes numbered JPEGs at `sampling_fps`; frame timestamps are
  sample-timeline times (`index / sampling_fps`), not source PTS. Duration
  comes from ffprobe. Fail-closed on missing tools, decode errors, numbering
  gaps, or unusable durations.
- **S1 paired embeddings** (`retrieve/clip_onnx.py`, `retrieve/tokenizer.py`,
  `retrieve/config.py`): local CLIP ONNX vision and text towers behind
  fail-closed adapters; both return **L2-normalized unit vectors**, so
  cross-modal similarity is exactly a dot product of unit vectors.
  `PairedEncoderProvenance` establishes the same-space assumption via shared
  model family + shared pinned export revision, verifying each file against
  its own sha256 (the two file digests are never compared to each other).
- **S2 score curve** (`retrieve/curve.py`): per-frame cosine → centered
  odd-width box mean (boundary windows average only available samples; no
  padding) → mean-subtracted contrast. Downstream consumes contrast only.
- **S3 proposal** (`retrieve/proposals.py`): duration-constrained maximum-sum
  subarray via the O(n) prefix-sum + monotonic-deque formulation (never
  unconstrained-then-clipped). Search objective = accumulated contrast;
  detection statistic = contrast margin (0.0 by convention for whole-video
  windows). Deterministic tie-break: sum, then earliest start, then shortest
  duration. Frozen interval mapping: `start = i_start/fps`,
  `end = min((i_end+1)/fps, duration)`. S3 always proposes; it never decides.
- **S5 decision/output** (`retrieve/solve.py`): `event_detected =
  contrast_margin >= tau_e002`; `confidence = sigmoid(alpha * margin)` — an
  uncalibrated positive-event score, never flipped on negatives; negatives
  carry null timestamps. Output is a benchmark predictions JSON with full
  config/provenance/diagnostics in the `system` block; timings are isolated
  so the predictions array is byte-stable.

No verification stage, no language model, no network access, no worker or
product wiring.

## Reused vs built

Reused: the benchmark's strict manifest/predictions loaders and schema; the
fail-closed adapter pattern (path/size/I-O-name validation, lazy 1-thread
sessions, `network_access -> False`) and the exact CLIP preprocessing recipe
from the worker's frame-embedding provider — **vendored, not imported**,
because the Apache benchmarks package has an enforced no-BUSL-import boundary
and an empty runtime dependency list; `sigmoid` from the E001 baseline.

Built new: text tower adapter + tokenizer adapter, paired provenance, box
smoothing, contrast curve, constrained max-subarray, interval mapping, the
decision rule, TOML config, and the CLI.

## Tokenizer / text-tower contract (resolved)

The tokenizer loads the export's own `tokenizer.json` via the `tokenizers`
runtime (optional `clip` extra) — the smallest maintainable choice versus
hand-rolling CLIP BPE. It asserts structure instead of assuming it: CLIP
special tokens must exist, truncation/padding are configured explicitly to a
77-id context padded with `<|endoftext|>`, and every encoding is checked for
exact length and BOS/EOS structure. The text ONNX contract is enforced
fail-closed as `input_ids` (+ optional `attention_mask`) -> `text_embeds`.
Verification against the real Xenova text export is still pending because the
artifact is not present locally (see below).

## Model artifacts

Nothing is downloaded automatically and no weights are committed. Runtimes
(`numpy`, `onnxruntime`, `pillow`, `tokenizers`) live in the optional `clip`
extra and are imported lazily at model-load time; the package core remains
dependency-free.

- Vision tower: `Xenova/clip-vit-base-patch32` revision `d15189d7…`,
  `onnx/vision_model.onnx`, sha256 `fd6e1402…` — present in the local HF
  cache; **live inference verified** (512-d, unit-normalized, deterministic).
- Text tower (`onnx/text_model.onnx`) and `tokenizer.json` from the same
  revision — **absent locally; live text inference NOT verified.** Manual
  fetch, matching existing repo practice:
  `huggingface-cli download Xenova/clip-vit-base-patch32 onnx/text_model.onnx tokenizer.json --revision d15189d7028b43f1d3e65039190477f6af591c2a`
  then pin its sha256 in the solver TOML. The export's licence remains
  unresolved (docs/investigation-architecture.md) and applies to both towers.

## Tests

46 new tests (118 total in the package, all passing; `make benchmarks-check`
green): unit-vector/cosine/smoothing/contrast math against hand-computed
values; frozen boundary and interval-mapping conventions; constrained-search
correctness incl. cases where the unconstrained optimum is illegal;
deterministic tie-breaks; whole-video degradation and its 0.0 margin; the
accumulated-sum vs margin distinction; provenance acceptance/rejection matrix
(family, revision, per-file digests; differing file digests explicitly OK);
provider I/O-contract validation with fake sessions; S5 threshold edges and
confidence orientation; ffmpeg-dependent sampling tests (skip when absent);
end-to-end `run()` with injected fake embedders whose output round-trips
through the strict predictions loader.

## Uncalibrated placeholders (NOT A BENCHMARK RESULT)

`sampling_fps=2.0` and `smoothing_width=21` are literature anchors;
`min/max_proposal_duration_s=1.0/30.0`, `tau_e002=0.0`, `alpha=1.0` are
arbitrary. None were tuned against any data; every predictions file carries
this statement in its `system` block. Honest selection happens only on a dev
split of real rights-cleared clips — which do not exist yet, so no
benchmark-quality numbers exist for E002.
