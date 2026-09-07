# E001 — naive visual baseline (proposal)

Status: proposed, not scheduled.

## Goal

Produce the first real predictions file for ID Event Light Bench v0 using the
smallest sensible visual baseline, establishing the floor that every later
system must beat. The baseline is built from first principles — no pretrained
weights, no ML dependencies — so every number it produces is attributable to
visible arithmetic.

## The smallest sensible baseline

A query-agnostic motion/appearance-change detector. It cannot interpret the
natural-language query at all — which is exactly why it is the right floor: it
measures how far simple pixel statistics get before any semantics are needed.

Pipeline, deliberately one primitive operation per stage:

1. **Frame sampling.** Decode the video into frames at a fixed rate (e.g.
   2 fps), giving a T×H×W×3 array.
2. **Resizing.** Downsample every frame to a small fixed size (e.g. 64×64) so
   later stages are cheap and shape-stable.
3. **Normalization.** Convert to grayscale, scale values to [0, 1], subtract
   the per-video mean, so thresholds and similarity scores are comparable
   across videos.
4. **Convolution.** Slide one hand-written 3×3 kernel (e.g. a Sobel edge
   filter) over each frame to get an edge map — the same nine weights reused
   everywhere, with no learned parameters.
5. **Feature vectors.** Reduce each frame to a small vector (e.g. mean edge
   energy in a 4×4 grid of cells → 16 dimensions). Everything the vector does
   not encode is deliberately discarded.
6. **Change scoring.** Score change between consecutive frame vectors with a
   dot product (or cosine similarity).
7. **Confidence mapping.** Map the per-video peak change score s through
   sigma(s) = 1 / (1 + e^(-(s - b) / k)) to a confidence in (0, 1), with the
   bias b and scale k stated in the predictions file's `system` block.
8. **Detection threshold.** `event_detected = confidence >= t`, with t chosen
   by sweeping the precision/recall trade-off on the benchmark itself and
   reported alongside the results.
9. **Localization.** Report the contiguous window where the change score
   stays above threshold as (start_time, end_time).

## Why this is the right floor

The hard-negative bucket should demolish this baseline — a door opening moves
pixels just like entering a vehicle does. That failure is the point: it makes
the case for semantics (pretrained embeddings, temporal models, VLMs in later
baselines) concrete and measurable rather than assumed, and it exercises every
section of the evaluator's report, including the false-positive rate that the
hard negatives exist to expose.

## Explicit non-goals

- No training, no learned weights, no downloads of models or datasets.
- No new dependencies decided here — even NumPy vs pure Python is an
  implementation-time decision.
- No claim that the resulting numbers mean anything beyond placeholder data
  until real, rights-cleared clips exist in the manifest.
