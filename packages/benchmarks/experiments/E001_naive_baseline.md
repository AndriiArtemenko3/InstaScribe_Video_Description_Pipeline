# E001 — naive visual baseline (proposal only, nothing implemented)

Status: proposed. Do not implement without explicit further instruction.

## Goal

Produce the first real predictions file for ID Event Light Bench v0 using the
smallest sensible visual baseline (the future "B0" rung), so that every later
system has a floor to beat — and so the underlying mathematics is learned by
building it, not by importing it.

## The smallest sensible baseline

A query-agnostic motion/appearance-change detector. It cannot understand the
natural-language query at all — which is exactly why it is the right floor: it
measures how far simple pixel statistics get before any semantics are needed.

Pipeline, one stage per concept:

1. **Frames as tensors.** Decode the video into frames sampled at a fixed rate
   (e.g. 2 fps). Each frame is an H×W×3 array of numbers — a tensor. Concept:
   images are just arrays; time adds one more axis (T×H×W×3).
2. **Resizing.** Downsample every frame to a small fixed size (e.g. 64×64).
   Concept: resolution vs information; why models fix their input shape.
3. **Normalization.** Convert to grayscale, scale values to [0, 1], subtract
   the per-video mean. Concept: putting inputs on a common scale so that
   thresholds and dot products mean the same thing across videos.
4. **Convolution.** Slide one hand-written 3×3 kernel (e.g. a Sobel edge
   filter) over each frame to get an edge map. Concept: convolution as a
   local weighted sum; the same nine weights reused everywhere — the seed
   idea behind every CNN.
5. **Feature vectors.** Reduce each frame to a small vector (e.g. mean edge
   energy in a 4×4 grid of cells → a 16-dimensional vector). Concept: a
   feature vector is a deliberate summary; everything not encoded is lost.
6. **Dot products.** Score change between consecutive frame vectors with a
   dot product (or cosine similarity). Concept: dot product as similarity;
   the geometry behind embeddings and retrieval.
7. **Sigmoid probabilities.** Map the per-video peak change score s through
   sigma(s) = 1 / (1 + e^(-(s - b) / k)) to get a confidence in (0, 1).
   Concept: squashing an unbounded score into a probability-shaped number,
   and why b (bias) and k (scale) matter.
8. **Classification threshold.** `event_detected = confidence >= t`. Sweep t
   and watch precision and recall move against each other. Concept: the
   precision/recall trade-off is a choice, not a property of the model.
9. **Localization.** Report the contiguous window where the change score
   stays above threshold as (start_time, end_time). Concept: temporal IoU
   punishes both late starts and greedy windows.

## What it will teach

Each benchmark metric becomes an experience rather than a formula: the
hard-negative bucket should demolish this baseline (a door opening moves
pixels just like entering a vehicle does), making the case for semantics
(later rungs: pretrained embeddings, temporal models, VLMs) concrete and
measurable.

## Explicit non-goals

- No training, no learned weights, no downloads of models or datasets.
- No new dependencies decided here — even NumPy vs pure Python is a decision
  to make at implementation time.
- No claim that the resulting numbers mean anything beyond placeholder data
  until real, rights-cleared clips exist in the manifest.
