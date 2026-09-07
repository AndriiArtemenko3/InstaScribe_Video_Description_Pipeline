"""E002 retrieve-only solver for ID Event Light Bench v0.

Stages S0 (decode/sample), S1 (paired CLIP image/text embeddings),
S2 (score curve + smoothing + contrast), S3 (duration-constrained temporal
proposal) and S5 (retrieve-only detection/output). No verification stage, no
language model, no network access — a standalone offline benchmark submitter
that writes predictions JSON only.

The CLI module (``solve``) is deliberately not imported here.
"""
