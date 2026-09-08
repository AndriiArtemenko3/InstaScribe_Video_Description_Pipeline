"""Model-blind dataset-construction tooling for ID Event Light Bench v0.

Everything here operates on media files, human-authored metadata, hashes and
split rules. Nothing in this package may invoke a solver, an embedding model,
or any inference — candidate clips are never scored during collection. The
only benchmark import is the manifest schema (the contract the dataset must
ultimately satisfy).

The CLI module (``cli``) is deliberately not imported here.
"""
