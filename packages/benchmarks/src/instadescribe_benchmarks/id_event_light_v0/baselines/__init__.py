"""Baseline solvers for ID Event Light Bench v0.

Each baseline is a standalone submitter: it reads a benchmark manifest, runs
its own inference, and writes a predictions JSON that the evaluator scores.
Baselines never share scoring logic with the evaluator.

The CLI modules are deliberately not imported here (importing them from the
package ``__init__`` would make ``python -m`` execute them twice).
"""
