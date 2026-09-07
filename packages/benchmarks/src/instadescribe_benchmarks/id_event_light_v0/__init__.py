"""ID Event Light Bench v0 — natural-language event detection and localization.

The CLI modules (``evaluator``, ``validate_manifest``) are deliberately not
imported here: importing them from the package ``__init__`` would make
``python -m instadescribe_benchmarks.id_event_light_v0.evaluator`` execute the
module twice (once via this import, once as ``__main__``). Import ``evaluate``
directly from ``.evaluator`` when using the library API.
"""

from .metrics import ClassificationCounts, temporal_iou
from .schema import (
    BenchmarkItem,
    Difficulty,
    ManifestError,
    Prediction,
    load_manifest,
    load_predictions,
)

__all__ = [
    "BenchmarkItem",
    "ClassificationCounts",
    "Difficulty",
    "ManifestError",
    "Prediction",
    "load_manifest",
    "load_predictions",
    "temporal_iou",
]
