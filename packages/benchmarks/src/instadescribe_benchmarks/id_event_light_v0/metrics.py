"""Metric formulas for ID Event Light Bench v0, written out in full.

Every function here is pure: counts or numbers in, one number (or ``None``)
out. The zero-denominator policy is uniform and deliberate:

    An undefined metric returns ``None`` (rendered as "n/a"), never 0.0 and
    never an exception. 0.0 would fake a terrible score where no score
    exists; raising would kill an otherwise valid report half-way through.

Empty buckets are legitimate benchmark states (e.g. a manifest with no
negatives yet), so they must not crash the evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationCounts:
    """Binary confusion-matrix counts for event detection.

    positive class = "the event occurred" (label ``true``).
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )


def accuracy(counts: ClassificationCounts) -> float | None:
    """(TP + TN) / (TP + FP + TN + FN); None on an empty count set."""

    if counts.total == 0:
        return None
    return (counts.true_positives + counts.true_negatives) / counts.total


def precision(counts: ClassificationCounts) -> float | None:
    """TP / (TP + FP); None when the system predicted no positives at all."""

    predicted_positives = counts.true_positives + counts.false_positives
    if predicted_positives == 0:
        return None
    return counts.true_positives / predicted_positives


def recall(counts: ClassificationCounts) -> float | None:
    """TP / (TP + FN); None when the benchmark contains no actual positives."""

    actual_positives = counts.true_positives + counts.false_negatives
    if actual_positives == 0:
        return None
    return counts.true_positives / actual_positives


def f1_score(counts: ClassificationCounts) -> float | None:
    """2 * P * R / (P + R), the harmonic mean of precision and recall.

    None when either component is undefined; 0.0 when both are defined but
    zero (the harmonic mean of two zeros is zero, not undefined).
    """

    p = precision(counts)
    r = recall(counts)
    if p is None or r is None:
        return None
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def temporal_iou(
    truth_start: float,
    truth_end: float,
    predicted_start: float,
    predicted_end: float,
) -> float:
    """Intersection over union of two time intervals, in [0, 1].

        intersection = max(0, min(ends) - max(starts))
        union        = truth_length + predicted_length - intersection
        IoU          = intersection / union

    The manifest guarantees truth_start < truth_end, and the evaluator never
    passes a reversed predicted interval; both guards below exist so the
    function stays honest for direct callers. A zero-length *predicted*
    interval is accepted and yields 0.0 naturally (intersection length 0,
    union > 0 because the truth interval has positive length).
    """

    if truth_end <= truth_start:
        raise ValueError("truth interval must have positive length")
    if predicted_end < predicted_start:
        raise ValueError("predicted interval must not be reversed")

    intersection = max(0.0, min(truth_end, predicted_end) - max(truth_start, predicted_start))
    union = (truth_end - truth_start) + (predicted_end - predicted_start) - intersection
    return intersection / union


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean; None on an empty sequence (zero denominator)."""

    if not values:
        return None
    return sum(values) / len(values)
