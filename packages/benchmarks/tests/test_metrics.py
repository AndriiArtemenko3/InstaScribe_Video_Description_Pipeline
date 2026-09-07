"""Metric formulas against hand-computed values."""

from __future__ import annotations

import pytest

from instadescribe_benchmarks.id_event_light_v0.metrics import (
    ClassificationCounts,
    accuracy,
    f1_score,
    mean,
    precision,
    recall,
    temporal_iou,
)


def counts(tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0) -> ClassificationCounts:
    return ClassificationCounts(
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn
    )


def test_accuracy_known_value() -> None:
    # (3 TP + 2 TN) / 8 total
    assert accuracy(counts(tp=3, fp=2, tn=2, fn=1)) == pytest.approx(0.625)


def test_accuracy_none_on_zero_total() -> None:
    assert accuracy(counts()) is None


def test_precision_recall_f1_known_values() -> None:
    c = counts(tp=3, fp=2, tn=2, fn=1)
    assert precision(c) == pytest.approx(3 / 5)
    assert recall(c) == pytest.approx(3 / 4)
    # harmonic mean: 2 * 0.6 * 0.75 / 1.35
    assert f1_score(c) == pytest.approx(2 * 0.6 * 0.75 / 1.35)


def test_precision_none_when_no_predicted_positives() -> None:
    assert precision(counts(tn=4, fn=2)) is None


def test_recall_none_when_no_actual_positives() -> None:
    assert recall(counts(fp=2, tn=4)) is None


def test_f1_none_when_component_undefined() -> None:
    assert f1_score(counts(tn=4, fn=2)) is None  # precision undefined
    assert f1_score(counts(fp=2, tn=4)) is None  # recall undefined


def test_f1_zero_when_precision_and_recall_zero() -> None:
    # predicted positives and actual positives both exist, but no overlap
    assert f1_score(counts(fp=3, fn=2)) == 0.0


def test_temporal_iou_perfect_overlap() -> None:
    assert temporal_iou(2.0, 5.0, 2.0, 5.0) == pytest.approx(1.0)


def test_temporal_iou_partial_overlap() -> None:
    # truth 10-14 (len 4), pred 12-18 (len 6): intersection 2, union 8
    assert temporal_iou(10.0, 14.0, 12.0, 18.0) == pytest.approx(0.25)


def test_temporal_iou_no_overlap() -> None:
    assert temporal_iou(2.0, 5.0, 6.0, 9.0) == 0.0


def test_temporal_iou_zero_length_prediction() -> None:
    # a point "interval" inside the truth window still localizes nothing
    assert temporal_iou(2.0, 5.0, 3.0, 3.0) == 0.0


def test_temporal_iou_rejects_zero_length_truth() -> None:
    with pytest.raises(ValueError, match="truth interval"):
        temporal_iou(3.0, 3.0, 2.0, 5.0)


def test_temporal_iou_rejects_reversed_prediction() -> None:
    with pytest.raises(ValueError, match="reversed"):
        temporal_iou(2.0, 5.0, 5.0, 2.0)


def test_mean_known_value() -> None:
    assert mean([1.0, 2.0, 6.0]) == pytest.approx(3.0)


def test_mean_none_on_empty() -> None:
    assert mean([]) is None
