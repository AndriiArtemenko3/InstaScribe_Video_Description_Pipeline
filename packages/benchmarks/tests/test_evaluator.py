"""Scoring rules of the evaluator, one behavior per test."""

from __future__ import annotations

import json

import pytest
from conftest import item, prediction

from instadescribe_benchmarks.id_event_light_v0.evaluator import (
    evaluate,
    render_report,
    result_to_dict,
)
from instadescribe_benchmarks.id_event_light_v0.schema import Difficulty


def test_perfect_predictions_full_marks() -> None:
    items = [item("p1"), item("n1", difficulty=Difficulty.NEGATIVE)]
    predictions = [prediction("p1"), prediction("n1", detected=False, start=None, end=None)]
    result = evaluate(items, predictions)
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.mean_iou == pytest.approx(1.0)
    assert result.mean_start_error == 0.0
    assert result.mean_end_error == 0.0


def test_false_negative_counted() -> None:
    result = evaluate([item("p1")], [prediction("p1", detected=False)])
    assert result.counts.false_negatives == 1
    # a missed event never enters the temporal set
    assert result.n_true_positives == 0
    assert result.mean_iou is None


def test_negatives_score_true_negative() -> None:
    result = evaluate(
        [item("n1", difficulty=Difficulty.NEGATIVE)],
        [prediction("n1", detected=False, start=None, end=None)],
    )
    assert result.counts.true_negatives == 1


def test_hard_negative_false_positive_rate() -> None:
    items = [
        item("h1", difficulty=Difficulty.HARD_NEGATIVE),
        item("h2", difficulty=Difficulty.HARD_NEGATIVE),
    ]
    predictions = [prediction("h1"), prediction("h2", detected=False, start=None, end=None)]
    result = evaluate(items, predictions)
    bucket = next(b for b in result.buckets if b.difficulty is Difficulty.HARD_NEGATIVE)
    assert (bucket.n, bucket.correct, bucket.incorrect) == (2, 1, 1)
    assert bucket.false_positive_rate == pytest.approx(0.5)


def test_empty_bucket_reports_none_fp_rate() -> None:
    result = evaluate([item("p1")], [prediction("p1")])
    negative = next(b for b in result.buckets if b.difficulty is Difficulty.NEGATIVE)
    assert (negative.n, negative.false_positive_rate) == (0, None)
    positive = next(b for b in result.buckets if b.difficulty is Difficulty.POSITIVE)
    assert positive.false_positive_rate is None  # never defined for positives


def test_temporal_only_over_true_positives() -> None:
    # a false positive's timestamps must not leak into temporal metrics
    items = [item("p1"), item("n1", difficulty=Difficulty.NEGATIVE)]
    predictions = [prediction("p1"), prediction("n1", detected=True, start=0.0, end=99.0)]
    result = evaluate(items, predictions)
    assert set(result.iou_by_id) == {"p1"}


def test_detected_positive_missing_timestamps_scores_zero_iou() -> None:
    result = evaluate([item("p1")], [prediction("p1", start=None, end=None)])
    assert result.iou_by_id["p1"] == 0.0
    assert (result.n_true_positives, result.n_localized) == (1, 0)
    # excluded from the error means: no usable endpoints to measure against
    assert result.mean_start_error is None
    assert result.mean_end_error is None


def test_malformed_predicted_interval_scores_zero_iou() -> None:
    result = evaluate([item("p1")], [prediction("p1", start=9.0, end=1.0)])
    assert result.iou_by_id["p1"] == 0.0
    assert result.n_localized == 0
    assert result.mean_start_error is None
    # still a true positive for classification
    assert result.counts.true_positives == 1


def test_zero_length_predicted_interval_is_localized() -> None:
    # item gt is 2.0-5.0; a point prediction at 3.0 has IoU 0 but real errors
    result = evaluate([item("p1")], [prediction("p1", start=3.0, end=3.0)])
    assert result.iou_by_id["p1"] == 0.0
    assert result.n_localized == 1
    assert result.mean_start_error == pytest.approx(1.0)
    assert result.mean_end_error == pytest.approx(2.0)


def test_mean_iou_averages_over_all_true_positives() -> None:
    # one perfect localization + one omitted interval: the omission must drag
    # the mean to 0.5 rather than being silently skipped
    items = [item("p1"), item("p2", start=10.0, end=14.0)]
    predictions = [
        prediction("p1", start=2.0, end=5.0),
        prediction("p2", start=None, end=None),
    ]
    result = evaluate(items, predictions)
    assert result.mean_iou == pytest.approx(0.5)


def test_missing_prediction_raises() -> None:
    with pytest.raises(ValueError, match="missing predictions.*p1"):
        evaluate([item("p1")], [])


def test_extra_prediction_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown benchmark items.*typo"):
        evaluate([item("p1")], [prediction("p1"), prediction("typo")])


def test_empty_benchmark_raises() -> None:
    with pytest.raises(ValueError, match="at least one benchmark item"):
        evaluate([], [])


def test_no_true_positives_temporal_metrics_none() -> None:
    result = evaluate(
        [item("n1", difficulty=Difficulty.NEGATIVE)],
        [prediction("n1", detected=False, start=None, end=None)],
    )
    assert result.mean_iou is None
    assert result.mean_start_error is None
    assert result.mean_end_error is None


def test_system_slot_passed_through() -> None:
    result = evaluate([item("p1")], [prediction("p1")], system={"model": "demo", "latency_s": 1.2})
    assert result.system == {"model": "demo", "latency_s": 1.2}


def test_render_report_contains_all_sections() -> None:
    items = [item("p1"), item("h1", difficulty=Difficulty.HARD_NEGATIVE)]
    predictions = [prediction("p1"), prediction("h1")]
    report = render_report(evaluate(items, predictions, system={"model": "demo"}))
    for fragment in (
        "ID Event Light Bench v0",
        "Samples: 2",
        "Accuracy:",
        "F1:",
        "Temporal IoU:",
        "Mean start error:",
        "Hard negatives:",
        "False positive rate:",
        "System (reported by submitter, not verified):",
    ):
        assert fragment in report


def test_result_to_dict_is_json_serializable() -> None:
    result = evaluate([item("p1")], [prediction("p1", start=None, end=None)])
    payload = json.loads(json.dumps(result_to_dict(result)))
    assert payload["samples"] == 1
    assert payload["temporal"]["mean_start_error"] is None  # None survives as null
