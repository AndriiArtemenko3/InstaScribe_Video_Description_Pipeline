"""Score a predictions file against an ID Event Light Bench v0 manifest.

Scoring rules (each is a deliberate decision, see docs/benchmark-decisions.md):

1. Every manifest item must have exactly one prediction. A missing id is an
   error, not an implicit "no event" — "the system said nothing" and "the
   system said no" are different claims. Extra ids are errors too (almost
   always a typo that would otherwise drop a real prediction).
2. Classification is driven by ``event_detected`` alone. A detected positive
   with a garbage interval is still a true positive for classification; the
   localization failure shows up in the temporal block. That separation is
   the point of scoring detection and localization independently.
3. Temporal metrics cover true positives only (label true AND detected
   true). Within those, an item is "localized" when both predicted
   timestamps are present and start <= end:
   - localized      -> real IoU, and |start error| / |end error| enter the
                       error means (a zero-length interval counts: its IoU
                       is 0.0 but its endpoint errors are real numbers);
   - not localized  -> IoU 0.0 (a non-interval localizes nothing) and
                       EXCLUDED from the error means (absolute errors
                       against absent or reversed endpoints are noise).
   Mean IoU averages over ALL true positives, so omitting timestamps can
   never improve the headline localization number.
4. Negative items and false negatives get no temporal scoring; any predicted
   timestamps on them are ignored.

Usage:
    python -m instadescribe_benchmarks.id_event_light_v0.evaluator \
        --manifest .../sample_manifest.json \
        --predictions .../sample_predictions.json [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .metrics import (
    ClassificationCounts,
    accuracy,
    f1_score,
    mean,
    precision,
    recall,
    temporal_iou,
)
from .schema import (
    BenchmarkItem,
    Difficulty,
    ManifestError,
    Prediction,
    load_manifest,
    load_predictions,
)


@dataclass(frozen=True, slots=True)
class BucketBreakdown:
    """Per-difficulty result: how many items, how many judged correctly.

    ``false_positive_rate`` is incorrect/n for the two negative buckets
    (an incorrect negative IS a false positive) and None for the positive
    bucket and for empty buckets.
    """

    difficulty: Difficulty
    n: int
    correct: int
    incorrect: int
    false_positive_rate: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    counts: ClassificationCounts
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    n_true_positives: int
    n_localized: int
    mean_iou: float | None
    mean_start_error: float | None
    mean_end_error: float | None
    iou_by_id: Mapping[str, float]
    buckets: tuple[BucketBreakdown, ...]
    system: Mapping[str, object] | None


def _is_localized(prediction: Prediction) -> bool:
    """True when the prediction carries a usable interval (start <= end)."""

    return (
        prediction.start_time is not None
        and prediction.end_time is not None
        and prediction.start_time <= prediction.end_time
    )


def evaluate(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    *,
    system: Mapping[str, object] | None = None,
) -> EvaluationResult:
    if not items:
        raise ValueError("at least one benchmark item is required")

    by_id = {prediction.item_id: prediction for prediction in predictions}
    manifest_ids = {item.item_id for item in items}
    missing = sorted(manifest_ids - by_id.keys())
    if missing:
        raise ValueError(f"missing predictions for benchmark items: {missing}")
    extra = sorted(by_id.keys() - manifest_ids)
    if extra:
        raise ValueError(f"predictions for unknown benchmark items: {extra}")

    # --- classification: event_detected vs label, nothing else -------------
    tp = fp = tn = fn = 0
    for item in items:
        detected = by_id[item.item_id].event_detected
        if item.label and detected:
            tp += 1
        elif item.label and not detected:
            fn += 1
        elif not item.label and detected:
            fp += 1
        else:
            tn += 1
    counts = ClassificationCounts(
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn
    )

    # --- temporal localization: true positives only ------------------------
    iou_by_id: dict[str, float] = {}
    start_errors: list[float] = []
    end_errors: list[float] = []
    n_localized = 0
    for item in items:
        prediction = by_id[item.item_id]
        if not (item.label and prediction.event_detected):
            continue
        assert item.start_time is not None and item.end_time is not None
        if _is_localized(prediction):
            assert prediction.start_time is not None and prediction.end_time is not None
            n_localized += 1
            iou_by_id[item.item_id] = temporal_iou(
                item.start_time, item.end_time, prediction.start_time, prediction.end_time
            )
            start_errors.append(abs(prediction.start_time - item.start_time))
            end_errors.append(abs(prediction.end_time - item.end_time))
        else:
            iou_by_id[item.item_id] = 0.0

    # --- per-difficulty breakdown ------------------------------------------
    buckets: list[BucketBreakdown] = []
    for difficulty in Difficulty:
        bucket_items = [item for item in items if item.difficulty is difficulty]
        correct = sum(
            1 for item in bucket_items if by_id[item.item_id].event_detected == item.label
        )
        incorrect = len(bucket_items) - correct
        if difficulty is Difficulty.POSITIVE or not bucket_items:
            fp_rate = None
        else:
            fp_rate = incorrect / len(bucket_items)
        buckets.append(
            BucketBreakdown(
                difficulty=difficulty,
                n=len(bucket_items),
                correct=correct,
                incorrect=incorrect,
                false_positive_rate=fp_rate,
            )
        )

    return EvaluationResult(
        counts=counts,
        accuracy=accuracy(counts),
        precision=precision(counts),
        recall=recall(counts),
        f1=f1_score(counts),
        n_true_positives=tp,
        n_localized=n_localized,
        # Mean over ALL true positives: unlocalized ones contribute their 0.0.
        mean_iou=mean(list(iou_by_id.values())),
        mean_start_error=mean(start_errors),
        mean_end_error=mean(end_errors),
        iou_by_id=iou_by_id,
        buckets=tuple(buckets),
        system=system,
    )


def result_to_dict(result: EvaluationResult) -> dict[str, object]:
    """JSON-safe snake_case view of a result (None stays null)."""

    return {
        "samples": result.counts.total,
        "counts": {
            "true_positives": result.counts.true_positives,
            "false_positives": result.counts.false_positives,
            "true_negatives": result.counts.true_negatives,
            "false_negatives": result.counts.false_negatives,
        },
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "temporal": {
            "n_true_positives": result.n_true_positives,
            "n_localized": result.n_localized,
            "mean_iou": result.mean_iou,
            "mean_start_error": result.mean_start_error,
            "mean_end_error": result.mean_end_error,
            "iou_by_id": dict(result.iou_by_id),
        },
        "buckets": [
            {
                "difficulty": bucket.difficulty.value,
                "n": bucket.n,
                "correct": bucket.correct,
                "incorrect": bucket.incorrect,
                "false_positive_rate": bucket.false_positive_rate,
            }
            for bucket in result.buckets
        ],
        "system": dict(result.system) if result.system is not None else None,
    }


def _fmt(value: float | None, *, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def render_report(result: EvaluationResult) -> str:
    lines = [
        "ID Event Light Bench v0",
        "",
        f"Samples: {result.counts.total}",
        "",
        f"Accuracy: {_fmt(result.accuracy)}",
        f"Precision: {_fmt(result.precision)}",
        f"Recall: {_fmt(result.recall)}",
        f"F1: {_fmt(result.f1)}",
        "",
        f"True positives: {result.n_true_positives} (localized: {result.n_localized})",
        f"Temporal IoU: {_fmt(result.mean_iou)}",
        f"Mean start error: {_fmt(result.mean_start_error, suffix='s')}",
        f"Mean end error: {_fmt(result.mean_end_error, suffix='s')}",
        "",
    ]
    for bucket in result.buckets:
        lines.append(f"{bucket.difficulty.value.replace('_', ' ').capitalize()}s:")
        lines.append(f"  Correct: {bucket.correct}/{bucket.n}")
        if bucket.false_positive_rate is not None:
            lines.append(f"  False positive rate: {bucket.false_positive_rate:.3f}")
    if result.system is not None:
        lines.append("")
        lines.append("System (reported by submitter, not verified):")
        for key, value in result.system.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="benchmark manifest JSON")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions JSON")
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of text"
    )
    args = parser.parse_args(argv)

    try:
        # Scoring reads labels and predictions, never pixels, so video-file
        # existence is not this CLI's concern (validate_manifest owns it).
        items = load_manifest(args.manifest, check_video_files=False)
        predictions, system = load_predictions(args.predictions)
        result = evaluate(items, predictions, system=system)
    except (ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result_to_dict(result), indent=2))
    else:
        print(render_report(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
