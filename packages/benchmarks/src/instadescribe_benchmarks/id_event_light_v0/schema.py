"""Data contracts for ID Event Light Bench v0.

One asymmetry drives everything in this module:

- Ground truth (the manifest) is validated *fail closed*: any inconsistency
  raises :class:`ManifestError`, because a broken manifest silently corrupts
  every score computed from it.
- Predictions are validated only against *file corruption* (wrong types,
  non-finite numbers, duplicate ids). A prediction whose interval is missing
  or reversed loads fine — that is a system behavior the benchmark scores,
  not a broken file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path

MANIFEST_SCHEMA_VERSION = "id-event-light-bench-manifest/0"
PREDICTIONS_SCHEMA_VERSION = "id-event-light-bench-predictions/0"

REQUIRED_ITEM_FIELDS = frozenset(
    {"id", "video", "query", "label", "start_time", "end_time", "difficulty"}
)
KNOWN_ITEM_FIELDS = REQUIRED_ITEM_FIELDS | {"notes"}
REQUIRED_PREDICTION_FIELDS = frozenset(
    {"id", "event_detected", "start_time", "end_time", "confidence"}
)


class ManifestError(ValueError):
    """Any contract violation — validation never degrades to a warning."""


class Difficulty(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    HARD_NEGATIVE = "hard_negative"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _as_finite_number(value: object, field_name: str, owner: str) -> float | None:
    """Return a finite float, None, or fail closed.

    bool is excluded explicitly because in Python ``True`` is an ``int`` and
    would otherwise slip through as the timestamp 1.0.
    """

    if value is None:
        return None
    _require(
        isinstance(value, int | float) and not isinstance(value, bool),
        f"{owner}: {field_name} must be a number or null",
    )
    number = float(value)  # type: ignore[arg-type]
    _require(isfinite(number), f"{owner}: {field_name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One ground-truth record: a video, a query, and what actually happened."""

    item_id: str
    video: str
    query: str
    label: bool
    start_time: float | None
    end_time: float | None
    difficulty: Difficulty
    notes: str | None = None

    def __post_init__(self) -> None:
        owner = f"item {self.item_id!r}" if self.item_id else "item"
        for field_name in ("item_id", "video", "query"):
            value = getattr(self, field_name)
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{owner}: {field_name} must be a non-empty string",
            )
        _require(isinstance(self.label, bool), f"{owner}: label must be a boolean")
        _require(
            isinstance(self.difficulty, Difficulty),
            f"{owner}: difficulty must be one of {[d.value for d in Difficulty]}",
        )
        # label and difficulty are deliberately redundant: the pair is a
        # consistency check for hand-edited manifests, not two sources of truth.
        _require(
            self.label == (self.difficulty is Difficulty.POSITIVE),
            f"{owner}: label {self.label} contradicts difficulty {self.difficulty.value!r}",
        )
        if self.label:
            _require(
                self.start_time is not None and self.end_time is not None,
                f"{owner}: a positive item must have both start_time and end_time",
            )
            assert self.start_time is not None and self.end_time is not None
            _require(
                self.start_time >= 0.0,
                f"{owner}: start_time must be >= 0",
            )
            # Strictly before: a zero-length ground-truth event has no duration
            # to anchor an IoU denominator, so it is rejected outright.
            _require(
                self.start_time < self.end_time,
                f"{owner}: start_time must be strictly before end_time",
            )
        else:
            _require(
                self.start_time is None and self.end_time is None,
                f"{owner}: a negative item must have null start_time and end_time",
            )
        if self.notes is not None:
            _require(isinstance(self.notes, str), f"{owner}: notes must be a string or omitted")


@dataclass(frozen=True, slots=True)
class Prediction:
    """One system output for one benchmark item.

    Only type/domain corruption is rejected here. A missing or reversed
    interval is allowed through so the evaluator can score it as a
    localization failure.
    """

    item_id: str
    event_detected: bool
    start_time: float | None
    end_time: float | None
    confidence: float | None

    def __post_init__(self) -> None:
        owner = f"prediction {self.item_id!r}" if self.item_id else "prediction"
        _require(
            isinstance(self.item_id, str) and bool(self.item_id.strip()),
            f"{owner}: id must be a non-empty string",
        )
        _require(
            isinstance(self.event_detected, bool),
            f"{owner}: event_detected must be a boolean",
        )
        for field_name in ("start_time", "end_time"):
            value = getattr(self, field_name)
            if value is not None:
                _require(
                    isinstance(value, float) and isfinite(value),
                    f"{owner}: {field_name} must be a finite number or null",
                )
        if self.confidence is not None:
            _require(
                isinstance(self.confidence, float) and isfinite(self.confidence),
                f"{owner}: confidence must be a finite number or null",
            )
            _require(
                0.0 <= self.confidence <= 1.0,
                f"{owner}: confidence must be within [0, 1]",
            )


def _load_json_object(path: Path, what: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"{what} file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"{what} file {path} is not valid JSON: {error}") from error
    _require(isinstance(raw, dict), f"{what} file {path} must contain a JSON object")
    return raw  # type: ignore[return-value]


def _parse_item(record: object, index: int) -> BenchmarkItem:
    _require(isinstance(record, dict), f"items[{index}] must be a JSON object")
    assert isinstance(record, dict)
    owner = f"items[{index}]"
    missing = REQUIRED_ITEM_FIELDS - record.keys()
    _require(not missing, f"{owner}: missing required fields {sorted(missing)}")
    unknown = record.keys() - KNOWN_ITEM_FIELDS
    # Unknown keys fail closed to catch typos like "starttime", which would
    # otherwise silently coexist with a null "start_time".
    _require(not unknown, f"{owner}: unknown fields {sorted(unknown)}")

    item_id = record["id"]
    _require(isinstance(item_id, str), f"{owner}: id must be a string")
    difficulty_raw = record["difficulty"]
    try:
        difficulty = Difficulty(difficulty_raw)  # type: ignore[arg-type]
    except ValueError as error:
        raise ManifestError(
            f"item {item_id!r}: difficulty {difficulty_raw!r} is not one of "
            f"{[d.value for d in Difficulty]}"
        ) from error

    label = record["label"]
    _require(isinstance(label, bool), f"item {item_id!r}: label must be a boolean")
    return BenchmarkItem(
        item_id=item_id,  # type: ignore[arg-type]
        video=record["video"],  # type: ignore[arg-type]
        query=record["query"],  # type: ignore[arg-type]
        label=label,
        start_time=_as_finite_number(record["start_time"], "start_time", f"item {item_id!r}"),
        end_time=_as_finite_number(record["end_time"], "end_time", f"item {item_id!r}"),
        difficulty=difficulty,
        notes=record.get("notes"),  # type: ignore[arg-type]
    )


def load_manifest(path: Path, *, check_video_files: bool = True) -> tuple[BenchmarkItem, ...]:
    """Load and fully validate a benchmark manifest.

    ``check_video_files=False`` skips only the file-existence check, so
    template/sample manifests with placeholder paths stay validatable.
    Video paths are resolved relative to the manifest's own directory.
    """

    raw = _load_json_object(path, "manifest")
    _require(
        raw.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}, "
        f"got {raw.get('schema_version')!r}",
    )
    items_raw = raw.get("items")
    _require(isinstance(items_raw, list), "manifest items must be a list")
    assert isinstance(items_raw, list)
    _require(len(items_raw) > 0, "manifest must contain at least one item")

    items = tuple(_parse_item(record, index) for index, record in enumerate(items_raw))

    seen: set[str] = set()
    for item in items:
        _require(item.item_id not in seen, f"duplicate item id {item.item_id!r}")
        seen.add(item.item_id)

    if check_video_files:
        base = path.resolve().parent
        for item in items:
            video_path = base / item.video
            _require(
                video_path.is_file(),
                f"item {item.item_id!r}: video file not found: {video_path}",
            )
    return items


def _parse_prediction(record: object, index: int) -> Prediction:
    _require(isinstance(record, dict), f"predictions[{index}] must be a JSON object")
    assert isinstance(record, dict)
    owner = f"predictions[{index}]"
    missing = REQUIRED_PREDICTION_FIELDS - record.keys()
    _require(not missing, f"{owner}: missing required fields {sorted(missing)}")
    unknown = record.keys() - REQUIRED_PREDICTION_FIELDS
    _require(not unknown, f"{owner}: unknown fields {sorted(unknown)}")

    item_id = record["id"]
    _require(isinstance(item_id, str), f"{owner}: id must be a string")
    return Prediction(
        item_id=item_id,  # type: ignore[arg-type]
        event_detected=record["event_detected"],  # type: ignore[arg-type]
        start_time=_as_finite_number(record["start_time"], "start_time", f"prediction {item_id!r}"),
        end_time=_as_finite_number(record["end_time"], "end_time", f"prediction {item_id!r}"),
        confidence=_as_finite_number(record["confidence"], "confidence", f"prediction {item_id!r}"),
    )


def load_predictions(path: Path) -> tuple[tuple[Prediction, ...], dict[str, object] | None]:
    """Load a predictions file.

    Returns the predictions plus the optional free-form ``system`` object
    (model name, latency, cost, ...) passed through verbatim — the evaluator
    reports it but never computes with it.
    """

    raw = _load_json_object(path, "predictions")
    _require(
        raw.get("schema_version") == PREDICTIONS_SCHEMA_VERSION,
        f"predictions schema_version must be {PREDICTIONS_SCHEMA_VERSION!r}, "
        f"got {raw.get('schema_version')!r}",
    )
    unknown = raw.keys() - {"schema_version", "system", "predictions"}
    _require(not unknown, f"predictions file: unknown top-level fields {sorted(unknown)}")

    system = raw.get("system")
    if system is not None:
        _require(isinstance(system, dict), "predictions system must be a JSON object")

    records = raw.get("predictions")
    _require(isinstance(records, list), "predictions must be a list")
    assert isinstance(records, list)
    predictions = tuple(_parse_prediction(record, index) for index, record in enumerate(records))

    seen: set[str] = set()
    for prediction in predictions:
        _require(prediction.item_id not in seen, f"duplicate prediction id {prediction.item_id!r}")
        seen.add(prediction.item_id)
    return predictions, system  # type: ignore[return-value]
