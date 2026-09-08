"""Canonical item metadata model and validation for dataset construction.

One record per benchmark item candidate, progressing through the lifecycle

    candidate -> annotated -> reviewed -> assigned  (freeze is dataset-level)

Validation is status-aware and fail-closed: a record claiming a stage must
satisfy every requirement of that stage and the stages before it. ``None``
means "not provided yet"; empty strings are never used as missing values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from enum import StrEnum
from math import isfinite
from pathlib import Path

from ..schema import Difficulty

_SHA256_HEX = 64


class EventCategory(StrEnum):
    STATIC_STATE = "static_state"
    SIMPLE_MOTION = "simple_motion"
    INTERACTION = "interaction"
    TRANSITION = "transition"
    DIRECTIONAL_TRANSITION = "directional_transition"
    STATE_CHANGE = "state_change"


class HardNegativeType(StrEnum):
    PRECONDITION_WITHOUT_ACTION = "precondition_without_action"
    POSTCONDITION_WITHOUT_TRANSITION = "postcondition_without_transition"
    RELATED_OBJECT_INTERACTION = "related_object_interaction"
    WRONG_DIRECTION = "wrong_direction"
    PARTIAL_ACTION = "partial_action"
    SIMILAR_MOTION = "similar_motion"
    SAME_SCENE_WRONG_EVENT = "same_scene_wrong_event"
    TEMPORAL_NEAR_MISS = "temporal_near_miss"


class SourceType(StrEnum):
    SELF_RECORDED = "self_recorded"
    PURPOSE_RECORDED = "purpose_recorded"
    PERMISSIVELY_LICENSED = "permissively_licensed"
    PUBLIC_DOMAIN = "public_domain"
    OTHER_RIGHTS_CLEARED = "other_rights_cleared"


# Externally sourced media requires explicit licence evidence before freeze.
EXTERNAL_SOURCE_TYPES = frozenset(
    {SourceType.PERMISSIVELY_LICENSED, SourceType.PUBLIC_DOMAIN, SourceType.OTHER_RIGHTS_CLEARED}
)


class ExclusionReason(StrEnum):
    AMBIGUOUS_EVENT = "ambiguous_event"
    AMBIGUOUS_QUERY = "ambiguous_query"
    CRITICAL_TRANSITION_OCCLUDED = "critical_transition_occluded"
    MULTIPLE_OCCURRENCES_NOT_REPRESENTABLE = "multiple_occurrences_not_representable"
    UNUSABLE_MEDIA = "unusable_media"
    RIGHTS_UNRESOLVED = "rights_unresolved"
    NEAR_DUPLICATE = "near_duplicate"
    OTHER = "other"


class ItemStatus(StrEnum):
    CANDIDATE = "candidate"
    ANNOTATED = "annotated"
    REVIEWED = "reviewed"
    ASSIGNED = "assigned"


_STATUS_ORDER = {
    ItemStatus.CANDIDATE: 0,
    ItemStatus.ANNOTATED: 1,
    ItemStatus.REVIEWED: 2,
    ItemStatus.ASSIGNED: 3,
}


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def sha256_file(path: Path) -> str:
    """Hash a local file; never trust a user-entered digest when the file exists."""

    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_timestamp(value: float) -> tuple[float, bool]:
    """Normalize to 0.1 s precision; the boolean says whether input changed.

    Callers surfacing annotator input must show the normalized value before
    saving when ``changed`` is True — never round silently.
    """

    if not (isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value)):
        raise ValueError("timestamp must be a finite number")
    normalized = round(float(value), 1)
    return normalized, abs(normalized - float(value)) > 1e-9


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """One candidate/final benchmark item. All lifecycle fields in one place."""

    # identity
    item_id: str
    source_group_id: str
    status: ItemStatus
    # source / provenance
    source_type: SourceType
    source_reference: str | None = None
    source_title: str | None = None
    source_creator: str | None = None
    licence_id: str | None = None
    licence_evidence: str | None = None
    acquisition_date: str | None = None  # ISO date, human-entered
    # media
    original_filename: str | None = None
    original_file_sha256: str | None = None
    clip_filename: str | None = None
    derived_clip_sha256: str | None = None
    transform_log: tuple[str, ...] = ()
    # annotation
    query: str | None = None
    label: bool | None = None
    difficulty: Difficulty | None = None
    event_category: EventCategory | None = None
    hard_negative_type: HardNegativeType | None = None
    start_time: float | None = None
    end_time: float | None = None
    annotation_notes: str | None = None
    annotator: str | None = None
    annotation_date: str | None = None
    # review
    review_status: str | None = None  # "approved" is the only accepting value
    reviewer: str | None = None
    review_date: str | None = None
    review_notes: str | None = None
    # split
    split: Split | None = None

    def __post_init__(self) -> None:
        # Always-true structural invariants, regardless of lifecycle stage.
        for name in ("item_id", "source_group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("original_file_sha256", "derived_clip_sha256"):
            value = getattr(self, name)
            if value is not None and not is_sha256(value):
                raise ValueError(f"{name} must be a 64-character hex sha256")
        if self.query is not None:
            if not self.query.strip():
                raise ValueError("query must not be empty when provided")
            if "\n" in self.query or "\r" in self.query:
                raise ValueError("query must not contain newlines")
            if len(self.query) > 300:
                raise ValueError("query is unreasonably long (> 300 characters)")
        if self.start_time is not None or self.end_time is not None:
            if self.start_time is None or self.end_time is None:
                raise ValueError("start_time and end_time must be provided together")
            for name, value in (("start_time", self.start_time), ("end_time", self.end_time)):
                normalized, changed = normalize_timestamp(value)
                if changed:
                    raise ValueError(f"{name} must be stored at 0.1 s precision (got {value})")
            if not 0 <= self.start_time < self.end_time:
                raise ValueError("require 0 <= start_time < end_time")
        if self.label is not None and self.difficulty is not None:
            if self.label != (self.difficulty is Difficulty.POSITIVE):
                raise ValueError(
                    f"label {self.label} contradicts difficulty {self.difficulty.value!r}"
                )
        if self.difficulty is not None:
            if self.difficulty is Difficulty.HARD_NEGATIVE:
                pass  # hard_negative_type requirement enforced per-stage below
            elif self.hard_negative_type is not None:
                raise ValueError("hard_negative_type must be null unless difficulty=hard_negative")
        if self.review_status is not None and self.review_status not in ("pending", "approved"):
            raise ValueError("review_status must be 'pending' or 'approved'")


def validation_errors(item: DatasetItem) -> list[str]:
    """Stage-aware requirements; a record must satisfy its claimed status."""

    errors: list[str] = []
    stage = _STATUS_ORDER[item.status]

    # candidate: identity + media registration
    for name in ("clip_filename", "derived_clip_sha256", "acquisition_date", "source_reference"):
        if getattr(item, name) is None:
            errors.append(f"{name} is required from candidate stage")

    if stage >= _STATUS_ORDER[ItemStatus.ANNOTATED]:
        for name in (
            "query",
            "label",
            "difficulty",
            "event_category",
            "annotator",
            "annotation_date",
        ):
            if getattr(item, name) is None:
                errors.append(f"{name} is required once annotated")
        if item.difficulty is Difficulty.POSITIVE:
            if item.start_time is None or item.end_time is None:
                errors.append("positive items require start_time and end_time")
            if item.hard_negative_type is not None:
                errors.append("positive items must have null hard_negative_type")
        elif item.difficulty is Difficulty.NEGATIVE:
            if item.start_time is not None or item.end_time is not None:
                errors.append("negative items must have null timestamps")
            if item.hard_negative_type is not None:
                errors.append("ordinary negatives must have null hard_negative_type")
        elif item.difficulty is Difficulty.HARD_NEGATIVE:
            if item.start_time is not None or item.end_time is not None:
                errors.append("hard negatives must have null timestamps")
            if item.hard_negative_type is None:
                errors.append("hard negatives require a hard_negative_type")

    if stage >= _STATUS_ORDER[ItemStatus.REVIEWED]:
        if item.review_status != "approved":
            errors.append("reviewed items require review_status='approved'")
        for name in ("reviewer", "review_date"):
            if getattr(item, name) is None:
                errors.append(f"{name} is required once reviewed")

    if stage >= _STATUS_ORDER[ItemStatus.ASSIGNED] and item.split is None:
        errors.append("assigned items require a split")

    return errors


def rights_errors(item: DatasetItem) -> list[str]:
    """Provenance completeness required before freeze.

    The tool validates that evidence was supplied; it never concludes whether
    a licence is legally valid.
    """

    errors: list[str] = []
    if item.source_reference is None:
        errors.append("source_reference is required")
    if item.source_type in EXTERNAL_SOURCE_TYPES:
        if item.licence_id is None:
            errors.append("licence_id is required for externally sourced media")
        if item.licence_evidence is None:
            errors.append("licence_evidence is required for externally sourced media")
    return errors


_ENUM_FIELDS: dict[str, type[StrEnum]] = {
    "status": ItemStatus,
    "source_type": SourceType,
    "difficulty": Difficulty,
    "event_category": EventCategory,
    "hard_negative_type": HardNegativeType,
    "split": Split,
}
_FIELD_NAMES = tuple(field.name for field in fields(DatasetItem))


def item_to_json(item: DatasetItem) -> dict[str, object]:
    """Deterministic, complete serialization (every field, sorted keys)."""

    record: dict[str, object] = {}
    for name in sorted(_FIELD_NAMES):
        value = getattr(item, name)
        if isinstance(value, StrEnum):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        record[name] = value
    return record


def item_from_json(record: object) -> DatasetItem:
    """Strict parse: unknown keys, bad enum values, wrong types fail closed."""

    if not isinstance(record, dict):
        raise ValueError("item record must be a JSON object")
    unknown = record.keys() - set(_FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown item fields {sorted(unknown)}")
    kwargs: dict[str, object] = dict(record)
    for name, enum_type in _ENUM_FIELDS.items():
        value = kwargs.get(name)
        if value is not None:
            try:
                kwargs[name] = enum_type(value)  # type: ignore[arg-type]
            except ValueError as error:
                raise ValueError(
                    f"{name} value {value!r} is not one of {[e.value for e in enum_type]}"
                ) from error
    transform = kwargs.get("transform_log", ())
    if transform is not None:
        if not isinstance(transform, list | tuple) or not all(
            isinstance(entry, str) for entry in transform
        ):
            raise ValueError("transform_log must be a list of strings")
        kwargs["transform_log"] = tuple(transform)
    return DatasetItem(**kwargs)  # type: ignore[arg-type]
