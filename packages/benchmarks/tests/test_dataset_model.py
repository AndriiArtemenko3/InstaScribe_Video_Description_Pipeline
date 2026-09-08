"""Dataset item metadata: validation matrix, serialization, model-blindness."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from instadescribe_benchmarks.id_event_light_v0.dataset.model import (
    DatasetItem,
    EventCategory,
    HardNegativeType,
    ItemStatus,
    SourceType,
    item_from_json,
    item_to_json,
    normalize_timestamp,
    rights_errors,
    sha256_file,
    validation_errors,
)
from instadescribe_benchmarks.id_event_light_v0.schema import Difficulty

SHA = "a" * 64


def item(**overrides: object) -> DatasetItem:
    fields: dict[str, object] = {
        "item_id": "item-001",
        "source_group_id": "g-001",
        "status": ItemStatus.ANNOTATED,
        "source_type": SourceType.SELF_RECORDED,
        "source_reference": "session 2026-09-08, phone camera",
        "acquisition_date": "2026-09-08",
        "clip_filename": "item-001.mp4",
        "derived_clip_sha256": SHA,
        "query": "a person enters a vehicle",
        "label": True,
        "difficulty": Difficulty.POSITIVE,
        "event_category": EventCategory.DIRECTIONAL_TRANSITION,
        "start_time": 2.0,
        "end_time": 5.5,
        "annotator": "owner",
        "annotation_date": "2026-09-08",
    }
    fields.update(overrides)
    return DatasetItem(**fields)  # type: ignore[arg-type]


def test_valid_positive_negative_hard_negative() -> None:
    assert validation_errors(item()) == []
    negative = item(
        query="a dog runs through the frame",
        label=False,
        difficulty=Difficulty.NEGATIVE,
        start_time=None,
        end_time=None,
    )
    assert validation_errors(negative) == []
    hard = item(
        label=False,
        difficulty=Difficulty.HARD_NEGATIVE,
        start_time=None,
        end_time=None,
        hard_negative_type=HardNegativeType.PARTIAL_ACTION,
    )
    assert validation_errors(hard) == []


def test_invalid_positive_without_interval() -> None:
    errors = validation_errors(item(start_time=None, end_time=None))
    assert any("start_time and end_time" in message for message in errors)


def test_invalid_negative_with_interval() -> None:
    broken = item(label=False, difficulty=Difficulty.NEGATIVE)
    errors = validation_errors(broken)
    assert any("null timestamps" in message for message in errors)


def test_invalid_hard_negative_without_type() -> None:
    broken = item(label=False, difficulty=Difficulty.HARD_NEGATIVE, start_time=None, end_time=None)
    assert any("hard_negative_type" in message for message in validation_errors(broken))


def test_hard_negative_type_forbidden_elsewhere() -> None:
    with pytest.raises(ValueError, match="hard_negative_type must be null"):
        item(hard_negative_type=HardNegativeType.WRONG_DIRECTION)


def test_label_difficulty_contradiction_rejected() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        item(label=False)


def test_invalid_controlled_vocabulary_rejected() -> None:
    record = item_to_json(item())
    record["event_category"] = "vibes"
    with pytest.raises(ValueError, match="event_category"):
        item_from_json(record)
    record = item_to_json(item())
    record["unexpected_field"] = 1
    with pytest.raises(ValueError, match="unknown item fields"):
        item_from_json(record)


def test_malformed_hash_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        item(derived_clip_sha256="not-a-hash")


def test_timestamp_precision_enforced_and_normalizer_reports_change() -> None:
    with pytest.raises(ValueError, match="0.1 s precision"):
        item(start_time=2.04)
    normalized, changed = normalize_timestamp(2.04)
    assert (normalized, changed) == (2.0, True)
    assert normalize_timestamp(2.0) == (2.0, False)


def test_review_stage_requires_explicit_approval() -> None:
    reviewed = item(status=ItemStatus.REVIEWED)
    assert any("review_status" in message for message in validation_errors(reviewed))


def test_rights_requirements_by_source_type() -> None:
    assert rights_errors(item()) == []  # self-recorded: reference is enough
    external = item(source_type=SourceType.PERMISSIVELY_LICENSED)
    messages = rights_errors(external)
    assert any("licence_id" in m for m in messages)
    assert any("licence_evidence" in m for m in messages)


def test_serialization_round_trip_and_determinism() -> None:
    original = item()
    record = item_to_json(original)
    assert item_from_json(record) == original
    assert list(record) == sorted(record)  # deterministic key order
    assert record["hard_negative_type"] is None  # None survives, never ""


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"media-bytes")
    assert sha256_file(path) == hashlib.sha256(b"media-bytes").hexdigest()
    with pytest.raises(ValueError, match="not found"):
        sha256_file(tmp_path / "missing.bin")


def test_dataset_tooling_is_model_blind() -> None:
    """Static assertion: no dataset module imports any solver/model runtime."""

    forbidden = (
        "retrieve",
        "baselines",
        "onnxruntime",
        "tokenizers",
        "numpy",
        "PIL",
        "huggingface",
    )
    dataset_dir = (
        Path(__file__).parent.parent
        / "src"
        / "instadescribe_benchmarks"
        / "id_event_light_v0"
        / "dataset"
    )
    for source in sorted(dataset_dir.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            for name in names:
                for banned in forbidden:
                    assert banned not in name, f"{source.name} imports {name!r} ({banned})"
