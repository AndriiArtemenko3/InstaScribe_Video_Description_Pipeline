"""Fail-closed manifest validation and permissive prediction loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from instadescribe_benchmarks.id_event_light_v0.schema import (
    MANIFEST_SCHEMA_VERSION,
    PREDICTIONS_SCHEMA_VERSION,
    ManifestError,
    load_manifest,
    load_predictions,
)

SAMPLE_DIR = (
    Path(__file__).parent.parent / "src" / "instadescribe_benchmarks" / "id_event_light_v0" / "data"
)


def item_record(item_id: str = "pos-001", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": item_id,
        "video": f"videos/{item_id}.mp4",
        "query": "Does a person enter the vehicle?",
        "label": True,
        "start_time": 2.0,
        "end_time": 5.0,
        "difficulty": "positive",
    }
    record.update(overrides)
    return record


def write_manifest(tmp_path: Path, items: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, "items": items}))
    return path


def load(tmp_path: Path, items: list[dict[str, object]]):
    return load_manifest(write_manifest(tmp_path, items), check_video_files=False)


def prediction_record(item_id: str = "pos-001", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": item_id,
        "event_detected": True,
        "start_time": 2.1,
        "end_time": 4.8,
        "confidence": 0.9,
    }
    record.update(overrides)
    return record


def write_predictions(tmp_path: Path, records: list[dict[str, object]], **top: object) -> Path:
    path = tmp_path / "predictions.json"
    payload: dict[str, object] = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "predictions": records,
    }
    payload.update(top)
    path.write_text(json.dumps(payload))
    return path


def test_valid_manifest_loads(tmp_path: Path) -> None:
    items = load(
        tmp_path,
        [
            item_record("pos-001"),
            item_record(
                "neg-001", label=False, difficulty="negative", start_time=None, end_time=None
            ),
        ],
    )
    assert [i.item_id for i in items] == ["pos-001", "neg-001"]
    assert items[0].start_time == 2.0 and items[1].start_time is None


def test_sample_manifest_is_valid() -> None:
    items = load_manifest(SAMPLE_DIR / "sample_manifest.json", check_video_files=False)
    assert len(items) == 8


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "wrong/9", "items": [item_record()]}))
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(path, check_video_files=False)


def test_empty_items_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="at least one item"):
        load(tmp_path, [])


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="duplicate item id"):
        load(tmp_path, [item_record("pos-001"), item_record("pos-001")])


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    record = item_record()
    del record["query"]
    with pytest.raises(ManifestError, match="missing required fields"):
        load(tmp_path, [record])


def test_unknown_item_field_rejected(tmp_path: Path) -> None:
    # a typo like "starttime" must fail, not silently coexist with start_time
    with pytest.raises(ManifestError, match="unknown fields"):
        load(tmp_path, [item_record(starttime=2.0)])


def test_malformed_field_type_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="label must be a boolean"):
        load(tmp_path, [item_record(label="yes")])


def test_invalid_difficulty_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="difficulty"):
        load(tmp_path, [item_record(difficulty="easy")])


def test_label_difficulty_mismatch_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="contradicts"):
        load(tmp_path, [item_record(label=False)])


def test_positive_without_timestamps_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="must have both"):
        load(tmp_path, [item_record(start_time=None, end_time=None)])


def test_negative_with_timestamps_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="null start_time"):
        load(tmp_path, [item_record(label=False, difficulty="negative", end_time=None)])


def test_truth_start_not_before_end_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="strictly before"):
        load(tmp_path, [item_record(start_time=5.0, end_time=2.0)])
    # zero-length ground truth cannot anchor an IoU denominator
    with pytest.raises(ManifestError, match="strictly before"):
        load(tmp_path, [item_record(start_time=2.0, end_time=2.0)])


def test_non_finite_timestamp_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="finite"):
        load(tmp_path, [item_record(start_time=float("inf"))])


def test_missing_video_file_rejected_when_checked(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [item_record()])
    with pytest.raises(ManifestError, match="video file not found"):
        load_manifest(path, check_video_files=True)


def test_missing_video_file_allowed_when_skipped(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [item_record()])
    assert len(load_manifest(path, check_video_files=False)) == 1


def test_predictions_load_with_system_slot(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, [prediction_record()], system={"name": "demo"})
    predictions, system = load_predictions(path)
    assert len(predictions) == 1
    assert system == {"name": "demo"}


def test_prediction_malformed_interval_loads(tmp_path: Path) -> None:
    # start > end is a scored system behavior, not file corruption
    path = write_predictions(tmp_path, [prediction_record(start_time=9.0, end_time=1.0)])
    predictions, _ = load_predictions(path)
    assert predictions[0].start_time == 9.0


def test_duplicate_prediction_ids_rejected(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, [prediction_record(), prediction_record()])
    with pytest.raises(ManifestError, match="duplicate prediction id"):
        load_predictions(path)


def test_prediction_confidence_out_of_range_rejected(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, [prediction_record(confidence=1.5)])
    with pytest.raises(ManifestError, match=r"within \[0, 1\]"):
        load_predictions(path)


def test_sample_predictions_file_is_valid() -> None:
    predictions, system = load_predictions(SAMPLE_DIR / "sample_predictions.json")
    assert len(predictions) == 8
    assert system is not None and system["name"] == "hand-written-example"
