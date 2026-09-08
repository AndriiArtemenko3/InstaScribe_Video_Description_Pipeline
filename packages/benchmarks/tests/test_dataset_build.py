"""Store, exclusion ledger, status, manifests, freeze, and drift detection."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from instadescribe_benchmarks.id_event_light_v0.dataset.build import (
    build_manifests,
    dataset_errors,
    freeze_dataset,
    status_report,
    verify_freeze,
)
from instadescribe_benchmarks.id_event_light_v0.dataset.model import (
    DatasetItem,
    EventCategory,
    ExclusionReason,
    HardNegativeType,
    ItemStatus,
    SourceType,
    Split,
    sha256_file,
)
from instadescribe_benchmarks.id_event_light_v0.dataset.store import (
    Workspace,
    append_exclusion,
    load_exclusions,
    load_items,
    save_items,
)
from instadescribe_benchmarks.id_event_light_v0.schema import Difficulty, load_manifest

FAKE_DURATION = 60.0


def fake_probe(path: Path) -> float:
    return FAKE_DURATION


def build_workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(root=tmp_path / "dataset")
    workspace.ensure_layout()
    return workspace


def make_full_item(workspace: Workspace, index: int, difficulty: Difficulty) -> DatasetItem:
    """A fully assigned item with a real (synthetic) clip file on disk."""

    clip_name = f"item-{index:03d}.mp4"
    clip_path = workspace.clips_dir / clip_name
    clip_path.write_bytes(f"synthetic-clip-{index}".encode())
    dev_quota = {Difficulty.POSITIVE: 6, Difficulty.NEGATIVE: 6, Difficulty.HARD_NEGATIVE: 6}
    position_in_bucket = index % 16
    split = Split.DEV if position_in_bucket < dev_quota[difficulty] else Split.TEST
    return DatasetItem(
        item_id=f"item-{index:03d}",
        source_group_id=f"g-{index:03d}",
        status=ItemStatus.ASSIGNED,
        source_type=SourceType.SELF_RECORDED,
        source_reference="recording session A",
        acquisition_date="2026-09-08",
        clip_filename=clip_name,
        derived_clip_sha256=sha256_file(clip_path),
        query="a person enters a vehicle",
        label=difficulty is Difficulty.POSITIVE,
        difficulty=difficulty,
        event_category=EventCategory.DIRECTIONAL_TRANSITION,
        hard_negative_type=(
            HardNegativeType.RELATED_OBJECT_INTERACTION
            if difficulty is Difficulty.HARD_NEGATIVE
            else None
        ),
        start_time=1.0 if difficulty is Difficulty.POSITIVE else None,
        end_time=4.5 if difficulty is Difficulty.POSITIVE else None,
        annotator="owner",
        annotation_date="2026-09-08",
        review_status="approved",
        reviewer="owner",
        review_date="2026-09-08",
        split=split,
    )


def full_workspace(tmp_path: Path) -> Workspace:
    workspace = build_workspace(tmp_path)
    items: dict[str, DatasetItem] = {}
    index = 0
    for difficulty in (Difficulty.POSITIVE, Difficulty.NEGATIVE, Difficulty.HARD_NEGATIVE):
        for _ in range(16):
            entry = make_full_item(workspace, index, difficulty)
            items[entry.item_id] = entry
            index += 1
    save_items(workspace, items)
    return workspace


def test_store_round_trip_and_duplicate_rejection(tmp_path: Path) -> None:
    workspace = full_workspace(tmp_path)
    items = load_items(workspace)
    assert len(items) == 48
    save_items(workspace, items)
    assert load_items(workspace) == items  # deterministic rewrite
    # Corrupt with a duplicate line: loader fails closed.
    line = workspace.items_path.read_text().splitlines()[0]
    workspace.items_path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate item_id"):
        load_items(workspace)


def test_exclusion_ledger_preserves_history(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path)
    append_exclusion(
        workspace,
        candidate_id="item-000",
        source_group_id="g-000",
        decision="excluded",
        reason=ExclusionReason.AMBIGUOUS_EVENT,
        notes="cannot tell whether the door fully opens",
        solver_output_seen=False,
    )
    append_exclusion(
        workspace,
        candidate_id="item-000",
        source_group_id="g-000",
        decision="reconsidered",
        reason=ExclusionReason.OTHER,
        notes="re-watched at full resolution",
        solver_output_seen=False,
    )
    entries = load_exclusions(workspace)
    assert [entry["decision"] for entry in entries] == ["excluded", "reconsidered"]
    assert all(entry["solver_output_seen"] is False for entry in entries)
    with pytest.raises(ValueError, match="decision"):
        append_exclusion(
            workspace,
            candidate_id="x",
            source_group_id="g",
            decision="deleted",
            reason=ExclusionReason.OTHER,
            notes="",
            solver_output_seen=False,
        )


def test_status_report_counts_incomplete_state(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path)
    items = {}
    for index, difficulty in ((0, Difficulty.POSITIVE), (1, Difficulty.NEGATIVE)):
        entry = make_full_item(workspace, index, difficulty)
        items[entry.item_id] = entry
    save_items(workspace, items)
    append_exclusion(
        workspace,
        candidate_id="item-999",
        source_group_id="g-999",
        decision="excluded",
        reason=ExclusionReason.UNUSABLE_MEDIA,
        notes="",
        solver_output_seen=False,
    )
    report = status_report(workspace)
    assert "included items:     2 / 48" in report
    assert "excluded (current): 1" in report
    assert "positive=1/16" in report and "negative=1/16" in report
    assert "digest mismatches:  0" in report


def test_dataset_errors_flags_digest_mismatch_and_duration(tmp_path: Path) -> None:
    workspace = full_workspace(tmp_path)
    # Tamper with one clip: silent media replacement must be impossible.
    (workspace.clips_dir / "item-000.mp4").write_bytes(b"replaced")
    errors = dataset_errors(workspace, probe=fake_probe)
    assert any("digest mismatch" in message for message in errors)
    # An annotation beyond the (fake) duration fails.
    workspace2 = full_workspace(tmp_path / "second")
    items = load_items(workspace2)
    items["item-001"] = dataclasses.replace(items["item-001"], start_time=1.0, end_time=90.0)
    save_items(workspace2, items)
    errors = dataset_errors(workspace2, probe=fake_probe)
    assert any("exceeds clip duration" in message for message in errors)


def test_build_manifests_round_trips_strict_loader(tmp_path: Path) -> None:
    workspace = full_workspace(tmp_path)
    paths = build_manifests(workspace)
    dev_items = load_manifest(paths["dev"], check_video_files=True)
    test_items = load_manifest(paths["test"], check_video_files=True)
    assert len(dev_items) == 18 and len(test_items) == 30
    assert [i.item_id for i in dev_items] == sorted(i.item_id for i in dev_items)
    negatives = [i for i in test_items if not i.label]
    assert negatives and all(i.start_time is None and i.end_time is None for i in negatives)
    # Deterministic output.
    first = paths["dev"].read_bytes()
    build_manifests(workspace)
    assert paths["dev"].read_bytes() == first


def test_freeze_rejects_incomplete_and_invalid_states(tmp_path: Path) -> None:
    # Incomplete dataset.
    workspace = build_workspace(tmp_path / "incomplete")
    entry = make_full_item(workspace, 0, Difficulty.POSITIVE)
    save_items(workspace, {entry.item_id: entry})
    with pytest.raises(ValueError, match="freeze requires 48"):
        freeze_dataset(workspace, version="V_TEST", probe=fake_probe)

    # Unreviewed item.
    workspace = full_workspace(tmp_path / "unreviewed")
    items = load_items(workspace)
    items["item-005"] = dataclasses.replace(
        items["item-005"],
        status=ItemStatus.ANNOTATED,
        review_status="pending",
        reviewer=None,
        review_date=None,
        split=None,
    )
    save_items(workspace, items)
    with pytest.raises(ValueError, match="requires 'assigned'"):
        freeze_dataset(workspace, version="V_TEST", probe=fake_probe)

    # Unresolved rights.
    workspace = full_workspace(tmp_path / "rights")
    items = load_items(workspace)
    items["item-007"] = dataclasses.replace(
        items["item-007"], source_type=SourceType.PERMISSIVELY_LICENSED
    )
    save_items(workspace, items)
    with pytest.raises(ValueError, match="licence_id"):
        freeze_dataset(workspace, version="V_TEST", probe=fake_probe)

    # Digest mismatch.
    workspace = full_workspace(tmp_path / "tamper")
    (workspace.clips_dir / "item-003.mp4").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="digest mismatch"):
        freeze_dataset(workspace, version="V_TEST", probe=fake_probe)

    # Source leakage.
    workspace = full_workspace(tmp_path / "leak")
    items = load_items(workspace)
    items["item-000"] = dataclasses.replace(items["item-000"], source_group_id="shared")
    items["item-006"] = dataclasses.replace(items["item-006"], source_group_id="shared")
    save_items(workspace, items)  # item-000 dev, item-006 test
    with pytest.raises(ValueError, match="both DEV and TEST"):
        freeze_dataset(workspace, version="V_TEST", probe=fake_probe)


def test_freeze_succeeds_and_is_deterministic(tmp_path: Path) -> None:
    workspace = full_workspace(tmp_path)
    record_path = freeze_dataset(
        workspace,
        version="DATASET_FREEZE_TEST",
        probe=fake_probe,
        now="2026-09-08T00:00:00+00:00",
        tool_version="test",
    )
    first = record_path.read_bytes()
    record = json.loads(first)
    assert record["counts"]["total"] == 48
    assert record["counts"]["difficulty"] == {"positive": 16, "negative": 16, "hard_negative": 16}
    assert record["counts"]["split"]["dev"] == {"positive": 6, "negative": 6, "hard_negative": 6}
    assert len(record["items"]) == 48
    assert (workspace.freeze_dir / "DATASET_FREEZE_TEST.json.sha256").is_file()
    # Deterministic given fixed timestamp/tool version.
    freeze_dataset(
        workspace,
        version="DATASET_FREEZE_TEST",
        probe=fake_probe,
        now="2026-09-08T00:00:00+00:00",
        tool_version="test",
    )
    assert record_path.read_bytes() == first


def test_verify_freeze_detects_drift(tmp_path: Path) -> None:
    workspace = full_workspace(tmp_path)
    record_path = freeze_dataset(
        workspace,
        version="V1",
        probe=fake_probe,
        now="2026-09-08T00:00:00+00:00",
        tool_version="test",
    )
    verify_freeze(workspace, record_path)  # clean state passes

    # Media drift.
    (workspace.clips_dir / "item-010.mp4").write_bytes(b"drifted")
    with pytest.raises(ValueError, match="FAILED"):
        verify_freeze(workspace, record_path)

    # Metadata drift (restore media first).
    workspace2 = full_workspace(tmp_path / "meta")
    record_path2 = freeze_dataset(
        workspace2,
        version="V1",
        probe=fake_probe,
        now="2026-09-08T00:00:00+00:00",
        tool_version="test",
    )
    items = load_items(workspace2)
    items["item-000"] = dataclasses.replace(items["item-000"], query="a different query")
    save_items(workspace2, items)
    with pytest.raises(ValueError, match="metadata digest drifted"):
        verify_freeze(workspace2, record_path2)
