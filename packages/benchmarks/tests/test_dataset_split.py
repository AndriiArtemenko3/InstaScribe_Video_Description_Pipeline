"""Deterministic source-group-aware split assignment."""

from __future__ import annotations

import pytest

from instadescribe_benchmarks.id_event_light_v0.dataset.model import (
    DatasetItem,
    EventCategory,
    HardNegativeType,
    ItemStatus,
    SourceType,
    Split,
)
from instadescribe_benchmarks.id_event_light_v0.dataset.split import (
    assign_split,
    validate_no_leakage,
)
from instadescribe_benchmarks.id_event_light_v0.schema import Difficulty

SHA = "b" * 64


def make_item(index: int, difficulty: Difficulty, group: str) -> DatasetItem:
    return DatasetItem(
        item_id=f"item-{index:03d}",
        source_group_id=group,
        status=ItemStatus.REVIEWED,
        source_type=SourceType.SELF_RECORDED,
        source_reference="session",
        acquisition_date="2026-09-08",
        clip_filename=f"item-{index:03d}.mp4",
        derived_clip_sha256=SHA,
        query="a person enters a vehicle",
        label=difficulty is Difficulty.POSITIVE,
        difficulty=difficulty,
        event_category=EventCategory.TRANSITION,
        hard_negative_type=(
            HardNegativeType.PARTIAL_ACTION if difficulty is Difficulty.HARD_NEGATIVE else None
        ),
        start_time=1.0 if difficulty is Difficulty.POSITIVE else None,
        end_time=3.0 if difficulty is Difficulty.POSITIVE else None,
        annotator="owner",
        annotation_date="2026-09-08",
        review_status="approved",
        reviewer="owner",
        review_date="2026-09-08",
    )


def full_population(group_of=None) -> dict[str, DatasetItem]:
    """48 items, 16 per bucket; group assignment injectable per index."""

    items: dict[str, DatasetItem] = {}
    index = 0
    for difficulty in (Difficulty.POSITIVE, Difficulty.NEGATIVE, Difficulty.HARD_NEGATIVE):
        for _ in range(16):
            group = group_of(index) if group_of else f"g-{index:03d}"
            entry = make_item(index, difficulty, group)
            items[entry.item_id] = entry
            index += 1
    return items


def test_deterministic_and_exact_counts() -> None:
    items = full_population()
    first = assign_split(items)
    second = assign_split(items)
    assert first == second  # fully deterministic, no seed involved
    dev = [g for g, split in first.items() if split is Split.DEV]
    assert len(dev) == 18  # singleton groups: 6+6+6


def test_source_group_never_crosses_split() -> None:
    # Pair items across buckets into shared groups: each group has one
    # positive and one negative.
    def group_of(index: int) -> str:
        return f"pair-{index % 16:02d}" if index < 32 else f"solo-{index:03d}"

    items = full_population(group_of)
    assignment = assign_split(items)
    splits_per_group: dict[str, set[Split]] = {}
    for item in items.values():
        splits_per_group.setdefault(item.source_group_id, set()).add(
            assignment[item.source_group_id]
        )
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_impossible_grouping_fails_instead_of_leaking() -> None:
    # One giant group holding 7 positives cannot fit DEV (6) and cannot be
    # split; with the rest singletons the exact target is unreachable only if
    # every other combination fails — construct the sharp case: all 16
    # positives in ONE group.
    def group_of(index: int) -> str:
        return "mega-positive" if index < 16 else f"solo-{index:03d}"

    with pytest.raises(ValueError, match="no leakage-free split"):
        assign_split(full_population(group_of))


def test_same_location_independent_groups_is_fine() -> None:
    # Two independent recordings at the same physical location are separate
    # source groups by rule — the split must succeed.
    def group_of(index: int) -> str:
        return f"parking-lot-recording-{index:03d}"  # same place, distinct sessions

    assignment = assign_split(full_population(group_of))
    assert len(assignment) == 48


def test_incomplete_population_rejected() -> None:
    items = full_population()
    items.pop("item-000")
    with pytest.raises(ValueError, match="complete dataset"):
        assign_split(items)


def test_validate_no_leakage_reports_offending_group() -> None:
    import dataclasses

    items = full_population()
    a = dataclasses.replace(items["item-000"], source_group_id="shared", split=Split.DEV)
    b = dataclasses.replace(items["item-001"], source_group_id="shared", split=Split.TEST)
    problems = validate_no_leakage({"a": a, "b": b})
    assert problems and "shared" in problems[0]
