"""Deterministic, source-group-aware dev/test split assignment.

The leakage unit is the source group (same original video, same continuous
recording/session, near-duplicate family) — never geography. A source group
is atomic: all of its items land on one side of the split.

Priority order, hard-coded: NO source leakage > exact bucket counts. If group
composition makes the exact targets impossible, the search STOPS and reports
why; it never splits a group and never consults solver output.

The algorithm is a deterministic depth-first search over groups in sorted
group-id order (no randomness, no seed needed): find the first subset of
groups whose summed per-difficulty counts equal the DEV target exactly; the
remaining groups form TEST, which is then verified against its own target.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..schema import Difficulty
from .model import DatasetItem, Split

DEV_TARGET: dict[Difficulty, int] = {
    Difficulty.POSITIVE: 6,
    Difficulty.NEGATIVE: 6,
    Difficulty.HARD_NEGATIVE: 6,
}
TEST_TARGET: dict[Difficulty, int] = {
    Difficulty.POSITIVE: 10,
    Difficulty.NEGATIVE: 10,
    Difficulty.HARD_NEGATIVE: 10,
}
_SEARCH_BUDGET = 2_000_000

_BUCKETS = (Difficulty.POSITIVE, Difficulty.NEGATIVE, Difficulty.HARD_NEGATIVE)


def group_counts(items: Mapping[str, DatasetItem]) -> dict[str, tuple[int, int, int]]:
    """Per-source-group item counts as (positive, negative, hard_negative)."""

    counts: dict[str, list[int]] = {}
    for item in items.values():
        if item.difficulty is None:
            raise ValueError(f"item {item.item_id!r} has no difficulty; annotate before splitting")
        row = counts.setdefault(item.source_group_id, [0, 0, 0])
        row[_BUCKETS.index(item.difficulty)] += 1
    return {group: (row[0], row[1], row[2]) for group, row in counts.items()}


def assign_split(items: Mapping[str, DatasetItem]) -> dict[str, Split]:
    """Return a deterministic {source_group_id: Split} assignment.

    Requires the complete target population (totals must equal DEV+TEST
    targets exactly) so the assignment happens once, on the full reviewed
    dataset, and is visible before freeze.
    """

    counts = group_counts(items)
    totals = tuple(sum(row[i] for row in counts.values()) for i in range(3))
    expected = tuple(DEV_TARGET[b] + TEST_TARGET[b] for b in _BUCKETS)
    if totals != expected:
        raise ValueError(
            "split assignment requires the complete dataset: "
            f"have (pos, neg, hardneg) = {totals}, need {expected}"
        )

    groups = sorted(counts)  # deterministic ordering IS the algorithm's seed
    target = tuple(DEV_TARGET[b] for b in _BUCKETS)

    # Suffix sums for feasibility pruning: what is still obtainable from the
    # remaining groups at each search depth.
    remaining = [(0, 0, 0)] * (len(groups) + 1)
    for index in range(len(groups) - 1, -1, -1):
        row = counts[groups[index]]
        nxt = remaining[index + 1]
        remaining[index] = (nxt[0] + row[0], nxt[1] + row[1], nxt[2] + row[2])

    budget = _SEARCH_BUDGET
    chosen: list[str] = []

    def search(index: int, need: tuple[int, int, int]) -> bool:
        nonlocal budget
        budget -= 1
        if budget <= 0:
            raise ValueError(
                "split search budget exceeded; simplify source-group composition "
                "or assign the split manually (never by splitting a group)"
            )
        if need == (0, 0, 0):
            return True
        if index == len(groups):
            return False
        left = remaining[index]
        if any(need[i] > left[i] for i in range(3)):
            return False  # not enough remaining in some bucket
        row = counts[groups[index]]
        # Branch 1 (preferred deterministically): take this group into DEV.
        if all(row[i] <= need[i] for i in range(3)):
            chosen.append(groups[index])
            if search(index + 1, tuple(need[i] - row[i] for i in range(3))):  # type: ignore[arg-type]
                return True
            chosen.pop()
        # Branch 2: leave this group to TEST.
        return search(index + 1, need)

    if not search(0, target):
        raise ValueError(
            "no leakage-free split reaches the exact targets "
            f"DEV {target} / TEST {tuple(TEST_TARGET[b] for b in _BUCKETS)}; "
            "source groups are atomic and will not be divided. Group "
            f"composition: { {g: counts[g] for g in groups} }"
        )

    dev_groups = set(chosen)
    return {group: (Split.DEV if group in dev_groups else Split.TEST) for group in groups}


def validate_no_leakage(items: Mapping[str, DatasetItem]) -> list[str]:
    """Every source group must live entirely inside one split."""

    seen: dict[str, set[Split]] = {}
    for item in items.values():
        if item.split is not None:
            seen.setdefault(item.source_group_id, set()).add(item.split)
    return [
        f"source group {group!r} appears in both DEV and TEST"
        for group, splits in sorted(seen.items())
        if len(splits) > 1
    ]
