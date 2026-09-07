"""E002 S3: constrained max-sum search, margins, interval mapping."""

from __future__ import annotations

import pytest

from instadescribe_benchmarks.id_event_light_v0.retrieve.config import RetrieveSolverConfig
from instadescribe_benchmarks.id_event_light_v0.retrieve.proposals import (
    best_proposal,
    constrained_max_subarray,
    contrast_margin,
    frame_length_bounds,
    map_interval,
)


def config(**overrides: object) -> RetrieveSolverConfig:
    defaults: dict[str, object] = {
        "sampling_fps": 2.0,
        "smoothing_width": 1,
        "min_proposal_duration_s": 1.0,  # L_min = 2 frames at 2 fps
        "max_proposal_duration_s": 2.0,  # L_max = 4 frames at 2 fps
    }
    defaults.update(overrides)
    return RetrieveSolverConfig(**defaults)  # type: ignore[arg-type]


def test_known_constrained_example() -> None:
    # Best window of length 2..3 in [1, -5, 4, 3, -5]: [4, 3] at indices 2..3.
    start, end, total = constrained_max_subarray([1.0, -5.0, 4.0, 3.0, -5.0], 2, 3)
    assert (start, end) == (2, 3)
    assert total == pytest.approx(7.0)


def test_minimum_length_forces_neighbor_inclusion() -> None:
    # Unconstrained optimum is the single spike [2..2]; L_min=2 must extend it.
    start, end, total = constrained_max_subarray([-1.0, -1.0, 10.0, -2.0, -1.0], 2, 3)
    assert (start, end) == (1, 2)  # spike plus its cheaper neighbor (-1 < -2)
    assert total == pytest.approx(9.0)


def test_maximum_length_caps_broad_region() -> None:
    # All-positive region longer than L_max: the best L_max-length window wins.
    start, end, total = constrained_max_subarray([1.0, 2.0, 3.0, 2.0, 1.0], 1, 3)
    assert (start, end) == (1, 3)
    assert total == pytest.approx(7.0)


def test_unconstrained_optimum_illegal_constrained_correct() -> None:
    # Unconstrained Kadane would take all five (sum 6); L_max=2 forbids it.
    start, end, total = constrained_max_subarray([2.0, 1.0, 0.0, 1.0, 2.0], 1, 2)
    assert (start, end) == (0, 1)
    assert total == pytest.approx(3.0)


def test_tie_break_earliest_start() -> None:
    # Two identical windows [3, 3]: the earlier one must win.
    start, end, _ = constrained_max_subarray([3.0, -9.0, 3.0], 1, 1)
    assert (start, end) == (0, 0)


def test_tie_break_shorter_duration_on_equal_sum_and_start() -> None:
    # [5, 0]: window [0..0] and [0..1] both sum to 5 -> shorter wins.
    start, end, _ = constrained_max_subarray([5.0, 0.0], 1, 2)
    assert (start, end) == (0, 0)


def test_bounds_validation() -> None:
    with pytest.raises(ValueError, match="window bounds"):
        constrained_max_subarray([1.0], 2, 3)  # l_min > n
    with pytest.raises(ValueError, match="window bounds"):
        constrained_max_subarray([1.0, 2.0], 2, 1)  # l_max < l_min


def test_contrast_margin_distinct_from_accumulated_sum() -> None:
    # The search objective (sum inside) and the detection statistic (mean
    # inside minus mean outside) genuinely order windows differently:
    # [0..2] has the LARGER sum (6.1 > 6.0) but the SMALLER margin.
    values = [3.0, 3.0, 0.1, -3.0, -3.0, -3.0]
    sum_short, sum_long = sum(values[0:2]), sum(values[0:3])
    assert sum_long > sum_short
    margin_short = contrast_margin(values, 0, 1)
    margin_long = contrast_margin(values, 0, 2)
    assert margin_short == pytest.approx(3.0 - ((0.1 - 9.0) / 4.0))
    assert margin_long == pytest.approx((6.1 / 3.0) - (-3.0))
    assert margin_long < margin_short


def test_contrast_margin_whole_video_is_zero_by_convention() -> None:
    assert contrast_margin([0.5, 0.7], 0, 1) == 0.0


def test_frame_length_bounds_degrades_for_short_video() -> None:
    # min 1.0 s at 2 fps -> L_min 2; a single-frame video degrades.
    l_min, l_max, degraded = frame_length_bounds(config(), 1)
    assert (l_min, l_max, degraded) == (1, 1, True)


def test_frame_length_bounds_rejects_empty_and_impossible() -> None:
    with pytest.raises(ValueError, match="at least one sampled frame"):
        frame_length_bounds(config(), 0)
    # 0.3 s min and 0.4 s max at 2 fps: ceil(0.6)=1, floor(0.8)=0 -> no length.
    bad = config(min_proposal_duration_s=0.3, max_proposal_duration_s=0.4)
    with pytest.raises(ValueError, match="admit no window length"):
        frame_length_bounds(bad, 5)


def test_map_interval_frozen_convention() -> None:
    # Inclusive [2, 4] at 2 fps: cells 1.0 .. 2.5.
    assert map_interval(2, 4, sampling_fps=2.0, video_duration_s=10.0) == (1.0, 2.5)


def test_map_interval_one_frame_positive_duration() -> None:
    start_s, end_s = map_interval(3, 3, sampling_fps=2.0, video_duration_s=10.0)
    assert (start_s, end_s) == (1.5, 2.0)
    assert end_s > start_s


def test_map_interval_end_clamped_to_duration() -> None:
    # Last cell would end at 2.0 but the video is 1.9 s long.
    assert map_interval(3, 3, sampling_fps=2.0, video_duration_s=1.9) == (1.5, 1.9)


def test_map_interval_invariant_violation_raises() -> None:
    # A frame index beyond the video duration cannot produce a valid interval.
    with pytest.raises(ValueError, match="invariant"):
        map_interval(10, 10, sampling_fps=2.0, video_duration_s=1.0)


def test_best_proposal_whole_video_degradation() -> None:
    proposal, degraded = best_proposal([0.4], config=config(), video_duration_s=0.5)
    assert degraded is True
    assert (proposal.start_index, proposal.end_index) == (0, 0)
    assert proposal.contrast_margin == 0.0  # frozen neutral convention
    assert (proposal.start_s, proposal.end_s) == (0.0, 0.5)


def test_best_proposal_reports_both_quantities() -> None:
    scores = [-1.0, 3.0, 3.0, -1.0, -1.0, -1.0]
    proposal, degraded = best_proposal(scores, config=config(), video_duration_s=3.0)
    assert degraded is False
    assert (proposal.start_index, proposal.end_index) == (1, 2)
    assert proposal.accumulated_contrast == pytest.approx(6.0)
    assert proposal.contrast_margin == pytest.approx(3.0 - (-1.0))
    assert (proposal.start_s, proposal.end_s) == (0.5, 1.5)
    assert proposal.rank == 1


def test_best_proposal_emitted_even_when_all_scores_negative() -> None:
    # S3 proposes regardless of sign; it never decides event existence.
    proposal, _ = best_proposal([-5.0, -1.0, -1.0, -5.0], config=config(), video_duration_s=2.0)
    assert (proposal.start_index, proposal.end_index) == (1, 2)
    assert proposal.accumulated_contrast < 0
