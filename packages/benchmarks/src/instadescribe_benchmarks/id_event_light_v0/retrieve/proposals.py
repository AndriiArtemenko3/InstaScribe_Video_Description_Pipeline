"""S3 — duration-constrained maximum-sum temporal proposal.

Two deliberately distinct quantities (never conflated):

- ``accumulated_contrast`` — sum of contrast scores inside a window — is the
  SEARCH objective the constrained maximization actually optimizes;
- ``contrast_margin`` — mean(contrast inside) minus mean(contrast outside) —
  is the E002 DETECTION statistic and confidence input.

The search is the deterministic O(n) prefix-sum + monotonic-deque formulation
of the duration-constrained maximum-sum subarray (NOT unconstrained Kadane
followed by clipping): maximize ``prefix[i] - min(prefix[j])`` over
``i - L_max <= j <= i - L_min``.

S3 proposes; it never decides event existence — if at least one admissible
window exists, the best admissible proposal is always emitted regardless of
its score's sign.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor

from .config import RetrieveSolverConfig


@dataclass(frozen=True, slots=True)
class TemporalProposal:
    start_s: float
    end_s: float
    start_index: int  # inclusive sampled-frame index
    end_index: int  # inclusive sampled-frame index
    accumulated_contrast: float
    contrast_margin: float  # 0.0 by convention when the window covers all frames
    rank: int


def frame_length_bounds(config: RetrieveSolverConfig, n_frames: int) -> tuple[int, int, bool]:
    """Window-length bounds in frames, plus the short-video degradation flag.

    ``L_min = ceil(min_s * fps)``, ``L_max = floor(max_s * fps)``. If the
    bounds admit no length at this fps the config is unusable: fail closed.
    If the video itself is shorter than the minimum, the constraint degrades
    to "the whole video is the only admissible window" (flagged, logged).
    """

    if n_frames < 1:
        raise ValueError("at least one sampled frame is required")
    l_min = max(1, ceil(config.min_proposal_duration_s * config.sampling_fps))
    l_max = floor(config.max_proposal_duration_s * config.sampling_fps)
    if l_max < l_min:
        raise ValueError(
            "proposal duration bounds admit no window length at "
            f"sampling_fps={config.sampling_fps} (L_min={l_min}, L_max={l_max})"
        )
    if l_min > n_frames:
        return n_frames, n_frames, True
    return l_min, min(l_max, n_frames), False


def constrained_max_subarray(
    values: Sequence[float], l_min: int, l_max: int
) -> tuple[int, int, float]:
    """Best window of length in [l_min, l_max]: (start, end_inclusive, sum).

    Deterministic tie-break: highest sum, then earliest start, then shortest
    duration. O(n): the deque keeps admissible prefix-minimum candidates in
    increasing prefix order; equal prefixes keep the EARLIER index (strict
    ``>`` pop), so the front is always the earliest admissible minimum.
    """

    n = len(values)
    if not 1 <= l_min <= l_max <= n:
        raise ValueError("window bounds must satisfy 1 <= l_min <= l_max <= n")
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    best: tuple[float, int, int] | None = None  # (sum, start, length)
    candidates: deque[int] = deque()
    for i in range(l_min, n + 1):
        entering = i - l_min
        while candidates and prefix[candidates[-1]] > prefix[entering]:
            candidates.pop()
        candidates.append(entering)
        while candidates[0] < i - l_max:
            candidates.popleft()
        start = candidates[0]
        total = prefix[i] - prefix[start]
        length = i - start
        if (
            best is None
            or total > best[0]
            or (total == best[0] and start < best[1])
            or (total == best[0] and start == best[1] and length < best[2])
        ):
            best = (total, start, length)
    assert best is not None  # loop runs at least once because l_min <= n
    total, start, length = best
    return start, start + length - 1, total


def contrast_margin(values: Sequence[float], start: int, end_inclusive: int) -> float:
    """mean(inside) - mean(outside); 0.0 when there is no outside region.

    The 0.0 is the frozen neutral convention ("no evidence either way"), not
    a mathematical estimate — it fires only for whole-video windows.
    """

    inside = values[start : end_inclusive + 1]
    outside_count = len(values) - len(inside)
    if outside_count == 0:
        return 0.0
    outside_sum = sum(values) - sum(inside)
    return (sum(inside) / len(inside)) - (outside_sum / outside_count)


def map_interval(
    start_index: int, end_index: int, *, sampling_fps: float, video_duration_s: float
) -> tuple[float, float]:
    """Frozen frame-index -> temporal-boundary mapping (normative).

    Each sampled frame represents one sample-time cell:

        start_s = i_start / sampling_fps
        end_s   = min((i_end + 1) / sampling_fps, video_duration_s)

    so a one-frame proposal has positive duration when the video permits.
    Violating 0 <= start_s <= end_s <= duration is a solver bug: raise.
    """

    start_s = start_index / sampling_fps
    end_s = min((end_index + 1) / sampling_fps, video_duration_s)
    if not 0.0 <= start_s <= end_s <= video_duration_s:
        raise ValueError(
            f"interval mapping violated its invariant: start={start_s}, "
            f"end={end_s}, duration={video_duration_s}"
        )
    return start_s, end_s


def best_proposal(
    contrast_scores: Sequence[float],
    *,
    config: RetrieveSolverConfig,
    video_duration_s: float,
) -> tuple[TemporalProposal, bool]:
    """The top-1 admissible proposal, plus the whole-video degradation flag."""

    l_min, l_max, degraded = frame_length_bounds(config, len(contrast_scores))
    start, end, total = constrained_max_subarray(contrast_scores, l_min, l_max)
    margin = contrast_margin(contrast_scores, start, end)
    start_s, end_s = map_interval(
        start, end, sampling_fps=config.sampling_fps, video_duration_s=video_duration_s
    )
    proposal = TemporalProposal(
        start_s=start_s,
        end_s=end_s,
        start_index=start,
        end_index=end,
        accumulated_contrast=total,
        contrast_margin=margin,
        rank=1,
    )
    return proposal, degraded
