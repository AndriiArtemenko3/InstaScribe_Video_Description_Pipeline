"""S2 — unit vectors, box smoothing, and the contrast score curve.

Vector rule: both towers are L2-normalized once at embed time, so every
similarity below is a plain dot product of unit vectors — mathematically the
cosine. Cosine is already scale-invariant; normalizing changes no value or
ranking. It is done anyway so the representation invariant is explicit and a
future implementation can never accidentally mix raw dot products with
cosine-scored values.

Everything downstream of this module consumes ``contrast_score`` (smoothed
minus the video mean), never absolute similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class FrameEmbedding:
    timestamp_s: float
    vector: tuple[float, ...]  # L2-normalized (unit) vector


@dataclass(frozen=True, slots=True)
class ScorePoint:
    timestamp_s: float
    raw_similarity: float
    smoothed_score: float
    contrast_score: float


def unit_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize; fails closed on empty, non-finite, or zero-norm input."""

    if not vector:
        raise ValueError("embedding must not be empty")
    if any(not isfinite(value) for value in vector):
        raise ValueError("embedding must be finite")
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("embedding must have a positive L2 norm")
    return tuple(value / norm for value in vector)


def unit_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two unit vectors == their cosine similarity.

    Callers must pass unit vectors (the providers guarantee it); the clamp
    only absorbs float rounding at the boundaries.
    """

    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


def box_smooth(values: Sequence[float], width: int) -> list[float]:
    """Centered box mean of odd ``width``; boundary windows use only the
    available samples (no zero padding, no reflection) — the frozen v0
    convention, so independent implementations cannot disagree.
    """

    if width < 1 or width % 2 == 0:
        raise ValueError("smoothing width must be an odd integer >= 1")
    half = width // 2
    smoothed = []
    for index in range(len(values)):
        lo = max(0, index - half)
        hi = min(len(values), index + half + 1)
        window = values[lo:hi]
        smoothed.append(sum(window) / len(window))
    return smoothed


def contrast(values: Sequence[float]) -> list[float]:
    """Subtract the mean: background suppression. The result sums to ~0."""

    if not values:
        raise ValueError("cannot compute contrast of an empty sequence")
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def build_score_curve(
    query_vector: Sequence[float],
    frames: Sequence[FrameEmbedding],
    *,
    smoothing_width: int,
) -> tuple[ScorePoint, ...]:
    """raw cosine -> box smooth -> mean-subtracted contrast, per frame."""

    if not frames:
        raise ValueError("cannot build a score curve without frames")
    raw = [unit_cosine(query_vector, frame.vector) for frame in frames]
    smoothed = box_smooth(raw, smoothing_width)
    contrasted = contrast(smoothed)
    return tuple(
        ScorePoint(
            timestamp_s=frame.timestamp_s,
            raw_similarity=raw[index],
            smoothed_score=smoothed[index],
            contrast_score=contrasted[index],
        )
        for index, frame in enumerate(frames)
    )
