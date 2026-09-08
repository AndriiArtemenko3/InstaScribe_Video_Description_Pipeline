"""E002 S2: unit vectors, cosine, box smoothing, contrast."""

from __future__ import annotations

import pytest

from instadescribe_benchmarks.id_event_light_v0.retrieve.curve import (
    FrameEmbedding,
    box_smooth,
    build_score_curve,
    contrast,
    unit_cosine,
    unit_vector,
)


def test_unit_vector_normalizes() -> None:
    assert unit_vector((3.0, 4.0)) == pytest.approx((0.6, 0.8))


def test_unit_vector_rejects_zero_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="positive L2 norm"):
        unit_vector((0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        unit_vector((1.0, float("nan")))
    with pytest.raises(ValueError, match="empty"):
        unit_vector(())


def test_unit_cosine_known_values() -> None:
    assert unit_cosine((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert unit_cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert unit_cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="dimensions differ"):
        unit_cosine((1.0,), (1.0, 0.0))


def test_box_smooth_known_values() -> None:
    # w=3 centered; boundary windows average only the available samples.
    assert box_smooth([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])


def test_box_smooth_width_one_is_identity() -> None:
    values = [0.4, -0.2, 0.9]
    assert box_smooth(values, 1) == pytest.approx(values)


def test_box_smooth_rejects_even_or_nonpositive_width() -> None:
    for width in (0, -1, 2, 4):
        with pytest.raises(ValueError, match="odd"):
            box_smooth([1.0, 2.0], width)


def test_box_smooth_wide_window_truncates_to_available() -> None:
    # Window wider than the sequence: every output is the global mean.
    assert box_smooth([1.0, 2.0, 3.0], 21) == pytest.approx([2.0, 2.0, 2.0])


def test_contrast_sums_to_zero() -> None:
    values = contrast([0.2, 0.4, 0.9, 0.1])
    assert sum(values) == pytest.approx(0.0, abs=1e-12)


def test_build_score_curve_deterministic_and_complete() -> None:
    query = (1.0, 0.0)
    frames = [
        FrameEmbedding(0.0, (1.0, 0.0)),  # cosine 1
        FrameEmbedding(0.5, (0.0, 1.0)),  # cosine 0
        FrameEmbedding(1.0, (1.0, 0.0)),  # cosine 1
    ]
    curve = build_score_curve(query, frames, smoothing_width=1)
    assert [point.raw_similarity for point in curve] == pytest.approx([1.0, 0.0, 1.0])
    # w=1: smoothed == raw; contrast subtracts the mean 2/3.
    assert [point.contrast_score for point in curve] == pytest.approx([1 / 3, -2 / 3, 1 / 3])
    assert curve == build_score_curve(query, frames, smoothing_width=1)


def test_build_score_curve_rejects_empty() -> None:
    with pytest.raises(ValueError, match="without frames"):
        build_score_curve((1.0, 0.0), [], smoothing_width=1)
