"""E001 naive visual baseline: pure stages, decisions, and the CLI contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from instadescribe_benchmarks.id_event_light_v0.baselines.naive_visual import (
    FrameAnalysis,
    NaiveBaselineConfig,
    analyse_frames,
    change_score,
    decode_frames,
    frame_features,
    grid_mean_energy,
    main,
    sigmoid,
    sobel_edge_magnitude,
)
from instadescribe_benchmarks.id_event_light_v0.schema import load_manifest, load_predictions

FFMPEG = shutil.which("ffmpeg")

# A small config with an explicit, test-owned decision geometry. Values are
# chosen for arithmetic clarity, not tuned against any data.
CONFIG = NaiveBaselineConfig(
    sampling_fps=2.0,
    frame_size=8,
    grid_cells=2,
    sigmoid_bias=0.5,
    sigmoid_scale=0.1,
    detection_threshold=0.5,
)


def blank_frame(size: int = 8) -> bytes:
    return bytes(size * size)


def edge_frame(size: int = 8) -> bytes:
    # Left half black, right half white: a strong vertical edge.
    row = bytes(size // 2) + bytes([255] * (size // 2))
    return row * size


def test_config_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="sampling_fps"):
        NaiveBaselineConfig(sampling_fps=0.0)
    with pytest.raises(ValueError, match="finite"):
        NaiveBaselineConfig(sigmoid_bias=float("nan"))
    with pytest.raises(ValueError, match="sigmoid_scale"):
        NaiveBaselineConfig(sigmoid_scale=0.0)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        NaiveBaselineConfig(detection_threshold=1.5)
    with pytest.raises(ValueError, match="divisible"):
        NaiveBaselineConfig(frame_size=10, grid_cells=4)


def test_sobel_flat_frame_has_zero_energy() -> None:
    edges = sobel_edge_magnitude([0.5] * 64, 8)
    assert edges == [0.0] * 64  # constant input: both gradients vanish


def test_sobel_vertical_edge_concentrates_energy() -> None:
    pixels = [value / 255.0 for value in edge_frame()]
    edges = sobel_edge_magnitude(pixels, 8)
    # Energy sits on the boundary columns (3 and 4), interior rows only.
    assert max(edges) > 0.0
    hottest = edges.index(max(edges))
    assert hottest % 8 in (3, 4)
    # The one-pixel border stays exactly zero.
    assert all(edges[col] == 0.0 for col in range(8))


def test_grid_mean_energy_known_values() -> None:
    # 4x4 frame, 2x2 grid: each cell averages its own 2x2 block.
    edges = [float(index) for index in range(16)]
    features = grid_mean_energy(edges, 4, 2)
    assert features == (
        (0 + 1 + 4 + 5) / 4,
        (2 + 3 + 6 + 7) / 4,
        (8 + 9 + 12 + 13) / 4,
        (10 + 11 + 14 + 15) / 4,
    )


def test_change_score_identical_frames_is_zero() -> None:
    features = frame_features(edge_frame(), config=CONFIG)
    assert change_score(features, features) == pytest.approx(0.0, abs=1e-12)


def test_change_score_zero_norm_conventions() -> None:
    assert change_score((0.0, 0.0), (0.0, 0.0)) == 0.0  # blank -> blank: no change
    assert change_score((0.0, 0.0), (1.0, 2.0)) == 1.0  # structure appears: max change


def test_sigmoid_is_stable_and_monotonic() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)  # no overflow
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(1.0) > sigmoid(0.5) > sigmoid(-0.5)


def test_analyse_detects_abrupt_change_and_localizes_it() -> None:
    # Steps: blank->blank (0.0), blank->edge (1.0), edge->edge (0.0).
    frames = [blank_frame(), blank_frame(), edge_frame(), edge_frame()]
    analysis = analyse_frames(frames, config=CONFIG)
    assert analysis.change_scores == pytest.approx((0.0, 1.0, 0.0))
    assert analysis.event_detected is True
    # Only step 1 crosses the gate: run [1, 1] -> [1/fps, 2/fps] at 2 fps.
    assert analysis.start_time == pytest.approx(0.5)
    assert analysis.end_time == pytest.approx(1.0)
    assert analysis.confidence == pytest.approx(sigmoid((1.0 - 0.5) / 0.1))


def test_analyse_static_video_is_negative_with_low_confidence() -> None:
    analysis = analyse_frames([edge_frame()] * 5, config=CONFIG)
    assert analysis.event_detected is False
    assert analysis.start_time is None and analysis.end_time is None
    # Positive-event score orientation: still populated, just small.
    assert 0.0 < analysis.confidence < 0.5


def test_analyse_fewer_than_two_frames_is_negative() -> None:
    for frames in ([], [blank_frame()]):
        analysis = analyse_frames(frames, config=CONFIG)
        assert analysis == FrameAnalysis((), (), False, None, None, 0.0)


def test_analyse_run_extends_around_peak() -> None:
    # Steps: 1.0, 1.0, 0.0 -> both above-threshold steps join one run [0, 1].
    frames = [blank_frame(), edge_frame(), blank_frame(), blank_frame()]
    analysis = analyse_frames(frames, config=CONFIG)
    assert analysis.event_detected is True
    assert analysis.start_time == pytest.approx(0.0)
    assert analysis.end_time == pytest.approx(1.0)


def test_analyse_is_deterministic() -> None:
    frames = [blank_frame(), edge_frame(), blank_frame(), edge_frame()]
    assert analyse_frames(frames, config=CONFIG) == analyse_frames(frames, config=CONFIG)


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_decode_frames_shapes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=64x64:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        timeout=60,
    )
    config = NaiveBaselineConfig(sampling_fps=2.0, frame_size=8, grid_cells=2)
    frames = decode_frames(video, config=config)
    # 3 s at 2 fps: allow for ffmpeg's boundary rounding.
    assert 5 <= len(frames) <= 7
    assert all(len(frame) == 64 for frame in frames)


def test_decode_frames_missing_file_fails_closed(tmp_path: Path) -> None:
    if FFMPEG is None:
        pytest.skip("ffmpeg not available on PATH")
    with pytest.raises(ValueError, match="ffmpeg failed"):
        decode_frames(tmp_path / "missing.mp4", config=CONFIG)


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_cli_end_to_end_emits_valid_benchmark_predictions(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(videos / "pos.mp4"),
        ],
        check=True,
        timeout=60,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "id-event-light-bench-manifest/0",
                "items": [
                    {
                        "id": "pos-001",
                        "video": "videos/pos.mp4",
                        "query": "Does the test pattern change?",
                        "label": True,
                        "start_time": 0.5,
                        "end_time": 1.5,
                        "difficulty": "positive",
                    }
                ],
            }
        )
    )
    out_path = tmp_path / "predictions.json"
    assert main(["--manifest", str(manifest_path), "--out", str(out_path)]) == 0

    # The output must round-trip through the benchmark's own strict loader.
    predictions, system = load_predictions(out_path)
    assert [p.item_id for p in predictions] == ["pos-001"]
    assert system is not None and system["name"] == "e001-naive-visual"
    assert system["config"]["detection_threshold"] == 0.5
    # No outcome assertion on default thresholds: defaults are uncalibrated
    # placeholders and tests must not turn them into empirical claims.


def test_prediction_record_round_trips_via_schema(tmp_path: Path) -> None:
    # No pixels on disk needed: build a record from analyse_frames directly
    # and prove the emitted shape survives the benchmark's strict loader.
    items = load_manifest(_write_min_manifest(tmp_path), check_video_files=False)
    analysis = analyse_frames([blank_frame(), edge_frame()], config=CONFIG)
    record = {
        "id": items[0].item_id,
        "event_detected": analysis.event_detected,
        "start_time": analysis.start_time,
        "end_time": analysis.end_time,
        "confidence": analysis.confidence,
    }
    payload = {
        "schema_version": "id-event-light-bench-predictions/0",
        "predictions": [record],
    }
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))
    loaded, _ = load_predictions(path)
    assert loaded[0].item_id == items[0].item_id


def _write_min_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "id-event-light-bench-manifest/0",
                "items": [
                    {
                        "id": "pos-001",
                        "video": "videos/pos.mp4",
                        "query": "Does anything change?",
                        "label": True,
                        "start_time": 0.5,
                        "end_time": 1.0,
                        "difficulty": "positive",
                    }
                ],
            }
        )
    )
    return path
