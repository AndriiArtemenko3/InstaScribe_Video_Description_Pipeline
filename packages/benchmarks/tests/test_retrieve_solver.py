"""E002 S0/S1 contracts, S5 decisions, and the end-to-end predictions output."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from instadescribe_benchmarks.id_event_light_v0.retrieve.clip_onnx import (
    OnnxClipTextEmbeddingProvider,
    OnnxClipVisionEmbeddingProvider,
)
from instadescribe_benchmarks.id_event_light_v0.retrieve.config import RetrieveSolverConfig
from instadescribe_benchmarks.id_event_light_v0.retrieve.proposals import TemporalProposal
from instadescribe_benchmarks.id_event_light_v0.retrieve.sampling import (
    probe_duration_s,
    sample_frames,
)
from instadescribe_benchmarks.id_event_light_v0.retrieve.solve import decide, run
from instadescribe_benchmarks.id_event_light_v0.schema import load_predictions

FFMPEG = shutil.which("ffmpeg")

CONFIG = RetrieveSolverConfig(
    sampling_fps=2.0,
    smoothing_width=1,
    min_proposal_duration_s=0.5,
    max_proposal_duration_s=2.0,
    tau_e002=0.1,  # test-owned value for arithmetic clarity, not a default
    alpha=1.0,
)


def proposal(margin: float) -> TemporalProposal:
    return TemporalProposal(
        start_s=1.0,
        end_s=2.0,
        start_index=2,
        end_index=3,
        accumulated_contrast=margin * 2,
        contrast_margin=margin,
        rank=1,
    )


class FakeVision:
    """Deterministic unit vectors derived from the frame file bytes."""

    def embed_frame(self, frame_path: Path) -> tuple[float, ...]:
        seed = sum(frame_path.read_bytes()[:512]) % 97
        # Angle varies with content; always a 2-d unit vector.
        from math import cos, sin

        return (cos(seed), sin(seed))


class FakeText:
    def embed_text(self, query: str) -> tuple[float, ...]:
        return (1.0, 0.0)


# --- S5 decision rule ------------------------------------------------------


def test_decide_below_threshold_negative_with_null_interval() -> None:
    prediction = decide(proposal(margin=0.05), config=CONFIG)
    assert prediction.event_detected is False
    assert prediction.start_s is None and prediction.end_s is None
    # Positive-event orientation: confidence populated, > 0.5 iff margin > 0.
    assert 0.5 < prediction.confidence < 0.6


def test_decide_at_threshold_is_positive() -> None:
    # The rule is >=, so margin == tau detects.
    prediction = decide(proposal(margin=0.1), config=CONFIG)
    assert prediction.event_detected is True
    assert (prediction.start_s, prediction.end_s) == (1.0, 2.0)


def test_decide_above_threshold_localizes() -> None:
    prediction = decide(proposal(margin=0.9), config=CONFIG)
    assert prediction.event_detected is True
    assert prediction.confidence > 0.7


def test_decide_confidence_orientation_never_flips() -> None:
    low = decide(proposal(margin=-2.0), config=CONFIG)
    high = decide(proposal(margin=2.0), config=CONFIG)
    assert low.event_detected is False and high.event_detected is True
    assert low.confidence < high.confidence  # larger margin => larger confidence


# --- S1 provider contracts (fake sessions; no real model needed) -----------


class _Port:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, inputs: list[str], outputs: list[str], vector: list[float]) -> None:
        self._inputs = [_Port(name) for name in inputs]
        self._outputs = [_Port(name) for name in outputs]
        self._vector = vector
        self.last_feeds: dict[str, Any] | None = None

    def get_inputs(self) -> list[_Port]:
        return self._inputs

    def get_outputs(self) -> list[_Port]:
        return self._outputs

    def run(self, output_names: object, feeds: dict[str, Any]) -> list[Any]:
        self.last_feeds = feeds
        return [[self._vector]]


def onnx_stub(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"stub")
    return path


def test_text_provider_validates_io_contract(tmp_path: Path) -> None:
    bad = FakeSession(["wrong_input"], ["text_embeds"], [1.0, 0.0])
    provider = OnnxClipTextEmbeddingProvider(
        onnx_stub(tmp_path, "text.onnx"),
        tokenize=lambda text: [1, 2, 3],
        session_factory=lambda path: bad,
        tensorize=lambda ids: list(ids),
    )
    with pytest.raises(ValueError, match="not a CLIP text export"):
        provider.embed_text("query")


def test_text_provider_embeds_and_normalizes(tmp_path: Path) -> None:
    session = FakeSession(["input_ids", "attention_mask"], ["text_embeds"], [3.0, 4.0])
    provider = OnnxClipTextEmbeddingProvider(
        onnx_stub(tmp_path, "text.onnx"),
        tokenize=lambda text: [1, 2, 3],
        session_factory=lambda path: session,
        tensorize=lambda ids: list(ids),
    )
    vector = provider.embed_text("query")
    assert vector == pytest.approx((0.6, 0.8))  # unit-normalized at embed time
    assert provider.network_access is False
    assert session.last_feeds is not None and "attention_mask" in session.last_feeds


def test_vision_provider_validates_io_contract(tmp_path: Path) -> None:
    bad = FakeSession(["pixel_values", "extra"], ["image_embeds"], [1.0])
    provider = OnnxClipVisionEmbeddingProvider(
        onnx_stub(tmp_path, "vision.onnx"),
        session_factory=lambda path: bad,
        preprocess=lambda path: "unused",
    )
    with pytest.raises(ValueError, match="not a CLIP vision export"):
        provider.embed_frame(tmp_path / "frame.jpg")


def test_vision_provider_rejects_relative_and_wrong_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        OnnxClipVisionEmbeddingProvider(Path("relative.onnx"))
    not_onnx = tmp_path / "model.bin"
    not_onnx.write_bytes(b"x")
    with pytest.raises(ValueError, match=".onnx"):
        OnnxClipVisionEmbeddingProvider(not_onnx)


# --- S0 sampling (ffmpeg-dependent) ---------------------------------------


def make_video(path: Path, *, duration: int = 3) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=64x64:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        timeout=60,
    )


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_sample_frames_deterministic_indexing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    make_video(video)
    frames = sample_frames(video, tmp_path / "ws", sampling_fps=2.0)
    assert 5 <= len(frames) <= 7
    assert [frame.index for frame in frames] == list(range(len(frames)))
    assert frames[3].timestamp_s == pytest.approx(1.5)  # index / sampling_fps
    # Re-decoding into a fresh workspace yields identical structure.
    again = sample_frames(video, tmp_path / "ws2", sampling_fps=2.0)
    assert [f.timestamp_s for f in again] == [f.timestamp_s for f in frames]


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_sample_frames_missing_video_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ffmpeg failed"):
        sample_frames(tmp_path / "missing.mp4", tmp_path / "ws", sampling_fps=2.0)


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_probe_duration(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    make_video(video, duration=2)
    assert probe_duration_s(video) == pytest.approx(2.0, abs=0.2)
    with pytest.raises(ValueError, match="ffprobe failed"):
        probe_duration_s(tmp_path / "missing.mp4")


# --- End-to-end: run() with injected embedders ----------------------------


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available on PATH")
def test_run_emits_predictions_that_round_trip(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    make_video(videos / "a.mp4")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "id-event-light-bench-manifest/0",
                "items": [
                    {
                        "id": "pos-001",
                        "video": "videos/a.mp4",
                        "query": "Does the pattern change?",
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
    count = run(
        manifest_path,
        out_path,
        config=CONFIG,
        vision=FakeVision(),
        text=FakeText(),
        provenance=None,
    )
    assert count == 1

    predictions, system = load_predictions(out_path)
    assert [p.item_id for p in predictions] == ["pos-001"]
    assert system is not None
    assert system["name"] == "e002-retrieve-visual"
    assert system["config"]["candidate_count"] == 1
    assert system["config"]["transition_decomposition"] == "disabled"
    assert "no verified model provenance" in system["models"]["status"]
    assert "UNCALIBRATED PLACEHOLDER" in system["note"]
    # Timestamps, when present, respect the interval invariant.
    record = predictions[0]
    if record.event_detected:
        assert record.start_time is not None and record.end_time is not None
        assert 0.0 <= record.start_time <= record.end_time
    else:
        assert record.start_time is None and record.end_time is None
