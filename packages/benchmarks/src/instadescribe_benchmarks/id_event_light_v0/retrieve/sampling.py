"""S0 — deterministic video sampling for the E002 retrieve solver.

One ffmpeg pass per video writes numbered JPEG frames at ``sampling_fps``.
Frame timestamps are SAMPLE-TIMELINE times (``index / sampling_fps``), a
deterministic solver-defined timeline — not a claim about the source
container's presentation timestamps.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

_SUBPROCESS_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class SampledFrame:
    index: int
    timestamp_s: float
    path: Path


def _which(tool: str) -> str:
    resolved = shutil.which(tool)
    if resolved is None:
        raise ValueError(f"{tool} is required and was not found on PATH")
    return resolved


def probe_duration_s(video_path: Path) -> float:
    """Container duration in seconds via ffprobe; fails closed on anything odd."""

    ffprobe = _which("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:400]
        raise ValueError(f"ffprobe failed for {video_path}: {detail}")
    text = result.stdout.decode("utf-8", "replace").strip()
    try:
        duration = float(text)
    except ValueError as error:
        raise ValueError(
            f"ffprobe returned a non-numeric duration for {video_path}: {text!r}"
        ) from error
    if not isfinite(duration) or duration <= 0:
        raise ValueError(f"unusable video duration {duration!r} for {video_path}")
    return duration


def sample_frames(
    video_path: Path, workspace: Path, *, sampling_fps: float
) -> tuple[SampledFrame, ...]:
    """Decode ``video_path`` into numbered JPEGs in ``workspace``.

    Deterministic ordering: ffmpeg numbers frames sequentially from 0 and we
    sort by that number; ``timestamp_s = index / sampling_fps``.
    """

    ffmpeg = _which("ffmpeg")
    workspace.mkdir(parents=True, exist_ok=True)
    pattern = workspace / "frame_%06d.jpg"
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sampling_fps}",
            "-q:v",
            "3",
            "-start_number",
            "0",
            str(pattern),
        ],
        capture_output=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:400]
        raise ValueError(f"ffmpeg failed to decode {video_path}: {detail}")

    paths = sorted(workspace.glob("frame_*.jpg"))
    if not paths:
        raise ValueError(f"ffmpeg produced no frames for {video_path}")
    frames = []
    for index, path in enumerate(paths):
        expected = workspace / f"frame_{index:06d}.jpg"
        if path != expected:
            # A gap in the numbering means the decode was not the single
            # contiguous pass this contract requires.
            raise ValueError(f"frame numbering gap: expected {expected.name}, found {path.name}")
        frames.append(SampledFrame(index=index, timestamp_s=index / sampling_fps, path=path))
    return tuple(frames)
