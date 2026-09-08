"""E001 naive visual baseline for ID Event Light Bench v0.

A deliberately primitive, query-agnostic appearance-change detector. It cannot
interpret the natural-language query at all — which is exactly why it is the
right floor: it measures how far simple pixel statistics get before any
semantics are needed. Every later system must beat this on the benchmark,
especially on hard-negative false-positive rate, to justify its complexity.

Pipeline, one primitive operation per stage (ffmpeg does all pixel decoding so
this package stays dependency-free; everything after decode is pure Python):

1. sample:   one ffmpeg pass -> grayscale frames, ``frame_size`` square, at
             ``sampling_fps`` (raw bytes, one byte per pixel)
2. scale:    pixel values to [0, 1]  (per-video mean subtraction is omitted
             on purpose: the Sobel gradients in stage 3 are invariant to a
             constant offset, so subtracting the mean would be a no-op)
3. edges:    hand-written 3x3 Sobel convolution -> per-pixel edge magnitude
4. features: mean edge energy over a ``grid_cells`` x ``grid_cells`` grid ->
             one small non-negative feature vector per frame
5. change:   change(i) = 1 - cosine(features[i], features[i+1]) in [0, 1]
6. squash:   confidence(i) = sigmoid((change(i) - sigmoid_bias) / sigmoid_scale)
7. decide:   event_detected = peak confidence >= detection_threshold
8. localize: the contiguous run of steps with confidence >= threshold that
             contains the peak step (earliest peak on ties); change step i
             involves frames i and i+1, so the run [s, e] maps to
             start_time = s / sampling_fps, end_time = (e + 1) / sampling_fps

The emitted ``confidence`` is always the peak step confidence — an
uncalibrated positive-event score (larger => stronger evidence the event
occurred), reported for negatives too.

All sigmoid and threshold defaults are uncalibrated placeholders. They were
not tuned against any data, and no accuracy or performance result produced
with them may be reported.

Usage:
    python -m instadescribe_benchmarks.id_event_light_v0.baselines.naive_visual \\
        --manifest <manifest.json> --out <predictions.json>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import exp, isfinite, sqrt
from pathlib import Path

from ..schema import PREDICTIONS_SCHEMA_VERSION, BenchmarkItem, load_manifest

BASELINE_NAME = "e001-naive-visual"
_FFMPEG_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class NaiveBaselineConfig:
    """Every scalar the baseline uses; validated fail-closed on construction.

    The defaults are arbitrary, uncalibrated placeholders — deliberately NOT
    tuned against the sample data. Honest tuning happens only on a dev split
    of real, rights-cleared clips.
    """

    sampling_fps: float = 2.0
    frame_size: int = 64
    grid_cells: int = 4
    sigmoid_bias: float = 0.1
    sigmoid_scale: float = 0.05
    detection_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("sampling_fps", "sigmoid_bias", "sigmoid_scale", "detection_threshold"):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be > 0")
        if self.sigmoid_scale <= 0:
            raise ValueError("sigmoid_scale must be > 0")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be within [0, 1]")
        if self.grid_cells < 1:
            raise ValueError("grid_cells must be >= 1")
        # Sobel needs an interior, and the grid must tile the frame exactly.
        if self.frame_size < 3 or self.frame_size % self.grid_cells != 0:
            raise ValueError("frame_size must be >= 3 and divisible by grid_cells")


def decode_frames(video_path: Path, *, config: NaiveBaselineConfig) -> list[bytes]:
    """Decode a video into raw grayscale frames via one ffmpeg pass.

    Returns one ``bytes`` object of length frame_size**2 per sampled frame
    (row-major, one byte per pixel). Fails closed on a missing decoder,
    a decode error, or a truncated stream.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("ffmpeg is required to decode video and was not found on PATH")
    size = config.frame_size
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-vf",
        f"fps={config.sampling_fps},scale={size}:{size},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(
        command, capture_output=True, timeout=_FFMPEG_TIMEOUT_SECONDS, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:400]
        raise ValueError(f"ffmpeg failed to decode {video_path}: {detail}")
    frame_bytes = size * size
    if len(result.stdout) == 0 or len(result.stdout) % frame_bytes != 0:
        raise ValueError(
            f"ffmpeg produced a truncated frame stream for {video_path} "
            f"({len(result.stdout)} bytes, frame size {frame_bytes})"
        )
    return [
        result.stdout[offset : offset + frame_bytes]
        for offset in range(0, len(result.stdout), frame_bytes)
    ]


def sobel_edge_magnitude(pixels: Sequence[float], size: int) -> list[float]:
    """Per-pixel gradient magnitude from the two 3x3 Sobel kernels.

        Gx = [-1 0 +1; -2 0 +2; -1 0 +1]    Gy = Gx transposed
        magnitude = sqrt(gx^2 + gy^2)

    Computed for interior pixels only; the one-pixel border is left at 0.0 so
    the output needs no padding convention.
    """

    if len(pixels) != size * size:
        raise ValueError("pixel buffer does not match the declared frame size")
    edges = [0.0] * (size * size)
    for row in range(1, size - 1):
        for col in range(1, size - 1):
            top = (row - 1) * size + col
            mid = row * size + col
            bot = (row + 1) * size + col
            gx = (
                -pixels[top - 1]
                + pixels[top + 1]
                - 2.0 * pixels[mid - 1]
                + 2.0 * pixels[mid + 1]
                - pixels[bot - 1]
                + pixels[bot + 1]
            )
            gy = (
                -pixels[top - 1]
                - 2.0 * pixels[top]
                - pixels[top + 1]
                + pixels[bot - 1]
                + 2.0 * pixels[bot]
                + pixels[bot + 1]
            )
            edges[mid] = sqrt(gx * gx + gy * gy)
    return edges


def grid_mean_energy(edges: Sequence[float], size: int, grid_cells: int) -> tuple[float, ...]:
    """Mean edge magnitude per cell of a grid_cells x grid_cells grid.

    This is the deliberate lossy summary: everything the grid does not encode
    is discarded. All entries are >= 0 by construction.
    """

    cell = size // grid_cells
    features: list[float] = []
    for cell_row in range(grid_cells):
        for cell_col in range(grid_cells):
            total = 0.0
            for row in range(cell_row * cell, (cell_row + 1) * cell):
                base = row * size + cell_col * cell
                total += sum(edges[base : base + cell])
            features.append(total / (cell * cell))
    return tuple(features)


def frame_features(frame: bytes, *, config: NaiveBaselineConfig) -> tuple[float, ...]:
    """Stages 2-4 for one frame: scale to [0, 1], Sobel, grid pooling."""

    pixels = [value / 255.0 for value in frame]
    edges = sobel_edge_magnitude(pixels, config.frame_size)
    return grid_mean_energy(edges, config.frame_size, config.grid_cells)


def change_score(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity between two non-negative feature vectors, in [0, 1].

    Zero-norm convention (featureless frames have zero edge energy):
    both zero -> 0.0 (nothing changed between two blank frames);
    exactly one zero -> 1.0 (structure appeared or vanished: maximal change).
    """

    norm_a = sqrt(sum(value * value for value in a))
    norm_b = sqrt(sum(value * value for value in b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine = sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)
    # Non-negative vectors give cosine in [0, 1]; clamp guards float error.
    return min(1.0, max(0.0, 1.0 - cosine))


def sigmoid(x: float) -> float:
    """Numerically stable logistic function 1 / (1 + e^-x)."""

    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    return exp(x) / (1.0 + exp(x))


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    """Deterministic outcome of stages 5-8 over one video's frames."""

    change_scores: tuple[float, ...]
    confidences: tuple[float, ...]
    event_detected: bool
    start_time: float | None
    end_time: float | None
    confidence: float


def analyse_frames(frames: Sequence[bytes], *, config: NaiveBaselineConfig) -> FrameAnalysis:
    """Run stages 2-8 on decoded frames. Pure and deterministic."""

    features = [frame_features(frame, config=config) for frame in frames]
    changes = tuple(change_score(features[i], features[i + 1]) for i in range(len(features) - 1))
    confidences = tuple(
        sigmoid((change - config.sigmoid_bias) / config.sigmoid_scale) for change in changes
    )
    if not confidences:
        # Fewer than two sampled frames: no change is observable at all.
        return FrameAnalysis(changes, confidences, False, None, None, 0.0)

    peak_confidence = max(confidences)
    # Earliest peak on ties (max() over the raw tuple already scans left to
    # right, but the index lookup makes the tie-break explicit).
    peak_index = confidences.index(peak_confidence)
    if peak_confidence < config.detection_threshold:
        return FrameAnalysis(changes, confidences, False, None, None, peak_confidence)

    # Localization: grow the contiguous >=threshold run around the peak step.
    run_start = peak_index
    while run_start > 0 and confidences[run_start - 1] >= config.detection_threshold:
        run_start -= 1
    run_end = peak_index
    while run_end + 1 < len(confidences) and confidences[run_end + 1] >= config.detection_threshold:
        run_end += 1

    # Change step i involves frames i and i+1: run [s, e] spans those frames'
    # sample-timeline cells.
    start_time = run_start / config.sampling_fps
    end_time = (run_end + 1) / config.sampling_fps
    return FrameAnalysis(changes, confidences, True, start_time, end_time, peak_confidence)


def solve_item(
    item: BenchmarkItem, *, manifest_dir: Path, config: NaiveBaselineConfig
) -> dict[str, object]:
    """Produce one benchmark prediction record for one manifest item."""

    frames = decode_frames(manifest_dir / item.video, config=config)
    analysis = analyse_frames(frames, config=config)
    return {
        "id": item.item_id,
        "event_detected": analysis.event_detected,
        "start_time": analysis.start_time,
        "end_time": analysis.end_time,
        "confidence": analysis.confidence,
    }


def build_predictions(
    items: Sequence[BenchmarkItem], *, manifest_dir: Path, config: NaiveBaselineConfig
) -> dict[str, object]:
    """Assemble the full predictions document, including provenance."""

    return {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "system": {
            "name": BASELINE_NAME,
            "algorithm": (
                "uniform grayscale sampling; 3x3 Sobel edge magnitude; grid mean edge "
                "energy; 1 - cosine change between consecutive feature vectors; sigmoid "
                "confidence; threshold gate with contiguous-run localization"
            ),
            "config": asdict(config),
            "note": (
                "Deliberately primitive query-agnostic floor baseline. Sigmoid and "
                "threshold parameters are uncalibrated placeholders; no accuracy or "
                "performance result derived from them may be reported."
            ),
        },
        "predictions": [
            solve_item(item, manifest_dir=manifest_dir, config=config) for item in items
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="benchmark manifest JSON")
    parser.add_argument("--out", type=Path, required=True, help="predictions JSON to write")
    args = parser.parse_args(argv)

    config = NaiveBaselineConfig()
    try:
        # The solver reads pixels, so video files must actually exist.
        items = load_manifest(args.manifest, check_video_files=True)
        document = build_predictions(
            items, manifest_dir=args.manifest.resolve().parent, config=config
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
