"""S5 — E002 retrieve-only detection, output, and CLI.

The E002 decision rule, exactly and only:

    event_detected = winning_proposal.contrast_margin >= tau_e002
    confidence     = sigmoid(alpha * winning_proposal.contrast_margin)

Confidence is an uncalibrated POSITIVE-EVENT score: larger always means
stronger evidence the event occurred; it is never flipped on a negative
prediction. A negative prediction carries null timestamps and the same
positive-event confidence.

Standalone-solver boundary: this module writes a benchmark predictions JSON
and nothing else — no worker wiring, no evidence, no beliefs, no
calibrated_confidence.

Usage:
    python -m instadescribe_benchmarks.id_event_light_v0.retrieve.solve \\
        --manifest <manifest.json> --out <predictions.json> --config <solver.toml>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..baselines.naive_visual import sigmoid
from ..schema import PREDICTIONS_SCHEMA_VERSION, BenchmarkItem, load_manifest
from .clip_onnx import OnnxClipTextEmbeddingProvider, OnnxClipVisionEmbeddingProvider
from .config import (
    PairedEncoderProvenance,
    RetrieveSolverConfig,
    TokenizerArtifact,
    load_solver_toml,
)
from .curve import FrameEmbedding, build_score_curve
from .proposals import TemporalProposal, best_proposal
from .sampling import probe_duration_s, sample_frames
from .tokenizer import ClipTokenizerAdapter

SOLVER_NAME = "e002-retrieve-visual"


class VisionEmbedder(Protocol):
    def embed_frame(self, frame_path: Path) -> tuple[float, ...]: ...


class TextEmbedder(Protocol):
    def embed_text(self, query: str) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class SolverPrediction:
    event_detected: bool
    start_s: float | None
    end_s: float | None
    confidence: float


def decide(proposal: TemporalProposal, *, config: RetrieveSolverConfig) -> SolverPrediction:
    """The frozen E002 rule; nothing else influences detection."""

    confidence = sigmoid(config.alpha * proposal.contrast_margin)
    if proposal.contrast_margin >= config.tau_e002:
        return SolverPrediction(True, proposal.start_s, proposal.end_s, confidence)
    return SolverPrediction(False, None, None, confidence)


def solve_video(
    video_path: Path,
    query: str,
    *,
    config: RetrieveSolverConfig,
    vision: VisionEmbedder,
    text: TextEmbedder,
    workspace: Path,
) -> tuple[SolverPrediction, dict[str, object]]:
    """Run S0-S3 + S5 for one (video, query) pair.

    Returns the prediction plus per-video diagnostics for the system block.
    """

    duration_s = probe_duration_s(video_path)
    frames = sample_frames(video_path, workspace, sampling_fps=config.sampling_fps)

    query_vector = text.embed_text(query)
    embeddings = []
    for frame in frames:
        vector = vision.embed_frame(frame.path)
        if len(vector) != len(query_vector):
            raise ValueError(
                f"embedding dimension mismatch: query {len(query_vector)}, frame {len(vector)}"
            )
        embeddings.append(FrameEmbedding(timestamp_s=frame.timestamp_s, vector=vector))

    curve = build_score_curve(query_vector, embeddings, smoothing_width=config.smoothing_width)
    proposal, degraded = best_proposal(
        [point.contrast_score for point in curve],
        config=config,
        video_duration_s=duration_s,
    )
    prediction = decide(proposal, config=config)
    diagnostics: dict[str, object] = {
        "frames_sampled": len(frames),
        "whole_video_degradation": degraded,
        "accumulated_contrast": proposal.accumulated_contrast,
        "contrast_margin": proposal.contrast_margin,
    }
    return prediction, diagnostics


def run(
    manifest_path: Path,
    out_path: Path,
    *,
    config: RetrieveSolverConfig,
    vision: VisionEmbedder,
    text: TextEmbedder,
    provenance: PairedEncoderProvenance | None,
    tokenizer: TokenizerArtifact | None = None,
) -> int:
    """Solve every manifest item and write the predictions document."""

    items: Sequence[BenchmarkItem] = load_manifest(manifest_path, check_video_files=True)
    manifest_dir = manifest_path.resolve().parent

    records: list[dict[str, object]] = []
    per_video_diagnostics: dict[str, object] = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="e002-frames-") as scratch:
        for index, item in enumerate(items):
            prediction, diagnostics = solve_video(
                manifest_dir / item.video,
                item.query,
                config=config,
                vision=vision,
                text=text,
                workspace=Path(scratch) / f"item-{index:04d}",
            )
            records.append(
                {
                    "id": item.item_id,
                    "event_detected": prediction.event_detected,
                    "start_time": prediction.start_s,
                    "end_time": prediction.end_s,
                    "confidence": prediction.confidence,
                }
            )
            per_video_diagnostics[item.item_id] = diagnostics
    elapsed = time.monotonic() - started

    if provenance is None:
        models: dict[str, object] = {"status": "injected providers - no verified model provenance"}
    else:
        models = {
            "model_family": provenance.model_family,
            "export_revision": provenance.export_revision,
            "vision_file": provenance.vision.path.name,
            "vision_digest": provenance.vision.expected_digest,
            "text_file": provenance.text.path.name,
            "text_digest": provenance.text.expected_digest,
        }
        if tokenizer is not None:
            models["tokenizer_file"] = tokenizer.path.name
            models["tokenizer_digest"] = tokenizer.expected_digest

    document = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "system": {
            "name": SOLVER_NAME,
            "stage_scope": "S0 decode/sample; S1 paired CLIP embeddings; S2 box-smoothed "
            "contrast curve; S3 duration-constrained max-sum proposal; S5 margin rule",
            "models": models,
            "config": {
                "sampling_fps": config.sampling_fps,
                "smoothing": "box",
                "smoothing_width": config.smoothing_width,
                "min_proposal_duration_s": config.min_proposal_duration_s,
                "max_proposal_duration_s": config.max_proposal_duration_s,
                "candidate_count": 1,
                "transition_decomposition": "disabled",
                "tau_e002": config.tau_e002,
                "alpha": config.alpha,
            },
            "diagnostics": per_video_diagnostics,
            "note": "UNCALIBRATED PLACEHOLDER configuration - NOT A BENCHMARK RESULT. "
            "Thresholds, smoothing width, proposal bounds and alpha await selection "
            "on a dev split of real rights-cleared clips.",
            # Timings isolated under one key so the predictions array stays
            # byte-stable across identical runs.
            "timings": {"total_seconds": round(elapsed, 3)},
        },
        "predictions": records,
    }
    out_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return len(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="benchmark manifest JSON")
    parser.add_argument("--out", type=Path, required=True, help="predictions JSON to write")
    parser.add_argument(
        "--config", type=Path, required=True, help="solver TOML ([solver] + [models])"
    )
    args = parser.parse_args(argv)

    try:
        config, provenance, tokenizer_artifact = load_solver_toml(args.config)
        provenance.verify_files()  # each file against its OWN pinned digest
        tokenizer_artifact.verify_file()
        tokenizer = ClipTokenizerAdapter(tokenizer_artifact.path)
        vision = OnnxClipVisionEmbeddingProvider(provenance.vision.path)
        text = OnnxClipTextEmbeddingProvider(provenance.text.path, tokenize=tokenizer)
        count = run(
            args.manifest,
            args.out,
            config=config,
            vision=vision,
            text=text,
            provenance=provenance,
            tokenizer=tokenizer_artifact,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {count} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
