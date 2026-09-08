"""E002 config scalars and paired-encoder provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from instadescribe_benchmarks.id_event_light_v0.retrieve.config import (
    EncoderFile,
    PairedEncoderProvenance,
    RetrieveSolverConfig,
    load_solver_toml,
)

FAMILY = "CLIP ViT-B/32"
REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"


def encoder_file(path: Path, content: bytes, **overrides: str) -> EncoderFile:
    path.write_bytes(content)
    fields: dict[str, str] = {
        "expected_digest": hashlib.sha256(content).hexdigest(),
        "model_family": FAMILY,
        "export_revision": REVISION,
    }
    fields.update(overrides)
    return EncoderFile(path=path, **fields)  # type: ignore[arg-type]


def test_config_scalar_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="sampling_fps"):
        RetrieveSolverConfig(sampling_fps=0.0)
    with pytest.raises(ValueError, match="odd"):
        RetrieveSolverConfig(smoothing_width=4)
    with pytest.raises(ValueError, match="min_proposal_duration_s"):
        RetrieveSolverConfig(min_proposal_duration_s=0.0)
    with pytest.raises(ValueError, match="min_proposal_duration_s"):
        RetrieveSolverConfig(min_proposal_duration_s=5.0, max_proposal_duration_s=1.0)
    with pytest.raises(ValueError, match="finite"):
        RetrieveSolverConfig(tau_e002=float("inf"))
    with pytest.raises(ValueError, match="alpha must be > 0"):
        RetrieveSolverConfig(alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        RetrieveSolverConfig(alpha=float("nan"))


def test_paired_provenance_accepts_correct_metadata(tmp_path: Path) -> None:
    vision = encoder_file(tmp_path / "vision.onnx", b"vision-bytes")
    text = encoder_file(tmp_path / "text.onnx", b"text-bytes")
    pair = PairedEncoderProvenance(vision=vision, text=text)
    pair.verify_files()  # both files hash to their OWN pinned digests
    assert pair.model_family == FAMILY
    assert pair.export_revision == REVISION
    # The two files are different, so their digests differ — and that is fine.
    assert vision.expected_digest != text.expected_digest


def test_paired_provenance_rejects_mismatched_family(tmp_path: Path) -> None:
    vision = encoder_file(tmp_path / "vision.onnx", b"vision-bytes")
    text = encoder_file(tmp_path / "text.onnx", b"text-bytes", model_family="SigLIP 2")
    with pytest.raises(ValueError, match="different model families"):
        PairedEncoderProvenance(vision=vision, text=text)


def test_paired_provenance_rejects_mismatched_revision(tmp_path: Path) -> None:
    vision = encoder_file(tmp_path / "vision.onnx", b"vision-bytes")
    text = encoder_file(tmp_path / "text.onnx", b"text-bytes", export_revision="deadbeef")
    with pytest.raises(ValueError, match="different export revisions"):
        PairedEncoderProvenance(vision=vision, text=text)


def test_paired_provenance_rejects_wrong_vision_digest(tmp_path: Path) -> None:
    vision = encoder_file(tmp_path / "vision.onnx", b"vision-bytes", expected_digest="0" * 64)
    text = encoder_file(tmp_path / "text.onnx", b"text-bytes")
    with pytest.raises(ValueError, match="vision model digest mismatch"):
        PairedEncoderProvenance(vision=vision, text=text).verify_files()


def test_paired_provenance_rejects_wrong_text_digest(tmp_path: Path) -> None:
    vision = encoder_file(tmp_path / "vision.onnx", b"vision-bytes")
    text = encoder_file(tmp_path / "text.onnx", b"text-bytes", expected_digest="0" * 64)
    with pytest.raises(ValueError, match="text model digest mismatch"):
        PairedEncoderProvenance(vision=vision, text=text).verify_files()


def test_encoder_file_rejects_malformed_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64-character hex"):
        EncoderFile(
            path=tmp_path / "x.onnx",
            expected_digest="not-a-digest",
            model_family=FAMILY,
            export_revision=REVISION,
        )


def test_load_solver_toml_round_trip(tmp_path: Path) -> None:
    vision = tmp_path / "vision.onnx"
    text = tmp_path / "text.onnx"
    tokenizer = tmp_path / "tokenizer.json"
    config_path = tmp_path / "solver.toml"
    config_path.write_text(
        f'''
[solver]
sampling_fps = 2.0
smoothing_width = 5
min_proposal_duration_s = 0.5
max_proposal_duration_s = 4.0
tau_e002 = 0.01
alpha = 2.0

[models]
model_family = "{FAMILY}"
export_revision = "{REVISION}"
vision_path = "{vision}"
vision_digest = "{"a" * 64}"
text_path = "{text}"
text_digest = "{"b" * 64}"
tokenizer_path = "{tokenizer}"
tokenizer_digest = "{"c" * 64}"
'''
    )
    config, provenance, tokenizer_artifact = load_solver_toml(config_path)
    assert config.smoothing_width == 5
    assert provenance.vision.expected_digest == "a" * 64
    assert provenance.text.expected_digest == "b" * 64
    assert tokenizer_artifact.path == tokenizer
    assert tokenizer_artifact.expected_digest == "c" * 64


def test_tokenizer_artifact_digest_verification(tmp_path: Path) -> None:
    import hashlib as _hashlib

    from instadescribe_benchmarks.id_event_light_v0.retrieve.config import TokenizerArtifact

    path = tmp_path / "tokenizer.json"
    path.write_bytes(b"tokenizer-bytes")
    good = TokenizerArtifact(
        path=path, expected_digest=_hashlib.sha256(b"tokenizer-bytes").hexdigest()
    )
    good.verify_file()
    bad = TokenizerArtifact(path=path, expected_digest="0" * 64)
    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        bad.verify_file()


def test_load_solver_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "solver.toml"
    config_path.write_text("[solver]\ntypo_key = 1\n")
    with pytest.raises(ValueError, match=r"unknown \[solver\] keys"):
        load_solver_toml(config_path)
