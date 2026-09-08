"""Configuration and paired-encoder provenance for the E002 retrieve solver.

Two rules dominate this module:

- Every scalar used in scoring or decisions is validated fail-closed at
  construction (finite, correctly bounded) — a bad config never produces a
  prediction.
- The image/text same-embedding-space assumption is established by shared
  model family and shared pinned export revision, with each model file
  verified against its OWN pinned sha256. The two files are different files;
  their digests are never compared to each other.

All scalar defaults are UNCALIBRATED PLACEHOLDERS — NOT A BENCHMARK RESULT.
They exist so the mechanics can run; honest values are selected on a dev
split of real rights-cleared clips only.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

_DIGEST_HEX_LENGTH = 64


def _require_finite(value: object, name: str) -> float:
    if not (isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class EncoderFile:
    """One encoder artifact: its location, pinned digest, and declared origin."""

    path: Path
    expected_digest: str
    model_family: str
    export_revision: str

    def __post_init__(self) -> None:
        if not self.model_family.strip():
            raise ValueError("model_family must not be empty")
        if not self.export_revision.strip():
            raise ValueError("export_revision must not be empty")
        digest = self.expected_digest.lower()
        if len(digest) != _DIGEST_HEX_LENGTH or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("expected_digest must be a 64-character hex sha256")


@dataclass(frozen=True, slots=True)
class PairedEncoderProvenance:
    """Fail-closed pairing of the vision and text towers.

    The same-space guarantee comes from shared ``model_family`` and shared
    ``export_revision`` — never from comparing the two file hashes, which are
    necessarily different because the files are different.
    """

    vision: EncoderFile
    text: EncoderFile

    def __post_init__(self) -> None:
        if self.vision.model_family != self.text.model_family:
            raise ValueError(
                "vision and text towers declare different model families: "
                f"{self.vision.model_family!r} vs {self.text.model_family!r}"
            )
        if self.vision.export_revision != self.text.export_revision:
            raise ValueError(
                "vision and text towers declare different export revisions: "
                f"{self.vision.export_revision!r} vs {self.text.export_revision!r}"
            )

    @property
    def model_family(self) -> str:
        return self.vision.model_family

    @property
    def export_revision(self) -> str:
        return self.vision.export_revision

    def verify_files(self) -> None:
        """Hash each file and compare against its OWN pinned digest."""

        for label, entry in (("vision", self.vision), ("text", self.text)):
            actual = _sha256_file(entry.path)
            if actual != entry.expected_digest.lower():
                raise ValueError(
                    f"{label} model digest mismatch for {entry.path}: "
                    f"expected {entry.expected_digest}, found {actual}"
                )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"model file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TokenizerArtifact:
    """The tokenizer file sits outside the two-entry encoder pair by design,
    but its provenance is pinned and verified the same way."""

    path: Path
    expected_digest: str

    def __post_init__(self) -> None:
        digest = self.expected_digest.lower()
        if len(digest) != _DIGEST_HEX_LENGTH or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("tokenizer expected_digest must be a 64-character hex sha256")

    def verify_file(self) -> None:
        actual = _sha256_file(self.path)
        if actual != self.expected_digest.lower():
            raise ValueError(
                f"tokenizer digest mismatch for {self.path}: "
                f"expected {self.expected_digest}, found {actual}"
            )


@dataclass(frozen=True, slots=True)
class RetrieveSolverConfig:
    """Every scalar the E002 solver uses.

    Defaults: sampling_fps=2.0 and smoothing_width=21 are literature anchors;
    everything else is an arbitrary placeholder. ALL of them are
    UNCALIBRATED PLACEHOLDERS — NOT A BENCHMARK RESULT.
    """

    sampling_fps: float = 2.0
    smoothing_width: int = 21
    min_proposal_duration_s: float = 1.0
    max_proposal_duration_s: float = 30.0
    tau_e002: float = 0.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        fps = _require_finite(self.sampling_fps, "sampling_fps")
        if fps <= 0:
            raise ValueError("sampling_fps must be > 0")
        if not isinstance(self.smoothing_width, int) or isinstance(self.smoothing_width, bool):
            raise ValueError("smoothing_width must be an integer")
        if self.smoothing_width < 1 or self.smoothing_width % 2 == 0:
            raise ValueError("smoothing_width must be an odd integer >= 1")
        minimum = _require_finite(self.min_proposal_duration_s, "min_proposal_duration_s")
        maximum = _require_finite(self.max_proposal_duration_s, "max_proposal_duration_s")
        if not 0 < minimum <= maximum:
            raise ValueError("require 0 < min_proposal_duration_s <= max_proposal_duration_s")
        _require_finite(self.tau_e002, "tau_e002")
        alpha = _require_finite(self.alpha, "alpha")
        if alpha <= 0:
            # The frozen confidence orientation (larger => stronger positive
            # evidence) requires a positive sigmoid slope.
            raise ValueError("alpha must be > 0")


_KNOWN_SOLVER_KEYS = frozenset(
    {
        "sampling_fps",
        "smoothing_width",
        "min_proposal_duration_s",
        "max_proposal_duration_s",
        "tau_e002",
        "alpha",
    }
)
_KNOWN_MODEL_KEYS = frozenset(
    {
        "model_family",
        "export_revision",
        "vision_path",
        "vision_digest",
        "text_path",
        "text_digest",
        "tokenizer_path",
        "tokenizer_digest",
    }
)


def load_solver_toml(
    path: Path,
) -> tuple[RetrieveSolverConfig, PairedEncoderProvenance, TokenizerArtifact]:
    """Parse the CLI's TOML config: [solver] scalars and [models] artifacts.

    Returns (config, provenance, tokenizer artifact). Unknown keys fail
    closed — a typo must never silently fall back to a default.
    """

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read solver config {path}: {error}") from error

    unknown_sections = raw.keys() - {"solver", "models"}
    if unknown_sections:
        raise ValueError(f"unknown config sections {sorted(unknown_sections)}")

    solver_raw = raw.get("solver", {})
    unknown = solver_raw.keys() - _KNOWN_SOLVER_KEYS
    if unknown:
        raise ValueError(f"unknown [solver] keys {sorted(unknown)}")
    config = RetrieveSolverConfig(**solver_raw)

    models_raw = raw.get("models")
    if not isinstance(models_raw, dict):
        raise ValueError("config must contain a [models] section")
    unknown = models_raw.keys() - _KNOWN_MODEL_KEYS
    if unknown:
        raise ValueError(f"unknown [models] keys {sorted(unknown)}")
    missing = _KNOWN_MODEL_KEYS - models_raw.keys()
    if missing:
        raise ValueError(f"missing [models] keys {sorted(missing)}")

    family = models_raw["model_family"]
    revision = models_raw["export_revision"]
    provenance = PairedEncoderProvenance(
        vision=EncoderFile(
            path=Path(models_raw["vision_path"]),
            expected_digest=models_raw["vision_digest"],
            model_family=family,
            export_revision=revision,
        ),
        text=EncoderFile(
            path=Path(models_raw["text_path"]),
            expected_digest=models_raw["text_digest"],
            model_family=family,
            export_revision=revision,
        ),
    )
    tokenizer = TokenizerArtifact(
        path=Path(models_raw["tokenizer_path"]),
        expected_digest=models_raw["tokenizer_digest"],
    )
    return config, provenance, tokenizer
