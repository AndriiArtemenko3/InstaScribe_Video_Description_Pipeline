"""OPTIONAL live-model integration tests for the E002 CLIP pair.

These run ONLY when the pinned export is explicitly supplied locally:

    export INSTADESCRIBE_TEST_CLIP_EXPORT_DIR=\\
        ~/.cache/huggingface/hub/models--Xenova--clip-vit-base-patch32/snapshots/<revision>

and the optional ``clip`` extra is installed. The ordinary unit suite never
downloads models and never depends on these; CI does not set the variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_EXPORT_DIR = os.environ.get("INSTADESCRIBE_TEST_CLIP_EXPORT_DIR")

pytestmark = pytest.mark.skipif(
    _EXPORT_DIR is None,
    reason="set INSTADESCRIBE_TEST_CLIP_EXPORT_DIR to run live CLIP integration tests",
)


@pytest.fixture(scope="module")
def export_dir() -> Path:
    assert _EXPORT_DIR is not None
    directory = Path(_EXPORT_DIR).expanduser()
    for required in ("onnx/text_model.onnx", "onnx/vision_model.onnx", "tokenizer.json"):
        if not (directory / required).is_file():
            pytest.skip(f"live export incomplete: missing {required}")
    return directory


@pytest.fixture(scope="module")
def tokenizer(export_dir: Path):
    pytest.importorskip("tokenizers")
    from instadescribe_benchmarks.id_event_light_v0.retrieve.tokenizer import (
        ClipTokenizerAdapter,
    )

    return ClipTokenizerAdapter(export_dir / "tokenizer.json")


def test_live_tokenizer_contract(tokenizer) -> None:
    ids = tokenizer("a person entering a vehicle")
    assert len(ids) == 77
    assert ids[0] == 49406  # <|startoftext|>
    assert 49407 in ids  # <|endoftext|> (also the pad token)
    # Overlong queries truncate deterministically to the same 77 ids.
    overlong = "a person walking " * 40
    assert tokenizer(overlong) == tokenizer(overlong)
    assert len(tokenizer(overlong)) == 77


def test_live_text_tower_unit_vectors(export_dir: Path, tokenizer) -> None:
    pytest.importorskip("onnxruntime")
    from instadescribe_benchmarks.id_event_light_v0.retrieve.clip_onnx import (
        OnnxClipTextEmbeddingProvider,
    )

    provider = OnnxClipTextEmbeddingProvider(
        export_dir / "onnx" / "text_model.onnx", tokenize=tokenizer
    )
    vector = provider.embed_text("a door opening")
    assert len(vector) == 512
    assert sum(value * value for value in vector) == pytest.approx(1.0)
    assert provider.embed_text("a door opening") == vector  # deterministic


def test_live_pair_shares_dimension_and_crosses_modalities(
    export_dir: Path, tokenizer, tmp_path: Path
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")
    from PIL import Image

    from instadescribe_benchmarks.id_event_light_v0.retrieve.clip_onnx import (
        OnnxClipTextEmbeddingProvider,
        OnnxClipVisionEmbeddingProvider,
    )
    from instadescribe_benchmarks.id_event_light_v0.retrieve.curve import unit_cosine

    frame = tmp_path / "red.jpg"
    Image.new("RGB", (224, 224), (220, 30, 30)).save(frame, "JPEG")

    text = OnnxClipTextEmbeddingProvider(
        export_dir / "onnx" / "text_model.onnx", tokenize=tokenizer
    )
    vision = OnnxClipVisionEmbeddingProvider(export_dir / "onnx" / "vision_model.onnx")
    text_vector = text.embed_text("a plain red image")
    image_vector = vision.embed_frame(frame)
    assert len(text_vector) == len(image_vector) == 512
    similarity = unit_cosine(image_vector, text_vector)
    assert -1.0 <= similarity <= 1.0  # cross-modal execution, not a quality claim
