"""S1 — local ONNX CLIP vision and text embedding providers.

These providers vendor the fail-closed pattern of the worker's frame-embedding
adapter (`services/worker/instadescribe_worker/frame_embeddings.py`) rather
than importing it: the benchmarks package is Apache-2.0 with an enforced
no-BUSL-imports boundary and an empty runtime dependency list, so the heavy
runtimes (onnxruntime, numpy, Pillow) are imported lazily and declared under
the optional ``clip`` extra.

Unlike the worker provider (which returns raw model output), BOTH providers
here return **L2-normalized unit vectors** — the frozen E002 representation:
cross-modal similarity is then exactly a dot product of unit vectors, and raw
dot products can never be mixed with cosine-scored values.

Strictly local: ``network_access`` is False by construction; loading and
inference read only the given file paths.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from .curve import unit_vector

_MAX_MODEL_BYTES = 2_000_000_000
_VISION_INPUT = "pixel_values"
_VISION_OUTPUT = "image_embeds"
_TEXT_REQUIRED_INPUT = "input_ids"
_TEXT_OPTIONAL_INPUT = "attention_mask"
_TEXT_OUTPUT = "text_embeds"

# CLIP feature-extractor recipe (same constants as the worker provider).
_IMAGE_SIZE = 224
_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class EmbeddingSession(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def run(self, output_names: object, feeds: dict[str, Any]) -> Sequence[Any]: ...


SessionFactory = Callable[[Path], EmbeddingSession]


def _onnxruntime_session(model_path: Path) -> EmbeddingSession:
    import onnxruntime  # lazy: optional `clip` extra

    options = onnxruntime.SessionOptions()
    # Single-threaded for deterministic float accumulation order.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _validate_model_path(model_path: Path) -> Path:
    if not model_path.is_absolute():
        raise ValueError("model path must be absolute")
    if model_path.suffix != ".onnx":
        raise ValueError("model path must point to a .onnx file")
    resolved = model_path.resolve()
    if not resolved.is_file():
        raise ValueError(f"model file not found: {model_path}")
    if resolved.stat().st_size > _MAX_MODEL_BYTES:
        raise ValueError("model file exceeds the 2 GB safety bound")
    return resolved


def _clip_preprocess(frame_path: Path) -> Any:
    """CLIP recipe: RGB, bicubic shortest edge 224, centre crop, normalize, CHW."""

    import numpy  # lazy: optional `clip` extra
    from PIL import Image  # lazy: optional `clip` extra

    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = _IMAGE_SIZE / min(width, height)
        resized = rgb.resize(
            (max(_IMAGE_SIZE, round(width * scale)), max(_IMAGE_SIZE, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        left = (resized.width - _IMAGE_SIZE) // 2
        top = (resized.height - _IMAGE_SIZE) // 2
        cropped = resized.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))
        array = numpy.asarray(cropped, dtype=numpy.float32) / 255.0
    mean = numpy.asarray(_IMAGE_MEAN, dtype=numpy.float32)
    std = numpy.asarray(_IMAGE_STD, dtype=numpy.float32)
    normalized = (array - mean) / std
    return numpy.ascontiguousarray(normalized.transpose(2, 0, 1)[numpy.newaxis, :])


def _int64_batch(token_ids: Sequence[int]) -> Any:
    import numpy  # lazy: optional `clip` extra

    return numpy.asarray([list(token_ids)], dtype=numpy.int64)


def _output_vector(outputs: Sequence[Any], what: str) -> tuple[float, ...]:
    if len(outputs) != 1:
        raise ValueError(f"{what} model returned {len(outputs)} outputs, expected 1")
    batch = outputs[0]
    if len(batch) != 1:
        raise ValueError(f"{what} model returned batch size {len(batch)}, expected 1")
    return tuple(float(value) for value in batch[0])


class OnnxClipVisionEmbeddingProvider:
    """CLIP vision tower behind a fail-closed, lazily loaded ONNX session."""

    def __init__(
        self,
        model_path: Path,
        *,
        session_factory: SessionFactory = _onnxruntime_session,
        preprocess: Callable[[Path], Any] = _clip_preprocess,
    ) -> None:
        self._model_path = _validate_model_path(model_path)
        self._session_factory = session_factory
        self._preprocess = preprocess
        self._session: EmbeddingSession | None = None

    @property
    def network_access(self) -> bool:
        return False

    def _load(self) -> EmbeddingSession:
        if self._session is None:
            session = self._session_factory(self._model_path)
            input_names = [item.name for item in session.get_inputs()]
            output_names = [item.name for item in session.get_outputs()]
            if input_names != [_VISION_INPUT] or _VISION_OUTPUT not in output_names:
                raise ValueError(
                    f"model is not a CLIP vision export (inputs {input_names}, "
                    f"outputs {output_names})"
                )
            self._session = session
        return self._session

    def embed_frame(self, frame_path: Path) -> tuple[float, ...]:
        session = self._load()
        outputs = session.run([_VISION_OUTPUT], {_VISION_INPUT: self._preprocess(frame_path)})
        return unit_vector(_output_vector(outputs, "vision"))


class OnnxClipTextEmbeddingProvider:
    """CLIP text tower; the tokenizer is injected (see ``tokenizer.py``).

    Fail-closed I/O contract: the export must take ``input_ids`` (optionally
    ``attention_mask``) and produce ``text_embeds``. Context length, padding
    and truncation are owned by the tokenizer, which must emit exactly the
    fixed-length id sequence the export expects.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        tokenize: Callable[[str], Sequence[int]],
        session_factory: SessionFactory = _onnxruntime_session,
        tensorize: Callable[[Sequence[int]], Any] = _int64_batch,
    ) -> None:
        self._model_path = _validate_model_path(model_path)
        self._tokenize = tokenize
        self._session_factory = session_factory
        self._tensorize = tensorize
        self._session: EmbeddingSession | None = None
        self._wants_attention_mask = False

    @property
    def network_access(self) -> bool:
        return False

    def _load(self) -> EmbeddingSession:
        if self._session is None:
            session = self._session_factory(self._model_path)
            input_names = [item.name for item in session.get_inputs()]
            output_names = [item.name for item in session.get_outputs()]
            allowed = {_TEXT_REQUIRED_INPUT, _TEXT_OPTIONAL_INPUT}
            if (
                _TEXT_REQUIRED_INPUT not in input_names
                or not set(input_names) <= allowed
                or _TEXT_OUTPUT not in output_names
            ):
                raise ValueError(
                    f"model is not a CLIP text export (inputs {input_names}, "
                    f"outputs {output_names})"
                )
            self._wants_attention_mask = _TEXT_OPTIONAL_INPUT in input_names
            self._session = session
        return self._session

    def embed_text(self, query: str) -> tuple[float, ...]:
        session = self._load()
        token_ids = list(self._tokenize(query))
        if not token_ids:
            raise ValueError("tokenizer produced no token ids")
        feeds: dict[str, Any] = {_TEXT_REQUIRED_INPUT: self._tensorize(token_ids)}
        if self._wants_attention_mask:
            feeds[_TEXT_OPTIONAL_INPUT] = self._tensorize([1] * len(token_ids))
        outputs = session.run([_TEXT_OUTPUT], feeds)
        return unit_vector(_output_vector(outputs, "text"))
