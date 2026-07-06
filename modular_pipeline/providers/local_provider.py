"""Local, keyless providers.

  * Vision and text run on Ollama through its OpenAI-compatible endpoint
    (OLLAMA_BASE_URL, default http://localhost:11434/v1). Install Ollama, then:
        ollama pull qwen2.5vl:7b     # vision (scene captioning)
        ollama pull qwen2.5:7b       # text  (Smart Fill rewrite)
  * TTS runs on Kokoro (kokoro-82M, Apache-2.0) in-process; the ~330MB weights
    download once on first use. Requires `pip install -r requirements-local.txt`.

Quality trails the hosted OpenAI models on ambiguous frames; the human
edit-and-approve loop is the intended mitigation. See docs/local-models.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import openai

from .base import CaptionResult, ProviderError, TextResult

_RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)


def _ollama_client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # Ollama ignores the key
    )


def _wrap(exc: Exception) -> ProviderError:
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(
            f"Cannot reach Ollama — is `ollama serve` running and OLLAMA_BASE_URL correct? ({exc})",
            retryable=True,
        )
    return ProviderError(f"{type(exc).__name__}: {exc}", retryable=isinstance(exc, _RETRYABLE))


class OllamaVisionProvider:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5vl:7b") -> None:
        self.model = model

    def caption_chunk(
        self, *, developer_prompt, user_text, frames, schema, image_detail="low"
    ) -> CaptionResult:
        client = _ollama_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for local_idx, f in enumerate(frames):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"FRAME {local_idx}\n"
                        f"global_frame_index={f.index}\n"
                        f"timestamp={f.timestamp:.1f}s"
                    ),
                }
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f.image_b64}"}}
            )
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": developer_prompt},
                    {"role": "user", "content": content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema["name"],
                        "schema": schema["schema"],
                        "strict": schema.get("strict", True),
                    },
                },
                temperature=0,
            )
        except openai.OpenAIError as exc:
            raise _wrap(exc) from exc

        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)  # JSONDecodeError -> caller retries
        usage = getattr(completion, "usage", None)
        usage_d = {"total_tokens": getattr(usage, "total_tokens", None)} if usage else None
        return CaptionResult(data=data, model=self.model, usage=usage_d)


class OllamaTextProvider:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b") -> None:
        self.model = model

    def rewrite(self, *, system, user, temperature=0.4, max_tokens=400) -> TextResult:
        client = _ollama_client()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.OpenAIError as exc:
            raise _wrap(exc) from exc
        text = (completion.choices[0].message.content or "").strip()
        usage = getattr(completion, "usage", None)
        tokens = (getattr(usage, "total_tokens", 0) or 0) if usage else 0
        return TextResult(text=text, model=self.model, tokens=tokens)


# InstaScribe's provider-neutral voice names -> Kokoro voices (a = American English).
_KOKORO_VOICE_MAP = {
    "onyx": "am_michael",
    "echo": "am_adam",
    "fable": "bm_george",
    "nova": "af_heart",
    "shimmer": "af_bella",
    "alloy": "af_sarah",
}
_DEFAULT_KOKORO_VOICE = "am_michael"


class KokoroTTSProvider:
    name = "kokoro"
    voices = tuple(_KOKORO_VOICE_MAP.keys())

    def __init__(self, lang_code: str = "a") -> None:
        self.lang_code = lang_code
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise ProviderError(
                    "Kokoro TTS is not installed. Run: pip install -r requirements-local.txt",
                    retryable=False,
                ) from exc
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synthesize(self, *, text, voice, out_path: Path) -> Path:
        import subprocess
        import tempfile

        import numpy as np
        import soundfile as sf

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = self._get_pipeline()
        kvoice = _KOKORO_VOICE_MAP.get((voice or "").strip().lower(), _DEFAULT_KOKORO_VOICE)

        chunks: list[Any] = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=kvoice):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr)
        if not chunks:
            raise ProviderError("Kokoro produced no audio for the given text", retryable=False)
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        # Kokoro emits 24kHz float32. Write a wav, then transcode to mp3 for parity
        # with the OpenAI path (the mixer downstream expects an mp3 on disk).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            sf.write(str(wav_path), audio, 24000)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "192k", str(out_path)],
                check=True,
                capture_output=True,
            )
        finally:
            wav_path.unlink(missing_ok=True)
        return out_path
