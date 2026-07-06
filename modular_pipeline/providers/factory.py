"""Runtime provider selection.

Pick a backend for everything at once with INSTASCRIBE_BACKEND, or override one
capability at a time:

    INSTASCRIBE_BACKEND = openai | local | fake      (default: openai)
    VISION_PROVIDER / TEXT_PROVIDER / TTS_PROVIDER    (override per capability)

`local` means Ollama for vision + text and Kokoro for TTS. Model ids default per
backend and can be overridden with VISION_MODEL / TEXT_MODEL / TTS_MODEL (vision
on the OpenAI backend keeps reading JOB_MODEL for backward compatibility).
"""

from __future__ import annotations

import os

from .base import TextProvider, TTSProvider, VisionProvider

# `local` is the friendly umbrella name; vision + text under it run on Ollama.
_ALIASES = {"local": "ollama"}


def _resolve(capability_env: str, default_backend: str = "openai") -> str:
    backend = os.getenv(capability_env) or os.getenv("INSTASCRIBE_BACKEND") or default_backend
    backend = backend.strip().lower()
    return _ALIASES.get(backend, backend)


def get_vision_provider() -> VisionProvider:
    backend = _resolve("VISION_PROVIDER")
    if backend == "openai":
        from .openai_provider import OpenAIVisionProvider

        return OpenAIVisionProvider(model=os.getenv("JOB_MODEL", "gpt-4.1"))
    if backend == "ollama":
        from .local_provider import OllamaVisionProvider

        return OllamaVisionProvider(model=os.getenv("VISION_MODEL", "qwen2.5vl:7b"))
    if backend == "fake":
        from .fake_provider import FakeVisionProvider

        return FakeVisionProvider()
    raise ValueError(f"Unknown vision backend: {backend!r} (want openai | local | fake)")


def get_text_provider() -> TextProvider:
    backend = _resolve("TEXT_PROVIDER")
    if backend == "openai":
        from .openai_provider import OpenAITextProvider

        return OpenAITextProvider(model=os.getenv("TEXT_MODEL", "gpt-4o-mini"))
    if backend == "ollama":
        from .local_provider import OllamaTextProvider

        return OllamaTextProvider(model=os.getenv("TEXT_MODEL", "qwen2.5:7b"))
    if backend == "fake":
        from .fake_provider import FakeTextProvider

        return FakeTextProvider()
    raise ValueError(f"Unknown text backend: {backend!r} (want openai | local | fake)")


def get_tts_provider() -> TTSProvider:
    backend = _resolve("TTS_PROVIDER")
    if backend == "openai":
        from .openai_provider import OpenAITTSProvider

        return OpenAITTSProvider(model=os.getenv("TTS_MODEL", "tts-1-hd"))
    if backend in ("ollama", "kokoro"):  # `local` resolves to ollama; local TTS is Kokoro
        from .local_provider import KokoroTTSProvider

        return KokoroTTSProvider()
    if backend == "fake":
        from .fake_provider import FakeTTSProvider

        return FakeTTSProvider()
    raise ValueError(f"Unknown TTS backend: {backend!r} (want openai | local | fake)")
