"""Runtime provider selection.

Pick a backend for everything at once with INSTASCRIBE_BACKEND, or override one
capability at a time:

    INSTASCRIBE_BACKEND = openai | local | anthropic | gemini | fake   (default: openai)
    VISION_PROVIDER / TEXT_PROVIDER / TTS_PROVIDER                     (per capability)

`local` means Ollama for vision + text and Kokoro for TTS. `anthropic` (Claude)
and `gemini` cover vision + text only; their TTS falls back to OpenAI (set
TTS_PROVIDER=local for a keyless Kokoro voice). Model ids default per backend and
can be overridden with VISION_MODEL / TEXT_MODEL / TTS_MODEL (and the per-backend
ANTHROPIC_MODEL / GEMINI_MODEL); OpenAI vision keeps reading JOB_MODEL.
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
    if backend == "anthropic":
        from .anthropic_provider import AnthropicVisionProvider

        return AnthropicVisionProvider(
            model=_model("VISION_MODEL", "ANTHROPIC_MODEL", "claude-opus-4-8")
        )
    if backend == "gemini":
        from .gemini_provider import GeminiVisionProvider

        return GeminiVisionProvider(
            model=_model("VISION_MODEL", "GEMINI_MODEL", "gemini-2.5-flash")
        )
    if backend == "fake":
        from .fake_provider import FakeVisionProvider

        return FakeVisionProvider()
    raise ValueError(
        f"Unknown vision backend: {backend!r} (want openai|local|anthropic|gemini|fake)"
    )


def get_text_provider() -> TextProvider:
    backend = _resolve("TEXT_PROVIDER")
    if backend == "openai":
        from .openai_provider import OpenAITextProvider

        return OpenAITextProvider(model=os.getenv("TEXT_MODEL", "gpt-4o-mini"))
    if backend == "ollama":
        from .local_provider import OllamaTextProvider

        return OllamaTextProvider(model=os.getenv("TEXT_MODEL", "qwen2.5:7b"))
    if backend == "anthropic":
        from .anthropic_provider import AnthropicTextProvider

        return AnthropicTextProvider(
            model=_model("TEXT_MODEL", "ANTHROPIC_MODEL", "claude-opus-4-8")
        )
    if backend == "gemini":
        from .gemini_provider import GeminiTextProvider

        return GeminiTextProvider(model=_model("TEXT_MODEL", "GEMINI_MODEL", "gemini-2.5-flash"))
    if backend == "fake":
        from .fake_provider import FakeTextProvider

        return FakeTextProvider()
    raise ValueError(f"Unknown text backend: {backend!r} (want openai|local|anthropic|gemini|fake)")


def get_tts_provider() -> TTSProvider:
    backend = _resolve("TTS_PROVIDER")
    if backend in ("ollama", "kokoro"):  # `local` resolves to ollama; local TTS is Kokoro
        from .local_provider import KokoroTTSProvider

        return KokoroTTSProvider()
    if backend == "fake":
        from .fake_provider import FakeTTSProvider

        return FakeTTSProvider()
    # openai + the two vision/text-only cloud backends (anthropic, gemini): use OpenAI
    # TTS. For a keyless voice under those backends, set TTS_PROVIDER=local (Kokoro).
    if backend in ("openai", "anthropic", "gemini"):
        from .openai_provider import OpenAITTSProvider

        return OpenAITTSProvider(model=os.getenv("TTS_MODEL", "tts-1-hd"))
    raise ValueError(f"Unknown TTS backend: {backend!r} (want openai|local|fake)")


def _model(capability_env: str, backend_env: str, default: str) -> str:
    # Per-capability override (VISION_MODEL / TEXT_MODEL) wins, then the per-backend
    # override (ANTHROPIC_MODEL / GEMINI_MODEL), then the default.
    return os.getenv(capability_env) or os.getenv(backend_env) or default
