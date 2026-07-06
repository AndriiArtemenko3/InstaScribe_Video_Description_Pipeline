"""Model-provider seam for the three model-backed pipeline stages.

Call sites import the factory, never a vendor SDK:

    from providers import get_vision_provider, get_text_provider, get_tts_provider

Backend is chosen at runtime from environment variables (see factory.py).
"""

from .base import (
    CaptionResult,
    Frame,
    ProviderError,
    TextProvider,
    TextResult,
    TTSProvider,
    VisionProvider,
)
from .factory import get_text_provider, get_tts_provider, get_vision_provider

__all__ = [
    "CaptionResult",
    "Frame",
    "ProviderError",
    "TextProvider",
    "TextResult",
    "TTSProvider",
    "VisionProvider",
    "get_text_provider",
    "get_tts_provider",
    "get_vision_provider",
]
