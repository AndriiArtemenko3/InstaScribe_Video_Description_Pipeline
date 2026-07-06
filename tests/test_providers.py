"""Tests for the model-provider seam. Exercises the factory's backend selection
and the fake backend end to end — no API key, no network, no heavy deps."""

import pytest
from providers import Frame, get_text_provider, get_tts_provider, get_vision_provider
from providers.fake_provider import FakeTextProvider, FakeTTSProvider, FakeVisionProvider
from schemas import SCENE_SCHEMA

_ENV_KEYS = ("INSTASCRIBE_BACKEND", "VISION_PROVIDER", "TEXT_PROVIDER", "TTS_PROVIDER")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_backend_is_openai(clean_env):
    assert get_vision_provider().name == "openai"
    assert get_text_provider().name == "openai"
    assert get_tts_provider().name == "openai"


def test_fake_backend_selected_for_all_capabilities(clean_env):
    clean_env.setenv("INSTASCRIBE_BACKEND", "fake")
    assert isinstance(get_vision_provider(), FakeVisionProvider)
    assert isinstance(get_text_provider(), FakeTextProvider)
    assert isinstance(get_tts_provider(), FakeTTSProvider)


def test_local_alias_maps_to_ollama_and_kokoro(clean_env):
    clean_env.setenv("INSTASCRIBE_BACKEND", "local")
    assert get_vision_provider().name == "ollama"
    assert get_text_provider().name == "ollama"
    assert get_tts_provider().name == "kokoro"


def test_per_capability_override_beats_global(clean_env):
    clean_env.setenv("INSTASCRIBE_BACKEND", "openai")
    clean_env.setenv("VISION_PROVIDER", "fake")
    assert get_vision_provider().name == "fake"
    assert get_text_provider().name == "openai"  # untouched by the vision override


def test_unknown_backend_raises(clean_env):
    clean_env.setenv("VISION_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        get_vision_provider()


def test_constructing_local_providers_needs_no_server(clean_env):
    # A provider must construct without reaching its backend; the call does that.
    from providers.local_provider import OllamaVisionProvider

    assert OllamaVisionProvider().name == "ollama"


def test_fake_vision_output_conforms_to_scene_schema_shape():
    frames = [Frame(index=i, timestamp=float(i), image_b64="") for i in range(3)]
    result = FakeVisionProvider().caption_chunk(
        developer_prompt="dev", user_text="usr", frames=frames, schema=SCENE_SCHEMA
    )
    data = result.data
    assert all(k in data for k in SCENE_SCHEMA["schema"]["required"])
    assert len(data["scenes"]) == 3
    scene_required = SCENE_SCHEMA["schema"]["properties"]["scenes"]["items"]["required"]
    assert all(k in data["scenes"][0] for k in scene_required)
    assert data["memory_updates"] == {"seen_character_ids": [], "new_characters": []}


def test_fake_text_respects_word_budget():
    user = (
        "Time budget: 3.0 seconds (~7 words at typical AD pace).\n\n"
        "Current description:\n"
        "A tall man in a dark coat walks slowly across the wet platform to the train.\n\n"
        "Rewrite to fit within the time budget."
    )
    out = FakeTextProvider().rewrite(system="editor", user=user)
    assert len(out.text.split()) <= 7
    assert out.model == "fake"


def test_fake_tts_writes_a_file(tmp_path):
    out = tmp_path / "line.mp3"
    FakeTTSProvider().synthesize(text="hello", voice="onyx", out_path=out)
    assert out.exists()
