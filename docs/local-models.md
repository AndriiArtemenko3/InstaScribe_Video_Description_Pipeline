# Model providers and running locally

InstaScribe calls a model for exactly three things: **vision** (describe a chunk
of frames as structured JSON), **text** (rewrite one description to fit a time
budget, the Smart Fill), and **TTS** (speak a description to audio). Speech
detection and transcription already run locally (silero-vad + faster-whisper) and
never touched a hosted API.

Each of the three is a small interface in `modular_pipeline/providers/`, and the
backend is chosen at runtime. No call site imports a vendor SDK, so swapping a
model is a config change, not a code change.

## Choosing a backend

Set one environment variable:

| `INSTASCRIBE_BACKEND` | Vision | Text | TTS | Key? |
|---|---|---|---|---|
| `openai` (default) | gpt-4.1 | gpt-4o-mini | tts-1-hd | yes |
| `local` | Ollama qwen2.5vl:7b | Ollama qwen2.5:7b | Kokoro | no |
| `fake` | placeholder JSON | deterministic trim | silence | no |

Override one stage at a time with `VISION_PROVIDER`, `TEXT_PROVIDER`, or
`TTS_PROVIDER` (each `openai | local | fake`). For example, keep hosted vision
but speak with local TTS: `VISION_PROVIDER=openai TTS_PROVIDER=local`.

`fake` makes no network call and needs no model; it powers the test suite and a
keyless server smoke run.

## Running fully local (no API key)

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull qwen2.5vl:7b   # vision — scene captioning (~6GB)
   ollama pull qwen2.5:7b     # text  — Smart Fill rewrite (~5GB)
   ```

2. Install the local TTS dependency (Kokoro; weights download on first use):

   ```bash
   pip install -r requirements-local.txt
   ```

3. Point InstaScribe at the local backend and run it:

   ```bash
   INSTASCRIBE_BACKEND=local python modular_pipeline/server.py
   ```

The pipeline now runs end to end with no key and no data leaving the machine.

## Model choices and the quality tradeoff

- **Vision — Qwen2.5-VL-7B** (Apache-2.0). Strong open vision-language model that
  handles the strict-JSON scene schema through Ollama's structured-output mode. It
  trails GPT-4.1 on ambiguous or cluttered frames — a real, visible gap. The
  human edit-and-approve loop is the intended mitigation: a description a blind
  listener relies on is reviewed before it ships regardless of which model drafted
  it. Larger Qwen2.5-VL variants close the gap at a higher memory cost.
- **Text — Qwen2.5-7B** (Apache-2.0). Smart Fill is a short, constrained rewrite;
  the local gap here is small.
- **TTS — Kokoro-82M** (Apache-2.0). Runs in-process on CPU near real time. Its
  even, neutral delivery suits audio-description narration. The gap to hosted TTS
  is narrower than the vision gap.
- **ASR — faster-whisper** stays as-is; it is already the local standard.

Licenses were chosen so a commercial or portfolio deploy is unencumbered. Avoided
as defaults: Llama 3.2 Vision (restrictive license), XTTS-v2 (non-commercial),
F5-TTS (CC-BY-NC).

## The `OPENAI_BASE_URL` shortcut

The OpenAI backend also honors `OPENAI_BASE_URL`, so you can point the standard
client at any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenRouter)
without touching the `local` provider. This covers vision and text; hosted TTS
has no local-compatible endpoint, so a keyless setup still uses Kokoro for speech.

## Why a hand-rolled seam and not LiteLLM

A gateway like LiteLLM would abstract vision and text, but its transcription
wrapper only covers hosted ASR, so it cannot touch the existing local
faster-whisper stage, and it adds a dependency and an indirection layer for three
call sites. Three small typed interfaces (`VisionProvider`, `TextProvider`,
`TTSProvider`) plus a factory keep the seam readable in one file, let the demo run
through a `fake` provider with no special-casing, and leave the door open to point
the OpenAI client at a compatible server for free. The interface is the part that
matters; the specific model behind it is one line of config.
