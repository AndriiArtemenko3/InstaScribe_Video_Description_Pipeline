import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

MAX_CALLS = 100
DEFAULT_MAX_TOKENS = 20000
_call_count = 0


def _bump():
    global _call_count
    _call_count += 1
    if _call_count > MAX_CALLS:
        raise RuntimeError(f"Exceeded MAX_CALLS={MAX_CALLS} (safety limit).")


def get_client() -> OpenAI:
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL")  # point at Ollama/vLLM/LM Studio/OpenRouter
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        if base_url:
            key = "not-needed"  # local OpenAI-compatible servers ignore the key
        else:
            raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=key, base_url=base_url or None)


def safe_create_response(client: OpenAI, **kwargs):
    _bump()
    kwargs.setdefault("max_output_tokens", DEFAULT_MAX_TOKENS)
    return client.responses.create(**kwargs)
