"""CLIP text tokenizer adapter for the E002 solver.

Resolved implementation choice (previously parked): rather than hand-rolling
CLIP's byte-pair encoding — whose byte encoder, lowercasing and merge rules
are easy to get subtly wrong — this adapter loads the export's own
``tokenizer.json`` through the ``tokenizers`` runtime (optional ``clip``
extra). The smallest maintainable solution: the artifact that shipped with
the model defines the exact tokenization, and we assert its structure instead
of assuming it.

Fail-closed structure checks (performed once at load):
- the special tokens ``<|startoftext|>`` and ``<|endoftext|>`` must exist;
- every encoding must come back exactly ``context_length`` ids long
  (truncation and padding are configured explicitly here, not assumed);
- the sequence must start with BOS and contain EOS.

The exact padding id follows the artifact's declared EOS token, matching the
CLIP convention of padding with ``<|endoftext|>``.
"""

from __future__ import annotations

from pathlib import Path

_BOS_TOKEN = "<|startoftext|>"
_EOS_TOKEN = "<|endoftext|>"
_CONTEXT_LENGTH = 77


class ClipTokenizerAdapter:
    def __init__(self, tokenizer_path: Path, *, context_length: int = _CONTEXT_LENGTH) -> None:
        if not tokenizer_path.is_file():
            raise ValueError(f"tokenizer file not found: {tokenizer_path}")
        if context_length < 3:  # BOS + at least one token + EOS
            raise ValueError("context_length must be >= 3")
        from tokenizers import Tokenizer  # lazy: optional `clip` extra

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        bos_id = tokenizer.token_to_id(_BOS_TOKEN)
        eos_id = tokenizer.token_to_id(_EOS_TOKEN)
        if bos_id is None or eos_id is None:
            raise ValueError(
                f"tokenizer at {tokenizer_path} lacks the CLIP special tokens "
                f"{_BOS_TOKEN!r}/{_EOS_TOKEN!r}"
            )
        tokenizer.enable_truncation(max_length=context_length)
        tokenizer.enable_padding(length=context_length, pad_id=eos_id, pad_token=_EOS_TOKEN)
        self._tokenizer = tokenizer
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._context_length = context_length

    def __call__(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if len(ids) != self._context_length:
            raise ValueError(f"tokenizer produced {len(ids)} ids, expected {self._context_length}")
        if ids[0] != self._bos_id or self._eos_id not in ids:
            raise ValueError("tokenizer output missing the expected BOS/EOS structure")
        return ids
