"""Workspace layout and deterministic on-disk storage.

The workspace root is user-chosen and lives OUTSIDE the public repository —
media and licensing-sensitive provenance never enter git history. Layout:

    <root>/
      clips/                 derived benchmark clips (media)
      sources/               original media (optional local storage)
      metadata/items.jsonl   one JSON object per item, canonical store
      metadata/exclusions.jsonl  append-only decision ledger
      manifest.dev.json      built by the manifest builder
      manifest.test.json
      freeze/                freeze records

``items.jsonl`` is rewritten deterministically (sorted by item_id, sorted
keys); ``exclusions.jsonl`` is append-only — reconsidering a candidate adds a
new event, it never erases history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .model import DatasetItem, ExclusionReason, item_from_json, item_to_json


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def items_path(self) -> Path:
        return self.root / "metadata" / "items.jsonl"

    @property
    def exclusions_path(self) -> Path:
        return self.root / "metadata" / "exclusions.jsonl"

    @property
    def freeze_dir(self) -> Path:
        return self.root / "freeze"

    def manifest_path(self, split_name: str) -> Path:
        return self.root / f"manifest.{split_name}.json"

    def ensure_layout(self) -> None:
        for directory in (
            self.clips_dir,
            self.sources_dir,
            self.items_path.parent,
            self.freeze_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def serialize_items(items: dict[str, DatasetItem]) -> str:
    """The canonical byte-stable representation (also hashed at freeze)."""

    lines = [
        json.dumps(item_to_json(items[item_id]), sort_keys=True, ensure_ascii=True)
        for item_id in sorted(items)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def load_items(workspace: Workspace) -> dict[str, DatasetItem]:
    if not workspace.items_path.is_file():
        return {}
    items: dict[str, DatasetItem] = {}
    for line_number, line in enumerate(
        workspace.items_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = item_from_json(json.loads(line))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            raise ValueError(f"items.jsonl line {line_number}: {error}") from error
        if item.item_id in items:
            raise ValueError(f"duplicate item_id {item.item_id!r} in items.jsonl")
        items[item.item_id] = item
    return items


def save_items(workspace: Workspace, items: dict[str, DatasetItem]) -> None:
    workspace.ensure_layout()
    workspace.items_path.write_text(serialize_items(items), encoding="utf-8")


def append_exclusion(
    workspace: Workspace,
    *,
    candidate_id: str,
    source_group_id: str,
    decision: str,
    reason: ExclusionReason,
    notes: str,
    solver_output_seen: bool,
    timestamp: str | None = None,
) -> None:
    """Append one decision event; history is never rewritten or deleted."""

    if decision not in ("excluded", "reconsidered"):
        raise ValueError("decision must be 'excluded' or 'reconsidered'")
    workspace.ensure_layout()
    entry = {
        "candidate_id": candidate_id,
        "source_group_id": source_group_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": decision,
        "reason": reason.value,
        "notes": notes,
        # Explicit, never defaulted: whether ANY solver output existed/was
        # seen when this decision was made. Must be false for every
        # construction-time decision before the first model run.
        "solver_output_seen": solver_output_seen,
    }
    with workspace.exclusions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")


def load_exclusions(workspace: Workspace) -> list[dict[str, object]]:
    if not workspace.exclusions_path.is_file():
        return []
    entries = []
    for line_number, line in enumerate(
        workspace.exclusions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"exclusions.jsonl line {line_number}: {error}") from error
    return entries


def currently_excluded_ids(entries: list[dict[str, object]]) -> set[str]:
    """Latest decision per candidate wins; earlier history stays on disk."""

    state: dict[str, str] = {}
    for entry in entries:
        state[str(entry["candidate_id"])] = str(entry["decision"])
    return {candidate for candidate, decision in state.items() if decision == "excluded"}
