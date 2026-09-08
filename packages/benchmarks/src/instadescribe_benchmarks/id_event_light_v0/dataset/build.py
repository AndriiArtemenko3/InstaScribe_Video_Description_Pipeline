"""Validation, status, manifest building, freeze and drift verification.

Freeze is the only operation that demands the complete dataset contract
(48 items, 16/16/16; DEV 6/6/6, TEST 10/10/10). Every earlier command
tolerates an incomplete construction state so intermediate work is never
forced or discarded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from ..schema import MANIFEST_SCHEMA_VERSION, Difficulty, load_manifest
from .model import DatasetItem, ItemStatus, rights_errors, sha256_file, validation_errors
from .split import DEV_TARGET, TEST_TARGET, validate_no_leakage
from .store import Workspace, load_exclusions, load_items, serialize_items
from .store import currently_excluded_ids as _excluded_ids

DurationProbe = Callable[[Path], float]

_BUCKETS = (Difficulty.POSITIVE, Difficulty.NEGATIVE, Difficulty.HARD_NEGATIVE)


def ffprobe_duration_s(video_path: Path) -> float:
    """Container duration via ffprobe (the tooling's only media inspection)."""

    import shutil

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe is required for duration validation and was not found on PATH")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:300]
        raise ValueError(f"ffprobe failed for {video_path}: {detail}")
    try:
        return float(result.stdout.decode("utf-8", "replace").strip())
    except ValueError as error:
        raise ValueError(f"ffprobe returned a non-numeric duration for {video_path}") from error


def active_items(workspace: Workspace) -> dict[str, DatasetItem]:
    """All stored items minus those currently excluded by the ledger."""

    items = load_items(workspace)
    excluded = _excluded_ids(load_exclusions(workspace))
    return {item_id: item for item_id, item in items.items() if item_id not in excluded}


def item_media_errors(
    item: DatasetItem, workspace: Workspace, *, probe: DurationProbe | None
) -> list[str]:
    """Media availability, digest integrity, and annotation-vs-duration checks."""

    errors: list[str] = []
    if item.clip_filename is None or item.derived_clip_sha256 is None:
        return errors  # stage validation reports the missing registration
    clip_path = workspace.clips_dir / item.clip_filename
    if not clip_path.is_file():
        errors.append(f"{item.item_id}: clip file missing: {clip_path}")
        return errors
    actual = sha256_file(clip_path)
    if actual != item.derived_clip_sha256.lower():
        errors.append(
            f"{item.item_id}: derived clip digest mismatch "
            f"(recorded {item.derived_clip_sha256}, actual {actual})"
        )
        return errors  # do not trust a tampered file for further checks
    if probe is not None and item.end_time is not None:
        duration = probe(clip_path)
        if item.end_time > duration:
            errors.append(
                f"{item.item_id}: end_time {item.end_time} exceeds clip duration {duration:.2f}"
            )
    return errors


def dataset_errors(
    workspace: Workspace, *, probe: DurationProbe | None, check_media: bool = True
) -> list[str]:
    """All construction-state problems, without requiring completeness."""

    errors: list[str] = []
    items = active_items(workspace)
    for item_id in sorted(items):
        item = items[item_id]
        errors.extend(f"{item_id}: {message}" for message in validation_errors(item))
        if check_media:
            errors.extend(item_media_errors(item, workspace, probe=probe))
    errors.extend(validate_no_leakage(items))
    return errors


def status_report(workspace: Workspace) -> str:
    """Human-readable construction progress; no model inference anywhere."""

    all_items = load_items(workspace)
    exclusions = load_exclusions(workspace)
    excluded = _excluded_ids(exclusions)
    items = {i: item for i, item in all_items.items() if i not in excluded}

    status_counts = Counter(item.status.value for item in items.values())
    difficulty_counts = Counter(
        item.difficulty.value for item in items.values() if item.difficulty is not None
    )
    category_counts = Counter(
        item.event_category.value for item in items.values() if item.event_category is not None
    )
    hard_negative_counts = Counter(
        item.hard_negative_type.value
        for item in items.values()
        if item.hard_negative_type is not None
    )
    source_counts = Counter(item.source_type.value for item in items.values())
    split_counts = Counter(item.split.value for item in items.values() if item.split is not None)

    rights_incomplete = sum(1 for item in items.values() if rights_errors(item))
    missing_media = 0
    digest_mismatch = 0
    for item in items.values():
        if item.clip_filename is None:
            continue
        clip_path = workspace.clips_dir / item.clip_filename
        if not clip_path.is_file():
            missing_media += 1
        elif item.derived_clip_sha256 and sha256_file(clip_path) != item.derived_clip_sha256:
            digest_mismatch += 1

    target_total = sum(DEV_TARGET.values()) + sum(TEST_TARGET.values())
    lines = [
        f"included items:     {len(items)} / {target_total}",
        f"excluded (current): {len(excluded)}   (ledger events: {len(exclusions)})",
        "",
        "lifecycle: "
        + "  ".join(
            f"{status.value}={status_counts.get(status.value, 0)}" for status in ItemStatus
        ),
        "difficulty: "
        + "  ".join(
            f"{bucket.value}={difficulty_counts.get(bucket.value, 0)}/16" for bucket in _BUCKETS
        ),
        "split: "
        + ("  ".join(f"{k}={v}" for k, v in sorted(split_counts.items())) or "unassigned"),
        "",
        "event categories:   "
        + (", ".join(f"{k}={v}" for k, v in sorted(category_counts.items())) or "none yet"),
        "hard-negative types: "
        + (", ".join(f"{k}={v}" for k, v in sorted(hard_negative_counts.items())) or "none yet"),
        "source types:       "
        + (", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) or "none yet"),
        "",
        f"rights incomplete:  {rights_incomplete}",
        f"missing media:      {missing_media}",
        f"digest mismatches:  {digest_mismatch}",
    ]
    return "\n".join(lines)


def _manifest_record(item: DatasetItem) -> dict[str, object]:
    assert item.clip_filename is not None and item.query is not None
    record: dict[str, object] = {
        "id": item.item_id,
        "video": f"clips/{item.clip_filename}",
        "query": item.query,
        "label": item.label,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "difficulty": item.difficulty.value if item.difficulty else None,
    }
    if item.annotation_notes is not None:
        record["notes"] = item.annotation_notes
    return record


def build_manifests(workspace: Workspace) -> dict[str, Path]:
    """Write manifest.dev.json / manifest.test.json and re-load them through
    the strict benchmark loader. Loader failure fails construction."""

    items = active_items(workspace)
    paths: dict[str, Path] = {}
    for split_name in ("dev", "test"):
        members = sorted(
            (
                item
                for item in items.values()
                if item.split is not None and item.split.value == split_name
            ),
            key=lambda item: item.item_id,
        )
        if not members:
            raise ValueError(f"no items assigned to the {split_name} split yet")
        document = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": f"ID Event Light Bench v0 - {split_name} split",
            "items": [_manifest_record(item) for item in members],
        }
        path = workspace.manifest_path(split_name)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        load_manifest(path, check_video_files=True)  # strict round-trip, media required
        paths[split_name] = path
    return paths


def _split_counts(items: Mapping[str, DatasetItem]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"dev": {}, "test": {}}
    for item in items.values():
        if item.split is None or item.difficulty is None:
            continue
        bucket = counts[item.split.value]
        bucket[item.difficulty.value] = bucket.get(item.difficulty.value, 0) + 1
    return counts


def _tool_version(package_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            timeout=10,
            check=False,
            cwd=package_dir,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except OSError:
        pass
    return "unknown"


def freeze_dataset(
    workspace: Workspace,
    *,
    version: str,
    probe: DurationProbe | None,
    now: str | None = None,
    tool_version: str | None = None,
) -> Path:
    """The deliberate final gate. Fails closed unless EVERY contract holds."""

    if not version.strip() or "/" in version:
        raise ValueError("freeze version must be a non-empty name without '/'")
    items = active_items(workspace)
    errors: list[str] = []

    # Completeness contract.
    total_target = sum(DEV_TARGET.values()) + sum(TEST_TARGET.values())
    if len(items) != total_target:
        errors.append(f"dataset has {len(items)} items, freeze requires {total_target}")
    difficulty_counts = Counter(
        item.difficulty for item in items.values() if item.difficulty is not None
    )
    for bucket in _BUCKETS:
        expected = DEV_TARGET[bucket] + TEST_TARGET[bucket]
        if difficulty_counts.get(bucket, 0) != expected:
            errors.append(
                f"{bucket.value}: {difficulty_counts.get(bucket, 0)} items, need {expected}"
            )
    split_counts = _split_counts(items)
    for split_name, target in (("dev", DEV_TARGET), ("test", TEST_TARGET)):
        for bucket in _BUCKETS:
            have = split_counts[split_name].get(bucket.value, 0)
            if have != target[bucket]:
                errors.append(f"{split_name} {bucket.value}: {have}, need {target[bucket]}")

    # Item-level contracts.
    for item_id in sorted(items):
        item = items[item_id]
        if item.status is not ItemStatus.ASSIGNED:
            errors.append(f"{item_id}: status {item.status.value!r}, freeze requires 'assigned'")
        errors.extend(f"{item_id}: {message}" for message in validation_errors(item))
        errors.extend(f"{item_id}: rights - {message}" for message in rights_errors(item))
        errors.extend(item_media_errors(item, workspace, probe=probe))
    errors.extend(validate_no_leakage(items))

    clip_hashes = Counter(
        item.derived_clip_sha256 for item in items.values() if item.derived_clip_sha256
    )
    for digest, count in sorted(clip_hashes.items()):
        if count > 1:
            errors.append(f"derived clip hash {digest} appears {count} times")

    if errors:
        raise ValueError("freeze rejected:\n  " + "\n  ".join(errors))

    manifest_paths = build_manifests(workspace)

    record = {
        "freeze_version": version,
        "timestamp": now or datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_metadata_sha256": hashlib.sha256(
            serialize_items(items).encode("utf-8")
        ).hexdigest(),
        "dev_manifest_sha256": sha256_file(manifest_paths["dev"]),
        "test_manifest_sha256": sha256_file(manifest_paths["test"]),
        "items": [
            {"item_id": item_id, "derived_clip_sha256": items[item_id].derived_clip_sha256}
            for item_id in sorted(items)
        ],
        "source_groups": {
            group: sorted(i for i, item in items.items() if item.source_group_id == group)
            for group in sorted({item.source_group_id for item in items.values()})
        },
        "counts": {
            "total": len(items),
            "difficulty": {b.value: difficulty_counts.get(b, 0) for b in _BUCKETS},
            "split": split_counts,
            "event_category": dict(
                sorted(
                    Counter(
                        item.event_category.value for item in items.values() if item.event_category
                    ).items()
                )
            ),
            "hard_negative_type": dict(
                sorted(
                    Counter(
                        item.hard_negative_type.value
                        for item in items.values()
                        if item.hard_negative_type
                    ).items()
                )
            ),
        },
        "tool_version": tool_version or _tool_version(workspace.root),
    }
    workspace.freeze_dir.mkdir(parents=True, exist_ok=True)
    record_path = workspace.freeze_dir / f"{version}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace.freeze_dir / f"{version}.json.sha256").write_text(
        sha256_file(record_path) + "\n", encoding="utf-8"
    )
    return record_path


def verify_freeze(workspace: Workspace, record_path: Path) -> None:
    """Any drift between the frozen record and current state fails closed.

    Changing a clip, query, label, difficulty, boundary, source group or
    split after freeze creates a NEW dataset version — this check is how that
    rule is enforced.
    """

    record = json.loads(record_path.read_text(encoding="utf-8"))
    items = active_items(workspace)
    problems: list[str] = []

    actual_metadata = hashlib.sha256(serialize_items(items).encode("utf-8")).hexdigest()
    if actual_metadata != record["dataset_metadata_sha256"]:
        problems.append("dataset metadata digest drifted from the freeze record")

    for split_name in ("dev", "test"):
        manifest = workspace.manifest_path(split_name)
        if not manifest.is_file():
            problems.append(f"manifest.{split_name}.json is missing")
        elif sha256_file(manifest) != record[f"{split_name}_manifest_sha256"]:
            problems.append(f"manifest.{split_name}.json drifted from the freeze record")

    recorded = {entry["item_id"]: entry["derived_clip_sha256"] for entry in record["items"]}
    if set(recorded) != set(items):
        problems.append("item id set differs from the freeze record")
    for item_id in sorted(set(recorded) & set(items)):
        item = items[item_id]
        if item.derived_clip_sha256 != recorded[item_id]:
            problems.append(f"{item_id}: recorded clip digest differs from metadata")
            continue
        if item.clip_filename is not None:
            clip_path = workspace.clips_dir / item.clip_filename
            if not clip_path.is_file():
                problems.append(f"{item_id}: clip file missing")
            elif sha256_file(clip_path) != recorded[item_id]:
                problems.append(f"{item_id}: clip file drifted from the frozen digest")

    if problems:
        raise ValueError("freeze verification FAILED:\n  " + "\n  ".join(problems))
