"""One coherent CLI for dataset construction. Model-blind by design.

Usage:
    python -m instadescribe_benchmarks.id_event_light_v0.dataset.cli \\
        --root <workspace> <command> [options]

Commands: status, add, annotate, review, exclude, assign-split, validate,
build-manifests, freeze, verify-freeze, hash-file.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ..schema import Difficulty
from .build import (
    build_manifests,
    dataset_errors,
    ffprobe_duration_s,
    freeze_dataset,
    status_report,
    verify_freeze,
)
from .model import (
    DatasetItem,
    EventCategory,
    ExclusionReason,
    HardNegativeType,
    ItemStatus,
    SourceType,
    normalize_timestamp,
    sha256_file,
)
from .split import assign_split
from .store import Workspace, append_exclusion, load_items, save_items


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _replace(item: DatasetItem, **changes: object) -> DatasetItem:
    return dataclasses.replace(item, **changes)  # type: ignore[arg-type]


def _require_item(items: dict[str, DatasetItem], item_id: str) -> DatasetItem:
    if item_id not in items:
        raise ValueError(f"unknown item id {item_id!r}")
    return items[item_id]


def _timestamp_arg(raw: str) -> float:
    normalized, changed = normalize_timestamp(float(raw))
    if changed:
        print(f"note: timestamp {raw} normalized to 0.1 s precision -> {normalized}")
    return normalized


def cmd_add(workspace: Workspace, args: argparse.Namespace) -> None:
    items = load_items(workspace)
    if args.item_id in items:
        raise ValueError(f"item {args.item_id!r} already exists")
    clip_path = Path(args.clip)
    if not clip_path.is_file():
        raise ValueError(f"clip file not found: {clip_path}")
    destination = workspace.clips_dir / clip_path.name
    if destination.resolve() != clip_path.resolve():
        raise ValueError(
            f"place the derived clip inside the workspace first: expected {destination}"
        )
    item = DatasetItem(
        item_id=args.item_id,
        source_group_id=args.source_group,
        status=ItemStatus.CANDIDATE,
        source_type=SourceType(args.source_type),
        source_reference=args.source_reference,
        source_title=args.source_title,
        source_creator=args.source_creator,
        licence_id=args.licence_id,
        licence_evidence=args.licence_evidence,
        acquisition_date=args.acquisition_date or _today(),
        original_filename=Path(args.original).name if args.original else None,
        original_file_sha256=sha256_file(Path(args.original)) if args.original else None,
        clip_filename=clip_path.name,
        derived_clip_sha256=sha256_file(clip_path),  # never user-entered
        transform_log=tuple(args.transform or ()),
    )
    items[args.item_id] = item
    save_items(workspace, items)
    print(f"registered candidate {args.item_id} (clip sha256 {item.derived_clip_sha256[:16]}...)")


def cmd_annotate(workspace: Workspace, args: argparse.Namespace) -> None:
    items = load_items(workspace)
    item = _require_item(items, args.item_id)
    difficulty = Difficulty(args.difficulty)
    start = _timestamp_arg(args.start) if args.start is not None else None
    end = _timestamp_arg(args.end) if args.end is not None else None
    updated = _replace(
        item,
        query=args.query,
        difficulty=difficulty,
        label=difficulty is Difficulty.POSITIVE,
        event_category=EventCategory(args.category),
        hard_negative_type=(
            HardNegativeType(args.hard_negative_type) if args.hard_negative_type else None
        ),
        start_time=start,
        end_time=end,
        annotation_notes=args.notes,
        annotator=args.annotator,
        annotation_date=_today(),
        status=ItemStatus.ANNOTATED,
        review_status="pending",
    )
    items[args.item_id] = updated
    save_items(workspace, items)
    print(f"annotated {args.item_id}: {difficulty.value}, query={args.query!r}")


def cmd_review(workspace: Workspace, args: argparse.Namespace) -> None:
    items = load_items(workspace)
    item = _require_item(items, args.item_id)
    if item.status is not ItemStatus.ANNOTATED:
        raise ValueError(
            f"{args.item_id} is {item.status.value!r}; review requires 'annotated' "
            "(review is an explicit action, never automatic)"
        )
    items[args.item_id] = _replace(
        item,
        review_status="approved",
        reviewer=args.reviewer,
        review_date=_today(),
        review_notes=args.notes,
        status=ItemStatus.REVIEWED,
    )
    save_items(workspace, items)
    print(f"reviewed {args.item_id} (reviewer: {args.reviewer})")


def cmd_exclude(workspace: Workspace, args: argparse.Namespace) -> None:
    items = load_items(workspace)
    item = _require_item(items, args.item_id)
    append_exclusion(
        workspace,
        candidate_id=args.item_id,
        source_group_id=item.source_group_id,
        decision=args.decision,
        reason=ExclusionReason(args.reason),
        notes=args.notes or "",
        solver_output_seen=args.solver_output_seen,
    )
    print(f"{args.decision}: {args.item_id} ({args.reason}); ledger appended")


def cmd_assign_split(workspace: Workspace, args: argparse.Namespace) -> None:
    items = load_items(workspace)
    not_reviewed = sorted(
        item_id
        for item_id, item in items.items()
        if item.status not in (ItemStatus.REVIEWED, ItemStatus.ASSIGNED)
    )
    if not_reviewed:
        raise ValueError(f"split assignment requires every item reviewed; pending: {not_reviewed}")
    assignment = assign_split(items)
    for item_id, item in sorted(items.items()):
        items[item_id] = _replace(
            item, split=assignment[item.source_group_id], status=ItemStatus.ASSIGNED
        )
    save_items(workspace, items)
    for group in sorted(assignment):
        print(f"{assignment[group].value:>4}  {group}")
    print("split assigned; review with 'status' and 'validate' before freeze")


def cmd_validate(workspace: Workspace, args: argparse.Namespace) -> None:
    probe = None if args.skip_duration_check else ffprobe_duration_s
    errors = dataset_errors(workspace, probe=probe)
    if errors:
        print("validation FAILED:", file=sys.stderr)
        for message in errors:
            print(f"  {message}", file=sys.stderr)
        raise SystemExit(1)
    print("validation OK")


def cmd_build_manifests(workspace: Workspace, args: argparse.Namespace) -> None:
    paths = build_manifests(workspace)
    for path in sorted(paths.values()):
        print(f"wrote {path} (strict loader OK)")


def cmd_freeze(workspace: Workspace, args: argparse.Namespace) -> None:
    probe = None if args.skip_duration_check else ffprobe_duration_s
    record_path = freeze_dataset(workspace, version=args.version, probe=probe)
    print(f"FROZEN: {record_path}")
    print(f"record sha256: {sha256_file(record_path)}")


def cmd_verify_freeze(workspace: Workspace, args: argparse.Namespace) -> None:
    verify_freeze(workspace, Path(args.record))
    print("freeze verification OK: no drift detected")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, required=True, help="dataset workspace root (outside the repo)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="construction progress overview")

    add = commands.add_parser("add", help="register a candidate item")
    add.add_argument("--item-id", required=True)
    add.add_argument("--source-group", required=True)
    add.add_argument("--source-type", required=True, choices=[s.value for s in SourceType])
    add.add_argument("--clip", required=True, help="derived clip already inside <root>/clips/")
    add.add_argument("--original", help="original media file (hashed if given)")
    add.add_argument("--source-reference", required=True)
    add.add_argument("--source-title")
    add.add_argument("--source-creator")
    add.add_argument("--licence-id")
    add.add_argument("--licence-evidence")
    add.add_argument("--acquisition-date")
    add.add_argument("--transform", action="append", help="one transform-log entry (repeatable)")

    annotate = commands.add_parser("annotate", help="record the human annotation")
    annotate.add_argument("--item-id", required=True)
    annotate.add_argument("--query", required=True)
    annotate.add_argument("--difficulty", required=True, choices=[d.value for d in Difficulty])
    annotate.add_argument("--category", required=True, choices=[c.value for c in EventCategory])
    annotate.add_argument("--hard-negative-type", choices=[h.value for h in HardNegativeType])
    annotate.add_argument("--start", help="positive start_time (seconds, 0.1 s precision)")
    annotate.add_argument("--end", help="positive end_time (seconds, 0.1 s precision)")
    annotate.add_argument("--notes")
    annotate.add_argument("--annotator", required=True)

    review = commands.add_parser("review", help="approve an annotated item (explicit second pass)")
    review.add_argument("--item-id", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes")

    exclude = commands.add_parser("exclude", help="append an exclusion/reconsideration event")
    exclude.add_argument("--item-id", required=True)
    exclude.add_argument("--decision", default="excluded", choices=["excluded", "reconsidered"])
    exclude.add_argument("--reason", required=True, choices=[r.value for r in ExclusionReason])
    exclude.add_argument("--notes")
    exclude.add_argument(
        "--solver-output-seen",
        action="store_true",
        help="set ONLY if any solver output existed when deciding (must stay unset before freeze)",
    )

    commands.add_parser("assign-split", help="deterministic source-group-aware dev/test split")

    validate = commands.add_parser("validate", help="all construction-state checks")
    validate.add_argument("--skip-duration-check", action="store_true")

    commands.add_parser("build-manifests", help="write + strict-load both split manifests")

    freeze = commands.add_parser("freeze", help="final dataset freeze (all contracts enforced)")
    freeze.add_argument("--version", required=True, help="e.g. DATASET_FREEZE_V1")
    freeze.add_argument("--skip-duration-check", action="store_true")

    verify = commands.add_parser("verify-freeze", help="detect drift against a freeze record")
    verify.add_argument("record", help="path to the freeze record JSON")

    hash_cmd = commands.add_parser("hash-file", help="sha256 of a local file")
    hash_cmd.add_argument("path")

    args = parser.parse_args(argv)
    workspace = Workspace(root=args.root)
    handlers = {
        "status": lambda: print(status_report(workspace)),
        "add": lambda: cmd_add(workspace, args),
        "annotate": lambda: cmd_annotate(workspace, args),
        "review": lambda: cmd_review(workspace, args),
        "exclude": lambda: cmd_exclude(workspace, args),
        "assign-split": lambda: cmd_assign_split(workspace, args),
        "validate": lambda: cmd_validate(workspace, args),
        "build-manifests": lambda: cmd_build_manifests(workspace, args),
        "freeze": lambda: cmd_freeze(workspace, args),
        "verify-freeze": lambda: cmd_verify_freeze(workspace, args),
        "hash-file": lambda: print(sha256_file(Path(args.path))),
    }
    try:
        handlers[args.command]()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
