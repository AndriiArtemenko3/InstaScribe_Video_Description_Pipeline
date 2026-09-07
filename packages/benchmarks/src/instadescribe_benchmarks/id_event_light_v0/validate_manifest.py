"""Validate an ID Event Light Bench v0 manifest, failing closed on any defect.

Usage:
    python -m instadescribe_benchmarks.id_event_light_v0.validate_manifest \
        --manifest .../sample_manifest.json [--skip-video-check] [--json]

``--skip-video-check`` exists for template/sample manifests whose video paths
are placeholders; every other rule still applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .schema import ManifestError, load_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="benchmark manifest JSON")
    parser.add_argument(
        "--skip-video-check",
        action="store_true",
        help="skip the video file-existence check (for placeholder manifests)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of text"
    )
    args = parser.parse_args(argv)

    checked = not args.skip_video_check
    try:
        items = load_manifest(args.manifest, check_video_files=checked)
    except ManifestError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"ok": True, "items": len(items), "video_files_checked": checked}))
    else:
        suffix = "checked" if checked else "not checked"
        print(f"manifest OK: {len(items)} items (video files {suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
