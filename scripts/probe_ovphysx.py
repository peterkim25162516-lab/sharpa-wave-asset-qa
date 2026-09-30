#!/usr/bin/env python3
"""Emit a structured kit-less OVPhysX Gate 0 capability probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from wave_asset_qa.adapters.ovphysx import load_manifest_json, probe_ovphysx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Isaac Lab's kit-less OVPhysX environment. This script never "
            "starts Kit, a renderer, a camera, or a GUI."
        )
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="check pinned package imports/symbols only; do not open USD or step physics",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="approved root used to resolve and confine a relative USD path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="canonical scenario manifest JSON containing the resolved USD path",
    )
    parser.add_argument(
        "--resolved-usd",
        type=Path,
        help="explicit resolved USD for the headless schema/runtime capability probe",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        help="hand to select when --manifest contains both canonical hands",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is also written to stdout",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="explicit logical device, for example cuda:0 after CUDA_VISIBLE_DEVICES is set",
    )
    return parser


def _failure(mode: str, error: Exception) -> dict[str, object]:
    return {
        "backend": "ovphysx",
        "mode": mode,
        "status": "error",
        "message": f"{type(error).__name__}: {error}",
        "capabilities": {
            "kitless_contract": True,
            "renderer_disabled": True,
            "camera_disabled": True,
            "pinned_ovphysx_wheel_version": False,
            "resolved_usd_open": False,
            "physics_step": False,
        },
        "provenance": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _write(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "import_only" if args.import_only else "resolved_usd_headless"
    try:
        manifest = load_manifest_json(args.manifest) if args.manifest is not None else None
        if not args.import_only and manifest is None and args.resolved_usd is None:
            raise ValueError(
                "resolved USD mode requires --manifest or --resolved-usd; "
                "use --import-only for a package check"
            )
        result = probe_ovphysx(
            import_only=args.import_only,
            manifest=manifest,
            asset_root=args.asset_root,
            resolved_usd=args.resolved_usd,
            hand=args.hand,
            device=args.device,
        )
        payload = result.to_dict()
    except Exception as error:
        payload = _failure(mode, error)

    _write(payload, args.output)
    status = payload["status"]
    if status == "available":
        return 0
    if status == "capability_error":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
