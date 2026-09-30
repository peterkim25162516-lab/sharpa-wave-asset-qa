#!/usr/bin/env python3
"""Record one portable, formal resolved-USD Gate 0 schema probe."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from wave_asset_qa.adapters.ovphysx import load_manifest_json, probe_ovphysx
from wave_asset_qa.parity.bundle import write_json_atomic
from wave_asset_qa.parity.contracts import HandSide
from wave_asset_qa.parity.scenarios import load_manifest, manifest_sha256


EXPECTED_APPROVED_ROOT = Path("/data/home/exampleuser/sharpa-wave-asset-qa-gate0")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("left", "right"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda:0",))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--asset-tree-sha256", required=True)
    return parser


def _confined(path: Path, root: Path, label: str, *, strict: bool = True) -> Path:
    expanded = path.expanduser()
    if expanded.exists() and expanded.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = expanded.resolve(strict=strict)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the approved root") from error
    return resolved


def _portable_probe_payload(
    payload: Mapping[str, object],
    *,
    model_path: str,
    hand: str,
    session_id: str,
    source_revision: str,
    canonical_manifest_sha256: str,
    manifest_file_sha256: str,
    asset_tree_sha256: str,
    asset_commit: str,
    asset_git_tree: str,
) -> dict[str, object]:
    portable = dict(payload)
    raw_provenance = portable.get("provenance")
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}

    raw_modules = provenance.get("imported_modules")
    if isinstance(raw_modules, Mapping):
        modules: dict[str, object] = {}
        for name, raw_record in raw_modules.items():
            if isinstance(raw_record, Mapping):
                record = dict(raw_record)
                record.pop("file", None)
                modules[str(name)] = record
        provenance["imported_modules"] = modules

    raw_inspection = provenance.get("usd_inspection")
    if isinstance(raw_inspection, Mapping):
        inspection = dict(raw_inspection)
        inspection["source_path"] = model_path
        provenance["usd_inspection"] = inspection

    provenance.update(
        {
            "hand": hand,
            "model_path": model_path,
            "session_id": session_id,
            "source_revision": source_revision,
            "manifest_sha256": canonical_manifest_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "asset_tree_sha256": asset_tree_sha256,
            "asset_commit": asset_commit,
            "asset_git_tree": asset_git_tree,
            "schema_probe_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    portable["provenance"] = provenance
    return portable


def _run(args: argparse.Namespace) -> dict[str, object]:
    root = args.approved_root.expanduser().resolve(strict=True)
    if root != EXPECTED_APPROVED_ROOT.resolve(strict=True):
        raise ValueError("unexpected approved root")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("approved root must be a real directory")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("display variables must be unset for the kit-less schema probe")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one physical GPU index")
    if _SESSION_RE.fullmatch(args.session_id) is None:
        raise ValueError("--session-id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(args.source_revision) is None:
        raise ValueError("--source-revision must be a lowercase 40-digit Git commit")
    if _SHA256_RE.fullmatch(args.asset_tree_sha256) is None:
        raise ValueError("--asset-tree-sha256 must be a lowercase SHA-256")

    asset_root = _confined(args.asset_root, root, "asset root")
    manifest_path = _confined(args.manifest, root, "manifest path")
    results_root = _confined(root / "results", root, "results root")
    output = _confined(args.output, results_root, "output path", strict=False)
    if output.exists():
        raise FileExistsError(f"schema probe refuses to overwrite existing output: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    if args.asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise ValueError("asset-tree SHA-256 does not match the canonical manifest")
    hand_side = HandSide(args.hand)
    hand = manifest.hand(hand_side)
    raw_manifest = load_manifest_json(manifest_path)
    result = probe_ovphysx(
        import_only=False,
        manifest=raw_manifest,
        asset_root=asset_root,
        hand=args.hand,
        device=args.device,
    )
    payload = _portable_probe_payload(
        result.to_dict(),
        model_path=hand.model_paths.ovphysx,
        hand=args.hand,
        session_id=args.session_id,
        source_revision=args.source_revision,
        canonical_manifest_sha256=manifest_sha256(manifest),
        manifest_file_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        asset_tree_sha256=args.asset_tree_sha256,
        asset_commit=manifest.provenance.commit,
        asset_git_tree=manifest.provenance.asset_git_tree,
    )
    write_json_atomic(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _run(args)
    except Exception as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2
    summary = {
        "status": payload["status"],
        "hand": args.hand,
        "output": args.output.name,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
    return 0 if payload["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
