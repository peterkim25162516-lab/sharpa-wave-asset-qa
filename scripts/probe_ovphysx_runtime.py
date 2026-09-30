#!/usr/bin/env python3
"""Execute exactly one canonical Wave case in a fresh kit-less OVPhysX process."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Sequence

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.adapters.ovphysx import OVPhysXAdapter
from wave_asset_qa.parity.bundle import write_json_atomic
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.runner import adapter_run_result_from_dict
from wave_asset_qa.parity.scenarios import (
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPECTED_APPROVED_ROOT = Path("/data/home/exampleuser/sharpa-wave-asset-qa-gate0")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--asset-tree-sha256", required=True)
    return parser


def _confined(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the approved root") from error
    return resolved


def _select_case(manifest: object, case_id: str) -> object:
    matches = [case for case in expand_scenario_cases(manifest) if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError(f"manifest does not contain exactly one case {case_id!r}")
    case = matches[0]
    if case.simulator is not Simulator.OVPHYSX:
        raise ValueError("the OVPhysX worker accepts only canonical ovphysx cases")
    return case


def _validated_result(
    result: AdapterRunResult,
    *,
    case: object,
    canonical_manifest_sha256: str,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
    asset_tree_sha256: str,
    asset_commit: str,
    asset_git_tree: str,
) -> AdapterRunResult:
    provenance = dict(result.provenance)
    # Raw worker JSON is part of the portable evidence bundle.  Keep the
    # asset identity but never persist the private absolute Server 6 path.
    provenance["source_path"] = case.model_path
    provenance.update(
        {
            "case_id": case.case_id,
            "manifest_sha256": canonical_manifest_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "session_id": session_id,
            "source_revision": source_revision,
            "asset_tree_sha256": asset_tree_sha256,
            "mapping_schema_version": 1,
            "asset_commit": asset_commit,
            "asset_git_tree": asset_git_tree,
            "worker_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "worker_pid": os.getpid(),
        }
    )
    enriched = replace(result, provenance=provenance)
    strict = adapter_run_result_from_dict(enriched.to_dict())
    if strict.backend != Simulator.OVPHYSX.value:
        raise ValueError("worker result backend is not ovphysx")
    if strict.scenario_id != case.scenario_id:
        raise ValueError("worker result scenario_id does not match the selected case")
    if not math.isclose(strict.dt, case.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("worker result dt does not match the selected case")
    return strict


def _run(args: argparse.Namespace) -> AdapterRunResult:
    root = args.approved_root.expanduser().resolve()
    if root != EXPECTED_APPROVED_ROOT.resolve():
        raise ValueError("unexpected approved root")
    if not root.is_dir():
        raise FileNotFoundError(f"approved root does not exist: {root}")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("display variables must be unset for the kit-less worker")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one physical GPU index")
    if _SESSION_RE.fullmatch(args.session_id) is None:
        raise ValueError("--session-id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(args.source_revision) is None:
        raise ValueError("--source-revision must be a lowercase 40-digit Git commit")
    if not re.fullmatch(r"[0-9a-f]{64}", args.asset_tree_sha256):
        raise ValueError("--asset-tree-sha256 must be a lowercase SHA-256")

    asset_root = _confined(args.asset_root, root, "asset root")
    manifest_path = _confined(args.manifest, root, "manifest path")
    output = _confined(args.output, root / "results", "output path")
    if not asset_root.is_dir():
        raise FileNotFoundError(f"asset root does not exist: {asset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"worker refuses to overwrite existing output: {output.name}")

    manifest = load_manifest(manifest_path)
    if args.asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise ValueError("asset-tree SHA-256 does not match the canonical manifest")
    case = _select_case(manifest, args.case_id)
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    adapter = OVPhysXAdapter(device=args.device, asset_root=asset_root)
    result = adapter.run_scenario(hand, scenario, dt_override=case.dt_s)
    result = _validated_result(
        result,
        case=case,
        canonical_manifest_sha256=manifest_sha256(manifest),
        manifest_file_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        session_id=args.session_id,
        source_revision=args.source_revision,
        asset_tree_sha256=args.asset_tree_sha256,
        asset_commit=manifest.provenance.commit,
        asset_git_tree=manifest.provenance.asset_git_tree,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result.to_dict())
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except Exception as error:
        summary = {
            "status": "worker_error",
            "case_id": args.case_id,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        sys.stdout.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        return 2

    summary = {
        "status": result.status,
        "case_id": args.case_id,
        "requested_steps": result.requested_steps,
        "completed_steps": result.completed_steps,
        "output": args.output.name,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
