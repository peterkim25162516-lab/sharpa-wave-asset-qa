#!/usr/bin/env python3
"""Collect one Freeze A OVPhysX effective-parameter record without a trace step."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Sequence

from wave_asset_qa.adapters.ovphysx import OVPhysXAdapter
from wave_asset_qa.parity.bundle import write_json_atomic
from wave_asset_qa.parity.contracts import HandSide
from wave_asset_qa.parity.diagnostics import asset_tree_sha256, is_link_like
from wave_asset_qa.parity.effective_readback import (
    EffectiveReadbackError,
    build_readback_instance,
    canonical_json_sha256,
    load_frozen_contract,
    sha256_file,
)


EXPECTED_APPROVED_ROOT = Path("/data/home/exampleuser/sharpa-wave-asset-qa-gate0")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--gate0-manifest", required=True, type=Path)
    parser.add_argument("--freeze-config", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--instance-index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda:0",))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--asset-tree-sha256", required=True)
    return parser


def _confined(
    path: Path,
    root: Path,
    label: str,
    *,
    require_exists: bool = True,
) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    for ancestor in (raw, *raw.parents):
        if ancestor.exists() and is_link_like(ancestor):
            raise EffectiveReadbackError(f"{label} has a symbolic-link ancestor")
        if ancestor == root:
            break
    try:
        resolved = raw.resolve(strict=require_exists)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EffectiveReadbackError(f"{label} escapes the approved root") from exc
    return resolved


def _read_marker(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or is_link_like(path):
        raise EffectiveReadbackError(f"missing or linked {label} marker")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EffectiveReadbackError(f"cannot read {label} marker") from exc
    if lines != [expected]:
        raise EffectiveReadbackError(f"{label} marker does not match worker input")


def _distribution_version(name: str, *, absent: str | None = None) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if absent is not None:
            return absent
        raise EffectiveReadbackError(f"required distribution is not installed: {name}")


def _gpu_identity(physical_index: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise EffectiveReadbackError(f"cannot read selected GPU identity: {detail}")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise EffectiveReadbackError("selected GPU query did not return exactly one row")
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if len(fields) != 3 or any(not field for field in fields):
        raise EffectiveReadbackError("selected GPU identity row is malformed")
    return {"gpu_uuid": fields[0], "gpu_name": fields[1], "driver_version": fields[2]}


def _process_identity(machine_id: str, case_id: str, session_id: str) -> tuple[str, str]:
    pid = os.getpid()
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        fields = Path("/proc/self/stat").read_text(encoding="ascii").split()
        start_ticks = fields[21]
    except (OSError, UnicodeError, IndexError) as exc:
        raise EffectiveReadbackError(
            "cannot establish Linux fresh-process identity"
        ) from exc
    if not boot_id or not start_ticks.isdigit():
        raise EffectiveReadbackError("Linux fresh-process identity is malformed")
    private_process_id = canonical_json_sha256(
        {
            "machine_id": machine_id,
            "boot_id": boot_id,
            "pid": pid,
            "process_start_ticks": start_ticks,
        }
    )
    instance_id = canonical_json_sha256(
        {
            "case_id": case_id,
            "fresh_process_id": private_process_id,
            "session_id": session_id,
        }
    )
    return private_process_id, instance_id


def _validate_source_snapshot(
    *,
    root: Path,
    source_revision: str,
    source_tree: str,
    source_archive_sha256: str,
) -> Path:
    for value, regex, label in (
        (source_revision, _HEX40_RE, "source revision"),
        (source_tree, _HEX40_RE, "source tree"),
        (source_archive_sha256, _SHA256_RE, "source archive SHA-256"),
    ):
        if regex.fullmatch(value) is None:
            raise EffectiveReadbackError(f"invalid {label}")
    project_root = _confined(
        root / "project" / source_tree, root, "project snapshot"
    )
    if project_root != Path(__file__).resolve(strict=True).parents[1]:
        raise EffectiveReadbackError("worker script is not executing from the selected snapshot")
    if any(is_link_like(path) for path in project_root.rglob("*")):
        raise EffectiveReadbackError("project snapshot contains a symbolic link")
    _read_marker(project_root / ".source-revision", source_revision, "source revision")
    _read_marker(project_root / ".source-tree", source_tree, "source tree")
    _read_marker(
        project_root / ".archive-sha256",
        source_archive_sha256,
        "source archive SHA-256",
    )
    archive = _confined(
        root / "downloads" / f"{source_archive_sha256}.tar",
        root,
        "source archive",
    )
    if sha256_file(archive) != source_archive_sha256:
        raise EffectiveReadbackError("deployed source archive content hash mismatch")
    return project_root


def _run(args: argparse.Namespace) -> dict[str, object]:
    root = args.approved_root.expanduser().resolve(strict=True)
    if root != EXPECTED_APPROVED_ROOT.resolve(strict=True):
        raise EffectiveReadbackError("unexpected approved root")
    if not root.is_dir() or is_link_like(root):
        raise EffectiveReadbackError("approved root must be a real directory")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise EffectiveReadbackError("display variables must be unset")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise EffectiveReadbackError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU index"
        )
    if _SESSION_RE.fullmatch(args.session_id) is None:
        raise EffectiveReadbackError("invalid session id")
    project_root = _validate_source_snapshot(
        root=root,
        source_revision=args.source_revision,
        source_tree=args.source_tree,
        source_archive_sha256=args.source_archive_sha256,
    )
    asset_root = _confined(args.asset_root, root, "asset root")
    manifest_path = _confined(args.gate0_manifest, project_root, "Gate 0 manifest")
    freeze_path = _confined(args.freeze_config, project_root, "Freeze A config")
    expected_manifest = project_root / "configs" / "parity" / "gate0.json"
    expected_freeze = (
        project_root / "configs" / "parity" / "ovphysx_effective_readback.json"
    )
    if manifest_path != expected_manifest or freeze_path != expected_freeze:
        raise EffectiveReadbackError("worker inputs are not the snapshot's canonical configs")
    output = _confined(
        args.output,
        (root / "results").resolve(strict=True),
        "worker output",
        require_exists=False,
    )
    if output.exists() or is_link_like(output):
        raise FileExistsError(f"worker refuses to overwrite output: {output}")
    if not output.parent.is_dir() or is_link_like(output.parent):
        raise EffectiveReadbackError("worker output parent must already be a real directory")
    if _SHA256_RE.fullmatch(args.asset_tree_sha256) is None:
        raise EffectiveReadbackError("invalid asset tree SHA-256")

    plan, manifest = load_frozen_contract(freeze_path, manifest_path)
    expected_row = next(
        (
            row
            for row in plan["expected_instances"]
            if row["side"] == args.side
            and row["instance_index"] == args.instance_index
        ),
        None,
    )
    if expected_row is None or expected_row["case_id"] != args.case_id:
        raise EffectiveReadbackError("worker case does not match Freeze A")
    if args.asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise EffectiveReadbackError("asset tree argument differs from Gate 0")
    canonical_asset_subtree = _confined(
        asset_root / manifest.provenance.asset_root,
        asset_root,
        "canonical asset subtree",
    )
    if asset_tree_sha256(canonical_asset_subtree) != args.asset_tree_sha256:
        raise EffectiveReadbackError("materialized asset tree hash mismatch")

    gpu = _gpu_identity(visible)
    machine_id = socket.gethostname()
    if not machine_id:
        raise EffectiveReadbackError("machine identity is unavailable")
    fresh_process_id, fresh_instance_id = _process_identity(
        machine_id, args.case_id, args.session_id
    )
    hand = manifest.hand(HandSide(args.side))
    zero_hold = manifest.scenario("zero_hold")
    adapter = OVPhysXAdapter(device=args.device, asset_root=asset_root)
    try:
        adapter.open(hand, dt_override=zero_hold.dt_s)
        snapshot = adapter.effective_parameter_readback()
    finally:
        adapter.close()

    script_hash = sha256_file(Path(__file__).resolve(strict=True))
    module_path = (
        project_root / "src" / "wave_asset_qa" / "parity" / "effective_readback.py"
    )
    snapshot_marker = project_root / ".snapshot-sha256"
    if not snapshot_marker.is_file() or is_link_like(snapshot_marker):
        raise EffectiveReadbackError("snapshot SHA-256 marker is missing")
    snapshot_sha256 = snapshot_marker.read_text(encoding="ascii").strip()
    if _SHA256_RE.fullmatch(snapshot_sha256) is None:
        raise EffectiveReadbackError("snapshot SHA-256 marker is invalid")
    provenance: dict[str, object] = {
        "source_revision": args.source_revision,
        "source_tree": args.source_tree,
        "source_archive_sha256": args.source_archive_sha256,
        "source_snapshot_sha256": snapshot_sha256,
        "probe_source_sha256": script_hash,
        "pipeline_module_sha256": sha256_file(module_path),
        "freeze_a_config_sha256": sha256_file(freeze_path),
        "gate0_manifest_file_sha256": sha256_file(manifest_path),
        "gate0_manifest_semantic_sha256": plan["frozen_context"][
            "gate0_manifest_semantic_sha256"
        ],
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "canonical_lf_asset_tree_sha256": args.asset_tree_sha256,
        "python_version": platform.python_version(),
        "isaac_lab_version": _distribution_version("isaaclab"),
        "isaac_sim_version": _distribution_version(
            "isaacsim", absent="not-installed-kitless"
        ),
        "physx_version": _distribution_version("ovphysx"),
        "pytorch_version": _distribution_version("torch"),
        "device": args.device,
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "driver_version": gpu["driver_version"],
        "machine_id": machine_id,
        "session_id": args.session_id,
        "worker_pid": os.getpid(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "constraints": {
            "kitless": True,
            "headless": True,
            "renderer": False,
            "camera": False,
            "fresh_process": True,
            "trace_step_calls": 0,
        },
    }
    payload = build_readback_instance(
        plan=plan,
        manifest=manifest,
        adapter_snapshot=snapshot,
        side=args.side,
        instance_index=args.instance_index,
        case_id=args.case_id,
        fresh_instance_id=fresh_instance_id,
        fresh_process_id=fresh_process_id,
        provenance=provenance,
    )
    write_json_atomic(output, payload)
    if sha256(output.read_bytes()).hexdigest() != sha256_file(output):
        raise EffectiveReadbackError("worker output hash verification failed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _run(args)
    except Exception as exc:
        summary = {
            "status": "worker_error",
            "case_id": args.case_id,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        sys.stdout.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stdout.flush()
        sys.stderr.flush()
        return 2
    summary = {
        "status": "collected",
        "case_id": payload["case_id"],
        "output": args.output.name,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
