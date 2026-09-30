"""Fresh-process worker shared by the two Contact Gate C0 backends."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import platform
import re
import sys

from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.diagnostics import asset_tree_sha256, is_link_like
from wave_asset_qa.parity.process_identity import (
    current_os_process_identity,
    os_process_identity_sha256,
)

from .bundle import (
    canonical_json_sha256,
    sha256_file,
    verify_adapter_private_evidence,
    write_json_exclusive,
)
from .runner import select_contact_case, validate_contact_run, write_contact_run
from .scenarios import contact_manifest_sha256, load_contact_manifest


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ContactWorkerError(RuntimeError):
    """Raised when a worker cannot prove one complete formal case."""


def build_worker_parser(*, backend: Simulator) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run one fresh-process Contact C0 {backend.value} case"
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--asset-tree-sha256", required=True)
    if backend is Simulator.OVPHYSX:
        parser.add_argument("--device", default="cuda:0")
    return parser


def _checked_file(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    for candidate in (raw, *raw.parents):
        if candidate.exists() and is_link_like(candidate):
            raise ContactWorkerError(
                f"{label} has a symbolic-link or junction ancestor"
            )
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ContactWorkerError(f"{label} must be a regular file")
    return resolved


def _checked_directory(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    for candidate in (raw, *raw.parents):
        if candidate.exists() and is_link_like(candidate):
            raise ContactWorkerError(
                f"{label} has a symbolic-link or junction ancestor"
            )
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ContactWorkerError(f"{label} must be a regular directory")
    return resolved


def _validate_scalar_arguments(args: argparse.Namespace) -> None:
    if _SESSION.fullmatch(args.session_id) is None:
        raise ContactWorkerError("session ID is not portable")
    if _HEX40.fullmatch(args.source_revision) is None:
        raise ContactWorkerError("source revision must be lowercase Git hex")
    if _HEX40.fullmatch(args.source_tree) is None:
        raise ContactWorkerError("source tree must be lowercase Git hex")
    if _HEX64.fullmatch(args.asset_tree_sha256) is None:
        raise ContactWorkerError("asset tree hash must be lowercase SHA-256")


def run_contact_worker(
    args: argparse.Namespace,
    *,
    backend: Simulator,
    experimental_campaign: str | None = None,
) -> object:
    """Execute and persist one case; return the strict :class:`ContactRun`."""

    _validate_scalar_arguments(args)
    if experimental_campaign is not None:
        from .async_campaign import CAMPAIGN_ID
        if experimental_campaign != CAMPAIGN_ID:
            raise ContactWorkerError('unrecognized experimental campaign')
    project_root = Path(__file__).resolve(strict=True).parents[3]
    manifest_path = _checked_file(args.manifest, "manifest")
    expected_manifest = (
        project_root / "configs" / "parity" / "contact_c0.json"
    ).resolve(strict=True)
    if manifest_path != expected_manifest:
        raise ContactWorkerError("worker requires the checked-in Contact C0 manifest")
    asset_root = _checked_directory(args.asset_root, "asset root")
    output_dir = _checked_directory(args.output_dir, "output directory")
    for name in (
        f"{args.case_id}.run.json",
        "private-process.json",
        "private-adapter-evidence.json",
        "private-campaign.json",
    ):
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"worker refuses to overwrite {target.name}")

    manifest = load_contact_manifest(manifest_path)
    case = select_contact_case(manifest, args.case_id, simulator=backend)
    if args.asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise ContactWorkerError("requested asset hash differs from the manifest")

    process_identity = current_os_process_identity()
    fresh_process_hash = os_process_identity_sha256(process_identity)
    process_record = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": case.case_id,
        "backend": backend.value,
        "session_id": args.session_id,
        "source_revision": args.source_revision,
        "source_tree": args.source_tree,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_semantic_sha256": contact_manifest_sha256(manifest),
        "asset_tree_sha256": args.asset_tree_sha256,
        "python_version": platform.python_version(),
        "python_executable_realpath": os.path.realpath(sys.executable),
        "worker_module_realpath": os.path.realpath(__file__),
        "worker_module_sha256": sha256_file(__file__),
        "os_process_identity": process_identity,
        "fresh_process_identity_sha256": fresh_process_hash,
    }
    if experimental_campaign is not None:
        process_record['experimental_campaign'] = experimental_campaign
    write_json_exclusive(output_dir / "private-process.json", process_record)

    asset_subtree = (asset_root / manifest.provenance.asset_root).resolve(strict=True)
    try:
        asset_subtree.relative_to(asset_root)
    except ValueError as error:
        raise ContactWorkerError("manifest asset subtree escapes the asset root") from error
    observed_asset_hash = asset_tree_sha256(asset_subtree)
    if observed_asset_hash != args.asset_tree_sha256:
        raise ContactWorkerError("materialized asset tree hash differs from the freeze")

    # Importing adapters is delayed until the exact worker identity is durable.
    from .adapters import MuJoCoContactAdapter, OVPhysXContactAdapter

    ov_adapter = OVPhysXContactAdapter
    if experimental_campaign is not None and backend is Simulator.OVPHYSX:
        from .async_campaign import AsyncCandidateContactAdapter
        ov_adapter = AsyncCandidateContactAdapter

    adapter = (
        MuJoCoContactAdapter(
            source_revision=args.source_revision,
            source_tree=args.source_tree,
            fresh_process_identity_sha256=fresh_process_hash,
        )
        if backend is Simulator.MUJOCO
        else ov_adapter(
            source_revision=args.source_revision,
            source_tree=args.source_tree,
            fresh_process_identity_sha256=fresh_process_hash,
        )
    )
    run = adapter.run_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=asset_root,
        device=(None if backend is Simulator.MUJOCO else args.device),
    )
    validate_contact_run(
        run,
        manifest=manifest,
        case=case,
        source_revision=args.source_revision,
        source_tree=args.source_tree,
        fresh_process_identity_sha256=fresh_process_hash,
    )
    private_evidence = dict(adapter.last_private_evidence)
    verify_adapter_private_evidence(private_evidence, run)
    intervention = None
    if experimental_campaign is not None and backend is Simulator.OVPHYSX:
        from .async_campaign import validate_candidate_evidence
        intervention = validate_candidate_evidence(private_evidence, run, dt_s=case.dt_s)
    fixture = private_evidence.get("fixture_overlay")
    if not isinstance(fixture, dict):
        raise ContactWorkerError("adapter fixture preimage is absent")
    collision_preimage = fixture.get("collision_inventory_hash_preimage")
    if canonical_json_sha256(collision_preimage) != run.fixture_readback.get(
        "collision_inventory_sha256"
    ):
        raise ContactWorkerError("collision inventory preimage hash mismatch")

    write_json_exclusive(
        output_dir / "private-adapter-evidence.json", private_evidence
    )
    write_contact_run(output_dir / f"{case.case_id}.run.json", run)
    if experimental_campaign is not None:
        write_json_exclusive(output_dir/'private-campaign.json', {
            'schema_version': 1, 'experimental_campaign': experimental_campaign,
            'formal_c0_replacement': False, 'backend': backend.value,
            'case_id': case.case_id, 'source_revision': args.source_revision,
            'source_tree': args.source_tree, 'manifest_sha256': contact_manifest_sha256(manifest),
            'intervention': intervention,
            'run_sha256': sha256_file(output_dir/f'{case.case_id}.run.json'),
            'private_adapter_sha256': sha256_file(output_dir/'private-adapter-evidence.json'),
            'private_process_sha256': sha256_file(output_dir/'private-process.json'),
        })
    return run


def worker_main(
    argv: Sequence[str] | None,
    *,
    backend: Simulator,
    experimental_campaign: str | None = None,
) -> int:
    parser = build_worker_parser(backend=backend)
    args = parser.parse_args(argv)
    try:
        run = run_contact_worker(args, backend=backend, experimental_campaign=experimental_campaign)
    except Exception as error:
        sys.stdout.write(
            json.dumps(
                {
                    "status": "worker_error",
                    "case_id": getattr(args, "case_id", None),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": "completed",
                "case_id": run.case.case_id,
                "completed_steps": run.execution.completed_steps,
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


__all__ = [
    "ContactWorkerError",
    "build_worker_parser",
    "run_contact_worker",
    "worker_main",
]
