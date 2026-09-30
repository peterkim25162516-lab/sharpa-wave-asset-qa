#!/usr/bin/env python3
"""Finalize four remote OVPhysX Freeze A records into one private hash bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

from wave_asset_qa.parity.bundle import write_bundle_manifest, write_json_atomic
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    bundle_payload_paths,
    create_staging_root,
    is_link_like,
    promote_staging_root,
    validate_git_source,
    validated_results_output_path,
    verify_exact_bundle,
    write_text_exclusive,
)
from wave_asset_qa.parity.effective_readback import (
    EffectiveReadbackError,
    assemble_readback_payload,
    load_frozen_contract,
    load_json_strict,
    render_readback_report,
    sanitize_readback_payload,
    sha256_file,
)


_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_INVENTORY_RE = re.compile(r"^([0-9a-f]{64})  (\./[^\r\n]+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate0-manifest", required=True, type=Path)
    parser.add_argument("--freeze-config", required=True, type=Path)
    parser.add_argument("--raw-evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    return parser


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EffectiveReadbackError(f"{path} must be an object with string keys")
    return value


def _strict_source(path: str | Path, label: str, *, directory: bool) -> Path:
    raw = Path(os.path.abspath(Path(path).expanduser()))
    for ancestor in (raw, *raw.parents):
        if ancestor.exists() and is_link_like(ancestor):
            raise EffectiveReadbackError(f"{label} has a symbolic-link or junction ancestor")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise EffectiveReadbackError(f"{label} does not exist") from exc
    if directory and not resolved.is_dir():
        raise EffectiveReadbackError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise EffectiveReadbackError(f"{label} must be a regular file")
    return resolved


def _safe_inventory_path(raw: str) -> str:
    if not raw.startswith("./"):
        raise EffectiveReadbackError("inventory paths must start with ./")
    relative = raw[2:]
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EffectiveReadbackError(f"unsafe inventory path: {raw!r}")
    return relative


def _verify_sha256_inventory(
    root: Path,
    inventory_path: Path,
    *,
    excluded: set[str],
) -> dict[str, str]:
    try:
        lines = inventory_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EffectiveReadbackError(f"cannot read evidence inventory: {inventory_path}") from exc
    records: dict[str, str] = {}
    for line in lines:
        match = _INVENTORY_RE.fullmatch(line)
        if match is None:
            raise EffectiveReadbackError(f"malformed evidence inventory line: {line!r}")
        digest, raw_path = match.groups()
        relative = _safe_inventory_path(raw_path)
        if relative in records:
            raise EffectiveReadbackError(f"duplicate evidence inventory path: {relative}")
        member = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        try:
            member.relative_to(root)
        except ValueError as exc:
            raise EffectiveReadbackError("inventory member escapes evidence root") from exc
        if not member.is_file() or is_link_like(member):
            raise EffectiveReadbackError(f"inventory member is not a regular file: {relative}")
        if sha256_file(member) != digest:
            raise EffectiveReadbackError(f"evidence inventory hash mismatch: {relative}")
        records[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in excluded
    }
    if set(records) != actual:
        raise EffectiveReadbackError(
            "evidence inventory is not exact: "
            f"missing={sorted(actual - set(records))}, extra={sorted(set(records) - actual)}"
        )
    return records


def _verify_no_links(root: Path) -> None:
    linked = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if is_link_like(path)
    ]
    if linked:
        raise EffectiveReadbackError(
            "raw evidence contains a symbolic link or junction: " + ", ".join(linked[:5])
        )


def _copy_tree_strict(source: Path, destination: Path) -> None:
    if destination.exists() or is_link_like(destination):
        raise EffectiveReadbackError("private evidence destination already exists")
    destination.mkdir()
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        target = destination / relative
        if is_link_like(path):
            raise EffectiveReadbackError(f"raw evidence contains a link: {relative.as_posix()}")
        if path.is_dir():
            target.mkdir()
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        else:
            raise EffectiveReadbackError(f"raw evidence contains a special entry: {relative}")


def _validate_launcher(
    raw_root: Path,
    *,
    plan: Mapping[str, Any],
    source_revision: str,
    source_tree: str,
    freeze_sha256: str,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    launcher = raw_root / "launcher"
    if not launcher.is_dir() or is_link_like(launcher):
        raise EffectiveReadbackError("raw evidence launcher directory is missing")
    status = load_json_strict(launcher / "status.json", label="launcher status")
    provenance = load_json_strict(
        launcher / "provenance.json", label="launcher provenance"
    )
    expected_status_keys = {
        "schema_version",
        "experiment_id",
        "status",
        "scientific_verdict",
        "exit_code",
        "run_id",
        "session_id",
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "snapshot_sha256",
        "freeze_a_config_sha256",
        "selected_gpu_index",
        "selected_gpu_uuid",
        "completed_case_count",
        "expected_case_count",
        "last_case_id",
        "elapsed_s",
        "gate0_root_bytes",
        "ended_at_utc",
    }
    if set(status) != expected_status_keys:
        raise EffectiveReadbackError("launcher status fields changed")
    if (
        status.get("schema_version") != 1
        or status.get("experiment_id") != plan.get("experiment_id")
        or status.get("status") != "collected"
        or status.get("scientific_verdict") != "pending_local_validation"
        or status.get("exit_code") != 0
        or status.get("completed_case_count") != 4
        or status.get("expected_case_count") != 4
    ):
        raise EffectiveReadbackError("remote launcher did not complete all four records")
    expected_case_ids = [str(row["case_id"]) for row in plan["expected_instances"]]
    if status.get("last_case_id") != expected_case_ids[-1]:
        raise EffectiveReadbackError(
            "successful launcher status does not name the frozen final case"
        )
    for field in ("run_id", "session_id"):
        value = status.get(field)
        if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
            raise EffectiveReadbackError(f"launcher status has invalid {field}")
    if status.get("source_revision") != source_revision or status.get("source_tree") != source_tree:
        raise EffectiveReadbackError("launcher source identity differs from local source")
    if status.get("freeze_a_config_sha256") != freeze_sha256:
        raise EffectiveReadbackError("launcher Freeze A hash mismatch")
    for field in ("source_archive_sha256", "snapshot_sha256", "selected_gpu_uuid"):
        value = status.get(field)
        if not isinstance(value, str) or (field != "selected_gpu_uuid" and _SHA256_RE.fullmatch(value) is None):
            raise EffectiveReadbackError(f"launcher status has invalid {field}")
    for field in ("selected_gpu_index", "elapsed_s", "gate0_root_bytes"):
        value = status.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EffectiveReadbackError(f"launcher status has invalid {field}")

    expected_provenance_keys = {
        "schema_version",
        "experiment_id",
        "run_id",
        "session_id",
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "snapshot_sha256",
        "project_path",
        "launcher_sha256",
        "probe_source_sha256",
        "freeze_a_config_sha256",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "asset_commit",
        "asset_git_tree",
        "canonical_lf_asset_tree_sha256",
        "isaaclab_commit",
        "selected_gpu_index",
        "selected_gpu_uuid",
        "selected_gpu_name",
        "driver_version",
        "expected_case_ids",
        "constraints",
        "environment",
        "recorded_at_utc",
    }
    if set(provenance) != expected_provenance_keys:
        raise EffectiveReadbackError("launcher provenance fields changed")
    expected_pairs = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "run_id": status["run_id"],
        "session_id": status["session_id"],
        "source_revision": source_revision,
        "source_tree": source_tree,
        "source_archive_sha256": status["source_archive_sha256"],
        "snapshot_sha256": status["snapshot_sha256"],
        "project_path": f"project/{source_tree}",
        "freeze_a_config_sha256": freeze_sha256,
        "gate0_manifest_file_sha256": manifest_file_sha256,
        "gate0_manifest_semantic_sha256": manifest_semantic_sha256,
        "asset_commit": plan["frozen_context"]["asset_commit"],
        "asset_git_tree": plan["frozen_context"]["asset_git_tree"],
        "canonical_lf_asset_tree_sha256": plan["frozen_context"][
            "canonical_lf_asset_tree_sha256"
        ],
        "selected_gpu_index": status["selected_gpu_index"],
        "selected_gpu_uuid": status["selected_gpu_uuid"],
        "expected_case_ids": expected_case_ids,
    }
    for field, expected in expected_pairs.items():
        if provenance.get(field) != expected:
            raise EffectiveReadbackError(f"launcher provenance mismatch: {field}")
    local_hashes = {
        "launcher_sha256": sha256_file(
            project_root / "scripts" / "run_ovphysx_effective_readback_remote.sh"
        ),
        "probe_source_sha256": sha256_file(
            project_root / "scripts" / "probe_ovphysx_effective_params.py"
        ),
    }
    for field, expected in local_hashes.items():
        if provenance.get(field) != expected:
            raise EffectiveReadbackError(f"launcher provenance source hash mismatch: {field}")
    constraints = _mapping(provenance.get("constraints"), "launcher.constraints")
    if constraints != {"headless": True, "kit": False, "renderer": False, "camera": False}:
        raise EffectiveReadbackError("launcher constraints changed")
    if not isinstance(provenance.get("environment"), Mapping):
        raise EffectiveReadbackError("launcher environment is missing")
    return status, provenance


def _load_instances(
    raw_root: Path,
    *,
    plan: Mapping[str, Any],
    manifest: object,
    expected_provenance: Mapping[str, object],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cases_root = raw_root / "cases"
    if not cases_root.is_dir() or is_link_like(cases_root):
        raise EffectiveReadbackError("raw evidence cases directory is missing")
    expected_ids = [str(row["case_id"]) for row in plan["expected_instances"]]
    actual_dirs = sorted(path.name for path in cases_root.iterdir() if path.is_dir())
    if actual_dirs != sorted(expected_ids):
        raise EffectiveReadbackError("raw evidence case directories are not exactly Freeze A")
    unexpected = [path.name for path in cases_root.iterdir() if not path.is_dir()]
    if unexpected:
        raise EffectiveReadbackError(f"cases root contains unexpected entries: {unexpected}")
    instances: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    from wave_asset_qa.parity.effective_readback import validate_readback_instance

    for case_id in expected_ids:
        case_dir = cases_root / case_id
        inventory = case_dir / "evidence.sha256"
        _verify_sha256_inventory(
            case_dir,
            inventory,
            excluded={"evidence.sha256"},
        )
        exit_code_path = case_dir / "worker.exit-code.txt"
        try:
            if exit_code_path.read_text(encoding="ascii").splitlines() != ["0"]:
                raise EffectiveReadbackError(f"worker did not exit cleanly: {case_id}")
        except (OSError, UnicodeError) as exc:
            raise EffectiveReadbackError(f"cannot read worker exit code: {case_id}") from exc
        payload_path = case_dir / f"{case_id}.json"
        payload = load_json_strict(payload_path, label=f"worker payload {case_id}")
        validated = validate_readback_instance(
            plan,
            manifest,  # type: ignore[arg-type]
            payload,
            expected_provenance=expected_provenance,
        )
        instances.append(validated)
        hashes[case_id] = sha256_file(payload_path)
    json_candidates = sorted(
        path.relative_to(cases_root).as_posix()
        for path in cases_root.rglob("*.json")
    )
    expected_json = sorted(f"{case_id}/{case_id}.json" for case_id in expected_ids)
    if json_candidates != expected_json:
        raise EffectiveReadbackError("raw case JSON inventory contains unexpected files")
    return instances, hashes


def finalize_effective_readback(
    *,
    gate0_manifest_path: str | Path,
    freeze_config_path: str | Path,
    raw_evidence_root: str | Path,
    output_dir: str | Path,
    source_revision: str,
) -> tuple[Path, dict[str, object]]:
    """Validate, sanitize, hash, and atomically promote one Freeze A bundle."""

    project_root = Path(__file__).resolve(strict=True).parents[1]
    manifest_path = _strict_source(gate0_manifest_path, "Gate 0 manifest", directory=False)
    freeze_path = _strict_source(freeze_config_path, "Freeze A config", directory=False)
    if manifest_path != project_root / "configs" / "parity" / "gate0.json":
        raise EffectiveReadbackError("finalizer requires the canonical Gate 0 manifest")
    if freeze_path != project_root / "configs" / "parity" / "ovphysx_effective_readback.json":
        raise EffectiveReadbackError("finalizer requires the canonical Freeze A config")
    required_tracked = (
        "configs/parity/gate0.json",
        "configs/parity/ovphysx_effective_readback.json",
        "src/wave_asset_qa/parity/effective_readback.py",
        "src/wave_asset_qa/adapters/ovphysx.py",
        "scripts/probe_ovphysx_effective_params.py",
        "scripts/run_ovphysx_effective_readback_remote.sh",
        "scripts/finalize_ovphysx_effective_readback.py",
    )
    source_tree = validate_git_source(
        project_root,
        source_revision,
        required_tracked_paths=required_tracked,
        sensitive_untracked_paths=("configs", "src", "scripts"),
    )
    plan, manifest = load_frozen_contract(freeze_path, manifest_path)
    freeze_sha = sha256_file(freeze_path)
    manifest_file_sha = sha256_file(manifest_path)
    manifest_semantic_sha = str(plan["frozen_context"]["gate0_manifest_semantic_sha256"])
    raw_root = _strict_source(raw_evidence_root, "raw evidence root", directory=True)
    _verify_no_links(raw_root)
    status, launcher_provenance = _validate_launcher(
        raw_root,
        plan=plan,
        source_revision=source_revision,
        source_tree=source_tree,
        freeze_sha256=freeze_sha,
        manifest_file_sha256=manifest_file_sha,
        manifest_semantic_sha256=manifest_semantic_sha,
        project_root=project_root,
    )
    _verify_sha256_inventory(
        raw_root,
        raw_root / "launcher" / "evidence.sha256",
        excluded={"launcher/evidence.sha256"},
    )
    worker_expected = {
        "source_revision": source_revision,
        "source_tree": source_tree,
        "source_archive_sha256": launcher_provenance["source_archive_sha256"],
        "source_snapshot_sha256": launcher_provenance["snapshot_sha256"],
        "probe_source_sha256": launcher_provenance["probe_source_sha256"],
        "pipeline_module_sha256": sha256_file(
            project_root
            / "src"
            / "wave_asset_qa"
            / "parity"
            / "effective_readback.py"
        ),
        "freeze_a_config_sha256": freeze_sha,
        "gate0_manifest_file_sha256": manifest_file_sha,
        "gate0_manifest_semantic_sha256": manifest_semantic_sha,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "canonical_lf_asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "device": "cuda:0",
        "gpu_name": launcher_provenance["selected_gpu_name"],
        "gpu_uuid": launcher_provenance["selected_gpu_uuid"],
        "driver_version": launcher_provenance["driver_version"],
        "session_id": launcher_provenance["session_id"],
    }
    instances, instance_hashes = _load_instances(
        raw_root,
        plan=plan,
        manifest=manifest,
        expected_provenance=worker_expected,
    )
    payload = assemble_readback_payload(
        plan,
        manifest,
        instances,
        expected_provenance=worker_expected,
    )
    summary = sanitize_readback_payload(
        plan, payload, instance_file_sha256=instance_hashes
    )
    report = render_readback_report(summary)

    destination = validated_results_output_path(project_root, output_dir)
    try:
        destination.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise EffectiveReadbackError(
            "output directory must not be inside the raw evidence root"
        )
    try:
        raw_root.relative_to(destination)
    except ValueError:
        pass
    else:
        raise EffectiveReadbackError(
            "raw evidence root must not be inside the output directory"
        )
    staging = create_staging_root(destination)
    source_hashes = {
        "finalizer_script_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "pipeline_module_sha256": sha256_file(
            project_root / "src" / "wave_asset_qa" / "parity" / "effective_readback.py"
        ),
        "adapter_module_sha256": sha256_file(
            project_root / "src" / "wave_asset_qa" / "adapters" / "ovphysx.py"
        ),
        "worker_script_sha256": sha256_file(
            project_root / "scripts" / "probe_ovphysx_effective_params.py"
        ),
        "launcher_script_sha256": sha256_file(
            project_root / "scripts" / "run_ovphysx_effective_readback_remote.sh"
        ),
    }
    try:
        private_dir = staging / "private"
        results_dir = staging / "results"
        inputs_dir = staging / "inputs"
        source_dir = inputs_dir / "source"
        private_dir.mkdir()
        results_dir.mkdir()
        inputs_dir.mkdir()
        source_dir.mkdir()
        _copy_tree_strict(raw_root, private_dir / "remote-evidence")
        write_json_atomic(private_dir / "readback.json", payload)
        shutil.copyfile(manifest_path, inputs_dir / "gate0.json")
        shutil.copyfile(freeze_path, inputs_dir / "ovphysx_effective_readback.json")
        for relative in (
            "src/wave_asset_qa/parity/effective_readback.py",
            "src/wave_asset_qa/adapters/ovphysx.py",
            "scripts/probe_ovphysx_effective_params.py",
            "scripts/run_ovphysx_effective_readback_remote.sh",
            "scripts/finalize_ovphysx_effective_readback.py",
        ):
            source = project_root / relative
            destination_source = source_dir / Path(relative).name
            shutil.copyfile(source, destination_source)
        write_json_atomic(results_dir / "summary.json", summary)
        write_text_exclusive(staging / "report.md", report)
        write_json_atomic(
            results_dir / "finalization.json",
            {
                "schema_version": 1,
                "experiment_id": plan["experiment_id"],
                "status": payload["status"],
                "private_bundle": True,
                "raw_evidence_visibility": "private",
                "public_safe_outputs": ["results/summary.json", "report.md"],
                "source_revision": source_revision,
                "source_tree": source_tree,
                "source_hashes": source_hashes,
                "freeze_a_config_sha256": freeze_sha,
                "gate0_manifest_file_sha256": manifest_file_sha,
                "gate0_manifest_semantic_sha256": manifest_semantic_sha,
                "remote_launcher_run_id": status["run_id"],
                "instance_file_sha256": instance_hashes,
                "formal_gate0_status_unchanged": "DIVERGENT",
                "formal_gate0_pass_ready_unchanged": False,
                "freeze_b_required": True,
            },
        )
        observed_tree = validate_git_source(
            project_root,
            source_revision,
            required_tracked_paths=required_tracked,
            sensitive_untracked_paths=("configs", "src", "scripts"),
        )
        if observed_tree != source_tree:
            raise EffectiveReadbackError("local source tree changed during finalization")
        observed_hashes = {
            "finalizer_script_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "pipeline_module_sha256": sha256_file(
                project_root / "src" / "wave_asset_qa" / "parity" / "effective_readback.py"
            ),
            "adapter_module_sha256": sha256_file(
                project_root / "src" / "wave_asset_qa" / "adapters" / "ovphysx.py"
            ),
            "worker_script_sha256": sha256_file(
                project_root / "scripts" / "probe_ovphysx_effective_params.py"
            ),
            "launcher_script_sha256": sha256_file(
                project_root / "scripts" / "run_ovphysx_effective_readback_remote.sh"
            ),
        }
        if observed_hashes != source_hashes:
            raise EffectiveReadbackError("local source files changed during finalization")
        write_bundle_manifest(staging, bundle_payload_paths(staging))
        verification = dict(verify_exact_bundle(staging))
        verification.update(
            {
                "status": payload["status"],
                "coverage_44_of_44": True,
                "fresh_process_count": 4,
                "repeatability_exact": True,
                "formal_gate0_unchanged": True,
                "private_bundle": True,
            }
        )
        promoted = promote_staging_root(staging, destination)
        final_verification = dict(verify_exact_bundle(promoted))
        if final_verification["root_sha256"] != verification["root_sha256"]:
            raise EffectiveReadbackError("bundle changed during atomic promotion")
        return promoted, verification
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, verification = finalize_effective_readback(
            gate0_manifest_path=args.gate0_manifest,
            freeze_config_path=args.freeze_config,
            raw_evidence_root=args.raw_evidence_root,
            output_dir=args.output_dir,
            source_revision=args.source_revision,
        )
    except (
        DiagnosticEvidenceError,
        EffectiveReadbackError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"Effective readback finalization failed: {exc}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {"output_dir": str(output), **verification},
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
