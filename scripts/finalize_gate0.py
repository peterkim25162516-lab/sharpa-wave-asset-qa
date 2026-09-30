#!/usr/bin/env python3
"""Validate, compare, report, and hash one complete 32-case Gate 0 run."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile

import wave_asset_qa

from wave_asset_qa.parity.bundle import (
    DEFAULT_MANIFEST_PATH,
    verify_bundle,
    write_bundle_manifest,
    write_json_atomic,
)
from wave_asset_qa.parity.compare import CollectedRun, compare_gate0_runs
from wave_asset_qa.parity.contracts import ParityManifest, Simulator
from wave_asset_qa.parity.report import write_reports
from wave_asset_qa.parity.runner import RUN_FILE_SUFFIX, load_collected_runs
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPECTED_BACKEND_CASE_COUNT = 16
EXPECTED_TOTAL_CASE_COUNT = 32
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_EXPECTED_DRIVER_VERSION = "570.158.01"
_EXPECTED_GPU_NAME_FRAGMENT = "A800-SXM4-40GB"
_MAX_IDLE_MEMORY_MIB = 32
_EXPECTED_ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
_EXPECTED_REMOTE_ENVIRONMENT = {
    "python": "3.12.14",
    "isaaclab": "6.1.14",
    "isaaclab-ovphysx": "3.0.2",
    "ovphysx": "0.4.13",
    "torch": "2.10.0+cu128",
    "usd-core": "25.11",
    "kitless": "true",
}
_PROBE_TOP_LEVEL_KEYS = {
    "backend",
    "mode",
    "status",
    "message",
    "capabilities",
    "provenance",
    "error",
}
_REQUIRED_SCHEMA_CAPABILITIES = {
    "kitless_contract",
    "renderer_disabled",
    "camera_disabled",
    "pinned_ovphysx_wheel_version",
    "isaaclab_import",
    "simulation_cfg_symbol",
    "build_simulation_context_symbol",
    "ovphysx_cfg_symbol",
    "forbidden_modules_absent",
    "resolved_usd_open",
    "stage_fully_composed",
    "articulation_root_schema",
    "canonical_joint_count",
    "position_drive_schema",
    "position_drive_values_finite",
    "position_drive_stiffness_positive",
    "physx_properties_authored",
    "physx_max_joint_velocity_values_finite_positive",
    "canonical_distal_frame_count",
}


class Gate0FinalizationError(RuntimeError):
    """Raised when evidence identity, topology, or provenance is invalid."""


def _resolved_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Gate0FinalizationError(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Gate0FinalizationError(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise Gate0FinalizationError(f"{label} must be a directory: {resolved}")
    return resolved


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Gate0FinalizationError(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Gate0FinalizationError(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise Gate0FinalizationError(f"{label} must be a regular file: {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _new_destination(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output path already exists; refusing overwrite: {requested}")
    if requested.name in {"", ".", ".."}:
        raise Gate0FinalizationError("output path must name a new directory")
    parent = _resolved_directory(requested.parent, "output parent")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"output path already exists; refusing overwrite: {destination}"
        )
    return parent, destination


def _validate_regular_tree(root: Path, label: str) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Gate0FinalizationError(
                f"{label} contains a symbolic link: {path.relative_to(root)}"
            )
        mode = path.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise Gate0FinalizationError(
                f"{label} contains a non-regular filesystem entry: "
                f"{path.relative_to(root)}"
            )


def _copy_tree_strict(source: Path, destination: Path, label: str) -> None:
    _validate_regular_tree(source, label)
    destination.mkdir()
    for source_path in sorted(
        source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()
    ):
        relative = source_path.relative_to(source)
        target = destination / relative
        if source_path.is_dir():
            target.mkdir()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Gate0FinalizationError(
            f"cannot verify the finalizer Git source: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise Gate0FinalizationError(
            "cannot verify the finalizer Git source"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    """Return the commit tree after proving this finalizer uses that clean checkout."""

    root = project_root.resolve(strict=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise Gate0FinalizationError(
            f"finalizer project root is not the Git top level: {root} != {top_level}"
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise Gate0FinalizationError(
            f"source_revision does not match local HEAD: {source_revision} != {head}"
        )
    source_tree = _git_text(
        root, "rev-parse", "--verify", f"{source_revision}^{{tree}}"
    )
    if _GIT_SHA1_RE.fullmatch(source_tree) is None:
        raise Gate0FinalizationError("source_revision did not resolve to a Git tree")
    tracked_changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_changes:
        raise Gate0FinalizationError(
            "tracked worktree/index must be clean during Gate 0 finalization"
        )

    tracked_sources = (
        "scripts/finalize_gate0.py",
        "scripts/probe_ovphysx_schema.py",
        "scripts/run_gate0_local.py",
        "scripts/run_gate0_remote.sh",
        "scripts/probe_ovphysx_runtime.py",
        "src/wave_asset_qa/parity/bundle.py",
        "src/wave_asset_qa/parity/compare.py",
        "src/wave_asset_qa/parity/contracts.py",
        "src/wave_asset_qa/parity/mapping.py",
        "src/wave_asset_qa/parity/report.py",
        "src/wave_asset_qa/parity/runner.py",
        "src/wave_asset_qa/parity/scenarios.py",
    )
    for relative in tracked_sources:
        _git_text(root, "ls-files", "--error-unmatch", "--", relative)

    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise Gate0FinalizationError(
            "wave_asset_qa was imported from a source tree other than this checkout"
        )
    return source_tree


def _strict_json_file(path: Path) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Gate0FinalizationError(
                    f"duplicate JSON key {key!r} in {path.name}"
                )
            result[key] = value
        return result

    def no_constant(token: str) -> object:
        raise Gate0FinalizationError(
            f"non-finite JSON value {token!r} in {path.name}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Gate0FinalizationError(
            f"cannot read formal schema probe {path.name}: {exc}"
        ) from exc


def _string_set(value: object, label: str) -> set[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise Gate0FinalizationError(f"{label} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise Gate0FinalizationError(f"{label} must not contain duplicates")
    return set(value)


def _exact_int(value: object, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise Gate0FinalizationError(f"{label} must equal {expected}")


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "finite and positive" if positive else "finite"
        raise Gate0FinalizationError(f"{label} must be {qualifier}")
    return float(value)


def _validate_usd_value_records(
    inspection: Mapping[str, object],
    *,
    hand_side: str,
    expected_joints: set[str],
) -> None:
    drive_records = inspection.get("angular_drive_records")
    if not isinstance(drive_records, list) or len(drive_records) != 22:
        raise Gate0FinalizationError(
            f"{hand_side} angular_drive_records must contain exactly 22 records"
        )
    observed_drive_joints: set[str] = set()
    drive_keys = {
        "joint_name",
        "type",
        "stiffness",
        "damping",
        "max_force",
        "target_position",
    }
    for index, record in enumerate(drive_records):
        if not isinstance(record, Mapping) or set(record) != drive_keys:
            raise Gate0FinalizationError(
                f"{hand_side} angular_drive_records[{index}] has invalid fields"
            )
        joint_name = record["joint_name"]
        if not isinstance(joint_name, str) or not joint_name:
            raise Gate0FinalizationError(
                f"{hand_side} angular drive joint_name is invalid"
            )
        observed_drive_joints.add(joint_name)
        if record["type"] != "force":
            raise Gate0FinalizationError(
                f"{hand_side} angular drive {joint_name!r} is not force mode"
            )
        _finite_number(
            record["stiffness"],
            f"{hand_side} angular drive {joint_name!r} stiffness",
            positive=True,
        )
        for field in ("damping", "max_force", "target_position"):
            _finite_number(
                record[field],
                f"{hand_side} angular drive {joint_name!r} {field}",
            )
    if observed_drive_joints != expected_joints or len(observed_drive_joints) != 22:
        raise Gate0FinalizationError(
            f"{hand_side} angular_drive_records do not cover the canonical joints"
        )

    velocity_records = inspection.get("physx_velocity_records")
    if not isinstance(velocity_records, list) or len(velocity_records) != 22:
        raise Gate0FinalizationError(
            f"{hand_side} physx_velocity_records must contain exactly 22 records"
        )
    observed_velocity_joints: set[str] = set()
    for index, record in enumerate(velocity_records):
        if not isinstance(record, Mapping) or set(record) != {
            "joint_name",
            "max_joint_velocity",
        }:
            raise Gate0FinalizationError(
                f"{hand_side} physx_velocity_records[{index}] has invalid fields"
            )
        joint_name = record["joint_name"]
        if not isinstance(joint_name, str) or not joint_name:
            raise Gate0FinalizationError(
                f"{hand_side} PhysX velocity joint_name is invalid"
            )
        observed_velocity_joints.add(joint_name)
        _finite_number(
            record["max_joint_velocity"],
            f"{hand_side} PhysX max velocity {joint_name!r}",
            positive=True,
        )
    if (
        observed_velocity_joints != expected_joints
        or len(observed_velocity_joints) != 22
    ):
        raise Gate0FinalizationError(
            f"{hand_side} physx_velocity_records do not cover the canonical joints"
        )


def _verify_schema_hashes(schema_dir: Path) -> None:
    hash_path = schema_dir / "evidence.sha256"
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Gate0FinalizationError("cannot read schema evidence.sha256") from exc
    expected_names = {
        f"{hand}.{suffix}"
        for hand in ("left", "right")
        for suffix in ("json", "stdout", "stderr", "exitcode")
    }
    records: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None:
            raise Gate0FinalizationError(
                "schema evidence.sha256 contains a non-canonical record"
            )
        digest, name = match.groups()
        if name in records:
            raise Gate0FinalizationError(
                f"schema evidence.sha256 repeats {name!r}"
            )
        records[name] = digest
    if set(records) != expected_names:
        raise Gate0FinalizationError(
            "schema evidence.sha256 must cover exactly the eight formal probe files"
        )
    for name, expected_digest in records.items():
        if _sha256_file(schema_dir / name) != expected_digest:
            raise Gate0FinalizationError(f"formal schema probe hash mismatch: {name}")


def _validate_schema_probe(
    payload: object,
    *,
    hand_side: str,
    manifest: ParityManifest,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
    schema_probe_script_sha256: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _PROBE_TOP_LEVEL_KEYS:
        raise Gate0FinalizationError(
            f"{hand_side} formal schema probe has invalid top-level fields"
        )
    if (
        payload["backend"] != Simulator.OVPHYSX.value
        or payload["mode"] != "resolved_usd_headless"
        or payload["status"] != "available"
        or payload["error"] is not None
    ):
        raise Gate0FinalizationError(
            f"{hand_side} formal schema probe is not resolved-USD available"
        )
    capabilities = payload["capabilities"]
    if not isinstance(capabilities, Mapping):
        raise Gate0FinalizationError(
            f"{hand_side} formal schema capabilities must be an object"
        )
    failed_capabilities = sorted(
        name for name in _REQUIRED_SCHEMA_CAPABILITIES if capabilities.get(name) is not True
    )
    if failed_capabilities:
        raise Gate0FinalizationError(
            f"{hand_side} formal schema probe failed capabilities: "
            + ", ".join(failed_capabilities)
        )
    if capabilities.get("physics_step") is not False:
        raise Gate0FinalizationError(
            f"{hand_side} schema probe must not claim physics stepping"
        )

    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise Gate0FinalizationError(
            f"{hand_side} formal schema provenance must be an object"
        )
    expected_identity: dict[str, object] = {
        "hand": hand_side,
        "session_id": session_id,
        "source_revision": source_revision,
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_file_sha256": manifest_file_sha256,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "schema_probe_script_sha256": schema_probe_script_sha256,
    }
    mismatches = [
        key for key, value in expected_identity.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise Gate0FinalizationError(
            f"{hand_side} formal schema probe has invalid provenance field(s): "
            + ", ".join(sorted(mismatches))
        )
    inspection = provenance.get("usd_inspection")
    if not isinstance(inspection, Mapping):
        raise Gate0FinalizationError(
            f"{hand_side} formal schema probe lacks usd_inspection"
        )
    if inspection.get("stage_load_policy") != "load_all":
        raise Gate0FinalizationError(
            f"{hand_side} resolved USD stage was not fully composed"
        )
    prim_count = inspection.get("prim_count")
    if isinstance(prim_count, bool) or not isinstance(prim_count, int) or prim_count < 1:
        raise Gate0FinalizationError(
            f"{hand_side} resolved USD stage must contain at least one prim"
        )
    _exact_int(
        inspection.get("revolute_joint_count"), 22, f"{hand_side} revolute_joint_count"
    )
    _exact_int(
        inspection.get("angular_drive_count"), 22, f"{hand_side} angular_drive_count"
    )
    _exact_int(
        inspection.get("physx_velocity_joint_count"),
        22,
        f"{hand_side} physx_velocity_joint_count",
    )
    _exact_int(
        inspection.get("distal_frame_count"), 5, f"{hand_side} distal_frame_count"
    )
    roots = _string_set(
        inspection.get("articulation_root_paths"),
        f"{hand_side} articulation_root_paths",
    )
    if len(roots) != 1:
        raise Gate0FinalizationError(
            f"{hand_side} resolved USD must have exactly one articulation root"
        )

    hand = manifest.hand(next(item.side for item in manifest.hands if item.side.value == hand_side))
    expected_joints = set(hand.joint_names)
    expected_frames = set(hand.distal_frame_names)
    _validate_usd_value_records(
        inspection,
        hand_side=hand_side,
        expected_joints=expected_joints,
    )
    for field in (
        "revolute_joint_names",
        "angular_drive_joint_names",
        "physx_velocity_joint_names",
    ):
        if _string_set(inspection.get(field), f"{hand_side} {field}") != expected_joints:
            raise Gate0FinalizationError(
                f"{hand_side} {field} does not match the 22 canonical joints"
            )
    if (
        _string_set(
            inspection.get("distal_frame_names"), f"{hand_side} distal_frame_names"
        )
        != expected_frames
    ):
        raise Gate0FinalizationError(
            f"{hand_side} distal_frame_names do not match the five canonical frames"
        )
    source_path = inspection.get("source_path")
    expected_model_path = hand.model_paths.ovphysx.replace("\\", "/")
    if source_path != expected_model_path:
        raise Gate0FinalizationError(
            f"{hand_side} schema probe source_path must be the portable manifest USD path"
        )
    if provenance.get("model_path") != expected_model_path:
        raise Gate0FinalizationError(
            f"{hand_side} schema probe model_path does not match the manifest USD"
        )
    imported_modules = provenance.get("imported_modules")
    if not isinstance(imported_modules, Mapping) or any(
        not isinstance(record, Mapping) or "file" in record
        for record in imported_modules.values()
    ):
        raise Gate0FinalizationError(
            f"{hand_side} schema probe imported-module provenance is not portable"
        )
    if (
        not isinstance(inspection.get("source_sha256"), str)
        or _SHA256_RE.fullmatch(str(inspection.get("source_sha256"))) is None
    ):
        raise Gate0FinalizationError(
            f"{hand_side} schema probe source_sha256 is invalid"
        )


def _validate_formal_schema_evidence(
    remote_evidence_root: Path,
    manifest: ParityManifest,
    *,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
    schema_probe_script_sha256: str,
) -> None:
    schema_dir = remote_evidence_root / "launcher" / "schema"
    if schema_dir.is_symlink() or not schema_dir.is_dir():
        raise Gate0FinalizationError(
            "OVPhysX evidence lacks launcher/schema formal resolved-USD probes"
        )
    expected_names = {
        "evidence.sha256",
        *(
            f"{hand}.{suffix}"
            for hand in ("left", "right")
            for suffix in ("json", "stdout", "stderr", "exitcode")
        ),
    }
    entries = tuple(schema_dir.iterdir())
    observed_names = {path.name for path in entries}
    if (
        observed_names != expected_names
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise Gate0FinalizationError(
            "launcher/schema must contain exactly two complete formal probe evidence sets"
        )
    _verify_schema_hashes(schema_dir)
    for hand_side in ("left", "right"):
        exitcode = (schema_dir / f"{hand_side}.exitcode").read_text(
            encoding="utf-8"
        )
        if exitcode != "0\n":
            raise Gate0FinalizationError(
                f"{hand_side} formal schema probe exitcode must be exactly zero"
            )
        _validate_schema_probe(
            _strict_json_file(schema_dir / f"{hand_side}.json"),
            hand_side=hand_side,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            session_id=session_id,
            source_revision=source_revision,
            schema_probe_script_sha256=schema_probe_script_sha256,
        )


def _parse_key_value_file(path: Path, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Gate0FinalizationError(f"cannot read {label}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or "=" not in line:
            raise Gate0FinalizationError(f"{label} contains a non-canonical line")
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None or not value:
            raise Gate0FinalizationError(f"{label} contains an invalid key/value")
        if key in values:
            raise Gate0FinalizationError(f"{label} repeats key {key!r}")
        values[key] = value
    return values


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Gate0FinalizationError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_gpu_state(path: Path, label: str) -> tuple[dict[str, str], tuple[str, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Gate0FinalizationError(f"cannot read {label}") from exc
    try:
        begin = lines.index("compute_processes_begin")
        end = lines.index("compute_processes_end")
    except ValueError as exc:
        raise Gate0FinalizationError(
            f"{label} lacks the compute-process evidence envelope"
        ) from exc
    if begin >= end or end != len(lines) - 1:
        raise Gate0FinalizationError(
            f"{label} has an invalid compute-process evidence envelope"
        )
    fields: dict[str, str] = {}
    for line in lines[:begin]:
        if "=" not in line:
            raise Gate0FinalizationError(f"{label} contains an invalid field")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise Gate0FinalizationError(f"{label} contains a duplicate/empty field")
        fields[key] = value
    processes = tuple(line.strip() for line in lines[begin + 1 : end] if line.strip())
    required = {
        "recorded_at_utc",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "driver_version",
        "memory_used_mib",
        "utilization_percent",
    }
    if set(fields) != required:
        raise Gate0FinalizationError(f"{label} has invalid GPU-state fields")
    return fields, processes


def _validate_idle_gpu_state(
    path: Path,
    *,
    label: str,
    gpu_index: str,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
) -> None:
    fields, processes = _parse_gpu_state(path, label)
    expected = {
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "utilization_percent": "0",
    }
    mismatches = [
        key for key, value in expected.items() if fields.get(key) != value
    ]
    memory_text = fields.get("memory_used_mib", "")
    if re.fullmatch(r"[0-9]+", memory_text) is None:
        mismatches.append("memory_used_mib")
    elif int(memory_text) > _MAX_IDLE_MEMORY_MIB:
        mismatches.append("memory_used_mib")
    if mismatches or processes:
        detail = ", ".join(sorted(set(mismatches))) or "compute_processes"
        raise Gate0FinalizationError(f"{label} does not prove an idle pinned GPU: {detail}")


def _normalize_hash_member(raw: str, label: str) -> str:
    candidate = raw[2:] if raw.startswith("./") else raw
    if not candidate or "\\" in candidate:
        raise Gate0FinalizationError(f"{label} contains an unsafe hash path")
    pure = PurePosixPath(candidate)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise Gate0FinalizationError(f"{label} contains an unsafe hash path")
    return pure.as_posix()


def _verify_evidence_hash_file(root: Path, hash_path: Path, label: str) -> None:
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Gate0FinalizationError(f"cannot read {label}") from exc
    records: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise Gate0FinalizationError(f"{label} contains a non-canonical record")
        digest, _mode, raw_path = match.groups()
        relative = _normalize_hash_member(raw_path, label)
        if relative in records:
            raise Gate0FinalizationError(f"{label} repeats {relative!r}")
        records[relative] = digest

    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != hash_path
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise Gate0FinalizationError(
            f"{label} does not cover every raw case file: missing={missing}, extra={extra}"
        )
    for relative, expected_digest in records.items():
        member = (root / PurePosixPath(relative)).resolve()
        if not _is_within(member, root) or not member.is_file() or member.is_symlink():
            raise Gate0FinalizationError(f"{label} references an unsafe member")
        if _sha256_file(member) != expected_digest:
            raise Gate0FinalizationError(
                f"{label} digest mismatch for {relative!r}"
            )


def _validate_local_launcher_evidence(
    local_root: Path,
    manifest: ParityManifest,
    *,
    expected_source_tree: str,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
) -> None:
    launcher = local_root / "launcher"
    if launcher.is_symlink() or not launcher.is_dir():
        raise Gate0FinalizationError("MuJoCo evidence lacks a regular launcher directory")
    expected_names = {
        "completed.json",
        "evidence.sha256",
        "gate0.manifest.json",
        "matrix.json",
        "provenance.json",
    }
    entries = tuple(launcher.iterdir())
    if (
        {path.name for path in entries} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise Gate0FinalizationError(
            "MuJoCo launcher must contain exactly its canonical five evidence files"
        )
    _verify_evidence_hash_file(
        launcher,
        launcher / "evidence.sha256",
        "MuJoCo launcher/evidence.sha256",
    )

    manifest_snapshot = launcher / "gate0.manifest.json"
    if load_manifest(manifest_snapshot) != manifest:
        raise Gate0FinalizationError("MuJoCo launcher manifest snapshot is not canonical")
    project_root = Path(__file__).resolve().parents[1]
    common: dict[str, object] = {
        "schema_version": 1,
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": expected_source_tree,
        "launcher_script_sha256": _sha256_file(
            project_root / "scripts" / "run_gate0_local.py"
        ),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "manifest_snapshot_sha256": _sha256_file(manifest_snapshot),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
    }
    expected_cases = _expected_backend_cases(manifest, Simulator.MUJOCO)
    case_ids = [case.case_id for case in expected_cases]
    expected_payloads: dict[str, dict[str, object]] = {
        "provenance.json": dict(common),
        "matrix.json": {
            **common,
            "case_count": EXPECTED_BACKEND_CASE_COUNT,
            "case_ids": case_ids,
        },
        "completed.json": {
            **common,
            "completed_case_count": EXPECTED_BACKEND_CASE_COUNT,
            "case_ids": case_ids,
        },
    }
    for name, expected in expected_payloads.items():
        observed = _strict_json_file(launcher / name)
        if observed != expected:
            raise Gate0FinalizationError(
                f"MuJoCo launcher/{name} does not match the pinned run identity"
            )


def _validate_local_case_evidence(
    local_root: Path,
    manifest: ParityManifest,
    *,
    session_id: str,
    source_revision: str,
) -> None:
    cases_dir = local_root / "cases"
    for case in _expected_backend_cases(manifest, Simulator.MUJOCO):
        case_dir = cases_dir / case.case_id
        expected_names = {
            "evidence.sha256",
            "launcher.command.json",
            "launcher.exitcode.txt",
            "launcher.stderr.log",
            "launcher.stdout.log",
            f"{case.case_id}{RUN_FILE_SUFFIX}",
        }
        entries = tuple(case_dir.iterdir())
        if (
            {path.name for path in entries} != expected_names
            or any(path.is_symlink() or not path.is_file() for path in entries)
        ):
            raise Gate0FinalizationError(
                f"{case.case_id} local evidence topology is not canonical"
            )
        hash_path = case_dir / "evidence.sha256"
        _verify_evidence_hash_file(
            case_dir, hash_path, f"{case.case_id}/evidence.sha256"
        )
        if (case_dir / "launcher.exitcode.txt").read_text(encoding="utf-8") != "0\n":
            raise Gate0FinalizationError(
                f"{case.case_id} local launcher exit code must be exactly zero"
            )
        command = _strict_json_file(case_dir / "launcher.command.json")
        expected_argv = [
            "<python>",
            "-P",
            "-m",
            "wave_asset_qa.parity.runner",
            "--backend",
            Simulator.MUJOCO.value,
            "--asset-root",
            "<asset-root>",
            "--manifest",
            "launcher/gate0.manifest.json",
            "--output-dir",
            f"cases/{case.case_id}",
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--case-id",
            case.case_id,
        ]
        if (
            not isinstance(command, Mapping)
            or set(command) != {"argv", "case_id", "schema_version"}
            or command.get("schema_version") != 1
            or command.get("case_id") != case.case_id
            or command.get("argv") != expected_argv
        ):
            raise Gate0FinalizationError(
                f"{case.case_id} local launcher command record is invalid"
            )


def _validate_remote_launcher_evidence(
    remote_root: Path,
    manifest: ParityManifest,
    *,
    expected_source_tree: str,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
) -> dict[str, str]:
    launcher = remote_root / "launcher"
    if launcher.is_symlink() or not launcher.is_dir():
        raise Gate0FinalizationError("OVPhysX evidence lacks a regular launcher directory")
    provenance_path = launcher / "provenance.txt"
    status_path = launcher / "status.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (provenance_path, status_path)
    ):
        raise Gate0FinalizationError(
            "OVPhysX launcher provenance.txt and status.json are required"
        )
    provenance = _parse_key_value_file(
        provenance_path, "OVPhysX launcher/provenance.txt"
    )
    status = _strict_json_file(status_path)
    if not isinstance(status, Mapping):
        raise Gate0FinalizationError("OVPhysX launcher/status.json must be an object")
    status_expected: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "scientific_verdict": "pending_local_compare",
        "exit_code": 0,
        "session_id": session_id,
        "source_revision": source_revision,
        "completed_case_count": EXPECTED_BACKEND_CASE_COUNT,
        "completed_schema_probe_count": 2,
        "last_case_id": None,
    }
    status_mismatches = [
        key for key, value in status_expected.items() if status.get(key) != value
    ]
    if status_mismatches:
        raise Gate0FinalizationError(
            "OVPhysX launcher/status.json is not a completed 16-case run: "
            + ", ".join(sorted(status_mismatches))
        )

    required_provenance: dict[str, str] = {
        "schema_version": "1",
        "session_id": session_id,
        "source_revision": source_revision,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_lf_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "isaaclab_commit": _EXPECTED_ISAACLAB_COMMIT,
        "driver_version": _EXPECTED_DRIVER_VERSION,
        **_EXPECTED_REMOTE_ENVIRONMENT,
    }
    provenance_mismatches = [
        key for key, value in required_provenance.items() if provenance.get(key) != value
    ]
    if provenance_mismatches:
        raise Gate0FinalizationError(
            "OVPhysX launcher/provenance.txt has invalid pinned field(s): "
            + ", ".join(sorted(provenance_mismatches))
        )
    for key in (
        "archive_sha256",
        "launcher_sha256",
        "worker_sha256",
    ):
        _require_digest(provenance.get(key), f"launcher provenance {key}")
    source_tree = provenance.get("source_tree")
    if source_tree is None or _GIT_SHA1_RE.fullmatch(source_tree) is None:
        raise Gate0FinalizationError("launcher provenance source_tree is invalid")
    if source_tree != expected_source_tree:
        raise Gate0FinalizationError(
            "remote source_tree does not match source_revision^{tree}"
        )
    if provenance.get("project_path") != f"project/{source_tree}":
        raise Gate0FinalizationError("launcher provenance project_path/source_tree mismatch")
    gpu_uuid = provenance.get("gpu_uuid", "")
    gpu_index = provenance.get("gpu_index", "")
    gpu_name = provenance.get("gpu_name", "")
    if (
        _GPU_UUID_RE.fullmatch(gpu_uuid) is None
        or re.fullmatch(r"[0-7]", gpu_index) is None
        or _EXPECTED_GPU_NAME_FRAGMENT not in gpu_name
    ):
        raise Gate0FinalizationError("launcher provenance GPU identity is invalid")
    if (
        status.get("run_id") != provenance.get("run_id")
        or status.get("source_tree") != source_tree
        or status.get("selected_gpu_uuid") != gpu_uuid
        or status.get("selected_gpu_index") != int(gpu_index)
    ):
        raise Gate0FinalizationError(
            "launcher status/provenance do not bind the same run, source tree, and GPU"
        )

    project_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        "launcher_sha256": _sha256_file(project_root / "scripts" / "run_gate0_remote.sh"),
        "worker_sha256": _sha256_file(project_root / "scripts" / "probe_ovphysx_runtime.py"),
    }
    changed_sources = [
        key for key, value in source_hashes.items() if provenance.get(key) != value
    ]
    if changed_sources:
        raise Gate0FinalizationError(
            "remote launcher used source files different from this finalizer revision: "
            + ", ".join(sorted(changed_sources))
        )

    _validate_idle_gpu_state(
        launcher / "gpu-selection.txt",
        label="launcher GPU selection",
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version=_EXPECTED_DRIVER_VERSION,
    )
    return {
        "run_id": str(provenance["run_id"]),
        "source_tree": source_tree,
        "archive_sha256": str(provenance["archive_sha256"]),
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": _EXPECTED_DRIVER_VERSION,
        "worker_sha256": str(provenance["worker_sha256"]),
    }


def _validate_remote_case_evidence(
    remote_root: Path,
    manifest: ParityManifest,
    remote_identity: Mapping[str, str],
) -> None:
    cases_dir = remote_root / "cases"
    for case in _expected_backend_cases(manifest, Simulator.OVPHYSX):
        case_dir = cases_dir / case.case_id
        hash_path = case_dir / "evidence.sha256"
        if hash_path.is_symlink() or not hash_path.is_file():
            raise Gate0FinalizationError(
                f"{case.case_id} lacks its raw evidence.sha256"
            )
        _verify_evidence_hash_file(
            case_dir, hash_path, f"{case.case_id}/evidence.sha256"
        )
        required_files = {
            f"{case.case_id}{RUN_FILE_SUFFIX}",
            f"{case.case_id}.worker.result.json",
            f"{case.case_id}.worker.stdout.log",
            f"{case.case_id}.worker.stderr.log",
            f"{case.case_id}.worker.exitcode.txt",
            "launcher.stdout.txt",
            "launcher.stderr.txt",
            "launcher.exit-code.txt",
            "preflight.txt",
        }
        observed_files = {
            path.relative_to(case_dir).as_posix()
            for path in case_dir.rglob("*")
            if path.is_file()
        }
        missing = sorted(required_files - observed_files)
        if missing:
            raise Gate0FinalizationError(
                f"{case.case_id} lacks required raw evidence: {missing}"
            )
        for exit_name in (
            f"{case.case_id}.worker.exitcode.txt",
            "launcher.exit-code.txt",
        ):
            if (case_dir / exit_name).read_text(encoding="utf-8") != "0\n":
                raise Gate0FinalizationError(
                    f"{case.case_id} raw exit code is nonzero: {exit_name}"
                )
        idle_kwargs = {
            "gpu_index": remote_identity["gpu_index"],
            "gpu_uuid": remote_identity["gpu_uuid"],
            "gpu_name": remote_identity["gpu_name"],
            "driver_version": remote_identity["driver_version"],
        }
        _validate_idle_gpu_state(
            case_dir / "preflight.txt",
            label=f"{case.case_id} preflight",
            **idle_kwargs,
        )
        postflights = tuple(sorted(case_dir.glob("postflight-*.txt")))
        if not postflights:
            raise Gate0FinalizationError(
                f"{case.case_id} lacks postflight GPU evidence"
            )
        # The launcher may retain failed drain attempts; its final numbered
        # postflight must prove that the same device returned fully idle.
        for path in postflights:
            fields, _processes = _parse_gpu_state(
                path, f"{case.case_id} {path.name}"
            )
            if fields.get("gpu_uuid") != remote_identity["gpu_uuid"]:
                raise Gate0FinalizationError(
                    f"{case.case_id} postflight changed GPU UUID"
                )
        _validate_idle_gpu_state(
            postflights[-1],
            label=f"{case.case_id} final postflight",
            **idle_kwargs,
        )


def _expected_backend_cases(
    manifest: ParityManifest, simulator: Simulator
) -> tuple[ScenarioCase, ...]:
    cases = tuple(
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is simulator
    )
    ids = tuple(case.case_id for case in cases)
    if len(cases) != EXPECTED_BACKEND_CASE_COUNT or len(set(ids)) != len(ids):
        raise Gate0FinalizationError(
            f"manifest must define exactly {EXPECTED_BACKEND_CASE_COUNT} unique "
            f"{simulator.value} cases; found {len(cases)}"
        )
    return cases


def _load_exact_backend_evidence(
    evidence_root: Path,
    manifest: ParityManifest,
    simulator: Simulator,
) -> tuple[CollectedRun, ...]:
    expected_cases = _expected_backend_cases(manifest, simulator)
    expected_by_id = {case.case_id: case for case in expected_cases}
    cases_dir = evidence_root / "cases"
    if cases_dir.is_symlink() or not cases_dir.is_dir():
        raise Gate0FinalizationError(
            f"{simulator.value} evidence must contain a regular cases directory"
        )
    entries = tuple(cases_dir.iterdir())
    bad_entries = [entry.name for entry in entries if entry.is_symlink() or not entry.is_dir()]
    observed_ids = {entry.name for entry in entries if entry.is_dir() and not entry.is_symlink()}
    expected_ids = set(expected_by_id)
    if bad_entries or observed_ids != expected_ids or len(entries) != len(expected_ids):
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise Gate0FinalizationError(
            f"{simulator.value} evidence is not the exact 16-case directory set: "
            f"missing={missing}, extra={extra}, invalid={sorted(bad_entries)}"
        )

    expected_run_paths = {
        (cases_dir / case_id / f"{case_id}{RUN_FILE_SUFFIX}").resolve()
        for case_id in expected_ids
    }
    observed_run_paths = {
        path.resolve() for path in evidence_root.rglob(f"*{RUN_FILE_SUFFIX}")
    }
    if observed_run_paths != expected_run_paths:
        missing = sorted(path.relative_to(evidence_root).as_posix() for path in expected_run_paths - observed_run_paths)
        extra = sorted(path.relative_to(evidence_root).as_posix() for path in observed_run_paths - expected_run_paths)
        raise Gate0FinalizationError(
            f"{simulator.value} evidence must contain exactly its 16 canonical run files: "
            f"missing={missing}, extra={extra}"
        )

    loaded: list[CollectedRun] = []
    for case in expected_cases:
        run_path = cases_dir / case.case_id / f"{case.case_id}{RUN_FILE_SUFFIX}"
        runs = load_collected_runs(run_path)
        if len(runs) != 1 or runs[0].case != case:
            raise Gate0FinalizationError(
                f"collected run does not match canonical case {case.case_id}"
            )
        if runs[0].bundle_root_sha256 is not None:
            raise Gate0FinalizationError(
                f"raw case {case.case_id} references a prior bundle"
            )
        loaded.append(runs[0])
    return tuple(loaded)


def _validate_provenance(
    runs: Iterable[CollectedRun],
    *,
    manifest: ParityManifest,
    session_id: str,
    source_revision: str,
) -> None:
    digest = manifest_sha256(manifest)
    expected: dict[str, object] = {
        "manifest_sha256": digest,
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "mapping_schema_version": 1,
    }
    for run in runs:
        provenance = run.result.provenance
        if not isinstance(provenance, Mapping):
            raise Gate0FinalizationError(
                f"{run.case.case_id} provenance must be an object"
            )
        mismatches = [
            key for key, value in expected.items() if provenance.get(key) != value
        ]
        if mismatches:
            raise Gate0FinalizationError(
                f"{run.case.case_id} has invalid provenance field(s): "
                + ", ".join(sorted(mismatches))
            )
        if run.case.simulator is Simulator.OVPHYSX:
            _validate_ovphysx_runtime_provenance(run, manifest)


def _validate_ovphysx_runtime_provenance(
    run: CollectedRun, manifest: ParityManifest
) -> None:
    """Hard-bind a remote run to its effective physics/control settings."""

    provenance = run.result.provenance
    scenario = manifest.scenario(run.case.scenario_id)
    hand = manifest.hand(run.case.hand)
    configuration = provenance.get("simulation_configuration")
    if not isinstance(configuration, Mapping) or configuration.get("verified") is not True:
        raise Gate0FinalizationError(
            f"{run.case.case_id} lacks verified simulation_configuration provenance"
        )
    for field in ("requested_dt_s", "cfg_dt_s", "backend_dt_s"):
        value = configuration.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), run.case.dt_s, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise Gate0FinalizationError(
                f"{run.case.case_id} has invalid {field} readback"
            )
    for field, tolerance in (
        ("requested_gravity_m_s2", 1e-12),
        ("cfg_gravity_m_s2", 1e-6),
        ("physics_scene_gravity_m_s2", 1e-5),
    ):
        values = configuration.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
            or any(
                not math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=tolerance
                )
                for actual, expected in zip(values, scenario.gravity_m_s2)
            )
        ):
            raise Gate0FinalizationError(
                f"{run.case.case_id} has invalid {field} readback"
            )
    if not isinstance(configuration.get("physics_prim_path"), str) or not str(
        configuration.get("physics_prim_path")
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} lacks PhysicsScene path provenance"
        )

    actuation_contract_version = provenance.get("actuation_contract_version")
    if (
        isinstance(actuation_contract_version, bool)
        or not isinstance(actuation_contract_version, int)
        or actuation_contract_version != 2
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} actuation_contract_version must be 2"
        )
    if provenance.get("actuator_model") != "IdealPDActuator":
        raise Gate0FinalizationError(
            f"{run.case.case_id} actuator_model must be IdealPDActuator"
        )
    if provenance.get("control_path") != "explicit_pd_effort":
        raise Gate0FinalizationError(
            f"{run.case.case_id} control_path must be explicit_pd_effort"
        )

    for field, positive in (
        ("controller_dof_stiffness", True),
        ("controller_dof_damping", False),
        ("controller_dof_effort_limit", True),
        ("controller_dof_effort_limit_sim", True),
    ):
        values = provenance.get(field)
        if not isinstance(values, list) or len(values) != len(hand.joint_names):
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} must contain 22 readbacks"
            )
        for value in values:
            number = _finite_number(value, f"{run.case.case_id} {field}")
            if (positive and number <= 0.0) or (not positive and number < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise Gate0FinalizationError(
                    f"{run.case.case_id} {field} must be {qualifier}"
                )
    if provenance.get("controller_parameter_source") != "ideal_pd_actuator_tensor":
        raise Gate0FinalizationError(
            f"{run.case.case_id} controller parameter source is invalid"
        )

    for field in ("backend_dof_stiffness", "backend_dof_damping"):
        values = provenance.get(field)
        if not isinstance(values, list) or len(values) != len(hand.joint_names):
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} must contain 22 readbacks"
            )
        if any(
            abs(_finite_number(value, f"{run.case.case_id} {field}")) > 1e-8
            for value in values
        ):
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} must contain zeroed PhysX drive readbacks"
            )
    if provenance.get("backend_dof_drive_readback_source") != "root_view_cpu_numpy_binding":
        raise Gate0FinalizationError(
            f"{run.case.case_id} drive readback source is invalid"
        )

    if provenance.get("position_target_readback_verified") is not True:
        raise Gate0FinalizationError(
            f"{run.case.case_id} position-target binding was not verified"
        )
    if (
        provenance.get("position_target_readback_source")
        != "articulation_data_joint_pos_target_torch"
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} position-target readback source is invalid"
        )
    if provenance.get("zero_velocity_target_verified") is not True:
        raise Gate0FinalizationError(
            f"{run.case.case_id} zero velocity target was not verified"
        )
    if provenance.get("zero_feedforward_effort_target_verified") is not True:
        raise Gate0FinalizationError(
            f"{run.case.case_id} zero feedforward effort target was not verified"
        )
    expected_steps = round(scenario.duration_s / run.case.dt_s)
    count = provenance.get("position_target_readback_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < expected_steps + 2:
        raise Gate0FinalizationError(
            f"{run.case.case_id} position-target readback coverage is incomplete"
        )
    error = provenance.get("position_target_readback_max_abs_error_rad")
    if (
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not math.isfinite(float(error))
        or float(error) > 1e-6
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} position-target readback error is invalid"
        )
    values = provenance.get("position_target_readback_values_rad")
    expected_target = scenario.initial_position_rad + (
        scenario.target_position_rad if scenario.kind.value == "small_step" else 0.0
    )
    if (
        not isinstance(values, list)
        or len(values) != len(hand.joint_names)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected_target, rel_tol=0.0, abs_tol=1e-6
            )
            for value in values
        )
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} final position-target readback values are invalid"
        )
    expected_nonzero = scenario.kind.value == "small_step"
    if provenance.get("position_target_nonzero_readback_observed") is not expected_nonzero:
        raise Gate0FinalizationError(
            f"{run.case.case_id} nonzero target readback flag is invalid"
        )

    for field in ("computed_effort_peak_abs_nm", "applied_effort_peak_abs_nm"):
        values = provenance.get(field)
        if not isinstance(values, list) or len(values) != len(hand.joint_names):
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} must contain 22 observations"
            )
        if any(
            _finite_number(value, f"{run.case.case_id} {field}") < 0.0
            for value in values
        ):
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} must be non-negative"
            )
    effort_observation_count = provenance.get("effort_observation_count")
    if (
        isinstance(effort_observation_count, bool)
        or not isinstance(effort_observation_count, int)
        or effort_observation_count < expected_steps + 1
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} effort observation coverage is incomplete"
        )
    for field, tolerance in (
        ("effort_formula_max_abs_error_nm", 1e-5),
        ("effort_clip_max_abs_error_nm", 1e-6),
    ):
        value = _finite_number(
            provenance.get(field), f"{run.case.case_id} {field}"
        )
        if value < 0.0 or value > tolerance:
            raise Gate0FinalizationError(
                f"{run.case.case_id} {field} exceeds its validation tolerance"
            )
    effort_clip_count = provenance.get("effort_clip_count")
    if (
        isinstance(effort_clip_count, bool)
        or not isinstance(effort_clip_count, int)
        or effort_clip_count < 0
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} effort_clip_count must be a non-negative integer"
        )
    if (
        provenance.get("effort_command_source")
        != "articulation_data_computed_and_applied_torque_torch"
    ):
        raise Gate0FinalizationError(
            f"{run.case.case_id} effort command source is invalid"
        )


def _payload_paths(root: Path) -> tuple[str, ...]:
    manifest_path = (root / DEFAULT_MANIFEST_PATH).resolve()
    paths: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Gate0FinalizationError(
                f"final bundle contains a symbolic link: {path.relative_to(root)}"
            )
        if path.is_file():
            if path.resolve() == manifest_path:
                continue
            paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def finalize_gate0(
    *,
    manifest_path: str | Path,
    mujoco_evidence_root: str | Path,
    ovphysx_evidence_root: str | Path,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
) -> tuple[Path, dict[str, object]]:
    """Create an immutable, content-hashed report bundle from exact evidence."""

    if _SESSION_RE.fullmatch(session_id) is None:
        raise Gate0FinalizationError("session_id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise Gate0FinalizationError(
            "source_revision must be a lowercase 40-character project code commit"
        )
    project_root = Path(__file__).resolve(strict=True).parents[1]
    local_source_tree = _validate_source_identity(project_root, source_revision)
    schema_probe_script_sha256 = _sha256_file(
        project_root / "scripts" / "probe_ovphysx_schema.py"
    )
    finalizer_source_hashes = {
        "finalizer_script_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "bundle_module_sha256": _sha256_file(
            project_root / "src" / "wave_asset_qa" / "parity" / "bundle.py"
        ),
        "compare_module_sha256": _sha256_file(
            project_root / "src" / "wave_asset_qa" / "parity" / "compare.py"
        ),
        "report_module_sha256": _sha256_file(
            project_root / "src" / "wave_asset_qa" / "parity" / "report.py"
        ),
        "schema_probe_script_sha256": schema_probe_script_sha256,
    }
    manifest_source = _resolved_file(manifest_path, "manifest")
    mujoco_source = _resolved_directory(mujoco_evidence_root, "MuJoCo evidence root")
    ovphysx_source = _resolved_directory(
        ovphysx_evidence_root, "OVPhysX evidence root"
    )
    if (
        mujoco_source == ovphysx_source
        or _is_within(mujoco_source, ovphysx_source)
        or _is_within(ovphysx_source, mujoco_source)
    ):
        raise Gate0FinalizationError("evidence roots must be separate, non-nested trees")
    parent, destination = _new_destination(output_dir)
    for source, label in (
        (mujoco_source, "MuJoCo evidence"),
        (ovphysx_source, "OVPhysX evidence"),
    ):
        if _is_within(destination, source) or _is_within(source, destination):
            raise Gate0FinalizationError(
                f"output and {label} paths must not contain one another"
            )

    manifest = load_manifest(manifest_source)
    expected_all = expand_scenario_cases(manifest)
    if (
        len(expected_all) != EXPECTED_TOTAL_CASE_COUNT
        or len({case.case_id for case in expected_all}) != EXPECTED_TOTAL_CASE_COUNT
    ):
        raise Gate0FinalizationError(
            "canonical manifest must expand to exactly 32 unique Gate 0 cases"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial.", dir=parent)
    ).resolve(strict=True)
    try:
        evidence_destination = staging / "evidence"
        evidence_destination.mkdir()
        _copy_tree_strict(
            mujoco_source, evidence_destination / "mujoco", "MuJoCo evidence"
        )
        _copy_tree_strict(
            ovphysx_source, evidence_destination / "ovphysx", "OVPhysX evidence"
        )

        manifest_dir = staging / "manifest"
        manifest_dir.mkdir()
        manifest_snapshot = manifest_dir / "gate0.json"
        with manifest_source.open("rb") as reader, manifest_snapshot.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        snapshot_manifest = load_manifest(manifest_snapshot)
        if snapshot_manifest != manifest:
            raise Gate0FinalizationError("copied manifest snapshot changed during finalization")

        remote_identity = _validate_remote_launcher_evidence(
            evidence_destination / "ovphysx",
            manifest,
            expected_source_tree=local_source_tree,
            manifest_file_sha256=_sha256_file(manifest_snapshot),
            session_id=session_id,
            source_revision=source_revision,
        )
        _validate_formal_schema_evidence(
            evidence_destination / "ovphysx",
            manifest,
            manifest_file_sha256=_sha256_file(manifest_snapshot),
            session_id=session_id,
            source_revision=source_revision,
            schema_probe_script_sha256=schema_probe_script_sha256,
        )
        _validate_remote_case_evidence(
            evidence_destination / "ovphysx", manifest, remote_identity
        )
        _validate_local_launcher_evidence(
            evidence_destination / "mujoco",
            manifest,
            expected_source_tree=local_source_tree,
            manifest_file_sha256=_sha256_file(manifest_snapshot),
            session_id=session_id,
            source_revision=source_revision,
        )
        _validate_local_case_evidence(
            evidence_destination / "mujoco",
            manifest,
            session_id=session_id,
            source_revision=source_revision,
        )
        mujoco_runs = _load_exact_backend_evidence(
            evidence_destination / "mujoco", manifest, Simulator.MUJOCO
        )
        ovphysx_runs = _load_exact_backend_evidence(
            evidence_destination / "ovphysx", manifest, Simulator.OVPHYSX
        )
        runs = (*mujoco_runs, *ovphysx_runs)
        if len(runs) != EXPECTED_TOTAL_CASE_COUNT or len(
            {run.case.case_id for run in runs}
        ) != EXPECTED_TOTAL_CASE_COUNT:
            raise Gate0FinalizationError("final evidence is not exactly 32 unique runs")
        _validate_provenance(
            runs,
            manifest=manifest,
            session_id=session_id,
            source_revision=source_revision,
        )
        _verify_evidence_hash_file(
            evidence_destination / "ovphysx",
            evidence_destination / "ovphysx" / "launcher" / "evidence.sha256",
            "OVPhysX launcher/evidence.sha256",
        )

        comparison = compare_gate0_runs(manifest, runs)
        results_dir = staging / "results"
        results_dir.mkdir()
        write_json_atomic(results_dir / "comparison.json", comparison.to_dict())
        write_reports(comparison, results_dir)
        write_json_atomic(
            results_dir / "finalization.json",
            {
                "schema_version": 1,
                "manifest_sha256": manifest_sha256(manifest),
                "session_id": session_id,
                "source_revision": source_revision,
                "source_tree": local_source_tree,
                "expected_case_count": EXPECTED_TOTAL_CASE_COUNT,
                "received_case_count": len(runs),
                "case_ids": [run.case.case_id for run in runs],
                "execution_status": comparison.execution_status.value,
                "comparison_status": comparison.comparison_status.value,
                "raw_evidence_roots": ["evidence/mujoco", "evidence/ovphysx"],
                "resolved_usd_schema_probes": {
                    "left": "available",
                    "right": "available",
                    "physics_step_claimed": False,
                },
                "remote_execution_identity": remote_identity,
                "local_finalizer_identity": finalizer_source_hashes,
            },
        )

        observed_source_tree = _validate_source_identity(project_root, source_revision)
        if observed_source_tree != local_source_tree or any(
            _sha256_file(
                {
                    "finalizer_script_sha256": Path(__file__).resolve(strict=True),
                    "bundle_module_sha256": project_root
                    / "src"
                    / "wave_asset_qa"
                    / "parity"
                    / "bundle.py",
                    "compare_module_sha256": project_root
                    / "src"
                    / "wave_asset_qa"
                    / "parity"
                    / "compare.py",
                    "report_module_sha256": project_root
                    / "src"
                    / "wave_asset_qa"
                    / "parity"
                    / "report.py",
                    "schema_probe_script_sha256": project_root
                    / "scripts"
                    / "probe_ovphysx_schema.py",
                }[name]
            )
            != digest
            for name, digest in finalizer_source_hashes.items()
        ):
            raise Gate0FinalizationError("local finalizer source changed during finalization")
        write_bundle_manifest(staging, _payload_paths(staging))
        verification: dict[str, object] = dict(verify_bundle(staging))
        execution_status = comparison.execution_status.value
        comparison_status = comparison.comparison_status.value
        execution_complete = execution_status == "completed"
        comparison_conclusive = comparison_status != "inconclusive"
        successful = execution_complete and comparison_conclusive
        verification.update(
            {
                "gate0_execution_status": execution_status,
                "gate0_comparison_status": comparison_status,
                "execution_complete": execution_complete,
                "comparison_conclusive": comparison_conclusive,
                "successful": successful,
                "successful_semantics": (
                    "execution complete and comparison conclusive; use pass_ready "
                    "for the Gate 0 acceptance decision"
                ),
                "pass_ready": successful and comparison_status == "within_tolerance",
            }
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"output path appeared during finalization; refusing overwrite: {destination}"
            )
        staging.replace(destination)
        return destination.resolve(strict=True), verification
    except BaseException:
        # This directory was created by this invocation under the already
        # validated output parent.  Inputs are never touched.
        if staging.exists() and _is_within(staging, parent):
            shutil.rmtree(staging)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mujoco-evidence-root", required=True, type=Path)
    parser.add_argument("--ovphysx-evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output, verification = finalize_gate0(
            manifest_path=args.manifest,
            mujoco_evidence_root=args.mujoco_evidence_root,
            ovphysx_evidence_root=args.ovphysx_evidence_root,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
        )
    except (Gate0FinalizationError, FileExistsError, OSError, ValueError) as exc:
        print(f"Gate 0 finalization failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output_dir": str(output), **verification},
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if verification.get("successful") is True else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
