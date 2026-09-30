#!/usr/bin/env python3
"""Finalize the formal 32-case Contact Gate C0 evidence set locally.

This program never imports either simulator.  It verifies the two launcher
trees, recomputes every worker preimage hash, runs the frozen comparison, and
publishes only a two-file de-identified report.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np

import wave_asset_qa
from wave_asset_qa.contact.bundle import (
    ContactEvidenceError,
    canonical_json_sha256,
    inventory_root_sha256,
    read_json_strict,
    regular_tree_records,
    verify_adapter_private_evidence,
    write_json_exclusive,
)
from wave_asset_qa.contact.compare import EvidenceStatus, evaluate_contact_c0
from wave_asset_qa.contact.records import ContactRun
from wave_asset_qa.contact.report import public_summary, render_contact_markdown
from wave_asset_qa.contact.runner import load_and_validate_contact_run
from wave_asset_qa.contact.scenarios import expand_contact_cases, load_contact_manifest
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.diagnostics import is_link_like
from wave_asset_qa.parity.process_identity import (
    os_process_identity_sha256,
    validate_os_process_identity,
)


EXPECTED_CASES_PER_BACKEND = 16
EXPECTED_TOTAL_CASES = 32
MAX_EVIDENCE_BYTES_PER_BACKEND = 2 * 1024 * 1024 * 1024
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_UTC_SECOND = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_REMOTE_ROOT = PurePosixPath("/data/home/exampleuser/sharpa-wave-asset-qa-gate0")
_OVPHYSX_SDK_ARCHIVE = (
    _REMOTE_ROOT
    / "dependencies"
    / "ovphysx-0.4.13"
    / "ovphysx-linux-x86_64-0.4.13.tar.gz"
)
_OVPHYSX_SDK_ARCHIVE_SHA256 = (
    "191dcaff34980f6fdf94bb783c8faacbab058aa4c2671e31ac306fcd89cdb1e7"
)
_PHYSX_NATIVE_MASS_READBACK_SOURCE = (
    "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
    "post_first_reset"
)
_NATIVE_HELPER_SOURCE_RELATIVE = PurePosixPath(
    "src/wave_asset_qa/contact/native/physx_ccd_readback.cpp"
)
_NATIVE_HELPER_FILENAME = "libwaveqa_physx_ccd_readback.so"
_REMOTE_NATIVE_FILES = frozenset(
    {
        "build-provenance.json",
        "headers-manifest.json",
        _NATIVE_HELPER_FILENAME,
    }
)
_PRIVATE_PROCESS_KEYS = {
    "schema_version",
    "visibility",
    "case_id",
    "backend",
    "session_id",
    "source_revision",
    "source_tree",
    "manifest_file_sha256",
    "manifest_semantic_sha256",
    "asset_tree_sha256",
    "python_version",
    "python_executable_realpath",
    "worker_module_realpath",
    "worker_module_sha256",
    "os_process_identity",
    "fresh_process_identity_sha256",
}

_LOCAL_LAUNCHER_FILES = frozenset({"provenance.json", "status.json"})
_LOCAL_CASE_FILES = frozenset(
    {
        "private-process.json",
        "private-adapter-evidence.json",
        "stdout.txt",
        "stderr.txt",
        "exit-code.txt",
    }
)
_REMOTE_LAUNCHER_FILES = frozenset(
    {
        "stdout.txt",
        "stderr.txt",
        "gpu-selection.txt",
        "snapshot-verifications.txt",
        "matrix.json",
        "provenance.json",
        "status.json",
    }
)
_REMOTE_CASE_FIXED_FILES = frozenset(
    {
        "private-process.json",
        "private-adapter-evidence.json",
        "stdout.txt",
        "stderr.txt",
        "exit-code.txt",
        "command.json",
        "owned-process-group.txt",
        "process.json",
        "worker-group-binding.json",
        "gpu-monitor.txt",
        "gpu-preflight-01.txt",
        "gpu-preflight-02.txt",
        "gpu-preflight-03.txt",
    }
)
_MAX_RUNTIME_LOG_BYTES = 128 * 1024 * 1024
_RUNTIME_FATAL_LOG_PATTERNS = (
    (
        "python_traceback",
        re.compile(r"(?i)(?<![a-z0-9])trace[\s_-]*back(?![a-z0-9])"),
    ),
    (
        "cuda_error",
        re.compile(
            r"(?i)(?<![a-z0-9])cuda(?:[\s_:-]*runtime)?[\s_:-]*"
            r"(?:error|failure|failed)(?![a-z0-9])|"
            r"(?<![a-z0-9])cudaerror[a-z0-9_]*"
        ),
    ),
    (
        "out_of_memory",
        re.compile(
            r"(?i)(?<![a-z0-9])out[\s_-]*of[\s_-]*memory(?![a-z0-9])|"
            r"(?<![a-z0-9])oom(?![a-z0-9])|"
            r"(?<![a-z0-9])memoryerror(?![a-z0-9])|"
            r"std::bad_alloc|cannot[\s_-]+allocate[\s_-]+memory"
        ),
    ),
)

_OV_MASS_SOURCE_KEYS = frozenset(
    {
        "physics:mass",
        "physics:density",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    }
)
_OV_RUNTIME_MASS_CHECK_KEYS = frozenset(
    {
        "body_name_exact",
        "mass_abs_error_kg",
        "center_of_mass_max_abs_error_m",
        "inertia_link_tensor_max_abs_error_kg_m2",
        "inertia_symmetry_max_abs_error_kg_m2",
        "runtime_inertia_min_eigenvalue_kg_m2",
        "principal_axes_representation_angle_error_rad",
        "expected_link_inertia_tensor_kg_m2",
        "runtime_link_inertia_tensor_kg_m2",
        "expected_inertia_eigenvalues_kg_m2",
        "runtime_inertia_eigenvalues_kg_m2",
        "tolerances",
    }
)
_OV_RUNTIME_MASS_TOLERANCES = {
    "mass_relative": 5.0e-5,
    "mass_absolute_kg": 1.0e-8,
    "center_of_mass_absolute_m": 1.0e-7,
    "inertia_relative": 5.0e-5,
    "inertia_absolute_kg_m2": 1.0e-12,
}
_PHYSX_MASS_QUATERNION_NORM_ABS_TOLERANCE = 1.0e-5

_LOCAL_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "backend",
        "session_id",
        "source_revision",
        "source_tree",
        "manifest_id",
        "manifest_file_sha256",
        "manifest_semantic_sha256",
        "manifest_canonical_json_sha256",
        "asset_repository",
        "asset_commit",
        "asset_git_tree",
        "asset_subtree_relative_path",
        "asset_tree_sha256",
        "asset_tree_verification",
        "local_runtime_identity",
        "launcher_script_sha256",
        "worker_script_sha256",
        "worker_module_sha256",
        "fresh_process_per_case",
        "expected_case_count",
        "case_ids",
        "worker_command_template",
    }
)
_LOCAL_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "backend",
        "status",
        "message",
        "session_id",
        "source_revision",
        "source_tree",
        "expected_case_count",
        "completed_case_count",
        "unique_process_count",
        "failed_case_id",
    }
)
_REMOTE_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "backend",
        "run_id",
        "session_id",
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "snapshot_sha256",
        "manifest_id",
        "manifest_sha256",
        "asset_commit",
        "asset_git_tree",
        "asset_tree_sha256",
        "isaaclab_commit",
        "selected_gpu",
        "environment",
        "native_physx_ccd_helper",
        "claim_boundary",
        "started_at_utc",
    }
)
_REMOTE_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "campaign",
        "backend",
        "status",
        "scientific_verdict",
        "exit_code",
        "run_id",
        "session_id",
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "snapshot_sha256",
        "selected_gpu_index",
        "selected_gpu_uuid",
        "completed_case_count",
        "unique_process_count",
        "last_case_id",
        "elapsed_s",
        "gpu_elapsed_s",
        "ended_at_utc",
    }
)


class ContactC0FinalizationError(RuntimeError):
    """Raised when formal evidence or publication is not admissible."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ContactC0FinalizationError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object, expected_keys: Sequence[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    data = _mapping(value, label)
    expected = set(expected_keys)
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise ContactC0FinalizationError(
            f"{label} fields are not exact: missing={missing}, extra={extra}"
        )
    return data


def _nonnegative_int(value: object, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContactC0FinalizationError(f"{label} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ContactC0FinalizationError(f"{label} exceeds its frozen maximum")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContactC0FinalizationError(f"{label} must be a non-empty string")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContactC0FinalizationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ContactC0FinalizationError(f"{label} must be a finite number")
    return number


def _finite_vector(value: object, *, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ContactC0FinalizationError(
            f"{label} must be a {length}-element number array"
        )
    return np.asarray(
        [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)],
        dtype=np.float64,
    )


def _finite_matrix3(value: object, label: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in value)
    ):
        raise ContactC0FinalizationError(f"{label} must be a 3x3 number array")
    return np.asarray(
        [
            [
                _finite_float(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
            for row_index, row in enumerate(value)
        ],
        dtype=np.float64,
    )


def _normalized_quaternion_xyzw(value: object, label: str) -> np.ndarray:
    quaternion = _finite_vector(value, length=4, label=label)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ContactC0FinalizationError(f"{label} has zero norm")
    if abs(norm - 1.0) > _PHYSX_MASS_QUATERNION_NORM_ABS_TOLERANCE:
        raise ContactC0FinalizationError(f"{label} is not unit length")
    return quaternion / norm


def _quaternion_rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _assert_recorded_scalar(
    value: object, expected: float, label: str, *, nonnegative: bool = True
) -> None:
    recorded = _finite_float(value, label)
    if nonnegative and recorded < 0.0:
        raise ContactC0FinalizationError(f"{label} must be non-negative")
    if not math.isclose(recorded, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ContactC0FinalizationError(f"{label} differs from independent recompute")


def _assert_recorded_array(value: object, expected: np.ndarray, label: str) -> None:
    recorded = (
        _finite_vector(value, length=3, label=label)
        if expected.shape == (3,)
        else _finite_matrix3(value, label)
    )
    if recorded.shape != expected.shape or not np.allclose(
        recorded, expected, rtol=1.0e-12, atol=1.0e-15
    ):
        raise ContactC0FinalizationError(f"{label} differs from independent recompute")


def _mass_source_value(
    source: Mapping[str, Any], name: str, *, length: int | None = None
) -> float | np.ndarray:
    record = _exact_mapping(
        source.get(name),
        {"valid", "authored", "value", "type"},
        f"OV authored mass property {name}",
    )
    if record.get("valid") is not True or record.get("authored") is not True:
        raise ContactC0FinalizationError(
            f"OV authored mass property {name} is not valid and authored"
        )
    _nonempty_string(record.get("type"), f"OV authored mass property {name} type")
    if length is None:
        return _finite_float(record.get("value"), f"OV authored mass property {name}")
    return _finite_vector(
        record.get("value"), length=length, label=f"OV authored mass property {name}"
    )


def _validate_ov_runtime_mass_proof(
    *,
    source_value: object,
    runtime_effective_value: object,
    runtime_checks_value: object,
    expected_body_name: str,
    expected_body_prim_path: str,
    expected_helper_sha256: str,
) -> None:
    """Independently recompute the post-cook mass/COM/inertia proof."""

    source = _exact_mapping(
        source_value, _OV_MASS_SOURCE_KEYS, "OV authored mass-property source"
    )
    # Density is part of the exact USD readback even though explicit mass and
    # inertia, not density, define this proof.
    density = _exact_mapping(
        source.get("physics:density"),
        {"valid", "authored", "value", "type"},
        "OV authored mass property physics:density",
    )
    if not isinstance(density.get("valid"), bool) or not isinstance(
        density.get("authored"), bool
    ):
        raise ContactC0FinalizationError(
            "OV authored mass property physics:density flags are not booleans"
        )
    density_type = density.get("type")
    if density_type is not None and not isinstance(density_type, str):
        raise ContactC0FinalizationError(
            "OV authored mass property physics:density type is invalid"
        )

    expected_mass = float(_mass_source_value(source, "physics:mass"))
    expected_com = np.asarray(
        _mass_source_value(source, "physics:centerOfMass", length=3),
        dtype=np.float64,
    )
    expected_diagonal = np.asarray(
        _mass_source_value(source, "physics:diagonalInertia", length=3),
        dtype=np.float64,
    )
    expected_quaternion = _normalized_quaternion_xyzw(
        _mass_source_value(source, "physics:principalAxes", length=4).tolist(),
        "OV authored principalAxes",
    )
    if expected_mass <= 0.0 or np.any(expected_diagonal <= 0.0):
        raise ContactC0FinalizationError(
            "OV authored mass and diagonal inertia must be positive"
        )

    runtime_effective = _exact_mapping(
        runtime_effective_value,
        {
            "body_name",
            "body_prim_path",
            "source",
            "physx_version",
            "helper_sha256",
            "mass_kg",
            "center_of_mass_pose_b",
            "mass_space_inertia_diagonal_kg_m2",
        },
        "OV runtime effective mass properties",
    )
    if runtime_effective.get("body_name") != expected_body_name:
        raise ContactC0FinalizationError(
            "OV runtime mass body differs from sensor body"
        )
    if (
        runtime_effective.get("body_prim_path") != expected_body_prim_path
        or runtime_effective.get("source") != _PHYSX_NATIVE_MASS_READBACK_SOURCE
        or runtime_effective.get("physx_version") != "5.9.0"
        or runtime_effective.get("helper_sha256") != expected_helper_sha256
    ):
        raise ContactC0FinalizationError(
            "OV native runtime mass-property provenance differs"
        )
    _require_hex(
        runtime_effective.get("helper_sha256"),
        "OV native runtime mass-property helper SHA-256",
        length=64,
    )
    runtime_mass = _finite_float(runtime_effective.get("mass_kg"), "OV runtime mass")
    if runtime_mass <= 0.0:
        raise ContactC0FinalizationError("OV runtime mass must be positive")
    center_pose = _exact_mapping(
        runtime_effective.get("center_of_mass_pose_b"),
        {"position_m", "quaternion_xyzw"},
        "OV runtime center-of-mass pose",
    )
    runtime_com = _finite_vector(
        center_pose.get("position_m"), length=3, label="OV runtime center of mass"
    )
    runtime_quaternion = _normalized_quaternion_xyzw(
        center_pose.get("quaternion_xyzw"), "OV runtime COM pose"
    )
    runtime_diagonal = _finite_vector(
        runtime_effective.get("mass_space_inertia_diagonal_kg_m2"),
        length=3,
        label="OV native runtime mass-space inertia diagonal",
    )
    if np.any(runtime_diagonal <= 0.0):
        raise ContactC0FinalizationError(
            "OV native runtime mass-space inertia must be positive"
        )

    mass_error = abs(runtime_mass - expected_mass)
    com_error = float(np.max(np.abs(runtime_com - expected_com)))
    runtime_eigenvalues = np.sort(runtime_diagonal)
    expected_eigenvalues = np.sort(expected_diagonal)
    expected_rotation = _quaternion_rotation_matrix_xyzw(expected_quaternion)
    runtime_rotation = _quaternion_rotation_matrix_xyzw(runtime_quaternion)
    expected_link_inertia = (
        expected_rotation @ np.diag(expected_diagonal) @ expected_rotation.T
    )
    runtime_link_inertia = (
        runtime_rotation @ np.diag(runtime_diagonal) @ runtime_rotation.T
    )
    symmetry_error = float(
        np.max(np.abs(runtime_link_inertia - runtime_link_inertia.T))
    )
    inertia_error = float(
        np.max(np.abs(runtime_link_inertia - expected_link_inertia))
    )
    quaternion_dot = float(abs(np.dot(expected_quaternion, runtime_quaternion)))
    representation_angle = 2.0 * math.acos(min(1.0, max(0.0, quaternion_dot)))

    checks = _exact_mapping(
        runtime_checks_value, _OV_RUNTIME_MASS_CHECK_KEYS, "OV runtime mass checks"
    )
    if checks.get("body_name_exact") is not True:
        raise ContactC0FinalizationError("OV runtime mass body proof is false")
    tolerances = _exact_mapping(
        checks.get("tolerances"),
        frozenset(_OV_RUNTIME_MASS_TOLERANCES),
        "OV runtime mass tolerances",
    )
    for name, expected_value in _OV_RUNTIME_MASS_TOLERANCES.items():
        _assert_recorded_scalar(
            tolerances.get(name), expected_value, f"OV runtime tolerance {name}"
        )

    _assert_recorded_scalar(
        checks.get("mass_abs_error_kg"), mass_error, "OV runtime mass absolute error"
    )
    _assert_recorded_scalar(
        checks.get("center_of_mass_max_abs_error_m"),
        com_error,
        "OV runtime center-of-mass error",
    )
    _assert_recorded_scalar(
        checks.get("inertia_link_tensor_max_abs_error_kg_m2"),
        inertia_error,
        "OV runtime link-frame inertia error",
    )
    _assert_recorded_scalar(
        checks.get("inertia_symmetry_max_abs_error_kg_m2"),
        symmetry_error,
        "OV runtime inertia symmetry error",
    )
    _assert_recorded_scalar(
        checks.get("runtime_inertia_min_eigenvalue_kg_m2"),
        float(np.min(runtime_eigenvalues)),
        "OV runtime inertia minimum eigenvalue",
        nonnegative=False,
    )
    _assert_recorded_scalar(
        checks.get("principal_axes_representation_angle_error_rad"),
        representation_angle,
        "OV principal-axes representation angle",
    )
    _assert_recorded_array(
        checks.get("expected_link_inertia_tensor_kg_m2"),
        expected_link_inertia,
        "OV expected link-frame inertia tensor",
    )
    _assert_recorded_array(
        checks.get("runtime_link_inertia_tensor_kg_m2"),
        runtime_link_inertia,
        "OV runtime link-frame inertia tensor",
    )
    _assert_recorded_array(
        checks.get("expected_inertia_eigenvalues_kg_m2"),
        expected_eigenvalues,
        "OV expected inertia eigenvalues",
    )
    _assert_recorded_array(
        checks.get("runtime_inertia_eigenvalues_kg_m2"),
        runtime_eigenvalues,
        "OV runtime inertia eigenvalues",
    )

    mass_limit = max(
        _OV_RUNTIME_MASS_TOLERANCES["mass_absolute_kg"],
        _OV_RUNTIME_MASS_TOLERANCES["mass_relative"] * abs(expected_mass),
    )
    inertia_limit = max(
        _OV_RUNTIME_MASS_TOLERANCES["inertia_absolute_kg_m2"],
        _OV_RUNTIME_MASS_TOLERANCES["inertia_relative"]
        * float(np.max(np.abs(expected_link_inertia))),
    )
    if (
        mass_error > mass_limit
        or com_error > _OV_RUNTIME_MASS_TOLERANCES["center_of_mass_absolute_m"]
        or symmetry_error > _OV_RUNTIME_MASS_TOLERANCES["inertia_absolute_kg_m2"]
        or np.any(runtime_eigenvalues <= 0.0)
        or inertia_error > inertia_limit
    ):
        raise ContactC0FinalizationError(
            "OV runtime mass/COM/link-frame inertia tolerance proof failed"
        )


def _file_sha256(path: Path) -> str:
    try:
        source = path.resolve(strict=True)
    except OSError as error:
        raise ContactC0FinalizationError(f"source file does not exist: {path}") from error
    if is_link_like(source) or not source.is_file():
        raise ContactC0FinalizationError(f"source file is not regular: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex(value: object, label: str, *, length: int) -> str:
    pattern = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContactC0FinalizationError(f"{label} must be lowercase hex{length}")
    return value


def _reject_link_ancestors(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    for candidate in (raw, *raw.parents):
        if candidate.exists() and is_link_like(candidate):
            raise ContactC0FinalizationError(
                f"{label} has a symbolic-link or junction ancestor: {candidate}"
            )
    return raw


def _resolved_directory(path: str | Path, label: str) -> Path:
    raw = _reject_link_ancestors(Path(path), label)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ContactC0FinalizationError(f"{label} does not exist") from error
    if not resolved.is_dir():
        raise ContactC0FinalizationError(f"{label} must be a regular directory")
    return resolved


def _new_directory_path(path: str | Path, label: str) -> Path:
    raw = _reject_link_ancestors(Path(path), label)
    if raw.exists() or is_link_like(raw):
        raise FileExistsError(f"{label} already exists; refusing overwrite: {raw}")
    parent = _resolved_directory(raw.parent, f"{label} parent")
    return parent / raw.name


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_disjoint(paths: Sequence[tuple[Path, str]]) -> None:
    for index, (left, left_label) in enumerate(paths):
        for right, right_label in paths[index + 1 :]:
            if left == right or _is_within(left, right) or _is_within(right, left):
                raise ContactC0FinalizationError(
                    f"{left_label} and {right_label} must be disjoint trees"
                )


def _git_text(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContactC0FinalizationError("cannot inspect local Git source") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ContactC0FinalizationError(
            "local Git source inspection failed" + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _source_bytecode(root: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for name in ("src", "scripts"):
        source = root / name
        for path in source.rglob("*"):
            if path.name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo"}:
                result.append(path)
    return tuple(result)


def _validate_source(root: Path, revision: str) -> str:
    if _HEX40.fullmatch(revision) is None:
        raise ContactC0FinalizationError("source revision is malformed")
    if Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True) != root:
        raise ContactC0FinalizationError("project root is not the Git top level")
    if _git_text(root, "rev-parse", "HEAD") != revision:
        raise ContactC0FinalizationError("source revision does not match HEAD")
    tree = _git_text(root, "rev-parse", f"{revision}^{{tree}}")
    _require_hex(tree, "source tree", length=40)
    if _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ContactC0FinalizationError("formal finalization requires a clean worktree")
    bytecode = _source_bytecode(root)
    if bytecode:
        raise ContactC0FinalizationError(
            "formal source contains ignored Python bytecode: "
            + bytecode[0].relative_to(root).as_posix()
        )
    relative = Path(__file__).resolve(strict=True).relative_to(root).as_posix()
    _git_text(root, "ls-files", "--error-unmatch", "--", relative)
    imported = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    expected = (root / "src" / "wave_asset_qa").resolve(strict=True)
    if imported != expected:
        raise ContactC0FinalizationError("wave_asset_qa imported from another source tree")
    return tree


def _git_archive_bytes(root: Path, revision: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=root,
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContactC0FinalizationError("cannot create source archive") from error
    if completed.returncode != 0:
        raise ContactC0FinalizationError("git archive failed")
    return completed.stdout


def _virtual_snapshot_sha256(
    archive_bytes: bytes,
    *,
    source_revision: str,
    source_tree: str,
    archive_sha256: str,
) -> str:
    """Reproduce the immutable snapshot installer hash without extraction."""

    if sha256(archive_bytes).hexdigest() != archive_sha256:
        raise ContactC0FinalizationError("source archive digest mismatch")
    entries: dict[str, tuple[bytes, int, bytes | None]] = {}

    def parents(path: PurePosixPath) -> None:
        for length in range(1, len(path.parts)):
            name = PurePosixPath(*path.parts[:length]).as_posix()
            existing = entries.get(name)
            if existing is not None and existing[0] != b"D":
                raise ContactC0FinalizationError("archive file is used as a parent")
            entries.setdefault(name, (b"D", 0, None))

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as bundle:
            for member in bundle.getmembers():
                raw = member.name.rstrip("/")
                path = PurePosixPath(raw)
                if (
                    not raw
                    or "\\" in raw
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or ":" in path.parts[0]
                    or path.parts[0]
                    in {
                        ".source-revision",
                        ".source-tree",
                        ".archive-sha256",
                        ".snapshot-sha256",
                    }
                ):
                    raise ContactC0FinalizationError("source archive path is unsafe")
                name = path.as_posix()
                parents(path)
                if name in entries:
                    raise ContactC0FinalizationError("source archive member is duplicate")
                if member.isdir():
                    entries[name] = (b"D", 0, None)
                elif member.isfile():
                    handle = bundle.extractfile(member)
                    if handle is None:
                        raise ContactC0FinalizationError("archive member cannot be read")
                    content = handle.read()
                    if len(content) != member.size:
                        raise ContactC0FinalizationError("archive member size drifted")
                    entries[name] = (b"F", int(bool(member.mode & 0o111)), content)
                else:
                    raise ContactC0FinalizationError("archive contains a special member")
    except (tarfile.TarError, OSError) as error:
        raise ContactC0FinalizationError("cannot inspect source archive") from error
    for name, value in {
        ".source-revision": source_revision,
        ".source-tree": source_tree,
        ".archive-sha256": archive_sha256,
    }.items():
        entries[name] = (b"F", 0, (value + "\n").encode("ascii"))
    digest = sha256(b"waveqa-source-snapshot-v1\0")
    for name in sorted(entries):
        kind, executable, content = entries[name]
        encoded = name.encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(executable.to_bytes(1, "big"))
        if kind == b"F":
            assert content is not None
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def _validate_evidence_manifest(root: Path, expected_scope: str) -> dict[str, object]:
    path = root / "evidence-manifest.json"
    data = _mapping(read_json_strict(path), f"{expected_scope} evidence manifest")
    if set(data) != {"schema_version", "scope", "records", "root_sha256"}:
        raise ContactC0FinalizationError("evidence manifest fields are not exact")
    if data["schema_version"] != 1 or data["scope"] != expected_scope:
        raise ContactC0FinalizationError("evidence manifest identity differs")
    raw_records = data["records"]
    if not isinstance(raw_records, list):
        raise ContactC0FinalizationError("evidence manifest records must be an array")
    observed = list(regular_tree_records(root, exclude=("evidence-manifest.json",)))
    if raw_records != observed:
        raise ContactC0FinalizationError("evidence manifest does not match exact tree inventory")
    root_hash = inventory_root_sha256(raw_records)
    if data["root_sha256"] != root_hash:
        raise ContactC0FinalizationError("evidence inventory root hash mismatch")
    total = sum(int(record["size"]) for record in raw_records)
    if total > MAX_EVIDENCE_BYTES_PER_BACKEND:
        raise ContactC0FinalizationError("backend evidence exceeds the 2 GiB cap")
    return {"root_sha256": root_hash, "file_count": len(raw_records), "bytes": total}


def _validate_exact_directory_entries(
    root: Path,
    *,
    files: set[str] | frozenset[str],
    directories: set[str] | frozenset[str],
    label: str,
) -> None:
    entries = tuple(root.iterdir())
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in entries:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or is_link_like(path):
            raise ContactC0FinalizationError(f"{label} contains a link: {path.name}")
        if stat.S_ISREG(mode):
            observed_files.add(path.name)
        elif stat.S_ISDIR(mode):
            observed_directories.add(path.name)
        else:
            raise ContactC0FinalizationError(
                f"{label} contains a special entry: {path.name}"
            )
    if observed_files != set(files) or observed_directories != set(directories):
        raise ContactC0FinalizationError(
            f"{label} topology is not exact: files={sorted(observed_files)}, "
            f"directories={sorted(observed_directories)}"
        )


def _audit_runtime_log(path: Path, label: str) -> dict[str, object]:
    if is_link_like(path) or not path.is_file():
        raise ContactC0FinalizationError(f"{label} is not a regular log")
    size = path.stat().st_size
    if size > _MAX_RUNTIME_LOG_BYTES:
        raise ContactC0FinalizationError(f"{label} exceeds the 128 MiB log cap")
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline=None) as handle:
            for line_number, line in enumerate(handle, start=1):
                for pattern_id, pattern in _RUNTIME_FATAL_LOG_PATTERNS:
                    if pattern.search(line) is not None:
                        raise ContactC0FinalizationError(
                            f"{label} contains fatal runtime signal {pattern_id} "
                            f"at line {line_number}"
                        )
    except UnicodeError as error:
        raise ContactC0FinalizationError(f"{label} is not strict UTF-8") from error
    return {"path": label, "size": size, "sha256": _file_sha256(path)}


def _validate_backend_topology(
    root: Path,
    *,
    backend: Simulator,
    expected_case_ids: set[str],
    experimental_campaign: str | None = None,
) -> list[dict[str, object]]:
    _validate_exact_directory_entries(
        root,
        files={"evidence-manifest.json"},
        directories={"launcher", "cases"},
        label=f"{backend.value} evidence root",
    )
    launcher = root / "launcher"
    _validate_exact_directory_entries(
        launcher,
        files=(
            _LOCAL_LAUNCHER_FILES
            if backend is Simulator.MUJOCO
            else _REMOTE_LAUNCHER_FILES
        ),
        directories=(set() if backend is Simulator.MUJOCO else {"native"}),
        label=f"{backend.value} launcher",
    )
    if backend is Simulator.OVPHYSX:
        _validate_exact_directory_entries(
            launcher / "native",
            files=_REMOTE_NATIVE_FILES,
            directories={"include"},
            label="ovphysx native helper evidence",
        )
    cases_root = root / "cases"
    _validate_exact_directory_entries(
        cases_root,
        files=set(),
        directories=expected_case_ids,
        label=f"{backend.value} cases root",
    )
    logs: list[dict[str, object]] = []
    if backend is Simulator.OVPHYSX:
        logs.extend(
            _audit_runtime_log(launcher / name, f"{backend.value}/launcher/{name}")
            for name in ("stdout.txt", "stderr.txt")
        )
    for case_id in sorted(expected_case_ids):
        case_root = cases_root / case_id
        run_name = f"{case_id}.run.json"
        if backend is Simulator.MUJOCO:
            expected_files = {*_LOCAL_CASE_FILES, run_name}
        else:
            direct_files = {
                path.name
                for path in case_root.iterdir()
                if stat.S_ISREG(path.lstat().st_mode) and not is_link_like(path)
            }
            observed_postflights = sorted(
                name
                for name in direct_files
                if re.fullmatch(r"gpu-postflight-[0-9]{2}\.txt", name)
            )
            expected_postflights = [
                f"gpu-postflight-{index:02d}.txt"
                for index in range(1, len(observed_postflights) + 1)
            ]
            if (
                not 1 <= len(observed_postflights) <= 3
                or observed_postflights != expected_postflights
            ):
                raise ContactC0FinalizationError(
                    f"{case_id} GPU postflights are not consecutive 01..03"
                )
            expected_files = {
                *_REMOTE_CASE_FIXED_FILES,
                run_name,
                *expected_postflights,
            }
        if experimental_campaign is not None:
            if experimental_campaign != 'contact-c0-async-v1':
                raise ContactC0FinalizationError('unknown candidate campaign')
            expected_files.add('private-campaign.json')
        _validate_exact_directory_entries(
            case_root,
            files=expected_files,
            directories=set(),
            label=f"{backend.value} case {case_id}",
        )
        if (case_root / "exit-code.txt").read_text(encoding="utf-8") != "0\n":
            raise ContactC0FinalizationError(f"{case_id} worker exit code is nonzero")
        logs.extend(
            _audit_runtime_log(
                case_root / name, f"{backend.value}/cases/{case_id}/{name}"
            )
            for name in ("stdout.txt", "stderr.txt")
        )
    return logs


def _validate_local_launcher_identity(
    root: Path,
    *,
    manifest: object,
    expected_case_ids: list[str],
    source_revision: str,
    source_tree: str,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    project_root: Path,
    experimental_campaign: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    launcher = root / "launcher"
    provenance = _exact_mapping(
        read_json_strict(launcher / "provenance.json"),
        _LOCAL_PROVENANCE_KEYS,
        "MuJoCo launcher provenance",
    )
    status = _exact_mapping(
        read_json_strict(launcher / "status.json"),
        _LOCAL_STATUS_KEYS,
        "MuJoCo launcher status",
    )
    session_id = status.get("session_id")
    if not isinstance(session_id, str) or _SESSION.fullmatch(session_id) is None:
        raise ContactC0FinalizationError("MuJoCo session ID is invalid")
    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise ContactC0FinalizationError('unknown candidate campaign')
    worker_relative = ('scripts/run_contact_async_worker.py' if experimental_campaign
                       else 'scripts/run_contact_c0_mujoco_worker.py')
    worker_template = [
        "<python>",
        "-P",
        worker_relative,
        *(['--backend','mujoco'] if experimental_campaign else []),
        "--asset-root",
        "<asset-root>",
        "--manifest",
        "configs/parity/contact_c0.json",
        "--case-id",
        "<case-id>",
        "--output-dir",
        "cases/<case-id>",
        "--session-id",
        session_id,
        "--source-revision",
        source_revision,
        "--source-tree",
        source_tree,
        "--asset-tree-sha256",
        manifest.provenance.canonical_lf_asset_tree_sha256,
    ]
    fixed = {
        "schema_version": 1,
        "campaign": experimental_campaign or "contact_c0",
        "backend": "mujoco",
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "manifest_id": manifest.manifest_id,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "manifest_canonical_json_sha256": manifest_semantic_sha256,
        "asset_repository": manifest.provenance.repository,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_subtree_relative_path": manifest.provenance.asset_root,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "launcher_script_sha256": _file_sha256(
            project_root / "scripts" / "run_contact_c0_local.py"
        ),
        "worker_script_sha256": _file_sha256(
            project_root / worker_relative
        ),
        "worker_module_sha256": _file_sha256(
            project_root / "src" / "wave_asset_qa" / "contact" / "worker.py"
        ),
        "fresh_process_per_case": True,
        "expected_case_count": EXPECTED_CASES_PER_BACKEND,
        "case_ids": expected_case_ids,
        "worker_command_template": worker_template,
    }
    mismatches = sorted(key for key, value in fixed.items() if provenance.get(key) != value)
    if mismatches:
        raise ContactC0FinalizationError(
            "MuJoCo launcher provenance differs: " + ", ".join(mismatches)
        )
    runtime = _exact_mapping(
        provenance.get("local_runtime_identity"),
        {
            "schema_version",
            "python_version",
            "mujoco_version",
            "numpy_version",
            "python_implementation",
            "platform",
            "machine",
            "python_executable_sha256",
        },
        "MuJoCo local runtime identity",
    )
    expected_runtime = {
        "schema_version": 1,
        "python_version": "3.12.6",
        "mujoco_version": "3.12.0",
        "numpy_version": "2.5.2",
        "python_implementation": "CPython",
        "platform": "Windows-11-10.0.26200-SP0",
        "machine": "AMD64",
        "python_executable_sha256": (
            "3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238"
        ),
    }
    if dict(runtime) != expected_runtime:
        raise ContactC0FinalizationError("MuJoCo launcher runtime identity drifted")
    expected_status = {
        "schema_version": 1,
        "campaign": experimental_campaign or "contact_c0",
        "backend": "mujoco",
        "status": "completed",
        "message": None,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "expected_case_count": EXPECTED_CASES_PER_BACKEND,
        "completed_case_count": EXPECTED_CASES_PER_BACKEND,
        "unique_process_count": EXPECTED_CASES_PER_BACKEND,
        "failed_case_id": None,
    }
    if dict(status) != expected_status:
        raise ContactC0FinalizationError("MuJoCo launcher status is not canonical completed")
    return status, runtime


def _read_strict_text(path: Path, label: str) -> str:
    if is_link_like(path) or not path.is_file():
        raise ContactC0FinalizationError(f"{label} is not a regular file")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as error:
        raise ContactC0FinalizationError(f"{label} is not strict UTF-8") from error


def _parse_gpu_state(path: Path, label: str) -> dict[str, object]:
    """Parse one launcher-produced nvidia-smi state without trusting its status."""

    text = _read_strict_text(path, label)
    if not text.endswith("\n"):
        raise ContactC0FinalizationError(f"{label} is not newline terminated")
    lines = text.splitlines()
    fixed_names = (
        "recorded_at_utc",
        "query_status",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "driver_version",
        "memory_used_mib",
        "utilization_percent",
    )
    if len(lines) < len(fixed_names) + 2:
        raise ContactC0FinalizationError(f"{label} is truncated")
    values: dict[str, str] = {}
    for line, expected_name in zip(lines[: len(fixed_names)], fixed_names, strict=True):
        name, separator, value = line.partition("=")
        if separator != "=" or name != expected_name or not value:
            raise ContactC0FinalizationError(f"{label} fields are not exact")
        values[name] = value
    if (
        lines[len(fixed_names)] != "compute_processes_begin"
        or lines[-1] != "compute_processes_end"
    ):
        raise ContactC0FinalizationError(f"{label} compute-process envelope is invalid")
    processes: list[dict[str, object]] = []
    for row in lines[len(fixed_names) + 1 : -1]:
        columns = [column.strip() for column in row.split(",")]
        if len(columns) != 3:
            raise ContactC0FinalizationError(f"{label} compute-process row is malformed")
        process_uuid, process_pid, used_memory = columns
        if _GPU_UUID.fullmatch(process_uuid) is None:
            raise ContactC0FinalizationError(f"{label} compute-process UUID is invalid")
        try:
            pid = int(process_pid)
            memory = int(used_memory)
        except ValueError as error:
            raise ContactC0FinalizationError(
                f"{label} compute-process values are invalid"
            ) from error
        if pid <= 0 or memory < 0:
            raise ContactC0FinalizationError(
                f"{label} compute-process values are invalid"
            )
        processes.append({"gpu_uuid": process_uuid, "pid": pid, "memory_mib": memory})
    try:
        parsed = {
            "recorded_at_utc": values["recorded_at_utc"],
            "query_status": int(values["query_status"]),
            "gpu_index": int(values["gpu_index"]),
            "gpu_uuid": values["gpu_uuid"],
            "gpu_name": values["gpu_name"],
            "driver_version": values["driver_version"],
            "memory_used_mib": int(values["memory_used_mib"]),
            "utilization_percent": int(values["utilization_percent"]),
            "compute_processes": processes,
        }
    except ValueError as error:
        raise ContactC0FinalizationError(f"{label} numeric fields are invalid") from error
    if _UTC_SECOND.fullmatch(str(parsed["recorded_at_utc"])) is None:
        raise ContactC0FinalizationError(f"{label} timestamp is invalid")
    if (
        parsed["query_status"] not in {0, 1}
        or not 0 <= int(parsed["gpu_index"]) <= 7
        or _GPU_UUID.fullmatch(str(parsed["gpu_uuid"])) is None
        or int(parsed["memory_used_mib"]) < 0
        or int(parsed["utilization_percent"]) < 0
    ):
        raise ContactC0FinalizationError(f"{label} values are invalid")
    return parsed


def _gpu_state_is_strictly_idle(
    state: Mapping[str, object],
    *,
    gpu_index: int,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
) -> bool:
    return (
        state.get("gpu_index") == gpu_index
        and state.get("gpu_uuid") == gpu_uuid
        and state.get("gpu_name") == gpu_name
        and state.get("driver_version") == driver_version
        and int(state.get("memory_used_mib", -1)) <= 32
        and state.get("utilization_percent") == 0
        and state.get("compute_processes") == []
    )


def _validate_remote_launcher_security_evidence(
    launcher: Path,
    *,
    expected_case_ids: Sequence[str],
    snapshot_sha256: str,
    gpu_index: int,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
) -> None:
    selection = _parse_gpu_state(
        launcher / "gpu-selection.txt", "OVPhysX launcher GPU selection"
    )
    if selection["query_status"] != 0 or not _gpu_state_is_strictly_idle(
        selection,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version=driver_version,
    ):
        raise ContactC0FinalizationError(
            "OVPhysX launcher GPU selection was not strictly idle"
        )

    text = _read_strict_text(
        launcher / "snapshot-verifications.txt",
        "OVPhysX snapshot verifications",
    )
    if not text.endswith("\n"):
        raise ContactC0FinalizationError(
            "OVPhysX snapshot verifications are not newline terminated"
        )
    expected_checkpoints = [
        *(f"before-{case_id}" for case_id in expected_case_ids),
        "post-matrix",
    ]
    observed_checkpoints: list[str] = []
    pattern = re.compile(
        r"^(?P<time>[0-9T:Z-]+) checkpoint=(?P<checkpoint>[^ ]+) "
        r"snapshot_sha256=(?P<digest>[0-9a-f]{64})$"
    )
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match is None or _UTC_SECOND.fullmatch(match.group("time")) is None:
            raise ContactC0FinalizationError(
                "OVPhysX snapshot verification record is malformed"
            )
        if match.group("digest") != snapshot_sha256:
            raise ContactC0FinalizationError(
                "OVPhysX snapshot verification source identity differs"
            )
        observed_checkpoints.append(match.group("checkpoint"))
    if observed_checkpoints != expected_checkpoints:
        raise ContactC0FinalizationError(
            "OVPhysX snapshot verification checkpoints are not exact"
        )


def _validate_remote_native_helper(
    root: Path,
    *,
    run_id: str,
    source_tree: str,
    project_root: Path,
    launcher_summary_value: object,
) -> dict[str, str]:
    """Verify the extracted SDK headers, exact build recipe, and helper bytes."""

    native = root / "launcher" / "native"
    include = native / "include"
    headers_manifest_path = native / "headers-manifest.json"
    build_provenance_path = native / "build-provenance.json"
    helper_path = native / _NATIVE_HELPER_FILENAME
    remote_native = _REMOTE_ROOT / "results" / run_id / "launcher" / "native"
    remote_include = remote_native / "include"
    remote_headers_manifest = remote_native / "headers-manifest.json"
    remote_build_provenance = remote_native / "build-provenance.json"
    remote_helper = remote_native / _NATIVE_HELPER_FILENAME
    remote_source = (
        _REMOTE_ROOT / "project" / source_tree / _NATIVE_HELPER_SOURCE_RELATIVE
    )

    headers_manifest = _exact_mapping(
        read_json_strict(headers_manifest_path),
        {
            "schema_version",
            "sdk_archive_path",
            "sdk_archive_sha256",
            "include_root_in_archive",
            "file_count",
            "total_bytes",
            "records",
            "root_sha256",
        },
        "OVPhysX native header manifest",
    )
    if (
        headers_manifest.get("schema_version") != 1
        or headers_manifest.get("sdk_archive_path") != str(_OVPHYSX_SDK_ARCHIVE)
        or headers_manifest.get("sdk_archive_sha256")
        != _OVPHYSX_SDK_ARCHIVE_SHA256
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native header manifest SDK identity differs"
        )
    include_root_in_archive = _nonempty_string(
        headers_manifest.get("include_root_in_archive"),
        "OVPhysX native include root in archive",
    )
    include_root_posix = PurePosixPath(include_root_in_archive)
    if (
        include_root_posix.is_absolute()
        or ".." in include_root_posix.parts
        or include_root_posix.as_posix() != include_root_in_archive
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native include root in archive is unsafe"
        )
    records = headers_manifest.get("records")
    if not isinstance(records, list):
        raise ContactC0FinalizationError(
            "OVPhysX native header manifest records must be an array"
        )
    try:
        observed_records = list(regular_tree_records(include))
        observed_root = inventory_root_sha256(observed_records)
    except ContactEvidenceError as error:
        raise ContactC0FinalizationError(
            "OVPhysX native header tree is not regular and exact"
        ) from error
    if records != observed_records:
        raise ContactC0FinalizationError(
            "OVPhysX native header manifest does not match exact header inventory"
        )
    file_count = _nonnegative_int(
        headers_manifest.get("file_count"),
        "OVPhysX native header file count",
        maximum=20_000,
    )
    total_bytes = _nonnegative_int(
        headers_manifest.get("total_bytes"),
        "OVPhysX native header byte count",
        maximum=256 * 1024 * 1024,
    )
    if (
        file_count != len(observed_records)
        or file_count < 3
        or total_bytes != sum(int(record["size"]) for record in observed_records)
        or total_bytes <= 0
        or headers_manifest.get("root_sha256") != observed_root
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native header inventory summary differs"
        )
    header_paths = {str(record["path"]) for record in observed_records}
    if not {
        "PxArticulationLink.h",
        "PxScene.h",
        "PxRigidBody.h",
        "foundation/PxPhysicsVersion.h",
    }.issubset(header_paths):
        raise ContactC0FinalizationError(
            "OVPhysX native header inventory lacks the required PhysX API"
        )

    headers_manifest_sha256 = _file_sha256(headers_manifest_path)
    build_provenance = _exact_mapping(
        read_json_strict(build_provenance_path),
        {
            "schema_version",
            "helper_id",
            "sdk_archive",
            "headers",
            "source",
            "compiler",
            "build",
        },
        "OVPhysX native helper build provenance",
    )
    if (
        build_provenance.get("schema_version") != 1
        or build_provenance.get("helper_id") != "waveqa_physx_ccd_readback"
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native helper build identity differs"
        )
    sdk = _exact_mapping(
        build_provenance.get("sdk_archive"),
        {"package", "version", "path", "sha256"},
        "OVPhysX native helper SDK archive provenance",
    )
    if dict(sdk) != {
        "package": "ovphysx",
        "version": "0.4.13",
        "path": str(_OVPHYSX_SDK_ARCHIVE),
        "sha256": _OVPHYSX_SDK_ARCHIVE_SHA256,
    }:
        raise ContactC0FinalizationError(
            "OVPhysX native helper SDK archive provenance differs"
        )
    headers = _exact_mapping(
        build_provenance.get("headers"),
        {
            "directory",
            "manifest_path",
            "manifest_sha256",
            "include_root_in_archive",
            "file_count",
            "total_bytes",
            "root_sha256",
        },
        "OVPhysX native helper header provenance",
    )
    expected_headers = {
        "directory": str(remote_include),
        "manifest_path": str(remote_headers_manifest),
        "manifest_sha256": headers_manifest_sha256,
        "include_root_in_archive": include_root_in_archive,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "root_sha256": observed_root,
    }
    if dict(headers) != expected_headers:
        raise ContactC0FinalizationError(
            "OVPhysX native helper header build provenance differs"
        )
    source = _exact_mapping(
        build_provenance.get("source"),
        {"relative_path", "path", "sha256"},
        "OVPhysX native helper source provenance",
    )
    local_source = project_root.joinpath(*_NATIVE_HELPER_SOURCE_RELATIVE.parts)
    expected_source_sha256 = _file_sha256(local_source)
    if dict(source) != {
        "relative_path": _NATIVE_HELPER_SOURCE_RELATIVE.as_posix(),
        "path": str(remote_source),
        "sha256": expected_source_sha256,
    }:
        raise ContactC0FinalizationError(
            "OVPhysX native helper immutable source provenance differs"
        )
    compiler = _exact_mapping(
        build_provenance.get("compiler"),
        {"path", "sha256", "version"},
        "OVPhysX native helper compiler provenance",
    )
    compiler_path = _nonempty_string(
        compiler.get("path"), "OVPhysX native helper compiler path"
    )
    compiler_posix = PurePosixPath(compiler_path)
    compiler_version = _nonempty_string(
        compiler.get("version"), "OVPhysX native helper compiler version"
    )
    if (
        not compiler_posix.is_absolute()
        or compiler_posix.parent != PurePosixPath("/usr/bin")
        or "g++" not in compiler_posix.name
        or len(compiler_version.encode("utf-8")) > 32 * 1024
        or "\x00" in compiler_version
        or "g++" not in compiler_version.splitlines()[0]
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native helper compiler identity is invalid"
        )
    _require_hex(
        compiler.get("sha256"), "OVPhysX native helper compiler SHA-256", length=64
    )
    build = _exact_mapping(
        build_provenance.get("build"),
        {"argv", "output_path", "output_sha256"},
        "OVPhysX native helper build invocation",
    )
    expected_argv = [
        compiler_path,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(remote_include),
        str(remote_source),
        "-o",
        str(remote_helper),
    ]
    helper_sha256 = _file_sha256(helper_path)
    if helper_path.stat().st_size <= 0 or helper_path.stat().st_size > 32 * 1024 * 1024:
        raise ContactC0FinalizationError(
            "OVPhysX native helper size is outside the frozen bound"
        )
    elf_header = helper_path.read_bytes()[:20]
    if (
        len(elf_header) < 20
        or elf_header[:4] != b"\x7fELF"
        or elf_header[4:6] != b"\x02\x01"
        or int.from_bytes(elf_header[16:18], "little") != 3
        or int.from_bytes(elf_header[18:20], "little") != 62
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native helper is not an x86-64 ELF shared object"
        )
    if (
        build.get("argv") != expected_argv
        or build.get("output_path") != str(remote_helper)
        or build.get("output_sha256") != helper_sha256
    ):
        raise ContactC0FinalizationError(
            "OVPhysX native helper build invocation/output differs"
        )
    build_provenance_sha256 = _file_sha256(build_provenance_path)
    launcher_summary = _exact_mapping(
        launcher_summary_value,
        {
            "build_provenance_path",
            "build_provenance_sha256",
            "helper_path",
            "helper_sha256",
        },
        "OVPhysX native helper launcher summary",
    )
    expected_launcher_summary = {
        "build_provenance_path": str(remote_build_provenance),
        "build_provenance_sha256": build_provenance_sha256,
        "helper_path": str(remote_helper),
        "helper_sha256": helper_sha256,
    }
    if dict(launcher_summary) != expected_launcher_summary:
        raise ContactC0FinalizationError(
            "OVPhysX native helper launcher summary differs"
        )
    return {"path": str(remote_helper), "sha256": helper_sha256}


def _validate_remote_launcher_identity(
    root: Path,
    *,
    manifest: object,
    expected_case_ids: list[str],
    source_revision: str,
    source_tree: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    project_root: Path,
    experimental_campaign: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise ContactC0FinalizationError('unknown candidate campaign')
    launcher = root / "launcher"
    provenance = _exact_mapping(
        read_json_strict(launcher / "provenance.json"),
        set(_REMOTE_PROVENANCE_KEYS) | ({'slurm_binding'} if experimental_campaign else set()),
        "OVPhysX launcher provenance",
    )
    status = _exact_mapping(
        read_json_strict(launcher / "status.json"),
        _REMOTE_STATUS_KEYS,
        "OVPhysX launcher status",
    )
    matrix = _exact_mapping(
        read_json_strict(launcher / "matrix.json"),
        {
            "schema_version",
            "manifest_id",
            "manifest_file_sha256",
            "manifest_semantic_sha256",
            "backend",
            "case_ids",
        },
        "OVPhysX launcher matrix",
    )
    expected_matrix = {
        "schema_version": 1,
        "manifest_id": manifest.manifest_id,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "backend": "ovphysx",
        "case_ids": expected_case_ids,
    }
    if dict(matrix) != expected_matrix:
        raise ContactC0FinalizationError("OVPhysX launcher matrix is not canonical")
    session_id = status.get("session_id")
    run_id = status.get("run_id")
    if not isinstance(session_id, str) or _SESSION.fullmatch(session_id) is None:
        raise ContactC0FinalizationError("OVPhysX session ID is invalid")
    if not isinstance(run_id, str) or _SESSION.fullmatch(run_id) is None:
        raise ContactC0FinalizationError("OVPhysX run ID is invalid")
    fixed = {
        "schema_version": 1,
        "campaign": experimental_campaign or "contact_c0",
        "backend": "ovphysx",
        "run_id": run_id,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "source_archive_sha256": source_archive_sha256,
        "snapshot_sha256": remote_snapshot_sha256,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest_semantic_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "isaaclab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
    }
    mismatches = sorted(key for key, value in fixed.items() if provenance.get(key) != value)
    if mismatches:
        raise ContactC0FinalizationError(
            "OVPhysX launcher provenance differs: " + ", ".join(mismatches)
        )
    selected_gpu = _exact_mapping(
        provenance.get("selected_gpu"),
        {("query_index" if experimental_campaign else "physical_index"), "uuid", "name", "driver_version", "worker_visible_device"},
        "OVPhysX selected GPU",
    )
    gpu_index = _nonnegative_int(
        selected_gpu.get("query_index" if experimental_campaign else "physical_index"), "OVPhysX query GPU index", maximum=7
    )
    gpu_uuid = _nonempty_string(selected_gpu.get("uuid"), "OVPhysX GPU UUID")
    if experimental_campaign:
        from wave_asset_qa.contact.slurm_binding import validate_binding
        binding = validate_binding(provenance.get('slurm_binding'))
        if gpu_index != binding['query_index'] or gpu_uuid != binding['gpu_uuid']:
            raise ContactC0FinalizationError('GPU selection differs from Slurm binding')
    gpu_name = _nonempty_string(selected_gpu.get("name"), "OVPhysX GPU name")
    driver_version = _nonempty_string(
        selected_gpu.get("driver_version"), "OVPhysX driver version"
    )
    if selected_gpu.get("worker_visible_device") != "cuda:0":
        raise ContactC0FinalizationError("OVPhysX worker device is not cuda:0")
    environment = _exact_mapping(
        provenance.get("environment"),
        {"python", "packages", "torch_cuda_build", "kitless", "renderer", "camera"},
        "OVPhysX environment",
    )
    expected_packages = {
        "isaaclab": "6.1.14",
        "isaaclab-ovphysx": "3.0.2",
        "ovphysx": "0.4.13",
        "torch": "2.10.0+cu128",
        "usd-core": "25.11",
    }
    if (
        environment.get("python") != "3.12.14"
        or environment.get("packages") != expected_packages
        or environment.get("torch_cuda_build") != "12.8"
        or environment.get("kitless") is not True
        or environment.get("renderer") is not False
        or environment.get("camera") is not False
    ):
        raise ContactC0FinalizationError("OVPhysX frozen dependency/runtime identity drifted")
    boundary = _exact_mapping(
        provenance.get("claim_boundary"),
        {
            "synthetic_fixture_only",
            "kitless_ovphysx",
            "full_isaac_sim",
            "renderer",
            "camera",
            "dual_hand",
            "floating_base",
            "hardware",
            "sim2real",
        },
        "OVPhysX launcher claim boundary",
    )
    if dict(boundary) != {
        "synthetic_fixture_only": True,
        "kitless_ovphysx": True,
        "full_isaac_sim": False,
        "renderer": False,
        "camera": False,
        "dual_hand": False,
        "floating_base": False,
        "hardware": False,
        "sim2real": False,
    }:
        raise ContactC0FinalizationError("OVPhysX launcher claim boundary expanded")
    started = provenance.get("started_at_utc")
    ended = status.get("ended_at_utc")
    if (
        not isinstance(started, str)
        or _UTC_SECOND.fullmatch(started) is None
        or not isinstance(ended, str)
        or _UTC_SECOND.fullmatch(ended) is None
    ):
        raise ContactC0FinalizationError("OVPhysX launcher timestamps are not canonical UTC")
    expected_status = {
        "schema_version": 1,
        "campaign": experimental_campaign or "contact_c0",
        "backend": "ovphysx",
        "status": "completed",
        "scientific_verdict": "pending_local_compare",
        "exit_code": 0,
        "run_id": run_id,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "source_archive_sha256": source_archive_sha256,
        "snapshot_sha256": remote_snapshot_sha256,
        "selected_gpu_index": gpu_index,
        "selected_gpu_uuid": gpu_uuid,
        "completed_case_count": EXPECTED_CASES_PER_BACKEND,
        "unique_process_count": EXPECTED_CASES_PER_BACKEND,
        "last_case_id": None,
        "elapsed_s": status.get("elapsed_s"),
        "gpu_elapsed_s": status.get("gpu_elapsed_s"),
        "ended_at_utc": ended,
    }
    _nonnegative_int(status.get("elapsed_s"), "OVPhysX elapsed_s")
    _nonnegative_int(status.get("gpu_elapsed_s"), "OVPhysX gpu_elapsed_s", maximum=14_400)
    if dict(status) != expected_status:
        raise ContactC0FinalizationError("OVPhysX launcher status is not canonical completed")
    native_helper = _validate_remote_native_helper(
        root,
        run_id=run_id,
        source_tree=source_tree,
        project_root=project_root,
        launcher_summary_value=provenance.get("native_physx_ccd_helper"),
    )
    _validate_remote_launcher_security_evidence(
        launcher,
        expected_case_ids=expected_case_ids,
        snapshot_sha256=remote_snapshot_sha256,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version=driver_version,
    )
    return status, environment, native_helper


def _validate_process_record(
    value: object,
    *,
    case_id: str,
    backend: str,
    session_id: str,
    source_revision: str,
    source_tree: str,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    asset_tree_sha256: str,
    worker_module_sha256: str,
    experimental_campaign: str | None = None,
) -> tuple[dict[str, object], str]:
    data = _mapping(value, f"{case_id} private process")
    keys = set(_PRIVATE_PROCESS_KEYS)
    if experimental_campaign is not None:
        if experimental_campaign != 'contact-c0-async-v1':
            raise ContactC0FinalizationError('unknown candidate campaign')
        keys.add('experimental_campaign')
    if set(data) != keys:
        raise ContactC0FinalizationError(f"{case_id} private process fields are not exact")
    expected = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": case_id,
        "backend": backend,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "asset_tree_sha256": asset_tree_sha256,
        "worker_module_sha256": worker_module_sha256,
    }
    if experimental_campaign is not None:
        expected['experimental_campaign'] = experimental_campaign
    mismatches = [key for key, target in expected.items() if data.get(key) != target]
    if mismatches:
        raise ContactC0FinalizationError(
            f"{case_id} private process differs: " + ", ".join(sorted(mismatches))
        )
    identity = validate_os_process_identity(data["os_process_identity"])
    fresh_hash = os_process_identity_sha256(identity)
    if data["fresh_process_identity_sha256"] != fresh_hash:
        raise ContactC0FinalizationError(f"{case_id} fresh process hash mismatch")
    for key in ("python_version", "python_executable_realpath", "worker_module_realpath"):
        if not isinstance(data[key], str) or not data[key]:
            raise ContactC0FinalizationError(f"{case_id} private process {key} is invalid")
    _require_hex(data["worker_module_sha256"], f"{case_id} worker module hash", length=64)
    return dict(data), fresh_hash


def _validate_remote_case_security_evidence(
    case_root: Path,
    *,
    case_id: str,
    run_id: str,
    session_id: str,
    source_revision: str,
    source_tree: str,
    asset_tree_sha256: str,
    gpu_index: int,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
    native_helper_path: str,
    native_helper_sha256: str,
    process: Mapping[str, object],
    fresh_process_identity_sha256: str,
    experimental_campaign: str | None = None,
) -> None:
    """Recompute the remote command, GPU, and owned-process-group bindings."""

    project_dir = _REMOTE_ROOT / "project" / source_tree
    remote_case_dir = _REMOTE_ROOT / "results" / run_id / "cases" / case_id
    worker_python = _REMOTE_ROOT / "env" / "bin" / "python"
    worker_script = project_dir / "scripts" / ("run_contact_async_worker.py" if experimental_campaign else "run_contact_c0_ovphysx_worker.py")
    manifest_path = project_dir / "configs" / "parity" / "contact_c0.json"

    command = _exact_mapping(
        read_json_strict(case_root / "command.json"),
        {"schema_version", "case_id", "argv", "environment"},
        f"{case_id} remote command",
    )
    expected_argv = [
        str(worker_python),
        "-P",
        str(worker_script),
        *(["--backend", "ovphysx"] if experimental_campaign else []),
        "--asset-root",
        str(_REMOTE_ROOT / "assets"),
        "--manifest",
        str(manifest_path),
        "--case-id",
        case_id,
        "--output-dir",
        str(remote_case_dir),
        "--session-id",
        session_id,
        "--source-revision",
        source_revision,
        "--source-tree",
        source_tree,
        "--asset-tree-sha256",
        asset_tree_sha256,
        "--device",
        "cuda:0",
    ]
    command_environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "WAVEQA_PHYSX_CCD_HELPER_PATH": native_helper_path,
        "WAVEQA_PHYSX_CCD_HELPER_SHA256": native_helper_sha256,
        "worker_device": "cuda:0",
        "DISPLAY": None,
        "WAYLAND_DISPLAY": None,
    }
    if (
        command.get("schema_version") != 1
        or command.get("case_id") != case_id
        or command.get("argv") != expected_argv
        or command.get("environment") != command_environment
    ):
        raise ContactC0FinalizationError(
            f"{case_id} remote command/source/output-directory binding differs"
        )

    process_identity = validate_os_process_identity(process.get("os_process_identity"))
    if process_identity.get("platform") != "posix":
        raise ContactC0FinalizationError(f"{case_id} remote worker is not POSIX")
    expected_worker_module = str(
        project_dir / "src" / "wave_asset_qa" / "contact" / "worker.py"
    )
    if process.get("worker_module_realpath") != expected_worker_module:
        raise ContactC0FinalizationError(
            f"{case_id} remote worker module escaped the immutable source tree"
        )
    python_realpath = process.get("python_executable_realpath")
    if not isinstance(python_realpath, str) or not (
        python_realpath.startswith(str(_REMOTE_ROOT / "env") + "/")
        or python_realpath.startswith(str(_REMOTE_ROOT / "toolchain" / "python") + "/")
    ):
        raise ContactC0FinalizationError(
            f"{case_id} remote Python escaped the approved runtime root"
        )

    group = _exact_mapping(
        read_json_strict(case_root / "process.json"),
        {
            "schema_version",
            "visibility",
            "case_id",
            "transport_pid",
            "owned_process_group_id",
        },
        f"{case_id} owned process group",
    )
    transport_pid = group.get("transport_pid")
    owned_pgid = group.get("owned_process_group_id")
    if (
        group.get("schema_version") != 1
        or group.get("visibility") != "private_not_for_publication"
        or group.get("case_id") != case_id
        or isinstance(transport_pid, bool)
        or not isinstance(transport_pid, int)
        or transport_pid <= 0
        or owned_pgid != transport_pid
    ):
        raise ContactC0FinalizationError(f"{case_id} owned process group is invalid")
    if _read_strict_text(
        case_root / "owned-process-group.txt", f"{case_id} owned PGID handshake"
    ) != f"{owned_pgid}\n":
        raise ContactC0FinalizationError(
            f"{case_id} owned process-group handshake differs"
        )
    binding = _exact_mapping(
        read_json_strict(case_root / "worker-group-binding.json"),
        {
            "schema_version",
            "visibility",
            "case_id",
            "worker_pid",
            "owned_process_group_id",
            "fresh_process_identity_sha256",
        },
        f"{case_id} worker-group binding",
    )
    expected_binding = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": case_id,
        "worker_pid": process_identity["pid"],
        "owned_process_group_id": owned_pgid,
        "fresh_process_identity_sha256": fresh_process_identity_sha256,
    }
    if dict(binding) != expected_binding:
        raise ContactC0FinalizationError(
            f"{case_id} worker-to-owned-process-group binding differs"
        )

    for suffix in ("01", "02", "03"):
        state = _parse_gpu_state(
            case_root / f"gpu-preflight-{suffix}.txt",
            f"{case_id} GPU preflight {suffix}",
        )
        if state["query_status"] != 0 or not _gpu_state_is_strictly_idle(
            state,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version=driver_version,
        ):
            raise ContactC0FinalizationError(
                f"{case_id} GPU preflight {suffix} was not strictly idle"
            )

    postflight_paths = sorted(case_root.glob("gpu-postflight-[0-9][0-9].txt"))
    if not postflight_paths:
        raise ContactC0FinalizationError(f"{case_id} GPU postflight is absent")
    for index, path in enumerate(postflight_paths):
        state = _parse_gpu_state(path, f"{case_id} GPU postflight {index + 1:02d}")
        identity_matches = all(
            (
                state.get("gpu_index") == gpu_index,
                state.get("gpu_uuid") == gpu_uuid,
                state.get("gpu_name") == gpu_name,
                state.get("driver_version") == driver_version,
            )
        )
        if not identity_matches:
            raise ContactC0FinalizationError(
                f"{case_id} GPU postflight device identity differs"
            )
        idle = _gpu_state_is_strictly_idle(
            state,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version=driver_version,
        )
        final = index == len(postflight_paths) - 1
        if final and (state["query_status"] != 0 or not idle):
            raise ContactC0FinalizationError(
                f"{case_id} final GPU postflight was not strictly idle"
            )
        if not final and state["query_status"] == 0:
            raise ContactC0FinalizationError(
                f"{case_id} retained a postflight after an earlier successful check"
            )

    monitor = _read_strict_text(case_root / "gpu-monitor.txt", f"{case_id} GPU monitor")
    if not monitor.endswith("\n"):
        raise ContactC0FinalizationError(f"{case_id} GPU monitor is not newline terminated")
    lines = monitor.splitlines()
    sample_count = 0
    cursor = 0
    header_pattern = re.compile(
        r"^sample_at_utc=(?P<time>[0-9T:Z-]+) owner_pgid=(?P<pgid>[1-9][0-9]*)$"
    )
    while cursor < len(lines):
        match = header_pattern.fullmatch(lines[cursor])
        if (
            match is None
            or _UTC_SECOND.fullmatch(match.group("time")) is None
            or int(match.group("pgid")) != owned_pgid
        ):
            raise ContactC0FinalizationError(
                f"{case_id} GPU monitor owner-PGID record differs"
            )
        cursor += 1
        if cursor >= len(lines) or lines[cursor] != "compute_processes_begin":
            raise ContactC0FinalizationError(
                f"{case_id} GPU monitor process envelope is invalid"
            )
        cursor += 1
        while cursor < len(lines) and lines[cursor] != "compute_processes_end":
            columns = [column.strip() for column in lines[cursor].split(",")]
            if len(columns) != 3 or columns[0] != gpu_uuid:
                raise ContactC0FinalizationError(
                    f"{case_id} GPU monitor observed a foreign-device process"
                )
            try:
                process_pid = int(columns[1])
                used_memory = int(columns[2])
            except ValueError as error:
                raise ContactC0FinalizationError(
                    f"{case_id} GPU monitor process row is invalid"
                ) from error
            if (
                process_pid <= 0
                or used_memory < 0
                or process_pid != process_identity["pid"]
            ):
                raise ContactC0FinalizationError(
                    f"{case_id} GPU monitor process is not the bound worker"
                )
            cursor += 1
        if cursor >= len(lines):
            raise ContactC0FinalizationError(
                f"{case_id} GPU monitor process envelope is truncated"
            )
        cursor += 1
        sample_count += 1
    if sample_count == 0:
        raise ContactC0FinalizationError(f"{case_id} GPU monitor has no samples")


def _validate_private_adapter(
    value: object,
    run: ContactRun,
    *,
    condition: str,
    native_helper_sha256: str | None = None,
    experimental_campaign: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    private = verify_adapter_private_evidence(value, run)
    backend = run.case.simulator
    expected_private_keys = (
        {"fixture_overlay", "runtime_fingerprint", "unexpected_pair_observations"}
        if backend == "mujoco"
        else {"fixture_overlay", "runtime_fingerprint", "unexpected_pair_observation"}
    )
    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise ContactC0FinalizationError('unknown candidate campaign')
    if experimental_campaign and backend == 'ovphysx':
        expected_private_keys.add('async_step_audit')
    _exact_mapping(private, expected_private_keys, f"{backend} adapter evidence")
    fixture_keys = {
        "schema_version",
        "backend",
        "overlay",
        "geometry",
        "collision",
        "material",
        "mass_properties",
        "native_collision_disabled_inventory",
        "solver",
        "collision_inventory_hash_preimage",
    }
    if backend == "ovphysx":
        fixture_keys.update({"final_collision_inventory", "sensor_readback"})
    fixture = _exact_mapping(
        private.get("fixture_overlay"), fixture_keys, f"{backend} fixture overlay"
    )
    runtime_keys = (
        {
            "backend",
            "backend_version",
            "python_version",
            "platform",
            "device",
            "solver",
            "integrator",
            "control_path",
            "contact_path",
        }
        if backend == "mujoco"
        else {
            "backend",
            "python_version",
            "platform",
            "device",
            "dt_s",
            "module_versions",
            "solver_config_type",
            "control_path",
            "contact_path",
            "sensor_update_order",
            "kitless",
            "renderer",
            "camera",
            "scene_inventory_verification_phase",
            "forbidden_module_verification_phase",
            "forbidden_module_inventory",
            "camera_prim_paths",
            "created_sensor_types",
        }
    )
    if experimental_campaign and backend == 'ovphysx':
        runtime_keys.add('stepping_intervention')
        from wave_asset_qa.contact.async_campaign import validate_candidate_evidence
        validate_candidate_evidence(private,run,dt_s=run.case.dt_s)
    runtime = _exact_mapping(
        private.get("runtime_fingerprint"), runtime_keys, f"{backend} runtime fingerprint"
    )
    if (
        fixture.get("schema_version") != 1
        or fixture.get("backend") != backend
        or runtime.get("backend") != backend
    ):
        raise ContactC0FinalizationError("adapter backend preimage differs from its run")
    run_fixture = _mapping(run.fixture_readback, "run fixture readback")
    run_observation = _mapping(run.contact_observation, "run contact observation")
    geometry = _exact_mapping(
        fixture.get("geometry"), {"probe", "target"}, "fixture geometry"
    )
    probe = _mapping(geometry.get("probe"), "fixture probe geometry")
    target = _mapping(geometry.get("target"), "fixture target geometry")
    expected_frame = str(run_fixture.get("probe_parent_frame_name"))
    if backend == "mujoco":
        _exact_mapping(
            probe,
            {
                "canonical_id",
                "parent_frame",
                "compiled_geom_id",
                "shape",
                "local_center_m",
                "radius_m",
            },
            "MuJoCo probe geometry",
        )
        _exact_mapping(
            target,
            {
                "canonical_id",
                "compiled_geom_id",
                "shape",
                "center_world_m",
                "half_extents_m",
                "top_world_m",
                "static",
            },
            "MuJoCo target geometry",
        )
        parent_frame = probe.get("parent_frame")
    else:
        _exact_mapping(
            probe,
            {
                "canonical_id",
                "parent_prim_path",
                "prim_path",
                "shape",
                "local_center_m",
                "radius_m",
            },
            "OV probe geometry",
        )
        _exact_mapping(
            target,
            {
                "canonical_id",
                "prim_path",
                "shape",
                "center_world_m",
                "half_extents_m",
                "top_world_m",
                "static",
            },
            "OV target geometry",
        )
        parent_path = _nonempty_string(
            probe.get("parent_prim_path"), "OV probe parent path"
        )
        parent_frame = parent_path.rsplit("/", 1)[-1]
    geometry_expected = {
        "probe_id": "synthetic_index_probe",
        "probe_shape": "sphere",
        "probe_center": run_fixture.get("probe_local_center_m"),
        "probe_radius": run_fixture.get("probe_radius_m"),
        "target_id": "static_box",
        "target_shape": "box",
        "target_center": run_fixture.get("target_world_center_m"),
        "target_half_extents": run_fixture.get("target_half_extents_m"),
        "target_top": run_fixture.get("target_top_surface_z_m"),
    }
    geometry_observed = {
        "probe_id": probe.get("canonical_id"),
        "probe_shape": probe.get("shape"),
        "probe_center": probe.get("local_center_m"),
        "probe_radius": probe.get("radius_m"),
        "target_id": target.get("canonical_id"),
        "target_shape": target.get("shape"),
        "target_center": target.get("center_world_m"),
        "target_half_extents": target.get("half_extents_m"),
        "target_top": target.get("top_world_m"),
    }
    if (
        parent_frame != expected_frame
        or geometry_observed != geometry_expected
        or target.get("static") is not True
    ):
        raise ContactC0FinalizationError(
            "adapter geometry preimage differs from run readback"
        )
    collision = _mapping(fixture.get("collision"), "fixture collision")
    pair_enabled = condition == "contact"
    if (
        collision.get("allowed_pair_id") != "synthetic_index_probe__static_box"
        or collision.get("pair_collision_enabled") is not pair_enabled
    ):
        raise ContactC0FinalizationError("adapter selected-pair readback differs")
    native = _mapping(
        fixture.get("native_collision_disabled_inventory"),
        "native collision inventory",
    )
    _exact_mapping(
        native,
        {"count", "all_disabled", "items"},
        "native collision inventory",
    )
    items = native.get("items")
    if (
        native.get("all_disabled") is not True
        or not isinstance(items, list)
        or native.get("count") != len(items)
        or len(items) < 1
        or run_fixture.get("native_collision_prim_count") != len(items)
        or run_fixture.get("native_collision_disabled_count") != len(items)
    ):
        raise ContactC0FinalizationError("native collision inventory is incomplete")
    if backend == "mujoco":
        if any(
            not isinstance(item, Mapping)
            or item.get("disabled") is not True
            or item.get("compiled_contype") != 0
            or item.get("compiled_conaffinity") != 0
            for item in items
        ):
            raise ContactC0FinalizationError(
                "MuJoCo native collision mask is not disabled"
            )
    elif any(
        not isinstance(item, Mapping)
        or item.get("effective_collision_enabled") is not False
        for item in items
    ):
        raise ContactC0FinalizationError("OV native collision mask is not disabled")
    mass = _mapping(fixture.get("mass_properties"), "mass properties")
    _exact_mapping(
        mass,
        (
            {"preserved", "source", "effective", "synthetic_probe_density"}
            if backend == "mujoco"
            else {
                "preserved",
                "usd_composed_preserved",
                "runtime_verified",
                "source",
                "effective",
                "synthetic_probe_mass_api_applied",
                "runtime_effective",
                "runtime_checks",
            }
        ),
        f"{backend} mass properties",
    )
    if (
        mass.get("preserved") is not True
        or mass.get("source") != mass.get("effective")
    ):
        raise ContactC0FinalizationError("synthetic fixture changed mass properties")
    collision_preimage = fixture.get("collision_inventory_hash_preimage")
    collision_preimage_map = _exact_mapping(
        collision_preimage,
        {"native_collision_items", "synthetic_enabled_shapes"},
        "collision inventory hash preimage",
    )
    if (
        collision_preimage_map.get("native_collision_items") != items
        or collision_preimage_map.get("synthetic_enabled_shapes")
        != ["synthetic_index_probe", "static_box"]
    ):
        raise ContactC0FinalizationError("collision inventory preimage contents differ")
    if canonical_json_sha256(collision_preimage) != run_fixture.get(
        "collision_inventory_sha256"
    ):
        raise ContactC0FinalizationError("collision inventory preimage hash mismatch")
    overlay = _mapping(fixture.get("overlay"), "adapter overlay")
    _exact_mapping(
        overlay,
        (
            {
                "method",
                "canonical_source_edited",
                "source_sha256_before",
                "source_sha256_after",
            }
            if backend == "mujoco"
            else {
                "method",
                "canonical_source_edited",
                "first_reset_count_at_readback",
            }
        ),
        f"{backend} adapter overlay",
    )
    if overlay.get("canonical_source_edited") is not False:
        raise ContactC0FinalizationError("adapter reports editing the canonical source")
    if backend == "mujoco":
        _exact_mapping(
            collision,
            {
                "allowed_pair_id",
                "condition",
                "pair_collision_enabled",
                "pair_mask_source",
                "compiled_explicit_pair_count",
                "compiled_condim",
                "probe_contype",
                "probe_conaffinity",
                "target_contype",
                "target_conaffinity",
                "ccd_enabled",
            },
            "MuJoCo collision readback",
        )
        if (
            collision.get("condition") != condition
            or collision.get("compiled_explicit_pair_count")
            != (1 if pair_enabled else 0)
            or (pair_enabled and collision.get("compiled_condim") != 1)
            or any(
                collision.get(name) != 0
                for name in (
                    "probe_contype",
                    "probe_conaffinity",
                    "target_contype",
                    "target_conaffinity",
                )
            )
        ):
            raise ContactC0FinalizationError("MuJoCo explicit pair readback differs")
        if private.get("unexpected_pair_observations") != []:
            raise ContactC0FinalizationError("MuJoCo recorded an unexpected pair")
        if overlay.get("source_sha256_before") != overlay.get("source_sha256_after"):
            raise ContactC0FinalizationError("MuJoCo source changed during overlay")
        _require_hex(
            overlay.get("source_sha256_before"),
            "MuJoCo overlay source SHA-256",
            length=64,
        )
        material = _mapping(fixture.get("material"), "MuJoCo material readback")
        _exact_mapping(
            material,
            {
                "friction_coefficient",
                "restitution_coefficient",
                "compiled_pair_friction",
                "compiled_pair_solref",
                "restitution_semantics",
            },
            "MuJoCo material readback",
        )
        pair_friction = material.get("compiled_pair_friction")
        if (
            material.get("friction_coefficient")
            != run_fixture.get("static_friction")
            or material.get("restitution_coefficient")
            != run_fixture.get("restitution")
            or (
                pair_enabled
                and (
                    not isinstance(pair_friction, list)
                    or any(float(component) != 0.0 for component in pair_friction)
                )
            )
            or (not pair_enabled and pair_friction is not None)
        ):
            raise ContactC0FinalizationError("MuJoCo material readback differs")
        if collision.get("ccd_enabled") is not False:
            raise ContactC0FinalizationError("MuJoCo CCD readback is not disabled")
        if runtime.get("device") != "cpu":
            raise ContactC0FinalizationError("MuJoCo runtime device is not CPU")
    else:
        _exact_mapping(
            collision,
            {
                "allowed_pair_id",
                "pair_collision_enabled",
                "pair_mask_source",
                "filtered_pair_targets",
                "probe_collision_enabled",
                "target_collision_enabled",
                "self_collision_enabled",
                "ccd_enabled",
                "ccd_readback_source",
                "ccd_physx_version",
                "ccd_helper_sha256",
                "ccd_scene_prim_path",
                "ccd_body_prim_path",
                "ccd_scene_flags",
                "ccd_body_flags",
                "ccd_scene_mask",
                "ccd_body_mask",
                "ccd_scene_enabled",
                "ccd_body_enabled",
                "ccd_usd_input_source",
                "ccd_usd_scene_enabled",
                "ccd_usd_body_enabled",
            },
            "OV collision readback",
        )
        ccd_helper_sha256 = _require_hex(
            collision.get("ccd_helper_sha256"),
            "OV native CCD helper SHA-256",
            length=64,
        )
        scene_flags = _nonnegative_int(
            collision.get("ccd_scene_flags"),
            "OV PhysX scene flags",
            maximum=0xFFFFFFFF,
        )
        body_flags = _nonnegative_int(
            collision.get("ccd_body_flags"),
            "OV PhysX rigid-body flags",
            maximum=0xFFFFFFFF,
        )
        if (
            overlay.get("first_reset_count_at_readback") != 0
            or collision.get("probe_collision_enabled") is not True
            or collision.get("target_collision_enabled") is not True
            or collision.get("self_collision_enabled") is not False
            or collision.get("ccd_enabled") is not False
            or collision.get("ccd_readback_source")
            != (
                "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_"
                "post_first_reset"
            )
            or collision.get("ccd_physx_version") != "5.9.0"
            or (
                native_helper_sha256 is not None
                and ccd_helper_sha256 != native_helper_sha256
            )
            or not isinstance(collision.get("ccd_scene_prim_path"), str)
            or not str(collision.get("ccd_scene_prim_path")).startswith("/")
            or collision.get("ccd_body_prim_path") != parent_path
            or collision.get("ccd_scene_mask") != 2
            or collision.get("ccd_body_mask") != 4
            or bool(scene_flags & 2)
            or bool(body_flags & 4)
            or collision.get("ccd_scene_enabled") is not False
            or collision.get("ccd_body_enabled") is not False
            or collision.get("ccd_usd_input_source")
            != "post_first_reset_live_composed_usd"
            or collision.get("ccd_usd_scene_enabled") is not False
            or collision.get("ccd_usd_body_enabled") is not False
            or mass.get("synthetic_probe_mass_api_applied") is not False
        ):
            raise ContactC0FinalizationError("OV overlay/collision boundary differs")
        sensor = _mapping(fixture.get("sensor_readback"), "OV sensor readback")
        _exact_mapping(
            sensor,
            {
                "body_prim_path",
                "filter_prim_paths_expr",
                "filter_count",
                "update_period_s",
                "debug_visualization",
                "created_before_first_reset",
                "sample_update_force_recompute",
            },
            "OV sensor readback",
        )
        filters = sensor.get("filter_prim_paths_expr")
        target_path = _nonempty_string(target.get("prim_path"), "OV target path")
        probe_parent_path = _nonempty_string(
            probe.get("parent_prim_path"), "OV probe parent path"
        )
        if (
            not isinstance(filters, list)
            or filters != [target_path]
            or sensor.get("body_prim_path") != probe_parent_path
            or sensor.get("filter_count") != 1
            or sensor.get("created_before_first_reset") is not True
            or sensor.get("sample_update_force_recompute") is not True
            or run_observation.get("filtered_sensor_body_count") != 1
            or run_observation.get("filtered_target_count") != 1
        ):
            raise ContactC0FinalizationError("OV filtered sensor readback differs")
        filtered_pairs = collision.get("filtered_pair_targets")
        if filtered_pairs != ([] if pair_enabled else [target_path]):
            raise ContactC0FinalizationError("OV contact/sham pair mask differs")
        final_inventory = _mapping(
            fixture.get("final_collision_inventory"),
            "OV final collision inventory",
        )
        _exact_mapping(
            final_inventory,
            {"count", "enabled_count", "enabled_paths", "items"},
            "OV final collision inventory",
        )
        final_items = final_inventory.get("items")
        enabled_paths = (
            sorted(
                str(item.get("prim_path"))
                for item in final_items
                if isinstance(item, Mapping)
                and item.get("collision_enabled") is True
            )
            if isinstance(final_items, list)
            else []
        )
        expected_enabled = sorted(
            [
                _nonempty_string(probe.get("prim_path"), "OV probe path"),
                target_path,
            ]
        )
        if (
            not isinstance(final_items, list)
            or final_inventory.get("count") != len(final_items)
            or final_inventory.get("enabled_count") != 2
            or final_inventory.get("enabled_paths") != expected_enabled
            or enabled_paths != expected_enabled
            or run_fixture.get("enabled_collision_shape_count") != 2
        ):
            raise ContactC0FinalizationError(
                "OV final enabled collision inventory differs"
            )
        unexpected = _mapping(
            private.get("unexpected_pair_observation"),
            "OV unexpected-pair evidence",
        )
        _exact_mapping(
            unexpected,
            {"status", "unexpected_pairs"},
            "OV unexpected-pair evidence",
        )
        if (
            unexpected.get("unexpected_pairs") != []
            or unexpected.get("status")
            != "proved_empty_by_exclusive_enabled_shape_inventory"
        ):
            raise ContactC0FinalizationError("OV unexpected-pair evidence differs")
        if (
            mass.get("usd_composed_preserved") is not True
            or mass.get("runtime_verified") is not True
        ):
            raise ContactC0FinalizationError(
                "OV mass properties lack runtime verification"
            )
        _validate_ov_runtime_mass_proof(
            source_value=mass.get("source"),
            runtime_effective_value=mass.get("runtime_effective"),
            runtime_checks_value=mass.get("runtime_checks"),
            expected_body_name=expected_frame,
            expected_body_prim_path=parent_path,
            expected_helper_sha256=ccd_helper_sha256,
        )
        material = _mapping(fixture.get("material"), "OV material readback")
        _exact_mapping(
            material,
            {
                "prim_path",
                "static_friction",
                "dynamic_friction",
                "friction_coefficient",
                "restitution_coefficient",
            },
            "OV material readback",
        )
        if any(
            material.get(name) != run_fixture.get(target_name)
            for name, target_name in (
                ("static_friction", "static_friction"),
                ("dynamic_friction", "dynamic_friction"),
                ("restitution_coefficient", "restitution"),
            )
        ):
            raise ContactC0FinalizationError("OV material readback differs")
        if (
            runtime.get("kitless") is not True
            or runtime.get("renderer") is not False
            or runtime.get("camera") is not False
            or runtime.get("device") != "cuda:0"
            or runtime.get("scene_inventory_verification_phase")
            != "after_full_trajectory_before_cleanup"
            or runtime.get("forbidden_module_verification_phase")
            != "after_runtime_cleanup"
            or runtime.get("forbidden_module_inventory") != []
            or runtime.get("camera_prim_paths") != []
            or runtime.get("created_sensor_types") != ["ContactSensor"]
        ):
            raise ContactC0FinalizationError("OV runtime is not kit-less/headless")
    return {"fixture": fixture, "runtime": runtime}


def _status_value(status: Mapping[str, Any], *names: str) -> object:
    for name in names:
        if name in status:
            return status[name]
    return None


def _validate_remote_ccd_scene_path_consistency(paths: set[str]) -> None:
    if len(paths) != 1:
        raise ContactC0FinalizationError(
            "OV PhysicsScene CCD readback path differs across canonical cases"
        )


def _load_backend_evidence(
    root: Path,
    *,
    backend: Simulator,
    manifest: object,
    source_revision: str,
    source_tree: str,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str | None,
    project_root: Path,
    experimental_campaign: str | None = None,
) -> tuple[list[ContactRun], dict[str, object], set[str]]:
    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise ContactC0FinalizationError('unknown candidate campaign')
    expected_cases = tuple(
        case for case in expand_contact_cases(manifest) if case.simulator is backend
    )
    if (
        len(expected_cases) != EXPECTED_CASES_PER_BACKEND
        or len({case.case_id for case in expected_cases})
        != EXPECTED_CASES_PER_BACKEND
    ):
        raise ContactC0FinalizationError(
            f"{backend.value} manifest matrix is not exactly 16 cases"
        )
    expected_id_list = [case.case_id for case in expected_cases]
    expected_ids = {case.case_id for case in expected_cases}
    log_audit = _validate_backend_topology(
        root, backend=backend, expected_case_ids=expected_ids,
        experimental_campaign=experimental_campaign,
    )
    remote_native_helper: dict[str, str] | None = None
    if backend is Simulator.MUJOCO:
        status, launcher_runtime = _validate_local_launcher_identity(
            root,
            manifest=manifest,
            expected_case_ids=expected_id_list,
            source_revision=source_revision,
            source_tree=source_tree,
            manifest_file_sha256=manifest_file_sha256,
            manifest_semantic_sha256=manifest_semantic_sha256,
            project_root=project_root,
            experimental_campaign=experimental_campaign,
        )
    else:
        if remote_snapshot_sha256 is None:
            raise ContactC0FinalizationError("remote snapshot identity is missing")
        status, launcher_runtime, remote_native_helper = _validate_remote_launcher_identity(
            root,
            manifest=manifest,
            expected_case_ids=expected_id_list,
            source_revision=source_revision,
            source_tree=source_tree,
            source_archive_sha256=source_archive_sha256,
            remote_snapshot_sha256=remote_snapshot_sha256,
            manifest_file_sha256=manifest_file_sha256,
            manifest_semantic_sha256=manifest_semantic_sha256,
            project_root=project_root,
            experimental_campaign=experimental_campaign,
        )
        remote_provenance = read_json_strict(root / "launcher" / "provenance.json")
        remote_gpu = _mapping(
            _mapping(remote_provenance, "OVPhysX launcher provenance").get(
                "selected_gpu"
            ),
            "OVPhysX selected GPU",
        )
    session_id = str(status["session_id"])
    cases_root = _resolved_directory(root / "cases", f"{backend.value} cases")
    runs: list[ContactRun] = []
    process_hashes: set[str] = set()
    remote_ccd_scene_paths: set[str] = set()
    asset_hash = manifest.provenance.canonical_lf_asset_tree_sha256
    worker_hash = _file_sha256(
        project_root / "src" / "wave_asset_qa" / "contact" / "worker.py"
    )
    for case in expected_cases:
        case_root = cases_root / case.case_id
        process, fresh_hash = _validate_process_record(
            read_json_strict(case_root / "private-process.json"),
            case_id=case.case_id,
            backend=backend.value,
            session_id=session_id,
            source_revision=source_revision,
            source_tree=source_tree,
            manifest_file_sha256=manifest_file_sha256,
            manifest_semantic_sha256=manifest_semantic_sha256,
            asset_tree_sha256=asset_hash,
            worker_module_sha256=worker_hash,
            experimental_campaign=experimental_campaign,
        )
        if fresh_hash in process_hashes:
            raise ContactC0FinalizationError(f"{backend.value} reused a worker process")
        if backend is Simulator.OVPHYSX:
            assert remote_native_helper is not None
            _validate_remote_case_security_evidence(
                case_root,
                case_id=case.case_id,
                run_id=str(status["run_id"]),
                session_id=session_id,
                source_revision=source_revision,
                source_tree=source_tree,
                asset_tree_sha256=asset_hash,
                gpu_index=int(remote_gpu["query_index" if experimental_campaign else "physical_index"]),
                gpu_uuid=str(remote_gpu["uuid"]),
                gpu_name=str(remote_gpu["name"]),
                driver_version=str(remote_gpu["driver_version"]),
                native_helper_path=remote_native_helper["path"],
                native_helper_sha256=remote_native_helper["sha256"],
                process=process,
                fresh_process_identity_sha256=fresh_hash,
                experimental_campaign=experimental_campaign,
            )
        run = load_and_validate_contact_run(
            case_root / f"{case.case_id}.run.json",
            manifest=manifest,
            case=case,
            source_revision=source_revision,
            source_tree=source_tree,
            fresh_process_identity_sha256=fresh_hash,
        )
        adapter = _validate_private_adapter(
            read_json_strict(case_root / "private-adapter-evidence.json"),
            run,
            condition=case.condition.value,
            native_helper_sha256=(
                remote_native_helper["sha256"]
                if remote_native_helper is not None
                else None
            ),
            experimental_campaign=experimental_campaign,
        )
        if experimental_campaign:
            from wave_asset_qa.contact.async_campaign import load_candidate_case
            load_candidate_case(case_root,manifest=manifest,case=case,
                                source_revision=source_revision,source_tree=source_tree)
        runtime = adapter["runtime"]
        if backend is Simulator.OVPHYSX:
            adapter_collision = _mapping(
                adapter["fixture"].get("collision"), "OV collision readback"
            )
            remote_ccd_scene_paths.add(str(adapter_collision["ccd_scene_prim_path"]))
        expected_python = (
            launcher_runtime.get("python_version")
            if backend is Simulator.MUJOCO
            else launcher_runtime.get("python")
        )
        if (
            process.get("python_version") != expected_python
            or runtime.get("python_version") != expected_python
        ):
            raise ContactC0FinalizationError(
                f"{case.case_id} process/adapter Python identity differs from launcher"
            )
        if backend is Simulator.MUJOCO and runtime.get(
            "backend_version"
        ) != launcher_runtime.get("mujoco_version"):
            raise ContactC0FinalizationError(
                f"{case.case_id} MuJoCo runtime version differs from launcher"
            )
        if backend is Simulator.OVPHYSX and runtime.get("dt_s") != case.dt_s:
            raise ContactC0FinalizationError(
                f"{case.case_id} OV runtime timestep differs from case"
            )
        process_hashes.add(fresh_hash)
        runs.append(run)
    if backend is Simulator.OVPHYSX:
        _validate_remote_ccd_scene_path_consistency(remote_ccd_scene_paths)
    return (
        runs,
        {
            "session_id": session_id,
            "status": dict(status),
            "runtime_log_audit": log_audit,
        },
        process_hashes,
    )


def _copy_tree_strict(source: Path, destination: Path) -> None:
    regular_tree_records(source)
    shutil.copytree(source, destination, symlinks=False)
    if regular_tree_records(source) != regular_tree_records(destination):
        raise ContactC0FinalizationError("copied evidence tree differs from source")


def _validate_existing_gate0(project_root: Path) -> dict[str, object]:
    gate0 = _mapping(
        read_json_strict(project_root / "results" / "gate0" / "summary.json"),
        "existing Gate 0 summary",
    )
    gate0_status = _mapping(gate0.get("status"), "existing Gate 0 status")
    waveform = _mapping(
        read_json_strict(
            project_root / "results" / "waveform-t1" / "summary.json"
        ),
        "existing Trajectory T1 summary",
    )
    waveform_gate0 = _mapping(
        waveform.get("formal_gate0"), "existing Trajectory T1 formal Gate 0"
    )
    if (
        str(gate0_status.get("comparison_status")).upper() != "DIVERGENT"
        or gate0_status.get("pass_ready") is not False
        or waveform.get("scientific_label") != "DIVERGENT"
        or waveform_gate0.get("comparison_status") != "DIVERGENT"
        or waveform_gate0.get("pass_ready") is not False
        or waveform_gate0.get("unchanged") is not True
    ):
        raise ContactC0FinalizationError(
            "existing Gate 0/T1 no longer states DIVERGENT and pass_ready=false"
        )
    return {
        "trajectory_t1_science_status": "DIVERGENT",
        "pass_ready": False,
        "unchanged_by_contact_c0": True,
        "gate0_summary_sha256": _file_sha256(
            project_root / "results" / "gate0" / "summary.json"
        ),
        "waveform_t1_summary_sha256": _file_sha256(
            project_root / "results" / "waveform-t1" / "summary.json"
        ),
    }


def _audit_public(summary: Mapping[str, Any], markdown: str, sessions: Sequence[str]) -> None:
    serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
    combined = serialized + "\n" + markdown
    forbidden = [
        "selected_pair_force_norm_n",
        "native_contact_observation",
        "private-process",
        "private-adapter",
        "gpu_uuid",
        "cuda_visible_devices",
        "camera_prim_paths",
        "created_sensor_types",
        "scene_inventory_verification_phase",
        "forbidden_module_verification_phase",
        "forbidden_module_inventory",
        "gpu6.example.invalid",
        "exampleuser@",
        "/data/home/",
        "/mnt/ceph2",
        "c:\\users\\",
    ]
    forbidden.extend(sessions)
    lowered = combined.lower()
    leaked = [token for token in forbidden if token.lower() in lowered]
    if leaked:
        raise ContactC0FinalizationError(
            "public projection contains private material: " + ", ".join(leaked)
        )
    boundary = _mapping(summary.get("claim_boundary"), "public claim boundary")
    if (
        boundary.get("hardware_in_scope") is not False
        or boundary.get("sim2real_claimed") is not False
        or boundary.get("native_hand_collision_geometry_validated") is not False
        or boundary.get("dual_hand_interaction_in_scope") is not False
        or boundary.get("floating_base_in_scope") is not False
        or boundary.get("full_isaac_sim_in_scope") is not False
        or boundary.get("synthetic_fixture_only") is not True
    ):
        raise ContactC0FinalizationError("public claim boundary expanded beyond C0")
    existing = _mapping(summary.get("existing_gate0"), "public existing Gate 0")
    if existing != {
        "trajectory_t1_science_status": "DIVERGENT",
        "pass_ready": False,
        "unchanged_by_contact_c0": True,
    }:
        raise ContactC0FinalizationError("public existing Gate 0 status changed")


def finalize_contact_c0(
    *,
    manifest_path: str | Path,
    local_evidence_root: str | Path,
    remote_evidence_root: str | Path,
    output_dir: str | Path,
    public_output_dir: str | Path,
    source_revision: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str,
    experimental_campaign: str | None = None,
) -> dict[str, object]:
    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise ContactC0FinalizationError('unknown candidate campaign')
    project_root = Path(__file__).resolve(strict=True).parents[1]
    revision = _require_hex(source_revision, "source revision", length=40)
    archive_hash = _require_hex(
        source_archive_sha256, "source archive SHA-256", length=64
    )
    snapshot_hash = _require_hex(
        remote_snapshot_sha256, "remote snapshot SHA-256", length=64
    )
    source_tree = _validate_source(project_root, revision)
    archive_bytes = _git_archive_bytes(project_root, revision)
    if sha256(archive_bytes).hexdigest() != archive_hash:
        raise ContactC0FinalizationError("source archive is not the exact commit archive")
    if _virtual_snapshot_sha256(
        archive_bytes,
        source_revision=revision,
        source_tree=source_tree,
        archive_sha256=archive_hash,
    ) != snapshot_hash:
        raise ContactC0FinalizationError("remote snapshot is not the exact source archive")

    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    expected_manifest = (
        project_root / "configs" / "parity" / "contact_c0.json"
    ).resolve(strict=True)
    if manifest_file != expected_manifest or is_link_like(manifest_file):
        raise ContactC0FinalizationError("finalizer requires the checked-in C0 manifest")
    manifest = load_contact_manifest(manifest_file)
    manifest_file_hash = sha256(manifest_file.read_bytes()).hexdigest()
    manifest_semantic_hash = canonical_json_sha256(manifest.to_dict())
    existing_gate0 = _validate_existing_gate0(project_root)

    local_root = _resolved_directory(local_evidence_root, "local evidence")
    remote_root = _resolved_directory(remote_evidence_root, "remote evidence")
    destination = _new_directory_path(output_dir, "private bundle output")
    public_destination = _new_directory_path(public_output_dir, "public output")
    _validate_disjoint(
        (
            (local_root, "local evidence"),
            (remote_root, "remote evidence"),
            (destination, "private bundle output"),
            (public_destination, "public output"),
        )
    )
    scope = experimental_campaign or 'contact-c0'
    local_inventory = _validate_evidence_manifest(local_root, scope + "-mujoco")
    remote_inventory = _validate_evidence_manifest(remote_root, scope + "-ovphysx")

    local_runs, local_meta, local_processes = _load_backend_evidence(
        local_root,
        backend=Simulator.MUJOCO,
        manifest=manifest,
        source_revision=revision,
        source_tree=source_tree,
        manifest_file_sha256=manifest_file_hash,
        manifest_semantic_sha256=manifest_semantic_hash,
        source_archive_sha256=archive_hash,
        remote_snapshot_sha256=None,
        project_root=project_root,
        experimental_campaign=experimental_campaign,
    )
    remote_runs, remote_meta, remote_processes = _load_backend_evidence(
        remote_root,
        backend=Simulator.OVPHYSX,
        manifest=manifest,
        source_revision=revision,
        source_tree=source_tree,
        manifest_file_sha256=manifest_file_hash,
        manifest_semantic_sha256=manifest_semantic_hash,
        source_archive_sha256=archive_hash,
        remote_snapshot_sha256=snapshot_hash,
        project_root=project_root,
        experimental_campaign=experimental_campaign,
    )
    if local_processes & remote_processes or len(local_processes | remote_processes) != 32:
        raise ContactC0FinalizationError("32 globally unique fresh processes were not proven")
    result = evaluate_contact_c0(manifest, (*local_runs, *remote_runs))
    if result.evidence_status is not EvidenceStatus.VALID:
        raise ContactC0FinalizationError(
            "C0 evidence is INVALID: " + "; ".join(result.invalid_reasons)
        )

    partial = destination.parent / f".{destination.name}.partial.{os.getpid()}"
    public_partial = public_destination.parent / (
        f".{public_destination.name}.partial.{os.getpid()}"
    )
    for path in (partial, public_partial):
        if path.exists() or is_link_like(path):
            raise FileExistsError(f"finalizer staging path already exists: {path}")
    partial.mkdir(mode=0o700)
    (partial / "raw").mkdir()
    (partial / "analysis").mkdir()
    (partial / "public").mkdir()
    _copy_tree_strict(local_root, partial / "raw" / "mujoco")
    _copy_tree_strict(remote_root, partial / "raw" / "ovphysx")
    write_json_exclusive(partial / "analysis" / "gate-result.json", result.to_dict())
    source_identity = {
        "schema_version": 1,
        "source_revision": revision,
        "source_tree": source_tree,
        "source_archive_sha256": archive_hash,
        "remote_snapshot_sha256": snapshot_hash,
        "manifest_file_sha256": manifest_file_hash,
        "manifest_semantic_sha256": manifest_semantic_hash,
        "local_evidence": local_inventory,
        "remote_evidence": remote_inventory,
        "fresh_process_count": 32,
        "existing_gate0": existing_gate0,
        "runtime_log_gate": {
            "fatal_pattern_set_version": 1,
            "passed": True,
            "records": [
                *local_meta["runtime_log_audit"],
                *remote_meta["runtime_log_audit"],
            ],
        },
    }
    write_json_exclusive(
        partial / "analysis" / "source-and-evidence-identity.json", source_identity
    )

    summary = public_summary(result)
    if experimental_campaign:
        summary['experimental_campaign'] = experimental_campaign
        summary['formal_c0_replacement'] = False
        summary['intervention'] = 'diagnostic_async_wait_v1'
    summary["existing_gate0"] = {
        "trajectory_t1_science_status": existing_gate0[
            "trajectory_t1_science_status"
        ],
        "pass_ready": existing_gate0["pass_ready"],
        "unchanged_by_contact_c0": existing_gate0["unchanged_by_contact_c0"],
    }
    markdown = render_contact_markdown(summary)
    if experimental_campaign:
        markdown = ('# Experimental async + wait C0 comparison\n\n'
                    'This is a separately collected intervention campaign, not the original C0. '
                    'The original C0 VALID / INCONCLUSIVE result remains unchanged.\n\n') + markdown
    markdown += (
        "\n## Relationship to the existing Gate 0\n\n"
        "Contacts C0 is additive. The existing Trajectory T1 result remains "
        "`DIVERGENT` with `pass_ready=false`; C0 does not reinterpret it.\n"
    )
    sessions = (str(local_meta["session_id"]), str(remote_meta["session_id"]))
    _audit_public(summary, markdown, sessions)
    write_json_exclusive(partial / "public" / "summary.json", summary)
    report_path = partial / "public" / "report.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
        handle.flush()
        os.fsync(handle.fileno())

    bundle_records = list(regular_tree_records(partial, exclude=("bundle-manifest.json",)))
    bundle_root = inventory_root_sha256(bundle_records)
    write_json_exclusive(
        partial / "bundle-manifest.json",
        {
            "schema_version": 1,
            "scope": scope + "-private-bundle",
            "records": bundle_records,
            "root_sha256": bundle_root,
        },
    )
    if list(regular_tree_records(partial, exclude=("bundle-manifest.json",))) != bundle_records:
        raise ContactC0FinalizationError("private bundle changed while sealing")

    public_partial.mkdir(mode=0o700)
    shutil.copy2(partial / "public" / "summary.json", public_partial / "summary.json")
    shutil.copy2(partial / "public" / "report.md", public_partial / "report.md")
    if {path.name for path in public_partial.iterdir()} != {"summary.json", "report.md"}:
        raise ContactC0FinalizationError("public tree is not the exact two-file projection")
    if (
        (public_partial / "summary.json").read_bytes()
        != (partial / "public" / "summary.json").read_bytes()
        or (public_partial / "report.md").read_bytes()
        != (partial / "public" / "report.md").read_bytes()
    ):
        raise ContactC0FinalizationError("public projection differs from private bundle copy")

    partial.rename(destination)
    public_partial.rename(public_destination)
    return {
        "status": "finalized",
        "evidence_status": result.evidence_status.value,
        "science_status": result.science_status.value,
        "pass_ready": bool(summary["pass_ready"]),
        "completed_case_count": result.completed_run_count,
        "joint_mapping": summary["mapping"]["joint_mapping"],
        "distal_frame_mapping": summary["mapping"]["distal_frame_mapping"],
        "bundle_root_sha256": bundle_root,
        "public_file_count": 2,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-evidence-root", type=Path, required=True)
    parser.add_argument("--remote-evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--remote-snapshot-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = finalize_contact_c0(
            manifest_path=args.manifest,
            local_evidence_root=args.local_evidence_root,
            remote_evidence_root=args.remote_evidence_root,
            output_dir=args.output_dir,
            public_output_dir=args.public_output_dir,
            source_revision=args.source_revision,
            source_archive_sha256=args.source_archive_sha256,
            remote_snapshot_sha256=args.remote_snapshot_sha256,
        )
    except Exception as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    raise SystemExit(main())
