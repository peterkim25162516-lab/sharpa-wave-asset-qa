#!/usr/bin/env python3
"""Fail-closed local finalization for the 24-case OVPhysX Freeze B study.

The finalizer never executes a simulator.  It verifies the preregistered
private/public contracts, the immutable A->B->C->D->E->F->G campaign and
G->H->I analysis-only source transitions, the frozen R1 and formal Gate 0
bundles, and the complete local/remote evidence trees.  A scientific label is
assigned only after every validity gate passes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
from typing import Any

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.parity.bundle import sha256_file, write_bundle_manifest, write_json_atomic
from wave_asset_qa.parity.compare import (
    MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S,
    MAX_ABS_CANONICAL_JOINT_POSITION_RAD,
    MAX_ABS_CANONICAL_POSITION_TARGET_RAD,
    MAX_FRAME_ORIGIN_DISTANCE_M,
    UNIT_QUATERNION_NORM_TOLERANCE,
    CollectedRun,
    trace_delta,
)
from wave_asset_qa.parity.contracts import HandSide, Simulator
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    bundle_payload_paths,
    create_staging_root,
    is_link_like,
    promote_staging_root,
    validated_results_output_path,
    verify_copied_bundle_subset,
    verify_exact_bundle,
    validate_diagnostic_run,
    write_text_exclusive,
)
from wave_asset_qa.parity.effective_readback import _solver_runtime
from wave_asset_qa.parity.runner import adapter_run_result_from_dict, load_collected_runs
from wave_asset_qa.parity.scenarios import (
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)
from wave_asset_qa.parity.sensitivity import (
    FreezeBDecision,
    SensitivityStatus,
    SensitivityTraceSet,
    TIME_GRID_ABS_TOLERANCE_S,
    canonical_frame_names,
    canonical_json_sha256,
    canonical_joint_names,
    evaluate_freeze_b,
    load_private_plan,
    load_public_protocol,
    private_plan_sha256,
)


PROTOCOL_ID = "ovphysx-legacy-joint-friction-freeze-b-v1"
PUBLIC_PROTOCOL_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b.json"
)
CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_corrigendum.json"
)
ADMISSION_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_admission_corrigendum.json"
)
ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_asset_preflight_corrigendum.json"
)
RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_runtime_order_corrigendum.json"
)
TIME_GRID_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_time_grid_corrigendum.json"
)
TIME_GRID_CORRIGENDUM_CANONICAL_SHA256 = (
    "5d6ea66579508d255b17f26a118433d4c0f01e078d14b48d1baa9b1b776bbf20"
)
GATE0_MANIFEST_RELATIVE_PATH = Path("configs/parity/gate0.json")
EXPECTED_R1_STATUS = "READBACK_VALID"
EXPECTED_REMOTE_ENVIRONMENT = {
    "python": "3.12.14",
    "isaaclab": "6.1.14",
    "isaaclab-ovphysx": "3.0.2",
    "ovphysx": "0.4.13",
    "torch": "2.10.0+cu128",
    "usd-core": "25.11",
    "kit": "false",
    "renderer": "false",
    "camera": "false",
}
EXPECTED_DRIVER_VERSION = "570.158.01"
EXPECTED_GPU_NAME_FRAGMENT = "A800-SXM4-40GB"
EXPECTED_REMOTE_HOSTNAME = "example-gpu-node-6"
REQUIRED_SOURCE_PATHS = (
    "configs/parity/gate0.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b_corrigendum.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b_admission_corrigendum.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b_asset_preflight_corrigendum.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b_runtime_order_corrigendum.json",
    "configs/parity/ovphysx_legacy_friction_freeze_b_time_grid_corrigendum.json",
    "scripts/finalize_ovphysx_freeze_b.py",
    "scripts/prepare_ovphysx_freeze_b.py",
    "scripts/probe_ovphysx_freeze_b.py",
    "scripts/run_freeze_b_mujoco_worker.py",
    "scripts/run_ovphysx_freeze_b_local.py",
    "scripts/run_ovphysx_freeze_b_remote.sh",
    "src/wave_asset_qa/adapters/ovphysx.py",
    "src/wave_asset_qa/parity/runner.py",
    "src/wave_asset_qa/parity/process_identity.py",
    "src/wave_asset_qa/parity/sensitivity.py",
    "tests/test_freeze_b_finalizer.py",
    "tests/test_freeze_b_sensitivity.py",
)

PREREGISTRATION_REVISION = "608f83f1f93042d713346bd102de09b0b999b588"
PREREGISTRATION_TREE = "1d1467d9f156d1115dbcb8ff54d522f0189454cc"
PUBLIC_PROTOCOL_BLOB_OID = "c716ce9a4496d7e26384aa94b25437b8d5f0ce08"
PUBLIC_PROTOCOL_FILE_SHA256 = (
    "9967098e45e063e2a4ca6e3db8644219f55dab808e99723c892fa72d53dc79c7"
)
PUBLIC_PROTOCOL_CANONICAL_SHA256 = (
    "de2a97ac4d73d95f564f730a5655ddf26e9b01426eefff30c73135d932cdf66b"
)
PRIVATE_PLAN_SHA256 = (
    "b4416b6869bdae90743ace253212ef43f7e592e30a93093f470c648261bab1ea"
)
PRIVATE_PLAN_FILE_SHA256 = (
    "6017453e9e13a40f64cc53a4ad933944868b8da4540f646b2986e7138ae5c519"
)
IMPLEMENTATION_TO_PREREGISTRATION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b.json"
]
PREREGISTRATION_TO_CORRECTION_DIFF = [
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "A\tscripts/run_freeze_b_mujoco_worker.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "A\tsrc/wave_asset_qa/parity/process_identity.py",
    "M\ttests/test_freeze_b_finalizer.py",
    "A\ttests/test_freeze_b_mujoco_worker.py",
    "M\ttests/test_freeze_b_preparation_local.py",
    "A\ttests/test_process_identity.py",
]
CORRECTION_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_corrigendum.json"
]
FIRST_EXECUTION_REVISION = "8bb84691c82ab0582254d21953630725d64e9395"
FIRST_EXECUTION_TREE = "6052d16ddb4f66c93046a170f44a575dfe4e385e"
FIRST_CORRIGENDUM_BLOB_OID = "0ec18c5738aee6f0654ccbfcdb5105aeae24e4ed"
FIRST_CORRIGENDUM_FILE_SHA256 = (
    "41eba7988668c7396b042d5785dfe1b627709f5df46ac77aa94eda77d0b5595b"
)
FIRST_CORRIGENDUM_CANONICAL_SHA256 = (
    "ccf01aebaca218f50f9adce9250f150d7d6e00e88bca8bd42db694c55808eff5"
)
FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF = [
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "M\tscripts/run_ovphysx_freeze_b_remote.sh",
    "M\ttests/test_freeze_b_finalizer.py",
    "M\ttests/test_freeze_b_preparation_local.py",
    "M\ttests/test_freeze_b_remote.py",
]
ADMISSION_CORRECTION_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_admission_corrigendum.json"
]
ADMISSION_CORRECTION_REVISION = "4156bd4cc0df996b6fab1be0e48fdbb04a65f393"
ADMISSION_CORRECTION_TREE = "c5966f06ef32672f490e9c40746c15e92fa96761"
SECOND_EXECUTION_REVISION = "554e59a45a0b9596ca70ad677ba2aba858002536"
SECOND_EXECUTION_TREE = "0335c0e5046025fae27311b0aa1b84003d811939"
ADMISSION_CORRIGENDUM_BLOB_OID = "a7e50ece608fb1aa3eadb2f88bc472204eed5d9f"
ADMISSION_CORRIGENDUM_FILE_SHA256 = (
    "6651af497d3ff1451153e3f6f3b1a03c977ded9879e5fc6389fc1b3a41c7172e"
)
ADMISSION_CORRIGENDUM_CANONICAL_SHA256 = (
    "fccba33068aaa6d06e8269be80a3c507b95a94384272803e59e61111293237ee"
)
INCIDENT_SOURCE_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_asset_preflight_corrigendum.json",
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "M\ttests/test_freeze_b_finalizer.py",
    "M\ttests/test_freeze_b_preparation_local.py",
]
CAMPAIGN_EXECUTION_REVISION = "d890f8ee83741cf643aadf2d9d1f256bf41c09cc"
CAMPAIGN_EXECUTION_TREE = "29b6d6846e181897c4be2c3e19361e061524b061"
CAMPAIGN_SOURCE_ARCHIVE_SHA256 = (
    "b019ba666ffa0d6882915a81203a446503f4cd0e955aea48225e9d645812359d"
)
REMOTE_SOURCE_SNAPSHOT_SHA256 = (
    "202c06828a76cec02a62ca5240fc0483f011b155caad29db541050a4f26a2743"
)
ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID = (
    "205acc39d041ae9017edf94e4ed901dd842651d7"
)
ASSET_PREFLIGHT_CORRIGENDUM_FILE_SHA256 = (
    "ca95f4acac13b2a0743530d1a97bfa18e7570260fc6a36850be8c80920e89884"
)
ASSET_PREFLIGHT_CORRIGENDUM_CANONICAL_SHA256 = (
    "d538a94a385c86c141730a7d83c10d435902ea604c9c203cc617f06dec660d1e"
)
CAMPAIGN_TO_ANALYSIS_CORRECTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_runtime_order_corrigendum.json",
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\ttests/test_freeze_b_finalizer.py",
]
FIRST_ANALYSIS_REVISION = "ba2fb4cb881bda82ac5469ef2103cdf4e2eb0eea"
FIRST_ANALYSIS_TREE = "a6e38d493d1a3057617436e0e5b8ef0101007f31"
RUNTIME_ORDER_CORRIGENDUM_BLOB_OID = (
    "7d7a5acc04113b3fbf276eb7e7132f8551c8ae34"
)
RUNTIME_ORDER_CORRIGENDUM_FILE_SHA256 = (
    "531addc277e62ac59e9ea84d8af3475d8782e8e8c1160a4f63e103e2d07a55ea"
)
RUNTIME_ORDER_CORRIGENDUM_CANONICAL_SHA256 = (
    "ae275f12c328f18f0d4461a45d4a46fffc32be768d2cc715e3e46bf822318bad"
)
ANALYSIS_TO_TIME_GRID_CORRECTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_time_grid_corrigendum.json",
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\tsrc/wave_asset_qa/parity/sensitivity.py",
    "M\ttests/test_freeze_b_finalizer.py",
    "M\ttests/test_freeze_b_sensitivity.py",
]
SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY: dict[str, object] = {
    "classification": "local_unpublished_superseded_analysis_bundle",
    "analysis_revision": FIRST_ANALYSIS_REVISION,
    "analysis_tree": FIRST_ANALYSIS_TREE,
    "root_sha256": (
        "55375eab768452da6988e644c3cb0f4c2193c1cd8f499981afe09f777cb05614"
    ),
    "file_count": 398,
    "total_bytes": 92_868_012,
    "bundle_json_sha256": (
        "50d0aeb2cb666ff14bb3011a035d529735414f15805e4c97faf2912686bced8b"
    ),
    "private_decision_sha256": (
        "688c2b9651213ad6523a4622b28b6c7833e03f4247a808823cbf34cb8e0a0575"
    ),
    "public_summary_sha256": (
        "059b3c620b8f804020030f5d41863d923ef18e56860225b8bc3103c60f94d4b4"
    ),
    "public_report_sha256": (
        "1e46a69a79983374ba11002dd06c19bf1d05b4cbe4f65a289a1f88c57b51bd85"
    ),
    "private_finalization_sha256": (
        "cd8e66e73bc61110251c368d7776da26e4674af95959d6620425f7d42d32b39e"
    ),
    "private_source_identity_sha256": (
        "5cf0b5b788377c64a0431a43ed2feaec31d765f02eeea3bfaa3c16586e30e8d5"
    ),
    "full_tree_entry_count": 492,
    "full_tree_file_count": 399,
    "full_tree_total_bytes": 92_958_145,
    "full_tree_root_sha256": (
        "45709ef95f791ffcd8489217fa64d35268f8a43a0908eab435d27a7c2a3e03db"
    ),
    "public_pair_file_count": 2,
    "public_pair_total_bytes": 11_740,
    "public_pair_root_sha256": (
        "6410ff9bd061d9502e5ffe8438530ae4b77571a3915e52bd93e73ebf5299fb46"
    ),
    "validation_status": "INVALID",
    "scientific_label": None,
    "published": False,
    "reuse_as_final_result_allowed": False,
}
SUPERSEDED_ANALYSIS_SUBSET_PATHS = (
    "bundle.json",
    "private/decision.json",
    "private/finalization.json",
    "private/source-identity.json",
    "public/results/freeze-b/summary.json",
    "public/results/freeze-b/report.md",
)
ACTIVE_EVIDENCE_IDENTITIES: dict[str, dict[str, object]] = {
    "local-a4": {
        "classification": "active_campaign_local_evidence",
        "case_count": 8,
        "entry_count": 82,
        "file_count": 72,
        "total_bytes": 14_855_295,
        "root_sha256": (
            "cae24f6532e79fa1ac33e606fe9d034d86482543513376904fcfb7cc4a17c9b8"
        ),
        "source_revision": CAMPAIGN_EXECUTION_REVISION,
        "source_tree": CAMPAIGN_EXECUTION_TREE,
    },
    "remote-a4": {
        "classification": "active_campaign_remote_evidence",
        "case_count": 16,
        "entry_count": 171,
        "file_count": 153,
        "total_bytes": 28_771_413,
        "root_sha256": (
            "5800da811451ac5162ddce95aa17817ed52ce5cff9e5a564dfb002fe13a1f6b7"
        ),
        "source_revision": CAMPAIGN_EXECUTION_REVISION,
        "source_tree": CAMPAIGN_EXECUTION_TREE,
        "runtime_joint_count_per_case": 22,
        "runtime_unique_joint_name_count_per_case": 22,
        "backend_joint_order_canonical_set_match_case_count": 16,
        "records_aligned_to_backend_joint_order_case_count": 16,
    },
}
ADMISSION_EXCLUDED_EVIDENCE_LABELS = ("local-a1", "local-a2", "remote-a2")
ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES: dict[str, dict[str, object]] = {
    "local-a1": {
        "classification": "invalid_local_worker_identity_collection",
        "entry_count": 13,
        "file_count": 10,
        "total_bytes": 1_282_435,
        "root_sha256": "fb661580d635ae84fd6bf5d7eaa33e838c96e13b8d1b5c242ea9fedc1be10498",
    },
    "local-a2": {
        "classification": "excluded_structurally_valid_local_campaign_evidence",
        "entry_count": 82,
        "file_count": 72,
        "total_bytes": 14_852_439,
        "root_sha256": "998a75cf133b0baac0f1d5c2baf53469fa2d6d955bb3c277c7c621712accec1b",
    },
    "remote-a2": {
        "classification": "invalid_remote_payload_admission_collection",
        "entry_count": 20,
        "file_count": 17,
        "total_bytes": 1_245_586,
        "root_sha256": "155fdebc6515aa49a7f5e0aafbacb629cf348829028b2da4fedcc5a1603d3abe",
    },
}
EXCLUDED_EVIDENCE_LABELS = (
    "local-a1",
    "local-a2",
    "remote-a2",
    "local-a3",
)
EXCLUDED_EVIDENCE_IDENTITIES: dict[str, dict[str, object]] = {
    **ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES,
    "local-a3": {
        "classification": "failed_pre_simulation_asset_tree_preflight",
        "entry_count": 16,
        "file_count": 13,
        "total_bytes": 36_508,
        "root_sha256": "bff7ab75459f9ed7b8ab6b0bfd44997c7f544ec72f3ce8ba5644c0996a30eb79",
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_INVENTORY_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_ABSOLUTE_WINDOWS_RE = re.compile(r"(?:^|[\s\"'])[A-Za-z]:[\\/]")
_PUBLIC_FORBIDDEN_KEY_PARTS = (
    "session_id",
    "run_id",
    "gpu_uuid",
    "gpu_index",
    "selected_gpu",
    "worker_pid",
    "fresh_process_id",
    "os_process_identity",
    "ownership_token",
    "hostname",
    "joint_prim_path",
    "expected_pre_value",
    "sham_write_value",
    "zero_write_value",
    "per_joint",
)


class FreezeBFinalizationError(RuntimeError):
    """An input identity, evidence topology, or packaging invariant failed."""


@dataclass(frozen=True, slots=True)
class DeltaMetrics:
    joint_max_abs_rad: float
    frame_position_max_m: float
    frame_orientation_max_rad: float

    @classmethod
    def between(cls, reference: CollectedRun, candidate: CollectedRun, hand: object) -> "DeltaMetrics":
        values = trace_delta(reference.result, candidate.result, hand)
        return cls(*(float(value) for value in values))

    @classmethod
    def maximum(cls, values: Sequence["DeltaMetrics"]) -> "DeltaMetrics":
        if not values:
            return cls(0.0, 0.0, 0.0)
        return cls(
            max(item.joint_max_abs_rad for item in values),
            max(item.frame_position_max_m for item in values),
            max(item.frame_orientation_max_rad for item in values),
        )

    def within(self, *, joint: float, position: float, orientation: float) -> bool:
        return (
            self.joint_max_abs_rad <= joint
            and self.frame_position_max_m <= position
            and self.frame_orientation_max_rad <= orientation
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "joint_max_abs_rad": self.joint_max_abs_rad,
            "frame_position_max_m": self.frame_position_max_m,
            "frame_orientation_max_rad": self.frame_orientation_max_rad,
        }


def _reject_constant(token: str) -> object:
    raise FreezeBFinalizationError(f"non-finite JSON constant is forbidden: {token}")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeBFinalizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeBFinalizationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeBFinalizationError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FreezeBFinalizationError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FreezeBFinalizationError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FreezeBFinalizationError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeBFinalizationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FreezeBFinalizationError(f"{label} must be a finite number")
    return result


def _float32_hex(value: object, label: str) -> str:
    numeric = _finite(value, label)
    try:
        packed = struct.pack("!f", numeric)
        rounded = struct.unpack("!f", packed)[0]
    except (OverflowError, struct.error) as exc:
        raise FreezeBFinalizationError(f"{label} is not finite float32") from exc
    if not math.isfinite(rounded):
        raise FreezeBFinalizationError(f"{label} is not finite float32")
    return packed.hex()


def _resolved_real(path: str | Path, label: str, *, directory: bool) -> Path:
    raw = Path(path).expanduser()
    if is_link_like(raw):
        raise FreezeBFinalizationError(f"{label} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise FreezeBFinalizationError(f"{label} does not exist") from exc
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        kind = "directory" if directory else "file"
        raise FreezeBFinalizationError(f"{label} must be a regular {kind}")
    return resolved


def _validate_regular_tree(root: Path, label: str) -> None:
    for path in root.rglob("*"):
        if is_link_like(path):
            raise FreezeBFinalizationError(
                f"{label} contains a symbolic link: {path.relative_to(root).as_posix()}"
            )
        mode = path.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise FreezeBFinalizationError(
                f"{label} contains a non-regular entry: {path.relative_to(root).as_posix()}"
            )


def _copy_tree_strict(source: Path, destination: Path, label: str) -> None:
    _validate_regular_tree(source, label)
    destination.mkdir()
    for item in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()):
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with item.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())


def _tree_fingerprint(root: Path, label: str) -> dict[str, object]:
    """Bind one input tree's exact regular-entry topology, sizes, and bytes."""

    _validate_regular_tree(root, label)
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            records.append({"path": relative, "kind": "directory"})
        else:
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": path.stat(follow_symlinks=False).st_size,
                    "sha256": sha256_file(path),
                }
            )
    files = [record for record in records if record["kind"] == "file"]
    return {
        "entry_count": len(records),
        "file_count": len(files),
        "total_bytes": sum(int(record["size"]) for record in files),
        "root_sha256": canonical_json_sha256(records),
    }


def _validate_excluded_evidence_fingerprints(
    excluded_evidence: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Bind every excluded tree without deserializing any trajectory payload."""

    if set(excluded_evidence) != set(EXCLUDED_EVIDENCE_LABELS):
        raise FreezeBFinalizationError(
            "excluded evidence labels must be exactly local-a1, local-a2, "
            "remote-a2, local-a3"
        )
    observed: dict[str, dict[str, object]] = {}
    for label in EXCLUDED_EVIDENCE_LABELS:
        fingerprint = _tree_fingerprint(
            excluded_evidence[label], f"excluded evidence {label}"
        )
        expected = {
            key: value
            for key, value in EXCLUDED_EVIDENCE_IDENTITIES[label].items()
            if key != "classification"
        }
        if fingerprint != expected:
            raise FreezeBFinalizationError(
                f"excluded evidence {label} fingerprint drifted"
            )
        observed[label] = {
            "classification": EXCLUDED_EVIDENCE_IDENTITIES[label]["classification"],
            **fingerprint,
            "reuse_allowed": False,
        }
    return observed


def _validate_active_evidence_fingerprints(
    *,
    local_evidence: Path,
    remote_evidence: Path,
) -> dict[str, dict[str, object]]:
    """Bind the only two evidence trees admitted from campaign G."""

    observed: dict[str, dict[str, object]] = {}
    for label, root in (
        ("local-a4", local_evidence),
        ("remote-a4", remote_evidence),
    ):
        fingerprint = _tree_fingerprint(root, f"active evidence {label}")
        expected = {
            key: value
            for key, value in ACTIVE_EVIDENCE_IDENTITIES[label].items()
            if key
            in {
                "entry_count",
                "file_count",
                "total_bytes",
                "root_sha256",
            }
        }
        if fingerprint != expected:
            raise FreezeBFinalizationError(
                f"active evidence {label} fingerprint drifted"
            )
        observed[label] = {
            **ACTIVE_EVIDENCE_IDENTITIES[label],
            **fingerprint,
        }
    return observed


def _file_identity(path: Path, label: str) -> dict[str, object]:
    if is_link_like(path) or not path.is_file():
        raise FreezeBFinalizationError(f"{label} must be a regular file")
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def _require_file_identity(
    path: Path,
    identity: Mapping[str, object],
    label: str,
) -> None:
    if (
        is_link_like(path)
        or not path.is_file()
        or path.stat().st_size != identity.get("size")
        or sha256_file(path) != identity.get("sha256")
    ):
        raise FreezeBFinalizationError(f"{label} changed during analysis/copy")


def _validate_output_destinations(
    project_root: Path,
    *,
    output_dir: Path,
    public_output_dir: Path | None,
    input_trees: Sequence[Path],
) -> tuple[Path, Path | None]:
    destination = validated_results_output_path(project_root, output_dir)
    public_destination = (
        validated_results_output_path(project_root, public_output_dir)
        if public_output_dir is not None
        else None
    )
    if public_destination is not None and public_destination == destination:
        raise FreezeBFinalizationError("private and public outputs must be distinct")
    for selected in (destination, public_destination):
        if selected is None:
            continue
        for source in input_trees:
            try:
                selected.relative_to(source)
            except ValueError:
                pass
            else:
                raise FreezeBFinalizationError(
                    "output must not be inside an input evidence tree"
                )
    return destination, public_destination


def _validate_disjoint_input_trees(input_trees: Sequence[Path]) -> None:
    """Reject equal or nested evidence roles before any copy or analysis."""

    for index, left in enumerate(input_trees):
        for right in input_trees[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise FreezeBFinalizationError(
                    "active and excluded evidence trees must be disjoint"
                )


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Verified, private bytes used by every finalization loader and metric."""

    root: Path
    readback_bundle: Path
    formal_bundle: Path
    local_evidence: Path
    remote_evidence: Path
    excluded_evidence: dict[str, Path]
    private_plan: Path
    public_protocol: Path
    corrigendum: Path
    admission_corrigendum: Path
    asset_preflight_corrigendum: Path
    runtime_order_corrigendum: Path
    time_grid_corrigendum: Path
    superseded_analysis_bundle: Path
    manifest: Path
    source_root: Path


def _prepare_analysis_snapshot(
    staging: Path,
    *,
    project_root: Path,
    readback_bundle: Path,
    formal_bundle: Path,
    local_evidence: Path,
    remote_evidence: Path,
    excluded_evidence: Mapping[str, Path],
    private_plan_path: Path,
    public_protocol_path: Path,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
    runtime_order_corrigendum_path: Path,
    time_grid_corrigendum_path: Path,
    superseded_analysis_bundle: Path,
) -> AnalysisSnapshot:
    """Copy and bind every input before any semantic or numeric analysis.

    Both the source endpoint and copied endpoint must equal the same pre-copy
    fingerprint.  Analysis subsequently receives only paths below ``staging``;
    a source that is temporarily replaced and restored cannot influence the
    published computation.
    """

    tree_sources = {
        "readback_bundle": (readback_bundle, "R1 bundle"),
        "formal_bundle": (formal_bundle, "formal bundle"),
        "local_evidence": (local_evidence, "local evidence"),
        "remote_evidence": (remote_evidence, "remote evidence"),
        "superseded_analysis_bundle": (
            superseded_analysis_bundle,
            "superseded H analysis bundle",
        ),
        **{
            f"excluded_{label}": (path, f"excluded evidence {label}")
            for label, path in excluded_evidence.items()
        },
    }
    manifest_path = project_root / GATE0_MANIFEST_RELATIVE_PATH
    file_sources = {
        "private_plan": (private_plan_path, "private plan"),
        "public_protocol": (public_protocol_path, "public protocol"),
        "corrigendum": (corrigendum_path, "corrigendum"),
        "admission_corrigendum": (
            admission_corrigendum_path,
            "admission corrigendum",
        ),
        "asset_preflight_corrigendum": (
            asset_preflight_corrigendum_path,
            "asset preflight corrigendum",
        ),
        "runtime_order_corrigendum": (
            runtime_order_corrigendum_path,
            "runtime order corrigendum",
        ),
        "time_grid_corrigendum": (
            time_grid_corrigendum_path,
            "time-grid corrigendum",
        ),
        "manifest": (manifest_path, "canonical manifest"),
    }
    tracked_sources = {
        relative: (
            project_root / PurePosixPath(relative),
            f"tracked source {relative}",
        )
        for relative in REQUIRED_SOURCE_PATHS
    }
    tree_fingerprints = {
        key: _tree_fingerprint(path, label)
        for key, (path, label) in tree_sources.items()
    }
    file_identities = {
        key: _file_identity(path, label)
        for key, (path, label) in file_sources.items()
    }
    tracked_identities = {
        relative: _file_identity(path, label)
        for relative, (path, label) in tracked_sources.items()
    }

    trees_root = staging / "trees"
    files_root = staging / "files"
    source_root = staging / "source"
    trees_root.mkdir()
    files_root.mkdir()
    source_root.mkdir()
    copied_trees = {
        key: trees_root / key for key in tree_sources
    }
    copied_files = {
        "private_plan": files_root / "freeze_b.private-plan.json",
        "public_protocol": files_root / "public-protocol.json",
        "corrigendum": files_root / "corrigendum.json",
        "admission_corrigendum": files_root / "admission-corrigendum.json",
        "asset_preflight_corrigendum": (
            files_root / "asset-preflight-corrigendum.json"
        ),
        "runtime_order_corrigendum": (
            files_root / "runtime-order-corrigendum.json"
        ),
        "time_grid_corrigendum": files_root / "time-grid-corrigendum.json",
        "manifest": files_root / "gate0.json",
    }
    for key, (source, label) in tree_sources.items():
        _copy_tree_strict(source, copied_trees[key], label)
    for key, (source, _label) in file_sources.items():
        _copy_file_exclusive(source, copied_files[key])
    for relative, (source, _label) in tracked_sources.items():
        _copy_file_exclusive(source, source_root / PurePosixPath(relative))

    for key, (source, label) in tree_sources.items():
        expected = tree_fingerprints[key]
        if (
            _tree_fingerprint(source, f"source {label}") != expected
            or _tree_fingerprint(copied_trees[key], f"copied {label}")
            != expected
        ):
            raise FreezeBFinalizationError(
                f"{label} changed while creating the analysis snapshot"
            )
    for key, (source, label) in file_sources.items():
        identity = file_identities[key]
        _require_file_identity(source, identity, f"source {label}")
        _require_file_identity(copied_files[key], identity, f"copied {label}")
    for relative, (source, label) in tracked_sources.items():
        identity = tracked_identities[relative]
        _require_file_identity(source, identity, f"source {label}")
        _require_file_identity(
            source_root / PurePosixPath(relative),
            identity,
            f"copied {label}",
        )

    return AnalysisSnapshot(
        root=staging,
        readback_bundle=copied_trees["readback_bundle"],
        formal_bundle=copied_trees["formal_bundle"],
        local_evidence=copied_trees["local_evidence"],
        remote_evidence=copied_trees["remote_evidence"],
        excluded_evidence={
            label: copied_trees[f"excluded_{label}"]
            for label in EXCLUDED_EVIDENCE_LABELS
        },
        private_plan=copied_files["private_plan"],
        public_protocol=copied_files["public_protocol"],
        corrigendum=copied_files["corrigendum"],
        admission_corrigendum=copied_files["admission_corrigendum"],
        asset_preflight_corrigendum=copied_files[
            "asset_preflight_corrigendum"
        ],
        runtime_order_corrigendum=copied_files[
            "runtime_order_corrigendum"
        ],
        time_grid_corrigendum=copied_files["time_grid_corrigendum"],
        superseded_analysis_bundle=copied_trees[
            "superseded_analysis_bundle"
        ],
        manifest=copied_files["manifest"],
        source_root=source_root,
    )


def _git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FreezeBFinalizationError(f"cannot verify Git identity: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise FreezeBFinalizationError(f"Git identity check failed: {detail}")
    return completed.stdout.strip()


def _git_archive_bytes(project_root: Path, revision: str) -> bytes:
    if _GIT_OID_RE.fullmatch(revision) is None:
        raise FreezeBFinalizationError("archive revision must be a 40-character Git OID")
    try:
        completed = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=project_root,
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FreezeBFinalizationError(f"cannot build immutable source archive: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeBFinalizationError(f"Git source archive failed: {detail}")
    return completed.stdout


def _git_archive_sha256(project_root: Path, revision: str) -> str:
    """Hash the exact binary tar stream deployed for execution commit D."""

    return sha256(_git_archive_bytes(project_root, revision)).hexdigest()


def _virtual_snapshot_sha256(
    archive_bytes: bytes,
    *,
    source_revision: str,
    source_tree: str,
    archive_sha256: str,
) -> str:
    """Reproduce install_gate0_snapshot's portable materialized-tree hash.

    This operates directly on the immutable Git tar stream, including
    implicit parent directories, Git executable bits, and the three installer
    markers.  Only the circular ``.snapshot-sha256`` marker is absent.
    """

    if (
        _GIT_OID_RE.fullmatch(source_revision) is None
        or _GIT_OID_RE.fullmatch(source_tree) is None
        or _SHA256_RE.fullmatch(archive_sha256) is None
    ):
        raise FreezeBFinalizationError("virtual snapshot identity is malformed")
    if sha256(archive_bytes).hexdigest() != archive_sha256:
        raise FreezeBFinalizationError("virtual snapshot archive digest mismatch")
    entries: dict[str, tuple[bytes, int, bytes | None]] = {}

    def ensure_parents(path: PurePosixPath) -> None:
        for length in range(1, len(path.parts)):
            parent = PurePosixPath(*path.parts[:length]).as_posix()
            existing = entries.get(parent)
            if existing is not None and existing[0] != b"D":
                raise FreezeBFinalizationError(
                    "source archive materializes a file as a parent directory"
                )
            entries.setdefault(parent, (b"D", 0, None))

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as bundle:
            for member in bundle.getmembers():
                raw_name = member.name.rstrip("/")
                if not raw_name or "\\" in raw_name:
                    raise FreezeBFinalizationError("source archive member path is invalid")
                path = PurePosixPath(raw_name)
                if (
                    path.is_absolute()
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
                    raise FreezeBFinalizationError("source archive member path is unsafe")
                name = path.as_posix()
                ensure_parents(path)
                if name in entries:
                    raise FreezeBFinalizationError(
                        f"source archive member is duplicate or out of order: {name}"
                    )
                if member.isdir():
                    entries[name] = (b"D", 0, None)
                elif member.isfile():
                    source = bundle.extractfile(member)
                    if source is None:
                        raise FreezeBFinalizationError(
                            f"source archive member cannot be read: {name}"
                        )
                    content = source.read()
                    if len(content) != member.size:
                        raise FreezeBFinalizationError(
                            f"source archive member size drifted: {name}"
                        )
                    entries[name] = (
                        b"F",
                        int(bool(member.mode & 0o111)),
                        content,
                    )
                else:
                    raise FreezeBFinalizationError(
                        f"source archive contains a non-regular member: {name}"
                    )
    except (tarfile.TarError, OSError) as exc:
        raise FreezeBFinalizationError(f"cannot inspect source archive: {exc}") from exc
    if not any(kind == b"F" for kind, _executable, _content in entries.values()):
        raise FreezeBFinalizationError("source archive contains no regular files")
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


def _validate_remote_source_archive(
    project_root: Path,
    *,
    execution_revision: str,
    remote_archive_sha256: object,
) -> str:
    observed = remote_archive_sha256
    if not isinstance(observed, str) or _SHA256_RE.fullmatch(observed) is None:
        raise FreezeBFinalizationError("remote source archive SHA-256 is malformed")
    expected = _git_archive_sha256(project_root, execution_revision)
    if observed != expected:
        raise FreezeBFinalizationError(
            "remote source archive is not the exact local execution-commit archive"
        )
    return expected


def _validate_remote_source_snapshot(
    project_root: Path,
    *,
    execution_revision: str,
    execution_tree: str,
    remote_archive_sha256: object,
    remote_snapshot_sha256: object,
) -> dict[str, str]:
    if not isinstance(remote_archive_sha256, str) or _SHA256_RE.fullmatch(
        remote_archive_sha256
    ) is None:
        raise FreezeBFinalizationError("remote source archive SHA-256 is malformed")
    if not isinstance(remote_snapshot_sha256, str) or _SHA256_RE.fullmatch(
        remote_snapshot_sha256
    ) is None:
        raise FreezeBFinalizationError("remote source snapshot SHA-256 is malformed")
    archive_bytes = _git_archive_bytes(project_root, execution_revision)
    expected_archive = sha256(archive_bytes).hexdigest()
    if remote_archive_sha256 != expected_archive:
        raise FreezeBFinalizationError(
            "remote source archive is not the exact local execution-commit archive"
        )
    expected_snapshot = _virtual_snapshot_sha256(
        archive_bytes,
        source_revision=execution_revision,
        source_tree=execution_tree,
        archive_sha256=expected_archive,
    )
    if remote_snapshot_sha256 != expected_snapshot:
        raise FreezeBFinalizationError(
            "remote source snapshot does not match the locally reproduced installer snapshot"
        )
    return {
        "source_archive_sha256": expected_archive,
        "source_snapshot_sha256": expected_snapshot,
    }


def _validate_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
) -> dict[str, object]:
    data = _mapping(value, "corrigendum")
    _exact_keys(
        data,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_preregistration",
            "incident",
            "correction",
            "correction_source",
            "source_transition_policy",
        },
        "corrigendum",
    )
    expected_identity = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-evidence-corrigendum-v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "state": (
            "POST_PREREGISTRATION_EVIDENCE_CORRECTION_BEFORE_ANY_ADMITTED_"
            "FREEZE_B_EVIDENCE"
        ),
    }
    if any(data.get(key) != expected for key, expected in expected_identity.items()):
        raise FreezeBFinalizationError("corrigendum identity literals drifted")
    frozen = _mapping(data.get("frozen_preregistration"), "frozen preregistration")
    _exact_keys(
        frozen,
        {
            "implementation_revision",
            "implementation_tree",
            "private_plan_sha256",
            "private_plan_file_sha256",
            "preregistration_revision",
            "preregistration_tree",
            "public_protocol_path",
            "public_protocol_blob_oid",
            "public_protocol_file_sha256",
            "public_protocol_canonical_sha256",
            "public_protocol_unchanged",
        },
        "frozen preregistration",
    )
    expected_frozen = {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "private_plan_sha256": PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": PRIVATE_PLAN_FILE_SHA256,
        "preregistration_revision": PREREGISTRATION_REVISION,
        "preregistration_tree": PREREGISTRATION_TREE,
        "public_protocol_path": PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
        "public_protocol_blob_oid": PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
    }
    if dict(frozen) != expected_frozen:
        raise FreezeBFinalizationError("corrigendum frozen preregistration drifted")
    incident = _mapping(data.get("incident"), "corrigendum incident")
    _exact_keys(
        incident,
        {
            "attempt_label",
            "attempt_classification",
            "raw_local_run_file_count",
            "admitted_evidence_case_count",
            "planned_evidence_case_count",
            "ovphysx_case_count_executed",
            "remote_or_gpu_execution_occurred",
            "terminal_completion_ledger_written",
            "failure_class",
            "detected_by",
            "trajectory_file_deserialized_for_structural_validation",
            "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction",
            "scientific_metrics_computed",
            "scientific_label_assigned",
            "reuse_allowed",
        },
        "corrigendum incident",
    )
    expected_incident = {
        "attempt_label": "local-a1",
        "attempt_classification": "invalid_collection_attempt",
        "raw_local_run_file_count": 1,
        "admitted_evidence_case_count": 0,
        "planned_evidence_case_count": 24,
        "ovphysx_case_count_executed": 0,
        "remote_or_gpu_execution_occurred": False,
        "terminal_completion_ledger_written": False,
        "failure_class": (
            "windows_venv_launcher_pid_was_not_actual_python_worker_pid"
        ),
        "detected_by": "fresh_process_worker_pid_binding_gate",
        "trajectory_file_deserialized_for_structural_validation": True,
        "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction": False,
        "scientific_metrics_computed": False,
        "scientific_label_assigned": False,
        "reuse_allowed": False,
    }
    if dict(incident) != expected_incident:
        raise FreezeBFinalizationError("corrigendum incident facts drifted")
    correction = _mapping(data.get("correction"), "corrigendum correction")
    _exact_keys(
        correction,
        {
            "scope",
            "worker_identity_authority",
            "worker_identity_capture",
            "launcher_record_relation",
            "finalizer_relation",
            "full_matrix_rerun_required",
            "failed_attempt_artifacts_must_remain_unmodified",
            "scientific_protocol_fields_changed",
            "adapter_changed",
            "scenario_or_manifest_changed",
            "control_or_target_changed",
            "dt_or_case_order_changed",
            "metrics_or_thresholds_changed",
            "formal_gate0_status_unchanged",
            "formal_gate0_pass_ready_unchanged",
        },
        "corrigendum correction",
    )
    expected_correction = {
        "scope": "evidence_plumbing_only",
        "worker_identity_authority": "actual_python_worker_kernel_identity_v1",
        "worker_identity_capture": "same_python_process_before_runner_main",
        "launcher_record_relation": (
            "exact_copy_of_worker_identity_with_recomputed_hash"
        ),
        "finalizer_relation": (
            "worker_sidecar_fresh_record_run_worker_pid_three_way_binding"
        ),
        "full_matrix_rerun_required": True,
        "failed_attempt_artifacts_must_remain_unmodified": True,
        "scientific_protocol_fields_changed": [],
        "adapter_changed": False,
        "scenario_or_manifest_changed": False,
        "control_or_target_changed": False,
        "dt_or_case_order_changed": False,
        "metrics_or_thresholds_changed": False,
        "formal_gate0_status_unchanged": "DIVERGENT",
        "formal_gate0_pass_ready_unchanged": False,
    }
    if dict(correction) != expected_correction:
        raise FreezeBFinalizationError("corrigendum correction scope drifted")
    correction_source = _mapping(
        data.get("correction_source"), "corrigendum correction source"
    )
    _exact_keys(correction_source, {"revision", "tree"}, "correction source")
    if any(
        not isinstance(correction_source.get(key), str)
        or _GIT_OID_RE.fullmatch(str(correction_source[key])) is None
        for key in ("revision", "tree")
    ):
        raise FreezeBFinalizationError("corrigendum correction source is invalid")
    policy = _mapping(data.get("source_transition_policy"), "transition policy")
    _exact_keys(
        policy,
        {
            "implementation_to_preregistration_name_status",
            "preregistration_to_correction_name_status",
            "correction_to_execution_name_status",
            "renames_allowed",
            "other_paths_allowed",
        },
        "transition policy",
    )
    expected_policy = {
        "implementation_to_preregistration_name_status": (
            IMPLEMENTATION_TO_PREREGISTRATION_DIFF
        ),
        "preregistration_to_correction_name_status": (
            PREREGISTRATION_TO_CORRECTION_DIFF
        ),
        "correction_to_execution_name_status": CORRECTION_TO_EXECUTION_DIFF,
        "renames_allowed": False,
        "other_paths_allowed": False,
    }
    if dict(policy) != expected_policy:
        raise FreezeBFinalizationError("corrigendum transition policy drifted")
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_admission_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
) -> dict[str, object]:
    """Validate the post-a2 admission-only correction without scope drift."""

    data = _mapping(value, "admission corrigendum")
    _exact_keys(
        data,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_history",
            "incident",
            "correction",
            "excluded_evidence",
            "correction_source",
            "source_transition_policy",
        },
        "admission corrigendum",
    )
    expected_identity = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-remote-admission-corrigendum-v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "state": (
            "POST_PREREGISTRATION_REMOTE_ADMISSION_CORRECTION_AFTER_ABORTED_A2_"
            "COLLECTION_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
    }
    if any(data.get(key) != expected for key, expected in expected_identity.items()):
        raise FreezeBFinalizationError("admission corrigendum identity literals drifted")

    frozen = _mapping(data.get("frozen_history"), "admission frozen history")
    expected_frozen = {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "preregistration_revision": PREREGISTRATION_REVISION,
        "preregistration_tree": PREREGISTRATION_TREE,
        "first_correction_revision": first_correction_revision,
        "first_correction_tree": first_correction_tree,
        "first_execution_revision": FIRST_EXECUTION_REVISION,
        "first_execution_tree": FIRST_EXECUTION_TREE,
        "private_plan_sha256": PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": PRIVATE_PLAN_FILE_SHA256,
        "public_protocol_path": PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
        "public_protocol_blob_oid": PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
        "first_corrigendum_path": CORRIGENDUM_RELATIVE_PATH.as_posix(),
        "first_corrigendum_blob_oid": FIRST_CORRIGENDUM_BLOB_OID,
        "first_corrigendum_file_sha256": FIRST_CORRIGENDUM_FILE_SHA256,
        "first_corrigendum_canonical_sha256": FIRST_CORRIGENDUM_CANONICAL_SHA256,
        "first_corrigendum_unchanged": True,
    }
    _exact_keys(frozen, set(expected_frozen), "admission frozen history")
    if dict(frozen) != expected_frozen:
        raise FreezeBFinalizationError("admission corrigendum frozen history drifted")

    incident = _mapping(data.get("incident"), "admission incident")
    expected_incident = {
        "campaign_label": "a2",
        "attempt_classification": "aborted_mixed_local_remote_collection_campaign",
        "local_attempt_label": "local-a2",
        "remote_attempt_label": "remote-a2",
        "local_mujoco_case_count_executed": 8,
        "local_mujoco_case_count_automated_admission_validated": 8,
        "remote_ovphysx_case_count_executed": 1,
        "remote_worker_completed_case_count": 1,
        "remote_launcher_admitted_case_count": 0,
        "final_analysis_admitted_case_count": 0,
        "remote_or_gpu_execution_occurred": True,
        "terminal_success_completion_ledger_written": False,
        "failure_class": "post_worker_payload_validator_disabled_pinned_site_packages",
        "detected_by": "remote_payload_admission_gate",
        "root_cause": "python_no_site_flag_blocked_pinned_numpy_import",
        "automated_structural_or_numerical_validity_validation_occurred": True,
        "trajectory_sample_values_may_have_been_reviewed": True,
        "trajectory_sample_values_used_to_select_or_design_correction": False,
        "scientific_metrics_computed": False,
        "scientific_label_assigned": False,
        "reuse_allowed": False,
    }
    _exact_keys(incident, set(expected_incident), "admission incident")
    if dict(incident) != expected_incident:
        raise FreezeBFinalizationError("admission incident facts drifted")

    correction = _mapping(data.get("correction"), "admission correction")
    expected_correction = {
        "scientific_execution_change_scope": (
            "remote_post_worker_payload_admission_only"
        ),
        "supporting_provenance_and_finalizer_plumbing_changed": True,
        "validator_python_flags_before": ["-P", "-S", "-"],
        "validator_python_flags_after": ["-P", "-"],
        "pinned_environment_site_packages_enabled_after": True,
        "user_site_packages_remain_disabled": True,
        "worker_execution_command_changed": False,
        "worker_or_adapter_changed": False,
        "scenario_or_manifest_changed": False,
        "control_or_target_changed": False,
        "dt_or_case_order_changed": False,
        "metrics_or_thresholds_changed": False,
        "scientific_protocol_fields_changed": [],
        "full_24_case_matrix_rerun_required": True,
        "failed_attempt_artifacts_must_remain_unmodified": True,
        "formal_gate0_status_unchanged": "DIVERGENT",
        "formal_gate0_pass_ready_unchanged": False,
    }
    _exact_keys(correction, set(expected_correction), "admission correction")
    if dict(correction) != expected_correction:
        raise FreezeBFinalizationError("admission correction scope drifted")

    excluded = _mapping(data.get("excluded_evidence"), "excluded evidence")
    _exact_keys(
        excluded,
        set(ADMISSION_EXCLUDED_EVIDENCE_LABELS),
        "excluded evidence",
    )
    for label in ADMISSION_EXCLUDED_EVIDENCE_LABELS:
        record = _mapping(excluded.get(label), f"excluded evidence {label}")
        expected_record = {
            **ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES[label],
            "reuse_allowed": False,
        }
        _exact_keys(record, set(expected_record), f"excluded evidence {label}")
        if dict(record) != expected_record:
            raise FreezeBFinalizationError(
                f"excluded evidence {label} identity drifted"
            )

    correction_source = _mapping(
        data.get("correction_source"), "admission correction source"
    )
    _exact_keys(correction_source, {"revision", "tree"}, "admission correction source")
    if any(
        not isinstance(correction_source.get(key), str)
        or _GIT_OID_RE.fullmatch(str(correction_source[key])) is None
        for key in ("revision", "tree")
    ):
        raise FreezeBFinalizationError("admission correction source is invalid")

    policy = _mapping(
        data.get("source_transition_policy"), "admission transition policy"
    )
    expected_policy = {
        "first_execution_to_admission_correction_name_status": (
            FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF
        ),
        "admission_correction_to_execution_name_status": (
            ADMISSION_CORRECTION_TO_EXECUTION_DIFF
        ),
        "renames_allowed": False,
        "other_paths_allowed": False,
    }
    _exact_keys(policy, set(expected_policy), "admission transition policy")
    if dict(policy) != expected_policy:
        raise FreezeBFinalizationError("admission transition policy drifted")
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_asset_preflight_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    """Validate the cumulative local-a3 pre-simulation incident ledger."""

    data = _mapping(value, "asset preflight corrigendum")
    _exact_keys(
        data,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_history",
            "incident",
            "correction",
            "excluded_evidence",
            "incident_source",
            "source_transition_policy",
        },
        "asset preflight corrigendum",
    )
    expected = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-asset-preflight-"
            "corrigendum-v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "state": (
            "POST_PREREGISTRATION_OPERATOR_ASSET_PREFLIGHT_INCIDENT_AFTER_"
            "ABORTED_LOCAL_A3_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
        "frozen_history": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "preregistration_revision": PREREGISTRATION_REVISION,
            "preregistration_tree": PREREGISTRATION_TREE,
            "first_correction_revision": first_correction_revision,
            "first_correction_tree": first_correction_tree,
            "first_execution_revision": FIRST_EXECUTION_REVISION,
            "first_execution_tree": FIRST_EXECUTION_TREE,
            "admission_correction_revision": admission_correction_revision,
            "admission_correction_tree": admission_correction_tree,
            "private_plan_sha256": PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": PUBLIC_PROTOCOL_CANONICAL_SHA256,
            "public_protocol_unchanged": True,
            "first_corrigendum_path": CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": (
                FIRST_CORRIGENDUM_CANONICAL_SHA256
            ),
            "first_corrigendum_unchanged": True,
            "admission_corrigendum_path": (
                ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ),
            "admission_corrigendum_blob_oid": ADMISSION_CORRIGENDUM_BLOB_OID,
            "admission_corrigendum_file_sha256": (
                ADMISSION_CORRIGENDUM_FILE_SHA256
            ),
            "admission_corrigendum_canonical_sha256": (
                ADMISSION_CORRIGENDUM_CANONICAL_SHA256
            ),
            "admission_corrigendum_unchanged": True,
        },
        "incident": {
            "attempt_label": "local-a3",
            "attempt_classification": "aborted_local_pre_simulation_attempt",
            "worker_process_count_started": 1,
            "worker_process_exit_verified": True,
            "runner_asset_preflight_entered": True,
            "adapter_instance_created": False,
            "mujoco_runtime_imported": False,
            "model_created": False,
            "trajectory_case_count_executed": 0,
            "simulator_step_count": 0,
            "run_payload_count": 0,
            "launcher_admitted_case_count": 0,
            "final_analysis_admitted_case_count": 0,
            "remote_or_gpu_execution_occurred": False,
            "terminal_success_completion_ledger_written": False,
            "failure_class": (
                "materialized_asset_tree_did_not_match_frozen_canonical_lf_hash"
            ),
            "detected_by": "runner_canonical_lf_asset_tree_sha256_preflight",
            "expected_canonical_lf_asset_tree_sha256": (
                "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
            ),
            "operator_disclosed_noncanonical_asset_materialization": True,
            "actual_materialized_asset_root_recorded_in_evidence": False,
            "actual_materialized_asset_tree_sha256_recorded_in_evidence": False,
            "trajectory_sample_values_existed": False,
            "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction": (
                False
            ),
            "scientific_metrics_computed": False,
            "scientific_label_assigned": False,
            "reuse_allowed": False,
        },
        "correction": {
            "scientific_execution_change_scope": (
                "operator_asset_materialization_compliance_only"
            ),
            "operator_invocation_changed_to_match_frozen_asset_identity": True,
            "supporting_provenance_finalizer_and_local_launcher_plumbing_changed": (
                True
            ),
            "adapter_or_runner_scientific_behavior_changed": False,
            "scenario_or_manifest_changed": False,
            "control_or_target_changed": False,
            "dt_or_case_order_changed": False,
            "metrics_or_thresholds_changed": False,
            "scientific_protocol_fields_changed": [],
            "full_24_case_matrix_rerun_required": True,
            "failed_attempt_artifacts_must_remain_unmodified": True,
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
        },
        "excluded_evidence": {
            label: {
                **EXCLUDED_EVIDENCE_IDENTITIES[label],
                "reuse_allowed": False,
            }
            for label in EXCLUDED_EVIDENCE_LABELS
        },
        "incident_source": {
            "revision": SECOND_EXECUTION_REVISION,
            "tree": SECOND_EXECUTION_TREE,
        },
        "source_transition_policy": {
            "incident_source_to_execution_name_status": (
                INCIDENT_SOURCE_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }
    if dict(data) != expected:
        raise FreezeBFinalizationError(
            "asset preflight corrigendum facts or scope drifted"
        )
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_runtime_order_corrigendum(value: object) -> dict[str, object]:
    """Validate the post-collection, analysis-only G -> H correction."""

    data = _mapping(value, "runtime order corrigendum")
    expected = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-runtime-order-"
            "corrigendum-v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "state": (
            "POST_COLLECTION_ANALYSIS_ONLY_RUNTIME_ORDER_CORRECTION_BEFORE_"
            "FINAL_BUNDLE_PUBLICATION"
        ),
        "frozen_campaign": {
            "campaign_execution_revision": CAMPAIGN_EXECUTION_REVISION,
            "campaign_execution_tree": CAMPAIGN_EXECUTION_TREE,
            "campaign_source_archive_sha256": CAMPAIGN_SOURCE_ARCHIVE_SHA256,
            "remote_source_snapshot_sha256": REMOTE_SOURCE_SNAPSHOT_SHA256,
            "private_plan_sha256": PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": (
                PUBLIC_PROTOCOL_CANONICAL_SHA256
            ),
            "public_protocol_unchanged": True,
            "first_corrigendum_path": CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": (
                FIRST_CORRIGENDUM_CANONICAL_SHA256
            ),
            "first_corrigendum_unchanged": True,
            "admission_corrigendum_path": (
                ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ),
            "admission_corrigendum_blob_oid": ADMISSION_CORRIGENDUM_BLOB_OID,
            "admission_corrigendum_file_sha256": (
                ADMISSION_CORRIGENDUM_FILE_SHA256
            ),
            "admission_corrigendum_canonical_sha256": (
                ADMISSION_CORRIGENDUM_CANONICAL_SHA256
            ),
            "admission_corrigendum_unchanged": True,
            "asset_preflight_corrigendum_path": (
                ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ),
            "asset_preflight_corrigendum_blob_oid": (
                ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID
            ),
            "asset_preflight_corrigendum_file_sha256": (
                ASSET_PREFLIGHT_CORRIGENDUM_FILE_SHA256
            ),
            "asset_preflight_corrigendum_canonical_sha256": (
                ASSET_PREFLIGHT_CORRIGENDUM_CANONICAL_SHA256
            ),
            "asset_preflight_corrigendum_unchanged": True,
        },
        "active_evidence": ACTIVE_EVIDENCE_IDENTITIES,
        "incident": {
            "active_evidence_case_count": 24,
            "collection_complete": True,
            "combined_formal_repeat_dt_sensitivity_metrics_computed": False,
            "correction_rule_source": (
                "existing_mapping_schema_and_freeze_a_exact_id_join_semantics"
            ),
            "correction_selected_from_scientific_effect_or_threshold": False,
            "detected_by": (
                "post_collection_finalizer_runtime_friction_readback_gate"
            ),
            "failure_class": (
                "analysis_assumed_backend_runtime_joint_order_equaled_"
                "expected_private_plan_order"
            ),
            "final_bundle_published": False,
            "raw_evidence_invalidated": False,
            "runtime_readback_diagnostic_result_reviewed": True,
            "runtime_readback_values_reviewed": True,
            "runtime_readback_case_count": 16,
            "diagnostic_canonical_projection_repeat_group_match_count": 8,
            "diagnostic_canonical_projection_sham_r1_match_case_count": 8,
            "diagnostic_canonical_projection_zero_changed_from_r1_case_count": 0,
            "scientific_protocol_defect": False,
            "scientific_label_assigned": False,
            "simulator_execution_defect": False,
            "trajectory_payloads_deserialized": True,
            "trajectory_values_finite_and_sanity_checked": True,
            "trajectory_values_used_to_choose_correction": False,
        },
        "correction": {
            "scope": "post_collection_analysis_only",
            "canonicalization_key": "canonical_id",
            "runtime_order_gate": (
                "22_unique_canonical_ids_and_set_equals_expected_private_plan_"
                "joint_names"
            ),
            "record_alignment_gate": (
                "records[i].backend_index_equals_i_and_backend_name_equals_"
                "backend_joint_order[i]"
            ),
            "mapping_inverse_gate": (
                "runtime_record_canonical_id_to_backend_name_and_index_equals_"
                "joint_mapping"
            ),
            "analysis_order": "expected_private_plan_joint_names_order",
            "position_target_vector_order": (
                "joint_mapping.index_from_backend_tensor_then_expected_"
                "private_plan_joint_names_order"
            ),
            "raw_evidence_modified": False,
            "numeric_values_changed": False,
            "protocol_changed": False,
            "control_or_target_changed": False,
            "dt_or_case_order_changed": False,
            "metrics_or_thresholds_changed": False,
            "simulator_rerun_required": False,
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
        },
        "source_transition_policy": {
            "campaign_execution_to_analysis_correction_name_status": (
                CAMPAIGN_TO_ANALYSIS_CORRECTION_DIFF
            ),
            "direct_single_parent_required": True,
            "analysis_tree_bound_to_head": True,
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
        "claim_boundary": {
            "study": (
                "unofficial_simulation_only_kitless_ovphysx_native_input_"
                "sensitivity"
            ),
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
            "cross_engine_parameter_equivalence_claimed": False,
            "official_backend_bug_claimed": False,
            "hardware_or_sim2real_claimed": False,
        },
    }
    if dict(data) != expected:
        raise FreezeBFinalizationError(
            "runtime order corrigendum facts or scope drifted"
        )
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_time_grid_corrigendum(value: object) -> dict[str, object]:
    """Bind every disclosed H incident and the analysis-only I rule."""

    data = _mapping(value, "time-grid corrigendum")
    _exact_keys(
        data,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_history",
            "superseded_analysis_bundle",
            "incident",
            "outcome_exposure",
            "diagnosis",
            "correction",
            "source_transition_policy",
            "claim_boundary",
        },
        "time-grid corrigendum",
    )
    if canonical_json_sha256(data) != TIME_GRID_CORRIGENDUM_CANONICAL_SHA256:
        raise FreezeBFinalizationError(
            "time-grid corrigendum facts or scope drifted"
        )
    frozen = _mapping(data["frozen_history"], "time-grid frozen history")
    expected_frozen = {
        "campaign_execution_revision": CAMPAIGN_EXECUTION_REVISION,
        "campaign_execution_tree": CAMPAIGN_EXECUTION_TREE,
        "first_analysis_revision": FIRST_ANALYSIS_REVISION,
        "first_analysis_tree": FIRST_ANALYSIS_TREE,
        "private_plan_sha256": PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": PRIVATE_PLAN_FILE_SHA256,
        "public_protocol_blob_oid": PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
        "runtime_order_corrigendum_blob_oid": (
            RUNTIME_ORDER_CORRIGENDUM_BLOB_OID
        ),
        "runtime_order_corrigendum_file_sha256": (
            RUNTIME_ORDER_CORRIGENDUM_FILE_SHA256
        ),
        "runtime_order_corrigendum_canonical_sha256": (
            RUNTIME_ORDER_CORRIGENDUM_CANONICAL_SHA256
        ),
        "runtime_order_corrigendum_unchanged": True,
    }
    if dict(frozen) != expected_frozen:
        raise FreezeBFinalizationError("time-grid frozen history drifted")
    if dict(
        _mapping(
            data["superseded_analysis_bundle"],
            "time-grid superseded bundle",
        )
    ) != SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY:
        raise FreezeBFinalizationError(
            "time-grid superseded bundle identity drifted"
        )
    transition = _mapping(
        data["source_transition_policy"],
        "time-grid source transition policy",
    )
    if transition.get(
        "analysis_to_time_grid_correction_name_status"
    ) != ANALYSIS_TO_TIME_GRID_CORRECTION_DIFF:
        raise FreezeBFinalizationError("time-grid source transition drifted")
    correction = _mapping(
        data["correction"], "time-grid correction policy"
    )
    if (
        correction.get("compatibility_abs_tolerance_s")
        != TIME_GRID_ABS_TOLERANCE_S
        or correction.get(
            "base_baseline_and_zero_vs_mujoco_window_sample_count"
        )
        != 200
        or correction.get(
            "halved_baseline_and_zero_vs_mujoco_window_sample_count"
        )
        != 400
        or correction.get("base_treatment_vs_sham_window_sample_count")
        != 201
        or correction.get("halved_treatment_vs_sham_window_sample_count")
        != 401
    ):
        raise FreezeBFinalizationError(
            "time-grid tolerance or frozen window counts drifted"
        )
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_superseded_analysis_bundle(root: Path) -> dict[str, object]:
    """Verify the complete local-only H bundle before admitting its subset."""

    verification = verify_exact_bundle(root)
    if {
        "file_count": verification.get("file_count"),
        "total_bytes": verification.get("total_size"),
        "root_sha256": verification.get("root_sha256"),
        "exact_inventory": verification.get("exact_inventory"),
    } != {
        "file_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY["file_count"],
        "total_bytes": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY["total_bytes"],
        "root_sha256": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY["root_sha256"],
        "exact_inventory": True,
    }:
        raise FreezeBFinalizationError(
            "superseded H analysis bundle inventory or root drifted"
        )
    if _tree_fingerprint(root, "superseded H analysis bundle") != {
        "entry_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_entry_count"
        ],
        "file_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_file_count"
        ],
        "total_bytes": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_total_bytes"
        ],
        "root_sha256": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_root_sha256"
        ],
    }:
        raise FreezeBFinalizationError(
            "superseded H analysis bundle full-tree fingerprint drifted"
        )
    public_pair = root / "public/results/freeze-b"
    if _tree_fingerprint(public_pair, "superseded H public pair") != {
        "entry_count": 2,
        "file_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "public_pair_file_count"
        ],
        "total_bytes": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "public_pair_total_bytes"
        ],
        "root_sha256": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "public_pair_root_sha256"
        ],
    }:
        raise FreezeBFinalizationError(
            "superseded H public pair fingerprint drifted"
        )
    bound_files = {
        "bundle_json_sha256": root / "bundle.json",
        "private_decision_sha256": root / "private/decision.json",
        "private_finalization_sha256": root / "private/finalization.json",
        "private_source_identity_sha256": root / "private/source-identity.json",
        "public_summary_sha256": public_pair / "summary.json",
        "public_report_sha256": public_pair / "report.md",
    }
    if any(
        sha256_file(path) != SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[key]
        for key, path in bound_files.items()
    ):
        raise FreezeBFinalizationError(
            "superseded H analysis bundle bound-file digest drifted"
        )
    decision = _strict_json(
        root / "private/decision.json", "superseded H private decision"
    )
    if decision != {
        "schema_version": 1,
        "validation_status": "INVALID",
        "scientific_label": None,
        "cells": [],
        "s_values": [],
        "validity_failures": [
            "metric traces lack an exact recorded time grid"
        ],
    }:
        raise FreezeBFinalizationError(
            "superseded H private decision facts drifted"
        )
    summary = _strict_json(
        public_pair / "summary.json", "superseded H public summary"
    )
    validity = _mapping(summary.get("validity"), "superseded H validity")
    checks = _mapping(validity.get("checks"), "superseded H checks")
    if (
        summary.get("validation_status") != "INVALID"
        or summary.get("scientific_label") is not None
        or summary.get("primary_metrics") is not None
        or validity.get("all_passed") is not True
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise FreezeBFinalizationError(
            "superseded H public summary incident facts drifted"
        )
    report = (public_pair / "report.md").read_text(encoding="utf-8")
    if (
        "Validation status: **INVALID**" not in report
        or "All validity gates passed: true" not in report
        or "at least one preregistered validity gate failed" not in report
    ):
        raise FreezeBFinalizationError(
            "superseded H public report contradiction facts drifted"
        )
    finalization = _strict_json(
        root / "private/finalization.json",
        "superseded H private finalization",
    )
    source_identity = _strict_json(
        root / "private/source-identity.json",
        "superseded H source identity",
    )
    if (
        finalization.get("validation_status") != "INVALID"
        or finalization.get("scientific_label") is not None
        or source_identity.get("analysis_revision") != FIRST_ANALYSIS_REVISION
        or source_identity.get("analysis_tree") != FIRST_ANALYSIS_TREE
    ):
        raise FreezeBFinalizationError(
            "superseded H private provenance drifted"
        )
    return json.loads(
        json.dumps(SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY, allow_nan=False)
    )


def _git_name_status(project_root: Path, older: str, newer: str) -> list[str]:
    output = _git(
        project_root,
        "diff",
        "--name-status",
        "--no-renames",
        older,
        newer,
        "--",
    )
    return output.splitlines() if output else []


def _validate_source_identity(
    project_root: Path,
    *,
    implementation_revision: str,
    implementation_tree: str,
    campaign_revision: str,
    analysis_revision: str,
    public_protocol_path: Path,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
    runtime_order_corrigendum_path: Path,
    time_grid_corrigendum_path: Path,
    canonical_public_protocol_path: Path | None = None,
    canonical_corrigendum_path: Path | None = None,
    canonical_admission_corrigendum_path: Path | None = None,
    canonical_asset_preflight_corrigendum_path: Path | None = None,
    canonical_runtime_order_corrigendum_path: Path | None = None,
    canonical_time_grid_corrigendum_path: Path | None = None,
) -> dict[str, object]:
    """Prove exact A -> B -> C -> D -> E -> F -> G -> H -> I transitions."""

    if not all(
        _GIT_OID_RE.fullmatch(item)
        for item in (
            implementation_revision,
            implementation_tree,
            campaign_revision,
            CAMPAIGN_EXECUTION_TREE,
            FIRST_ANALYSIS_REVISION,
            FIRST_ANALYSIS_TREE,
            analysis_revision,
        )
    ):
        raise FreezeBFinalizationError("source identities must be 40-character Git OIDs")
    if campaign_revision != CAMPAIGN_EXECUTION_REVISION:
        raise FreezeBFinalizationError(
            "campaign revision must be the frozen G execution"
        )
    top = Path(_git(project_root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != project_root:
        raise FreezeBFinalizationError("project root is not the Git top-level directory")
    head = _git(project_root, "rev-parse", "HEAD")
    if head != analysis_revision:
        raise FreezeBFinalizationError("analysis revision is not current HEAD")
    analysis_tree = _git(project_root, "rev-parse", "HEAD^{tree}")
    committed_protocol_path = (
        public_protocol_path
        if canonical_public_protocol_path is None
        else canonical_public_protocol_path
    )
    committed_corrigendum_path = (
        corrigendum_path
        if canonical_corrigendum_path is None
        else canonical_corrigendum_path
    )
    committed_admission_corrigendum_path = (
        admission_corrigendum_path
        if canonical_admission_corrigendum_path is None
        else canonical_admission_corrigendum_path
    )
    committed_asset_preflight_corrigendum_path = (
        asset_preflight_corrigendum_path
        if canonical_asset_preflight_corrigendum_path is None
        else canonical_asset_preflight_corrigendum_path
    )
    committed_runtime_order_corrigendum_path = (
        runtime_order_corrigendum_path
        if canonical_runtime_order_corrigendum_path is None
        else canonical_runtime_order_corrigendum_path
    )
    committed_time_grid_corrigendum_path = (
        time_grid_corrigendum_path
        if canonical_time_grid_corrigendum_path is None
        else canonical_time_grid_corrigendum_path
    )
    relative_protocol = committed_protocol_path.relative_to(project_root).as_posix()
    if relative_protocol != PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix():
        raise FreezeBFinalizationError("public protocol is not the canonical Freeze B config")
    relative_corrigendum = committed_corrigendum_path.relative_to(
        project_root
    ).as_posix()
    if relative_corrigendum != CORRIGENDUM_RELATIVE_PATH.as_posix():
        raise FreezeBFinalizationError("corrigendum is not the canonical Freeze B file")
    relative_admission_corrigendum = committed_admission_corrigendum_path.relative_to(
        project_root
    ).as_posix()
    if (
        relative_admission_corrigendum
        != ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
    ):
        raise FreezeBFinalizationError(
            "admission corrigendum is not the canonical Freeze B file"
        )
    relative_asset_preflight_corrigendum = (
        committed_asset_preflight_corrigendum_path.relative_to(
            project_root
        ).as_posix()
    )
    if (
        relative_asset_preflight_corrigendum
        != ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix()
    ):
        raise FreezeBFinalizationError(
            "asset preflight corrigendum is not the canonical Freeze B file"
        )
    relative_runtime_order_corrigendum = (
        committed_runtime_order_corrigendum_path.relative_to(
            project_root
        ).as_posix()
    )
    if (
        relative_runtime_order_corrigendum
        != RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH.as_posix()
    ):
        raise FreezeBFinalizationError(
            "runtime order corrigendum is not the canonical Freeze B file"
        )
    relative_time_grid_corrigendum = (
        committed_time_grid_corrigendum_path.relative_to(
            project_root
        ).as_posix()
    )
    if (
        relative_time_grid_corrigendum
        != TIME_GRID_CORRIGENDUM_RELATIVE_PATH.as_posix()
    ):
        raise FreezeBFinalizationError(
            "time-grid corrigendum is not the canonical Freeze B file"
        )
    corrigendum = _validate_corrigendum(
        _strict_json(corrigendum_path, "Freeze B corrigendum"),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
    )
    correction_source = _mapping(
        corrigendum["correction_source"], "corrigendum correction source"
    )
    correction_revision = str(correction_source["revision"])
    correction_tree = str(correction_source["tree"])
    admission_corrigendum = _validate_admission_corrigendum(
        _strict_json(
            admission_corrigendum_path, "Freeze B admission corrigendum"
        ),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
    )
    admission_correction_source = _mapping(
        admission_corrigendum["correction_source"],
        "admission corrigendum correction source",
    )
    admission_correction_revision = str(admission_correction_source["revision"])
    admission_correction_tree = str(admission_correction_source["tree"])
    asset_preflight_corrigendum = _validate_asset_preflight_corrigendum(
        _strict_json(
            asset_preflight_corrigendum_path,
            "Freeze B asset preflight corrigendum",
        ),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
        admission_correction_revision=admission_correction_revision,
        admission_correction_tree=admission_correction_tree,
    )
    runtime_order_corrigendum = _validate_runtime_order_corrigendum(
        _strict_json(
            runtime_order_corrigendum_path,
            "Freeze B runtime order corrigendum",
        )
    )
    time_grid_corrigendum = _validate_time_grid_corrigendum(
        _strict_json(
            time_grid_corrigendum_path,
            "Freeze B time-grid corrigendum",
        )
    )
    identities = (
        implementation_revision,
        PREREGISTRATION_REVISION,
        correction_revision,
        FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        SECOND_EXECUTION_REVISION,
        campaign_revision,
        FIRST_ANALYSIS_REVISION,
        analysis_revision,
    )
    if len(set(identities)) != 9:
        raise FreezeBFinalizationError(
            "A, B, C, D, E, F, G, H, and I must be distinct"
        )
    expected_parents = {
        PREREGISTRATION_REVISION: implementation_revision,
        correction_revision: PREREGISTRATION_REVISION,
        FIRST_EXECUTION_REVISION: correction_revision,
        admission_correction_revision: FIRST_EXECUTION_REVISION,
        SECOND_EXECUTION_REVISION: admission_correction_revision,
        campaign_revision: SECOND_EXECUTION_REVISION,
        FIRST_ANALYSIS_REVISION: campaign_revision,
        analysis_revision: FIRST_ANALYSIS_REVISION,
    }
    for revision, expected_parent in expected_parents.items():
        ancestry_line = _git(
            project_root, "rev-list", "--parents", "-n", "1", revision
        ).split()
        if ancestry_line != [revision, expected_parent]:
            raise FreezeBFinalizationError(
                "A, B, C, D, E, F, G, H, and I must be direct single-parent commits"
            )
    expected_trees = {
        implementation_revision: implementation_tree,
        PREREGISTRATION_REVISION: PREREGISTRATION_TREE,
        correction_revision: correction_tree,
        FIRST_EXECUTION_REVISION: FIRST_EXECUTION_TREE,
        admission_correction_revision: admission_correction_tree,
        SECOND_EXECUTION_REVISION: SECOND_EXECUTION_TREE,
        campaign_revision: CAMPAIGN_EXECUTION_TREE,
        FIRST_ANALYSIS_REVISION: FIRST_ANALYSIS_TREE,
        analysis_revision: analysis_tree,
    }
    for revision, expected_tree in expected_trees.items():
        if _git(project_root, "rev-parse", f"{revision}^{{tree}}") != expected_tree:
            raise FreezeBFinalizationError("source commit/tree pair drifted")
    for older, newer in zip(identities, identities[1:]):
        _git(project_root, "merge-base", "--is-ancestor", older, newer)
    observed_diffs = (
        _git_name_status(
            project_root, implementation_revision, PREREGISTRATION_REVISION
        ),
        _git_name_status(
            project_root, PREREGISTRATION_REVISION, correction_revision
        ),
        _git_name_status(
            project_root, correction_revision, FIRST_EXECUTION_REVISION
        ),
        _git_name_status(
            project_root, FIRST_EXECUTION_REVISION, admission_correction_revision
        ),
        _git_name_status(
            project_root, admission_correction_revision, SECOND_EXECUTION_REVISION
        ),
        _git_name_status(
            project_root, SECOND_EXECUTION_REVISION, campaign_revision
        ),
        _git_name_status(
            project_root, campaign_revision, FIRST_ANALYSIS_REVISION
        ),
        _git_name_status(
            project_root, FIRST_ANALYSIS_REVISION, analysis_revision
        ),
    )
    expected_diffs = (
        IMPLEMENTATION_TO_PREREGISTRATION_DIFF,
        PREREGISTRATION_TO_CORRECTION_DIFF,
        CORRECTION_TO_EXECUTION_DIFF,
        FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF,
        ADMISSION_CORRECTION_TO_EXECUTION_DIFF,
        INCIDENT_SOURCE_TO_EXECUTION_DIFF,
        CAMPAIGN_TO_ANALYSIS_CORRECTION_DIFF,
        ANALYSIS_TO_TIME_GRID_CORRECTION_DIFF,
    )
    if observed_diffs != expected_diffs:
        raise FreezeBFinalizationError(
            "source transition differs from the exact corrigendum"
        )
    if _git(project_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise FreezeBFinalizationError("worktree/index must be clean during finalization")
    for relative in REQUIRED_SOURCE_PATHS:
        _git(project_root, "ls-files", "--error-unmatch", "--", relative)
    for revision in (
        PREREGISTRATION_REVISION,
        correction_revision,
        FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        SECOND_EXECUTION_REVISION,
        campaign_revision,
        FIRST_ANALYSIS_REVISION,
        analysis_revision,
    ):
        if _git(project_root, "rev-parse", f"{revision}:{relative_protocol}") != PUBLIC_PROTOCOL_BLOB_OID:
            raise FreezeBFinalizationError("frozen public protocol blob changed after B")
    if (
        sha256_file(public_protocol_path) != PUBLIC_PROTOCOL_FILE_SHA256
        or canonical_json_sha256(load_public_protocol(public_protocol_path))
        != PUBLIC_PROTOCOL_CANONICAL_SHA256
    ):
        raise FreezeBFinalizationError("frozen public protocol hashes drifted")
    for revision in (
        FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        SECOND_EXECUTION_REVISION,
        campaign_revision,
        FIRST_ANALYSIS_REVISION,
        analysis_revision,
    ):
        if (
            _git(project_root, "rev-parse", f"{revision}:{relative_corrigendum}")
            != FIRST_CORRIGENDUM_BLOB_OID
        ):
            raise FreezeBFinalizationError("first corrigendum changed after D")
    if (
        _git(project_root, "hash-object", relative_corrigendum)
        != FIRST_CORRIGENDUM_BLOB_OID
    ):
        raise FreezeBFinalizationError("corrigendum bytes differ from execution commit")
    for revision in (
        SECOND_EXECUTION_REVISION,
        campaign_revision,
        FIRST_ANALYSIS_REVISION,
        analysis_revision,
    ):
        if (
            _git(
                project_root,
                "rev-parse",
                f"{revision}:{relative_admission_corrigendum}",
            )
            != ADMISSION_CORRIGENDUM_BLOB_OID
        ):
            raise FreezeBFinalizationError(
                "admission corrigendum changed after F"
            )
    if (
        _git(project_root, "hash-object", relative_admission_corrigendum)
        != ADMISSION_CORRIGENDUM_BLOB_OID
    ):
        raise FreezeBFinalizationError(
            "admission corrigendum bytes differ from execution commit"
        )
    for revision in (
        campaign_revision,
        FIRST_ANALYSIS_REVISION,
        analysis_revision,
    ):
        if (
            _git(
                project_root,
                "rev-parse",
                f"{revision}:{relative_asset_preflight_corrigendum}",
            )
            != ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID
        ):
            raise FreezeBFinalizationError(
                "asset preflight corrigendum changed after G"
            )
    if (
        _git(
            project_root,
            "hash-object",
            relative_asset_preflight_corrigendum,
        )
        != ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID
    ):
        raise FreezeBFinalizationError(
            "asset preflight corrigendum bytes differ from campaign G"
        )
    if any(
        _git(
            project_root,
            "rev-parse",
            f"{revision}:{relative_runtime_order_corrigendum}",
        )
        != RUNTIME_ORDER_CORRIGENDUM_BLOB_OID
        for revision in (FIRST_ANALYSIS_REVISION, analysis_revision)
    ) or _git(
        project_root, "hash-object", relative_runtime_order_corrigendum
    ) != RUNTIME_ORDER_CORRIGENDUM_BLOB_OID:
        raise FreezeBFinalizationError(
            "runtime order corrigendum changed after H"
        )
    if (
        sha256_file(runtime_order_corrigendum_path)
        != RUNTIME_ORDER_CORRIGENDUM_FILE_SHA256
        or canonical_json_sha256(runtime_order_corrigendum)
        != RUNTIME_ORDER_CORRIGENDUM_CANONICAL_SHA256
    ):
        raise FreezeBFinalizationError(
            "runtime order corrigendum hashes drifted after H"
        )
    if _git(
        project_root,
        "rev-parse",
        f"{analysis_revision}:{relative_time_grid_corrigendum}",
    ) != _git(
        project_root,
        "hash-object",
        relative_time_grid_corrigendum,
    ):
        raise FreezeBFinalizationError(
            "time-grid corrigendum bytes differ from analysis commit I"
        )
    return {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "preregistration_revision": PREREGISTRATION_REVISION,
        "preregistration_tree": PREREGISTRATION_TREE,
        "correction_revision": correction_revision,
        "correction_tree": correction_tree,
        "first_execution_revision": FIRST_EXECUTION_REVISION,
        "first_execution_tree": FIRST_EXECUTION_TREE,
        "admission_correction_revision": admission_correction_revision,
        "admission_correction_tree": admission_correction_tree,
        "incident_source_revision": SECOND_EXECUTION_REVISION,
        "incident_source_tree": SECOND_EXECUTION_TREE,
        "execution_revision": campaign_revision,
        "execution_tree": CAMPAIGN_EXECUTION_TREE,
        "campaign_execution_revision": campaign_revision,
        "campaign_execution_tree": CAMPAIGN_EXECUTION_TREE,
        "first_analysis_revision": FIRST_ANALYSIS_REVISION,
        "first_analysis_tree": FIRST_ANALYSIS_TREE,
        "analysis_revision": analysis_revision,
        "analysis_tree": analysis_tree,
        "corrigendum_file_sha256": sha256_file(corrigendum_path),
        "corrigendum_canonical_sha256": canonical_json_sha256(corrigendum),
        "admission_corrigendum_file_sha256": sha256_file(
            admission_corrigendum_path
        ),
        "admission_corrigendum_canonical_sha256": canonical_json_sha256(
            admission_corrigendum
        ),
        "asset_preflight_corrigendum_file_sha256": sha256_file(
            asset_preflight_corrigendum_path
        ),
        "asset_preflight_corrigendum_canonical_sha256": canonical_json_sha256(
            asset_preflight_corrigendum
        ),
        "runtime_order_corrigendum_file_sha256": sha256_file(
            runtime_order_corrigendum_path
        ),
        "runtime_order_corrigendum_canonical_sha256": canonical_json_sha256(
            runtime_order_corrigendum
        ),
        "time_grid_corrigendum_file_sha256": sha256_file(
            time_grid_corrigendum_path
        ),
        "time_grid_corrigendum_canonical_sha256": canonical_json_sha256(
            time_grid_corrigendum
        ),
        "source_transitions_exact": True,
    }


def _safe_inventory_name(raw: str) -> str:
    value = raw[2:] if raw.startswith("./") else raw
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FreezeBFinalizationError(f"unsafe evidence inventory path: {raw!r}")
    return path.as_posix()


def _verify_hash_inventory(scope: Path, inventory: Path, *, recursive: bool = True) -> None:
    """Verify one sha256sum-style inventory is exact for its declared scope."""

    if is_link_like(inventory) or not inventory.is_file():
        raise FreezeBFinalizationError(f"missing evidence inventory: {inventory}")
    records: list[tuple[str, str]] = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        match = _INVENTORY_RE.fullmatch(line)
        if match is None:
            raise FreezeBFinalizationError(f"malformed evidence inventory line in {inventory.name}")
        records.append((match.group(1), _safe_inventory_name(match.group(2))))
    names = [name for _digest, name in records]
    if names != sorted(names) or len(names) != len(set(names)):
        raise FreezeBFinalizationError("evidence inventory paths must be sorted and unique")
    iterator = scope.rglob("*") if recursive else scope.glob("*")
    actual = sorted(
        path.relative_to(scope).as_posix()
        for path in iterator
        if path.is_file() and path.resolve(strict=True) != inventory.resolve(strict=True)
    )
    if names != actual:
        raise FreezeBFinalizationError(
            f"evidence inventory is not exact: missing={sorted(set(actual) - set(names))}, "
            f"extra={sorted(set(names) - set(actual))}"
        )
    for expected, name in records:
        if sha256_file(scope / PurePosixPath(name)) != expected:
            raise FreezeBFinalizationError(f"evidence hash mismatch: {name}")


def _case_by_id(manifest: object) -> dict[str, object]:
    cases = expand_scenario_cases(manifest)  # type: ignore[arg-type]
    return {case.case_id: case for case in cases}


def _assert_case_trace(
    run: CollectedRun,
    *,
    canonical_case: object,
    backend: Simulator,
    manifest: object,
) -> None:
    case = run.case
    result = run.result
    if case != canonical_case or case.simulator is not backend:
        raise FreezeBFinalizationError("run case differs from the canonical manifest case")
    try:
        validate_diagnostic_run(result, case, manifest)  # type: ignore[arg-type]
    except DiagnosticEvidenceError as exc:
        raise FreezeBFinalizationError(
            f"run failed the canonical diagnostic contract: {case.case_id}: {exc}"
        ) from exc
    for sample in result.samples:
        if any(
            abs(float(value)) > MAX_ABS_CANONICAL_JOINT_POSITION_RAD
            for value in sample.joint_positions.values()
        ):
            raise FreezeBFinalizationError("run exceeds the canonical joint-position sanity bound")
        if any(
            abs(float(value)) > MAX_ABS_CANONICAL_POSITION_TARGET_RAD
            for value in sample.position_targets.values()
        ):
            raise FreezeBFinalizationError("run exceeds the canonical target sanity bound")
        if any(
            abs(float(value)) > MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S
            for value in sample.qvel
        ):
            raise FreezeBFinalizationError("run exceeds the backend velocity sanity bound")
        for pose in sample.frame_poses.values():
            origin_distance = math.sqrt(math.fsum(float(value) ** 2 for value in pose[:3]))
            quaternion_norm = math.sqrt(math.fsum(float(value) ** 2 for value in pose[3:]))
            if origin_distance > MAX_FRAME_ORIGIN_DISTANCE_M:
                raise FreezeBFinalizationError("run exceeds the frame-origin sanity bound")
            if abs(quaternion_norm - 1.0) > UNIT_QUATERNION_NORM_TOLERANCE:
                raise FreezeBFinalizationError("run has a non-unit frame quaternion")
    hand = HandSide(case.hand)
    if (
        case.scenario_id != "small_step"
        or result.backend != backend.value
        or result.scenario_id != "small_step"
        or not result.completed
        or result.status != "completed"
        or result.error is not None
        or result.completed_steps != result.requested_steps
        or len(result.samples) != result.requested_steps + 1
        or result.joint_names != canonical_joint_names(hand)
        or result.frame_names != canonical_frame_names(hand)
    ):
        raise FreezeBFinalizationError("Freeze B run is incomplete or has wrong trace coverage")
    expected_steps = round(0.5 / float(case.dt_s))
    if result.requested_steps != expected_steps:
        raise FreezeBFinalizationError("Freeze B small_step trace has an unexpected duration")
    previous_time = -math.inf
    for index, sample in enumerate(result.samples):
        if sample.step != index or not math.isclose(
            sample.time_s, index * case.dt_s, rel_tol=0.0, abs_tol=1e-12
        ):
            raise FreezeBFinalizationError("Freeze B trace time grid is not canonical")
        if sample.time_s <= previous_time:
            raise FreezeBFinalizationError("Freeze B trace times are not strictly increasing")
        previous_time = sample.time_s
        if (
            set(sample.joint_positions) != set(result.joint_names)
            or set(sample.frame_poses) != set(result.frame_names)
        ):
            # Stable JSON output sorts object keys.  Canonical order is carried
            # by result.joint_names/frame_names and used for every lookup; map
            # objects therefore prove exact coverage, not insertion order.
            raise FreezeBFinalizationError("Freeze B sample name coverage is not canonical")
        if (
            len(sample.qpos) != 22
            or len(sample.qvel) != 22
            or set(sample.position_targets) != set(result.joint_names)
        ):
            raise FreezeBFinalizationError(
                "Freeze B sample generalized state/target coverage is incomplete"
            )
        for name in result.frame_names:
            pose = sample.frame_poses[name]
            if len(pose) != 7 or not math.isclose(
                math.sqrt(math.fsum(value * value for value in pose[3:])),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                raise FreezeBFinalizationError(
                    "Freeze B sample contains an invalid frame quaternion"
                )
        if backend is Simulator.MUJOCO and sample.contact_count != 0:
            raise FreezeBFinalizationError("MuJoCo Freeze B trace observed contact")
        if backend is Simulator.OVPHYSX and sample.contact_count is not None:
            raise FreezeBFinalizationError("OVPhysX contact_count must remain null/unobserved")


def _canonicalize_indexed_records(
    *,
    backend_order: object,
    records: object,
    expected_order: Sequence[str],
    label: str,
    index_key: str,
    require_exact_backend_coverage: bool,
    records_follow_backend_order: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Validate exact IDs/indices, then expose records in expected plan order."""

    expected = tuple(expected_order)
    raw_backend_order = _array(backend_order, f"{label} backend order")
    raw_records = _array(records, f"{label} records")
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(not isinstance(name, str) for name in expected)
        or any(not isinstance(name, str) for name in raw_backend_order)
        or len(set(raw_backend_order)) != len(raw_backend_order)
        or len(raw_records) != len(expected)
        or (
            require_exact_backend_coverage
            and len(raw_backend_order) != len(expected)
        )
    ):
        raise FreezeBFinalizationError(f"{label} coverage is incomplete")
    by_name: dict[str, Mapping[str, Any]] = {}
    indices: set[int] = set()
    for record_position, raw in enumerate(raw_records):
        record = _mapping(raw, f"{label} record")
        canonical_id = record.get("canonical_id")
        backend_name = record.get("backend_name")
        index = record.get(index_key)
        if (
            not isinstance(canonical_id, str)
            or canonical_id not in expected
            or canonical_id in by_name
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(raw_backend_order)
            or index in indices
            or not isinstance(backend_name, str)
            or raw_backend_order[index] != backend_name
            or (records_follow_backend_order and record_position != index)
        ):
            raise FreezeBFinalizationError(f"{label} ID/index binding drifted")
        by_name[canonical_id] = record
        indices.add(index)
    if set(by_name) != set(expected):
        raise FreezeBFinalizationError(f"{label} canonical IDs are incomplete")
    if require_exact_backend_coverage and len(indices) != len(raw_backend_order):
        raise FreezeBFinalizationError(f"{label} backend indices are incomplete")
    return tuple(by_name[name] for name in expected)


def _canonical_mapping_records(
    result: AdapterRunResult,
    *,
    side: str,
    key: str,
    expected_names: Sequence[str],
    unit: str,
) -> tuple[Mapping[str, Any], ...]:
    backend_key = {
        "joint_mapping": "backend_joint_names",
        "frame_mapping": "backend_frame_names",
    }[key]
    records = _canonicalize_indexed_records(
        backend_order=result.provenance.get(backend_key),
        records=result.provenance.get(key),
        expected_order=expected_names,
        label=key,
        index_key="index",
        require_exact_backend_coverage=key == "joint_mapping",
        records_follow_backend_order=False,
    )
    for record in records:
        if (
            record.get("backend") != result.backend
            or record.get("scope") != side
            or record.get("sign") != 1.0
            or record.get("offset") != 0.0
            or record.get("unit") != unit
        ):
            raise FreezeBFinalizationError(f"{key} record drifted")
    return records


def _validate_mapping_provenance(result: AdapterRunResult, *, side: str) -> None:
    _canonical_mapping_records(
        result,
        side=side,
        key="joint_mapping",
        expected_names=canonical_joint_names(side),
        unit="rad",
    )
    _canonical_mapping_records(
        result,
        side=side,
        key="frame_mapping",
        expected_names=canonical_frame_names(side),
        unit="xyz_m_qwxyz",
    )


def _launcher_identity_common(
    value: Mapping[str, Any],
    *,
    protocol_id: str,
    plan_digest: str,
    plan_file_digest: str,
    execution_revision: str,
    execution_tree: str,
) -> None:
    expected = {
        "protocol_id": protocol_id,
        "private_plan_sha256": plan_digest,
        "private_plan_file_sha256": plan_file_digest,
        "source_revision": execution_revision,
        "source_tree": execution_tree,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise FreezeBFinalizationError(f"launcher identity drifted at {key}")


def _validate_local_fresh_process_record(
    value: object,
    *,
    experiment_case_id: str,
    canonical_case_id: str,
) -> str:
    record = _mapping(value, "local fresh-process record")
    _exact_keys(
        record,
        {
            "schema_version",
            "experiment_case_id",
            "canonical_case_id",
            "os_process_identity",
            "fresh_process_id",
        },
        "local fresh-process record",
    )
    if (
        record.get("schema_version") != 1
        or record.get("experiment_case_id") != experiment_case_id
        or record.get("canonical_case_id") != canonical_case_id
    ):
        raise FreezeBFinalizationError("local fresh-process case identity drifted")
    identity = _mapping(record.get("os_process_identity"), "local OS process identity")
    platform = identity.get("platform")
    if platform == "windows":
        _exact_keys(
            identity,
            {"platform", "pid", "process_creation_filetime"},
            "local Windows process identity",
        )
        numeric_fields = (identity.get("pid"), identity.get("process_creation_filetime"))
    elif platform == "posix":
        _exact_keys(
            identity,
            {"platform", "boot_id", "pid", "process_start_ticks"},
            "local POSIX process identity",
        )
        boot_id = identity.get("boot_id")
        if not isinstance(boot_id, str) or not boot_id.strip():
            raise FreezeBFinalizationError("local POSIX boot ID is invalid")
        numeric_fields = (identity.get("pid"), identity.get("process_start_ticks"))
    else:
        raise FreezeBFinalizationError("local process identity platform is invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in numeric_fields
    ):
        raise FreezeBFinalizationError("local OS process identity is invalid")
    fresh_id = record.get("fresh_process_id")
    if (
        not isinstance(fresh_id, str)
        or _SHA256_RE.fullmatch(fresh_id) is None
        or fresh_id != canonical_json_sha256(identity)
    ):
        raise FreezeBFinalizationError("local fresh-process digest is invalid")
    return fresh_id


def _validate_local_worker_process_record(
    value: object,
) -> tuple[dict[str, Any], str, int]:
    record = _mapping(value, "local worker process record")
    _exact_keys(
        record,
        {
            "schema_version",
            "worker_pid",
            "os_process_identity",
            "fresh_process_id",
        },
        "local worker process record",
    )
    if record.get("schema_version") != 1:
        raise FreezeBFinalizationError("local worker process schema is invalid")
    identity = _mapping(
        record.get("os_process_identity"), "local worker OS process identity"
    )
    platform = identity.get("platform")
    if platform == "windows":
        _exact_keys(
            identity,
            {"platform", "pid", "process_creation_filetime"},
            "local worker Windows process identity",
        )
        numeric_fields = (identity.get("pid"), identity.get("process_creation_filetime"))
    elif platform == "posix":
        _exact_keys(
            identity,
            {"platform", "boot_id", "pid", "process_start_ticks"},
            "local worker POSIX process identity",
        )
        boot_id = identity.get("boot_id")
        if not isinstance(boot_id, str) or _BOOT_ID_RE.fullmatch(boot_id) is None:
            raise FreezeBFinalizationError("local worker POSIX boot ID is invalid")
        numeric_fields = (identity.get("pid"), identity.get("process_start_ticks"))
    else:
        raise FreezeBFinalizationError("local worker process platform is invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in numeric_fields
    ):
        raise FreezeBFinalizationError("local worker OS process identity is invalid")
    worker_pid = record.get("worker_pid")
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid != identity.get("pid")
    ):
        raise FreezeBFinalizationError(
            "local worker OS process identity is not bound to worker_pid"
        )
    fresh_id = record.get("fresh_process_id")
    if (
        not isinstance(fresh_id, str)
        or _SHA256_RE.fullmatch(fresh_id) is None
        or fresh_id != canonical_json_sha256(identity)
    ):
        raise FreezeBFinalizationError("local worker fresh-process digest is invalid")
    return dict(identity), fresh_id, worker_pid


def _load_local_runs(
    root: Path,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    manifest: object,
    execution_revision: str,
    execution_tree: str,
    project_root: Path,
    source_identity: Mapping[str, object],
) -> tuple[dict[str, CollectedRun], set[str]]:
    _validate_regular_tree(root, "local evidence")
    if {item.name for item in root.iterdir()} != {"launcher", "cases"}:
        raise FreezeBFinalizationError("local evidence must contain exactly launcher/ and cases/")
    launcher = root / "launcher"
    cases_root = root / "cases"
    expected_launcher_inventory = {
        "attempt-ownership.json",
        "completed.json",
        "evidence.sha256",
        "freeze_b.private-plan.json",
        "fresh-process-ledger.json",
        "gate0.manifest.json",
        "matrix.json",
        "provenance.json",
    }
    if {item.name for item in launcher.iterdir()} != expected_launcher_inventory:
        raise FreezeBFinalizationError(
            "local launcher inventory differs from a completed corrected attempt"
        )
    _verify_hash_inventory(launcher, launcher / "evidence.sha256")
    plan_digest = private_plan_sha256(plan)
    plan_file_digest = sha256_file(plan_path)
    snapshot = launcher / "freeze_b.private-plan.json"
    if sha256_file(snapshot) != plan_file_digest:
        raise FreezeBFinalizationError("local private-plan snapshot bytes drifted")
    load_private_plan(snapshot, expected_sha256=plan_digest)
    provenance = _strict_json(launcher / "provenance.json", "local launcher provenance")
    matrix = _strict_json(launcher / "matrix.json", "local launcher matrix")
    completed = _strict_json(launcher / "completed.json", "local completion record")
    ownership = _strict_json(
        launcher / "attempt-ownership.json", "local attempt ownership"
    )
    _exact_keys(
        ownership,
        {"schema_version", "ownership_token", "session_id", "source_revision"},
        "local attempt ownership",
    )
    _launcher_identity_common(
        provenance,
        protocol_id=str(plan["protocol_id"]),
        plan_digest=plan_digest,
        plan_file_digest=plan_file_digest,
        execution_revision=execution_revision,
        execution_tree=execution_tree,
    )
    if (
        ownership.get("schema_version") != 1
        or not isinstance(ownership.get("ownership_token"), str)
        or _SHA256_RE.fullmatch(str(ownership["ownership_token"])) is None
        or ownership.get("session_id") != provenance.get("session_id")
        or ownership.get("source_revision") != execution_revision
    ):
        raise FreezeBFinalizationError("local attempt ownership record drifted")
    for key in (
        "manifest_file_sha256",
        "manifest_semantic_sha256",
        "asset_commit",
        "asset_git_tree",
        "asset_tree_sha256",
    ):
        expected = {
            "manifest_file_sha256": plan["inputs"]["gate0_manifest_file_sha256"],
            "manifest_semantic_sha256": plan["inputs"]["gate0_manifest_semantic_sha256"],
            "asset_commit": plan["inputs"]["asset_commit"],
            "asset_git_tree": plan["inputs"]["asset_git_tree"],
            "asset_tree_sha256": plan["inputs"]["canonical_lf_asset_tree_sha256"],
        }[key]
        if provenance.get(key) != expected:
            raise FreezeBFinalizationError(f"local launcher {key} drifted")
    if (
        provenance.get("implementation_source_revision")
        != plan["inputs"]["freeze_b_source_revision"]
        or provenance.get("implementation_source_tree")
        != plan["inputs"]["freeze_b_source_tree"]
    ):
        raise FreezeBFinalizationError("local implementation A identity drifted")
    source_fields = {
        "preregistration_source_revision": source_identity[
            "preregistration_revision"
        ],
        "preregistration_source_tree": source_identity["preregistration_tree"],
        "correction_source_revision": source_identity["correction_revision"],
        "correction_source_tree": source_identity["correction_tree"],
        "first_execution_source_revision": source_identity[
            "first_execution_revision"
        ],
        "first_execution_source_tree": source_identity["first_execution_tree"],
        "admission_correction_source_revision": source_identity[
            "admission_correction_revision"
        ],
        "admission_correction_source_tree": source_identity[
            "admission_correction_tree"
        ],
        "asset_preflight_incident_source_revision": source_identity[
            "incident_source_revision"
        ],
        "asset_preflight_incident_source_tree": source_identity[
            "incident_source_tree"
        ],
        "corrigendum_file_sha256": source_identity["corrigendum_file_sha256"],
        "corrigendum_canonical_sha256": source_identity[
            "corrigendum_canonical_sha256"
        ],
        "admission_corrigendum_file_sha256": source_identity[
            "admission_corrigendum_file_sha256"
        ],
        "admission_corrigendum_canonical_sha256": source_identity[
            "admission_corrigendum_canonical_sha256"
        ],
        "asset_preflight_corrigendum_file_sha256": source_identity[
            "asset_preflight_corrigendum_file_sha256"
        ],
        "asset_preflight_corrigendum_canonical_sha256": source_identity[
            "asset_preflight_corrigendum_canonical_sha256"
        ],
        "worker_wrapper_sha256": sha256_file(
            project_root / "scripts/run_freeze_b_mujoco_worker.py"
        ),
        "process_identity_source_sha256": sha256_file(
            project_root / "src/wave_asset_qa/parity/process_identity.py"
        ),
        "fresh_process_identity_authority": (
            "actual_python_worker_kernel_identity_v1"
        ),
    }
    if any(provenance.get(key) != expected for key, expected in source_fields.items()):
        raise FreezeBFinalizationError(
            "local correction/corrigendum identity drifted"
        )
    if any(
        matrix.get(key) != expected or completed.get(key) != expected
        for key, expected in provenance.items()
    ):
        raise FreezeBFinalizationError(
            "local matrix/completion identity differs from launcher provenance"
        )
    if provenance.get("launcher_sha256") != sha256_file(project_root / "scripts/run_ovphysx_freeze_b_local.py"):
        raise FreezeBFinalizationError("local launcher source hash drifted")
    local_rows = list(plan["mujoco_cases"])
    experiment_ids = [str(row["experiment_case_id"]) for row in local_rows]
    canonical_ids = [str(row["canonical_case_id"]) for row in local_rows]
    ledger = _strict_json(
        launcher / "fresh-process-ledger.json", "local fresh-process ledger"
    )
    _exact_keys(
        ledger,
        {"schema_version", "case_count", "records"},
        "local fresh-process ledger",
    )
    ledger_records = _array(ledger.get("records"), "local fresh-process records")
    if ledger.get("schema_version") != 1 or ledger.get("case_count") != 8 or len(ledger_records) != 8:
        raise FreezeBFinalizationError("local fresh-process ledger is incomplete")
    fresh_ids: list[str] = []
    for row, raw_record in zip(local_rows, ledger_records):
        fresh_ids.append(
            _validate_local_fresh_process_record(
                raw_record,
                experiment_case_id=str(row["experiment_case_id"]),
                canonical_case_id=str(row["canonical_case_id"]),
            )
        )
    if len(set(fresh_ids)) != 8:
        raise FreezeBFinalizationError("local fresh-process identities are reused")
    if (
        matrix.get("case_count") != 8
        or matrix.get("experiment_case_ids") != experiment_ids
        or matrix.get("canonical_case_ids") != canonical_ids
        or completed.get("completed_case_count") != 8
        or completed.get("experiment_case_ids") != experiment_ids
        or completed.get("fresh_process_ids") != fresh_ids
    ):
        raise FreezeBFinalizationError("local launcher matrix/completion differs from the plan")
    if {item.name for item in cases_root.iterdir()} != set(experiment_ids):
        raise FreezeBFinalizationError("local case-directory inventory differs from the plan")
    canonical = _case_by_id(manifest)
    runs: dict[str, CollectedRun] = {}
    for row_index, row in enumerate(local_rows):
        experiment_id = str(row["experiment_case_id"])
        canonical_id = str(row["canonical_case_id"])
        case_dir = cases_root / experiment_id
        expected_case_inventory = {
            "evidence.sha256",
            "fresh-process.json",
            "worker-process.json",
            "launcher.command.json",
            "launcher.exitcode.txt",
            "launcher.stderr.log",
            "launcher.stdout.log",
            f"{canonical_id}.run.json",
        }
        if {item.name for item in case_dir.iterdir()} != expected_case_inventory:
            raise FreezeBFinalizationError(
                "local case file inventory differs from the corrected contract"
            )
        _verify_hash_inventory(case_dir, case_dir / "evidence.sha256")
        worker_identity, worker_fresh_id, worker_pid = (
            _validate_local_worker_process_record(
                _strict_json(
                    case_dir / "worker-process.json",
                    "local worker process record",
                )
            )
        )
        case_fresh_record = _strict_json(
            case_dir / "fresh-process.json", "local case fresh-process record"
        )
        if case_fresh_record != ledger_records[row_index]:
            raise FreezeBFinalizationError(
                "local case fresh-process record differs from launcher ledger"
            )
        if (
            case_fresh_record.get("os_process_identity") != worker_identity
            or case_fresh_record.get("fresh_process_id") != worker_fresh_id
        ):
            raise FreezeBFinalizationError(
                "local fresh-process record differs from actual worker sidecar"
            )
        run_path = case_dir / f"{canonical_id}.run.json"
        loaded = load_collected_runs(run_path)
        if len(loaded) != 1 or loaded[0].bundle_root_sha256 is not None:
            raise FreezeBFinalizationError("local raw case must contain one unbundled run")
        run = loaded[0]
        _assert_case_trace(
            run,
            canonical_case=canonical[canonical_id],
            backend=Simulator.MUJOCO,
            manifest=manifest,
        )
        _validate_mapping_provenance(run.result, side=str(row["hand"]))
        command = _strict_json(case_dir / "launcher.command.json", "local case command")
        expected_command = {
            "schema_version": 1,
            "experiment_case_id": experiment_id,
            "canonical_case_id": canonical_id,
            "argv": [
                "<python>",
                "-P",
                "scripts/run_freeze_b_mujoco_worker.py",
                "--identity-output",
                f"cases/{experiment_id}/worker-process.json",
                "--asset-root",
                "<asset-root>",
                "--manifest",
                "launcher/gate0.manifest.json",
                "--output-dir",
                f"cases/{experiment_id}",
                "--session-id",
                provenance["session_id"],
                "--source-revision",
                execution_revision,
                "--case-id",
                canonical_id,
            ],
        }
        if (
            command != expected_command
            or (case_dir / "launcher.exitcode.txt").read_text(encoding="ascii").strip() != "0"
        ):
            raise FreezeBFinalizationError("local case command/exit evidence drifted")
        run_provenance = run.result.provenance
        process_identity = _mapping(
            case_fresh_record.get("os_process_identity"),
            "local case OS process identity",
        )
        process_pid = process_identity.get("pid")
        run_worker_pid = run_provenance.get("worker_pid")
        if (
            isinstance(process_pid, bool)
            or not isinstance(process_pid, int)
            or process_pid <= 0
            or isinstance(run_worker_pid, bool)
            or not isinstance(run_worker_pid, int)
            or run_worker_pid != process_pid
            or run_worker_pid != worker_pid
        ):
            raise FreezeBFinalizationError(
                "local fresh-process identity is not bound to run worker_pid"
            )
        expected_provenance = {
            "manifest_sha256": plan["inputs"]["gate0_manifest_semantic_sha256"],
            "session_id": provenance["session_id"],
            "source_revision": execution_revision,
            "asset_tree_sha256": plan["inputs"]["canonical_lf_asset_tree_sha256"],
            "asset_commit": plan["inputs"]["asset_commit"],
            "asset_git_tree": plan["inputs"]["asset_git_tree"],
        }
        if any(run_provenance.get(key) != expected for key, expected in expected_provenance.items()):
            raise FreezeBFinalizationError("local run provenance drifted")
        runs[experiment_id] = run
    return runs, set(fresh_ids)


_INTERVENTION_KEYS = {
    "schema_version",
    "role",
    "private_plan_sha256",
    "attribute",
    "edit_strategy",
    "source_asset_sha256_before",
    "source_asset_sha256_after_cleanup",
    "source_asset_sha256_unchanged",
    "joint_count",
    "joint_order",
    "exact_manifest_coverage_verified",
    "all_properties_prevalidated_before_write",
    "session_layer_anonymous",
    "original_edit_layer_anonymous",
    "override_layer_anonymous",
    "source_asset_was_edit_target",
    "articulation_uninitialized_before_write",
    "articulation_initialized_after_reset",
    "applied_before_first_simulation_reset",
    "post_write_verified",
    "post_reset_verified",
    "cleanup_status",
    "records",
}
_INTERVENTION_RECORD_KEYS = {
    "canonical_id",
    "prim_path",
    "property_path",
    "observed_pre_value",
    "observed_pre_float32_hex",
    "expected_pre_value",
    "expected_pre_float32_hex",
    "write_value",
    "write_float32_hex",
    "post_write_value",
    "post_write_float32_hex",
    "post_reset_value",
    "post_reset_float32_hex",
    "post_cleanup_float32_hex",
}
_DT_V2_KEYS = {
    "schema_version",
    "dt_claim_scope",
    "requested_dt_s",
    "simulation_cfg_dt_s",
    "simulation_context_config_accessor_dt_s",
    "python_configuration_dt_exact_match",
    "runtime_effective_dt_s",
    "runtime_effective_dt_status",
    "runtime_effective_dt_verified",
    "requested_gravity_m_s2",
    "simulation_cfg_gravity_m_s2",
    "physics_scene_gravity_m_s2",
    "physics_prim_path",
}


def _hand_plan(plan: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    matches = [item for item in plan["hand_plans"] if item["hand"] == side]
    if len(matches) != 1:
        raise FreezeBFinalizationError("private plan hand inventory is ambiguous")
    return matches[0]


def _validate_intervention(
    result: AdapterRunResult,
    *,
    row: Mapping[str, Any],
    hand_plan: Mapping[str, Any],
    plan_digest: str,
) -> bool:
    evidence = _mapping(
        result.provenance.get("legacy_joint_friction_intervention"),
        "legacy friction intervention",
    )
    _exact_keys(evidence, _INTERVENTION_KEYS, "legacy friction intervention")
    role = str(row["role"])
    expected_literals = {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": plan_digest,
        "attribute": "physxJoint:jointFriction",
        "edit_strategy": "anonymous_overlay_as_strongest_session_sublayer",
        "joint_count": 22,
        "exact_manifest_coverage_verified": True,
        "all_properties_prevalidated_before_write": True,
        "session_layer_anonymous": True,
        "original_edit_layer_anonymous": True,
        "override_layer_anonymous": True,
        "source_asset_was_edit_target": False,
        "articulation_uninitialized_before_write": True,
        "articulation_initialized_after_reset": True,
        "applied_before_first_simulation_reset": True,
        "post_write_verified": True,
        "post_reset_verified": True,
        "cleanup_status": "restored_pre_values",
        "source_asset_sha256_unchanged": True,
    }
    valid = all(type(evidence.get(key)) is type(expected) and evidence.get(key) == expected for key, expected in expected_literals.items())
    names = list(hand_plan["joint_names"])
    valid = valid and evidence.get("joint_order") == names
    before = evidence.get("source_asset_sha256_before")
    valid = valid and isinstance(before, str) and _SHA256_RE.fullmatch(before) is not None
    valid = valid and evidence.get("source_asset_sha256_after_cleanup") == before
    records = _array(evidence.get("records"), "intervention records")
    if len(records) != 22:
        raise FreezeBFinalizationError("intervention must contain exactly 22 records")
    write_map = hand_plan[f"{role}_write_values"]
    expected_map = hand_plan["expected_pre_values"]
    prim_paths = hand_plan["joint_prim_paths"]
    for index, raw in enumerate(records):
        record = _mapping(raw, f"intervention record {index}")
        _exact_keys(record, _INTERVENTION_RECORD_KEYS, f"intervention record {index}")
        name = names[index]
        expected_hex = _float32_hex(expected_map[name], f"expected {name}")
        write_hex = _float32_hex(write_map[name], f"write {name}")
        expected_identity = (
            record.get("canonical_id") == name
            and record.get("prim_path") == prim_paths[name]
            and record.get("property_path") == f"{prim_paths[name]}.physxJoint:jointFriction"
        )
        expected_bits = {
            "observed_pre_float32_hex": expected_hex,
            "expected_pre_float32_hex": expected_hex,
            "write_float32_hex": write_hex,
            "post_write_float32_hex": write_hex,
            "post_reset_float32_hex": write_hex,
            "post_cleanup_float32_hex": expected_hex,
        }
        numeric_bits = {
            "observed_pre_value": expected_hex,
            "expected_pre_value": expected_hex,
            "write_value": write_hex,
            "post_write_value": write_hex,
            "post_reset_value": write_hex,
        }
        valid = valid and expected_identity
        valid = valid and all(record.get(key) == expected for key, expected in expected_bits.items())
        valid = valid and all(_float32_hex(record.get(key), f"record {name}.{key}") == expected for key, expected in numeric_bits.items())
    return bool(valid)


def _validate_dt_v2(result: AdapterRunResult, dt_s: float) -> bool:
    config = _mapping(result.provenance.get("simulation_configuration_v2"), "simulation_configuration_v2")
    _exact_keys(config, _DT_V2_KEYS, "simulation_configuration_v2")
    literals = {
        "schema_version": 2,
        "dt_claim_scope": "python_configuration_only_not_compiled_runtime",
        "python_configuration_dt_exact_match": True,
        "runtime_effective_dt_s": None,
        "runtime_effective_dt_status": "not_exposed_by_pinned_kitless_ovphysx",
        "runtime_effective_dt_verified": False,
        "physics_prim_path": "/physicsScene",
    }
    valid = all(type(config.get(key)) is type(expected) and config.get(key) == expected for key, expected in literals.items())
    for key in ("requested_dt_s", "simulation_cfg_dt_s", "simulation_context_config_accessor_dt_s"):
        valid = valid and math.isclose(_finite(config.get(key), key), dt_s, rel_tol=0.0, abs_tol=1e-12)
    for key in ("requested_gravity_m_s2", "simulation_cfg_gravity_m_s2", "physics_scene_gravity_m_s2"):
        gravity = config.get(key)
        valid = valid and isinstance(gravity, list) and len(gravity) == 3
        if isinstance(gravity, list):
            valid = valid and all(abs(_finite(value, key)) <= 1e-12 for value in gravity)
    return bool(valid)


def _validate_position_target_readback(
    result: AdapterRunResult,
    *,
    case: object,
    manifest: object,
    expected_joint_names: Sequence[str],
    max_abs_error_rad: float,
) -> tuple[bool, float]:
    """Validate the OVPhysX target binding and return its aggregate error."""

    limit = _finite(max_abs_error_rad, "position target readback limit")
    if limit != 1e-6:
        raise FreezeBFinalizationError(
            "position target readback limit must remain preregistered at 1e-6 rad"
        )
    scenario = manifest.scenario(case.scenario_id)  # type: ignore[attr-defined]
    hand = manifest.hand(case.hand)  # type: ignore[attr-defined]
    expected_names = tuple(expected_joint_names)
    if expected_names != tuple(hand.joint_names):
        raise FreezeBFinalizationError(
            "position target plan order differs from the canonical manifest"
        )
    provenance = result.provenance
    count = provenance.get("position_target_readback_count")
    error = provenance.get("position_target_readback_max_abs_error_rad")
    if isinstance(count, bool) or not isinstance(count, int):
        raise FreezeBFinalizationError("position target readback count is malformed")
    observed_error = _finite(error, "position target readback max error")
    if observed_error < 0.0:
        raise FreezeBFinalizationError("position target readback max error is negative")
    values = provenance.get("position_target_readback_values_rad")
    backend_joint_names = _array(
        provenance.get("backend_joint_names"),
        "position target backend joint order",
    )
    if not isinstance(values, list) or len(values) != len(backend_joint_names):
        raise FreezeBFinalizationError("position target readback coverage is incomplete")
    backend_values = [
        _finite(value, "position target readback value") for value in values
    ]
    mapping_records = _canonical_mapping_records(
        result,
        side=HandSide(case.hand).value,  # type: ignore[attr-defined]
        key="joint_mapping",
        expected_names=expected_names,
        unit="rad",
    )
    canonical_values = [
        backend_values[int(record["index"])] for record in mapping_records
    ]
    expected = canonical_position_targets(
        scenario,
        expected_names,
        step_index=result.requested_steps,
        dt_s=float(case.dt_s),
    )
    expected_steps = round(float(scenario.duration_s) / float(case.dt_s))
    valid = (
        result.backend == Simulator.OVPHYSX.value
        and provenance.get("position_target_readback_verified") is True
        and provenance.get("position_target_readback_source")
        == "articulation_data_joint_pos_target_torch"
        and count >= expected_steps + 2
        and observed_error <= limit
        and all(
            math.isclose(
                value,
                float(expected[name]),
                rel_tol=0.0,
                abs_tol=limit,
            )
            for name, value in zip(expected_names, canonical_values)
        )
        and provenance.get("position_target_nonzero_readback_observed") is True
        and provenance.get("zero_velocity_target_verified") is True
        and provenance.get("zero_feedforward_effort_target_verified") is True
    )
    return bool(valid), observed_error


def _effective_snapshot(result: AdapterRunResult) -> Mapping[str, Any]:
    snapshot = _mapping(result.provenance.get("effective_parameter_readback"), "effective parameter readback")
    if (
        snapshot.get("capture_phase") != "post_reset_pre_trace_step"
        or snapshot.get("step_index") != 0
        or not isinstance(snapshot.get("backend_joint_dynamics"), Mapping)
        or not isinstance(snapshot.get("physics_solver_contract"), Mapping)
    ):
        raise FreezeBFinalizationError("effective parameter readback phase is invalid")
    dynamics = _mapping(snapshot["backend_joint_dynamics"], "backend joint dynamics")
    if (
        dynamics.get("runtime_effective_readback") is not True
        or dynamics.get("joint_count") != 22
        or dynamics.get("capture_phase") != "post_reset_pre_trace_step"
        or len(_array(dynamics.get("records"), "backend joint records")) != 22
    ):
        raise FreezeBFinalizationError("effective joint readback is incomplete")
    return snapshot


def _canonical_runtime_joint_records(
    result: AdapterRunResult,
    *,
    side: str,
    expected_joint_names: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Join runtime records to canonical IDs through the frozen mapping."""

    snapshot = _effective_snapshot(result)
    dynamics = _mapping(snapshot["backend_joint_dynamics"], "backend dynamics")
    backend_order = _array(
        dynamics.get("joint_order"), "runtime backend joint order"
    )
    provenance_backend_order = _array(
        result.provenance.get("backend_joint_names"),
        "mapping backend joint order",
    )
    if backend_order != provenance_backend_order:
        raise FreezeBFinalizationError(
            "runtime joint order differs from mapping backend joint order"
        )
    mapping_records = _canonical_mapping_records(
        result,
        side=side,
        key="joint_mapping",
        expected_names=expected_joint_names,
        unit="rad",
    )
    runtime_records = _canonicalize_indexed_records(
        backend_order=backend_order,
        records=dynamics.get("records"),
        expected_order=expected_joint_names,
        label="runtime joint readback",
        index_key="backend_index",
        require_exact_backend_coverage=True,
        records_follow_backend_order=True,
    )
    for mapping_record, runtime_record in zip(
        mapping_records, runtime_records, strict=True
    ):
        if (
            runtime_record.get("canonical_id")
            != mapping_record.get("canonical_id")
            or runtime_record.get("backend_name")
            != mapping_record.get("backend_name")
            or runtime_record.get("backend_index")
            != mapping_record.get("index")
        ):
            raise FreezeBFinalizationError(
                "runtime joint record differs from the canonical mapping inverse"
            )
    return runtime_records


def _validate_intervention_readback(
    result: AdapterRunResult,
    *,
    row: Mapping[str, Any],
    hand_plan: Mapping[str, Any],
) -> bool:
    names = list(hand_plan["joint_names"])
    records = _canonical_runtime_joint_records(
        result,
        side=str(hand_plan["hand"]),
        expected_joint_names=names,
    )
    write_map = hand_plan[f"{row['role']}_write_values"]
    valid = True
    for name, raw in zip(names, records):
        record = _mapping(raw, f"backend joint {name}")
        legacy = _mapping(record.get("legacy_joint_friction_usd"), f"legacy friction {name}")
        expected_hex = _float32_hex(write_map[name], f"write readback {name}")
        valid = valid and record.get("canonical_id") == name
        valid = valid and legacy.get("attribute") == "physxJoint:jointFriction"
        valid = valid and legacy.get("has_authored_value_opinion") is True
        valid = valid and _float32_hex(legacy.get("authored_value"), name) == expected_hex
        valid = valid and _float32_hex(legacy.get("resolved_value"), name) == expected_hex
    return bool(valid)


def _normalized_non_target_contract(result: AdapterRunResult) -> object:
    snapshot = deepcopy(dict(_effective_snapshot(result)))
    dynamics = _mapping(snapshot["backend_joint_dynamics"], "backend joint dynamics")
    for raw in _array(dynamics["records"], "backend joint records"):
        record = _mapping(raw, "backend joint record")
        legacy = _mapping(record["legacy_joint_friction_usd"], "legacy friction")
        legacy["authored_value"] = "<freeze-b-target>"
        legacy["resolved_value"] = "<freeze-b-target>"
        # The target family includes both the authored legacy scalar and any
        # runtime friction slots that the backend may derive from it.  A zero
        # treatment changing these slots is an observation, not non-target
        # configuration drift.
        record["friction_properties_raw"] = "<freeze-b-target-runtime>"
        record["controller_static_friction"] = "<freeze-b-target-runtime>"
        record["controller_dynamic_friction"] = "<freeze-b-target-runtime>"
        record["controller_viscous_friction"] = "<freeze-b-target-runtime>"
        exact = record.get("controller_binding_exact_match")
        if isinstance(exact, dict):
            for key in ("static_friction", "dynamic_friction", "viscous_friction"):
                if key in exact:
                    exact[key] = "<freeze-b-target-runtime>"
    aggregate_exact = dynamics.get(
        "controller_joint_dynamics_binding_exact_match"
    )
    if isinstance(aggregate_exact, dict):
        for key in ("overall", "static_friction", "dynamic_friction", "viscous_friction"):
            if key in aggregate_exact:
                aggregate_exact[key] = "<freeze-b-target-runtime>"
    provenance = result.provenance
    return {
        "effective_parameter_readback": snapshot,
        "joint_mapping": provenance.get("joint_mapping"),
        "frame_mapping": provenance.get("frame_mapping"),
        "simulation_configuration_v2": provenance.get("simulation_configuration_v2"),
        "actuation_contract_version": provenance.get("actuation_contract_version"),
        "actuator_model": provenance.get("actuator_model"),
        "control_path": provenance.get("control_path"),
        "controller_dof_stiffness": provenance.get("controller_dof_stiffness"),
        "controller_dof_damping": provenance.get("controller_dof_damping"),
        "controller_dof_effort_limit": provenance.get("controller_dof_effort_limit"),
        "controller_dof_effort_limit_sim": provenance.get("controller_dof_effort_limit_sim"),
        "backend_dof_stiffness": provenance.get("backend_dof_stiffness"),
        "backend_dof_damping": provenance.get("backend_dof_damping"),
    }


def _runtime_friction_signature(
    result: AdapterRunResult,
    *,
    side: str,
    expected_joint_names: Sequence[str],
) -> str:
    """Validate and hash the target runtime family without publishing values.

    The treatment is allowed to change these values.  This routine therefore
    checks identity, complete coverage, finite numeric values, and truthful
    backend/controller comparison flags, but deliberately does not require a
    treatment value to equal the frozen R1 readback.
    """

    snapshot = _effective_snapshot(result)
    dynamics = _mapping(snapshot["backend_joint_dynamics"], "backend dynamics")
    expected = tuple(expected_joint_names)
    records = _canonical_runtime_joint_records(
        result,
        side=side,
        expected_joint_names=expected,
    )
    semantics = _mapping(dynamics.get("friction_semantics"), "runtime friction semantics")
    if semantics.get("raw_slot_order") != ["static", "dynamic", "viscous"]:
        raise FreezeBFinalizationError("runtime friction slot order drifted")

    payload: list[dict[str, object]] = []
    per_field_matches = {
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }
    for expected_name, raw in zip(expected, records):
        record = _mapping(raw, f"runtime friction {expected_name}")
        raw_slots = record.get("friction_properties_raw")
        if not isinstance(raw_slots, list) or len(raw_slots) != 3:
            raise FreezeBFinalizationError("runtime friction slots must contain three values")
        slots = [_finite(value, "runtime friction slot") for value in raw_slots]
        controller = [
            _finite(record.get(key), f"runtime friction {key}")
            for key in (
                "controller_static_friction",
                "controller_dynamic_friction",
                "controller_viscous_friction",
            )
        ]
        if record.get("canonical_id") != expected_name:
            raise FreezeBFinalizationError("runtime friction joint identity drifted")
        exact = _mapping(
            record.get("controller_binding_exact_match"),
            f"runtime friction binding flags {expected_name}",
        )
        _exact_keys(
            exact,
            {"armature", "static_friction", "dynamic_friction", "viscous_friction"},
            f"runtime friction binding flags {expected_name}",
        )
        for index, key in enumerate(per_field_matches):
            observed = exact.get(key)
            expected_match = slots[index] == controller[index]
            if type(observed) is not bool or observed is not expected_match:
                raise FreezeBFinalizationError(
                    f"runtime friction binding flag is inconsistent: {expected_name}.{key}"
                )
            per_field_matches[key] &= expected_match
        payload.append(
            {
                "canonical_id": expected_name,
                "runtime_slots": slots,
                "controller_buffers": controller,
            }
        )

    aggregate = _mapping(
        dynamics.get("controller_joint_dynamics_binding_exact_match"),
        "aggregate runtime binding flags",
    )
    _exact_keys(
        aggregate,
        {"overall", "armature", "static_friction", "dynamic_friction", "viscous_friction"},
        "aggregate runtime binding flags",
    )
    for key, expected_match in per_field_matches.items():
        if type(aggregate.get(key)) is not bool or aggregate.get(key) is not expected_match:
            raise FreezeBFinalizationError(f"aggregate runtime friction flag drifted: {key}")
    expected_overall = bool(aggregate.get("armature")) and all(per_field_matches.values())
    if type(aggregate.get("armature")) is not bool or aggregate.get("overall") is not expected_overall:
        raise FreezeBFinalizationError("aggregate runtime binding overall flag drifted")
    return canonical_json_sha256(payload)


def _r1_runtime_friction_signature(
    instance: Mapping[str, Any],
    *,
    expected_joint_names: Sequence[str] | None = None,
) -> str:
    records = _array(instance.get("dof_records"), "R1 DOF records")
    if len(records) != 22:
        raise FreezeBFinalizationError("R1 runtime friction coverage is incomplete")
    by_name: dict[str, dict[str, object]] = {}
    for raw in records:
        record = _mapping(raw, "R1 DOF record")
        mapping = _mapping(record.get("mapping"), "R1 mapping")
        name = mapping.get("canonical_joint_name")
        if not isinstance(name, str) or name in by_name:
            raise FreezeBFinalizationError("R1 runtime friction joint identity drifted")
        runtime = _mapping(
            _mapping(record.get("friction"), "R1 friction").get(
                "runtime_effective_value"
            ),
            "R1 runtime friction",
        )
        slots = _mapping(
            runtime.get("dof_friction_properties_binding"),
            "R1 runtime friction slots",
        )
        buffers = _mapping(
            runtime.get("idealpd_controller_buffers"),
            "R1 controller friction buffers",
        )
        _exact_keys(slots, {"static", "dynamic", "viscous"}, "R1 runtime friction slots")
        _exact_keys(buffers, {"static", "dynamic", "viscous"}, "R1 controller friction buffers")
        by_name[name] = {
            "canonical_id": name,
            "runtime_slots": [
                _finite(slots.get(key), f"R1 runtime friction {key}")
                for key in ("static", "dynamic", "viscous")
            ],
            "controller_buffers": [
                _finite(buffers.get(key), f"R1 controller friction {key}")
                for key in ("static", "dynamic", "viscous")
            ],
        }
    expected = (
        tuple(expected_joint_names)
        if expected_joint_names is not None
        else tuple(sorted(by_name))
    )
    if len(expected) != 22 or set(by_name) != set(expected):
        raise FreezeBFinalizationError("R1 runtime friction coverage is incomplete")
    return canonical_json_sha256([by_name[name] for name in expected])


def _summarize_runtime_friction_observation(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_signatures: Mapping[str, str],
    r1_signatures: Mapping[str, str],
) -> tuple[bool, dict[str, object], dict[str, object]]:
    """Return a gate plus public/private summaries of target-family readback.

    Only sham-to-R1 equality and exact fresh-process repeatability are gates.
    Whether the zero treatment changes the target runtime slots is reported as
    a descriptive observation and cannot by itself make the study valid or
    invalid.
    """

    expected_ids = {str(row["experiment_case_id"]) for row in rows}
    if set(case_signatures) != expected_ids or set(r1_signatures) != {"left", "right"}:
        raise FreezeBFinalizationError("runtime friction signature inventory drifted")
    groups: dict[tuple[str, str, str], list[str]] = {}
    zero_changed_flags: list[bool] = []
    sham_exact_flags: list[bool] = []
    for row in rows:
        experiment_id = str(row["experiment_case_id"])
        hand = str(row["hand"])
        variant = str(row["timestep_variant"])
        role = str(row["role"])
        if hand not in r1_signatures or role not in {"sham", "zero"}:
            raise FreezeBFinalizationError("runtime friction case identity drifted")
        signature = case_signatures[experiment_id]
        if not isinstance(signature, str) or _SHA256_RE.fullmatch(signature) is None:
            raise FreezeBFinalizationError("runtime friction signature is malformed")
        groups.setdefault((hand, variant, role), []).append(signature)
        if role == "sham":
            sham_exact_flags.append(signature == r1_signatures[hand])
        else:
            zero_changed_flags.append(signature != r1_signatures[hand])
    fresh_repeat_exact = all(
        len(values) == 2 and values[0] == values[1] for values in groups.values()
    )
    sham_exact_r1 = bool(sham_exact_flags) and all(sham_exact_flags)
    gate = sham_exact_r1 and fresh_repeat_exact
    public = {
        "schema_version": 1,
        "sham_exact_r1": sham_exact_r1,
        "fresh_repeat_exact": fresh_repeat_exact,
        "zero_changed_case_count_from_r1": sum(zero_changed_flags),
        "zero_case_count": len(zero_changed_flags),
        "zero_any_changed_from_r1": any(zero_changed_flags),
        "zero_change_is_descriptive_not_validity_failure": True,
        "values_published": False,
    }
    private = {
        **public,
        "r1_runtime_signatures": dict(sorted(r1_signatures.items())),
        "case_runtime_signatures": dict(sorted(case_signatures.items())),
    }
    return gate, public, private


def _validate_remote_case_runtime_envelope(
    provenance: Mapping[str, Any],
    *,
    launcher: Mapping[str, Any],
) -> None:
    selected_uuid = launcher.get("selected_gpu_uuid")
    selected_name = launcher.get("selected_gpu_name")
    selected_driver = launcher.get("driver_version")
    if (
        not isinstance(selected_uuid, str)
        or _GPU_UUID_RE.fullmatch(selected_uuid) is None
        or not isinstance(selected_name, str)
        or EXPECTED_GPU_NAME_FRAGMENT not in selected_name
        or selected_driver != EXPECTED_DRIVER_VERSION
    ):
        raise FreezeBFinalizationError("remote launcher GPU identity is invalid")
    if (
        provenance.get("gpu_uuid") != selected_uuid
        or provenance.get("gpu_name") != selected_name
        or provenance.get("driver_version") != selected_driver
        or provenance.get("forbidden_modules") != []
        or provenance.get("constraints")
        != {
            "kitless": True,
            "headless": True,
            "renderer": False,
            "camera": False,
            "fresh_process": True,
            "scenario_id": "small_step",
        }
    ):
        raise FreezeBFinalizationError(
            "remote case GPU identity or execution constraints differ from launcher"
        )


def _validate_remote_process_identity(provenance: Mapping[str, Any]) -> str:
    """Recompute one private worker identity and bind it to its run process."""

    identity = _mapping(
        provenance.get("os_process_identity"),
        "remote OS process identity",
    )
    _exact_keys(
        identity,
        {"boot_id", "hostname", "pid", "process_start_ticks"},
        "remote OS process identity",
    )
    boot_id = identity.get("boot_id")
    hostname = identity.get("hostname")
    pid = identity.get("pid")
    start_ticks = identity.get("process_start_ticks")
    worker_pid = provenance.get("worker_pid")
    if (
        not isinstance(boot_id, str)
        or _BOOT_ID_RE.fullmatch(boot_id) is None
        or hostname != EXPECTED_REMOTE_HOSTNAME
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks < 0
        or isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid != pid
    ):
        raise FreezeBFinalizationError(
            "remote OS process identity is not bound to worker_pid"
        )
    fresh = provenance.get("fresh_process_id")
    if (
        not isinstance(fresh, str)
        or _SHA256_RE.fullmatch(fresh) is None
        or canonical_json_sha256(identity) != fresh
    ):
        raise FreezeBFinalizationError(
            "remote fresh-process hash differs from OS process identity"
        )
    return fresh


def _load_remote_runs(
    root: Path,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    manifest: object,
    execution_revision: str,
    execution_tree: str,
    project_root: Path,
    public_protocol_file_sha256: str,
    public_protocol_canonical_sha256: str,
) -> tuple[dict[str, CollectedRun], Mapping[str, Any], set[str]]:
    _validate_regular_tree(root, "remote evidence")
    if {item.name for item in root.iterdir()} != {"launcher", "cases"}:
        raise FreezeBFinalizationError("remote evidence must contain exactly launcher/ and cases/")
    _verify_hash_inventory(root, root / "launcher/evidence.sha256")
    launcher = root / "launcher"
    status = _strict_json(launcher / "status.json", "remote launcher status")
    provenance = _strict_json(launcher / "provenance.json", "remote launcher provenance")
    plan_digest = private_plan_sha256(plan)
    plan_file_digest = sha256_file(plan_path)
    _launcher_identity_common(
        provenance,
        protocol_id=str(plan["protocol_id"]),
        plan_digest=plan_digest,
        plan_file_digest=plan_file_digest,
        execution_revision=execution_revision,
        execution_tree=execution_tree,
    )
    if (
        status.get("status") != "collected"
        or status.get("scientific_verdict") != "pending_local_validation"
        or status.get("exit_code") != 0
        or status.get("completed_case_count") != 16
        or status.get("source_revision") != execution_revision
        or status.get("source_tree") != execution_tree
        or status.get("private_plan_sha256") != plan_digest
        or status.get("formal_gate0_status_unchanged") != "DIVERGENT"
        or status.get("formal_gate0_pass_ready_unchanged") is not False
    ):
        raise FreezeBFinalizationError("remote launcher status is not a complete collection")
    if (
        provenance.get("case_count") != 16
        or provenance.get("headless") is not True
        or provenance.get("kitless") is not True
        or provenance.get("renderer") is not False
        or provenance.get("camera") is not False
        or provenance.get("driver_version") != EXPECTED_DRIVER_VERSION
        or not isinstance(provenance.get("selected_gpu_uuid"), str)
        or _GPU_UUID_RE.fullmatch(str(provenance.get("selected_gpu_uuid"))) is None
        or EXPECTED_GPU_NAME_FRAGMENT not in str(provenance.get("selected_gpu_name"))
        or provenance.get("maximum_gpu_hours") != 0.5
        or provenance.get("environment") != EXPECTED_REMOTE_ENVIRONMENT
        or provenance.get("launcher_sha256") != sha256_file(project_root / "scripts/run_ovphysx_freeze_b_remote.sh")
        or provenance.get("worker_sha256") != sha256_file(project_root / "scripts/probe_ovphysx_freeze_b.py")
        or provenance.get("public_protocol_file_sha256")
        != public_protocol_file_sha256
        or provenance.get("public_protocol_canonical_sha256")
        != public_protocol_canonical_sha256
        or provenance.get("implementation_source_revision")
        != plan["inputs"]["freeze_b_source_revision"]
        or provenance.get("implementation_source_tree")
        != plan["inputs"]["freeze_b_source_tree"]
        or provenance.get("deployment_source_revision") != execution_revision
        or provenance.get("deployment_source_tree") != execution_tree
        or provenance.get("implementation_to_deployment_relation_scope")
        != "locally_audited_before_gitless_remote_deployment_not_recomputed_here"
    ):
        raise FreezeBFinalizationError("remote launcher environment/source contract drifted")
    for key in ("source_archive_sha256", "source_snapshot_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise FreezeBFinalizationError(f"remote launcher {key} is malformed")
    rows = list(plan["ovphysx_cases"])
    experiment_ids = [str(row["experiment_case_id"]) for row in rows]
    expected_case_lines = [
        "|".join(
            str(row[key])
            for key in (
                "experiment_case_id",
                "canonical_case_id",
                "role",
                "hand",
                "timestep_variant",
                "dt_s",
                "repeat_index",
            )
        )
        for row in rows
    ]
    if (launcher / "cases.txt").read_text(encoding="utf-8").splitlines() != expected_case_lines:
        raise FreezeBFinalizationError("remote cases.txt differs from private run order")
    cases_root = root / "cases"
    if {item.name for item in cases_root.iterdir()} != set(experiment_ids):
        raise FreezeBFinalizationError("remote case-directory inventory differs from the plan")
    canonical = _case_by_id(manifest)
    runs: dict[str, CollectedRun] = {}
    fresh_ids: dict[str, str] = {}
    for row in rows:
        experiment_id = str(row["experiment_case_id"])
        canonical_id = str(row["canonical_case_id"])
        case_dir = cases_root / experiment_id
        _verify_hash_inventory(case_dir, case_dir / "evidence.sha256")
        raw = _strict_json(case_dir / f"{experiment_id}.json", "remote worker result")
        result = adapter_run_result_from_dict(raw)
        run = CollectedRun(case=canonical[canonical_id], result=result)
        _assert_case_trace(
            run,
            canonical_case=canonical[canonical_id],
            backend=Simulator.OVPHYSX,
            manifest=manifest,
        )
        _validate_mapping_provenance(result, side=str(row["hand"]))
        p = result.provenance
        expected = {
            "freeze_b_protocol_id": plan["protocol_id"],
            "private_plan_sha256": plan_digest,
            "private_plan_file_sha256": plan_file_digest,
            "experiment_case_id": experiment_id,
            "canonical_case_id": canonical_id,
            "freeze_b_role": row["role"],
            "hand": row["hand"],
            "timestep_variant": row["timestep_variant"],
            "repeat_index": row["repeat_index"],
            "manifest_sha256": plan["inputs"]["gate0_manifest_semantic_sha256"],
            "manifest_file_sha256": plan["inputs"]["gate0_manifest_file_sha256"],
            "session_id": provenance["session_id"],
            "source_revision": execution_revision,
            "source_tree": execution_tree,
            "source_snapshot_sha256": provenance["source_snapshot_sha256"],
            "asset_tree_sha256": plan["inputs"]["canonical_lf_asset_tree_sha256"],
            "asset_commit": plan["inputs"]["asset_commit"],
            "asset_git_tree": plan["inputs"]["asset_git_tree"],
            "worker_script_sha256": provenance["worker_sha256"],
            "device": "cuda:0",
            "public_protocol_file_sha256": public_protocol_file_sha256,
            "public_protocol_canonical_sha256": public_protocol_canonical_sha256,
            "deployment_source_revision": execution_revision,
            "deployment_source_tree": execution_tree,
            "implementation_source_revision": plan["inputs"][
                "freeze_b_source_revision"
            ],
            "implementation_source_tree": plan["inputs"][
                "freeze_b_source_tree"
            ],
            "implementation_to_deployment_relation_scope": (
                "locally_audited_before_gitless_remote_deployment_not_recomputed_here"
            ),
        }
        if any(p.get(key) != value for key, value in expected.items()):
            raise FreezeBFinalizationError(f"remote run provenance drifted: {experiment_id}")
        _validate_remote_case_runtime_envelope(p, launcher=provenance)
        if p.get("contact_check_performed") is not False:
            raise FreezeBFinalizationError("OVPhysX contact observation boundary was not disclosed")
        fresh = _validate_remote_process_identity(p)
        if fresh in fresh_ids.values():
            raise FreezeBFinalizationError("remote fresh-process identities are invalid or reused")
        fresh_ids[experiment_id] = fresh
        if (case_dir / "worker.exit-code.txt").read_text(encoding="ascii").strip() != "0":
            raise FreezeBFinalizationError("remote worker exit code is nonzero")
        runs[experiment_id] = run
    fresh_lines = (launcher / "fresh-process-ids.txt").read_text(encoding="ascii").splitlines()
    if fresh_lines != [f"{case_id}|{fresh_ids[case_id]}" for case_id in experiment_ids]:
        raise FreezeBFinalizationError("fresh-process launcher ledger differs from worker evidence")
    return runs, provenance, set(fresh_ids.values())


def _load_formal_runs(
    root: Path,
    *,
    plan: Mapping[str, Any],
    manifest: object,
) -> dict[tuple[str, str], CollectedRun]:
    verification = verify_exact_bundle(root)
    formal_root_sha256 = str(verification["root_sha256"])
    if formal_root_sha256 != plan["inputs"]["formal_gate0_bundle_root_sha256"]:
        raise FreezeBFinalizationError("formal Gate 0 bundle root drifted")
    copied_manifest = root / "manifest/gate0.json"
    if sha256_file(copied_manifest) != plan["inputs"]["gate0_manifest_file_sha256"]:
        raise FreezeBFinalizationError("formal Gate 0 manifest file drifted")
    if manifest_sha256(load_manifest(copied_manifest)) != plan["inputs"]["gate0_manifest_semantic_sha256"]:
        raise FreezeBFinalizationError("formal Gate 0 manifest semantics drifted")
    formal_finalization = _strict_json(root / "results/finalization.json", "formal finalization")
    formal_report = _strict_json(root / "results/report.json", "formal report")
    if (
        formal_finalization.get("comparison_status") != "divergent"
        or _mapping(formal_report.get("summary"), "formal report summary").get("comparison_status") != "divergent"
    ):
        raise FreezeBFinalizationError("formal Gate 0 is not the frozen DIVERGENT result")
    canonical = _case_by_id(manifest)
    runs: dict[tuple[str, str], CollectedRun] = {}
    for backend in (Simulator.MUJOCO, Simulator.OVPHYSX):
        for row in plan[f"{backend.value}_cases"]:
            case_id = str(row["canonical_case_id"])
            path = root / "evidence" / backend.value / "cases" / case_id / f"{case_id}.run.json"
            loaded = load_collected_runs(path)
            if len(loaded) != 1:
                raise FreezeBFinalizationError("formal case inventory is ambiguous")
            raw_run = loaded[0]
            run = CollectedRun(
                case=raw_run.case,
                result=raw_run.result,
                bundle_root_sha256=formal_root_sha256,
            )
            _assert_case_trace(
                run,
                canonical_case=canonical[case_id],
                backend=backend,
                manifest=manifest,
            )
            runs[(backend.value, case_id)] = run
    if len(runs) != 16:
        raise FreezeBFinalizationError("formal small_step subset must contain 16 unique runs")
    return runs


def _load_r1(
    root: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    verification = verify_exact_bundle(root)
    if verification["root_sha256"] != plan["inputs"]["readback_bundle_root_sha256"]:
        raise FreezeBFinalizationError("R1 readback bundle root drifted")
    summary_path = root / "results/summary.json"
    if sha256_file(summary_path) != plan["inputs"]["readback_summary_sha256"]:
        raise FreezeBFinalizationError("R1 public summary hash drifted")
    readback_cases = _mapping(plan["inputs"]["readback_case_sha256"], "R1 case hashes")
    for case_id, expected in readback_cases.items():
        path = root / "private/remote-evidence/cases" / case_id / f"{case_id}.json"
        if sha256_file(path) != expected:
            raise FreezeBFinalizationError(f"R1 case payload hash drifted: {case_id}")
    payload = _strict_json(root / "private/readback.json", "R1 private readback")
    if payload.get("status") != EXPECTED_R1_STATUS:
        raise FreezeBFinalizationError("R1 private readback is not READBACK_VALID")
    instances = _array(payload.get("instances"), "R1 instances")
    if len(instances) != 4:
        raise FreezeBFinalizationError("R1 must contain four fresh readback instances")
    by_hand: dict[str, list[Mapping[str, Any]]] = {"left": [], "right": []}
    for raw in instances:
        instance = _mapping(raw, "R1 instance")
        side = instance.get("side")
        if side not in by_hand:
            raise FreezeBFinalizationError("R1 instance side is invalid")
        by_hand[str(side)].append(instance)
    result: dict[str, Mapping[str, Any]] = {}
    for side, values in by_hand.items():
        if len(values) != 2:
            raise FreezeBFinalizationError("R1 must contain two instances per hand")
        comparable = (values[0].get("dof_records"), values[0].get("solver_runtime"))
        if comparable != (values[1].get("dof_records"), values[1].get("solver_runtime")):
            raise FreezeBFinalizationError("R1 per-hand readback is not exact-repeatable")
        result[side] = values[0]
    return result


def _r1_bridge(
    run: CollectedRun,
    *,
    r1_instance: Mapping[str, Any],
    row: Mapping[str, Any],
    hand_plan: Mapping[str, Any],
) -> bool:
    role = row.get("role")
    if role not in {"sham", "zero"}:
        raise FreezeBFinalizationError("R1 bridge role is invalid")
    expected_names = tuple(hand_plan["joint_names"])
    new_records = _canonical_runtime_joint_records(
        run.result,
        side=str(hand_plan["hand"]),
        expected_joint_names=expected_names,
    )
    old_records = _array(r1_instance.get("dof_records"), "R1 DOF records")
    if len(new_records) != 22 or len(old_records) != 22:
        return False
    old_by_name: dict[str, Mapping[str, Any]] = {}
    for old_raw in old_records:
        old = _mapping(old_raw, "R1 DOF record")
        mapping = _mapping(old.get("mapping"), "R1 mapping")
        canonical_name = mapping.get("canonical_joint_name")
        if (
            not isinstance(canonical_name, str)
            or canonical_name not in expected_names
            or canonical_name in old_by_name
        ):
            raise FreezeBFinalizationError(
                "R1 bridge has invalid or duplicate canonical IDs"
            )
        old_by_name[canonical_name] = old
    if set(old_by_name) != set(expected_names):
        raise FreezeBFinalizationError("R1 bridge canonical coverage is incomplete")
    intervention_records = _array(
        _mapping(
            run.result.provenance["legacy_joint_friction_intervention"],
            "intervention",
        )["records"],
        "intervention records",
    )
    intervention_by_name: dict[str, Mapping[str, Any]] = {}
    for raw_intervention in intervention_records:
        intervention_record = _mapping(raw_intervention, "intervention record")
        canonical_id = intervention_record.get("canonical_id")
        if not isinstance(canonical_id, str) or canonical_id in intervention_by_name:
            raise FreezeBFinalizationError(
                "intervention records have invalid or duplicate canonical IDs"
            )
        intervention_by_name[canonical_id] = intervention_record
    valid = True
    for canonical_name, new_raw in zip(expected_names, new_records, strict=True):
        new = _mapping(new_raw, "new backend record")
        old = old_by_name[canonical_name]
        mapping = _mapping(old.get("mapping"), "R1 mapping")
        armature = _mapping(_mapping(old.get("armature"), "R1 armature").get("runtime_effective_value"), "R1 armature runtime")
        friction = _mapping(old.get("friction"), "R1 friction")
        friction_authored = _mapping(friction.get("authored_value"), "R1 friction authored")
        friction_resolved = _mapping(friction.get("resolved_value"), "R1 friction resolved")
        control = _mapping(old.get("control_drive"), "R1 control")
        effort = _mapping(_mapping(control["effort_limit"], "R1 effort").get("runtime_effective_value"), "R1 effort runtime")
        intervention = intervention_by_name.get(str(mapping["canonical_joint_name"]))
        valid = valid and new.get("canonical_id") == mapping.get("canonical_joint_name")
        valid = valid and new.get("backend_name") == mapping.get("backend_joint_name")
        valid = valid and new.get("backend_index") == mapping.get("backend_joint_index")
        valid = valid and new.get("joint_prim_path") == mapping.get("joint_prim_path")
        if intervention is None:
            valid = False
        else:
            valid = valid and _float32_hex(
                intervention.get("observed_pre_value"), "R1 pre bridge"
            ) == _float32_hex(
                friction_authored.get("legacy_joint_friction_scalar"),
                "R1 authored",
            )
            valid = valid and _float32_hex(
                intervention.get("observed_pre_value"), "R1 resolved bridge"
            ) == _float32_hex(
                friction_resolved.get("legacy_joint_friction_scalar"),
                "R1 resolved",
            )
        valid = valid and new.get("armature") == armature.get("dof_armature_binding")
        valid = valid and new.get("controller_armature") == armature.get("idealpd_controller_buffer")
        if role == "sham":
            # The sham must reproduce the R1 runtime-zero-slot bridge exactly.
            # The zero treatment is intentionally exempt: a changed runtime
            # friction family is an experimental observation, not drift.
            friction_runtime = _mapping(
                friction.get("runtime_effective_value"),
                "R1 friction runtime",
            )
            old_slots = _mapping(
                friction_runtime.get("dof_friction_properties_binding"),
                "R1 runtime friction slots",
            )
            valid = valid and new.get("friction_properties_raw") == [
                old_slots.get(key) for key in ("static", "dynamic", "viscous")
            ]
            valid = valid and {
                "static": new.get("controller_static_friction"),
                "dynamic": new.get("controller_dynamic_friction"),
                "viscous": new.get("controller_viscous_friction"),
            } == friction_runtime.get("idealpd_controller_buffers")
        valid = valid and new.get("controller_stiffness") == _mapping(control["kp"], "R1 kp").get("runtime_effective_value")
        valid = valid and new.get("controller_damping") == _mapping(control["kd"], "R1 kd").get("runtime_effective_value")
        valid = valid and new.get("controller_effort_limit") == effort.get("controller")
        valid = valid and new.get("controller_effort_limit_sim") == effort.get("simulation")
        valid = valid and new.get("backend_drive_stiffness") == _mapping(control["backend_drive_stiffness"], "R1 backend kp").get("runtime_effective_value")
        valid = valid and new.get("backend_drive_damping") == _mapping(control["backend_drive_damping"], "R1 backend kd").get("runtime_effective_value")
    transformed_solver = _solver_runtime(_effective_snapshot(run.result))
    old_solver = deepcopy(r1_instance.get("solver_runtime"))
    if row["timestep_variant"] == "halved":
        # The halved case intentionally changes only the configured timestep.
        transformed_solver["physics_dt_s"] = "<dt-variant>"
        old_solver["physics_dt_s"] = "<dt-variant>"
    valid = valid and transformed_solver == old_solver
    return bool(valid)


def _formal_and_repeat_metrics(
    *,
    plan: Mapping[str, Any],
    manifest: object,
    formal: Mapping[tuple[str, str], CollectedRun],
    local: Mapping[str, CollectedRun],
    remote: Mapping[str, CollectedRun],
) -> tuple[dict[str, object], list[SensitivityTraceSet]]:
    hands = {side: manifest.hand(HandSide(side)) for side in ("left", "right")}  # type: ignore[attr-defined]
    bridges: list[DeltaMetrics] = []
    repeats: list[DeltaMetrics] = []
    dt_values: list[DeltaMetrics] = []
    trace_sets: list[SensitivityTraceSet] = []
    lookup_local = {(row["hand"], row["timestep_variant"], row["repeat_index"]): local[row["experiment_case_id"]] for row in plan["mujoco_cases"]}
    lookup_remote = {(row["hand"], row["timestep_variant"], row["repeat_index"], row["role"]): remote[row["experiment_case_id"]] for row in plan["ovphysx_cases"]}
    for side in ("left", "right"):
        hand = hands[side]
        for variant in ("base", "halved"):
            for repeat in (1, 2):
                mj_id = f"mujoco.{side}.small_step.{variant}.r{repeat:02d}"
                ov_id = f"ovphysx.{side}.small_step.{variant}.r{repeat:02d}"
                bridges.append(DeltaMetrics.between(formal[("mujoco", mj_id)], lookup_local[(side, variant, repeat)], hand))
                bridges.append(DeltaMetrics.between(formal[("ovphysx", ov_id)], lookup_remote[(side, variant, repeat, "sham")], hand))
            repeats.append(DeltaMetrics.between(lookup_local[(side, variant, 1)], lookup_local[(side, variant, 2)], hand))
            for role in ("sham", "zero"):
                repeats.append(DeltaMetrics.between(lookup_remote[(side, variant, 1, role)], lookup_remote[(side, variant, 2, role)], hand))
            mj_id = f"mujoco.{side}.small_step.{variant}.r01"
            ov_id = f"ovphysx.{side}.small_step.{variant}.r01"
            trace_sets.append(
                SensitivityTraceSet(
                    hand=HandSide(side),
                    timestep_variant=lookup_local[(side, variant, 1)].case.timestep_variant,
                    formal_mujoco=formal[("mujoco", mj_id)],
                    formal_ovphysx=formal[("ovphysx", ov_id)],
                    sham_ovphysx=lookup_remote[(side, variant, 1, "sham")],
                    zero_ovphysx=lookup_remote[(side, variant, 1, "zero")],
                )
            )
        dt_values.append(DeltaMetrics.between(lookup_local[(side, "base", 1)], lookup_local[(side, "halved", 1)], hand))
        for role in ("sham", "zero"):
            dt_values.append(DeltaMetrics.between(lookup_remote[(side, "base", 1, role)], lookup_remote[(side, "halved", 1, role)], hand))
    bridge_max = DeltaMetrics.maximum(bridges)
    repeat_max = DeltaMetrics.maximum(repeats)
    dt_max = DeltaMetrics.maximum(dt_values)
    return {
        "formal_bridge": bridge_max.to_dict(),
        "repeatability": repeat_max.to_dict(),
        "dt_halving": dt_max.to_dict(),
    }, trace_sets


def _evaluate_identity_bound_freeze_b(
    trace_sets: Sequence[SensitivityTraceSet],
    *,
    inputs: Mapping[str, Any],
    private_plan: Mapping[str, Any],
    public_protocol: Mapping[str, Any],
    public_protocol_file_sha256: str,
    public_protocol_canonical_sha256: str,
    deployment_source_revision: str,
    deployment_source_tree: str,
    deployment_source_snapshot_sha256: str,
    validity_checks: Mapping[str, bool],
) -> tuple[FreezeBDecision, dict[str, str]]:
    evidence_bindings = {
        "private_plan_sha256": private_plan_sha256(private_plan),
        "public_protocol_file_sha256": public_protocol_file_sha256,
        "public_protocol_canonical_sha256": public_protocol_canonical_sha256,
        "deployment_source_revision": deployment_source_revision,
        "deployment_source_tree": deployment_source_tree,
        "deployment_source_snapshot_sha256": deployment_source_snapshot_sha256,
    }
    decision = evaluate_freeze_b(
        trace_sets,
        frozen_inputs=inputs,
        private_plan=private_plan,
        public_protocol=public_protocol,
        evidence_bindings=evidence_bindings,
        validity_checks=validity_checks,
    )
    return decision, evidence_bindings


def _validate_public_safe(value: object) -> None:
    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).lower()
                if any(part in lowered for part in _PUBLIC_FORBIDDEN_KEY_PARTS):
                    raise FreezeBFinalizationError(f"public output contains forbidden field: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            lowered = item.lower()
            if (
                "/data/home/" in lowered
                or "/mnt/ceph" in lowered
                or "gpu-" in lowered
                or _ABSOLUTE_WINDOWS_RE.search(item) is not None
                or any(name.lower() in lowered for side in ("left", "right") for name in canonical_joint_names(side))
            ):
                raise FreezeBFinalizationError(f"public output contains private identifier at {path}")
    visit(value, "public")


def _validate_summary_decision_consistency(
    summary: Mapping[str, Any],
    *,
    decision: FreezeBDecision,
    checks: Mapping[str, bool],
) -> None:
    """Reject any public state that contradicts the private decision."""

    validity = _mapping(summary.get("validity"), "public validity")
    checks_all_passed = bool(checks) and all(value is True for value in checks.values())
    primary_metric_evaluation_passed = decision.execution_valid
    all_passed = checks_all_passed and primary_metric_evaluation_passed
    expected_status = "VALID" if decision.execution_valid else "INVALID"
    expected_label = (
        decision.scientific_label.value
        if decision.scientific_label is not None
        else None
    )
    primary_metrics = summary.get("primary_metrics")
    if (
        validity.get("checks_all_passed") is not checks_all_passed
        or validity.get("primary_metric_evaluation_passed")
        is not primary_metric_evaluation_passed
        or validity.get("all_passed") is not all_passed
        or summary.get("validation_status") != expected_status
        or summary.get("scientific_label") != expected_label
        or dict(_mapping(validity.get("checks"), "public validity checks"))
        != dict(sorted(checks.items()))
    ):
        raise FreezeBFinalizationError(
            "public summary contradicts evidence checks or private decision"
        )
    if decision.execution_valid:
        metrics = _mapping(primary_metrics, "public primary metrics")
        if (
            metrics.get("cells") != [cell.to_dict() for cell in decision.cells]
            or metrics.get("s_values") != list(decision.s_values)
            or not all_passed
        ):
            raise FreezeBFinalizationError(
                "VALID decision lacks consistent public primary metrics"
            )
    elif primary_metrics is not None or all_passed or expected_label is not None:
        raise FreezeBFinalizationError(
            "INVALID decision cannot publish metrics, a label, or all_passed=true"
        )


def _render_report(summary: Mapping[str, Any]) -> str:
    validation_status = str(summary["validation_status"])
    scientific_label = summary["scientific_label"]
    validity = _mapping(summary["validity"], "public validity")
    runtime_observation = _mapping(
        summary["runtime_friction_observation"],
        "runtime friction observation",
    )
    lines = [
        "# WaveSimParity Freeze B",
        "",
        f"Validation status: **{validation_status}**",
        (
            f"Scientific label: **{scientific_label}**"
            if scientific_label is not None
            else "Scientific label: not assigned"
        ),
        "",
        "This is an unofficial, simulation-only, kit-less OVPhysX native-input sensitivity study. "
        "It does not establish cross-engine parameter equivalence, an upstream bug, hardware behavior, or Sim2Real evidence.",
        "",
        "## Corrigenda",
        "",
        "- One earlier local collection attempt produced one raw MuJoCo run file but admitted 0/24 evidence cases because launcher and actual-worker process identities did not bind.",
        "- That excluded attempt ran 0 OVPhysX cases and used no remote GPU.",
        "- Its trajectory file was deserialized by the failed launcher for structural validation, but trajectory sample values were not reviewed or used for metrics, correction choices, or a scientific label.",
        "- A later a2 campaign completed and automatically validated 8/8 local MuJoCo cases, then executed one completed OVPhysX worker case on a remote GPU; the post-worker payload admission subprocess failed before admitting any remote case because `-S` hid pinned NumPy.",
        "- The a2 trajectory payloads may have been reviewed during diagnosis, but they were not used to select or design the one-flag admission fix, compute final metrics, or assign a scientific label.",
        "- A later local-a3 operator invocation disclosed a noncanonical asset materialization. One worker entered the runner's canonical-LF asset-tree preflight and exited on the recorded hash-mismatch error before adapter construction, model creation, trajectory output, or any simulator step; it admitted 0 cases and used no remote GPU.",
        "- All four prior evidence trees are immutable and excluded. The three earlier scientific-evidence trees and the operator-preflight-only local-a3 tree remain private; the corrected G execution reruns the full 24-case matrix under one source identity.",
        "- After G collection, the finalizer deserialized the R1, formal, local, and all remote traces and completed structural, finite-value, and sanity checks before exposing an analysis-only backend-order assumption. Combined formal/repeatability/dt/sensitivity metrics had not run, no scientific label was assigned, and no final bundle was published.",
        "- Runtime readback metadata and values were reviewed during diagnosis. The H correction follows the existing canonical-ID/backend-index mapping contract, not a trajectory effect or threshold: it validates the backend order and mapping inverse, then projects records and target readback into the frozen plan order without changing values or evidence bytes.",
        "- Active local-a4 and remote-a4 evidence remains campaign-G evidence. H and I only analyze that immutable 24-case evidence; neither reruns either simulator.",
        "- H produced one local, unpublished INVALID/null bundle after every evidence check passed but primary metric construction rejected sub-picosecond floating-accumulation timestamp differences. Its public summary and report therefore contradicted the private decision; that bundle is verified, retained as a private superseded subset, and is not reused as a final result.",
        "- The I correction applies the already-frozen 1e-12 single-trace time semantics only as a same-step, same-length cross-trace compatibility check. It does not alter or round timestamps; window selection still uses the reference trace's original recorded times. Baseline and zero-vs-MuJoCo use 200 base/400 halved samples, treatment-vs-sham uses 201 base/401 halved samples, and frozen baselines are revalidated exactly.",
        "- Runtime readback, trajectory values, primary metrics, and the diagnostic final S values and label were reviewed before I. The compatibility rule comes from the frozen preparer and existing timestamp contract, not from effects, thresholds, or a desired label.",
        "- The hypothesis, intervention, scenario, case order, metrics, thresholds, claim boundary, and formal Gate 0 status remain unchanged.",
        "",
        "## Evidence gates",
        "",
        f"- Corrected execution completed cases: {summary['execution']['completed_case_count']}/24",
        "- Process isolation: 24/24 cases ran under distinct, scheme-bound process identities; identifiers remain private.",
        f"- Canonical mapping: {summary['mapping']['joint_mapping']} joints and {summary['mapping']['distal_frame_mapping']} fingertip frames",
        "- OVPhysX contacts were not measured; `contact_count=null` records that observation boundary and is not evidence of zero contacts or contact absence.",
        "- OVPhysX timestep evidence verifies requested/configured dt and same-step, same-length recorded timestamp compatibility within 1e-12 s for comparisons only; the pinned kit-less stack exposes no compiled-runtime effective-dt getter.",
        (
            "- OVPhysX position-target binding: 16/16 cases verified; aggregate maximum "
            f"absolute readback error was {summary['position_target_readback']['aggregate_max_abs_error_rad']} rad "
            f"against a {summary['position_target_readback']['limit_rad']} rad limit."
        ),
        "- Every trace passed finite-value and preregistered sanity bounds for joints, targets, velocity, frame origins, and unit quaternions.",
        (
            "- Runtime friction readback: the sham matched frozen R1 exactly; "
            f"{runtime_observation['zero_changed_case_count_from_r1']}/"
            f"{runtime_observation['zero_case_count']} zero-treatment cases differed "
            "from R1. That count is descriptive and does not itself decide validity or the scientific label."
        ),
        f"- Preregistered evidence checks passed: {str(validity['checks_all_passed']).lower()}",
        f"- Primary metric construction and evaluation passed: {str(validity['primary_metric_evaluation_passed']).lower()}",
        f"- Overall checks-and-metrics disposition passed: {str(validity['all_passed']).lower()}",
        "",
        "## Interpretation",
        "",
    ]
    if validation_status == "INVALID":
        if validity["checks_all_passed"] is not True:
            lines.append(
                "No scientific label is assigned because at least one preregistered evidence check failed."
            )
        else:
            lines.append(
                "No scientific label is assigned because primary metric construction or evaluation failed after every preregistered evidence check passed."
            )
    elif scientific_label == SensitivityStatus.SUPPORTED.value:
        lines.append("All eight preregistered normalized sensitivity axes met the SUPPORT threshold.")
    elif scientific_label == SensitivityStatus.NOT_SUPPORTED.value:
        lines.append("All eight preregistered normalized sensitivity axes stayed at or below the NOT_SUPPORTED threshold.")
    else:
        lines.append("The eight preregistered axes did not satisfy either all-axis decision boundary, so the result is INCONCLUSIVE.")
    lines.extend(
        [
            "",
            "The original formal Gate 0 result remains **DIVERGENT** with `pass_ready=false`; Freeze B does not rewrite it.",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _copy_parent_subset(
    source_root: Path,
    destination_root: Path,
    relative_paths: Sequence[str],
) -> None:
    for relative in relative_paths:
        _copy_file_exclusive(source_root / PurePosixPath(relative), destination_root / PurePosixPath(relative))


def _verify_superseded_analysis_subset(root: Path) -> None:
    """Bind both H manifest bytes and every selected payload to H's root."""

    manifest = root / "bundle.json"
    if sha256_file(manifest) != SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
        "bundle_json_sha256"
    ]:
        raise FreezeBFinalizationError(
            "copied superseded H bundle manifest bytes drifted"
        )
    verify_copied_bundle_subset(
        manifest,
        expected_root_sha256=str(
            SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY["root_sha256"]
        ),
        copied_payloads={
            relative: root / PurePosixPath(relative)
            for relative in SUPERSEDED_ANALYSIS_SUBSET_PATHS
            if relative != "bundle.json"
        },
    )


def _remove_owned_tree(path: Path, *, expected_path: Path) -> None:
    """Remove only a tree created by this finalization attempt."""

    absolute = path.absolute()
    expected = expected_path.absolute()
    if absolute != expected or absolute.parent != expected.parent or is_link_like(absolute):
        raise FreezeBFinalizationError("refusing unsafe finalization cleanup")
    if absolute.exists():
        shutil.rmtree(absolute)


def _promote_output_pair(
    *,
    private_staging: Path,
    private_destination: Path,
    expected_private_root_sha256: str,
    public_staging: Path | None,
    public_destination: Path | None,
) -> tuple[Path, Path | None]:
    """Promote both outputs transactionally or remove this attempt's private tree."""

    if (public_staging is None) != (public_destination is None):
        raise FreezeBFinalizationError(
            "public staging and destination must either both exist or both be absent"
        )
    private_promoted: Path | None = None
    public_promoted: Path | None = None
    try:
        private_promoted = promote_staging_root(
            private_staging, private_destination
        )
        if (
            verify_exact_bundle(private_promoted)["root_sha256"]
            != expected_private_root_sha256
        ):
            raise FreezeBFinalizationError(
                "bundle changed during atomic promotion"
            )
        if public_staging is not None and public_destination is not None:
            public_promoted = promote_staging_root(
                public_staging, public_destination
            )
            _validate_public_safe(
                _strict_json(public_promoted / "summary.json", "public summary")
            )
            _validate_public_safe(
                (public_promoted / "report.md").read_text(encoding="utf-8")
            )
    except BaseException:
        for candidate, expected in (
            (public_promoted, public_destination),
            (private_promoted, private_destination),
            (public_staging, public_staging),
            (private_staging, private_staging),
        ):
            if candidate is not None and expected is not None and candidate.exists():
                _remove_owned_tree(candidate, expected_path=expected)
        raise
    assert private_promoted is not None
    return private_promoted, public_promoted


def _finalize_freeze_b_from_snapshot(
    *,
    project_root: Path,
    public_protocol_path: Path,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
    runtime_order_corrigendum_path: Path,
    time_grid_corrigendum_path: Path,
    private_plan_path: Path,
    readback_bundle: Path,
    formal_bundle: Path,
    local_evidence: Path,
    remote_evidence: Path,
    superseded_analysis_bundle: Path,
    excluded_evidence: Mapping[str, Path],
    output_dir: Path,
    public_output_dir: Path | None,
    campaign_revision: str,
    analysis_revision: str,
    analysis_manifest_path: Path | None = None,
    analysis_source_root: Path | None = None,
    canonical_public_protocol_path: Path | None = None,
    canonical_corrigendum_path: Path | None = None,
    canonical_admission_corrigendum_path: Path | None = None,
    canonical_asset_preflight_corrigendum_path: Path | None = None,
    canonical_runtime_order_corrigendum_path: Path | None = None,
    canonical_time_grid_corrigendum_path: Path | None = None,
    canonical_superseded_analysis_bundle: Path | None = None,
) -> tuple[Path, Path | None, dict[str, object]]:
    project_root = _resolved_real(project_root, "project root", directory=True)
    public_protocol_path = _resolved_real(public_protocol_path, "public protocol", directory=False)
    corrigendum_path = _resolved_real(corrigendum_path, "corrigendum", directory=False)
    admission_corrigendum_path = _resolved_real(
        admission_corrigendum_path, "admission corrigendum", directory=False
    )
    asset_preflight_corrigendum_path = _resolved_real(
        asset_preflight_corrigendum_path,
        "asset preflight corrigendum",
        directory=False,
    )
    runtime_order_corrigendum_path = _resolved_real(
        runtime_order_corrigendum_path,
        "runtime order corrigendum",
        directory=False,
    )
    time_grid_corrigendum_path = _resolved_real(
        time_grid_corrigendum_path,
        "time-grid corrigendum",
        directory=False,
    )
    private_plan_path = _resolved_real(private_plan_path, "private plan", directory=False)
    readback_bundle = _resolved_real(readback_bundle, "R1 bundle", directory=True)
    formal_bundle = _resolved_real(formal_bundle, "formal bundle", directory=True)
    local_evidence = _resolved_real(local_evidence, "local evidence", directory=True)
    remote_evidence = _resolved_real(remote_evidence, "remote evidence", directory=True)
    superseded_analysis_bundle = _resolved_real(
        superseded_analysis_bundle,
        "superseded H analysis bundle",
        directory=True,
    )
    superseded_analysis_source = (
        superseded_analysis_bundle
        if canonical_superseded_analysis_bundle is None
        else _resolved_real(
            canonical_superseded_analysis_bundle,
            "canonical superseded H analysis bundle",
            directory=True,
        )
    )
    if set(excluded_evidence) != set(EXCLUDED_EVIDENCE_LABELS):
        raise FreezeBFinalizationError(
            "excluded evidence labels must be exactly local-a1, local-a2, remote-a2, local-a3"
        )
    excluded_evidence = {
        label: _resolved_real(
            excluded_evidence[label], f"excluded evidence {label}", directory=True
        )
        for label in EXCLUDED_EVIDENCE_LABELS
    }
    source_bytes_root = (
        project_root
        if analysis_source_root is None
        else _resolved_real(
            analysis_source_root,
            "analysis source snapshot",
            directory=True,
        )
    )
    destination = validated_results_output_path(project_root, output_dir)
    public_destination = (
        validated_results_output_path(project_root, public_output_dir)
        if public_output_dir is not None
        else None
    )
    if public_destination is not None and public_destination == destination:
        raise FreezeBFinalizationError("private and public outputs must be distinct")
    for output in (destination, public_destination):
        if output is None:
            continue
        for source in (
            readback_bundle,
            formal_bundle,
            local_evidence,
            remote_evidence,
            superseded_analysis_bundle,
            *excluded_evidence.values(),
        ):
            try:
                output.relative_to(source)
            except ValueError:
                pass
            else:
                raise FreezeBFinalizationError("output must not be inside an input evidence tree")
    input_tree_fingerprints = {
        "readback_bundle": _tree_fingerprint(readback_bundle, "R1 bundle"),
        "formal_bundle": _tree_fingerprint(formal_bundle, "formal bundle"),
        "local_evidence": _tree_fingerprint(local_evidence, "local evidence"),
        "remote_evidence": _tree_fingerprint(remote_evidence, "remote evidence"),
        "superseded_analysis_bundle": _tree_fingerprint(
            superseded_analysis_bundle,
            "superseded H analysis bundle",
        ),
        **{
            f"excluded_{label}": _tree_fingerprint(
                path, f"excluded evidence {label}"
            )
            for label, path in excluded_evidence.items()
        },
    }
    input_file_identities = {
        "private_plan": _file_identity(private_plan_path, "private plan"),
        "public_protocol": _file_identity(public_protocol_path, "public protocol"),
        "corrigendum": _file_identity(corrigendum_path, "corrigendum"),
        "admission_corrigendum": _file_identity(
            admission_corrigendum_path, "admission corrigendum"
        ),
        "asset_preflight_corrigendum": _file_identity(
            asset_preflight_corrigendum_path,
            "asset preflight corrigendum",
        ),
        "runtime_order_corrigendum": _file_identity(
            runtime_order_corrigendum_path,
            "runtime order corrigendum",
        ),
        "time_grid_corrigendum": _file_identity(
            time_grid_corrigendum_path,
            "time-grid corrigendum",
        ),
    }
    tracked_source_identities = {
        relative: _file_identity(
            source_bytes_root / PurePosixPath(relative),
            f"tracked source {relative}",
        )
        for relative in REQUIRED_SOURCE_PATHS
    }
    public_protocol = load_public_protocol(public_protocol_path)
    plan = load_private_plan(private_plan_path, public_protocol=public_protocol)
    if plan["protocol_id"] != PROTOCOL_ID:
        raise FreezeBFinalizationError("Freeze B protocol ID drifted")
    inputs = _mapping(plan["inputs"], "private plan inputs")
    source_identity = _validate_source_identity(
        project_root,
        implementation_revision=str(inputs["freeze_b_source_revision"]),
        implementation_tree=str(inputs["freeze_b_source_tree"]),
        campaign_revision=campaign_revision,
        analysis_revision=analysis_revision,
        public_protocol_path=public_protocol_path,
        corrigendum_path=corrigendum_path,
        admission_corrigendum_path=admission_corrigendum_path,
        asset_preflight_corrigendum_path=asset_preflight_corrigendum_path,
        runtime_order_corrigendum_path=runtime_order_corrigendum_path,
        time_grid_corrigendum_path=time_grid_corrigendum_path,
        canonical_public_protocol_path=canonical_public_protocol_path,
        canonical_corrigendum_path=canonical_corrigendum_path,
        canonical_admission_corrigendum_path=(
            canonical_admission_corrigendum_path
        ),
        canonical_asset_preflight_corrigendum_path=(
            canonical_asset_preflight_corrigendum_path
        ),
        canonical_runtime_order_corrigendum_path=(
            canonical_runtime_order_corrigendum_path
        ),
        canonical_time_grid_corrigendum_path=(
            canonical_time_grid_corrigendum_path
        ),
    )
    campaign_tree = str(source_identity["campaign_execution_tree"])
    analysis_tree = str(source_identity["analysis_tree"])
    expected_superseded_tree_fingerprint = {
        "entry_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_entry_count"
        ],
        "file_count": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_file_count"
        ],
        "total_bytes": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_total_bytes"
        ],
        "root_sha256": SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY[
            "full_tree_root_sha256"
        ],
    }
    if (
        input_tree_fingerprints["superseded_analysis_bundle"]
        != expected_superseded_tree_fingerprint
        or _tree_fingerprint(
            superseded_analysis_source,
            "source superseded H analysis bundle",
        )
        != expected_superseded_tree_fingerprint
    ):
        raise FreezeBFinalizationError(
            "superseded H analysis source/snapshot fingerprint drifted"
        )
    # The full source was copied before entry to this function.  Verify the
    # complete snapshot before deserializing any H JSON, then parse H only from
    # that verified snapshot.
    superseded_analysis_identity = _validate_superseded_analysis_bundle(
        superseded_analysis_bundle
    )
    source_identity["superseded_analysis_bundle"] = (
        superseded_analysis_identity
    )
    manifest_path = (
        project_root / GATE0_MANIFEST_RELATIVE_PATH
        if analysis_manifest_path is None
        else _resolved_real(
            analysis_manifest_path,
            "analysis manifest snapshot",
            directory=False,
        )
    )
    manifest = load_manifest(manifest_path)
    if (
        sha256_file(manifest_path) != inputs["gate0_manifest_file_sha256"]
        or manifest_sha256(manifest) != inputs["gate0_manifest_semantic_sha256"]
        or manifest.provenance.commit != inputs["asset_commit"]
        or manifest.provenance.asset_git_tree != inputs["asset_git_tree"]
        or manifest.provenance.canonical_lf_asset_tree_sha256 != inputs["canonical_lf_asset_tree_sha256"]
    ):
        raise FreezeBFinalizationError("canonical manifest/asset identity drifted")
    active_evidence_identities = _validate_active_evidence_fingerprints(
        local_evidence=local_evidence,
        remote_evidence=remote_evidence,
    )
    r1 = _load_r1(readback_bundle, plan=plan)
    formal = _load_formal_runs(formal_bundle, plan=plan, manifest=manifest)
    local, local_fresh_process_ids = _load_local_runs(
        local_evidence,
        plan=plan,
        plan_path=private_plan_path,
        manifest=manifest,
        execution_revision=campaign_revision,
        execution_tree=campaign_tree,
        project_root=source_bytes_root,
        source_identity=source_identity,
    )
    public_protocol_file_digest = sha256_file(public_protocol_path)
    public_protocol_canonical_digest = canonical_json_sha256(public_protocol)
    corrigendum = _validate_corrigendum(
        _strict_json(corrigendum_path, "Freeze B corrigendum"),
        implementation_revision=str(inputs["freeze_b_source_revision"]),
        implementation_tree=str(inputs["freeze_b_source_tree"]),
    )
    _validate_public_safe(corrigendum)
    corrigendum_file_digest = sha256_file(corrigendum_path)
    corrigendum_canonical_digest = canonical_json_sha256(corrigendum)
    admission_corrigendum = _validate_admission_corrigendum(
        _strict_json(
            admission_corrigendum_path, "Freeze B admission corrigendum"
        ),
        implementation_revision=str(inputs["freeze_b_source_revision"]),
        implementation_tree=str(inputs["freeze_b_source_tree"]),
        first_correction_revision=str(source_identity["correction_revision"]),
        first_correction_tree=str(source_identity["correction_tree"]),
    )
    _validate_public_safe(admission_corrigendum)
    admission_corrigendum_file_digest = sha256_file(admission_corrigendum_path)
    admission_corrigendum_canonical_digest = canonical_json_sha256(
        admission_corrigendum
    )
    asset_preflight_corrigendum = _validate_asset_preflight_corrigendum(
        _strict_json(
            asset_preflight_corrigendum_path,
            "Freeze B asset preflight corrigendum",
        ),
        implementation_revision=str(inputs["freeze_b_source_revision"]),
        implementation_tree=str(inputs["freeze_b_source_tree"]),
        first_correction_revision=str(source_identity["correction_revision"]),
        first_correction_tree=str(source_identity["correction_tree"]),
        admission_correction_revision=str(
            source_identity["admission_correction_revision"]
        ),
        admission_correction_tree=str(
            source_identity["admission_correction_tree"]
        ),
    )
    _validate_public_safe(asset_preflight_corrigendum)
    asset_preflight_corrigendum_file_digest = sha256_file(
        asset_preflight_corrigendum_path
    )
    asset_preflight_corrigendum_canonical_digest = canonical_json_sha256(
        asset_preflight_corrigendum
    )
    runtime_order_corrigendum = _validate_runtime_order_corrigendum(
        _strict_json(
            runtime_order_corrigendum_path,
            "Freeze B runtime order corrigendum",
        )
    )
    _validate_public_safe(runtime_order_corrigendum)
    runtime_order_corrigendum_file_digest = sha256_file(
        runtime_order_corrigendum_path
    )
    runtime_order_corrigendum_canonical_digest = canonical_json_sha256(
        runtime_order_corrigendum
    )
    time_grid_corrigendum = _validate_time_grid_corrigendum(
        _strict_json(
            time_grid_corrigendum_path,
            "Freeze B time-grid corrigendum",
        )
    )
    _validate_public_safe(time_grid_corrigendum)
    time_grid_corrigendum_file_digest = sha256_file(
        time_grid_corrigendum_path
    )
    time_grid_corrigendum_canonical_digest = canonical_json_sha256(
        time_grid_corrigendum
    )
    excluded_evidence_identities = _validate_excluded_evidence_fingerprints(
        excluded_evidence
    )
    remote, remote_launcher, remote_fresh_process_ids = _load_remote_runs(
        remote_evidence,
        plan=plan,
        plan_path=private_plan_path,
        manifest=manifest,
        execution_revision=campaign_revision,
        execution_tree=campaign_tree,
        project_root=source_bytes_root,
        public_protocol_file_sha256=public_protocol_file_digest,
        public_protocol_canonical_sha256=public_protocol_canonical_digest,
    )
    local_source_snapshot_identity = _validate_remote_source_snapshot(
        project_root,
        execution_revision=campaign_revision,
        execution_tree=campaign_tree,
        remote_archive_sha256=remote_launcher.get("source_archive_sha256"),
        remote_snapshot_sha256=remote_launcher.get("source_snapshot_sha256"),
    )
    if (
        local_source_snapshot_identity.get("source_archive_sha256")
        != CAMPAIGN_SOURCE_ARCHIVE_SHA256
        or local_source_snapshot_identity.get("source_snapshot_sha256")
        != REMOTE_SOURCE_SNAPSHOT_SHA256
    ):
        raise FreezeBFinalizationError(
            "campaign G archive or remote snapshot identity drifted"
        )
    source_identity.update(local_source_snapshot_identity)
    thresholds = _mapping(public_protocol["validity_thresholds"], "validity thresholds")
    position_target_limit = _finite(
        thresholds.get("position_target_readback_max_abs_rad"),
        "position target readback threshold",
    )
    sanity_thresholds = {
        "max_abs_canonical_joint_position_rad": MAX_ABS_CANONICAL_JOINT_POSITION_RAD,
        "max_abs_canonical_position_target_rad": MAX_ABS_CANONICAL_POSITION_TARGET_RAD,
        "max_abs_backend_joint_velocity_rad_s": MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S,
        "max_frame_origin_distance_m": MAX_FRAME_ORIGIN_DISTANCE_M,
        "quaternion_norm_abs_tolerance": UNIT_QUATERNION_NORM_TOLERANCE,
    }
    if any(
        _finite(thresholds.get(key), f"public validity threshold {key}") != expected
        for key, expected in sanity_thresholds.items()
    ):
        raise FreezeBFinalizationError("public sanity thresholds drifted from compare.py")
    tagged_fresh_process_ids = {
        *(("local_os_identity_v1", value) for value in local_fresh_process_ids),
        *(("remote_worker_identity_v1", value) for value in remote_fresh_process_ids),
    }
    checks: dict[str, bool] = {
        "case_completion_24_of_24": len(local) == 8 and len(remote) == 16,
        "joint_mapping_44_of_44": True,
        "distal_frame_mapping_10_of_10": True,
        "fresh_process_24_of_24": (
            len(local_fresh_process_ids) == 8
            and len(remote_fresh_process_ids) == 16
            and len(tagged_fresh_process_ids) == 24
        ),
        "ovphysx_contact_unobserved_disclosed": True,
        "mujoco_zero_contacts": True,
        "dt_v2_evidence": True,
        "position_target_readback_within_1e_6": True,
        "intervention_write_readback": True,
        "non_target_configuration_invariance": True,
        "r1_legacy_and_runtime_bridge": True,
        "finite_and_sanity_bounds": True,
    }
    plan_digest = private_plan_sha256(plan)
    r1_runtime_signatures = {
        side: _r1_runtime_friction_signature(
            r1[side],
            expected_joint_names=_hand_plan(plan, side)["joint_names"],
        )
        for side in ("left", "right")
    }
    runtime_signatures: dict[str, str] = {}
    position_target_errors: list[float] = []
    groups: dict[tuple[str, str], list[object]] = {}
    for row in plan["ovphysx_cases"]:
        run = remote[row["experiment_case_id"]]
        side = str(row["hand"])
        hand_plan = _hand_plan(plan, side)
        checks["dt_v2_evidence"] &= _validate_dt_v2(run.result, float(row["dt_s"]))
        target_valid, target_error = _validate_position_target_readback(
            run.result,
            case=run.case,
            manifest=manifest,
            expected_joint_names=hand_plan["joint_names"],
            max_abs_error_rad=position_target_limit,
        )
        checks["position_target_readback_within_1e_6"] &= target_valid
        position_target_errors.append(target_error)
        checks["intervention_write_readback"] &= _validate_intervention(
            run.result, row=row, hand_plan=hand_plan, plan_digest=plan_digest
        )
        checks["intervention_write_readback"] &= _validate_intervention_readback(
            run.result, row=row, hand_plan=hand_plan
        )
        checks["r1_legacy_and_runtime_bridge"] &= _r1_bridge(
            run,
            r1_instance=r1[side],
            row=row,
            hand_plan=hand_plan,
        )
        runtime_signature = _runtime_friction_signature(
            run.result,
            side=side,
            expected_joint_names=hand_plan["joint_names"],
        )
        runtime_signatures[str(row["experiment_case_id"])] = runtime_signature
        if row["role"] == "sham":
            checks["r1_legacy_and_runtime_bridge"] &= (
                runtime_signature == r1_runtime_signatures[side]
            )
        groups.setdefault((side, str(row["timestep_variant"])), []).append(
            _normalized_non_target_contract(run.result)
        )
    checks["non_target_configuration_invariance"] = all(
        all(item == values[0] for item in values[1:]) for values in groups.values()
    )
    runtime_gate, runtime_friction_observation, private_runtime_friction_observation = (
        _summarize_runtime_friction_observation(
            plan["ovphysx_cases"],
            case_signatures=runtime_signatures,
            r1_signatures=r1_runtime_signatures,
        )
    )
    checks["r1_legacy_and_runtime_bridge"] &= runtime_gate
    metric_groups, trace_sets = _formal_and_repeat_metrics(
        plan=plan,
        manifest=manifest,
        formal=formal,
        local=local,
        remote=remote,
    )
    bridge = DeltaMetrics(**{
        "joint_max_abs_rad": metric_groups["formal_bridge"]["joint_max_abs_rad"],
        "frame_position_max_m": metric_groups["formal_bridge"]["frame_position_max_m"],
        "frame_orientation_max_rad": metric_groups["formal_bridge"]["frame_orientation_max_rad"],
    })
    repeat = DeltaMetrics(**{
        "joint_max_abs_rad": metric_groups["repeatability"]["joint_max_abs_rad"],
        "frame_position_max_m": metric_groups["repeatability"]["frame_position_max_m"],
        "frame_orientation_max_rad": metric_groups["repeatability"]["frame_orientation_max_rad"],
    })
    dt_delta = DeltaMetrics(**{
        "joint_max_abs_rad": metric_groups["dt_halving"]["joint_max_abs_rad"],
        "frame_position_max_m": metric_groups["dt_halving"]["frame_position_max_m"],
        "frame_orientation_max_rad": metric_groups["dt_halving"]["frame_orientation_max_rad"],
    })
    bridge_limit = float(thresholds["same_source_bridge_max_abs"])
    repeat_limit = float(thresholds["repeatability_max_abs"])
    checks["formal_same_source_bridge"] = bridge.within(
        joint=bridge_limit, position=bridge_limit, orientation=bridge_limit
    )
    checks["fresh_repeatability"] = repeat.within(
        joint=repeat_limit, position=repeat_limit, orientation=repeat_limit
    )
    checks["dt_halving"] = dt_delta.within(
        joint=float(thresholds["dt_halving_joint_max_abs_rad"]),
        position=float(thresholds["dt_halving_frame_position_max_m"]),
        orientation=float(thresholds["dt_halving_frame_orientation_max_rad"]),
    )
    decision, evidence_bindings = _evaluate_identity_bound_freeze_b(
        trace_sets,
        inputs=inputs,
        private_plan=plan,
        public_protocol=public_protocol,
        public_protocol_file_sha256=public_protocol_file_digest,
        public_protocol_canonical_sha256=public_protocol_canonical_digest,
        deployment_source_revision=campaign_revision,
        deployment_source_tree=campaign_tree,
        deployment_source_snapshot_sha256=str(
            remote_launcher["source_snapshot_sha256"]
        ),
        validity_checks=checks,
    )
    decision_payload = decision.to_dict()
    validation_status = "VALID" if decision.execution_valid else "INVALID"
    scientific_label = (
        decision.scientific_label.value
        if decision.scientific_label is not None
        else None
    )
    checks_all_passed = bool(checks) and all(
        value is True for value in checks.values()
    )
    primary_metric_evaluation_passed = decision.execution_valid
    all_passed = checks_all_passed and primary_metric_evaluation_passed
    public_protocol_digest = public_protocol_file_digest
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "validation_status": validation_status,
        "scientific_label": scientific_label,
        "execution": {
            "expected_case_count": 24,
            "completed_case_count": len(local) + len(remote),
            "mujoco_case_count": len(local),
            "ovphysx_case_count": len(remote),
            "completion_fraction": (len(local) + len(remote)) / 24.0,
        },
        "corrigendum": {
            "state": (
                "two_post_preregistration_evidence_plumbing_corrections_plus_"
                "one_operator_pre_adapter_incident_plus_one_post_collection_"
                "analysis_order_correction_plus_one_post_collection_time_grid_"
                "correction"
            ),
            "corrigendum_count": 5,
            "evidence_plumbing_correction_count": 2,
            "post_collection_analysis_only_correction_count": 2,
            "excluded_evidence_tree_count": 4,
            "invalid_collection_campaign_count": 2,
            "operator_pre_adapter_failed_attempt_count": 1,
            "operator_pre_adapter_worker_process_count": 1,
            "operator_pre_adapter_trajectory_case_count": 0,
            "operator_pre_adapter_simulator_step_count": 0,
            "valid_under_prior_source_but_superseded_tree_count": 1,
            "mujoco_cases_executed_in_excluded_evidence": 9,
            "ovphysx_cases_executed_in_excluded_evidence": 1,
            "final_analysis_cases_admitted_from_excluded_evidence": 0,
            "remote_or_gpu_execution_occurred_in_excluded_evidence": True,
            "automated_structural_or_numerical_validation_occurred": True,
            "trajectory_sample_values_may_have_been_reviewed": True,
            "trajectory_sample_values_used_to_select_or_design_correction": False,
            "active_trajectory_payloads_deserialized_before_order_correction": True,
            "active_trajectory_values_finite_and_sanity_checked": True,
            "active_trajectory_values_used_to_choose_order_correction": False,
            "runtime_readback_diagnostic_result_reviewed": True,
            "combined_metrics_computed_before_order_correction": False,
            "scientific_label_assigned_before_order_correction": False,
            "bundle_published_before_order_correction": False,
            "order_correction_required_simulator_rerun": False,
            "first_analysis_bundle_created_locally": True,
            "first_analysis_bundle_published": False,
            "first_analysis_validation_status": "INVALID",
            "first_analysis_scientific_label": None,
            "first_analysis_evidence_checks_all_passed": True,
            "first_analysis_primary_metric_evaluation_passed": False,
            "first_analysis_public_state_internally_consistent": False,
            "first_analysis_unique_metric_failure_count": 1,
            "time_grid_correction_required_simulator_rerun": False,
            "time_grid_correction_selected_from_effect_threshold_or_label": False,
            "trajectory_values_reviewed_before_time_grid_correction": True,
            "primary_metrics_and_final_s_values_reviewed_before_time_grid_correction": True,
            "scientific_metrics_computed_from_excluded_evidence": False,
            "scientific_label_assigned_to_excluded_evidence": False,
            "excluded_evidence_reused": False,
            "full_matrix_rerun": True,
            "scientific_protocol_changed": False,
        },
        "process_isolation": {
            "verified_case_count": len(tagged_fresh_process_ids),
            "expected_case_count": 24,
            "local_identity_domain": "os_process_identity_v1",
            "remote_identity_domain": "worker_process_identity_v1",
            "identifiers_published": False,
        },
        "mapping": {
            "joint_mapping": "44/44",
            "distal_frame_mapping": "10/10",
        },
        "contacts": {
            "mujoco_observed_contact_count": 0,
            "ovphysx_contact_observation_performed": False,
            "ovphysx_contact_count": None,
            "ovphysx_interpretation": "not_measured_not_zero_or_absence_evidence",
        },
        "dt_claim_boundary": {
            "requested_and_configured_dt_verified": True,
            "recorded_timestamp_grid_same_step_verified": True,
            "recorded_timestamp_grid_same_length_verified": True,
            "recorded_timestamp_grid_compatibility_abs_tolerance_s": (
                TIME_GRID_ABS_TOLERANCE_S
            ),
            "recorded_timestamp_values_modified_or_rounded": False,
            "compiled_runtime_effective_dt_exposed": False,
            "dt_halving_is_a_trajectory_validity_check": True,
        },
        "position_target_readback": {
            "verified_case_count": sum(
                1
                for row in plan["ovphysx_cases"]
                if remote[row["experiment_case_id"]].result.provenance.get(
                    "position_target_readback_verified"
                )
                is True
            ),
            "expected_case_count": 16,
            "aggregate_max_abs_error_rad": max(position_target_errors),
            "limit_rad": position_target_limit,
            "source": "articulation_data_joint_pos_target_torch",
            "values_published": False,
        },
        "runtime_friction_observation": runtime_friction_observation,
        "validity": {
            "checks_all_passed": checks_all_passed,
            "primary_metric_evaluation_passed": (
                primary_metric_evaluation_passed
            ),
            "all_passed": all_passed,
            "checks": dict(sorted(checks.items())),
            **metric_groups,
        },
        "primary_metrics": {
            "cells": [cell.to_dict() for cell in decision.cells],
            "s_values": list(decision.s_values),
        } if decision.status is not SensitivityStatus.INVALID else None,
        "provenance": {
            "private_plan_sha256": plan_digest,
            "public_protocol_sha256": public_protocol_digest,
            "public_protocol_canonical_sha256": public_protocol_canonical_digest,
            "corrigendum_sha256": corrigendum_file_digest,
            "corrigendum_canonical_sha256": corrigendum_canonical_digest,
            "admission_corrigendum_sha256": admission_corrigendum_file_digest,
            "admission_corrigendum_canonical_sha256": (
                admission_corrigendum_canonical_digest
            ),
            "asset_preflight_corrigendum_sha256": (
                asset_preflight_corrigendum_file_digest
            ),
            "asset_preflight_corrigendum_canonical_sha256": (
                asset_preflight_corrigendum_canonical_digest
            ),
            "runtime_order_corrigendum_sha256": (
                runtime_order_corrigendum_file_digest
            ),
            "runtime_order_corrigendum_canonical_sha256": (
                runtime_order_corrigendum_canonical_digest
            ),
            "time_grid_corrigendum_sha256": (
                time_grid_corrigendum_file_digest
            ),
            "time_grid_corrigendum_canonical_sha256": (
                time_grid_corrigendum_canonical_digest
            ),
            "implementation_revision": inputs["freeze_b_source_revision"],
            "implementation_tree": inputs["freeze_b_source_tree"],
            "preregistration_revision": source_identity["preregistration_revision"],
            "preregistration_tree": source_identity["preregistration_tree"],
            "correction_revision": source_identity["correction_revision"],
            "correction_tree": source_identity["correction_tree"],
            "first_execution_revision": source_identity["first_execution_revision"],
            "first_execution_tree": source_identity["first_execution_tree"],
            "admission_correction_revision": source_identity[
                "admission_correction_revision"
            ],
            "admission_correction_tree": source_identity[
                "admission_correction_tree"
            ],
            "asset_preflight_incident_source_revision": source_identity[
                "incident_source_revision"
            ],
            "asset_preflight_incident_source_tree": source_identity[
                "incident_source_tree"
            ],
            "execution_revision": campaign_revision,
            "execution_tree": campaign_tree,
            "first_analysis_revision": FIRST_ANALYSIS_REVISION,
            "first_analysis_tree": FIRST_ANALYSIS_TREE,
            "analysis_revision": analysis_revision,
            "analysis_tree": analysis_tree,
            "r1_readback_bundle_root_sha256": inputs["readback_bundle_root_sha256"],
            "formal_gate0_bundle_root_sha256": inputs["formal_gate0_bundle_root_sha256"],
            "superseded_analysis_bundle": {
                "classification": superseded_analysis_identity[
                    "classification"
                ],
                "analysis_revision": superseded_analysis_identity[
                    "analysis_revision"
                ],
                "analysis_tree": superseded_analysis_identity[
                    "analysis_tree"
                ],
                "root_sha256": superseded_analysis_identity["root_sha256"],
                "file_count": superseded_analysis_identity["file_count"],
                "total_bytes": superseded_analysis_identity["total_bytes"],
                "validation_status": superseded_analysis_identity[
                    "validation_status"
                ],
                "scientific_label": None,
                "published": False,
                "reused_as_final_result": False,
            },
        },
        "claim_boundary": {
            "study": "unofficial_simulation_only_kitless_ovphysx_native_input_sensitivity",
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
            "cross_engine_parameter_equivalence_claimed": False,
            "official_bug_claimed": False,
            "hardware_or_sim2real_claimed": False,
        },
    }
    _validate_summary_decision_consistency(
        summary,
        decision=decision,
        checks=checks,
    )
    _validate_public_safe(summary)
    report = _render_report(summary)
    _validate_public_safe(report)
    staging = create_staging_root(destination)
    public_staging: Path | None = None
    promoted: Path | None = None
    try:
        (staging / "private").mkdir()
        (staging / "inputs/source").mkdir(parents=True)
        (staging / "public/results/freeze-b").mkdir(parents=True)
        _copy_file_exclusive(private_plan_path, staging / "private/freeze_b.private-plan.json")
        _copy_tree_strict(local_evidence, staging / "private/local-evidence", "local evidence")
        _copy_tree_strict(remote_evidence, staging / "private/remote-evidence", "remote evidence")
        (staging / "private/excluded-evidence").mkdir()
        for label in EXCLUDED_EVIDENCE_LABELS:
            _copy_tree_strict(
                excluded_evidence[label],
                staging / "private/excluded-evidence" / label,
                f"excluded evidence {label}",
            )
        r1_subset = [
            "bundle.json",
            "private/readback.json",
            "results/summary.json",
            *[
                f"private/remote-evidence/cases/{case_id}/{case_id}.json"
                for case_id in inputs["readback_case_sha256"]
            ],
        ]
        _copy_parent_subset(readback_bundle, staging / "private/r1-subset", r1_subset)
        formal_subset = [
            "bundle.json",
            "manifest/gate0.json",
            "results/finalization.json",
            "results/report.json",
            *[
                f"evidence/{backend}/cases/{case_id}/{case_id}.run.json"
                for backend, case_id in sorted(formal)
            ],
        ]
        _copy_parent_subset(formal_bundle, staging / "private/formal-subset", formal_subset)
        _copy_file_exclusive(public_protocol_path, staging / "inputs/public-protocol.json")
        _copy_file_exclusive(corrigendum_path, staging / "inputs/corrigendum.json")
        _copy_file_exclusive(
            admission_corrigendum_path,
            staging / "inputs/admission-corrigendum.json",
        )
        _copy_file_exclusive(
            asset_preflight_corrigendum_path,
            staging / "inputs/asset-preflight-corrigendum.json",
        )
        _copy_file_exclusive(
            runtime_order_corrigendum_path,
            staging / "inputs/runtime-order-corrigendum.json",
        )
        _copy_file_exclusive(
            time_grid_corrigendum_path,
            staging / "inputs/time-grid-corrigendum.json",
        )
        superseded_subset = list(SUPERSEDED_ANALYSIS_SUBSET_PATHS)
        _copy_parent_subset(
            superseded_analysis_bundle,
            staging / "private/superseded-analysis-subset",
            superseded_subset,
        )
        _copy_file_exclusive(manifest_path, staging / "inputs/gate0.json")
        for relative in REQUIRED_SOURCE_PATHS:
            source = source_bytes_root / PurePosixPath(relative)
            _copy_file_exclusive(source, staging / "inputs/source" / Path(relative).name)
        write_json_atomic(staging / "private/decision.json", decision_payload)
        write_json_atomic(staging / "private/evidence-bindings.json", evidence_bindings)
        write_json_atomic(
            staging / "private/runtime-friction-observation.json",
            private_runtime_friction_observation,
        )
        write_json_atomic(staging / "private/source-identity.json", source_identity)
        write_json_atomic(
            staging / "private/excluded-evidence.json",
            {
                "schema_version": 1,
                "included_in_final_metrics": False,
                "identities": excluded_evidence_identities,
            },
        )
        write_json_atomic(
            staging / "private/active-evidence.json",
            {
                "schema_version": 1,
                "included_in_final_metrics": True,
                "campaign_revision": campaign_revision,
                "campaign_tree": campaign_tree,
                "identities": active_evidence_identities,
            },
        )
        write_json_atomic(
            staging / "private/input-tree-fingerprints.json",
            input_tree_fingerprints,
        )
        write_json_atomic(staging / "public/results/freeze-b/summary.json", summary)
        write_text_exclusive(staging / "public/results/freeze-b/report.md", report)
        write_json_atomic(
            staging / "private/finalization.json",
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "validation_status": validation_status,
                "scientific_label": scientific_label,
                "private_bundle": True,
                "public_safe_outputs": [
                    "public/results/freeze-b/summary.json",
                    "public/results/freeze-b/report.md",
                ],
                "public_protocol_sha256": public_protocol_digest,
                "corrigendum_sha256": corrigendum_file_digest,
                "corrigendum_canonical_sha256": corrigendum_canonical_digest,
                "admission_corrigendum_sha256": (
                    admission_corrigendum_file_digest
                ),
                "admission_corrigendum_canonical_sha256": (
                    admission_corrigendum_canonical_digest
                ),
                "asset_preflight_corrigendum_sha256": (
                    asset_preflight_corrigendum_file_digest
                ),
                "asset_preflight_corrigendum_canonical_sha256": (
                    asset_preflight_corrigendum_canonical_digest
                ),
                "runtime_order_corrigendum_sha256": (
                    runtime_order_corrigendum_file_digest
                ),
                "runtime_order_corrigendum_canonical_sha256": (
                    runtime_order_corrigendum_canonical_digest
                ),
                "time_grid_corrigendum_sha256": (
                    time_grid_corrigendum_file_digest
                ),
                "time_grid_corrigendum_canonical_sha256": (
                    time_grid_corrigendum_canonical_digest
                ),
                "superseded_analysis_bundle": (
                    superseded_analysis_identity
                ),
                "superseded_analysis_subset": superseded_subset,
                "active_evidence": active_evidence_identities,
                "excluded_evidence": excluded_evidence_identities,
                "private_plan_sha256": plan_digest,
                "source_identity": source_identity,
                "remote_source_archive_sha256": remote_launcher["source_archive_sha256"],
                "remote_source_snapshot_sha256": remote_launcher["source_snapshot_sha256"],
                "formal_gate0_status_unchanged": "DIVERGENT",
                "formal_gate0_pass_ready_unchanged": False,
            },
        )

        # Close the source-to-bundle TOCTOU window.  The complete input trees
        # must still match their pre-analysis fingerprints, and the full local
        # and remote evidence copies must match those same fingerprints.
        current_input_fingerprints = {
            "readback_bundle": _tree_fingerprint(readback_bundle, "R1 bundle"),
            "formal_bundle": _tree_fingerprint(formal_bundle, "formal bundle"),
            "local_evidence": _tree_fingerprint(local_evidence, "local evidence"),
            "remote_evidence": _tree_fingerprint(remote_evidence, "remote evidence"),
            "superseded_analysis_bundle": _tree_fingerprint(
                superseded_analysis_bundle,
                "superseded H analysis bundle",
            ),
            **{
                f"excluded_{label}": _tree_fingerprint(
                    path, f"excluded evidence {label}"
                )
                for label, path in excluded_evidence.items()
            },
        }
        if current_input_fingerprints != input_tree_fingerprints:
            raise FreezeBFinalizationError("an input evidence tree changed during analysis")
        if _tree_fingerprint(
            superseded_analysis_source,
            "source superseded H analysis bundle",
        ) != expected_superseded_tree_fingerprint:
            raise FreezeBFinalizationError(
                "source superseded H analysis bundle changed during analysis"
            )
        if _tree_fingerprint(
            staging / "private/local-evidence", "copied local evidence"
        ) != input_tree_fingerprints["local_evidence"]:
            raise FreezeBFinalizationError("copied local evidence differs from analyzed bytes")
        if _tree_fingerprint(
            staging / "private/remote-evidence", "copied remote evidence"
        ) != input_tree_fingerprints["remote_evidence"]:
            raise FreezeBFinalizationError("copied remote evidence differs from analyzed bytes")
        for label in EXCLUDED_EVIDENCE_LABELS:
            if _tree_fingerprint(
                staging / "private/excluded-evidence" / label,
                f"copied excluded evidence {label}",
            ) != input_tree_fingerprints[f"excluded_{label}"]:
                raise FreezeBFinalizationError(
                    f"copied excluded evidence {label} differs from analyzed bytes"
                )

        copied_plan = staging / "private/freeze_b.private-plan.json"
        copied_protocol = staging / "inputs/public-protocol.json"
        copied_corrigendum = staging / "inputs/corrigendum.json"
        copied_admission_corrigendum = staging / "inputs/admission-corrigendum.json"
        copied_asset_preflight_corrigendum = (
            staging / "inputs/asset-preflight-corrigendum.json"
        )
        copied_runtime_order_corrigendum = (
            staging / "inputs/runtime-order-corrigendum.json"
        )
        copied_time_grid_corrigendum = (
            staging / "inputs/time-grid-corrigendum.json"
        )
        for source, copied, identity, label in (
            (private_plan_path, copied_plan, input_file_identities["private_plan"], "private plan"),
            (public_protocol_path, copied_protocol, input_file_identities["public_protocol"], "public protocol"),
            (corrigendum_path, copied_corrigendum, input_file_identities["corrigendum"], "corrigendum"),
            (
                admission_corrigendum_path,
                copied_admission_corrigendum,
                input_file_identities["admission_corrigendum"],
                "admission corrigendum",
            ),
            (
                asset_preflight_corrigendum_path,
                copied_asset_preflight_corrigendum,
                input_file_identities["asset_preflight_corrigendum"],
                "asset preflight corrigendum",
            ),
            (
                runtime_order_corrigendum_path,
                copied_runtime_order_corrigendum,
                input_file_identities["runtime_order_corrigendum"],
                "runtime order corrigendum",
            ),
            (
                time_grid_corrigendum_path,
                copied_time_grid_corrigendum,
                input_file_identities["time_grid_corrigendum"],
                "time-grid corrigendum",
            ),
        ):
            _require_file_identity(source, identity, f"source {label}")
            _require_file_identity(copied, identity, f"copied {label}")
        for relative, identity in tracked_source_identities.items():
            _require_file_identity(
                source_bytes_root / PurePosixPath(relative),
                identity,
                f"source {relative}",
            )
            if source_bytes_root != project_root:
                _require_file_identity(
                    project_root / PurePosixPath(relative),
                    identity,
                    f"working-tree source {relative}",
                )
            _require_file_identity(
                staging / "inputs/source" / Path(relative).name,
                identity,
                f"copied source {relative}",
            )
        _require_file_identity(
            staging / "inputs/gate0.json",
            tracked_source_identities[GATE0_MANIFEST_RELATIVE_PATH.as_posix()],
            "copied canonical manifest",
        )
        _require_file_identity(
            copied_protocol,
            tracked_source_identities[PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix()],
            "copied public protocol",
        )
        _require_file_identity(
            copied_corrigendum,
            tracked_source_identities[CORRIGENDUM_RELATIVE_PATH.as_posix()],
            "copied corrigendum",
        )
        _require_file_identity(
            copied_admission_corrigendum,
            tracked_source_identities[
                ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ],
            "copied admission corrigendum",
        )
        _require_file_identity(
            copied_asset_preflight_corrigendum,
            tracked_source_identities[
                ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ],
            "copied asset preflight corrigendum",
        )
        _require_file_identity(
            copied_runtime_order_corrigendum,
            tracked_source_identities[
                RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ],
            "copied runtime order corrigendum",
        )
        _require_file_identity(
            copied_time_grid_corrigendum,
            tracked_source_identities[
                TIME_GRID_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ],
            "copied time-grid corrigendum",
        )
        copied_public = load_public_protocol(copied_protocol)
        copied_corrigendum_value = _validate_corrigendum(
            _strict_json(copied_corrigendum, "copied corrigendum"),
            implementation_revision=str(inputs["freeze_b_source_revision"]),
            implementation_tree=str(inputs["freeze_b_source_tree"]),
        )
        copied_private = load_private_plan(copied_plan, public_protocol=copied_public)
        copied_admission_corrigendum_value = _validate_admission_corrigendum(
            _strict_json(
                copied_admission_corrigendum, "copied admission corrigendum"
            ),
            implementation_revision=str(inputs["freeze_b_source_revision"]),
            implementation_tree=str(inputs["freeze_b_source_tree"]),
            first_correction_revision=str(source_identity["correction_revision"]),
            first_correction_tree=str(source_identity["correction_tree"]),
        )
        copied_asset_preflight_corrigendum_value = (
            _validate_asset_preflight_corrigendum(
                _strict_json(
                    copied_asset_preflight_corrigendum,
                    "copied asset preflight corrigendum",
                ),
                implementation_revision=str(inputs["freeze_b_source_revision"]),
                implementation_tree=str(inputs["freeze_b_source_tree"]),
                first_correction_revision=str(
                    source_identity["correction_revision"]
                ),
                first_correction_tree=str(source_identity["correction_tree"]),
                admission_correction_revision=str(
                    source_identity["admission_correction_revision"]
                ),
                admission_correction_tree=str(
                    source_identity["admission_correction_tree"]
                ),
            )
        )
        copied_runtime_order_corrigendum_value = (
            _validate_runtime_order_corrigendum(
                _strict_json(
                    copied_runtime_order_corrigendum,
                    "copied runtime order corrigendum",
                )
            )
        )
        copied_time_grid_corrigendum_value = (
            _validate_time_grid_corrigendum(
                _strict_json(
                    copied_time_grid_corrigendum,
                    "copied time-grid corrigendum",
                )
            )
        )
        if (
            private_plan_sha256(copied_private) != plan_digest
            or canonical_json_sha256(copied_public) != public_protocol_canonical_digest
            or canonical_json_sha256(copied_corrigendum_value)
            != corrigendum_canonical_digest
            or canonical_json_sha256(copied_admission_corrigendum_value)
            != admission_corrigendum_canonical_digest
            or canonical_json_sha256(
                copied_asset_preflight_corrigendum_value
            )
            != asset_preflight_corrigendum_canonical_digest
            or canonical_json_sha256(
                copied_runtime_order_corrigendum_value
            )
            != runtime_order_corrigendum_canonical_digest
            or canonical_json_sha256(
                copied_time_grid_corrigendum_value
            )
            != time_grid_corrigendum_canonical_digest
        ):
            raise FreezeBFinalizationError("copied private/public protocol binding drifted")

        copied_local = staging / "private/local-evidence"
        _verify_hash_inventory(
            copied_local / "launcher", copied_local / "launcher/evidence.sha256"
        )
        for case_dir in (copied_local / "cases").iterdir():
            _verify_hash_inventory(case_dir, case_dir / "evidence.sha256")
        copied_remote = staging / "private/remote-evidence"
        _verify_hash_inventory(
            copied_remote, copied_remote / "launcher/evidence.sha256"
        )
        for case_dir in (copied_remote / "cases").iterdir():
            _verify_hash_inventory(case_dir, case_dir / "evidence.sha256")
        verify_copied_bundle_subset(
            staging / "private/r1-subset/bundle.json",
            expected_root_sha256=str(inputs["readback_bundle_root_sha256"]),
            copied_payloads={
                relative: staging / "private/r1-subset" / PurePosixPath(relative)
                for relative in r1_subset
                if relative != "bundle.json"
            },
        )
        verify_copied_bundle_subset(
            staging / "private/formal-subset/bundle.json",
            expected_root_sha256=str(inputs["formal_gate0_bundle_root_sha256"]),
            copied_payloads={
                relative: staging / "private/formal-subset" / PurePosixPath(relative)
                for relative in formal_subset
                if relative != "bundle.json"
            },
        )
        _verify_superseded_analysis_subset(
            staging / "private/superseded-analysis-subset"
        )
        if (
            _git(project_root, "rev-parse", "HEAD") != analysis_revision
            or _git(project_root, "rev-parse", "HEAD^{tree}") != analysis_tree
            or _git(
                project_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        ):
            raise FreezeBFinalizationError(
                "HEAD/tree/worktree changed before final bundle publication"
            )
        write_bundle_manifest(staging, bundle_payload_paths(staging))
        verification = dict(verify_exact_bundle(staging))

        if public_destination is not None:
            public_staging = create_staging_root(public_destination)
            write_json_atomic(public_staging / "summary.json", summary)
            write_text_exclusive(public_staging / "report.md", report)
            _validate_public_safe(_strict_json(public_staging / "summary.json", "public summary"))
            _validate_public_safe((public_staging / "report.md").read_text(encoding="utf-8"))
        promoted, public_destination = _promote_output_pair(
            private_staging=staging,
            private_destination=destination,
            expected_private_root_sha256=str(verification["root_sha256"]),
            public_staging=public_staging,
            public_destination=public_destination,
        )
    except BaseException:
        if staging.exists():
            _remove_owned_tree(staging, expected_path=staging)
        if public_staging is not None and public_staging.exists():
            _remove_owned_tree(public_staging, expected_path=public_staging)
        if promoted is not None and promoted.exists():
            _remove_owned_tree(promoted, expected_path=destination)
        raise
    assert promoted is not None
    return promoted, public_destination, {
        **verification,
        "validation_status": validation_status,
        "scientific_label": scientific_label,
        "completed_case_count": len(local) + len(remote),
        "joint_mapping": "44/44",
        "distal_frame_mapping": "10/10",
        "formal_gate0_status_unchanged": "DIVERGENT",
        "formal_gate0_pass_ready_unchanged": False,
    }


def finalize_freeze_b(
    *,
    project_root: Path,
    public_protocol_path: Path,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
    runtime_order_corrigendum_path: Path,
    time_grid_corrigendum_path: Path,
    private_plan_path: Path,
    readback_bundle: Path,
    formal_bundle: Path,
    local_evidence: Path,
    remote_evidence: Path,
    superseded_analysis_bundle: Path,
    excluded_evidence: Mapping[str, Path],
    output_dir: Path,
    public_output_dir: Path | None,
    campaign_revision: str,
    analysis_revision: str,
) -> tuple[Path, Path | None, dict[str, object]]:
    """Finalize exclusively from a verified copy-first analysis snapshot."""

    project_root = _resolved_real(project_root, "project root", directory=True)
    public_protocol_path = _resolved_real(
        public_protocol_path,
        "public protocol",
        directory=False,
    )
    corrigendum_path = _resolved_real(
        corrigendum_path,
        "corrigendum",
        directory=False,
    )
    admission_corrigendum_path = _resolved_real(
        admission_corrigendum_path,
        "admission corrigendum",
        directory=False,
    )
    asset_preflight_corrigendum_path = _resolved_real(
        asset_preflight_corrigendum_path,
        "asset preflight corrigendum",
        directory=False,
    )
    runtime_order_corrigendum_path = _resolved_real(
        runtime_order_corrigendum_path,
        "runtime order corrigendum",
        directory=False,
    )
    time_grid_corrigendum_path = _resolved_real(
        time_grid_corrigendum_path,
        "time-grid corrigendum",
        directory=False,
    )
    private_plan_path = _resolved_real(
        private_plan_path,
        "private plan",
        directory=False,
    )
    readback_bundle = _resolved_real(readback_bundle, "R1 bundle", directory=True)
    formal_bundle = _resolved_real(formal_bundle, "formal bundle", directory=True)
    local_evidence = _resolved_real(local_evidence, "local evidence", directory=True)
    remote_evidence = _resolved_real(remote_evidence, "remote evidence", directory=True)
    superseded_analysis_bundle = _resolved_real(
        superseded_analysis_bundle,
        "superseded H analysis bundle",
        directory=True,
    )
    if set(excluded_evidence) != set(EXCLUDED_EVIDENCE_LABELS):
        raise FreezeBFinalizationError(
            "excluded evidence labels must be exactly local-a1, local-a2, remote-a2, local-a3"
        )
    excluded_evidence = {
        label: _resolved_real(
            excluded_evidence[label], f"excluded evidence {label}", directory=True
        )
        for label in EXCLUDED_EVIDENCE_LABELS
    }
    all_evidence_trees = (
        readback_bundle,
        formal_bundle,
        local_evidence,
        remote_evidence,
        superseded_analysis_bundle,
        *excluded_evidence.values(),
    )
    _validate_disjoint_input_trees(all_evidence_trees)
    destination, _public_destination = _validate_output_destinations(
        project_root,
        output_dir=output_dir,
        public_output_dir=public_output_dir,
        input_trees=all_evidence_trees,
    )
    snapshot_final_path = destination.with_name(
        f"{destination.name}.analysis-snapshot"
    )
    analysis_staging = create_staging_root(snapshot_final_path)
    try:
        snapshot = _prepare_analysis_snapshot(
            analysis_staging,
            project_root=project_root,
            readback_bundle=readback_bundle,
            formal_bundle=formal_bundle,
            local_evidence=local_evidence,
            remote_evidence=remote_evidence,
            excluded_evidence=excluded_evidence,
            private_plan_path=private_plan_path,
            public_protocol_path=public_protocol_path,
            corrigendum_path=corrigendum_path,
            admission_corrigendum_path=admission_corrigendum_path,
            asset_preflight_corrigendum_path=(
                asset_preflight_corrigendum_path
            ),
            runtime_order_corrigendum_path=runtime_order_corrigendum_path,
            time_grid_corrigendum_path=time_grid_corrigendum_path,
            superseded_analysis_bundle=superseded_analysis_bundle,
        )
        return _finalize_freeze_b_from_snapshot(
            project_root=project_root,
            public_protocol_path=snapshot.public_protocol,
            corrigendum_path=snapshot.corrigendum,
            admission_corrigendum_path=snapshot.admission_corrigendum,
            asset_preflight_corrigendum_path=(
                snapshot.asset_preflight_corrigendum
            ),
            runtime_order_corrigendum_path=(
                snapshot.runtime_order_corrigendum
            ),
            time_grid_corrigendum_path=(
                snapshot.time_grid_corrigendum
            ),
            private_plan_path=snapshot.private_plan,
            readback_bundle=snapshot.readback_bundle,
            formal_bundle=snapshot.formal_bundle,
            local_evidence=snapshot.local_evidence,
            remote_evidence=snapshot.remote_evidence,
            superseded_analysis_bundle=(
                snapshot.superseded_analysis_bundle
            ),
            excluded_evidence=snapshot.excluded_evidence,
            output_dir=output_dir,
            public_output_dir=public_output_dir,
            campaign_revision=campaign_revision,
            analysis_revision=analysis_revision,
            analysis_manifest_path=snapshot.manifest,
            analysis_source_root=snapshot.source_root,
            canonical_public_protocol_path=public_protocol_path,
            canonical_corrigendum_path=corrigendum_path,
            canonical_admission_corrigendum_path=admission_corrigendum_path,
            canonical_asset_preflight_corrigendum_path=(
                asset_preflight_corrigendum_path
            ),
            canonical_runtime_order_corrigendum_path=(
                runtime_order_corrigendum_path
            ),
            canonical_time_grid_corrigendum_path=(
                time_grid_corrigendum_path
            ),
            canonical_superseded_analysis_bundle=(
                superseded_analysis_bundle
            ),
        )
    finally:
        if analysis_staging.exists():
            _remove_owned_tree(
                analysis_staging,
                expected_path=analysis_staging,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--public-protocol", type=Path, required=True)
    parser.add_argument("--corrigendum", type=Path, required=True)
    parser.add_argument("--admission-corrigendum", type=Path, required=True)
    parser.add_argument("--asset-preflight-corrigendum", type=Path, required=True)
    parser.add_argument("--runtime-order-corrigendum", type=Path, required=True)
    parser.add_argument("--time-grid-corrigendum", type=Path, required=True)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--readback-bundle", type=Path, required=True)
    parser.add_argument("--formal-bundle", type=Path, required=True)
    parser.add_argument("--local-evidence", type=Path, required=True)
    parser.add_argument("--remote-evidence", type=Path, required=True)
    parser.add_argument("--superseded-analysis-bundle", type=Path, required=True)
    parser.add_argument(
        "--excluded-evidence",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--campaign-revision", required=True)
    parser.add_argument("--analysis-revision", required=True)
    return parser


def _parse_excluded_evidence_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or label not in EXCLUDED_EVIDENCE_LABELS or not raw_path:
            raise FreezeBFinalizationError(
                "excluded evidence must use local-a1=PATH, local-a2=PATH, "
                "remote-a2=PATH, or local-a3=PATH"
            )
        if label in result:
            raise FreezeBFinalizationError(
                f"duplicate excluded evidence label: {label}"
            )
        result[label] = Path(raw_path)
    if set(result) != set(EXCLUDED_EVIDENCE_LABELS):
        raise FreezeBFinalizationError(
            "excluded evidence labels must be exactly local-a1, local-a2, "
            "remote-a2, local-a3"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, public_output, verification = finalize_freeze_b(
            project_root=args.project_root,
            public_protocol_path=args.public_protocol,
            corrigendum_path=args.corrigendum,
            admission_corrigendum_path=args.admission_corrigendum,
            asset_preflight_corrigendum_path=(
                args.asset_preflight_corrigendum
            ),
            runtime_order_corrigendum_path=args.runtime_order_corrigendum,
            time_grid_corrigendum_path=args.time_grid_corrigendum,
            private_plan_path=args.private_plan,
            readback_bundle=args.readback_bundle,
            formal_bundle=args.formal_bundle,
            local_evidence=args.local_evidence,
            remote_evidence=args.remote_evidence,
            superseded_analysis_bundle=args.superseded_analysis_bundle,
            excluded_evidence=_parse_excluded_evidence_arguments(
                args.excluded_evidence
            ),
            output_dir=args.output,
            public_output_dir=args.public_output,
            campaign_revision=args.campaign_revision,
            analysis_revision=args.analysis_revision,
        )
    except Exception as exc:
        sys.stderr.write(f"Freeze B finalization failed: {type(exc).__name__}: {exc}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "output": output.name,
                "public_output": public_output.name if public_output is not None else None,
                **verification,
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
