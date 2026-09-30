#!/usr/bin/env python3
"""Run one frozen OVPhysX legacy-joint-friction sensitivity case.

This worker is deliberately private-plan driven.  It executes exactly one
``small_step`` case in one fresh, headless, kit-less process and writes the
backend-neutral :class:`AdapterRunResult` JSON used elsewhere by the project.
It does not compare simulators or assign a scientific verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import sys
from typing import Mapping, Sequence

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.adapters.ovphysx import OVPhysXAdapter
from wave_asset_qa.parity.bundle import sha256_file, write_json_atomic
from wave_asset_qa.parity.contracts import HandSide, Simulator
from wave_asset_qa.parity.diagnostics import asset_tree_sha256, is_link_like
from wave_asset_qa.parity.runner import adapter_run_result_from_dict
from wave_asset_qa.parity.scenarios import (
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)
from wave_asset_qa.parity.sensitivity import (
    canonical_json_sha256,
    experiment_case,
    load_private_plan,
    load_public_protocol,
    private_plan_sha256,
)


EXPECTED_APPROVED_ROOT = Path("/data/home/exampleuser/sharpa-wave-asset-qa-gate0")
_PLAN_TOP_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "inputs",
        "hand_plans",
        "ovphysx_cases",
        "mujoco_cases",
        "run_order",
    }
)
_HAND_PLAN_KEYS = frozenset(
    {
        "hand",
        "joint_names",
        "joint_prim_paths",
        "expected_pre_values",
        "sham_write_values",
        "zero_write_values",
        "expected_pre_float32_sha256",
    }
)
_CASE_KEYS = frozenset(
    {
        "experiment_case_id",
        "canonical_case_id",
        "backend",
        "role",
        "hand",
        "timestep_variant",
        "dt_s",
        "repeat_index",
    }
)
_INPUT_KEYS = frozenset(
    {
        "freeze_a_config_sha256",
        "readback_bundle_root_sha256",
        "readback_summary_sha256",
        "readback_source_revision",
        "readback_source_tree",
        "readback_case_sha256",
        "formal_gate0_bundle_root_sha256",
        "formal_gate0_source_revision",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "asset_commit",
        "asset_git_tree",
        "canonical_lf_asset_tree_sha256",
        "freeze_b_source_revision",
        "freeze_b_source_tree",
        "formal_crosssim_window_rmse",
    }
)
_INTERVENTION_EVIDENCE_KEYS = frozenset(
    {
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
)
_INTERVENTION_RECORD_KEYS = frozenset(
    {
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
)
_SIMULATION_CONFIGURATION_V2_KEYS = frozenset(
    {
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
)
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_EXPERIMENT_CASE_RE = re.compile(
    r"^freeze_b\.ovphysx\.(left|right)\.small_step\."
    r"(base|halved)\.(sham|zero)\.r(01|02)$"
)
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FreezeBWorkerError(RuntimeError):
    """A fail-closed worker input, environment, or evidence error."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--public-protocol", required=True, type=Path)
    parser.add_argument("--public-protocol-file-sha256", required=True)
    parser.add_argument("--public-protocol-canonical-sha256", required=True)
    parser.add_argument("--private-plan", required=True, type=Path)
    parser.add_argument("--private-plan-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda:0",))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--asset-tree-sha256", required=True)
    return parser


def _exact_object(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise FreezeBWorkerError(
            f"{label} must contain exactly: " + ", ".join(sorted(keys))
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FreezeBWorkerError(f"{label} must be a non-empty string")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeBWorkerError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise FreezeBWorkerError(f"{label} must be a positive finite number")
    return number


def _float32(value: object, label: str) -> tuple[float, bytes]:
    number = _finite(value, label)
    try:
        payload = struct.pack("!f", number)
        rounded = struct.unpack("!f", payload)[0]
    except (OverflowError, struct.error) as exc:
        raise FreezeBWorkerError(f"{label} is not finite float32") from exc
    if not math.isfinite(rounded):
        raise FreezeBWorkerError(f"{label} is not finite float32")
    return rounded, payload


def _float32_vector_sha256(names: Sequence[str], values: Mapping[str, float]) -> str:
    """Hash names plus binary32 values so order and exact bits are committed."""

    digest = sha256(b"waveqa-freeze-b-float32-v1\0")
    for name in names:
        encoded = name.encode("utf-8")
        _, payload = _float32(values[name], f"expected_pre_values.{name}")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(payload)
    return digest.hexdigest()


def _number_map(
    value: object,
    names: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(names):
        raise FreezeBWorkerError(f"{label} must exactly cover the 22 manifest joints")
    result: dict[str, float] = {}
    for name in names:
        rounded, _ = _float32(value[name], f"{label}.{name}")
        result[name] = rounded
    return result


def _validate_hand_plan(
    raw: object,
    *,
    hand: object,
) -> dict[str, object]:
    plan = _exact_object(raw, _HAND_PLAN_KEYS, "private hand plan")
    side = hand.side.value
    if plan["hand"] != side:
        raise FreezeBWorkerError("private hand plan side does not match the case")
    raw_names = plan["joint_names"]
    raw_paths = plan["joint_prim_paths"]
    if not isinstance(raw_names, list) or tuple(raw_names) != hand.joint_names:
        raise FreezeBWorkerError("private hand plan joint order differs from manifest")
    if (
        not isinstance(raw_paths, dict)
        or set(raw_paths) != set(hand.joint_names)
        or len(set(raw_paths.values())) != 22
        or any(
            not isinstance(path, str)
            or not path.startswith("/")
            or ".." in path.split("/")
            for path in raw_paths.values()
        )
    ):
        raise FreezeBWorkerError("private hand plan joint prim paths are invalid")
    names = hand.joint_names
    for name in names:
        if raw_paths[name] != f"/World/Env_0/Robot/joints/{name}":
            raise FreezeBWorkerError("private hand plan joint prim path drifted")
    expected = _number_map(plan["expected_pre_values"], names, "expected_pre_values")
    sham = _number_map(plan["sham_write_values"], names, "sham_write_values")
    zero = _number_map(plan["zero_write_values"], names, "zero_write_values")
    for name in names:
        expected_value, expected_bits = _float32(expected[name], f"expected.{name}")
        sham_value, sham_bits = _float32(sham[name], f"sham.{name}")
        zero_value, zero_bits = _float32(zero[name], f"zero.{name}")
        if expected_value <= 0.0:
            raise FreezeBWorkerError("Freeze B expected pre-values must be positive")
        if sham_value != expected_value or sham_bits != expected_bits:
            raise FreezeBWorkerError("sham writes must exactly preserve float32 bits")
        if zero_value != 0.0 or zero_bits != b"\x00\x00\x00\x00":
            raise FreezeBWorkerError("zero writes must be positive float32 zero")
    expected_digest = _string(
        plan["expected_pre_float32_sha256"], "expected_pre_float32_sha256"
    )
    if _SHA256_RE.fullmatch(expected_digest) is None:
        raise FreezeBWorkerError("expected_pre_float32_sha256 is malformed")
    # The contract module and adapter remain authoritative.  Checking the
    # locally reproducible digest catches transport/order corruption before GPU use.
    if _float32_vector_sha256(names, expected) != expected_digest:
        raise FreezeBWorkerError("expected pre-value float32 vector hash mismatch")
    return {
        "hand": side,
        "joint_names": list(names),
        "joint_prim_paths": {name: raw_paths[name] for name in names},
        "expected_pre_values": expected,
        "sham_write_values": sham,
        "zero_write_values": zero,
        "expected_pre_float32_sha256": expected_digest,
    }


def _validate_case(raw: object, *, backend: str) -> dict[str, object]:
    case = _exact_object(raw, _CASE_KEYS, f"private {backend} case")
    experiment_id = _string(case["experiment_case_id"], "experiment_case_id")
    canonical_id = _string(case["canonical_case_id"], "canonical_case_id")
    if case["backend"] != backend:
        raise FreezeBWorkerError(f"private case backend must be {backend}")
    expected_roles = {"sham", "zero"} if backend == "ovphysx" else {"control"}
    if case["role"] not in expected_roles:
        raise FreezeBWorkerError(
            f"private {backend} case role must be one of {sorted(expected_roles)}"
        )
    if case["hand"] not in {"left", "right"}:
        raise FreezeBWorkerError("private case hand must be left or right")
    if case["timestep_variant"] not in {"base", "halved"}:
        raise FreezeBWorkerError("private case timestep_variant is invalid")
    dt = _finite(case["dt_s"], "private case dt_s", positive=True)
    repeat = case["repeat_index"]
    if isinstance(repeat, bool) or repeat not in {1, 2}:
        raise FreezeBWorkerError("private case repeat_index must be 1 or 2")
    if backend == "ovphysx":
        match = _EXPERIMENT_CASE_RE.fullmatch(experiment_id)
        if match is None:
            raise FreezeBWorkerError("OVPhysX experiment_case_id is not canonical Freeze B")
        expected_experiment = (
            f"freeze_b.ovphysx.{case['hand']}.small_step."
            f"{case['timestep_variant']}.{case['role']}.r{repeat:02d}"
        )
        expected_canonical = (
            f"ovphysx.{case['hand']}.small_step."
            f"{case['timestep_variant']}.r{repeat:02d}"
        )
        if experiment_id != expected_experiment or canonical_id != expected_canonical:
            raise FreezeBWorkerError("OVPhysX case identity fields disagree")
    else:
        expected_experiment = (
            f"freeze_b.mujoco.{case['hand']}.small_step."
            f"{case['timestep_variant']}.control.r{repeat:02d}"
        )
        expected_canonical = (
            f"mujoco.{case['hand']}.small_step."
            f"{case['timestep_variant']}.r{repeat:02d}"
        )
        if experiment_id != expected_experiment or canonical_id != expected_canonical:
            raise FreezeBWorkerError("MuJoCo case identity fields disagree")
    return {**case, "dt_s": dt}


def _validated_plan_matrix(plan: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw_ov = plan["ovphysx_cases"]
    raw_mj = plan["mujoco_cases"]
    raw_order = plan["run_order"]
    if not isinstance(raw_ov, list) or not isinstance(raw_mj, list):
        raise FreezeBWorkerError("private plan cases must be arrays")
    ov_cases = tuple(_validate_case(item, backend="ovphysx") for item in raw_ov)
    mj_cases = tuple(_validate_case(item, backend="mujoco") for item in raw_mj)
    if len(ov_cases) != 16:
        raise FreezeBWorkerError("Freeze B requires exactly 16 OVPhysX cases")
    if len(mj_cases) != 8:
        raise FreezeBWorkerError("Freeze B requires exactly 8 local MuJoCo controls")
    ids = [str(item["experiment_case_id"]) for item in (*ov_cases, *mj_cases)]
    if len(ids) != len(set(ids)):
        raise FreezeBWorkerError("private plan experiment case IDs must be unique")
    if (
        not isinstance(raw_order, list)
        or any(not isinstance(item, str) for item in raw_order)
        or len(raw_order) != len(ov_cases)
        or set(raw_order)
        != {str(item["experiment_case_id"]) for item in ov_cases}
    ):
        raise FreezeBWorkerError(
            "private plan run_order must list all 16 OVPhysX cases exactly once"
        )
    observed = {
        (
            item["hand"],
            item["role"],
            item["timestep_variant"],
            item["repeat_index"],
        )
        for item in ov_cases
    }
    expected = {
        (hand, role, timestep, repeat)
        for hand in ("left", "right")
        for role in ("sham", "zero")
        for timestep in ("base", "halved")
        for repeat in (1, 2)
    }
    if observed != expected:
        raise FreezeBWorkerError("OVPhysX matrix is not the frozen 2x2x2x2 design")
    expected_order: list[str] = []
    for hand in ("left", "right"):
        for timestep in ("base", "halved"):
            for repeat in (1, 2):
                roles = ("sham", "zero") if repeat == 1 else ("zero", "sham")
                for role in roles:
                    expected_order.append(
                        f"freeze_b.ovphysx.{hand}.small_step."
                        f"{timestep}.{role}.r{repeat:02d}"
                    )
    if raw_order != expected_order:
        raise FreezeBWorkerError("private run_order is not the counterbalanced Freeze B order")
    return ov_cases


def _validated_hand_plan_map(
    value: object, *, manifest: object
) -> dict[str, dict[str, object]]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, dict) for item in value)
        or [item.get("hand") for item in value] != ["left", "right"]
    ):
        raise FreezeBWorkerError("private plan hand order must be exactly left, right")
    return {
        side: _validate_hand_plan(
            value[index], hand=manifest.hand(HandSide(side))
        )
        for index, side in enumerate(("left", "right"))
    }


def _validate_plan_inputs(
    value: object,
    *,
    manifest: object,
    manifest_path: Path,
    source_revision: str,
    source_tree: str,
    asset_tree_sha256_value: str,
) -> None:
    inputs = _exact_object(value, _INPUT_KEYS, "private plan inputs")
    sha_fields = (
        "freeze_a_config_sha256",
        "readback_bundle_root_sha256",
        "readback_summary_sha256",
        "formal_gate0_bundle_root_sha256",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "canonical_lf_asset_tree_sha256",
    )
    for field in sha_fields:
        if not isinstance(inputs[field], str) or _SHA256_RE.fullmatch(inputs[field]) is None:
            raise FreezeBWorkerError(f"private plan inputs.{field} is malformed")
    for field in (
        "readback_source_revision",
        "readback_source_tree",
        "formal_gate0_source_revision",
        "asset_commit",
        "asset_git_tree",
        "freeze_b_source_revision",
        "freeze_b_source_tree",
    ):
        if not isinstance(inputs[field], str) or _HEX40_RE.fullmatch(inputs[field]) is None:
            raise FreezeBWorkerError(f"private plan inputs.{field} is malformed")
    if inputs["gate0_manifest_file_sha256"] != sha256_file(manifest_path):
        raise FreezeBWorkerError("private plan Gate 0 manifest file hash drifted")
    if inputs["gate0_manifest_semantic_sha256"] != manifest_sha256(manifest):
        raise FreezeBWorkerError("private plan Gate 0 manifest semantic hash drifted")
    if inputs["asset_commit"] != manifest.provenance.commit:
        raise FreezeBWorkerError("private plan asset commit drifted")
    if inputs["asset_git_tree"] != manifest.provenance.asset_git_tree:
        raise FreezeBWorkerError("private plan asset Git tree drifted")
    if inputs["canonical_lf_asset_tree_sha256"] != asset_tree_sha256_value:
        raise FreezeBWorkerError("private plan asset-tree SHA-256 drifted")
    # The private plan is frozen from implementation commit A.  Execution is
    # intentionally deployed from preregistration commit B, whose only added
    # source is the public protocol.  The launcher validates the public/private
    # bridge; this worker records, but never conflates, the two identities.
    readback_cases = inputs["readback_case_sha256"]
    expected_readback_ids = {
        "ovphysx.left.effective_readback.r01",
        "ovphysx.left.effective_readback.r02",
        "ovphysx.right.effective_readback.r01",
        "ovphysx.right.effective_readback.r02",
    }
    if not isinstance(readback_cases, dict) or set(readback_cases) != expected_readback_ids:
        raise FreezeBWorkerError("private plan R1 readback case inventory drifted")
    if any(
        not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        for digest in readback_cases.values()
    ):
        raise FreezeBWorkerError("private plan R1 readback case hash is malformed")
    baselines = inputs["formal_crosssim_window_rmse"]
    if not isinstance(baselines, dict) or set(baselines) != {"left", "right"}:
        raise FreezeBWorkerError("private plan formal baseline hands are invalid")
    for side in ("left", "right"):
        variants = baselines[side]
        if not isinstance(variants, dict) or set(variants) != {"base", "halved"}:
            raise FreezeBWorkerError("private plan formal baseline variants are invalid")
        for variant in ("base", "halved"):
            metrics = variants[variant]
            if not isinstance(metrics, dict) or set(metrics) != {
                "joint_rmse_rad",
                "frame_position_rmse_m",
            }:
                raise FreezeBWorkerError("private plan formal baseline metrics are invalid")
            for name, raw in metrics.items():
                _finite(raw, f"formal baseline {side}.{variant}.{name}", positive=True)


def _float32_hex(value: object, label: str) -> str:
    return _float32(value, label)[1].hex()


def _validate_intervention_evidence(
    provenance: Mapping[str, object],
    *,
    role: str,
    private_plan_sha256: str,
    hand_plan: Mapping[str, object],
) -> None:
    evidence = _exact_object(
        provenance.get("legacy_joint_friction_intervention"),
        _INTERVENTION_EVIDENCE_KEYS,
        "legacy joint-friction intervention evidence",
    )
    expected_literals = {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": private_plan_sha256,
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
    }
    for field, expected in expected_literals.items():
        if type(evidence[field]) is not type(expected) or evidence[field] != expected:
            raise FreezeBWorkerError(f"intervention evidence {field} drifted")
    names = tuple(hand_plan["joint_names"])
    if evidence["joint_order"] != list(names):
        raise FreezeBWorkerError("intervention evidence joint order drifted")
    before = evidence["source_asset_sha256_before"]
    after = evidence["source_asset_sha256_after_cleanup"]
    if (
        not isinstance(before, str)
        or _SHA256_RE.fullmatch(before) is None
        or after != before
        or evidence["source_asset_sha256_unchanged"] is not True
    ):
        raise FreezeBWorkerError("intervention touched or failed to verify the source asset")
    records = evidence["records"]
    if not isinstance(records, list) or len(records) != 22:
        raise FreezeBWorkerError("intervention evidence must contain 22 records")
    expected_values = hand_plan["expected_pre_values"]
    write_values = hand_plan[f"{role}_write_values"]
    prim_paths = hand_plan["joint_prim_paths"]
    for index, raw_record in enumerate(records):
        record = _exact_object(
            raw_record,
            _INTERVENTION_RECORD_KEYS,
            f"intervention record {index}",
        )
        name = names[index]
        prim_path = prim_paths[name]
        expected_hex = _float32_hex(expected_values[name], f"expected {name}")
        write_hex = _float32_hex(write_values[name], f"write {name}")
        if (
            record["canonical_id"] != name
            or record["prim_path"] != prim_path
            or record["property_path"]
            != f"{prim_path}.physxJoint:jointFriction"
        ):
            raise FreezeBWorkerError("intervention record identity drifted")
        expected_record = {
            "observed_pre_float32_hex": expected_hex,
            "expected_pre_float32_hex": expected_hex,
            "write_float32_hex": write_hex,
            "post_write_float32_hex": write_hex,
            "post_reset_float32_hex": write_hex,
            "post_cleanup_float32_hex": expected_hex,
        }
        for field, expected in expected_record.items():
            if record[field] != expected:
                raise FreezeBWorkerError(f"intervention record {name} {field} drifted")
        numeric_record = {
            "observed_pre_value": expected_hex,
            "expected_pre_value": expected_hex,
            "write_value": write_hex,
            "post_write_value": write_hex,
            "post_reset_value": write_hex,
        }
        for field, expected in numeric_record.items():
            if _float32_hex(record[field], f"record {name}.{field}") != expected:
                raise FreezeBWorkerError(f"intervention record {name} {field} drifted")


def _validate_simulation_configuration_v2(
    provenance: Mapping[str, object], *, dt_s: float
) -> None:
    config = _exact_object(
        provenance.get("simulation_configuration_v2"),
        _SIMULATION_CONFIGURATION_V2_KEYS,
        "simulation_configuration_v2",
    )
    literals = {
        "schema_version": 2,
        "dt_claim_scope": "python_configuration_only_not_compiled_runtime",
        "python_configuration_dt_exact_match": True,
        "runtime_effective_dt_s": None,
        "runtime_effective_dt_status": "not_exposed_by_pinned_kitless_ovphysx",
        "runtime_effective_dt_verified": False,
        "physics_prim_path": "/physicsScene",
    }
    for field, expected in literals.items():
        if type(config[field]) is not type(expected) or config[field] != expected:
            raise FreezeBWorkerError(f"simulation_configuration_v2.{field} drifted")
    for field in (
        "requested_dt_s",
        "simulation_cfg_dt_s",
        "simulation_context_config_accessor_dt_s",
    ):
        if not math.isclose(
            _finite(config[field], f"simulation_configuration_v2.{field}", positive=True),
            dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FreezeBWorkerError(f"simulation_configuration_v2.{field} differs from case")
    for field in (
        "requested_gravity_m_s2",
        "simulation_cfg_gravity_m_s2",
        "physics_scene_gravity_m_s2",
    ):
        gravity = config[field]
        if (
            not isinstance(gravity, list)
            or len(gravity) != 3
            or any(abs(_finite(item, f"{field} item")) > 1e-12 for item in gravity)
        ):
            raise FreezeBWorkerError(f"simulation_configuration_v2.{field} is not zero")


def _confined(
    path: Path,
    root: Path,
    label: str,
    *,
    require_exists: bool = True,
) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    try:
        resolved = raw.resolve(strict=require_exists)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise FreezeBWorkerError(f"{label} escapes the approved root") from exc
    cursor = resolved if require_exists else resolved.parent
    while True:
        if cursor.exists() and is_link_like(cursor):
            raise FreezeBWorkerError(f"{label} has a symbolic-link ancestor")
        if cursor == root:
            break
        if root not in cursor.parents:
            raise FreezeBWorkerError(f"{label} escapes the approved root")
        cursor = cursor.parent
    return resolved


def _marker(path: Path, regex: re.Pattern[str], label: str) -> str:
    if not path.is_file() or is_link_like(path):
        raise FreezeBWorkerError(f"missing or linked {label} marker")
    value = path.read_text(encoding="ascii").strip()
    if regex.fullmatch(value) is None:
        raise FreezeBWorkerError(f"invalid {label} marker")
    return value


def _source_snapshot(root: Path, source_revision: str) -> tuple[Path, str, str]:
    project_root = Path(__file__).resolve(strict=True).parents[1]
    _confined(project_root, root, "project snapshot")
    source_tree = _marker(project_root / ".source-tree", _HEX40_RE, "source tree")
    if project_root != (root / "project" / source_tree).resolve(strict=True):
        raise FreezeBWorkerError("worker is not running from its immutable tree snapshot")
    if _marker(project_root / ".source-revision", _HEX40_RE, "source revision") != source_revision:
        raise FreezeBWorkerError("source revision marker differs from worker input")
    _marker(project_root / ".archive-sha256", _SHA256_RE, "source archive SHA-256")
    snapshot_sha256 = _marker(
        project_root / ".snapshot-sha256", _SHA256_RE, "source snapshot SHA-256"
    )
    return project_root, source_tree, snapshot_sha256


def _gpu_identity(index: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
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
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if completed.returncode != 0 or len(rows) != 1:
        raise FreezeBWorkerError("cannot establish selected GPU identity")
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if len(fields) != 3 or any(not field for field in fields):
        raise FreezeBWorkerError("selected GPU identity is malformed")
    return {"gpu_uuid": fields[0], "gpu_name": fields[1], "driver_version": fields[2]}


def _fresh_process_id_from_os_identity(identity: Mapping[str, object]) -> str:
    expected_keys = {
        "boot_id",
        "hostname",
        "pid",
        "process_start_ticks",
    }
    if set(identity) != expected_keys:
        raise FreezeBWorkerError("Linux process identity fields are not exact")
    boot_id = identity["boot_id"]
    hostname = identity["hostname"]
    pid = identity["pid"]
    start_ticks = identity["process_start_ticks"]
    if (
        not isinstance(boot_id, str)
        or not boot_id
        or not isinstance(hostname, str)
        or not hostname
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks < 0
    ):
        raise FreezeBWorkerError("Linux process identity is malformed")
    return sha256(
        json.dumps(
            dict(identity),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _os_process_identity() -> dict[str, object]:
    """Read the worker's auditable Linux identity without case/session data."""

    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        stat_text = Path("/proc/self/stat").read_text(encoding="ascii")
        _prefix, separator, suffix = stat_text.rpartition(")")
        fields_after_comm = suffix.strip().split()
        if not separator or len(fields_after_comm) <= 19:
            raise ValueError("malformed /proc/self/stat")
        start_ticks = int(fields_after_comm[19])
    except (OSError, UnicodeError, ValueError) as exc:
        raise FreezeBWorkerError("cannot establish fresh Linux process identity") from exc
    identity: dict[str, object] = {
        "boot_id": boot_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_start_ticks": start_ticks,
    }
    _fresh_process_id_from_os_identity(identity)
    return identity


def _fresh_process_id() -> str:
    return _fresh_process_id_from_os_identity(_os_process_identity())


def _select_canonical_case(manifest: object, case_id: str) -> object:
    matches = [item for item in expand_scenario_cases(manifest) if item.case_id == case_id]
    if len(matches) != 1 or matches[0].simulator is not Simulator.OVPHYSX:
        raise FreezeBWorkerError("canonical case is not one manifest OVPhysX case")
    case = matches[0]
    if case.scenario_id != "small_step":
        raise FreezeBWorkerError("Freeze B worker accepts only small_step")
    return case


def _run(args: argparse.Namespace) -> AdapterRunResult:
    root = args.approved_root.expanduser().resolve(strict=True)
    if root != EXPECTED_APPROVED_ROOT.resolve(strict=True):
        raise FreezeBWorkerError("unexpected approved root")
    if not root.is_dir() or is_link_like(root):
        raise FreezeBWorkerError("approved root must be a real directory")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise FreezeBWorkerError("display variables must be unset")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"[0-7]", visible) is None:
        raise FreezeBWorkerError("CUDA_VISIBLE_DEVICES must expose exactly one Server 6 GPU")
    if _SESSION_RE.fullmatch(args.session_id) is None:
        raise FreezeBWorkerError("invalid session id")
    if _HEX40_RE.fullmatch(args.source_revision) is None:
        raise FreezeBWorkerError("invalid source revision")
    if _SHA256_RE.fullmatch(args.asset_tree_sha256) is None:
        raise FreezeBWorkerError("invalid asset-tree SHA-256")
    if _SHA256_RE.fullmatch(args.private_plan_sha256) is None:
        raise FreezeBWorkerError("invalid private-plan SHA-256")
    if _SHA256_RE.fullmatch(args.public_protocol_file_sha256) is None:
        raise FreezeBWorkerError("invalid public-protocol file SHA-256")
    if _SHA256_RE.fullmatch(args.public_protocol_canonical_sha256) is None:
        raise FreezeBWorkerError("invalid public-protocol canonical SHA-256")

    project_root, source_tree, snapshot_sha256 = _source_snapshot(
        root, args.source_revision
    )
    asset_root = _confined(args.asset_root, root, "asset root")
    manifest_path = _confined(args.manifest, project_root, "manifest")
    expected_manifest = project_root / "configs" / "parity" / "gate0.json"
    if manifest_path != expected_manifest.resolve(strict=True):
        raise FreezeBWorkerError("manifest is not the snapshot's canonical Gate 0 config")
    public_protocol_path = _confined(
        args.public_protocol, project_root, "public protocol"
    )
    expected_public_protocol = (
        project_root
        / "configs"
        / "parity"
        / "ovphysx_legacy_friction_freeze_b.json"
    )
    if public_protocol_path != expected_public_protocol.resolve(strict=True):
        raise FreezeBWorkerError("public protocol is not the canonical Freeze B config")
    if sha256_file(public_protocol_path) != args.public_protocol_file_sha256:
        raise FreezeBWorkerError("public protocol file SHA-256 mismatch")
    public_protocol = load_public_protocol(public_protocol_path)
    if canonical_json_sha256(public_protocol) != args.public_protocol_canonical_sha256:
        raise FreezeBWorkerError("public protocol canonical SHA-256 mismatch")
    private_plan_path = _confined(args.private_plan, root, "private plan")
    if not private_plan_path.is_file() or is_link_like(private_plan_path):
        raise FreezeBWorkerError("private plan must be a real file")
    expected_private_plan = (
        root
        / "inputs"
        / "freeze-b"
        / f"{args.private_plan_sha256}.json"
    ).resolve(strict=True)
    if private_plan_path != expected_private_plan:
        raise FreezeBWorkerError("private plan path is not the hash-addressed input path")
    private_stat = private_plan_path.stat()
    try:
        import pwd

        private_owner = pwd.getpwuid(private_stat.st_uid).pw_name
    except (ImportError, KeyError) as exc:
        raise FreezeBWorkerError("cannot establish private plan owner") from exc
    if (
        private_owner != "exampleuser"
        or private_stat.st_uid != root.stat().st_uid
        or private_stat.st_mode & 0o077
    ):
        raise FreezeBWorkerError(
            "private plan owner or group/other permission boundary is invalid"
        )
    results_root = (root / "results").resolve(strict=True)
    output = _confined(
        args.output, results_root, "worker output", require_exists=False
    )
    if output.exists() or is_link_like(output):
        raise FileExistsError(f"worker refuses to overwrite output: {output.name}")
    if not output.parent.is_dir() or is_link_like(output.parent):
        raise FreezeBWorkerError("worker output parent must already be a real directory")

    manifest = load_manifest(manifest_path)
    if args.asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise FreezeBWorkerError("asset-tree argument differs from Gate 0 manifest")
    asset_subtree = _confined(
        asset_root / manifest.provenance.asset_root,
        asset_root,
        "canonical asset subtree",
    )
    if asset_tree_sha256(asset_subtree) != args.asset_tree_sha256:
        raise FreezeBWorkerError("materialized canonical asset tree hash mismatch")

    plan = load_private_plan(
        private_plan_path,
        public_protocol=public_protocol,
        expected_sha256=args.private_plan_sha256,
    )
    canonical_plan_sha256 = private_plan_sha256(plan)
    if canonical_plan_sha256 != args.private_plan_sha256:
        raise FreezeBWorkerError("private plan canonical SHA-256 mismatch")
    if plan["protocol_id"] != "ovphysx-legacy-joint-friction-freeze-b-v1":
        raise FreezeBWorkerError("private plan protocol_id drifted")
    _validate_plan_inputs(
        plan["inputs"],
        manifest=manifest,
        manifest_path=manifest_path,
        source_revision=args.source_revision,
        source_tree=source_tree,
        asset_tree_sha256_value=args.asset_tree_sha256,
    )
    ov_cases = _validated_plan_matrix(plan)
    selected = [item for item in ov_cases if item["experiment_case_id"] == args.case_id]
    if len(selected) != 1:
        raise FreezeBWorkerError("private plan does not contain exactly one requested case")
    plan_case = selected[0]
    resolved_experiment = experiment_case(plan, args.case_id)
    if resolved_experiment.get("case") != plan_case:
        raise FreezeBWorkerError("private case resolution is not stable")
    canonical_case = _select_canonical_case(
        manifest, str(plan_case["canonical_case_id"])
    )
    if (
        canonical_case.hand.value != plan_case["hand"]
        or canonical_case.timestep_variant.value != plan_case["timestep_variant"]
        or canonical_case.repeat_index != plan_case["repeat_index"]
        or not math.isclose(
            canonical_case.dt_s,
            float(plan_case["dt_s"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise FreezeBWorkerError("private case differs from its canonical manifest case")
    hand = manifest.hand(canonical_case.hand)
    raw_hand_plans = plan["hand_plans"]
    validated_hand_plans = _validated_hand_plan_map(
        raw_hand_plans, manifest=manifest
    )
    matching_hand_plans = [
        item
        for item in raw_hand_plans
        if isinstance(item, dict) and item.get("hand") == canonical_case.hand.value
    ]
    if len(matching_hand_plans) != 1:
        raise FreezeBWorkerError("private plan must contain one plan per Wave hand")
    hand_plan = validated_hand_plans[canonical_case.hand.value]
    role = str(plan_case["role"])
    intervention = {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": args.private_plan_sha256,
        "expected_pre_values": hand_plan["expected_pre_values"],
        "write_values": hand_plan[f"{role}_write_values"],
    }

    scenario = manifest.scenario("small_step")
    gpu = _gpu_identity(visible)
    os_process_identity = _os_process_identity()
    fresh_id = _fresh_process_id_from_os_identity(os_process_identity)
    adapter = OVPhysXAdapter(
        device=args.device,
        asset_root=asset_root,
        legacy_joint_friction_intervention=intervention,
    )
    result = adapter.run_scenario(
        hand, scenario, dt_override=float(plan_case["dt_s"])
    )
    provenance = dict(result.provenance)
    if result.completed:
        _validate_intervention_evidence(
            provenance,
            role=role,
            private_plan_sha256=args.private_plan_sha256,
            hand_plan=hand_plan,
        )
        _validate_simulation_configuration_v2(
            provenance, dt_s=float(plan_case["dt_s"])
        )
    adapter_source = provenance.pop("source_path", None)
    if not isinstance(adapter_source, str) or (
        _confined(Path(adapter_source), asset_root, "adapter source path")
        != (asset_root / canonical_case.model_path).resolve(strict=True)
    ):
        raise FreezeBWorkerError("adapter source path differs from the canonical case")
    adapter_owned_literals = {
        "device": args.device,
        "python_version": platform.python_version(),
        "source_revision": args.source_revision,
        "worker_pid": os.getpid(),
    }
    for key, expected_value in adapter_owned_literals.items():
        if provenance.get(key) != expected_value:
            raise FreezeBWorkerError(f"adapter provenance drifted before enrichment: {key}")
    if provenance.get("forbidden_modules") != []:
        raise FreezeBWorkerError(
            "forbidden Kit, renderer, or Isaac Sim modules were loaded during execution"
        )
    worker_provenance = {
        "source_path": canonical_case.model_path,
        "freeze_b_schema_version": 1,
        "freeze_b_protocol_id": plan["protocol_id"],
        "private_plan_sha256": args.private_plan_sha256,
        "private_plan_file_sha256": sha256_file(private_plan_path),
        "public_protocol_file_sha256": args.public_protocol_file_sha256,
        "public_protocol_canonical_sha256": (
            args.public_protocol_canonical_sha256
        ),
        "experiment_case_id": args.case_id,
        "canonical_case_id": canonical_case.case_id,
        "freeze_b_role": role,
        "hand": canonical_case.hand.value,
        "timestep_variant": canonical_case.timestep_variant.value,
        "repeat_index": canonical_case.repeat_index,
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_file_sha256": sha256_file(manifest_path),
        "session_id": args.session_id,
        "source_tree": source_tree,
        "deployment_source_revision": args.source_revision,
        "deployment_source_tree": source_tree,
        "implementation_source_revision": plan["inputs"][
            "freeze_b_source_revision"
        ],
        "implementation_source_tree": plan["inputs"]["freeze_b_source_tree"],
        "implementation_to_deployment_relation_scope": (
            "locally_audited_before_gitless_remote_deployment_not_recomputed_here"
        ),
        "source_snapshot_sha256": snapshot_sha256,
        "asset_tree_sha256": args.asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "worker_script_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "os_process_identity": os_process_identity,
        "fresh_process_id": fresh_id,
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "driver_version": gpu["driver_version"],
        "constraints": {
            "kitless": True,
            "headless": True,
            "renderer": False,
            "camera": False,
            "fresh_process": True,
            "scenario_id": "small_step",
        },
    }
    duplicate_provenance_keys = sorted(set(provenance) & set(worker_provenance))
    if duplicate_provenance_keys:
        raise FreezeBWorkerError(
            "worker provenance enrichment would overwrite adapter keys: "
            + ", ".join(duplicate_provenance_keys)
        )
    provenance.update(worker_provenance)
    strict = adapter_run_result_from_dict(replace(result, provenance=provenance).to_dict())
    if strict.backend != "ovphysx" or strict.scenario_id != "small_step":
        raise FreezeBWorkerError("worker result identity is invalid")
    if not math.isclose(
        strict.dt, float(plan_case["dt_s"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise FreezeBWorkerError("worker result timestep differs from private case")
    write_json_atomic(output, strict.to_dict())
    if sha256_file(output) != sha256(output.read_bytes()).hexdigest():
        raise FreezeBWorkerError("worker output hash verification failed")
    return strict


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
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
