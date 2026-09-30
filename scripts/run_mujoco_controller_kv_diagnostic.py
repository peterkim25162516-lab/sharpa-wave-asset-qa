#!/usr/bin/env python3
"""Align total viscous damping via controller-Kv under zero dry friction."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Mapping, Sequence

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.parity.bundle import (
    sha256_file,
    write_bundle_manifest,
    write_json_atomic,
)
from wave_asset_qa.parity.compare import CollectedRun, trace_delta
from wave_asset_qa.parity.contracts import (
    HandSide,
    HandSpec,
    ParityManifest,
    Simulator,
)
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    bundle_payload_paths,
    create_staging_root,
    promote_staging_root,
    validate_git_source,
    validated_results_output_path,
    verify_copied_bundle_subset,
    verify_exact_bundle,
    write_text_exclusive,
)
from wave_asset_qa.parity.runner import load_collected_runs, run_backend_cases
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    canonical_initial_positions,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPERIMENT_ID = "mujoco-total-viscous-alignment-via-controller-kv-v1"
AUTHORITATIVE_FRICTION_ROOT_SHA256 = (
    "0804626ef0b7009a1e6c223284c210550e568593852046bc9b2b31d5bf3e4803"
)
AUTHORITATIVE_FRICTION_MANIFEST_SHA256 = (
    "6950a2cc14e4c1b7951c9d332e57ca748af05b6d2b08300bdaed857635e88f06"
)
AUTHORITATIVE_FRICTION_SUMMARY_SHA256 = (
    "545c2515be009822618f272f3fdd1d8500027c71f1a178e4c36f0f9d785fca45"
)
AUTHORITATIVE_ZERO_RUN_SHA256 = (
    "b9e547cee1bced18aae93d0b21e6418e059a2f49f45d9cdbb8118c083fbaf758"
)
AUTHORITATIVE_OV_RUN_SHA256 = (
    "008ed804406ee32e58a5b701bc9feb9f4123a23bc13d521392a2525d0de2dc30"
)
AUTHORITATIVE_FORMAL_ROOT_SHA256 = (
    "d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505"
)
AUTHORITATIVE_FORMAL_MANIFEST_SHA256 = (
    "d3b64ff8c2bea108a724e62f32994127b8dabb174701deb61a73611f671eb690"
)
AUTHORITATIVE_FORMAL_FINALIZATION_SHA256 = (
    "2a393c39777b01d79d56ba1dd3f2fea789f347e3412042de7ddd7b990bed28af"
)
AUTHORITATIVE_FORMAL_SOURCE_REVISION = (
    "c571d9bb37a619fe9e867447d3be448acc615e3c"
)
AUTHORITATIVE_FORMAL_SOURCE_TREE = (
    "fe86e1ce279770db1a70ce74609166b282f151ff"
)
AUTHORITATIVE_MANIFEST_FILE_SHA256 = (
    "a95a1da0d38abac8b3eccf18ea0b618b4898e35a6102017a2f0e024a9a4d5244"
)
AUTHORITATIVE_MANIFEST_SHA256 = (
    "576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e"
)
AUTHORITATIVE_ASSET_TREE_SHA256 = (
    "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
)
AUTHORITATIVE_MODEL_SHA256 = (
    "3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11"
)
AUTHORITATIVE_FRICTION_SOURCE_REVISION = (
    "e238b29934f2e341703ba1aa7decdab7cae53688"
)
AUTHORITATIVE_ASSET_COMMIT = "6eea427eb24189519f32b9f21674cd534d3f973c"
AUTHORITATIVE_ASSET_GIT_TREE = "bb00a9d5527b8a76de576ce876ebece67d8ffde1"
AUTHORITATIVE_OV_CANONICAL_KD_SHA256 = (
    "f97922838c9ed5cf392a7035b32e244860de7441bcf6f321e88b6bdb4a6e4d75"
)
MUJOCO_CASE_ID = "mujoco.left.small_step.base.r01"
OVPHYSX_CASE_ID = "ovphysx.left.small_step.base.r01"
PREREG_RELATIVE_PATH = "configs/parity/controller_kv_diagnostic.json"
AUTHORITATIVE_PREREGISTRATION_SHA256 = (
    "5acff5d5ce9b34a133032830e62cc831f5c583087c5e18e244367edd8464d5ba"
)
CONTROL_ROLE = "same_source_control"
TREATMENT_ROLE = "total_viscous_aligned_via_controller_kv_treatment"
CLAIM_BOUNDARY = (
    "This two-run endpoint diagnostic can only test the local effect of adjusting "
    "MuJoCo controller Kv so controller Kv plus MuJoCo passive joint damping equals "
    "frozen OV explicit-PD Kd under friction=0 for one left small_step trajectory. "
    "It does not align passive PhysX damping, establish physical parameter equivalence, "
    "fully explain the formal divergence, identify a backend bug, change Gate 0 pass "
    "readiness, or generalize to other hands, scenarios, hardware, or Sim2Real."
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ControllerKvDiagnosticError(DiagnosticEvidenceError):
    """Raised when the controller-Kv experiment violates its contract."""


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerKvDiagnosticError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerKvDiagnosticError(f"{label} must be a JSON object")
    return value


def _load_one(path: Path, label: str) -> CollectedRun:
    runs = load_collected_runs(path)
    if len(runs) != 1:
        raise ControllerKvDiagnosticError(f"{label} must contain exactly one run")
    return runs[0]


def _number(value: object, context: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerKvDiagnosticError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ControllerKvDiagnosticError(
            f"{context} must be a finite number"
        ) from exc
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ControllerKvDiagnosticError(f"{context} must be a finite number")
    return number


def _number_list(
    value: object,
    context: str,
    *,
    length: int | None = None,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = f" with length {length}" if length is not None else ""
        raise ControllerKvDiagnosticError(f"{context} must be an array{suffix}")
    return tuple(
        _number(item, f"{context}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(value)
    )


def _compact_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_preregistration(
    path: Path,
    manifest: ParityManifest,
) -> tuple[dict[str, object], dict[str, float]]:
    prereg = _load_json_object(path, "controller-Kv preregistration")
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "classification",
        "scope",
        "frozen_inputs",
        "cases",
        "runs",
        "intervention",
        "baseline_drift",
        "primary_metrics",
        "secondary_metrics",
        "decision_rule",
        "claim_boundary",
    }
    if set(prereg) != expected_top_level:
        raise ControllerKvDiagnosticError("preregistration top-level schema changed")
    if prereg.get("schema_version") != 1 or prereg.get("experiment_id") != EXPERIMENT_ID:
        raise ControllerKvDiagnosticError("unexpected controller-Kv preregistration")
    frozen = prereg.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise ControllerKvDiagnosticError("preregistration lacks frozen_inputs")
    expected_frozen = {
        "formal_bundle_root_sha256": AUTHORITATIVE_FORMAL_ROOT_SHA256,
        "formal_bundle_manifest_sha256": AUTHORITATIVE_FORMAL_MANIFEST_SHA256,
        "formal_finalization_sha256": AUTHORITATIVE_FORMAL_FINALIZATION_SHA256,
        "friction_bundle_root_sha256": AUTHORITATIVE_FRICTION_ROOT_SHA256,
        "friction_bundle_manifest_sha256": AUTHORITATIVE_FRICTION_MANIFEST_SHA256,
        "friction_scale_0_run_sha256": AUTHORITATIVE_ZERO_RUN_SHA256,
        "friction_summary_sha256": AUTHORITATIVE_FRICTION_SUMMARY_SHA256,
        "friction_source_revision": AUTHORITATIVE_FRICTION_SOURCE_REVISION,
        "formal_ovphysx_run_sha256": AUTHORITATIVE_OV_RUN_SHA256,
        "formal_source_revision": AUTHORITATIVE_FORMAL_SOURCE_REVISION,
        "formal_source_tree": AUTHORITATIVE_FORMAL_SOURCE_TREE,
        "gate0_manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
        "gate0_manifest_file_sha256": AUTHORITATIVE_MANIFEST_FILE_SHA256,
        "canonical_lf_asset_tree_sha256": AUTHORITATIVE_ASSET_TREE_SHA256,
        "mujoco_model_sha256": AUTHORITATIVE_MODEL_SHA256,
        "asset_commit": AUTHORITATIVE_ASSET_COMMIT,
        "asset_git_tree": AUTHORITATIVE_ASSET_GIT_TREE,
    }
    if dict(frozen) != expected_frozen:
        raise ControllerKvDiagnosticError(
            "preregistration frozen_inputs are not exactly authoritative"
        )
    if (
        prereg.get("classification") != "diagnostic_only_not_formal_gate0"
        or prereg.get("scope")
        != "left small_step base r01 under zero MuJoCo dry friction"
    ):
        raise ControllerKvDiagnosticError("preregistration scope changed")
    cases = prereg.get("cases")
    if cases != {"mujoco": MUJOCO_CASE_ID, "ovphysx": OVPHYSX_CASE_ID}:
        raise ControllerKvDiagnosticError("preregistration has unexpected cases")
    intervention = prereg.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ControllerKvDiagnosticError("preregistration lacks intervention")
    expected_intervention_keys = {
        "name",
        "ov_target_source",
        "ov_target_provenance_field",
        "mujoco_effective_controller_kv_source",
        "mujoco_total_viscous_damping_definition",
        "target_unit",
        "unchanged",
        "targets_in_canonical_joint_order",
    }
    expected_unchanged = [
        "canonical asset files",
        "dof_frictionloss_scale=0",
        "position targets",
        "initial state",
        "dt",
        "controller Kp",
        "joint passive damping",
        "armature",
        "gear and transmission",
        "force limits",
        "solver and gravity",
    ]
    if (
        set(intervention) != expected_intervention_keys
        or intervention.get("name")
        != "per_joint_total_viscous_alignment_via_controller_kv"
        or intervention.get("ov_target_source")
        != "IdealPDActuator.controller_dof_damping"
        or intervention.get("ov_target_provenance_field")
        != "controller_dof_damping"
        or intervention.get("mujoco_effective_controller_kv_source")
        != "-model.actuator_biasprm[actuator_id, 2]"
        or intervention.get("mujoco_total_viscous_damping_definition")
        != "effective_controller_kv + model.dof_damping[dof_index]"
        or intervention.get("target_unit") != "N*m*s/rad"
        or intervention.get("unchanged") != expected_unchanged
    ):
        raise ControllerKvDiagnosticError("preregistration target semantics changed")
    targets = intervention.get("targets_in_canonical_joint_order")
    hand = manifest.hand(HandSide.LEFT)
    if not isinstance(targets, list) or len(targets) != len(hand.joint_names):
        raise ControllerKvDiagnosticError("preregistration must contain 22 targets")
    target_map: dict[str, float] = {}
    observed_names: list[str] = []
    for index, record in enumerate(targets):
        if not isinstance(record, Mapping) or set(record) != {
            "canonical_id",
            "target_total_viscous_damping_nm_s_per_rad",
        }:
            raise ControllerKvDiagnosticError(
                f"invalid preregistration target record {index}"
            )
        name = record["canonical_id"]
        if not isinstance(name, str):
            raise ControllerKvDiagnosticError("target canonical_id must be text")
        observed_names.append(name)
        target_map[name] = _number(
            record["target_total_viscous_damping_nm_s_per_rad"],
            f"target {name}",
            nonnegative=True,
        )
    if tuple(observed_names) != hand.joint_names or len(target_map) != len(targets):
        raise ControllerKvDiagnosticError(
            "preregistered targets are not in canonical joint order"
        )
    if _compact_sha256([target_map[name] for name in hand.joint_names]) != (
        AUTHORITATIVE_OV_CANONICAL_KD_SHA256
    ):
        raise ControllerKvDiagnosticError("preregistered target vector hash changed")
    expected_runs = [
        {
            "role": CONTROL_ROLE,
            "dof_frictionloss_scale": 0.0,
            "controller_kv_mode": "canonical_no_override",
        },
        {
            "role": TREATMENT_ROLE,
            "dof_frictionloss_scale": 0.0,
            "controller_kv_mode": "diagnostic_total_damping_override",
        },
    ]
    if prereg.get("runs") != expected_runs:
        raise ControllerKvDiagnosticError("preregistered run intervention changed")
    metrics = prereg.get("primary_metrics")
    expected_metrics = [
        {
            "metric_id": "left_pinky_CMC_window_peak",
            "kind": "joint_max_abs_rad",
            "canonical_id": "left_pinky_CMC",
            "window_start_s": 0.11,
            "window_end_s": 0.16,
            "frozen_baseline_value": 0.00908693500546992,
            "frozen_baseline_peak_time_s": 0.13,
        },
        {
            "metric_id": "left_thumb_DP_window_peak",
            "kind": "frame_position_max_m",
            "canonical_id": "left_thumb_DP",
            "window_start_s": 0.11,
            "window_end_s": 0.17,
            "frozen_baseline_value": 0.0008506666265246331,
            "frozen_baseline_peak_time_s": 0.132,
        },
    ]
    if metrics != expected_metrics:
        raise ControllerKvDiagnosticError("preregistered primary metrics changed")
    expected_secondary_metrics = [
        {
            "metric_id": "global_joint_max_abs_rad",
            "definition": (
                "maximum absolute canonical joint-position difference over all 251 "
                "samples and 22 joints"
            ),
        },
        {
            "metric_id": "global_frame_position_max_m",
            "definition": (
                "maximum Euclidean distal-frame XYZ difference over all 251 samples "
                "and 5 frames"
            ),
        },
        {
            "metric_id": "global_frame_orientation_max_rad",
            "definition": (
                "maximum shortest-arc normalized-quaternion geodesic angle over all "
                "251 samples and 5 frames"
            ),
        },
        {
            "metric_id": "joint_rmse_rad",
            "definition": (
                "sqrt(sum of squared canonical joint-position differences divided by "
                "251*22)"
            ),
        },
        {
            "metric_id": "frame_position_rms_m",
            "definition": (
                "sqrt(sum of squared distal-frame XYZ Euclidean distances divided by "
                "251*5)"
            ),
        },
    ]
    if prereg.get("secondary_metrics") != expected_secondary_metrics:
        raise ControllerKvDiagnosticError("preregistered secondary metrics changed")
    decision = prereg.get("decision_rule")
    expected_decision = {
        "supports_registered_local_gap_reduction_under_total_viscous_retuning_if_each_reduction_at_least": 0.25,
        "rejects_total_viscous_alignment_as_common_major_local_explanation_if_each_reduction_at_most": 0.1,
        "otherwise": "mixed_or_inconclusive",
        "global_joint_frame_orientation_and_rms_metrics": "secondary_exploratory",
    }
    if decision != expected_decision:
        raise ControllerKvDiagnosticError("preregistered decision thresholds changed")
    if prereg.get("baseline_drift") != {
        "reference": "frozen friction scale=0 MuJoCo run",
        "maximum_joint_rad": 1e-9,
        "maximum_frame_position_m": 1e-9,
        "maximum_frame_orientation_rad": 1e-9,
        "require_exact_samples": True,
    }:
        raise ControllerKvDiagnosticError("preregistered baseline-drift rule changed")
    if prereg.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ControllerKvDiagnosticError("preregistered claim boundary changed")
    return prereg, target_map


def _extract_ov_controller_kd(
    ov_run: CollectedRun,
    hand: HandSpec,
) -> tuple[dict[str, float], dict[str, object]]:
    provenance = ov_run.result.provenance
    expected = {
        "actuation_contract_version": 2,
        "actuator_model": "IdealPDActuator",
        "control_path": "explicit_pd_effort",
        "controller_parameter_source": "ideal_pd_actuator_tensor",
        "backend_dof_drive_zero_verified": True,
        "zero_velocity_target_verified": True,
        "zero_feedforward_effort_target_verified": True,
        "source_revision": AUTHORITATIVE_FORMAL_SOURCE_REVISION,
        "asset_commit": AUTHORITATIVE_ASSET_COMMIT,
        "asset_git_tree": AUTHORITATIVE_ASSET_GIT_TREE,
        "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ControllerKvDiagnosticError(
                f"frozen OV provenance.{field} is not authoritative"
            )
    backend_names = provenance.get("backend_joint_names")
    mappings = provenance.get("joint_mapping")
    if (
        not isinstance(backend_names, list)
        or len(backend_names) != len(hand.joint_names)
        or len(set(backend_names)) != len(backend_names)
        or not isinstance(mappings, list)
        or len(mappings) != len(hand.joint_names)
    ):
        raise ControllerKvDiagnosticError("frozen OV joint inventory is invalid")
    raw_kd = _number_list(
        provenance.get("controller_dof_damping"),
        "frozen OV controller_dof_damping",
        length=len(backend_names),
        nonnegative=True,
    )
    for field in ("backend_dof_stiffness", "backend_dof_damping"):
        values = _number_list(
            provenance.get(field),
            f"frozen OV {field}",
            length=len(backend_names),
            nonnegative=True,
        )
        if any(value != 0.0 for value in values):
            raise ControllerKvDiagnosticError(f"frozen OV {field} is not zero")
    target_map: dict[str, float] = {}
    mapped_indices: set[int] = set()
    for position, raw_mapping in enumerate(mappings):
        if not isinstance(raw_mapping, Mapping):
            raise ControllerKvDiagnosticError(f"invalid frozen OV mapping {position}")
        canonical_id = raw_mapping.get("canonical_id")
        backend_name = raw_mapping.get("backend_name")
        index = raw_mapping.get("index")
        if (
            not isinstance(canonical_id, str)
            or not isinstance(backend_name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(backend_names)
            or backend_names[index] != backend_name
            or raw_mapping.get("backend") != Simulator.OVPHYSX.value
            or raw_mapping.get("sign") != 1.0
            or raw_mapping.get("offset") != 0.0
            or raw_mapping.get("unit") != "rad"
            or canonical_id in target_map
            or index in mapped_indices
        ):
            raise ControllerKvDiagnosticError(f"invalid frozen OV mapping {position}")
        target_map[canonical_id] = raw_kd[index]
        mapped_indices.add(index)
    if (
        set(target_map) != set(hand.joint_names)
        or mapped_indices != set(range(len(backend_names)))
    ):
        raise ControllerKvDiagnosticError("frozen OV mapping does not cover canonical joints")
    ordered = [target_map[name] for name in hand.joint_names]
    audit = {
        "source_field": "result.provenance.controller_dof_damping",
        "source_semantics": "IdealPDActuator explicit velocity-error gain",
        "derived_unit": "N*m*s/rad",
        "backend_order": list(backend_names),
        "backend_order_values": list(raw_kd),
        "backend_order_values_sha256": _compact_sha256(list(raw_kd)),
        "canonical_order": list(hand.joint_names),
        "canonical_order_values": ordered,
        "canonical_order_values_sha256": _compact_sha256(ordered),
        "mapping_method": "joint_mapping canonical_id/backend_name/index",
    }
    if audit["canonical_order_values_sha256"] != AUTHORITATIVE_OV_CANONICAL_KD_SHA256:
        raise ControllerKvDiagnosticError("frozen OV canonical Kd vector hash changed")
    return {name: target_map[name] for name in hand.joint_names}, audit


def _validate_trace_samples(
    result: AdapterRunResult,
    case: ScenarioCase,
    manifest: ParityManifest,
) -> None:
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    expected_steps = round(scenario.duration_s / case.dt_s)
    if (
        not result.completed
        or result.error is not None
        or result.backend != case.simulator.value
        or result.scenario_id != case.scenario_id
        or not math.isclose(result.dt, case.dt_s, rel_tol=0.0, abs_tol=1e-12)
        or result.requested_steps != expected_steps
        or result.completed_steps != expected_steps
        or len(result.samples) != expected_steps + 1
        or result.joint_names != hand.joint_names
        or result.frame_names != hand.distal_frame_names
    ):
        raise ControllerKvDiagnosticError(
            f"run is not a complete canonical trace: {case.case_id}"
        )
    backend_names = result.provenance.get("backend_joint_names")
    raw_mapping = result.provenance.get("joint_mapping")
    if (
        not isinstance(backend_names, list)
        or len(backend_names) != len(hand.joint_names)
        or len(set(backend_names)) != len(backend_names)
        or not isinstance(raw_mapping, list)
        or len(raw_mapping) != len(hand.joint_names)
    ):
        raise ControllerKvDiagnosticError("run has invalid joint mapping provenance")
    mapping_by_canonical: dict[str, tuple[int, float, float]] = {}
    mapped_indices: set[int] = set()
    observed_canonical_order: list[str] = []
    for position, raw_record in enumerate(raw_mapping):
        if not isinstance(raw_record, Mapping):
            raise ControllerKvDiagnosticError(
                f"run has invalid joint mapping record {position}"
            )
        canonical_id = raw_record.get("canonical_id")
        backend_name = raw_record.get("backend_name")
        index = raw_record.get("index")
        sign = raw_record.get("sign")
        offset = raw_record.get("offset")
        if (
            not isinstance(canonical_id, str)
            or canonical_id not in hand.joint_names
            or not isinstance(backend_name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(backend_names)
            or backend_names[index] != backend_name
            or raw_record.get("backend") != result.backend
            or raw_record.get("scope") != case.hand.value
            or raw_record.get("unit") != "rad"
            or isinstance(sign, bool)
            or not isinstance(sign, (int, float))
            or float(sign) not in (-1.0, 1.0)
            or isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or canonical_id in mapping_by_canonical
            or index in mapped_indices
        ):
            raise ControllerKvDiagnosticError(
                f"run has invalid joint mapping record {position}"
            )
        observed_canonical_order.append(canonical_id)
        mapped_indices.add(index)
        mapping_by_canonical[canonical_id] = (index, float(sign), float(offset))
    if (
        tuple(observed_canonical_order) != hand.joint_names
        or mapped_indices != set(range(len(hand.joint_names)))
    ):
        raise ControllerKvDiagnosticError("run joint mapping coverage or order differs")
    if case.simulator is Simulator.OVPHYSX:
        if (
            result.provenance.get("contact_check_performed") is not False
            or result.provenance.get("contact_observation_capability")
            != "not_evaluated"
        ):
            raise ControllerKvDiagnosticError(
                "frozen OV run has unexpected contact-observation provenance"
            )
        expected_contact_count: int | None = None
    else:
        expected_contact_count = 0
    initial = canonical_initial_positions(scenario, hand.joint_names)
    for step, sample in enumerate(result.samples):
        if sample.step != step or not math.isclose(
            sample.time_s,
            step * case.dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ControllerKvDiagnosticError("run has a non-canonical time axis")
        targets = canonical_position_targets(
            scenario,
            hand.joint_names,
            step_index=step,
            dt_s=case.dt_s,
        )
        if dict(sample.position_targets) != targets:
            raise ControllerKvDiagnosticError("run has non-canonical position targets")
        if (
            set(sample.joint_positions) != set(hand.joint_names)
            or set(sample.frame_poses) != set(hand.distal_frame_names)
            or len(sample.qpos) != len(hand.joint_names)
            or len(sample.qvel) != len(hand.joint_names)
            or any(len(pose) != 7 for pose in sample.frame_poses.values())
            or sample.contact_count != expected_contact_count
        ):
            raise ControllerKvDiagnosticError("run has invalid sampled state structure")
        for canonical_id, (index, sign, offset) in mapping_by_canonical.items():
            mapped_position = sign * float(sample.qpos[index]) + offset
            if mapped_position != float(sample.joint_positions[canonical_id]):
                raise ControllerKvDiagnosticError(
                    "run qpos does not reproduce canonical joint_positions through mapping"
                )
        values = (
            sample.time_s,
            *sample.qpos,
            *sample.qvel,
            *sample.joint_positions.values(),
            *(value for pose in sample.frame_poses.values() for value in pose),
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ControllerKvDiagnosticError("run contains a non-finite sample")
    first = result.samples[0]
    if (
        dict(first.joint_positions) != initial
        or any(value != 0.0 for value in first.qvel)
    ):
        raise ControllerKvDiagnosticError("run does not start from canonical state")


def _compiled_joints(result: AdapterRunResult) -> list[Mapping[str, object]]:
    compiled = result.provenance.get("compiled_control_parameters")
    if not isinstance(compiled, Mapping):
        raise ControllerKvDiagnosticError("run lacks compiled control provenance")
    joints = compiled.get("joints")
    if not isinstance(joints, list) or any(not isinstance(item, Mapping) for item in joints):
        raise ControllerKvDiagnosticError("compiled control joint records are invalid")
    return joints


def _validate_new_mujoco_run(
    run: CollectedRun,
    *,
    expected_case: ScenarioCase,
    manifest: ParityManifest,
    session_id: str,
    source_revision: str,
    source_sha256: str,
    target_map: Mapping[str, float] | None,
) -> None:
    if run.case != expected_case:
        raise ControllerKvDiagnosticError("new diagnostic returned the wrong case")
    _validate_trace_samples(run.result, expected_case, manifest)
    hand = manifest.hand(expected_case.hand)
    provenance = run.result.provenance
    expected_common = {
        "manifest_sha256": manifest_sha256(manifest),
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "backend": Simulator.MUJOCO.value,
        "device": "cpu",
        "headless": True,
        "renderer": False,
        "source_path": expected_case.model_path,
        "source_sha256": source_sha256,
        "frame_pose_source": "mujoco_data_xpos_xquat_after_mj_kinematics",
        "frame_pose_state_alignment": "current_sample_qpos",
        "dof_frictionloss_scale": 0.0,
        "dof_frictionloss_mode": "diagnostic_scaled_override",
        "dof_frictionloss_scope": "all_compiled_dofs",
        "dof_frictionloss_order": "model_dof_index_ascending",
        "joint_total_damping_mode": (
            "diagnostic_total_damping_override"
            if target_map is not None
            else "canonical_no_override"
        ),
        "joint_total_damping_scope": "all_compiled_canonical_joints",
        "joint_total_damping_order": "model_joint_index_ascending",
        "joint_total_damping_semantics": (
            "total=actuator_controller_kv+model_dof_damping"
        ),
    }
    for field, value in expected_common.items():
        if provenance.get(field) != value:
            raise ControllerKvDiagnosticError(f"new run provenance.{field} differs")
    if provenance.get("backend_joint_names") != list(hand.joint_names):
        raise ControllerKvDiagnosticError("new run backend joint order differs")
    original_friction = _number_list(
        provenance.get("dof_frictionloss_original"),
        "new run original friction",
        length=len(hand.joint_names),
        nonnegative=True,
    )
    effective_friction = _number_list(
        provenance.get("dof_frictionloss_effective"),
        "new run effective friction",
        length=len(hand.joint_names),
        nonnegative=True,
    )
    if not original_friction or any(value != 0.0 for value in effective_friction):
        raise ControllerKvDiagnosticError("new run friction intervention is invalid")
    if provenance.get("dof_frictionloss_dof_indices") != list(
        range(len(hand.joint_names))
    ):
        raise ControllerKvDiagnosticError("new run friction indices differ")

    target_names = provenance.get("joint_total_damping_target_joint_names")
    expected_target_names = list(hand.joint_names) if target_map is not None else []
    if target_names != expected_target_names:
        raise ControllerKvDiagnosticError("new run damping target coverage differs")
    records = provenance.get("joint_total_damping_records")
    if not isinstance(records, list) or len(records) != len(hand.joint_names):
        raise ControllerKvDiagnosticError("new run damping records are incomplete")
    compiled = _compiled_joints(run.result)
    if len(compiled) != len(hand.joint_names):
        raise ControllerKvDiagnosticError("new run compiled joints are incomplete")
    compiled_by_name = {
        item.get("canonical_id"): item for item in compiled if isinstance(item, Mapping)
    }
    for index, (joint_name, record) in enumerate(zip(hand.joint_names, records)):
        if not isinstance(record, Mapping) or record.get("canonical_id") != joint_name:
            raise ControllerKvDiagnosticError("new run damping record order differs")
        joint_damping = _number(
            record.get("joint_damping"),
            f"{joint_name} joint damping",
            nonnegative=True,
        )
        original_kv = _number(
            record.get("original_controller_damping_kv"),
            f"{joint_name} original Kv",
            nonnegative=True,
        )
        original_total = _number(
            record.get("original_total_damping"),
            f"{joint_name} original total damping",
            nonnegative=True,
        )
        requested_total = _number(
            record.get("requested_total_damping"),
            f"{joint_name} requested total damping",
            nonnegative=True,
        )
        effective_kv = _number(
            record.get("effective_controller_damping_kv"),
            f"{joint_name} effective Kv",
            nonnegative=True,
        )
        effective_total = _number(
            record.get("effective_total_damping"),
            f"{joint_name} effective total damping",
            nonnegative=True,
        )
        if (
            record.get("dof_index") != index
            or record.get("unit") != "N*m*s/rad"
            or not math.isclose(
                original_total, original_kv + joint_damping, rel_tol=0.0, abs_tol=1e-15
            )
            or not math.isclose(
                effective_total, effective_kv + joint_damping, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ControllerKvDiagnosticError(f"{joint_name} damping arithmetic differs")
        original_gain = _number_list(
            record.get("original_actuator_gainprm"),
            f"{joint_name} original gainprm",
        )
        effective_gain = _number_list(
            record.get("effective_actuator_gainprm"),
            f"{joint_name} effective gainprm",
        )
        original_bias = _number_list(
            record.get("original_actuator_biasprm"),
            f"{joint_name} original biasprm",
        )
        effective_bias = _number_list(
            record.get("effective_actuator_biasprm"),
            f"{joint_name} effective biasprm",
        )
        if (
            len(original_bias) < 3
            or len(original_bias) != len(effective_bias)
            or len(original_gain) != len(effective_gain)
            or not math.isclose(
                original_bias[2], -original_kv, rel_tol=0.0, abs_tol=1e-15
            )
            or not math.isclose(
                effective_bias[2], -effective_kv, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ControllerKvDiagnosticError(
                f"{joint_name} actuator parameter readback is inconsistent"
            )
        observed_gain_changes = [
            offset
            for offset, (before, after) in enumerate(zip(original_gain, effective_gain))
            if before != after
        ]
        observed_bias_changes = [
            offset
            for offset, (before, after) in enumerate(zip(original_bias, effective_bias))
            if before != after
        ]
        if (
            original_gain != effective_gain
            or record.get("changed_gainprm_indices") != observed_gain_changes
        ):
            raise ControllerKvDiagnosticError(f"{joint_name} actuator gainprm changed")
        if record.get("changed_biasprm_indices") != observed_bias_changes:
            raise ControllerKvDiagnosticError(
                f"{joint_name} actuator biasprm change audit differs"
            )
        if target_map is None:
            if (
                record.get("override_applied") is not False
                or requested_total != original_total
                or effective_total != original_total
                or effective_kv != original_kv
                or original_bias != effective_bias
                or record.get("changed_biasprm_indices") != []
            ):
                raise ControllerKvDiagnosticError(
                    f"control run unexpectedly changed {joint_name} damping"
                )
        else:
            target = target_map[joint_name]
            if (
                record.get("override_applied") is not True
                or not math.isclose(requested_total, target, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(effective_total, target, rel_tol=0.0, abs_tol=1e-12)
                or len(original_bias) != len(effective_bias)
                or record.get("changed_biasprm_indices") not in ([], [2])
                or any(
                    before != after
                    for bias_index, (before, after) in enumerate(
                        zip(original_bias, effective_bias)
                    )
                    if bias_index != 2
                )
            ):
                raise ControllerKvDiagnosticError(
                    f"treatment run changed more than {joint_name} controller Kv"
                )
        compiled_joint = compiled_by_name.get(joint_name)
        if (
            not isinstance(compiled_joint, Mapping)
            or compiled_joint.get("joint_id") != record.get("joint_id")
            or compiled_joint.get("dof_index") != record.get("dof_index")
            or compiled_joint.get("actuator_id") != record.get("actuator_id")
            or not math.isclose(
                _number(
                    compiled_joint.get("controller_damping_kv"),
                    f"{joint_name} compiled Kv",
                    nonnegative=True,
                ),
                effective_kv,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _number(
                    compiled_joint.get("joint_damping"),
                    f"{joint_name} compiled joint damping",
                    nonnegative=True,
                ),
                joint_damping,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or _number(
                compiled_joint.get("joint_frictionloss"),
                f"{joint_name} compiled frictionloss",
                nonnegative=True,
            )
            != 0.0
        ):
            raise ControllerKvDiagnosticError(f"{joint_name} compiled Kv readback differs")


def _command_signature(
    result: AdapterRunResult,
    hand: HandSpec,
) -> tuple[tuple[int, float, tuple[float, ...]], ...]:
    return tuple(
        (
            sample.step,
            sample.time_s,
            tuple(float(sample.position_targets[name]) for name in hand.joint_names),
        )
        for sample in result.samples
    )


def _initial_signature(
    result: AdapterRunResult,
    hand: HandSpec,
) -> tuple[object, ...]:
    sample = result.samples[0]
    return (
        tuple(sample.qpos),
        tuple(sample.qvel),
        tuple(float(sample.joint_positions[name]) for name in hand.joint_names),
        tuple(
            tuple(float(value) for value in sample.frame_poses[name])
            for name in hand.distal_frame_names
        ),
    )


def _normalized_non_kv_provenance(result: AdapterRunResult) -> dict[str, object]:
    value = json.loads(json.dumps(dict(result.provenance), allow_nan=False))
    for field in (
        "joint_total_damping_mode",
        "joint_total_damping_target_joint_names",
        "joint_total_damping_records",
    ):
        value.pop(field, None)
    compiled = value.get("compiled_control_parameters")
    if not isinstance(compiled, dict) or not isinstance(compiled.get("joints"), list):
        raise ControllerKvDiagnosticError("cannot normalize compiled provenance")
    for joint in compiled["joints"]:
        if not isinstance(joint, dict):
            raise ControllerKvDiagnosticError("cannot normalize compiled joint provenance")
        joint.pop("controller_damping_kv", None)
    return value


def _damping_source_signature(result: AdapterRunResult) -> tuple[object, ...]:
    """Return the fresh compiled damping state before any diagnostic override."""

    records = result.provenance.get("joint_total_damping_records")
    if not isinstance(records, list):
        raise ControllerKvDiagnosticError("run lacks damping audit records")
    signature: list[object] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ControllerKvDiagnosticError("run has an invalid damping audit record")
        original_bias = _number_list(
            record.get("original_actuator_biasprm"),
            "original actuator biasprm",
        )
        signature.append(
            (
                record.get("canonical_id"),
                record.get("joint_id"),
                record.get("dof_index"),
                record.get("actuator_id"),
                tuple(
                    _number_list(
                        record.get("original_actuator_gainprm"),
                        "original actuator gainprm",
                    )
                ),
                tuple(original_bias),
                _number(record.get("joint_damping"), "joint damping"),
                _number(
                    record.get("original_controller_damping_kv"),
                    "original controller Kv",
                ),
                _number(record.get("original_total_damping"), "original total damping"),
                record.get("unit"),
            )
        )
    return tuple(signature)


def _validate_experiment_pair(
    *,
    frozen_zero: CollectedRun,
    control: CollectedRun,
    treatment: CollectedRun,
    hand: HandSpec,
) -> tuple[float, float, float]:
    """Prove the two-run intervention and exact same-source control."""

    if control.result.samples != frozen_zero.result.samples:
        raise ControllerKvDiagnosticError(
            "same-source control samples drifted from frozen friction=0 baseline"
        )
    baseline_drift = trace_delta(frozen_zero.result, control.result, hand)
    if any(value > 1e-9 for value in baseline_drift):
        raise ControllerKvDiagnosticError("exact control matched but metric drift did not")
    if (
        _command_signature(control.result, hand)
        != _command_signature(treatment.result, hand)
        or _initial_signature(control.result, hand)
        != _initial_signature(treatment.result, hand)
    ):
        raise ControllerKvDiagnosticError(
            "control and treatment commands or initial state differ"
        )
    control_friction = control.result.provenance.get("dof_frictionloss_original")
    treatment_friction = treatment.result.provenance.get("dof_frictionloss_original")
    if control_friction != treatment_friction:
        raise ControllerKvDiagnosticError(
            "control and treatment original friction differ"
        )
    if _normalized_non_kv_provenance(
        control.result
    ) != _normalized_non_kv_provenance(treatment.result):
        raise ControllerKvDiagnosticError(
            "control and treatment provenance differ beyond controller Kv"
        )
    if _damping_source_signature(control.result) != _damping_source_signature(
        treatment.result
    ):
        raise ControllerKvDiagnosticError(
            "control and treatment did not start from the same compiled damping state"
        )
    return baseline_drift


def _sample_by_step(result: AdapterRunResult) -> dict[int, TraceSample]:
    samples = {sample.step: sample for sample in result.samples}
    if len(samples) != len(result.samples):
        raise ControllerKvDiagnosticError("trace contains duplicate steps")
    return samples


def _window_joint_peak(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    *,
    joint_name: str,
    start_step: int,
    end_step: int,
) -> dict[str, object]:
    ref = _sample_by_step(reference)
    cand = _sample_by_step(candidate)
    best: dict[str, object] | None = None
    best_delta = -1.0
    for step in range(start_step, end_step + 1):
        if step not in ref or step not in cand:
            raise ControllerKvDiagnosticError("joint window is missing a sample step")
        ref_value = float(ref[step].joint_positions[joint_name])
        candidate_value = float(cand[step].joint_positions[joint_name])
        delta = abs(ref_value - candidate_value)
        if delta > best_delta:
            best_delta = delta
            best = {
                "canonical_id": joint_name,
                "start_step": start_step,
                "end_step": end_step,
                "sample_count": end_step - start_step + 1,
                "peak_step": step,
                "peak_time_s": ref[step].time_s,
                "peak_abs_rad": delta,
                "reference_position_rad": ref_value,
                "candidate_position_rad": candidate_value,
                "signed_reference_minus_candidate_rad": ref_value - candidate_value,
            }
    assert best is not None
    return best


def _window_frame_peak(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    *,
    frame_name: str,
    start_step: int,
    end_step: int,
) -> dict[str, object]:
    ref = _sample_by_step(reference)
    cand = _sample_by_step(candidate)
    best: dict[str, object] | None = None
    best_delta = -1.0
    for step in range(start_step, end_step + 1):
        if step not in ref or step not in cand:
            raise ControllerKvDiagnosticError("frame window is missing a sample step")
        reference_xyz = tuple(float(value) for value in ref[step].frame_poses[frame_name][:3])
        candidate_xyz = tuple(float(value) for value in cand[step].frame_poses[frame_name][:3])
        vector = tuple(a - b for a, b in zip(reference_xyz, candidate_xyz))
        delta = math.sqrt(sum(value * value for value in vector))
        if delta > best_delta:
            best_delta = delta
            best = {
                "canonical_id": frame_name,
                "start_step": start_step,
                "end_step": end_step,
                "sample_count": end_step - start_step + 1,
                "peak_step": step,
                "peak_time_s": ref[step].time_s,
                "peak_position_distance_m": delta,
                "reference_xyz_m": list(reference_xyz),
                "candidate_xyz_m": list(candidate_xyz),
                "reference_minus_candidate_xyz_m": list(vector),
            }
    assert best is not None
    return best


def _trajectory_rms(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    hand: HandSpec,
) -> tuple[float, float]:
    ref = _sample_by_step(reference)
    cand = _sample_by_step(candidate)
    joint_sum = 0.0
    joint_count = 0
    frame_sum = 0.0
    frame_count = 0
    for step in sorted(ref):
        for name in hand.joint_names:
            delta = float(ref[step].joint_positions[name]) - float(
                cand[step].joint_positions[name]
            )
            joint_sum += delta * delta
            joint_count += 1
        for name in hand.distal_frame_names:
            pose_a = ref[step].frame_poses[name]
            pose_b = cand[step].frame_poses[name]
            distance_squared = sum(
                (float(a) - float(b)) ** 2 for a, b in zip(pose_a[:3], pose_b[:3])
            )
            frame_sum += distance_squared
            frame_count += 1
    return math.sqrt(joint_sum / joint_count), math.sqrt(frame_sum / frame_count)


def _analysis_row(
    role: str,
    ov: CollectedRun,
    candidate: CollectedRun,
    hand: HandSpec,
) -> dict[str, object]:
    joint, frame, orientation = trace_delta(ov.result, candidate.result, hand)
    joint_window = _window_joint_peak(
        ov.result,
        candidate.result,
        joint_name="left_pinky_CMC",
        start_step=55,
        end_step=80,
    )
    frame_window = _window_frame_peak(
        ov.result,
        candidate.result,
        frame_name="left_thumb_DP",
        start_step=55,
        end_step=85,
    )
    joint_rms, frame_rms = _trajectory_rms(ov.result, candidate.result, hand)
    return {
        "role": role,
        "case_id": candidate.case.case_id,
        "run_json_sha256": None,
        "primary": {
            "left_pinky_CMC_window": joint_window,
            "left_thumb_DP_window": frame_window,
        },
        "secondary": {
            "global_joint_max_abs_rad": joint,
            "global_frame_position_max_m": frame,
            "global_frame_orientation_max_rad": orientation,
            "joint_rmse_rad": joint_rms,
            "frame_position_rms_m": frame_rms,
        },
    }


def _audit_staged_runs(
    runs: Mapping[str, tuple[CollectedRun, Path]],
    *,
    frozen_zero: CollectedRun,
    frozen_ov: CollectedRun,
    mujoco_case: ScenarioCase,
    manifest: ParityManifest,
    hand: HandSpec,
    session_id: str,
    source_revision: str,
    treatment_targets: Mapping[str, float],
) -> tuple[
    tuple[float, float, float],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Reload run JSON bytes and recompute the complete diagnostic result."""

    expected_roles = {CONTROL_ROLE, TREATMENT_ROLE}
    if set(runs) != expected_roles:
        raise ControllerKvDiagnosticError("staged run roles are incomplete")
    reloaded: dict[str, CollectedRun] = {}
    for role in (CONTROL_ROLE, TREATMENT_ROLE):
        original, run_path = runs[role]
        observed = _load_one(run_path, f"staged {role} run")
        if observed != original:
            raise ControllerKvDiagnosticError(
                f"staged {role} run differs from the validated in-memory result"
            )
        targets = treatment_targets if role == TREATMENT_ROLE else None
        _validate_new_mujoco_run(
            observed,
            expected_case=mujoco_case,
            manifest=manifest,
            session_id=session_id,
            source_revision=source_revision,
            source_sha256=AUTHORITATIVE_MODEL_SHA256,
            target_map=targets,
        )
        reloaded[role] = observed

    control = reloaded[CONTROL_ROLE]
    treatment = reloaded[TREATMENT_ROLE]
    baseline_drift = _validate_experiment_pair(
        frozen_zero=frozen_zero,
        control=control,
        treatment=treatment,
        hand=hand,
    )
    control_row = _analysis_row(CONTROL_ROLE, frozen_ov, control, hand)
    treatment_row = _analysis_row(TREATMENT_ROLE, frozen_ov, treatment, hand)
    control_row["run_json_sha256"] = sha256_file(runs[CONTROL_ROLE][1])
    treatment_row["run_json_sha256"] = sha256_file(runs[TREATMENT_ROLE][1])
    decision = _evaluate_decision(
        control_row,
        treatment_row,
        baseline_drift_ok=True,
    )
    return baseline_drift, control_row, treatment_row, decision


def _evaluate_decision(
    control_row: Mapping[str, object],
    treatment_row: Mapping[str, object],
    *,
    baseline_drift_ok: bool,
) -> dict[str, object]:
    control_primary = control_row["primary"]
    treatment_primary = treatment_row["primary"]
    assert isinstance(control_primary, Mapping) and isinstance(treatment_primary, Mapping)
    control_joint = float(
        control_primary["left_pinky_CMC_window"]["peak_abs_rad"]  # type: ignore[index]
    )
    treatment_joint = float(
        treatment_primary["left_pinky_CMC_window"]["peak_abs_rad"]  # type: ignore[index]
    )
    control_frame = float(
        control_primary["left_thumb_DP_window"]["peak_position_distance_m"]  # type: ignore[index]
    )
    treatment_frame = float(
        treatment_primary["left_thumb_DP_window"]["peak_position_distance_m"]  # type: ignore[index]
    )
    if control_joint <= 0.0 or control_frame <= 0.0:
        raise ControllerKvDiagnosticError("primary control metric must be positive")
    if not baseline_drift_ok:
        raise ControllerKvDiagnosticError(
            "same-source control drift violates the preregistered exact-baseline rule"
        )
    joint_reduction = (control_joint - treatment_joint) / control_joint
    frame_reduction = (control_frame - treatment_frame) / control_frame
    if joint_reduction >= 0.25 and frame_reduction >= 0.25:
        status = (
            "supports_registered_local_gap_reduction_under_total_viscous_retuning"
        )
    elif joint_reduction <= 0.10 and frame_reduction <= 0.10:
        status = (
            "rejects_total_viscous_alignment_as_common_major_local_explanation"
        )
    else:
        status = "mixed_or_inconclusive"
    return {
        "scientific_status": status,
        "prediction_supported": status
        == "supports_registered_local_gap_reduction_under_total_viscous_retuning",
        "support_minimum_reduction_fraction": 0.25,
        "rejection_maximum_reduction_fraction": 0.10,
        "left_pinky_CMC_reduction_fraction": joint_reduction,
        "left_thumb_DP_reduction_fraction": frame_reduction,
        "baseline_drift_within_tolerance": baseline_drift_ok,
    }


def _render_report(summary: Mapping[str, object]) -> str:
    decision = summary["decision"]
    results = summary["results"]
    baseline = summary["baseline_drift"]
    assert isinstance(decision, Mapping)
    assert isinstance(results, list)
    assert isinstance(baseline, Mapping)
    lines = [
        "# MuJoCo total-viscous alignment via controller-Kv diagnostic",
        "",
        "> Local, simulation-only endpoint diagnostic under zero MuJoCo dry friction.",
        "> The intervention sets MuJoCo controller Kv so `Kv + MuJoCo passive joint damping = frozen OV explicit-PD Kd`; it does not align passive PhysX damping or establish physical equivalence.",
        "",
        "## Result",
        "",
        f"Scientific status: `{decision['scientific_status']}`.",
        f"Prediction supported: `{str(decision['prediction_supported']).lower()}`.",
        f"Same-source control exactly reproduces frozen friction=0 samples: `{str(baseline['exact_samples']).lower()}`.",
        "",
        "| Run | pinky_CMC peak 0.11–0.16 s (rad) | thumb_DP peak 0.11–0.17 s (mm) | Global joint max (rad) | Global position max (mm) | Global orientation max (rad) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        assert isinstance(row, Mapping)
        primary = row["primary"]
        secondary = row["secondary"]
        assert isinstance(primary, Mapping) and isinstance(secondary, Mapping)
        lines.append(
            "| "
            f"{row['role']} | "
            f"{float(primary['left_pinky_CMC_window']['peak_abs_rad']):.9g} | "  # type: ignore[index]
            f"{float(primary['left_thumb_DP_window']['peak_position_distance_m']) * 1000.0:.9g} | "  # type: ignore[index]
            f"{float(secondary['global_joint_max_abs_rad']):.9g} | "
            f"{float(secondary['global_frame_position_max_m']) * 1000.0:.9g} | "
            f"{float(secondary['global_frame_orientation_max_rad']):.9g} |"
        )
    lines.extend(
        [
            "",
            "Secondary whole-trace RMS (251 samples):",
            "",
            "| Run | Joint RMSE (rad) | Frame position RMS (mm) |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in results:
        assert isinstance(row, Mapping)
        secondary = row["secondary"]
        assert isinstance(secondary, Mapping)
        lines.append(
            "| "
            f"{row['role']} | "
            f"{float(secondary['joint_rmse_rad']):.9g} | "
            f"{float(secondary['frame_position_rms_m']) * 1000.0:.9g} |"
        )
    lines.extend(
        [
            "",
            "Registered endpoint reductions:",
            "",
            f"- `left_pinky_CMC`: `{float(decision['left_pinky_CMC_reduction_fraction']) * 100.0:.4f}%`",
            f"- `left_thumb_DP`: `{float(decision['left_thumb_DP_reduction_fraction']) * 100.0:.4f}%`",
            "- Support requires both reductions to be at least `25%`; rejection requires both to be at most `10%`; all other outcomes are mixed/inconclusive.",
            "",
            "## Evidence identity",
            "",
            f"- Frozen friction bundle root: `{summary['frozen_friction_bundle_root_sha256']}`",
            f"- Diagnostic source revision/tree: `{summary['diagnostic_source_revision']}` / `{summary['diagnostic_source_tree']}`",
            f"- Preregistration SHA-256: `{summary['preregistration_file_sha256']}`",
            "",
            "## Interpretation boundary",
            "",
            str(summary["interpretation"]),
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostic(
    *,
    asset_root: str | Path,
    manifest_path: str | Path,
    preregistration_path: str | Path,
    friction_bundle: str | Path,
    expected_friction_root_sha256: str,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
) -> tuple[Path, dict[str, object]]:
    if expected_friction_root_sha256 != AUTHORITATIVE_FRICTION_ROOT_SHA256:
        raise ControllerKvDiagnosticError(
            "this diagnostic accepts only the authoritative friction bundle"
        )
    if _HEX64.fullmatch(expected_friction_root_sha256) is None:
        raise ControllerKvDiagnosticError("friction root must be a lowercase SHA-256")
    if _HEX40.fullmatch(source_revision) is None:
        raise ControllerKvDiagnosticError("source_revision must be a code commit")
    if _SESSION.fullmatch(session_id) is None:
        raise ControllerKvDiagnosticError("session_id must be portable and non-empty")

    project_root = Path(__file__).resolve(strict=True).parents[1]
    source_files = (
        "scripts/run_mujoco_controller_kv_diagnostic.py",
        "src/wave_asset_qa/__init__.py",
        "src/wave_asset_qa/adapters/__init__.py",
        "src/wave_asset_qa/adapters/base.py",
        "src/wave_asset_qa/adapters/mujoco.py",
        "src/wave_asset_qa/parity/__init__.py",
        "src/wave_asset_qa/parity/bundle.py",
        "src/wave_asset_qa/parity/compare.py",
        "src/wave_asset_qa/parity/contracts.py",
        "src/wave_asset_qa/parity/diagnostics.py",
        "src/wave_asset_qa/parity/mapping.py",
        "src/wave_asset_qa/parity/runner.py",
        "src/wave_asset_qa/parity/scenarios.py",
        PREREG_RELATIVE_PATH,
    )
    source_tree = validate_git_source(
        project_root,
        source_revision,
        required_tracked_paths=source_files,
        sensitive_untracked_paths=(
            "src/wave_asset_qa",
            "scripts/run_mujoco_controller_kv_diagnostic.py",
            PREREG_RELATIVE_PATH,
            "configs/parity/gate0.json",
        ),
    )
    source_hashes = {
        relative: sha256_file(project_root / relative) for relative in source_files
    }
    final_output = validated_results_output_path(project_root, output_dir)

    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    expected_manifest_file = (project_root / "configs/parity/gate0.json").resolve(
        strict=True
    )
    if manifest_file != expected_manifest_file:
        raise ControllerKvDiagnosticError("diagnostic manifest path is not canonical")
    if sha256_file(manifest_file) != AUTHORITATIVE_MANIFEST_FILE_SHA256:
        raise ControllerKvDiagnosticError("diagnostic manifest bytes changed")
    manifest = load_manifest(manifest_file)
    if manifest_sha256(manifest) != AUTHORITATIVE_MANIFEST_SHA256:
        raise ControllerKvDiagnosticError("diagnostic semantic manifest changed")
    prereg_file = Path(preregistration_path).expanduser().resolve(strict=True)
    expected_prereg_file = (project_root / PREREG_RELATIVE_PATH).resolve(strict=True)
    if prereg_file != expected_prereg_file:
        raise ControllerKvDiagnosticError("preregistration path is not canonical")
    prereg_file_sha256 = sha256_file(prereg_file)
    if prereg_file_sha256 != AUTHORITATIVE_PREREGISTRATION_SHA256:
        raise ControllerKvDiagnosticError("canonical preregistration bytes changed")
    prereg, prereg_targets = _load_preregistration(prereg_file, manifest)

    assets = Path(asset_root).expanduser().resolve(strict=True)
    expected_cases = {case.case_id: case for case in expand_scenario_cases(manifest)}
    mujoco_case = expected_cases[MUJOCO_CASE_ID]
    ov_case = expected_cases[OVPHYSX_CASE_ID]
    hand = manifest.hand(mujoco_case.hand)
    model_path = (assets / mujoco_case.model_path).resolve(strict=True)
    try:
        model_path.relative_to(assets)
    except ValueError as exc:
        raise ControllerKvDiagnosticError("MuJoCo model escapes the asset root") from exc
    if sha256_file(model_path) != AUTHORITATIVE_MODEL_SHA256:
        raise ControllerKvDiagnosticError("MuJoCo model bytes changed")

    frozen_root = Path(friction_bundle).expanduser().resolve(strict=True)
    frozen_verification = verify_exact_bundle(frozen_root)
    if frozen_verification["root_sha256"] != expected_friction_root_sha256:
        raise ControllerKvDiagnosticError("frozen friction bundle root mismatch")
    frozen_manifest_path = frozen_root / "bundle.json"
    frozen_summary_path = frozen_root / "summary.json"
    frozen_report_path = frozen_root / "report.md"
    frozen_zero_path = (
        frozen_root
        / "diagnostic"
        / "scale-0"
        / f"{MUJOCO_CASE_ID}.run.json"
    )
    frozen_ov_path = frozen_root / "baseline" / f"{OVPHYSX_CASE_ID}.run.json"
    if (
        sha256_file(frozen_manifest_path) != AUTHORITATIVE_FRICTION_MANIFEST_SHA256
        or sha256_file(frozen_summary_path) != AUTHORITATIVE_FRICTION_SUMMARY_SHA256
        or sha256_file(frozen_zero_path) != AUTHORITATIVE_ZERO_RUN_SHA256
        or sha256_file(frozen_ov_path) != AUTHORITATIVE_OV_RUN_SHA256
    ):
        raise ControllerKvDiagnosticError("frozen friction input hash mismatch")
    frozen_summary = _load_json_object(frozen_summary_path, "frozen friction summary")
    if (
        frozen_summary.get("diagnostic_source_revision")
        != AUTHORITATIVE_FRICTION_SOURCE_REVISION
        or frozen_summary.get("scientific_status") != "supports_local_hypothesis"
    ):
        raise ControllerKvDiagnosticError("frozen friction summary identity changed")
    frozen_zero = _load_one(frozen_zero_path, "frozen friction=0 run")
    frozen_ov = _load_one(frozen_ov_path, "frozen OV run")
    if frozen_zero.case != mujoco_case or frozen_ov.case != ov_case:
        raise ControllerKvDiagnosticError("frozen run cases changed")
    _validate_trace_samples(frozen_zero.result, mujoco_case, manifest)
    _validate_trace_samples(frozen_ov.result, ov_case, manifest)
    if (
        frozen_zero.result.provenance.get("source_revision")
        != AUTHORITATIVE_FRICTION_SOURCE_REVISION
        or frozen_zero.result.provenance.get("dof_frictionloss_scale") != 0.0
        or any(
            value != 0.0
            for value in _number_list(
                frozen_zero.result.provenance.get("dof_frictionloss_effective"),
                "frozen zero effective friction",
                length=len(hand.joint_names),
                nonnegative=True,
            )
        )
    ):
        raise ControllerKvDiagnosticError("frozen friction=0 provenance changed")
    ov_targets, ov_target_audit = _extract_ov_controller_kd(frozen_ov, hand)
    if ov_targets != prereg_targets:
        raise ControllerKvDiagnosticError(
            "preregistered targets do not match mapped frozen OV Kd"
        )

    baseline_joint = _window_joint_peak(
        frozen_ov.result,
        frozen_zero.result,
        joint_name="left_pinky_CMC",
        start_step=55,
        end_step=80,
    )
    baseline_frame = _window_frame_peak(
        frozen_ov.result,
        frozen_zero.result,
        frame_name="left_thumb_DP",
        start_step=55,
        end_step=85,
    )
    if (
        not math.isclose(
            float(baseline_joint["peak_abs_rad"]),
            0.00908693500546992,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or baseline_joint["peak_step"] != 65
        or not math.isclose(
            float(baseline_frame["peak_position_distance_m"]),
            0.0008506666265246331,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or baseline_frame["peak_step"] != 66
    ):
        raise ControllerKvDiagnosticError("frozen primary metric baseline changed")

    staging = create_staging_root(final_output)
    inputs = staging / "inputs"
    current_source_root = inputs / "current-source"
    for relative in source_files:
        destination = current_source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative, destination)
        if sha256_file(destination) != source_hashes[relative]:
            raise ControllerKvDiagnosticError(f"source copy changed: {relative}")
    copied_manifest = inputs / "gate0.manifest.json"
    copied_prereg = inputs / "controller-kv.preregistration.json"
    inputs.mkdir(exist_ok=True)
    shutil.copyfile(manifest_file, copied_manifest)
    shutil.copyfile(prereg_file, copied_prereg)
    if (
        sha256_file(copied_manifest) != AUTHORITATIVE_MANIFEST_FILE_SHA256
        or sha256_file(copied_prereg) != prereg_file_sha256
    ):
        raise ControllerKvDiagnosticError(
            "copied manifest or preregistration bytes changed"
        )

    frozen_inputs = inputs / "frozen"
    frozen_inputs.mkdir()
    copy_spec = {
        "summary.json": frozen_inputs / "friction-summary.json",
        "report.md": frozen_inputs / "friction-report.md",
        frozen_zero_path.relative_to(frozen_root).as_posix(): (
            frozen_inputs / f"{MUJOCO_CASE_ID}.run.json"
        ),
        frozen_ov_path.relative_to(frozen_root).as_posix(): (
            frozen_inputs / f"{OVPHYSX_CASE_ID}.run.json"
        ),
        "baseline/formal-bundle.json": frozen_inputs / "formal-bundle.json",
        "baseline/formal-finalization.json": (
            frozen_inputs / "formal-finalization.json"
        ),
    }
    for parent_relative, destination in copy_spec.items():
        shutil.copyfile(frozen_root / parent_relative, destination)
    copied_frozen_zero = copy_spec[
        frozen_zero_path.relative_to(frozen_root).as_posix()
    ]
    copied_frozen_ov = copy_spec[frozen_ov_path.relative_to(frozen_root).as_posix()]
    copied_zero_run = _load_one(
        copied_frozen_zero,
        "copied frozen friction=0 run",
    )
    copied_ov_run = _load_one(copied_frozen_ov, "copied frozen OV run")
    if (
        copied_zero_run != frozen_zero
        or copied_ov_run != frozen_ov
    ):
        raise ControllerKvDiagnosticError("copied frozen run semantics changed")
    _validate_trace_samples(copied_zero_run.result, mujoco_case, manifest)
    _validate_trace_samples(copied_ov_run.result, ov_case, manifest)
    copied_ov_targets, _ = _extract_ov_controller_kd(copied_ov_run, hand)
    if copied_ov_targets != ov_targets:
        raise ControllerKvDiagnosticError("copied frozen OV Kd mapping changed")
    copied_formal_manifest = copy_spec["baseline/formal-bundle.json"]
    copied_formal_finalization = copy_spec["baseline/formal-finalization.json"]
    if (
        sha256_file(copied_formal_manifest) != AUTHORITATIVE_FORMAL_MANIFEST_SHA256
        or sha256_file(copied_formal_finalization)
        != AUTHORITATIVE_FORMAL_FINALIZATION_SHA256
    ):
        raise ControllerKvDiagnosticError("copied formal evidence hash changed")
    formal_manifest = _load_json_object(
        copied_formal_manifest,
        "copied formal bundle manifest",
    )
    formal_finalization = _load_json_object(
        copied_formal_finalization,
        "copied formal finalization",
    )
    if formal_manifest.get("root_sha256") != AUTHORITATIVE_FORMAL_ROOT_SHA256:
        raise ControllerKvDiagnosticError("copied formal bundle root changed")
    expected_finalization = {
        "schema_version": 1,
        "execution_status": "completed",
        "comparison_status": "divergent",
        "source_revision": AUTHORITATIVE_FORMAL_SOURCE_REVISION,
        "source_tree": AUTHORITATIVE_FORMAL_SOURCE_TREE,
        "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
    }
    for field, expected in expected_finalization.items():
        if formal_finalization.get(field) != expected:
            raise ControllerKvDiagnosticError(
                f"copied formal finalization.{field} changed"
            )
    copied_friction_manifest = frozen_inputs / "friction-bundle.json"
    shutil.copyfile(frozen_manifest_path, copied_friction_manifest)
    if sha256_file(copied_friction_manifest) != AUTHORITATIVE_FRICTION_MANIFEST_SHA256:
        raise ControllerKvDiagnosticError("copied friction manifest changed")
    verify_copied_bundle_subset(
        copied_friction_manifest,
        expected_root_sha256=AUTHORITATIVE_FRICTION_ROOT_SHA256,
        copied_payloads=copy_spec,
    )
    frozen_source_root = frozen_root / "inputs" / "source"
    copied_frozen_source_root = inputs / "friction-source"
    copied_frozen_source_records: dict[str, Path] = {}
    for source in frozen_source_root.rglob("*"):
        if not source.is_file():
            continue
        relative_inside_source = source.relative_to(frozen_source_root)
        destination = copied_frozen_source_root / relative_inside_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        parent_relative = source.relative_to(frozen_root).as_posix()
        copied_frozen_source_records[parent_relative] = destination
    verify_copied_bundle_subset(
        copied_friction_manifest,
        expected_root_sha256=AUTHORITATIVE_FRICTION_ROOT_SHA256,
        copied_payloads=copied_frozen_source_records,
    )
    runs: dict[str, tuple[CollectedRun, Path]] = {}
    for role, targets in ((CONTROL_ROLE, None), (TREATMENT_ROLE, ov_targets)):
        run_dir = staging / "runs" / role
        adapter_factory = partial(
            MuJoCoAdapter,
            dof_frictionloss_scale=0.0,
            joint_total_damping_targets=targets,
        )
        collected = run_backend_cases(
            assets,
            manifest,
            run_dir,
            backend=Simulator.MUJOCO,
            case_ids=(MUJOCO_CASE_ID,),
            adapter_factory=adapter_factory,
            session_id=session_id,
            source_revision=source_revision,
            verify_asset_tree=True,
        )
        if len(collected) != 1:
            raise ControllerKvDiagnosticError(f"{role} did not produce exactly one run")
        _validate_new_mujoco_run(
            collected[0],
            expected_case=mujoco_case,
            manifest=manifest,
            session_id=session_id,
            source_revision=source_revision,
            source_sha256=AUTHORITATIVE_MODEL_SHA256,
            target_map=targets,
        )
        run_path = run_dir / f"{MUJOCO_CASE_ID}.run.json"
        if not run_path.is_file():
            raise ControllerKvDiagnosticError(f"{role} run JSON is missing")
        runs[role] = (collected[0], run_path)

    control = runs[CONTROL_ROLE][0]
    treatment = runs[TREATMENT_ROLE][0]
    (
        baseline_drift,
        control_row,
        treatment_row,
        decision,
    ) = _audit_staged_runs(
        runs,
        frozen_zero=frozen_zero,
        frozen_ov=frozen_ov,
        mujoco_case=mujoco_case,
        manifest=manifest,
        hand=hand,
        session_id=session_id,
        source_revision=source_revision,
        treatment_targets=ov_targets,
    )
    status = str(decision["scientific_status"])
    if (
        status
        == "supports_registered_local_gap_reduction_under_total_viscous_retuning"
    ):
        interpretation = (
            "For this one left small_step endpoint diagnostic under friction=0, adjusting "
            "MuJoCo controller Kv so controller Kv plus MuJoCo passive joint damping equals "
            "the frozen OV explicit-PD Kd reduced both registered transient gaps by at least "
            "25%. This shows that both registered local gaps are sensitive to this exact "
            "retuning and is consistent with a total-viscous parameter contribution. It "
            "does not prove a cross-backend parameter mismatch: the adjustment may "
            "compensate for other backend differences and does not establish damping "
            "semantics or physical equivalence."
        )
    elif (
        status
        == "rejects_total_viscous_alignment_as_common_major_local_explanation"
    ):
        interpretation = (
            "Both registered transient gaps improved by at most 10%, so this endpoint test "
            "rejects this total-viscous alignment via controller-Kv adjustment as their "
            "common major residual explanation."
        )
    else:
        interpretation = (
            "The two registered transient gaps did not satisfy the same support or rejection "
            "rule. The total-viscous endpoint result is mixed or inconclusive."
        )
    interpretation += (
        " The formal Gate 0 result remains DIVERGENT / pass_ready=false; this local "
        "simulation-only diagnostic is not a backend bug, hardware, safety, or Sim2Real claim."
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "completed",
        "scientific_status": status,
        "classification": "diagnostic_only_not_formal_gate0",
        "frozen_friction_bundle_root_sha256": AUTHORITATIVE_FRICTION_ROOT_SHA256,
        "frozen_friction_bundle_verification": frozen_verification,
        "frozen_zero_run_sha256": AUTHORITATIVE_ZERO_RUN_SHA256,
        "frozen_ovphysx_run_sha256": AUTHORITATIVE_OV_RUN_SHA256,
        "diagnostic_source_revision": source_revision,
        "diagnostic_source_tree": source_tree,
        "diagnostic_source_files_sha256": source_hashes,
        "session_id": session_id,
        "manifest_file_sha256": sha256_file(copied_manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration": prereg,
        "ov_controller_kd_target_audit": ov_target_audit,
        "baseline_drift": {
            "exact_samples": control.result.samples == frozen_zero.result.samples,
            "tolerance": 1e-9,
            "joint_max_abs_rad": baseline_drift[0],
            "frame_position_max_m": baseline_drift[1],
            "frame_orientation_max_rad": baseline_drift[2],
        },
        "single_variable_checks": {
            "same_source_and_session": True,
            "same_commands_and_initial_state": True,
            "same_original_and_effective_zero_friction": True,
            "non_kv_provenance_identical": True,
            "fresh_compiled_damping_source_identical": True,
            "all_22_total_viscous_damping_targets_read_back": True,
            "actuator_gainprm_unchanged": True,
            "only_actuator_biasprm_index_2_may_change": True,
            "canonical_asset_files_modified": False,
        },
        "frozen_primary_baseline": {
            "left_pinky_CMC_window": baseline_joint,
            "left_thumb_DP_window": baseline_frame,
        },
        "results": [control_row, treatment_row],
        "decision": decision,
        "interpretation": interpretation,
        "limits": [
            "Only one left small_step base r01 endpoint treatment is tested.",
            "The control and treatment each have one execution; this diagnostic does not establish repeatability or dt-halving stability.",
            "The numeric target comes from frozen OV explicit-PD Kd, not passive PhysX damping.",
            "The experiment does not establish equivalent continuous or discrete damping semantics.",
            "Global and RMS metrics are secondary exploratory observations.",
            "Formal Gate 0 thresholds and status are unchanged.",
        ],
    }

    observed_tree = validate_git_source(
        project_root,
        source_revision,
        required_tracked_paths=source_files,
        sensitive_untracked_paths=(
            "src/wave_asset_qa",
            "scripts/run_mujoco_controller_kv_diagnostic.py",
            PREREG_RELATIVE_PATH,
            "configs/parity/gate0.json",
        ),
    )
    if observed_tree != source_tree:
        raise ControllerKvDiagnosticError("diagnostic source tree changed during run")
    if {
        relative: sha256_file(project_root / relative) for relative in source_files
    } != source_hashes:
        raise ControllerKvDiagnosticError("diagnostic source files changed during run")
    if (
        sha256_file(manifest_file) != AUTHORITATIVE_MANIFEST_FILE_SHA256
        or sha256_file(prereg_file) != prereg_file_sha256
        or verify_exact_bundle(frozen_root)["root_sha256"]
        != AUTHORITATIVE_FRICTION_ROOT_SHA256
    ):
        raise ControllerKvDiagnosticError("frozen input changed during run")

    copied_source_hashes = {
        relative: sha256_file(current_source_root / relative)
        for relative in source_files
    }
    if copied_source_hashes != source_hashes:
        raise ControllerKvDiagnosticError("staged current-source copies changed")
    if (
        sha256_file(copied_manifest) != AUTHORITATIVE_MANIFEST_FILE_SHA256
        or manifest_sha256(load_manifest(copied_manifest))
        != AUTHORITATIVE_MANIFEST_SHA256
        or sha256_file(copied_prereg) != prereg_file_sha256
    ):
        raise ControllerKvDiagnosticError("staged manifest or preregistration changed")
    copied_preregistration, copied_targets = _load_preregistration(
        copied_prereg,
        manifest,
    )
    if copied_preregistration != prereg or copied_targets != prereg_targets:
        raise ControllerKvDiagnosticError(
            "staged preregistration semantics changed"
        )
    if (
        sha256_file(copied_friction_manifest)
        != AUTHORITATIVE_FRICTION_MANIFEST_SHA256
        or sha256_file(copied_formal_manifest)
        != AUTHORITATIVE_FORMAL_MANIFEST_SHA256
        or sha256_file(copied_formal_finalization)
        != AUTHORITATIVE_FORMAL_FINALIZATION_SHA256
    ):
        raise ControllerKvDiagnosticError("staged frozen evidence identity changed")
    verify_copied_bundle_subset(
        copied_friction_manifest,
        expected_root_sha256=AUTHORITATIVE_FRICTION_ROOT_SHA256,
        copied_payloads=copy_spec,
    )
    verify_copied_bundle_subset(
        copied_friction_manifest,
        expected_root_sha256=AUTHORITATIVE_FRICTION_ROOT_SHA256,
        copied_payloads=copied_frozen_source_records,
    )
    final_copied_zero = _load_one(
        copied_frozen_zero,
        "final copied frozen friction=0 run",
    )
    final_copied_ov = _load_one(
        copied_frozen_ov,
        "final copied frozen OV run",
    )
    if final_copied_zero != frozen_zero or final_copied_ov != frozen_ov:
        raise ControllerKvDiagnosticError(
            "staged frozen run semantics changed"
        )
    _validate_trace_samples(final_copied_zero.result, mujoco_case, manifest)
    _validate_trace_samples(final_copied_ov.result, ov_case, manifest)
    final_ov_targets, _ = _extract_ov_controller_kd(final_copied_ov, hand)
    if final_ov_targets != ov_targets:
        raise ControllerKvDiagnosticError("staged frozen OV Kd mapping changed")
    (
        reloaded_baseline_drift,
        reloaded_control_row,
        reloaded_treatment_row,
        reloaded_decision,
    ) = _audit_staged_runs(
        runs,
        frozen_zero=frozen_zero,
        frozen_ov=frozen_ov,
        mujoco_case=mujoco_case,
        manifest=manifest,
        hand=hand,
        session_id=session_id,
        source_revision=source_revision,
        treatment_targets=ov_targets,
    )
    if (
        reloaded_baseline_drift != baseline_drift
        or reloaded_control_row != control_row
        or reloaded_treatment_row != treatment_row
        or reloaded_decision != decision
    ):
        raise ControllerKvDiagnosticError(
            "staged run analysis changed before finalization"
        )

    summary_path = staging / "summary.json"
    report_path = staging / "report.md"
    report_content = _render_report(summary)
    write_json_atomic(summary_path, summary)
    write_text_exclusive(report_path, report_content)
    if (
        _load_json_object(summary_path, "staged diagnostic summary") != summary
        or report_path.read_text(encoding="utf-8") != report_content
    ):
        raise ControllerKvDiagnosticError(
            "staged summary or report failed semantic readback"
        )

    expected_payload_paths = {
        (current_source_root / relative).relative_to(staging).as_posix()
        for relative in source_files
    }
    expected_payload_paths.update(
        destination.relative_to(staging).as_posix()
        for destination in copy_spec.values()
    )
    expected_payload_paths.update(
        destination.relative_to(staging).as_posix()
        for destination in copied_frozen_source_records.values()
    )
    expected_payload_paths.update(
        {
            copied_manifest.relative_to(staging).as_posix(),
            copied_prereg.relative_to(staging).as_posix(),
            copied_friction_manifest.relative_to(staging).as_posix(),
            runs[CONTROL_ROLE][1].relative_to(staging).as_posix(),
            runs[TREATMENT_ROLE][1].relative_to(staging).as_posix(),
            summary_path.relative_to(staging).as_posix(),
            report_path.relative_to(staging).as_posix(),
        }
    )
    observed_payload_paths = set(bundle_payload_paths(staging))
    if observed_payload_paths != expected_payload_paths:
        raise ControllerKvDiagnosticError(
            "staged payload inventory differs from the closed diagnostic schema: "
            f"missing={sorted(expected_payload_paths - observed_payload_paths)}, "
            f"extra={sorted(observed_payload_paths - expected_payload_paths)}"
        )
    write_bundle_manifest(staging, sorted(expected_payload_paths))
    staged_verification = verify_exact_bundle(staging)
    if (
        _load_json_object(summary_path, "final staged diagnostic summary") != summary
        or report_path.read_text(encoding="utf-8") != report_content
    ):
        raise ControllerKvDiagnosticError(
            "staged summary or report changed during bundle finalization"
        )
    final_stage_audit = _audit_staged_runs(
        runs,
        frozen_zero=frozen_zero,
        frozen_ov=frozen_ov,
        mujoco_case=mujoco_case,
        manifest=manifest,
        hand=hand,
        session_id=session_id,
        source_revision=source_revision,
        treatment_targets=ov_targets,
    )
    if final_stage_audit != (
        baseline_drift,
        control_row,
        treatment_row,
        decision,
    ) or verify_exact_bundle(staging) != staged_verification:
        raise ControllerKvDiagnosticError(
            "staged semantic evidence changed during bundle finalization"
        )
    promoted = promote_staging_root(staging, final_output)
    final_verification = verify_exact_bundle(promoted)
    if staged_verification != final_verification:
        raise ControllerKvDiagnosticError("bundle identity changed during promotion")
    return promoted, final_verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--friction-bundle", required=True, type=Path)
    parser.add_argument("--expected-friction-root-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, verification = run_diagnostic(
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            preregistration_path=args.preregistration,
            friction_bundle=args.friction_bundle,
            expected_friction_root_sha256=args.expected_friction_root_sha256,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
        )
    except (DiagnosticEvidenceError, OSError, ValueError) as exc:
        print(
            f"MuJoCo total-viscous/controller-Kv diagnostic failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": output.name,
                **verification,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
