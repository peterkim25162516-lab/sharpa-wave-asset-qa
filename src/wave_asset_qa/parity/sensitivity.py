"""CPU-only contracts and primary metrics for WaveSimParity Freeze B.

The public protocol contains publishable aggregate evidence.  The separate
private plan contains the two 22-joint authored-value maps and the exact case
inventory.  Nothing here assigns a MuJoCo meaning or unit to the native
OVPhysX input under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

from wave_asset_qa.adapters.base import TraceSample

from .compare import (
    MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S,
    MAX_ABS_CANONICAL_JOINT_POSITION_RAD,
    MAX_ABS_CANONICAL_POSITION_TARGET_RAD,
    MAX_FRAME_ORIGIN_DISTANCE_M,
    CollectedRun,
)
from .contracts import DISTAL_FRAME_SUFFIXES, JOINT_SUFFIXES, HandSide, Simulator
from .scenarios import TimestepVariant


PUBLIC_PROTOCOL_SCHEMA_VERSION = 1
PRIVATE_PLAN_SCHEMA_VERSION = 1
WINDOW_START_S = 0.1
WINDOW_END_S = 0.5
SUPPORTED_MIN_S = 0.25
NOT_SUPPORTED_MAX_S = 0.10
EXPECTED_S_COUNT = 8
BASELINE_RECHECK_ABS_TOLERANCE = 1e-15
TIME_GRID_ABS_TOLERANCE_S = 1e-12
REQUIRED_VALIDITY_CHECKS = frozenset(
    {
        "case_completion_24_of_24",
        "joint_mapping_44_of_44",
        "distal_frame_mapping_10_of_10",
        "fresh_process_24_of_24",
        "ovphysx_contact_unobserved_disclosed",
        "mujoco_zero_contacts",
        "dt_v2_evidence",
        "intervention_write_readback",
        "non_target_configuration_invariance",
        "r1_legacy_and_runtime_bridge",
        "formal_same_source_bridge",
        "fresh_repeatability",
        "dt_halving",
        "position_target_readback_within_1e_6",
        "finite_and_sanity_bounds",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_HANDS = (HandSide.LEFT, HandSide.RIGHT)
_VARIANTS = (TimestepVariant.BASE, TimestepVariant.HALVED)
_DT = {TimestepVariant.BASE: 0.002, TimestepVariant.HALVED: 0.001}
_SEMANTIC_FAMILIES = ("armature", "control_drive", "friction", "solver_runtime")
_READBACK_CASE_IDS = (
    "ovphysx.left.effective_readback.r01",
    "ovphysx.left.effective_readback.r02",
    "ovphysx.right.effective_readback.r01",
    "ovphysx.right.effective_readback.r02",
)

_PRIVATE_KEYS = {
    "schema_version",
    "protocol_id",
    "inputs",
    "hand_plans",
    "ovphysx_cases",
    "mujoco_cases",
    "run_order",
}
_INPUT_KEYS = {
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
_HAND_PLAN_KEYS = {
    "hand",
    "joint_names",
    "joint_prim_paths",
    "expected_pre_values",
    "sham_write_values",
    "zero_write_values",
    "expected_pre_float32_sha256",
}
_CASE_KEYS = {
    "experiment_case_id",
    "canonical_case_id",
    "backend",
    "role",
    "hand",
    "timestep_variant",
    "dt_s",
    "repeat_index",
}
_PUBLIC_KEYS = {
    "schema_version",
    "protocol_id",
    "state",
    "private_plan_sha256",
    "inputs",
    "semantic_classification_per_field_family",
    "candidate_hypothesis",
    "intervention",
    "control_and_force_law",
    "scenario",
    "run_matrix",
    "controls",
    "metrics",
    "decision_thresholds",
    "validity_thresholds",
    "bounded_compute_budget",
    "privacy",
    "claim_boundary",
}
_EVIDENCE_BINDING_KEYS = {
    "private_plan_sha256",
    "public_protocol_file_sha256",
    "public_protocol_canonical_sha256",
    "deployment_source_revision",
    "deployment_source_tree",
    "deployment_source_snapshot_sha256",
}


class SensitivityValidationError(ValueError):
    """A protocol, private plan, or trace failed closed."""


class SensitivityStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SensitivityValidationError(f"{path} must be an object with string keys")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise SensitivityValidationError(f"{path} must be an array")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise SensitivityValidationError(f"{path} fields invalid: {'; '.join(details)}")


def _literal(value: object, expected: object, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise SensitivityValidationError(f"{path} must be {expected!r}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SensitivityValidationError(f"{path} must be a non-empty string")
    return value


def _number(
    value: object,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensitivityValidationError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SensitivityValidationError(f"{path} must be finite")
    if positive and result <= 0.0:
        raise SensitivityValidationError(f"{path} must be positive")
    if nonnegative and result < 0.0:
        raise SensitivityValidationError(f"{path} must be non-negative")
    return result


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SensitivityValidationError(f"{path} must be a lowercase SHA-256")
    return value


def _git_oid(value: object, path: str) -> str:
    if not isinstance(value, str) or not _GIT_OID_RE.fullmatch(value):
        raise SensitivityValidationError(f"{path} must be a lowercase 40-character Git OID")
    return value


def _json_copy(value: object) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SensitivityValidationError("value is not canonical JSON data") from exc
    return json.loads(encoded)


def canonical_json_sha256(value: object) -> str:
    copied = _json_copy(value)
    encoded = json.dumps(
        copied,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _strict_json(path: str | Path, label: str) -> object:
    source = Path(path)

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SensitivityValidationError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise SensitivityValidationError(f"{label} contains non-finite constant {token}")

    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensitivityValidationError(f"cannot load {label} {source}: {exc}") from exc


def canonical_joint_names(hand: HandSide | str) -> tuple[str, ...]:
    side = HandSide(hand)
    return tuple(f"{side.value}_{suffix}" for suffix in JOINT_SUFFIXES)


def canonical_frame_names(hand: HandSide | str) -> tuple[str, ...]:
    side = HandSide(hand)
    return tuple(f"{side.value}_{suffix}" for suffix in DISTAL_FRAME_SUFFIXES)


def _baseline_map(value: object, path: str) -> dict[str, object]:
    hands = _mapping(value, path)
    _exact(hands, {"left", "right"}, path)
    for hand in ("left", "right"):
        variants = _mapping(hands[hand], f"{path}.{hand}")
        _exact(variants, {"base", "halved"}, f"{path}.{hand}")
        for variant in ("base", "halved"):
            metrics = _mapping(variants[variant], f"{path}.{hand}.{variant}")
            _exact(
                metrics,
                {"joint_rmse_rad", "frame_position_rmse_m"},
                f"{path}.{hand}.{variant}",
            )
            _number(
                metrics["joint_rmse_rad"],
                f"{path}.{hand}.{variant}.joint_rmse_rad",
                positive=True,
            )
            _number(
                metrics["frame_position_rmse_m"],
                f"{path}.{hand}.{variant}.frame_position_rmse_m",
                positive=True,
            )
    return _json_copy(hands)


def _validate_inputs(value: object, path: str) -> dict[str, object]:
    data = _mapping(value, path)
    _exact(data, _INPUT_KEYS, path)
    for key in (
        "freeze_a_config_sha256",
        "readback_bundle_root_sha256",
        "readback_summary_sha256",
        "formal_gate0_bundle_root_sha256",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "canonical_lf_asset_tree_sha256",
    ):
        _digest(data[key], f"{path}.{key}")
    for key in (
        "readback_source_revision",
        "readback_source_tree",
        "formal_gate0_source_revision",
        "asset_commit",
        "asset_git_tree",
        "freeze_b_source_revision",
        "freeze_b_source_tree",
    ):
        _git_oid(data[key], f"{path}.{key}")
    readback = _mapping(data["readback_case_sha256"], f"{path}.readback_case_sha256")
    _exact(readback, set(_READBACK_CASE_IDS), f"{path}.readback_case_sha256")
    for case_id in _READBACK_CASE_IDS:
        _digest(readback[case_id], f"{path}.readback_case_sha256.{case_id}")
    _baseline_map(data["formal_crosssim_window_rmse"], f"{path}.formal_crosssim_window_rmse")
    return _json_copy(data)


def _float32(value: object, path: str) -> tuple[float, bytes]:
    numeric = _number(value, path)
    try:
        payload = struct.pack("!f", numeric)
        rounded = struct.unpack("!f", payload)[0]
    except (OverflowError, struct.error) as exc:
        raise SensitivityValidationError(f"{path} is not finite float32") from exc
    if not math.isfinite(rounded):
        raise SensitivityValidationError(f"{path} is not finite float32")
    return rounded, payload


def expected_pre_float32_sha256(
    names: Sequence[str], values: Mapping[str, object]
) -> str:
    """Hash ordered names and big-endian binary32 values with a domain tag."""

    ordered_names = tuple(names)
    if (
        len(ordered_names) != 22
        or len(set(ordered_names)) != 22
        or any(not isinstance(name, str) or not name for name in ordered_names)
    ):
        raise SensitivityValidationError("float32 vector must have 22 unique names")
    value_map = _mapping(values, "expected_pre_values")
    _exact(value_map, set(ordered_names), "expected_pre_values")
    digest = sha256(b"waveqa-freeze-b-float32-v1\0")
    for name in ordered_names:
        encoded = name.encode("utf-8")
        _, payload = _float32(value_map[name], f"expected_pre_values.{name}")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(payload)
    return digest.hexdigest()


def expected_case_matrix(backend: Simulator | str) -> tuple[dict[str, object], ...]:
    simulator = Simulator(backend)
    cases: list[dict[str, object]] = []
    for hand in _HANDS:
        for variant in _VARIANTS:
            for repeat in (1, 2):
                canonical = (
                    f"{simulator.value}.{hand.value}.small_step."
                    f"{variant.value}.r{repeat:02d}"
                )
                roles = (
                    (("sham", "zero") if repeat == 1 else ("zero", "sham"))
                    if simulator is Simulator.OVPHYSX
                    else ("control",)
                )
                for role in roles:
                    experiment = (
                        f"freeze_b.{simulator.value}.{hand.value}.small_step."
                        f"{variant.value}.{role}.r{repeat:02d}"
                    )
                    cases.append(
                        {
                            "experiment_case_id": experiment,
                            "canonical_case_id": canonical,
                            "backend": simulator.value,
                            "role": role,
                            "hand": hand.value,
                            "timestep_variant": variant.value,
                            "dt_s": _DT[variant],
                            "repeat_index": repeat,
                        }
                    )
    return tuple(cases)


def expected_run_order() -> tuple[str, ...]:
    return tuple(
        str(case["experiment_case_id"])
        for case in expected_case_matrix(Simulator.OVPHYSX)
    )


def _number_map(
    value: object,
    names: tuple[str, ...],
    path: str,
) -> dict[str, float]:
    data = _mapping(value, path)
    _exact(data, set(names), path)
    result: dict[str, float] = {}
    for name in names:
        rounded, _ = _float32(data[name], f"{path}.{name}")
        result[name] = rounded
    return result


def _validate_hand_plan(
    value: object, expected_hand: HandSide, path: str
) -> dict[str, object]:
    data = _mapping(value, path)
    _exact(data, _HAND_PLAN_KEYS, path)
    _literal(data["hand"], expected_hand.value, f"{path}.hand")
    names = canonical_joint_names(expected_hand)
    _literal(data["joint_names"], list(names), f"{path}.joint_names")
    paths = _mapping(data["joint_prim_paths"], f"{path}.joint_prim_paths")
    _exact(paths, set(names), f"{path}.joint_prim_paths")
    for name in names:
        _literal(
            paths[name],
            f"/World/Env_0/Robot/joints/{name}",
            f"{path}.joint_prim_paths.{name}",
        )
    expected = _number_map(data["expected_pre_values"], names, f"{path}.expected_pre_values")
    sham = _number_map(data["sham_write_values"], names, f"{path}.sham_write_values")
    zero = _number_map(data["zero_write_values"], names, f"{path}.zero_write_values")
    for name in names:
        expected_value, expected_bits = _float32(expected[name], f"{path}.expected.{name}")
        sham_value, sham_bits = _float32(sham[name], f"{path}.sham.{name}")
        zero_value, zero_bits = _float32(zero[name], f"{path}.zero.{name}")
        if expected_value <= 0.0:
            raise SensitivityValidationError(f"{path} expected values must be positive")
        if sham_value != expected_value or sham_bits != expected_bits:
            raise SensitivityValidationError(f"{path} sham values must preserve exact float32 bits")
        if zero_value != 0.0 or zero_bits != b"\0\0\0\0":
            raise SensitivityValidationError(f"{path} zero values must be positive float32 zero")
    observed = _digest(
        data["expected_pre_float32_sha256"],
        f"{path}.expected_pre_float32_sha256",
    )
    if observed != expected_pre_float32_sha256(names, expected):
        raise SensitivityValidationError(f"{path}.expected_pre_float32_sha256 mismatch")
    return _json_copy(data)


def _validate_cases(
    value: object, backend: Simulator, path: str
) -> list[dict[str, object]]:
    rows = _array(value, path)
    expected = expected_case_matrix(backend)
    if len(rows) != len(expected):
        raise SensitivityValidationError(f"{path} has the wrong case count")
    for index, row in enumerate(rows):
        _exact(_mapping(row, f"{path}[{index}]"), _CASE_KEYS, f"{path}[{index}]")
    if list(rows) != list(expected):
        raise SensitivityValidationError(f"{path} differs from the deterministic matrix")
    return _json_copy(rows)


def validate_private_plan(value: object) -> dict[str, object]:
    """Validate the exact seven-key private plan produced by the preparer."""

    data = _mapping(value, "private_plan")
    _exact(data, _PRIVATE_KEYS, "private_plan")
    _literal(data["schema_version"], PRIVATE_PLAN_SCHEMA_VERSION, "private_plan.schema_version")
    protocol_id = _string(data["protocol_id"], "private_plan.protocol_id")
    if not _IDENTIFIER_RE.fullmatch(protocol_id):
        raise SensitivityValidationError("private_plan.protocol_id is not a safe identifier")
    _validate_inputs(data["inputs"], "private_plan.inputs")
    hands = _array(data["hand_plans"], "private_plan.hand_plans")
    if len(hands) != 2:
        raise SensitivityValidationError("private_plan.hand_plans must contain two hands")
    for index, hand in enumerate(_HANDS):
        _validate_hand_plan(hands[index], hand, f"private_plan.hand_plans[{index}]")
    ov_cases = _validate_cases(
        data["ovphysx_cases"], Simulator.OVPHYSX, "private_plan.ovphysx_cases"
    )
    _validate_cases(
        data["mujoco_cases"], Simulator.MUJOCO, "private_plan.mujoco_cases"
    )
    _literal(data["run_order"], list(expected_run_order()), "private_plan.run_order")
    if {case["experiment_case_id"] for case in ov_cases} != set(data["run_order"]):
        raise SensitivityValidationError("private_plan.run_order must cover the 16 OV cases")
    return _json_copy(data)


def private_plan_sha256(value: object) -> str:
    """Return the external canonical hash of the whole validated private plan."""

    return canonical_json_sha256(validate_private_plan(value))


def load_private_plan(
    path: str | Path,
    *,
    public_protocol: object | None = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    plan = validate_private_plan(_strict_json(path, "private plan"))
    observed = private_plan_sha256(plan)
    if expected_sha256 is not None and observed != _digest(expected_sha256, "expected_sha256"):
        raise SensitivityValidationError("private plan SHA-256 mismatch")
    if public_protocol is not None:
        public = validate_public_protocol(public_protocol)
        if plan["protocol_id"] != public["protocol_id"]:
            raise SensitivityValidationError("private/public protocol_id mismatch")
        if plan["inputs"] != public["inputs"]:
            raise SensitivityValidationError("private/public inputs mismatch")
        if observed != public["private_plan_sha256"]:
            raise SensitivityValidationError("private plan hash is not pinned by public protocol")
    return plan


def experiment_case(private_plan: object, experiment_case_id: str) -> dict[str, object]:
    """Resolve one case and, for OVPhysX, its expected and write value maps."""

    plan = validate_private_plan(private_plan)
    requested = _string(experiment_case_id, "experiment_case_id")
    cases = [*plan["ovphysx_cases"], *plan["mujoco_cases"]]
    matches = [case for case in cases if case["experiment_case_id"] == requested]
    if len(matches) != 1:
        raise SensitivityValidationError(f"unknown experiment_case_id: {requested}")
    case = dict(matches[0])
    result: dict[str, object] = {"case": case}
    if case["backend"] == "ovphysx":
        hand_plan = next(
            item for item in plan["hand_plans"] if item["hand"] == case["hand"]
        )
        role = str(case["role"])
        result.update(
            {
                "joint_names": list(hand_plan["joint_names"]),
                "joint_prim_paths": dict(hand_plan["joint_prim_paths"]),
                "expected_pre_values": dict(hand_plan["expected_pre_values"]),
                "write_values": dict(hand_plan[f"{role}_write_values"]),
                "expected_pre_float32_sha256": hand_plan[
                    "expected_pre_float32_sha256"
                ],
            }
        )
    else:
        result.update(
            {
                "joint_names": None,
                "joint_prim_paths": None,
                "expected_pre_values": None,
                "write_values": None,
                "expected_pre_float32_sha256": None,
            }
        )
    return result


def _semantic_map(value: object, path: str) -> None:
    data = _mapping(value, path)
    _exact(data, set(_SEMANTIC_FAMILIES), path)
    for family in _SEMANTIC_FAMILIES:
        _literal(data[family], "UNMAPPABLE", f"{path}.{family}")


_INTERVENTION = {
    "backend": "ovphysx",
    "source_attribute": "physxJoint:jointFriction",
    "application_time": (
        "live_composed_stage_after_articulation_construction_while_"
        "articulation_uninitialized_before_first_simulation_reset"
    ),
    "sham": "write_exact_frozen_authored_float32_value",
    "treatment": "write_literal_positive_float32_zero",
    "treatment_value": 0.0,
    "semantic_boundary": "native_ovphysx_input_sensitivity_only_no_cross_engine_transfer",
}
_CONTROL_LAW = {
    "mounting": "fixed_base",
    "control_mode": "position",
    "actuator_model": "IdealPDActuator_explicit_PD_effort",
    "target": "canonical_small_step_all_22_joints",
    "mujoco_mutation": "none",
    "non_intervention_configuration": "invariant_between_ovphysx_sham_and_zero",
}
_SCENARIO = {
    "scenario_id": "small_step",
    "duration_s": 0.5,
    "window_start_s": 0.1,
    "window_end_s": 0.5,
    "window_selection": "recorded_time_s >= start and recorded_time_s <= end",
    "base_dt_s": 0.002,
    "halved_dt_s": 0.001,
    "hands": ["left", "right"],
    "repeat_count": 2,
    "environment": "zero_gravity_no_ground_noncontact_target",
}
_RUN_MATRIX = {
    "ovphysx_case_count": 16,
    "mujoco_case_count": 8,
    "ovphysx_roles": ["sham", "zero"],
    "mujoco_roles": ["control"],
    "fresh_process_per_case": True,
    "ovphysx_order_pinned_in_private_plan": True,
}
_CONTROLS = [
    "same_source_unchanged_mujoco_control",
    "same_source_ovphysx_sham",
    "frozen_formal_mujoco_reference",
    "frozen_formal_ovphysx_reference",
    "left_and_right_hand_symmetry_check",
    "fresh_process_repeatability",
    "dt_halving",
    "non_target_readback_invariance",
]
_METRICS = {
    "Dq": "elementwise_RMSE_over_window_times_22_joint_positions_rad",
    "Dp": "elementwise_RMSE_over_window_times_5_frame_xyz_positions_m",
    "S": "treatment_vs_sham_RMSE_divided_by_frozen_crosssim_baseline_RMSE",
    "secondary_R": (
        "1_minus_zero_vs_frozen_mujoco_RMSE_divided_by_frozen_crosssim_"
        "baseline_RMSE"
    ),
    "secondary_R_role": "descriptive_gap_direction_only_cannot_change_primary_status",
    "axes": [
        "left.base.Dq",
        "left.base.Dp",
        "left.halved.Dq",
        "left.halved.Dp",
        "right.base.Dq",
        "right.base.Dp",
        "right.halved.Dq",
        "right.halved.Dp",
    ],
    "orientation": "validity_repeatability_and_dt_halving_only_not_in_S",
    "repeat_collapse": (
        "primary_metrics_use_r01_only_after_r01_r02_repeatability_gate_passes_"
        "for_every_backend_hand_dt_role;_r02_is_gate_only_and_must_not_be_"
        "selected_for_favorable_results"
    ),
}
_DECISION = {
    "expected_s_count": 8,
    "SUPPORTED_if_all_S_at_least": 0.25,
    "NOT_SUPPORTED_if_all_S_at_most": 0.10,
    "otherwise": "INCONCLUSIVE",
    "validity_failure": "validation_status_INVALID_and_scientific_label_null",
}
_VALIDITY = {
    "formal_baseline_recheck_abs_tolerance": 1e-15,
    "same_source_bridge_max_abs": 1e-9,
    "repeatability_max_abs": 1e-9,
    "dt_halving_joint_max_abs_rad": 0.01,
    "dt_halving_frame_position_max_m": 0.001,
    "dt_halving_frame_orientation_max_rad": 0.02,
    "quaternion_norm_abs_tolerance": 0.001,
    "minimum_completion_fraction": 1.0,
    "maximum_nonfinite_count": 0,
    "position_target_readback_max_abs_rad": 1e-6,
    "max_abs_canonical_joint_position_rad": (
        MAX_ABS_CANONICAL_JOINT_POSITION_RAD
    ),
    "max_abs_canonical_position_target_rad": (
        MAX_ABS_CANONICAL_POSITION_TARGET_RAD
    ),
    "max_abs_backend_joint_velocity_rad_s": (
        MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S
    ),
    "max_frame_origin_distance_m": MAX_FRAME_ORIGIN_DISTANCE_M,
    "mujoco_contact_count": 0,
    "ovphysx_contact_count": None,
    "contact_observation_boundary": "ovphysx_contact_check_performed_false",
}
_PRIVACY = {
    "public_contains_per_joint_values": False,
    "private_plan_visibility": "private",
    "machine_gpu_session_identifiers_public": False,
    "aggregate_baseline_rmse_public": True,
}

FROZEN_CANDIDATE_HYPOTHESIS = (
    "In the pinned kit-less OVPhysX stack with explicit IdealPD control, "
    "changing only the live composed USD legacy physxJoint:jointFriction "
    "opinions from exact R1 values to literal positive float32 0.0 produces "
    "a bilateral, dt-stable small_step trajectory change."
)
FROZEN_ALLOWED_CLAIM = (
    "Report only whether the preregistered native OVPhysX input-sensitivity "
    "pattern is SUPPORTED, NOT_SUPPORTED, or INCONCLUSIVE after every "
    "validity gate passes; validity failure receives no scientific label, "
    "and no cross-engine parameter equivalence, upstream bug, hardware, "
    "Sim2Real, or formal Gate 0 rewrite claim is made."
)
FROZEN_MAXIMUM_GPU_HOURS = 0.5


def build_public_protocol(
    private_plan: object,
    *,
    candidate_hypothesis: str,
    allowed_claim: str,
    maximum_gpu_hours: float,
) -> dict[str, object]:
    private = validate_private_plan(private_plan)
    hypothesis = _string(candidate_hypothesis, "candidate_hypothesis")
    claim = _string(allowed_claim, "allowed_claim")
    gpu_hours = _number(maximum_gpu_hours, "maximum_gpu_hours", positive=True)
    _literal(
        hypothesis,
        FROZEN_CANDIDATE_HYPOTHESIS,
        "candidate_hypothesis",
    )
    _literal(claim, FROZEN_ALLOWED_CLAIM, "allowed_claim")
    _literal(gpu_hours, FROZEN_MAXIMUM_GPU_HOURS, "maximum_gpu_hours")
    protocol = {
        "schema_version": 1,
        "protocol_id": private["protocol_id"],
        "state": "PREREGISTERED_BEFORE_ANY_FREEZE_B_TRAJECTORY",
        "private_plan_sha256": private_plan_sha256(private),
        "inputs": _json_copy(private["inputs"]),
        "semantic_classification_per_field_family": {
            family: "UNMAPPABLE" for family in _SEMANTIC_FAMILIES
        },
        "candidate_hypothesis": hypothesis,
        "intervention": _json_copy(_INTERVENTION),
        "control_and_force_law": _json_copy(_CONTROL_LAW),
        "scenario": _json_copy(_SCENARIO),
        "run_matrix": _json_copy(_RUN_MATRIX),
        "controls": list(_CONTROLS),
        "metrics": _json_copy(_METRICS),
        "decision_thresholds": _json_copy(_DECISION),
        "validity_thresholds": _json_copy(_VALIDITY),
        "bounded_compute_budget": {
            "maximum_ovphysx_cases": 16,
            "maximum_mujoco_cases": 8,
            "maximum_gpu_hours": gpu_hours,
            "manual_value_tuning_allowed": False,
        },
        "privacy": _json_copy(_PRIVACY),
        "claim_boundary": {
            "allowed": claim,
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
            "forbidden": [
                "cross_engine_parameter_equivalence",
                "official_bug_claim",
                "hardware_or_Sim2Real_claim",
                "formal_Gate_0_rewrite",
            ],
        },
    }
    return validate_public_protocol(protocol)


def _literal_object(value: object, expected: Mapping[str, object], path: str) -> None:
    data = _mapping(value, path)
    _exact(data, set(expected), path)
    for key, expected_value in expected.items():
        _literal(data[key], expected_value, f"{path}.{key}")


def validate_public_protocol(value: object) -> dict[str, object]:
    data = _mapping(value, "public_protocol")
    _exact(data, _PUBLIC_KEYS, "public_protocol")
    _literal(data["schema_version"], 1, "public_protocol.schema_version")
    protocol_id = _string(data["protocol_id"], "public_protocol.protocol_id")
    if not _IDENTIFIER_RE.fullmatch(protocol_id):
        raise SensitivityValidationError("public_protocol.protocol_id is invalid")
    _literal(
        data["state"],
        "PREREGISTERED_BEFORE_ANY_FREEZE_B_TRAJECTORY",
        "public_protocol.state",
    )
    _digest(data["private_plan_sha256"], "public_protocol.private_plan_sha256")
    _validate_inputs(data["inputs"], "public_protocol.inputs")
    _semantic_map(
        data["semantic_classification_per_field_family"],
        "public_protocol.semantic_classification_per_field_family",
    )
    _literal(
        data["candidate_hypothesis"],
        FROZEN_CANDIDATE_HYPOTHESIS,
        "public_protocol.candidate_hypothesis",
    )
    _literal_object(data["intervention"], _INTERVENTION, "public_protocol.intervention")
    _literal_object(
        data["control_and_force_law"],
        _CONTROL_LAW,
        "public_protocol.control_and_force_law",
    )
    _literal_object(data["scenario"], _SCENARIO, "public_protocol.scenario")
    _literal_object(data["run_matrix"], _RUN_MATRIX, "public_protocol.run_matrix")
    _literal(data["controls"], _CONTROLS, "public_protocol.controls")
    _literal_object(data["metrics"], _METRICS, "public_protocol.metrics")
    _literal_object(
        data["decision_thresholds"],
        _DECISION,
        "public_protocol.decision_thresholds",
    )
    _literal_object(
        data["validity_thresholds"],
        _VALIDITY,
        "public_protocol.validity_thresholds",
    )
    budget = _mapping(data["bounded_compute_budget"], "public_protocol.bounded_compute_budget")
    _exact(
        budget,
        {
            "maximum_ovphysx_cases",
            "maximum_mujoco_cases",
            "maximum_gpu_hours",
            "manual_value_tuning_allowed",
        },
        "public_protocol.bounded_compute_budget",
    )
    _literal(budget["maximum_ovphysx_cases"], 16, "bounded_compute_budget.maximum_ovphysx_cases")
    _literal(budget["maximum_mujoco_cases"], 8, "bounded_compute_budget.maximum_mujoco_cases")
    _literal(
        budget["maximum_gpu_hours"],
        FROZEN_MAXIMUM_GPU_HOURS,
        "bounded_compute_budget.maximum_gpu_hours",
    )
    _literal(budget["manual_value_tuning_allowed"], False, "bounded_compute_budget.manual_value_tuning_allowed")
    _literal_object(data["privacy"], _PRIVACY, "public_protocol.privacy")
    claim = _mapping(data["claim_boundary"], "public_protocol.claim_boundary")
    _exact(
        claim,
        {
            "allowed",
            "formal_gate0_status_unchanged",
            "formal_gate0_pass_ready_unchanged",
            "forbidden",
        },
        "public_protocol.claim_boundary",
    )
    _literal(
        claim["allowed"],
        FROZEN_ALLOWED_CLAIM,
        "public_protocol.claim_boundary.allowed",
    )
    _literal(claim["formal_gate0_status_unchanged"], "DIVERGENT", "claim_boundary.formal_gate0_status_unchanged")
    _literal(claim["formal_gate0_pass_ready_unchanged"], False, "claim_boundary.formal_gate0_pass_ready_unchanged")
    _literal(
        claim["forbidden"],
        [
            "cross_engine_parameter_equivalence",
            "official_bug_claim",
            "hardware_or_Sim2Real_claim",
            "formal_Gate_0_rewrite",
        ],
        "claim_boundary.forbidden",
    )
    return _json_copy(data)


def load_public_protocol(path: str | Path) -> dict[str, object]:
    return validate_public_protocol(_strict_json(path, "public protocol"))


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    sample_count: int
    joint_element_count: int
    position_element_count: int
    joint_rmse_rad: float
    frame_position_rmse_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "joint_element_count": self.joint_element_count,
            "position_element_count": self.position_element_count,
            "joint_rmse_rad": self.joint_rmse_rad,
            "frame_position_rmse_m": self.frame_position_rmse_m,
        }


def _trace(
    run: CollectedRun,
    hand: HandSide,
    variant: TimestepVariant,
    backend: Simulator,
) -> tuple[TraceSample, ...]:
    if not isinstance(run, CollectedRun):
        raise SensitivityValidationError("trace must be a CollectedRun")
    case, result = run.case, run.result
    if case.hand is not hand or case.timestep_variant is not variant or case.simulator is not backend:
        raise SensitivityValidationError("trace case does not match its hand/dt/backend role")
    if case.scenario_id != "small_step" or result.scenario_id != "small_step":
        raise SensitivityValidationError("Freeze B metrics accept only small_step")
    if float(case.dt_s) != _DT[variant] or float(result.dt) != _DT[variant]:
        raise SensitivityValidationError("trace dt is not the frozen base/halved value")
    if result.backend != backend.value or not result.completed or result.status != "completed":
        raise SensitivityValidationError("trace backend or completion is invalid")
    expected_steps = round(0.5 / _DT[variant])
    if (
        result.requested_steps != expected_steps
        or result.completed_steps != expected_steps
        or len(result.samples) != expected_steps + 1
    ):
        raise SensitivityValidationError("trace sample inventory is incomplete")
    joints, frames = canonical_joint_names(hand), canonical_frame_names(hand)
    if tuple(result.joint_names) != joints or tuple(result.frame_names) != frames:
        raise SensitivityValidationError("trace result names/order are not canonical")
    contact_disclosure = result.provenance.get("contact_check_performed")
    if backend is Simulator.OVPHYSX and contact_disclosure is not False:
        raise SensitivityValidationError("OVPhysX must disclose contact_check_performed=false")
    previous = -math.inf
    for index, sample in enumerate(result.samples):
        if not isinstance(sample, TraceSample) or sample.step != index:
            raise SensitivityValidationError("trace steps are not canonical")
        time_s = _number(sample.time_s, f"trace.samples[{index}].time_s")
        if time_s <= previous or not math.isclose(
            time_s,
            index * _DT[variant],
            rel_tol=0.0,
            abs_tol=TIME_GRID_ABS_TOLERANCE_S,
        ):
            raise SensitivityValidationError("trace recorded time grid is invalid")
        previous = time_s
        if (
            set(sample.joint_positions) != set(joints)
            or set(sample.position_targets) != set(joints)
            or set(sample.frame_poses) != set(frames)
        ):
            raise SensitivityValidationError(
                "sample mappings do not have exact canonical coverage"
            )
        if backend is Simulator.MUJOCO and sample.contact_count != 0:
            raise SensitivityValidationError("MuJoCo trace must report contact_count=0")
        if backend is Simulator.OVPHYSX and sample.contact_count is not None:
            raise SensitivityValidationError("OVPhysX contact_count must remain null/unobserved")
        if len(sample.qpos) != 22 or len(sample.qvel) != 22:
            raise SensitivityValidationError(
                "trace qpos/qvel must each contain exactly 22 values"
            )
        for dof_index, value in enumerate(sample.qpos):
            _number(value, f"sample.qpos[{dof_index}]")
        for dof_index, value in enumerate(sample.qvel):
            velocity = _number(value, f"sample.qvel[{dof_index}]")
            if abs(velocity) > MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S:
                raise SensitivityValidationError(
                    "trace exceeds the backend joint-velocity sanity bound"
                )
        for name in joints:
            position = _number(
                sample.joint_positions[name], f"sample.joint_positions.{name}"
            )
            target = _number(
                sample.position_targets[name], f"sample.position_targets.{name}"
            )
            if abs(position) > MAX_ABS_CANONICAL_JOINT_POSITION_RAD:
                raise SensitivityValidationError(
                    "trace exceeds the canonical joint-position sanity bound"
                )
            if abs(target) > MAX_ABS_CANONICAL_POSITION_TARGET_RAD:
                raise SensitivityValidationError(
                    "trace exceeds the canonical position-target sanity bound"
                )
        for name in frames:
            pose = sample.frame_poses[name]
            if not isinstance(pose, tuple) or len(pose) != 7:
                raise SensitivityValidationError("frame pose must be a 7-tuple")
            values = tuple(_number(item, f"sample.frame_poses.{name}") for item in pose)
            origin_distance = math.sqrt(math.fsum(item * item for item in values[:3]))
            if origin_distance > MAX_FRAME_ORIGIN_DISTANCE_M:
                raise SensitivityValidationError(
                    "trace exceeds the frame-origin-distance sanity bound"
                )
            norm = math.sqrt(math.fsum(item * item for item in values[3:]))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
                raise SensitivityValidationError("frame quaternion is not unit length")
    return result.samples


def _window_pair(
    reference: CollectedRun,
    candidate: CollectedRun,
    *,
    hand: HandSide,
    variant: TimestepVariant,
    reference_backend: Simulator,
    candidate_backend: Simulator,
) -> WindowMetrics:
    first = _trace(reference, hand, variant, reference_backend)
    second = _trace(candidate, hand, variant, candidate_backend)
    if len(first) != len(second) or any(
        left.step != right.step
        or not math.isclose(
            left.time_s,
            right.time_s,
            rel_tol=0.0,
            abs_tol=TIME_GRID_ABS_TOLERANCE_S,
        )
        for left, right in zip(first, second, strict=True)
    ):
        raise SensitivityValidationError(
            "metric traces lack a compatible recorded time grid"
        )
    indices = [
        index
        for index, sample in enumerate(first)
        if sample.time_s >= WINDOW_START_S and sample.time_s <= WINDOW_END_S
    ]
    if not indices:
        raise SensitivityValidationError("inclusive 0.1-0.5 window is empty")
    joints, frames = canonical_joint_names(hand), canonical_frame_names(hand)
    q_squared: list[float] = []
    p_squared: list[float] = []
    for index in indices:
        for name in joints:
            delta = float(first[index].joint_positions[name]) - float(
                second[index].joint_positions[name]
            )
            q_squared.append(delta * delta)
        for name in frames:
            for axis in range(3):
                delta = float(first[index].frame_poses[name][axis]) - float(
                    second[index].frame_poses[name][axis]
                )
                p_squared.append(delta * delta)
    return WindowMetrics(
        sample_count=len(indices),
        joint_element_count=len(q_squared),
        position_element_count=len(p_squared),
        joint_rmse_rad=math.sqrt(math.fsum(q_squared) / len(q_squared)),
        frame_position_rmse_m=math.sqrt(math.fsum(p_squared) / len(p_squared)),
    )


@dataclass(frozen=True, slots=True)
class SensitivityTraceSet:
    hand: HandSide
    timestep_variant: TimestepVariant
    formal_mujoco: CollectedRun
    formal_ovphysx: CollectedRun
    sham_ovphysx: CollectedRun
    zero_ovphysx: CollectedRun

    def __post_init__(self) -> None:
        if not isinstance(self.hand, HandSide) or not isinstance(
            self.timestep_variant, TimestepVariant
        ):
            raise SensitivityValidationError(
                "SensitivityTraceSet hand/timestep_variant must use frozen enums"
            )
        expected = (
            ("formal_mujoco", self.formal_mujoco, Simulator.MUJOCO),
            ("formal_ovphysx", self.formal_ovphysx, Simulator.OVPHYSX),
            ("sham_ovphysx", self.sham_ovphysx, Simulator.OVPHYSX),
            ("zero_ovphysx", self.zero_ovphysx, Simulator.OVPHYSX),
        )
        for slot, run, backend in expected:
            if not isinstance(run, CollectedRun):
                raise SensitivityValidationError(f"{slot} must be a CollectedRun")
            case = run.case
            if (
                case.simulator is not backend
                or case.hand is not self.hand
                or case.scenario_id != "small_step"
                or case.timestep_variant is not self.timestep_variant
                or case.repeat_index != 1
                or not math.isclose(
                    float(case.dt_s),
                    _DT[self.timestep_variant],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise SensitivityValidationError(
                    f"{slot} does not match the canonical repeat-1 Freeze B case"
                )


@dataclass(frozen=True, slots=True)
class PrimaryMetricCell:
    hand: HandSide
    timestep_variant: TimestepVariant
    crosssim_baseline: WindowMetrics
    treatment_vs_sham: WindowMetrics
    zero_vs_mujoco: WindowMetrics
    S_Dq: float
    S_Dp: float
    R_Dq: float
    R_Dp: float

    def to_dict(self) -> dict[str, object]:
        return {
            "hand": self.hand.value,
            "timestep_variant": self.timestep_variant.value,
            "crosssim_baseline": self.crosssim_baseline.to_dict(),
            "treatment_vs_sham": self.treatment_vs_sham.to_dict(),
            "zero_vs_mujoco": self.zero_vs_mujoco.to_dict(),
            "S_Dq": self.S_Dq,
            "S_Dp": self.S_Dp,
            "R_Dq": self.R_Dq,
            "R_Dp": self.R_Dp,
        }


@dataclass(frozen=True, slots=True)
class FreezeBDecision:
    """Execution disposition plus an optional scientific label.

    ``INVALID`` is an execution-validity status only.  It is deliberately not
    exposed as a scientific label.
    """

    status: SensitivityStatus
    cells: tuple[PrimaryMetricCell, ...] = ()
    validity_failures: tuple[str, ...] = ()

    @property
    def s_values(self) -> tuple[float, ...]:
        return tuple(value for cell in self.cells for value in (cell.S_Dq, cell.S_Dp))

    @property
    def scientific_label(self) -> SensitivityStatus | None:
        if self.status is SensitivityStatus.INVALID:
            return None
        return self.status

    @property
    def execution_valid(self) -> bool:
        return self.status is not SensitivityStatus.INVALID

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "validation_status": "VALID" if self.execution_valid else "INVALID",
            "scientific_label": (
                self.scientific_label.value
                if self.scientific_label is not None
                else None
            ),
            "cells": [cell.to_dict() for cell in self.cells],
            "s_values": list(self.s_values),
            "validity_failures": list(self.validity_failures),
        }


def _validated_study_bindings(
    *,
    frozen_inputs: object,
    private_plan: object,
    public_protocol: object,
    evidence_bindings: object,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Cross-bind public/private protocols, source identity, and evidence roots."""

    inputs = _validate_inputs(frozen_inputs, "frozen_inputs")
    plan = validate_private_plan(private_plan)
    public = validate_public_protocol(public_protocol)
    bindings_raw = _mapping(evidence_bindings, "evidence_bindings")
    _exact(bindings_raw, _EVIDENCE_BINDING_KEYS, "evidence_bindings")
    bindings = _json_copy(bindings_raw)
    plan_digest = private_plan_sha256(plan)
    public_canonical_digest = canonical_json_sha256(public)
    expected_binding_values = {
        "private_plan_sha256": plan_digest,
        "public_protocol_canonical_sha256": public_canonical_digest,
    }
    for key, expected in expected_binding_values.items():
        observed = _digest(bindings[key], f"evidence_bindings.{key}")
        if observed != expected:
            raise SensitivityValidationError(f"evidence_bindings.{key} mismatch")
    for key in (
        "public_protocol_file_sha256",
        "deployment_source_snapshot_sha256",
    ):
        _digest(bindings[key], f"evidence_bindings.{key}")
    for key in ("deployment_source_revision", "deployment_source_tree"):
        _git_oid(bindings[key], f"evidence_bindings.{key}")
    if plan["inputs"] != inputs or public["inputs"] != inputs:
        raise SensitivityValidationError(
            "frozen inputs differ across private plan/public protocol/metric call"
        )
    if plan["protocol_id"] != public["protocol_id"]:
        raise SensitivityValidationError("private/public protocol_id mismatch")
    if public["private_plan_sha256"] != plan_digest:
        raise SensitivityValidationError("public protocol does not pin the private plan")
    return inputs, plan, public, bindings


def _exact_provenance_subset(
    provenance: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    path: str,
) -> None:
    mismatches = [
        key
        for key, value in expected.items()
        if type(provenance.get(key)) is not type(value)
        or provenance.get(key) != value
    ]
    if mismatches:
        raise SensitivityValidationError(
            f"{path} provenance identity mismatch: {', '.join(sorted(mismatches))}"
        )


def _validate_metric_trace_identities(
    trace_sets: Sequence[SensitivityTraceSet],
    *,
    inputs: Mapping[str, object],
    plan: Mapping[str, object],
    public: Mapping[str, object],
    bindings: Mapping[str, object],
) -> None:
    """Bind every metric input to the frozen case matrix and immutable evidence."""

    formal_root = str(inputs["formal_gate0_bundle_root_sha256"])
    formal_common = {
        "source_revision": inputs["formal_gate0_source_revision"],
        "manifest_sha256": inputs["gate0_manifest_semantic_sha256"],
        "asset_commit": inputs["asset_commit"],
        "asset_git_tree": inputs["asset_git_tree"],
        "asset_tree_sha256": inputs["canonical_lf_asset_tree_sha256"],
    }
    remote_common = {
        "freeze_b_protocol_id": plan["protocol_id"],
        "private_plan_sha256": bindings["private_plan_sha256"],
        "public_protocol_file_sha256": bindings["public_protocol_file_sha256"],
        "public_protocol_canonical_sha256": bindings[
            "public_protocol_canonical_sha256"
        ],
        "manifest_sha256": inputs["gate0_manifest_semantic_sha256"],
        "manifest_file_sha256": inputs["gate0_manifest_file_sha256"],
        "asset_commit": inputs["asset_commit"],
        "asset_git_tree": inputs["asset_git_tree"],
        "asset_tree_sha256": inputs["canonical_lf_asset_tree_sha256"],
        "source_revision": bindings["deployment_source_revision"],
        "source_tree": bindings["deployment_source_tree"],
        "deployment_source_revision": bindings["deployment_source_revision"],
        "deployment_source_tree": bindings["deployment_source_tree"],
        "implementation_source_revision": inputs["freeze_b_source_revision"],
        "implementation_source_tree": inputs["freeze_b_source_tree"],
        "source_snapshot_sha256": bindings[
            "deployment_source_snapshot_sha256"
        ],
    }
    if public["private_plan_sha256"] != bindings["private_plan_sha256"]:
        raise SensitivityValidationError("trace bindings do not match public protocol")

    for traces in trace_sets:
        hand = traces.hand.value
        variant = traces.timestep_variant.value
        canonical_mujoco = f"mujoco.{hand}.small_step.{variant}.r01"
        canonical_ovphysx = f"ovphysx.{hand}.small_step.{variant}.r01"
        formal_runs = (
            ("formal_mujoco", traces.formal_mujoco, canonical_mujoco),
            ("formal_ovphysx", traces.formal_ovphysx, canonical_ovphysx),
        )
        for slot, run, canonical_id in formal_runs:
            if run.case.case_id != canonical_id:
                raise SensitivityValidationError(
                    f"{slot} canonical case identity mismatch"
                )
            if run.bundle_root_sha256 != formal_root:
                raise SensitivityValidationError(
                    f"{slot} is not bound to the frozen formal bundle root"
                )
            _exact_provenance_subset(
                run.result.provenance,
                formal_common,
                path=slot,
            )

        remote_runs = (
            ("sham_ovphysx", traces.sham_ovphysx, "sham"),
            ("zero_ovphysx", traces.zero_ovphysx, "zero"),
        )
        for slot, run, role in remote_runs:
            experiment_id = (
                f"freeze_b.ovphysx.{hand}.small_step.{variant}.{role}.r01"
            )
            expected = {
                **remote_common,
                "experiment_case_id": experiment_id,
                "canonical_case_id": canonical_ovphysx,
                "freeze_b_role": role,
                "hand": hand,
                "timestep_variant": variant,
                "repeat_index": 1,
            }
            if run.case.case_id != canonical_ovphysx:
                raise SensitivityValidationError(
                    f"{slot} canonical case identity mismatch"
                )
            if run.bundle_root_sha256 is not None:
                raise SensitivityValidationError(
                    f"{slot} must be a raw pre-finalization case, not a foreign bundle"
                )
            _exact_provenance_subset(run.result.provenance, expected, path=slot)


def compute_primary_metrics(
    trace_sets: Sequence[SensitivityTraceSet],
    *,
    frozen_inputs: object,
    private_plan: object,
    public_protocol: object,
    evidence_bindings: object,
) -> tuple[PrimaryMetricCell, ...]:
    """Compute 4 cells/8 S values only from fully identity-bound evidence."""

    axes = tuple((hand, variant) for hand in _HANDS for variant in _VARIANTS)
    if len(trace_sets) != 4:
        raise SensitivityValidationError("exactly four hand/dt trace sets are required")
    for index, ((hand, variant), traces) in enumerate(zip(axes, trace_sets)):
        if (
            not isinstance(traces, SensitivityTraceSet)
            or traces.hand is not hand
            or traces.timestep_variant is not variant
        ):
            raise SensitivityValidationError(
                f"trace_sets[{index}] is not canonical {hand.value}.{variant.value}"
            )
    inputs, plan, public, bindings = _validated_study_bindings(
        frozen_inputs=frozen_inputs,
        private_plan=private_plan,
        public_protocol=public_protocol,
        evidence_bindings=evidence_bindings,
    )
    _validate_metric_trace_identities(
        trace_sets,
        inputs=inputs,
        plan=plan,
        public=public,
        bindings=bindings,
    )
    cells: list[PrimaryMetricCell] = []
    for (hand, variant), traces in zip(axes, trace_sets):
        baseline = _window_pair(
            traces.formal_mujoco,
            traces.formal_ovphysx,
            hand=hand,
            variant=variant,
            reference_backend=Simulator.MUJOCO,
            candidate_backend=Simulator.OVPHYSX,
        )
        effect = _window_pair(
            traces.sham_ovphysx,
            traces.zero_ovphysx,
            hand=hand,
            variant=variant,
            reference_backend=Simulator.OVPHYSX,
            candidate_backend=Simulator.OVPHYSX,
        )
        zero_vs_mujoco = _window_pair(
            traces.formal_mujoco,
            traces.zero_ovphysx,
            hand=hand,
            variant=variant,
            reference_backend=Simulator.MUJOCO,
            candidate_backend=Simulator.OVPHYSX,
        )
        if baseline.joint_rmse_rad <= 0.0 or baseline.frame_position_rmse_m <= 0.0:
            raise SensitivityValidationError("crosssim baselines must be positive")
        frozen = inputs["formal_crosssim_window_rmse"][hand.value][variant.value]
        for observed, expected, metric in (
            (
                baseline.joint_rmse_rad,
                float(frozen["joint_rmse_rad"]),
                "joint_rmse_rad",
            ),
            (
                baseline.frame_position_rmse_m,
                float(frozen["frame_position_rmse_m"]),
                "frame_position_rmse_m",
            ),
        ):
            if not math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=BASELINE_RECHECK_ABS_TOLERANCE,
            ):
                raise SensitivityValidationError(
                    f"frozen baseline mismatch: {hand.value}.{variant.value}.{metric}"
                )
        cells.append(
            PrimaryMetricCell(
                hand=hand,
                timestep_variant=variant,
                crosssim_baseline=baseline,
                treatment_vs_sham=effect,
                zero_vs_mujoco=zero_vs_mujoco,
                S_Dq=effect.joint_rmse_rad / baseline.joint_rmse_rad,
                S_Dp=effect.frame_position_rmse_m / baseline.frame_position_rmse_m,
                R_Dq=(
                    1.0
                    - zero_vs_mujoco.joint_rmse_rad
                    / baseline.joint_rmse_rad
                ),
                R_Dp=(
                    1.0
                    - zero_vs_mujoco.frame_position_rmse_m
                    / baseline.frame_position_rmse_m
                ),
            )
        )
    return tuple(cells)


def decide_freeze_b(
    metrics: Sequence[PrimaryMetricCell],
    *,
    validity_checks: Mapping[str, object] | None = None,
) -> FreezeBDecision:
    """Apply the all-eight rule only after every frozen validity gate passes."""

    failures = _validity_check_failures(validity_checks)
    if failures:
        return FreezeBDecision(
            SensitivityStatus.INVALID,
            validity_failures=failures,
        )
    if len(metrics) != 4 or not all(
        isinstance(item, PrimaryMetricCell) for item in metrics
    ):
        return FreezeBDecision(
            SensitivityStatus.INVALID,
            validity_failures=("expected_exactly_four_primary_metric_cells",),
        )
    cells = tuple(metrics)
    expected_axes = tuple((hand, variant) for hand in _HANDS for variant in _VARIANTS)
    observed_axes = tuple(
        (cell.hand, cell.timestep_variant) for cell in cells
    )
    if observed_axes != expected_axes:
        return FreezeBDecision(
            SensitivityStatus.INVALID,
            validity_failures=("primary_metric_cells_are_not_in_canonical_order",),
        )
    values: tuple[object, ...] = tuple(
        value for cell in cells for value in (cell.S_Dq, cell.S_Dp)
    )
    converted: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            return FreezeBDecision(
                SensitivityStatus.INVALID,
                validity_failures=(
                    "S_values_must_be_finite_nonnegative_numbers",
                ),
            )
        converted.append(float(value))
    if all(value >= SUPPORTED_MIN_S for value in converted):
        status = SensitivityStatus.SUPPORTED
    elif all(value <= NOT_SUPPORTED_MAX_S for value in converted):
        status = SensitivityStatus.NOT_SUPPORTED
    else:
        status = SensitivityStatus.INCONCLUSIVE
    return FreezeBDecision(status, cells)


def _validity_check_failures(
    validity_checks: Mapping[str, object] | None,
) -> tuple[str, ...]:
    if not isinstance(validity_checks, Mapping) or any(
        not isinstance(key, str) for key in validity_checks
    ):
        return ("validity_checks_not_an_object",)
    observed = set(validity_checks)
    failures = [
        *(
            f"missing_validity_check:{key}"
            for key in sorted(REQUIRED_VALIDITY_CHECKS - observed)
        ),
        *(
            f"unknown_validity_check:{key}"
            for key in sorted(observed - REQUIRED_VALIDITY_CHECKS)
        ),
        *(
            f"validity_check_not_true:{key}"
            for key in sorted(REQUIRED_VALIDITY_CHECKS & observed)
            if validity_checks[key] is not True
        ),
    ]
    return tuple(failures)


def evaluate_freeze_b(
    trace_sets: Sequence[SensitivityTraceSet],
    *,
    frozen_inputs: object,
    private_plan: object | None = None,
    public_protocol: object | None = None,
    evidence_bindings: object | None = None,
    validity_checks: Mapping[str, object] | None = None,
) -> FreezeBDecision:
    """Fail-closed convenience entrypoint from raw traces to a final decision."""

    failures = _validity_check_failures(validity_checks)
    if failures:
        return FreezeBDecision(
            SensitivityStatus.INVALID,
            validity_failures=failures,
        )
    try:
        metrics = compute_primary_metrics(
            trace_sets,
            frozen_inputs=frozen_inputs,
            private_plan=private_plan,
            public_protocol=public_protocol,
            evidence_bindings=evidence_bindings,
        )
    except (SensitivityValidationError, TypeError, ValueError) as exc:
        return FreezeBDecision(
            SensitivityStatus.INVALID,
            validity_failures=(str(exc),),
        )
    return decide_freeze_b(metrics, validity_checks=validity_checks)


__all__ = [
    "BASELINE_RECHECK_ABS_TOLERANCE",
    "EXPECTED_S_COUNT",
    "FreezeBDecision",
    "NOT_SUPPORTED_MAX_S",
    "PrimaryMetricCell",
    "REQUIRED_VALIDITY_CHECKS",
    "SUPPORTED_MIN_S",
    "TIME_GRID_ABS_TOLERANCE_S",
    "SensitivityStatus",
    "SensitivityTraceSet",
    "SensitivityValidationError",
    "WINDOW_END_S",
    "WINDOW_START_S",
    "WindowMetrics",
    "build_public_protocol",
    "canonical_frame_names",
    "canonical_json_sha256",
    "canonical_joint_names",
    "compute_primary_metrics",
    "decide_freeze_b",
    "evaluate_freeze_b",
    "expected_case_matrix",
    "expected_pre_float32_sha256",
    "expected_run_order",
    "experiment_case",
    "load_private_plan",
    "load_public_protocol",
    "private_plan_sha256",
    "validate_private_plan",
    "validate_public_protocol",
]
