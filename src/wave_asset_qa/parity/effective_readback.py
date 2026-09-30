"""Fail-closed contracts for the descriptive OVPhysX parameter readback.

This module is deliberately CPU-safe.  The remote worker owns all imports of
the kit-less OVPhysX runtime and passes the adapter's detached JSON snapshot
into :func:`build_readback_instance`.  Local finalization can therefore audit,
compare, sanitize, and report the four fresh-process records without loading a
GPU backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from .contracts import HandSide, ParityManifest
from .scenarios import load_manifest, manifest_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class EffectiveReadbackError(RuntimeError):
    """Raised when Freeze A evidence is incomplete, inconsistent, or unsafe."""


def _reject_constant(token: str) -> object:
    raise EffectiveReadbackError(f"JSON contains forbidden non-finite token: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EffectiveReadbackError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path, *, label: str) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate keys."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except EffectiveReadbackError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EffectiveReadbackError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EffectiveReadbackError(f"{label} must be a JSON object")
    _validate_finite_json(value, label)
    return value


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EffectiveReadbackError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash the exact normalized encoding frozen by the Freeze A contract."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EffectiveReadbackError("value is not finite canonical JSON") from exc
    return sha256(encoded).hexdigest()


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EffectiveReadbackError(f"{path} must be an object with string keys")
    return value


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise EffectiveReadbackError(f"{path} must be an array")
    return value


def _require_keys(value: Mapping[str, Any], required: Sequence[str], path: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise EffectiveReadbackError(f"{path} missing required keys: {missing}")


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing or unknown:
        raise EffectiveReadbackError(
            f"{path} has invalid keys: missing={missing}, unknown={unknown}"
        )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EffectiveReadbackError(f"{path} must be a non-empty string")
    return value


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EffectiveReadbackError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EffectiveReadbackError(f"{path} must be finite")
    return number


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EffectiveReadbackError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise EffectiveReadbackError(f"{path} must be at least {minimum}")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EffectiveReadbackError(f"{path} must be a lowercase SHA-256")
    return value


def _hex40(value: object, path: str) -> str:
    if not isinstance(value, str) or _HEX40_RE.fullmatch(value) is None:
        raise EffectiveReadbackError(f"{path} must be a lowercase 40-digit Git id")
    return value


def _validate_finite_json(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EffectiveReadbackError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EffectiveReadbackError(f"{path} contains a non-string object key")
        for key, child in value.items():
            _validate_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{path}[{index}]")


def _json_copy(value: object) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise EffectiveReadbackError("readback contains non-JSON or non-finite data") from exc


def load_frozen_contract(
    freeze_config_path: str | Path,
    gate0_manifest_path: str | Path,
) -> tuple[dict[str, Any], ParityManifest]:
    """Load and cross-check Freeze A against the canonical Gate 0 manifest."""

    freeze_path = Path(freeze_config_path)
    manifest_path = Path(gate0_manifest_path)
    plan = load_json_strict(freeze_path, label="Freeze A config")
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise EffectiveReadbackError(f"cannot load Gate 0 manifest: {exc}") from exc

    if plan.get("schema_version") != 1:
        raise EffectiveReadbackError("Freeze A schema_version must be 1")
    if plan.get("classification") != (
        "descriptive_effective_parameter_readback_not_formal_gate0"
    ):
        raise EffectiveReadbackError("Freeze A classification changed")
    if plan.get("plan_stage") != (
        "freeze_a_before_any_new_ovphysx_parameter_value_is_observed"
    ):
        raise EffectiveReadbackError("Freeze A plan stage changed")

    frozen = _mapping(plan.get("frozen_context"), "Freeze A frozen_context")
    expected = {
        "gate0_manifest_file_sha256": sha256_file(manifest_path),
        "gate0_manifest_semantic_sha256": manifest_sha256(manifest),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "canonical_lf_asset_tree_sha256": (
            manifest.provenance.canonical_lf_asset_tree_sha256
        ),
    }
    for key, observed in expected.items():
        if frozen.get(key) != observed:
            raise EffectiveReadbackError(
                f"Freeze A frozen_context does not match Gate 0: {key}"
            )

    rows = _array(plan.get("expected_instances"), "Freeze A expected_instances")
    observed_rows: list[tuple[str, int, str]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"Freeze A expected_instances[{index}]")
        observed_rows.append(
            (
                _nonempty_string(row.get("side"), f"expected_instances[{index}].side"),
                _integer(
                    row.get("instance_index"),
                    f"expected_instances[{index}].instance_index",
                    minimum=1,
                ),
                _nonempty_string(
                    row.get("case_id"), f"expected_instances[{index}].case_id"
                ),
            )
        )
    required_rows = [
        ("left", 1, "ovphysx.left.effective_readback.r01"),
        ("left", 2, "ovphysx.left.effective_readback.r02"),
        ("right", 1, "ovphysx.right.effective_readback.r01"),
        ("right", 2, "ovphysx.right.effective_readback.r02"),
    ]
    if observed_rows != required_rows:
        raise EffectiveReadbackError("Freeze A fresh-process matrix changed")

    execution = _mapping(plan.get("execution_contract"), "execution_contract")
    required_execution = {
        "fresh_instances_per_hand": 2,
        "total_instance_count": 4,
        "fresh_process_per_instance": True,
        "reuse_of_articulation_or_physics_scene_forbidden": True,
        "user_or_experimental_trace_dynamics_advance_allowed": False,
        "required_user_trace_simulation_step_call_count": 0,
        "required_trajectory_sample_count": 0,
        "required_trace_time_s": 0.0,
        "required_initialization_state_write_performed": True,
        "experimental_control_command_allowed": False,
        "experimental_target_write_allowed": False,
        "required_experimental_target_write_performed": False,
        "renderer_allowed": False,
        "camera_allowed": False,
        "private_raw_evidence_required": True,
    }
    for key, value in required_execution.items():
        if execution.get(key) != value:
            raise EffectiveReadbackError(f"Freeze A execution contract changed: {key}")
    if tuple(hand.side.value for hand in manifest.hands) != ("left", "right"):
        raise EffectiveReadbackError("Gate 0 must contain canonical left and right hands")
    if any(len(hand.joint_names) != 22 for hand in manifest.hands):
        raise EffectiveReadbackError("Gate 0 must contain 22 joints per hand")
    return plan, manifest


def expected_instance(
    plan: Mapping[str, Any], *, side: str, instance_index: int
) -> Mapping[str, Any]:
    matches = [
        _mapping(row, "expected instance")
        for row in _array(plan.get("expected_instances"), "expected_instances")
        if isinstance(row, Mapping)
        and row.get("side") == side
        and row.get("instance_index") == instance_index
    ]
    if len(matches) != 1:
        raise EffectiveReadbackError(
            f"Freeze A has no unique expected instance for {side} r{instance_index:02d}"
        )
    return matches[0]


def _available_evidence(
    value: object,
    *,
    unit: object,
    semantics: str,
    source_locator: str,
    authored_value: object | None = None,
    resolved_value: object | None = None,
) -> dict[str, Any]:
    return {
        "availability": "available",
        "authored_value": _json_copy(value if authored_value is None else authored_value),
        "resolved_value": _json_copy(value if resolved_value is None else resolved_value),
        "runtime_effective_value": _json_copy(value),
        "unavailable_layers": [],
        "unit": _json_copy(unit),
        "semantics": semantics,
        "source_locator": source_locator,
        "unavailable_reason": None,
    }


def _runtime_only_evidence(
    value: object, *, unit: object, semantics: str, source_locator: str
) -> dict[str, Any]:
    return {
        "availability": "partially_available",
        "authored_value": None,
        "resolved_value": None,
        "runtime_effective_value": _json_copy(value),
        "unavailable_layers": ["authored", "resolved"],
        "unit": _json_copy(unit),
        "semantics": semantics,
        "source_locator": source_locator,
        "unavailable_reason": (
            "This adapter contract captures an exposed runtime value but does not "
            "separately attribute authored and resolved values."
        ),
    }


def _layered_evidence(
    *,
    authored_value: object | None,
    resolved_value: object,
    runtime_effective_value: object,
    unit: object,
    semantics: str,
    source_locator: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Keep independently observed authored/resolved/runtime layers distinct."""

    if authored_value is None:
        availability = "partially_available"
        unavailable_layers = ["authored"]
        reason = unavailable_reason or "No authored value opinion is present."
    else:
        availability = "available"
        unavailable_layers = []
        reason = None
    return {
        "availability": availability,
        "authored_value": _json_copy(authored_value),
        "resolved_value": _json_copy(resolved_value),
        "runtime_effective_value": _json_copy(runtime_effective_value),
        "unavailable_layers": unavailable_layers,
        "unit": _json_copy(unit),
        "semantics": semantics,
        "source_locator": source_locator,
        "unavailable_reason": reason,
    }


def _unavailable_evidence(
    *, unit: object, semantics: str, source_locator: str, reason: str
) -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "authored_value": None,
        "resolved_value": None,
        "runtime_effective_value": None,
        "unavailable_layers": ["authored", "resolved", "runtime_effective"],
        "unit": _json_copy(unit),
        "semantics": semantics,
        "source_locator": source_locator,
        "unavailable_reason": reason,
    }


def _composed_input_evidence(
    value: object,
    *,
    authored: bool,
    unit: object,
    semantics: str,
    source_locator: str,
) -> dict[str, Any]:
    unavailable = ["runtime_effective"]
    authored_value: object | None = _json_copy(value)
    if not authored:
        unavailable.insert(0, "authored")
        authored_value = None
        reason = (
            "No authored value opinion is present; the resolved live composed-USD "
            "input is available, but the pinned Python API does not expose the "
            "compiled C++ solver state."
        )
    else:
        reason = (
            "The pinned Python API exposes this live composed-USD input, not the "
            "compiled C++ solver state."
        )
    return {
        "availability": "partially_available",
        "authored_value": authored_value,
        "resolved_value": _json_copy(value),
        "runtime_effective_value": None,
        "unavailable_layers": unavailable,
        "unit": _json_copy(unit),
        "semantics": semantics,
        "source_locator": source_locator,
        "unavailable_reason": reason,
    }


def _composed_solver_record(
    container: Mapping[str, Any], key: str, path: str
) -> dict[str, Any]:
    record = _mapping(container.get(key), f"{path}.{key}")
    _exact_keys(
        record,
        (
            "resolved_available",
            "resolved_value",
            "has_authored_value_opinion",
            "source_locator",
            "unavailable_reason",
        ),
        f"{path}.{key}",
    )
    available = record.get("resolved_available")
    if not isinstance(available, bool):
        raise EffectiveReadbackError(f"{path}.{key}.resolved_available must be boolean")
    value = record.get("resolved_value")
    authored = record.get("has_authored_value_opinion")
    locator = _nonempty_string(record.get("source_locator"), f"{path}.{key}.source_locator")
    reason = record.get("unavailable_reason")
    if available:
        if value is None or not isinstance(authored, bool) or reason is not None:
            raise EffectiveReadbackError(f"{path}.{key} has an invalid available record")
    else:
        if value is not None or authored is not None:
            raise EffectiveReadbackError(f"{path}.{key} unavailable record contains a value")
        reason = _nonempty_string(reason, f"{path}.{key}.unavailable_reason")
    return {
        "resolved_available": available,
        "resolved_value": _json_copy(value),
        "has_authored_value_opinion": authored,
        "source_locator": locator,
        "unavailable_reason": reason,
    }


def _unavailable_solver_family(
    records: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    unit: object,
    semantics: str,
) -> dict[str, Any] | None:
    missing = [
        (label, record)
        for label, record in records
        if record.get("resolved_available") is not True
    ]
    if not missing:
        return None
    locator = " | ".join(str(record["source_locator"]) for _, record in records)
    reason = "Required live composed USD inputs are unavailable: " + "; ".join(
        f"{label}: {record['unavailable_reason']}" for label, record in missing
    )
    return _unavailable_evidence(
        unit=unit,
        semantics=semantics,
        source_locator=locator,
        reason=reason,
    )


def _axis_vector(axis: object, path: str) -> list[float]:
    token = _nonempty_string(axis, path)
    vectors = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
    if token not in vectors:
        raise EffectiveReadbackError(f"{path} must be X, Y, or Z")
    return vectors[token]


def _solver_runtime(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    solver = _mapping(snapshot.get("physics_solver_contract"), "snapshot.physics_solver_contract")
    _exact_keys(
        solver,
        (
            "schema_version",
            "source",
            "compiled_runtime_readback",
            "capture_phase",
            "physics_scene_path",
            "articulation_root_path",
            "scene",
            "articulation",
            "derived_clamped_requested_iterations",
            "derived_values_are_compiled_runtime_readback",
            "stepping",
        ),
        "snapshot.physics_solver_contract",
    )
    if (
        solver.get("schema_version") != 2
        or solver.get("source") != "composed_usd_input"
        or solver.get("compiled_runtime_readback") is not False
        or solver.get("capture_phase") != "post_runtime_reset_pre_trace_step"
        or solver.get("derived_values_are_compiled_runtime_readback") is not False
    ):
        raise EffectiveReadbackError("adapter solver readback contract changed")
    _nonempty_string(solver.get("physics_scene_path"), "physics_scene_path")
    _nonempty_string(solver.get("articulation_root_path"), "articulation_root_path")
    scene = _mapping(solver.get("scene"), "physics_solver_contract.scene")
    articulation = _mapping(
        solver.get("articulation"), "physics_solver_contract.articulation"
    )
    derived = _mapping(
        solver.get("derived_clamped_requested_iterations"),
        "physics_solver_contract.derived_clamped_requested_iterations",
    )
    stepping = _mapping(solver.get("stepping"), "physics_solver_contract.stepping")
    scene_keys = (
        "solver_type",
        "enable_stabilization",
        "min_position_iteration_count",
        "max_position_iteration_count",
        "min_velocity_iteration_count",
        "max_velocity_iteration_count",
        "time_steps_per_second",
    )
    articulation_keys = (
        "solver_position_iteration_count",
        "solver_velocity_iteration_count",
        "sleep_threshold",
        "stabilization_threshold",
    )
    _exact_keys(scene, scene_keys, "physics_solver_contract.scene")
    _exact_keys(articulation, articulation_keys, "physics_solver_contract.articulation")
    _exact_keys(
        derived,
        ("position", "velocity"),
        "physics_solver_contract.derived_clamped_requested_iterations",
    )
    _exact_keys(
        stepping,
        (
            "requested_dt_s",
            "cfg_dt_s",
            "config_accessor_dt_s",
            "config_accessor_semantics",
            "backend_step_api",
            "backend_step_calls_per_adapter_step",
            "external_substeps",
            "internal_solver_substeps",
            "internal_solver_substeps_status",
            "physics_scene_time_steps_per_second_controls_effective_dt",
            "render_interval",
            "render_interval_is_physics_substeps",
            "gpu_warmup_outside_trace",
            "gpu_warmup_timestep_status",
        ),
        "physics_solver_contract.stepping",
    )
    scene_records = {
        key: _composed_solver_record(scene, key, "physics_solver_contract.scene")
        for key in scene_keys
    }
    articulation_records = {
        key: _composed_solver_record(
            articulation, key, "physics_solver_contract.articulation"
        )
        for key in articulation_keys
    }

    solver_type_record = scene_records["solver_type"]
    solver_type_evidence = _unavailable_solver_family(
        [("solver_type", solver_type_record)],
        unit="enum",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if solver_type_evidence is None:
        solver_type = _nonempty_string(
            solver_type_record["resolved_value"], "scene.solver_type"
        )
        if solver_type not in {"TGS", "PGS"}:
            raise EffectiveReadbackError("scene.solver_type must be TGS or PGS")
        solver_type_evidence = _composed_input_evidence(
            solver_type,
            authored=solver_type_record["has_authored_value_opinion"] is True,
            unit="enum",
            semantics="live_composed_usd_solver_input_not_compiled_solver_state",
            source_locator=str(solver_type_record["source_locator"]),
        )

    position_records = [
        ("requested", articulation_records["solver_position_iteration_count"]),
        ("scene_min", scene_records["min_position_iteration_count"]),
        ("scene_max", scene_records["max_position_iteration_count"]),
    ]
    position_evidence = _unavailable_solver_family(
        position_records,
        unit="count",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if position_evidence is None:
        requested_position = _integer(
            position_records[0][1]["resolved_value"],
            "articulation.solver_position_iteration_count",
            minimum=1,
        )
        min_position = _integer(
            position_records[1][1]["resolved_value"], "scene.min_position", minimum=1
        )
        max_position = _integer(
            position_records[2][1]["resolved_value"], "scene.max_position", minimum=1
        )
        expected_position = min(max(requested_position, min_position), max_position)
        clamped_position = _integer(derived.get("position"), "derived.position", minimum=1)
        if clamped_position != expected_position:
            raise EffectiveReadbackError("derived position iterations are inconsistent")
        position_evidence = _composed_input_evidence(
            {
                "requested": requested_position,
                "clamped_requested": clamped_position,
                "scene_min": min_position,
                "scene_max": max_position,
            },
            authored=all(
                record["has_authored_value_opinion"] is True
                for _, record in position_records
            ),
            unit="count",
            semantics="requested_and_scene_clamped_composed_inputs_not_compiled_solver_state",
            source_locator=" | ".join(
                str(record["source_locator"]) for _, record in position_records
            ),
        )
    elif derived.get("position") is not None:
        raise EffectiveReadbackError("unavailable position inputs produced a derived value")

    velocity_records = [
        ("requested", articulation_records["solver_velocity_iteration_count"]),
        ("scene_min", scene_records["min_velocity_iteration_count"]),
        ("scene_max", scene_records["max_velocity_iteration_count"]),
    ]
    velocity_evidence = _unavailable_solver_family(
        velocity_records,
        unit="count",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if velocity_evidence is None:
        requested_velocity = _integer(
            velocity_records[0][1]["resolved_value"],
            "articulation.solver_velocity_iteration_count",
            minimum=0,
        )
        min_velocity = _integer(
            velocity_records[1][1]["resolved_value"], "scene.min_velocity", minimum=0
        )
        max_velocity = _integer(
            velocity_records[2][1]["resolved_value"], "scene.max_velocity", minimum=0
        )
        expected_velocity = min(max(requested_velocity, min_velocity), max_velocity)
        clamped_velocity = _integer(derived.get("velocity"), "derived.velocity", minimum=0)
        if clamped_velocity != expected_velocity:
            raise EffectiveReadbackError("derived velocity iterations are inconsistent")
        velocity_evidence = _composed_input_evidence(
            {
                "requested": requested_velocity,
                "clamped_requested": clamped_velocity,
                "scene_min": min_velocity,
                "scene_max": max_velocity,
            },
            authored=all(
                record["has_authored_value_opinion"] is True
                for _, record in velocity_records
            ),
            unit="count",
            semantics="requested_and_scene_clamped_composed_inputs_not_compiled_solver_state",
            source_locator=" | ".join(
                str(record["source_locator"]) for _, record in velocity_records
            ),
        )
    elif derived.get("velocity") is not None:
        raise EffectiveReadbackError("unavailable velocity inputs produced a derived value")

    config_accessor_dt = _finite_number(
        stepping.get("config_accessor_dt_s"), "stepping.config_accessor_dt_s"
    )
    cfg_dt = _finite_number(stepping.get("cfg_dt_s"), "stepping.cfg_dt_s")
    requested_dt = _finite_number(stepping.get("requested_dt_s"), "stepping.requested_dt_s")
    if (
        config_accessor_dt <= 0.0
        or config_accessor_dt != cfg_dt
        or config_accessor_dt != requested_dt
        or stepping.get("config_accessor_semantics")
        != (
            "SimulationContext.get_physics_dt_returns_SimulationCfg.dt_in_"
            "the_pinned_kitless_stack_not_compiled_backend_state"
        )
    ):
        raise EffectiveReadbackError("adapter configured-timestep readbacks are inconsistent")
    external_substeps = _integer(
        stepping.get("external_substeps"), "stepping.external_substeps", minimum=1
    )
    backend_step_calls = _integer(
        stepping.get("backend_step_calls_per_adapter_step"),
        "stepping.backend_step_calls_per_adapter_step",
        minimum=1,
    )
    _integer(
        stepping.get("render_interval"), "stepping.render_interval", minimum=1
    )
    if (
        external_substeps != 1
        or backend_step_calls != 1
        or stepping.get("backend_step_api") != "ovphysx.PhysX.step_sync"
        or stepping.get("physics_scene_time_steps_per_second_controls_effective_dt")
        is not False
        or stepping.get("render_interval_is_physics_substeps") is not False
        or stepping.get("gpu_warmup_outside_trace") is not True
        or stepping.get("gpu_warmup_timestep_status")
        != "backend_internal_minimal_not_exposed"
    ):
        raise EffectiveReadbackError("adapter stepping disclosure changed")
    if stepping.get("internal_solver_substeps") is not None:
        raise EffectiveReadbackError("adapter unexpectedly inferred internal solver substeps")
    if stepping.get("internal_solver_substeps_status") != "not_exposed_not_inferred":
        raise EffectiveReadbackError("adapter internal-substep disclosure changed")

    stabilization_record = scene_records["enable_stabilization"]
    stabilization_evidence = _unavailable_solver_family(
        [("enable_stabilization", stabilization_record)],
        unit="boolean",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if stabilization_evidence is None:
        stabilization = stabilization_record["resolved_value"]
        if not isinstance(stabilization, bool):
            raise EffectiveReadbackError("scene.enable_stabilization must be boolean")
        stabilization_evidence = _composed_input_evidence(
            stabilization,
            authored=stabilization_record["has_authored_value_opinion"] is True,
            unit="boolean",
            semantics="live_composed_usd_scene_input_not_compiled_solver_state",
            source_locator=str(stabilization_record["source_locator"]),
        )

    tolerance_records = [
        (
            "stabilization_threshold",
            articulation_records["stabilization_threshold"],
        ),
        ("scene_time_steps_per_second", scene_records["time_steps_per_second"]),
    ]
    tolerance_evidence = _unavailable_solver_family(
        tolerance_records,
        unit="mixed_composed_inputs",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if tolerance_evidence is None:
        tolerances = {
            "stabilization_threshold": _finite_number(
                tolerance_records[0][1]["resolved_value"],
                "articulation.stabilization_threshold",
            ),
            "scene_time_steps_per_second": _integer(
                tolerance_records[1][1]["resolved_value"],
                "scene.time_steps_per_second",
                minimum=1,
            ),
        }
        tolerance_evidence = _composed_input_evidence(
            tolerances,
            authored=all(
                record["has_authored_value_opinion"] is True
                for _, record in tolerance_records
            ),
            unit="mixed_composed_inputs",
            semantics="descriptive_composed_inputs_not_cross_backend_equivalent_tolerances",
            source_locator=" | ".join(
                str(record["source_locator"]) for _, record in tolerance_records
            ),
        )

    sleep_record = articulation_records["sleep_threshold"]
    sleep_evidence = _unavailable_solver_family(
        [("sleep_threshold", sleep_record)],
        unit="backend_native_threshold",
        semantics="live_composed_usd_attribute_probe_only_not_compiled_runtime_state",
    )
    if sleep_evidence is None:
        sleep_evidence = _composed_input_evidence(
            _finite_number(
                sleep_record["resolved_value"], "articulation.sleep_threshold"
            ),
            authored=sleep_record["has_authored_value_opinion"] is True,
            unit="backend_native_threshold",
            semantics="live_composed_usd_articulation_input_not_compiled_solver_state",
            source_locator=str(sleep_record["source_locator"]),
        )

    return {
        "solver_type": solver_type_evidence,
        "position_iterations": position_evidence,
        "velocity_iterations": velocity_evidence,
        "physics_dt_s": {
            "availability": "partially_available",
            "authored_value": requested_dt,
            "resolved_value": cfg_dt,
            "runtime_effective_value": None,
            "unavailable_layers": ["runtime_effective"],
            "unit": "s",
            "semantics": (
                "requested_and_SimulationCfg_dt_with_config_accessor_echo_not_"
                "compiled_backend_timestep"
            ),
            "source_locator": (
                "OVPhysXAdapter.requested_dt/SimulationCfg.dt/"
                "SimulationContext.get_physics_dt_config_accessor"
            ),
            "unavailable_reason": (
                "In the pinned kit-less stack get_physics_dt reads the Python "
                "SimulationCfg value; no compiled backend timestep getter was "
                "exposed and no experimental trace step was executed."
            ),
        },
        "substeps": _unavailable_evidence(
            unit="count",
            semantics="composite_substep_family_has_unexposed_internal_solver_count",
            source_locator="OVPhysXAdapter.step/ovphysx.PhysX.step_sync",
            reason=(
                f"The adapter declares {external_substeps} external step_sync call per "
                "adapter step, but the pinned backend does not expose internal solver "
                "substeps; the composite family is conservatively unavailable."
            ),
        ),
        "integrator": _unavailable_evidence(
            unit="enum",
            semantics="compiled_integrator_not_exposed_by_pinned_python_api",
            source_locator="isaaclab_ovphysx.physics.OvPhysxCfg",
            reason="No compiled integrator getter is exposed by the pinned kit-less API.",
        ),
        "stabilization": stabilization_evidence,
        "tolerances": tolerance_evidence,
        "sleep_threshold": sleep_evidence,
        "execution_pipeline": _runtime_only_evidence(
            "gpu",
            unit="enum",
            semantics="observed_kitless_headless_OVPhysX_GPU_execution_pipeline",
            source_locator="OVPhysXAdapter.device/runtime_api",
        ),
    }


def _vector_value(record: Mapping[str, Any], key: str) -> object:
    if key == "friction_runtime_effective":
        return _mapping(record.get("friction"), "record.friction").get(
            "runtime_effective_value"
        )
    if key == "armature_runtime_effective":
        return _mapping(record.get("armature"), "record.armature").get(
            "runtime_effective_value"
        )
    field = key.removesuffix("_runtime_effective")
    drive = _mapping(record.get("control_drive"), "record.control_drive")
    return _mapping(drive.get(field), f"record.control_drive.{field}").get(
        "runtime_effective_value"
    )


def canonical_vector_hash(records: Sequence[Mapping[str, Any]], key: str) -> str:
    return canonical_json_sha256([_vector_value(record, key) for record in records])


def build_readback_instance(
    *,
    plan: Mapping[str, Any],
    manifest: ParityManifest,
    adapter_snapshot: Mapping[str, Any],
    side: str,
    instance_index: int,
    case_id: str,
    fresh_instance_id: str,
    fresh_process_id: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one pre-step adapter snapshot into the frozen portable schema."""

    expected = expected_instance(plan, side=side, instance_index=instance_index)
    if expected.get("case_id") != case_id:
        raise EffectiveReadbackError("worker case_id does not match Freeze A")
    try:
        hand = manifest.hand(HandSide(side))
    except (ValueError, KeyError) as exc:
        raise EffectiveReadbackError(f"invalid hand side: {side}") from exc
    if (
        adapter_snapshot.get("readback_contract_version") != 1
        or adapter_snapshot.get("capture_phase") != "post_reset_pre_trace_step"
        or adapter_snapshot.get("step_index") != 0
    ):
        raise EffectiveReadbackError("adapter snapshot was not captured pre-trace-step")
    dynamics = _mapping(
        adapter_snapshot.get("backend_joint_dynamics"),
        "adapter_snapshot.backend_joint_dynamics",
    )
    if (
        dynamics.get("schema_version") != 1
        or dynamics.get("source") != "ovphysx_tensor_binding_post_load"
        or dynamics.get("runtime_effective_readback") is not True
        or dynamics.get("capture_phase") != "post_reset_pre_trace_step"
        or dynamics.get("joint_count") != 22
        or dynamics.get("controller_consistency_verified") is not True
        or dynamics.get("controller_joint_dynamics_binding_comparison_performed")
        is not True
        or dynamics.get("controller_joint_dynamics_binding_match_policy")
        != "exact_float32_value"
        or dynamics.get("backend_drive_zero_verified") is not True
    ):
        raise EffectiveReadbackError("adapter joint-dynamics contract changed or failed")
    backend_order = tuple(
        _nonempty_string(name, "backend joint name")
        for name in _array(dynamics.get("joint_order"), "joint_order")
    )
    raw_records = _array(dynamics.get("records"), "backend_joint_dynamics.records")
    if len(backend_order) != 22 or len(set(backend_order)) != 22 or len(raw_records) != 22:
        raise EffectiveReadbackError("adapter readback does not contain exactly 22 unique joints")
    friction_semantics = _mapping(
        dynamics.get("friction_semantics"), "backend_joint_dynamics.friction_semantics"
    )
    if friction_semantics.get("raw_slot_order") != ["static", "dynamic", "viscous"]:
        raise EffectiveReadbackError("OVPhysX friction slot order changed")
    if friction_semantics.get("cross_engine_equivalence") != "not_claimed":
        raise EffectiveReadbackError("adapter overstates cross-engine friction equivalence")
    reported_top_matches = _mapping(
        dynamics.get("controller_joint_dynamics_binding_exact_match"),
        "backend_joint_dynamics.controller_joint_dynamics_binding_exact_match",
    )
    if set(reported_top_matches) != {
        "overall",
        "armature",
        "static_friction",
        "dynamic_friction",
        "viscous_friction",
    } or any(not isinstance(value, bool) for value in reported_top_matches.values()):
        raise EffectiveReadbackError("adapter top-level binding match flags are invalid")

    dof_records: list[dict[str, Any]] = []
    recomputed_matches = {
        "armature": True,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }
    for canonical_index, canonical_name in enumerate(hand.joint_names):
        matches = [
            _mapping(raw, f"adapter record[{index}]")
            for index, raw in enumerate(raw_records)
            if isinstance(raw, Mapping) and raw.get("canonical_id") == canonical_name
        ]
        if len(matches) != 1:
            raise EffectiveReadbackError(
                f"adapter does not map canonical joint exactly once: {canonical_name}"
            )
        raw = matches[0]
        backend_index = _integer(raw.get("backend_index"), f"{canonical_name}.backend_index", minimum=0)
        backend_name = _nonempty_string(raw.get("backend_name"), f"{canonical_name}.backend_name")
        if backend_index >= 22 or backend_order[backend_index] != backend_name:
            raise EffectiveReadbackError(f"backend mapping mismatch for {canonical_name}")
        friction_raw = _array(
            raw.get("friction_properties_raw"), f"{canonical_name}.friction_properties_raw"
        )
        if len(friction_raw) != 3:
            raise EffectiveReadbackError(f"friction readback must have three slots: {canonical_name}")
        friction = {
            slot: _finite_number(friction_raw[index], f"{canonical_name}.friction.{slot}")
            for index, slot in enumerate(("static", "dynamic", "viscous"))
        }
        if any(value < 0.0 for value in friction.values()):
            raise EffectiveReadbackError(f"negative friction readback: {canonical_name}")
        armature = _finite_number(raw.get("armature"), f"{canonical_name}.armature")
        if armature < 0.0:
            raise EffectiveReadbackError(f"negative armature readback: {canonical_name}")
        controller_armature = _finite_number(
            raw.get("controller_armature"),
            f"{canonical_name}.controller_armature",
        )
        controller_friction = {
            "static": _finite_number(
                raw.get("controller_static_friction"),
                f"{canonical_name}.controller_static_friction",
            ),
            "dynamic": _finite_number(
                raw.get("controller_dynamic_friction"),
                f"{canonical_name}.controller_dynamic_friction",
            ),
            "viscous": _finite_number(
                raw.get("controller_viscous_friction"),
                f"{canonical_name}.controller_viscous_friction",
            ),
        }
        if controller_armature < 0.0 or any(
            value < 0.0 for value in controller_friction.values()
        ):
            raise EffectiveReadbackError(
                f"negative IdealPD joint-dynamics buffer: {canonical_name}"
            )
        computed_record_matches = {
            "armature": armature == controller_armature,
            "static_friction": friction["static"] == controller_friction["static"],
            "dynamic_friction": friction["dynamic"]
            == controller_friction["dynamic"],
            "viscous_friction": friction["viscous"]
            == controller_friction["viscous"],
        }
        reported_record_matches = _mapping(
            raw.get("controller_binding_exact_match"),
            f"{canonical_name}.controller_binding_exact_match",
        )
        if (
            set(reported_record_matches) != set(computed_record_matches)
            or any(not isinstance(value, bool) for value in reported_record_matches.values())
            or dict(reported_record_matches) != computed_record_matches
        ):
            raise EffectiveReadbackError(
                f"adapter binding match flags do not reproduce raw values: {canonical_name}"
            )
        for family, matched in computed_record_matches.items():
            recomputed_matches[family] = recomputed_matches[family] and matched
        kp = _finite_number(raw.get("controller_stiffness"), f"{canonical_name}.kp")
        kd = _finite_number(raw.get("controller_damping"), f"{canonical_name}.kd")
        effort = _finite_number(raw.get("controller_effort_limit"), f"{canonical_name}.effort_limit")
        effort_sim = _finite_number(
            raw.get("controller_effort_limit_sim"), f"{canonical_name}.effort_limit_sim"
        )
        backend_kp = _finite_number(
            raw.get("backend_drive_stiffness"), f"{canonical_name}.backend_drive_stiffness"
        )
        backend_kd = _finite_number(
            raw.get("backend_drive_damping"), f"{canonical_name}.backend_drive_damping"
        )
        if min(kp, kd, effort, effort_sim, backend_kp, backend_kd) < 0.0:
            raise EffectiveReadbackError(f"negative control readback: {canonical_name}")
        joint_path = _nonempty_string(raw.get("joint_prim_path"), f"{canonical_name}.joint_prim_path")
        if not joint_path.startswith("/World/"):
            raise EffectiveReadbackError("joint prim path is not portable live-stage provenance")
        armature_usd = _mapping(
            raw.get("armature_usd"), f"{canonical_name}.armature_usd"
        )
        legacy_friction_usd = _mapping(
            raw.get("legacy_joint_friction_usd"),
            f"{canonical_name}.legacy_joint_friction_usd",
        )
        if (
            armature_usd.get("attribute") != "physxJoint:armature"
            or armature_usd.get("source") != "composed_usd_input"
            or armature_usd.get("compiled_runtime_readback") is not False
            or legacy_friction_usd.get("attribute")
            != "physxJoint:jointFriction"
            or legacy_friction_usd.get("source") != "composed_usd_input"
            or legacy_friction_usd.get("compiled_runtime_readback") is not False
            or legacy_friction_usd.get("runtime_binding_equality_check")
            != "not_performed"
        ):
            raise EffectiveReadbackError(
                f"composed USD parameter provenance changed: {canonical_name}"
            )
        armature_authored = armature_usd.get("has_authored_value_opinion")
        friction_authored = legacy_friction_usd.get("has_authored_value_opinion")
        if not isinstance(armature_authored, bool) or not isinstance(
            friction_authored, bool
        ):
            raise EffectiveReadbackError(
                f"composed USD authored-state is invalid: {canonical_name}"
            )
        armature_authored_value = armature_usd.get("authored_value")
        if armature_authored:
            armature_authored_value = _finite_number(
                armature_authored_value, f"{canonical_name}.armature_usd.authored"
            )
        elif armature_authored_value is not None:
            raise EffectiveReadbackError(
                f"unauthored armature has a value: {canonical_name}"
            )
        armature_resolved = _finite_number(
            armature_usd.get("resolved_value"),
            f"{canonical_name}.armature_usd.resolved",
        )
        friction_authored_value = legacy_friction_usd.get("authored_value")
        if friction_authored:
            friction_authored_value = _finite_number(
                friction_authored_value,
                f"{canonical_name}.legacy_joint_friction_usd.authored",
            )
        elif friction_authored_value is not None:
            raise EffectiveReadbackError(
                f"unauthored legacy friction has a value: {canonical_name}"
            )
        friction_resolved = _finite_number(
            legacy_friction_usd.get("resolved_value"),
            f"{canonical_name}.legacy_joint_friction_usd.resolved",
        )
        if min(armature_resolved, friction_resolved) < 0.0:
            raise EffectiveReadbackError(
                f"negative composed USD parameter: {canonical_name}"
            )
        armature_locator = _nonempty_string(
            armature_usd.get("source_locator"),
            f"{canonical_name}.armature_usd.source_locator",
        )
        friction_locator = _nonempty_string(
            legacy_friction_usd.get("source_locator"),
            f"{canonical_name}.legacy_joint_friction_usd.source_locator",
        )
        dof_records.append(
            {
                "mapping": {
                    "side": side,
                    "canonical_joint_name": canonical_name,
                    "backend_joint_name": backend_name,
                    "backend_joint_index": backend_index,
                    "joint_prim_path": joint_path,
                    "joint_type": _nonempty_string(raw.get("joint_type"), f"{canonical_name}.joint_type"),
                    "axis": _axis_vector(raw.get("joint_axis"), f"{canonical_name}.joint_axis"),
                    "canonical_sign": 1.0,
                    "canonical_offset_rad": 0.0,
                    "canonical_unit": "rad",
                    "source_locator": f"composed-usd:{joint_path}#physics:axis",
                },
                "friction": _layered_evidence(
                    authored_value=(
                        {"legacy_joint_friction_scalar": friction_authored_value}
                        if friction_authored
                        else None
                    ),
                    resolved_value={
                        "legacy_joint_friction_scalar": friction_resolved
                    },
                    runtime_effective_value={
                        "dof_friction_properties_binding": friction,
                        "idealpd_controller_buffers": controller_friction,
                        "binding_controller_exact_match": {
                            "overall": all(
                                computed_record_matches[name]
                                for name in (
                                    "static_friction",
                                    "dynamic_friction",
                                    "viscous_friction",
                                )
                            ),
                            "static_friction": computed_record_matches[
                                "static_friction"
                            ],
                            "dynamic_friction": computed_record_matches[
                                "dynamic_friction"
                            ],
                            "viscous_friction": computed_record_matches[
                                "viscous_friction"
                            ],
                        },
                    },
                    unit={
                        "authored_and_resolved_legacy_scalar": (
                            "documentation_conflict_not_assigned"
                        ),
                        "runtime_static_dynamic": (
                            "documentation_conflict_not_assigned"
                        ),
                        "runtime_viscous": (
                            "N*m*s/rad_for_revolute_joint"
                        ),
                    },
                    semantics=(
                        "authored/resolved values are the legacy composed-USD scalar; "
                        "runtime_effective_value preserves the distinct raw OVPhysX "
                        "[static,dynamic,viscous] binding, IdealPD buffers, and exact "
                        "comparison flags. No USD-to-runtime or cross-engine physical "
                        "equivalence is performed or implied"
                    ),
                    source_locator=(
                        f"composed-usd:{friction_locator};"
                        "ovphysx.TensorDataType.DOF_FRICTION_PROPERTIES/"
                        "root_view_cpu_numpy_binding"
                    ),
                ),
                "armature": _layered_evidence(
                    authored_value=(
                        armature_authored_value if armature_authored else None
                    ),
                    resolved_value=armature_resolved,
                    runtime_effective_value={
                        "dof_armature_binding": armature,
                        "idealpd_controller_buffer": controller_armature,
                        "binding_controller_exact_match": computed_record_matches[
                            "armature"
                        ],
                    },
                    unit="kg*m^2_for_revolute_joint",
                    semantics=(
                        "authored/resolved composed-USD added joint-space inertia and "
                        "independent compiled DOF_ARMATURE and IdealPD-buffer runtime "
                        "readbacks with their exact comparison flag"
                    ),
                    source_locator=(
                        f"composed-usd:{armature_locator};"
                        "ovphysx.TensorDataType.DOF_ARMATURE/"
                        "root_view_cpu_numpy_binding"
                    ),
                ),
                "control_drive": {
                    "controller_model": _runtime_only_evidence(
                        "IdealPDActuator",
                        unit="enum",
                        semantics="explicit_PD_effort_controller_model",
                        source_locator="isaaclab.actuators.IdealPDActuator",
                    ),
                    "kp": _runtime_only_evidence(
                        kp,
                        unit="N*m/rad_for_revolute_joint",
                        semantics="IdealPD_explicit_effort_stiffness",
                        source_locator="IdealPDActuator.stiffness_torch",
                    ),
                    "kd": _runtime_only_evidence(
                        kd,
                        unit="N*m*s/rad_for_revolute_joint",
                        semantics="IdealPD_explicit_effort_damping",
                        source_locator="IdealPDActuator.damping_torch",
                    ),
                    "backend_drive_stiffness": _runtime_only_evidence(
                        backend_kp,
                        unit="N*m/rad_for_revolute_joint",
                        semantics="compiled_PhysX_drive_stiffness_zeroed_for_explicit_PD",
                        source_locator="ovphysx.TensorDataType.DOF_STIFFNESS/root_view_cpu_numpy_binding",
                    ),
                    "backend_drive_damping": _runtime_only_evidence(
                        backend_kd,
                        unit="N*m*s/rad_for_revolute_joint",
                        semantics="compiled_PhysX_drive_damping_zeroed_for_explicit_PD",
                        source_locator="ovphysx.TensorDataType.DOF_DAMPING/root_view_cpu_numpy_binding",
                    ),
                    "passive_joint_damping": _unavailable_evidence(
                        unit="N*m*s/rad_for_revolute_joint",
                        semantics="separate_passive_joint_damping_not_exposed",
                        source_locator="pinned-isaaclab-ovphysx:ArticulationData",
                        reason="The pinned OVPhysX API does not expose a separate passive damping term.",
                    ),
                    "gear_or_transmission": _unavailable_evidence(
                        unit="ratio",
                        semantics="gear_or_transmission_not_exposed",
                        source_locator="pinned-isaaclab-ovphysx:ArticulationData",
                        reason="The pinned OVPhysX tensor API does not expose a gear or transmission ratio.",
                    ),
                    "effort_limit": _runtime_only_evidence(
                        {"controller": effort, "simulation": effort_sim},
                        unit="N*m_for_revolute_joint",
                        semantics="IdealPD_controller_and_simulation_effort_limits",
                        source_locator="IdealPDActuator.effort_limit_and_effort_limit_sim_torch",
                    ),
                    "velocity_limit": _unavailable_evidence(
                        unit="rad/s",
                        semantics="not_captured_by_this_frozen_adapter_contract",
                        source_locator="OVPhysXAdapter.backend_joint_dynamics.velocity_limit_readback",
                        reason="Freeze A does not infer velocity limits without a captured runtime value.",
                    ),
                    "target_interpretation": _runtime_only_evidence(
                        "canonical_joint_position_rad_explicit_PD_effort",
                        unit="enum",
                        semantics="position_target_consumed_by_explicit_IdealPD_effort_law",
                        source_locator="OVPhysXAdapter.set_position_targets/IdealPDActuator.compute",
                    ),
                },
            }
        )

    recomputed_top_matches = {
        "overall": all(recomputed_matches.values()),
        **recomputed_matches,
    }
    if dict(reported_top_matches) != recomputed_top_matches:
        raise EffectiveReadbackError(
            "adapter top-level binding match flags do not reproduce per-joint values"
        )

    schema = _mapping(plan.get("readback_schema"), "readback_schema")
    hash_keys = _array(
        schema.get("canonical_vector_sha256_required_keys"),
        "canonical_vector_sha256_required_keys",
    )
    vector_hashes = {
        _nonempty_string(key, "canonical vector hash key"): canonical_vector_hash(
            dof_records, str(key)
        )
        for key in hash_keys
    }
    payload = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "case_id": case_id,
        "side": side,
        "instance_index": instance_index,
        "fresh_instance_id": _nonempty_string(fresh_instance_id, "fresh_instance_id"),
        "fresh_process_id": _nonempty_string(fresh_process_id, "fresh_process_id"),
        "readback_only_after_initialization": True,
        "initialization_state_write_performed": True,
        "experimental_trace_dynamics_advance_performed": False,
        "experimental_control_command_applied": False,
        "experimental_target_write_performed": False,
        "user_trace_simulation_step_call_count": 0,
        "trajectory_sample_count": 0,
        "trace_time_s": 0.0,
        "gpu_warmup_outside_trace": _runtime_only_evidence(
            True,
            unit="boolean",
            semantics="adapter_runtime_initialization_only_not_user_or_experimental_trace",
            source_locator="physics_solver_contract.stepping.gpu_warmup_outside_trace",
        ),
        "initialization_internal_advance_exposure": _unavailable_evidence(
            unit="step_count",
            semantics="backend_internal_initialization_advance_not_exposed_not_inferred",
            source_locator="physics_solver_contract.stepping.gpu_warmup_timestep_status",
            reason="The pinned backend does not quantify internal initialization advances.",
        ),
        "provenance": _json_copy(dict(provenance)),
        "solver_runtime": _solver_runtime(adapter_snapshot),
        "dof_records": dof_records,
        "canonical_vector_sha256": vector_hashes,
    }
    validate_readback_instance(plan, manifest, payload)
    return payload


def _validate_evidence(plan: Mapping[str, Any], raw: object, path: str) -> Mapping[str, Any]:
    evidence = _mapping(raw, path)
    schema = _mapping(plan.get("readback_schema"), "readback_schema")
    fields = [str(value) for value in _array(schema.get("required_evidence_fields"), "required_evidence_fields")]
    _exact_keys(evidence, fields, path)
    availability = evidence.get("availability")
    allowed = set(_array(schema.get("allowed_availability"), "allowed_availability"))
    if availability not in allowed:
        raise EffectiveReadbackError(f"{path}.availability is invalid")
    layers = _array(evidence.get("unavailable_layers"), f"{path}.unavailable_layers")
    if any(layer not in {"authored", "resolved", "runtime_effective"} for layer in layers):
        raise EffectiveReadbackError(f"{path} names an invalid unavailable layer")
    if len(layers) != len(set(layers)):
        raise EffectiveReadbackError(f"{path} repeats an unavailable layer")
    values = {
        "authored": evidence.get("authored_value"),
        "resolved": evidence.get("resolved_value"),
        "runtime_effective": evidence.get("runtime_effective_value"),
    }
    for layer, value in values.items():
        if value is None and layer not in layers:
            raise EffectiveReadbackError(f"{path} leaves null layer {layer!r} undisclosed")
        if value is not None and layer in layers:
            raise EffectiveReadbackError(f"{path} marks present layer {layer!r} unavailable")
    reason = evidence.get("unavailable_reason")
    if availability == "available":
        if layers or reason is not None or any(value is None for value in values.values()):
            raise EffectiveReadbackError(f"{path} contradicts available status")
    elif not layers or not isinstance(reason, str) or not reason.strip():
        raise EffectiveReadbackError(f"{path} must disclose unavailable layers and reason")
    if availability == "unavailable" and any(value is not None for value in values.values()):
        raise EffectiveReadbackError(f"{path} unavailable status contains a value")
    if availability == "partially_available" and all(value is None for value in values.values()):
        raise EffectiveReadbackError(f"{path} partial status contains no evidence")
    _nonempty_string(evidence.get("semantics"), f"{path}.semantics")
    _nonempty_string(evidence.get("source_locator"), f"{path}.source_locator")
    _validate_finite_json(evidence, path)
    return evidence


def _expected_names(manifest: ParityManifest, side: str) -> tuple[str, ...]:
    try:
        return manifest.hand(HandSide(side)).joint_names
    except (ValueError, KeyError) as exc:
        raise EffectiveReadbackError(f"unknown hand side: {side}") from exc


def _validate_armature_layers(evidence: Mapping[str, Any], path: str) -> None:
    authored = evidence.get("authored_value")
    if authored is not None and _finite_number(authored, f"{path}.authored_value") < 0.0:
        raise EffectiveReadbackError(f"{path}.authored_value must be non-negative")
    if _finite_number(evidence.get("resolved_value"), f"{path}.resolved_value") < 0.0:
        raise EffectiveReadbackError(f"{path}.resolved_value must be non-negative")
    runtime = _mapping(evidence.get("runtime_effective_value"), f"{path}.runtime")
    if set(runtime) != {
        "dof_armature_binding",
        "idealpd_controller_buffer",
        "binding_controller_exact_match",
    }:
        raise EffectiveReadbackError(f"{path}.runtime has invalid fields")
    binding = _finite_number(runtime.get("dof_armature_binding"), f"{path}.binding")
    controller = _finite_number(
        runtime.get("idealpd_controller_buffer"), f"{path}.controller_buffer"
    )
    match = runtime.get("binding_controller_exact_match")
    if min(binding, controller) < 0.0 or not isinstance(match, bool):
        raise EffectiveReadbackError(f"{path}.runtime values are invalid")
    if match is not (binding == controller):
        raise EffectiveReadbackError(f"{path}.runtime match flag is false evidence")


def _validate_friction_layers(evidence: Mapping[str, Any], path: str) -> None:
    for layer in ("authored_value", "resolved_value"):
        value = evidence.get(layer)
        if value is None and layer == "authored_value":
            continue
        wrapper = _mapping(value, f"{path}.{layer}")
        if set(wrapper) != {"legacy_joint_friction_scalar"}:
            raise EffectiveReadbackError(f"{path}.{layer} has invalid fields")
        if _finite_number(
            wrapper.get("legacy_joint_friction_scalar"),
            f"{path}.{layer}.legacy_joint_friction_scalar",
        ) < 0.0:
            raise EffectiveReadbackError(f"{path}.{layer} must be non-negative")
    runtime = _mapping(evidence.get("runtime_effective_value"), f"{path}.runtime")
    if set(runtime) != {
        "dof_friction_properties_binding",
        "idealpd_controller_buffers",
        "binding_controller_exact_match",
    }:
        raise EffectiveReadbackError(f"{path}.runtime has invalid fields")
    binding = _mapping(
        runtime.get("dof_friction_properties_binding"), f"{path}.runtime.binding"
    )
    controller = _mapping(
        runtime.get("idealpd_controller_buffers"), f"{path}.runtime.controller"
    )
    slots = {"static", "dynamic", "viscous"}
    if set(binding) != slots or set(controller) != slots:
        raise EffectiveReadbackError(f"{path}.runtime friction slots are invalid")
    binding_values = {
        slot: _finite_number(binding[slot], f"{path}.binding.{slot}")
        for slot in slots
    }
    controller_values = {
        slot: _finite_number(controller[slot], f"{path}.controller.{slot}")
        for slot in slots
    }
    if min(*binding_values.values(), *controller_values.values()) < 0.0:
        raise EffectiveReadbackError(f"{path}.runtime friction values are negative")
    flags = _mapping(
        runtime.get("binding_controller_exact_match"), f"{path}.runtime.match"
    )
    expected_flags = {
        "overall": all(
            binding_values[slot] == controller_values[slot] for slot in slots
        ),
        "static_friction": binding_values["static"] == controller_values["static"],
        "dynamic_friction": binding_values["dynamic"]
        == controller_values["dynamic"],
        "viscous_friction": binding_values["viscous"]
        == controller_values["viscous"],
    }
    if (
        set(flags) != set(expected_flags)
        or any(not isinstance(value, bool) for value in flags.values())
        or dict(flags) != expected_flags
    ):
        raise EffectiveReadbackError(f"{path}.runtime friction match flags are false")


def validate_readback_instance(
    plan: Mapping[str, Any],
    manifest: ParityManifest,
    raw: object,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate one worker payload and return a detached JSON-safe copy."""

    instance = _mapping(raw, "readback instance")
    schema = _mapping(plan.get("readback_schema"), "readback_schema")
    required_instance = [str(value) for value in _array(schema.get("required_instance_fields"), "required_instance_fields")]
    _exact_keys(
        instance,
        ["schema_version", "experiment_id", *required_instance],
        "readback instance",
    )
    if instance.get("schema_version") != 1:
        raise EffectiveReadbackError("instance schema_version must be 1")
    if instance.get("experiment_id") != plan.get("experiment_id"):
        raise EffectiveReadbackError("instance experiment_id does not match Freeze A")
    side = _nonempty_string(instance.get("side"), "instance.side")
    index = _integer(instance.get("instance_index"), "instance.instance_index", minimum=1)
    expected = expected_instance(plan, side=side, instance_index=index)
    if instance.get("case_id") != expected.get("case_id"):
        raise EffectiveReadbackError("instance case_id does not match Freeze A")
    for field in ("fresh_instance_id", "fresh_process_id"):
        value = _nonempty_string(instance.get(field), f"instance.{field}")
        if _IDENTIFIER_RE.fullmatch(value) is None and _SHA256_RE.fullmatch(value) is None:
            raise EffectiveReadbackError(f"instance.{field} is not a portable identifier")
    required_flags = {
        "readback_only_after_initialization": True,
        "initialization_state_write_performed": True,
        "experimental_trace_dynamics_advance_performed": False,
        "experimental_control_command_applied": False,
        "experimental_target_write_performed": False,
        "user_trace_simulation_step_call_count": 0,
        "trajectory_sample_count": 0,
        "trace_time_s": 0.0,
    }
    for field, expected_value in required_flags.items():
        if instance.get(field) != expected_value:
            raise EffectiveReadbackError(f"instance violates readback-only contract: {field}")
    _validate_evidence(plan, instance.get("gpu_warmup_outside_trace"), "gpu_warmup_outside_trace")
    _validate_evidence(
        plan,
        instance.get("initialization_internal_advance_exposure"),
        "initialization_internal_advance_exposure",
    )

    provenance = _mapping(instance.get("provenance"), "instance.provenance")
    provenance_fields = [str(value) for value in _array(schema.get("required_provenance_fields"), "required_provenance_fields")]
    _require_keys(provenance, provenance_fields, "instance.provenance")
    for field in ("source_revision", "source_tree"):
        _hex40(provenance.get(field), f"provenance.{field}")
    for field in (
        "source_archive_sha256",
        "source_snapshot_sha256",
        "probe_source_sha256",
        "pipeline_module_sha256",
        "freeze_a_config_sha256",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "canonical_lf_asset_tree_sha256",
    ):
        _sha256(provenance.get(field), f"provenance.{field}")
    _hex40(provenance.get("asset_commit"), "provenance.asset_commit")
    _hex40(provenance.get("asset_git_tree"), "provenance.asset_git_tree")
    frozen = _mapping(plan.get("frozen_context"), "frozen_context")
    for field in (
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "asset_commit",
        "asset_git_tree",
        "canonical_lf_asset_tree_sha256",
    ):
        if provenance.get(field) != frozen.get(field):
            raise EffectiveReadbackError(f"frozen provenance mismatch: {field}")
    for field in (
        "python_version",
        "isaac_lab_version",
        "isaac_sim_version",
        "physx_version",
        "pytorch_version",
        "device",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "machine_id",
        "session_id",
        "recorded_at_utc",
    ):
        _nonempty_string(provenance.get(field), f"provenance.{field}")
    if provenance.get("device") != "cuda:0":
        raise EffectiveReadbackError("worker device must be cuda:0")
    try:
        recorded = datetime.fromisoformat(
            str(provenance["recorded_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise EffectiveReadbackError("recorded_at_utc is invalid") from exc
    if recorded.tzinfo is None or recorded.utcoffset() != timezone.utc.utcoffset(recorded):
        raise EffectiveReadbackError("recorded_at_utc must be UTC")
    if expected_provenance is not None:
        for field, expected_value in expected_provenance.items():
            if provenance.get(field) != expected_value:
                raise EffectiveReadbackError(f"worker provenance mismatch: {field}")

    solver = _mapping(instance.get("solver_runtime"), "instance.solver_runtime")
    solver_fields = [str(value) for value in _array(schema.get("required_solver_runtime_fields"), "required_solver_runtime_fields")]
    _exact_keys(solver, solver_fields, "instance.solver_runtime")
    for field in solver_fields:
        _validate_evidence(plan, solver[field], f"solver_runtime.{field}")

    records = _array(instance.get("dof_records"), "instance.dof_records")
    names = _expected_names(manifest, side)
    if len(records) != 22:
        raise EffectiveReadbackError("each instance must contain exactly 22 DoF records")
    dof_fields = [str(value) for value in _array(schema.get("required_dof_record_fields"), "required_dof_record_fields")]
    mapping_fields = [str(value) for value in _array(schema.get("required_mapping_fields"), "required_mapping_fields")]
    control_fields = [str(value) for value in _array(schema.get("required_control_drive_fields"), "required_control_drive_fields")]
    observed_names: list[str] = []
    backend_indices: list[int] = []
    typed_records: list[Mapping[str, Any]] = []
    for record_index, raw_record in enumerate(records):
        path = f"dof_records[{record_index}]"
        record = _mapping(raw_record, path)
        _exact_keys(record, dof_fields, path)
        mapping = _mapping(record.get("mapping"), f"{path}.mapping")
        _exact_keys(mapping, mapping_fields, f"{path}.mapping")
        if mapping.get("side") != side:
            raise EffectiveReadbackError(f"{path}.mapping side mismatch")
        observed_names.append(
            _nonempty_string(mapping.get("canonical_joint_name"), f"{path}.canonical_joint_name")
        )
        _nonempty_string(mapping.get("backend_joint_name"), f"{path}.backend_joint_name")
        backend_indices.append(
            _integer(mapping.get("backend_joint_index"), f"{path}.backend_joint_index", minimum=0)
        )
        if mapping.get("joint_type") != "revolute" or mapping.get("canonical_unit") != "rad":
            raise EffectiveReadbackError(f"{path}.mapping joint type/unit changed")
        axis = _array(mapping.get("axis"), f"{path}.axis")
        if len(axis) != 3 or sum(abs(_finite_number(value, f"{path}.axis")) for value in axis) != 1.0:
            raise EffectiveReadbackError(f"{path}.axis must be a Cartesian unit axis")
        if _finite_number(mapping.get("canonical_sign"), f"{path}.canonical_sign") not in (-1.0, 1.0):
            raise EffectiveReadbackError(f"{path}.canonical_sign must be +/-1")
        _finite_number(mapping.get("canonical_offset_rad"), f"{path}.canonical_offset_rad")
        _nonempty_string(mapping.get("joint_prim_path"), f"{path}.joint_prim_path")
        _nonempty_string(mapping.get("source_locator"), f"{path}.source_locator")
        friction_evidence = _validate_evidence(
            plan, record.get("friction"), f"{path}.friction"
        )
        armature_evidence = _validate_evidence(
            plan, record.get("armature"), f"{path}.armature"
        )
        _validate_friction_layers(friction_evidence, f"{path}.friction")
        _validate_armature_layers(armature_evidence, f"{path}.armature")
        control = _mapping(record.get("control_drive"), f"{path}.control_drive")
        _exact_keys(control, control_fields, f"{path}.control_drive")
        for field in control_fields:
            _validate_evidence(plan, control[field], f"{path}.control_drive.{field}")
        typed_records.append(record)
    if tuple(observed_names) != names or len(set(observed_names)) != 22:
        raise EffectiveReadbackError("canonical DoF coverage/order differs from Gate 0")
    if len(set(backend_indices)) != 22 or set(backend_indices) != set(range(22)):
        raise EffectiveReadbackError("backend DoF indices are not a unique 0..21 mapping")

    hashes = _mapping(instance.get("canonical_vector_sha256"), "canonical_vector_sha256")
    hash_keys = [str(value) for value in _array(schema.get("canonical_vector_sha256_required_keys"), "canonical_vector_sha256_required_keys")]
    _exact_keys(hashes, hash_keys, "canonical_vector_sha256")
    for key in hash_keys:
        if hashes[key] != canonical_vector_hash(typed_records, key):
            raise EffectiveReadbackError(f"canonical vector hash mismatch: {key}")
    _validate_finite_json(instance, "readback instance")
    return _json_copy(instance)


def assemble_readback_payload(
    plan: Mapping[str, Any],
    manifest: ParityManifest,
    instances: Sequence[object],
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate four records and require exact within-hand reproduction."""

    validated = [
        validate_readback_instance(
            plan, manifest, item, expected_provenance=expected_provenance
        )
        for item in instances
    ]
    expected_rows = [
        (str(row["side"]), int(row["instance_index"]), str(row["case_id"]))
        for row in _array(plan.get("expected_instances"), "expected_instances")
        if isinstance(row, Mapping)
    ]
    actual_rows = [
        (str(row["side"]), int(row["instance_index"]), str(row["case_id"]))
        for row in validated
    ]
    if actual_rows != expected_rows:
        raise EffectiveReadbackError("worker payload order/matrix differs from Freeze A")
    instance_ids = [str(row["fresh_instance_id"]) for row in validated]
    process_ids = [str(row["fresh_process_id"]) for row in validated]
    if len(set(instance_ids)) != 4:
        raise EffectiveReadbackError("fresh_instance_id was reused")
    if len(set(process_ids)) != 4:
        raise EffectiveReadbackError("fresh_process_id was reused")

    stable_provenance_fields = (
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "source_snapshot_sha256",
        "probe_source_sha256",
        "pipeline_module_sha256",
        "freeze_a_config_sha256",
        "gate0_manifest_file_sha256",
        "gate0_manifest_semantic_sha256",
        "asset_commit",
        "asset_git_tree",
        "canonical_lf_asset_tree_sha256",
        "python_version",
        "isaac_lab_version",
        "isaac_sim_version",
        "physx_version",
        "pytorch_version",
        "device",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "machine_id",
        "session_id",
    )
    first_provenance = _mapping(validated[0]["provenance"], "provenance")
    for row in validated[1:]:
        provenance = _mapping(row["provenance"], "provenance")
        for field in stable_provenance_fields:
            if provenance.get(field) != first_provenance.get(field):
                raise EffectiveReadbackError(
                    f"fresh processes disagree on stable provenance: {field}"
                )

    for side in ("left", "right"):
        first = next(row for row in validated if row["side"] == side and row["instance_index"] == 1)
        second = next(row for row in validated if row["side"] == side and row["instance_index"] == 2)
        if first["dof_records"] != second["dof_records"]:
            raise EffectiveReadbackError(f"{side} DoF readback is not exactly reproducible")
        if first["solver_runtime"] != second["solver_runtime"]:
            raise EffectiveReadbackError(f"{side} solver readback is not exactly reproducible")
        if first["canonical_vector_sha256"] != second["canonical_vector_sha256"]:
            raise EffectiveReadbackError(f"{side} vector hashes are not exactly reproducible")

    payload = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "status": plan["fail_closed"]["valid_status"],
        "instances": validated,
        "coverage": {
            "canonical_joint_count_per_hand": {"left": 22, "right": 22},
            "canonical_joint_count_total": 44,
            "covered_joint_count_total": 44,
            "complete": True,
        },
        "reproducibility": {
            "mapping_exact": True,
            "solver_runtime_exact": True,
            "canonical_numeric_vector_hashes_exact": True,
        },
        "claim_boundary": _json_copy(plan["claim_boundary"]),
        "freeze_b_required": True,
    }
    _validate_finite_json(payload)
    return payload


def _availability_counts(evidences: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    counts = {"available": 0, "partially_available": 0, "unavailable": 0}
    layers = {"authored": 0, "resolved": 0, "runtime_effective": 0}
    locators: set[str] = set()
    for evidence in evidences:
        counts[str(evidence["availability"])] += 1
        for layer, key in (
            ("authored", "authored_value"),
            ("resolved", "resolved_value"),
            ("runtime_effective", "runtime_effective_value"),
        ):
            if evidence[key] is not None:
                layers[layer] += 1
        locators.add(str(evidence["source_locator"]))
    return {
        "record_count": len(evidences),
        "availability_counts": counts,
        "present_layer_counts": layers,
        "source_locators": sorted(locators),
    }


def sanitize_readback_payload(
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    instance_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Create the public-safe allowlisted result; raw numeric values stay private."""

    instances = _array(payload.get("instances"), "payload.instances")
    record_families: dict[str, list[Mapping[str, Any]]] = {
        "friction": [],
        "armature": [],
    }
    control_names = [
        str(name)
        for name in plan["readback_schema"]["required_control_drive_fields"]
    ]
    for name in control_names:
        record_families[f"control_drive.{name}"] = []
    solver_names = [
        str(name)
        for name in plan["readback_schema"]["required_solver_runtime_fields"]
    ]
    for name in solver_names:
        record_families[f"solver_runtime.{name}"] = []

    for raw_instance in instances:
        instance = _mapping(raw_instance, "instance")
        for raw_record in _array(instance["dof_records"], "dof_records"):
            record = _mapping(raw_record, "dof record")
            for family in ("friction", "armature"):
                record_families[family].append(_mapping(record[family], family))
            control = _mapping(record["control_drive"], "control_drive")
            for name in control_names:
                record_families[f"control_drive.{name}"].append(
                    _mapping(control[name], name)
                )
        solver = _mapping(instance["solver_runtime"], "solver_runtime")
        for name in solver_names:
            record_families[f"solver_runtime.{name}"].append(
                _mapping(solver[name], name)
            )

    matrix = {
        family: _availability_counts(values)
        for family, values in sorted(record_families.items())
    }
    solver_field_classifications: dict[str, dict[str, str]] = {}
    for name in solver_names:
        field_matrix = matrix[f"solver_runtime.{name}"]
        availability = _mapping(
            field_matrix["availability_counts"], "availability_counts"
        )
        fully_available = int(availability["available"]) == int(
            field_matrix["record_count"]
        )
        if not fully_available:
            solver_field_classifications[name] = {
                "classification": "UNMAPPABLE",
                "reason": (
                    "At least one frozen instance lacks an authored, resolved, or "
                    "runtime-effective layer; Freeze A classifies incomplete or "
                    "ambiguous evidence as UNMAPPABLE."
                ),
            }
        else:
            solver_field_classifications[name] = {
                "classification": "SENSITIVITY_ONLY",
                "reason": (
                    "Observed configuration evidence may be used only as a sensitivity "
                    "variable, not as a cross-backend equivalent parameter."
                ),
            }
    solver_has_unavailable = any(
        row["classification"] == "UNMAPPABLE"
        for row in solver_field_classifications.values()
    )
    semantic = {
        "friction": {
            "classification": "UNMAPPABLE",
            "reason": (
                "Pinned wrapper and authoritative API labels conflict for non-viscous "
                "slots; numeric equality or closeness cannot establish MuJoCo equivalence."
            ),
            "source_locators": matrix["friction"]["source_locators"],
        },
        "armature": {
            "classification": "UNMAPPABLE",
            "reason": (
                "This descriptive OVPhysX-only readback does not by itself verify every "
                "cross-backend physical/action-law requirement for DIRECT_TRANSFERABLE."
            ),
            "source_locators": matrix["armature"]["source_locators"],
        },
        "control_drive": {
            "classification": "UNMAPPABLE",
            "reason": (
                "Same names or numbers do not prove equal controller, transmission, "
                "limit, or passive-force laws across engines."
            ),
            "source_locators": sorted(
                {
                    locator
                    for key, row in matrix.items()
                    if key.startswith("control_drive.")
                    for locator in row["source_locators"]
                }
            ),
        },
        "solver_runtime": {
            "classification": (
                "UNMAPPABLE" if solver_has_unavailable else "SENSITIVITY_ONLY"
            ),
            "reason": (
                "Incomplete or ambiguous solver evidence takes precedence and is "
                "UNMAPPABLE; only fully available solver evidence could remain "
                "SENSITIVITY_ONLY, never an equivalent parameter."
                if solver_has_unavailable
                else "Solver, iteration, stabilization, tolerance, substep, and "
                "integrator settings are sensitivity variables, not equivalent "
                "parameters."
            ),
            "field_classifications": solver_field_classifications,
            "source_locators": sorted(
                {
                    locator
                    for key, row in matrix.items()
                    if key.startswith("solver_runtime.")
                    for locator in row["source_locators"]
                }
            ),
        },
    }
    instance_hash_rows: list[dict[str, Any]] = []
    for raw_instance in instances:
        instance = _mapping(raw_instance, "instance")
        case_id = str(instance["case_id"])
        digest = instance_file_sha256.get(case_id)
        _sha256(digest, f"instance_file_sha256[{case_id}]")
        instance_hash_rows.append(
            {
                "case_id": case_id,
                "worker_payload_sha256": digest,
                "canonical_vector_sha256": _json_copy(
                    instance["canonical_vector_sha256"]
                ),
            }
        )
    first_provenance = _mapping(
        _mapping(instances[0], "first instance")["provenance"], "provenance"
    )
    summary = {
        "schema_version": 1,
        "experiment_id": payload["experiment_id"],
        "classification": plan["classification"],
        "status": payload["status"],
        "coverage": _json_copy(payload["coverage"]),
        "reproducibility": _json_copy(payload["reproducibility"]),
        "execution_observed": {
            "fresh_process_count": 4,
            "instances_per_hand": {"left": 2, "right": 2},
            "initialization_state_write_performed_all_instances": True,
            "user_or_experimental_trace_simulation_step_call_count": 0,
            "experimental_control_command_count": 0,
            "experimental_target_write_count": 0,
            "trajectory_sample_count": 0,
            "trace_time_s": 0.0,
            "gpu_warmup_disclosed_separately": True,
            "internal_initialization_advance_not_inferred": True,
        },
        "field_availability_matrix": matrix,
        "semantic_classification": semantic,
        "instance_content_hashes": instance_hash_rows,
        "safe_provenance": {
            field: first_provenance[field]
            for field in (
                "source_revision",
                "source_tree",
                "source_archive_sha256",
                "probe_source_sha256",
                "freeze_a_config_sha256",
                "gate0_manifest_file_sha256",
                "gate0_manifest_semantic_sha256",
                "asset_commit",
                "asset_git_tree",
                "canonical_lf_asset_tree_sha256",
                "python_version",
                "isaac_lab_version",
                "isaac_sim_version",
                "physx_version",
                "pytorch_version",
                "device",
                "gpu_name",
                "driver_version",
            )
        },
        "private_raw_evidence": {
            "included_in_private_bundle": True,
            "public_values_omitted": True,
            "omitted_identity_field_count": len(
                plan["output_policy"]["private_fields"]
            ),
        },
        "freeze_b_required": True,
        "formal_gate0": {
            "status": "DIVERGENT",
            "pass_ready": False,
            "unchanged_by_this_descriptive_readback": True,
            "formal_bundle_root_sha256": plan["frozen_context"][
                "formal_gate0_bundle_root_sha256"
            ],
        },
        "claim_boundary": _json_copy(plan["claim_boundary"]),
    }
    forbidden = {
        "machine_id",
        "gpu_uuid",
        "session_id",
        "fresh_process_id",
        "worker_pid",
    }
    rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if any(f'"{name}"' in rendered for name in forbidden):
        raise EffectiveReadbackError("sanitized summary contains a private field")
    return summary


def render_readback_report(summary: Mapping[str, Any]) -> str:
    """Render a compact public-safe Markdown report from the sanitized summary."""

    coverage = _mapping(summary.get("coverage"), "summary.coverage")
    reproduction = _mapping(summary.get("reproducibility"), "summary.reproducibility")
    provenance = _mapping(summary.get("safe_provenance"), "summary.safe_provenance")
    semantics = _mapping(summary.get("semantic_classification"), "summary.semantic_classification")
    lines = [
        "# OVPhysX Effective-Parameter Readback (Freeze A)",
        "",
        f"- Status: `{summary['status']}`",
        "- Scope: descriptive, kit-less OVPhysX, fixed-base Wave hands, pre-trace readback only",
        (
            "- Coverage: "
            f"`{coverage['covered_joint_count_total']}/{coverage['canonical_joint_count_total']}` "
            "canonical left/right joints"
        ),
        "- Fresh processes: `4` (two per hand)",
        (
            "- Exact repeatability: mapping="
            f"`{str(reproduction['mapping_exact']).lower()}`, solver evidence records="
            f"`{str(reproduction['solver_runtime_exact']).lower()}`, vectors="
            f"`{str(reproduction['canonical_numeric_vector_hashes_exact']).lower()}`"
        ),
        "- Experimental trace steps/commands/targets/samples: `0 / 0 / 0 / 0`",
        "- Adapter initialization/reset zero-state and zero-target writes: disclosed and performed",
        "- Backend-internal warmup/advance: disclosed separately; exact internal advance is not exposed or inferred",
        "",
        "## Conservative semantic boundary",
        "",
    ]
    for family in ("friction", "armature", "control_drive", "solver_runtime"):
        row = _mapping(semantics[family], f"semantic_classification.{family}")
        lines.append(
            f"- `{family}`: **{row['classification']}** — {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Frozen identities",
            "",
            f"- Source revision/tree: `{provenance['source_revision']}` / `{provenance['source_tree']}`",
            f"- Freeze A config SHA-256: `{provenance['freeze_a_config_sha256']}`",
            f"- Gate 0 manifest semantic SHA-256: `{provenance['gate0_manifest_semantic_sha256']}`",
            f"- Canonical asset tree SHA-256: `{provenance['canonical_lf_asset_tree_sha256']}`",
            "",
            "## Claim boundary",
            "",
            "This is descriptive configuration evidence only. It does not identify the cause of a trajectory difference, establish cross-engine parameter equivalence, infer realized torque, or support a backend/upstream bug claim.",
            "",
            "Formal Gate 0 remains **DIVERGENT** with `pass_ready=false`. Freeze B is still required before any trajectory intervention.",
            "",
            "The exact raw records and private machine/process identifiers remain in the private content-hashed bundle; this report intentionally omits them.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    for private in ("machine_id", "gpu_uuid", "session_id", "fresh_process_id"):
        if f"`{private}`" in rendered or f'"{private}"' in rendered:
            raise EffectiveReadbackError("public report contains a private field")
    return rendered


__all__ = [
    "EffectiveReadbackError",
    "assemble_readback_payload",
    "build_readback_instance",
    "canonical_json_sha256",
    "canonical_vector_hash",
    "expected_instance",
    "load_frozen_contract",
    "load_json_strict",
    "render_readback_report",
    "sanitize_readback_payload",
    "sha256_file",
    "validate_readback_instance",
]
