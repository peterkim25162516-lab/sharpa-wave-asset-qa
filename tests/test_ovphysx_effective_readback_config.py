from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "parity" / "ovphysx_effective_readback.json"
GATE0_PATH = ROOT / "configs" / "parity" / "gate0.json"


class ReadbackValidationError(ValueError):
    """Raised by the test-only executable contract validator."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_keys(value: dict[str, Any], required: list[str], path: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise ReadbackValidationError(f"{path} missing required keys: {missing}")


def _validate_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReadbackValidationError(f"{path} contains a non-finite value")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")


def _canonical_joint_names(gate0: dict[str, Any]) -> dict[str, list[str]]:
    return {hand["side"]: hand["joint_names"] for hand in gate0["hands"]}


def _evidence(
    effective: Any,
    *,
    unit: str,
    semantics: str = "synthetic_contract_fixture",
) -> dict[str, Any]:
    return {
        "availability": "available",
        "authored_value": effective,
        "resolved_value": effective,
        "runtime_effective_value": effective,
        "unavailable_layers": [],
        "unit": unit,
        "semantics": semantics,
        "source_locator": "synthetic://contract-fixture",
        "unavailable_reason": None,
    }


def _validate_evidence(
    plan: dict[str, Any], value: dict[str, Any], path: str
) -> None:
    schema = plan["readback_schema"]
    _require_keys(value, schema["required_evidence_fields"], path)
    availability = value["availability"]
    if availability not in schema["allowed_availability"]:
        raise ReadbackValidationError(f"{path} has invalid availability")
    if availability == "available":
        if value["runtime_effective_value"] is None:
            raise ReadbackValidationError(f"{path} lacks runtime-effective readback")
        if value["unavailable_layers"]:
            raise ReadbackValidationError(f"{path} contradicts available status")
        if value["unavailable_reason"] is not None:
            raise ReadbackValidationError(f"{path} has a spurious unavailable reason")
    else:
        if not value["unavailable_layers"]:
            raise ReadbackValidationError(f"{path} must name unavailable layers")
        if not isinstance(value["unavailable_reason"], str) or not value[
            "unavailable_reason"
        ].strip():
            raise ReadbackValidationError(f"{path} must explain unavailability")
        null_layers = {
            name
            for name, key in (
                ("authored", "authored_value"),
                ("resolved", "resolved_value"),
                ("runtime_effective", "runtime_effective_value"),
            )
            if value[key] is None
        }
        if not null_layers.issubset(set(value["unavailable_layers"])):
            raise ReadbackValidationError(f"{path} has an unmarked unavailable layer")
    if not isinstance(value["source_locator"], str) or not value[
        "source_locator"
    ].strip():
        raise ReadbackValidationError(f"{path} lacks a source locator")


def _vector_value(record: dict[str, Any], key: str) -> Any:
    if key == "friction_runtime_effective":
        return record["friction"]["runtime_effective_value"]
    if key == "armature_runtime_effective":
        return record["armature"]["runtime_effective_value"]
    control_key = key.removesuffix("_runtime_effective")
    return record["control_drive"][control_key]["runtime_effective_value"]


def _canonical_vector_hash(records: list[dict[str, Any]], key: str) -> str:
    values = [_vector_value(record, key) for record in records]
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_dof_record(side: str, name: str, index: int) -> dict[str, Any]:
    return {
        "mapping": {
            "side": side,
            "canonical_joint_name": name,
            "backend_joint_name": f"synthetic_{name}",
            "backend_joint_index": index,
            "joint_prim_path": f"/synthetic/{side}/{name}",
            "joint_type": "revolute",
            "axis": [1.0, 0.0, 0.0],
            "canonical_sign": 1.0,
            "canonical_offset_rad": 0.0,
            "canonical_unit": "rad",
            "source_locator": "synthetic://mapping",
        },
        "friction": _evidence(
            0.0,
            unit="synthetic_unit",
            semantics="synthetic_not_a_real_ovphysx_value",
        ),
        "armature": _evidence(
            0.0,
            unit="kg*m^2",
            semantics="synthetic_not_a_real_ovphysx_value",
        ),
        "control_drive": {
            "controller_model": _evidence("synthetic_pd", unit="enum"),
            "kp": _evidence(1.0, unit="synthetic_stiffness_unit"),
            "kd": _evidence(1.0, unit="synthetic_damping_unit"),
            "backend_drive_stiffness": _evidence(
                1.0, unit="synthetic_stiffness_unit"
            ),
            "backend_drive_damping": _evidence(
                1.0, unit="synthetic_damping_unit"
            ),
            "passive_joint_damping": _evidence(
                0.0, unit="synthetic_damping_unit"
            ),
            "gear_or_transmission": _evidence(1.0, unit="ratio"),
            "effort_limit": _evidence(1.0, unit="synthetic_effort_unit"),
            "velocity_limit": _evidence(1.0, unit="rad/s"),
            "target_interpretation": _evidence(
                "synthetic_canonical_position", unit="enum"
            ),
        },
    }


def _make_solver_runtime() -> dict[str, Any]:
    return {
        "solver_type": _evidence("synthetic_solver", unit="enum"),
        "position_iterations": _evidence(1, unit="count"),
        "velocity_iterations": _evidence(1, unit="count"),
        "physics_dt_s": _evidence(0.001, unit="s"),
        "substeps": _evidence(1, unit="count"),
        "integrator": _evidence("synthetic_integrator", unit="enum"),
        "stabilization": _evidence(False, unit="boolean"),
        "tolerances": _evidence(
            {"synthetic_tolerance": 0.0}, unit="synthetic_unit"
        ),
        "sleep_threshold": _evidence(0.0, unit="synthetic_unit"),
        "execution_pipeline": _evidence("synthetic_gpu", unit="enum"),
    }


def _make_payload(
    plan: dict[str, Any], gate0: dict[str, Any]
) -> dict[str, Any]:
    joint_names = _canonical_joint_names(gate0)
    instances: list[dict[str, Any]] = []
    for expected in plan["expected_instances"]:
        side = expected["side"]
        index = expected["instance_index"]
        records = [
            _make_dof_record(side, name, dof_index)
            for dof_index, name in enumerate(joint_names[side])
        ]
        vector_hashes = {
            key: _canonical_vector_hash(records, key)
            for key in plan["readback_schema"][
                "canonical_vector_sha256_required_keys"
            ]
        }
        instances.append(
            {
                "case_id": expected["case_id"],
                "side": side,
                "instance_index": index,
                "fresh_instance_id": f"synthetic-{side}-instance-{index}",
                "fresh_process_id": f"synthetic-{side}-process-{index}",
                "readback_only_after_initialization": True,
                "initialization_state_write_performed": True,
                "experimental_trace_dynamics_advance_performed": False,
                "experimental_control_command_applied": False,
                "experimental_target_write_performed": False,
                "user_trace_simulation_step_call_count": 0,
                "trajectory_sample_count": 0,
                "trace_time_s": 0.0,
                "gpu_warmup_outside_trace": _evidence(
                    True,
                    unit="boolean",
                    semantics="synthetic_initialization_only_not_trace_dynamics",
                ),
                "initialization_internal_advance_exposure": {
                    "availability": "unavailable",
                    "authored_value": None,
                    "resolved_value": None,
                    "runtime_effective_value": None,
                    "unavailable_layers": [
                        "authored",
                        "resolved",
                        "runtime_effective",
                    ],
                    "unit": "s",
                    "semantics": "synthetic_exact_internal_advance_not_exposed",
                    "source_locator": "synthetic://contract-fixture",
                    "unavailable_reason": "synthetic backend API does not quantify internal initialization advance",
                },
                "provenance": {
                    "source_revision": "synthetic-source-revision",
                    "source_tree": "synthetic-source-tree",
                    "source_archive_sha256": "a" * 64,
                    "source_snapshot_sha256": "d" * 64,
                    "probe_source_sha256": "b" * 64,
                    "pipeline_module_sha256": "e" * 64,
                    "freeze_a_config_sha256": "c" * 64,
                    "gate0_manifest_file_sha256": plan["frozen_context"][
                        "gate0_manifest_file_sha256"
                    ],
                    "gate0_manifest_semantic_sha256": plan["frozen_context"][
                        "gate0_manifest_semantic_sha256"
                    ],
                    "asset_commit": plan["frozen_context"]["asset_commit"],
                    "asset_git_tree": plan["frozen_context"]["asset_git_tree"],
                    "canonical_lf_asset_tree_sha256": plan["frozen_context"][
                        "canonical_lf_asset_tree_sha256"
                    ],
                    "python_version": "synthetic",
                    "isaac_lab_version": "synthetic",
                    "isaac_sim_version": "synthetic",
                    "physx_version": "synthetic",
                    "pytorch_version": "synthetic",
                    "device": "synthetic",
                    "gpu_name": "synthetic",
                    "gpu_uuid": "synthetic-private",
                    "driver_version": "synthetic",
                    "machine_id": "synthetic-private",
                    "session_id": f"synthetic-session-{side}-{index}",
                    "recorded_at_utc": "2000-01-01T00:00:00Z",
                },
                "solver_runtime": _make_solver_runtime(),
                "dof_records": records,
                "canonical_vector_sha256": vector_hashes,
            }
        )

    return {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "status": "READBACK_VALID",
        "instances": instances,
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
        "claim_boundary": plan["claim_boundary"],
        "freeze_b_required": True,
    }


def _validate_payload(
    plan: dict[str, Any], gate0: dict[str, Any], payload: dict[str, Any]
) -> None:
    schema = plan["readback_schema"]
    _require_keys(payload, schema["required_top_level_fields"], "readback")
    _validate_finite(payload)
    if payload["status"] != plan["fail_closed"]["valid_status"]:
        raise ReadbackValidationError("readback is not valid")
    if payload["experiment_id"] != plan["experiment_id"]:
        raise ReadbackValidationError("experiment id mismatch")
    if payload["freeze_b_required"] is not True:
        raise ReadbackValidationError("Freeze B must remain required")

    expected_rows = {
        (row["side"], row["instance_index"]): row
        for row in plan["expected_instances"]
    }
    instances = payload["instances"]
    if len(instances) != plan["execution_contract"]["total_instance_count"]:
        raise ReadbackValidationError("unexpected instance count")
    actual_rows = {(row["side"], row["instance_index"]): row for row in instances}
    if len(actual_rows) != len(instances) or set(actual_rows) != set(expected_rows):
        raise ReadbackValidationError("instances do not match the frozen matrix")
    instance_ids = [row["fresh_instance_id"] for row in instances]
    process_ids = [row["fresh_process_id"] for row in instances]
    if len(set(instance_ids)) != len(instance_ids):
        raise ReadbackValidationError("fresh instance id reused")
    if len(set(process_ids)) != len(process_ids):
        raise ReadbackValidationError("fresh process id reused")

    expected_names = _canonical_joint_names(gate0)
    global_coverage: set[tuple[str, str]] = set()
    for pair, instance in actual_rows.items():
        _require_keys(instance, schema["required_instance_fields"], f"instance{pair}")
        expected = expected_rows[pair]
        if instance["case_id"] != expected["case_id"]:
            raise ReadbackValidationError("case id mismatch")
        if (
            instance["readback_only_after_initialization"] is not True
            or instance["initialization_state_write_performed"] is not True
            or instance["experimental_trace_dynamics_advance_performed"] is not False
            or instance["experimental_control_command_applied"] is not False
            or instance["experimental_target_write_performed"] is not False
            or instance["user_trace_simulation_step_call_count"] != 0
            or instance["trajectory_sample_count"] != 0
            or instance["trace_time_s"] != 0.0
        ):
            raise ReadbackValidationError(
                "readback advanced or commanded user/experimental trace dynamics"
            )
        _validate_evidence(
            plan,
            instance["gpu_warmup_outside_trace"],
            f"instance{pair}.gpu_warmup_outside_trace",
        )
        _validate_evidence(
            plan,
            instance["initialization_internal_advance_exposure"],
            f"instance{pair}.initialization_internal_advance_exposure",
        )
        _require_keys(
            instance["provenance"],
            schema["required_provenance_fields"],
            f"instance{pair}.provenance",
        )
        provenance = instance["provenance"]
        for field in (
            "gate0_manifest_file_sha256",
            "gate0_manifest_semantic_sha256",
            "asset_commit",
            "asset_git_tree",
            "canonical_lf_asset_tree_sha256",
        ):
            if provenance[field] != plan["frozen_context"][field]:
                raise ReadbackValidationError(f"frozen provenance mismatch: {field}")

        _require_keys(
            instance["solver_runtime"],
            schema["required_solver_runtime_fields"],
            f"instance{pair}.solver_runtime",
        )
        for field in schema["required_solver_runtime_fields"]:
            _validate_evidence(
                plan,
                instance["solver_runtime"][field],
                f"instance{pair}.solver_runtime.{field}",
            )

        records = instance["dof_records"]
        if len(records) != plan["coverage_contract"][
            "canonical_joint_count_per_hand"
        ]:
            raise ReadbackValidationError("wrong per-instance DoF count")
        names: list[str] = []
        backend_indices: list[int] = []
        for record_index, record in enumerate(records):
            record_path = f"instance{pair}.dof_records[{record_index}]"
            _require_keys(record, schema["required_dof_record_fields"], record_path)
            _require_keys(
                record["mapping"],
                schema["required_mapping_fields"],
                f"{record_path}.mapping",
            )
            if record["mapping"]["side"] != instance["side"]:
                raise ReadbackValidationError("mapping side mismatch")
            names.append(record["mapping"]["canonical_joint_name"])
            backend_indices.append(record["mapping"]["backend_joint_index"])
            _validate_evidence(plan, record["friction"], f"{record_path}.friction")
            _validate_evidence(plan, record["armature"], f"{record_path}.armature")
            _require_keys(
                record["control_drive"],
                schema["required_control_drive_fields"],
                f"{record_path}.control_drive",
            )
            for field in schema["required_control_drive_fields"]:
                _validate_evidence(
                    plan,
                    record["control_drive"][field],
                    f"{record_path}.control_drive.{field}",
                )
        if names != expected_names[instance["side"]]:
            raise ReadbackValidationError("canonical mapping is incomplete or unordered")
        if len(set(names)) != len(names):
            raise ReadbackValidationError("duplicate canonical joint")
        if len(set(backend_indices)) != len(backend_indices):
            raise ReadbackValidationError("duplicate backend joint index")
        global_coverage.update((instance["side"], name) for name in names)

        hashes = instance["canonical_vector_sha256"]
        _require_keys(
            hashes,
            schema["canonical_vector_sha256_required_keys"],
            f"instance{pair}.canonical_vector_sha256",
        )
        for key in schema["canonical_vector_sha256_required_keys"]:
            if hashes[key] != _canonical_vector_hash(records, key):
                raise ReadbackValidationError(f"canonical vector hash mismatch: {key}")

    if len(global_coverage) != plan["coverage_contract"][
        "canonical_joint_count_total"
    ]:
        raise ReadbackValidationError("canonical coverage is not 44 of 44")
    if payload["coverage"] != {
        "canonical_joint_count_per_hand": {"left": 22, "right": 22},
        "canonical_joint_count_total": 44,
        "covered_joint_count_total": 44,
        "complete": True,
    }:
        raise ReadbackValidationError("declared coverage does not match 44 of 44")

    for side in plan["scope"]["hands"]:
        first = actual_rows[(side, 1)]
        second = actual_rows[(side, 2)]
        if first["dof_records"] != second["dof_records"]:
            raise ReadbackValidationError("fresh-instance DoF readback mismatch")
        if first["solver_runtime"] != second["solver_runtime"]:
            raise ReadbackValidationError("fresh-instance solver readback mismatch")
        if first["canonical_vector_sha256"] != second["canonical_vector_sha256"]:
            raise ReadbackValidationError("fresh-instance vector hash mismatch")
    if payload["reproducibility"] != {
        "mapping_exact": True,
        "solver_runtime_exact": True,
        "canonical_numeric_vector_hashes_exact": True,
    }:
        raise ReadbackValidationError("declared reproducibility is not exact")


def _classify_semantics(
    plan: dict[str, Any], family: str, facts: dict[str, Any]
) -> str:
    classes = set(plan["semantic_classification"]["allowed_classes"])
    if facts.get("availability") != "available" or facts.get("ambiguous", False):
        result = "UNMAPPABLE"
    elif family == "solver_runtime":
        result = "SENSITIVITY_ONLY"
    elif (
        family == "friction"
        and facts.get("physx_semantics")
        in plan["semantic_classification"][
            "legacy_or_unitless_physx_friction_semantics"
        ]
    ):
        value = facts.get("runtime_effective_value")
        exact_zero = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0.0
        )
        result = (
            "ZERO_ONLY"
            if exact_zero
            and facts.get("source_api_semantics_verify_zero_disables_term") is True
            else "UNMAPPABLE"
        )
    elif all(
        facts.get(requirement) is True
        for requirement in plan["semantic_classification"][
            "direct_transferable_requires_all"
        ]
    ):
        result = "DIRECT_TRANSFERABLE"
    elif all(
        facts.get(requirement) is True
        for requirement in plan["semantic_classification"][
            "zero_only_requires_all"
        ]
    ):
        result = "ZERO_ONLY"
    else:
        result = "UNMAPPABLE"
    assert result in classes
    return result


def test_freeze_a_is_two_hands_times_two_fresh_instances_and_readback_only() -> None:
    plan = _load_json(PLAN_PATH)

    assert plan["plan_stage"] == (
        "freeze_a_before_any_new_ovphysx_parameter_value_is_observed"
    )
    assert [(row["side"], row["instance_index"]) for row in plan["expected_instances"]] == [
        ("left", 1),
        ("left", 2),
        ("right", 1),
        ("right", 2),
    ]
    execution = plan["execution_contract"]
    assert execution["fresh_instances_per_hand"] == 2
    assert execution["total_instance_count"] == 4
    assert execution["fresh_process_per_instance"] is True
    assert execution["user_or_experimental_trace_dynamics_advance_allowed"] is False
    assert execution["required_user_trace_simulation_step_call_count"] == 0
    assert execution["required_trajectory_sample_count"] == 0
    assert execution["required_trace_time_s"] == 0.0
    assert execution["initialization_reset_zero_state_and_target_writes_allowed"] is True
    assert execution["required_initialization_state_write_performed"] is True
    assert execution["experimental_control_command_allowed"] is False
    assert execution["experimental_target_write_allowed"] is False
    assert execution["required_experimental_target_write_performed"] is False
    assert execution[
        "gpu_warmup_and_internal_initialization_advance_must_be_disclosed_separately"
    ] is True


def test_frozen_context_matches_canonical_gate0_manifest() -> None:
    plan = _load_json(PLAN_PATH)
    gate0 = _load_json(GATE0_PATH)

    assert plan["frozen_context"]["asset_commit"] == gate0["provenance"]["commit"]
    assert plan["frozen_context"]["asset_git_tree"] == gate0["provenance"][
        "asset_git_tree"
    ]
    assert plan["frozen_context"]["canonical_lf_asset_tree_sha256"] == gate0[
        "provenance"
    ]["canonical_lf_asset_tree_sha256"]
    names = _canonical_joint_names(gate0)
    assert set(names) == {"left", "right"}
    assert all(len(side_names) == 22 for side_names in names.values())
    assert len({(side, name) for side, side_names in names.items() for name in side_names}) == 44


def test_required_field_families_and_hashes_are_closed() -> None:
    plan = _load_json(PLAN_PATH)
    schema = plan["readback_schema"]

    assert schema["required_dof_record_fields"] == [
        "mapping",
        "friction",
        "armature",
        "control_drive",
    ]
    assert {
        "controller_model",
        "kp",
        "kd",
        "backend_drive_stiffness",
        "backend_drive_damping",
        "passive_joint_damping",
        "gear_or_transmission",
        "effort_limit",
        "velocity_limit",
        "target_interpretation",
    } == set(schema["required_control_drive_fields"])
    assert {
        "solver_type",
        "position_iterations",
        "velocity_iterations",
        "physics_dt_s",
        "substeps",
        "integrator",
        "stabilization",
        "tolerances",
        "sleep_threshold",
        "execution_pipeline",
    } == set(schema["required_solver_runtime_fields"])
    assert len(schema["canonical_vector_sha256_required_keys"]) == 10
    assert plan["value_contract"]["unavailable_is_never_interpreted_as_zero"] is True


def test_synthetic_contract_payload_is_valid_and_covers_44_of_44() -> None:
    plan = _load_json(PLAN_PATH)
    gate0 = _load_json(GATE0_PATH)
    payload = _make_payload(plan, gate0)

    _validate_payload(plan, gate0, payload)

    assert len(payload["instances"]) == 4
    assert sum(len(row["dof_records"]) for row in payload["instances"]) == 88
    assert payload["coverage"]["covered_joint_count_total"] == 44


def _delete_required_field(payload: dict[str, Any]) -> None:
    del payload["instances"][0]["provenance"]["physx_version"]


def _insert_nonfinite_value(payload: dict[str, Any]) -> None:
    payload["instances"][0]["dof_records"][0]["friction"][
        "runtime_effective_value"
    ] = math.nan


def _drop_joint(payload: dict[str, Any]) -> None:
    payload["instances"][0]["dof_records"].pop()


def _duplicate_canonical_joint(payload: dict[str, Any]) -> None:
    records = payload["instances"][0]["dof_records"]
    records[-1]["mapping"]["canonical_joint_name"] = records[0]["mapping"][
        "canonical_joint_name"
    ]


def _reuse_process(payload: dict[str, Any]) -> None:
    payload["instances"][1]["fresh_process_id"] = payload["instances"][0][
        "fresh_process_id"
    ]


def _advance_dynamics(payload: dict[str, Any]) -> None:
    payload["instances"][0]["experimental_trace_dynamics_advance_performed"] = True
    payload["instances"][0]["user_trace_simulation_step_call_count"] = 1


def _omit_internal_advance_disclosure(payload: dict[str, Any]) -> None:
    del payload["instances"][0]["initialization_internal_advance_exposure"]


def _deny_initialization_write(payload: dict[str, Any]) -> None:
    payload["instances"][0]["initialization_state_write_performed"] = False


def _change_repeat_without_rehash(payload: dict[str, Any]) -> None:
    payload["instances"][1]["dof_records"][0]["armature"][
        "runtime_effective_value"
    ] = 0.25


@pytest.mark.parametrize(
    "mutate",
    [
        _delete_required_field,
        _insert_nonfinite_value,
        _drop_joint,
        _duplicate_canonical_joint,
        _reuse_process,
        _advance_dynamics,
        _omit_internal_advance_disclosure,
        _deny_initialization_write,
        _change_repeat_without_rehash,
    ],
    ids=[
        "missing-field",
        "nonfinite",
        "missing-joint",
        "duplicate-joint",
        "reused-process",
        "dynamics-advance",
        "missing-internal-advance-disclosure",
        "missing-initialization-write-attestation",
        "fresh-instance-mismatch",
    ],
)
def test_readback_contract_fails_closed(
    mutate: Callable[[dict[str, Any]], None]
) -> None:
    plan = _load_json(PLAN_PATH)
    gate0 = _load_json(GATE0_PATH)
    payload = _make_payload(plan, gate0)
    mutate(payload)

    with pytest.raises(ReadbackValidationError):
        _validate_payload(plan, gate0, payload)


def test_explicit_unavailable_is_valid_evidence_but_semantically_unmappable() -> None:
    plan = _load_json(PLAN_PATH)
    unavailable = _evidence(0.0, unit="unknown")
    unavailable.update(
        {
            "availability": "unavailable",
            "authored_value": None,
            "resolved_value": None,
            "runtime_effective_value": None,
            "unavailable_layers": ["authored", "resolved", "runtime_effective"],
            "unavailable_reason": "synthetic API omission",
        }
    )

    _validate_evidence(plan, unavailable, "synthetic.unavailable")
    assert _classify_semantics(
        plan,
        "armature",
        {"availability": "unavailable", "runtime_effective_value": None},
    ) == "UNMAPPABLE"


def test_semantic_classification_boundaries_are_fail_closed() -> None:
    plan = _load_json(PLAN_PATH)
    direct = {
        "availability": "available",
        "runtime_effective_readback_available": True,
        "same_physical_quantity": True,
        "same_unit": True,
        "same_canonical_coordinate": True,
        "same_action_law": True,
    }
    same_name_only = {
        **direct,
        "same_action_law": False,
        "same_name": True,
        "numerically_close": True,
    }
    legacy_friction = {
        "availability": "available",
        "physx_semantics": "legacy_unitless_joint_friction_coefficient",
        "runtime_effective_value": 0.5,
        "source_api_semantics_verify_zero_disables_term": True,
    }
    legacy_friction_zero = {
        **legacy_friction,
        "runtime_effective_value": 0.0,
    }

    assert _classify_semantics(plan, "armature", direct) == "DIRECT_TRANSFERABLE"
    assert _classify_semantics(plan, "armature", same_name_only) == "UNMAPPABLE"
    assert _classify_semantics(plan, "friction", legacy_friction) == "UNMAPPABLE"
    assert _classify_semantics(plan, "friction", legacy_friction_zero) == "ZERO_ONLY"
    assert _classify_semantics(plan, "solver_runtime", direct) == "SENSITIVITY_ONLY"
    assert _classify_semantics(
        plan, "armature", {**direct, "ambiguous": True}
    ) == "UNMAPPABLE"


def test_freeze_b_is_still_required_before_any_trajectory() -> None:
    plan = _load_json(PLAN_PATH)
    freeze_b = plan["freeze_b_contract"]

    assert freeze_b["state_at_freeze_a"] == "REQUIRED_NOT_CREATED"
    assert freeze_b["required_before_any_trajectory"] is True
    assert freeze_b["trajectory_authorized_by_freeze_a"] is False
    assert freeze_b["manual_value_tuning_allowed"] is False
    assert freeze_b["unforeseen_or_ambiguous_semantics_policy"] == (
        "STOP_SEMANTICS_UNMAPPABLE"
    )
    assert {
        "freeze_a_config_sha256",
        "readback_bundle_root_sha256",
        "exact_per_joint_treatment_values",
        "both_hands_run_matrix",
        "preregistered_metrics",
        "preregistered_decision_thresholds",
        "bounded_compute_budget",
    }.issubset(freeze_b["required_frozen_fields"])


def test_claim_boundary_forbids_causal_equivalence_and_bug_claims() -> None:
    plan = _load_json(PLAN_PATH)
    boundary = plan["claim_boundary"]
    forbidden = " ".join(boundary["forbidden"]).lower()

    assert "descriptive evidence" in boundary["allowed"].lower()
    assert "without any user or experimental trace simulation.step call" in boundary[
        "allowed"
    ].lower()
    assert "backend-internal gpu warmup" in boundary["allowed"].lower()
    assert "causal" in forbidden
    assert "cross-engine parameter equivalence" in forbidden
    assert "backend or upstream bug" in forbidden
    assert "hardware or sim2real" in forbidden
    assert "divergent" in boundary["formal_status_unchanged"].lower()
    assert "pass_ready=false" in boundary["formal_status_unchanged"]
