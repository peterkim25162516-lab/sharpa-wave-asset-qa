from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from wave_asset_qa.parity.effective_readback import (
    EffectiveReadbackError,
    assemble_readback_payload,
    build_readback_instance,
    canonical_json_sha256,
    load_frozen_contract,
    load_json_strict,
    render_readback_report,
    sanitize_readback_payload,
    sha256_file,
    validate_readback_instance,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs" / "parity" / "ovphysx_effective_readback.json"
GATE0 = ROOT / "configs" / "parity" / "gate0.json"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solver_record(
    value: object,
    *,
    authored: bool = True,
    locator: str,
) -> dict[str, object]:
    return {
        "resolved_available": True,
        "resolved_value": value,
        "has_authored_value_opinion": authored,
        "source_locator": locator,
        "unavailable_reason": None,
    }


def _unavailable_solver_record(*, locator: str, reason: str) -> dict[str, object]:
    return {
        "resolved_available": False,
        "resolved_value": None,
        "has_authored_value_opinion": None,
        "source_locator": locator,
        "unavailable_reason": reason,
    }


def _snapshot(side: str, names: tuple[str, ...]) -> dict[str, object]:
    records = []
    for index, name in enumerate(names):
        joint_path = f"/World/Env_0/Robot/{side}/{name}"
        records.append(
            {
                "canonical_id": name,
                "backend_name": name,
                "backend_index": index,
                "joint_prim_path": joint_path,
                "joint_type": "revolute",
                "joint_usd_type_name": "PhysicsRevoluteJoint",
                "joint_axis": "X",
                "joint_axis_authored": True,
                "joint_descriptor_source": "composed_usd_input",
                "joint_descriptor_is_compiled_runtime_readback": False,
                "armature_usd": {
                    "attribute": "physxJoint:armature",
                    "has_authored_value_opinion": True,
                    "authored_value": 0.01 + index * 0.001,
                    "resolved_value": 0.01 + index * 0.001,
                    "unit": "kg_m2_for_revolute_joint",
                    "source": "composed_usd_input",
                    "compiled_runtime_readback": False,
                    "source_locator": f"{joint_path}.physxJoint:armature",
                },
                "legacy_joint_friction_usd": {
                    "attribute": "physxJoint:jointFriction",
                    "has_authored_value_opinion": True,
                    "authored_value": 0.2 + index * 0.001,
                    "resolved_value": 0.2 + index * 0.001,
                    "unit": "documentation_conflict_not_assigned",
                    "source": "composed_usd_input",
                    "compiled_runtime_readback": False,
                    "source_locator": f"{joint_path}.physxJoint:jointFriction",
                    "runtime_binding_equality_check": "not_performed",
                },
                "armature": 0.03 + index * 0.001,
                "armature_unit": "kg_m2_for_revolute_joint",
                "friction_properties_raw": [
                    0.4 + index * 0.001,
                    0.5 + index * 0.001,
                    0.6 + index * 0.001,
                ],
                "runtime_binding_source": "root_view_cpu_numpy_binding",
                "controller_armature": 0.03 + index * 0.001,
                "controller_static_friction": 0.4 + index * 0.001,
                "controller_dynamic_friction": 0.5 + index * 0.001,
                "controller_viscous_friction": 0.6 + index * 0.001,
                "controller_binding_exact_match": {
                    "armature": True,
                    "static_friction": True,
                    "dynamic_friction": True,
                    "viscous_friction": True,
                },
                "controller_stiffness": 1.0 + index,
                "controller_damping": 0.5 + index,
                "controller_effort_limit": 10.0,
                "controller_effort_limit_sim": 10.0,
                "backend_drive_stiffness": 0.0,
                "backend_drive_damping": 0.0,
                "velocity_limit": None,
                "passive_joint_damping": None,
                "gear": None,
            }
        )
    return {
        "readback_contract_version": 1,
        "capture_phase": "post_reset_pre_trace_step",
        "step_index": 0,
        "backend_joint_dynamics": {
            "schema_version": 1,
            "source": "ovphysx_tensor_binding_post_load",
            "runtime_effective_readback": True,
            "capture_phase": "post_reset_pre_trace_step",
            "scope": "all_backend_dofs",
            "joint_count": 22,
            "joint_order": list(names),
            "records": records,
            "controller_consistency_verified": True,
            "controller_joint_dynamics_binding_comparison_performed": True,
            "controller_joint_dynamics_binding_exact_match": {
                "overall": True,
                "armature": True,
                "static_friction": True,
                "dynamic_friction": True,
                "viscous_friction": True,
            },
            "controller_joint_dynamics_binding_match_policy": "exact_float32_value",
            "controller_model": "IdealPDActuator",
            "controller_mode": "explicit_pd_effort",
            "backend_drive_zero_verified": True,
            "backend_drive_zero_abs_tolerance": 1e-8,
            "friction_semantics": {
                "raw_slot_order": ["static", "dynamic", "viscous"],
                "pinned_wrapper_labels": [],
                "authoritative_api_labels": [],
                "semantic_conflict": True,
                "static_dynamic_units": "documentation_conflict_not_assigned",
                "viscous_unit": "N_m_s_per_rad_for_revolute_joint",
                "legacy_equality_check": "not_performed",
                "cross_engine_equivalence": "not_claimed",
            },
            "velocity_limit_readback": {"status": "not_captured_in_this_contract", "value": None},
            "passive_joint_damping_readback": {"status": "not_exposed_separately_by_pinned_ovphysx", "value": None},
            "gear_readback": {"status": "not_exposed_by_pinned_ovphysx_tensor_api", "value": None},
        },
        "physics_solver_contract": {
            "schema_version": 2,
            "source": "composed_usd_input",
            "compiled_runtime_readback": False,
            "capture_phase": "post_runtime_reset_pre_trace_step",
            "physics_scene_path": "/physicsScene",
            "articulation_root_path": "/World/Env_0/Robot",
            "scene": {
                "solver_type": _solver_record(
                    "TGS", locator="live-composed-usd-probe:/physicsScene#physxScene:solverType"
                ),
                "enable_stabilization": _solver_record(
                    True,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:enableStabilization",
                ),
                "min_position_iteration_count": _solver_record(
                    1,
                    authored=False,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:minPositionIterationCount",
                ),
                "max_position_iteration_count": _solver_record(
                    255,
                    authored=False,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:maxPositionIterationCount",
                ),
                "min_velocity_iteration_count": _solver_record(
                    0,
                    authored=False,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:minVelocityIterationCount",
                ),
                "max_velocity_iteration_count": _solver_record(
                    255,
                    authored=False,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:maxVelocityIterationCount",
                ),
                "time_steps_per_second": _solver_record(
                    500,
                    locator="live-composed-usd-probe:/physicsScene#physxScene:timeStepsPerSecond",
                ),
            },
            "articulation": {
                "solver_position_iteration_count": _solver_record(
                    4,
                    locator="live-composed-usd-probe:/World/Env_0/Robot#physxArticulation:solverPositionIterationCount",
                ),
                "solver_velocity_iteration_count": _solver_record(
                    1,
                    locator="live-composed-usd-probe:/World/Env_0/Robot#physxArticulation:solverVelocityIterationCount",
                ),
                "sleep_threshold": _solver_record(
                    0.005,
                    authored=False,
                    locator="live-composed-usd-probe:/World/Env_0/Robot#physxArticulation:sleepThreshold",
                ),
                "stabilization_threshold": _solver_record(
                    0.001,
                    authored=False,
                    locator="live-composed-usd-probe:/World/Env_0/Robot#physxArticulation:stabilizationThreshold",
                ),
            },
            "derived_clamped_requested_iterations": {"position": 4, "velocity": 1},
            "derived_values_are_compiled_runtime_readback": False,
            "stepping": {
                "requested_dt_s": 0.002,
                "cfg_dt_s": 0.002,
                "config_accessor_dt_s": 0.002,
                "config_accessor_semantics": (
                    "SimulationContext.get_physics_dt_returns_SimulationCfg.dt_in_"
                    "the_pinned_kitless_stack_not_compiled_backend_state"
                ),
                "backend_step_api": "ovphysx.PhysX.step_sync",
                "backend_step_calls_per_adapter_step": 1,
                "external_substeps": 1,
                "internal_solver_substeps": None,
                "internal_solver_substeps_status": "not_exposed_not_inferred",
                "physics_scene_time_steps_per_second_controls_effective_dt": False,
                "render_interval": 1000,
                "render_interval_is_physics_substeps": False,
                "gpu_warmup_outside_trace": True,
                "gpu_warmup_timestep_status": "backend_internal_minimal_not_exposed",
            },
        },
    }


def _provenance(plan: dict[str, object], suffix: str) -> dict[str, object]:
    frozen = plan["frozen_context"]
    assert isinstance(frozen, dict)
    return {
        "source_revision": "a" * 40,
        "source_tree": "b" * 40,
        "source_archive_sha256": "c" * 64,
        "source_snapshot_sha256": "d" * 64,
        "probe_source_sha256": "e" * 64,
        "pipeline_module_sha256": "f" * 64,
        "freeze_a_config_sha256": sha256_file(FREEZE),
        "gate0_manifest_file_sha256": frozen["gate0_manifest_file_sha256"],
        "gate0_manifest_semantic_sha256": frozen["gate0_manifest_semantic_sha256"],
        "asset_commit": frozen["asset_commit"],
        "asset_git_tree": frozen["asset_git_tree"],
        "canonical_lf_asset_tree_sha256": frozen["canonical_lf_asset_tree_sha256"],
        "python_version": "3.12.14",
        "isaac_lab_version": "6.1.14",
        "isaac_sim_version": "not-installed-kitless",
        "physx_version": "0.4.13",
        "pytorch_version": "2.10.0+cu128",
        "device": "cuda:0",
        "gpu_name": "NVIDIA A800-SXM4-40GB",
        "gpu_uuid": "GPU-private",
        "driver_version": "570.158.01",
        "machine_id": "private-machine",
        "session_id": "private-session",
        "worker_pid": int(suffix),
        "recorded_at_utc": "2026-08-28T00:00:00Z",
    }


def _instances():
    plan, manifest = load_frozen_contract(FREEZE, GATE0)
    instances = []
    for ordinal, row in enumerate(plan["expected_instances"], start=1):
        side = row["side"]
        hand = manifest.hand(next(item for item in type(manifest.hands[0].side) if item.value == side))
        instances.append(
            build_readback_instance(
                plan=plan,
                manifest=manifest,
                adapter_snapshot=_snapshot(side, hand.joint_names),
                side=side,
                instance_index=row["instance_index"],
                case_id=row["case_id"],
                fresh_instance_id=canonical_json_sha256({"instance": ordinal}),
                fresh_process_id=canonical_json_sha256({"process": ordinal}),
                provenance=_provenance(plan, str(ordinal)),
            )
        )
    return plan, manifest, instances


def test_worker_instance_preserves_distinct_authored_resolved_runtime_layers() -> None:
    plan, manifest, instances = _instances()
    instance = instances[0]
    record = instance["dof_records"][0]

    assert record["armature"]["authored_value"] == pytest.approx(0.01)
    assert record["armature"]["resolved_value"] == pytest.approx(0.01)
    assert record["armature"]["runtime_effective_value"] == {
        "dof_armature_binding": pytest.approx(0.03),
        "idealpd_controller_buffer": pytest.approx(0.03),
        "binding_controller_exact_match": True,
    }
    assert record["friction"]["authored_value"] == {
        "legacy_joint_friction_scalar": pytest.approx(0.2)
    }
    assert record["friction"]["resolved_value"] == {
        "legacy_joint_friction_scalar": pytest.approx(0.2)
    }
    assert record["friction"]["runtime_effective_value"] == {
        "dof_friction_properties_binding": {
            "static": pytest.approx(0.4),
            "dynamic": pytest.approx(0.5),
            "viscous": pytest.approx(0.6),
        },
        "idealpd_controller_buffers": {
            "static": pytest.approx(0.4),
            "dynamic": pytest.approx(0.5),
            "viscous": pytest.approx(0.6),
        },
        "binding_controller_exact_match": {
            "overall": True,
            "static_friction": True,
            "dynamic_friction": True,
            "viscous_friction": True,
        },
    }
    assert "No USD-to-runtime" in record["friction"]["semantics"]
    for field in (
        "controller_model",
        "kp",
        "kd",
        "effort_limit",
        "target_interpretation",
    ):
        evidence = record["control_drive"][field]
        assert evidence["availability"] == "partially_available"
        assert evidence["authored_value"] is None
        assert evidence["resolved_value"] is None
        assert evidence["unavailable_layers"] == ["authored", "resolved"]
    dt_evidence = instance["solver_runtime"]["physics_dt_s"]
    assert dt_evidence["availability"] == "partially_available"
    assert dt_evidence["runtime_effective_value"] is None
    assert "not_compiled_backend_timestep" in dt_evidence["semantics"]
    validate_readback_instance(plan, manifest, instance)


def test_missing_solver_attributes_are_explicitly_unavailable() -> None:
    plan, manifest = load_frozen_contract(FREEZE, GATE0)
    hand = manifest.hands[0]
    snapshot = _snapshot("left", hand.joint_names)
    solver = snapshot["physics_solver_contract"]
    scene = solver["scene"]
    scene["solver_type"] = _unavailable_solver_record(
        locator="live-composed-usd-probe:/physicsScene#physxScene:solverType",
        reason="synthetic absent solverType; no default inferred",
    )
    scene["min_position_iteration_count"] = _unavailable_solver_record(
        locator=(
            "live-composed-usd-probe:/physicsScene#"
            "physxScene:minPositionIterationCount"
        ),
        reason="synthetic absent minimum; no default inferred",
    )
    solver["derived_clamped_requested_iterations"]["position"] = None

    instance = build_readback_instance(
        plan=plan,
        manifest=manifest,
        adapter_snapshot=snapshot,
        side="left",
        instance_index=1,
        case_id="ovphysx.left.effective_readback.r01",
        fresh_instance_id="1" * 64,
        fresh_process_id="2" * 64,
        provenance=_provenance(plan, "1"),
    )

    solver_runtime = instance["solver_runtime"]
    for name in ("solver_type", "position_iterations"):
        evidence = solver_runtime[name]
        assert evidence["availability"] == "unavailable"
        assert evidence["authored_value"] is None
        assert evidence["resolved_value"] is None
        assert evidence["runtime_effective_value"] is None
        assert evidence["unavailable_layers"] == [
            "authored",
            "resolved",
            "runtime_effective",
        ]
        assert "no default" in evidence["unavailable_reason"]


def test_solver_snapshot_requires_exact_stepping_disclosure() -> None:
    plan, manifest = load_frozen_contract(FREEZE, GATE0)
    hand = manifest.hands[0]
    snapshot = _snapshot("left", hand.joint_names)
    del snapshot["physics_solver_contract"]["stepping"]["gpu_warmup_outside_trace"]

    with pytest.raises(EffectiveReadbackError, match="invalid keys"):
        build_readback_instance(
            plan=plan,
            manifest=manifest,
            adapter_snapshot=snapshot,
            side="left",
            instance_index=1,
            case_id="ovphysx.left.effective_readback.r01",
            fresh_instance_id="1" * 64,
            fresh_process_id="2" * 64,
            provenance=_provenance(plan, "1"),
        )


def test_binding_controller_mismatch_is_preserved_not_rejected() -> None:
    plan, manifest = load_frozen_contract(FREEZE, GATE0)
    hand = manifest.hands[0]
    snapshot = _snapshot("left", hand.joint_names)
    dynamics = snapshot["backend_joint_dynamics"]
    dynamics["records"][0]["controller_armature"] = 0.031
    dynamics["records"][0]["controller_binding_exact_match"]["armature"] = False
    dynamics["controller_joint_dynamics_binding_exact_match"] = {
        "overall": False,
        "armature": False,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }
    instance = build_readback_instance(
        plan=plan,
        manifest=manifest,
        adapter_snapshot=snapshot,
        side="left",
        instance_index=1,
        case_id="ovphysx.left.effective_readback.r01",
        fresh_instance_id="1" * 64,
        fresh_process_id="2" * 64,
        provenance=_provenance(plan, "1"),
    )

    observed = instance["dof_records"][0]["armature"]["runtime_effective_value"]
    assert observed == {
        "dof_armature_binding": pytest.approx(0.03),
        "idealpd_controller_buffer": pytest.approx(0.031),
        "binding_controller_exact_match": False,
    }


def test_four_fresh_processes_assemble_exactly_and_sanitize_private_values() -> None:
    plan, manifest, instances = _instances()
    payload = assemble_readback_payload(plan, manifest, instances)
    hashes = {
        row["case_id"]: canonical_json_sha256(row) for row in instances
    }
    summary = sanitize_readback_payload(
        plan, payload, instance_file_sha256=hashes
    )
    report = render_readback_report(summary)

    assert payload["coverage"]["covered_joint_count_total"] == 44
    assert payload["reproducibility"] == {
        "mapping_exact": True,
        "solver_runtime_exact": True,
        "canonical_numeric_vector_hashes_exact": True,
    }
    rendered = json.dumps(summary, sort_keys=True)
    for private in ("GPU-private", "private-machine", "private-session"):
        assert private not in rendered
    assert summary["semantic_classification"]["friction"]["classification"] == "UNMAPPABLE"
    solver_semantic = summary["semantic_classification"]["solver_runtime"]
    assert solver_semantic["classification"] == "UNMAPPABLE"
    assert solver_semantic["field_classifications"]["integrator"]["classification"] == "UNMAPPABLE"
    assert (
        solver_semantic["field_classifications"]["physics_dt_s"]["classification"]
        == "UNMAPPABLE"
    )
    assert "Formal Gate 0 remains **DIVERGENT**" in report
    assert "pass_ready=false" in report


def test_readback_rejects_trace_activity_and_repeatability_drift() -> None:
    plan, manifest, instances = _instances()
    stepped = copy.deepcopy(instances[0])
    stepped["user_trace_simulation_step_call_count"] = 1
    with pytest.raises(EffectiveReadbackError, match="readback-only"):
        validate_readback_instance(plan, manifest, stepped)

    drifted = copy.deepcopy(instances)
    armature_runtime = drifted[1]["dof_records"][0]["armature"][
        "runtime_effective_value"
    ]
    armature_runtime["dof_armature_binding"] += 0.1
    armature_runtime["idealpd_controller_buffer"] += 0.1
    key = "armature_runtime_effective"
    values = [
        record["armature"]["runtime_effective_value"]
        for record in drifted[1]["dof_records"]
    ]
    drifted[1]["canonical_vector_sha256"][key] = canonical_json_sha256(values)
    with pytest.raises(EffectiveReadbackError, match="not exactly reproducible"):
        assemble_readback_payload(plan, manifest, drifted)

    solver_drifted = copy.deepcopy(instances)
    solver_drifted[1]["solver_runtime"]["integrator"]["unavailable_reason"] += " drift"
    with pytest.raises(EffectiveReadbackError, match="solver readback"):
        assemble_readback_payload(plan, manifest, solver_drifted)


def test_source_snapshot_and_pipeline_hashes_are_required_and_stable() -> None:
    plan, manifest, instances = _instances()
    malformed = copy.deepcopy(instances[0])
    malformed["provenance"]["pipeline_module_sha256"] = "not-a-sha"
    with pytest.raises(EffectiveReadbackError, match="pipeline_module_sha256"):
        validate_readback_instance(plan, manifest, malformed)

    mixed = copy.deepcopy(instances)
    mixed[1]["provenance"]["source_snapshot_sha256"] = "0" * 64
    with pytest.raises(EffectiveReadbackError, match="stable provenance"):
        assemble_readback_payload(plan, manifest, mixed)


def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(EffectiveReadbackError, match="duplicate"):
        load_json_strict(duplicate, label="fixture")
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(EffectiveReadbackError, match="non-finite"):
        load_json_strict(nonfinite, label="fixture")


def test_finalizer_inventory_is_exact_and_detects_tampering(tmp_path: Path) -> None:
    finalizer = _load_script("finalize_ovphysx_effective_readback.py")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    inventory = tmp_path / "evidence.sha256"
    inventory.write_text(
        "\n".join(
            f"{sha256_file(tmp_path / name)}  ./{name}" for name in ("a.txt", "b.txt")
        )
        + "\n",
        encoding="utf-8",
    )
    records = finalizer._verify_sha256_inventory(
        tmp_path, inventory, excluded={"evidence.sha256"}
    )
    assert set(records) == {"a.txt", "b.txt"}

    injected = tmp_path / "payload.tmp.injected"
    injected.write_text("unlisted\n", encoding="utf-8")
    with pytest.raises(EffectiveReadbackError, match="not exact"):
        finalizer._verify_sha256_inventory(
            tmp_path, inventory, excluded={"evidence.sha256"}
        )
    injected.unlink()

    (tmp_path / "b.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(EffectiveReadbackError, match="hash mismatch"):
        finalizer._verify_sha256_inventory(
            tmp_path, inventory, excluded={"evidence.sha256"}
        )


def test_worker_and_finalizer_cli_are_frozen() -> None:
    worker = _load_script("probe_ovphysx_effective_params.py")
    finalizer = _load_script("finalize_ovphysx_effective_readback.py")
    worker_actions = {action.dest for action in worker._parser()._actions}
    assert {
        "approved_root",
        "asset_root",
        "gate0_manifest",
        "freeze_config",
        "case_id",
        "side",
        "instance_index",
        "output",
        "device",
        "session_id",
        "source_revision",
        "source_tree",
        "source_archive_sha256",
        "asset_tree_sha256",
    }.issubset(worker_actions)
    finalizer_actions = {action.dest for action in finalizer._parser()._actions}
    assert {
        "gate0_manifest",
        "freeze_config",
        "raw_evidence_root",
        "output_dir",
        "source_revision",
    }.issubset(finalizer_actions)
