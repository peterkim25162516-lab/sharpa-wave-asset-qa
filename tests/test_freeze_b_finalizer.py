from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import HandSide, Simulator
from wave_asset_qa.parity.scenarios import (
    canonical_initial_positions,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
)
from wave_asset_qa.parity.sensitivity import (
    REQUIRED_VALIDITY_CHECKS,
    canonical_frame_names,
    canonical_joint_names,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_finalizer():
    path = ROOT / "scripts/finalize_ovphysx_freeze_b.py"
    spec = importlib.util.spec_from_file_location("test_freeze_b_finalizer_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corrigendum_fixture(
    finalizer: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    correction_revision: str,
    correction_tree: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-evidence-corrigendum-v1"
        ),
        "protocol_id": finalizer.PROTOCOL_ID,
        "state": (
            "POST_PREREGISTRATION_EVIDENCE_CORRECTION_BEFORE_ANY_ADMITTED_"
            "FREEZE_B_EVIDENCE"
        ),
        "frozen_preregistration": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "private_plan_sha256": finalizer.PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": finalizer.PRIVATE_PLAN_FILE_SHA256,
            "preregistration_revision": finalizer.PREREGISTRATION_REVISION,
            "preregistration_tree": finalizer.PREREGISTRATION_TREE,
            "public_protocol_path": finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": finalizer.PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": finalizer.PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": (
                finalizer.PUBLIC_PROTOCOL_CANONICAL_SHA256
            ),
            "public_protocol_unchanged": True,
        },
        "incident": {
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
        },
        "correction": {
            "scope": "evidence_plumbing_only",
            "worker_identity_authority": (
                "actual_python_worker_kernel_identity_v1"
            ),
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
        },
        "correction_source": {
            "revision": correction_revision,
            "tree": correction_tree,
        },
        "source_transition_policy": {
            "implementation_to_preregistration_name_status": (
                finalizer.IMPLEMENTATION_TO_PREREGISTRATION_DIFF
            ),
            "preregistration_to_correction_name_status": (
                finalizer.PREREGISTRATION_TO_CORRECTION_DIFF
            ),
            "correction_to_execution_name_status": (
                finalizer.CORRECTION_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }


def _admission_corrigendum_fixture(
    finalizer: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-remote-admission-corrigendum-v1"
        ),
        "protocol_id": finalizer.PROTOCOL_ID,
        "state": (
            "POST_PREREGISTRATION_REMOTE_ADMISSION_CORRECTION_AFTER_ABORTED_A2_"
            "COLLECTION_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
        "frozen_history": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "preregistration_revision": finalizer.PREREGISTRATION_REVISION,
            "preregistration_tree": finalizer.PREREGISTRATION_TREE,
            "first_correction_revision": first_correction_revision,
            "first_correction_tree": first_correction_tree,
            "first_execution_revision": finalizer.FIRST_EXECUTION_REVISION,
            "first_execution_tree": finalizer.FIRST_EXECUTION_TREE,
            "private_plan_sha256": finalizer.PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": finalizer.PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": finalizer.PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": finalizer.PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": finalizer.PUBLIC_PROTOCOL_CANONICAL_SHA256,
            "public_protocol_unchanged": True,
            "first_corrigendum_path": finalizer.CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": finalizer.FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": finalizer.FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": finalizer.FIRST_CORRIGENDUM_CANONICAL_SHA256,
            "first_corrigendum_unchanged": True,
        },
        "incident": {
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
        },
        "correction": {
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
        },
        "excluded_evidence": {
            label: {
                **finalizer.ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES[label],
                "reuse_allowed": False,
            }
            for label in finalizer.ADMISSION_EXCLUDED_EVIDENCE_LABELS
        },
        "correction_source": {
            "revision": admission_correction_revision,
            "tree": admission_correction_tree,
        },
        "source_transition_policy": {
            "first_execution_to_admission_correction_name_status": (
                finalizer.FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF
            ),
            "admission_correction_to_execution_name_status": (
                finalizer.ADMISSION_CORRECTION_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }


def _asset_preflight_corrigendum_fixture(
    finalizer: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    value = json.loads(
        (ROOT / finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    value["frozen_history"] = {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "preregistration_revision": finalizer.PREREGISTRATION_REVISION,
        "preregistration_tree": finalizer.PREREGISTRATION_TREE,
        "first_correction_revision": first_correction_revision,
        "first_correction_tree": first_correction_tree,
        "first_execution_revision": finalizer.FIRST_EXECUTION_REVISION,
        "first_execution_tree": finalizer.FIRST_EXECUTION_TREE,
        "admission_correction_revision": admission_correction_revision,
        "admission_correction_tree": admission_correction_tree,
        "private_plan_sha256": finalizer.PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": finalizer.PRIVATE_PLAN_FILE_SHA256,
        "public_protocol_path": finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
        "public_protocol_blob_oid": finalizer.PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": finalizer.PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": finalizer.PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
        "first_corrigendum_path": finalizer.CORRIGENDUM_RELATIVE_PATH.as_posix(),
        "first_corrigendum_blob_oid": finalizer.FIRST_CORRIGENDUM_BLOB_OID,
        "first_corrigendum_file_sha256": finalizer.FIRST_CORRIGENDUM_FILE_SHA256,
        "first_corrigendum_canonical_sha256": finalizer.FIRST_CORRIGENDUM_CANONICAL_SHA256,
        "first_corrigendum_unchanged": True,
        "admission_corrigendum_path": finalizer.ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        "admission_corrigendum_blob_oid": finalizer.ADMISSION_CORRIGENDUM_BLOB_OID,
        "admission_corrigendum_file_sha256": finalizer.ADMISSION_CORRIGENDUM_FILE_SHA256,
        "admission_corrigendum_canonical_sha256": finalizer.ADMISSION_CORRIGENDUM_CANONICAL_SHA256,
        "admission_corrigendum_unchanged": True,
    }
    value["excluded_evidence"] = {
        label: {
            **finalizer.EXCLUDED_EVIDENCE_IDENTITIES[label],
            "reuse_allowed": False,
        }
        for label in finalizer.EXCLUDED_EVIDENCE_LABELS
    }
    value["incident_source"] = {
        "revision": finalizer.SECOND_EXECUTION_REVISION,
        "tree": finalizer.SECOND_EXECUTION_TREE,
    }
    value["source_transition_policy"] = {
        "incident_source_to_execution_name_status": (
            finalizer.INCIDENT_SOURCE_TO_EXECUTION_DIFF
        ),
        "renames_allowed": False,
        "other_paths_allowed": False,
    }
    return value


def _runtime_order_corrigendum_fixture(finalizer: object) -> dict[str, object]:
    value = json.loads(
        (
            ROOT / finalizer.RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH
        ).read_text(encoding="utf-8")
    )
    value["frozen_campaign"].update(
        {
            "admission_corrigendum_blob_oid": (
                finalizer.ADMISSION_CORRIGENDUM_BLOB_OID
            ),
            "admission_corrigendum_file_sha256": (
                finalizer.ADMISSION_CORRIGENDUM_FILE_SHA256
            ),
            "admission_corrigendum_canonical_sha256": (
                finalizer.ADMISSION_CORRIGENDUM_CANONICAL_SHA256
            ),
        }
    )
    return value


def _time_grid_corrigendum_fixture(finalizer: object) -> dict[str, object]:
    value = json.loads(
        (
            ROOT / finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH
        ).read_text(encoding="utf-8")
    )
    value["frozen_history"].update(
        {
            "runtime_order_corrigendum_blob_oid": (
                finalizer.RUNTIME_ORDER_CORRIGENDUM_BLOB_OID
            ),
            "runtime_order_corrigendum_file_sha256": (
                finalizer.RUNTIME_ORDER_CORRIGENDUM_FILE_SHA256
            ),
            "runtime_order_corrigendum_canonical_sha256": (
                finalizer.RUNTIME_ORDER_CORRIGENDUM_CANONICAL_SHA256
            ),
        }
    )
    return value


def _load_sensitivity_fixtures():
    path = ROOT / "tests/test_freeze_b_sensitivity.py"
    spec = importlib.util.spec_from_file_location(
        "test_freeze_b_finalizer_sensitivity_fixtures", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_snapshot_installer():
    path = ROOT / "scripts/install_gate0_snapshot.py"
    spec = importlib.util.spec_from_file_location(
        "test_freeze_b_finalizer_snapshot_installer", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hand_plan(side: str) -> dict[str, object]:
    names = canonical_joint_names(side)
    expected = {name: 1.25 for name in names}
    return {
        "hand": side,
        "joint_names": list(names),
        "joint_prim_paths": {
            name: f"/World/Env_0/Robot/joints/{name}" for name in names
        },
        "expected_pre_values": expected,
        "sham_write_values": dict(expected),
        "zero_write_values": {name: 0.0 for name in names},
    }


def _intervention(role: str, plan_digest: str, hand_plan: dict[str, object]) -> dict[str, object]:
    names = hand_plan["joint_names"]
    assert isinstance(names, list)
    expected = hand_plan["expected_pre_values"]
    writes = hand_plan[f"{role}_write_values"]
    paths = hand_plan["joint_prim_paths"]
    assert isinstance(expected, dict) and isinstance(writes, dict) and isinstance(paths, dict)
    records = []
    for name in names:
        expected_hex = "3fa00000"
        write_hex = expected_hex if role == "sham" else "00000000"
        write_value = expected[name] if role == "sham" else 0.0
        records.append(
            {
                "canonical_id": name,
                "prim_path": paths[name],
                "property_path": f"{paths[name]}.physxJoint:jointFriction",
                "observed_pre_value": expected[name],
                "observed_pre_float32_hex": expected_hex,
                "expected_pre_value": expected[name],
                "expected_pre_float32_hex": expected_hex,
                "write_value": write_value,
                "write_float32_hex": write_hex,
                "post_write_value": write_value,
                "post_write_float32_hex": write_hex,
                "post_reset_value": write_value,
                "post_reset_float32_hex": write_hex,
                "post_cleanup_float32_hex": expected_hex,
            }
        )
    digest = "a" * 64
    return {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": plan_digest,
        "attribute": "physxJoint:jointFriction",
        "edit_strategy": "anonymous_overlay_as_strongest_session_sublayer",
        "source_asset_sha256_before": digest,
        "source_asset_sha256_after_cleanup": digest,
        "source_asset_sha256_unchanged": True,
        "joint_count": 22,
        "joint_order": names,
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
        "records": records,
    }


def _dt_v2(dt: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "dt_claim_scope": "python_configuration_only_not_compiled_runtime",
        "requested_dt_s": dt,
        "simulation_cfg_dt_s": dt,
        "simulation_context_config_accessor_dt_s": dt,
        "python_configuration_dt_exact_match": True,
        "runtime_effective_dt_s": None,
        "runtime_effective_dt_status": "not_exposed_by_pinned_kitless_ovphysx",
        "runtime_effective_dt_verified": False,
        "requested_gravity_m_s2": [0.0, 0.0, 0.0],
        "simulation_cfg_gravity_m_s2": [0.0, 0.0, 0.0],
        "physics_scene_gravity_m_s2": [0.0, 0.0, 0.0],
        "physics_prim_path": "/physicsScene",
    }


def _result(
    role: str,
    plan_digest: str,
    hand_plan: dict[str, object],
    *,
    runtime_friction: float = 0.0,
    armature: float = 0.01,
) -> AdapterRunResult:
    names = hand_plan["joint_names"]
    writes = hand_plan[f"{role}_write_values"]
    assert isinstance(names, list) and isinstance(writes, dict)
    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "canonical_id": name,
                "backend_name": name,
                "backend_index": index,
                "joint_prim_path": hand_plan["joint_prim_paths"][name],
                "legacy_joint_friction_usd": {
                    "attribute": "physxJoint:jointFriction",
                    "has_authored_value_opinion": True,
                    "authored_value": writes[name],
                    "resolved_value": writes[name],
                },
                "armature": armature,
                "controller_armature": armature,
                "friction_properties_raw": [runtime_friction] * 3,
                "controller_static_friction": runtime_friction,
                "controller_dynamic_friction": runtime_friction,
                "controller_viscous_friction": runtime_friction,
                "controller_binding_exact_match": {
                    "armature": True,
                    "static_friction": True,
                    "dynamic_friction": True,
                    "viscous_friction": True,
                },
                "controller_stiffness": 1.0,
                "controller_damping": 0.5,
                "controller_effort_limit": 3.3,
                "controller_effort_limit_sim": 1e9,
                "backend_drive_stiffness": 0.0,
                "backend_drive_damping": 0.0,
            }
        )
    return AdapterRunResult(
        backend="ovphysx",
        scenario_id="small_step",
        status="completed",
        message="fixture",
        dt=0.002,
        requested_steps=1,
        completed_steps=1,
        joint_names=tuple(names),
        frame_names=canonical_frame_names("left"),
        provenance={
            "backend_joint_names": list(names),
            "joint_mapping": [
                {
                    "canonical_id": name,
                    "backend_name": name,
                    "backend": "ovphysx",
                    "scope": str(hand_plan["hand"]),
                    "index": index,
                    "sign": 1.0,
                    "offset": 0.0,
                    "unit": "rad",
                }
                for index, name in enumerate(names)
            ],
            "legacy_joint_friction_intervention": _intervention(
                role, plan_digest, hand_plan
            ),
            "simulation_configuration_v2": _dt_v2(0.002),
            "effective_parameter_readback": {
                "capture_phase": "post_reset_pre_trace_step",
                "step_index": 0,
                "backend_joint_dynamics": {
                    "runtime_effective_readback": True,
                    "capture_phase": "post_reset_pre_trace_step",
                    "joint_count": 22,
                    "joint_order": names,
                    "records": records,
                    "friction_semantics": {
                        "raw_slot_order": ["static", "dynamic", "viscous"]
                    },
                    "controller_joint_dynamics_binding_exact_match": {
                        "overall": True,
                        "armature": True,
                        "static_friction": True,
                        "dynamic_friction": True,
                        "viscous_friction": True,
                    },
                },
                "physics_solver_contract": {},
            },
        },
    )


def _backend_order_result(
    result: AdapterRunResult,
    canonical_backend_order: list[str],
    *,
    backend_name_prefix: str = "",
) -> AdapterRunResult:
    """Represent the same values in backend order with an exact inverse map."""

    provenance = deepcopy(dict(result.provenance))
    mapping_by_name = {
        record["canonical_id"]: record
        for record in provenance["joint_mapping"]
    }
    runtime = provenance["effective_parameter_readback"][
        "backend_joint_dynamics"
    ]
    runtime_by_name = {
        record["canonical_id"]: record for record in runtime["records"]
    }
    backend_names = {
        name: f"{backend_name_prefix}{name}" for name in canonical_backend_order
    }
    for index, name in enumerate(canonical_backend_order):
        mapping_by_name[name]["index"] = index
        mapping_by_name[name]["backend_name"] = backend_names[name]
        runtime_by_name[name]["backend_index"] = index
        runtime_by_name[name]["backend_name"] = backend_names[name]
    provenance["backend_joint_names"] = [
        backend_names[name] for name in canonical_backend_order
    ]
    runtime["joint_order"] = [
        backend_names[name] for name in canonical_backend_order
    ]
    runtime["records"] = [
        runtime_by_name[name] for name in canonical_backend_order
    ]
    return replace(result, provenance=provenance)


def _r1_instance(
    hand_plan: dict[str, object],
    *,
    runtime_friction: float = 0.0,
) -> dict[str, object]:
    names = hand_plan["joint_names"]
    assert isinstance(names, list)
    return {
        "dof_records": [
            {
                "mapping": {"canonical_joint_name": name},
                "friction": {
                    "runtime_effective_value": {
                        "dof_friction_properties_binding": {
                            "static": runtime_friction,
                            "dynamic": runtime_friction,
                            "viscous": runtime_friction,
                        },
                        "idealpd_controller_buffers": {
                            "static": runtime_friction,
                            "dynamic": runtime_friction,
                            "viscous": runtime_friction,
                        },
                    }
                },
            }
            for name in names
        ]
    }


@pytest.mark.parametrize("role", ["sham", "zero"])
def test_intervention_and_dt_evidence_are_exact(role: str) -> None:
    finalizer = _load_finalizer()
    plan_digest = "b" * 64
    hand_plan = _hand_plan("left")
    result = _result(role, plan_digest, hand_plan)
    row = {"role": role}

    assert finalizer._validate_intervention(
        result, row=row, hand_plan=hand_plan, plan_digest=plan_digest
    )
    assert finalizer._validate_intervention_readback(
        result, row=row, hand_plan=hand_plan
    )
    assert finalizer._validate_dt_v2(result, 0.002)

    result.provenance["simulation_configuration_v2"]["runtime_effective_dt_verified"] = True
    assert not finalizer._validate_dt_v2(result, 0.002)


def test_position_target_readback_uses_mapping_index_and_rejects_tampered_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(ROOT / "configs/parity/gate0.json")
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.case_id == "ovphysx.left.small_step.base.r01"
    )
    scenario = manifest.scenario("small_step")
    hand = manifest.hand(HandSide.LEFT)
    hand_plan = _hand_plan("left")
    names = list(hand.joint_names)
    backend_order = names[5:13] + names[:5] + names[13:]
    result = _backend_order_result(
        _result("sham", "b" * 64, hand_plan),
        backend_order,
        backend_name_prefix="backend::",
    )
    expected_steps = round(scenario.duration_s / case.dt_s)
    result = replace(
        result,
        requested_steps=expected_steps,
        completed_steps=expected_steps,
    )
    expected = {name: (index + 1) / 100.0 for index, name in enumerate(names)}
    monkeypatch.setattr(
        finalizer,
        "canonical_position_targets",
        lambda *_args, **_kwargs: expected,
    )
    result.provenance.update(
        {
            "position_target_readback_verified": True,
            "position_target_readback_source": (
                "articulation_data_joint_pos_target_torch"
            ),
            "position_target_readback_count": expected_steps + 2,
            "position_target_readback_max_abs_error_rad": 0.0,
            "position_target_readback_values_rad": [
                expected[name] for name in backend_order
            ],
            "position_target_nonzero_readback_observed": True,
            "zero_velocity_target_verified": True,
            "zero_feedforward_effort_target_verified": True,
        }
    )
    valid, observed = finalizer._validate_position_target_readback(
        result,
        case=case,
        manifest=manifest,
        expected_joint_names=hand_plan["joint_names"],
        max_abs_error_rad=1e-6,
    )
    assert valid is True and observed == 0.0

    result.provenance["position_target_readback_source"] = "tampered"
    valid, _ = finalizer._validate_position_target_readback(
        result,
        case=case,
        manifest=manifest,
        expected_joint_names=hand_plan["joint_names"],
        max_abs_error_rad=1e-6,
    )
    assert valid is False


def test_remote_case_gpu_and_constraint_envelope_is_launcher_bound() -> None:
    finalizer = _load_finalizer()
    launcher = {
        "selected_gpu_uuid": "GPU-f4c114dd-f889-5df3-9aed-0bc7436185a6",
        "selected_gpu_name": "NVIDIA A800-SXM4-40GB",
        "driver_version": "570.158.01",
    }
    provenance = {
        "gpu_uuid": launcher["selected_gpu_uuid"],
        "gpu_name": launcher["selected_gpu_name"],
        "driver_version": launcher["driver_version"],
        "forbidden_modules": [],
        "constraints": {
            "kitless": True,
            "headless": True,
            "renderer": False,
            "camera": False,
            "fresh_process": True,
            "scenario_id": "small_step",
        },
    }
    finalizer._validate_remote_case_runtime_envelope(
        provenance, launcher=launcher
    )
    provenance["constraints"]["renderer"] = True
    with pytest.raises(finalizer.FreezeBFinalizationError, match="constraints"):
        finalizer._validate_remote_case_runtime_envelope(
            provenance, launcher=launcher
        )

    provenance["constraints"]["renderer"] = False
    provenance["forbidden_modules"] = ["omni.kit.app"]
    with pytest.raises(finalizer.FreezeBFinalizationError, match="constraints"):
        finalizer._validate_remote_case_runtime_envelope(
            provenance, launcher=launcher
        )


def test_remote_process_identity_hash_is_recomputed_and_worker_bound() -> None:
    finalizer = _load_finalizer()
    identity = {
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "hostname": "example-gpu-node-6",
        "pid": 4242,
        "process_start_ticks": 987654,
    }
    provenance = {
        "os_process_identity": identity,
        "worker_pid": 4242,
        "fresh_process_id": finalizer.canonical_json_sha256(identity),
    }

    assert (
        finalizer._validate_remote_process_identity(provenance)
        == provenance["fresh_process_id"]
    )
    provenance["worker_pid"] = 4243
    with pytest.raises(finalizer.FreezeBFinalizationError, match="worker_pid"):
        finalizer._validate_remote_process_identity(provenance)
    provenance["worker_pid"] = 4242
    provenance["fresh_process_id"] = "0" * 64
    with pytest.raises(finalizer.FreezeBFinalizationError, match="hash"):
        finalizer._validate_remote_process_identity(provenance)


def test_local_worker_identity_is_independently_recomputed_and_bound() -> None:
    finalizer = _load_finalizer()
    identity = {
        "platform": "windows",
        "pid": 4242,
        "process_creation_filetime": 123456789,
    }
    record = {
        "schema_version": 1,
        "worker_pid": 4242,
        "os_process_identity": identity,
        "fresh_process_id": finalizer.canonical_json_sha256(identity),
    }

    assert finalizer._validate_local_worker_process_record(record) == (
        identity,
        record["fresh_process_id"],
        4242,
    )
    record["worker_pid"] = 4243
    with pytest.raises(finalizer.FreezeBFinalizationError, match="worker_pid"):
        finalizer._validate_local_worker_process_record(record)
    record["worker_pid"] = 4242
    record["fresh_process_id"] = "0" * 64
    with pytest.raises(finalizer.FreezeBFinalizationError, match="digest"):
        finalizer._validate_local_worker_process_record(record)


def test_corrigendum_schema_and_source_transitions_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    implementation_revision = "a" * 40
    implementation_tree = "b" * 40
    correction_revision = "c" * 40
    correction_tree = "d" * 40
    admission_correction_revision = "e" * 40
    admission_correction_tree = "1" * 40
    second_execution_revision = "f" * 40
    second_execution_tree = "2" * 40
    campaign_revision = finalizer.CAMPAIGN_EXECUTION_REVISION
    campaign_tree = finalizer.CAMPAIGN_EXECUTION_TREE
    analysis_revision = "3" * 40
    analysis_tree = "4" * 40
    project = tmp_path / "repo"
    protocol = project / finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH
    corrigendum_path = project / finalizer.CORRIGENDUM_RELATIVE_PATH
    admission_corrigendum_path = (
        project / finalizer.ADMISSION_CORRIGENDUM_RELATIVE_PATH
    )
    asset_preflight_corrigendum_path = (
        project / finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH
    )
    runtime_order_corrigendum_path = (
        project / finalizer.RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH
    )
    time_grid_corrigendum_path = (
        project / finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((ROOT / finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH).read_bytes())
    corrigendum = _corrigendum_fixture(
        finalizer,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        correction_revision=correction_revision,
        correction_tree=correction_tree,
    )
    corrigendum_path.write_text(
        json.dumps(corrigendum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission_corrigendum = _admission_corrigendum_fixture(
        finalizer,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
        admission_correction_revision=admission_correction_revision,
        admission_correction_tree=admission_correction_tree,
    )
    admission_corrigendum_path.write_text(
        json.dumps(admission_corrigendum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        finalizer,
        "ADMISSION_CORRIGENDUM_FILE_SHA256",
        finalizer.sha256_file(admission_corrigendum_path),
    )
    monkeypatch.setattr(
        finalizer,
        "ADMISSION_CORRIGENDUM_CANONICAL_SHA256",
        finalizer.canonical_json_sha256(admission_corrigendum),
    )
    monkeypatch.setattr(
        finalizer,
        "ADMISSION_CORRIGENDUM_BLOB_OID",
        "5" * 40,
    )
    monkeypatch.setattr(
        finalizer, "SECOND_EXECUTION_REVISION", second_execution_revision
    )
    monkeypatch.setattr(
        finalizer, "SECOND_EXECUTION_TREE", second_execution_tree
    )
    asset_preflight_corrigendum = _asset_preflight_corrigendum_fixture(
        finalizer,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
        admission_correction_revision=admission_correction_revision,
        admission_correction_tree=admission_correction_tree,
    )
    asset_preflight_corrigendum_path.write_text(
        json.dumps(asset_preflight_corrigendum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_order_corrigendum = _runtime_order_corrigendum_fixture(finalizer)
    runtime_order_corrigendum_path.write_text(
        json.dumps(runtime_order_corrigendum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        finalizer,
        "RUNTIME_ORDER_CORRIGENDUM_FILE_SHA256",
        finalizer.sha256_file(runtime_order_corrigendum_path),
    )
    monkeypatch.setattr(
        finalizer,
        "RUNTIME_ORDER_CORRIGENDUM_CANONICAL_SHA256",
        finalizer.canonical_json_sha256(runtime_order_corrigendum),
    )
    monkeypatch.setattr(
        finalizer,
        "RUNTIME_ORDER_CORRIGENDUM_BLOB_OID",
        "6" * 40,
    )
    time_grid_corrigendum = _time_grid_corrigendum_fixture(finalizer)
    time_grid_corrigendum_path.write_text(
        json.dumps(time_grid_corrigendum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        finalizer,
        "TIME_GRID_CORRIGENDUM_CANONICAL_SHA256",
        finalizer.canonical_json_sha256(time_grid_corrigendum),
    )
    validated = finalizer._validate_corrigendum(
        corrigendum,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
    )
    assert validated == corrigendum
    unknown = deepcopy(corrigendum)
    unknown["notes"] = "post-hoc scope expansion"
    with pytest.raises(finalizer.FreezeBFinalizationError, match="fields"):
        finalizer._validate_corrigendum(
            unknown,
            implementation_revision=implementation_revision,
            implementation_tree=implementation_tree,
        )
    assert finalizer._validate_admission_corrigendum(
        admission_corrigendum,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
    ) == admission_corrigendum
    finalizer._validate_public_safe(admission_corrigendum)
    drifted_admission = deepcopy(admission_corrigendum)
    drifted_admission["incident"]["remote_launcher_admitted_case_count"] = 1
    with pytest.raises(finalizer.FreezeBFinalizationError, match="incident facts"):
        finalizer._validate_admission_corrigendum(
            drifted_admission,
            implementation_revision=implementation_revision,
            implementation_tree=implementation_tree,
            first_correction_revision=correction_revision,
            first_correction_tree=correction_tree,
        )
    assert finalizer._validate_asset_preflight_corrigendum(
        asset_preflight_corrigendum,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
        admission_correction_revision=admission_correction_revision,
        admission_correction_tree=admission_correction_tree,
    ) == asset_preflight_corrigendum
    finalizer._validate_public_safe(asset_preflight_corrigendum)
    drifted_preflight = deepcopy(asset_preflight_corrigendum)
    drifted_preflight["incident"]["trajectory_case_count_executed"] = 1
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="asset preflight corrigendum facts",
    ):
        finalizer._validate_asset_preflight_corrigendum(
            drifted_preflight,
            implementation_revision=implementation_revision,
            implementation_tree=implementation_tree,
            first_correction_revision=correction_revision,
            first_correction_tree=correction_tree,
            admission_correction_revision=admission_correction_revision,
            admission_correction_tree=admission_correction_tree,
        )
    assert finalizer._validate_runtime_order_corrigendum(
        runtime_order_corrigendum
    ) == runtime_order_corrigendum
    finalizer._validate_public_safe(runtime_order_corrigendum)
    drifted_runtime_order = deepcopy(runtime_order_corrigendum)
    drifted_runtime_order["correction"]["numeric_values_changed"] = True
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="runtime order corrigendum facts",
    ):
        finalizer._validate_runtime_order_corrigendum(drifted_runtime_order)
    assert finalizer._validate_time_grid_corrigendum(
        time_grid_corrigendum
    ) == time_grid_corrigendum
    finalizer._validate_public_safe(time_grid_corrigendum)
    drifted_time_grid = deepcopy(time_grid_corrigendum)
    drifted_time_grid["correction"]["compatibility_abs_tolerance_s"] = 1e-9
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="time-grid corrigendum facts",
    ):
        finalizer._validate_time_grid_corrigendum(drifted_time_grid)

    tree_by_revision = {
        implementation_revision: implementation_tree,
        finalizer.PREREGISTRATION_REVISION: finalizer.PREREGISTRATION_TREE,
        correction_revision: correction_tree,
        finalizer.FIRST_EXECUTION_REVISION: finalizer.FIRST_EXECUTION_TREE,
        admission_correction_revision: admission_correction_tree,
        second_execution_revision: second_execution_tree,
        campaign_revision: campaign_tree,
        finalizer.FIRST_ANALYSIS_REVISION: finalizer.FIRST_ANALYSIS_TREE,
        analysis_revision: analysis_tree,
    }
    diffs = {
        (implementation_revision, finalizer.PREREGISTRATION_REVISION): (
            finalizer.IMPLEMENTATION_TO_PREREGISTRATION_DIFF
        ),
        (finalizer.PREREGISTRATION_REVISION, correction_revision): (
            finalizer.PREREGISTRATION_TO_CORRECTION_DIFF
        ),
        (correction_revision, finalizer.FIRST_EXECUTION_REVISION): (
            finalizer.CORRECTION_TO_EXECUTION_DIFF
        ),
        (finalizer.FIRST_EXECUTION_REVISION, admission_correction_revision): (
            finalizer.FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF
        ),
        (admission_correction_revision, second_execution_revision): (
            finalizer.ADMISSION_CORRECTION_TO_EXECUTION_DIFF
        ),
        (second_execution_revision, campaign_revision): (
            finalizer.INCIDENT_SOURCE_TO_EXECUTION_DIFF
        ),
        (campaign_revision, finalizer.FIRST_ANALYSIS_REVISION): (
            finalizer.CAMPAIGN_TO_ANALYSIS_CORRECTION_DIFF
        ),
        (finalizer.FIRST_ANALYSIS_REVISION, analysis_revision): (
            finalizer.ANALYSIS_TO_TIME_GRID_CORRECTION_DIFF
        ),
    }
    parents = {
        finalizer.PREREGISTRATION_REVISION: implementation_revision,
        correction_revision: finalizer.PREREGISTRATION_REVISION,
        finalizer.FIRST_EXECUTION_REVISION: correction_revision,
        admission_correction_revision: finalizer.FIRST_EXECUTION_REVISION,
        second_execution_revision: admission_correction_revision,
        campaign_revision: second_execution_revision,
        finalizer.FIRST_ANALYSIS_REVISION: campaign_revision,
        analysis_revision: finalizer.FIRST_ANALYSIS_REVISION,
    }

    def fake_git(root: Path, *arguments: str) -> str:
        assert root == project.resolve()
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(project.resolve())
        if arguments == ("rev-parse", "HEAD"):
            return analysis_revision
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return analysis_tree
        if len(arguments) == 5 and arguments[:4] == (
            "rev-list",
            "--parents",
            "-n",
            "1",
        ):
            revision = arguments[4]
            return f"{revision} {parents[revision]}"
        if len(arguments) == 2 and arguments[0] == "rev-parse":
            value = arguments[1]
            if value.endswith("^{tree}"):
                return tree_by_revision[value.removesuffix("^{tree}")]
            if value.endswith(":" + finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix()):
                return finalizer.PUBLIC_PROTOCOL_BLOB_OID
            if value.endswith(":" + finalizer.CORRIGENDUM_RELATIVE_PATH.as_posix()):
                return finalizer.FIRST_CORRIGENDUM_BLOB_OID
            if value.endswith(
                ":" + finalizer.ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ):
                return finalizer.ADMISSION_CORRIGENDUM_BLOB_OID
            if value.endswith(
                ":"
                + finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ):
                return finalizer.ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID
            if value.endswith(
                ":"
                + finalizer.RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ):
                return finalizer.RUNTIME_ORDER_CORRIGENDUM_BLOB_OID
            if value == (
                analysis_revision
                + ":"
                + finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ):
                return "7" * 40
        if len(arguments) == 4 and arguments[:2] == (
            "merge-base",
            "--is-ancestor",
        ):
            return ""
        if arguments[:3] == ("diff", "--name-status", "--no-renames"):
            older, newer = arguments[3], arguments[4]
            return "\n".join(diffs[(older, newer)])
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return arguments[-1]
        if arguments == (
            "hash-object",
            finalizer.CORRIGENDUM_RELATIVE_PATH.as_posix(),
        ):
            return finalizer.FIRST_CORRIGENDUM_BLOB_OID
        if arguments == (
            "hash-object",
            finalizer.ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        ):
            return finalizer.ADMISSION_CORRIGENDUM_BLOB_OID
        if arguments == (
            "hash-object",
            finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        ):
            return finalizer.ASSET_PREFLIGHT_CORRIGENDUM_BLOB_OID
        if arguments == (
            "hash-object",
            finalizer.RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        ):
            return finalizer.RUNTIME_ORDER_CORRIGENDUM_BLOB_OID
        if arguments == (
            "hash-object",
            finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        ):
            return "7" * 40
        raise AssertionError(f"unexpected git call: {arguments}")

    monkeypatch.setattr(finalizer, "_git", fake_git)
    source = finalizer._validate_source_identity(
        project.resolve(),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        campaign_revision=campaign_revision,
        analysis_revision=analysis_revision,
        public_protocol_path=protocol.resolve(),
        corrigendum_path=corrigendum_path.resolve(),
        admission_corrigendum_path=admission_corrigendum_path.resolve(),
        asset_preflight_corrigendum_path=(
            asset_preflight_corrigendum_path.resolve()
        ),
        runtime_order_corrigendum_path=(
            runtime_order_corrigendum_path.resolve()
        ),
        time_grid_corrigendum_path=time_grid_corrigendum_path.resolve(),
    )
    assert source["correction_revision"] == correction_revision
    assert source["admission_correction_revision"] == admission_correction_revision
    assert source["incident_source_revision"] == second_execution_revision
    assert source["campaign_execution_revision"] == campaign_revision
    assert source["first_analysis_revision"] == finalizer.FIRST_ANALYSIS_REVISION
    assert source["first_analysis_tree"] == finalizer.FIRST_ANALYSIS_TREE
    assert source["analysis_revision"] == analysis_revision
    assert source["time_grid_corrigendum_canonical_sha256"] == (
        finalizer.TIME_GRID_CORRIGENDUM_CANONICAL_SHA256
    )
    assert source["source_transitions_exact"] is True

    snapshot_root = tmp_path / "analysis-snapshot"
    snapshot_root.mkdir()
    snapshot_protocol = snapshot_root / "public-protocol.json"
    snapshot_corrigendum = snapshot_root / "corrigendum.json"
    snapshot_admission_corrigendum = snapshot_root / "admission-corrigendum.json"
    snapshot_asset_preflight_corrigendum = (
        snapshot_root / "asset-preflight-corrigendum.json"
    )
    snapshot_runtime_order_corrigendum = (
        snapshot_root / "runtime-order-corrigendum.json"
    )
    snapshot_time_grid_corrigendum = (
        snapshot_root / "time-grid-corrigendum.json"
    )
    snapshot_protocol.write_bytes(protocol.read_bytes())
    snapshot_corrigendum.write_bytes(corrigendum_path.read_bytes())
    snapshot_admission_corrigendum.write_bytes(
        admission_corrigendum_path.read_bytes()
    )
    snapshot_asset_preflight_corrigendum.write_bytes(
        asset_preflight_corrigendum_path.read_bytes()
    )
    snapshot_runtime_order_corrigendum.write_bytes(
        runtime_order_corrigendum_path.read_bytes()
    )
    snapshot_time_grid_corrigendum.write_bytes(
        time_grid_corrigendum_path.read_bytes()
    )
    snapshot_source = finalizer._validate_source_identity(
        project.resolve(),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        campaign_revision=campaign_revision,
        analysis_revision=analysis_revision,
        public_protocol_path=snapshot_protocol,
        corrigendum_path=snapshot_corrigendum,
        admission_corrigendum_path=snapshot_admission_corrigendum,
        asset_preflight_corrigendum_path=(
            snapshot_asset_preflight_corrigendum
        ),
        runtime_order_corrigendum_path=(
            snapshot_runtime_order_corrigendum
        ),
        time_grid_corrigendum_path=snapshot_time_grid_corrigendum,
        canonical_public_protocol_path=protocol.resolve(),
        canonical_corrigendum_path=corrigendum_path.resolve(),
        canonical_admission_corrigendum_path=admission_corrigendum_path.resolve(),
        canonical_asset_preflight_corrigendum_path=(
            asset_preflight_corrigendum_path.resolve()
        ),
        canonical_runtime_order_corrigendum_path=(
            runtime_order_corrigendum_path.resolve()
        ),
        canonical_time_grid_corrigendum_path=(
            time_grid_corrigendum_path.resolve()
        ),
    )
    assert snapshot_source == source

    parents[analysis_revision] = finalizer.PREREGISTRATION_REVISION
    with pytest.raises(finalizer.FreezeBFinalizationError, match="direct single-parent"):
        finalizer._validate_source_identity(
            project.resolve(),
            implementation_revision=implementation_revision,
            implementation_tree=implementation_tree,
            campaign_revision=campaign_revision,
            analysis_revision=analysis_revision,
            public_protocol_path=protocol.resolve(),
            corrigendum_path=corrigendum_path.resolve(),
            admission_corrigendum_path=admission_corrigendum_path.resolve(),
            asset_preflight_corrigendum_path=(
                asset_preflight_corrigendum_path.resolve()
            ),
            runtime_order_corrigendum_path=(
                runtime_order_corrigendum_path.resolve()
            ),
            time_grid_corrigendum_path=time_grid_corrigendum_path.resolve(),
        )
    parents[analysis_revision] = finalizer.FIRST_ANALYSIS_REVISION

    diffs[(finalizer.FIRST_ANALYSIS_REVISION, analysis_revision)] = [
        *finalizer.ANALYSIS_TO_TIME_GRID_CORRECTION_DIFF,
        "M\tsrc/wave_asset_qa/adapters/mujoco.py",
    ]
    with pytest.raises(finalizer.FreezeBFinalizationError, match="exact corrigendum"):
        finalizer._validate_source_identity(
            project.resolve(),
            implementation_revision=implementation_revision,
            implementation_tree=implementation_tree,
            campaign_revision=campaign_revision,
            analysis_revision=analysis_revision,
            public_protocol_path=protocol.resolve(),
            corrigendum_path=corrigendum_path.resolve(),
            admission_corrigendum_path=admission_corrigendum_path.resolve(),
            asset_preflight_corrigendum_path=(
                asset_preflight_corrigendum_path.resolve()
            ),
            runtime_order_corrigendum_path=(
                runtime_order_corrigendum_path.resolve()
            ),
            time_grid_corrigendum_path=time_grid_corrigendum_path.resolve(),
        )


def test_public_sanitizer_allows_process_aggregate_but_rejects_identities() -> None:
    finalizer = _load_finalizer()

    finalizer._validate_public_safe(
        {
            "validity": {
                "checks": {"fresh_process_24_of_24": True},
                "process_isolation": "24/24 distinct; identifiers remain private",
            }
        }
    )
    for private_value in (
        {"fresh_process_id": "0" * 64},
        {"fresh_process_ids": ["0" * 64]},
        {"os_process_identity": {"pid": 42}},
        {"ownership_token": "0" * 64},
    ):
        with pytest.raises(finalizer.FreezeBFinalizationError, match="forbidden"):
            finalizer._validate_public_safe(private_value)


def test_runtime_friction_sham_exact_zero_change_valid_and_nonfriction_drift_invalid() -> None:
    finalizer = _load_finalizer()
    hand_plan = _hand_plan("left")
    names = hand_plan["joint_names"]
    assert isinstance(names, list)
    plan_digest = "b" * 64
    sham = _result("sham", plan_digest, hand_plan, runtime_friction=0.0)
    zero = _result("zero", plan_digest, hand_plan, runtime_friction=0.25)
    r1_signature = finalizer._r1_runtime_friction_signature(
        _r1_instance(hand_plan), expected_joint_names=names
    )
    sham_signature = finalizer._runtime_friction_signature(
        sham, side="left", expected_joint_names=names
    )
    zero_signature = finalizer._runtime_friction_signature(
        zero, side="left", expected_joint_names=names
    )
    assert sham_signature == r1_signature
    assert zero_signature != r1_signature
    assert finalizer._normalized_non_target_contract(sham) == (
        finalizer._normalized_non_target_contract(zero)
    )

    rows = [
        {
            "experiment_case_id": "sham-r01",
            "hand": "left",
            "timestep_variant": "base",
            "role": "sham",
        },
        {
            "experiment_case_id": "sham-r02",
            "hand": "left",
            "timestep_variant": "base",
            "role": "sham",
        },
        {
            "experiment_case_id": "zero-r01",
            "hand": "left",
            "timestep_variant": "base",
            "role": "zero",
        },
        {
            "experiment_case_id": "zero-r02",
            "hand": "left",
            "timestep_variant": "base",
            "role": "zero",
        },
    ]
    gate, public, private = finalizer._summarize_runtime_friction_observation(
        rows,
        case_signatures={
            "sham-r01": sham_signature,
            "sham-r02": sham_signature,
            "zero-r01": zero_signature,
            "zero-r02": zero_signature,
        },
        r1_signatures={"left": r1_signature, "right": "f" * 64},
    )
    assert gate is True
    assert public["sham_exact_r1"] is True
    assert public["zero_any_changed_from_r1"] is True
    assert public["zero_change_is_descriptive_not_validity_failure"] is True
    assert "case_runtime_signatures" not in public
    assert "case_runtime_signatures" in private

    nonfriction_drift = _result(
        "zero",
        plan_digest,
        hand_plan,
        runtime_friction=0.25,
        armature=0.02,
    )
    assert finalizer._normalized_non_target_contract(sham) != (
        finalizer._normalized_non_target_contract(nonfriction_drift)
    )


def test_backend_order_is_validated_then_canonicalized_for_all_runtime_joins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    hand_plan = _hand_plan("left")
    names = hand_plan["joint_names"]
    assert isinstance(names, list)
    backend_order = names[5:13] + names[:5] + names[13:]
    plan_digest = "b" * 64
    canonical = _result("sham", plan_digest, hand_plan)
    backend = _backend_order_result(
        canonical,
        backend_order,
        backend_name_prefix="backend::",
    )

    assert finalizer._runtime_friction_signature(
        backend,
        side="left",
        expected_joint_names=names,
    ) == finalizer._runtime_friction_signature(
        canonical,
        side="left",
        expected_joint_names=names,
    )
    row = {
        "role": "sham",
        "hand": "left",
        "timestep_variant": "base",
    }
    assert finalizer._validate_intervention_readback(
        backend,
        row=row,
        hand_plan=hand_plan,
    )

    runtime_records = finalizer._canonical_runtime_joint_records(
        backend,
        side="left",
        expected_joint_names=names,
    )
    r1_records = []
    for record in runtime_records:
        name = record["canonical_id"]
        r1_records.append(
            {
                "mapping": {
                    "canonical_joint_name": name,
                    "backend_joint_name": record["backend_name"],
                    "backend_joint_index": record["backend_index"],
                    "joint_prim_path": record["joint_prim_path"],
                },
                "armature": {
                    "runtime_effective_value": {
                        "dof_armature_binding": record["armature"],
                        "idealpd_controller_buffer": record[
                            "controller_armature"
                        ],
                    }
                },
                "friction": {
                    "authored_value": {
                        "legacy_joint_friction_scalar": 1.25
                    },
                    "resolved_value": {
                        "legacy_joint_friction_scalar": 1.25
                    },
                    "runtime_effective_value": {
                        "dof_friction_properties_binding": {
                            key: record["friction_properties_raw"][index]
                            for index, key in enumerate(
                                ("static", "dynamic", "viscous")
                            )
                        },
                        "idealpd_controller_buffers": {
                            "static": record["controller_static_friction"],
                            "dynamic": record["controller_dynamic_friction"],
                            "viscous": record["controller_viscous_friction"],
                        },
                    },
                },
                "control_drive": {
                    "effort_limit": {
                        "runtime_effective_value": {
                            "controller": record["controller_effort_limit"],
                            "simulation": record[
                                "controller_effort_limit_sim"
                            ],
                        }
                    },
                    "kp": {
                        "runtime_effective_value": record[
                            "controller_stiffness"
                        ]
                    },
                    "kd": {
                        "runtime_effective_value": record[
                            "controller_damping"
                        ]
                    },
                    "backend_drive_stiffness": {
                        "runtime_effective_value": record[
                            "backend_drive_stiffness"
                        ]
                    },
                    "backend_drive_damping": {
                        "runtime_effective_value": record[
                            "backend_drive_damping"
                        ]
                    },
                },
            }
        )
    monkeypatch.setattr(finalizer, "_solver_runtime", lambda _snapshot: {})
    manifest = load_manifest(ROOT / "configs/parity/gate0.json")
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.case_id == "ovphysx.left.small_step.base.r01"
    )
    assert finalizer._r1_bridge(
        CollectedRun(case=case, result=backend),
        r1_instance={"dof_records": r1_records, "solver_runtime": {}},
        row=row,
        hand_plan=hand_plan,
    )

    drifted = deepcopy(dict(backend.provenance))
    drifted["effective_parameter_readback"]["backend_joint_dynamics"][
        "records"
    ][0]["backend_index"] = 1
    with pytest.raises(finalizer.FreezeBFinalizationError, match="ID/index"):
        finalizer._canonical_runtime_joint_records(
            replace(backend, provenance=drifted),
            side="left",
            expected_joint_names=names,
        )


def test_identity_bound_full_true_traces_receive_a_scientific_status() -> None:
    finalizer = _load_finalizer()
    fixtures = _load_sensitivity_fixtures()
    kwargs = fixtures._metric_kwargs()
    bindings = kwargs["evidence_bindings"]
    decision, observed_bindings = finalizer._evaluate_identity_bound_freeze_b(
        fixtures._trace_sets(),
        inputs=kwargs["frozen_inputs"],
        private_plan=kwargs["private_plan"],
        public_protocol=kwargs["public_protocol"],
        public_protocol_file_sha256=bindings["public_protocol_file_sha256"],
        public_protocol_canonical_sha256=bindings[
            "public_protocol_canonical_sha256"
        ],
        deployment_source_revision=bindings["deployment_source_revision"],
        deployment_source_tree=bindings["deployment_source_tree"],
        deployment_source_snapshot_sha256=bindings[
            "deployment_source_snapshot_sha256"
        ],
        validity_checks=fixtures._validity_checks(),
    )

    assert decision.status.value != "INVALID"
    assert decision.scientific_label is not None
    assert observed_bindings == bindings


def test_hash_inventory_is_exact_and_detects_tampering(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    (tmp_path / "nested").mkdir()
    payload = tmp_path / "nested/payload.txt"
    payload.write_text("evidence\n", encoding="utf-8")
    inventory = tmp_path / "evidence.sha256"
    inventory.write_text(
        f"{sha256(payload.read_bytes()).hexdigest()}  nested/payload.txt\n",
        encoding="utf-8",
    )
    finalizer._verify_hash_inventory(tmp_path, inventory)

    payload.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(finalizer.FreezeBFinalizationError, match="hash mismatch"):
        finalizer._verify_hash_inventory(tmp_path, inventory)


def test_tree_fingerprint_and_owned_cleanup_fail_closed(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    tree = tmp_path / "tree"
    tree.mkdir()
    payload = tree / "payload.txt"
    payload.write_text("one\n", encoding="utf-8")
    first = finalizer._tree_fingerprint(tree, "fixture")
    payload.write_text("two\n", encoding="utf-8")
    assert finalizer._tree_fingerprint(tree, "fixture") != first

    with pytest.raises(finalizer.FreezeBFinalizationError, match="unsafe"):
        finalizer._remove_owned_tree(tree, expected_path=tmp_path / "other")
    finalizer._remove_owned_tree(tree, expected_path=tree)
    assert not tree.exists()


def test_excluded_evidence_cli_and_exact_fingerprints_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    actions = {action.dest: action for action in finalizer._parser()._actions}
    assert actions["time_grid_corrigendum"].required is True
    assert actions["superseded_analysis_bundle"].required is True
    parsed = finalizer._parse_excluded_evidence_arguments(
        [f"{label}={tmp_path / label}" for label in finalizer.EXCLUDED_EVIDENCE_LABELS]
    )
    assert tuple(parsed) == finalizer.EXCLUDED_EVIDENCE_LABELS
    with pytest.raises(finalizer.FreezeBFinalizationError, match="exactly"):
        finalizer._parse_excluded_evidence_arguments(
            [f"local-a1={tmp_path / 'local-a1'}"]
        )
    with pytest.raises(finalizer.FreezeBFinalizationError, match="duplicate"):
        finalizer._parse_excluded_evidence_arguments(
            [
                f"local-a1={tmp_path / 'one'}",
                f"local-a1={tmp_path / 'two'}",
                f"local-a2={tmp_path / 'three'}",
                f"remote-a2={tmp_path / 'four'}",
            ]
        )

    trees: dict[str, Path] = {}
    for label in finalizer.EXCLUDED_EVIDENCE_LABELS:
        tree = tmp_path / label
        tree.mkdir()
        trees[label] = tree

    fingerprints = {
        label: {
            key: value
            for key, value in finalizer.EXCLUDED_EVIDENCE_IDENTITIES[label].items()
            if key != "classification"
        }
        for label in finalizer.EXCLUDED_EVIDENCE_LABELS
    }
    monkeypatch.setattr(
        finalizer,
        "_tree_fingerprint",
        lambda _path, label: fingerprints[label.rsplit(" ", 1)[-1]],
    )
    observed = finalizer._validate_excluded_evidence_fingerprints(trees)
    assert set(observed) == set(finalizer.EXCLUDED_EVIDENCE_LABELS)
    assert all(record["reuse_allowed"] is False for record in observed.values())

    original = finalizer.EXCLUDED_EVIDENCE_IDENTITIES["remote-a2"]["root_sha256"]
    finalizer.EXCLUDED_EVIDENCE_IDENTITIES["remote-a2"]["root_sha256"] = "0" * 64
    try:
        with pytest.raises(finalizer.FreezeBFinalizationError, match="remote-a2"):
            finalizer._validate_excluded_evidence_fingerprints(trees)
    finally:
        finalizer.EXCLUDED_EVIDENCE_IDENTITIES["remote-a2"]["root_sha256"] = original


def test_active_a4_evidence_fingerprints_are_exact_and_campaign_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    local = tmp_path / "local-a4"
    remote = tmp_path / "remote-a4"
    local.mkdir()
    remote.mkdir()
    fingerprints = {
        label: {
            key: value
            for key, value in identity.items()
            if key
            in {"entry_count", "file_count", "total_bytes", "root_sha256"}
        }
        for label, identity in finalizer.ACTIVE_EVIDENCE_IDENTITIES.items()
    }
    monkeypatch.setattr(
        finalizer,
        "_tree_fingerprint",
        lambda _path, label: fingerprints[label.rsplit(" ", 1)[-1]],
    )
    observed = finalizer._validate_active_evidence_fingerprints(
        local_evidence=local,
        remote_evidence=remote,
    )
    assert observed["local-a4"]["source_revision"] == (
        finalizer.CAMPAIGN_EXECUTION_REVISION
    )
    assert observed["remote-a4"]["source_tree"] == (
        finalizer.CAMPAIGN_EXECUTION_TREE
    )
    monkeypatch.setitem(
        finalizer.ACTIVE_EVIDENCE_IDENTITIES["remote-a4"],
        "root_sha256",
        "0" * 64,
    )
    with pytest.raises(finalizer.FreezeBFinalizationError, match="remote-a4"):
        finalizer._validate_active_evidence_fingerprints(
            local_evidence=local,
            remote_evidence=remote,
        )


def test_actual_superseded_h_analysis_bundle_is_exact_when_available() -> None:
    finalizer = _load_finalizer()
    root = ROOT / "results/freeze-b-bundle-ba2fb4c-20260830-a4"
    if not root.is_dir():
        pytest.skip("ignored superseded H analysis bundle is not available")
    observed = finalizer._validate_superseded_analysis_bundle(root)
    assert observed == finalizer.SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY
    assert observed["root_sha256"] == (
        "55375eab768452da6988e644c3cb0f4c2193c1cd8f499981afe09f777cb05614"
    )
    assert observed["full_tree_root_sha256"] == (
        "45709ef95f791ffcd8489217fa64d35268f8a43a0908eab435d27a7c2a3e03db"
    )


def test_superseded_h_subset_binds_manifest_bytes_when_available(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    source = ROOT / "results/freeze-b-bundle-ba2fb4c-20260830-a4"
    if not source.is_dir():
        pytest.skip("ignored superseded H analysis bundle is not available")
    subset = tmp_path / "superseded-analysis-subset"
    finalizer._copy_parent_subset(
        source,
        subset,
        finalizer.SUPERSEDED_ANALYSIS_SUBSET_PATHS,
    )
    finalizer._verify_superseded_analysis_subset(subset)
    manifest_path = subset / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finalizer.verify_copied_bundle_subset(
        manifest_path,
        expected_root_sha256=str(
            finalizer.SUPERSEDED_ANALYSIS_BUNDLE_IDENTITY["root_sha256"]
        ),
        copied_payloads={
            relative: subset / relative
            for relative in finalizer.SUPERSEDED_ANALYSIS_SUBSET_PATHS
            if relative != "bundle.json"
        },
    )
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="manifest bytes drifted",
    ):
        finalizer._verify_superseded_analysis_subset(subset)


def test_actual_a4_evidence_replays_valid_inconclusive_when_available() -> None:
    finalizer = _load_finalizer()
    h_bundle = ROOT / "results/freeze-b-bundle-ba2fb4c-20260830-a4"
    formal_bundle = ROOT / "results/gate0-bundle-c571d9b-20260827-a1"
    readback_bundle = (
        ROOT / "results/ovphysx-effective-readback-bundle-44571ea-20260828-a3"
    )
    if not all(path.is_dir() for path in (h_bundle, formal_bundle, readback_bundle)):
        pytest.skip("ignored Freeze B inputs are not all available")
    public_path = h_bundle / "inputs/public-protocol.json"
    plan_path = h_bundle / "private/freeze_b.private-plan.json"
    public = finalizer.load_public_protocol(public_path)
    plan = finalizer.load_private_plan(plan_path, public_protocol=public)
    manifest = load_manifest(h_bundle / "inputs/gate0.json")
    source_identity = finalizer._strict_json(
        h_bundle / "private/source-identity.json",
        "actual H source identity",
    )
    finalizer._load_r1(readback_bundle, plan=plan)
    formal = finalizer._load_formal_runs(
        formal_bundle,
        plan=plan,
        manifest=manifest,
    )
    local, local_ids = finalizer._load_local_runs(
        h_bundle / "private/local-evidence",
        plan=plan,
        plan_path=plan_path,
        manifest=manifest,
        execution_revision=finalizer.CAMPAIGN_EXECUTION_REVISION,
        execution_tree=finalizer.CAMPAIGN_EXECUTION_TREE,
        project_root=ROOT,
        source_identity=source_identity,
    )
    remote, launcher, remote_ids = finalizer._load_remote_runs(
        h_bundle / "private/remote-evidence",
        plan=plan,
        plan_path=plan_path,
        manifest=manifest,
        execution_revision=finalizer.CAMPAIGN_EXECUTION_REVISION,
        execution_tree=finalizer.CAMPAIGN_EXECUTION_TREE,
        project_root=ROOT,
        public_protocol_file_sha256=finalizer.sha256_file(public_path),
        public_protocol_canonical_sha256=(
            finalizer.canonical_json_sha256(public)
        ),
    )
    metric_groups, trace_sets = finalizer._formal_and_repeat_metrics(
        plan=plan,
        manifest=manifest,
        formal=formal,
        local=local,
        remote=remote,
    )
    decision, _bindings = finalizer._evaluate_identity_bound_freeze_b(
        trace_sets,
        inputs=plan["inputs"],
        private_plan=plan,
        public_protocol=public,
        public_protocol_file_sha256=finalizer.sha256_file(public_path),
        public_protocol_canonical_sha256=(
            finalizer.canonical_json_sha256(public)
        ),
        deployment_source_revision=finalizer.CAMPAIGN_EXECUTION_REVISION,
        deployment_source_tree=finalizer.CAMPAIGN_EXECUTION_TREE,
        deployment_source_snapshot_sha256=str(
            launcher["source_snapshot_sha256"]
        ),
        validity_checks={key: True for key in REQUIRED_VALIDITY_CHECKS},
    )
    assert (len(local), len(remote), len(local_ids), len(remote_ids)) == (
        8,
        16,
        8,
        16,
    )
    assert metric_groups["formal_bridge"] == {
        "joint_max_abs_rad": 0.0,
        "frame_position_max_m": 0.0,
        "frame_orientation_max_rad": 0.0,
    }
    assert decision.status is finalizer.SensitivityStatus.INCONCLUSIVE
    assert decision.execution_valid is True
    assert decision.validity_failures == ()
    assert len(decision.cells) == 4
    assert list(decision.s_values) == [
        0.09305827319415706,
        0.10210679734906178,
        0.09702777058089153,
        0.10679712580803115,
        0.09000382487930997,
        0.09914972602988767,
        0.09384820757062738,
        0.10367760461408876,
    ]


def test_time_grid_corrigendum_binds_outcome_exposure_and_window_counts() -> None:
    finalizer = _load_finalizer()
    path = ROOT / finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH
    value = finalizer._strict_json(path, "time-grid corrigendum")
    assert finalizer._validate_time_grid_corrigendum(value) == value
    correction = value["correction"]
    assert correction["base_baseline_and_zero_vs_mujoco_window_sample_count"] == 200
    assert correction["halved_baseline_and_zero_vs_mujoco_window_sample_count"] == 400
    assert correction["base_treatment_vs_sham_window_sample_count"] == 201
    assert correction["halved_treatment_vs_sham_window_sample_count"] == 401
    outcome = value["outcome_exposure"]
    assert outcome["diagnostic_validity_failure_count"] == 0
    assert (
        outcome["diagnostic_s_at_or_below_not_supported_count"],
        outcome["diagnostic_s_above_not_supported_count"],
        outcome["diagnostic_s_at_or_above_supported_count"],
    ) == (5, 3, 0)


def test_asset_preflight_corrigendum_is_public_safe_and_pre_adapter_only() -> None:
    finalizer = _load_finalizer()
    path = ROOT / finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH
    value = finalizer._strict_json(path, "asset preflight corrigendum")
    frozen = value["frozen_history"]
    validated = finalizer._validate_asset_preflight_corrigendum(
        value,
        implementation_revision=frozen["implementation_revision"],
        implementation_tree=frozen["implementation_tree"],
        first_correction_revision=frozen["first_correction_revision"],
        first_correction_tree=frozen["first_correction_tree"],
        admission_correction_revision=frozen["admission_correction_revision"],
        admission_correction_tree=frozen["admission_correction_tree"],
    )
    finalizer._validate_public_safe(validated)
    incident = validated["incident"]
    assert incident["worker_process_count_started"] == 1
    assert incident["runner_asset_preflight_entered"] is True
    assert incident["adapter_instance_created"] is False
    assert incident["trajectory_case_count_executed"] == 0
    assert incident["simulator_step_count"] == 0
    assert incident["remote_or_gpu_execution_occurred"] is False
    assert incident["actual_materialized_asset_root_recorded_in_evidence"] is False
    assert incident["actual_materialized_asset_tree_sha256_recorded_in_evidence"] is False
    assert validated["excluded_evidence"]["local-a3"] == {
        **finalizer.EXCLUDED_EVIDENCE_IDENTITIES["local-a3"],
        "reuse_allowed": False,
    }


def _analysis_snapshot_fixture(tmp_path: Path, finalizer: object) -> dict[str, object]:
    project = tmp_path / "repo"
    (project / "results").mkdir(parents=True)
    for relative in finalizer.REQUIRED_SOURCE_PATHS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source: {relative}\n", encoding="utf-8")
    private_plan = tmp_path / "freeze_b.private-plan.json"
    private_plan.write_text("frozen private plan\n", encoding="utf-8")
    tree_paths: dict[str, Path] = {}
    for name in (
        "readback_bundle",
        "formal_bundle",
        "local_evidence",
        "remote_evidence",
        "superseded_analysis_bundle",
    ):
        root = tmp_path / name
        root.mkdir()
        (root / "payload.txt").write_text(
            f"frozen {name}\n",
            encoding="utf-8",
        )
        tree_paths[name] = root
    excluded_evidence: dict[str, Path] = {}
    for label in finalizer.EXCLUDED_EVIDENCE_LABELS:
        root = tmp_path / f"excluded-{label}"
        root.mkdir()
        (root / "payload.txt").write_text(
            f"frozen excluded {label}\n", encoding="utf-8"
        )
        excluded_evidence[label] = root
    return {
        "project_root": project,
        "public_protocol_path": project / finalizer.PUBLIC_PROTOCOL_RELATIVE_PATH,
        "corrigendum_path": project / finalizer.CORRIGENDUM_RELATIVE_PATH,
        "admission_corrigendum_path": (
            project / finalizer.ADMISSION_CORRIGENDUM_RELATIVE_PATH
        ),
        "asset_preflight_corrigendum_path": (
            project / finalizer.ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH
        ),
        "runtime_order_corrigendum_path": (
            project / finalizer.RUNTIME_ORDER_CORRIGENDUM_RELATIVE_PATH
        ),
        "time_grid_corrigendum_path": (
            project / finalizer.TIME_GRID_CORRIGENDUM_RELATIVE_PATH
        ),
        "private_plan_path": private_plan,
        **tree_paths,
        "excluded_evidence": excluded_evidence,
        "output_dir": project / "results/freeze-b-private",
        "public_output_dir": None,
        "campaign_revision": finalizer.CAMPAIGN_EXECUTION_REVISION,
        "analysis_revision": "a" * 40,
    }


def test_copy_first_snapshot_isolates_analysis_from_temporary_source_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    arguments = _analysis_snapshot_fixture(tmp_path, finalizer)
    original_local = Path(arguments["local_evidence"]) / "payload.txt"
    observed: dict[str, object] = {}

    def analyze_snapshot(**kwargs: object):
        source_root = Path(kwargs["analysis_source_root"])
        analysis_root = source_root.parent
        snapshot_local = Path(kwargs["local_evidence"]) / "payload.txt"
        for key in (
            "public_protocol_path",
            "corrigendum_path",
            "admission_corrigendum_path",
            "asset_preflight_corrigendum_path",
            "runtime_order_corrigendum_path",
            "time_grid_corrigendum_path",
            "private_plan_path",
            "readback_bundle",
            "formal_bundle",
            "local_evidence",
            "remote_evidence",
            "superseded_analysis_bundle",
            "analysis_manifest_path",
            "analysis_source_root",
        ):
            Path(kwargs[key]).relative_to(analysis_root)
        excluded = kwargs["excluded_evidence"]
        assert isinstance(excluded, dict)
        assert set(excluded) == set(finalizer.EXCLUDED_EVIDENCE_LABELS)
        for path in excluded.values():
            Path(path).relative_to(analysis_root)
        original_local.write_text("temporary replacement\n", encoding="utf-8")
        try:
            observed["analyzed"] = snapshot_local.read_text(encoding="utf-8")
        finally:
            original_local.write_text(
                "frozen local_evidence\n",
                encoding="utf-8",
            )
        return Path(kwargs["output_dir"]), None, {"snapshot_only": True}

    monkeypatch.setattr(
        finalizer,
        "_finalize_freeze_b_from_snapshot",
        analyze_snapshot,
    )
    _private, _public, verification = finalizer.finalize_freeze_b(**arguments)
    assert observed == {"analyzed": "frozen local_evidence\n"}
    assert verification == {"snapshot_only": True}
    assert not tuple((Path(arguments["project_root"]) / "results").glob(".*analysis-snapshot*"))


def test_copy_first_snapshot_rejects_replace_restore_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    arguments = _analysis_snapshot_fixture(tmp_path, finalizer)
    original_copy = finalizer._copy_tree_strict
    local_payload = Path(arguments["local_evidence"]) / "payload.txt"

    def copy_with_replace_restore(
        source: Path,
        destination: Path,
        label: str,
    ) -> None:
        if label != "local evidence":
            original_copy(source, destination, label)
            return
        local_payload.write_text("temporary replacement\n", encoding="utf-8")
        try:
            original_copy(source, destination, label)
        finally:
            local_payload.write_text(
                "frozen local_evidence\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(finalizer, "_copy_tree_strict", copy_with_replace_restore)
    monkeypatch.setattr(
        finalizer,
        "_finalize_freeze_b_from_snapshot",
        lambda **_kwargs: pytest.fail("analysis must not start after a raced copy"),
    )
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="changed while creating the analysis snapshot",
    ):
        finalizer.finalize_freeze_b(**arguments)
    assert not tuple((Path(arguments["project_root"]) / "results").glob(".*analysis-snapshot*"))


def test_public_promotion_failure_rolls_back_private_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    private_staging = tmp_path / "private.staging"
    public_staging = tmp_path / "public.staging"
    private_destination = tmp_path / "private-final"
    public_destination = tmp_path / "public-final"
    private_staging.mkdir()
    public_staging.mkdir()
    (private_staging / "payload.txt").write_text("private\n", encoding="utf-8")
    finalizer.write_bundle_manifest(
        private_staging,
        finalizer.bundle_payload_paths(private_staging),
    )
    private_root = finalizer.verify_exact_bundle(private_staging)["root_sha256"]
    (public_staging / "summary.json").write_text(
        '{"schema_version":1}\n', encoding="utf-8"
    )
    (public_staging / "report.md").write_text(
        "# Public\n", encoding="utf-8"
    )
    original_promote = finalizer.promote_staging_root
    calls = 0

    def fail_public(staging: Path, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected public promotion failure")
        return original_promote(staging, destination)

    monkeypatch.setattr(finalizer, "promote_staging_root", fail_public)
    with pytest.raises(RuntimeError, match="injected"):
        finalizer._promote_output_pair(
            private_staging=private_staging,
            private_destination=private_destination,
            expected_private_root_sha256=str(private_root),
            public_staging=public_staging,
            public_destination=public_destination,
        )

    assert calls == 2
    assert not private_staging.exists()
    assert not public_staging.exists()
    assert not private_destination.exists()
    assert not public_destination.exists()


def test_file_identity_rejects_source_or_staged_tampering(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    source = tmp_path / "source.txt"
    copied = tmp_path / "copied.txt"
    source.write_text("frozen\n", encoding="utf-8")
    copied.write_text("frozen\n", encoding="utf-8")
    identity = finalizer._file_identity(source, "source")
    finalizer._require_file_identity(copied, identity, "copied")

    copied.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(finalizer.FreezeBFinalizationError, match="changed"):
        finalizer._require_file_identity(copied, identity, "copied")
    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(finalizer.FreezeBFinalizationError, match="changed"):
        finalizer._require_file_identity(source, identity, "source")


def test_private_and_public_output_destinations_must_be_distinct(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    candidate = ROOT / "results" / f"freeze-b-finalizer-{tmp_path.name}"
    with pytest.raises(finalizer.FreezeBFinalizationError, match="must be distinct"):
        finalizer._validate_output_destinations(
            ROOT,
            output_dir=candidate,
            public_output_dir=candidate,
            input_trees=(),
        )

    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    with pytest.raises(finalizer.FreezeBFinalizationError, match="disjoint"):
        finalizer._validate_disjoint_input_trees((outer.resolve(), inner.resolve()))


@pytest.mark.parametrize("platform", ["windows", "posix"])
def test_local_fresh_process_identity_is_exact_and_hashed(platform: str) -> None:
    finalizer = _load_finalizer()
    identity = (
        {
            "platform": "windows",
            "pid": 123,
            "process_creation_filetime": 456,
        }
        if platform == "windows"
        else {
            "platform": "posix",
            "boot_id": "boot-id",
            "pid": 123,
            "process_start_ticks": 456,
        }
    )
    record = {
        "schema_version": 1,
        "experiment_case_id": "experiment",
        "canonical_case_id": "canonical",
        "os_process_identity": identity,
        "fresh_process_id": finalizer.canonical_json_sha256(identity),
    }
    assert finalizer._validate_local_fresh_process_record(
        record,
        experiment_case_id="experiment",
        canonical_case_id="canonical",
    ) == record["fresh_process_id"]
    record["fresh_process_id"] = "0" * 64
    with pytest.raises(finalizer.FreezeBFinalizationError, match="digest"):
        finalizer._validate_local_fresh_process_record(
            record,
            experiment_case_id="experiment",
            canonical_case_id="canonical",
        )


def test_remote_archive_hash_must_match_exact_local_git_archive() -> None:
    finalizer = _load_finalizer()
    revision = finalizer._git(ROOT, "rev-parse", "HEAD")
    tree = finalizer._git(ROOT, "rev-parse", "HEAD^{tree}")
    archive_bytes = finalizer._git_archive_bytes(ROOT, revision)
    archive_sha256 = finalizer._git_archive_sha256(ROOT, revision)
    snapshot_sha256 = finalizer._virtual_snapshot_sha256(
        archive_bytes,
        source_revision=revision,
        source_tree=tree,
        archive_sha256=archive_sha256,
    )

    assert finalizer._validate_remote_source_archive(
        ROOT,
        execution_revision=revision,
        remote_archive_sha256=archive_sha256,
    ) == archive_sha256
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="not the exact local execution-commit archive",
    ):
        finalizer._validate_remote_source_archive(
            ROOT,
            execution_revision=revision,
            remote_archive_sha256="0" * 64,
        )
    assert finalizer._validate_remote_source_snapshot(
        ROOT,
        execution_revision=revision,
        execution_tree=tree,
        remote_archive_sha256=archive_sha256,
        remote_snapshot_sha256=snapshot_sha256,
    ) == {
        "source_archive_sha256": archive_sha256,
        "source_snapshot_sha256": snapshot_sha256,
    }
    altered_marker_snapshot = finalizer._virtual_snapshot_sha256(
        archive_bytes,
        source_revision="f" * 40,
        source_tree=tree,
        archive_sha256=archive_sha256,
    )
    assert altered_marker_snapshot != snapshot_sha256
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="locally reproduced installer snapshot",
    ):
        finalizer._validate_remote_source_snapshot(
            ROOT,
            execution_revision=revision,
            execution_tree=tree,
            remote_archive_sha256=archive_sha256,
            remote_snapshot_sha256=altered_marker_snapshot,
        )


def test_virtual_snapshot_digest_matches_real_installer(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    installer = _load_snapshot_installer()
    revision = finalizer._git(ROOT, "rev-parse", "HEAD")
    tree = finalizer._git(ROOT, "rev-parse", "HEAD^{tree}")
    archive_bytes = finalizer._git_archive_bytes(ROOT, revision)
    archive_sha256 = sha256(archive_bytes).hexdigest()
    root = tmp_path / "gate0"
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    (root / "project").mkdir()
    archive = downloads / "source.tar"
    archive.write_bytes(archive_bytes)

    installed = installer.install_snapshot(
        root,
        archive,
        expected_archive_sha256=archive_sha256,
        source_revision=revision,
        source_tree=tree,
    )
    assert finalizer._virtual_snapshot_sha256(
        archive_bytes,
        source_revision=revision,
        source_tree=tree,
        archive_sha256=archive_sha256,
    ) == installed["snapshot_sha256"]


def test_public_redaction_rejects_paths_machine_ids_and_joint_values() -> None:
    finalizer = _load_finalizer()
    finalizer._validate_public_safe(
        {
            "validation_status": "VALID",
            "scientific_label": "INCONCLUSIVE",
            "mapping": {"joint_mapping": "44/44"},
            "aggregate": [0.1, 0.2],
        }
    )
    for payload in (
        {"session_id": "private"},
        {"note": "/data/home/exampleuser/private"},
        {"note": "GPU-11111111-2222"},
        {"expected_pre_values": [1.0]},
        {"note": canonical_joint_names("left")[0]},
    ):
        with pytest.raises(finalizer.FreezeBFinalizationError):
            finalizer._validate_public_safe(payload)


def _diagnostic_collected() -> tuple[object, CollectedRun]:
    manifest = load_manifest(ROOT / "configs/parity/gate0.json")
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.case_id == "mujoco.left.small_step.base.r01"
    )
    hand = manifest.hand(HandSide.LEFT)
    scenario = manifest.scenario("small_step")
    initial = canonical_initial_positions(scenario, hand.joint_names)
    samples = []
    for step in range(round(scenario.duration_s / case.dt_s) + 1):
        positions = dict(initial)
        samples.append(
            TraceSample(
                step=step,
                time_s=step * case.dt_s,
                qpos=tuple(positions[name] for name in hand.joint_names),
                qvel=(0.0,) * 22,
                joint_positions=positions,
                frame_poses={
                    name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                    for name in hand.distal_frame_names
                },
                position_targets=canonical_position_targets(
                    scenario,
                    hand.joint_names,
                    step_index=step,
                    dt_s=case.dt_s,
                ),
                contact_count=0,
            )
        )
    provenance = {
        "backend_joint_names": list(hand.joint_names),
        "joint_mapping": [
            {
                "canonical_id": name,
                "backend_name": name,
                "backend": "mujoco",
                "scope": "left",
                "index": index,
                "sign": 1.0,
                "offset": 0.0,
                "unit": "rad",
            }
            for index, name in enumerate(hand.joint_names)
        ],
        "backend_frame_names": list(hand.distal_frame_names),
        "frame_mapping": [
            {
                "canonical_id": name,
                "backend_name": name,
                "backend": "mujoco",
                "scope": "left",
                "index": index,
                "sign": 1.0,
                "offset": 0.0,
                "unit": "xyz_m_qwxyz",
            }
            for index, name in enumerate(hand.distal_frame_names)
        ],
    }
    return manifest, CollectedRun(
        case=case,
        result=AdapterRunResult(
            backend="mujoco",
            scenario_id="small_step",
            status="completed",
            message="diagnostic fixture",
            dt=case.dt_s,
            requested_steps=len(samples) - 1,
            completed_steps=len(samples) - 1,
            joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
            samples=tuple(samples),
            provenance=provenance,
        ),
    )


@pytest.mark.parametrize(
    "tamper",
    ["target", "result_dt", "mapping_index", "qpos", "huge_qvel", "huge_position"],
)
def test_case_trace_rejects_any_canonical_diagnostic_tamper(tamper: str) -> None:
    finalizer = _load_finalizer()
    manifest, run = _diagnostic_collected()
    finalizer._assert_case_trace(
        run,
        canonical_case=run.case,
        backend=Simulator.MUJOCO,
        manifest=manifest,
    )

    result = run.result
    if tamper == "target":
        samples = list(result.samples)
        targets = dict(samples[1].position_targets)
        targets[result.joint_names[0]] += 0.01
        samples[1] = replace(samples[1], position_targets=targets)
        result = replace(result, samples=tuple(samples))
    elif tamper == "result_dt":
        result = replace(result, dt=result.dt * 2.0)
    elif tamper == "mapping_index":
        provenance = deepcopy(dict(result.provenance))
        provenance["joint_mapping"][0]["index"] = 1
        result = replace(result, provenance=provenance)
    elif tamper == "qpos":
        samples = list(result.samples)
        qpos = list(samples[0].qpos)
        qpos[0] += 0.01
        samples[0] = replace(samples[0], qpos=tuple(qpos))
        result = replace(result, samples=tuple(samples))
    elif tamper == "huge_qvel":
        samples = list(result.samples)
        qvel = list(samples[1].qvel)
        qvel[0] = 1001.0
        samples[1] = replace(samples[1], qvel=tuple(qvel))
        result = replace(result, samples=tuple(samples))
    else:
        samples = list(result.samples)
        qpos = list(samples[1].qpos)
        positions = dict(samples[1].joint_positions)
        qpos[0] = 7.0
        positions[result.joint_names[0]] = 7.0
        samples[1] = replace(
            samples[1], qpos=tuple(qpos), joint_positions=positions
        )
        result = replace(result, samples=tuple(samples))

    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="canonical diagnostic contract|sanity bound",
    ):
        finalizer._assert_case_trace(
            replace(run, result=result),
            canonical_case=run.case,
            backend=Simulator.MUJOCO,
            manifest=manifest,
        )


def _collected(delta: float) -> CollectedRun:
    manifest = load_manifest(ROOT / "configs/parity/gate0.json")
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.case_id == "mujoco.left.small_step.base.r01"
    )
    hand = manifest.hand(HandSide.LEFT)
    samples = []
    for step, time_s in enumerate((0.0, 0.002)):
        samples.append(
            TraceSample(
                step=step,
                time_s=time_s,
                qpos=(delta,) * 22,
                qvel=(0.0,) * 22,
                joint_positions={name: delta for name in hand.joint_names},
                frame_poses={
                    name: (delta, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                    for name in hand.distal_frame_names
                },
                contact_count=0,
            )
        )
    result = AdapterRunResult(
        backend="mujoco",
        scenario_id="small_step",
        status="completed",
        message="fixture",
        dt=case.dt_s,
        requested_steps=1,
        completed_steps=1,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=tuple(samples),
    )
    return CollectedRun(case=case, result=result)


def test_delta_metrics_cover_bridge_repeat_and_dt_numerics() -> None:
    finalizer = _load_finalizer()
    reference = _collected(0.0)
    candidate = _collected(1e-10)
    hand = load_manifest(ROOT / "configs/parity/gate0.json").hand(HandSide.LEFT)
    delta = finalizer.DeltaMetrics.between(reference, candidate, hand)
    assert delta.joint_max_abs_rad == pytest.approx(1e-10)
    assert delta.frame_position_max_m == pytest.approx(1e-10)
    assert delta.frame_orientation_max_rad == 0.0
    assert delta.within(joint=1e-9, position=1e-9, orientation=1e-9)


def test_invalid_report_assigns_no_scientific_label() -> None:
    finalizer = _load_finalizer()
    summary = {
        "validation_status": "INVALID",
        "scientific_label": None,
        "execution": {"completed_case_count": 24},
        "mapping": {"joint_mapping": "44/44", "distal_frame_mapping": "10/10"},
        "runtime_friction_observation": {
            "zero_changed_case_count_from_r1": 0,
            "zero_case_count": 8,
        },
        "position_target_readback": {
            "aggregate_max_abs_error_rad": 0.0,
            "limit_rad": 1e-6,
        },
        "validity": {
            "checks_all_passed": True,
            "primary_metric_evaluation_passed": False,
            "all_passed": False,
        },
    }
    report = finalizer._render_report(summary)
    assert "No scientific label is assigned" in report
    assert "Validation status: **INVALID**" in report
    assert "Scientific label: not assigned" in report
    assert "formal Gate 0 result remains **DIVERGENT**" in report
    assert "OVPhysX contacts were not measured" in report
    assert "not evidence of zero contacts or contact absence" in report
    assert "requested/configured dt" in report
    assert "same-step, same-length recorded timestamp compatibility" in report
    assert "within 1e-12 s" in report
    assert "200 base/400 halved" in report
    assert "201 base/401 halved" in report
    assert "effective-dt getter" in report
    assert "All four prior evidence trees" in report
    assert "operator invocation disclosed a noncanonical asset materialization" in report
    assert "before adapter construction" in report
    assert "corrected G execution" in report
    assert "Preregistered evidence checks passed: true" in report
    assert "Primary metric construction and evaluation passed: false" in report
    assert (
        "primary metric construction or evaluation failed after every "
        "preregistered evidence check passed"
    ) in report
    assert "at least one preregistered validity gate failed" not in report
    assert "local, unpublished INVALID/null bundle" in report
    assert "does not alter or round timestamps" in report
    assert "external/lfstd" not in report


def test_invalid_metric_decision_cannot_publish_all_passed_true() -> None:
    finalizer = _load_finalizer()
    checks = {"frozen_evidence_gate": True}
    decision = finalizer.FreezeBDecision(
        finalizer.SensitivityStatus.INVALID,
        validity_failures=(
            "metric traces lack a compatible recorded time grid",
        ),
    )
    summary = {
        "validation_status": "INVALID",
        "scientific_label": None,
        "primary_metrics": None,
        "validity": {
            "checks_all_passed": True,
            "primary_metric_evaluation_passed": False,
            "all_passed": False,
            "checks": dict(checks),
        },
    }
    finalizer._validate_summary_decision_consistency(
        summary,
        decision=decision,
        checks=checks,
    )
    summary["validity"]["all_passed"] = True
    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="contradicts",
    ):
        finalizer._validate_summary_decision_consistency(
            summary,
            decision=decision,
            checks=checks,
        )


def test_legacy_a1_inventory_is_rejected_before_any_run_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "legacy-a1"
    launcher = root / "launcher"
    case = (
        root
        / "cases"
        / "freeze_b.mujoco.left.small_step.base.control.r01"
    )
    launcher.mkdir(parents=True)
    case.mkdir(parents=True)
    for name in (
        "freeze_b.private-plan.json",
        "gate0.manifest.json",
        "matrix.json",
        "provenance.json",
    ):
        (launcher / name).write_text("{}\n", encoding="utf-8")
    for name in (
        "fresh-process.json",
        "launcher.command.json",
        "launcher.exitcode.txt",
        "launcher.stderr.log",
        "launcher.stdout.log",
        "mujoco.left.small_step.base.r01.run.json",
    ):
        (case / name).write_text("{}\n", encoding="utf-8")
    deserialized = False

    def forbidden_deserialization(_path: Path) -> object:
        nonlocal deserialized
        deserialized = True
        raise AssertionError("legacy a1 run must never be deserialized by finalization")

    monkeypatch.setattr(finalizer, "load_collected_runs", forbidden_deserialization)

    with pytest.raises(
        finalizer.FreezeBFinalizationError,
        match="completed corrected attempt",
    ):
        finalizer._load_local_runs(
            root,
            plan={},
            plan_path=tmp_path / "unused-plan.json",
            manifest=object(),
            execution_revision="d" * 40,
            execution_tree="e" * 40,
            project_root=tmp_path,
            source_identity={},
        )
    assert deserialized is False
