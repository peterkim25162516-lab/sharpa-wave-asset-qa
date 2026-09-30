from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import struct

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import HandSide, Simulator
from wave_asset_qa.parity.scenarios import ScenarioCase, TimestepVariant
from wave_asset_qa.parity.runner import adapter_run_result_from_dict
from wave_asset_qa.parity import sensitivity


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _inputs() -> dict[str, object]:
    readback_ids = (
        "ovphysx.left.effective_readback.r01",
        "ovphysx.left.effective_readback.r02",
        "ovphysx.right.effective_readback.r01",
        "ovphysx.right.effective_readback.r02",
    )
    baselines = {
        hand: {
            variant: {
                "joint_rmse_rad": 0.02,
                "frame_position_rmse_m": 0.003,
            }
            for variant in ("base", "halved")
        }
        for hand in ("left", "right")
    }
    return {
        "freeze_a_config_sha256": "a" * 64,
        "readback_bundle_root_sha256": "b" * 64,
        "readback_summary_sha256": "c" * 64,
        "readback_source_revision": "1" * 40,
        "readback_source_tree": "2" * 40,
        "readback_case_sha256": {
            case_id: f"{index:x}" * 64
            for index, case_id in enumerate(readback_ids, start=3)
        },
        "formal_gate0_bundle_root_sha256": "d" * 64,
        "formal_gate0_source_revision": "4" * 40,
        "gate0_manifest_file_sha256": "e" * 64,
        "gate0_manifest_semantic_sha256": "f" * 64,
        "asset_commit": "5" * 40,
        "asset_git_tree": "6" * 40,
        "canonical_lf_asset_tree_sha256": "0" * 64,
        "freeze_b_source_revision": "7" * 40,
        "freeze_b_source_tree": "8" * 40,
        "formal_crosssim_window_rmse": baselines,
    }


def _private_plan() -> dict[str, object]:
    hand_plans: list[dict[str, object]] = []
    for hand in (HandSide.LEFT, HandSide.RIGHT):
        names = sensitivity.canonical_joint_names(hand)
        values = {
            name: _float32(0.125 + index / 1024.0)
            for index, name in enumerate(names)
        }
        hand_plans.append(
            {
                "hand": hand.value,
                "joint_names": list(names),
                "joint_prim_paths": {
                    name: f"/World/Env_0/Robot/joints/{name}" for name in names
                },
                "expected_pre_values": dict(values),
                "sham_write_values": dict(values),
                "zero_write_values": {name: 0.0 for name in names},
                "expected_pre_float32_sha256": (
                    sensitivity.expected_pre_float32_sha256(names, values)
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": "fixture-freeze-b-v1",
        "inputs": _inputs(),
        "hand_plans": hand_plans,
        "ovphysx_cases": list(
            sensitivity.expected_case_matrix(Simulator.OVPHYSX)
        ),
        "mujoco_cases": list(
            sensitivity.expected_case_matrix(Simulator.MUJOCO)
        ),
        "run_order": list(sensitivity.expected_run_order()),
    }


def _public_protocol(plan: object) -> dict[str, object]:
    return sensitivity.build_public_protocol(
        plan,
        candidate_hypothesis=sensitivity.FROZEN_CANDIDATE_HYPOTHESIS,
        allowed_claim=sensitivity.FROZEN_ALLOWED_CLAIM,
        maximum_gpu_hours=sensitivity.FROZEN_MAXIMUM_GPU_HOURS,
    )


def _evidence_bindings(
    plan: object | None = None,
    public: object | None = None,
) -> dict[str, object]:
    selected_plan = _private_plan() if plan is None else plan
    selected_public = (
        _public_protocol(selected_plan) if public is None else public
    )
    return {
        "private_plan_sha256": sensitivity.private_plan_sha256(selected_plan),
        "public_protocol_file_sha256": "9" * 64,
        "public_protocol_canonical_sha256": sensitivity.canonical_json_sha256(
            selected_public
        ),
        "deployment_source_revision": "9" * 40,
        "deployment_source_tree": "a" * 40,
        "deployment_source_snapshot_sha256": "b" * 64,
    }


def _metric_kwargs(
    inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_inputs = _inputs() if inputs is None else deepcopy(inputs)
    plan = _private_plan()
    plan["inputs"] = deepcopy(selected_inputs)
    public = _public_protocol(plan)
    return {
        "frozen_inputs": selected_inputs,
        "private_plan": plan,
        "public_protocol": public,
        "evidence_bindings": _evidence_bindings(plan, public),
    }


def _validity_checks() -> dict[str, bool]:
    return {name: True for name in sensitivity.REQUIRED_VALIDITY_CHECKS}


def _decision_cells(
    values: list[float] | tuple[float, ...],
) -> tuple[sensitivity.PrimaryMetricCell, ...]:
    if len(values) != 8:
        raise ValueError("decision fixture requires eight S values")
    window = sensitivity.WindowMetrics(
        sample_count=1,
        joint_element_count=22,
        position_element_count=15,
        joint_rmse_rad=0.02,
        frame_position_rmse_m=0.003,
    )
    cells: list[sensitivity.PrimaryMetricCell] = []
    index = 0
    for hand in (HandSide.LEFT, HandSide.RIGHT):
        for variant in (TimestepVariant.BASE, TimestepVariant.HALVED):
            cells.append(
                sensitivity.PrimaryMetricCell(
                    hand=hand,
                    timestep_variant=variant,
                    crosssim_baseline=window,
                    treatment_vs_sham=window,
                    zero_vs_mujoco=window,
                    S_Dq=values[index],
                    S_Dp=values[index + 1],
                    R_Dq=0.0,
                    R_Dp=0.0,
                )
            )
            index += 2
    return tuple(cells)


def test_private_plan_exact_schema_hash_and_case_inventory() -> None:
    plan = _private_plan()
    validated = sensitivity.validate_private_plan(plan)

    assert set(validated) == {
        "schema_version",
        "protocol_id",
        "inputs",
        "hand_plans",
        "ovphysx_cases",
        "mujoco_cases",
        "run_order",
    }
    assert len(validated["hand_plans"]) == 2
    assert all(len(item["joint_names"]) == 22 for item in validated["hand_plans"])
    assert len(validated["ovphysx_cases"]) == 16
    assert len(validated["mujoco_cases"]) == 8
    assert len(validated["run_order"]) == 16
    assert validated["run_order"][:4] == [
        "freeze_b.ovphysx.left.small_step.base.sham.r01",
        "freeze_b.ovphysx.left.small_step.base.zero.r01",
        "freeze_b.ovphysx.left.small_step.base.zero.r02",
        "freeze_b.ovphysx.left.small_step.base.sham.r02",
    ]
    first_mujoco = validated["mujoco_cases"][0]
    assert first_mujoco["role"] == "control"
    assert first_mujoco["experiment_case_id"] == (
        "freeze_b.mujoco.left.small_step.base.control.r01"
    )
    digest = sensitivity.private_plan_sha256(plan)
    assert len(digest) == 64
    assert digest == sensitivity.canonical_json_sha256(validated)
    assert "private_plan_sha256" not in validated
    serialized = json.dumps(validated)
    assert "C:\\\\" not in serialized and "/data/home/" not in serialized


def test_private_plan_float32_domain_hash_and_sham_zero_contract() -> None:
    plan = _private_plan()
    left = plan["hand_plans"][0]
    names = tuple(left["joint_names"])
    values = left["expected_pre_values"]

    assert sensitivity.expected_pre_float32_sha256(names, values) == left[
        "expected_pre_float32_sha256"
    ]
    resolved = sensitivity.experiment_case(
        plan, "freeze_b.ovphysx.left.small_step.base.sham.r01"
    )
    assert resolved["case"]["canonical_case_id"] == (
        "ovphysx.left.small_step.base.r01"
    )
    assert resolved["write_values"] == resolved["expected_pre_values"]
    zero = sensitivity.experiment_case(
        plan, "freeze_b.ovphysx.left.small_step.base.zero.r01"
    )
    assert set(zero["write_values"].values()) == {0.0}

    bad = deepcopy(plan)
    bad["hand_plans"][0]["sham_write_values"][names[0]] += 0.01
    with pytest.raises(sensitivity.SensitivityValidationError, match="sham"):
        sensitivity.validate_private_plan(bad)


def test_private_plan_rejects_unknown_fields_matrix_drift_and_bad_hash() -> None:
    plan = _private_plan()
    extra = deepcopy(plan)
    extra["machine_id"] = "forbidden"
    with pytest.raises(sensitivity.SensitivityValidationError, match="unknown"):
        sensitivity.validate_private_plan(extra)

    nested = deepcopy(plan)
    nested["inputs"]["unknown_hash"] = "0" * 64
    with pytest.raises(sensitivity.SensitivityValidationError, match="unknown"):
        sensitivity.validate_private_plan(nested)

    reordered = deepcopy(plan)
    reordered["ovphysx_cases"][0], reordered["ovphysx_cases"][1] = (
        reordered["ovphysx_cases"][1],
        reordered["ovphysx_cases"][0],
    )
    with pytest.raises(sensitivity.SensitivityValidationError, match="matrix"):
        sensitivity.validate_private_plan(reordered)

    bad_vector_hash = deepcopy(plan)
    bad_vector_hash["hand_plans"][0]["expected_pre_float32_sha256"] = "0" * 64
    with pytest.raises(sensitivity.SensitivityValidationError, match="mismatch"):
        sensitivity.validate_private_plan(bad_vector_hash)


def test_public_protocol_is_exact_redacted_and_pins_private_plan() -> None:
    plan = _private_plan()
    public = _public_protocol(plan)
    validated = sensitivity.validate_public_protocol(public)

    assert validated["private_plan_sha256"] == sensitivity.private_plan_sha256(plan)
    assert validated["inputs"] == plan["inputs"]
    assert validated["decision_thresholds"] == {
        "expected_s_count": 8,
        "SUPPORTED_if_all_S_at_least": 0.25,
        "NOT_SUPPORTED_if_all_S_at_most": 0.1,
        "otherwise": "INCONCLUSIVE",
        "validity_failure": (
            "validation_status_INVALID_and_scientific_label_null"
        ),
    }
    assert validated["metrics"]["repeat_collapse"] == (
        "primary_metrics_use_r01_only_after_r01_r02_repeatability_gate_passes_"
        "for_every_backend_hand_dt_role;_r02_is_gate_only_and_must_not_be_"
        "selected_for_favorable_results"
    )
    assert validated["validity_thresholds"] == {
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
        "max_abs_canonical_joint_position_rad": math.tau,
        "max_abs_canonical_position_target_rad": math.tau,
        "max_abs_backend_joint_velocity_rad_s": 1_000.0,
        "max_frame_origin_distance_m": 10.0,
        "mujoco_contact_count": 0,
        "ovphysx_contact_count": None,
        "contact_observation_boundary": "ovphysx_contact_check_performed_false",
    }
    assert len(sensitivity.REQUIRED_VALIDITY_CHECKS) == 15
    assert "fresh_process_24_of_24" in sensitivity.REQUIRED_VALIDITY_CHECKS
    assert "fresh_process_remote_16_of_16" not in (
        sensitivity.REQUIRED_VALIDITY_CHECKS
    )
    assert "position_target_readback_within_1e_6" in (
        sensitivity.REQUIRED_VALIDITY_CHECKS
    )
    assert "finite_and_sanity_bounds" in sensitivity.REQUIRED_VALIDITY_CHECKS
    assert validated["validity_thresholds"]["ovphysx_contact_count"] is None
    assert validated["validity_thresholds"]["contact_observation_boundary"] == (
        "ovphysx_contact_check_performed_false"
    )
    assert validated["intervention"]["application_time"] == (
        "live_composed_stage_after_articulation_construction_while_"
        "articulation_uninitialized_before_first_simulation_reset"
    )
    serialized = json.dumps(validated)
    assert "expected_pre_values" not in serialized
    assert "/World/Env_0/Robot/joints/" not in serialized

    assert (
        validated["candidate_hypothesis"]
        == sensitivity.FROZEN_CANDIDATE_HYPOTHESIS
    )
    assert (
        validated["claim_boundary"]["allowed"]
        == sensitivity.FROZEN_ALLOWED_CLAIM
    )
    assert validated["bounded_compute_budget"]["maximum_gpu_hours"] == 0.5

    for field, replacement in (
        ("candidate_hypothesis", "private host path C:/secret"),
        ("claim_boundary.allowed", "an expanded post-hoc claim"),
        ("bounded_compute_budget.maximum_gpu_hours", 2.0),
    ):
        tampered = deepcopy(public)
        if "." in field:
            parent, child = field.split(".", 1)
            tampered[parent][child] = replacement
        else:
            tampered[field] = replacement
        with pytest.raises(sensitivity.SensitivityValidationError):
            sensitivity.validate_public_protocol(tampered)

    with pytest.raises(sensitivity.SensitivityValidationError):
        sensitivity.build_public_protocol(
            plan,
            candidate_hypothesis="a different hypothesis",
            allowed_claim=sensitivity.FROZEN_ALLOWED_CLAIM,
            maximum_gpu_hours=sensitivity.FROZEN_MAXIMUM_GPU_HOURS,
        )

    unknown = deepcopy(public)
    unknown["notes"] = "post-hoc"
    with pytest.raises(sensitivity.SensitivityValidationError, match="unknown"):
        sensitivity.validate_public_protocol(unknown)


def test_strict_loaders_detect_hash_drift_and_duplicate_keys(tmp_path: Path) -> None:
    plan = _private_plan()
    public = _public_protocol(plan)
    plan_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    public_path.write_text(json.dumps(public), encoding="utf-8")

    assert sensitivity.load_public_protocol(public_path) == public
    assert sensitivity.load_private_plan(
        plan_path,
        public_protocol=public,
        expected_sha256=sensitivity.private_plan_sha256(plan),
    ) == plan
    with pytest.raises(sensitivity.SensitivityValidationError, match="SHA-256"):
        sensitivity.load_private_plan(plan_path, expected_sha256="0" * 64)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(sensitivity.SensitivityValidationError, match="duplicate"):
        sensitivity.load_private_plan(duplicate)


def _run(
    backend: Simulator,
    hand: HandSide,
    variant: TimestepVariant,
    *,
    joint_value: float,
    frame_value: float,
    origin: str,
    role: str | None = None,
    study_inputs: dict[str, object] | None = None,
) -> CollectedRun:
    dt_s = 0.002 if variant is TimestepVariant.BASE else 0.001
    steps = round(0.5 / dt_s)
    joints = sensitivity.canonical_joint_names(hand)
    frames = sensitivity.canonical_frame_names(hand)
    samples: list[TraceSample] = []
    for step in range(steps + 1):
        time_s = step * dt_s
        # Large pre-window differences prove that selection uses recorded time.
        q_value = 6.0 if time_s < 0.1 else joint_value
        p_value = 5.0 if time_s < 0.1 else frame_value
        positions = {name: q_value for name in joints}
        poses = {
            name: (p_value, p_value, p_value, 1.0, 0.0, 0.0, 0.0)
            for name in frames
        }
        samples.append(
            TraceSample(
                step=step,
                time_s=time_s,
                qpos=tuple(positions.values()),
                qvel=(0.0,) * 22,
                joint_positions=positions,
                frame_poses=poses,
                position_targets=dict(positions),
                contact_count=0 if backend is Simulator.MUJOCO else None,
            )
        )
    case = ScenarioCase(
        simulator=backend,
        hand=hand,
        scenario_id="small_step",
        repeat_index=1,
        timestep_variant=variant,
        dt_s=dt_s,
        model_path=f"models/{hand.value}.{'xml' if backend is Simulator.MUJOCO else 'usda'}",
    )
    inputs = _inputs() if study_inputs is None else deepcopy(study_inputs)
    plan = _private_plan()
    plan["inputs"] = deepcopy(inputs)
    public = _public_protocol(plan)
    bindings = _evidence_bindings(plan, public)
    if origin == "formal":
        provenance: dict[str, object] = {
            "source_revision": inputs["formal_gate0_source_revision"],
            "manifest_sha256": inputs["gate0_manifest_semantic_sha256"],
            "asset_commit": inputs["asset_commit"],
            "asset_git_tree": inputs["asset_git_tree"],
            "asset_tree_sha256": inputs[
                "canonical_lf_asset_tree_sha256"
            ],
        }
        bundle_root = str(inputs["formal_gate0_bundle_root_sha256"])
    elif origin == "remote" and backend is Simulator.OVPHYSX and role in {
        "sham",
        "zero",
    }:
        experiment_id = (
            f"freeze_b.ovphysx.{hand.value}.small_step."
            f"{variant.value}.{role}.r01"
        )
        provenance = {
            "contact_check_performed": False,
            "freeze_b_protocol_id": plan["protocol_id"],
            "private_plan_sha256": bindings["private_plan_sha256"],
            "public_protocol_file_sha256": bindings[
                "public_protocol_file_sha256"
            ],
            "public_protocol_canonical_sha256": bindings[
                "public_protocol_canonical_sha256"
            ],
            "experiment_case_id": experiment_id,
            "canonical_case_id": case.case_id,
            "freeze_b_role": role,
            "hand": hand.value,
            "timestep_variant": variant.value,
            "repeat_index": 1,
            "manifest_sha256": inputs["gate0_manifest_semantic_sha256"],
            "manifest_file_sha256": inputs["gate0_manifest_file_sha256"],
            "asset_commit": inputs["asset_commit"],
            "asset_git_tree": inputs["asset_git_tree"],
            "asset_tree_sha256": inputs[
                "canonical_lf_asset_tree_sha256"
            ],
            "source_revision": bindings["deployment_source_revision"],
            "source_tree": bindings["deployment_source_tree"],
            "deployment_source_revision": bindings[
                "deployment_source_revision"
            ],
            "deployment_source_tree": bindings["deployment_source_tree"],
            "implementation_source_revision": inputs[
                "freeze_b_source_revision"
            ],
            "implementation_source_tree": inputs["freeze_b_source_tree"],
            "source_snapshot_sha256": bindings[
                "deployment_source_snapshot_sha256"
            ],
        }
        bundle_root = None
    else:
        raise AssertionError("synthetic run origin/role is invalid")
    if backend is Simulator.OVPHYSX:
        provenance["contact_check_performed"] = False
    return CollectedRun(
        case=case,
        result=AdapterRunResult(
            backend=backend.value,
            scenario_id="small_step",
            status="completed",
            message="synthetic",
            dt=dt_s,
            requested_steps=steps,
            completed_steps=steps,
            joint_names=joints,
            frame_names=frames,
            samples=tuple(samples),
            provenance=provenance,
        ),
        bundle_root_sha256=bundle_root,
    )


def _trace_sets(
    effect_scale: float = 0.5,
    *,
    study_inputs: dict[str, object] | None = None,
) -> tuple[sensitivity.SensitivityTraceSet, ...]:
    result: list[sensitivity.SensitivityTraceSet] = []
    for hand in (HandSide.LEFT, HandSide.RIGHT):
        for variant in (TimestepVariant.BASE, TimestepVariant.HALVED):
            mujoco = _run(
                Simulator.MUJOCO,
                hand,
                variant,
                joint_value=0.0,
                frame_value=0.0,
                origin="formal",
                study_inputs=study_inputs,
            )
            formal_ov = _run(
                Simulator.OVPHYSX,
                hand,
                variant,
                joint_value=0.02,
                frame_value=0.003,
                origin="formal",
                study_inputs=study_inputs,
            )
            sham = _run(
                Simulator.OVPHYSX,
                hand,
                variant,
                joint_value=0.02,
                frame_value=0.003,
                origin="remote",
                role="sham",
                study_inputs=study_inputs,
            )
            zero = _run(
                Simulator.OVPHYSX,
                hand,
                variant,
                joint_value=0.02 + 0.02 * effect_scale,
                frame_value=0.003 + 0.003 * effect_scale,
                origin="remote",
                role="zero",
                study_inputs=study_inputs,
            )
            result.append(
                sensitivity.SensitivityTraceSet(
                    hand=hand,
                    timestep_variant=variant,
                    formal_mujoco=mujoco,
                    formal_ovphysx=formal_ov,
                    sham_ovphysx=sham,
                    zero_ovphysx=zero,
                )
            )
    return tuple(result)


def test_primary_metrics_are_elementwise_rmse_on_inclusive_window() -> None:
    metrics = sensitivity.compute_primary_metrics(_trace_sets(), **_metric_kwargs())

    assert len(metrics) == 4
    assert metrics[0].crosssim_baseline.sample_count == 201
    assert metrics[1].crosssim_baseline.sample_count == 401
    assert metrics[0].crosssim_baseline.joint_element_count == 201 * 22
    assert metrics[0].crosssim_baseline.position_element_count == 201 * 5 * 3
    for cell in metrics:
        assert cell.crosssim_baseline.joint_rmse_rad == pytest.approx(0.02)
        assert cell.crosssim_baseline.frame_position_rmse_m == pytest.approx(0.003)
        assert cell.treatment_vs_sham.joint_rmse_rad == pytest.approx(0.01)
        assert cell.treatment_vs_sham.frame_position_rmse_m == pytest.approx(0.0015)
        assert cell.S_Dq == pytest.approx(0.5)
        assert cell.S_Dp == pytest.approx(0.5)
        assert cell.R_Dq == pytest.approx(-0.5)
        assert cell.R_Dp == pytest.approx(-0.5)
    decision = sensitivity.decide_freeze_b(
        metrics, validity_checks=_validity_checks()
    )
    assert decision.status is sensitivity.SensitivityStatus.SUPPORTED
    assert decision.scientific_label is sensitivity.SensitivityStatus.SUPPORTED
    serialized = decision.to_dict()
    assert serialized["validation_status"] == "VALID"
    assert serialized["scientific_label"] == "SUPPORTED"
    assert "status" not in serialized
    assert "execution_valid" not in serialized


def test_metric_time_grid_accepts_sub_ulp_drift_without_changing_recorded_window() -> None:
    traces = list(_trace_sets())
    for index, trace_set in enumerate(traces):
        reference = trace_set.formal_mujoco
        accumulated_time = 0.0
        samples: list[TraceSample] = []
        for sample in reference.result.samples:
            samples.append(replace(sample, time_s=accumulated_time))
            accumulated_time += reference.result.dt
        traces[index] = replace(
            trace_set,
            formal_mujoco=replace(
                reference,
                result=replace(reference.result, samples=tuple(samples)),
            ),
        )

    metrics = sensitivity.compute_primary_metrics(traces, **_metric_kwargs())

    assert [cell.crosssim_baseline.sample_count for cell in metrics] == [
        200,
        400,
        200,
        400,
    ]
    assert [cell.zero_vs_mujoco.sample_count for cell in metrics] == [
        200,
        400,
        200,
        400,
    ]
    assert [cell.treatment_vs_sham.sample_count for cell in metrics] == [
        201,
        401,
        201,
        401,
    ]
    assert all(
        cell.crosssim_baseline.joint_rmse_rad == 0.02
        and cell.crosssim_baseline.frame_position_rmse_m == 0.003
        for cell in metrics
    )
    assert traces[0].formal_mujoco.result.samples[-1].time_s > 0.5
    assert traces[0].formal_ovphysx.result.samples[-1].time_s == 0.5


def test_metric_time_grid_rejects_cross_trace_drift_over_tolerance() -> None:
    traces = list(_trace_sets())
    trace_set = traces[0]
    reference = trace_set.formal_mujoco
    candidate = trace_set.formal_ovphysx
    reference_samples = list(reference.result.samples)
    candidate_samples = list(candidate.result.samples)
    sample_index = 100
    canonical_time = sample_index * reference.result.dt
    reference_samples[sample_index] = replace(
        reference_samples[sample_index],
        time_s=canonical_time + 0.75e-12,
    )
    candidate_samples[sample_index] = replace(
        candidate_samples[sample_index],
        time_s=canonical_time - 0.75e-12,
    )
    traces[0] = replace(
        trace_set,
        formal_mujoco=replace(
            reference,
            result=replace(reference.result, samples=tuple(reference_samples)),
        ),
        formal_ovphysx=replace(
            candidate,
            result=replace(candidate.result, samples=tuple(candidate_samples)),
        ),
    )

    with pytest.raises(
        sensitivity.SensitivityValidationError,
        match="compatible recorded time grid",
    ):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())


@pytest.mark.parametrize("failure", ["wrong_step", "nonmonotonic_time"])
def test_metric_time_grid_still_rejects_invalid_single_trace_axes(
    failure: str,
) -> None:
    traces = list(_trace_sets())
    trace_set = traces[0]
    candidate = trace_set.formal_ovphysx
    samples = list(candidate.result.samples)
    if failure == "wrong_step":
        samples[10] = replace(samples[10], step=11)
    else:
        samples[10] = replace(samples[10], time_s=samples[9].time_s)
    traces[0] = replace(
        trace_set,
        formal_ovphysx=replace(
            candidate,
            result=replace(candidate.result, samples=tuple(samples)),
        ),
    )

    with pytest.raises(sensitivity.SensitivityValidationError):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())


def test_sort_keys_round_trip_preserves_coverage_without_requiring_dict_order() -> None:
    traces = list(_trace_sets())
    formal = traces[0].formal_mujoco
    serialized = json.dumps(formal.result.to_dict(), sort_keys=True)
    round_tripped = adapter_run_result_from_dict(json.loads(serialized))
    canonical = sensitivity.canonical_joint_names(HandSide.LEFT)
    assert tuple(round_tripped.samples[0].joint_positions) != canonical
    assert tuple(round_tripped.samples[0].position_targets) != canonical
    traces[0] = replace(
        traces[0],
        formal_mujoco=replace(formal, result=round_tripped),
    )

    metrics = sensitivity.compute_primary_metrics(traces, **_metric_kwargs())

    assert len(metrics) == 4
    assert metrics[0].S_Dq == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("experiment_case_id", "freeze_b.ovphysx.left.small_step.base.zero.r01"),
        ("canonical_case_id", "ovphysx.left.small_step.base.r02"),
        ("freeze_b_role", "zero"),
        ("hand", "right"),
        ("timestep_variant", "halved"),
        ("repeat_index", 2),
        ("private_plan_sha256", "0" * 64),
        ("public_protocol_file_sha256", "0" * 64),
        ("public_protocol_canonical_sha256", "0" * 64),
        ("implementation_source_revision", "0" * 40),
        ("deployment_source_tree", "0" * 40),
    ],
)
def test_remote_metric_identity_cannot_be_bypassed(
    field: str, bad_value: object
) -> None:
    traces = list(_trace_sets())
    sham = traces[0].sham_ovphysx
    provenance = dict(sham.result.provenance)
    provenance[field] = bad_value
    traces[0] = replace(
        traces[0],
        sham_ovphysx=replace(
            sham,
            result=replace(sham.result, provenance=provenance),
        ),
    )

    with pytest.raises(
        sensitivity.SensitivityValidationError,
        match="provenance identity mismatch",
    ):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())


def test_formal_bundle_and_deployment_bindings_cannot_be_self_reported_away() -> None:
    traces = list(_trace_sets())
    formal = traces[0].formal_mujoco
    traces[0] = replace(
        traces[0],
        formal_mujoco=replace(formal, bundle_root_sha256="0" * 64),
    )
    with pytest.raises(sensitivity.SensitivityValidationError, match="formal bundle root"):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())

    traces = list(_trace_sets())
    kwargs = _metric_kwargs()
    bindings = dict(kwargs["evidence_bindings"])
    bindings["deployment_source_revision"] = "0" * 40
    kwargs["evidence_bindings"] = bindings
    with pytest.raises(
        sensitivity.SensitivityValidationError,
        match="provenance identity mismatch",
    ):
        sensitivity.compute_primary_metrics(traces, **kwargs)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("simulator", Simulator.MUJOCO),
        ("hand", HandSide.RIGHT),
        ("scenario_id", "zero_hold"),
        ("timestep_variant", TimestepVariant.HALVED),
        ("repeat_index", 2),
    ],
)
def test_trace_set_rejects_noncanonical_case_components(
    field: str, bad_value: object
) -> None:
    traces = _trace_sets()
    original = traces[0]
    run = original.zero_ovphysx
    bad_case = replace(run.case, **{field: bad_value})
    with pytest.raises(
        sensitivity.SensitivityValidationError,
        match="canonical repeat-1",
    ):
        replace(original, zero_ovphysx=replace(run, case=bad_case))


def test_primary_metrics_fail_on_baseline_or_contact_boundary_drift() -> None:
    bad_inputs = _inputs()
    bad_inputs["formal_crosssim_window_rmse"]["left"]["base"][
        "joint_rmse_rad"
    ] += 2e-15
    with pytest.raises(sensitivity.SensitivityValidationError, match="baseline mismatch"):
        sensitivity.compute_primary_metrics(
            _trace_sets(study_inputs=bad_inputs),
            **_metric_kwargs(bad_inputs),
        )

    traces = list(_trace_sets())
    formal_ov = traces[0].formal_ovphysx
    bad_sample = replace(formal_ov.result.samples[0], contact_count=0)
    bad_result = replace(
        formal_ov.result,
        samples=(bad_sample, *formal_ov.result.samples[1:]),
    )
    traces[0] = replace(traces[0], formal_ovphysx=replace(formal_ov, result=bad_result))
    with pytest.raises(sensitivity.SensitivityValidationError, match="null/unobserved"):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("short_duration", "sample inventory"),
        ("nonfinite_qpos", "must be finite"),
        ("short_qvel", "exactly 22"),
        ("excessive_qvel", "joint-velocity sanity bound"),
        ("excessive_position", "joint-position sanity bound"),
        ("excessive_target", "position-target sanity bound"),
        ("distant_frame", "frame-origin-distance sanity bound"),
    ],
)
def test_trace_rejects_wrong_duration_or_invalid_backend_state(
    failure: str, message: str
) -> None:
    traces = list(_trace_sets())
    formal = traces[0].formal_mujoco
    result = formal.result
    if failure == "short_duration":
        result = replace(
            result,
            requested_steps=result.requested_steps - 1,
            completed_steps=result.completed_steps - 1,
            samples=result.samples[:-1],
        )
    else:
        sample = result.samples[10]
        if failure == "nonfinite_qpos":
            sample = replace(sample, qpos=(math.nan, *sample.qpos[1:]))
        elif failure == "short_qvel":
            sample = replace(sample, qvel=sample.qvel[:-1])
        elif failure == "excessive_qvel":
            sample = replace(sample, qvel=(1_000.1, *sample.qvel[1:]))
        elif failure == "excessive_position":
            positions = dict(sample.joint_positions)
            positions[formal.result.joint_names[0]] = math.tau + 0.01
            sample = replace(sample, joint_positions=positions)
        elif failure == "excessive_target":
            targets = dict(sample.position_targets)
            targets[formal.result.joint_names[0]] = math.tau + 0.01
            sample = replace(sample, position_targets=targets)
        else:
            poses = dict(sample.frame_poses)
            poses[formal.result.frame_names[0]] = (
                10.01,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            sample = replace(sample, frame_poses=poses)
        samples = list(result.samples)
        samples[10] = sample
        result = replace(result, samples=tuple(samples))
    traces[0] = replace(
        traces[0], formal_mujoco=replace(formal, result=result)
    )

    with pytest.raises(sensitivity.SensitivityValidationError, match=message):
        sensitivity.compute_primary_metrics(traces, **_metric_kwargs())


@pytest.mark.parametrize(
    ("values", "status"),
    [
        ([0.25] * 8, sensitivity.SensitivityStatus.SUPPORTED),
        ([0.10] * 8, sensitivity.SensitivityStatus.NOT_SUPPORTED),
        ([0.25] * 7 + [0.10], sensitivity.SensitivityStatus.INCONCLUSIVE),
        ([0.11] * 8, sensitivity.SensitivityStatus.INCONCLUSIVE),
        ([0.25] * 7 + [math.nan], sensitivity.SensitivityStatus.INVALID),
    ],
)
def test_intersection_decision(values: list[float], status: object) -> None:
    assert (
        sensitivity.decide_freeze_b(
            _decision_cells(values), validity_checks=_validity_checks()
        ).status
        is status
    )


def test_decision_rejects_raw_S_values_and_noncanonical_cells() -> None:
    raw = sensitivity.decide_freeze_b(
        [0.3] * 8,  # type: ignore[arg-type]
        validity_checks=_validity_checks(),
    )
    assert raw.status is sensitivity.SensitivityStatus.INVALID
    assert raw.scientific_label is None
    assert raw.cells == ()
    assert raw.s_values == ()
    assert raw.validity_failures == (
        "expected_exactly_four_primary_metric_cells",
    )

    cells = _decision_cells([0.3] * 8)
    reordered = sensitivity.decide_freeze_b(
        (cells[1], cells[0], cells[2], cells[3]),
        validity_checks=_validity_checks(),
    )
    assert reordered.status is sensitivity.SensitivityStatus.INVALID
    assert reordered.scientific_label is None
    assert reordered.cells == ()
    assert reordered.s_values == ()
    assert reordered.validity_failures == (
        "primary_metric_cells_are_not_in_canonical_order",
    )


def test_any_external_validity_failure_returns_invalid() -> None:
    checks = _validity_checks()
    checks["dt_halving"] = False
    decision = sensitivity.decide_freeze_b(
        _decision_cells([0.5] * 8),
        validity_checks=checks,
    )
    assert decision.status is sensitivity.SensitivityStatus.INVALID
    assert decision.scientific_label is None
    assert decision.validity_failures == (
        "validity_check_not_true:dt_halving",
    )


@pytest.mark.parametrize("mode", ["missing_all", "missing_one", "unknown", "non_boolean"])
def test_validity_inventory_must_be_complete_exact_and_all_true(mode: str) -> None:
    checks: object
    if mode == "missing_all":
        checks = None
    else:
        selected = _validity_checks()
        if mode == "missing_one":
            del selected["fresh_repeatability"]
        elif mode == "unknown":
            selected["post_hoc_override"] = True
        else:
            selected["fresh_repeatability"] = 1  # type: ignore[assignment]
        checks = selected

    decision = sensitivity.decide_freeze_b(
        _decision_cells([0.5] * 8),
        validity_checks=checks,  # type: ignore[arg-type]
    )

    assert decision.status is sensitivity.SensitivityStatus.INVALID
    assert decision.execution_valid is False
    assert decision.scientific_label is None
    assert decision.cells == ()
    serialized = decision.to_dict()
    assert serialized["validation_status"] == "INVALID"
    assert serialized["scientific_label"] is None
    assert "status" not in serialized
    assert decision.validity_failures


def test_evaluate_does_not_compute_or_label_when_validity_inventory_is_missing() -> None:
    decision = sensitivity.evaluate_freeze_b(
        (),
        **_metric_kwargs(),
        validity_checks={"dt_halving": True},
    )

    assert decision.status is sensitivity.SensitivityStatus.INVALID
    assert decision.scientific_label is None
    assert any(
        failure.startswith("missing_validity_check:")
        for failure in decision.validity_failures
    )


def test_raw_trace_validity_failure_returns_invalid() -> None:
    traces = list(_trace_sets())
    formal_ov = traces[0].formal_ovphysx
    bad_provenance = dict(formal_ov.result.provenance)
    bad_provenance["contact_check_performed"] = True
    bad_result = replace(
        formal_ov.result,
        provenance=bad_provenance,
    )
    traces[0] = replace(
        traces[0], formal_ovphysx=replace(formal_ov, result=bad_result)
    )

    decision = sensitivity.evaluate_freeze_b(
        traces,
        **_metric_kwargs(),
        validity_checks=_validity_checks(),
    )

    assert decision.status is sensitivity.SensitivityStatus.INVALID
    assert "contact_check_performed=false" in decision.validity_failures[0]
