from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity import compare as compare_module
from wave_asset_qa.parity.compare import (
    CollectedRun,
    ComparisonThresholds,
    compare_gate0_runs,
)
from wave_asset_qa.parity.contracts import (
    ComparisonStatus,
    ExecutionStatus,
    ParityManifest,
    Simulator,
)
from wave_asset_qa.parity.mapping import FrameMapping, JointMapping
from wave_asset_qa.parity.scenarios import (
    TimestepVariant,
    canonical_position_targets,
    expand_scenario_cases,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_quaternion_angle_is_stable_at_zero_and_sign_invariant() -> None:
    quaternion = (0.9238795325112867, 0.0, 0.3826834323650898, 0.0)
    pose = (0.0, 0.0, 0.0, *quaternion)
    negated_pose = (0.0, 0.0, 0.0, *(-value for value in quaternion))
    quarter_turn = (0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    assert compare_module._quaternion_angle(pose, pose) == 0.0
    assert compare_module._quaternion_angle(pose, negated_pose) == 0.0
    assert compare_module._quaternion_angle(identity, quarter_turn) == pytest.approx(
        math.pi / 2.0
    )


def _short_manifest() -> ParityManifest:
    payload = json.loads(
        (ROOT / "configs" / "parity" / "gate0.json").read_text(encoding="utf-8")
    )
    for scenario in payload["scenarios"]:
        scenario["duration_s"] = 0.02
        scenario["dt_s"] = 0.01
        if scenario["kind"] == "small_step":
            scenario["step_start_s"] = 0.01
    return ParityManifest.from_dict(payload)


def _sample_value(scenario_id: str, time_s: float) -> float:
    if scenario_id == "small_step" and time_s >= 0.01:
        return 0.05 * (1.0 - pow(2.718281828, -(time_s - 0.01) * 10.0))
    if scenario_id == "gravity_settling":
        return -0.001 * time_s
    return 0.0


def _provenance(
    manifest: ParityManifest,
    case: object,
    hand: object,
) -> dict[str, object]:
    backend = getattr(getattr(case, "simulator"), "value")
    scope = getattr(getattr(case, "hand"), "value")
    joint_names = tuple(getattr(hand, "joint_names"))
    frame_names = tuple(getattr(hand, "distal_frame_names"))
    scenario = manifest.scenario(getattr(case, "scenario_id"))
    steps = round(scenario.duration_s / getattr(case, "dt_s"))
    is_small_step = scenario.scenario_id == "small_step"
    return {
        "synthetic": True,
        "manifest_sha256": manifest_sha256(manifest),
        "session_id": "fixture-session",
        "source_revision": "1" * 40,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "simulation_configuration": {
            "verified": True,
            "requested_dt_s": getattr(case, "dt_s"),
            "cfg_dt_s": getattr(case, "dt_s"),
            "backend_dt_s": getattr(case, "dt_s"),
            "requested_gravity_m_s2": list(scenario.gravity_m_s2),
            "cfg_gravity_m_s2": list(scenario.gravity_m_s2),
            "physics_scene_gravity_m_s2": list(scenario.gravity_m_s2),
            "physics_prim_path": "/physicsScene",
        },
        "actuation_contract_version": 2,
        "actuator_model": "IdealPDActuator",
        "control_path": "explicit_pd_effort",
        "controller_dof_stiffness": [1.0] * len(joint_names),
        "controller_dof_damping": [0.1] * len(joint_names),
        "controller_dof_effort_limit": [2.0] * len(joint_names),
        "controller_dof_effort_limit_sim": [3.0] * len(joint_names),
        "controller_parameter_source": "ideal_pd_actuator_tensor",
        "backend_dof_stiffness": [0.0] * len(joint_names),
        "backend_dof_damping": [0.0] * len(joint_names),
        "backend_dof_drive_readback_source": "root_view_cpu_numpy_binding",
        "position_target_readback_verified": True,
        "position_target_readback_source": "articulation_data_joint_pos_target_torch",
        "zero_velocity_target_verified": True,
        "zero_feedforward_effort_target_verified": True,
        "position_target_readback_count": steps + 2,
        "position_target_readback_max_abs_error_rad": 0.0,
        "position_target_readback_values_rad": [
            scenario.target_position_rad if is_small_step else 0.0
        ]
        * len(joint_names),
        "position_target_nonzero_readback_observed": is_small_step,
        "computed_effort_peak_abs_nm": [0.5] * len(joint_names),
        "applied_effort_peak_abs_nm": [0.5] * len(joint_names),
        "effort_observation_count": steps + 1,
        "effort_formula_max_abs_error_nm": 0.0,
        "effort_clip_max_abs_error_nm": 0.0,
        "effort_clip_count": 0,
        "effort_command_source": (
            "articulation_data_computed_and_applied_torque_torch"
        ),
        "mapping_schema_version": 1,
        "backend_joint_names": list(joint_names),
        "backend_frame_names": list(frame_names),
        "joint_mapping": [
            JointMapping(
                canonical_id=name,
                backend=backend,
                scope=scope,
                backend_name=name,
                index=index,
                sign=1.0,
                offset=0.0,
                unit="rad",
            ).to_dict()
            for index, name in enumerate(joint_names)
        ],
        "frame_mapping": [
            FrameMapping(
                canonical_id=name,
                backend=backend,
                scope=scope,
                backend_name=name,
                index=index,
                sign=1.0,
                offset=0.0,
                unit="xyz_m_qwxyz",
            ).to_dict()
            for index, name in enumerate(frame_names)
        ],
    }


def _synthetic_runs(manifest: ParityManifest) -> list[CollectedRun]:
    collected: list[CollectedRun] = []
    for case in expand_scenario_cases(manifest):
        hand = manifest.hand(case.hand)
        scenario = manifest.scenario(case.scenario_id)
        steps = round(scenario.duration_s / case.dt_s)
        samples: list[TraceSample] = []
        for step in range(steps + 1):
            time_s = step * case.dt_s
            value = _sample_value(case.scenario_id, time_s)
            joint_positions = {name: value for name in hand.joint_names}
            frame_poses = {
                name: (index * 0.001 + value * 0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                for index, name in enumerate(hand.distal_frame_names)
            }
            samples.append(
                TraceSample(
                    step=step,
                    time_s=time_s,
                    qpos=tuple(joint_positions.values()),
                    qvel=(0.0,) * len(hand.joint_names),
                    joint_positions=joint_positions,
                    frame_poses=frame_poses,
                    position_targets=canonical_position_targets(
                        scenario,
                        hand.joint_names,
                        step_index=step,
                        dt_s=case.dt_s,
                    ),
                    contact_count=0,
                )
            )
        collected.append(
            CollectedRun(
                case=case,
                result=AdapterRunResult(
                    backend=case.simulator.value,
                    scenario_id=case.scenario_id,
                    status="completed",
                    message="synthetic completed run",
                    dt=case.dt_s,
                    requested_steps=steps,
                    completed_steps=steps,
                    joint_names=hand.joint_names,
                    frame_names=hand.distal_frame_names,
                    samples=tuple(samples),
                    provenance=_provenance(manifest, case, hand),
                ),
                bundle_root_sha256="a" * 64,
            )
        )
    return collected


def _offset_joint(run: CollectedRun, offset: float) -> CollectedRun:
    samples = tuple(
        replace(
            sample,
            qpos=tuple(
                value + (offset if sample.step else 0.0) for value in sample.qpos
            ),
            joint_positions={
                name: value + (offset if sample.step else 0.0)
                for name, value in sample.joint_positions.items()
            },
        )
        for sample in run.result.samples
    )
    return replace(run, result=replace(run.result, samples=samples))


def test_complete_finite_repeated_and_dt_halved_runs_are_within_tolerance() -> None:
    manifest = _short_manifest()
    comparison = compare_gate0_runs(manifest, _synthetic_runs(manifest))

    assert comparison.execution_status is ExecutionStatus.COMPLETED
    assert comparison.comparison_status is ComparisonStatus.WITHIN_TOLERANCE
    assert comparison.expected_run_count == comparison.completed_run_count == 32
    assert comparison.completion_fraction == 1.0
    assert comparison.step_completion_fraction == 1.0
    assert comparison.nonfinite_run_count == 0
    assert comparison.observed_joint_mapping_count == 44
    assert comparison.observed_distal_frame_mapping_count == 10
    assert comparison.mapping_validation_complete
    assert len(comparison.results) == 6
    assert all(
        result.comparison.comparison_status is ComparisonStatus.WITHIN_TOLERANCE
        for result in comparison.results
    )


def test_crosssim_threshold_exceedance_is_divergent_but_execution_completed() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    runs = [
        _offset_joint(run, 0.1)
        if run.case.simulator.value == "ovphysx"
        and run.case.hand.value == "left"
        and run.case.scenario_id == "zero_hold"
        else run
        for run in runs
    ]

    comparison = compare_gate0_runs(manifest, runs)
    left_zero = next(
        result
        for result in comparison.results
        if result.hand.value == "left" and result.scenario_id == "zero_hold"
    )

    assert comparison.execution_status is ExecutionStatus.COMPLETED
    assert comparison.comparison_status is ComparisonStatus.DIVERGENT
    assert left_zero.comparison.comparison_status is ComparisonStatus.DIVERGENT
    assert "observation, not an official bug" in left_zero.comparison.message


def test_nonfinite_trace_is_execution_error_and_comparison_inconclusive() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = runs[0]
    first = replace(target.result.samples[0], qpos=(float("nan"),) + target.result.samples[0].qpos[1:])
    runs[0] = replace(
        target,
        result=replace(target.result, samples=(first, *target.result.samples[1:])),
    )

    comparison = compare_gate0_runs(manifest, runs)
    affected = comparison.results[0]

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert comparison.nonfinite_run_count == 1
    assert affected.executions[0].execution_status is ExecutionStatus.ERROR
    assert affected.comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert "NaN or Inf" in affected.executions[0].message


def test_contact_observation_contract_is_backend_specific() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    mujoco = next(
        run for run in runs if run.case.simulator is Simulator.MUJOCO
    )
    ovphysx = next(
        run for run in runs if run.case.simulator is Simulator.OVPHYSX
    )

    missing_mujoco_contact = replace(
        mujoco,
        result=replace(
            mujoco.result,
            samples=(
                replace(mujoco.result.samples[0], contact_count=None),
                *mujoco.result.samples[1:],
            ),
        ),
    )
    ov_samples = tuple(
        replace(sample, contact_count=None) for sample in ovphysx.result.samples
    )
    valid_unknown_ov_contact = replace(
        ovphysx,
        result=replace(ovphysx.result, samples=ov_samples),
    )
    mutated = [
        missing_mujoco_contact
        if run is mujoco
        else valid_unknown_ov_contact
        if run is ovphysx
        else run
        for run in runs
    ]

    comparison = compare_gate0_runs(manifest, mutated)
    affected = next(
        result
        for result in comparison.results
        if result.hand is mujoco.case.hand
        and result.scenario_id == mujoco.case.scenario_id
    )

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert any(
        execution.simulator is Simulator.MUJOCO
        and "contact_count must be an explicit integer" in execution.message
        for execution in affected.executions
    )
    assert not any(
        execution.simulator is Simulator.OVPHYSX
        and "contact_count" in (execution.message or "")
        for result in comparison.results
        for execution in result.executions
    )


def test_missing_run_reduces_completion_and_is_not_called_divergent() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)[:-1]

    comparison = compare_gate0_runs(manifest, runs)

    assert comparison.received_run_count == 31
    assert comparison.completed_run_count == 31
    assert comparison.completion_fraction == pytest.approx(31 / 32)
    assert not comparison.completion_target_met
    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert any("completion target" in item.lower() for item in comparison.observations)


def test_repeatability_and_dt_halving_have_independent_thresholds() -> None:
    manifest = _short_manifest()
    thresholds = ComparisonThresholds(
        repeat_joint_max_abs_rad=0.001,
        dt_halving_joint_max_abs_rad=0.001,
    )

    repeat_runs = [
        _offset_joint(run, 0.01)
        if run.case.scenario_id == "zero_hold" and run.case.repeat_index == 2
        else run
        for run in _synthetic_runs(manifest)
    ]
    repeat_comparison = compare_gate0_runs(
        manifest, repeat_runs, thresholds=thresholds
    )
    repeat_result = next(
        result
        for result in repeat_comparison.results
        if result.hand.value == "left" and result.scenario_id == "zero_hold"
    )
    assert repeat_result.comparison.comparison_status is ComparisonStatus.DIVERGENT
    assert repeat_result.comparison.metrics["repeat_joint_max_abs_rad"] == pytest.approx(
        0.01
    )

    halved_runs = [
        _offset_joint(run, 0.01)
        if run.case.scenario_id == "small_step"
        and run.case.timestep_variant is TimestepVariant.HALVED
        else run
        for run in _synthetic_runs(manifest)
    ]
    halved_comparison = compare_gate0_runs(
        manifest, halved_runs, thresholds=thresholds
    )
    halved_result = next(
        result
        for result in halved_comparison.results
        if result.hand.value == "left" and result.scenario_id == "small_step"
    )
    assert halved_result.comparison.comparison_status is ComparisonStatus.DIVERGENT
    assert halved_result.comparison.metrics[
        "dt_halving_joint_max_abs_rad"
    ] == pytest.approx(0.01)


def test_duplicate_or_unknown_case_is_rejected() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)

    with pytest.raises(ValueError, match="duplicate"):
        compare_gate0_runs(manifest, [*runs, runs[0]])

    bad_case = replace(runs[0].case, scenario_id="unknown")
    with pytest.raises(ValueError, match="unexpected"):
        compare_gate0_runs(manifest, [replace(runs[0], case=bad_case)])


def test_small_step_with_valid_targets_but_no_state_response_is_inconclusive() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    replaced: list[CollectedRun] = []
    for run in runs:
        if run.case.scenario_id != "small_step":
            replaced.append(run)
            continue
        samples = tuple(
            replace(
                sample,
                qpos=(0.0,) * len(sample.qpos),
                joint_positions={name: 0.0 for name in sample.joint_positions},
            )
            for sample in run.result.samples
        )
        replaced.append(replace(run, result=replace(run.result, samples=samples)))

    comparison = compare_gate0_runs(manifest, replaced)
    small_step = next(
        result for result in comparison.results if result.scenario_id == "small_step"
    )

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert small_step.comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert "no observable canonical joint response" in small_step.executions[0].message


def test_wrong_target_schedule_and_raw_qpos_mapping_are_data_errors() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = next(run for run in runs if run.case.scenario_id == "small_step")
    samples = list(target.result.samples)
    onset = next(
        index
        for index, sample in enumerate(samples)
        if any(value != 0.0 for value in sample.position_targets.values())
    )
    samples[onset] = replace(
        samples[onset],
        position_targets={name: 0.0 for name in samples[onset].position_targets},
    )
    first_after_zero = samples[1]
    samples[1] = replace(
        first_after_zero,
        qpos=(first_after_zero.qpos[0] + 0.01, *first_after_zero.qpos[1:]),
    )
    mutated = [
        replace(target, result=replace(target.result, samples=tuple(samples)))
        if run is target
        else run
        for run in runs
    ]

    comparison = compare_gate0_runs(manifest, mutated)
    affected = next(
        result
        for result in comparison.results
        if result.hand is target.case.hand and result.scenario_id == "small_step"
    )

    assert affected.executions[0].execution_status is ExecutionStatus.ERROR
    assert "target trajectory mismatch" in affected.executions[0].message
    assert "raw qpos does not map" in affected.executions[0].message
    assert affected.comparison.comparison_status is ComparisonStatus.INCONCLUSIVE


def test_missing_mapping_or_mixed_session_cannot_report_complete_mapping() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    missing = runs[0]
    provenance = deepcopy(dict(missing.result.provenance))
    provenance["joint_mapping"] = provenance["joint_mapping"][:-1]
    runs[0] = replace(missing, result=replace(missing.result, provenance=provenance))
    mixed = runs[1]
    second_provenance = dict(mixed.result.provenance)
    second_provenance["session_id"] = "different-session"
    runs[1] = replace(
        mixed, result=replace(mixed.result, provenance=second_provenance)
    )

    comparison = compare_gate0_runs(manifest, runs)

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert not comparison.mapping_validation_complete
    assert any(
        "multiple session_id" in execution.message
        for result in comparison.results
        for execution in result.executions
        if execution.message
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("manifest_sha256", "0" * 64, "canonical manifest"),
        ("session_id", "", "non-empty string"),
        ("source_revision", "not-a-commit", "40-character code commit"),
        ("asset_commit", "0" * 40, "upstream asset commit"),
        ("asset_git_tree", "0" * 40, "pinned asset Git tree"),
        ("asset_tree_sha256", "0" * 64, "canonical LF asset tree"),
    ),
)
def test_each_run_must_bind_to_the_expected_provenance(
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = runs[0]
    provenance = dict(target.result.provenance)
    provenance[field] = value
    runs[0] = replace(
        target,
        result=replace(target.result, provenance=provenance),
    )

    comparison = compare_gate0_runs(manifest, runs)

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert not comparison.mapping_validation_complete
    assert any(
        message in execution.message
        for result in comparison.results
        for execution in result.executions
        if execution.message
    )


def test_compare_rejects_forged_ovphysx_final_target_readback() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = next(
        run
        for run in runs
        if run.case.simulator.value == "ovphysx"
        and run.case.scenario_id == "small_step"
    )
    provenance = dict(target.result.provenance)
    provenance["position_target_readback_values_rad"] = [0.0] * 22
    mutated = [
        replace(target, result=replace(target.result, provenance=provenance))
        if run is target
        else run
        for run in runs
    ]

    comparison = compare_gate0_runs(manifest, mutated)
    affected = next(
        result
        for result in comparison.results
        if result.hand is target.case.hand and result.scenario_id == "small_step"
    )

    execution = next(
        item for item in affected.executions if item.simulator.value == "ovphysx"
    )
    assert execution.execution_status is ExecutionStatus.ERROR
    assert "final target readback" in execution.message
    assert affected.comparison.comparison_status is ComparisonStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("actuation_contract_version", 1),
        ("actuation_contract_version", 2.0),
        ("actuator_model", "ImplicitActuator"),
        ("control_path", "implicit_physx_drive"),
        ("controller_dof_stiffness", [0.0] * 22),
        ("controller_dof_damping", [-0.1] * 22),
        ("controller_dof_effort_limit", [0.0] * 22),
        ("controller_dof_effort_limit_sim", [0.0] * 22),
        ("controller_parameter_source", "usd_drive_schema"),
        ("backend_dof_stiffness", [1e-7] * 22),
        ("backend_dof_damping", [-1e-7] * 22),
        ("backend_dof_drive_readback_source", "cached_tensor"),
        ("position_target_readback_source", "root_view_cuda_warp_binding"),
        ("zero_velocity_target_verified", False),
        ("zero_feedforward_effort_target_verified", False),
        ("computed_effort_peak_abs_nm", [-0.1] * 22),
        ("applied_effort_peak_abs_nm", [0.1] * 21),
        ("effort_observation_count", 0),
        ("effort_formula_max_abs_error_nm", 1.1e-5),
        ("effort_clip_max_abs_error_nm", 1.1e-6),
        ("effort_clip_count", True),
        ("effort_command_source", "actuator_cache"),
    ),
)
def test_compare_rejects_invalid_ideal_pd_evidence(
    field: str, invalid_value: object
) -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = next(run for run in runs if run.case.simulator.value == "ovphysx")
    provenance = dict(target.result.provenance)
    provenance[field] = invalid_value
    mutated = [
        replace(target, result=replace(target.result, provenance=provenance))
        if run is target
        else run
        for run in runs
    ]

    comparison = compare_gate0_runs(manifest, mutated)

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert not comparison.mapping_validation_complete


def test_runs_from_different_code_revisions_cannot_be_mixed() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    target = runs[0]
    provenance = dict(target.result.provenance)
    provenance["source_revision"] = "2" * 40
    runs[0] = replace(
        target,
        result=replace(target.result, provenance=provenance),
    )

    comparison = compare_gate0_runs(manifest, runs)

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert not comparison.mapping_validation_complete
    assert any(
        "multiple source_revision" in execution.message
        for result in comparison.results
        for execution in result.executions
        if execution.message
    )


def test_identical_finite_explosion_and_nonunit_quaternion_are_not_divergence() -> None:
    manifest = _short_manifest()
    runs = _synthetic_runs(manifest)
    mutated: list[CollectedRun] = []
    for run in runs:
        if run.case.scenario_id != "zero_hold":
            mutated.append(run)
            continue
        samples = []
        for sample in run.result.samples:
            if sample.step == 0:
                samples.append(sample)
                continue
            bad_frames = dict(sample.frame_poses)
            first_frame = next(iter(bad_frames))
            pose = bad_frames[first_frame]
            bad_frames[first_frame] = (*pose[:3], 2.0, 0.0, 0.0, 0.0)
            samples.append(
                replace(
                    sample,
                    qpos=(1e8,) * len(sample.qpos),
                    joint_positions={name: 1e8 for name in sample.joint_positions},
                    frame_poses=bad_frames,
                )
            )
        mutated.append(
            replace(run, result=replace(run.result, samples=tuple(samples)))
        )

    comparison = compare_gate0_runs(manifest, mutated)

    assert comparison.execution_status is ExecutionStatus.ERROR
    assert comparison.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert any(
        "integrity bound" in execution.message or "not unit length" in execution.message
        for result in comparison.results
        for execution in result.executions
        if execution.message
    )
