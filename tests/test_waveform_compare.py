from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import wave_asset_qa.parity.compare as compare_module

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.compare import (
    CollectedRun,
    ComparisonThresholds,
    _RunAssessment,
    _assess_run,
    _comparison_record,
    _enforce_run_consistency,
    trace_delta,
)
from wave_asset_qa.parity.contracts import (
    ComparisonStatus,
    ExecutionRecord,
    ExecutionStatus,
    ParityManifest,
    Simulator,
)
from wave_asset_qa.parity.scenarios import (
    TimestepVariant,
    expand_scenario_cases,
    load_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "waveform_t1.json"


def _session_assessments(
    *, mixed_mujoco_session: bool = False
) -> tuple[ParityManifest, dict[str, _RunAssessment]]:
    manifest = load_manifest(MANIFEST_PATH)
    assessments: dict[str, _RunAssessment] = {}
    for case in expand_scenario_cases(manifest):
        hand = manifest.hand(case.hand)
        session_id = (
            "waveform-local-session"
            if case.simulator is Simulator.MUJOCO
            else "waveform-remote-session"
        )
        if mixed_mujoco_session and case.simulator is Simulator.MUJOCO:
            if (
                case.repeat_index == 2
                and case.timestep_variant is TimestepVariant.HALVED
            ):
                session_id = "waveform-second-local-session"
        result = replace(
            _result(case, hand, offset=0.0),
            provenance={
                "session_id": session_id,
                "source_revision": "1" * 40,
            },
        )
        assessments[case.case_id] = _RunAssessment(
            collected=CollectedRun(case=case, result=result),
            expected_steps=result.requested_steps,
            finite=True,
            errors=(),
        )
    return manifest, assessments


def test_waveform_sessions_are_consistent_per_backend() -> None:
    manifest, assessments = _session_assessments()

    checked = _enforce_run_consistency(assessments, manifest)

    assert all(not assessment.errors for assessment in checked.values())


def test_waveform_compare_accepts_distinct_backend_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, assessments = _session_assessments()
    monkeypatch.setattr(
        compare_module,
        "_assess_run",
        lambda collected, unused_manifest: assessments[collected.case.case_id],
    )

    comparison = compare_module.compare_gate0_runs(
        manifest,
        [assessment.collected for assessment in assessments.values()],
        thresholds=ComparisonThresholds(minimum_completion_fraction=0.99),
    )

    assert comparison.execution_status is ExecutionStatus.COMPLETED
    assert comparison.completed_run_count == 32
    assert comparison.completion_fraction == 1.0
    assert all(result.comparison.metrics for result in comparison.results)


def test_waveform_rejects_mixed_sessions_within_one_backend_only() -> None:
    manifest, assessments = _session_assessments(mixed_mujoco_session=True)

    checked = _enforce_run_consistency(assessments, manifest)

    mujoco = [
        assessment
        for assessment in checked.values()
        if assessment.collected.case.simulator is Simulator.MUJOCO
    ]
    ovphysx = [
        assessment
        for assessment in checked.values()
        if assessment.collected.case.simulator is Simulator.OVPHYSX
    ]
    assert all(
        any("multiple session_id" in error for error in assessment.errors)
        for assessment in mujoco
    )
    assert all(not assessment.errors for assessment in ovphysx)


def _result(case, hand, *, offset: float) -> AdapterRunResult:
    sample_steps = (0, 1) if case.timestep_variant is TimestepVariant.BASE else (0, 1, 2)
    samples = []
    for step in sample_steps:
        positions = {name: (0.0 if step == 0 else offset) for name in hand.joint_names}
        samples.append(
            TraceSample(
                step=step,
                time_s=step * case.dt_s,
                qpos=tuple(positions.values()),
                qvel=(0.0,) * len(hand.joint_names),
                joint_positions=positions,
                frame_poses={
                    name: (
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    for name in hand.distal_frame_names
                },
                position_targets={name: 0.0 for name in hand.joint_names},
                contact_count=(0 if case.simulator is Simulator.MUJOCO else None),
            )
        )
    return AdapterRunResult(
        backend=case.simulator.value,
        scenario_id=case.scenario_id,
        status="completed",
        message="synthetic",
        dt=case.dt_s,
        requested_steps=sample_steps[-1],
        completed_steps=sample_steps[-1],
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=tuple(samples),
    )


def _record_for_offsets(offsets: dict[tuple[str, str, int], float]):
    manifest = load_manifest(MANIFEST_PATH)
    hand = manifest.hands[0]
    scenario_id = "offset_sine"
    cases = tuple(
        case
        for case in expand_scenario_cases(manifest)
        if case.hand is hand.side and case.scenario_id == scenario_id
    )
    assessments = {}
    for case in cases:
        key = (
            case.simulator.value,
            case.timestep_variant.value,
            case.repeat_index,
        )
        result = _result(case, hand, offset=offsets.get(key, 0.0))
        collected = CollectedRun(case=case, result=result)
        assessments[case.case_id] = _RunAssessment(
            collected=collected,
            expected_steps=result.requested_steps,
            finite=True,
            errors=(),
        )
    executions = (
        ExecutionRecord(
            simulator=Simulator.MUJOCO,
            execution_status=ExecutionStatus.COMPLETED,
            completed_repeats=2,
            requested_repeats=2,
            finite=True,
        ),
        ExecutionRecord(
            simulator=Simulator.OVPHYSX,
            execution_status=ExecutionStatus.COMPLETED,
            completed_repeats=2,
            requested_repeats=2,
            finite=True,
        ),
    )
    thresholds = ComparisonThresholds(minimum_completion_fraction=0.99)
    return _comparison_record(
        manifest,
        hand,
        scenario_id,
        cases,
        assessments,
        executions,
        thresholds,
    )


def test_waveform_repeatability_failure_makes_crosssim_inconclusive() -> None:
    offsets = {
        (simulator, variant, 2): 0.01
        for simulator in ("mujoco", "ovphysx")
        for variant in ("base", "halved")
    }

    record = _record_for_offsets(offsets)

    assert record.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert record.metrics["repeat_joint_max_abs_rad"] == pytest.approx(0.01)
    assert "Cross-simulator status was not assigned" in (record.message or "")


def test_waveform_dt_failure_makes_crosssim_inconclusive() -> None:
    offsets = {
        (simulator, "halved", repeat): 0.02
        for simulator in ("mujoco", "ovphysx")
        for repeat in (1, 2)
    }

    record = _record_for_offsets(offsets)

    assert record.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert record.metrics["dt_halving_joint_max_abs_rad"] == pytest.approx(0.02)


def test_waveform_crosssim_exceedance_is_valid_divergence_after_prerequisites() -> None:
    offsets = {
        ("ovphysx", variant, repeat): 0.03
        for variant in ("base", "halved")
        for repeat in (1, 2)
    }

    record = _record_for_offsets(offsets)

    assert record.comparison_status is ComparisonStatus.DIVERGENT
    assert record.metrics["crosssim_joint_max_abs_rad"] == pytest.approx(0.03)
    assert record.metrics["repeat_joint_max_abs_rad"] == 0.0
    assert record.metrics["dt_halving_joint_max_abs_rad"] == 0.0


def test_waveform_crosssim_threshold_covers_halved_dt_cases() -> None:
    offsets = {
        ("ovphysx", "base", repeat): 0.0195
        for repeat in (1, 2)
    }
    offsets.update(
        {
            ("ovphysx", "halved", repeat): 0.0205
            for repeat in (1, 2)
        }
    )

    record = _record_for_offsets(offsets)

    assert record.comparison_status is ComparisonStatus.DIVERGENT
    assert record.metrics["crosssim_joint_max_abs_rad"] == pytest.approx(0.0205)
    assert record.metrics["dt_halving_joint_max_abs_rad"] == pytest.approx(0.001)


def test_waveform_v2_requires_null_ovphysx_contact_observation() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    hand = manifest.hands[0]
    case = next(
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.OVPHYSX
        and case.hand is hand.side
        and case.timestep_variant is TimestepVariant.BASE
        and case.repeat_index == 1
    )
    result = _result(case, hand, offset=0.0)
    result = replace(
        result,
        samples=tuple(
            replace(sample, contact_count=0) for sample in result.samples
        ),
    )

    assessment = _assess_run(CollectedRun(case=case, result=result), manifest)

    assert any(
        "OVPhysX contact_count must be null" in error
        for error in assessment.errors
    )


def test_trace_delta_uses_integer_step_ratio_not_timestamp_dictionary() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    hand = manifest.hands[0]
    cases = [
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.MUJOCO
        and case.hand is hand.side
        and case.scenario_id == "offset_sine"
        and case.repeat_index == 1
    ]
    base_case = next(
        case for case in cases if case.timestep_variant is TimestepVariant.BASE
    )
    halved_case = next(
        case for case in cases if case.timestep_variant is TimestepVariant.HALVED
    )
    base = _result(base_case, hand, offset=0.0)
    halved = _result(halved_case, hand, offset=0.0)

    assert trace_delta(base, halved, hand) == (0.0, 0.0, 0.0)

    malformed = replace(
        halved,
        samples=tuple(
            replace(sample, step=(3 if sample.step == 2 else sample.step))
            for sample in halved.samples
        ),
    )
    with pytest.raises(ValueError, match="canonical step 2"):
        trace_delta(base, malformed, hand)
