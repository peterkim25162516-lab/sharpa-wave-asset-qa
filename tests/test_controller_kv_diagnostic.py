from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Mapping

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.bundle import write_bundle_manifest
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    bundle_payload_paths,
    create_staging_root,
    promote_staging_root,
    validated_results_output_path,
    verify_copied_bundle_subset,
    verify_exact_bundle,
)
from wave_asset_qa.parity.scenarios import (
    canonical_initial_positions,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "gate0.json"
PREREGISTRATION_PATH = (
    ROOT / "configs" / "parity" / "controller_kv_diagnostic.json"
)


def _load_diagnostic():
    path = ROOT / "scripts" / "run_mujoco_controller_kv_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "test_run_mujoco_controller_kv_diagnostic",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(simulator: Simulator):
    manifest = load_manifest(MANIFEST_PATH)
    case_id = (
        "mujoco.left.small_step.base.r01"
        if simulator is Simulator.MUJOCO
        else "ovphysx.left.small_step.base.r01"
    )
    case = next(
        item for item in expand_scenario_cases(manifest) if item.case_id == case_id
    )
    return manifest, case


def _canonical_run(simulator: Simulator) -> tuple[CollectedRun, object]:
    manifest, case = _case(simulator)
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    initial = canonical_initial_positions(scenario, hand.joint_names)
    backend_names = (
        list(hand.joint_names)
        if simulator is Simulator.MUJOCO
        else list(reversed(hand.joint_names))
    )
    backend_index = {name: index for index, name in enumerate(backend_names)}
    mapping = [
        {
            "canonical_id": name,
            "backend": simulator.value,
            "scope": case.hand.value,
            "backend_name": name,
            "index": backend_index[name],
            "sign": 1.0,
            "offset": 0.0,
            "unit": "rad",
        }
        for name in hand.joint_names
    ]
    frame_poses = {
        name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        for name in hand.distal_frame_names
    }
    samples = tuple(
        TraceSample(
            step=step,
            time_s=step * case.dt_s,
            qpos=tuple(initial[name] for name in backend_names),
            qvel=(0.0,) * len(hand.joint_names),
            joint_positions=initial,
            frame_poses=frame_poses,
            position_targets=canonical_position_targets(
                scenario,
                hand.joint_names,
                step_index=step,
                dt_s=case.dt_s,
            ),
            contact_count=0 if simulator is Simulator.MUJOCO else None,
        )
        for step in range(scenario.steps + 1)
    )
    provenance: dict[str, object] = {
        "backend_joint_names": backend_names,
        "joint_mapping": mapping,
    }
    if simulator is Simulator.OVPHYSX:
        provenance.update(
            {
                "contact_check_performed": False,
                "contact_observation_capability": "not_evaluated",
            }
        )
    result = AdapterRunResult(
        backend=simulator.value,
        scenario_id=case.scenario_id,
        status="completed",
        message="fixture",
        dt=case.dt_s,
        requested_steps=scenario.steps,
        completed_steps=scenario.steps,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=samples,
        provenance=provenance,
    )
    return CollectedRun(case=case, result=result), manifest


def _metric_result(
    *,
    joint_peak: float = 0.0,
    frame_peak: float = 0.0,
) -> AdapterRunResult:
    manifest = load_manifest(MANIFEST_PATH)
    hand = manifest.hands[0]
    samples: list[TraceSample] = []
    for step in range(55, 86):
        joints = {name: 0.0 for name in hand.joint_names}
        frames = {
            name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            for name in hand.distal_frame_names
        }
        if step == 65:
            joints["left_pinky_CMC"] = joint_peak
        if step == 66:
            frames["left_thumb_DP"] = (
                frame_peak,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
        samples.append(
            TraceSample(
                step=step,
                time_s=step * 0.002,
                qpos=tuple(joints.values()),
                qvel=(0.0,) * len(hand.joint_names),
                joint_positions=joints,
                frame_poses=frames,
            )
        )
    return AdapterRunResult(
        backend="fixture",
        scenario_id="small_step",
        status="completed",
        message="fixture",
        dt=0.002,
        requested_steps=30,
        completed_steps=30,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=tuple(samples),
    )


def _decision_row(joint: float, frame: float) -> dict[str, object]:
    return {
        "primary": {
            "left_pinky_CMC_window": {"peak_abs_rad": joint},
            "left_thumb_DP_window": {"peak_position_distance_m": frame},
        }
    }


def test_preregistration_locks_targets_metrics_and_decision(tmp_path: Path) -> None:
    diagnostic = _load_diagnostic()
    manifest = load_manifest(MANIFEST_PATH)

    preregistration, targets = diagnostic._load_preregistration(
        PREREGISTRATION_PATH,
        manifest,
    )

    assert len(targets) == 22
    assert diagnostic._compact_sha256(list(targets.values())) == (
        diagnostic.AUTHORITATIVE_OV_CANONICAL_KD_SHA256
    )
    tampered = json.loads(json.dumps(preregistration))
    tampered["primary_metrics"][0]["window_end_s"] = 0.162
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        diagnostic.ControllerKvDiagnosticError,
        match="primary metrics changed",
    ):
        diagnostic._load_preregistration(tampered_path, manifest)
    tampered = json.loads(json.dumps(preregistration))
    tampered["unregistered_claim"] = "not allowed"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        diagnostic.ControllerKvDiagnosticError,
        match="top-level schema changed",
    ):
        diagnostic._load_preregistration(tampered_path, manifest)


def test_trace_validation_maps_ov_backend_qpos_and_unobserved_contacts() -> None:
    diagnostic = _load_diagnostic()
    run, manifest = _canonical_run(Simulator.OVPHYSX)

    diagnostic._validate_trace_samples(run.result, run.case, manifest)

    sample = run.result.samples[51]
    changed_qpos = list(sample.qpos)
    changed_qpos[0] += 0.001
    changed_samples = list(run.result.samples)
    changed_samples[51] = replace(sample, qpos=tuple(changed_qpos))
    tampered = replace(run.result, samples=tuple(changed_samples))
    with pytest.raises(
        diagnostic.ControllerKvDiagnosticError,
        match="qpos does not reproduce",
    ):
        diagnostic._validate_trace_samples(tampered, run.case, manifest)


def test_trace_validation_rejects_backend_specific_contact_mismatch() -> None:
    diagnostic = _load_diagnostic()
    ov, manifest = _canonical_run(Simulator.OVPHYSX)
    samples = list(ov.result.samples)
    samples[0] = replace(samples[0], contact_count=0)

    with pytest.raises(
        diagnostic.ControllerKvDiagnosticError,
        match="invalid sampled state structure",
    ):
        diagnostic._validate_trace_samples(
            replace(ov.result, samples=tuple(samples)),
            ov.case,
            manifest,
        )


def test_frozen_ov_kd_mapping_is_canonical_and_content_addressed() -> None:
    diagnostic = _load_diagnostic()
    run, manifest = _canonical_run(Simulator.OVPHYSX)
    hand = manifest.hand(run.case.hand)
    _, targets = diagnostic._load_preregistration(PREREGISTRATION_PATH, manifest)
    provenance = dict(run.result.provenance)
    backend_names = provenance["backend_joint_names"]
    assert isinstance(backend_names, list)
    provenance.update(
        {
            "actuation_contract_version": 2,
            "actuator_model": "IdealPDActuator",
            "control_path": "explicit_pd_effort",
            "controller_parameter_source": "ideal_pd_actuator_tensor",
            "backend_dof_drive_zero_verified": True,
            "zero_velocity_target_verified": True,
            "zero_feedforward_effort_target_verified": True,
            "source_revision": diagnostic.AUTHORITATIVE_FORMAL_SOURCE_REVISION,
            "asset_commit": diagnostic.AUTHORITATIVE_ASSET_COMMIT,
            "asset_git_tree": diagnostic.AUTHORITATIVE_ASSET_GIT_TREE,
            "manifest_sha256": diagnostic.AUTHORITATIVE_MANIFEST_SHA256,
            "controller_dof_damping": [targets[name] for name in backend_names],
            "backend_dof_stiffness": [0.0] * len(backend_names),
            "backend_dof_damping": [0.0] * len(backend_names),
        }
    )
    frozen = replace(run, result=replace(run.result, provenance=provenance))

    observed, audit = diagnostic._extract_ov_controller_kd(frozen, hand)

    assert observed == targets
    assert audit["canonical_order_values_sha256"] == (
        diagnostic.AUTHORITATIVE_OV_CANONICAL_KD_SHA256
    )


def test_registered_windows_and_decision_rule_are_exact() -> None:
    diagnostic = _load_diagnostic()
    reference = _metric_result()
    candidate = _metric_result(joint_peak=0.009, frame_peak=0.0008)

    joint = diagnostic._window_joint_peak(
        reference,
        candidate,
        joint_name="left_pinky_CMC",
        start_step=55,
        end_step=80,
    )
    frame = diagnostic._window_frame_peak(
        reference,
        candidate,
        frame_name="left_thumb_DP",
        start_step=55,
        end_step=85,
    )

    assert joint["peak_step"] == 65
    assert joint["peak_time_s"] == pytest.approx(0.13)
    assert joint["peak_abs_rad"] == pytest.approx(0.009)
    assert frame["peak_step"] == 66
    assert frame["peak_time_s"] == pytest.approx(0.132)
    assert frame["peak_position_distance_m"] == pytest.approx(0.0008)

    assert diagnostic._evaluate_decision(
        _decision_row(1.0, 1.0),
        _decision_row(0.75, 0.70),
        baseline_drift_ok=True,
    )["scientific_status"] == (
        "supports_registered_local_gap_reduction_under_total_viscous_retuning"
    )
    assert diagnostic._evaluate_decision(
        _decision_row(1.0, 1.0),
        _decision_row(0.90, 0.95),
        baseline_drift_ok=True,
    )["scientific_status"] == (
        "rejects_total_viscous_alignment_as_common_major_local_explanation"
    )
    assert diagnostic._evaluate_decision(
        _decision_row(1.0, 1.0),
        _decision_row(0.60, 0.95),
        baseline_drift_ok=True,
    )["scientific_status"] == "mixed_or_inconclusive"
    with pytest.raises(
        diagnostic.ControllerKvDiagnosticError,
        match="exact-baseline rule",
    ):
        diagnostic._evaluate_decision(
            _decision_row(1.0, 1.0),
            _decision_row(0.1, 0.1),
            baseline_drift_ok=False,
        )


def test_diagnostic_bundle_helpers_confine_promote_and_verify(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    final = validated_results_output_path(tmp_path, results / "diagnostic")
    with pytest.raises(DiagnosticEvidenceError, match="direct child"):
        validated_results_output_path(tmp_path, tmp_path / "outside")

    staging = create_staging_root(final)
    (staging / "payload.txt").write_text("evidence", encoding="utf-8")
    write_bundle_manifest(staging, bundle_payload_paths(staging))
    assert verify_exact_bundle(staging)["exact_inventory"] is True

    promoted = promote_staging_root(staging, final)
    assert promoted == final
    assert verify_exact_bundle(promoted)["exact_inventory"] is True
    (promoted / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(DiagnosticEvidenceError, match="inventory is not exact"):
        verify_exact_bundle(promoted)


def test_copied_parent_bundle_subset_is_byte_exact(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    payloads = {"one/a.json": b"a", "two/b.json": b"b"}
    for relative, content in payloads.items():
        path = parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_bundle_manifest(parent, payloads)
    verification = verify_exact_bundle(parent)

    copied = tmp_path / "copied"
    copied.mkdir()
    manifest_copy = copied / "parent-bundle.json"
    shutil.copyfile(parent / "bundle.json", manifest_copy)
    selected: dict[str, Path] = {}
    for index, relative in enumerate(payloads):
        destination = copied / f"payload-{index}.json"
        shutil.copyfile(parent / relative, destination)
        selected[relative] = destination
    verify_copied_bundle_subset(
        manifest_copy,
        expected_root_sha256=str(verification["root_sha256"]),
        copied_payloads=selected,
    )
    forged = json.loads(manifest_copy.read_text(encoding="utf-8"))
    forged["files"][0]["sha256"] = "0" * 64
    forged_manifest = copied / "forged-parent-bundle.json"
    forged_manifest.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(DiagnosticEvidenceError, match="wrong root hash"):
        verify_copied_bundle_subset(
            forged_manifest,
            expected_root_sha256=str(verification["root_sha256"]),
            copied_payloads=selected,
        )
    next(iter(selected.values())).write_text("tampered", encoding="utf-8")
    with pytest.raises(DiagnosticEvidenceError, match="differs from its parent"):
        verify_copied_bundle_subset(
            manifest_copy,
            expected_root_sha256=str(verification["root_sha256"]),
            copied_payloads=selected,
        )


def test_numeric_parser_rejects_huge_integer_without_overflow() -> None:
    diagnostic = _load_diagnostic()
    with pytest.raises(diagnostic.ControllerKvDiagnosticError, match="finite number"):
        diagnostic._number(10**400, "huge")
