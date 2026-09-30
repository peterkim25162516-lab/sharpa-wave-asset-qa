from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Mapping

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.bundle import write_bundle_manifest
from wave_asset_qa.parity.compare import CollectedRun, trace_delta
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.scenarios import (
    canonical_initial_positions,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "gate0.json"


def _load_scan():
    path = ROOT / "scripts" / "run_mujoco_friction_scan.py"
    spec = importlib.util.spec_from_file_location("test_run_mujoco_friction_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(
    *,
    joint_delta: float = 0.0,
    frame_delta: float = 0.0,
    backend: str = "fixture",
    provenance: Mapping[str, object] | None = None,
) -> AdapterRunResult:
    hand = load_manifest(MANIFEST).hands[0]
    joint_positions = {name: 0.0 for name in hand.joint_names}
    joint_positions["left_thumb_CMC_FE"] = joint_delta
    frame_poses = {
        name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        for name in hand.distal_frame_names
    }
    frame_poses["left_thumb_DP"] = (
        frame_delta,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    sample = TraceSample(
        step=0,
        time_s=0.0,
        qpos=tuple(joint_positions.values()),
        qvel=(0.0,) * len(hand.joint_names),
        joint_positions=joint_positions,
        frame_poses=frame_poses,
    )
    return AdapterRunResult(
        backend=backend,
        scenario_id="small_step",
        status="completed",
        message="fixture",
        dt=0.002,
        requested_steps=0,
        completed_steps=0,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=(sample,),
        provenance={} if provenance is None else provenance,
    )


def _diagnostic_run(scale: float) -> tuple[CollectedRun, object]:
    manifest = load_manifest(MANIFEST)
    case = next(
        case
        for case in expand_scenario_cases(manifest)
        if case.case_id == "mujoco.left.small_step.base.r01"
    )
    hand = manifest.hand(case.hand)
    original = [0.1 + index * 0.001 for index in range(len(hand.joint_names))]
    effective = [value * scale for value in original]
    provenance: dict[str, object] = {
        "manifest_sha256": manifest_sha256(manifest),
        "session_id": "diagnostic-session",
        "source_revision": "a" * 40,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "backend": "mujoco",
        "device": "cpu",
        "headless": True,
        "renderer": False,
        "source_path": case.model_path,
        "source_sha256": "b" * 64,
        "frame_pose_source": "mujoco_data_xpos_xquat_after_mj_kinematics",
        "dof_frictionloss_scale": scale,
        "dof_frictionloss_mode": (
            "canonical_no_override" if scale == 1.0 else "diagnostic_scaled_override"
        ),
        "dof_frictionloss_scope": "all_compiled_dofs",
        "dof_frictionloss_order": "model_dof_index_ascending",
        "mapping_schema_version": 1,
        "backend_joint_names": list(hand.joint_names),
        "backend_frame_names": list(hand.distal_frame_names),
        "joint_mapping": [
            {"canonical_id": name} for name in hand.joint_names
        ],
        "frame_mapping": [
            {"canonical_id": name} for name in hand.distal_frame_names
        ],
        "dof_frictionloss_dof_indices": list(range(len(original))),
        "dof_frictionloss_original": original,
        "dof_frictionloss_effective": effective,
        "frame_pose_state_alignment": "current_sample_qpos",
        "compiled_control_parameters": {
            "schema_version": 1,
            "source": "mujoco_compiled_mjmodel_arrays",
            "semantics": "compiled_configuration_only_not_measured_torque",
            "contains_measured_or_realized_torque": False,
            "joint_order": list(hand.joint_names),
            "joints": [
                {
                    "canonical_id": name,
                    "dof_index": index,
                    "joint_frictionloss": effective[index],
                }
                for index, name in enumerate(hand.joint_names)
            ],
        },
    }
    scenario = manifest.scenario(case.scenario_id)
    initial = canonical_initial_positions(scenario, hand.joint_names)
    frame_poses = {
        name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        for name in hand.distal_frame_names
    }
    samples = tuple(
        TraceSample(
            step=step,
            time_s=step * case.dt_s,
            qpos=tuple(initial[name] for name in hand.joint_names),
            qvel=(0.0,) * len(hand.joint_names),
            joint_positions=initial,
            frame_poses=frame_poses,
            position_targets=canonical_position_targets(
                scenario,
                hand.joint_names,
                step_index=step,
                dt_s=case.dt_s,
            ),
            contact_count=0,
        )
        for step in range(scenario.steps + 1)
    )
    result = AdapterRunResult(
        backend="mujoco",
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


def _trend_row(
    scale: float,
    *,
    selected_joint: float,
    selected_frame: float,
    global_joint: float,
    global_frame: float,
) -> dict[str, object]:
    return {
        "dof_frictionloss_scale": scale,
        "at_formal_peak_time": {
            "absolute_joint_delta_rad": selected_joint,
            "frame_position_distance_m": selected_frame,
        },
        "metrics_vs_formal_ovphysx": {
            "joint_max_abs_rad": global_joint,
            "frame_position_max_m": global_frame,
        },
    }


def test_public_trace_delta_matches_formal_metric_semantics() -> None:
    hand = load_manifest(MANIFEST).hands[0]
    reference = _result()
    candidate = _result(joint_delta=0.012, frame_delta=0.003)

    assert trace_delta(reference, candidate, hand) == pytest.approx(
        (0.012, 0.003, 0.0)
    )


def test_friction_scan_localizes_the_peak_dp_frame() -> None:
    scan = _load_scan()
    hand = load_manifest(MANIFEST).hands[0]

    peak = scan._peak_frame_delta(
        _result(),
        _result(frame_delta=0.003),
        hand,
    )

    assert peak["frame_name"] == "left_thumb_DP"
    assert peak["step"] == 0
    assert peak["time_s"] == pytest.approx(0.0)
    assert peak["position_distance_m"] == pytest.approx(0.003)
    assert peak["ov_minus_mujoco_xyz_m"] == pytest.approx([-0.003, 0.0, 0.0])


def test_friction_scan_uses_only_pre_registered_scales() -> None:
    scan = _load_scan()

    assert scan.DIAGNOSTIC_SCALES == (1.0, 0.5, 0.0)
    assert scan._scale_directory(1.0) == "scale-1"
    assert scan._scale_directory(0.5) == "scale-0p5"
    assert scan._scale_directory(0.0) == "scale-0"
    with pytest.raises(scan.FrictionScanError, match="unregistered diagnostic scale"):
        scan._scale_directory(0.25)


def test_friction_scan_prediction_requires_registered_effect_size() -> None:
    scan = _load_scan()

    assert scan._decreases_by([0.018, 0.009, 0.001], 0.001) is True
    assert scan._decreases_by([0.018, 0.0175, 0.001], 0.001) is False
    assert scan._decreases_by([0.018, 0.009, 0.010], 0.001) is False
    assert scan._decreases_by([0.018], 0.001) is False


def test_friction_scan_validates_every_dof_and_selected_compiled_readback() -> None:
    scan = _load_scan()
    run, manifest = _diagnostic_run(0.5)

    original, effective = scan._validate_diagnostic_run(
        run,
        expected_case=run.case,
        scale=0.5,
        session_id="diagnostic-session",
        source_revision="a" * 40,
        manifest_digest=manifest_sha256(manifest),
        manifest=manifest,
        expected_source_sha256="b" * 64,
    )

    assert len(original) == len(effective) == 22
    assert effective[0] == pytest.approx(original[0] * 0.5)


def test_friction_scan_rejects_tampered_friction_provenance() -> None:
    scan = _load_scan()
    run, manifest = _diagnostic_run(0.5)
    provenance = dict(run.result.provenance)
    effective = list(provenance["dof_frictionloss_effective"])
    effective[7] += 0.01
    provenance["dof_frictionloss_effective"] = effective
    tampered = replace(run, result=replace(run.result, provenance=provenance))

    with pytest.raises(scan.FrictionScanError, match="differs at DOF 7"):
        scan._validate_diagnostic_run(
            tampered,
            expected_case=run.case,
            scale=0.5,
            session_id="diagnostic-session",
            source_revision="a" * 40,
            manifest_digest=manifest_sha256(manifest),
            manifest=manifest,
            expected_source_sha256="b" * 64,
        )


def test_friction_scan_rejects_incomplete_or_noncanonical_trace() -> None:
    scan = _load_scan()
    run, manifest = _diagnostic_run(1.0)
    incomplete = replace(
        run,
        result=replace(
            run.result,
            requested_steps=0,
            completed_steps=0,
            samples=run.result.samples[:1],
        ),
    )
    with pytest.raises(scan.FrictionScanError, match="complete canonical trace"):
        scan._validate_diagnostic_run(
            incomplete,
            expected_case=run.case,
            scale=1.0,
            session_id="diagnostic-session",
            source_revision="a" * 40,
            manifest_digest=manifest_sha256(manifest),
            manifest=manifest,
            expected_source_sha256="b" * 64,
        )

    changed_targets = dict(run.result.samples[50].position_targets)
    changed_targets["left_thumb_CMC_FE"] = 0.123
    samples = list(run.result.samples)
    samples[50] = replace(samples[50], position_targets=changed_targets)
    noncanonical = replace(
        run,
        result=replace(run.result, samples=tuple(samples)),
    )
    with pytest.raises(scan.FrictionScanError, match="non-canonical position targets"):
        scan._validate_diagnostic_run(
            noncanonical,
            expected_case=run.case,
            scale=1.0,
            session_id="diagnostic-session",
            source_revision="a" * 40,
            manifest_digest=manifest_sha256(manifest),
            manifest=manifest,
            expected_source_sha256="b" * 64,
        )


def test_friction_scan_command_and_initial_state_signatures_match_across_scales() -> None:
    scan = _load_scan()
    scale_one, manifest = _diagnostic_run(1.0)
    scale_half, _ = _diagnostic_run(0.5)
    hand = manifest.hand(scale_one.case.hand)

    assert scan._command_signature(scale_one.result, hand) == scan._command_signature(
        scale_half.result, hand
    )
    assert scan._initial_state_signature(
        scale_one.result, hand
    ) == scan._initial_state_signature(scale_half.result, hand)


def test_friction_scan_baseline_drift_makes_result_inconclusive() -> None:
    scan = _load_scan()
    rows = [
        _trend_row(1.0, selected_joint=0.018, selected_frame=0.0021, global_joint=0.019, global_frame=0.0022),
        _trend_row(0.5, selected_joint=0.009, selected_frame=0.0010, global_joint=0.010, global_frame=0.0011),
        _trend_row(0.0, selected_joint=0.001, selected_frame=0.0002, global_joint=0.002, global_frame=0.0003),
    ]

    trend = scan._evaluate_trend(rows, (2e-9, 0.0, 0.0))

    assert trend["prediction_supported"] is False
    assert trend["scientific_status"] == "inconclusive_baseline_drift"


@pytest.mark.parametrize(
    "rows",
    [
        [
            _trend_row(1.0, selected_joint=0.018, selected_frame=0.0021, global_joint=0.019, global_frame=0.0022),
            _trend_row(0.5, selected_joint=0.017995, selected_frame=0.0010, global_joint=0.010, global_frame=0.0011),
            _trend_row(0.0, selected_joint=0.001, selected_frame=0.0002, global_joint=0.002, global_frame=0.0003),
        ],
        [
            _trend_row(1.0, selected_joint=0.018, selected_frame=0.0021, global_joint=0.019, global_frame=0.0022),
            _trend_row(0.5, selected_joint=0.009, selected_frame=0.0010, global_joint=0.010, global_frame=0.0011),
            _trend_row(0.0, selected_joint=0.010, selected_frame=0.0002, global_joint=0.002, global_frame=0.0003),
        ],
    ],
)
def test_friction_scan_small_or_nonmonotonic_effect_is_not_supported(
    rows: list[dict[str, object]],
) -> None:
    scan = _load_scan()

    trend = scan._evaluate_trend(rows, (0.0, 0.0, 0.0))

    assert trend["prediction_supported"] is False
    assert trend["scientific_status"] == "prediction_not_supported"


def test_friction_scan_refuses_to_reuse_an_output_directory(tmp_path: Path) -> None:
    scan = _load_scan()
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(scan.FrictionScanError, match="already exists"):
        scan._create_output_root(output)


def test_friction_scan_confines_output_to_direct_results_child(tmp_path: Path) -> None:
    scan = _load_scan()
    results = tmp_path / "results"
    results.mkdir()

    assert scan._validated_output_path(tmp_path, results / "scan") == results / "scan"
    with pytest.raises(scan.FrictionScanError, match="direct child"):
        scan._validated_output_path(tmp_path, tmp_path / "outside")
    nested = results / "nested"
    nested.mkdir()
    with pytest.raises(scan.FrictionScanError, match="direct child"):
        scan._validated_output_path(tmp_path, nested / "scan")


def test_friction_scan_promotes_only_a_verified_staging_sibling(tmp_path: Path) -> None:
    scan = _load_scan()
    results = tmp_path / "results"
    results.mkdir()
    final = results / "scan"
    staging = scan._create_staging_root(final)

    assert ".incomplete-" in staging.name
    (staging / "evidence.txt").write_text("verified", encoding="utf-8")
    write_bundle_manifest(staging, scan._bundle_payload_paths(staging))
    assert scan._verify_exact_bundle(staging)["exact_inventory"] is True

    promoted = scan._promote_staging_root(staging, final)
    assert promoted == final
    assert not staging.exists()
    assert scan._verify_exact_bundle(final)["exact_inventory"] is True


def test_friction_scan_bundle_inventory_includes_nested_evidence_exactly(
    tmp_path: Path,
) -> None:
    scan = _load_scan()
    (tmp_path / "baseline").mkdir()
    (tmp_path / "inputs" / "source").mkdir(parents=True)
    (tmp_path / "baseline" / "formal-bundle.json").write_text("{}", encoding="utf-8")
    (tmp_path / "baseline" / "formal-finalization.json").write_text("{}", encoding="utf-8")
    (tmp_path / "inputs" / "gate0.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "inputs" / "source" / "adapter.py").write_text("# source\n", encoding="utf-8")
    payload = scan._bundle_payload_paths(tmp_path)

    assert "baseline/formal-bundle.json" in payload
    assert "baseline/formal-finalization.json" in payload
    assert "inputs/gate0.manifest.json" in payload
    assert "inputs/source/adapter.py" in payload
    write_bundle_manifest(tmp_path, payload)
    assert scan._verify_exact_bundle(tmp_path)["exact_inventory"] is True

    (tmp_path / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(scan.FrictionScanError, match="inventory is not exact"):
        scan._verify_exact_bundle(tmp_path)


def test_friction_scan_copied_formal_subset_matches_frozen_manifest(
    tmp_path: Path,
) -> None:
    scan = _load_scan()
    formal = tmp_path / "formal"
    formal.mkdir()
    selected = {
        "results/finalization.json": b"finalization",
        "evidence/mujoco.run.json": b"mujoco",
        "evidence/ovphysx.run.json": b"ovphysx",
    }
    for relative, content in selected.items():
        path = formal / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_bundle_manifest(formal, selected)
    root_sha = scan.verify_bundle(formal)["root_sha256"]

    copied = tmp_path / "copied"
    copied.mkdir()
    copied_manifest = copied / "formal-bundle.json"
    shutil.copyfile(formal / "bundle.json", copied_manifest)
    copied_payloads: dict[str, Path] = {}
    for index, relative in enumerate(selected):
        destination = copied / f"payload-{index}.json"
        shutil.copyfile(formal / relative, destination)
        copied_payloads[relative] = destination

    scan._verify_copied_formal_subset(
        copied_manifest,
        expected_root_sha256=root_sha,
        copied_payloads=copied_payloads,
    )
    next(iter(copied_payloads.values())).write_text("tampered", encoding="utf-8")
    with pytest.raises(scan.FrictionScanError, match="differs from its manifest"):
        scan._verify_copied_formal_subset(
            copied_manifest,
            expected_root_sha256=root_sha,
            copied_payloads=copied_payloads,
        )


def test_friction_scan_rejects_a_different_formal_bundle_before_io() -> None:
    scan = _load_scan()

    with pytest.raises(scan.FrictionScanError, match="authoritative c571d9b"):
        scan.run_scan(
            asset_root="missing-assets",
            manifest_path="missing-manifest",
            formal_bundle="missing-bundle",
            expected_formal_root_sha256="0" * 64,
            output_dir="missing-output",
            session_id="test",
            source_revision="1" * 40,
        )
