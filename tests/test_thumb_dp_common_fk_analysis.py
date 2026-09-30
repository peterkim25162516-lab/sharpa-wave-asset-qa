from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import wave_asset_qa.parity.diagnostics as diagnostic_helpers

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.parity.contracts import (
    AssetProvenance,
    ControlMode,
    DISTAL_FRAME_SUFFIXES,
    HandSide,
    HandSpec,
    JOINT_SUFFIXES,
    ModelPaths,
    Mounting,
    ParityManifest,
    RunPolicy,
    ScenarioKind,
    ScenarioSpec,
    Simulator,
)
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    asset_tree_sha256,
    replay_mujoco_fk_named,
    validate_diagnostic_run,
)
from wave_asset_qa.parity.bundle import sha256_file, verify_bundle, write_bundle_manifest
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    TimestepVariant,
    canonical_initial_positions,
    canonical_position_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_analysis_script():
    path = ROOT / "scripts" / "run_thumb_dp_common_fk_analysis.py"
    spec = importlib.util.spec_from_file_location(
        "test_run_thumb_dp_common_fk_analysis",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def analysis_script():
    return _load_analysis_script()


def _synthetic_plan(module) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "left-thumb-dp-common-mujoco-fk-attribution-v1",
        "classification": "post_hoc_mechanistic_analysis_not_formal_gate0",
        "analysis_plan_status": (
            "frozen_after_single_step_feasibility_probe_before_full_window_computation"
        ),
        "scope": {},
        "feasibility_disclosure": {
            "inspected_before_plan_freeze": True,
            "step": 66,
            "time_s": 0.132,
        },
        "frozen_inputs": {
            "parent_bundle_root_sha256": module.PARENT_ROOT_SHA256,
            "parent_bundle_manifest_sha256": module.PARENT_MANIFEST_SHA256,
            "parent_bundle_payload_count": module.PARENT_PAYLOAD_COUNT,
            "parent_bundle_total_size_bytes": module.PARENT_TOTAL_SIZE_BYTES,
            "mujoco_model_sha256": module.MODEL_SHA256,
            "frozen_mujoco_version": module.MUJOCO_VERSION,
        },
        "parent_paths": {},
        "cases": {},
        "analysis": {
            "hand": "left",
            "frame": module.FRAME_NAME,
            "window_start_step": module.START_STEP,
            "window_end_step": module.END_STEP,
            "sample_count": module.END_STEP - module.START_STEP + 1,
            "dt_s": module.DT_S,
            "candidate_roles": list(module.ROLE_ORDER),
            "thumb_players_in_order": list(module.PLAYER_NAMES),
            "coalition_count_per_sample": 32,
            "backend_native_qpos_direct_use_forbidden": True,
            "coordinate_transform": "identity_only_no_fit",
            "advance_dynamics": False,
            "fit_translation_rotation_scale_or_time_shift": False,
        },
        "sanity_gates": {
            "empty_coalition_max_position_error_m": (
                module.EMPTY_COALITION_ATOL_M
            ),
            "all_thumb_coalition_vs_full_ov_fk_max_position_error_m": (
                module.FULL_COALITION_ATOL_M
            ),
            "vector_shapley_efficiency_max_abs_error_m": (
                module.VECTOR_EFFICIENCY_ATOL_M
            ),
            "energy_shapley_efficiency_max_abs_fraction": (
                module.SHAPLEY_EFFICIENCY_ATOL
            ),
            "closure_denominator_min_frobenius_m": module.RAW_NORM_MIN_M,
        },
        "closure_metrics": {},
        "closure_decision_rule": {
            "closes_if_window_ratio_at_most": module.CLOSE_RHO_MAX,
            "closes_if_peak_ratio_at_most": module.CLOSE_RHO_MAX,
            "does_not_close_if_window_ratio_at_least": (
                module.DOES_NOT_CLOSE_RHO_MIN
            ),
            "does_not_close_if_peak_ratio_at_least": (
                module.DOES_NOT_CLOSE_RHO_MIN
            ),
        },
        "shapley": {
            "dominant_positive_minimum_fraction_points": (
                module.DOMINANT_MIN_NORMALIZED_ATTRIBUTION
            ),
            "dominant_positive_minimum_share_of_positive_attribution": (
                module.DOMINANT_MIN_POSITIVE_SHARE
            ),
            "material_adverse_maximum_fraction_points": (
                module.MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION
            ),
        },
        "output_policy": {},
        "claim_boundary": {},
    }


def _write_plan(path: Path, plan: object) -> None:
    path.write_text(
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def test_plan_bytes_and_semantics_are_independently_locked(
    analysis_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _synthetic_plan(analysis_script)
    plan_path = tmp_path / "synthetic-plan.json"
    _write_plan(plan_path, plan)
    monkeypatch.setattr(
        analysis_script,
        "PLAN_SHA256",
        analysis_script.sha256_file(plan_path),
    )

    assert analysis_script._load_plan(plan_path) == plan

    # A byte-only change must fail even though parsed JSON semantics are unchanged.
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        analysis_script.CommonFkAnalysisError,
        match="analysis plan bytes changed",
    ):
        analysis_script._load_plan(plan_path)

    # Re-freezing the byte hash must not bypass the independent method lock.
    tampered = deepcopy(plan)
    tampered["analysis"]["coordinate_transform"] = "translation_fit"  # type: ignore[index]
    _write_plan(plan_path, tampered)
    monkeypatch.setattr(
        analysis_script,
        "PLAN_SHA256",
        analysis_script.sha256_file(plan_path),
    )
    with pytest.raises(
        analysis_script.CommonFkAnalysisError,
        match="analysis method changed",
    ):
        analysis_script._load_plan(plan_path)


def _synthetic_manifest() -> ParityManifest:
    def hand(side: HandSide) -> HandSpec:
        prefix = side.value
        return HandSpec(
            side=side,
            model_name=f"{prefix}_sharpa_wave",
            mounting=Mounting.FIXED_BASE,
            control_mode=ControlMode.POSITION,
            model_paths=ModelPaths(
                mujoco=f"synthetic/mujoco/{prefix}_sharpa_wave.xml",
                ovphysx=f"synthetic/ovphysx/{prefix}_sharpa_wave.usda",
            ),
            joint_names=tuple(f"{prefix}_{suffix}" for suffix in JOINT_SUFFIXES),
            distal_frame_names=tuple(
                f"{prefix}_{suffix}" for suffix in DISTAL_FRAME_SUFFIXES
            ),
        )

    scenarios = (
        ScenarioSpec(
            scenario_id="zero_hold",
            kind=ScenarioKind.ZERO_HOLD,
            duration_s=0.002,
            dt_s=0.002,
            gravity_m_s2=(0.0, 0.0, 0.0),
            initial_position_rad=0.0,
            target_position_rad=0.0,
            step_start_s=None,
            expect_no_contacts=True,
        ),
        ScenarioSpec(
            scenario_id="small_step",
            kind=ScenarioKind.SMALL_STEP,
            duration_s=0.004,
            dt_s=0.002,
            gravity_m_s2=(0.0, 0.0, 0.0),
            initial_position_rad=0.0,
            target_position_rad=0.1,
            step_start_s=0.002,
            expect_no_contacts=True,
        ),
        ScenarioSpec(
            scenario_id="gravity_settling",
            kind=ScenarioKind.GRAVITY_SETTLING,
            duration_s=0.002,
            dt_s=0.002,
            gravity_m_s2=(0.0, 0.0, -9.81),
            initial_position_rad=0.0,
            target_position_rad=0.0,
            step_start_s=None,
            expect_no_contacts=True,
        ),
    )
    return ParityManifest(
        schema_version=1,
        manifest_id="synthetic-gate0",
        provenance=AssetProvenance(
            repository="https://example.invalid/synthetic-assets.git",
            commit="a" * 40,
            asset_git_tree="b" * 40,
            canonical_lf_asset_tree_sha256="c" * 64,
            asset_root="synthetic_assets",
        ),
        simulators=(Simulator.MUJOCO, Simulator.OVPHYSX),
        hands=(hand(HandSide.LEFT), hand(HandSide.RIGHT)),
        scenarios=scenarios,
        run_policy=RunPolicy(
            repeat_count=2,
            dt_halving_scenario_ids=("zero_hold",),
            minimum_completion_fraction=1.0,
        ),
    )


def _synthetic_ovphysx_run() -> tuple[ParityManifest, ScenarioCase, AdapterRunResult]:
    manifest = _synthetic_manifest()
    hand = manifest.hand(HandSide.LEFT)
    scenario = manifest.scenario("zero_hold")
    case = ScenarioCase(
        simulator=Simulator.OVPHYSX,
        hand=HandSide.LEFT,
        scenario_id=scenario.scenario_id,
        repeat_index=1,
        timestep_variant=TimestepVariant.BASE,
        dt_s=scenario.dt_s,
        model_path=hand.model_paths.ovphysx,
    )
    backend_joint_names = tuple(reversed(hand.joint_names))
    backend_frame_names = tuple(reversed(hand.distal_frame_names))
    joint_index = {name: index for index, name in enumerate(backend_joint_names)}
    frame_index = {name: index for index, name in enumerate(backend_frame_names)}
    samples: list[TraceSample] = []
    for step in range(scenario.steps + 1):
        if step == 0:
            positions = canonical_initial_positions(scenario, hand.joint_names)
        else:
            positions = {
                name: (index + 1) * 0.001
                for index, name in enumerate(hand.joint_names)
            }
        samples.append(
            TraceSample(
                step=step,
                time_s=step * case.dt_s,
                qpos=tuple(positions[name] for name in backend_joint_names),
                qvel=(0.0,) * len(hand.joint_names),
                joint_positions=positions,
                frame_poses={
                    name: (
                        index * 0.001,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    for index, name in enumerate(hand.distal_frame_names)
                },
                position_targets=canonical_position_targets(
                    scenario,
                    hand.joint_names,
                    step_index=step,
                    dt_s=case.dt_s,
                ),
                contact_count=None,
            )
        )
    provenance = {
        "backend_joint_names": list(backend_joint_names),
        "joint_mapping": [
            {
                "canonical_id": name,
                "backend": Simulator.OVPHYSX.value,
                "scope": HandSide.LEFT.value,
                "backend_name": name,
                "index": joint_index[name],
                "sign": 1.0,
                "offset": 0.0,
                "unit": "rad",
            }
            for name in hand.joint_names
        ],
        "backend_frame_names": list(backend_frame_names),
        "frame_mapping": [
            {
                "canonical_id": name,
                "backend": Simulator.OVPHYSX.value,
                "scope": HandSide.LEFT.value,
                "backend_name": name,
                "index": frame_index[name],
                "sign": 1.0,
                "offset": 0.0,
                "unit": "xyz_m_qwxyz",
            }
            for name in hand.distal_frame_names
        ],
        "contact_check_performed": False,
        "contact_observation_capability": "not_evaluated",
    }
    result = AdapterRunResult(
        backend=Simulator.OVPHYSX.value,
        scenario_id=scenario.scenario_id,
        status="completed",
        message="synthetic fixture",
        dt=case.dt_s,
        requested_steps=scenario.steps,
        completed_steps=scenario.steps,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=tuple(samples),
        provenance=provenance,
    )
    return manifest, case, result


def test_validate_diagnostic_run_accepts_explicit_reordered_backend_mapping() -> None:
    manifest, case, result = _synthetic_ovphysx_run()

    validate_diagnostic_run(result, case, manifest)


@pytest.mark.parametrize(
    ("record_group", "field", "bad_value"),
    [
        ("joint_mapping", "backend_name", "not-the-indexed-backend-joint"),
        ("joint_mapping", "backend", "mujoco"),
        ("joint_mapping", "index", 0),
        ("frame_mapping", "backend_name", "not-the-indexed-backend-frame"),
        ("frame_mapping", "backend", "mujoco"),
        ("frame_mapping", "index", 0),
    ],
)
def test_validate_diagnostic_run_rejects_inconsistent_backend_mapping(
    record_group: str,
    field: str,
    bad_value: object,
) -> None:
    manifest, case, result = _synthetic_ovphysx_run()
    provenance = deepcopy(dict(result.provenance))
    provenance[record_group][0][field] = bad_value  # type: ignore[index]
    tampered = AdapterRunResult(
        backend=result.backend,
        scenario_id=result.scenario_id,
        status=result.status,
        message=result.message,
        dt=result.dt,
        requested_steps=result.requested_steps,
        completed_steps=result.completed_steps,
        joint_names=result.joint_names,
        frame_names=result.frame_names,
        samples=result.samples,
        provenance=provenance,
    )

    with pytest.raises(DiagnosticEvidenceError, match="mapping"):
        validate_diagnostic_run(tampered, case, manifest)


@pytest.mark.parametrize("record_group", ["joint_mapping", "frame_mapping"])
def test_validate_diagnostic_run_rejects_noncanonical_mapping_order(
    record_group: str,
) -> None:
    manifest, case, result = _synthetic_ovphysx_run()
    provenance = deepcopy(dict(result.provenance))
    records = provenance[record_group]
    records[0], records[1] = records[1], records[0]  # type: ignore[index]
    tampered = AdapterRunResult(
        backend=result.backend,
        scenario_id=result.scenario_id,
        status=result.status,
        message=result.message,
        dt=result.dt,
        requested_steps=result.requested_steps,
        completed_steps=result.completed_steps,
        joint_names=result.joint_names,
        frame_names=result.frame_names,
        samples=result.samples,
        provenance=provenance,
    )

    with pytest.raises(DiagnosticEvidenceError, match="mapping coverage or order"):
        validate_diagnostic_run(tampered, case, manifest)


@pytest.fixture
def tiny_two_joint_mjcf(tmp_path: Path) -> Path:
    source = tmp_path / "tiny-two-joint.xml"
    source.write_text(
        """
        <mujoco model="tiny_two_joint_fk">
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="base">
              <body name="link_a">
                <joint name="joint_a" axis="0 1 0" damping="0.1"/>
                <geom type="capsule" size="0.01 0.05" mass="0.1"/>
                <body name="link_b" pos="0 0 0.1">
                  <joint name="joint_b" axis="1 0 0" damping="0.2"/>
                  <geom type="capsule" size="0.01 0.05" mass="0.1"/>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="joint_a_ctrl" joint="joint_a" kp="5" kv="0.3"/>
            <position name="joint_b_ctrl" joint="joint_b" kp="6" kv="0.4"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return source


def test_replay_mujoco_fk_named_requires_the_complete_compiled_joint_set(
    tiny_two_joint_mjcf: Path,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter()
    adapter.open(
        {
            "mjcf_path": str(tiny_two_joint_mjcf),
            "fixed_base": True,
            "control_mode": "position",
            "expected_joint_names": ["joint_a", "joint_b"],
            "expected_frame_names": ["link_b"],
        }
    )
    try:
        sample = replay_mujoco_fk_named(
            adapter,
            {"joint_a": 0.2, "joint_b": -0.15},
            expected_joint_names=("joint_a", "joint_b"),
            frame_names=("link_b",),
            step=17,
        )
        assert tuple(sample.joint_positions) == ("joint_a", "joint_b")
        assert sample.joint_positions == pytest.approx(
            {"joint_a": 0.2, "joint_b": -0.15}
        )
        assert sample.time_s == 0.0
        assert set(sample.frame_poses) == {"link_b"}
        assert all(math.isfinite(value) for value in sample.frame_poses["link_b"])

        with pytest.raises(
            DiagnosticEvidenceError,
            match="exactly the compiled canonical joint set",
        ):
            replay_mujoco_fk_named(
                adapter,
                {"joint_a": 0.2},
                expected_joint_names=("joint_a", "joint_b"),
                frame_names=("link_b",),
            )
        with pytest.raises(
            DiagnosticEvidenceError,
            match="exactly the compiled canonical joint set",
        ):
            replay_mujoco_fk_named(
                adapter,
                {"joint_a": 0.2},
                expected_joint_names=("joint_a",),
                frame_names=("link_b",),
            )
    finally:
        adapter.close()


def _decision_row(status: str) -> dict[str, object]:
    return {"closure": {"status": status}}


def test_overall_decision_preserves_formal_gate0_and_requires_both_to_close(
    analysis_script,
) -> None:
    closes = analysis_script._overall_decision(
        [
            _decision_row(analysis_script.CLOSES),
            _decision_row(analysis_script.CLOSES),
        ]
    )
    assert closes["candidate_statuses"] == {
        analysis_script.ROLE_CONTROL: analysis_script.CLOSES,
        analysis_script.ROLE_TREATMENT: analysis_script.CLOSES,
    }
    assert closes["common_mujoco_fk_closure_supported_for_both_candidates"] is True
    assert closes["scientific_status"] == (
        "kinematically_consistent_with_recorded_joint_state_gap_under_common_mujoco_fk"
    )
    assert closes["formal_gate0_status"] == "DIVERGENT"
    assert closes["formal_gate0_pass_ready"] is False
    assert closes["formal_gate0_result_changed"] is False
    assert "formal_gate0_pass_ready_unchanged" not in closes

    not_closed = analysis_script._overall_decision(
        [
            _decision_row(analysis_script.CLOSES),
            _decision_row(analysis_script.DOES_NOT_CLOSE),
        ]
    )
    assert not_closed["common_mujoco_fk_closure_supported_for_both_candidates"] is False
    assert not_closed["scientific_status"] == (
        "common_mujoco_fk_closure_does_not_hold_for_both_candidates"
    )
    assert not_closed["formal_gate0_status"] == "DIVERGENT"

    mixed = analysis_script._overall_decision(
        [
            _decision_row(analysis_script.CLOSES),
            _decision_row("MIXED"),
        ]
    )
    assert mixed["common_mujoco_fk_closure_supported_for_both_candidates"] is False
    assert mixed["scientific_status"] == "mixed_or_inconclusive"

    with pytest.raises(
        analysis_script.CommonFkAnalysisError,
        match="invalid closure status",
    ):
        analysis_script._overall_decision(
            [
                _decision_row(analysis_script.CLOSES),
                _decision_row("UNREGISTERED"),
            ]
        )


def test_attribution_row_is_json_round_trip_stable(analysis_script) -> None:
    raw = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    fk = raw.copy()
    coalitions = np.zeros((32, 1, 3), dtype=np.float64)
    for mask in range(32):
        coalitions[mask, 0, 0] = float(mask.bit_count()) / 5.0

    row = analysis_script._attribution_row(
        analysis_script.ROLE_CONTROL,
        raw,
        fk,
        coalitions,
    )

    assert json.loads(json.dumps(row)) == row


def test_rendered_report_discloses_nonblind_feasibility_probe(analysis_script) -> None:
    summary = {
        "decision": {"scientific_status": "synthetic"},
        "feasibility_disclosure": {"step": 66, "time_s": 0.132},
        "candidate_results": [],
        "claim_boundary": "Synthetic boundary.",
        "analysis_plan_sha256": "a" * 64,
        "parent_bundle_root_sha256": "b" * 64,
        "analysis_source_revision": "c" * 40,
        "analysis_source_tree": "d" * 40,
        "mujoco_model_sha256": "e" * 64,
    }

    report = analysis_script._render_report(summary)

    assert "frozen after a disclosed one-step feasibility check" in report
    assert "step 66 (0.132 s)" in report
    assert "post-hoc analysis, not a blind confirmation" in report


def test_parent_member_accepts_only_confined_normalized_paths(
    analysis_script,
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "synthetic-parent"
    nested = parent_root / "runs"
    nested.mkdir(parents=True)
    member = nested / "control.json"
    member.write_text("{}", encoding="utf-8")

    assert analysis_script._parent_member(
        parent_root.resolve(),
        "runs/control.json",
    ) == member.resolve()

    for unsafe in (
        "../outside.json",
        "runs/../control.json",
        "/absolute/path.json",
        ".",
        "",
    ):
        with pytest.raises(
            analysis_script.CommonFkAnalysisError,
            match="parent member",
        ):
            analysis_script._parent_member(parent_root.resolve(), unsafe)


def test_asset_tree_and_cli_root_reject_root_symlink(
    analysis_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real = tmp_path / "real-assets"
    real.mkdir()
    (real / "asset.txt").write_text("asset", encoding="utf-8")

    monkeypatch.setattr(
        diagnostic_helpers,
        "is_link_like",
        lambda path: Path(path) == real,
    )

    with pytest.raises(DiagnosticEvidenceError, match="symbolic link or junction"):
        asset_tree_sha256(real)
    monkeypatch.setattr(analysis_script, "is_link_like", lambda _path: True)
    with pytest.raises(
        analysis_script.CommonFkAnalysisError,
        match="symbolic link or junction",
    ):
        analysis_script._validated_real_asset_root(real)


def test_frozen_copy_audit_rejects_post_copy_mutation(
    analysis_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent_run = parent / "runs" / "run.json"
    parent_run.parent.mkdir(parents=True)
    parent_run.write_text('{"evidence": 1}\n', encoding="utf-8")
    write_bundle_manifest(parent, ["runs/run.json"])
    parent_verification = verify_bundle(parent)

    copied = tmp_path / "copied"
    copied.mkdir()
    copied_manifest = copied / "bundle.json"
    copied_run = copied / "run.json"
    copied_model = copied / "model.xml"
    copied_source = copied / "source.py"
    shutil.copyfile(parent / "bundle.json", copied_manifest)
    shutil.copyfile(parent_run, copied_run)
    copied_model.write_text("<mujoco/>\n", encoding="utf-8")
    copied_source.write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        analysis_script,
        "PARENT_ROOT_SHA256",
        parent_verification["root_sha256"],
    )
    monkeypatch.setattr(
        analysis_script,
        "PARENT_MANIFEST_SHA256",
        sha256_file(parent / "bundle.json"),
    )
    monkeypatch.setattr(
        analysis_script,
        "MODEL_SHA256",
        sha256_file(copied_model),
    )
    monkeypatch.setattr(
        analysis_script,
        "SOURCE_RELATIVE_PATHS",
        ("source.py",),
    )
    plan = {"parent_paths": {"run": "runs/run.json"}}
    source_copies = {"source.py": copied_source}
    source_hashes = {"source.py": sha256_file(copied_source)}
    input_copies = {
        "run": copied_run,
        "bundle_manifest": copied_manifest,
        "model": copied_model,
    }

    analysis_script._verify_frozen_copies(
        plan=plan,
        source_copies=source_copies,
        source_hashes=source_hashes,
        input_copies=input_copies,
    )

    copied_source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        analysis_script.CommonFkAnalysisError,
        match="copied analysis source differs",
    ):
        analysis_script._verify_frozen_copies(
            plan=plan,
            source_copies=source_copies,
            source_hashes=source_hashes,
            input_copies=input_copies,
        )

    copied_source.write_text("VALUE = 1\n", encoding="utf-8")
    copied_run.write_text('{"evidence": 2}\n', encoding="utf-8")
    with pytest.raises(DiagnosticEvidenceError, match="copied payload differs"):
        analysis_script._verify_frozen_copies(
            plan=plan,
            source_copies=source_copies,
            source_hashes=source_hashes,
            input_copies=input_copies,
        )


def test_expected_payload_inventory_is_closed(analysis_script, tmp_path: Path) -> None:
    members = {
        "summary": tmp_path / "summary.json",
        "run": tmp_path / "run.json",
    }
    observed = analysis_script._expected_payload_paths(members)

    assert "bundle.json" not in observed
    assert "replay.json" in observed
    assert "inputs/parent/summary/summary.json" in observed
    assert "inputs/parent/run/run.json" in observed
    assert len(observed) == len(analysis_script.SOURCE_RELATIVE_PATHS) + 7
