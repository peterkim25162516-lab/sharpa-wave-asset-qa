from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wave_asset_qa.adapters.base import (
    AdapterLifecycle,
    AdapterRunResult,
    TraceSample,
)
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import ParityManifest, Simulator
from wave_asset_qa.parity.runner import (
    RUN_FILE_SUFFIX,
    RunPayloadValidationError,
    collected_run_from_dict,
    collected_run_to_dict,
    load_collected_runs,
    run_backend_cases,
    run_mujoco_cases,
)
from wave_asset_qa.parity.scenarios import expand_scenario_cases
import wave_asset_qa.parity.runner as runner


ROOT = Path(__file__).resolve().parents[1]


def _short_manifest() -> ParityManifest:
    payload = json.loads(
        (ROOT / "configs" / "parity" / "gate0.json").read_text(encoding="utf-8")
    )
    for scenario in payload["scenarios"]:
        scenario["duration_s"] = 0.004
        scenario["dt_s"] = 0.002
        if scenario["scenario_id"] == "small_step":
            scenario["step_start_s"] = 0.002
    return ParityManifest.from_dict(payload)


def _completed_result(
    hand: object,
    scenario: object,
    dt: float,
) -> AdapterRunResult:
    joint_names = tuple(getattr(hand, "joint_names"))
    frame_names = tuple(getattr(hand, "distal_frame_names"))
    requested_steps = round(float(getattr(scenario, "duration_s")) / dt)
    samples = tuple(
        TraceSample(
            step=step,
            time_s=step * dt,
            qpos=(0.0,) * len(joint_names),
            qvel=(0.0,) * len(joint_names),
            joint_positions={name: 0.0 for name in joint_names},
            frame_poses={
                name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                for name in frame_names
            },
            position_targets={name: 0.0 for name in joint_names},
            contact_count=0,
        )
        for step in range(requested_steps + 1)
    )
    return AdapterRunResult(
        backend="mujoco",
        scenario_id=str(getattr(scenario, "scenario_id")),
        status="completed",
        message="fixture completed",
        dt=dt,
        requested_steps=requested_steps,
        completed_steps=requested_steps,
        joint_names=joint_names,
        frame_names=frame_names,
        samples=samples,
        provenance={"fixture": True},
    )


class _FakeAdapter:
    instances: list[_FakeAdapter] = []

    def __init__(self, *, asset_root: Path) -> None:
        self.asset_root = asset_root
        self._lifecycle = AdapterLifecycle.CREATED
        self.close_calls = 0
        type(self).instances.append(self)

    @property
    def backend_name(self) -> str:
        return "mujoco"

    @property
    def lifecycle(self) -> AdapterLifecycle:
        return self._lifecycle

    def run_scenario(
        self,
        hand: object,
        scenario: object,
        *,
        dt_override: float | None = None,
    ) -> AdapterRunResult:
        self._lifecycle = AdapterLifecycle.READY
        assert dt_override is not None
        return _completed_result(hand, scenario, dt_override)

    def close(self) -> None:
        self.close_calls += 1
        self._lifecycle = AdapterLifecycle.CLOSED


def test_runner_filters_canonical_cases_and_uses_fresh_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _short_manifest()
    selected = [
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator.value == "mujoco"
    ][:2]
    _FakeAdapter.instances = []
    monkeypatch.setattr(runner, "MuJoCoAdapter", _FakeAdapter)

    output_dir = tmp_path / "runs"
    runs = run_mujoco_cases(tmp_path, manifest, output_dir, case_ids=selected)
    loaded = load_collected_runs(output_dir)

    assert [run.case.case_id for run in runs] == selected
    assert loaded == runs
    assert len(_FakeAdapter.instances) == len(selected)
    assert all(instance.close_calls == 1 for instance in _FakeAdapter.instances)
    assert all(run.result.provenance["worker_pid"] == os.getpid() for run in runs)
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        f"{case_id}{RUN_FILE_SUFFIX}" for case_id in selected
    )


def test_runner_persists_adapter_exception_as_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingAdapter(_FakeAdapter):
        def run_scenario(
            self,
            hand: object,
            scenario: object,
            *,
            dt_override: float | None = None,
        ) -> AdapterRunResult:
            del hand, scenario, dt_override
            raise RuntimeError("fixture explosion")

    manifest = _short_manifest()
    case_id = next(
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator.value == "mujoco"
    )
    monkeypatch.setattr(runner, "MuJoCoAdapter", ExplodingAdapter)

    returned = run_mujoco_cases(tmp_path, manifest, tmp_path / "runs", case_id)
    loaded = load_collected_runs(tmp_path / "runs")

    assert returned == loaded
    assert returned[0].result.status == "error"
    assert returned[0].result.error == {
        "type": "RuntimeError",
        "message": "fixture explosion",
    }
    assert returned[0].result.requested_steps == 2


def test_nonfinite_adapter_output_is_replaced_by_json_safe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NonFiniteAdapter(_FakeAdapter):
        def run_scenario(
            self,
            hand: object,
            scenario: object,
            *,
            dt_override: float | None = None,
        ) -> AdapterRunResult:
            assert dt_override is not None
            result = _completed_result(hand, scenario, dt_override)
            first = result.samples[0]
            bad = TraceSample(
                step=first.step,
                time_s=first.time_s,
                qpos=(float("nan"), *first.qpos[1:]),
                qvel=first.qvel,
                joint_positions=first.joint_positions,
                frame_poses=first.frame_poses,
                position_targets=first.position_targets,
                contact_count=first.contact_count,
            )
            return AdapterRunResult(
                backend=result.backend,
                scenario_id=result.scenario_id,
                status=result.status,
                message=result.message,
                dt=result.dt,
                requested_steps=result.requested_steps,
                completed_steps=result.completed_steps,
                joint_names=result.joint_names,
                frame_names=result.frame_names,
                samples=(bad, *result.samples[1:]),
                provenance=result.provenance,
            )

    manifest = _short_manifest()
    case_id = next(
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator.value == "mujoco"
    )
    monkeypatch.setattr(runner, "MuJoCoAdapter", NonFiniteAdapter)

    runs = run_mujoco_cases(tmp_path, manifest, tmp_path / "runs", case_id)
    payload_text = next((tmp_path / "runs").iterdir()).read_text(encoding="utf-8")

    assert runs[0].result.status == "error"
    assert "invalid/non-finite" in runs[0].result.message
    assert "NaN" not in payload_text


def test_collected_run_round_trip_is_strict_and_rejects_nonfinite() -> None:
    manifest = _short_manifest()
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator.value == "mujoco"
    )
    run = CollectedRun(
        case=case,
        result=_completed_result(
            manifest.hand(case.hand), manifest.scenario(case.scenario_id), case.dt_s
        ),
        bundle_root_sha256="a" * 64,
    )
    payload = collected_run_to_dict(run)

    assert collected_run_from_dict(payload) == run

    unknown = deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(RunPayloadValidationError, match="unknown"):
        collected_run_from_dict(unknown)

    nonfinite = deepcopy(payload)
    nonfinite["result"]["samples"][0]["qpos"][0] = float("inf")
    with pytest.raises(RunPayloadValidationError, match="finite"):
        collected_run_from_dict(nonfinite)


def test_load_rejects_nonstandard_filename_and_json_nan(tmp_path: Path) -> None:
    bad_name = tmp_path / f"wrong{RUN_FILE_SUFFIX}"
    bad_name.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(RunPayloadValidationError, match="non-finite"):
        load_collected_runs(bad_name)


def _write_canonical_mjcf(asset_root: Path, manifest: ParityManifest) -> Path:
    hand = manifest.hands[0]
    destination = asset_root / Path(hand.model_paths.mujoco)
    destination.parent.mkdir(parents=True)
    bodies = "\n".join(
        f"""
        <body name="{name}_link" pos="0 0 {0.002 * (index + 1):.6f}">
          <joint name="{name}" axis="0 1 0" range="-1 1" damping="0.1"/>
          <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
        </body>"""
        for index, name in enumerate(hand.joint_names)
    )
    frames = "\n".join(
        f'<body name="{name}" pos="{0.001 * index:.6f} 0 0"/>'
        for index, name in enumerate(hand.distal_frame_names)
    )
    actuators = "\n".join(
        f'<position name="{name}_ctrl" joint="{name}" kp="0.5"/>'
        for name in hand.joint_names
    )
    destination.write_text(
        f"""
        <mujoco model="canonical_micro">
          <compiler fusestatic="false"/>
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody><body name="base">{bodies}{frames}</body></worldbody>
          <actuator>{actuators}</actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return destination


def test_real_micro_mjcf_case_is_written_and_reloadable(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    manifest = _short_manifest()
    _write_canonical_mjcf(tmp_path, manifest)
    case_id = next(
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator.value == "mujoco"
        and case.hand.value == "left"
        and case.scenario_id == "zero_hold"
        and case.repeat_index == 1
    )

    runs = run_mujoco_cases(tmp_path, manifest, tmp_path / "runs", case_id)
    loaded = load_collected_runs(tmp_path / "runs")

    assert runs == loaded
    assert runs[0].result.completed, runs[0].result.message
    assert runs[0].result.provenance["source_path"] == runs[0].case.model_path
    assert len(runs[0].result.joint_names) == 22
    assert len(runs[0].result.frame_names) == 5


def test_canonical_dt_halving_recomputes_steps_from_duration(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    manifest = _short_manifest()
    _write_canonical_mjcf(tmp_path, manifest)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator.value == "mujoco"
        and item.hand.value == "left"
        and item.scenario_id == "small_step"
        and item.timestep_variant.value == "halved"
        and item.repeat_index == 1
    )

    run = run_mujoco_cases(
        tmp_path,
        manifest,
        tmp_path / "runs",
        case.case_id,
    )[0]

    expected_steps = round(manifest.scenario(case.scenario_id).duration_s / case.dt_s)
    assert expected_steps == 4
    assert run.result.requested_steps == expected_steps
    assert run.result.completed_steps == expected_steps
    assert len(run.result.samples) == expected_steps + 1
    assert run.result.samples[-1].time_s == pytest.approx(
        manifest.scenario(case.scenario_id).duration_s
    )


def _fake_ovphysx_worker_result(
    manifest: ParityManifest,
    case: object,
    *,
    manifest_file_sha256: str,
    worker_script_sha256: str,
    session_id: str,
    source_revision: str,
    asset_tree_sha256: str,
) -> AdapterRunResult:
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    steps = round(scenario.duration_s / case.dt_s)
    backend_joints = tuple(reversed(hand.joint_names))
    backend_frames = ("root", *tuple(reversed(hand.distal_frame_names)))
    samples = []
    for step in range(steps + 1):
        targets = runner.canonical_position_targets(
            scenario,
            hand.joint_names,
            step_index=step,
            dt_s=case.dt_s,
        )
        samples.append(
            TraceSample(
                step=step,
                time_s=step * case.dt_s,
                qpos=(0.0,) * 22,
                qvel=(0.0,) * 22,
                joint_positions={name: 0.0 for name in hand.joint_names},
                frame_poses={
                    name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                    for name in hand.distal_frame_names
                },
                position_targets=targets,
                contact_count=None,
            )
        )
    provenance = {
        "case_id": case.case_id,
        "manifest_sha256": runner.manifest_sha256(manifest),
        "manifest_file_sha256": manifest_file_sha256,
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "simulation_configuration": {
            "verified": True,
            "requested_dt_s": case.dt_s,
            "cfg_dt_s": case.dt_s,
            "backend_dt_s": case.dt_s,
            "requested_gravity_m_s2": list(scenario.gravity_m_s2),
            "cfg_gravity_m_s2": list(scenario.gravity_m_s2),
            "physics_scene_gravity_m_s2": list(scenario.gravity_m_s2),
            "physics_prim_path": "/physicsScene",
        },
        "actuation_contract_version": 2,
        "actuator_model": "IdealPDActuator",
        "control_path": "explicit_pd_effort",
        "controller_dof_stiffness": [1.0] * 22,
        "controller_dof_damping": [0.1] * 22,
        "controller_dof_effort_limit": [2.0] * 22,
        "controller_dof_effort_limit_sim": [3.0] * 22,
        "controller_parameter_source": "ideal_pd_actuator_tensor",
        "backend_dof_stiffness": [0.0] * 22,
        "backend_dof_damping": [0.0] * 22,
        "backend_dof_drive_readback_source": "root_view_cpu_numpy_binding",
        "position_target_readback_verified": True,
        "position_target_readback_source": "articulation_data_joint_pos_target_torch",
        "zero_velocity_target_verified": True,
        "zero_feedforward_effort_target_verified": True,
        "position_target_readback_count": steps + 2,
        "position_target_readback_max_abs_error_rad": 0.0,
        "position_target_readback_values_rad": list(
            runner.canonical_position_targets(
                scenario,
                backend_joints,
                step_index=steps,
                dt_s=case.dt_s,
            ).values()
        ),
        "position_target_nonzero_readback_observed": scenario.kind.value
        in {"small_step", "offset_sine", "offset_linear_chirp"},
        "computed_effort_peak_abs_nm": [0.5] * 22,
        "applied_effort_peak_abs_nm": [0.5] * 22,
        "effort_observation_count": steps + 1,
        "effort_formula_max_abs_error_nm": 0.0,
        "effort_clip_max_abs_error_nm": 0.0,
        "effort_clip_count": 0,
        "effort_command_source": (
            "articulation_data_computed_and_applied_torque_torch"
        ),
        "mapping_schema_version": 1,
        "worker_script_sha256": worker_script_sha256,
        "source_path": str(Path("/fixture") / Path(case.model_path)),
        "source_sha256": "b" * 64,
        "backend_joint_names": list(backend_joints),
        "backend_frame_names": list(backend_frames),
        "joint_mapping": [
            {
                "canonical_id": name,
                "backend": "ovphysx",
                "scope": case.hand.value,
                "backend_name": name,
                "index": backend_joints.index(name),
                "sign": 1.0,
                "offset": 0.0,
                "unit": "rad",
            }
            for name in hand.joint_names
        ],
        "frame_mapping": [
            {
                "canonical_id": name,
                "backend": "ovphysx",
                "scope": case.hand.value,
                "backend_name": name,
                "index": backend_frames.index(name),
                "sign": 1.0,
                "offset": 0.0,
                "unit": "xyz_m_qwxyz",
            }
            for name in hand.distal_frame_names
        ],
        "contact_check_performed": False,
        "forbidden_modules": [],
    }
    if manifest.schema_version == 2:
        full_digest = runner.canonical_target_sequence_sha256(
            scenario,
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=True,
        )
        prefix_digest = runner.canonical_target_sequence_sha256(
            scenario,
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=False,
        )
        provenance.update(
            {
                "position_target_canonical_readback_max_abs_error_rad": 0.0,
                "target_sequence_digest_schema_version": 1,
                "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
                "target_sequence_digest_projection": (
                    "ieee754_binary32_roundtrip"
                ),
                "target_sequence_canonical_joint_names": list(hand.joint_names),
                "scheduled_target_sequence_semantics": (
                    "q[0..N]; target q[k] is recorded at t_k and applies to "
                    "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
                ),
                "scheduled_target_sequence_count": steps + 1,
                "scheduled_target_sequence_sha256": full_digest,
                "requested_target_sequence_semantics": (
                    "q[0..N] passed to set_joint_position_target_index"
                ),
                "requested_target_sequence_count": steps + 1,
                "requested_target_sequence_sha256": full_digest,
                "immediate_target_readback_sequence_semantics": (
                    "q[0..N] read immediately from joint_pos_target after each request"
                ),
                "immediate_target_readback_sequence_count": steps + 1,
                "immediate_target_readback_sequence_sha256": full_digest,
                "pre_step_applied_target_readback_sequence_semantics": (
                    "q[0..N-1] read after write_data_to_sim and before each physics step"
                ),
                "pre_step_applied_target_readback_sequence_count": steps,
                "pre_step_applied_target_readback_sequence_sha256": prefix_digest,
            }
        )
    return AdapterRunResult(
        backend="ovphysx",
        scenario_id=case.scenario_id,
        status="completed",
        message="fake worker completed",
        dt=case.dt_s,
        requested_steps=steps,
        completed_steps=steps,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        samples=tuple(samples),
        provenance=provenance,
    )


def test_ovphysx_worker_rejects_forged_final_target_readback() -> None:
    manifest = _short_manifest()
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.OVPHYSX and item.scenario_id == "small_step"
    )
    manifest_file_sha256 = "c" * 64
    worker_script_sha256 = "d" * 64
    session_id = "target-readback"
    source_revision = "a" * 40
    asset_tree_sha256 = manifest.provenance.canonical_lf_asset_tree_sha256
    result = _fake_ovphysx_worker_result(
        manifest,
        case,
        manifest_file_sha256=manifest_file_sha256,
        worker_script_sha256=worker_script_sha256,
        session_id=session_id,
        source_revision=source_revision,
        asset_tree_sha256=asset_tree_sha256,
    )
    provenance = dict(result.provenance)
    provenance["position_target_readback_values_rad"] = [0.0] * 22
    forged = replace(result, provenance=provenance)

    with pytest.raises(
        RunPayloadValidationError,
        match="final target readback does not match the canonical scenario",
    ):
        runner._validate_ovphysx_worker_result(
            forged,
            case=case,
            manifest=manifest,
            expected_manifest_sha256=runner.manifest_sha256(manifest),
            expected_manifest_file_sha256=manifest_file_sha256,
            session_id=session_id,
            source_revision=source_revision,
            worker_script_sha256=worker_script_sha256,
            asset_tree_sha256=asset_tree_sha256,
        )


def test_waveform_worker_rejects_forged_intermediate_target_digest() -> None:
    manifest_path = ROOT / "configs" / "parity" / "waveform_t1.json"
    manifest = runner.load_manifest(manifest_path)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.OVPHYSX
        and item.hand.value == "left"
        and item.scenario_id == "offset_sine"
        and item.timestep_variant.value == "base"
        and item.repeat_index == 1
    )
    manifest_file_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    worker_script_sha256 = "d" * 64
    session_id = "waveform-digest"
    source_revision = "a" * 40
    asset_tree_sha256 = manifest.provenance.canonical_lf_asset_tree_sha256
    result = _fake_ovphysx_worker_result(
        manifest,
        case,
        manifest_file_sha256=manifest_file_sha256,
        worker_script_sha256=worker_script_sha256,
        session_id=session_id,
        source_revision=source_revision,
        asset_tree_sha256=asset_tree_sha256,
    )

    validated = runner._validate_ovphysx_worker_result(
        result,
        case=case,
        manifest=manifest,
        expected_manifest_sha256=runner.manifest_sha256(manifest),
        expected_manifest_file_sha256=manifest_file_sha256,
        session_id=session_id,
        source_revision=source_revision,
        worker_script_sha256=worker_script_sha256,
        asset_tree_sha256=asset_tree_sha256,
    )
    assert validated.completed

    provenance = dict(result.provenance)
    provenance["immediate_target_readback_sequence_sha256"] = "0" * 64
    forged = replace(result, provenance=provenance)
    with pytest.raises(
        RunPayloadValidationError,
        match="immediate_target_readback_sequence_sha256.*canonical schedule",
    ):
        runner._validate_ovphysx_worker_result(
            forged,
            case=case,
            manifest=manifest,
            expected_manifest_sha256=runner.manifest_sha256(manifest),
            expected_manifest_file_sha256=manifest_file_sha256,
            session_id=session_id,
            source_revision=source_revision,
            worker_script_sha256=worker_script_sha256,
            asset_tree_sha256=asset_tree_sha256,
        )


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
def test_ovphysx_worker_rejects_invalid_ideal_pd_evidence(
    field: str, invalid_value: object
) -> None:
    manifest = _short_manifest()
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.OVPHYSX
    )
    manifest_file_sha256 = "c" * 64
    worker_script_sha256 = "d" * 64
    session_id = "ideal-pd-contract"
    source_revision = "a" * 40
    asset_tree_sha256 = manifest.provenance.canonical_lf_asset_tree_sha256
    result = _fake_ovphysx_worker_result(
        manifest,
        case,
        manifest_file_sha256=manifest_file_sha256,
        worker_script_sha256=worker_script_sha256,
        session_id=session_id,
        source_revision=source_revision,
        asset_tree_sha256=asset_tree_sha256,
    )
    provenance = dict(result.provenance)
    provenance[field] = invalid_value

    with pytest.raises(RunPayloadValidationError):
        runner._validate_ovphysx_worker_result(
            replace(result, provenance=provenance),
            case=case,
            manifest=manifest,
            expected_manifest_sha256=runner.manifest_sha256(manifest),
            expected_manifest_file_sha256=manifest_file_sha256,
            session_id=session_id,
            source_revision=source_revision,
            worker_script_sha256=worker_script_sha256,
            asset_tree_sha256=asset_tree_sha256,
        )


def test_builtin_ovphysx_runs_each_case_in_fresh_subprocess_and_preserves_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _short_manifest()
    approved = tmp_path / "approved"
    asset_root = approved / "assets"
    (asset_root / manifest.provenance.asset_root).mkdir(parents=True)
    worker_script = approved / "project" / "scripts" / "probe_ovphysx_runtime.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# fake worker\n", encoding="utf-8")
    output_dir = approved / "results" / "unit-session"
    selected = [
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.OVPHYSX
    ][:2]
    calls: list[tuple[list[str], dict[str, object]]] = []
    expected_asset_hash = manifest.provenance.canonical_lf_asset_tree_sha256
    monkeypatch.setattr(runner, "_asset_tree_sha256", lambda path: expected_asset_hash)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        arguments = {
            command[index]: command[index + 1]
            for index in range(2, len(command), 2)
        }
        worker_manifest = runner.load_manifest(arguments["--manifest"])
        case = next(
            item
            for item in expand_scenario_cases(worker_manifest)
            if item.case_id == arguments["--case-id"]
        )
        result = _fake_ovphysx_worker_result(
            worker_manifest,
            case,
            manifest_file_sha256=sha256(
                Path(arguments["--manifest"]).read_bytes()
            ).hexdigest(),
            worker_script_sha256=sha256(Path(command[1]).read_bytes()).hexdigest(),
            session_id=arguments["--session-id"],
            source_revision=arguments["--source-revision"],
            asset_tree_sha256=arguments["--asset-tree-sha256"],
        )
        Path(arguments["--output"]).write_text(
            json.dumps(result.to_dict(), allow_nan=False), encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"completed {case.case_id}\n",
            stderr="[fixture cooking warning]\n",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runs = run_backend_cases(
        asset_root,
        manifest,
        output_dir,
        backend=Simulator.OVPHYSX,
        case_ids=selected,
        approved_root=approved,
        worker_script=worker_script,
        worker_python=sys.executable,
        worker_timeout_s=10.0,
        session_id="unit-session",
        source_revision="a" * 40,
    )

    assert len(calls) == len(selected) == 2
    assert all(command[0] == str(Path(sys.executable).absolute()) for command, _ in calls)
    assert all(kwargs["shell"] is False for _command, kwargs in calls)
    assert all(len(run.result.samples) == run.result.requested_steps + 1 for run in runs)
    assert all(run.result.completed for run in runs)
    assert all(run.result.provenance["session_id"] == "unit-session" for run in runs)
    assert all(run.result.provenance["source_revision"] == "a" * 40 for run in runs)
    assert all(run.result.provenance["asset_commit"] == manifest.provenance.commit for run in runs)
    assert all(run.result.provenance["asset_git_tree"] == manifest.provenance.asset_git_tree for run in runs)
    assert load_collected_runs(output_dir) == runs
    for case_id in selected:
        assert (output_dir / f"{case_id}.worker.result.json").is_file()
        assert "completed" in (
            output_dir / f"{case_id}.worker.stdout.log"
        ).read_text(encoding="utf-8")
        assert "cooking warning" in (
            output_dir / f"{case_id}.worker.stderr.log"
        ).read_text(encoding="utf-8")
        assert (output_dir / f"{case_id}.worker.exitcode.txt").read_text(
            encoding="utf-8"
        ) == "0\n"
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        run_backend_cases(
            asset_root,
            manifest,
            output_dir,
            backend=Simulator.OVPHYSX,
            case_ids=selected,
            approved_root=approved,
            worker_script=worker_script,
            worker_python=sys.executable,
            session_id="unit-session",
            source_revision="a" * 40,
        )


def test_worker_python_virtualenv_symlink_entrypoint_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "python-real"
    target.write_text("fixture\n", encoding="utf-8")
    entrypoint = tmp_path / "env" / "bin" / "python"
    entrypoint.parent.mkdir(parents=True)
    path_type = type(entrypoint)
    original_resolve = path_type.resolve

    def fake_resolve(path: Path, strict: bool = False) -> Path:
        if path == entrypoint:
            return target
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(path_type, "resolve", fake_resolve)

    validated = runner._validated_executable_entrypoint(entrypoint, "worker_python")

    assert validated == entrypoint.absolute()
    assert validated != target.resolve()


def test_builtin_ovphysx_worker_missing_output_becomes_structured_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _short_manifest()
    approved = tmp_path / "approved"
    asset_root = approved / "assets"
    (asset_root / manifest.provenance.asset_root).mkdir(parents=True)
    worker_script = approved / "project" / "probe.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# no output fixture\n", encoding="utf-8")
    case_id = next(
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.OVPHYSX
    )
    monkeypatch.setattr(
        runner,
        "_asset_tree_sha256",
        lambda path: manifest.provenance.canonical_lf_asset_tree_sha256,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="claimed success\n", stderr=""
        ),
    )

    run = run_backend_cases(
        asset_root,
        manifest,
        approved / "results" / "missing-output",
        backend="ovphysx",
        case_ids=case_id,
        approved_root=approved,
        worker_script=worker_script,
        worker_python=sys.executable,
        session_id="missing-output",
        source_revision="a" * 40,
    )[0]

    assert run.result.status == "error"
    assert "did not create" in run.result.message
    assert run.result.provenance["worker_exitcode"] == 0


def test_builtin_ovphysx_rejects_outside_output_before_creating_it(
    tmp_path: Path,
) -> None:
    manifest = _short_manifest()
    approved = tmp_path / "approved"
    asset_root = approved / "assets"
    (asset_root / manifest.provenance.asset_root).mkdir(parents=True)
    worker_script = approved / "project" / "probe.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# fixture\n", encoding="utf-8")
    outside = tmp_path / "outside" / "must-not-exist"

    with pytest.raises(ValueError, match="escapes the approved results root"):
        run_backend_cases(
            asset_root,
            manifest,
            outside,
            backend="ovphysx",
            approved_root=approved,
            worker_script=worker_script,
            worker_python=sys.executable,
            session_id="outside-output",
            source_revision="a" * 40,
        )

    assert not outside.exists()


def test_ovphysx_adapter_factory_remains_an_in_process_test_hook(tmp_path: Path) -> None:
    manifest = _short_manifest()
    case_id = next(
        case.case_id
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.OVPHYSX
    )

    class FakeOVAdapter(_FakeAdapter):
        def run_scenario(
            self,
            hand: object,
            scenario: object,
            *,
            dt_override: float | None = None,
        ) -> AdapterRunResult:
            assert dt_override is not None
            self._lifecycle = AdapterLifecycle.READY
            return replace(_completed_result(hand, scenario, dt_override), backend="ovphysx")

    run = run_backend_cases(
        tmp_path,
        manifest,
        tmp_path / "fake-ov",
        backend="ovphysx",
        case_ids=case_id,
        adapter_factory=FakeOVAdapter,
        session_id="fake-hook",
        source_revision="a" * 40,
    )[0]

    assert run.result.completed
    assert run.result.backend == "ovphysx"
    assert run.result.provenance["session_id"] == "fake-hook"
