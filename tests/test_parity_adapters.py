from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wave_asset_qa.adapters.base import (
    AdapterCapabilityError,
    AdapterLifecycle,
    AdapterProbeResult,
    AdapterRunResult,
    SimulationAdapter,
    TraceSample,
)
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.adapters import ovphysx
from wave_asset_qa.parity.scenarios import expand_scenario_cases, load_manifest


@pytest.fixture
def tiny_mjcf(tmp_path: Path) -> Path:
    source = tmp_path / "tiny.xml"
    source.write_text(
        """
        <mujoco model="tiny_fixed">
          <option timestep="0.002" gravity="0 0 0" solver="CG"
                  integrator="implicitfast" iterations="37"
                  ls_iterations="11" tolerance="1e-7"/>
          <worldbody>
            <body name="base">
              <body name="tip" pos="0 0 0.1">
                <joint name="hinge" axis="0 1 0" range="-1 1"
                       armature="0.003" damping="0.02" frictionloss="0.04"/>
                <geom type="capsule" size="0.01 0.05" mass="0.1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="hinge_ctrl" joint="hinge" kp="5" kv="0.7"
                      gear="1" forcelimited="true" forcerange="-2 3"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return source


@pytest.fixture
def two_joint_mjcf(tmp_path: Path) -> Path:
    source = tmp_path / "two_joint.xml"
    source.write_text(
        """
        <mujoco model="two_joint_fixed">
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="base">
              <body name="link_a">
                <joint name="joint_a" axis="0 1 0" damping="0.1"
                       frictionloss="0.04"/>
                <geom type="capsule" size="0.01 0.05" mass="0.1"/>
                <body name="link_b" pos="0 0 0.1">
                  <joint name="joint_b" axis="1 0 0" damping="0.2"
                         frictionloss="0.08"/>
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


def _manifest(path: Path) -> dict[str, object]:
    return {
        "mjcf_path": str(path),
        "fixed_base": True,
        "control_mode": "position",
        "expected_joint_names": ["hinge"],
        "expected_frame_names": ["tip"],
    }


def _two_joint_manifest(path: Path) -> dict[str, object]:
    return {
        "mjcf_path": str(path),
        "fixed_base": True,
        "control_mode": "position",
        "expected_joint_names": ["joint_a", "joint_b"],
        "expected_frame_names": ["link_b"],
    }


def test_base_trace_and_execution_result_are_json_ready() -> None:
    sample = TraceSample(
        step=1,
        time_s=0.01,
        qpos=(0.1,),
        qvel=(0.2,),
        joint_positions={"hinge": 0.1},
        frame_poses={"tip": (0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)},
    )
    result = AdapterRunResult(
        backend="fixture",
        scenario_id="case",
        status="completed",
        message="ok",
        dt=0.01,
        requested_steps=1,
        completed_steps=1,
        samples=(sample,),
    )

    payload = result.to_dict()
    assert result.completed
    assert payload["samples"][0]["frame_poses"]["tip"][-1] == 0.0
    json.dumps(payload, allow_nan=False)


def test_mujoco_adapter_matches_runtime_protocol() -> None:
    assert isinstance(MuJoCoAdapter(), SimulationAdapter)


@pytest.mark.parametrize(
    "scale",
    [
        True,
        False,
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        "1.0",
        None,
    ],
)
def test_mujoco_rejects_invalid_dof_frictionloss_scale(scale: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"dof_frictionloss_scale must be a finite number >= 0",
    ):
        MuJoCoAdapter(dof_frictionloss_scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "targets",
    [
        True,
        {"hinge": True},
        {"hinge": -0.1},
        {"hinge": float("nan")},
        {"hinge": float("inf")},
        {"hinge": 10**400},
        {"hinge": "0.7"},
        {"": 0.7},
        {1: 0.7},
    ],
)
def test_mujoco_rejects_invalid_joint_total_damping_targets(
    targets: object,
) -> None:
    with pytest.raises(ValueError, match="joint_total_damping_targets"):
        MuJoCoAdapter(  # type: ignore[arg-type]
            joint_total_damping_targets=targets,
        )


def test_mujoco_dof_frictionloss_scale_defaults_to_canonical_model(
    tiny_mjcf: Path,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter()
    adapter.open(_manifest(tiny_mjcf))
    try:
        provenance = adapter._provenance(joint_names=("hinge",))
    finally:
        adapter.close()

    assert provenance["dof_frictionloss_scale"] == pytest.approx(1.0)
    assert provenance["dof_frictionloss_mode"] == "canonical_no_override"
    assert provenance["dof_frictionloss_scope"] == "all_compiled_dofs"
    assert provenance["dof_frictionloss_order"] == "model_dof_index_ascending"
    assert provenance["dof_frictionloss_dof_indices"] == [0]
    assert provenance["dof_frictionloss_original"] == pytest.approx([0.04])
    assert provenance["dof_frictionloss_effective"] == pytest.approx([0.04])
    assert provenance["joint_total_damping_mode"] == "canonical_no_override"
    assert provenance["joint_total_damping_scope"] == (
        "all_compiled_canonical_joints"
    )
    assert provenance["joint_total_damping_order"] == (
        "model_joint_index_ascending"
    )
    assert provenance["joint_total_damping_target_joint_names"] == []
    damping_record = provenance["joint_total_damping_records"][0]
    assert damping_record["canonical_id"] == "hinge"
    assert damping_record["override_applied"] is False
    assert damping_record["original_actuator_gainprm"] == pytest.approx(
        damping_record["effective_actuator_gainprm"]
    )
    assert damping_record["original_actuator_biasprm"] == pytest.approx(
        damping_record["effective_actuator_biasprm"]
    )
    assert damping_record["changed_gainprm_indices"] == []
    assert damping_record["changed_biasprm_indices"] == []
    assert damping_record["original_controller_damping_kv"] == pytest.approx(0.7)
    assert damping_record["joint_damping"] == pytest.approx(0.02)
    assert damping_record["original_total_damping"] == pytest.approx(0.72)
    assert damping_record["requested_total_damping"] == pytest.approx(0.72)
    assert damping_record["effective_controller_damping_kv"] == pytest.approx(0.7)
    assert damping_record["effective_total_damping"] == pytest.approx(0.72)
    compiled = provenance["compiled_control_parameters"]
    assert compiled["joints"][0]["joint_frictionloss"] == pytest.approx(0.04)


def test_mujoco_dof_frictionloss_scale_changes_only_the_compiled_model(
    tiny_mjcf: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter(dof_frictionloss_scale=0.25)
    adapter.open(_manifest(tiny_mjcf))
    try:
        provenance = adapter._provenance(joint_names=("hinge",))
        assert adapter._model.dof_frictionloss[0] == pytest.approx(0.01)
    finally:
        adapter.close()

    assert provenance["dof_frictionloss_scale"] == pytest.approx(0.25)
    assert provenance["dof_frictionloss_mode"] == "diagnostic_scaled_override"
    assert provenance["dof_frictionloss_scope"] == "all_compiled_dofs"
    assert provenance["dof_frictionloss_order"] == "model_dof_index_ascending"
    assert provenance["dof_frictionloss_dof_indices"] == [0]
    assert provenance["dof_frictionloss_original"] == pytest.approx([0.04])
    assert provenance["dof_frictionloss_effective"] == pytest.approx([0.01])
    compiled = provenance["compiled_control_parameters"]
    assert compiled["joints"][0]["joint_frictionloss"] == pytest.approx(0.01)
    assert adapter._original_dof_frictionloss == ()

    canonical_model = mujoco.MjModel.from_xml_path(str(tiny_mjcf))
    assert canonical_model.dof_frictionloss[0] == pytest.approx(0.04)


def test_mujoco_zero_frictionloss_scale_preserves_original_audit_values(
    tiny_mjcf: Path,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter(dof_frictionloss_scale=0.0)
    adapter.open(_manifest(tiny_mjcf))
    try:
        provenance = adapter._provenance(joint_names=("hinge",))
    finally:
        adapter.close()

    assert provenance["dof_frictionloss_original"] == pytest.approx([0.04])
    assert provenance["dof_frictionloss_effective"] == pytest.approx([0.0])
    assert provenance["compiled_control_parameters"]["joints"][0][
        "joint_frictionloss"
    ] == pytest.approx(0.0)
    assert adapter._original_dof_frictionloss == ()


def test_mujoco_frictionloss_scale_does_not_accumulate_across_reopen(
    tiny_mjcf: Path,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter(dof_frictionloss_scale=0.25)
    observed: list[tuple[float, float]] = []
    for _ in range(2):
        adapter.open(_manifest(tiny_mjcf))
        provenance = adapter._provenance(joint_names=("hinge",))
        observed.append(
            (
                provenance["dof_frictionloss_original"][0],
                provenance["dof_frictionloss_effective"][0],
            )
        )
        adapter.close()

    assert observed == pytest.approx([(0.04, 0.01), (0.04, 0.01)])


def test_mujoco_joint_total_damping_override_is_ordered_combined_and_fresh(
    two_joint_mjcf: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    source_before = two_joint_mjcf.read_bytes()
    adapter = MuJoCoAdapter(
        dof_frictionloss_scale=0.0,
        joint_total_damping_targets={"joint_b": 0.9, "joint_a": 0.5},
    )
    observed: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    for _ in range(2):
        adapter.open(_two_joint_manifest(two_joint_mjcf))
        try:
            provenance = adapter._provenance(
                joint_names=("joint_a", "joint_b"),
            )
            records = provenance["joint_total_damping_records"]
            compiled_joints = provenance["compiled_control_parameters"]["joints"]
            observed.append(
                (
                    tuple(
                        record["original_controller_damping_kv"]
                        for record in records
                    ),
                    tuple(
                        record["effective_controller_damping_kv"]
                        for record in records
                    ),
                )
            )

            assert provenance["dof_frictionloss_original"] == pytest.approx(
                [0.04, 0.08]
            )
            assert provenance["dof_frictionloss_effective"] == pytest.approx(
                [0.0, 0.0]
            )
            assert provenance["joint_total_damping_mode"] == (
                "diagnostic_total_damping_override"
            )
            assert provenance["joint_total_damping_scope"] == (
                "all_compiled_canonical_joints"
            )
            assert provenance["joint_total_damping_order"] == (
                "model_joint_index_ascending"
            )
            assert provenance["joint_total_damping_target_joint_names"] == [
                "joint_a",
                "joint_b",
            ]
            assert [record["canonical_id"] for record in records] == [
                "joint_a",
                "joint_b",
            ]
            assert [record["override_applied"] for record in records] == [
                True,
                True,
            ]
            for record in records:
                original_gainprm = record["original_actuator_gainprm"]
                effective_gainprm = record["effective_actuator_gainprm"]
                original_biasprm = record["original_actuator_biasprm"]
                effective_biasprm = record["effective_actuator_biasprm"]
                assert original_gainprm == pytest.approx(effective_gainprm)
                assert record["changed_gainprm_indices"] == []
                assert record["changed_biasprm_indices"] == [2]
                assert len(original_biasprm) == len(effective_biasprm)
                assert original_biasprm[2] != pytest.approx(effective_biasprm[2])
                for parameter_index in range(len(original_biasprm)):
                    if parameter_index != 2:
                        assert original_biasprm[parameter_index] == pytest.approx(
                            effective_biasprm[parameter_index]
                        )
            assert [record["joint_damping"] for record in records] == pytest.approx(
                [0.1, 0.2]
            )
            assert [
                record["original_total_damping"] for record in records
            ] == pytest.approx([0.4, 0.6])
            assert [
                record["requested_total_damping"] for record in records
            ] == pytest.approx([0.5, 0.9])
            assert [
                record["effective_controller_damping_kv"] for record in records
            ] == pytest.approx([0.4, 0.7])
            assert [
                record["effective_total_damping"] for record in records
            ] == pytest.approx([0.5, 0.9])
            assert [
                joint["controller_damping_kv"] for joint in compiled_joints
            ] == pytest.approx([0.4, 0.7])
        finally:
            adapter.close()
        assert adapter._joint_total_damping_target_joint_names == ()
        assert adapter._joint_total_damping_records == ()

    assert observed == pytest.approx(
        [((0.3, 0.4), (0.4, 0.7)), ((0.3, 0.4), (0.4, 0.7))]
    )
    assert two_joint_mjcf.read_bytes() == source_before
    canonical_model = mujoco.MjModel.from_xml_path(str(two_joint_mjcf))
    assert canonical_model.dof_frictionloss == pytest.approx([0.04, 0.08])
    assert -canonical_model.actuator_biasprm[:, 2] == pytest.approx([0.3, 0.4])


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ({"missing": 0.5}, "unknown canonical joint"),
        ({"hinge": 0.01}, "must be >= compiled joint damping"),
    ],
)
def test_mujoco_rejects_unusable_joint_total_damping_targets_after_compile(
    tiny_mjcf: Path,
    targets: dict[str, float],
    message: str,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter(joint_total_damping_targets=targets)

    with pytest.raises(ValueError, match=message):
        adapter.open(_manifest(tiny_mjcf))

    assert adapter.lifecycle is AdapterLifecycle.FAILED
    assert adapter._model is None
    assert adapter._original_dof_frictionloss == ()
    assert adapter._joint_total_damping_target_joint_names == ()
    assert adapter._joint_total_damping_records == ()


def test_mujoco_open_failure_clears_frictionloss_audit_state(
    tiny_mjcf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter(dof_frictionloss_scale=0.25)

    def fail_reset() -> None:
        raise RuntimeError("forced reset failure")

    monkeypatch.setattr(adapter, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="forced reset failure"):
        adapter.open(_manifest(tiny_mjcf))

    assert adapter.lifecycle is AdapterLifecycle.FAILED
    assert adapter._mujoco is None
    assert adapter._model is None
    assert adapter._data is None
    assert adapter._source_path is None
    assert adapter._joint_names == ()
    assert adapter._joint_qpos_address == {}
    assert adapter._joint_actuator == {}
    assert adapter._position_targets == {}
    assert adapter._mapping_scope == "unscoped"
    assert adapter._original_dof_frictionloss == ()
    assert adapter._joint_total_damping_target_joint_names == ()
    assert adapter._joint_total_damping_records == ()


def test_mujoco_direct_lifecycle_and_named_trace(tiny_mjcf: Path) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter()

    adapter.open(_manifest(tiny_mjcf), dt_override=0.001)
    assert adapter.lifecycle is AdapterLifecycle.READY
    assert adapter.dt == pytest.approx(0.001)
    adapter.set_position_targets({"hinge": 0.05})
    adapter.step()
    sample = adapter.sample(step=1, joint_names=("hinge",), frame_names=("tip",))
    adapter.close()

    assert adapter.lifecycle is AdapterLifecycle.CLOSED
    assert sample.step == 1
    assert sample.time_s == pytest.approx(0.001)
    assert sample.position_targets == {"hinge": 0.05}
    assert len(sample.frame_poses["tip"]) == 7
    assert sample.contact_count == 0


def test_mujoco_frame_pose_matches_the_same_samples_current_qpos(
    tiny_mjcf: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter()

    adapter.open(_manifest(tiny_mjcf))
    try:
        adapter.set_position_targets({"hinge": 0.5})
        adapter.step()
        sample = adapter.sample(step=1, joint_names=("hinge",), frame_names=("tip",))
    finally:
        adapter.close()

    assert abs(sample.qpos[0]) > 1e-12
    replay_model = mujoco.MjModel.from_xml_path(str(tiny_mjcf))
    replay_data = mujoco.MjData(replay_model)
    replay_data.qpos[:] = sample.qpos
    mujoco.mj_kinematics(replay_model, replay_data)
    tip_id = mujoco.mj_name2id(replay_model, mujoco.mjtObj.mjOBJ_BODY, "tip")
    expected_pose = tuple(float(value) for value in replay_data.xpos[tip_id]) + tuple(
        float(value) for value in replay_data.xquat[tip_id]
    )

    assert sample.frame_poses["tip"] == pytest.approx(expected_pose, abs=1e-12)


def test_mujoco_provenance_records_compiled_control_parameters(
    tiny_mjcf: Path,
) -> None:
    pytest.importorskip("mujoco")
    adapter = MuJoCoAdapter()
    adapter.open(_manifest(tiny_mjcf))
    try:
        provenance = adapter._provenance(joint_names=("hinge",))
    finally:
        adapter.close()

    compiled = provenance["compiled_control_parameters"]
    assert provenance["frame_pose_source"] == (
        "mujoco_data_xpos_xquat_after_mj_kinematics"
    )
    assert provenance["frame_pose_state_alignment"] == "current_sample_qpos"
    assert isinstance(compiled, dict)
    assert compiled["schema_version"] == 1
    assert compiled["source"] == "mujoco_compiled_mjmodel_arrays"
    assert compiled["semantics"] == "compiled_configuration_only_not_measured_torque"
    assert compiled["contains_measured_or_realized_torque"] is False
    assert compiled["joint_order"] == ["hinge"]
    assert compiled["field_sources"]["controller_damping_kv"] == (
        "-model.actuator_biasprm[actuator_id, 2]"
    )

    joints = compiled["joints"]
    assert isinstance(joints, list) and len(joints) == 1
    control = joints[0]
    assert control["canonical_id"] == "hinge"
    assert control["joint_type"] == "hinge"
    assert control["actuator_name"] == "hinge_ctrl"
    assert control["transmission_type"] == "joint"
    assert control["direct_gear"] is True
    assert control["controller_stiffness_kp"] == pytest.approx(5.0)
    assert control["controller_damping_kv"] == pytest.approx(0.7)
    assert control["gear"] == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert control["forcelimited"] is True
    assert control["forcerange"] == pytest.approx([-2.0, 3.0])
    assert control["joint_armature"] == pytest.approx(0.003)
    assert control["joint_damping"] == pytest.approx(0.02)
    assert control["joint_frictionloss"] == pytest.approx(0.04)
    assert control["units"] == {
        "joint_coordinate": "rad",
        "controller_stiffness_kp": "N*m/rad",
        "controller_damping_kv": "N*m*s/rad",
        "gear": "dimensionless",
        "forcerange": "N*m",
        "joint_armature": "kg*m^2",
        "joint_damping": "N*m*s/rad",
        "joint_frictionloss": "N*m",
    }

    solver = compiled["solver"]
    assert solver["source"] == "model.opt"
    assert solver["solver"] == "cg"
    assert solver["integrator"] == "implicitfast"
    assert solver["timestep"] == pytest.approx(0.002)
    assert solver["iterations"] == 37
    assert solver["ls_iterations"] == 11
    assert solver["tolerance"] == pytest.approx(1e-7)
    assert solver["gravity"] == pytest.approx([0.0, 0.0, 0.0])
    json.dumps(provenance, allow_nan=False)


def test_mujoco_rejects_nonunity_position_actuator_gear(
    tiny_mjcf: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    source = tmp_path / "geared.xml"
    source.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace('gear="1"', 'gear="2"'),
        encoding="utf-8",
    )

    result = MuJoCoAdapter().run_scenario(
        _manifest(source),
        {"id": "direct_gear_only", "kind": "zero_hold", "steps": 1},
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.get("capability") == "direct_position_actuator"


@pytest.mark.parametrize(
    ("kind", "gravity", "target", "step_start"),
    [
        ("zero_hold", (0.0, 0.0, 0.0), 0.0, None),
        ("small_step", (0.0, 0.0, 0.0), 0.05, 0.002),
        ("gravity_settling", (0.0, 0.0, -9.81), 0.0, None),
    ],
)
def test_mujoco_runs_gate0_contract_scenarios(
    tiny_mjcf: Path,
    kind: str,
    gravity: tuple[float, float, float],
    target: float,
    step_start: float | None,
) -> None:
    pytest.importorskip("mujoco")
    scenario = SimpleNamespace(
        scenario_id=f"tiny_{kind}",
        kind=kind,
        duration_s=0.004,
        dt_s=0.001,
        gravity_m_s2=gravity,
        target_position_rad=target,
        step_start_s=step_start,
        expect_no_contacts=True,
    )

    adapter = MuJoCoAdapter()
    result = adapter.run_scenario(_manifest(tiny_mjcf), scenario)

    assert result.completed, result.message
    assert result.dt == pytest.approx(0.001)
    assert result.completed_steps == 4
    assert len(result.samples) == 5  # reset state plus one sample per completed step
    assert result.frame_names == ("tip",)
    assert result.provenance["source_sha256"]
    assert adapter.lifecycle is AdapterLifecycle.CLOSED
    backend_frames = result.provenance["backend_frame_names"]
    frame_mapping = result.provenance["frame_mapping"]
    assert isinstance(backend_frames, list) and len(backend_frames) > 1
    assert isinstance(frame_mapping, list) and len(frame_mapping) == 1
    assert backend_frames[frame_mapping[0]["index"]] == "tip"
    if kind == "small_step":
        assert result.samples[1].position_targets["hinge"] == 0.0
        assert result.samples[2].position_targets["hinge"] == pytest.approx(0.05)
        assert result.samples[2].joint_positions["hinge"] == pytest.approx(0.0, abs=1e-12)
        assert result.samples[3].joint_positions["hinge"] > 0.0


def test_mujoco_reports_nonfinite_target_without_stepping(tiny_mjcf: Path) -> None:
    pytest.importorskip("mujoco")
    result = MuJoCoAdapter().run_scenario(
        _manifest(tiny_mjcf),
        {
            "id": "bad_target",
            "kind": "small_step",
            "steps": 2,
            "dt_s": 0.001,
            "targets": {"hinge": float("nan")},
        },
    )

    assert result.status == "error"
    assert result.completed_steps == 0
    assert "finite" in result.message


def test_mujoco_rejects_non_position_actuator(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    source = tmp_path / "motor.xml"
    source.write_text(
        """
        <mujoco><worldbody><body><joint name="hinge"/>
        <geom size="0.1" mass="1"/></body></worldbody>
        <actuator><motor joint="hinge"/></actuator></mujoco>
        """,
        encoding="utf-8",
    )

    result = MuJoCoAdapter().run_scenario(
        {"mjcf_path": source}, {"id": "position_only", "kind": "zero_hold", "steps": 1}
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.get("capability") == "position_actuator_mapping"


def test_mujoco_rejects_floating_base(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    source = tmp_path / "floating.xml"
    source.write_text(
        """
        <mujoco><worldbody><body><freejoint name="root"/>
        <geom size="0.1" mass="1"/></body></worldbody></mujoco>
        """,
        encoding="utf-8",
    )

    result = MuJoCoAdapter().run_scenario(
        {"mjcf_path": source}, {"id": "fixed_only", "kind": "zero_hold", "steps": 1}
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.get("capability") == "fixed_base_only"


def test_mujoco_resolves_model_paths_relative_to_asset_root(tiny_mjcf: Path) -> None:
    pytest.importorskip("mujoco")
    hand = SimpleNamespace(
        model_paths=SimpleNamespace(mujoco=tiny_mjcf.name),
        mounting="fixed_base",
        control_mode="position",
    )
    adapter = MuJoCoAdapter(asset_root=tiny_mjcf.parent)
    adapter.open(hand)
    adapter.close()


def test_importing_ovphysx_adapter_does_not_import_heavy_runtime() -> None:
    command = (
        "import sys; before=set(sys.modules); "
        "import wave_asset_qa.adapters.ovphysx; "
        "added=set(sys.modules)-before; "
        "bad=[n for n in added if n.startswith(('isaaclab','isaaclab_ovphysx','omni.kit','isaacsim'))]; "
        "raise SystemExit(1 if bad else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", command], check=False)
    assert completed.returncode == 0


def test_ovphysx_import_only_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> object:
        raise ImportError(f"no module {name}")

    monkeypatch.setattr(ovphysx, "import_module", unavailable)
    result = ovphysx.probe_ovphysx(import_only=True)

    assert result.status == "capability_error"
    assert result.mode == "import_only"
    assert result.error is not None
    assert result.error["capability"] == "python_imports"
    assert result.capabilities["physics_step"] is False
    json.dumps(result.to_dict(), allow_nan=False)


def test_ovphysx_canonical_dt_halving_recomputes_requested_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(
        Path(__file__).resolve().parents[1] / "configs" / "parity" / "gate0.json"
    )
    scenario = manifest.scenario("small_step")

    def unavailable(
        self: ovphysx.OVPhysXAdapter,
        hand: object,
        *,
        dt_override: float | None = None,
    ) -> None:
        del self, hand, dt_override
        raise AdapterCapabilityError(
            "ovphysx",
            "fixture",
            "runtime intentionally unavailable in this unit test",
        )

    monkeypatch.setattr(ovphysx.OVPhysXAdapter, "open", unavailable)
    result = ovphysx.OVPhysXAdapter(device="cpu").run_scenario(
        manifest.hands[0],
        scenario,
        dt_override=scenario.dt_s / 2.0,
    )

    assert result.status == "error"
    assert result.requested_steps == 500


def test_ovphysx_reads_cpu_only_drive_bindings_without_cuda_buffer() -> None:
    class CpuOnlyBinding:
        shape = (1, 22)

        def __init__(self, value: float) -> None:
            self.value = value
            self.read_buffer: object | None = None

        def read(self, output: object) -> None:
            # Model the pinned OVPhysX contract: CPU-only bindings reject a
            # Torch/CUDA-style buffer and accept a host NumPy array.
            if hasattr(output, "detach") or hasattr(output, "__cuda_array_interface__"):
                raise RuntimeError("CPU-only binding received a CUDA buffer")
            self.read_buffer = output
            output[...] = self.value  # type: ignore[index]

    binding = CpuOnlyBinding(3.25)
    values = ovphysx._read_cpu_float_binding(
        binding,
        name="DOF_STIFFNESS",
        expected_shape=(1, 22),
    )

    assert values == pytest.approx((3.25,) * 22)
    assert binding.read_buffer is not None
    assert type(binding.read_buffer).__module__.startswith("numpy")


def test_ovphysx_reads_gpu_target_binding_into_reused_warp_buffer() -> None:
    class FakeWarpBuffer:
        def __init__(self) -> None:
            self.value = 0.0
            self.fill_count = 0
            self.device = "cuda:0"

        def fill_(self, value: float) -> None:
            self.value = value
            self.fill_count += 1

    class GpuBinding:
        def __init__(self) -> None:
            self.read_buffer: object | None = None

        def read(self, output: object) -> None:
            if hasattr(output, "detach"):
                raise RuntimeError("GPU binding received a Torch tensor")
            self.read_buffer = output
            output.value = 0.05  # type: ignore[attr-defined]

    class FakeWarp:
        sync_devices: list[str] = []

        @classmethod
        def synchronize_device(cls, device: str) -> None:
            cls.sync_devices.append(device)

        @staticmethod
        def to_torch(buffer: FakeWarpBuffer) -> tuple[str, float]:
            return ("torch-view", buffer.value)

    binding = GpuBinding()
    buffer = FakeWarpBuffer()
    first = ovphysx._read_gpu_float_binding(binding, buffer=buffer, warp=FakeWarp)
    second = ovphysx._read_gpu_float_binding(binding, buffer=buffer, warp=FakeWarp)

    assert first == second == ("torch-view", 0.05)
    assert binding.read_buffer is buffer
    assert buffer.fill_count == 2
    assert FakeWarp.sync_devices == ["cuda:0"] * 4


class _OVScenarioHarness(ovphysx.OVPhysXAdapter):
    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.gravity_seen: tuple[float, float, float] | None = None

    def open(self, manifest: object, *, dt_override: float | None = None) -> None:
        assert dt_override is not None
        self.gravity_seen = self._open_gravity
        self._dt = dt_override
        self._joint_names = tuple(getattr(manifest, "joint_names"))
        self._frame_names = tuple(getattr(manifest, "distal_frame_names"))
        self._position_targets = {name: 0.0 for name in self._joint_names}
        self.applied_targets: list[dict[str, float]] = []
        self._step_index = 0
        self._lifecycle = AdapterLifecycle.READY

    def set_position_targets(self, targets: object) -> None:
        self._position_targets.update(dict(targets))

    def step(self) -> None:
        self.applied_targets.append(dict(self._position_targets))
        self._step_index += 1

    def sample(
        self,
        *,
        step: int,
        joint_names: tuple[str, ...] = (),
        frame_names: tuple[str, ...] = (),
    ) -> TraceSample:
        return TraceSample(
            step=step,
            time_s=step * self._dt,
            qpos=(0.0,) * len(joint_names),
            qvel=(0.0,) * len(joint_names),
            joint_positions={name: 0.0 for name in joint_names},
            frame_poses={
                name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                for name in frame_names
            },
            position_targets=dict(self._position_targets),
            contact_count=None,
        )

    def _provenance(self) -> dict[str, object]:
        return {"harness": True}

    def close(self) -> None:
        self._lifecycle = AdapterLifecycle.CLOSED


def test_ovphysx_scenario_schedule_covers_gravity_step_and_dt_halving() -> None:
    manifest = load_manifest(
        Path(__file__).resolve().parents[1] / "configs" / "parity" / "gate0.json"
    )
    hand = manifest.hands[0]
    for scenario in manifest.scenarios:
        adapter = _OVScenarioHarness()
        dt = scenario.dt_s / 2.0 if scenario.scenario_id == "small_step" else scenario.dt_s
        result = adapter.run_scenario(hand, scenario, dt_override=dt)

        expected_steps = round(scenario.duration_s / dt)
        assert result.completed, result.message
        assert result.requested_steps == expected_steps
        assert result.completed_steps == expected_steps
        assert len(result.samples) == expected_steps + 1
        assert result.samples[-1].time_s == pytest.approx(scenario.duration_s)
        assert adapter.gravity_seen == scenario.gravity_m_s2
        assert all(sample.contact_count is None for sample in result.samples)
        if scenario.scenario_id == "small_step":
            onset = round(scenario.step_start_s / dt)
            assert result.samples[onset - 1].position_targets[hand.joint_names[0]] == 0.0
            assert result.samples[onset].position_targets[hand.joint_names[0]] == pytest.approx(
                scenario.target_position_rad
            )
            assert adapter.applied_targets[onset - 1][hand.joint_names[0]] == 0.0
            assert adapter.applied_targets[onset][hand.joint_names[0]] == pytest.approx(
                scenario.target_position_rad
            )


class _FakeTensorValue:
    def __init__(self, value: object) -> None:
        self.value = value

    def detach(self) -> _FakeTensorValue:
        return self

    def cpu(self) -> _FakeTensorValue:
        return self

    def item(self) -> float:
        return float(self.value)

    def tolist(self) -> object:
        return self.value


class _FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = value

    def __getitem__(self, key: object) -> _FakeTensorValue:
        value = self.value
        indices = key if isinstance(key, tuple) else (key,)
        for index in indices:
            value = value[index]
        return _FakeTensorValue(value)


def test_ovphysx_sample_preserves_backend_order_and_maps_canonical_qwxyz() -> None:
    manifest = load_manifest(
        Path(__file__).resolve().parents[1] / "configs" / "parity" / "gate0.json"
    )
    hand = manifest.hands[0]
    backend_joints = tuple(reversed(hand.joint_names))
    backend_frames = ("root", *tuple(reversed(hand.distal_frame_names)))
    joint_values = [float(index) / 100.0 for index in range(22)]
    velocity_values = [float(index) / 1000.0 for index in range(22)]
    body_values = [
        [float(index), 0.0, 0.0, 0.1, 0.2, 0.3, 0.9]
        for index in range(len(backend_frames))
    ]
    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._lifecycle = AdapterLifecycle.READY
    adapter._dt = 0.002
    adapter._joint_names = hand.joint_names
    adapter._frame_names = hand.distal_frame_names
    adapter._backend_joint_names = backend_joints
    adapter._backend_frame_names = backend_frames
    adapter._joint_mapping = tuple(
        {
            "canonical_id": name,
            "backend": "ovphysx",
            "scope": "left",
            "backend_name": name,
            "index": backend_joints.index(name),
            "sign": 1.0,
            "offset": 0.0,
            "unit": "rad",
        }
        for name in hand.joint_names
    )
    adapter._frame_mapping = tuple(
        {
            "canonical_id": name,
            "backend": "ovphysx",
            "scope": "left",
            "backend_name": name,
            "index": backend_frames.index(name),
            "sign": 1.0,
            "offset": 0.0,
            "unit": "xyz_m_qwxyz",
        }
        for name in hand.distal_frame_names
    )
    adapter._position_targets = {name: 0.0 for name in hand.joint_names}
    adapter._articulation = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=_FakeTensor([joint_values])),
            joint_vel=SimpleNamespace(torch=_FakeTensor([velocity_values])),
            body_link_pose_w=SimpleNamespace(torch=_FakeTensor([body_values])),
        )
    )

    sample = adapter.sample(
        step=0,
        joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
    )

    assert sample.qpos == tuple(joint_values)
    assert sample.qvel == tuple(velocity_values)
    first_joint = hand.joint_names[0]
    assert sample.joint_positions[first_joint] == joint_values[backend_joints.index(first_joint)]
    first_frame = hand.distal_frame_names[0]
    backend_index = backend_frames.index(first_frame)
    assert sample.frame_poses[first_frame] == (
        float(backend_index),
        0.0,
        0.0,
        0.9,
        0.1,
        0.2,
        0.3,
    )


def test_ovphysx_worker_derives_one_halved_case_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "probe_ovphysx_runtime.py"
    spec = importlib.util.spec_from_file_location("ovphysx_runtime_worker", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    approved = tmp_path / "approved"
    asset_root = approved / "assets"
    (asset_root / "wave_01").mkdir(parents=True)
    project = approved / "project"
    project.mkdir()
    (approved / "results").mkdir()
    manifest_path = project / "gate0.json"
    manifest_path.write_text(
        (Path(__file__).parents[1] / "configs" / "parity" / "gate0.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator.value == "ovphysx"
        and item.hand.value == "left"
        and item.scenario_id == "small_step"
        and item.timestep_variant.value == "halved"
        and item.repeat_index == 1
    )
    calls: list[tuple[object, object, float]] = []

    class FakeWorkerAdapter:
        def __init__(self, *, device: str, asset_root: Path) -> None:
            assert device == "cuda:0"
            assert asset_root == (approved / "assets").resolve()

        def run_scenario(
            self,
            hand: object,
            scenario: object,
            *,
            dt_override: float | None = None,
        ) -> AdapterRunResult:
            assert dt_override is not None
            calls.append((hand, scenario, dt_override))
            return _OVScenarioHarness().run_scenario(
                hand, scenario, dt_override=dt_override
            )

    monkeypatch.setattr(module, "EXPECTED_APPROVED_ROOT", approved)
    monkeypatch.setattr(module, "OVPhysXAdapter", FakeWorkerAdapter)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    output = approved / "results" / "case.worker.result.json"
    exit_code = module.main(
        [
            "--approved-root",
            str(approved),
            "--asset-root",
            str(asset_root),
            "--manifest",
            str(manifest_path),
            "--case-id",
            case.case_id,
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--session-id",
            "unit-session",
            "--source-revision",
            "a" * 40,
            "--asset-tree-sha256",
            manifest.provenance.canonical_lf_asset_tree_sha256,
        ]
    )

    assert exit_code == 0, capsys.readouterr()
    assert len(calls) == 1
    assert calls[0][2] == pytest.approx(0.001)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["requested_steps"] == 500
    assert len(payload["samples"]) == 501
    assert payload["provenance"]["manifest_sha256"]
    assert payload["provenance"]["session_id"] == "unit-session"
    assert payload["provenance"]["source_revision"] == "a" * 40
    assert payload["provenance"]["asset_tree_sha256"] == (
        manifest.provenance.canonical_lf_asset_tree_sha256
    )
    assert payload["provenance"]["mapping_schema_version"] == 1


def test_ovphysx_import_only_success_does_not_claim_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "isaaclab": SimpleNamespace(),
        "isaaclab.sim": SimpleNamespace(
            SimulationCfg=object,
            build_simulation_context=lambda config: config,
        ),
        "isaaclab_ovphysx.physics": SimpleNamespace(OvPhysxCfg=object),
    }
    monkeypatch.setattr(ovphysx, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        ovphysx,
        "_package_versions",
        lambda: {
            "isaaclab": "3.0.0b2",
            "isaaclab-ovphysx": "3.0.2",
            "ovphysx": "0.4.13",
            "torch": "fixture",
            "usd-core": "fixture",
        },
    )

    result = ovphysx.probe_ovphysx(import_only=True, device="cuda:0")

    assert result.available
    assert result.capabilities["build_simulation_context_symbol"] is True
    assert result.capabilities["pinned_ovphysx_wheel_version"] is True
    assert result.capabilities["physics_step"] is False
    assert "not exercised" in result.message


def test_ovphysx_import_only_rejects_wrong_wheel_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "isaaclab": SimpleNamespace(),
        "isaaclab.sim": SimpleNamespace(
            SimulationCfg=object,
            build_simulation_context=lambda config: config,
        ),
        "isaaclab_ovphysx.physics": SimpleNamespace(OvPhysxCfg=object),
    }
    monkeypatch.setattr(ovphysx, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        ovphysx,
        "_package_versions",
        lambda: {
            "isaaclab": "3.0.0b2",
            "isaaclab-ovphysx": "3.0.2",
            "ovphysx": "0.4.12",
            "torch": "fixture",
            "usd-core": "fixture",
        },
    )

    result = ovphysx.probe_ovphysx(import_only=True)

    assert result.status == "capability_error"
    assert result.error is not None
    assert result.error["capability"] == "ovphysx_wheel_version"
    assert result.capabilities["pinned_ovphysx_wheel_version"] is False


def test_ovphysx_import_only_requires_official_builder_not_context_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "isaaclab": SimpleNamespace(),
        "isaaclab.sim": SimpleNamespace(
            SimulationCfg=object,
            SimulationContext=object,
        ),
        "isaaclab_ovphysx.physics": SimpleNamespace(OvPhysxCfg=object),
    }
    monkeypatch.setattr(ovphysx, "import_module", modules.__getitem__)

    result = ovphysx.probe_ovphysx(import_only=True)

    assert result.status == "capability_error"
    assert result.error is not None
    assert result.error["capability"] == "pinned_api_surface"
    assert "build_simulation_context_symbol" in result.error["message"]


def test_ovphysx_rejects_renderer_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def should_not_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(ovphysx, "import_module", should_not_import)
    result = ovphysx.probe_ovphysx(
        import_only=True,
        manifest={"execution": {"renderer": True}},
    )

    assert result.status == "capability_error"
    assert result.error is not None
    assert result.error["capability"] == "kitless_headless_only"


def test_resolved_usd_probe_reports_static_availability_without_claiming_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "resolved.usd"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        ovphysx,
        "_load_import_capabilities",
        lambda: ({}, {"isaaclab_import": True}),
    )
    monkeypatch.setattr(
        ovphysx,
        "_package_versions",
        lambda: {
            "isaaclab": "3.0.0b2",
            "isaaclab-ovphysx": "3.0.2",
            "ovphysx": "0.4.13",
            "torch": "fixture",
            "usd-core": "fixture",
        },
    )
    monkeypatch.setattr(
        ovphysx,
        "_inspect_usd",
        lambda path: {
            "source_path": str(path),
            "stage_load_policy": "load_all",
            "prim_count": 100,
            "revolute_joint_count": 22,
            "angular_drive_count": 22,
            "angular_drive_records": [
                {
                    "joint_name": f"joint_{index}",
                    "type": "force",
                    "stiffness": 1.0,
                    "damping": 0.1,
                    "max_force": 2.0,
                    "target_position": 0.0,
                }
                for index in range(22)
            ],
            "physx_velocity_joint_count": 22,
            "physx_velocity_records": [
                {"joint_name": f"joint_{index}", "max_joint_velocity": 5.0}
                for index in range(22)
            ],
            "distal_frame_count": 5,
            "articulation_root_paths": ["/hand/root_joint"],
        },
    )

    result = ovphysx.probe_ovphysx(
        import_only=False,
        resolved_usd=source,
        asset_root=tmp_path,
    )

    assert result.status == "available"
    assert result.capabilities["resolved_usd_open"] is True
    assert result.capabilities["physics_step"] is False
    assert result.error is None
    assert "process-isolated runtime probe" in result.message


def test_resolved_usd_probe_rejects_incomplete_composed_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "resolved.usd"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        ovphysx,
        "_load_import_capabilities",
        lambda: ({}, {"isaaclab_import": True}),
    )
    monkeypatch.setattr(
        ovphysx,
        "_package_versions",
        lambda: {"ovphysx": "0.4.13"},
    )
    monkeypatch.setattr(
        ovphysx,
        "_inspect_usd",
        lambda path: {
            "source_path": str(path),
            "stage_load_policy": "load_all",
            "prim_count": 100,
            "revolute_joint_count": 21,
            "angular_drive_count": 21,
            "physx_velocity_joint_count": 21,
            "distal_frame_count": 5,
            "articulation_root_paths": ["/hand/root_joint"],
        },
    )

    result = ovphysx.probe_ovphysx(
        import_only=False,
        resolved_usd=source,
        asset_root=tmp_path,
    )

    assert result.status == "capability_error"
    assert result.capabilities["resolved_usd_open"] is True
    assert result.capabilities["canonical_joint_count"] is False
    assert result.error is not None
    assert result.error["capability"] == "resolved_usd_schema"


def test_canonical_manifest_requires_and_resolves_hand_selector(tmp_path: Path) -> None:
    left = tmp_path / "left.usd"
    right = tmp_path / "right.usd"
    left.touch()
    right.touch()
    manifest = {
        "hands": [
            {"side": "left", "model_paths": {"ovphysx": left.name}},
            {"side": "right", "model_paths": {"ovphysx": right.name}},
        ]
    }

    with pytest.raises(ValueError, match="hand selector"):
        ovphysx._resolve_usd_path(
            manifest, asset_root=tmp_path, resolved_usd=None
        )
    assert ovphysx._resolve_usd_path(
        manifest, asset_root=tmp_path, resolved_usd=None, hand="right"
    ) == right.resolve()


def test_probe_cli_writes_structured_json_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "probe_ovphysx.py"
    spec = importlib.util.spec_from_file_location("probe_ovphysx_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failure = AdapterProbeResult(
        backend="ovphysx",
        mode="import_only",
        status="capability_error",
        message="fixture unavailable",
        error={"type": "AdapterCapabilityError", "capability": "fixture"},
    )
    monkeypatch.setattr(module, "probe_ovphysx", lambda **kwargs: failure)
    destination = tmp_path / "probe.json"

    exit_code = module.main(["--import-only", "--output", str(destination)])
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(destination.read_text(encoding="utf-8"))

    assert exit_code == 3
    assert stdout == written
    assert written["status"] == "capability_error"
