from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from wave_asset_qa.adapters.base import AdapterLifecycle, TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.adapters.ovphysx import OVPhysXAdapter
from wave_asset_qa.parity.contracts import ScenarioKind, WaveformScenarioSpec
from wave_asset_qa.parity.scenarios import (
    canonical_position_targets,
    canonical_target_sequence_sha256,
)


def _scenario(kind: ScenarioKind) -> WaveformScenarioSpec:
    if kind is ScenarioKind.OFFSET_SINE:
        duration_s = 1.0
        start_hz = end_hz = 1.0
    else:
        duration_s = 2.0
        start_hz = 0.5
        end_hz = 1.5
    return WaveformScenarioSpec(
        scenario_id=kind.value,
        kind=kind,
        duration_s=duration_s,
        dt_s=0.01,
        gravity_m_s2=(0.0, 0.0, 0.0),
        initial_position_rad=0.0,
        command_offset_rad=0.025,
        command_amplitude_rad=0.025,
        frequency_start_hz=start_hz,
        frequency_end_hz=end_hz,
        expect_no_contacts=True,
    )


@pytest.fixture
def waveform_mjcf(tmp_path: Path) -> Path:
    source = tmp_path / "waveform.xml"
    source.write_text(
        """
        <mujoco model="waveform_fixed">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="base">
              <body name="tip" pos="0 0 0.1">
                <joint name="hinge" axis="0 1 0" range="-1 1"/>
                <geom type="capsule" size="0.01 0.05" mass="0.1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="hinge_ctrl" joint="hinge" kp="5" kv="0.7"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize(
    "kind",
    [ScenarioKind.OFFSET_SINE, ScenarioKind.OFFSET_LINEAR_CHIRP],
)
def test_mujoco_waveform_uses_canonical_schedule_and_terminal_digest(
    waveform_mjcf: Path,
    kind: ScenarioKind,
) -> None:
    pytest.importorskip("mujoco")
    scenario = _scenario(kind)
    manifest = {
        "mjcf_path": str(waveform_mjcf),
        "fixed_base": True,
        "control_mode": "position",
        "expected_joint_names": ["hinge"],
        "expected_frame_names": ["tip"],
    }

    result = MuJoCoAdapter().run_scenario(manifest, scenario)

    assert result.completed, result.message
    assert result.completed_steps == scenario.steps
    assert len(result.samples) == scenario.steps + 1
    for step_index, sample in enumerate(result.samples):
        expected = canonical_position_targets(
            scenario,
            ("hinge",),
            step_index=step_index,
        )
        assert sample.position_targets == expected
    # q[0] controls [t0,t1), so state[1] is still the zero-target response.
    assert result.samples[1].joint_positions["hinge"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    provenance = result.provenance
    assert provenance["target_sequence_digest_schema_version"] == 1
    assert provenance["target_sequence_digest_projection"] == (
        "ieee754_binary32_roundtrip"
    )
    assert provenance["target_sequence_canonical_joint_names"] == ["hinge"]
    assert provenance["scheduled_target_sequence_count"] == scenario.steps + 1
    assert provenance["scheduled_target_sequence_sha256"] == (
        canonical_target_sequence_sha256(scenario, ("hinge",))
    )


def _binary32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


class _OVWaveformHarness(OVPhysXAdapter):
    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.applied_targets: list[dict[str, float]] = []

    def open(self, manifest: object, *, dt_override: float | None = None) -> None:
        assert dt_override is not None
        self._dt = dt_override
        self._joint_names = tuple(getattr(manifest, "joint_names"))
        self._frame_names = tuple(getattr(manifest, "distal_frame_names"))
        self._backend_joint_names = tuple(reversed(self._joint_names))
        self._joint_mapping = tuple(
            {
                "canonical_id": name,
                "backend_name": name,
                "index": self._backend_joint_names.index(name),
                "sign": 1.0,
                "offset": 0.0,
            }
            for name in self._joint_names
        )
        self._position_targets = {name: 0.0 for name in self._joint_names}
        self._step_index = 0
        self._lifecycle = AdapterLifecycle.READY

    def set_position_targets(self, targets: object) -> None:
        self._position_targets.update(dict(targets))
        digest = self._immediate_target_readback_sequence_digest
        if digest is not None:
            digest.append(
                {
                    name: _binary32(self._position_targets[name])
                    for name in self._joint_names
                }
            )

    def step(self) -> None:
        projected = {
            name: _binary32(self._position_targets[name])
            for name in self._joint_names
        }
        digest = self._pre_step_applied_target_readback_sequence_digest
        if digest is not None:
            digest.append(projected)
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


@pytest.mark.parametrize(
    "kind",
    [ScenarioKind.OFFSET_SINE, ScenarioKind.OFFSET_LINEAR_CHIRP],
)
def test_ovphysx_waveform_hashes_full_and_applied_prefix_sequences(
    kind: ScenarioKind,
) -> None:
    scenario = _scenario(kind)
    hand = SimpleNamespace(
        joint_names=("joint_b", "joint_a"),
        distal_frame_names=("tip",),
    )
    adapter = _OVWaveformHarness()

    result = adapter.run_scenario(hand, scenario)

    assert result.completed, result.message
    provenance = result.provenance
    full_hash = canonical_target_sequence_sha256(
        scenario,
        hand.joint_names,
        include_terminal=True,
    )
    prefix_hash = canonical_target_sequence_sha256(
        scenario,
        hand.joint_names,
        include_terminal=False,
    )
    assert provenance["target_sequence_canonical_joint_names"] == list(
        hand.joint_names
    )
    for field_name in (
        "scheduled_target_sequence_sha256",
        "requested_target_sequence_sha256",
        "immediate_target_readback_sequence_sha256",
    ):
        assert provenance[field_name] == full_hash
    assert provenance["pre_step_applied_target_readback_sequence_sha256"] == (
        prefix_hash
    )
    for field_name in (
        "scheduled_target_sequence_count",
        "requested_target_sequence_count",
        "immediate_target_readback_sequence_count",
    ):
        assert provenance[field_name] == scenario.steps + 1
    assert provenance["pre_step_applied_target_readback_sequence_count"] == (
        scenario.steps
    )
    assert len(adapter.applied_targets) == scenario.steps
    for step_index, applied in enumerate(adapter.applied_targets):
        assert applied == canonical_position_targets(
            scenario,
            hand.joint_names,
            step_index=step_index,
        )


def test_ovphysx_readback_projection_uses_mapping_not_backend_order() -> None:
    adapter = OVPhysXAdapter(device="cpu")
    adapter._joint_names = ("joint_a", "joint_b")
    adapter._backend_joint_names = ("joint_b", "joint_a")
    adapter._joint_mapping = (
        {
            "canonical_id": "joint_a",
            "backend_name": "joint_a",
            "index": 1,
            "sign": -1.0,
            "offset": 0.1,
        },
        {
            "canonical_id": "joint_b",
            "backend_name": "joint_b",
            "index": 0,
            "sign": 1.0,
            "offset": 0.0,
        },
    )

    projected = adapter._canonical_position_target_readback((0.02, 0.03))

    assert list(projected) == ["joint_a", "joint_b"]
    assert projected == pytest.approx({"joint_a": 0.07, "joint_b": 0.02})
