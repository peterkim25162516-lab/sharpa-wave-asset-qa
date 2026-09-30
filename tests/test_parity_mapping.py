from __future__ import annotations

import json

import pytest

from wave_asset_qa.parity.mapping import (
    FrameMapping,
    JointMapping,
    MappingValidationError,
    validate_mapping,
)


BACKENDS = ("mujoco", "ovphysx")
JOINT_IDS = tuple(
    f"{side}/joint_{index:02d}" for side in ("left", "right") for index in range(22)
)
FRAME_IDS = tuple(
    f"{side}/{finger}_tip"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
)


def _joints() -> list[JointMapping]:
    return [
        JointMapping(
            canonical_id=canonical_id,
            backend=backend,
            scope=scope,
            backend_name=f"{backend}_{canonical_id.replace('/', '_')}",
            index=index,
            sign=-1.0 if canonical_id.endswith("_03") else 1.0,
            offset=0.125 if canonical_id.endswith("_03") else 0.0,
            unit="rad",
        )
        for backend in BACKENDS
        for scope in ("left", "right")
        for index, canonical_id in enumerate(
            item for item in JOINT_IDS if item.startswith(f"{scope}/")
        )
    ]


def _frames() -> list[FrameMapping]:
    return [
        FrameMapping(
            canonical_id=canonical_id,
            backend=backend,
            scope=scope,
            backend_name=f"{backend}_{canonical_id.replace('/', '_')}",
            index=index,
            sign=1.0,
            offset=0.0,
            unit="xyz_m_qwxyz",
        )
        for backend in BACKENDS
        for scope in ("left", "right")
        for index, canonical_id in enumerate(
            item for item in FRAME_IDS if item.startswith(f"{scope}/")
        )
    ]


def test_gate0_mapping_validates_44_joints_and_10_frames_per_backend() -> None:
    summary = validate_mapping(
        _joints(),
        _frames(),
        expected_joint_count=44,
        expected_frame_count=10,
        expected_joint_ids=JOINT_IDS,
        expected_frame_ids=FRAME_IDS,
        expected_backends=BACKENDS,
    )

    assert summary["joint_count"] == 44
    assert summary["frame_count"] == 10
    assert summary["by_backend"]["mujoco"]["joint_count"] == 44
    assert summary["by_backend"]["ovphysx"]["frame_count"] == 10
    assert summary["by_backend"]["mujoco"]["by_scope"]["left"] == {
        "joint_count": 22,
        "frame_count": 5,
    }
    json.dumps(summary)


def test_mapping_records_round_trip_every_explicit_field() -> None:
    joint = JointMapping(
        canonical_id="left/joint_03",
        backend="mujoco",
        scope="left",
        backend_name="left_j3",
        index=3,
        sign=-1.0,
        offset=0.125,
        unit="rad",
    )
    frame = FrameMapping(
        canonical_id="left/index_tip",
        backend="ovphysx",
        scope="left",
        backend_name="left_index_DP",
        index=1,
        sign=1.0,
        offset=0.0,
        unit="xyz_m_qwxyz",
    )

    assert JointMapping.from_dict(joint.to_dict()) == joint
    assert FrameMapping.from_dict(frame.to_dict()) == frame
    assert set(joint.to_dict()) == {
        "canonical_id",
        "backend",
        "scope",
        "backend_name",
        "index",
        "sign",
        "offset",
        "unit",
    }


@pytest.mark.parametrize("conflict", ["canonical", "name", "index"])
def test_mapping_rejects_duplicates_and_index_conflicts(conflict: str) -> None:
    first = JointMapping("joint/a", "mujoco", "left", "a", 0, 1.0, 0.0, "rad")
    values = {
        "canonical": JointMapping("joint/a", "mujoco", "left", "b", 1, 1.0, 0.0, "rad"),
        "name": JointMapping("joint/b", "mujoco", "left", "a", 1, 1.0, 0.0, "rad"),
        "index": JointMapping("joint/b", "mujoco", "left", "b", 0, 1.0, 0.0, "rad"),
    }

    with pytest.raises(MappingValidationError, match="duplicate|conflict"):
        validate_mapping([first, values[conflict]], [])


def test_backend_indices_restart_independently_for_each_hand_scope() -> None:
    left = JointMapping(
        "left/joint_00", "mujoco", "left", "left_joint_00", 0, 1.0, 0.0, "rad"
    )
    right = JointMapping(
        "right/joint_00", "mujoco", "right", "right_joint_00", 0, 1.0, 0.0, "rad"
    )

    summary = validate_mapping(
        [left, right],
        [],
        expected_joint_count=2,
        expected_backends=("mujoco",),
    )

    assert summary["by_backend"]["mujoco"]["joint_count"] == 2
    assert set(summary["by_backend"]["mujoco"]["by_scope"]) == {"left", "right"}


def test_mapping_rejects_missing_expected_identifier() -> None:
    joints = _joints()
    joints = [
        entry
        for entry in joints
        if not (entry.backend == "ovphysx" and entry.canonical_id == JOINT_IDS[-1])
    ]

    with pytest.raises(MappingValidationError, match="missing.*right/joint_21"):
        validate_mapping(
            joints,
            _frames(),
            expected_joint_ids=JOINT_IDS,
            expected_frame_ids=FRAME_IDS,
            expected_backends=BACKENDS,
        )


def test_mapping_rejects_invalid_conversion_and_strict_record_shape() -> None:
    with pytest.raises(MappingValidationError, match="sign"):
        JointMapping("joint/a", "mujoco", "left", "a", 0, 0.0, 0.0, "rad")

    record = JointMapping(
        "joint/a", "mujoco", "left", "a", 0, 1.0, 0.0, "rad"
    ).to_dict()
    record["extra"] = True
    with pytest.raises(MappingValidationError, match="unknown fields"):
        JointMapping.from_dict(record)
