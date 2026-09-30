import pytest

from wave_asset_qa.fk_compare import compare_forward_kinematics
from wave_asset_qa.models import FrameSpec, JointSpec, ParsedModel


def _model(model_format: str, *, offset: float = 0.0) -> ParsedModel:
    root = FrameSpec(name="right_hand_C_MC")
    child = FrameSpec(
        name="right_index_DP",
        parent="right_hand_C_MC",
        origin_xyz=(1.0 + offset, 0.0, 0.0),
        joint_name="right_index_PIP",
        metadata={"joint_names": ("right_index_PIP",)} if model_format == "mjcf" else {},
    )
    joint = JointSpec(
        name="right_index_PIP",
        joint_type="revolute",
        parent=root.name,
        child=child.name,
        axis=(0.0, 0.0, 1.0),
        lower=-1.0,
        upper=1.0,
        origin_xyz=child.origin_xyz,
    )
    return ParsedModel(
        format=model_format,
        name=model_format,
        source_path=__file__,
        links=(root.name, child.name),
        joints=(joint,),
        frames=(root, child),
    )


def test_identical_trees_have_near_zero_fk_error() -> None:
    result = compare_forward_kinematics(_model("urdf"), _model("mjcf"), samples=20, seed=7)

    assert result.passed
    assert result.sampled_joint_count == 1
    assert result.position_max_mm is not None and result.position_max_mm < 1e-9
    assert result.orientation_max_deg is not None and result.orientation_max_deg < 1e-9


def test_fixed_link_offset_is_detected() -> None:
    result = compare_forward_kinematics(
        _model("urdf"),
        _model("mjcf", offset=0.01),
        samples=10,
    )

    assert result.passed
    assert result.position_median_mm is not None
    assert result.position_median_mm == pytest.approx(10.0)


def test_joint_set_mismatch_is_a_structured_skip() -> None:
    mjcf = _model("mjcf")
    mjcf = ParsedModel(
        format=mjcf.format,
        name=mjcf.name,
        source_path=mjcf.source_path,
        links=mjcf.links,
        joints=(),
        frames=mjcf.frames,
    )
    result = compare_forward_kinematics(_model("urdf"), mjcf)

    assert result.status == "skip"
    assert "joint sets differ" in result.message
