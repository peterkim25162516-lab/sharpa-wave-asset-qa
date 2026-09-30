import numpy as np
import pytest

from wave_asset_qa.kinematics import (
    JointMotion,
    KinematicEdge,
    axis_angle_to_matrix,
    compose_transforms,
    edge_transform,
    forward_kinematics,
    invert_transform,
    joint_motion_transform,
    make_transform,
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rpy,
    pose_transform,
    quat_to_matrix,
    relative_transform,
    rotation_geodesic_error,
    rpy_to_matrix,
    skew,
    transform_points,
    transform_vectors,
)


ATOL = 1e-9


@pytest.mark.parametrize(
    "rotation_vector",
    [
        np.zeros(3),
        np.array([0.2, -0.4, 0.1]),
        np.array([np.pi, 0.0, 0.0]),
        np.array([0.0, -np.pi, 0.0]),
    ],
)
def test_axis_angle_round_trip(rotation_vector):
    rotation = axis_angle_to_matrix(rotation_vector)
    axis, angle = matrix_to_axis_angle(rotation)
    reconstructed = axis_angle_to_matrix(axis, angle)
    np.testing.assert_allclose(reconstructed, rotation, atol=ATOL)


def test_rodrigues_matches_expected_quarter_turn():
    rotation = axis_angle_to_matrix((0, 0, 1), np.pi / 2)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(skew((1, 2, 3)) @ np.array([4.0, 5.0, 6.0]), np.cross([1, 2, 3], [4, 5, 6]))


@pytest.mark.parametrize(
    "rpy",
    [
        np.zeros(3),
        np.array([0.3, -0.4, 1.2]),
        np.array([-2.0, 0.8, -1.0]),
    ],
)
def test_rpy_round_trip_away_from_gimbal_lock(rpy):
    rotation = rpy_to_matrix(rpy)
    recovered = matrix_to_rpy(rotation)
    np.testing.assert_allclose(rpy_to_matrix(recovered), rotation, atol=ATOL)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_rpy_gimbal_lock_returns_equivalent_rotation(pitch):
    rotation = rpy_to_matrix((0.7, pitch, -1.1))
    recovered = matrix_to_rpy(rotation)
    assert recovered[0] == 0.0
    np.testing.assert_allclose(rpy_to_matrix(recovered), rotation, atol=ATOL)


@pytest.mark.parametrize(
    "quaternion",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.7, 0.2, -0.1, 0.5]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_quaternion_round_trip_and_canonical_sign(quaternion):
    rotation = quat_to_matrix(quaternion)
    recovered = matrix_to_quat(rotation)
    assert recovered[0] >= 0.0
    np.testing.assert_allclose(quat_to_matrix(recovered), rotation, atol=ATOL)


def test_xyzw_quaternion_order_and_pose_constructor():
    xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    transform = pose_transform(xyz=(1, 2, 3), quat=xyzw, quat_order="xyzw")
    np.testing.assert_allclose(transform[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(transform[:3, :3], axis_angle_to_matrix((0, 0, 1), np.pi / 2), atol=ATOL)
    recovered = matrix_to_quat(transform[:3, :3], order="xyzw")
    np.testing.assert_allclose(quat_to_matrix(recovered, order="xyzw"), transform[:3, :3], atol=ATOL)


def test_rotation_geodesic_error_is_symmetric_and_handles_pi():
    first = rpy_to_matrix((0.1, -0.2, 0.3))
    delta = axis_angle_to_matrix((1, 2, 3), np.pi)
    second = first @ delta
    assert rotation_geodesic_error(first, first) == pytest.approx(0.0, abs=ATOL)
    assert rotation_geodesic_error(first, second) == pytest.approx(np.pi, abs=ATOL)
    assert rotation_geodesic_error(first, second) == pytest.approx(rotation_geodesic_error(second, first))


def test_transform_compose_inverse_relative_points_and_vectors():
    first = pose_transform(xyz=(1, 0, 0), rpy=(0, 0, np.pi / 2))
    second = pose_transform(xyz=(0, 2, 0), rpy=(np.pi / 2, 0, 0))
    combined = compose_transforms(first, second)

    np.testing.assert_allclose(invert_transform(combined) @ combined, np.eye(4), atol=ATOL)
    np.testing.assert_allclose(relative_transform(first, combined), second, atol=ATOL)

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    transformed = transform_points(first, points)
    np.testing.assert_allclose(transformed, [[1, 0, 0], [1, 1, 0]], atol=ATOL)
    np.testing.assert_allclose(transform_vectors(first, points[1]), [0, 1, 0], atol=ATOL)


def test_pose_transform_rejects_two_rotation_representations():
    with pytest.raises(ValueError, match="either rpy or quat"):
        pose_transform(rpy=(0, 0, 0), quat=(1, 0, 0, 0))


def test_revolute_motion_about_mjcf_style_local_pivot():
    motion = JointMotion("hinge", "hinge", axis=(0, 0, 2), pivot=(1, 0, 0))
    transform = joint_motion_transform(motion, np.pi / 2)
    # The pivot is invariant and a point one metre to its right swings upward.
    np.testing.assert_allclose(transform_points(transform, [1, 0, 0]), [1, 0, 0], atol=ATOL)
    np.testing.assert_allclose(transform_points(transform, [2, 0, 0]), [1, 1, 0], atol=ATOL)


def test_prismatic_and_fixed_motion():
    slide = JointMotion("slide", "slide", axis=(2, 0, 0))
    np.testing.assert_allclose(joint_motion_transform(slide, 0.25), make_transform(translation=(0.25, 0, 0)))
    fixed = JointMotion("mount", "fixed", axis=(0, 0, 0))
    np.testing.assert_allclose(joint_motion_transform(fixed, 123), np.eye(4))


def test_edge_applies_origin_then_multiple_motions_in_declaration_order():
    edge = KinematicEdge(
        "world",
        "tool",
        origin=make_transform(translation=(1, 0, 0)),
        motions=(
            JointMotion("yaw", "revolute", axis=(0, 0, 1)),
            JointMotion("extension", "prismatic", axis=(1, 0, 0)),
        ),
    )
    pose = edge_transform(edge, {"yaw": np.pi / 2, "extension": 2.0})
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 0], atol=ATOL)


def test_forward_kinematics_handles_unordered_branching_tree_and_root_pose():
    shoulder = KinematicEdge(
        "base",
        "upper",
        origin=make_transform(translation=(1, 0, 0)),
        motions=(JointMotion("shoulder", "revolute", axis=(0, 0, 1)),),
    )
    elbow = KinematicEdge(
        "upper",
        "tip",
        origin=make_transform(translation=(1, 0, 0)),
        motions=(JointMotion("extension", "prismatic", axis=(1, 0, 0)),),
    )
    camera = KinematicEdge("base", "camera", origin=make_transform(translation=(0, 0, 1)))
    world_base = make_transform(translation=(10, 0, 0))

    poses = forward_kinematics(
        [elbow, camera, shoulder],
        {"shoulder": np.pi / 2, "extension": 0.5},
        root_transform=world_base,
    )
    assert set(poses) == {"base", "upper", "tip", "camera"}
    np.testing.assert_allclose(poses["upper"][:3, 3], [11, 0, 0], atol=ATOL)
    np.testing.assert_allclose(poses["tip"][:3, 3], [11, 1.5, 0], atol=ATOL)
    np.testing.assert_allclose(poses["camera"][:3, 3], [10, 0, 1], atol=ATOL)


def test_forward_kinematics_defaults_missing_positions_to_zero():
    edge = KinematicEdge(
        "root",
        "child",
        origin=make_transform(translation=(1, 2, 3)),
        motions=(JointMotion("q", "revolute", axis=(0, 0, 1)),),
    )
    poses = forward_kinematics([edge])
    np.testing.assert_allclose(poses["child"], edge.origin)


@pytest.mark.parametrize(
    "edges, match",
    [
        (
            [KinematicEdge("root", "child"), KinematicEdge("other", "child")],
            "exactly one parent",
        ),
        (
            [KinematicEdge("a", "b"), KinematicEdge("b", "a")],
            "infer one root",
        ),
        (
            [KinematicEdge("root", "child"), KinematicEdge("orphan", "leaf")],
            "infer one root",
        ),
    ],
)
def test_forward_kinematics_rejects_invalid_graphs(edges, match):
    with pytest.raises(ValueError, match=match):
        forward_kinematics(edges)


def test_forward_kinematics_rejects_unknown_joint_name_by_default():
    edge = KinematicEdge(
        "root",
        "child",
        motions=(JointMotion("expected", "continuous", axis=(0, 1, 0)),),
    )
    with pytest.raises(KeyError, match="typo"):
        forward_kinematics([edge], {"typo": 0.1})
    poses = forward_kinematics([edge], {"typo": 0.1}, strict_positions=False)
    np.testing.assert_allclose(poses["child"], np.eye(4))


def test_forward_kinematics_rejects_duplicate_joint_names_across_edges():
    edges = [
        KinematicEdge("root", "left", motions=(JointMotion("duplicate", "hinge"),)),
        KinematicEdge("root", "right", motions=(JointMotion("duplicate", "slide"),)),
    ]
    with pytest.raises(ValueError, match="globally unique"):
        forward_kinematics(edges)


@pytest.mark.parametrize(
    "factory, match",
    [
        (lambda: axis_angle_to_matrix((0, 0, 0), 1.0), "non-zero"),
        (lambda: quat_to_matrix((0, 0, 0, 0)), "non-zero"),
        (lambda: JointMotion("q", "ball"), "unsupported"),
        (lambda: make_transform(np.diag([1, 1, -1])), "determinant"),
    ],
)
def test_invalid_inputs_raise_clear_errors(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
