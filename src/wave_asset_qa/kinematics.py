"""Small, NumPy-only rigid-body kinematics utilities.

Conventions
-----------
* Vectors are column vectors conceptually, while arrays of points are stored with
  the coordinate in the final axis.
* Rotations are active, right-handed rotations.
* Homogeneous transforms map coordinates from a child/local frame into its
  parent/world frame: ``p_parent = T_parent_child @ p_child``.
* Roll-pitch-yaw uses URDF's fixed-axis XYZ convention, equivalently
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
* Quaternions use ``wxyz`` ordering unless ``order="xyzw"`` is requested.

The tree representation deliberately separates a body's fixed reference pose
(``KinematicEdge.origin``) from its one or more joint motions.  A URDF joint is
normally one edge with one motion and a zero pivot.  An MJCF body can use its
``pos``/``quat`` as the edge origin, retain the joint's body-local ``pos`` as the
motion pivot, and keep multiple joints on the same edge in declaration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_EPS = 1e-12
_ROTATION_ATOL = 1e-7


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _rotation(value: ArrayLike, name: str = "rotation") -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(result.T @ result, np.eye(3), atol=_ROTATION_ATOL):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=_ROTATION_ATOL):
        raise ValueError(f"{name} must have determinant +1")
    return result


def _transform(value: ArrayLike, name: str = "transform") -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=_EPS):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    _rotation(result[:3, :3], f"{name} rotation")
    return result


def normalize(vector: ArrayLike, *, name: str = "vector") -> FloatArray:
    """Return a unit-length copy of a 3-vector.

    Raises ``ValueError`` instead of silently producing NaNs for a zero vector.
    """

    result = _vector(vector, 3, name)
    length = float(np.linalg.norm(result))
    if length <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return result / length


def skew(vector: ArrayLike) -> FloatArray:
    """Return the 3x3 cross-product matrix of ``vector``."""

    x, y, z = _vector(vector, 3, "vector")
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def axis_angle_to_matrix(axis: ArrayLike, angle: float | None = None) -> FloatArray:
    """Convert an axis/angle or rotation vector to a rotation matrix.

    When ``angle`` is omitted, ``axis`` is interpreted as a rotation vector:
    its direction is the axis and its magnitude is the angle in radians.
    """

    value = _vector(axis, 3, "axis" if angle is not None else "rotation vector")
    if angle is None:
        theta = float(np.linalg.norm(value))
        if theta <= _EPS:
            return np.eye(3)
        unit_axis = value / theta
    else:
        theta = float(angle)
        if not np.isfinite(theta):
            raise ValueError("angle must be finite")
        if abs(theta) <= _EPS and np.linalg.norm(value) <= _EPS:
            return np.eye(3)
        unit_axis = normalize(value, name="axis")

    cross = skew(unit_axis)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def quat_to_matrix(quaternion: ArrayLike, *, order: str = "wxyz") -> FloatArray:
    """Convert a non-zero quaternion to a rotation matrix.

    The quaternion is normalized, so callers may provide a scaled quaternion.
    """

    quat = _vector(quaternion, 4, "quaternion")
    if order == "xyzw":
        x, y, z, w = quat
    elif order == "wxyz":
        w, x, y, z = quat
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")

    length = float(np.linalg.norm(quat))
    if length <= _EPS:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = np.array((w, x, y, z)) / length

    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def matrix_to_quat(rotation: ArrayLike, *, order: str = "wxyz") -> FloatArray:
    """Convert a rotation matrix to a unit quaternion.

    The returned quaternion is canonicalized to a non-negative scalar component,
    removing the usual ``q`` versus ``-q`` ambiguity except at exactly 180 deg.
    """

    matrix = _rotation(rotation)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            x = 0.25 * scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            y = 0.25 * scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale

    quaternion = np.array((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion

    if order == "wxyz":
        return quaternion
    if order == "xyzw":
        return quaternion[[1, 2, 3, 0]]
    raise ValueError("order must be 'wxyz' or 'xyzw'")


def matrix_to_axis_angle(rotation: ArrayLike) -> tuple[FloatArray, float]:
    """Return the unit axis and principal angle (in ``[0, pi]``)."""

    quaternion = matrix_to_quat(rotation)
    scalar = float(np.clip(quaternion[0], -1.0, 1.0))
    vector = quaternion[1:]
    sine_half = float(np.linalg.norm(vector))
    if sine_half <= _EPS:
        return np.array((1.0, 0.0, 0.0)), 0.0
    axis = vector / sine_half
    angle = 2.0 * np.arctan2(sine_half, scalar)
    return axis, float(angle)


def rpy_to_matrix(rpy: ArrayLike) -> FloatArray:
    """Convert URDF-style fixed-axis XYZ roll, pitch, yaw to a matrix."""

    roll, pitch, yaw = _vector(rpy, 3, "rpy")
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def matrix_to_rpy(rotation: ArrayLike) -> FloatArray:
    """Convert a matrix to URDF-style fixed-axis XYZ roll, pitch, yaw.

    At gimbal lock, infinitely many solutions exist.  This function chooses
    roll=0 and returns the equivalent combined yaw deterministically.
    """

    matrix = _rotation(rotation)
    sine_pitch = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sine_pitch))
    cosine_pitch = float(np.cos(pitch))

    if abs(cosine_pitch) > _ROTATION_ATOL:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return np.array((roll, pitch, yaw))


def rotation_geodesic_error(first: ArrayLike, second: ArrayLike) -> float:
    """Return the shortest SO(3) angle between two rotations, in radians."""

    relative = _rotation(first, "first rotation").T @ _rotation(second, "second rotation")
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    skew_vector = np.array(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
    )
    sine_magnitude = float(np.linalg.norm(skew_vector) * 0.5)
    return float(np.arctan2(sine_magnitude, cosine))


def make_transform(
    rotation: ArrayLike | None = None,
    translation: ArrayLike | None = None,
) -> FloatArray:
    """Build a homogeneous transform from rotation and translation."""

    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = _rotation(rotation)
    if translation is not None:
        result[:3, 3] = _vector(translation, 3, "translation")
    return result


def pose_transform(
    *,
    xyz: ArrayLike = (0.0, 0.0, 0.0),
    rpy: ArrayLike | None = None,
    quat: ArrayLike | None = None,
    quat_order: str = "wxyz",
) -> FloatArray:
    """Build a transform from a translation and either RPY or quaternion pose."""

    if rpy is not None and quat is not None:
        raise ValueError("provide either rpy or quat, not both")
    if quat is not None:
        rotation = quat_to_matrix(quat, order=quat_order)
    elif rpy is not None:
        rotation = rpy_to_matrix(rpy)
    else:
        rotation = np.eye(3)
    return make_transform(rotation, xyz)


def compose_transforms(*transforms: ArrayLike) -> FloatArray:
    """Compose transforms from left to right; no arguments returns identity."""

    result = np.eye(4)
    for index, transform in enumerate(transforms):
        result = result @ _transform(transform, f"transform {index}")
    return result


def invert_transform(transform: ArrayLike) -> FloatArray:
    """Invert a rigid homogeneous transform without a general matrix inverse."""

    value = _transform(transform)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def relative_transform(reference: ArrayLike, target: ArrayLike) -> FloatArray:
    """Return ``T_reference_target`` from two transforms in the same frame."""

    return invert_transform(reference) @ _transform(target, "target")


def transform_points(transform: ArrayLike, points: ArrayLike) -> FloatArray:
    """Apply a transform to one point ``(3,)`` or an array ``(..., 3)``."""

    value = _transform(transform)
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"points must have final dimension 3, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return array @ value[:3, :3].T + value[:3, 3]


def transform_vectors(transform: ArrayLike, vectors: ArrayLike) -> FloatArray:
    """Rotate direction vectors, intentionally ignoring translation."""

    value = _transform(transform)
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"vectors must have final dimension 3, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors must contain only finite values")
    return array @ value[:3, :3].T


_JOINT_TYPE_ALIASES = {
    "continuous": "revolute",
    "hinge": "revolute",
    "slide": "prismatic",
}
_JOINT_TYPES = {"fixed", "revolute", "prismatic"}


@dataclass(frozen=True)
class JointMotion:
    """One scalar joint motion expressed in its edge/body reference frame.

    ``pivot`` matters for hinge/revolute joints and supports MJCF's body-local
    joint ``pos``.  It is ignored for fixed, slide, and prismatic joints.
    """

    name: str
    joint_type: str
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("joint name must be non-empty")
        canonical_type = _JOINT_TYPE_ALIASES.get(self.joint_type.lower(), self.joint_type.lower())
        if canonical_type not in _JOINT_TYPES:
            raise ValueError(f"unsupported joint type: {self.joint_type!r}")
        axis = _vector(self.axis, 3, "axis")
        pivot = _vector(self.pivot, 3, "pivot")
        if canonical_type != "fixed":
            axis = normalize(axis, name="axis")
        object.__setattr__(self, "joint_type", canonical_type)
        object.__setattr__(self, "axis", tuple(float(item) for item in axis))
        object.__setattr__(self, "pivot", tuple(float(item) for item in pivot))


@dataclass(frozen=True)
class KinematicEdge:
    """A parent-to-child tree edge with a fixed pose and ordered motions."""

    parent: str
    child: str
    origin: FloatArray = field(default_factory=lambda: np.eye(4))
    motions: tuple[JointMotion, ...] = ()

    def __post_init__(self) -> None:
        if not self.parent or not self.child:
            raise ValueError("parent and child frame names must be non-empty")
        if self.parent == self.child:
            raise ValueError("an edge cannot connect a frame to itself")
        origin = np.array(_transform(self.origin, "origin"), copy=True)
        origin.setflags(write=False)
        motions = tuple(self.motions)
        if not all(isinstance(motion, JointMotion) for motion in motions):
            raise TypeError("motions must contain only JointMotion objects")
        names = [motion.name for motion in motions]
        if len(names) != len(set(names)):
            raise ValueError("joint names on an edge must be unique")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "motions", motions)


def joint_motion_transform(motion: JointMotion, position: float = 0.0) -> FloatArray:
    """Return the local transform generated by one scalar joint motion."""

    value = float(position)
    if not np.isfinite(value):
        raise ValueError("joint position must be finite")
    axis = np.asarray(motion.axis)

    if motion.joint_type == "fixed":
        return np.eye(4)
    if motion.joint_type == "prismatic":
        return make_transform(translation=axis * value)

    rotation = make_transform(rotation=axis_angle_to_matrix(axis, value))
    pivot = np.asarray(motion.pivot)
    if np.linalg.norm(pivot) <= _EPS:
        return rotation
    return compose_transforms(
        make_transform(translation=pivot),
        rotation,
        make_transform(translation=-pivot),
    )


def edge_transform(edge: KinematicEdge, positions: Mapping[str, float] | None = None) -> FloatArray:
    """Evaluate one edge at the supplied joint positions (missing values are zero)."""

    positions = {} if positions is None else positions
    result = np.array(edge.origin, copy=True)
    for motion in edge.motions:
        result = result @ joint_motion_transform(motion, positions.get(motion.name, 0.0))
    return result


def forward_kinematics(
    edges: Iterable[KinematicEdge],
    positions: Mapping[str, float] | None = None,
    *,
    root: str | None = None,
    root_transform: ArrayLike | None = None,
    strict_positions: bool = True,
) -> dict[str, FloatArray]:
    """Evaluate an unordered tree and return world transforms for every frame.

    Parameters
    ----------
    edges:
        Parent-to-child edges.  Their input order does not matter.
    positions:
        Mapping from scalar joint names to radians or metres.  Missing names use
        zero.  With ``strict_positions=True``, unknown keys are rejected.
    root:
        Optional root frame.  It is inferred when the edges form one tree.
    root_transform:
        World pose of the root, identity by default.
    """

    edge_list = tuple(edges)
    if not edge_list:
        if root is None:
            raise ValueError("root is required when edges is empty")
        return {root: np.eye(4) if root_transform is None else np.array(_transform(root_transform), copy=True)}
    if not all(isinstance(edge, KinematicEdge) for edge in edge_list):
        raise TypeError("edges must contain only KinematicEdge objects")

    children = [edge.child for edge in edge_list]
    if len(children) != len(set(children)):
        raise ValueError("each child frame must have exactly one parent")

    parents = {edge.parent for edge in edge_list}
    child_set = set(children)
    inferred_roots = parents - child_set
    if root is None:
        if len(inferred_roots) != 1:
            raise ValueError(f"could not infer one root frame; candidates: {sorted(inferred_roots)}")
        root = next(iter(inferred_roots))
    if root in child_set:
        raise ValueError(f"root frame {root!r} also has a parent")

    position_values: Mapping[str, float] = {} if positions is None else positions
    joint_name_list = [motion.name for edge in edge_list for motion in edge.motions]
    joint_names = set(joint_name_list)
    if len(joint_name_list) != len(joint_names):
        duplicates = sorted({name for name in joint_names if joint_name_list.count(name) > 1})
        raise ValueError(f"joint names must be globally unique: {duplicates}")
    if strict_positions:
        unknown = set(position_values) - joint_names
        if unknown:
            raise KeyError(f"unknown joint position names: {sorted(unknown)}")

    by_parent: dict[str, list[KinematicEdge]] = {}
    for edge in edge_list:
        by_parent.setdefault(edge.parent, []).append(edge)

    root_pose = np.eye(4) if root_transform is None else np.array(_transform(root_transform), copy=True)
    poses: dict[str, FloatArray] = {root: root_pose}
    pending = [root]
    visited_edges = 0
    while pending:
        parent = pending.pop()
        for edge in by_parent.get(parent, ()):  # A tree means the child is not yet visited.
            if edge.child in poses:
                raise ValueError("kinematic graph contains a cycle")
            poses[edge.child] = poses[parent] @ edge_transform(edge, position_values)
            visited_edges += 1
            pending.append(edge.child)

    if visited_edges != len(edge_list):
        unreachable = sorted(set(children) - poses.keys())
        raise ValueError(f"kinematic graph is disconnected or cyclic; unreachable frames: {unreachable}")
    return poses


__all__ = [
    "JointMotion",
    "KinematicEdge",
    "axis_angle_to_matrix",
    "compose_transforms",
    "edge_transform",
    "forward_kinematics",
    "invert_transform",
    "joint_motion_transform",
    "make_transform",
    "matrix_to_axis_angle",
    "matrix_to_quat",
    "matrix_to_rpy",
    "normalize",
    "pose_transform",
    "quat_to_matrix",
    "relative_transform",
    "rotation_geodesic_error",
    "rpy_to_matrix",
    "skew",
    "transform_points",
    "transform_vectors",
]
