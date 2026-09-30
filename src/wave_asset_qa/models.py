"""Format-neutral data models used by the asset parsers and audit checks.

All angular joint limits exposed by this module are in radians.  Positions are
three-tuples and orientations are quaternions in ``(w, x, y, z)`` order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


Vector3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]

ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)


class ModelParseError(ValueError):
    """Raised when an asset cannot be represented safely by the parser.

    ``source_path`` is retained separately so callers can report concise audit
    findings without having to extract a path from the exception message.
    """

    def __init__(self, source_path: str | Path, message: str) -> None:
        self.source_path = Path(source_path)
        self.detail = message
        super().__init__(f"{self.source_path}: {message}")


@dataclass(frozen=True, slots=True)
class JointSpec:
    """A joint normalized across URDF, MJCF, and USDA overlays.

    ``origin_*`` is the zero-position transform from the parent frame to the
    child/body frame. ``anchor_xyz`` is the joint anchor expressed in the child
    body frame (MJCF ``joint@pos``); URDF joints use the zero vector because
    their joint frame is already encoded by ``origin_*``.
    """

    name: str
    joint_type: str
    parent: str | None = None
    child: str | None = None
    axis: Vector3 | None = None
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    origin_xyz: Vector3 = ZERO_VECTOR
    origin_quat_wxyz: QuaternionWXYZ = IDENTITY_QUATERNION
    anchor_xyz: Vector3 = ZERO_VECTOR
    limit_unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def has_complete_limits(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def limit_span(self) -> float | None:
        if not self.has_complete_limits:
            return None
        assert self.lower is not None and self.upper is not None
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """A link/body frame and its fixed zero-position parent transform."""

    name: str
    parent: str | None = None
    origin_xyz: Vector3 = ZERO_VECTOR
    origin_quat_wxyz: QuaternionWXYZ = IDENTITY_QUATERNION
    joint_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class AssetReference:
    """A file-backed asset reference found in a robot description."""

    kind: str
    raw_path: str
    resolved_path: Path | None
    name: str | None = None
    role: str | None = None
    referenced_by: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def exists(self) -> bool | None:
        """Return existence for resolved references, otherwise ``None``."""

        if self.resolved_path is None:
            return None
        return self.resolved_path.exists()


@dataclass(frozen=True, slots=True)
class ActuatorSpec:
    """An MJCF actuator binding retained for actuator/joint coverage audits."""

    name: str
    actuator_type: str
    joint: str | None = None
    control_lower: float | None = None
    control_upper: float | None = None
    gear: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def has_control_range(self) -> bool:
        return self.control_lower is not None and self.control_upper is not None


@dataclass(frozen=True, slots=True)
class ParsedModel:
    """The audit-facing, immutable representation of one source file."""

    format: str
    name: str
    source_path: Path
    links: tuple[str, ...] = ()
    joints: tuple[JointSpec, ...] = ()
    frames: tuple[FrameSpec, ...] = ()
    assets: tuple[AssetReference, ...] = ()
    actuators: tuple[ActuatorSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def joint_map(self) -> dict[str, JointSpec]:
        return {joint.name: joint for joint in self.joints}

    @property
    def frame_map(self) -> dict[str, FrameSpec]:
        return {frame.name: frame for frame in self.frames}

    @property
    def actuator_map(self) -> dict[str, ActuatorSpec]:
        return {actuator.name: actuator for actuator in self.actuators}

    @property
    def root_frames(self) -> tuple[str, ...]:
        return tuple(frame.name for frame in self.frames if frame.parent is None)

    @property
    def tree_edges(self) -> tuple[tuple[str, str, str | None], ...]:
        """Return ``(parent, child, joint_name)`` for non-root frames."""

        return tuple(
            (frame.parent, frame.name, frame.joint_name)
            for frame in self.frames
            if frame.parent is not None
        )

    @property
    def children_by_parent(self) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = {}
        for frame in self.frames:
            if frame.parent is not None:
                children.setdefault(frame.parent, []).append(frame.name)
        return {parent: tuple(names) for parent, names in children.items()}

    @property
    def unresolved_assets(self) -> tuple[AssetReference, ...]:
        return tuple(asset for asset in self.assets if asset.resolved_path is None)

    @property
    def missing_assets(self) -> tuple[AssetReference, ...]:
        return tuple(asset for asset in self.assets if asset.exists is False)

    def get_joint(self, name: str) -> JointSpec | None:
        return self.joint_map.get(name)

    def get_frame(self, name: str) -> FrameSpec | None:
        return self.frame_map.get(name)


__all__ = [
    "ActuatorSpec",
    "AssetReference",
    "FrameSpec",
    "IDENTITY_QUATERNION",
    "JointSpec",
    "ModelParseError",
    "ParsedModel",
    "QuaternionWXYZ",
    "Vector3",
    "ZERO_VECTOR",
]
