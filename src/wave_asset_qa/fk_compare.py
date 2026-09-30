"""Sampled forward-kinematics parity checks for URDF and MJCF models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np

from .kinematics import (
    JointMotion,
    KinematicEdge,
    forward_kinematics,
    invert_transform,
    pose_transform,
    rotation_geodesic_error,
)
from .models import JointSpec, ParsedModel


@dataclass(frozen=True, slots=True)
class FrameErrorSummary:
    frame: str
    position_median_mm: float
    position_p95_mm: float
    position_max_mm: float
    orientation_median_deg: float
    orientation_p95_deg: float
    orientation_max_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FKComparisonResult:
    status: str
    message: str
    samples: int
    seed: int
    base_frame: str | None
    terminal_frames: tuple[str, ...]
    sampled_joint_count: int
    position_median_mm: float | None = None
    position_p95_mm: float | None = None
    position_max_mm: float | None = None
    orientation_median_deg: float | None = None
    orientation_p95_deg: float | None = None
    orientation_max_deg: float | None = None
    per_frame: tuple[FrameErrorSummary, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["terminal_frames"] = list(self.terminal_frames)
        data["per_frame"] = [item.to_dict() for item in self.per_frame]
        return data


def _model_edges(model: ParsedModel) -> tuple[list[KinematicEdge], str, np.ndarray]:
    frame_map = model.frame_map
    roots = list(model.root_frames)
    if len(roots) != 1:
        raise ValueError(f"FK comparison needs exactly one root frame, found {len(roots)}")
    root_name = roots[0]
    root_frame = frame_map[root_name]
    root_transform = pose_transform(
        xyz=root_frame.origin_xyz,
        quat=root_frame.origin_quat_wxyz,
    )

    joint_map = model.joint_map
    edges: list[KinematicEdge] = []
    for frame in model.frames:
        if frame.parent is None:
            root_joint_names = tuple(frame.metadata.get("joint_names", ()))
            if root_joint_names:
                raise ValueError("root-frame joint motions are not supported by the v0.1 FK check")
            continue

        if model.format == "mjcf":
            joint_names = tuple(frame.metadata.get("joint_names", ()))
        else:
            joint_names = (frame.joint_name,) if frame.joint_name else ()

        motions: list[JointMotion] = []
        for joint_name in joint_names:
            joint = joint_map[joint_name]
            if joint.joint_type == "fixed":
                continue
            if joint.joint_type not in {"revolute", "prismatic"}:
                raise ValueError(f"unsupported scalar FK joint type {joint.joint_type!r}")
            if joint.axis is None:
                raise ValueError(f"joint {joint.name!r} has no axis")
            motions.append(
                JointMotion(
                    name=joint.name,
                    joint_type=joint.joint_type,
                    axis=joint.axis,
                    pivot=joint.anchor_xyz,
                )
            )
        edges.append(
            KinematicEdge(
                parent=frame.parent,
                child=frame.name,
                origin=pose_transform(
                    xyz=frame.origin_xyz,
                    quat=frame.origin_quat_wxyz,
                ),
                motions=tuple(motions),
            )
        )
    return edges, root_name, root_transform


def _scalar_joints(model: ParsedModel) -> dict[str, JointSpec]:
    return {
        joint.name: joint
        for joint in model.joints
        if joint.joint_type in {"revolute", "prismatic"}
    }


def _pick_base_frame(first: ParsedModel, second: ParsedModel) -> str:
    common = set(first.frame_map) & set(second.frame_map)
    candidates = sorted(name for name in common if name.endswith("hand_C_MC"))
    if len(candidates) == 1:
        return candidates[0]
    common_roots = sorted(set(first.root_frames) & set(second.root_frames))
    if len(common_roots) == 1:
        return common_roots[0]
    raise ValueError("could not identify one common hand base frame")


def _pick_terminal_frames(first: ParsedModel, second: ParsedModel) -> tuple[str, ...]:
    common = set(first.frame_map) & set(second.frame_map)
    terminals = tuple(sorted(name for name in common if name.endswith("_DP")))
    if not terminals:
        raise ValueError("no common distal-phalanx (*_DP) frames found")
    return terminals


def _sampling_ranges(first: ParsedModel, second: ParsedModel) -> dict[str, tuple[float, float]]:
    first_joints = _scalar_joints(first)
    second_joints = _scalar_joints(second)
    if set(first_joints) != set(second_joints):
        missing_first = sorted(set(second_joints) - set(first_joints))
        missing_second = sorted(set(first_joints) - set(second_joints))
        raise ValueError(
            f"joint sets differ (missing in first={missing_first}, missing in second={missing_second})"
        )

    ranges: dict[str, tuple[float, float]] = {}
    for name in sorted(first_joints):
        a, b = first_joints[name], second_joints[name]
        if None in (a.lower, a.upper, b.lower, b.upper):
            raise ValueError(f"joint {name!r} does not have complete limits in both formats")
        assert a.lower is not None and a.upper is not None
        assert b.lower is not None and b.upper is not None
        lower = max(a.lower, b.lower)
        upper = min(a.upper, b.upper)
        if not lower < upper:
            raise ValueError(f"joint {name!r} has no shared non-empty limit interval")
        ranges[name] = (lower, upper)
    return ranges


def _summarize(frame: str, positions_m: Iterable[float], orientations_rad: Iterable[float]) -> FrameErrorSummary:
    position = np.asarray(tuple(positions_m), dtype=np.float64) * 1000.0
    orientation = np.rad2deg(np.asarray(tuple(orientations_rad), dtype=np.float64))
    return FrameErrorSummary(
        frame=frame,
        position_median_mm=float(np.median(position)),
        position_p95_mm=float(np.percentile(position, 95)),
        position_max_mm=float(np.max(position)),
        orientation_median_deg=float(np.median(orientation)),
        orientation_p95_deg=float(np.percentile(orientation, 95)),
        orientation_max_deg=float(np.max(orientation)),
    )


def compare_forward_kinematics(
    urdf: ParsedModel,
    mjcf: ParsedModel,
    *,
    samples: int = 256,
    seed: int = 20260826,
) -> FKComparisonResult:
    """Compare common distal frames after one fixed hand-base alignment.

    Joint configurations are sampled from the intersection of the URDF and
    MJCF limits.  The base alignment is computed once at the zero joint pose;
    there is no per-sample or per-frame fitting.
    """

    if samples < 1:
        raise ValueError("samples must be at least one")
    if {urdf.format, mjcf.format} != {"urdf", "mjcf"}:
        raise ValueError("comparison requires one URDF and one MJCF model")
    if urdf.format != "urdf":
        urdf, mjcf = mjcf, urdf

    try:
        ranges = _sampling_ranges(urdf, mjcf)
        urdf_edges, _, urdf_root_transform = _model_edges(urdf)
        mjcf_edges, _, mjcf_root_transform = _model_edges(mjcf)
        base_frame = _pick_base_frame(urdf, mjcf)
        terminals = _pick_terminal_frames(urdf, mjcf)

        zeros = {name: 0.0 for name in ranges}
        urdf_zero = forward_kinematics(urdf_edges, zeros, root_transform=urdf_root_transform)
        mjcf_zero = forward_kinematics(mjcf_edges, zeros, root_transform=mjcf_root_transform)
        alignment = mjcf_zero[base_frame] @ invert_transform(urdf_zero[base_frame])

        rng = np.random.default_rng(seed)
        per_frame_positions: dict[str, list[float]] = {name: [] for name in terminals}
        per_frame_orientations: dict[str, list[float]] = {name: [] for name in terminals}
        for _ in range(samples):
            positions = {
                name: float(rng.uniform(lower, upper))
                for name, (lower, upper) in ranges.items()
            }
            urdf_poses = forward_kinematics(
                urdf_edges,
                positions,
                root_transform=urdf_root_transform,
            )
            mjcf_poses = forward_kinematics(
                mjcf_edges,
                positions,
                root_transform=mjcf_root_transform,
            )
            for frame in terminals:
                first = alignment @ urdf_poses[frame]
                second = mjcf_poses[frame]
                per_frame_positions[frame].append(float(np.linalg.norm(first[:3, 3] - second[:3, 3])))
                per_frame_orientations[frame].append(
                    rotation_geodesic_error(first[:3, :3], second[:3, :3])
                )

        summaries = tuple(
            _summarize(frame, per_frame_positions[frame], per_frame_orientations[frame])
            for frame in terminals
        )
        all_position = np.asarray(
            [value for values in per_frame_positions.values() for value in values]
        ) * 1000.0
        all_orientation = np.rad2deg(
            np.asarray([value for values in per_frame_orientations.values() for value in values])
        )
        return FKComparisonResult(
            status="pass",
            message="Compared distal frames after one zero-pose hand-base alignment.",
            samples=samples,
            seed=seed,
            base_frame=base_frame,
            terminal_frames=terminals,
            sampled_joint_count=len(ranges),
            position_median_mm=float(np.median(all_position)),
            position_p95_mm=float(np.percentile(all_position, 95)),
            position_max_mm=float(np.max(all_position)),
            orientation_median_deg=float(np.median(all_orientation)),
            orientation_p95_deg=float(np.percentile(all_orientation, 95)),
            orientation_max_deg=float(np.max(all_orientation)),
            per_frame=summaries,
        )
    except (KeyError, TypeError, ValueError) as error:
        return FKComparisonResult(
            status="skip",
            message=f"{type(error).__name__}: {error}",
            samples=samples,
            seed=seed,
            base_frame=None,
            terminal_frames=(),
            sampled_joint_count=0,
        )
