"""Dependency-light parsers for URDF, MJCF, and ASCII USDA overlays.

The USDA reader is intentionally a narrow overlay parser rather than a full
USD implementation.  It reads immediate children of an ``over \"joints\"``
block and normalizes PhysX angular limits from degrees to radians.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .models import (
    ActuatorSpec,
    AssetReference,
    FrameSpec,
    IDENTITY_QUATERNION,
    JointSpec,
    ModelParseError,
    ParsedModel,
    QuaternionWXYZ,
    Vector3,
    ZERO_VECTOR,
)


PathLike = str | Path
PackageRoots = Mapping[str, PathLike]


def parse_model(
    source_path: PathLike,
    *,
    package_roots: PackageRoots | None = None,
) -> ParsedModel:
    """Parse a supported description based on its filename extension."""

    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return parse_urdf(path, package_roots=package_roots)
    if suffix == ".xml":
        return parse_mjcf(path)
    if suffix == ".usda":
        return parse_usda_overlay(path)
    raise ModelParseError(
        path,
        f"unsupported model format {suffix or '<no extension>'!r}; "
        "expected .urdf, .xml, or ASCII .usda",
    )


def parse_urdf(
    source_path: PathLike,
    *,
    package_roots: PackageRoots | None = None,
) -> ParsedModel:
    """Parse links, joints, transforms, and mesh references from a URDF."""

    path, root = _parse_xml(source_path)
    if _local_name(root.tag) != "robot":
        raise ModelParseError(path, "URDF root element must be <robot>")

    model_name = _required_attr(root, "name", path, "robot")
    link_elements = list(_children(root, "link"))
    links: list[str] = []
    link_element_by_name: dict[str, ET.Element] = {}
    for index, link_element in enumerate(link_elements):
        link_name = _required_attr(link_element, "name", path, f"link[{index}]")
        links.append(link_name)
        link_element_by_name.setdefault(link_name, link_element)

    joints: list[JointSpec] = []
    child_joint: dict[str, JointSpec] = {}
    for index, joint_element in enumerate(_children(root, "joint")):
        context = f"joint[{index}]"
        name = _required_attr(joint_element, "name", path, context)
        raw_type = _required_attr(joint_element, "type", path, f"joint {name!r}")
        joint_type = _normalize_urdf_joint_type(raw_type)

        parent_element = _first_child(joint_element, "parent")
        child_element = _first_child(joint_element, "child")
        if parent_element is None or child_element is None:
            raise ModelParseError(path, f"joint {name!r} must define parent and child")
        parent = _required_attr(parent_element, "link", path, f"joint {name!r} parent")
        child = _required_attr(child_element, "link", path, f"joint {name!r} child")

        origin_element = _first_child(joint_element, "origin")
        if origin_element is None:
            origin_xyz = ZERO_VECTOR
            origin_quat = IDENTITY_QUATERNION
        else:
            origin_xyz = _vector_attr(
                origin_element,
                "xyz",
                3,
                path,
                f"joint {name!r} origin",
                ZERO_VECTOR,
            )
            rpy = _vector_attr(
                origin_element,
                "rpy",
                3,
                path,
                f"joint {name!r} origin",
                ZERO_VECTOR,
            )
            try:
                origin_quat = _rpy_to_quaternion(rpy)
            except ValueError as exc:
                raise ModelParseError(
                    path,
                    f"joint {name!r} has invalid origin rotation: {exc}",
                ) from exc

        if joint_type in {"revolute", "prismatic"}:
            axis_element = _first_child(joint_element, "axis")
            axis = (
                _vector_attr(
                    axis_element,
                    "xyz",
                    3,
                    path,
                    f"joint {name!r} axis",
                    (1.0, 0.0, 0.0),
                )
                if axis_element is not None
                else (1.0, 0.0, 0.0)
            )
        else:
            axis = None

        limit_element = _first_child(joint_element, "limit")
        lower = _optional_float_attr(limit_element, "lower", path, f"joint {name!r} limit")
        upper = _optional_float_attr(limit_element, "upper", path, f"joint {name!r} limit")
        effort = _optional_float_attr(limit_element, "effort", path, f"joint {name!r} limit")
        velocity = _optional_float_attr(limit_element, "velocity", path, f"joint {name!r} limit")

        metadata: dict[str, object] = {"source_joint_type": raw_type}
        mimic_element = _first_child(joint_element, "mimic")
        if mimic_element is not None:
            metadata["mimic"] = dict(mimic_element.attrib)

        joint = JointSpec(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            axis=axis,
            lower=lower,
            upper=upper,
            effort=effort,
            velocity=velocity,
            origin_xyz=origin_xyz,
            origin_quat_wxyz=origin_quat,
            anchor_xyz=ZERO_VECTOR,
            limit_unit=_limit_unit(joint_type),
            metadata=metadata,
        )
        joints.append(joint)
        child_joint.setdefault(child, joint)

    frames = tuple(
        FrameSpec(
            name=link_name,
            parent=child_joint[link_name].parent if link_name in child_joint else None,
            origin_xyz=(
                child_joint[link_name].origin_xyz
                if link_name in child_joint
                else ZERO_VECTOR
            ),
            origin_quat_wxyz=(
                child_joint[link_name].origin_quat_wxyz
                if link_name in child_joint
                else IDENTITY_QUATERNION
            ),
            joint_name=(child_joint[link_name].name if link_name in child_joint else None),
            metadata={"source_kind": "link"},
        )
        for link_name in links
    )

    assets: list[AssetReference] = []
    for link_name in links:
        link_element = link_element_by_name[link_name]
        for role in ("visual", "collision"):
            for role_element in _children(link_element, role):
                for mesh_element in _descendants(role_element, "mesh"):
                    raw_path = mesh_element.get("filename")
                    if not raw_path:
                        raise ModelParseError(
                            path,
                            f"{role} mesh on link {link_name!r} has no filename",
                        )
                    scale = _optional_vector_attr(
                        mesh_element,
                        "scale",
                        3,
                        path,
                        f"{role} mesh on link {link_name!r}",
                    )
                    metadata = {"scale": scale} if scale is not None else {}
                    assets.append(
                        AssetReference(
                            kind="mesh",
                            raw_path=raw_path,
                            resolved_path=_resolve_urdf_reference(
                                raw_path,
                                path,
                                package_roots,
                            ),
                            name=Path(raw_path).stem,
                            role=role,
                            referenced_by=link_name,
                            metadata=metadata,
                        )
                    )

    return ParsedModel(
        format="urdf",
        name=model_name,
        source_path=path,
        links=tuple(links),
        joints=tuple(joints),
        frames=frames,
        assets=tuple(assets),
        metadata={"angle_unit": "radian", "robot_attributes": dict(root.attrib)},
    )


def parse_mjcf(source_path: PathLike) -> ParsedModel:
    """Parse an MJCF model's body tree, joints, assets, and actuators."""

    path, root = _parse_xml(source_path)
    if _local_name(root.tag) != "mujoco":
        raise ModelParseError(path, "MJCF root element must be <mujoco>")

    model_name = root.get("model") or path.stem
    compiler = _first_child(root, "compiler")
    compiler_attributes = dict(compiler.attrib) if compiler is not None else {}
    source_angle_unit = compiler_attributes.get("angle", "degree").lower()
    if source_angle_unit not in {"degree", "radian"}:
        raise ModelParseError(
            path,
            f"unsupported MJCF compiler angle unit {source_angle_unit!r}",
        )
    euler_sequence = compiler_attributes.get("eulerseq", "xyz")
    if len(euler_sequence) != 3 or any(axis.lower() not in "xyz" for axis in euler_sequence):
        raise ModelParseError(path, f"unsupported MJCF eulerseq {euler_sequence!r}")

    mesh_directory = _mjcf_mesh_directory(path, compiler_attributes)
    assets: list[AssetReference] = []
    asset_element = _first_child(root, "asset")
    if asset_element is not None:
        for index, mesh_element in enumerate(_children(asset_element, "mesh")):
            raw_path = mesh_element.get("file")
            if raw_path is None:
                # MJCF also permits plugin/procedural mesh definitions without a file.
                continue
            mesh_name = mesh_element.get("name") or Path(raw_path).stem or f"mesh_{index}"
            scale = _optional_vector_attr(
                mesh_element,
                "scale",
                3,
                path,
                f"MJCF mesh {mesh_name!r}",
            )
            metadata = {
                key: value
                for key, value in mesh_element.attrib.items()
                if key not in {"name", "file"}
            }
            if scale is not None:
                metadata["scale"] = scale
            assets.append(
                AssetReference(
                    kind="mesh",
                    raw_path=raw_path,
                    resolved_path=_resolve_relative_reference(raw_path, mesh_directory),
                    name=mesh_name,
                    role="asset",
                    referenced_by=None,
                    metadata=metadata,
                )
            )

    worldbody = _first_child(root, "worldbody")
    if worldbody is None:
        raise ModelParseError(path, "MJCF must contain a <worldbody>")

    frames: list[FrameSpec] = []
    joints: list[JointSpec] = []
    body_counter = 0
    joint_counter = 0

    def visit_body(body_element: ET.Element, parent_name: str | None) -> None:
        nonlocal body_counter, joint_counter
        body_counter += 1
        generated_body_name = body_element.get("name") is None
        body_name = body_element.get("name") or f"__unnamed_body_{body_counter}"
        origin_xyz, origin_quat = _mjcf_body_transform(
            body_element,
            path,
            body_name,
            source_angle_unit,
            euler_sequence,
        )

        body_joint_names: list[str] = []
        for joint_element in body_element:
            tag = _local_name(joint_element.tag)
            if tag not in {"joint", "freejoint"}:
                continue
            joint_counter += 1
            generated_joint_name = joint_element.get("name") is None
            joint_name = joint_element.get("name") or f"__unnamed_joint_{joint_counter}"
            raw_type = "free" if tag == "freejoint" else joint_element.get("type", "hinge")
            joint_type = _normalize_mjcf_joint_type(raw_type)

            if joint_type in {"revolute", "prismatic"}:
                axis = _vector_attr(
                    joint_element,
                    "axis",
                    3,
                    path,
                    f"MJCF joint {joint_name!r}",
                    (0.0, 0.0, 1.0),
                )
            else:
                axis = None
            anchor = _vector_attr(
                joint_element,
                "pos",
                3,
                path,
                f"MJCF joint {joint_name!r}",
                ZERO_VECTOR,
            )

            range_values = _optional_vector_attr(
                joint_element,
                "range",
                2,
                path,
                f"MJCF joint {joint_name!r}",
            )
            if range_values is None:
                lower = upper = None
            else:
                lower, upper = range_values
                if joint_type in {"revolute", "spherical"} and source_angle_unit == "degree":
                    lower, upper = math.radians(lower), math.radians(upper)

            metadata: dict[str, object] = {
                "source_joint_type": raw_type,
                "generated_name": generated_joint_name,
            }
            metadata.update(
                {
                    key: value
                    for key, value in joint_element.attrib.items()
                    if key not in {"name", "type", "axis", "pos", "range"}
                }
            )
            joint = JointSpec(
                name=joint_name,
                joint_type=joint_type,
                parent=parent_name,
                child=body_name,
                axis=axis,
                lower=lower,
                upper=upper,
                origin_xyz=origin_xyz,
                origin_quat_wxyz=origin_quat,
                anchor_xyz=anchor,
                limit_unit=_limit_unit(joint_type),
                metadata=metadata,
            )
            joints.append(joint)
            body_joint_names.append(joint_name)

        frames.append(
            FrameSpec(
                name=body_name,
                parent=parent_name,
                origin_xyz=origin_xyz,
                origin_quat_wxyz=origin_quat,
                joint_name=body_joint_names[0] if body_joint_names else None,
                metadata={
                    "source_kind": "body",
                    "generated_name": generated_body_name,
                    "joint_names": tuple(body_joint_names),
                },
            )
        )

        for child_body in _children(body_element, "body"):
            visit_body(child_body, body_name)

    for top_level_body in _children(worldbody, "body"):
        visit_body(top_level_body, None)

    actuators: list[ActuatorSpec] = []
    actuator_element = _first_child(root, "actuator")
    if actuator_element is not None:
        for index, element in enumerate(list(actuator_element), start=1):
            actuator_type = _local_name(element.tag)
            name = element.get("name") or f"__unnamed_actuator_{index}"
            control_range = _optional_vector_attr(
                element,
                "ctrlrange",
                2,
                path,
                f"MJCF actuator {name!r}",
            )
            gear = _optional_float_sequence_attr(
                element,
                "gear",
                path,
                f"MJCF actuator {name!r}",
            )
            metadata = {
                key: value
                for key, value in element.attrib.items()
                if key not in {"name", "joint", "ctrlrange", "gear"}
            }
            actuators.append(
                ActuatorSpec(
                    name=name,
                    actuator_type=actuator_type,
                    joint=element.get("joint") or element.get("jointinparent"),
                    control_lower=(control_range[0] if control_range is not None else None),
                    control_upper=(control_range[1] if control_range is not None else None),
                    gear=gear or (),
                    metadata=metadata,
                )
            )

    return ParsedModel(
        format="mjcf",
        name=model_name,
        source_path=path,
        links=tuple(frame.name for frame in frames),
        joints=tuple(joints),
        frames=tuple(frames),
        assets=tuple(assets),
        actuators=tuple(actuators),
        metadata={
            "source_angle_unit": source_angle_unit,
            "angle_unit": "radian",
            "compiler": compiler_attributes,
            "mesh_directory": mesh_directory,
        },
    )


def parse_usda_overlay(source_path: PathLike) -> ParsedModel:
    """Parse joint limit overrides from an ASCII USDA layer.

    USD revolute-joint angular properties are authored in degrees.  Returned
    ``JointSpec.lower`` and ``JointSpec.upper`` values are converted to radians.
    Only joint prims immediately inside ``over/def \"joints\"`` are returned;
    nested prims cannot be mistaken for peer joints.
    """

    path = _existing_path(source_path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ModelParseError(path, f"could not read ASCII USDA: {exc}") from exc

    if not text.lstrip().startswith("#usda"):
        raise ModelParseError(path, "file is not an ASCII USDA layer")

    default_prim_match = re.search(r"\bdefaultPrim\s*=\s*\"([^\"]+)\"", text)
    model_name = default_prim_match.group(1) if default_prim_match else path.stem

    joint_blocks: list[tuple[str, str]] = []
    try:
        for block in _walk_usda_blocks(text):
            if block.name == "joints":
                joint_blocks.extend(
                    (child.name, child.body)
                    for child in _immediate_usda_blocks(block.body)
                )
    except ValueError as exc:
        raise ModelParseError(path, f"invalid USDA block structure: {exc}") from exc

    joints: list[JointSpec] = []
    for joint_name, body in joint_blocks:
        direct_body = _usda_direct_properties(body)
        lower_degrees = _usda_optional_scalar(
            direct_body,
            "physics:lowerLimit",
            path,
            joint_name,
        )
        upper_degrees = _usda_optional_scalar(
            direct_body,
            "physics:upperLimit",
            path,
            joint_name,
        )
        metadata: dict[str, object] = {
            "source_limit_unit": "degree",
            "source_prim_kind": "joint_overlay",
        }
        if lower_degrees is not None:
            metadata["source_lower_degrees"] = lower_degrees
        if upper_degrees is not None:
            metadata["source_upper_degrees"] = upper_degrees
        joints.append(
            JointSpec(
                name=joint_name,
                joint_type="revolute",
                lower=(math.radians(lower_degrees) if lower_degrees is not None else None),
                upper=(math.radians(upper_degrees) if upper_degrees is not None else None),
                origin_xyz=ZERO_VECTOR,
                origin_quat_wxyz=IDENTITY_QUATERNION,
                anchor_xyz=ZERO_VECTOR,
                limit_unit="radian",
                metadata=metadata,
            )
        )

    return ParsedModel(
        format="usda",
        name=model_name,
        source_path=path,
        joints=tuple(joints),
        metadata={
            "source_limit_unit": "degree",
            "angle_unit": "radian",
            "overlay_block_count": len(joint_blocks),
        },
    )


# Compatibility-friendly short name for callers that already know the layer is ASCII.
parse_usda = parse_usda_overlay


def _existing_path(source_path: PathLike) -> Path:
    path = Path(source_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ModelParseError(path, "file does not exist")
    return path


def _parse_xml(source_path: PathLike) -> tuple[Path, ET.Element]:
    path = _existing_path(source_path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError, UnicodeError) as exc:
        raise ModelParseError(path, f"invalid XML: {exc}") from exc
    return path, root


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, local_name: str) -> Iterator[ET.Element]:
    return (child for child in list(element) if _local_name(child.tag) == local_name)


def _descendants(element: ET.Element, local_name: str) -> Iterator[ET.Element]:
    return (
        descendant
        for descendant in element.iter()
        if descendant is not element and _local_name(descendant.tag) == local_name
    )


def _first_child(element: ET.Element, local_name: str) -> ET.Element | None:
    return next(_children(element, local_name), None)


def _required_attr(
    element: ET.Element,
    attribute: str,
    path: Path,
    context: str,
) -> str:
    value = element.get(attribute)
    if value is None or not value.strip():
        raise ModelParseError(path, f"{context} is missing required attribute {attribute!r}")
    return value.strip()


def _float(value: str, path: Path, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ModelParseError(path, f"{context} must be numeric, got {value!r}") from exc


def _optional_float_attr(
    element: ET.Element | None,
    attribute: str,
    path: Path,
    context: str,
) -> float | None:
    if element is None or attribute not in element.attrib:
        return None
    return _float(element.attrib[attribute], path, f"{context} {attribute}")


def _float_sequence(value: str, path: Path, context: str) -> tuple[float, ...]:
    tokens = value.replace(",", " ").split()
    return tuple(_float(token, path, context) for token in tokens)


def _optional_float_sequence_attr(
    element: ET.Element,
    attribute: str,
    path: Path,
    context: str,
) -> tuple[float, ...] | None:
    value = element.get(attribute)
    if value is None:
        return None
    return _float_sequence(value, path, f"{context} {attribute}")


def _vector_attr(
    element: ET.Element,
    attribute: str,
    length: int,
    path: Path,
    context: str,
    default: Sequence[float],
) -> tuple[float, ...]:
    value = element.get(attribute)
    if value is None:
        values = tuple(float(item) for item in default)
    else:
        values = _float_sequence(value, path, f"{context} {attribute}")
    if len(values) != length:
        raise ModelParseError(
            path,
            f"{context} {attribute} must contain {length} numbers, got {len(values)}",
        )
    return values


def _optional_vector_attr(
    element: ET.Element,
    attribute: str,
    length: int,
    path: Path,
    context: str,
) -> tuple[float, ...] | None:
    if attribute not in element.attrib:
        return None
    return _vector_attr(element, attribute, length, path, context, ())


def _normalize_urdf_joint_type(raw_type: str) -> str:
    return {
        "continuous": "revolute",
        "revolute": "revolute",
        "prismatic": "prismatic",
        "fixed": "fixed",
        "floating": "floating",
        "planar": "planar",
    }.get(raw_type.lower(), raw_type.lower())


def _normalize_mjcf_joint_type(raw_type: str) -> str:
    return {
        "hinge": "revolute",
        "slide": "prismatic",
        "ball": "spherical",
        "free": "floating",
    }.get(raw_type.lower(), raw_type.lower())


def _limit_unit(joint_type: str) -> str | None:
    if joint_type in {"revolute", "spherical"}:
        return "radian"
    if joint_type == "prismatic":
        return "metre"
    return None


def _resolve_urdf_reference(
    raw_path: str,
    source_path: Path,
    package_roots: PackageRoots | None,
) -> Path | None:
    if raw_path.startswith("package://"):
        package_reference = raw_path[len("package://") :]
        package_name, separator, relative = package_reference.partition("/")
        if not separator or not package_name or not relative:
            return None
        if package_roots and package_name in package_roots:
            package_root = Path(package_roots[package_name]).expanduser()
        else:
            package_root = next(
                (ancestor for ancestor in (source_path.parent, *source_path.parents) if ancestor.name == package_name),
                None,
            )
        if package_root is None:
            return None
        return (package_root / Path(relative)).resolve(strict=False)
    return _resolve_relative_reference(raw_path, source_path.parent)


def _resolve_relative_reference(raw_path: str, base_directory: Path) -> Path | None:
    if "://" in raw_path:
        parsed = urlparse(raw_path)
        if parsed.scheme != "file":
            return None
        file_path = unquote(parsed.path)
        if parsed.netloc:
            file_path = f"//{parsed.netloc}{file_path}"
        if re.match(r"^/[A-Za-z]:/", file_path):
            file_path = file_path[1:]
        return Path(file_path).resolve(strict=False)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return candidate.resolve(strict=False)


def _mjcf_mesh_directory(path: Path, compiler_attributes: Mapping[str, str]) -> Path:
    meshdir = compiler_attributes.get("meshdir")
    assetdir = compiler_attributes.get("assetdir")
    raw_directory = meshdir or assetdir or "."
    directory = Path(raw_directory)
    if not directory.is_absolute():
        directory = path.parent / directory
    return directory.resolve(strict=False)


def _rpy_to_quaternion(rpy: Sequence[float]) -> QuaternionWXYZ:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalize_quaternion(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _normalize_quaternion(quaternion: Sequence[float]) -> QuaternionWXYZ:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("quaternion must have a finite, non-zero norm")
    values = tuple(component / norm for component in quaternion)
    return values  # type: ignore[return-value]


def _quaternion_multiply(
    left: QuaternionWXYZ,
    right: QuaternionWXYZ,
) -> QuaternionWXYZ:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _axis_angle_quaternion(axis: Sequence[float], angle: float) -> QuaternionWXYZ:
    norm = math.sqrt(sum(component * component for component in axis))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("rotation axis must have a finite, non-zero norm")
    half_sine = math.sin(angle / 2.0)
    return (
        math.cos(angle / 2.0),
        axis[0] / norm * half_sine,
        axis[1] / norm * half_sine,
        axis[2] / norm * half_sine,
    )


def _mjcf_body_transform(
    body: ET.Element,
    path: Path,
    body_name: str,
    angle_unit: str,
    euler_sequence: str,
) -> tuple[Vector3, QuaternionWXYZ]:
    context = f"MJCF body {body_name!r}"
    position = _vector_attr(body, "pos", 3, path, context, ZERO_VECTOR)
    orientation_attributes = [
        attribute
        for attribute in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
        if attribute in body.attrib
    ]
    if len(orientation_attributes) > 1:
        raise ModelParseError(
            path,
            f"{context} defines multiple orientation attributes: {orientation_attributes}",
        )
    if not orientation_attributes:
        return position, IDENTITY_QUATERNION  # type: ignore[return-value]

    orientation_attribute = orientation_attributes[0]
    try:
        if orientation_attribute == "quat":
            quaternion = _vector_attr(
                body,
                "quat",
                4,
                path,
                context,
                IDENTITY_QUATERNION,
            )
            return position, _normalize_quaternion(quaternion)  # type: ignore[return-value]
        if orientation_attribute == "axisangle":
            axisangle = _vector_attr(body, "axisangle", 4, path, context, ())
            angle = axisangle[3]
            if angle_unit == "degree":
                angle = math.radians(angle)
            return position, _axis_angle_quaternion(axisangle[:3], angle)  # type: ignore[return-value]
        if orientation_attribute == "euler":
            angles = _vector_attr(body, "euler", 3, path, context, ())
            if angle_unit == "degree":
                angles = tuple(math.radians(angle) for angle in angles)
            quaternion = IDENTITY_QUATERNION
            for axis_name, angle in zip(euler_sequence.lower(), angles):
                axis = {
                    "x": (1.0, 0.0, 0.0),
                    "y": (0.0, 1.0, 0.0),
                    "z": (0.0, 0.0, 1.0),
                }[axis_name]
                quaternion = _quaternion_multiply(
                    quaternion,
                    _axis_angle_quaternion(axis, angle),
                )
            return position, _normalize_quaternion(quaternion)  # type: ignore[return-value]
        if orientation_attribute == "xyaxes":
            xyaxes = _vector_attr(body, "xyaxes", 6, path, context, ())
            return position, _quaternion_from_xy_axes(xyaxes[:3], xyaxes[3:])  # type: ignore[return-value]
        zaxis = _vector_attr(body, "zaxis", 3, path, context, ())
        return position, _quaternion_from_z_axis(zaxis)  # type: ignore[return-value]
    except ValueError as exc:
        raise ModelParseError(path, f"{context} has invalid orientation: {exc}") from exc


def _normalize_vector(vector: Sequence[float]) -> Vector3:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("vector must have a finite, non-zero norm")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _quaternion_from_xy_axes(x_axis: Sequence[float], y_axis: Sequence[float]) -> QuaternionWXYZ:
    x = _normalize_vector(x_axis)
    y_projection = tuple(y - _dot(y_axis, x) * x_component for y, x_component in zip(y_axis, x))
    y = _normalize_vector(y_projection)
    z = _normalize_vector(_cross(x, y))
    return _quaternion_from_matrix(
        (
            (x[0], y[0], z[0]),
            (x[1], y[1], z[1]),
            (x[2], y[2], z[2]),
        )
    )


def _quaternion_from_z_axis(z_axis: Sequence[float]) -> QuaternionWXYZ:
    target = _normalize_vector(z_axis)
    source = (0.0, 0.0, 1.0)
    cosine = max(-1.0, min(1.0, _dot(source, target)))
    if cosine > 1.0 - 1e-12:
        return IDENTITY_QUATERNION
    if cosine < -1.0 + 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    axis = _cross(source, target)
    return _axis_angle_quaternion(axis, math.acos(cosine))


def _quaternion_from_matrix(matrix: Sequence[Sequence[float]]) -> QuaternionWXYZ:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    return _normalize_quaternion(quaternion)


@dataclass(frozen=True, slots=True)
class _UsdaBlock:
    kind: str
    type_name: str | None
    name: str
    body: str


_USDA_DECLARATION = re.compile(
    r"(?P<kind>over|def|class)\s+"
    r"(?:(?P<type>[A-Za-z_][\w:]*)\s+)?"
    r'\"(?P<name>(?:\\.|[^\"\\])*)\"'
)


def _walk_usda_blocks(text: str) -> Iterator[_UsdaBlock]:
    for block in _immediate_usda_blocks(text):
        yield block
        yield from _walk_usda_blocks(block.body)


def _immediate_usda_blocks(text: str) -> Iterator[_UsdaBlock]:
    index = 0
    depth = 0
    while index < len(text):
        if text[index] == "#":
            index = _skip_line_comment(text, index)
            continue
        if text[index] == '"':
            index = _skip_quoted_string(text, index)
            continue
        if text[index] == "{":
            depth += 1
            index += 1
            continue
        if text[index] == "}":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0:
            match = _USDA_DECLARATION.match(text, index)
            if match is not None:
                open_brace = _find_usda_block_open(text, match.end())
                if open_brace is None:
                    index = match.end()
                    continue
                close_brace = _find_matching_brace(text, open_brace)
                if close_brace is None:
                    raise ValueError(f"unclosed USDA block {match.group('name')!r}")
                yield _UsdaBlock(
                    kind=match.group("kind"),
                    type_name=match.group("type"),
                    name=match.group("name").replace(r'\"', '"'),
                    body=text[open_brace + 1 : close_brace],
                )
                index = close_brace + 1
                continue
        index += 1


def _find_usda_block_open(text: str, start: int) -> int | None:
    parenthesis_depth = 0
    bracket_depth = 0
    index = start
    while index < len(text):
        character = text[index]
        if character == "#":
            index = _skip_line_comment(text, index)
            continue
        if character == '"':
            index = _skip_quoted_string(text, index)
            continue
        if character == "(":
            parenthesis_depth += 1
        elif character == ")":
            parenthesis_depth = max(0, parenthesis_depth - 1)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif character == "{" and parenthesis_depth == 0 and bracket_depth == 0:
            return index
        elif character in "\n;" and parenthesis_depth == 0 and bracket_depth == 0:
            # Declarations often put the opening brace on the next line, so a
            # newline is not terminal when the next code token is ``{``.
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] == "{":
                return lookahead
            if character == ";":
                return None
        index += 1
    return None


def _find_matching_brace(text: str, open_brace: int) -> int | None:
    depth = 0
    index = open_brace
    while index < len(text):
        character = text[index]
        if character == "#":
            index = _skip_line_comment(text, index)
            continue
        if character == '"':
            index = _skip_quoted_string(text, index)
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _skip_line_comment(text: str, start: int) -> int:
    newline = text.find("\n", start)
    return len(text) if newline == -1 else newline + 1


def _skip_quoted_string(text: str, start: int) -> int:
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return index + 1
        index += 1
    return len(text)


def _usda_direct_properties(body: str) -> str:
    """Mask nested brace content while retaining direct scalar assignments."""

    result = list(body)
    depth = 0
    index = 0
    while index < len(body):
        character = body[index]
        if character == "#":
            end = _skip_line_comment(body, index)
            for cursor in range(index, end):
                if body[cursor] != "\n":
                    result[cursor] = " "
            index = end
            continue
        if character == '"':
            end = _skip_quoted_string(body, index)
            if depth > 0:
                for cursor in range(index, end):
                    if body[cursor] != "\n":
                        result[cursor] = " "
            index = end
            continue
        if character == "{":
            if depth == 0:
                result[index] = " "
            depth += 1
        elif character == "}":
            result[index] = " "
            depth = max(0, depth - 1)
        elif depth > 0 and character != "\n":
            result[index] = " "
        index += 1
    return "".join(result)


def _usda_optional_scalar(
    direct_body: str,
    property_name: str,
    path: Path,
    joint_name: str,
) -> float | None:
    pattern = re.compile(
        rf"\b{re.escape(property_name)}\s*=\s*(?P<value>[^\s;,)}}]+)"
    )
    match = pattern.search(direct_body)
    if match is None:
        return None
    return _float(
        match.group("value"),
        path,
        f"USDA joint {joint_name!r} property {property_name}",
    )


__all__ = [
    "parse_mjcf",
    "parse_model",
    "parse_urdf",
    "parse_usda",
    "parse_usda_overlay",
]
