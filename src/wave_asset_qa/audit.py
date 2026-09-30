"""End-to-end Wave asset audit orchestration and rule evaluation."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import __version__
from .discovery import DiscoveredModel, discover_model_files, group_by_canonical_name, inventory_counts
from .fk_compare import compare_forward_kinematics
from .models import JointSpec, ModelParseError, ParsedModel
from .parsers import parse_model
from .simulation import run_mujoco_smoke


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "expected_active_joints_per_hand": 22,
    "axis_norm_tolerance": 1e-6,
    "joint_limit_tolerance_rad": 2e-4,
    "fk_position_warning_mm": 0.5,
    "fk_orientation_warning_deg": 0.1,
    "required_fingers": ["thumb", "index", "middle", "ring", "pinky"],
    "required_formats": ["urdf", "mjcf", "usda"],
    "strict_warnings": False,
}


@dataclass(frozen=True, slots=True)
class AuditCheck:
    status: str
    code: str
    model: str
    message: str
    reference: str | None = None
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class _Recorder:
    def __init__(self, known_issues: Iterable[Mapping[str, Any]]) -> None:
        self.checks: list[AuditCheck] = []
        self.known_issues = tuple(known_issues)

    def add(
        self,
        status: str,
        code: str,
        model: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> AuditCheck:
        reference = None
        if status == "fail":
            known = self._match_known(code, model, message, details)
            if known is not None:
                status = "known"
                reference = str(known.get("reference")) if known.get("reference") else None
                note = known.get("note")
                if note:
                    details = dict(details or {})
                    details["known_issue_note"] = str(note)
                    details["known_issue_id"] = str(known.get("id", "known"))
        check = AuditCheck(status, code, model, message, reference, details)
        self.checks.append(check)
        return check

    def _match_known(
        self,
        code: str,
        model: str,
        message: str,
        details: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        for issue in self.known_issues:
            codes = tuple(str(item) for item in issue.get("codes", ()))
            globs = tuple(str(item) for item in issue.get("path_globs", ()))
            tokens = tuple(str(item) for item in issue.get("message_contains", ()))
            if codes and code not in codes:
                continue
            if globs and not any(fnmatch.fnmatch(model, pattern) for pattern in globs):
                continue
            if tokens and not any(token in message for token in tokens):
                continue
            reference_globs = tuple(str(item) for item in issue.get("all_references_match", ()))
            if reference_globs:
                references = tuple(str(item) for item in (details or {}).get("references", ()))
                if not references or not all(
                    any(fnmatch.fnmatch(reference, pattern) for pattern in reference_globs)
                    for reference in references
                ):
                    continue
            expected_counts = issue.get("expected_reference_counts", {})
            if expected_counts:
                expected = expected_counts.get(model)
                references = {str(item) for item in (details or {}).get("references", ())}
                if expected is None or len(references) != int(expected):
                    continue
            expected_hashes = issue.get("reference_set_sha256", {})
            if expected_hashes:
                expected_hash = expected_hashes.get(model)
                references = sorted({str(item) for item in (details or {}).get("references", ())})
                actual_hash = hashlib.sha256("\n".join(references).encode("utf-8")).hexdigest()
                if expected_hash is None or actual_hash != expected_hash:
                    continue
            return issue
        return None


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    source = Path(path).expanduser().resolve()
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a JSON object: {source}")
    config.update(loaded)
    if int(config["expected_active_joints_per_hand"]) < 1:
        raise ValueError("expected_active_joints_per_hand must be positive")
    if float(config["joint_limit_tolerance_rad"]) < 0.0:
        raise ValueError("joint_limit_tolerance_rad must be non-negative")
    return config


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _git_dirty(path: Path) -> bool | None:
    if _git_head(path) is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _asset_tree_sha256(asset_root: Path) -> str:
    """Hash relative paths and bytes for every file below the supplied root."""

    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in asset_root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(asset_root).parts
        ),
        key=lambda path: path.relative_to(asset_root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(asset_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _validator_git_state() -> tuple[str | None, bool | None]:
    project_root = Path(__file__).resolve().parents[2]
    revision = _git_head(project_root)
    if revision is None:
        return None, None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return revision, None
    relevant = []
    for line in result.stdout.splitlines():
        path_text = line[3:].replace("\\", "/") if len(line) > 3 else line
        if path_text.startswith("results/"):
            continue
        relevant.append(line)
    return revision, bool(relevant)


def _cpu_model() -> str:
    candidates = [platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")]
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                candidates.insert(0, str(winreg.QueryValueEx(key, "ProcessorNameString")[0]))
        except (ImportError, OSError):
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    candidates.insert(0, line.split(":", 1)[1].strip())
                    break
        except OSError:
            pass
    return next((value.strip() for value in candidates if value and value.strip()), "unknown")


def _load_known_issues(
    path: str | Path | None,
    actual_upstream_commit: str | None,
    upstream_git_dirty: bool | None,
    asset_tree_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    if path is None:
        return ()
    source = Path(path).expanduser().resolve()
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("issues", []), list):
        raise ValueError(f"Known-issue registry has invalid schema: {source}")
    pinned = loaded.get("upstream_commit")
    pinned_tree = loaded.get("asset_tree_sha256")
    if pinned_tree:
        if asset_tree_sha256 != pinned_tree:
            return ()
    elif pinned:
        if actual_upstream_commit != pinned or upstream_git_dirty is not False:
            return ()
    return tuple(item for item in loaded.get("issues", []) if isinstance(item, dict))


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _scalar_joints(model: ParsedModel) -> list[JointSpec]:
    return [joint for joint in model.joints if joint.joint_type in {"revolute", "prismatic"}]


def _has_cycle(model: ParsedModel) -> bool:
    parent_by_child = {frame.name: frame.parent for frame in model.frames if frame.parent is not None}
    for start in parent_by_child:
        visited: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in visited:
                return True
            visited.add(current)
            current = parent_by_child.get(current)
    return False


def _path_has_exact_case(path: Path, asset_root: Path) -> bool | None:
    try:
        relative = path.relative_to(asset_root)
    except ValueError:
        return None
    current = asset_root
    for part in relative.parts:
        try:
            names = {child.name for child in current.iterdir()}
        except OSError:
            return None
        if part not in names:
            return False
        current = current / part
    return True


def _redact_asset_root(message: str, asset_root: Path) -> str:
    redacted = message.replace(str(asset_root), "<asset-root>")
    return redacted.replace(asset_root.as_posix(), "<asset-root>")


def _package_roots(asset_root: Path) -> dict[str, Path]:
    """Discover ROS package roots used by package:// URDF references."""

    roots: dict[str, Path] = {}
    for manifest in asset_root.rglob("package.xml"):
        try:
            root = ET.parse(manifest).getroot()
        except (ET.ParseError, OSError):
            continue
        name = root.findtext("name")
        if name:
            roots[name.strip()] = manifest.parent
    return roots


def _check_model(
    entry: DiscoveredModel,
    model: ParsedModel,
    asset_root: Path,
    config: Mapping[str, Any],
    recorder: _Recorder,
) -> None:
    label = entry.relative_path
    duplicate_joints = _duplicates(joint.name for joint in model.joints)
    duplicate_frames = _duplicates(frame.name for frame in model.frames)
    duplicate_actuators = _duplicates(actuator.name for actuator in model.actuators)
    if duplicate_joints or duplicate_frames or duplicate_actuators:
        recorder.add(
            "fail",
            "DUPLICATE_NAMES",
            label,
            "Duplicate names detected in normalized model structures.",
            details={
                "joints": duplicate_joints,
                "frames": duplicate_frames,
                "actuators": duplicate_actuators,
            },
        )
    else:
        recorder.add("pass", "UNIQUE_NAMES", label, "Joint, frame and actuator names are unique.")

    if model.format in {"urdf", "mjcf"}:
        frame_names = {frame.name for frame in model.frames}
        missing_parents = sorted(
            {frame.parent for frame in model.frames if frame.parent is not None and frame.parent not in frame_names}
        )
        invalid_joint_links = sorted(
            joint.name
            for joint in model.joints
            if (joint.parent is not None and joint.parent not in frame_names)
            or (joint.child is not None and joint.child not in frame_names)
        )
        roots = model.root_frames
        invalid_root_count = not roots or (model.format == "urdf" and len(roots) != 1)
        if missing_parents or invalid_joint_links or invalid_root_count or _has_cycle(model):
            recorder.add(
                "fail",
                "KINEMATIC_GRAPH_INVALID",
                label,
                "Kinematic frame graph is disconnected, cyclic or references missing frames.",
                details={
                    "roots": list(roots),
                    "missing_parents": missing_parents,
                    "invalid_joint_links": invalid_joint_links,
                    "cycle": _has_cycle(model),
                },
            )
        else:
            recorder.add(
                "pass",
                "KINEMATIC_GRAPH_VALID",
                label,
                f"Kinematic graph is acyclic with {len(roots)} root frame(s).",
            )

    scalar_joints = _scalar_joints(model)
    expected_per_hand = int(config["expected_active_joints_per_hand"])
    sides = ("left", "right") if entry.side == "dual" else (entry.side,)
    required_fingers = tuple(str(item) for item in config["required_fingers"])
    hand_counts = {
        side: sum(
            1
            for joint in scalar_joints
            if joint.name.startswith(f"{side}_")
            and any(f"_{finger}_" in joint.name or joint.name.startswith(f"{side}_{finger}_") for finger in required_fingers)
        )
        for side in sides
        if side in {"left", "right"}
    }
    bad_hand_counts = {side: count for side, count in hand_counts.items() if count != expected_per_hand}
    if not bad_hand_counts:
        auxiliary = len(scalar_joints) - sum(hand_counts.values())
        recorder.add(
            "pass",
            "JOINT_COUNT_EXPECTED",
            label,
            f"Found {expected_per_hand} articulated hand joints per hand and {auxiliary} auxiliary joint(s).",
        )
    else:
        recorder.add(
            "fail",
            "JOINT_COUNT_UNEXPECTED",
            label,
            f"Expected {expected_per_hand} articulated joints per hand; observed counts differ.",
            details={"expected_per_hand": expected_per_hand, "actual_by_hand": hand_counts},
        )

    bad_ranges = []
    bad_effort_velocity = []
    bad_axes = []
    non_unit_axes = []
    axis_tolerance = float(config["axis_norm_tolerance"])
    for joint in scalar_joints:
        no_authored_range = joint.lower is None and joint.upper is None
        if (model.format != "usda" and no_authored_range) or (
            not no_authored_range
            and (
                joint.lower is None
                or joint.upper is None
                or not math.isfinite(joint.lower)
                or not math.isfinite(joint.upper)
                or joint.lower >= joint.upper
            )
        ):
            bad_ranges.append(joint.name)
        if model.format == "usda":
            continue
        if joint.axis is None or not all(math.isfinite(value) for value in joint.axis):
            bad_axes.append(joint.name)
        elif float(np.linalg.norm(joint.axis)) <= 1e-12:
            bad_axes.append(joint.name)
        elif abs(float(np.linalg.norm(joint.axis)) - 1.0) > axis_tolerance:
            non_unit_axes.append(joint.name)
        if model.format == "urdf" and (
            joint.effort is None
            or joint.velocity is None
            or joint.effort <= 0.0
            or joint.velocity <= 0.0
        ):
            bad_effort_velocity.append(joint.name)

    if bad_ranges:
        recorder.add(
            "fail",
            "JOINT_LIMIT_INVALID",
            label,
            f"{len(bad_ranges)} scalar joint(s) have missing, non-finite or reversed ranges.",
            details={"joints": bad_ranges},
        )
    else:
        recorder.add("pass", "JOINT_LIMIT_VALID", label, "All scalar joint ranges are finite and ordered.")
    if bad_axes:
        recorder.add(
            "fail",
            "JOINT_AXIS_INVALID",
            label,
            f"{len(bad_axes)} scalar joint(s) have a missing, zero or non-finite axis.",
            details={"joints": bad_axes},
        )
    else:
        recorder.add("pass", "JOINT_AXIS_VALID", label, "All scalar joint axes are finite and non-zero.")
    if non_unit_axes:
        recorder.add(
            "warn",
            "JOINT_AXIS_NOT_UNIT",
            label,
            f"{len(non_unit_axes)} joint axis vector(s) are not unit length.",
            details={"joints": non_unit_axes},
        )
    if bad_effort_velocity:
        recorder.add(
            "fail",
            "URDF_LIMIT_METADATA_INVALID",
            label,
            f"{len(bad_effort_velocity)} URDF joint(s) lack positive effort/velocity limits.",
            details={"joints": bad_effort_velocity},
        )
    elif model.format == "urdf":
        recorder.add(
            "pass",
            "URDF_LIMIT_METADATA_VALID",
            label,
            "All scalar URDF joints define positive effort and velocity limits.",
        )

    if model.format in {"urdf", "mjcf"}:
        expected_terminals = {
            f"{side}_{finger}_DP"
            for side in sides
            if side in {"left", "right"}
            for finger in required_fingers
        }
        missing_terminals = sorted(expected_terminals - set(model.frame_map))
        if missing_terminals:
            recorder.add(
                "fail",
                "TERMINAL_FRAME_MISSING",
                label,
                f"Missing {len(missing_terminals)} expected distal frame(s).",
                details={"frames": missing_terminals},
            )
        else:
            recorder.add(
                "pass",
                "TERMINAL_FRAMES_PRESENT",
                label,
                f"All {len(expected_terminals)} expected distal frames are present.",
            )

    if model.format in {"urdf", "mjcf"}:
        unresolved = [asset.raw_path for asset in model.unresolved_assets]
        missing = [asset.raw_path for asset in model.missing_assets]
        outside = []
        case_mismatch = []
        for asset in model.assets:
            if asset.resolved_path is None:
                continue
            try:
                asset.resolved_path.relative_to(asset_root)
            except ValueError:
                outside.append(asset.raw_path)
            exact_case = _path_has_exact_case(asset.resolved_path, asset_root) if asset.exists else None
            if exact_case is False:
                case_mismatch.append(asset.raw_path)
        if unresolved:
            recorder.add(
                "fail",
                "ASSET_REFERENCE_UNRESOLVED",
                label,
                f"{len(unresolved)} asset reference(s) could not be resolved.",
                details={"references": sorted(set(unresolved))},
            )
        if missing:
            unique_missing = sorted(set(missing))
            message = f"{len(unique_missing)} unique mesh reference(s) are missing; examples: {', '.join(unique_missing[:3])}"
            recorder.add(
                "fail",
                "MESH_REFERENCE_MISSING",
                label,
                message,
                details={"references": unique_missing},
            )
        if outside:
            recorder.add(
                "warn",
                "ASSET_PATH_OUTSIDE_ROOT",
                label,
                f"{len(outside)} asset path(s) resolve outside the supplied asset root.",
                details={"references": sorted(set(outside))},
            )
        if case_mismatch:
            recorder.add(
                "warn",
                "ASSET_PATH_CASE_MISMATCH",
                label,
                f"{len(case_mismatch)} asset reference(s) differ from on-disk path case.",
                details={"references": sorted(set(case_mismatch))},
            )
        if not unresolved and not missing and not case_mismatch:
            recorder.add(
                "pass",
                "ASSET_REFERENCES_VALID",
                label,
                f"All {len(model.assets)} file-backed asset references resolve with exact case.",
            )

    if model.format == "mjcf":
        joint_names = {joint.name for joint in scalar_joints}
        bound = {actuator.joint for actuator in model.actuators if actuator.joint}
        invalid_targets = sorted(
            actuator.name
            for actuator in model.actuators
            if actuator.joint is None or actuator.joint not in joint_names
        )
        missing_bindings = sorted(joint_names - bound)
        bad_control_ranges = sorted(
            actuator.name
            for actuator in model.actuators
            if (actuator.control_lower is None) != (actuator.control_upper is None)
            or (
                actuator.control_lower is not None
                and actuator.control_upper is not None
                and actuator.control_lower >= actuator.control_upper
            )
        )
        if invalid_targets or missing_bindings or bad_control_ranges:
            recorder.add(
                "fail",
                "ACTUATOR_BINDING_INVALID",
                label,
                "MJCF actuator bindings do not provide one valid target for every scalar joint.",
                details={
                    "invalid_targets": invalid_targets,
                    "missing_joint_bindings": missing_bindings,
                    "invalid_control_ranges": bad_control_ranges,
                },
            )
        else:
            recorder.add(
                "pass",
                "ACTUATOR_BINDINGS_VALID",
                label,
                f"All {len(model.actuators)} actuators target valid scalar joints with full coverage.",
            )


def _joint_maps(models: Mapping[str, ParsedModel]) -> dict[str, dict[str, JointSpec]]:
    return {
        model_format: {joint.name: joint for joint in _scalar_joints(model)}
        for model_format, model in models.items()
    }


def _compare_group(
    variant: str,
    entries: list[DiscoveredModel],
    parsed: Mapping[str, ParsedModel],
    config: Mapping[str, Any],
    recorder: _Recorder,
) -> dict[str, Any]:
    position_entries = [entry for entry in entries if entry.control_mode == "position"]
    by_format = {
        entry.format: parsed[entry.relative_path]
        for entry in position_entries
        if entry.relative_path in parsed
    }
    required = set(str(item) for item in config["required_formats"])
    missing_formats = sorted(required - set(by_format))
    group_status = "pass"
    if missing_formats:
        recorder.add(
            "fail",
            "FORMAT_MATRIX_INCOMPLETE",
            variant,
            f"Missing required format(s): {', '.join(missing_formats)}.",
        )
        group_status = "fail"
    else:
        recorder.add("pass", "FORMAT_MATRIX_COMPLETE", variant, "URDF, MJCF and position USDA are present.")

    joint_maps = _joint_maps(by_format)
    common: set[str] = set.intersection(*(set(value) for value in joint_maps.values())) if joint_maps else set()
    union: set[str] = set.union(*(set(value) for value in joint_maps.values())) if joint_maps else set()
    missing_by_format = {
        model_format: sorted(union - set(joints))
        for model_format, joints in joint_maps.items()
        if union - set(joints)
    }
    missing_count = sum(len(value) for value in missing_by_format.values())
    if missing_count:
        recorder.add(
            "fail",
            "JOINT_SCHEMA_MISMATCH",
            variant,
            f"Cross-format joint sets differ in {missing_count} format/joint occurrence(s).",
            details={"missing_by_format": missing_by_format},
        )
        group_status = "fail"
    elif joint_maps:
        recorder.add(
            "pass",
            "JOINT_SCHEMA_MATCH",
            variant,
            f"All available formats share the same {len(common)} scalar joint names.",
        )

    max_limit_delta = 0.0
    incomplete_limit_pairs: list[str] = []
    formats = sorted(joint_maps)
    for joint_name in sorted(common):
        for first_index, first_format in enumerate(formats):
            for second_format in formats[first_index + 1 :]:
                first = joint_maps[first_format][joint_name]
                second = joint_maps[second_format][joint_name]
                if None in (first.lower, first.upper, second.lower, second.upper):
                    incomplete_limit_pairs.append(f"{joint_name}:{first_format}/{second_format}")
                    continue
                assert first.lower is not None and first.upper is not None
                assert second.lower is not None and second.upper is not None
                max_limit_delta = max(
                    max_limit_delta,
                    abs(first.lower - second.lower),
                    abs(first.upper - second.upper),
                )
    limit_tolerance = float(config["joint_limit_tolerance_rad"])
    if max_limit_delta > limit_tolerance:
        recorder.add(
            "fail",
            "JOINT_LIMIT_PARITY_MISMATCH",
            variant,
            f"Maximum cross-format limit delta is {math.degrees(max_limit_delta):.6g} deg.",
            details={
                "tolerance_deg": math.degrees(limit_tolerance),
                "inherited_or_unavailable_pairs": incomplete_limit_pairs,
            },
        )
        group_status = "fail"
    elif incomplete_limit_pairs:
        recorder.add(
            "warn",
            "JOINT_LIMIT_PARITY_INCOMPLETE",
            variant,
            f"Authored limits match within tolerance, but {len(incomplete_limit_pairs)} format/joint pair(s) inherit or omit limits.",
            details={"inherited_or_unavailable_pairs": incomplete_limit_pairs},
        )
        if group_status == "pass":
            group_status = "warn"
    elif joint_maps:
        recorder.add(
            "pass",
            "JOINT_LIMIT_PARITY_MATCH",
            variant,
            f"Maximum cross-format limit delta is {math.degrees(max_limit_delta):.6g} deg.",
        )

    max_axis_delta = 0.0
    if "urdf" in joint_maps and "mjcf" in joint_maps:
        for name in set(joint_maps["urdf"]) & set(joint_maps["mjcf"]):
            first = joint_maps["urdf"][name].axis
            second = joint_maps["mjcf"][name].axis
            if first is None or second is None:
                max_axis_delta = math.inf
                break
            first_array = np.asarray(first, dtype=float)
            second_array = np.asarray(second, dtype=float)
            first_norm = float(np.linalg.norm(first_array))
            second_norm = float(np.linalg.norm(second_array))
            if first_norm <= 1e-12 or second_norm <= 1e-12:
                max_axis_delta = math.inf
                break
            first_array /= first_norm
            second_array /= second_norm
            max_axis_delta = max(max_axis_delta, float(np.linalg.norm(first_array - second_array)))
        if max_axis_delta > float(config["axis_norm_tolerance"]):
            recorder.add(
                "fail",
                "JOINT_AXIS_PARITY_MISMATCH",
                variant,
                f"Maximum normalized URDF/MJCF axis delta is {max_axis_delta:.6g}.",
            )
            group_status = "fail"
        else:
            recorder.add(
                "pass",
                "JOINT_AXIS_PARITY_MATCH",
                variant,
                f"Maximum normalized URDF/MJCF axis delta is {max_axis_delta:.6g}.",
            )

    position_usda = next(
        (parsed[item.relative_path] for item in entries if item.format == "usda" and item.control_mode == "position" and item.relative_path in parsed),
        None,
    )
    mit_usda = next(
        (parsed[item.relative_path] for item in entries if item.format == "usda" and item.control_mode == "mit" and item.relative_path in parsed),
        None,
    )
    if position_usda is not None and mit_usda is not None:
        position_map = position_usda.joint_map
        mit_map = mit_usda.joint_map
        limit_delta = 0.0
        if set(position_map) == set(mit_map):
            for name in position_map:
                first, second = position_map[name], mit_map[name]
                if None not in (first.lower, first.upper, second.lower, second.upper):
                    assert first.lower is not None and first.upper is not None
                    assert second.lower is not None and second.upper is not None
                    limit_delta = max(
                        limit_delta,
                        abs(first.lower - second.lower),
                        abs(first.upper - second.upper),
                    )
        else:
            limit_delta = math.inf
        if set(position_map) == set(mit_map):
            recorder.add(
                "pass",
                "USDA_MODE_OVERLAY_SCHEMA_MATCH",
                variant,
                f"Position and MIT-mode USDA overlays share joint names; maximum authored limit delta is {math.degrees(limit_delta):.6g} deg.",
            )
        else:
            recorder.add(
                "fail",
                "USDA_MODE_OVERLAY_SCHEMA_MISMATCH",
                variant,
                "Position and MIT-mode USDA overlays differ in joint names.",
            )
            group_status = "fail"
    else:
        recorder.add(
            "fail",
            "USDA_MODE_OVERLAY_MISSING",
            variant,
            "Position or MIT-mode USDA overlay is missing or failed to parse.",
        )
        group_status = "fail"

    return {
        "variant": variant,
        "formats": sorted(by_format),
        "common_joint_count": len(common),
        "missing_joint_count": missing_count,
        "missing_by_format": missing_by_format,
        "max_limit_delta_deg": math.degrees(max_limit_delta),
        "incomplete_limit_pair_count": len(incomplete_limit_pairs),
        "max_axis_delta": max_axis_delta,
        "status": group_status,
    }


def _summary(checks: Iterable[AuditCheck]) -> dict[str, Any]:
    counts = {status: 0 for status in ("pass", "known", "warn", "fail", "skip")}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    if counts["fail"]:
        status = "fail"
    elif counts["known"]:
        status = "known"
    elif counts["warn"]:
        status = "warn"
    else:
        status = "pass"
    return {"status": status, **counts, "total_checks": sum(counts.values())}


def run_audit(
    asset_root: str | Path,
    *,
    config_path: str | Path | None = None,
    known_issues_path: str | Path | None = None,
    upstream_commit: str | None = None,
    expected_asset_tree_sha256: str | None = None,
    run_simulation: bool = True,
    simulation_steps: int = 50,
    fk_samples: int = 256,
    fk_seed: int = 20260826,
) -> dict[str, Any]:
    """Run the complete audit and return a report-ready dictionary."""

    root = Path(asset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Asset root is not a directory: {root}")
    if simulation_steps < 1:
        raise ValueError("simulation_steps must be at least one")
    if fk_samples < 1:
        raise ValueError("fk_samples must be at least one")

    config = _load_config(config_path)
    actual_upstream_commit = _git_head(root)
    upstream_git_dirty = _git_dirty(root)
    asset_tree_sha256 = _asset_tree_sha256(root)
    if upstream_commit and actual_upstream_commit and upstream_commit != actual_upstream_commit:
        raise ValueError(
            "Supplied upstream commit does not match the asset checkout HEAD: "
            f"expected {upstream_commit}, actual {actual_upstream_commit}"
        )
    if upstream_commit and actual_upstream_commit is None and expected_asset_tree_sha256 is None:
        raise ValueError(
            "Cannot verify the supplied upstream commit outside a Git checkout without an expected asset-tree hash."
        )
    if expected_asset_tree_sha256 and asset_tree_sha256 != expected_asset_tree_sha256:
        raise ValueError(
            "Asset-tree hash mismatch: "
            f"expected {expected_asset_tree_sha256}, actual {asset_tree_sha256}"
        )
    effective_upstream_commit = actual_upstream_commit or upstream_commit
    validator_commit, validator_dirty = _validator_git_state()
    known = _load_known_issues(
        known_issues_path,
        actual_upstream_commit,
        upstream_git_dirty,
        asset_tree_sha256,
    )
    recorder = _Recorder(known)
    discovered = discover_model_files(root)
    if not discovered:
        recorder.add(
            "fail",
            "NO_MODELS_DISCOVERED",
            root.name,
            "No supported URDF, MJCF or USDA model entry points were found.",
        )
    package_roots = _package_roots(root)
    parsed: dict[str, ParsedModel] = {}
    model_summaries: list[dict[str, Any]] = []

    for entry in discovered:
        try:
            model = parse_model(entry.path, package_roots=package_roots)
        except ModelParseError as error:
            recorder.add("fail", "MODEL_PARSE_FAILED", entry.relative_path, error.detail)
            model_summaries.append(
                {
                    "path": entry.relative_path,
                    "format": entry.format,
                    "variant": entry.canonical_name,
                    "status": "fail",
                    "message": error.detail,
                }
            )
            continue
        parsed[entry.relative_path] = model
        recorder.add("pass", "MODEL_PARSE_OK", entry.relative_path, f"Parsed {entry.format.upper()} entry point.")
        _check_model(entry, model, root, config, recorder)
        model_summaries.append(
            {
                "path": entry.relative_path,
                "format": entry.format,
                "variant": entry.canonical_name,
                "control_mode": entry.control_mode,
                "links": len(model.links),
                "frames": len(model.frames),
                "joints": len(model.joints),
                "scalar_joints": len(_scalar_joints(model)),
                "actuators": len(model.actuators),
                "asset_references": len(model.assets),
                "status": "parsed",
            }
        )

    comparisons = [
        _compare_group(variant, entries, parsed, config, recorder)
        for variant, entries in group_by_canonical_name(discovered).items()
    ]

    fk_comparisons: list[dict[str, Any]] = []
    for variant, entries in group_by_canonical_name(discovered).items():
        if any(entry.side == "dual" or entry.floating_base for entry in entries):
            continue
        urdf_entry = next((entry for entry in entries if entry.format == "urdf"), None)
        mjcf_entry = next((entry for entry in entries if entry.format == "mjcf"), None)
        if (
            urdf_entry is None
            or mjcf_entry is None
            or urdf_entry.relative_path not in parsed
            or mjcf_entry.relative_path not in parsed
        ):
            continue
        result = compare_forward_kinematics(
            parsed[urdf_entry.relative_path],
            parsed[mjcf_entry.relative_path],
            samples=fk_samples,
            seed=fk_seed,
        )
        record = result.to_dict()
        record["variant"] = variant
        fk_comparisons.append(record)
        if result.status == "skip":
            record["qa_status"] = "fail"
            recorder.add("fail", "FK_COMPARISON_FAILED", variant, result.message)
        else:
            position_bad = (
                result.position_max_mm is not None
                and result.position_max_mm > float(config["fk_position_warning_mm"])
            )
            orientation_bad = (
                result.orientation_max_deg is not None
                and result.orientation_max_deg > float(config["fk_orientation_warning_deg"])
            )
            if position_bad or orientation_bad:
                record["qa_status"] = "warn"
                recorder.add(
                    "warn",
                    "FK_PARITY_DELTA",
                    variant,
                    "Sampled distal-frame FK exceeds the reporting threshold after one fixed base alignment.",
                    details={
                        "position_max_mm": result.position_max_mm,
                        "orientation_max_deg": result.orientation_max_deg,
                        "position_threshold_mm": config["fk_position_warning_mm"],
                        "orientation_threshold_deg": config["fk_orientation_warning_deg"],
                    },
                )
            else:
                record["qa_status"] = "pass"
                recorder.add(
                    "pass",
                    "FK_PARITY_WITHIN_THRESHOLD",
                    variant,
                    "Sampled distal-frame FK is within the reporting threshold.",
                )

    simulations: list[dict[str, Any]] = []
    for entry in discovered:
        if entry.format != "mjcf":
            continue
        if run_simulation:
            result = run_mujoco_smoke(entry.path, steps=simulation_steps)
            record = result.to_dict()
            record.pop("source_path", None)
            record["model"] = entry.relative_path
            safe_message = _redact_asset_root(result.message, root)
            record["message"] = safe_message
            if result.status == "pass":
                recorder.add("pass", "MUJOCO_SMOKE_OK", entry.relative_path, safe_message)
            elif result.status == "skip":
                recorder.add("skip", "MUJOCO_UNAVAILABLE", entry.relative_path, safe_message)
            else:
                check = recorder.add("fail", "MUJOCO_LOAD_FAILED", entry.relative_path, safe_message)
                record["status"] = check.status
                if check.reference:
                    record["reference"] = check.reference
            simulations.append(record)
        else:
            recorder.add("skip", "MUJOCO_SMOKE_DISABLED", entry.relative_path, "MuJoCo checks disabled by caller.")

    try:
        import mujoco

        mujoco_version = getattr(mujoco, "__version__", None)
    except ImportError:
        mujoco_version = None

    checks = [check.to_dict() for check in recorder.checks]
    report = {
        "schema_version": 1,
        "metadata": {
            "run_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "validator_version": __version__,
            "validator_commit": validator_commit,
            "validator_dirty_excluding_results": validator_dirty,
            "upstream_repository": "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml",
            "upstream_commit": effective_upstream_commit,
            "upstream_commit_verified_from_git": (
                actual_upstream_commit is not None and upstream_git_dirty is False
            ),
            "upstream_git_dirty": upstream_git_dirty,
            "asset_tree_sha256": asset_tree_sha256,
            "asset_tree_matches_expected": (
                asset_tree_sha256 == expected_asset_tree_sha256
                if expected_asset_tree_sha256 is not None
                else None
            ),
            "asset_tree_hash_algorithm": "sha256(path_length || relative_path || file_size || file_bytes)",
            "asset_root": root.name,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco_version,
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "cpu_logical_count": os.cpu_count(),
            "fk_samples": fk_samples,
            "fk_seed": fk_seed,
            "simulation_steps": simulation_steps if run_simulation else 0,
        },
        "summary": _summary(recorder.checks),
        "inventory": inventory_counts(discovered),
        "models": sorted(model_summaries, key=lambda item: item["path"]),
        "comparisons": comparisons,
        "fk_comparisons": fk_comparisons,
        "mujoco_smoke": simulations,
        "checks": checks,
        "findings": [check for check in checks if check["status"] != "pass"],
    }
    return report


__all__ = ["AuditCheck", "DEFAULT_CONFIG", "run_audit"]
