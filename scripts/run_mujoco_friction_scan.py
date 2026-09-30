#!/usr/bin/env python3
"""Run a bounded local MuJoCo dry-friction diagnostic against formal OV evidence.

This is deliberately not a Gate 0 acceptance run.  It keeps one verified
OVPhysX trace frozen, changes only the in-memory MuJoCo ``dof_frictionloss``
array, and records enough provenance to decide whether a larger bilateral
friction experiment is justified.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

import wave_asset_qa
from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.parity.bundle import (
    sha256_file,
    verify_bundle,
    write_bundle_manifest,
    write_json_atomic,
)
from wave_asset_qa.parity.compare import CollectedRun, trace_delta
from wave_asset_qa.parity.contracts import HandSpec, ParityManifest, Simulator
from wave_asset_qa.parity.runner import load_collected_runs, run_backend_cases
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    canonical_initial_positions,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPERIMENT_ID = "mujoco-dry-friction-scale-v1"
AUTHORITATIVE_FORMAL_ROOT_SHA256 = (
    "d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505"
)
AUTHORITATIVE_FORMAL_SOURCE_REVISION = "c571d9bb37a619fe9e867447d3be448acc615e3c"
AUTHORITATIVE_FORMAL_MANIFEST_SHA256 = (
    "576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e"
)
AUTHORITATIVE_OVPHYSX_RUN_SHA256 = (
    "008ed804406ee32e58a5b701bc9feb9f4123a23bc13d521392a2525d0de2dc30"
)
FORMAL_MUJOCO_CASE_ID = "mujoco.left.small_step.base.r01"
FORMAL_OVPHYSX_CASE_ID = "ovphysx.left.small_step.base.r01"
DIAGNOSTIC_SCALES = (1.0, 0.5, 0.0)
SELECTED_JOINT = "left_thumb_CMC_FE"
SELECTED_FRAME = "left_thumb_DP"
BASELINE_DRIFT_TOLERANCE = 1e-9
MIN_JOINT_REDUCTION_PER_SCALE_STEP_RAD = 1e-5
MIN_FRAME_REDUCTION_PER_SCALE_STEP_M = 1e-6
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class FrictionScanError(RuntimeError):
    """Raised when the diagnostic cannot preserve its evidence contract."""


def _git_text(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FrictionScanError(f"cannot verify diagnostic Git source: {detail}")
    return completed.stdout.strip()


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    root = project_root.resolve(strict=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise FrictionScanError(
            f"diagnostic project root is not the Git top level: {root} != {top_level}"
        )
    if _HEX40.fullmatch(source_revision) is None:
        raise FrictionScanError("source_revision must be a lowercase 40-character commit")
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise FrictionScanError(
            f"source_revision does not match local HEAD: {source_revision} != {head}"
        )
    tree = _git_text(root, "rev-parse", "--verify", f"{source_revision}^{{tree}}")
    if _HEX40.fullmatch(tree) is None:
        raise FrictionScanError("diagnostic source commit did not resolve to a Git tree")
    tracked_changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_changes:
        raise FrictionScanError("tracked worktree/index must be clean for the diagnostic")
    untracked = _git_text(
        root,
        "ls-files",
        "--others",
        "--",
        "src/wave_asset_qa",
        "scripts/run_mujoco_friction_scan.py",
        "configs/parity/gate0.json",
    ).splitlines()
    sensitive_untracked = [
        item
        for item in untracked
        if "__pycache__" not in Path(item).parts and Path(item).suffix != ".pyc"
    ]
    if sensitive_untracked:
        raise FrictionScanError(
            "untracked files exist in diagnostic source/input paths: "
            + ", ".join(sensitive_untracked)
        )
    script = Path(__file__).resolve(strict=True)
    relative_script = script.relative_to(root).as_posix()
    _git_text(root, "ls-files", "--error-unmatch", "--", relative_script)
    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise FrictionScanError(
            "wave_asset_qa was imported from a source tree other than this checkout"
        )
    return tree


def _create_output_root(path: str | Path) -> Path:
    raw = Path(os.path.abspath(Path(path).expanduser()))
    if raw.exists() or _is_link_like(raw):
        raise FrictionScanError(f"output path already exists; refusing overwrite: {raw}")
    lexical_parent = raw.parent
    for ancestor in (lexical_parent, *lexical_parent.parents):
        if ancestor.exists() and _is_link_like(ancestor):
            raise FrictionScanError(
                f"output path has a symbolic-link or junction ancestor: {ancestor}"
            )
    parent = lexical_parent.resolve(strict=True)
    if not parent.is_dir():
        raise FrictionScanError("output parent must be a real existing directory")
    raw.mkdir()
    root = raw.resolve(strict=True)
    if root.parent != parent or root.is_symlink():
        raise FrictionScanError("output directory did not resolve below its requested parent")
    return root


def _validated_output_path(project_root: Path, path: str | Path) -> Path:
    """Confine immutable diagnostic outputs to one direct child of results/."""

    results_root = (project_root / "results").resolve(strict=True)
    raw = Path(os.path.abspath(Path(path).expanduser()))
    for ancestor in (raw.parent, *raw.parent.parents):
        if ancestor.exists() and _is_link_like(ancestor):
            raise FrictionScanError(
                f"diagnostic output has a symbolic-link or junction ancestor: {ancestor}"
            )
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise FrictionScanError(
            "diagnostic output parent must be the existing project results directory"
        ) from exc
    if parent != results_root:
        raise FrictionScanError(
            "diagnostic output must be a direct child of the project results directory"
        )
    return raw


def _create_staging_root(final_path: Path) -> Path:
    """Create a visibly incomplete sibling that can be promoted after verification."""

    if final_path.exists() or _is_link_like(final_path):
        raise FrictionScanError(
            f"output path already exists; refusing overwrite: {final_path}"
        )
    prefix = f".{final_path.name}.incomplete-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=final_path.parent))
    resolved = staging.resolve(strict=True)
    if resolved.parent != final_path.parent.resolve(strict=True) or _is_link_like(
        resolved
    ):
        raise FrictionScanError("staging directory escaped the approved results root")
    return resolved


def _promote_staging_root(staging: Path, final_path: Path) -> Path:
    if staging.parent != final_path.parent.resolve(strict=True):
        raise FrictionScanError("staging and final diagnostic paths are not siblings")
    if final_path.exists() or _is_link_like(final_path):
        raise FrictionScanError(
            f"output path appeared during the run; refusing overwrite: {final_path}"
        )
    try:
        os.rename(staging, final_path)
    except OSError as exc:
        raise FrictionScanError(f"cannot promote verified diagnostic output: {exc}") from exc
    promoted = final_path.resolve(strict=True)
    if promoted.parent != staging.parent:
        raise FrictionScanError("promoted diagnostic output escaped the results root")
    return promoted


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


def _write_text_exclusive(path: Path, content: str) -> Path:
    if path.exists() or path.is_symlink():
        raise FrictionScanError(f"refusing to overwrite diagnostic artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _bundle_payload_paths(root: Path) -> list[str]:
    root_manifest = root / "bundle.json"
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root_manifest
    )


def _verify_exact_bundle(root: Path) -> dict[str, object]:
    verification = dict(verify_bundle(root))
    try:
        manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        records = manifest["files"]
        listed = {record["path"] for record in records}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrictionScanError(f"cannot audit diagnostic bundle inventory: {exc}") from exc
    actual = set(_bundle_payload_paths(root))
    if listed != actual:
        raise FrictionScanError(
            "diagnostic bundle inventory is not exact: "
            f"missing={sorted(actual - listed)}, extra={sorted(listed - actual)}"
        )
    verification["exact_inventory"] = True
    return verification


def _verify_copied_formal_subset(
    copied_manifest: Path,
    *,
    expected_root_sha256: str,
    copied_payloads: Mapping[str, Path],
) -> None:
    try:
        manifest = json.loads(copied_manifest.read_text(encoding="utf-8"))
        if manifest["root_sha256"] != expected_root_sha256:
            raise FrictionScanError("copied formal manifest has the wrong root hash")
        records = {record["path"]: record for record in manifest["files"]}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrictionScanError(f"cannot audit copied formal manifest: {exc}") from exc
    for formal_path, copied_path in copied_payloads.items():
        record = records.get(formal_path)
        if not isinstance(record, Mapping):
            raise FrictionScanError(
                f"formal manifest does not list copied payload: {formal_path}"
            )
        if (
            record.get("size") != copied_path.stat().st_size
            or record.get("sha256") != sha256_file(copied_path)
        ):
            raise FrictionScanError(
                f"copied formal payload differs from its manifest: {formal_path}"
            )


def _formal_run_path(bundle: Path, case_id: str) -> Path:
    backend = case_id.split(".", 1)[0]
    return bundle / "evidence" / backend / "cases" / case_id / f"{case_id}.run.json"


def _load_one(path: Path) -> CollectedRun:
    runs = load_collected_runs(path)
    if len(runs) != 1:
        raise FrictionScanError(f"expected exactly one collected run in {path}")
    run = runs[0]
    if not run.result.completed:
        raise FrictionScanError(f"baseline run is not completed: {run.case.case_id}")
    return run


def _time_key(value: float) -> int:
    return round(float(value) * 1e12)


def _samples_by_time(result: AdapterRunResult) -> dict[int, TraceSample]:
    samples: dict[int, TraceSample] = {}
    for sample in result.samples:
        key = _time_key(sample.time_s)
        if key in samples:
            raise FrictionScanError("trace contains duplicate sample timestamps")
        samples[key] = sample
    return samples


def _position_distance(pose_a: Sequence[float], pose_b: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(pose_a[:3], pose_b[:3]))
    )


def _peak_frame_delta(
    ov_result: AdapterRunResult,
    mujoco_result: AdapterRunResult,
    hand: HandSpec,
) -> dict[str, object]:
    mujoco_by_time = _samples_by_time(mujoco_result)
    best_distance = -1.0
    best: dict[str, object] | None = None
    for ov_sample in ov_result.samples:
        mujoco_sample = mujoco_by_time.get(_time_key(ov_sample.time_s))
        if mujoco_sample is None:
            raise FrictionScanError(
                f"MuJoCo trace has no sample at t={ov_sample.time_s:.12g}s"
            )
        for frame_name in hand.distal_frame_names:
            ov_pose = ov_sample.frame_poses[frame_name]
            mujoco_pose = mujoco_sample.frame_poses[frame_name]
            distance = _position_distance(ov_pose, mujoco_pose)
            if distance > best_distance:
                best_distance = distance
                best = {
                    "frame_name": frame_name,
                    "step": ov_sample.step,
                    "time_s": ov_sample.time_s,
                    "position_distance_m": distance,
                    "ovphysx_xyz_m": [float(value) for value in ov_pose[:3]],
                    "mujoco_xyz_m": [float(value) for value in mujoco_pose[:3]],
                    "ov_minus_mujoco_xyz_m": [
                        float(ov_pose[index]) - float(mujoco_pose[index])
                        for index in range(3)
                    ],
                }
    if best is None:
        raise FrictionScanError("cannot locate a peak in an empty trace")
    return best


def _sample_at(result: AdapterRunResult, time_s: float) -> TraceSample:
    sample = _samples_by_time(result).get(_time_key(time_s))
    if sample is None:
        raise FrictionScanError(f"trace has no sample at t={time_s:.12g}s")
    return sample


def _compiled_joint(result: AdapterRunResult, joint_name: str) -> Mapping[str, object]:
    compiled = result.provenance.get("compiled_control_parameters")
    if not isinstance(compiled, Mapping):
        raise FrictionScanError("MuJoCo result lacks compiled control provenance")
    joints = compiled.get("joints")
    if not isinstance(joints, list):
        raise FrictionScanError("compiled control provenance lacks its joint list")
    matches = [
        item
        for item in joints
        if isinstance(item, Mapping) and item.get("canonical_id") == joint_name
    ]
    if len(matches) != 1:
        raise FrictionScanError(f"compiled provenance does not identify {joint_name}")
    return matches[0]


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrictionScanError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise FrictionScanError(f"{context} must be a finite number")
    return number


def _finite_number_array(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise FrictionScanError(f"{context} must be a non-empty array")
    numbers = tuple(
        _finite_number(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if any(item < 0.0 for item in numbers):
        raise FrictionScanError(f"{context} must contain non-negative values")
    return numbers


def _validate_diagnostic_run(
    run: CollectedRun,
    *,
    expected_case: ScenarioCase,
    scale: float,
    session_id: str,
    source_revision: str,
    manifest_digest: str,
    manifest: ParityManifest,
    expected_source_sha256: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate that one result really is the requested single-variable run."""

    if run.case != expected_case:
        raise FrictionScanError(f"diagnostic scale {scale} returned the wrong case")
    if not run.result.completed:
        raise FrictionScanError(f"diagnostic scale {scale} is not completed")
    hand = manifest.hand(run.case.hand)
    if run.result.backend != Simulator.MUJOCO.value:
        raise FrictionScanError(f"diagnostic scale {scale} is not a MuJoCo result")
    if run.result.scenario_id != run.case.scenario_id:
        raise FrictionScanError(f"diagnostic scale {scale} has the wrong scenario")
    scenario = manifest.scenario(run.case.scenario_id)
    expected_steps = round(scenario.duration_s / run.case.dt_s)
    if not math.isclose(run.result.dt, run.case.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise FrictionScanError(f"diagnostic scale {scale} has the wrong timestep")
    if (
        run.result.requested_steps != expected_steps
        or run.result.completed_steps != expected_steps
        or len(run.result.samples) != expected_steps + 1
        or run.result.error is not None
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} does not contain the complete canonical trace"
        )
    if run.result.joint_names != hand.joint_names:
        raise FrictionScanError(f"diagnostic scale {scale} has the wrong joint order")
    if run.result.frame_names != hand.distal_frame_names:
        raise FrictionScanError(f"diagnostic scale {scale} has the wrong frame order")
    canonical_initial = canonical_initial_positions(scenario, hand.joint_names)
    for expected_step, sample in enumerate(run.result.samples):
        expected_time = expected_step * run.case.dt_s
        if sample.step != expected_step or not math.isclose(
            sample.time_s, expected_time, rel_tol=0.0, abs_tol=1e-12
        ):
            raise FrictionScanError(
                f"diagnostic scale {scale} has a non-canonical sample time axis"
            )
        expected_targets = canonical_position_targets(
            scenario,
            hand.joint_names,
            step_index=expected_step,
            dt_s=run.case.dt_s,
        )
        if dict(sample.position_targets) != expected_targets:
            raise FrictionScanError(
                f"diagnostic scale {scale} has non-canonical position targets"
            )
        if set(sample.joint_positions) != set(hand.joint_names):
            raise FrictionScanError(
                f"diagnostic scale {scale} has incomplete sampled joints"
            )
        if set(sample.frame_poses) != set(hand.distal_frame_names):
            raise FrictionScanError(
                f"diagnostic scale {scale} has incomplete sampled frames"
            )
        ordered_joint_positions = tuple(
            float(sample.joint_positions[name]) for name in hand.joint_names
        )
        if tuple(sample.qpos) != ordered_joint_positions:
            raise FrictionScanError(
                f"diagnostic scale {scale} qpos and canonical joint samples differ"
            )
        numeric_values = (
            sample.time_s,
            *sample.qpos,
            *sample.qvel,
            *ordered_joint_positions,
            *(value for pose in sample.frame_poses.values() for value in pose),
        )
        if (
            len(sample.qpos) != len(hand.joint_names)
            or len(sample.qvel) != len(hand.joint_names)
            or any(len(pose) != 7 for pose in sample.frame_poses.values())
            or any(not math.isfinite(float(value)) for value in numeric_values)
            or sample.contact_count != 0
        ):
            raise FrictionScanError(
                f"diagnostic scale {scale} contains invalid sampled state"
            )
    initial_sample = run.result.samples[0]
    if (
        dict(initial_sample.joint_positions) != canonical_initial
        or tuple(initial_sample.qpos)
        != tuple(canonical_initial[name] for name in hand.joint_names)
        or any(value != 0.0 for value in initial_sample.qvel)
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} does not start from the canonical state"
        )

    provenance = run.result.provenance
    if not isinstance(provenance, Mapping):
        raise FrictionScanError(f"diagnostic scale {scale} lacks provenance")
    expected_values: dict[str, object] = {
        "manifest_sha256": manifest_digest,
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "backend": Simulator.MUJOCO.value,
        "device": "cpu",
        "headless": True,
        "renderer": False,
        "source_path": run.case.model_path,
        "source_sha256": expected_source_sha256,
        "frame_pose_source": "mujoco_data_xpos_xquat_after_mj_kinematics",
        "dof_frictionloss_mode": (
            "canonical_no_override" if scale == 1.0 else "diagnostic_scaled_override"
        ),
        "dof_frictionloss_scope": "all_compiled_dofs",
        "dof_frictionloss_order": "model_dof_index_ascending",
        "mapping_schema_version": 1,
        "frame_pose_state_alignment": "current_sample_qpos",
    }
    for field, expected in expected_values.items():
        if provenance.get(field) != expected:
            raise FrictionScanError(
                f"diagnostic scale {scale} provenance.{field} does not match"
            )
    backend_joint_names = provenance.get("backend_joint_names")
    backend_frame_names = provenance.get("backend_frame_names")
    joint_mapping = provenance.get("joint_mapping")
    frame_mapping = provenance.get("frame_mapping")
    if backend_joint_names != list(hand.joint_names):
        raise FrictionScanError(
            f"diagnostic scale {scale} backend joint order is inconsistent"
        )
    if (
        not isinstance(backend_frame_names, list)
        or len(backend_frame_names) != len(set(backend_frame_names))
        or not set(hand.distal_frame_names).issubset(backend_frame_names)
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} backend frame inventory is inconsistent"
        )
    if (
        not isinstance(joint_mapping, list)
        or len(joint_mapping) != len(hand.joint_names)
        or [
            item.get("canonical_id") if isinstance(item, Mapping) else None
            for item in joint_mapping
        ]
        != list(hand.joint_names)
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} canonical joint mapping is inconsistent"
        )
    if (
        not isinstance(frame_mapping, list)
        or len(frame_mapping) != len(hand.distal_frame_names)
        or [
            item.get("canonical_id") if isinstance(item, Mapping) else None
            for item in frame_mapping
        ]
        != list(hand.distal_frame_names)
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} canonical frame mapping is inconsistent"
        )
    recorded_scale = _finite_number(
        provenance.get("dof_frictionloss_scale"),
        f"diagnostic scale {scale} provenance.dof_frictionloss_scale",
    )
    if not math.isclose(recorded_scale, scale, rel_tol=0.0, abs_tol=0.0):
        raise FrictionScanError(
            f"diagnostic scale {scale} recorded a different friction scale"
        )

    original = _finite_number_array(
        provenance.get("dof_frictionloss_original"),
        f"diagnostic scale {scale} provenance.dof_frictionloss_original",
    )
    effective = _finite_number_array(
        provenance.get("dof_frictionloss_effective"),
        f"diagnostic scale {scale} provenance.dof_frictionloss_effective",
    )
    indices = provenance.get("dof_frictionloss_dof_indices")
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} friction DOF indices must be integers"
        )
    if indices != list(range(len(original))):
        raise FrictionScanError(
            f"diagnostic scale {scale} friction DOF indices are incomplete or unordered"
        )
    if len(effective) != len(original):
        raise FrictionScanError(
            f"diagnostic scale {scale} friction arrays have different lengths"
        )
    for index, (before, after) in enumerate(zip(original, effective)):
        if not math.isclose(
            after,
            before * scale,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise FrictionScanError(
                f"diagnostic scale {scale} effective friction differs at DOF {index}"
            )

    compiled = provenance.get("compiled_control_parameters")
    if not isinstance(compiled, Mapping):
        raise FrictionScanError(
            f"diagnostic scale {scale} lacks compiled control provenance"
        )
    expected_compiled_values = {
        "schema_version": 1,
        "source": "mujoco_compiled_mjmodel_arrays",
        "semantics": "compiled_configuration_only_not_measured_torque",
        "contains_measured_or_realized_torque": False,
        "joint_order": list(hand.joint_names),
    }
    if any(compiled.get(field) != value for field, value in expected_compiled_values.items()):
        raise FrictionScanError(
            f"diagnostic scale {scale} compiled control identity is inconsistent"
        )
    compiled_joints = compiled.get("joints")
    if not isinstance(compiled_joints, list) or len(compiled_joints) != len(
        hand.joint_names
    ):
        raise FrictionScanError(
            f"diagnostic scale {scale} compiled joint set is incomplete"
        )
    compiled_joint_names = [
        item.get("canonical_id") if isinstance(item, Mapping) else None
        for item in compiled_joints
    ]
    if compiled_joint_names != list(hand.joint_names):
        raise FrictionScanError(
            f"diagnostic scale {scale} compiled joint set is unknown or unordered"
        )
    compiled_dof_indices: set[int] = set()
    for joint_name in hand.joint_names:
        joint = _compiled_joint(run.result, joint_name)
        dof_index = joint.get("dof_index")
        if (
            isinstance(dof_index, bool)
            or not isinstance(dof_index, int)
            or dof_index < 0
            or dof_index >= len(effective)
            or dof_index in compiled_dof_indices
        ):
            raise FrictionScanError(
                f"diagnostic scale {scale} joint {joint_name} has an invalid DOF index"
            )
        compiled_dof_indices.add(dof_index)
        joint_friction = _finite_number(
            joint.get("joint_frictionloss"),
            f"diagnostic scale {scale} joint {joint_name} friction",
        )
        if not math.isclose(
            joint_friction,
            effective[dof_index],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise FrictionScanError(
                f"diagnostic scale {scale} joint {joint_name} friction is inconsistent"
            )
    if compiled_dof_indices != set(indices):
        raise FrictionScanError(
            f"diagnostic scale {scale} compiled joints do not cover all friction DOFs"
        )
    return original, effective


def _command_signature(
    result: AdapterRunResult,
    hand: HandSpec,
) -> tuple[tuple[int, float, tuple[float, ...]], ...]:
    return tuple(
        (
            sample.step,
            sample.time_s,
            tuple(float(sample.position_targets[name]) for name in hand.joint_names),
        )
        for sample in result.samples
    )


def _initial_state_signature(
    result: AdapterRunResult,
    hand: HandSpec,
) -> tuple[object, ...]:
    sample = result.samples[0]
    return (
        tuple(sample.qpos),
        tuple(sample.qvel),
        tuple(float(sample.joint_positions[name]) for name in hand.joint_names),
        tuple(
            tuple(float(value) for value in sample.frame_poses[name])
            for name in hand.distal_frame_names
        ),
    )


def _analysis_row(
    *,
    scale: float,
    role: str,
    formal_ov: CollectedRun,
    mujoco_run: CollectedRun,
    hand: HandSpec,
    formal_peak_time_s: float,
) -> dict[str, object]:
    joint_max, frame_position_max, frame_orientation_max = trace_delta(
        formal_ov.result,
        mujoco_run.result,
        hand,
    )
    ov_sample = _sample_at(formal_ov.result, formal_peak_time_s)
    mujoco_sample = _sample_at(mujoco_run.result, formal_peak_time_s)
    ov_joint = float(ov_sample.joint_positions[SELECTED_JOINT])
    mujoco_joint = float(mujoco_sample.joint_positions[SELECTED_JOINT])
    ov_pose = ov_sample.frame_poses[SELECTED_FRAME]
    mujoco_pose = mujoco_sample.frame_poses[SELECTED_FRAME]
    compiled_joint = _compiled_joint(mujoco_run.result, SELECTED_JOINT)
    return {
        "dof_frictionloss_scale": scale,
        "role": role,
        "case_id": mujoco_run.case.case_id,
        "run_source_revision": mujoco_run.result.provenance.get("source_revision"),
        "run_json_sha256": None,
        "compiled_selected_joint_frictionloss_nm": float(
            compiled_joint["joint_frictionloss"]
        ),
        "metrics_vs_formal_ovphysx": {
            "joint_max_abs_rad": joint_max,
            "frame_position_max_m": frame_position_max,
            "frame_orientation_max_rad": frame_orientation_max,
        },
        "own_peak_frame_delta": _peak_frame_delta(
            formal_ov.result, mujoco_run.result, hand
        ),
        "at_formal_peak_time": {
            "time_s": formal_peak_time_s,
            "selected_joint": SELECTED_JOINT,
            "ovphysx_joint_position_rad": ov_joint,
            "mujoco_joint_position_rad": mujoco_joint,
            "signed_ov_minus_mujoco_joint_rad": ov_joint - mujoco_joint,
            "absolute_joint_delta_rad": abs(ov_joint - mujoco_joint),
            "selected_frame": SELECTED_FRAME,
            "frame_position_distance_m": _position_distance(ov_pose, mujoco_pose),
        },
    }


def _scale_directory(scale: float) -> str:
    if scale == 1.0:
        return "scale-1"
    if scale == 0.5:
        return "scale-0p5"
    if scale == 0.0:
        return "scale-0"
    raise FrictionScanError(f"unregistered diagnostic scale: {scale}")


def _decreases_by(values: Sequence[float], minimum_reduction: float) -> bool:
    if len(values) < 2:
        return False
    return all(
        earlier - later >= minimum_reduction
        for earlier, later in zip(values, values[1:])
    )


def _evaluate_trend(
    rows: Sequence[Mapping[str, object]],
    baseline_drift: Sequence[float],
) -> dict[str, object]:
    if len(rows) != len(DIAGNOSTIC_SCALES):
        raise FrictionScanError("trend requires exactly the three diagnostic scales")
    observed_scales = [float(row["dof_frictionloss_scale"]) for row in rows]
    if observed_scales != list(DIAGNOSTIC_SCALES):
        raise FrictionScanError("trend rows are not in pre-registered scale order")
    if len(baseline_drift) != 3 or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in baseline_drift
    ):
        raise FrictionScanError("baseline drift must contain three finite maxima")

    selected_joint_gaps = [
        float(row["at_formal_peak_time"]["absolute_joint_delta_rad"])  # type: ignore[index]
        for row in rows
    ]
    selected_frame_gaps = [
        float(row["at_formal_peak_time"]["frame_position_distance_m"])  # type: ignore[index]
        for row in rows
    ]
    global_joint_gaps = [
        float(row["metrics_vs_formal_ovphysx"]["joint_max_abs_rad"])  # type: ignore[index]
        for row in rows
    ]
    global_frame_gaps = [
        float(row["metrics_vs_formal_ovphysx"]["frame_position_max_m"])  # type: ignore[index]
        for row in rows
    ]
    all_series = (
        selected_joint_gaps,
        selected_frame_gaps,
        global_joint_gaps,
        global_frame_gaps,
    )
    if any(any(not math.isfinite(value) or value < 0.0 for value in series) for series in all_series):
        raise FrictionScanError("trend metrics must be finite and non-negative")

    baseline_ok = all(
        float(value) <= BASELINE_DRIFT_TOLERANCE for value in baseline_drift
    )
    selected_joint_prediction = _decreases_by(
        selected_joint_gaps, MIN_JOINT_REDUCTION_PER_SCALE_STEP_RAD
    )
    global_joint_prediction = _decreases_by(
        global_joint_gaps, MIN_JOINT_REDUCTION_PER_SCALE_STEP_RAD
    )
    selected_frame_prediction = _decreases_by(
        selected_frame_gaps, MIN_FRAME_REDUCTION_PER_SCALE_STEP_M
    )
    global_frame_prediction = _decreases_by(
        global_frame_gaps, MIN_FRAME_REDUCTION_PER_SCALE_STEP_M
    )
    material_trend = all(
        (
            selected_joint_prediction,
            global_joint_prediction,
            selected_frame_prediction,
            global_frame_prediction,
        )
    )
    prediction_supported = baseline_ok and material_trend
    if not baseline_ok:
        scientific_status = "inconclusive_baseline_drift"
    elif prediction_supported:
        scientific_status = "supports_local_hypothesis"
    else:
        scientific_status = "prediction_not_supported"
    return {
        "scientific_status": scientific_status,
        "prediction_supported": prediction_supported,
        "baseline_drift_within_tolerance": baseline_ok,
        "baseline_drift_tolerance": BASELINE_DRIFT_TOLERANCE,
        "baseline_drift_formal_vs_new_scale_1": {
            "joint_max_abs_rad": float(baseline_drift[0]),
            "frame_position_max_m": float(baseline_drift[1]),
            "frame_orientation_max_rad": float(baseline_drift[2]),
        },
        "minimum_joint_reduction_per_scale_step_rad": (
            MIN_JOINT_REDUCTION_PER_SCALE_STEP_RAD
        ),
        "minimum_frame_reduction_per_scale_step_m": (
            MIN_FRAME_REDUCTION_PER_SCALE_STEP_M
        ),
        "selected_joint_gap_decreases_materially": selected_joint_prediction,
        "global_joint_max_decreases_materially": global_joint_prediction,
        "selected_frame_gap_decreases_materially": selected_frame_prediction,
        "global_frame_max_decreases_materially": global_frame_prediction,
        "selected_joint_gaps_rad_at_formal_peak": selected_joint_gaps,
        "selected_frame_gaps_m_at_formal_peak": selected_frame_gaps,
        "global_joint_max_abs_rad": global_joint_gaps,
        "global_frame_position_max_m": global_frame_gaps,
    }


def _render_report(summary: Mapping[str, object]) -> str:
    rows = summary["results"]
    assert isinstance(rows, list)
    trend = summary["trend"]
    assert isinstance(trend, Mapping)
    drift = trend["baseline_drift_formal_vs_new_scale_1"]
    assert isinstance(drift, Mapping)
    baseline = summary["formal_baseline"]
    assert isinstance(baseline, Mapping)
    lines = [
        "# MuJoCo dry-friction scaling diagnostic",
        "",
        "> Local, simulation-only diagnostic against one frozen formal OVPhysX trace.",
        "> This is not a new Gate 0 acceptance run and does not modify canonical assets.",
        "",
        "## Pre-registered question",
        "",
        str(summary["pre_registered_prediction"]),
        "",
        "## Result",
        "",
        f"Scientific status: `{trend['scientific_status']}`.",
        f"Prediction supported: `{str(trend['prediction_supported']).lower()}`.",
        f"New 1.0-scale run matches the old formal MuJoCo reference within 1e-9: `{str(trend['baseline_drift_within_tolerance']).lower()}`.",
        "",
        "Registered checks:",
        "",
        f"- Baseline drift (joint / position / orientation): `{float(drift['joint_max_abs_rad']):.9g}` rad / `{float(drift['frame_position_max_m']):.9g}` m / `{float(drift['frame_orientation_max_rad']):.9g}` rad (limit `{float(trend['baseline_drift_tolerance']):.9g}`).",
        f"- Minimum per-scale reduction: `{float(trend['minimum_joint_reduction_per_scale_step_rad']):.9g}` rad for joint metrics; `{float(trend['minimum_frame_reduction_per_scale_step_m']):.9g}` m for position metrics.",
        f"- Selected joint / global joint checks: `{str(trend['selected_joint_gap_decreases_materially']).lower()}` / `{str(trend['global_joint_max_decreases_materially']).lower()}`.",
        f"- Selected frame / global frame checks: `{str(trend['selected_frame_gap_decreases_materially']).lower()}` / `{str(trend['global_frame_max_decreases_materially']).lower()}`.",
        "",
        "| MuJoCo friction scale | Selected-joint friction (Nm) | Joint max (rad) | DP position max (m) | Orientation max (rad) | thumb_CMC_FE gap at formal peak (rad) | thumb_DP gap at formal peak (m) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        metrics = row["metrics_vs_formal_ovphysx"]
        peak = row["at_formal_peak_time"]
        assert isinstance(metrics, Mapping) and isinstance(peak, Mapping)
        lines.append(
            "| "
            f"{float(row['dof_frictionloss_scale']):.3g} | "
            f"{float(row['compiled_selected_joint_frictionloss_nm']):.9g} | "
            f"{float(metrics['joint_max_abs_rad']):.9g} | "
            f"{float(metrics['frame_position_max_m']):.9g} | "
            f"{float(metrics['frame_orientation_max_rad']):.9g} | "
            f"{float(peak['absolute_joint_delta_rad']):.9g} | "
            f"{float(peak['frame_position_distance_m']):.9g} |"
        )
    lines.extend(
        [
            "",
            "## Evidence identity",
            "",
            f"- Formal bundle root SHA-256: `{baseline['root_sha256']}`",
            f"- Formal source revision: `{baseline['source_revision']}`",
            f"- Diagnostic source revision: `{summary['diagnostic_source_revision']}`",
            f"- Canonical asset tree SHA-256: `{summary['canonical_lf_asset_tree_sha256']}`",
            "",
            "## Interpretation boundary",
            "",
            str(summary["interpretation"]),
            "",
        ]
    )
    return "\n".join(lines)


def run_scan(
    *,
    asset_root: str | Path,
    manifest_path: str | Path,
    formal_bundle: str | Path,
    expected_formal_root_sha256: str,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
) -> tuple[Path, dict[str, object]]:
    if _HEX64.fullmatch(expected_formal_root_sha256) is None:
        raise FrictionScanError("expected formal bundle root must be a lowercase SHA-256")
    if expected_formal_root_sha256 != AUTHORITATIVE_FORMAL_ROOT_SHA256:
        raise FrictionScanError(
            "this pre-registered scan accepts only the authoritative c571d9b formal bundle"
        )
    if _SESSION.fullmatch(session_id) is None:
        raise FrictionScanError("session_id must be a portable non-empty identifier")

    project_root = Path(__file__).resolve(strict=True).parents[1]
    source_tree = _validate_source_identity(project_root, source_revision)
    output_path = _validated_output_path(project_root, output_dir)
    source_files = (
        "scripts/run_mujoco_friction_scan.py",
        "src/wave_asset_qa/__init__.py",
        "src/wave_asset_qa/adapters/__init__.py",
        "src/wave_asset_qa/adapters/base.py",
        "src/wave_asset_qa/adapters/mujoco.py",
        "src/wave_asset_qa/parity/__init__.py",
        "src/wave_asset_qa/parity/bundle.py",
        "src/wave_asset_qa/parity/compare.py",
        "src/wave_asset_qa/parity/contracts.py",
        "src/wave_asset_qa/parity/mapping.py",
        "src/wave_asset_qa/parity/runner.py",
        "src/wave_asset_qa/parity/scenarios.py",
    )
    source_hashes = {
        relative: sha256_file(project_root / relative) for relative in source_files
    }
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    manifest_file_sha256 = sha256_file(manifest_file)
    manifest = load_manifest(manifest_file)
    manifest_digest = manifest_sha256(manifest)
    assets = Path(asset_root).expanduser().resolve(strict=True)
    bundle = Path(formal_bundle).expanduser().resolve(strict=True)
    output_candidate = output_path.resolve(strict=False)
    try:
        output_candidate.relative_to(bundle)
    except ValueError:
        pass
    else:
        raise FrictionScanError("diagnostic output must not modify the formal bundle")
    formal_verification = verify_bundle(bundle)
    if formal_verification["root_sha256"] != expected_formal_root_sha256:
        raise FrictionScanError(
            "formal bundle root mismatch: "
            f"expected {expected_formal_root_sha256}, "
            f"found {formal_verification['root_sha256']}"
        )
    finalization_path = bundle / "results" / "finalization.json"
    try:
        finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrictionScanError(f"cannot read formal finalization identity: {exc}") from exc
    if finalization.get("manifest_sha256") != manifest_digest:
        raise FrictionScanError("formal bundle and diagnostic manifest identities differ")
    if manifest_digest != AUTHORITATIVE_FORMAL_MANIFEST_SHA256:
        raise FrictionScanError("diagnostic manifest is not the pre-registered Gate 0 manifest")

    formal_mujoco_path = _formal_run_path(bundle, FORMAL_MUJOCO_CASE_ID)
    formal_ovphysx_path = _formal_run_path(bundle, FORMAL_OVPHYSX_CASE_ID)
    formal_mujoco = _load_one(formal_mujoco_path)
    formal_ovphysx = _load_one(formal_ovphysx_path)
    if sha256_file(formal_ovphysx_path) != AUTHORITATIVE_OVPHYSX_RUN_SHA256:
        raise FrictionScanError("formal OVPhysX baseline run hash is not authoritative")
    formal_source_revision = finalization.get("source_revision")
    if not isinstance(formal_source_revision, str) or _HEX40.fullmatch(
        formal_source_revision
    ) is None:
        raise FrictionScanError("formal bundle lacks a valid source revision")
    if formal_source_revision != AUTHORITATIVE_FORMAL_SOURCE_REVISION:
        raise FrictionScanError("formal baseline source is not c571d9b")
    for run in (formal_mujoco, formal_ovphysx):
        if run.result.provenance.get("source_revision") != formal_source_revision:
            raise FrictionScanError("formal baseline run source identity is inconsistent")

    expected_cases = {case.case_id: case for case in expand_scenario_cases(manifest)}
    if formal_mujoco.case != expected_cases[FORMAL_MUJOCO_CASE_ID]:
        raise FrictionScanError("formal MuJoCo baseline is not the canonical selected case")
    if formal_ovphysx.case != expected_cases[FORMAL_OVPHYSX_CASE_ID]:
        raise FrictionScanError("formal OVPhysX baseline is not the canonical selected case")
    hand = manifest.hand(formal_mujoco.case.hand)
    diagnostic_model_path = (assets / formal_mujoco.case.model_path).resolve(strict=True)
    try:
        diagnostic_model_path.relative_to(assets)
    except ValueError as exc:
        raise FrictionScanError("diagnostic MuJoCo model escapes the asset root") from exc
    expected_source_sha256 = sha256_file(diagnostic_model_path)

    root = _create_staging_root(output_path)
    inputs_dir = root / "inputs"
    inputs_dir.mkdir()
    copied_manifest = inputs_dir / "gate0.manifest.json"
    shutil.copyfile(manifest_file, copied_manifest)
    copied_source_root = inputs_dir / "source"
    for relative in source_files:
        destination = copied_source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative, destination)
        if sha256_file(destination) != source_hashes[relative]:
            raise FrictionScanError(f"copied diagnostic source hash differs: {relative}")
    baseline_dir = root / "baseline"
    baseline_dir.mkdir()
    formal_bundle_manifest_path = bundle / "bundle.json"
    copied_mujoco = baseline_dir / formal_mujoco_path.name
    copied_ovphysx = baseline_dir / formal_ovphysx_path.name
    copied_formal_bundle = baseline_dir / "formal-bundle.json"
    copied_finalization = baseline_dir / "formal-finalization.json"
    shutil.copyfile(formal_mujoco_path, copied_mujoco)
    shutil.copyfile(formal_ovphysx_path, copied_ovphysx)
    shutil.copyfile(formal_bundle_manifest_path, copied_formal_bundle)
    shutil.copyfile(finalization_path, copied_finalization)
    if sha256_file(copied_formal_bundle) != sha256_file(formal_bundle_manifest_path):
        raise FrictionScanError("copied formal bundle manifest differs from its source")
    _verify_copied_formal_subset(
        copied_formal_bundle,
        expected_root_sha256=expected_formal_root_sha256,
        copied_payloads={
            formal_mujoco_path.relative_to(bundle).as_posix(): copied_mujoco,
            formal_ovphysx_path.relative_to(bundle).as_posix(): copied_ovphysx,
            finalization_path.relative_to(bundle).as_posix(): copied_finalization,
        },
    )

    diagnostic_runs: dict[float, tuple[CollectedRun, Path]] = {}
    original_friction: tuple[float, ...] | None = None
    invariant_provenance: dict[str, object] | None = None
    command_signature: tuple[tuple[int, float, tuple[float, ...]], ...] | None = None
    initial_state_signature: tuple[object, ...] | None = None
    invariant_fields = (
        "source_sha256",
        "backend_version",
        "python_version",
        "platform",
        "backend_joint_names",
        "backend_frame_names",
        "joint_mapping",
        "frame_mapping",
        "compiled_control_parameters",
    )
    for scale in DIAGNOSTIC_SCALES:
        scale_dir = root / "diagnostic" / _scale_directory(scale)
        scale_dir.parent.mkdir(exist_ok=True)
        runs = run_backend_cases(
            assets,
            manifest,
            scale_dir,
            backend=Simulator.MUJOCO,
            case_ids=(FORMAL_MUJOCO_CASE_ID,),
            adapter_factory=partial(
                MuJoCoAdapter,
                dof_frictionloss_scale=scale,
            ),
            session_id=session_id,
            source_revision=source_revision,
            verify_asset_tree=True,
        )
        if len(runs) != 1:
            raise FrictionScanError(
                f"diagnostic scale {scale} did not produce exactly one run"
            )
        before, _ = _validate_diagnostic_run(
            runs[0],
            expected_case=expected_cases[FORMAL_MUJOCO_CASE_ID],
            scale=scale,
            session_id=session_id,
            source_revision=source_revision,
            manifest_digest=manifest_digest,
            manifest=manifest,
            expected_source_sha256=expected_source_sha256,
        )
        if original_friction is None:
            original_friction = before
        elif before != original_friction:
            raise FrictionScanError(
                "diagnostic scales were compiled from different original friction arrays"
            )
        provenance = runs[0].result.provenance
        observed_invariants = {field: provenance.get(field) for field in invariant_fields}
        # The effective friction values inside compiled_control_parameters are the
        # intentional intervention; compare the remaining compiled readback below.
        compiled = observed_invariants.pop("compiled_control_parameters")
        if not isinstance(compiled, Mapping):
            raise FrictionScanError("diagnostic run lacks compiled control provenance")
        normalized_compiled = dict(compiled)
        normalized_joints: list[dict[str, object]] = []
        joints = normalized_compiled.get("joints")
        if not isinstance(joints, list):
            raise FrictionScanError("diagnostic compiled provenance lacks joints")
        for item in joints:
            if not isinstance(item, Mapping):
                raise FrictionScanError("diagnostic compiled joint record is invalid")
            normalized = dict(item)
            normalized.pop("joint_frictionloss", None)
            normalized_joints.append(normalized)
        normalized_compiled["joints"] = normalized_joints
        observed_invariants["compiled_control_parameters_without_friction"] = (
            normalized_compiled
        )
        if invariant_provenance is None:
            invariant_provenance = observed_invariants
        elif observed_invariants != invariant_provenance:
            raise FrictionScanError(
                "diagnostic scales differ in provenance beyond the friction intervention"
            )
        observed_commands = _command_signature(runs[0].result, hand)
        observed_initial_state = _initial_state_signature(runs[0].result, hand)
        if command_signature is None:
            command_signature = observed_commands
            initial_state_signature = observed_initial_state
        elif (
            observed_commands != command_signature
            or observed_initial_state != initial_state_signature
        ):
            raise FrictionScanError(
                "diagnostic scales did not use identical commands and initial state"
            )
        run_path = scale_dir / f"{FORMAL_MUJOCO_CASE_ID}.run.json"
        if not run_path.is_file():
            raise FrictionScanError(f"diagnostic scale {scale} run file is missing")
        diagnostic_runs[scale] = (runs[0], run_path)

    formal_peak = _peak_frame_delta(formal_ovphysx.result, formal_mujoco.result, hand)
    formal_peak_time_s = float(formal_peak["time_s"])
    formal_reference = _analysis_row(
        scale=1.0,
        role="formal_mujoco_external_reference",
        formal_ov=formal_ovphysx,
        mujoco_run=formal_mujoco,
        hand=hand,
        formal_peak_time_s=formal_peak_time_s,
    )
    formal_reference["run_json_sha256"] = sha256_file(copied_mujoco)
    rows: list[dict[str, object]] = []
    for scale in DIAGNOSTIC_SCALES:
        run, run_path = diagnostic_runs[scale]
        row = _analysis_row(
            scale=scale,
            role=(
                "diagnostic_mujoco_canonical_scale"
                if scale == 1.0
                else "diagnostic_mujoco_scaled_override"
            ),
            formal_ov=formal_ovphysx,
            mujoco_run=run,
            hand=hand,
            formal_peak_time_s=formal_peak_time_s,
        )
        row["run_json_sha256"] = sha256_file(run_path)
        rows.append(row)

    baseline_drift = trace_delta(
        formal_mujoco.result,
        diagnostic_runs[1.0][0].result,
        hand,
    )
    trend = _evaluate_trend(rows, baseline_drift)
    scientific_status = str(trend["scientific_status"])
    if scientific_status == "inconclusive_baseline_drift":
        interpretation = (
            "The new canonical-scale MuJoCo trace drifted from the frozen formal MuJoCo "
            "reference, so this experiment cannot isolate friction and is scientifically "
            "inconclusive. The formal Gate 0 result remains unchanged."
        )
    elif scientific_status == "supports_local_hypothesis":
        interpretation = (
            "Within this one left-hand small_step trajectory, MuJoCo dry friction is "
            "supported as a material contributor to both joint and distal-frame-position "
            "differences. This does not establish equivalent PhysX friction, identify a "
            "backend bug, or replace the formal Gate 0 result."
        )
    else:
        interpretation = (
            "The pre-registered local dry-friction prediction was not supported. The formal "
            "Gate 0 result remains unchanged, and the friction hypothesis needs revision "
            "before any remote expansion."
        )
    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "completed",
        "scientific_status": scientific_status,
        "classification": "diagnostic_only_not_formal_gate0",
        "pre_registered_prediction": (
            "Using three newly generated runs from the same diagnostic source and session, "
            "each change in MuJoCo dof_frictionloss scale from 1.0 to 0.5 to 0.0 reduces "
            "both the selected thumb_CMC_FE / thumb_DP position gaps at the formal peak "
            "and the trajectory-wide joint / distal-frame-position maxima by at least the "
            "registered minimum effect sizes. The new 1.0 run must also reproduce the old "
            "formal MuJoCo trace within the registered baseline-drift tolerance."
        ),
        "formal_baseline": {
            "root_sha256": expected_formal_root_sha256,
            "source_revision": formal_source_revision,
            "verification": formal_verification,
            "mujoco_run_sha256": sha256_file(copied_mujoco),
            "ovphysx_run_sha256": sha256_file(copied_ovphysx),
            "bundle_manifest_sha256": sha256_file(copied_formal_bundle),
            "finalization_sha256": sha256_file(copied_finalization),
            "copied_files": {
                "bundle_manifest": copied_formal_bundle.relative_to(root).as_posix(),
                "finalization": copied_finalization.relative_to(root).as_posix(),
                "mujoco_run": copied_mujoco.relative_to(root).as_posix(),
                "ovphysx_run": copied_ovphysx.relative_to(root).as_posix(),
            },
        },
        "diagnostic_source_revision": source_revision,
        "diagnostic_source_tree": source_tree,
        "diagnostic_source_files_sha256": source_hashes,
        "diagnostic_selected_source_evidence_root": (
            copied_source_root.relative_to(root).as_posix()
        ),
        "session_id": session_id,
        "manifest_file_sha256": sha256_file(copied_manifest),
        "manifest_sha256": manifest_digest,
        "canonical_lf_asset_tree_sha256": (
            manifest.provenance.canonical_lf_asset_tree_sha256
        ),
        "case": {
            "mujoco_case_id": FORMAL_MUJOCO_CASE_ID,
            "ovphysx_case_id": FORMAL_OVPHYSX_CASE_ID,
            "selected_joint": SELECTED_JOINT,
            "selected_frame": SELECTED_FRAME,
            "formal_peak": formal_peak,
        },
        "intervention": {
            "field": "MuJoCo MjModel.dof_frictionloss",
            "scope": "all_compiled_dofs",
            "materialization": "in_memory_after_mjcf_compile",
            "canonical_asset_files_modified": False,
            "diagnostic_scales": list(DIAGNOSTIC_SCALES),
            "canonical_commands_verified": True,
            "identical_command_timeline_across_scales": True,
            "identical_initial_state_across_scales": True,
        },
        "results": rows,
        "formal_mujoco_external_reference": formal_reference,
        "trend": trend,
        "interpretation": interpretation,
        "limits": [
            "Only the left small_step base r01 trajectory is tested.",
            "The OVPhysX trace is frozen formal evidence; its friction is not changed or read back here.",
            "The diagnostic does not change formal thresholds or pass_ready=false.",
            "A supported trend motivates, but does not replace, bilateral friction and solver readback.",
        ],
    }
    observed_tree = _validate_source_identity(project_root, source_revision)
    if observed_tree != source_tree:
        raise FrictionScanError("diagnostic Git source changed during execution")
    observed_source_hashes = {
        relative: sha256_file(project_root / relative) for relative in source_files
    }
    if observed_source_hashes != source_hashes:
        raise FrictionScanError("diagnostic source files changed during execution")
    if sha256_file(manifest_file) != manifest_file_sha256:
        raise FrictionScanError("diagnostic manifest changed during execution")
    if verify_bundle(bundle)["root_sha256"] != expected_formal_root_sha256:
        raise FrictionScanError("formal baseline bundle changed during execution")

    write_json_atomic(root / "summary.json", summary)
    _write_text_exclusive(root / "report.md", _render_report(summary))
    payload_paths = _bundle_payload_paths(root)
    write_bundle_manifest(root, payload_paths)
    verification = _verify_exact_bundle(root)
    promoted = _promote_staging_root(root, output_path)
    final_verification = _verify_exact_bundle(promoted)
    if final_verification != verification:
        raise FrictionScanError("diagnostic bundle identity changed during promotion")
    return promoted, final_verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--formal-bundle", required=True, type=Path)
    parser.add_argument("--expected-formal-root-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, verification = run_scan(
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            formal_bundle=args.formal_bundle,
            expected_formal_root_sha256=args.expected_formal_root_sha256,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
        )
    except (FrictionScanError, OSError, ValueError) as exc:
        print(f"MuJoCo friction scan failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": output.name,
                **verification,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
