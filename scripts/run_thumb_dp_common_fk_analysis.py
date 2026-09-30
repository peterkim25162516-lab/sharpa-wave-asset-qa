"""Run the frozen left-thumb common-MuJoCo-FK attribution analysis.

This is an offline, post-hoc diagnostic.  It never advances dynamics and it
does not change the formal Gate 0 result.  The raw output bundle is private
because it copies frozen OVPhysX evidence containing host/process metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Mapping, Sequence

import numpy as np

from wave_asset_qa.adapters.base import TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter
from wave_asset_qa.parity.bundle import sha256_file, write_bundle_manifest
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import HandSide, ParityManifest, Simulator
from wave_asset_qa.parity.diagnostics import (
    DiagnosticEvidenceError,
    asset_tree_sha256,
    bundle_payload_paths,
    create_staging_root,
    is_link_like,
    promote_staging_root,
    replay_mujoco_fk_named,
    validate_diagnostic_run,
    validate_git_source,
    validated_results_output_path,
    verify_copied_bundle_subset,
    verify_exact_bundle,
    write_text_exclusive,
)
from wave_asset_qa.parity.kinematic_attribution import (
    CLOSES,
    DOES_NOT_CLOSE,
    CLOSE_RHO_MAX,
    DOES_NOT_CLOSE_RHO_MIN,
    DOMINANT_MIN_NORMALIZED_ATTRIBUTION,
    DOMINANT_MIN_POSITIVE_SHARE,
    EMPTY_COALITION_ATOL_M,
    FULL_COALITION_ATOL_M,
    MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION,
    RAW_NORM_MIN_M,
    SHAPLEY_EFFICIENCY_ATOL,
    VECTOR_EFFICIENCY_ATOL_M,
    analyze_kinematic_attribution,
)
from wave_asset_qa.parity.runner import load_collected_runs
from wave_asset_qa.parity.scenarios import load_manifest, manifest_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_RELATIVE_PATH = "configs/parity/thumb_dp_common_fk_analysis.json"
MANIFEST_RELATIVE_PATH = "configs/parity/gate0.json"
PLAN_SHA256 = "da63d86f48f9b40e7bfff034582f62c96d3787400ef46184a50fe90a429b30e2"
PARENT_ROOT_SHA256 = "a27826a0909afbf550f2bda198d7c7aab564e019a06a40aa63e829de92c0018c"
PARENT_MANIFEST_SHA256 = "2022d931615b678f8669c5847efe292dec18c08b6945c1476b034b49390bda81"
PARENT_PAYLOAD_COUNT = 39
PARENT_TOTAL_SIZE_BYTES = 5_750_736
MODEL_SHA256 = "3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11"
MUJOCO_VERSION = "3.12.0"
FRAME_NAME = "left_thumb_DP"
START_STEP = 55
END_STEP = 85
DT_S = 0.002
PLAYER_NAMES = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
)
ROLE_CONTROL = "same_source_control"
ROLE_TREATMENT = "total_viscous_aligned_via_controller_kv_treatment"
ROLE_ORDER = (ROLE_CONTROL, ROLE_TREATMENT)
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

SOURCE_RELATIVE_PATHS = (
    PLAN_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    "scripts/run_thumb_dp_common_fk_analysis.py",
    "src/wave_asset_qa/__init__.py",
    "src/wave_asset_qa/adapters/__init__.py",
    "src/wave_asset_qa/adapters/base.py",
    "src/wave_asset_qa/adapters/mujoco.py",
    "src/wave_asset_qa/parity/__init__.py",
    "src/wave_asset_qa/parity/bundle.py",
    "src/wave_asset_qa/parity/compare.py",
    "src/wave_asset_qa/parity/contracts.py",
    "src/wave_asset_qa/parity/diagnostics.py",
    "src/wave_asset_qa/parity/kinematic_attribution.py",
    "src/wave_asset_qa/parity/mapping.py",
    "src/wave_asset_qa/parity/runner.py",
    "src/wave_asset_qa/parity/scenarios.py",
    "tests/test_kinematic_attribution.py",
    "tests/test_thumb_dp_common_fk_analysis.py",
)


class CommonFkAnalysisError(RuntimeError):
    """Raised when the frozen analysis contract cannot be preserved."""


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CommonFkAnalysisError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> object:
    raise CommonFkAnalysisError(f"non-finite JSON number is forbidden: {token}")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except CommonFkAnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CommonFkAnalysisError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CommonFkAnalysisError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CommonFkAnalysisError(f"{label} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommonFkAnalysisError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CommonFkAnalysisError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise CommonFkAnalysisError(f"{label} must be a finite number")
    return number


def _compact_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_plan(path: Path) -> dict[str, object]:
    if sha256_file(path) != PLAN_SHA256:
        raise CommonFkAnalysisError("analysis plan bytes changed")
    plan = _load_json_object(path, "analysis plan")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "classification",
        "analysis_plan_status",
        "scope",
        "feasibility_disclosure",
        "frozen_inputs",
        "parent_paths",
        "cases",
        "analysis",
        "sanity_gates",
        "closure_metrics",
        "closure_decision_rule",
        "shapley",
        "output_policy",
        "claim_boundary",
    }
    if set(plan) != expected_keys:
        raise CommonFkAnalysisError("analysis plan top-level schema changed")
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment_id")
        != "left-thumb-dp-common-mujoco-fk-attribution-v1"
        or plan.get("classification")
        != "post_hoc_mechanistic_analysis_not_formal_gate0"
        or plan.get("analysis_plan_status")
        != "frozen_after_single_step_feasibility_probe_before_full_window_computation"
    ):
        raise CommonFkAnalysisError("analysis plan identity changed")
    disclosure = _mapping(plan["feasibility_disclosure"], "feasibility disclosure")
    if (
        disclosure.get("inspected_before_plan_freeze") is not True
        or disclosure.get("step") != 66
        or _finite(disclosure.get("time_s"), "feasibility time") != 0.132
    ):
        raise CommonFkAnalysisError("feasibility disclosure changed")
    frozen = _mapping(plan["frozen_inputs"], "frozen inputs")
    expected_frozen = {
        "parent_bundle_root_sha256": PARENT_ROOT_SHA256,
        "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_bundle_payload_count": PARENT_PAYLOAD_COUNT,
        "parent_bundle_total_size_bytes": PARENT_TOTAL_SIZE_BYTES,
        "mujoco_model_sha256": MODEL_SHA256,
        "frozen_mujoco_version": MUJOCO_VERSION,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise CommonFkAnalysisError(f"frozen input changed: {key}")
    analysis = _mapping(plan["analysis"], "analysis")
    if (
        analysis.get("hand") != "left"
        or analysis.get("frame") != FRAME_NAME
        or analysis.get("window_start_step") != START_STEP
        or analysis.get("window_end_step") != END_STEP
        or analysis.get("sample_count") != END_STEP - START_STEP + 1
        or _finite(analysis.get("dt_s"), "analysis dt") != DT_S
        or tuple(analysis.get("candidate_roles", ())) != ROLE_ORDER
        or tuple(analysis.get("thumb_players_in_order", ())) != PLAYER_NAMES
        or analysis.get("coalition_count_per_sample") != 32
        or analysis.get("backend_native_qpos_direct_use_forbidden") is not True
        or analysis.get("coordinate_transform") != "identity_only_no_fit"
        or analysis.get("advance_dynamics") is not False
        or analysis.get("fit_translation_rotation_scale_or_time_shift") is not False
    ):
        raise CommonFkAnalysisError("analysis method changed")
    gates = _mapping(plan["sanity_gates"], "sanity gates")
    expected_gates = {
        "empty_coalition_max_position_error_m": EMPTY_COALITION_ATOL_M,
        "all_thumb_coalition_vs_full_ov_fk_max_position_error_m": FULL_COALITION_ATOL_M,
        "vector_shapley_efficiency_max_abs_error_m": VECTOR_EFFICIENCY_ATOL_M,
        "energy_shapley_efficiency_max_abs_fraction": SHAPLEY_EFFICIENCY_ATOL,
        "closure_denominator_min_frobenius_m": RAW_NORM_MIN_M,
    }
    for key, expected in expected_gates.items():
        if _finite(gates.get(key), f"sanity gate {key}") != expected:
            raise CommonFkAnalysisError(f"sanity gate changed: {key}")
    rule = _mapping(plan["closure_decision_rule"], "closure decision rule")
    if (
        _finite(rule.get("closes_if_window_ratio_at_most"), "closure threshold")
        != CLOSE_RHO_MAX
        or _finite(rule.get("closes_if_peak_ratio_at_most"), "closure threshold")
        != CLOSE_RHO_MAX
        or _finite(
            rule.get("does_not_close_if_window_ratio_at_least"),
            "nonclosure threshold",
        )
        != DOES_NOT_CLOSE_RHO_MIN
        or _finite(
            rule.get("does_not_close_if_peak_ratio_at_least"),
            "nonclosure threshold",
        )
        != DOES_NOT_CLOSE_RHO_MIN
    ):
        raise CommonFkAnalysisError("closure thresholds changed")
    shapley = _mapping(plan["shapley"], "shapley")
    if (
        _finite(
            shapley.get("dominant_positive_minimum_fraction_points"),
            "dominant threshold",
        )
        != DOMINANT_MIN_NORMALIZED_ATTRIBUTION
        or _finite(
            shapley.get("dominant_positive_minimum_share_of_positive_attribution"),
            "positive-share threshold",
        )
        != DOMINANT_MIN_POSITIVE_SHARE
        or _finite(
            shapley.get("material_adverse_maximum_fraction_points"),
            "adverse threshold",
        )
        != MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION
    ):
        raise CommonFkAnalysisError("Shapley thresholds changed")
    return plan


def _parent_member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CommonFkAnalysisError("parent member path is invalid")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise CommonFkAnalysisError(f"parent member path is unsafe: {relative}")
    candidate = root.joinpath(*portable.parts)
    if candidate.is_symlink():
        raise CommonFkAnalysisError(f"parent member is a symbolic link: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CommonFkAnalysisError(f"parent member escapes bundle: {relative}") from exc
    if not resolved.is_file():
        raise CommonFkAnalysisError(f"parent member is not a file: {relative}")
    return resolved


def _validated_real_asset_root(path: Path) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    if is_link_like(raw):
        raise CommonFkAnalysisError(
            f"asset root must not be a symbolic link or junction: {raw}"
        )
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise CommonFkAnalysisError("asset root must be an existing directory")
    return resolved


def _load_one(path: Path, label: str) -> CollectedRun:
    runs = load_collected_runs(path)
    if len(runs) != 1:
        raise CommonFkAnalysisError(f"{label} must contain exactly one run")
    return runs[0]


def _validate_parent(
    parent_root: Path,
    plan: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, Path]]:
    verification = verify_exact_bundle(parent_root)
    if (
        verification.get("root_sha256") != PARENT_ROOT_SHA256
        or verification.get("file_count") != PARENT_PAYLOAD_COUNT
        or verification.get("total_size") != PARENT_TOTAL_SIZE_BYTES
        or verification.get("exact_inventory") is not True
        or sha256_file(parent_root / "bundle.json") != PARENT_MANIFEST_SHA256
    ):
        raise CommonFkAnalysisError("parent bundle identity or inventory changed")
    paths = _mapping(plan["parent_paths"], "parent paths")
    members = {name: _parent_member(parent_root, str(relative)) for name, relative in paths.items()}
    frozen = _mapping(plan["frozen_inputs"], "frozen inputs")
    expected_hashes = {
        "summary": frozen["parent_summary_sha256"],
        "controller_kv_preregistration": frozen[
            "controller_kv_preregistration_sha256"
        ],
        "gate0_manifest": frozen["gate0_manifest_file_sha256"],
        "formal_ovphysx_run": frozen["formal_ovphysx_run_sha256"],
        "friction_zero_run": frozen["friction_zero_run_sha256"],
        "same_source_control_run": frozen["same_source_control_run_sha256"],
        "total_viscous_treatment_run": frozen[
            "total_viscous_treatment_run_sha256"
        ],
    }
    for name, expected in expected_hashes.items():
        if sha256_file(members[name]) != expected:
            raise CommonFkAnalysisError(f"parent member hash changed: {name}")
    summary = _load_json_object(members["summary"], "parent summary")
    if (
        summary.get("execution_status") != "completed"
        or summary.get("scientific_status") != "mixed_or_inconclusive"
        or summary.get("classification") != "diagnostic_only_not_formal_gate0"
        or summary.get("diagnostic_source_revision")
        != frozen["parent_source_revision"]
        or summary.get("diagnostic_source_tree") != frozen["parent_source_tree"]
    ):
        raise CommonFkAnalysisError("parent summary identity changed")
    return verification, members


def _validate_manifest_and_assets(
    manifest_path: Path,
    asset_root: Path,
    plan: Mapping[str, object],
) -> tuple[ParityManifest, Path, str]:
    frozen = _mapping(plan["frozen_inputs"], "frozen inputs")
    if sha256_file(manifest_path) != frozen["gate0_manifest_file_sha256"]:
        raise CommonFkAnalysisError("Gate 0 manifest bytes changed")
    manifest = load_manifest(manifest_path)
    if manifest_sha256(manifest) != frozen["gate0_manifest_semantic_sha256"]:
        raise CommonFkAnalysisError("Gate 0 manifest semantics changed")
    if (
        manifest.provenance.commit != frozen["asset_commit"]
        or manifest.provenance.asset_git_tree != frozen["asset_git_tree"]
        or manifest.provenance.canonical_lf_asset_tree_sha256
        != frozen["canonical_lf_asset_tree_sha256"]
    ):
        raise CommonFkAnalysisError("manifest asset provenance changed")
    subtree_candidate = asset_root / manifest.provenance.asset_root
    if is_link_like(subtree_candidate):
        raise CommonFkAnalysisError(
            "manifest asset subtree must not be a symbolic link or junction"
        )
    asset_subtree = subtree_candidate.resolve(strict=True)
    try:
        asset_subtree.relative_to(asset_root)
    except ValueError as exc:
        raise CommonFkAnalysisError("manifest asset subtree escapes the asset root") from exc
    observed_tree = asset_tree_sha256(asset_subtree)
    if observed_tree != frozen["canonical_lf_asset_tree_sha256"]:
        raise CommonFkAnalysisError("materialized asset tree hash changed")
    model_relative = str(frozen["mujoco_model_relative_path"])
    model_candidate = asset_root.joinpath(*PurePosixPath(model_relative).parts)
    if is_link_like(model_candidate):
        raise CommonFkAnalysisError(
            "MuJoCo model must not be a symbolic link or junction"
        )
    model_path = model_candidate.resolve(strict=True)
    try:
        model_path.relative_to(asset_root)
    except ValueError as exc:
        raise CommonFkAnalysisError("MuJoCo model escapes the asset root") from exc
    if model_path.is_symlink() or sha256_file(model_path) != MODEL_SHA256:
        raise CommonFkAnalysisError("MuJoCo model bytes changed")
    hand = manifest.hand(HandSide.LEFT)
    if hand.model_paths.mujoco != model_relative:
        raise CommonFkAnalysisError("manifest MuJoCo path changed")
    return manifest, model_path, observed_tree


def _validate_run_provenance(
    run: CollectedRun,
    *,
    manifest: ParityManifest,
    plan: Mapping[str, object],
    role: str,
) -> None:
    validate_diagnostic_run(run.result, run.case, manifest)
    frozen = _mapping(plan["frozen_inputs"], "frozen inputs")
    provenance = run.result.provenance
    common = {
        "asset_commit": frozen["asset_commit"],
        "asset_git_tree": frozen["asset_git_tree"],
        "asset_tree_sha256": frozen["canonical_lf_asset_tree_sha256"],
        "manifest_sha256": frozen["gate0_manifest_semantic_sha256"],
    }
    for field, expected in common.items():
        if provenance.get(field) != expected:
            raise CommonFkAnalysisError(f"{role} provenance changed: {field}")
    cases = _mapping(plan["cases"], "cases")
    if role == "formal_ovphysx_run":
        expected_case = cases["ovphysx"]
        expected_backend = Simulator.OVPHYSX
        expected_source_revision = frozen["formal_source_revision"]
        expected_source_sha = frozen["formal_ovphysx_source_sha256"]
    elif role == "friction_zero_run":
        expected_case = cases["mujoco"]
        expected_backend = Simulator.MUJOCO
        expected_source_revision = frozen["friction_source_revision"]
        expected_source_sha = MODEL_SHA256
    else:
        expected_case = cases["mujoco"]
        expected_backend = Simulator.MUJOCO
        expected_source_revision = frozen["parent_source_revision"]
        expected_source_sha = MODEL_SHA256
    if (
        run.case.case_id != expected_case
        or run.case.simulator is not expected_backend
        or provenance.get("source_revision") != expected_source_revision
        or provenance.get("source_sha256") != expected_source_sha
    ):
        raise CommonFkAnalysisError(f"{role} case or source identity changed")
    if expected_backend is Simulator.MUJOCO and provenance.get("backend_version") != MUJOCO_VERSION:
        raise CommonFkAnalysisError(f"{role} MuJoCo version changed")


def _quaternion_angle(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    if len(a) != 4 or len(b) != 4:
        raise CommonFkAnalysisError("quaternion must have four values")
    qa = np.asarray(a, dtype=float)
    qb = np.asarray(b, dtype=float)
    if not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qb)):
        raise CommonFkAnalysisError("quaternion contains non-finite values")
    norm_a = math.hypot(*(float(value) for value in qa))
    norm_b = math.hypot(*(float(value) for value in qb))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise CommonFkAnalysisError("quaternion norm must be positive")
    unit_error = max(abs(norm_a - 1.0), abs(norm_b - 1.0))
    qa /= norm_a
    qb /= norm_b
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    difference_norm = math.hypot(
        *(float(x) - float(y) for x, y in zip(qa, qb))
    )
    sum_norm = math.hypot(
        *(float(x) + float(y) for x, y in zip(qa, qb))
    )
    return 4.0 * math.atan2(difference_norm, sum_norm), unit_error


def _pose_error(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    if len(a) != 7 or len(b) != 7:
        raise CommonFkAnalysisError("frame pose must have seven values")
    position = math.hypot(*(float(x) - float(y) for x, y in zip(a[:3], b[:3])))
    orientation, unit_error = _quaternion_angle(a[3:], b[3:])
    if not all(math.isfinite(value) for value in (position, orientation, unit_error)):
        raise CommonFkAnalysisError("derived pose error is non-finite")
    return position, orientation, unit_error


def _sample_map(run: CollectedRun) -> dict[int, TraceSample]:
    result = {sample.step: sample for sample in run.result.samples}
    if len(result) != len(run.result.samples):
        raise CommonFkAnalysisError("run contains duplicate sample steps")
    return result


def _topology_audit(adapter: MuJoCoAdapter) -> dict[str, object]:
    model = adapter._model
    mujoco = adapter._mujoco
    if model is None or mujoco is None:
        raise CommonFkAnalysisError("MuJoCo adapter is not open")
    if int(model.nq) != 22 or int(model.nv) != 22 or int(model.njnt) != 22:
        raise CommonFkAnalysisError("compiled model is not exactly 22 scalar joints")
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    records: list[dict[str, object]] = []
    qpos_addresses: list[int] = []
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not isinstance(name, str) or not name:
            raise CommonFkAnalysisError("compiled model contains an unnamed joint")
        joint_type = int(model.jnt_type[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        if joint_type != hinge:
            raise CommonFkAnalysisError(f"compiled joint is not scalar hinge: {name}")
        qpos_addresses.append(qpos_address)
        records.append(
            {
                "canonical_id": name,
                "joint_id": joint_id,
                "qpos_address": qpos_address,
                "dof_address": dof_address,
                "type": "hinge",
            }
        )
    if tuple(record["canonical_id"] for record in records) != adapter.joint_names:
        raise CommonFkAnalysisError("compiled joint order differs from adapter order")
    if sorted(qpos_addresses) != list(range(22)):
        raise CommonFkAnalysisError("compiled qpos addresses do not cover 0..21")

    body_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FRAME_NAME)
    )
    if body_id < 0:
        raise CommonFkAnalysisError("compiled model lacks left_thumb_DP")
    distal_to_root_segments: list[list[str]] = []
    cursor = body_id
    while cursor > 0:
        first = int(model.body_jntadr[cursor])
        count = int(model.body_jntnum[cursor])
        segment: list[str] = []
        for joint_id in range(first, first + count):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not isinstance(name, str) or not name:
                raise CommonFkAnalysisError("thumb ancestor contains unnamed joint")
            segment.append(name)
        if segment:
            distal_to_root_segments.append(segment)
        cursor = int(model.body_parentid[cursor])
    ancestor_chain = tuple(
        name
        for segment in reversed(distal_to_root_segments)
        for name in segment
    )
    if ancestor_chain != PLAYER_NAMES:
        raise CommonFkAnalysisError(
            f"thumb_DP ancestor chain changed: {ancestor_chain}"
        )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "joint_records": records,
        "thumb_dp_ancestor_joint_order": list(ancestor_chain),
        "thumb_dp_ancestor_joint_order_sha256": _compact_sha256(list(ancestor_chain)),
        "fixed_base": True,
        "scalar_hinge_joint_count": 22,
    }


def _identity_coordinate_audit(
    adapter: MuJoCoAdapter,
    *,
    ov: CollectedRun,
    candidates: Mapping[str, CollectedRun],
    manifest: ParityManifest,
    plan: Mapping[str, object],
) -> dict[str, object]:
    hand = manifest.hand(HandSide.LEFT)
    ov_zero = ov.result.samples[0]
    ov_replay = replay_mujoco_fk_named(
        adapter,
        ov_zero.joint_positions,
        expected_joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
    )
    gates = _mapping(plan["sanity_gates"], "sanity gates")
    max_ov_position = 0.0
    max_ov_orientation = 0.0
    max_unit_error = 0.0
    per_frame: dict[str, object] = {}
    for frame in hand.distal_frame_names:
        position, orientation, unit_error = _pose_error(
            ov_zero.frame_poses[frame], ov_replay.frame_poses[frame]
        )
        max_ov_position = max(max_ov_position, position)
        max_ov_orientation = max(max_ov_orientation, orientation)
        max_unit_error = max(max_unit_error, unit_error)
        per_frame[frame] = {
            "ov_raw_vs_common_fk_position_error_m": position,
            "ov_raw_vs_common_fk_orientation_error_rad": orientation,
        }
    candidate_rows: dict[str, object] = {}
    max_candidate_position = 0.0
    max_candidate_orientation = 0.0
    for role in ROLE_ORDER:
        sample = candidates[role].result.samples[0]
        replay = replay_mujoco_fk_named(
            adapter,
            sample.joint_positions,
            expected_joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
        )
        position_max = 0.0
        orientation_max = 0.0
        for frame in hand.distal_frame_names:
            position, orientation, unit_error = _pose_error(
                sample.frame_poses[frame], replay.frame_poses[frame]
            )
            position_max = max(position_max, position)
            orientation_max = max(orientation_max, orientation)
            max_unit_error = max(max_unit_error, unit_error)
        max_candidate_position = max(max_candidate_position, position_max)
        max_candidate_orientation = max(max_candidate_orientation, orientation_max)
        candidate_rows[role] = {
            "raw_vs_common_fk_position_error_max_m": position_max,
            "raw_vs_common_fk_orientation_error_max_rad": orientation_max,
        }

    disclosure = _mapping(
        plan["feasibility_disclosure"], "feasibility disclosure"
    )
    disclosed_step = int(disclosure["step"])
    probe_sample = ov.result.samples[disclosed_step]
    probe_replay = replay_mujoco_fk_named(
        adapter,
        probe_sample.joint_positions,
        expected_joint_names=hand.joint_names,
        frame_names=hand.distal_frame_names,
        step=disclosed_step,
    )
    probe_position_max = 0.0
    probe_orientation_max = 0.0
    for frame in hand.distal_frame_names:
        position, orientation, unit_error = _pose_error(
            probe_sample.frame_poses[frame], probe_replay.frame_poses[frame]
        )
        probe_position_max = max(probe_position_max, position)
        probe_orientation_max = max(probe_orientation_max, orientation)
        max_unit_error = max(max_unit_error, unit_error)
    if not math.isclose(
        probe_position_max,
        _finite(
            disclosure["maximum_same_q_position_error_m"],
            "disclosed feasibility position error",
        ),
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        probe_orientation_max,
        _finite(
            disclosure["maximum_same_q_orientation_error_rad"],
            "disclosed feasibility orientation error",
        ),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CommonFkAnalysisError(
            "frozen bytes do not reproduce the disclosed feasibility probe"
        )
    if max_ov_position > _finite(
        gates["identity_world_frame_t0_all_fingertips_max_position_error_m"],
        "identity position gate",
    ) or max_ov_orientation > _finite(
        gates["identity_world_frame_t0_all_fingertips_max_orientation_error_rad"],
        "identity orientation gate",
    ):
        raise CommonFkAnalysisError(
            "identity-only world-frame audit failed; XYZ attribution is forbidden"
        )
    if max_candidate_position > _finite(
        gates["mujoco_self_replay_max_position_error_m"], "self position gate"
    ) or max_candidate_orientation > _finite(
        gates["mujoco_self_replay_max_orientation_error_rad"],
        "self orientation gate",
    ):
        raise CommonFkAnalysisError("MuJoCo t=0 self replay failed")
    if max_unit_error > _finite(
        gates["quaternion_unit_norm_max_abs_error"], "quaternion unit gate"
    ):
        raise CommonFkAnalysisError("frame quaternion unit-norm audit failed")
    return {
        "transform": "identity_only_no_fit",
        "fit_performed": False,
        "validation_step": 0,
        "validation_time_s": 0.0,
        "ov_all_fingertips_position_error_max_m": max_ov_position,
        "ov_all_fingertips_orientation_error_max_rad": max_ov_orientation,
        "candidate_self_replay_position_error_max_m": max_candidate_position,
        "candidate_self_replay_orientation_error_max_rad": max_candidate_orientation,
        "quaternion_unit_norm_max_abs_error": max_unit_error,
        "per_frame": per_frame,
        "candidate_rows": candidate_rows,
        "disclosed_feasibility_probe": {
            "recomputed": True,
            "step": disclosed_step,
            "time_s": probe_sample.time_s,
            "all_fingertips_position_error_max_m": probe_position_max,
            "all_fingertips_orientation_error_max_rad": probe_orientation_max,
        },
        "passed": True,
    }


def _as_xyz(pose: Sequence[float]) -> np.ndarray:
    if len(pose) != 7:
        raise CommonFkAnalysisError("frame pose must have seven values")
    result = np.asarray(pose[:3], dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise CommonFkAnalysisError("frame position is not a finite XYZ vector")
    return result


def _candidate_replay(
    adapter: MuJoCoAdapter,
    *,
    role: str,
    ov: CollectedRun,
    candidate: CollectedRun,
    manifest: ParityManifest,
    plan: Mapping[str, object],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    hand = manifest.hand(HandSide.LEFT)
    ov_samples = _sample_map(ov)
    candidate_samples = _sample_map(candidate)
    raw_rows: list[np.ndarray] = []
    fk_rows: list[np.ndarray] = []
    coalition_rows = np.zeros((32, END_STEP - START_STEP + 1, 3), dtype=float)
    records: list[dict[str, object]] = []
    max_self_position = 0.0
    max_self_orientation = 0.0
    max_empty_error = 0.0
    max_full_error = 0.0
    max_nonthumb_error = 0.0
    gates = _mapping(plan["sanity_gates"], "sanity gates")
    nonthumb = tuple(name for name in hand.joint_names if name not in PLAYER_NAMES)
    for row_index, step in enumerate(range(START_STEP, END_STEP + 1)):
        if step not in ov_samples or step not in candidate_samples:
            raise CommonFkAnalysisError("frozen analysis window is missing a step")
        ov_sample = ov_samples[step]
        candidate_sample = candidate_samples[step]
        if not math.isclose(ov_sample.time_s, step * DT_S, rel_tol=0.0, abs_tol=1e-12):
            raise CommonFkAnalysisError("OV analysis window time axis changed")
        if not math.isclose(
            candidate_sample.time_s,
            step * DT_S,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CommonFkAnalysisError(
                "candidate analysis window time axis changed"
            )

        candidate_fk = replay_mujoco_fk_named(
            adapter,
            candidate_sample.joint_positions,
            expected_joint_names=hand.joint_names,
            frame_names=(FRAME_NAME,),
            step=step,
        )
        ov_fk = replay_mujoco_fk_named(
            adapter,
            ov_sample.joint_positions,
            expected_joint_names=hand.joint_names,
            frame_names=(FRAME_NAME,),
            step=step,
        )
        self_position, self_orientation, _ = _pose_error(
            candidate_sample.frame_poses[FRAME_NAME],
            candidate_fk.frame_poses[FRAME_NAME],
        )
        max_self_position = max(max_self_position, self_position)
        max_self_orientation = max(max_self_orientation, self_orientation)
        raw = _as_xyz(ov_sample.frame_poses[FRAME_NAME]) - _as_xyz(
            candidate_sample.frame_poses[FRAME_NAME]
        )
        fk = _as_xyz(ov_fk.frame_poses[FRAME_NAME]) - _as_xyz(
            candidate_fk.frame_poses[FRAME_NAME]
        )
        raw_rows.append(raw)
        fk_rows.append(fk)
        candidate_origin = _as_xyz(candidate_fk.frame_poses[FRAME_NAME])
        coalition_records: list[dict[str, object]] = []
        for mask in range(32):
            hybrid = {
                name: float(candidate_sample.joint_positions[name])
                for name in hand.joint_names
            }
            members: list[str] = []
            for player_index, name in enumerate(PLAYER_NAMES):
                if mask & (1 << player_index):
                    hybrid[name] = float(ov_sample.joint_positions[name])
                    members.append(name)
            hybrid_fk = replay_mujoco_fk_named(
                adapter,
                hybrid,
                expected_joint_names=hand.joint_names,
                frame_names=(FRAME_NAME,),
                step=step,
            )
            xyz = _as_xyz(hybrid_fk.frame_poses[FRAME_NAME])
            displacement = xyz - candidate_origin
            coalition_rows[mask, row_index] = displacement
            coalition_records.append(
                {
                    "mask": mask,
                    "members": members,
                    "common_fk_xyz_m": xyz.tolist(),
                    "displacement_from_candidate_fk_m": displacement.tolist(),
                }
            )
        empty_error = math.hypot(*(float(value) for value in coalition_rows[0, row_index]))
        full_error = math.hypot(
            *(float(value) for value in coalition_rows[-1, row_index] - fk)
        )
        max_empty_error = max(max_empty_error, empty_error)
        max_full_error = max(max_full_error, full_error)

        nonthumb_hybrid = {
            name: float(ov_sample.joint_positions[name]) for name in hand.joint_names
        }
        for name in nonthumb:
            nonthumb_hybrid[name] = float(candidate_sample.joint_positions[name])
        nonthumb_fk = replay_mujoco_fk_named(
            adapter,
            nonthumb_hybrid,
            expected_joint_names=hand.joint_names,
            frame_names=(FRAME_NAME,),
            step=step,
        )
        nonthumb_error = math.hypot(
            *(
                float(value)
                for value in _as_xyz(nonthumb_fk.frame_poses[FRAME_NAME])
                - _as_xyz(ov_fk.frame_poses[FRAME_NAME])
            )
        )
        max_nonthumb_error = max(max_nonthumb_error, nonthumb_error)
        records.append(
            {
                "step": step,
                "time_s": ov_sample.time_s,
                "raw_ov_xyz_m": list(ov_sample.frame_poses[FRAME_NAME][:3]),
                "raw_candidate_xyz_m": list(
                    candidate_sample.frame_poses[FRAME_NAME][:3]
                ),
                "common_fk_ov_xyz_m": list(ov_fk.frame_poses[FRAME_NAME][:3]),
                "common_fk_candidate_xyz_m": list(
                    candidate_fk.frame_poses[FRAME_NAME][:3]
                ),
                "raw_delta_m": raw.tolist(),
                "common_fk_delta_m": fk.tolist(),
                "closure_residual_m": (raw - fk).tolist(),
                "coalitions": coalition_records,
            }
        )
    raw_array = np.asarray(raw_rows, dtype=np.float64)
    fk_array = np.asarray(fk_rows, dtype=np.float64)
    if max_self_position > _finite(
        gates["mujoco_self_replay_max_position_error_m"], "self position gate"
    ) or max_self_orientation > _finite(
        gates["mujoco_self_replay_max_orientation_error_rad"],
        "self orientation gate",
    ):
        raise CommonFkAnalysisError(f"{role} MuJoCo self replay failed")
    if max_empty_error > EMPTY_COALITION_ATOL_M:
        raise CommonFkAnalysisError(f"{role} empty coalition is not identity")
    if max_full_error > FULL_COALITION_ATOL_M:
        raise CommonFkAnalysisError(f"{role} full coalition does not reach OV FK")
    if max_nonthumb_error > _finite(
        gates["non_thumb_joint_invariance_max_position_error_m"],
        "non-thumb invariance gate",
    ):
        raise CommonFkAnalysisError(f"{role} non-thumb invariance failed")
    return (
        {
            "role": role,
            "case_id": candidate.case.case_id,
            "window_start_step": START_STEP,
            "window_end_step": END_STEP,
            "sample_count": END_STEP - START_STEP + 1,
            "mujoco_self_replay_position_error_max_m": max_self_position,
            "mujoco_self_replay_orientation_error_max_rad": max_self_orientation,
            "empty_coalition_error_max_m": max_empty_error,
            "full_coalition_vs_ov_fk_error_max_m": max_full_error,
            "non_thumb_invariance_error_max_m": max_nonthumb_error,
            "samples": records,
        },
        raw_array,
        fk_array,
        coalition_rows,
    )


def _attribution_row(
    role: str,
    raw: np.ndarray,
    fk: np.ndarray,
    coalitions: np.ndarray,
) -> dict[str, object]:
    result = analyze_kinematic_attribution(raw, fk, coalitions)
    peak_step = START_STEP + result.closure.peak_index
    players = []
    for name, player in zip(PLAYER_NAMES, result.players):
        players.append(
            {
                "canonical_id": name,
                "player_index": player.player_index,
                "energy_attribution_m2": player.energy_attribution_m2,
                "normalized_attribution_fraction_points": player.normalized_attribution,
                "positive_attribution_share": player.positive_share,
                "label": player.label,
            }
        )
    ordered = sorted(
        players,
        key=lambda item: (
            -float(item["normalized_attribution_fraction_points"]),
            int(item["player_index"]),
        ),
    )
    row: dict[str, object] = {
        "role": role,
        "closure": {
            **asdict(result.closure),
            "peak_step": peak_step,
            "peak_time_s": peak_step * DT_S,
        },
        "raw_energy_m2": result.raw_energy_m2,
        "empty_coalition_value": result.empty_coalition_value,
        "full_coalition_value": result.full_coalition_value,
        "energy_efficiency_error_fraction": result.efficiency_error,
        "players_in_canonical_order": players,
        "players_ranked_by_signed_energy_fraction": ordered,
    }
    vector_values = getattr(result, "vector_attributions_m", None)
    vector_error = getattr(result, "vector_efficiency_max_error_m", None)
    if vector_values is None or vector_error is None:
        raise CommonFkAnalysisError("attribution module lacks vector Shapley audit")
    # Keep the in-memory summary JSON-native so the persisted round-trip audit
    # compares like-for-like instead of tuple objects against decoded lists.
    row["vector_shapley_m_by_player_sample_xyz"] = [
        [list(xyz) for xyz in player_samples]
        for player_samples in vector_values
    ]
    row["vector_efficiency_max_error_m"] = vector_error
    if float(vector_error) > VECTOR_EFFICIENCY_ATOL_M:
        raise CommonFkAnalysisError("vector Shapley efficiency gate failed")
    return row


def _overall_decision(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    statuses = []
    for row in rows:
        closure = _mapping(row["closure"], "closure")
        status = closure.get("status")
        if status not in {CLOSES, "MIXED", DOES_NOT_CLOSE}:
            raise CommonFkAnalysisError("invalid closure status")
        statuses.append(str(status))
    if all(status == CLOSES for status in statuses):
        scientific_status = (
            "kinematically_consistent_with_recorded_joint_state_gap_under_common_mujoco_fk"
        )
        supported = True
    elif any(status == DOES_NOT_CLOSE for status in statuses):
        scientific_status = (
            "common_mujoco_fk_closure_does_not_hold_for_both_candidates"
        )
        supported = False
    else:
        scientific_status = "mixed_or_inconclusive"
        supported = False
    return {
        "candidate_statuses": {
            role: status for role, status in zip(ROLE_ORDER, statuses)
        },
        "scientific_status": scientific_status,
        "common_mujoco_fk_closure_supported_for_both_candidates": supported,
        "formal_gate0_status": "DIVERGENT",
        "formal_gate0_pass_ready": False,
        "formal_gate0_result_changed": False,
    }


def _compute_analysis(
    *,
    asset_root: Path,
    manifest: ParityManifest,
    plan: Mapping[str, object],
    runs: Mapping[str, CollectedRun],
) -> tuple[dict[str, object], dict[str, object]]:
    ov = runs["formal_ovphysx_run"]
    candidates = {role: runs[role] for role in ROLE_ORDER}
    adapter = MuJoCoAdapter(asset_root=asset_root)
    adapter.open(manifest.hand(HandSide.LEFT))
    try:
        if getattr(adapter._mujoco, "__version__", None) != MUJOCO_VERSION:
            raise CommonFkAnalysisError("local MuJoCo version differs from frozen version")
        topology = _topology_audit(adapter)
        coordinate = _identity_coordinate_audit(
            adapter,
            ov=ov,
            candidates=candidates,
            manifest=manifest,
            plan=plan,
        )
        replay_candidates: list[dict[str, object]] = []
        attribution_rows: list[dict[str, object]] = []
        for role in ROLE_ORDER:
            replay, raw, fk, coalitions = _candidate_replay(
                adapter,
                role=role,
                ov=ov,
                candidate=candidates[role],
                manifest=manifest,
                plan=plan,
            )
            replay_candidates.append(replay)
            attribution_rows.append(_attribution_row(role, raw, fk, coalitions))
    finally:
        adapter.close()
    decision = _overall_decision(attribution_rows)
    replay_payload = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "fk_semantics": {
            "backend": "mujoco",
            "version": MUJOCO_VERSION,
            "device": "cpu",
            "headless": True,
            "renderer": False,
            "fixed_base": True,
            "joint_input": "all 22 canonical TraceSample.joint_positions",
            "reset_before_each_replay": True,
            "qvel_zero": True,
            "kinematics_refresh": "mj_forward then mj_kinematics through MuJoCoAdapter",
            "dynamics_steps": 0,
            "control_targets_used_for_fk": False,
            "frame_pose": "xpos metres and xquat qwxyz",
            "coordinate_transform": "identity_only_no_fit",
        },
        "topology_audit": topology,
        "coordinate_audit": coordinate,
        "candidates": replay_candidates,
    }
    derived = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "classification": plan["classification"],
        "analysis_plan_status": plan["analysis_plan_status"],
        "feasibility_disclosure": plan["feasibility_disclosure"],
        "frame": FRAME_NAME,
        "window_start_step": START_STEP,
        "window_end_step": END_STEP,
        "sample_count": END_STEP - START_STEP + 1,
        "candidate_results": attribution_rows,
        "decision": decision,
        "claim_boundary": plan["claim_boundary"],
    }
    return replay_payload, derived


def _render_report(summary: Mapping[str, object]) -> str:
    decision = _mapping(summary["decision"], "decision")
    disclosure = _mapping(
        summary["feasibility_disclosure"], "feasibility disclosure"
    )
    rows = summary["candidate_results"]
    if not isinstance(rows, list):
        raise CommonFkAnalysisError("summary candidate results must be an array")
    lines = [
        "# Left thumb_DP common-MuJoCo-FK attribution",
        "",
        "> Private raw diagnostic bundle. Sanitise before publication.",
        "> Offline, simulation-only, post-hoc analysis; not a formal Gate 0 rerun.",
        "",
        "## Result",
        "",
        f"Scientific status: `{decision['scientific_status']}`.",
        "Formal Gate 0 remains `DIVERGENT / pass_ready=false`.",
        "",
        "The method replays the recorded 22 canonical joint positions from both simulators through one fixed-base MuJoCo model. It uses no dynamics step and fits no translation, rotation, scale, offset, or time shift.",
        "",
        "The full-window plan was frozen after a disclosed one-step feasibility check at step {step} ({time:.3f} s). This is therefore a post-hoc analysis, not a blind confirmation.".format(
            step=disclosure["step"],
            time=float(disclosure["time_s"]),
        ),
        "",
        "## Closure",
        "",
        "| Candidate | Window residual/raw | Raw-peak residual/raw | Status | Raw peak |",
        "|---|---:|---:|---|---:|",
    ]
    for raw_row in rows:
        row = _mapping(raw_row, "candidate row")
        closure = _mapping(row["closure"], "candidate closure")
        lines.append(
            "| {role} | {window:.6g} | {peak:.6g} | {status} | step {step} ({time:.3f} s) |".format(
                role=row["role"],
                window=float(closure["rho_window"]),
                peak=float(closure["rho_peak"]),
                status=closure["status"],
                step=closure["peak_step"],
                time=float(closure["peak_time_s"]),
            )
        )
    lines.extend(["", "## Five-joint exact Shapley attribution", ""])
    for raw_row in rows:
        row = _mapping(raw_row, "candidate row")
        lines.extend(
            [
                f"### {row['role']}",
                "",
                "| Joint | Signed raw-gap energy fraction | Positive share | Label |",
                "|---|---:|---:|---|",
            ]
        )
        players = row["players_ranked_by_signed_energy_fraction"]
        if not isinstance(players, list):
            raise CommonFkAnalysisError("ranked players must be an array")
        for raw_player in players:
            player = _mapping(raw_player, "player")
            lines.append(
                "| {name} | {fraction:.3%} | {share:.3%} | {label} |".format(
                    name=player["canonical_id"],
                    fraction=float(player["normalized_attribution_fraction_points"]),
                    share=float(player["positive_attribution_share"]),
                    label=player["label"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "The attribution is a counterfactual allocation under one common kinematic model. It is not a dynamics cause, a physical parameter estimate, or evidence of an official backend bug.",
            "",
            "## Evidence identity",
            "",
            f"- Analysis plan SHA-256: `{summary['analysis_plan_sha256']}`",
            f"- Parent bundle root: `{summary['parent_bundle_root_sha256']}`",
            f"- Analysis source revision/tree: `{summary['analysis_source_revision']}` / `{summary['analysis_source_tree']}`",
            f"- MuJoCo model SHA-256: `{summary['mujoco_model_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> Path:
    try:
        content = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (OverflowError, TypeError, ValueError) as exc:
        raise CommonFkAnalysisError("analysis output is not finite JSON") from exc
    return write_text_exclusive(path, content)


def _copy_inputs(
    staging: Path,
    *,
    project_root: Path,
    parent_root: Path,
    parent_members: Mapping[str, Path],
    model_path: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    source_copies: dict[str, Path] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        source = (project_root / relative).resolve(strict=True)
        destination = staging / "inputs" / "analysis-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_copies[relative] = destination
    copied_parent_manifest = staging / "inputs" / "parent" / "bundle.json"
    copied_parent_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_root / "bundle.json", copied_parent_manifest)
    parent_copies: dict[str, Path] = {}
    for name, source in parent_members.items():
        destination = staging / "inputs" / "parent" / name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        parent_copies[name] = destination
    copied_model = staging / "inputs" / "asset" / "left_sharpa_wave.xml"
    copied_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, copied_model)
    return source_copies, {**parent_copies, "bundle_manifest": copied_parent_manifest, "model": copied_model}


def _verify_frozen_copies(
    *,
    plan: Mapping[str, object],
    source_copies: Mapping[str, Path],
    source_hashes: Mapping[str, str],
    input_copies: Mapping[str, Path],
) -> None:
    parent_paths = _mapping(plan["parent_paths"], "parent paths")
    expected_input_keys = set(parent_paths) | {"bundle_manifest", "model"}
    if set(input_copies) != expected_input_keys:
        raise CommonFkAnalysisError("copied frozen input inventory changed")
    if set(source_copies) != set(SOURCE_RELATIVE_PATHS) or set(source_hashes) != set(
        SOURCE_RELATIVE_PATHS
    ):
        raise CommonFkAnalysisError("copied analysis source inventory changed")
    if sha256_file(input_copies["bundle_manifest"]) != PARENT_MANIFEST_SHA256:
        raise CommonFkAnalysisError("copied parent bundle manifest differs")
    copied_payloads = {
        str(parent_paths[name]): input_copies[name] for name in parent_paths
    }
    verify_copied_bundle_subset(
        input_copies["bundle_manifest"],
        expected_root_sha256=PARENT_ROOT_SHA256,
        copied_payloads=copied_payloads,
    )
    if sha256_file(input_copies["model"]) != MODEL_SHA256:
        raise CommonFkAnalysisError("copied MuJoCo model differs")
    for relative in SOURCE_RELATIVE_PATHS:
        if sha256_file(source_copies[relative]) != source_hashes[relative]:
            raise CommonFkAnalysisError(f"copied analysis source differs: {relative}")


def _remove_failed_staging(staging: Path, project_root: Path) -> None:
    """Remove only the incomplete directory created for this analysis."""

    try:
        resolved = staging.resolve(strict=True)
        results_root = (project_root / "results").resolve(strict=True)
    except OSError as exc:
        raise CommonFkAnalysisError(
            "cannot verify failed staging path before cleanup"
        ) from exc
    if (
        resolved.parent != results_root
        or not resolved.name.startswith(".")
        or ".incomplete-" not in resolved.name
    ):
        raise CommonFkAnalysisError(
            f"refusing to recursively remove unverified staging path: {resolved}"
        )
    shutil.rmtree(resolved)


def _expected_payload_paths(
    parent_members: Mapping[str, Path],
) -> set[str]:
    expected = {
        f"inputs/analysis-source/{relative}" for relative in SOURCE_RELATIVE_PATHS
    }
    expected.add("inputs/parent/bundle.json")
    expected.update(
        f"inputs/parent/{name}/{source.name}"
        for name, source in parent_members.items()
    )
    expected.update(
        {
            "inputs/asset/left_sharpa_wave.xml",
            "replay.json",
            "summary.json",
            "report.md",
        }
    )
    return expected


def _verify_live_assets(
    *,
    manifest_path: Path,
    asset_root: Path,
    plan: Mapping[str, object],
    expected_model_path: Path,
    expected_tree_sha256: str,
) -> ParityManifest:
    manifest, model_path, tree_sha256 = _validate_manifest_and_assets(
        manifest_path, asset_root, plan
    )
    if model_path != expected_model_path or tree_sha256 != expected_tree_sha256:
        raise CommonFkAnalysisError("live asset identity changed during analysis")
    return manifest


def _audit_persisted_analysis(
    *,
    asset_root: Path,
    manifest_path: Path,
    plan_path: Path,
    staged_runs: Mapping[str, Path],
    expected_replay: Mapping[str, object],
    expected_derived: Mapping[str, object],
    expected_summary: Mapping[str, object],
    replay_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    audit_plan = _load_plan(plan_path)
    audit_manifest = load_manifest(manifest_path)
    replay_audit, derived_audit = _compute_analysis(
        asset_root=asset_root,
        manifest=audit_manifest,
        plan=audit_plan,
        runs={
            "formal_ovphysx_run": _load_one(staged_runs["formal_ovphysx_run"], "audit OV run"),
            ROLE_CONTROL: _load_one(staged_runs[ROLE_CONTROL], "audit control run"),
            ROLE_TREATMENT: _load_one(staged_runs[ROLE_TREATMENT], "audit treatment run"),
        },
    )
    if replay_audit != expected_replay or derived_audit != expected_derived:
        raise CommonFkAnalysisError("staged-byte recomputation changed the result")
    if _load_json_object(replay_path, "staged replay") != expected_replay:
        raise CommonFkAnalysisError("persisted replay differs from recomputation")
    if _load_json_object(summary_path, "staged summary") != expected_summary:
        raise CommonFkAnalysisError("persisted summary differs from recomputation")
    if report_path.read_text(encoding="utf-8") != _render_report(expected_summary):
        raise CommonFkAnalysisError("persisted report differs from recomputation")


def _run_analysis(
    *,
    project_root: Path,
    source_revision: str,
    asset_root: Path,
    parent_bundle: Path,
    manifest_path: Path,
    plan_path: Path,
    output_dir: Path,
    session_id: str,
) -> Path:
    if SESSION_PATTERN.fullmatch(session_id) is None:
        raise CommonFkAnalysisError("session_id is invalid")
    if project_root != PROJECT_ROOT:
        raise CommonFkAnalysisError("project root must be this checkout")
    if plan_path != (project_root / PLAN_RELATIVE_PATH).resolve(strict=True):
        raise CommonFkAnalysisError("analysis plan path is not canonical")
    if manifest_path != (project_root / MANIFEST_RELATIVE_PATH).resolve(strict=True):
        raise CommonFkAnalysisError("manifest path is not canonical")
    plan = _load_plan(plan_path)
    source_tree = validate_git_source(
        project_root,
        source_revision,
        required_tracked_paths=SOURCE_RELATIVE_PATHS,
        sensitive_untracked_paths=SOURCE_RELATIVE_PATHS,
    )
    source_hashes = {
        relative: sha256_file(project_root / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    parent_verification, parent_members = _validate_parent(parent_bundle, plan)
    manifest, model_path, observed_asset_tree = _validate_manifest_and_assets(
        manifest_path, asset_root, plan
    )

    runs = {
        name: _load_one(parent_members[name], name)
        for name in (
            "formal_ovphysx_run",
            "friction_zero_run",
            "same_source_control_run",
            "total_viscous_treatment_run",
        )
    }
    for name, run in runs.items():
        _validate_run_provenance(run, manifest=manifest, plan=plan, role=name)
    if (
        runs["same_source_control_run"].result.samples
        != runs["friction_zero_run"].result.samples
    ):
        raise CommonFkAnalysisError(
            "same-source control no longer exactly reproduces frozen friction=0 samples"
        )
    candidates = {
        ROLE_CONTROL: runs["same_source_control_run"],
        ROLE_TREATMENT: runs["total_viscous_treatment_run"],
    }
    compute_runs = {
        "formal_ovphysx_run": runs["formal_ovphysx_run"],
        **candidates,
    }

    final_path = validated_results_output_path(project_root, output_dir)
    staging = create_staging_root(final_path)
    try:
        source_copies, input_copies = _copy_inputs(
            staging,
            project_root=project_root,
            parent_root=parent_bundle,
            parent_members=parent_members,
            model_path=model_path,
        )
        _verify_frozen_copies(
            plan=plan,
            source_copies=source_copies,
            source_hashes=source_hashes,
            input_copies=input_copies,
        )

        copied_manifest = input_copies["gate0_manifest"]
        staged_manifest = _verify_live_assets(
            manifest_path=copied_manifest,
            asset_root=asset_root,
            plan=plan,
            expected_model_path=model_path,
            expected_tree_sha256=observed_asset_tree,
        )
        staged_run_paths = {
            "formal_ovphysx_run": input_copies["formal_ovphysx_run"],
            ROLE_CONTROL: input_copies["same_source_control_run"],
            ROLE_TREATMENT: input_copies["total_viscous_treatment_run"],
        }
        staged_runs = {
            "formal_ovphysx_run": _load_one(
                staged_run_paths["formal_ovphysx_run"], "staged OV run"
            ),
            ROLE_CONTROL: _load_one(
                staged_run_paths[ROLE_CONTROL], "staged control run"
            ),
            ROLE_TREATMENT: _load_one(
                staged_run_paths[ROLE_TREATMENT], "staged treatment run"
            ),
        }
        replay, derived = _compute_analysis(
            asset_root=asset_root,
            manifest=staged_manifest,
            plan=plan,
            runs=staged_runs,
        )
        replay_path = _write_json_exclusive(staging / "replay.json", replay)

        summary: dict[str, object] = {
            **derived,
            "execution_status": "completed",
            "session_id": session_id,
            "analysis_plan_sha256": PLAN_SHA256,
            "parent_bundle_root_sha256": PARENT_ROOT_SHA256,
            "parent_bundle_manifest_sha256": PARENT_MANIFEST_SHA256,
            "parent_bundle_verification": parent_verification,
            "parent_input_files_sha256": {
                name: sha256_file(path)
                for name, path in sorted(parent_members.items())
            },
            "parent_input_paths": dict(
                _mapping(plan["parent_paths"], "parent paths")
            ),
            "analysis_source_revision": source_revision,
            "analysis_source_tree": source_tree,
            "analysis_source_files_sha256": source_hashes,
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_semantic_sha256": manifest_sha256(manifest),
            "asset_commit": manifest.provenance.commit,
            "asset_git_tree": manifest.provenance.asset_git_tree,
            "canonical_lf_asset_tree_sha256": observed_asset_tree,
            "mujoco_model_sha256": sha256_file(model_path),
            "mujoco_version": MUJOCO_VERSION,
            "device": "cpu",
            "headless": True,
            "renderer": False,
            "raw_bundle_visibility": "private",
            "replay_sha256": sha256_file(replay_path),
        }
        summary_path = _write_json_exclusive(staging / "summary.json", summary)
        report_path = write_text_exclusive(
            staging / "report.md", _render_report(summary)
        )

        _verify_live_assets(
            manifest_path=copied_manifest,
            asset_root=asset_root,
            plan=plan,
            expected_model_path=model_path,
            expected_tree_sha256=observed_asset_tree,
        )
        _audit_persisted_analysis(
            asset_root=asset_root,
            manifest_path=copied_manifest,
            plan_path=source_copies[PLAN_RELATIVE_PATH],
            staged_runs=staged_run_paths,
            expected_replay=replay,
            expected_derived=derived,
            expected_summary=summary,
            replay_path=replay_path,
            summary_path=summary_path,
            report_path=report_path,
        )

        if validate_git_source(
            project_root,
            source_revision,
            required_tracked_paths=SOURCE_RELATIVE_PATHS,
            sensitive_untracked_paths=SOURCE_RELATIVE_PATHS,
        ) != source_tree:
            raise CommonFkAnalysisError("analysis Git source changed during execution")
        if {
            relative: sha256_file(project_root / relative)
            for relative in SOURCE_RELATIVE_PATHS
        } != source_hashes:
            raise CommonFkAnalysisError("analysis source files changed during execution")
        _verify_live_assets(
            manifest_path=copied_manifest,
            asset_root=asset_root,
            plan=plan,
            expected_model_path=model_path,
            expected_tree_sha256=observed_asset_tree,
        )
        _verify_frozen_copies(
            plan=plan,
            source_copies=source_copies,
            source_hashes=source_hashes,
            input_copies=input_copies,
        )
        expected_payloads = _expected_payload_paths(parent_members)
        observed_payloads = set(bundle_payload_paths(staging))
        if observed_payloads != expected_payloads:
            raise CommonFkAnalysisError(
                "analysis bundle inventory is not closed: "
                f"missing={sorted(expected_payloads - observed_payloads)}, "
                f"extra={sorted(observed_payloads - expected_payloads)}"
            )
        write_bundle_manifest(staging, sorted(expected_payloads))
        staged_verification = verify_exact_bundle(staging)

        _verify_live_assets(
            manifest_path=copied_manifest,
            asset_root=asset_root,
            plan=plan,
            expected_model_path=model_path,
            expected_tree_sha256=observed_asset_tree,
        )
        _audit_persisted_analysis(
            asset_root=asset_root,
            manifest_path=copied_manifest,
            plan_path=source_copies[PLAN_RELATIVE_PATH],
            staged_runs=staged_run_paths,
            expected_replay=replay,
            expected_derived=derived,
            expected_summary=summary,
            replay_path=replay_path,
            summary_path=summary_path,
            report_path=report_path,
        )
        _verify_live_assets(
            manifest_path=copied_manifest,
            asset_root=asset_root,
            plan=plan,
            expected_model_path=model_path,
            expected_tree_sha256=observed_asset_tree,
        )
        _verify_frozen_copies(
            plan=plan,
            source_copies=source_copies,
            source_hashes=source_hashes,
            input_copies=input_copies,
        )
        if validate_git_source(
            project_root,
            source_revision,
            required_tracked_paths=SOURCE_RELATIVE_PATHS,
            sensitive_untracked_paths=SOURCE_RELATIVE_PATHS,
        ) != source_tree:
            raise CommonFkAnalysisError(
                "analysis Git source changed during post-manifest audit"
            )
        post_audit_verification = verify_exact_bundle(staging)
        if post_audit_verification != staged_verification:
            raise CommonFkAnalysisError(
                "bundle bytes changed during post-manifest semantic audit"
            )
        promoted = promote_staging_root(staging, final_path)
        promoted_verification = verify_exact_bundle(promoted)
        if promoted_verification != staged_verification:
            raise CommonFkAnalysisError("bundle identity changed during promotion")
        return promoted
    except BaseException:
        if staging.exists():
            _remove_failed_staging(staging, project_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--parent-bundle", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / MANIFEST_RELATIVE_PATH
    )
    parser.add_argument(
        "--plan", type=Path, default=PROJECT_ROOT / PLAN_RELATIVE_PATH
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _run_analysis(
            project_root=PROJECT_ROOT,
            source_revision=args.source_revision,
            asset_root=_validated_real_asset_root(args.asset_root),
            parent_bundle=args.parent_bundle.expanduser().resolve(strict=True),
            manifest_path=args.manifest.expanduser().resolve(strict=True),
            plan_path=args.plan.expanduser().resolve(strict=True),
            output_dir=args.output_dir,
            session_id=args.session_id,
        )
    except (
        CommonFkAnalysisError,
        DiagnosticEvidenceError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"common-FK analysis failed: {exc}", file=sys.stderr)
        return 2
    verification = verify_exact_bundle(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bundle_root_sha256": verification["root_sha256"],
                "file_count": verification["file_count"],
                "total_size": verification["total_size"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
