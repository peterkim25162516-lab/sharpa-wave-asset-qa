#!/usr/bin/env python3
"""Derive the private, hash-frozen OVPhysX legacy-friction Freeze B plan.

The plan is generated only from the immutable R1 readback and formal Gate 0
bundles.  Per-joint legacy values remain in the ignored private plan; stdout
contains hashes and counts only.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Mapping, Sequence

from wave_asset_qa.parity.bundle import sha256_file
from wave_asset_qa.parity.contracts import HandSide
from wave_asset_qa.parity.diagnostics import verify_exact_bundle
from wave_asset_qa.parity.runner import load_collected_runs
from wave_asset_qa.parity.scenarios import load_manifest, manifest_sha256
from wave_asset_qa.parity.sensitivity import private_plan_sha256, validate_private_plan


PROTOCOL_ID = "ovphysx-legacy-joint-friction-freeze-b-v1"
READBACK_ROOT_SHA256 = (
    "858eee53e0ece9956a06b7cbd72a3d0a8348156e3c236ebf31ffd72fceb4fd4a"
)
FORMAL_ROOT_SHA256 = (
    "d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505"
)
FREEZE_A_CONFIG_SHA256 = (
    "12c89fd8e0cd3bfa849a96a2e133b22e9191fc0ca6188f18fde4960d4221ab4c"
)
READBACK_SUMMARY_SHA256 = (
    "d359ffd9622907658edc3c412b1870aab8ae717d1a5805f77bc7b5f530b2d055"
)
READBACK_SOURCE_REVISION = "44571ea60201f77ca93d49c9c14cb2d28152c30f"
READBACK_SOURCE_TREE = "d2a03e8ecceb46971e5c75debdb738baea63a4a3"
FORMAL_SOURCE_REVISION = "c571d9bb37a619fe9e867447d3be448acc615e3c"
GATE0_MANIFEST_FILE_SHA256 = (
    "a95a1da0d38abac8b3eccf18ea0b618b4898e35a6102017a2f0e024a9a4d5244"
)
GATE0_MANIFEST_SEMANTIC_SHA256 = (
    "576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e"
)
ASSET_COMMIT = "6eea427eb24189519f32b9f21674cd534d3f973c"
ASSET_GIT_TREE = "bb00a9d5527b8a76de576ce876ebece67d8ffde1"
ASSET_TREE_SHA256 = (
    "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
)
READBACK_CASE_SHA256 = {
    "ovphysx.left.effective_readback.r01": (
        "df41b21346d39a1d81d33817b35766525c8438dc1b3898023f5410f7deaf940f"
    ),
    "ovphysx.left.effective_readback.r02": (
        "2eed27934d2ddf6cd8017690b7005b03c209fdfc2bd994192c278f25ed30daa5"
    ),
    "ovphysx.right.effective_readback.r01": (
        "8286ce3b6bdee966beb588e8433f87ec7d49f12af694bda8cf77c2f616d6e3d5"
    ),
    "ovphysx.right.effective_readback.r02": (
        "71d1a6a8dbe446ce43747d66a7e4579529a97b30baa495a179e4727b348de00a"
    ),
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class FreezeBPreparationError(RuntimeError):
    """Raised when a frozen input or generated plan fails closed."""


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeBPreparationError(f"cannot read JSON input {path.name}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FreezeBPreparationError(f"JSON input {path.name} must be an object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeBPreparationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FreezeBPreparationError(f"{label} must be a finite number")
    return result


def _float32(value: object, label: str) -> tuple[float, bytes]:
    numeric = _finite(value, label)
    try:
        packed = struct.pack("!f", numeric)
    except (OverflowError, struct.error) as error:
        raise FreezeBPreparationError(f"{label} is not finite float32") from error
    canonical = struct.unpack("!f", packed)[0]
    if not math.isfinite(canonical):
        raise FreezeBPreparationError(f"{label} is not finite float32")
    return canonical, packed


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FreezeBPreparationError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FreezeBPreparationError(f"{label} must be an array")
    return value


def _validate_r1_runtime_friction(value: object, joint_name: str) -> None:
    runtime = _mapping(value, f"{joint_name} runtime friction")
    expected_runtime_keys = {
        "binding_controller_exact_match",
        "dof_friction_properties_binding",
        "idealpd_controller_buffers",
    }
    if set(runtime) != expected_runtime_keys:
        raise FreezeBPreparationError("R1 runtime friction fields are not exact")
    for field in (
        "dof_friction_properties_binding",
        "idealpd_controller_buffers",
    ):
        values = _mapping(runtime.get(field), f"{joint_name} {field}")
        if set(values) != {"static", "dynamic", "viscous"}:
            raise FreezeBPreparationError("R1 runtime friction slots are not exact")
        for key in ("static", "dynamic", "viscous"):
            if _float32(values.get(key), f"{joint_name} {field}.{key}")[1] != b"\0\0\0\0":
                raise FreezeBPreparationError(
                    "R1 runtime friction slots are not exact positive zero"
                )
    match = _mapping(
        runtime.get("binding_controller_exact_match"),
        f"{joint_name} runtime binding/controller match",
    )
    if match != {
        "overall": True,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }:
        raise FreezeBPreparationError(
            "R1 runtime binding/controller match is not exact"
        )


def _verify_git_source(project_root: Path, revision: str) -> str:
    if _HEX40.fullmatch(revision) is None:
        raise FreezeBPreparationError("source revision must be a 40-character commit")
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != revision:
        raise FreezeBPreparationError("source revision is not the checked-out HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise FreezeBPreparationError("source tree must be clean before plan generation")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _HEX40.fullmatch(tree) is None:
        raise FreezeBPreparationError("source tree identity is invalid")
    return tree


def _instance_maps(
    readback: Mapping[str, Any], manifest: object
) -> list[dict[str, object]]:
    if readback.get("status") != "READBACK_VALID":
        raise FreezeBPreparationError("R1 readback is not READBACK_VALID")
    instances = _array(readback.get("instances"), "readback.instances")
    if len(instances) != 4:
        raise FreezeBPreparationError("R1 readback must contain four fresh instances")
    by_side: dict[str, list[tuple[dict[str, float], dict[str, str]]]] = {
        "left": [],
        "right": [],
    }
    for raw_instance in instances:
        instance = _mapping(raw_instance, "readback instance")
        side = instance.get("side")
        if side not in by_side:
            raise FreezeBPreparationError("R1 instance has an invalid side")
        if instance.get("source_revision", READBACK_SOURCE_REVISION) not in {
            None,
            READBACK_SOURCE_REVISION,
        }:
            raise FreezeBPreparationError("R1 instance source revision drifted")
        provenance = _mapping(instance.get("provenance"), "instance.provenance")
        expected_provenance = {
            "source_revision": READBACK_SOURCE_REVISION,
            "source_tree": READBACK_SOURCE_TREE,
            "freeze_a_config_sha256": FREEZE_A_CONFIG_SHA256,
            "gate0_manifest_file_sha256": GATE0_MANIFEST_FILE_SHA256,
            "gate0_manifest_semantic_sha256": GATE0_MANIFEST_SEMANTIC_SHA256,
            "asset_commit": ASSET_COMMIT,
            "asset_git_tree": ASSET_GIT_TREE,
            "canonical_lf_asset_tree_sha256": ASSET_TREE_SHA256,
        }
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                raise FreezeBPreparationError(f"R1 provenance drifted at {key}")
        hand = manifest.hand(HandSide(side))
        records = _array(instance.get("dof_records"), "instance.dof_records")
        if len(records) != 22:
            raise FreezeBPreparationError("R1 instance must contain 22 DOF records")
        values: dict[str, float] = {}
        paths: dict[str, str] = {}
        for raw_record in records:
            record = _mapping(raw_record, "dof record")
            mapping = _mapping(record.get("mapping"), "dof.mapping")
            name = mapping.get("canonical_joint_name")
            if not isinstance(name, str) or name not in hand.joint_names or name in values:
                raise FreezeBPreparationError("R1 joint mapping is not exact")
            expected_path = f"/World/Env_0/Robot/joints/{name}"
            if mapping.get("joint_prim_path") != expected_path:
                raise FreezeBPreparationError("R1 joint prim path drifted")
            friction = _mapping(record.get("friction"), "dof.friction")
            authored = _mapping(friction.get("authored_value"), "friction.authored_value")
            resolved = _mapping(friction.get("resolved_value"), "friction.resolved_value")
            authored_value, authored_bits = _float32(
                authored.get("legacy_joint_friction_scalar"),
                f"{name} authored legacy friction",
            )
            resolved_value, resolved_bits = _float32(
                resolved.get("legacy_joint_friction_scalar"),
                f"{name} resolved legacy friction",
            )
            if authored_bits != resolved_bits or resolved_value <= 0.0:
                raise FreezeBPreparationError(
                    "R1 legacy friction must be positive and authored/resolved exact"
                )
            _validate_r1_runtime_friction(
                friction.get("runtime_effective_value"), name
            )
            values[name] = authored_value
            paths[name] = expected_path
        if tuple(values) != hand.joint_names:
            raise FreezeBPreparationError("R1 DOF record order differs from the manifest")
        by_side[str(side)].append((values, paths))

    plans: list[dict[str, object]] = []
    for side in ("left", "right"):
        pairs = by_side[side]
        if len(pairs) != 2 or pairs[0] != pairs[1]:
            raise FreezeBPreparationError(f"R1 {side} values are not exact across instances")
        values, paths = pairs[0]
        ordered_names = tuple(manifest.hand(HandSide(side)).joint_names)
        framed = b"waveqa-freeze-b-float32-v1\0"
        for name in ordered_names:
            encoded = name.encode("utf-8")
            framed += len(encoded).to_bytes(8, "big") + encoded
            framed += struct.pack("!f", values[name])
        plans.append(
            {
                "hand": side,
                "joint_names": list(ordered_names),
                "joint_prim_paths": {name: paths[name] for name in ordered_names},
                "expected_pre_values": {name: values[name] for name in ordered_names},
                "sham_write_values": {name: values[name] for name in ordered_names},
                "zero_write_values": {name: 0.0 for name in ordered_names},
                "expected_pre_float32_sha256": sha256(framed).hexdigest(),
            }
        )
    return plans


def _formal_run(formal_root: Path, backend: str, case_id: str) -> object:
    path = (
        formal_root
        / "evidence"
        / backend
        / "cases"
        / case_id
        / f"{case_id}.run.json"
    )
    runs = load_collected_runs(path)
    if len(runs) != 1 or not runs[0].result.completed:
        raise FreezeBPreparationError(f"formal case is unavailable: {case_id}")
    return runs[0]


def _window_rmse(left: object, right: object, *, frame_position: bool) -> float:
    left_samples = left.result.samples
    right_samples = right.result.samples
    if len(left_samples) != len(right_samples):
        raise FreezeBPreparationError("formal cross-sim sample counts differ")
    squared: list[float] = []
    for first, second in zip(left_samples, right_samples):
        if not math.isclose(first.time_s, second.time_s, rel_tol=0.0, abs_tol=1e-12):
            raise FreezeBPreparationError("formal cross-sim time axes differ")
        if first.time_s < 0.1 or first.time_s > 0.5:
            continue
        if frame_position:
            if tuple(first.frame_poses) != tuple(second.frame_poses):
                raise FreezeBPreparationError("formal frame order differs")
            for name in first.frame_poses:
                for axis in range(3):
                    delta = first.frame_poses[name][axis] - second.frame_poses[name][axis]
                    squared.append(delta * delta)
        else:
            if tuple(first.joint_positions) != tuple(second.joint_positions):
                raise FreezeBPreparationError("formal joint order differs")
            for name in first.joint_positions:
                delta = first.joint_positions[name] - second.joint_positions[name]
                squared.append(delta * delta)
    if not squared or not all(math.isfinite(value) for value in squared):
        raise FreezeBPreparationError("formal baseline window is empty or nonfinite")
    result = math.sqrt(math.fsum(squared) / len(squared))
    if not math.isfinite(result) or result <= 0.0:
        raise FreezeBPreparationError("formal baseline RMSE must be positive")
    return result


def _formal_baselines(formal_root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for side in ("left", "right"):
        side_result: dict[str, object] = {}
        for timestep in ("base", "halved"):
            mujoco_id = f"mujoco.{side}.small_step.{timestep}.r01"
            ovphysx_id = f"ovphysx.{side}.small_step.{timestep}.r01"
            mujoco = _formal_run(formal_root, "mujoco", mujoco_id)
            ovphysx = _formal_run(formal_root, "ovphysx", ovphysx_id)
            side_result[timestep] = {
                "joint_rmse_rad": _window_rmse(mujoco, ovphysx, frame_position=False),
                "frame_position_rmse_m": _window_rmse(
                    mujoco, ovphysx, frame_position=True
                ),
            }
        result[side] = side_result
    return result


def _cases() -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    ov_cases: list[dict[str, object]] = []
    mj_cases: list[dict[str, object]] = []
    run_order: list[str] = []
    for side in ("left", "right"):
        for timestep, dt_s in (("base", 0.002), ("halved", 0.001)):
            for repeat in (1, 2):
                roles = ("sham", "zero") if repeat == 1 else ("zero", "sham")
                for role in roles:
                    experiment_id = (
                        f"freeze_b.ovphysx.{side}.small_step.{timestep}.{role}.r{repeat:02d}"
                    )
                    ov_cases.append(
                        {
                            "experiment_case_id": experiment_id,
                            "canonical_case_id": (
                                f"ovphysx.{side}.small_step.{timestep}.r{repeat:02d}"
                            ),
                            "backend": "ovphysx",
                            "role": role,
                            "hand": side,
                            "timestep_variant": timestep,
                            "dt_s": dt_s,
                            "repeat_index": repeat,
                        }
                    )
                    run_order.append(experiment_id)
                mj_cases.append(
                    {
                        "experiment_case_id": (
                            f"freeze_b.mujoco.{side}.small_step.{timestep}.control.r{repeat:02d}"
                        ),
                        "canonical_case_id": (
                            f"mujoco.{side}.small_step.{timestep}.r{repeat:02d}"
                        ),
                        "backend": "mujoco",
                        "role": "control",
                        "hand": side,
                        "timestep_variant": timestep,
                        "dt_s": dt_s,
                        "repeat_index": repeat,
                    }
                )
    return ov_cases, mj_cases, run_order


def build_private_plan(
    *,
    project_root: Path,
    source_revision: str,
    manifest_path: Path,
    readback_root: Path,
    formal_root: Path,
) -> Mapping[str, object]:
    source_tree = _verify_git_source(project_root, source_revision)
    readback_verification = verify_exact_bundle(readback_root)
    formal_verification = verify_exact_bundle(formal_root)
    if readback_verification["root_sha256"] != READBACK_ROOT_SHA256:
        raise FreezeBPreparationError("R1 readback bundle root differs from Freeze A")
    if formal_verification["root_sha256"] != FORMAL_ROOT_SHA256:
        raise FreezeBPreparationError("formal Gate 0 bundle root changed")
    manifest = load_manifest(manifest_path)
    if sha256_file(manifest_path) != GATE0_MANIFEST_FILE_SHA256:
        raise FreezeBPreparationError("Gate 0 manifest file hash changed")
    if manifest_sha256(manifest) != GATE0_MANIFEST_SEMANTIC_SHA256:
        raise FreezeBPreparationError("Gate 0 manifest semantic hash changed")
    if (
        manifest.provenance.commit != ASSET_COMMIT
        or manifest.provenance.asset_git_tree != ASSET_GIT_TREE
        or manifest.provenance.canonical_lf_asset_tree_sha256 != ASSET_TREE_SHA256
    ):
        raise FreezeBPreparationError("Gate 0 asset identity changed")

    for case_id, expected in READBACK_CASE_SHA256.items():
        path = (
            readback_root
            / "private"
            / "remote-evidence"
            / "cases"
            / case_id
            / f"{case_id}.json"
        )
        if sha256_file(path) != expected:
            raise FreezeBPreparationError(f"R1 case hash changed: {case_id}")
    if sha256_file(readback_root / "results" / "summary.json") != READBACK_SUMMARY_SHA256:
        raise FreezeBPreparationError("R1 summary hash changed")

    readback = _strict_json(readback_root / "private" / "readback.json")
    hand_plans = _instance_maps(readback, manifest)
    baselines = _formal_baselines(formal_root)
    ov_cases, mj_cases, run_order = _cases()
    inputs = {
        "freeze_a_config_sha256": FREEZE_A_CONFIG_SHA256,
        "readback_bundle_root_sha256": READBACK_ROOT_SHA256,
        "readback_summary_sha256": READBACK_SUMMARY_SHA256,
        "readback_source_revision": READBACK_SOURCE_REVISION,
        "readback_source_tree": READBACK_SOURCE_TREE,
        "readback_case_sha256": dict(READBACK_CASE_SHA256),
        "formal_gate0_bundle_root_sha256": FORMAL_ROOT_SHA256,
        "formal_gate0_source_revision": FORMAL_SOURCE_REVISION,
        "gate0_manifest_file_sha256": GATE0_MANIFEST_FILE_SHA256,
        "gate0_manifest_semantic_sha256": GATE0_MANIFEST_SEMANTIC_SHA256,
        "asset_commit": ASSET_COMMIT,
        "asset_git_tree": ASSET_GIT_TREE,
        "canonical_lf_asset_tree_sha256": ASSET_TREE_SHA256,
        "freeze_b_source_revision": source_revision,
        "freeze_b_source_tree": source_tree,
        "formal_crosssim_window_rmse": baselines,
    }
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "inputs": inputs,
        "hand_plans": hand_plans,
        "ovphysx_cases": ov_cases,
        "mujoco_cases": mj_cases,
        "run_order": run_order,
    }
    return validate_private_plan(plan)


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FreezeBPreparationError(f"refusing to overwrite private plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readback-bundle", type=Path, required=True)
    parser.add_argument("--formal-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_root = args.project_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve()
        results_root = (project_root / "results").resolve(strict=True)
        if output.parent != results_root:
            raise FreezeBPreparationError("private plan must be a direct child of results/")
        plan = build_private_plan(
            project_root=project_root,
            source_revision=args.source_revision,
            manifest_path=args.manifest.expanduser().resolve(strict=True),
            readback_root=args.readback_bundle.expanduser().resolve(strict=True),
            formal_root=args.formal_bundle.expanduser().resolve(strict=True),
        )
        digest = private_plan_sha256(plan)
        _write_exclusive(output, plan)
        summary = {
            "status": "frozen",
            "protocol_id": PROTOCOL_ID,
            "private_plan_sha256": digest,
            "hand_count": len(plan["hand_plans"]),
            "ovphysx_case_count": len(plan["ovphysx_cases"]),
            "mujoco_case_count": len(plan["mujoco_cases"]),
        }
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(f"Freeze B preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
