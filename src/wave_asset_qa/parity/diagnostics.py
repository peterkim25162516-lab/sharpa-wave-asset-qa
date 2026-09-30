"""Fail-closed source, output, and bundle helpers for bounded diagnostics."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping, Sequence

import wave_asset_qa
from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample

from .bundle import BUNDLE_SCHEMA_VERSION, bundle_root_sha256, sha256_file, verify_bundle
from .contracts import ParityManifest, Simulator
from .scenarios import (
    ScenarioCase,
    canonical_initial_positions,
    canonical_position_targets,
)


_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class DiagnosticEvidenceError(RuntimeError):
    """Raised when a diagnostic cannot preserve its evidence contract."""


def asset_tree_sha256(root: str | Path) -> str:
    """Hash a materialized asset tree with the pinned Gate 0 algorithm."""

    raw_root = Path(os.path.abspath(Path(root).expanduser()))
    if is_link_like(raw_root):
        raise DiagnosticEvidenceError(
            f"asset root must not be a symbolic link or junction: {raw_root}"
        )
    asset_root = raw_root.resolve(strict=True)
    if not asset_root.is_dir():
        raise DiagnosticEvidenceError(
            f"asset root must be a real directory: {asset_root}"
        )
    digest = sha256()
    entries = tuple(asset_root.rglob("*"))
    linked = [
        path.relative_to(asset_root).as_posix()
        for path in entries
        if ".git" not in path.relative_to(asset_root).parts and is_link_like(path)
    ]
    if linked:
        raise DiagnosticEvidenceError(
            "asset tree contains symbolic-link or junction entries: "
            + ", ".join(sorted(linked)[:5])
        )
    files = sorted(
        (
            path
            for path in entries
            if path.is_file()
            and ".git" not in path.relative_to(asset_root).parts
        ),
        key=lambda path: path.relative_to(asset_root).as_posix(),
    )
    if not files:
        raise DiagnosticEvidenceError("asset root contains no regular files")
    for path in files:
        relative = path.relative_to(asset_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_diagnostic_run(
    result: AdapterRunResult,
    case: ScenarioCase,
    manifest: ParityManifest,
) -> None:
    """Require a complete canonical trace with auditable backend mappings.

    Backend-native ``qpos`` arrays are checked through each run's explicit
    mapping.  Downstream diagnostics should consume ``joint_positions`` only.
    """

    if not isinstance(result, AdapterRunResult):
        raise DiagnosticEvidenceError("diagnostic result has the wrong type")
    if not isinstance(case, ScenarioCase) or not isinstance(manifest, ParityManifest):
        raise DiagnosticEvidenceError("diagnostic case or manifest has the wrong type")
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    expected_steps = round(scenario.duration_s / case.dt_s)
    if (
        not result.completed
        or result.error is not None
        or result.backend != case.simulator.value
        or result.scenario_id != case.scenario_id
        or not math.isclose(result.dt, case.dt_s, rel_tol=0.0, abs_tol=1e-12)
        or result.requested_steps != expected_steps
        or result.completed_steps != expected_steps
        or len(result.samples) != expected_steps + 1
        or result.joint_names != hand.joint_names
        or result.frame_names != hand.distal_frame_names
    ):
        raise DiagnosticEvidenceError(
            f"run is not a complete canonical trace: {case.case_id}"
        )

    backend_joint_names = result.provenance.get("backend_joint_names")
    raw_joint_mapping = result.provenance.get("joint_mapping")
    if (
        not isinstance(backend_joint_names, list)
        or len(backend_joint_names) != len(hand.joint_names)
        or len(set(backend_joint_names)) != len(backend_joint_names)
        or any(not isinstance(name, str) or not name for name in backend_joint_names)
        or not isinstance(raw_joint_mapping, list)
        or len(raw_joint_mapping) != len(hand.joint_names)
    ):
        raise DiagnosticEvidenceError("run has invalid joint mapping provenance")
    joint_mapping: dict[str, tuple[int, float, float]] = {}
    mapped_joint_indices: set[int] = set()
    canonical_joint_order: list[str] = []
    for position, raw_record in enumerate(raw_joint_mapping):
        if not isinstance(raw_record, Mapping):
            raise DiagnosticEvidenceError(
                f"run has invalid joint mapping record {position}"
            )
        canonical_id = raw_record.get("canonical_id")
        backend_name = raw_record.get("backend_name")
        index = raw_record.get("index")
        sign = raw_record.get("sign")
        offset = raw_record.get("offset")
        if (
            not isinstance(canonical_id, str)
            or canonical_id not in hand.joint_names
            or not isinstance(backend_name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(backend_joint_names)
            or backend_joint_names[index] != backend_name
            or raw_record.get("backend") != result.backend
            or raw_record.get("scope") != case.hand.value
            or raw_record.get("unit") != "rad"
            or isinstance(sign, bool)
            or not isinstance(sign, (int, float))
            or float(sign) not in (-1.0, 1.0)
            or isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or canonical_id in joint_mapping
            or index in mapped_joint_indices
        ):
            raise DiagnosticEvidenceError(
                f"run has invalid joint mapping record {position}"
            )
        canonical_joint_order.append(canonical_id)
        mapped_joint_indices.add(index)
        joint_mapping[canonical_id] = (index, float(sign), float(offset))
    if (
        tuple(canonical_joint_order) != hand.joint_names
        or mapped_joint_indices != set(range(len(hand.joint_names)))
    ):
        raise DiagnosticEvidenceError("run joint mapping coverage or order differs")

    backend_frame_names = result.provenance.get("backend_frame_names")
    raw_frame_mapping = result.provenance.get("frame_mapping")
    if (
        not isinstance(backend_frame_names, list)
        or len(backend_frame_names) != len(set(backend_frame_names))
        or any(not isinstance(name, str) or not name for name in backend_frame_names)
        or not isinstance(raw_frame_mapping, list)
        or len(raw_frame_mapping) != len(hand.distal_frame_names)
    ):
        raise DiagnosticEvidenceError("run has invalid frame mapping provenance")
    mapped_frame_indices: set[int] = set()
    canonical_frame_order: list[str] = []
    for position, raw_record in enumerate(raw_frame_mapping):
        if not isinstance(raw_record, Mapping):
            raise DiagnosticEvidenceError(
                f"run has invalid frame mapping record {position}"
            )
        canonical_id = raw_record.get("canonical_id")
        backend_name = raw_record.get("backend_name")
        index = raw_record.get("index")
        sign = raw_record.get("sign")
        offset = raw_record.get("offset")
        if (
            not isinstance(canonical_id, str)
            or canonical_id not in hand.distal_frame_names
            or not isinstance(backend_name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(backend_frame_names)
            or backend_frame_names[index] != backend_name
            or raw_record.get("backend") != result.backend
            or raw_record.get("scope") != case.hand.value
            or raw_record.get("unit") != "xyz_m_qwxyz"
            or isinstance(sign, bool)
            or not isinstance(sign, (int, float))
            or not math.isfinite(float(sign))
            or float(sign) != 1.0
            or isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or float(offset) != 0.0
            or index in mapped_frame_indices
        ):
            raise DiagnosticEvidenceError(
                f"run has invalid frame mapping record {position}"
            )
        canonical_frame_order.append(canonical_id)
        mapped_frame_indices.add(index)
    if tuple(canonical_frame_order) != hand.distal_frame_names:
        raise DiagnosticEvidenceError("run frame mapping coverage or order differs")

    if case.simulator is Simulator.OVPHYSX:
        if (
            result.provenance.get("contact_check_performed") is not False
            or result.provenance.get("contact_observation_capability")
            != "not_evaluated"
        ):
            raise DiagnosticEvidenceError(
                "OVPhysX run has unexpected contact-observation provenance"
            )
        expected_contact_count: int | None = None
    else:
        expected_contact_count = 0

    initial = canonical_initial_positions(scenario, hand.joint_names)
    for step, sample in enumerate(result.samples):
        if sample.step != step or not math.isclose(
            sample.time_s,
            step * case.dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DiagnosticEvidenceError("run has a non-canonical time axis")
        targets = canonical_position_targets(
            scenario,
            hand.joint_names,
            step_index=step,
            dt_s=case.dt_s,
        )
        if dict(sample.position_targets) != targets:
            raise DiagnosticEvidenceError("run has non-canonical position targets")
        if (
            set(sample.joint_positions) != set(hand.joint_names)
            or set(sample.frame_poses) != set(hand.distal_frame_names)
            or len(sample.qpos) != len(hand.joint_names)
            or len(sample.qvel) != len(hand.joint_names)
            or any(len(pose) != 7 for pose in sample.frame_poses.values())
            or sample.contact_count != expected_contact_count
        ):
            raise DiagnosticEvidenceError("run has invalid sampled state structure")
        for canonical_id, (index, sign, offset) in joint_mapping.items():
            mapped = sign * float(sample.qpos[index]) + offset
            if mapped != float(sample.joint_positions[canonical_id]):
                raise DiagnosticEvidenceError(
                    "run qpos does not reproduce canonical joint_positions through mapping"
                )
        values = (
            sample.time_s,
            *sample.qpos,
            *sample.qvel,
            *sample.joint_positions.values(),
            *(value for pose in sample.frame_poses.values() for value in pose),
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise DiagnosticEvidenceError("run contains a non-finite sample")
    first = result.samples[0]
    if (
        dict(first.joint_positions) != initial
        or any(value != 0.0 for value in first.qvel)
    ):
        raise DiagnosticEvidenceError("run does not start from canonical state")


def replay_mujoco_fk_named(
    adapter: object,
    joint_positions: Mapping[str, float],
    *,
    expected_joint_names: Sequence[str],
    frame_names: Sequence[str],
    step: int = 0,
) -> TraceSample:
    """Replay one complete named canonical state without advancing dynamics."""

    from wave_asset_qa.adapters.mujoco import MuJoCoAdapter

    if not isinstance(adapter, MuJoCoAdapter):
        raise DiagnosticEvidenceError("common-FK replay requires MuJoCoAdapter")
    names = tuple(expected_joint_names)
    frames = tuple(frame_names)
    if (
        not names
        or len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
        or set(joint_positions) != set(names)
        or tuple(adapter.joint_names) != names
    ):
        raise DiagnosticEvidenceError(
            "common-FK replay must cover exactly the compiled canonical joint set"
        )
    if (
        not frames
        or len(frames) != len(set(frames))
        or any(not isinstance(name, str) or not name for name in frames)
    ):
        raise DiagnosticEvidenceError("common-FK frame selection is invalid")
    ordered: dict[str, float] = {}
    for name in names:
        raw = joint_positions[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise DiagnosticEvidenceError(
                f"common-FK joint position is not numeric: {name}"
            )
        try:
            value = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise DiagnosticEvidenceError(
                f"common-FK joint position is not finite: {name}"
            ) from exc
        if not math.isfinite(value):
            raise DiagnosticEvidenceError(
                f"common-FK joint position is not finite: {name}"
            )
        ordered[name] = value
    adapter.reset()
    adapter.set_joint_positions(ordered)
    sample = adapter.sample(
        step=step,
        joint_names=names,
        frame_names=frames,
    )
    if tuple(sample.joint_positions) != names or any(
        float(sample.joint_positions[name]) != ordered[name] for name in names
    ):
        raise DiagnosticEvidenceError("common-FK replay changed the requested joint state")
    if set(sample.frame_poses) != set(frames):
        raise DiagnosticEvidenceError("common-FK replay returned incomplete frame poses")
    if sample.time_s != 0.0:
        raise DiagnosticEvidenceError("common-FK replay unexpectedly advanced simulation time")
    return sample


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
        raise DiagnosticEvidenceError(f"cannot verify diagnostic Git source: {detail}")
    return completed.stdout.strip()


def validate_git_source(
    project_root: str | Path,
    source_revision: str,
    *,
    required_tracked_paths: Sequence[str],
    sensitive_untracked_paths: Sequence[str],
) -> str:
    """Require one clean, tracked checkout and return the committed tree hash."""

    root = Path(project_root).resolve(strict=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != root:
        raise DiagnosticEvidenceError(
            f"diagnostic project root is not the Git top level: {root} != {top_level}"
        )
    if _HEX40.fullmatch(source_revision) is None:
        raise DiagnosticEvidenceError(
            "source_revision must be a lowercase 40-character commit"
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise DiagnosticEvidenceError(
            f"source_revision does not match local HEAD: {source_revision} != {head}"
        )
    tree = _git_text(root, "rev-parse", "--verify", f"{source_revision}^{{tree}}")
    if _HEX40.fullmatch(tree) is None:
        raise DiagnosticEvidenceError(
            "diagnostic source commit did not resolve to a Git tree"
        )
    tracked_changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_changes:
        raise DiagnosticEvidenceError(
            "tracked worktree/index must be clean for the diagnostic"
        )
    untracked = _git_text(
        root,
        "ls-files",
        "--others",
        "--",
        *sensitive_untracked_paths,
    ).splitlines()
    material_untracked = [
        item
        for item in untracked
        if "__pycache__" not in Path(item).parts and Path(item).suffix != ".pyc"
    ]
    if material_untracked:
        raise DiagnosticEvidenceError(
            "untracked files exist in diagnostic source/input paths: "
            + ", ".join(material_untracked)
        )
    for relative in required_tracked_paths:
        _git_text(root, "ls-files", "--error-unmatch", "--", relative)
    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise DiagnosticEvidenceError(
            "wave_asset_qa was imported from a source tree other than this checkout"
        )
    return tree


def is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


def validated_results_output_path(
    project_root: str | Path,
    path: str | Path,
) -> Path:
    """Confine a final output to one direct, non-linked child of results/."""

    root = Path(project_root).resolve(strict=True)
    results_root = (root / "results").resolve(strict=True)
    raw = Path(os.path.abspath(Path(path).expanduser()))
    for ancestor in (raw.parent, *raw.parent.parents):
        if ancestor.exists() and is_link_like(ancestor):
            raise DiagnosticEvidenceError(
                f"diagnostic output has a symbolic-link or junction ancestor: {ancestor}"
            )
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise DiagnosticEvidenceError(
            "diagnostic output parent must be the existing project results directory"
        ) from exc
    if parent != results_root:
        raise DiagnosticEvidenceError(
            "diagnostic output must be a direct child of the project results directory"
        )
    if raw.exists() or is_link_like(raw):
        raise DiagnosticEvidenceError(
            f"output path already exists; refusing overwrite: {raw}"
        )
    return raw


def create_staging_root(final_path: str | Path) -> Path:
    """Create a visibly incomplete sibling for fail-closed finalization."""

    final = Path(final_path)
    if final.exists() or is_link_like(final):
        raise DiagnosticEvidenceError(
            f"output path already exists; refusing overwrite: {final}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final.name}.incomplete-",
            dir=final.parent,
        )
    ).resolve(strict=True)
    if staging.parent != final.parent.resolve(strict=True) or is_link_like(staging):
        raise DiagnosticEvidenceError(
            "staging directory escaped the approved results root"
        )
    return staging


def promote_staging_root(staging: str | Path, final_path: str | Path) -> Path:
    source = Path(staging).resolve(strict=True)
    final = Path(final_path)
    if source.parent != final.parent.resolve(strict=True):
        raise DiagnosticEvidenceError(
            "staging and final diagnostic paths are not siblings"
        )
    if final.exists() or is_link_like(final):
        raise DiagnosticEvidenceError(
            f"output path appeared during the run; refusing overwrite: {final}"
        )
    try:
        os.rename(source, final)
    except OSError as exc:
        raise DiagnosticEvidenceError(
            f"cannot promote verified diagnostic output: {exc}"
        ) from exc
    promoted = final.resolve(strict=True)
    if promoted.parent != source.parent:
        raise DiagnosticEvidenceError(
            "promoted diagnostic output escaped the results root"
        )
    return promoted


def bundle_payload_paths(root: str | Path) -> list[str]:
    bundle_root = Path(root).resolve(strict=True)
    root_manifest = bundle_root / "bundle.json"
    return sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path != root_manifest
    )


def verify_exact_bundle(root: str | Path) -> dict[str, object]:
    bundle_root = Path(root).resolve(strict=True)
    verification = dict(verify_bundle(bundle_root))
    try:
        manifest = json.loads(
            (bundle_root / "bundle.json").read_text(encoding="utf-8")
        )
        records = manifest["files"]
        listed = {record["path"] for record in records}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DiagnosticEvidenceError(
            f"cannot audit diagnostic bundle inventory: {exc}"
        ) from exc
    actual = set(bundle_payload_paths(bundle_root))
    if listed != actual:
        raise DiagnosticEvidenceError(
            "diagnostic bundle inventory is not exact: "
            f"missing={sorted(actual - listed)}, extra={sorted(listed - actual)}"
        )
    verification["exact_inventory"] = True
    return verification


def verify_copied_bundle_subset(
    copied_manifest: str | Path,
    *,
    expected_root_sha256: str,
    copied_payloads: Mapping[str, Path],
) -> None:
    """Match copied payload bytes to records in a frozen parent bundle."""

    manifest_path = Path(copied_manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "created_at_utc",
            "files",
            "root_sha256",
        }:
            raise DiagnosticEvidenceError(
                "copied parent manifest has an invalid top-level schema"
            )
        if (
            isinstance(manifest["schema_version"], bool)
            or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION
        ):
            raise DiagnosticEvidenceError(
                "copied parent manifest has an unsupported schema version"
            )
        raw_records = manifest["files"]
        if not isinstance(raw_records, list):
            raise DiagnosticEvidenceError(
                "copied parent manifest files must be an array"
            )
        created_at = manifest["created_at_utc"]
        if not isinstance(created_at, str) or not created_at:
            raise DiagnosticEvidenceError(
                "copied parent manifest has an invalid creation time"
            )
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if (
            parsed_created_at.tzinfo is None
            or parsed_created_at.utcoffset() != timezone.utc.utcoffset(parsed_created_at)
        ):
            raise DiagnosticEvidenceError(
                "copied parent manifest creation time is not UTC"
            )
        computed_root = bundle_root_sha256(raw_records)
        if (
            manifest["root_sha256"] != expected_root_sha256
            or computed_root != expected_root_sha256
        ):
            raise DiagnosticEvidenceError(
                "copied parent manifest has the wrong root hash"
            )
        paths = [record["path"] for record in raw_records]
        if paths != sorted(paths):
            raise DiagnosticEvidenceError(
                "copied parent manifest file records are not sorted"
            )
        records = {record["path"]: record for record in raw_records}
    except DiagnosticEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DiagnosticEvidenceError(
            f"cannot audit copied parent manifest: {exc}"
        ) from exc
    for parent_path, copied_path in copied_payloads.items():
        record = records.get(parent_path)
        if not isinstance(record, Mapping):
            raise DiagnosticEvidenceError(
                f"parent manifest does not list copied payload: {parent_path}"
            )
        if (
            record.get("size") != copied_path.stat().st_size
            or record.get("sha256") != sha256_file(copied_path)
        ):
            raise DiagnosticEvidenceError(
                f"copied payload differs from its parent manifest: {parent_path}"
            )


def write_text_exclusive(path: str | Path, content: str) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise DiagnosticEvidenceError(
            f"refusing to overwrite diagnostic artifact: {destination}"
        )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "DiagnosticEvidenceError",
    "asset_tree_sha256",
    "bundle_payload_paths",
    "create_staging_root",
    "is_link_like",
    "promote_staging_root",
    "replay_mujoco_fk_named",
    "validate_git_source",
    "validate_diagnostic_run",
    "validated_results_output_path",
    "verify_copied_bundle_subset",
    "verify_exact_bundle",
    "write_text_exclusive",
]
