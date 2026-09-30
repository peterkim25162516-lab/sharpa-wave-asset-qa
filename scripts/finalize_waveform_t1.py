#!/usr/bin/env python3
"""Finalize one exact 32-case WaveSimParity Waveform T1 evidence pair.

This program is local-only.  It copies, validates, compares, reports, and
content-addresses already-collected MuJoCo and kit-less OVPhysX evidence.  It
does not launch either simulator and does not access a remote host.

Raw launcher evidence remains inside the private bundle.  The only artifacts
eligible for publication are ``public/summary.json`` and ``public/report.md``;
their builder rejects private paths, process identifiers, GPU UUIDs, session
identifiers, non-finite values, and contradictory validity/science labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


class WaveformT1FinalizationError(RuntimeError):
    """Evidence, identity, publication, or bundle validation failed."""


def _source_bytecode_artifacts(project_root: Path) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    try:
        for source_root_name in ("src", "scripts"):
            source_root = project_root / source_root_name
            if not source_root.is_dir() or source_root.is_symlink():
                raise WaveformT1FinalizationError(
                    f"project {source_root_name}/ must be a regular directory"
                )
            for path in source_root.rglob("*"):
                if path.name == "__pycache__" or path.suffix.lower() in {
                    ".pyc",
                    ".pyo",
                }:
                    artifacts.append(path)
    except OSError as exc:
        raise WaveformT1FinalizationError(
            "cannot audit src/ and scripts/ for ignored Python bytecode"
        ) from exc
    return tuple(sorted(artifacts, key=lambda path: path.as_posix()))


def _reject_source_bytecode(project_root: Path) -> None:
    artifacts = _source_bytecode_artifacts(project_root)
    if artifacts:
        first = artifacts[0].relative_to(project_root).as_posix()
        raise WaveformT1FinalizationError(
            "src/ and scripts/ must contain no .pyc, .pyo, or __pycache__ "
            f"before formal finalization; first artifact: {first}"
        )


if __name__ == "__main__":
    # This guard is deliberately before any project or simulator package import.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    _reject_source_bytecode(Path(__file__).resolve(strict=True).parents[1])


import mujoco
import numpy
import wave_asset_qa

from wave_asset_qa.parity.bundle import (
    DEFAULT_MANIFEST_PATH,
    bundle_root_sha256,
    verify_bundle,
    write_bundle_manifest,
    write_json_atomic,
)
from wave_asset_qa.parity.compare import (
    CollectedRun,
    ComparisonThresholds,
    Gate0Comparison,
    compare_gate0_runs,
    trace_delta,
)
from wave_asset_qa.parity.contracts import (
    ComparisonStatus,
    HandSide,
    ParityManifest,
    Simulator,
)
from wave_asset_qa.parity.process_identity import (
    ProcessIdentityError,
    os_process_identity_sha256,
    validate_os_process_identity,
)
from wave_asset_qa.parity.runner import (
    RUN_FILE_SUFFIX,
    WORKER_EXITCODE_SUFFIX,
    WORKER_RESULT_SUFFIX,
    WORKER_STDERR_SUFFIX,
    WORKER_STDOUT_SUFFIX,
    load_collected_runs,
)
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    TimestepVariant,
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)
from wave_asset_qa.parity.waveform_metrics import (
    chirp_band_metrics,
    one_step_tracking_metrics,
    sine_response_metrics,
    state_state_metrics,
)
from wave_asset_qa.parity.waveform_report import (
    build_public_artifacts,
    canonical_summary_json,
    render_public_report,
)


EXPECTED_MANIFEST_ID = "wavesimparity-waveform-t1"
EXPECTED_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_BACKEND_CASE_COUNT = 16
EXPECTED_TOTAL_CASE_COUNT = 32
EXPECTED_JOINT_MAPPING_COUNT = 44
EXPECTED_DP_FRAME_MAPPING_COUNT = 10
EXPECTED_OVPHYSX_EFFORT_CASE_COUNT = 16
EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION = 22
OVPHYSX_EFFORT_CLIPPING_SCHEMA_VERSION = 1
MAX_PRIVATE_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
PRIVATE_BUNDLE_DERIVED_HEADROOM_BYTES = 64 * 1024 * 1024
EXPECTED_HANDS = (HandSide.LEFT, HandSide.RIGHT)
EXPECTED_SCENARIOS = ("offset_sine", "offset_linear_chirp")

# Keep the topology decisions in one place.  Formal T1 evidence must match the
# committed launcher exactly; accepting an unused alternate layout would make
# the supposedly exact inventory needlessly permissive.
LOCAL_LAUNCHER_FILES = frozenset(
    {
        "completed.json",
        "evidence.sha256",
        "matrix.json",
        "provenance.json",
        "waveform_t1.manifest.json",
    }
)
LOCAL_CASE_FILES = frozenset(
    {
        "evidence.sha256",
        "launcher.command.json",
        "launcher.exitcode.txt",
        "launcher.process.json",
        "launcher.stderr.log",
        "launcher.stdout.log",
        "worker-process.json",
    }
)
REMOTE_SCHEMA_RELATIVE_CANDIDATES = (
    PurePosixPath("launcher/schema"),
)
REMOTE_SCHEMA_FILES = frozenset(
    {
        "evidence.sha256",
        *(
            f"{hand}.{suffix}"
            for hand in ("left", "right")
            for suffix in ("json", "stdout", "stderr", "exitcode")
        ),
    }
)
REMOTE_LAUNCHER_REQUIRED_FILES = frozenset(
    {
        "cases.txt",
        "evidence.sha256",
        "gpu-selection.txt",
        "provenance.txt",
        "snapshot-verifications.txt",
        "status.json",
        "stderr.txt",
        "stdout.txt",
    }
)

TARGET_DIGEST_ENCODING = "utf8_json_lines_float_hex_v1"
TARGET_DIGEST_PROJECTION = "ieee754_binary32_roundtrip"
SCHEDULED_TARGET_SEMANTICS = (
    "q[0..N]; target q[k] is recorded at t_k and applies to "
    "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
)

# Frozen runtime-log gate.  These expressions deliberately target fatal
# execution diagnostics, not every line containing the word "error": the
# pinned kit-less stack emits a retained cooking-registry diagnostic even on
# successful runs.  Source identity and the private finalization payload bind
# this exact pattern set to the execution commit.
RUNTIME_LOG_GATE_SCHEMA_VERSION = 1
RUNTIME_FATAL_PATTERN_SET_VERSION = 1
RUNTIME_KNOWN_KITLESS_PATTERN_SET_VERSION = 1
RUNTIME_FATAL_LOG_PATTERNS = (
    (
        "python_traceback",
        re.compile(r"(?i)(?<![a-z0-9])trace[\s_-]*back(?![a-z0-9])"),
    ),
    (
        "cuda_error",
        re.compile(
            r"(?i)(?<![a-z0-9])cuda(?:[\s_:-]*runtime)?[\s_:-]*"
            r"(?:error|failure|failed)(?![a-z0-9])|"
            r"(?<![a-z0-9])cudaerror[a-z0-9_]*"
        ),
    ),
    (
        "out_of_memory",
        re.compile(
            r"(?i)(?<![a-z0-9])out[\s_-]*of[\s_-]*memory(?![a-z0-9])|"
            r"(?<![a-z0-9])oom(?![a-z0-9])|"
            r"(?<![a-z0-9])memoryerror(?![a-z0-9])|"
            r"std::bad_alloc|cannot[\s_-]+allocate[\s_-]+memory"
        ),
    ),
)
RUNTIME_KNOWN_KITLESS_LOG_PATTERNS = (
    (
        "missing_omni_physx_plugin",
        re.compile(
            r"(?i)getPluginDesc:\s*Failed to find a plugin with a name:\s*"
            r"omni\.physx\.plugin\."
        ),
    ),
    (
        "ujitso_cooking_service",
        re.compile(
            r"(?i)Failed to create a valid UJITSO Cooking Compute Service"
        ),
    ),
    (
        "ovphysx_sdk_unload_warning",
        re.compile(r"(?i)^\s*\[Warning\]\s*\[omni_physx_sdk\]"),
    ),
    (
        "material_binding_api_warning",
        re.compile(r"(?i)\bWarning\b.*\bMaterialBindingAPI\b"),
    ),
    (
        "usd_resolver_status",
        re.compile(r"(?i)^\s*Status(?:\s*\(secondary thread\))?:.*OmniUsdResolver"),
    ),
    (
        "cooking_registry_diagnostic",
        re.compile(
            r"(?i)^\s*\[Error\]\s*\[omni\.physx\.cooking\.plugin\]\s*"
            r"registry is false\.\s*$"
        ),
    ),
)
_RUNTIME_WARNING_LINE_RE = re.compile(r"(?i)(?<![a-z0-9])warn(?:ing)?(?![a-z0-9])")

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_HASH_LINE_RE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)")
_EXPECTED_DRIVER_VERSION = "570.158.01"
_EXPECTED_GPU_NAME_FRAGMENT = "A800-SXM4-40GB"
_MAX_REMOTE_GPU_ELAPSED_S = 2 * 60 * 60
_EXPECTED_ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
_EXPECTED_REMOTE_ENVIRONMENT = {
    "python": "3.12.14",
    "isaaclab": "6.1.14",
    "isaaclab-ovphysx": "3.0.2",
    "ovphysx": "0.4.13",
    "torch": "2.10.0+cu128",
    "usd-core": "25.11",
    "kitless": "true",
}
_REMOTE_APPROVED_ROOT = PurePosixPath(
    "/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
)
EXPECTED_LOCAL_PYTHON_VERSION = "3.12.6"
EXPECTED_LOCAL_MUJOCO_VERSION = "3.12.0"
EXPECTED_LOCAL_NUMPY_VERSION = "2.5.2"
EXPECTED_LOCAL_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_LOCAL_PLATFORM = "Windows-11-10.0.26200-SP0"
EXPECTED_LOCAL_MACHINE = "AMD64"
EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256 = (
    "3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238"
)


@dataclass(frozen=True, slots=True)
class _EvidenceSet:
    runs: tuple[CollectedRun, ...]
    local_identity: Mapping[str, object]
    remote_identity: Mapping[str, object]
    schema_relative_path: str


def _resolved_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise WaveformT1FinalizationError(f"{label} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise WaveformT1FinalizationError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise WaveformT1FinalizationError(f"{label} must be a directory")
    return resolved


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise WaveformT1FinalizationError(f"{label} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise WaveformT1FinalizationError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise WaveformT1FinalizationError(f"{label} must be a regular file")
    return resolved


def _new_destination(path: str | Path, label: str) -> tuple[Path, Path]:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"{label} already exists; refusing overwrite: {requested}")
    if requested.name in {"", ".", ".."}:
        raise WaveformT1FinalizationError(f"{label} must name a new directory")
    parent = _resolved_directory(requested.parent, f"{label} parent")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} already exists; refusing overwrite: {destination}")
    return parent, destination


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_separate_trees(paths: Sequence[tuple[Path, str]]) -> None:
    for index, (first, first_label) in enumerate(paths):
        for second, second_label in paths[index + 1 :]:
            if first == second or _is_within(first, second) or _is_within(second, first):
                raise WaveformT1FinalizationError(
                    f"{first_label} and {second_label} must be separate, non-nested trees"
                )


def _validate_regular_tree(root: Path, label: str) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WaveformT1FinalizationError(
                f"{label} contains a symbolic link: {path.relative_to(root).as_posix()}"
            )
        mode = path.stat(follow_symlinks=False).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise WaveformT1FinalizationError(
                f"{label} contains a non-regular filesystem entry: "
                f"{path.relative_to(root).as_posix()}"
            )


def _copy_tree_strict(source: Path, destination: Path, label: str) -> None:
    _validate_regular_tree(source, label)
    destination.mkdir()
    for source_path in sorted(
        source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
    ):
        relative = source_path.relative_to(source)
        target = destination / relative
        if source_path.is_dir():
            target.mkdir()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_runtime_identity() -> dict[str, object]:
    """Recompute the frozen finalizer interpreter and local package identity."""

    versions = {
        "python_version": platform.python_version(),
        "mujoco_version": str(getattr(mujoco, "__version__", "")),
        "numpy_version": str(getattr(numpy, "__version__", "")),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    expected_versions = {
        "python_version": EXPECTED_LOCAL_PYTHON_VERSION,
        "mujoco_version": EXPECTED_LOCAL_MUJOCO_VERSION,
        "numpy_version": EXPECTED_LOCAL_NUMPY_VERSION,
        "python_implementation": EXPECTED_LOCAL_PYTHON_IMPLEMENTATION,
        "platform": EXPECTED_LOCAL_PLATFORM,
        "machine": EXPECTED_LOCAL_MACHINE,
    }
    drift = [
        name for name, expected in expected_versions.items() if versions[name] != expected
    ]
    if drift:
        raise WaveformT1FinalizationError(
            "local Waveform T1 dependency version drift: "
            + ", ".join(sorted(drift))
        )
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise WaveformT1FinalizationError(
            "finalizer sys.executable is not a regular file"
        )
    executable_sha256 = _sha256_file(executable)
    if executable_sha256 != EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256:
        raise WaveformT1FinalizationError(
            "local Waveform T1 Python executable SHA-256 drifted from the freeze"
        )
    return {
        "schema_version": 1,
        **versions,
        "python_executable_sha256": executable_sha256,
    }


def _tree_records(root: Path) -> tuple[dict[str, object], ...]:
    _validate_regular_tree(root, "tree fingerprint input")
    return tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat(follow_symlinks=False).st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    )


def _tree_fingerprint(root: Path) -> dict[str, object]:
    records = _tree_records(root)
    return {
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size"]) for record in records),
        "root_sha256": bundle_root_sha256(records),
    }


def _validate_source_copy_budget(
    local_records: Sequence[Mapping[str, object]],
    remote_records: Sequence[Mapping[str, object]],
    *,
    manifest_size_bytes: int,
) -> dict[str, int]:
    """Reject oversized inputs before a private staging tree is created."""

    if (
        isinstance(manifest_size_bytes, bool)
        or not isinstance(manifest_size_bytes, int)
        or manifest_size_bytes < 0
    ):
        raise WaveformT1FinalizationError("manifest source size is invalid")
    local_bytes = sum(
        _nonnegative_int(record.get("size"), "local record size")
        for record in local_records
    )
    remote_bytes = sum(
        _nonnegative_int(record.get("size"), "remote record size")
        for record in remote_records
    )
    source_bytes = local_bytes + remote_bytes + manifest_size_bytes
    maximum_source_bytes = (
        MAX_PRIVATE_BUNDLE_BYTES - PRIVATE_BUNDLE_DERIVED_HEADROOM_BYTES
    )
    if maximum_source_bytes < 0 or source_bytes > maximum_source_bytes:
        raise WaveformT1FinalizationError(
            "Waveform T1 source evidence cannot fit the 2 GiB private-bundle "
            "hard limit with reserved derived-artifact headroom: "
            f"{source_bytes} > {maximum_source_bytes}"
        )
    return {
        "local_evidence_bytes": local_bytes,
        "remote_evidence_bytes": remote_bytes,
        "manifest_bytes": manifest_size_bytes,
        "source_payload_bytes": source_bytes,
        "derived_headroom_bytes": PRIVATE_BUNDLE_DERIVED_HEADROOM_BYTES,
        "hard_limit_bytes": MAX_PRIVATE_BUNDLE_BYTES,
    }


def _object_without_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON number: {token}")


def _strict_json_file(path: Path, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise WaveformT1FinalizationError(f"{label} must be a regular file")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WaveformT1FinalizationError(f"cannot read strict JSON from {label}") from exc


def _audit_runtime_log_file(
    path: Path,
    *,
    label: str,
    scope: str,
    stream: str,
) -> dict[str, object]:
    """Fail on frozen fatal diagnostics and count retained kit-less messages."""

    if path.is_symlink() or not path.is_file():
        raise WaveformT1FinalizationError(f"{label} must be a regular log file")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise WaveformT1FinalizationError(f"cannot read UTF-8 runtime log {label}") from exc
    if "\x00" in text:
        raise WaveformT1FinalizationError(f"runtime log {label} contains a NUL byte")

    fatal_match_counts = {
        pattern_id: len(tuple(pattern.finditer(text)))
        for pattern_id, pattern in RUNTIME_FATAL_LOG_PATTERNS
    }
    fatal_ids = [
        pattern_id for pattern_id, count in fatal_match_counts.items() if count > 0
    ]
    if fatal_ids:
        raise WaveformT1FinalizationError(
            f"fatal runtime log diagnostic in {label}: {', '.join(fatal_ids)}"
        )

    lines = text.splitlines()
    known_counts = {
        pattern_id: sum(1 for line in lines if pattern.search(line) is not None)
        for pattern_id, pattern in RUNTIME_KNOWN_KITLESS_LOG_PATTERNS
    }
    known_line_indexes = {
        index
        for index, line in enumerate(lines)
        if any(
            pattern.search(line) is not None
            for _pattern_id, pattern in RUNTIME_KNOWN_KITLESS_LOG_PATTERNS
        )
    }
    warning_line_indexes = {
        index
        for index, line in enumerate(lines)
        if _RUNTIME_WARNING_LINE_RE.search(line) is not None
    }
    return {
        "label": label,
        "scope": scope,
        "stream": stream,
        "byte_count": len(raw),
        "line_count": len(lines),
        "warning_line_count": len(warning_line_indexes),
        "known_kitless_diagnostic_line_count": len(known_line_indexes),
        "unclassified_warning_line_count": len(
            warning_line_indexes - known_line_indexes
        ),
        "fatal_match_counts": fatal_match_counts,
        "known_kitless_counts": known_counts,
    }


def _summarize_runtime_log_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fatal_ids = tuple(pattern_id for pattern_id, _pattern in RUNTIME_FATAL_LOG_PATTERNS)
    known_ids = tuple(
        pattern_id for pattern_id, _pattern in RUNTIME_KNOWN_KITLESS_LOG_PATTERNS
    )
    normalized: list[dict[str, object]] = []
    fatal_totals = {pattern_id: 0 for pattern_id in fatal_ids}
    known_totals = {pattern_id: 0 for pattern_id in known_ids}
    scalar_totals = {
        "byte_count": 0,
        "line_count": 0,
        "warning_line_count": 0,
        "known_kitless_diagnostic_line_count": 0,
        "unclassified_warning_line_count": 0,
    }
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"runtime log record {index}")
        expected_keys = {
            "label",
            "scope",
            "stream",
            *scalar_totals,
            "fatal_match_counts",
            "known_kitless_counts",
        }
        if set(record) != expected_keys:
            raise WaveformT1FinalizationError(
                f"runtime log record {index} fields are not exact"
            )
        for key in ("label", "scope", "stream"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise WaveformT1FinalizationError(
                    f"runtime log record {index}.{key} is invalid"
                )
        for key in scalar_totals:
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WaveformT1FinalizationError(
                    f"runtime log record {index}.{key} is invalid"
                )
            scalar_totals[key] += value
        fatal = _mapping(
            record.get("fatal_match_counts"),
            f"runtime log record {index}.fatal_match_counts",
        )
        known = _mapping(
            record.get("known_kitless_counts"),
            f"runtime log record {index}.known_kitless_counts",
        )
        if set(fatal) != set(fatal_ids) or set(known) != set(known_ids):
            raise WaveformT1FinalizationError(
                f"runtime log record {index} pattern counts are not exact"
            )
        for pattern_id in fatal_ids:
            count = fatal.get(pattern_id)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise WaveformT1FinalizationError(
                    f"runtime log record {index} fatal count is invalid"
                )
            fatal_totals[pattern_id] += count
        for pattern_id in known_ids:
            count = known.get(pattern_id)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise WaveformT1FinalizationError(
                    f"runtime log record {index} known-warning count is invalid"
                )
            known_totals[pattern_id] += count
        normalized.append(dict(record))
    fatal_match_count = sum(fatal_totals.values())
    if fatal_match_count != 0:
        raise WaveformT1FinalizationError(
            "runtime log summary contains a fatal diagnostic"
        )
    return {
        "schema_version": RUNTIME_LOG_GATE_SCHEMA_VERSION,
        "fatal_pattern_set_version": RUNTIME_FATAL_PATTERN_SET_VERSION,
        "known_kitless_pattern_set_version": (
            RUNTIME_KNOWN_KITLESS_PATTERN_SET_VERSION
        ),
        "fatal_pattern_ids": list(fatal_ids),
        "known_kitless_pattern_ids": list(known_ids),
        "file_count": len(normalized),
        "total_bytes": scalar_totals["byte_count"],
        "total_lines": scalar_totals["line_count"],
        "warning_line_count": scalar_totals["warning_line_count"],
        "known_kitless_diagnostic_line_count": scalar_totals[
            "known_kitless_diagnostic_line_count"
        ],
        "unclassified_warning_line_count": scalar_totals[
            "unclassified_warning_line_count"
        ],
        "fatal_match_count": fatal_match_count,
        "fatal_match_counts": fatal_totals,
        "known_kitless_counts": known_totals,
        "files": normalized,
    }


def _combine_runtime_log_audits(
    audits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records: list[Mapping[str, object]] = []
    for index, raw_audit in enumerate(audits):
        audit = _mapping(raw_audit, f"runtime log audit {index}")
        files = audit.get("files")
        if not isinstance(files, list):
            raise WaveformT1FinalizationError(
                f"runtime log audit {index}.files must be an array"
            )
        records.extend(
            _mapping(record, f"runtime log audit {index} file")
            for record in files
        )
    return _summarize_runtime_log_records(records)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise WaveformT1FinalizationError(f"{label} must be an object with string keys")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WaveformT1FinalizationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WaveformT1FinalizationError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative_vector(
    value: object,
    label: str,
    *,
    strictly_positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION:
        raise WaveformT1FinalizationError(
            f"{label} must contain exactly 22 numeric values"
        )
    result: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise WaveformT1FinalizationError(f"{label}[{index}] is not numeric")
        number = float(raw)
        if not math.isfinite(number) or (
            number <= 0.0 if strictly_positive else number < 0.0
        ):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise WaveformT1FinalizationError(
                f"{label}[{index}] must be finite and {qualifier}"
            )
        result.append(number)
    return tuple(result)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WaveformT1FinalizationError(f"{label} must be a lowercase SHA-256")
    return value


def _normalize_hash_member(raw: str, label: str) -> str:
    candidate = raw[2:] if raw.startswith("./") else raw
    if not candidate or "\\" in candidate:
        raise WaveformT1FinalizationError(f"{label} contains an unsafe hash path")
    pure = PurePosixPath(candidate)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise WaveformT1FinalizationError(f"{label} contains an unsafe hash path")
    return pure.as_posix()


def _verify_evidence_inventory(
    scope_root: Path,
    inventory_path: Path,
    label: str,
) -> dict[str, object]:
    """Verify an exact sha256sum inventory over *scope_root*.

    Local inventories live inside the directory they cover.  The remote
    launcher inventory lives at ``launcher/evidence.sha256`` but covers the
    complete remote run root; callers therefore pass the remote run root as
    ``scope_root``.
    """

    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise WaveformT1FinalizationError(f"{label} is missing")
    try:
        lines = inventory_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WaveformT1FinalizationError(f"cannot read {label}") from exc
    records: dict[str, str] = {}
    for line in lines:
        match = _HASH_LINE_RE.fullmatch(line)
        if match is None:
            raise WaveformT1FinalizationError(f"{label} contains a non-canonical record")
        digest, _mode, raw_path = match.groups()
        relative = _normalize_hash_member(raw_path, label)
        if relative in records:
            raise WaveformT1FinalizationError(f"{label} repeats {relative!r}")
        records[relative] = digest

    try:
        inventory_relative = inventory_path.resolve(strict=True).relative_to(scope_root)
    except ValueError as exc:
        raise WaveformT1FinalizationError(f"{label} is outside its inventory scope") from exc
    expected = {
        path.relative_to(scope_root).as_posix()
        for path in scope_root.rglob("*")
        if path.is_file() and path.relative_to(scope_root) != inventory_relative
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise WaveformT1FinalizationError(
            f"{label} is not exact: missing={missing}, extra={extra}"
        )
    total_size = 0
    manifest_records: list[dict[str, object]] = []
    for relative in sorted(records):
        member = scope_root.joinpath(*PurePosixPath(relative).parts)
        if member.is_symlink() or not member.is_file():
            raise WaveformT1FinalizationError(f"{label} references an unsafe member")
        digest = _sha256_file(member)
        if digest != records[relative]:
            raise WaveformT1FinalizationError(
                f"{label} digest mismatch for {relative!r}"
            )
        size = member.stat(follow_symlinks=False).st_size
        total_size += size
        manifest_records.append({"path": relative, "size": size, "sha256": digest})
    return {
        "file_count": len(records),
        "total_size_bytes": total_size,
        "root_sha256": bundle_root_sha256(manifest_records),
    }


def _parse_key_value_file(path: Path, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WaveformT1FinalizationError(f"cannot read {label}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or "=" not in line:
            raise WaveformT1FinalizationError(f"{label} contains a non-canonical line")
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None or not value or key in values:
            raise WaveformT1FinalizationError(f"{label} contains an invalid key/value")
        values[key] = value
    return values


def _git_text(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WaveformT1FinalizationError("cannot verify the finalizer Git source") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WaveformT1FinalizationError(
            "cannot verify the finalizer Git source" + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _validate_clean_committed_worktree(project_root: Path) -> None:
    """Reject every tracked or untracked non-ignored worktree/index change."""

    status = _git_text(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise WaveformT1FinalizationError(
            "worktree/index contains uncommitted or untracked non-ignored files"
        )


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    root = project_root.resolve(strict=True)
    _reject_source_bytecode(root)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise WaveformT1FinalizationError("finalizer project root is not the Git top level")
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise WaveformT1FinalizationError("source_revision does not match local HEAD")
    tree = _git_text(root, "rev-parse", "--verify", f"{source_revision}^{{tree}}")
    if _REVISION_RE.fullmatch(tree) is None:
        raise WaveformT1FinalizationError("source revision did not resolve to a Git tree")
    _validate_clean_committed_worktree(root)
    finalizer_relative = Path(__file__).resolve(strict=True).relative_to(root).as_posix()
    _git_text(root, "ls-files", "--error-unmatch", "--", finalizer_relative)
    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise WaveformT1FinalizationError(
            "wave_asset_qa was imported from a different source tree"
        )
    return tree


def _git_archive_bytes(project_root: Path, revision: str) -> bytes:
    """Return the exact binary ``git archive --format=tar`` deployment stream."""

    if _REVISION_RE.fullmatch(revision) is None:
        raise WaveformT1FinalizationError(
            "archive revision must be a 40-character Git OID"
        )
    try:
        completed = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=project_root,
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WaveformT1FinalizationError(
            "cannot build the immutable source archive"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise WaveformT1FinalizationError(
            "Git source archive failed" + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _virtual_snapshot_sha256(
    archive_bytes: bytes,
    *,
    source_revision: str,
    source_tree: str,
    archive_sha256: str,
) -> str:
    """Reproduce ``install_gate0_snapshot.py`` without materializing the tar."""

    if (
        _REVISION_RE.fullmatch(source_revision) is None
        or _REVISION_RE.fullmatch(source_tree) is None
        or _SHA256_RE.fullmatch(archive_sha256) is None
    ):
        raise WaveformT1FinalizationError("virtual snapshot identity is malformed")
    if sha256(archive_bytes).hexdigest() != archive_sha256:
        raise WaveformT1FinalizationError("virtual snapshot archive digest mismatch")
    entries: dict[str, tuple[bytes, int, bytes | None]] = {}

    def ensure_parents(path: PurePosixPath) -> None:
        for length in range(1, len(path.parts)):
            parent = PurePosixPath(*path.parts[:length]).as_posix()
            existing = entries.get(parent)
            if existing is not None and existing[0] != b"D":
                raise WaveformT1FinalizationError(
                    "source archive materializes a file as a parent directory"
                )
            entries.setdefault(parent, (b"D", 0, None))

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as bundle:
            for member in bundle.getmembers():
                raw_name = member.name.rstrip("/")
                if not raw_name or "\\" in raw_name:
                    raise WaveformT1FinalizationError(
                        "source archive member path is invalid"
                    )
                path = PurePosixPath(raw_name)
                if (
                    path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or ":" in path.parts[0]
                    or path.parts[0]
                    in {
                        ".source-revision",
                        ".source-tree",
                        ".archive-sha256",
                        ".snapshot-sha256",
                    }
                ):
                    raise WaveformT1FinalizationError(
                        "source archive member path is unsafe"
                    )
                name = path.as_posix()
                ensure_parents(path)
                if name in entries:
                    raise WaveformT1FinalizationError(
                        f"source archive member is duplicate: {name}"
                    )
                if member.isdir():
                    entries[name] = (b"D", 0, None)
                elif member.isfile():
                    source = bundle.extractfile(member)
                    if source is None:
                        raise WaveformT1FinalizationError(
                            f"source archive member cannot be read: {name}"
                        )
                    content = source.read()
                    if len(content) != member.size:
                        raise WaveformT1FinalizationError(
                            f"source archive member size drifted: {name}"
                        )
                    entries[name] = (
                        b"F",
                        int(bool(member.mode & 0o111)),
                        content,
                    )
                else:
                    raise WaveformT1FinalizationError(
                        f"source archive contains a non-regular member: {name}"
                    )
    except (tarfile.TarError, OSError) as exc:
        raise WaveformT1FinalizationError(
            "cannot inspect the source archive"
        ) from exc
    if not any(kind == b"F" for kind, _mode, _content in entries.values()):
        raise WaveformT1FinalizationError("source archive contains no regular files")
    for name, value in {
        ".source-revision": source_revision,
        ".source-tree": source_tree,
        ".archive-sha256": archive_sha256,
    }.items():
        entries[name] = (b"F", 0, (value + "\n").encode("ascii"))

    digest = sha256(b"waveqa-source-snapshot-v1\0")
    for name in sorted(entries):
        kind, executable, content = entries[name]
        encoded = name.encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(executable.to_bytes(1, "big"))
        if kind == b"F":
            assert content is not None
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def _validate_remote_source_snapshot(
    project_root: Path,
    *,
    source_revision: str,
    source_tree: str,
    remote_archive_sha256: object,
    remote_snapshot_sha256: object,
) -> dict[str, str]:
    archive_digest = _require_sha256(
        remote_archive_sha256, "remote source archive SHA-256"
    )
    snapshot_digest = _require_sha256(
        remote_snapshot_sha256, "remote source snapshot SHA-256"
    )
    archive_bytes = _git_archive_bytes(project_root, source_revision)
    expected_archive = sha256(archive_bytes).hexdigest()
    if archive_digest != expected_archive:
        raise WaveformT1FinalizationError(
            "remote source archive is not the exact local execution-commit archive"
        )
    expected_snapshot = _virtual_snapshot_sha256(
        archive_bytes,
        source_revision=source_revision,
        source_tree=source_tree,
        archive_sha256=expected_archive,
    )
    if snapshot_digest != expected_snapshot:
        raise WaveformT1FinalizationError(
            "remote source snapshot does not match the locally reproduced installer snapshot"
        )
    return {
        "source_archive_sha256": expected_archive,
        "source_snapshot_sha256": expected_snapshot,
    }


def _expected_backend_cases(
    manifest: ParityManifest, simulator: Simulator
) -> tuple[ScenarioCase, ...]:
    cases = tuple(
        case for case in expand_scenario_cases(manifest) if case.simulator is simulator
    )
    ids = tuple(case.case_id for case in cases)
    if len(cases) != EXPECTED_BACKEND_CASE_COUNT or len(set(ids)) != len(ids):
        raise WaveformT1FinalizationError(
            f"manifest must define exactly 16 unique {simulator.value} cases"
        )
    return cases


def _validate_manifest_contract(manifest: ParityManifest) -> tuple[ScenarioCase, ...]:
    if manifest.schema_version != EXPECTED_MANIFEST_SCHEMA_VERSION:
        raise WaveformT1FinalizationError("Waveform T1 requires manifest schema_version=2")
    if manifest.manifest_id != EXPECTED_MANIFEST_ID:
        raise WaveformT1FinalizationError(
            f"Waveform T1 manifest_id must be {EXPECTED_MANIFEST_ID!r}"
        )
    if tuple(hand.side for hand in manifest.hands) != EXPECTED_HANDS:
        raise WaveformT1FinalizationError("Waveform T1 hand order must be left, right")
    if tuple(scenario.scenario_id for scenario in manifest.scenarios) != EXPECTED_SCENARIOS:
        raise WaveformT1FinalizationError("Waveform T1 scenario order changed")
    if manifest.expected_joint_mapping_count != EXPECTED_JOINT_MAPPING_COUNT:
        raise WaveformT1FinalizationError("Waveform T1 must preregister 44 joint mappings")
    if manifest.expected_distal_frame_mapping_count != EXPECTED_DP_FRAME_MAPPING_COUNT:
        raise WaveformT1FinalizationError("Waveform T1 must preregister 10 DP frames")
    if not math.isclose(
        manifest.run_policy.minimum_completion_fraction, 0.99, rel_tol=0.0, abs_tol=0.0
    ):
        raise WaveformT1FinalizationError("Waveform T1 completion threshold must remain 0.99")
    cases = expand_scenario_cases(manifest)
    if len(cases) != EXPECTED_TOTAL_CASE_COUNT or len(
        {case.case_id for case in cases}
    ) != len(cases):
        raise WaveformT1FinalizationError("manifest must expand to exactly 32 unique cases")
    return cases


def _load_exact_backend_runs(
    evidence_root: Path,
    manifest: ParityManifest,
    simulator: Simulator,
) -> tuple[CollectedRun, ...]:
    expected = _expected_backend_cases(manifest, simulator)
    expected_by_id = {case.case_id: case for case in expected}
    cases_dir = evidence_root / "cases"
    if cases_dir.is_symlink() or not cases_dir.is_dir():
        raise WaveformT1FinalizationError(
            f"{simulator.value} evidence lacks a regular cases directory"
        )
    entries = tuple(cases_dir.iterdir())
    observed = {
        entry.name for entry in entries if entry.is_dir() and not entry.is_symlink()
    }
    invalid = sorted(
        entry.name for entry in entries if entry.is_symlink() or not entry.is_dir()
    )
    if observed != set(expected_by_id) or invalid or len(entries) != len(expected):
        raise WaveformT1FinalizationError(
            f"{simulator.value} evidence is not the exact 16-case directory set"
        )
    expected_paths = {
        (cases_dir / case_id / f"{case_id}{RUN_FILE_SUFFIX}").resolve()
        for case_id in expected_by_id
    }
    observed_paths = {path.resolve() for path in evidence_root.rglob(f"*{RUN_FILE_SUFFIX}")}
    if observed_paths != expected_paths:
        raise WaveformT1FinalizationError(
            f"{simulator.value} evidence does not contain exactly its 16 run payloads"
        )
    loaded: list[CollectedRun] = []
    for case in expected:
        path = cases_dir / case.case_id / f"{case.case_id}{RUN_FILE_SUFFIX}"
        runs = load_collected_runs(path)
        if len(runs) != 1 or runs[0].case != case:
            raise WaveformT1FinalizationError(
                f"collected run does not match canonical case {case.case_id}"
            )
        if runs[0].bundle_root_sha256 is not None:
            raise WaveformT1FinalizationError(
                f"raw case {case.case_id} references a prior bundle"
            )
        loaded.append(runs[0])
    return tuple(loaded)


def _validated_worker_process_record(value: object, label: str) -> dict[str, object]:
    record = _mapping(value, label)
    if set(record) != {
        "schema_version",
        "worker_pid",
        "os_process_identity",
        "fresh_process_id",
    } or record.get("schema_version") != 1:
        raise WaveformT1FinalizationError(f"{label} fields are not exact")
    try:
        identity = validate_os_process_identity(record.get("os_process_identity"))
        fresh = os_process_identity_sha256(identity)
    except ProcessIdentityError as exc:
        raise WaveformT1FinalizationError(f"{label} is invalid: {exc}") from exc
    worker_pid = _positive_int(record.get("worker_pid"), f"{label}.worker_pid")
    if worker_pid != identity["pid"] or record.get("fresh_process_id") != fresh:
        raise WaveformT1FinalizationError(f"{label} is not bound to its OS process identity")
    return {
        "schema_version": 1,
        "worker_pid": worker_pid,
        "os_process_identity": identity,
        "fresh_process_id": fresh,
    }


def _validate_remote_fresh_process_record(
    value: object,
    *,
    case: ScenarioCase,
    run: CollectedRun,
    source_tree: str,
) -> str:
    """Validate the private kernel identity of the actual OVPhysX worker."""

    label = f"{case.case_id} private fresh-process identity"
    record = _mapping(value, label)
    expected_keys = {
        "schema_version",
        "visibility",
        "case_id",
        "worker_command_identity",
        "os_process_identity",
        "fresh_process_id",
    }
    if set(record) != expected_keys:
        raise WaveformT1FinalizationError(f"{label} fields are not exact")
    if (
        record.get("schema_version") != 1
        or record.get("visibility") != "private_not_for_publication"
        or record.get("case_id") != case.case_id
    ):
        raise WaveformT1FinalizationError(f"{label} metadata is invalid")
    command = _mapping(record.get("worker_command_identity"), f"{label} command")
    if set(command) != {"python_realpath", "script_realpath"}:
        raise WaveformT1FinalizationError(f"{label} command fields are not exact")
    python_text = command.get("python_realpath")
    script_text = command.get("script_realpath")
    if not isinstance(python_text, str) or not isinstance(script_text, str):
        raise WaveformT1FinalizationError(f"{label} command paths are invalid")
    python_path = PurePosixPath(python_text)
    script_path = PurePosixPath(script_text)
    expected_script = (
        _REMOTE_APPROVED_ROOT
        / "project"
        / source_tree
        / "scripts"
        / "probe_ovphysx_runtime.py"
    )
    try:
        python_path.relative_to(_REMOTE_APPROVED_ROOT)
    except ValueError as exc:
        raise WaveformT1FinalizationError(
            f"{label} Python path escapes the approved root"
        ) from exc
    if (
        not python_path.is_absolute()
        or python_path.as_posix() != python_text
        or not python_path.name.startswith("python")
        or script_path != expected_script
        or script_path.as_posix() != script_text
    ):
        raise WaveformT1FinalizationError(f"{label} command identity is invalid")
    try:
        identity = validate_os_process_identity(record.get("os_process_identity"))
        fresh_process_id = os_process_identity_sha256(identity)
    except ProcessIdentityError as exc:
        raise WaveformT1FinalizationError(f"{label} is invalid: {exc}") from exc
    if identity.get("platform") != "posix":
        raise WaveformT1FinalizationError(f"{label} is not a POSIX kernel identity")
    if run.result.provenance.get("worker_pid") != identity.get("pid"):
        raise WaveformT1FinalizationError(
            f"{label} is not bound to run provenance.worker_pid"
        )
    if record.get("fresh_process_id") != fresh_process_id:
        raise WaveformT1FinalizationError(f"{label} hash is invalid")
    return fresh_process_id


def _validate_common_run_provenance(
    run: CollectedRun,
    manifest: ParityManifest,
    *,
    session_id: str,
    source_revision: str,
    local_runtime_identity: Mapping[str, object] | None = None,
) -> None:
    provenance = _mapping(run.result.provenance, f"{run.case.case_id} provenance")
    expected = {
        "manifest_sha256": manifest_sha256(manifest),
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "mapping_schema_version": 1,
    }
    if run.case.simulator is Simulator.MUJOCO:
        if local_runtime_identity is None:
            raise WaveformT1FinalizationError(
                f"{run.case.case_id} lacks the finalizer runtime identity"
            )
        expected.update(
            {
                "backend_version": local_runtime_identity["mujoco_version"],
                "python_version": local_runtime_identity["python_version"],
                "platform": local_runtime_identity["platform"],
                "local_runtime_identity": dict(local_runtime_identity),
            }
        )
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if mismatches:
        raise WaveformT1FinalizationError(
            f"{run.case.case_id} has invalid provenance: {', '.join(sorted(mismatches))}"
        )
    scenario = manifest.scenario(run.case.scenario_id)
    hand = manifest.hand(run.case.hand)
    expected_steps = round(scenario.duration_s / run.case.dt_s)
    full_digest = canonical_target_sequence_sha256(
        scenario, hand.joint_names, dt_s=run.case.dt_s, include_terminal=True
    )
    target_expected: dict[str, object] = {
        "target_sequence_digest_schema_version": 1,
        "target_sequence_digest_encoding": TARGET_DIGEST_ENCODING,
        "target_sequence_digest_projection": TARGET_DIGEST_PROJECTION,
        "target_sequence_canonical_joint_names": list(hand.joint_names),
        "scheduled_target_sequence_semantics": SCHEDULED_TARGET_SEMANTICS,
        "scheduled_target_sequence_count": expected_steps + 1,
        "scheduled_target_sequence_sha256": full_digest,
    }
    if run.case.simulator is Simulator.OVPHYSX:
        prefix_digest = canonical_target_sequence_sha256(
            scenario, hand.joint_names, dt_s=run.case.dt_s, include_terminal=False
        )
        target_expected.update(
            {
                "requested_target_sequence_semantics": (
                    "q[0..N] passed to set_joint_position_target_index"
                ),
                "immediate_target_readback_sequence_semantics": (
                    "q[0..N] read immediately from joint_pos_target after each request"
                ),
                "pre_step_applied_target_readback_sequence_semantics": (
                    "q[0..N-1] read after write_data_to_sim and before each physics step"
                ),
                "requested_target_sequence_count": expected_steps + 1,
                "immediate_target_readback_sequence_count": expected_steps + 1,
                "pre_step_applied_target_readback_sequence_count": expected_steps,
                "requested_target_sequence_sha256": full_digest,
                "immediate_target_readback_sequence_sha256": full_digest,
                "pre_step_applied_target_readback_sequence_sha256": prefix_digest,
            }
        )
        error = provenance.get("position_target_canonical_readback_max_abs_error_rad")
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or not 0.0 <= float(error) <= 1e-6
        ):
            raise WaveformT1FinalizationError(
                f"{run.case.case_id} canonical target readback is invalid"
            )
    target_mismatches = [
        key for key, value in target_expected.items() if provenance.get(key) != value
    ]
    if target_mismatches:
        raise WaveformT1FinalizationError(
            f"{run.case.case_id} target evidence is invalid: "
            + ", ".join(sorted(target_mismatches))
        )


def _aggregate_ovphysx_effort_clipping(
    manifest: ParityManifest,
    runs: Sequence[CollectedRun],
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate and aggregate observed IdealPD effort clipping from all OV runs.

    The public interpretation is deliberately not accepted from provenance.
    Only frozen numeric observations are consumed here; the report layer
    independently derives ``saturated/nonlinear`` from the aggregate count.
    Peak computed/applied torque vectors provide a second, independent
    consistency check against a forged clipping counter.
    """

    expected_cases = _expected_backend_cases(manifest, Simulator.OVPHYSX)
    expected_by_id = {case.case_id: case for case in expected_cases}
    ovphysx_runs = tuple(
        run for run in runs if run.case.simulator is Simulator.OVPHYSX
    )
    observed = {
        run.case.case_id: run
        for run in ovphysx_runs
    }
    if (
        len(ovphysx_runs) != EXPECTED_OVPHYSX_EFFORT_CASE_COUNT
        or len(observed) != EXPECTED_OVPHYSX_EFFORT_CASE_COUNT
        or set(observed) != set(expected_by_id)
    ):
        raise WaveformT1FinalizationError(
            "effort-clipping audit requires exactly the 16 canonical OVPhysX runs"
        )

    total_observations = 0
    total_clips = 0
    cases_with_clipping = 0
    case_records: list[dict[str, object]] = []
    for case in expected_cases:
        run = observed[case.case_id]
        if run.case != case:
            raise WaveformT1FinalizationError(
                f"{case.case_id} effort provenance is bound to the wrong case"
            )
        provenance = _mapping(
            run.result.provenance, f"{case.case_id} effort provenance"
        )
        expected_steps = round(
            manifest.scenario(case.scenario_id).duration_s / case.dt_s
        )
        expected_observations = expected_steps + 1
        observations = _positive_int(
            provenance.get("effort_observation_count"),
            f"{case.case_id} effort_observation_count",
        )
        if observations != expected_observations:
            raise WaveformT1FinalizationError(
                f"{case.case_id} effort observation coverage is not exact"
            )
        clips = _nonnegative_int(
            provenance.get("effort_clip_count"),
            f"{case.case_id} effort_clip_count",
        )
        if clips > observations * EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION:
            raise WaveformT1FinalizationError(
                f"{case.case_id} effort_clip_count exceeds observed joint efforts"
            )
        if (
            provenance.get("effort_command_source")
            != "articulation_data_computed_and_applied_torque_torch"
        ):
            raise WaveformT1FinalizationError(
                f"{case.case_id} effort observation source is invalid"
            )

        computed = _finite_nonnegative_vector(
            provenance.get("computed_effort_peak_abs_nm"),
            f"{case.case_id} computed_effort_peak_abs_nm",
        )
        applied = _finite_nonnegative_vector(
            provenance.get("applied_effort_peak_abs_nm"),
            f"{case.case_id} applied_effort_peak_abs_nm",
        )
        limits = _finite_nonnegative_vector(
            provenance.get("controller_dof_effort_limit"),
            f"{case.case_id} controller_dof_effort_limit",
            strictly_positive=True,
        )
        peak_saturated_joint_count = 0
        for index, (computed_peak, applied_peak, effort_limit) in enumerate(
            zip(computed, applied, limits)
        ):
            expected_applied_peak = min(computed_peak, effort_limit)
            if not math.isclose(
                applied_peak,
                expected_applied_peak,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise WaveformT1FinalizationError(
                    f"{case.case_id} applied effort peak {index} contradicts clipping"
                )
            peak_saturated_joint_count += int(computed_peak > effort_limit)
        if (
            (clips > 0) != (peak_saturated_joint_count > 0)
            or clips < peak_saturated_joint_count
        ):
            raise WaveformT1FinalizationError(
                f"{case.case_id} effort_clip_count contradicts observed torque peaks"
            )

        total_observations += observations
        total_clips += clips
        cases_with_clipping += int(clips > 0)
        case_records.append(
            {
                "case_id": case.case_id,
                "effort_observation_count": observations,
                "effort_clip_count": clips,
                "peak_saturated_joint_count": peak_saturated_joint_count,
            }
        )

    public_observation = {
        "schema_version": OVPHYSX_EFFORT_CLIPPING_SCHEMA_VERSION,
        "backend_case_count": EXPECTED_OVPHYSX_EFFORT_CASE_COUNT,
        "joint_count_per_observation": EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION,
        "effort_observation_count": total_observations,
        "effort_clip_count": total_clips,
    }
    private_audit = {
        **public_observation,
        "case_count_with_effort_clipping": cases_with_clipping,
        "cases": case_records,
    }
    return public_observation, private_audit


def _validate_local_evidence(
    root: Path,
    manifest: ParityManifest,
    *,
    manifest_file_sha256: str,
    source_revision: str,
    source_tree: str,
) -> tuple[tuple[CollectedRun, ...], dict[str, object]]:
    top_entries = tuple(root.iterdir())
    if {entry.name for entry in top_entries} != {"launcher", "cases"} or any(
        entry.is_symlink() or not entry.is_dir() for entry in top_entries
    ):
        raise WaveformT1FinalizationError(
            "MuJoCo evidence root must contain exactly launcher/ and cases/"
        )
    launcher = root / "launcher"
    launcher_entries = tuple(launcher.iterdir())
    if {entry.name for entry in launcher_entries} != LOCAL_LAUNCHER_FILES or any(
        entry.is_symlink() or not entry.is_file() for entry in launcher_entries
    ):
        raise WaveformT1FinalizationError("MuJoCo launcher evidence topology is not exact")
    launcher_inventory = _verify_evidence_inventory(
        launcher, launcher / "evidence.sha256", "MuJoCo launcher/evidence.sha256"
    )
    manifest_snapshot = launcher / "waveform_t1.manifest.json"
    if load_manifest(manifest_snapshot) != manifest:
        raise WaveformT1FinalizationError("MuJoCo launcher manifest snapshot changed")

    provenance = _mapping(
        _strict_json_file(launcher / "provenance.json", "MuJoCo provenance"),
        "MuJoCo provenance",
    )
    session_id = provenance.get("session_id")
    if not isinstance(session_id, str) or _SESSION_RE.fullmatch(session_id) is None:
        raise WaveformT1FinalizationError("MuJoCo session identity is invalid")
    runtime_identity = _local_runtime_identity()
    expected_common: dict[str, object] = {
        "schema_version": 1,
        "campaign": "waveform_t1",
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "launcher_script_sha256": _sha256_file(
            Path(__file__).resolve(strict=True).parent / "run_waveform_t1_local.py"
        ),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "manifest_snapshot_sha256": _sha256_file(manifest_snapshot),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification_required": "scanned_path_size_bytes_per_case",
        "local_runtime_identity": runtime_identity,
        "source_bridge_included": False,
    }
    if any(provenance.get(key) != value for key, value in expected_common.items()):
        raise WaveformT1FinalizationError("MuJoCo launcher provenance identity is invalid")
    launcher_pid = _positive_int(provenance.get("launcher_pid"), "MuJoCo launcher_pid")
    if set(provenance) != {*expected_common, "launcher_pid"}:
        raise WaveformT1FinalizationError("MuJoCo launcher provenance fields are not exact")

    expected_cases = _expected_backend_cases(manifest, Simulator.MUJOCO)
    case_ids = [case.case_id for case in expected_cases]
    matrix = _mapping(
        _strict_json_file(launcher / "matrix.json", "MuJoCo matrix"), "MuJoCo matrix"
    )
    completed = _mapping(
        _strict_json_file(launcher / "completed.json", "MuJoCo completion"),
        "MuJoCo completion",
    )
    if matrix != {**dict(provenance), "case_count": 16, "case_ids": case_ids}:
        raise WaveformT1FinalizationError("MuJoCo launcher matrix is not canonical")
    completed_fixed = {**dict(provenance), "completed_case_count": 16, "case_ids": case_ids}
    if any(completed.get(key) != value for key, value in completed_fixed.items()):
        raise WaveformT1FinalizationError("MuJoCo completion record is not canonical")
    worker_pids = completed.get("worker_pids")
    fresh_ids = completed.get("fresh_process_ids")
    if (
        not isinstance(worker_pids, list)
        or len(worker_pids) != 16
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in worker_pids
        )
        or len(set(worker_pids)) != 16
        or completed.get("fresh_worker_process_count") != 16
        or not isinstance(fresh_ids, list)
        or len(fresh_ids) != 16
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in fresh_ids
        )
        or len(set(fresh_ids)) != 16
    ):
        raise WaveformT1FinalizationError("MuJoCo completion lacks 16 fresh processes")
    if set(completed) != {
        *provenance,
        "completed_case_count",
        "case_ids",
        "worker_pids",
        "fresh_worker_process_count",
        "fresh_process_ids",
    }:
        raise WaveformT1FinalizationError("MuJoCo completion fields are not exact")

    runs = _load_exact_backend_runs(root, manifest, Simulator.MUJOCO)
    run_by_id = {run.case.case_id: run for run in runs}
    observed_worker_pids: list[int] = []
    observed_fresh_ids: list[str] = []
    runtime_log_records: list[dict[str, object]] = []
    for case in expected_cases:
        case_dir = root / "cases" / case.case_id
        entries = tuple(case_dir.iterdir())
        expected_files = {*LOCAL_CASE_FILES, f"{case.case_id}{RUN_FILE_SUFFIX}"}
        if {entry.name for entry in entries} != expected_files or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise WaveformT1FinalizationError(
                f"{case.case_id} local evidence topology is not exact"
            )
        _verify_evidence_inventory(
            case_dir, case_dir / "evidence.sha256", f"{case.case_id}/evidence.sha256"
        )
        for filename, stream in (
            ("launcher.stdout.log", "stdout"),
            ("launcher.stderr.log", "stderr"),
        ):
            runtime_log_records.append(
                _audit_runtime_log_file(
                    case_dir / filename,
                    label=f"mujoco/cases/{case.case_id}/{filename}",
                    scope="mujoco_case",
                    stream=stream,
                )
            )
        if (case_dir / "launcher.exitcode.txt").read_text(encoding="utf-8") != "0\n":
            raise WaveformT1FinalizationError(f"{case.case_id} local exit code is nonzero")
        command = _mapping(
            _strict_json_file(case_dir / "launcher.command.json", f"{case.case_id} command"),
            f"{case.case_id} command",
        )
        expected_argv = [
            "<python>",
            "-P",
            "scripts/run_waveform_t1_local.py",
            "_worker",
            "--identity-output",
            f"cases/{case.case_id}/worker-process.json",
            "--backend",
            "mujoco",
            "--asset-root",
            "<asset-root>",
            "--manifest",
            "launcher/waveform_t1.manifest.json",
            "--output-dir",
            f"cases/{case.case_id}",
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--case-id",
            case.case_id,
        ]
        if command != {"argv": expected_argv, "case_id": case.case_id, "schema_version": 1}:
            raise WaveformT1FinalizationError(f"{case.case_id} command record is invalid")

        process = _mapping(
            _strict_json_file(case_dir / "launcher.process.json", f"{case.case_id} process"),
            f"{case.case_id} process",
        )
        if set(process) != {
            "schema_version",
            "launcher_pid",
            "transport_pid",
            "worker_process",
            "returncode",
            "timed_out",
        } or process.get("schema_version") != 1 or process.get("launcher_pid") != launcher_pid:
            raise WaveformT1FinalizationError(f"{case.case_id} process record is invalid")
        transport_pid = _positive_int(
            process.get("transport_pid"), f"{case.case_id} transport PID"
        )
        if (
            transport_pid == launcher_pid
            or process.get("returncode") != 0
            or process.get("timed_out") is not False
        ):
            raise WaveformT1FinalizationError(f"{case.case_id} process did not complete cleanly")
        process_record = _validated_worker_process_record(
            process.get("worker_process"), f"{case.case_id} embedded worker process"
        )
        file_record = _validated_worker_process_record(
            _strict_json_file(case_dir / "worker-process.json", f"{case.case_id} worker process"),
            f"{case.case_id} worker process",
        )
        if process_record != file_record:
            raise WaveformT1FinalizationError(f"{case.case_id} worker process records differ")
        run = run_by_id[case.case_id]
        if run.result.provenance.get("worker_pid") != file_record["worker_pid"]:
            raise WaveformT1FinalizationError(f"{case.case_id} run worker PID is not bound")
        if not run.result.completed:
            raise WaveformT1FinalizationError(f"{case.case_id} did not complete")
        _validate_common_run_provenance(
            run,
            manifest,
            session_id=session_id,
            source_revision=source_revision,
            local_runtime_identity=runtime_identity,
        )
        observed_worker_pids.append(int(file_record["worker_pid"]))
        observed_fresh_ids.append(str(file_record["fresh_process_id"]))
    if observed_worker_pids != worker_pids or observed_fresh_ids != fresh_ids:
        raise WaveformT1FinalizationError("MuJoCo completion/fresh-process ledgers differ")

    return runs, {
        "session_id": session_id,
        "launcher_pid": launcher_pid,
        "fresh_process_count": len(observed_fresh_ids),
        "local_runtime_identity": runtime_identity,
        "launcher_inventory": launcher_inventory,
        "runtime_log_audit": _summarize_runtime_log_records(runtime_log_records),
    }


def _load_gate0_schema_contract(project_root: Path) -> object:
    path = project_root / "scripts" / "finalize_gate0.py"
    spec = importlib.util.spec_from_file_location("waveqa_t1_gate0_schema_contract", path)
    if spec is None or spec.loader is None:
        raise WaveformT1FinalizationError("cannot load the pinned schema validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema_directory(remote_root: Path) -> tuple[Path, str]:
    found: list[tuple[Path, str]] = []
    for relative in REMOTE_SCHEMA_RELATIVE_CANDIDATES:
        path = remote_root.joinpath(*relative.parts)
        if path.exists() or path.is_symlink():
            found.append((path, relative.as_posix()))
    if len(found) != 1:
        raise WaveformT1FinalizationError(
            "OVPhysX evidence must contain exactly one canonical schema-probe subtree"
        )
    path, relative = found[0]
    if path.is_symlink() or not path.is_dir():
        raise WaveformT1FinalizationError("OVPhysX schema evidence must be a directory")
    return path, relative


def _validate_remote_launcher_topology(
    remote_root: Path,
    schema_relative: str,
) -> dict[str, tuple[Path, ...]]:
    launcher = remote_root / "launcher"
    direct_files = {
        path.name for path in launcher.iterdir() if path.is_file()
    }
    direct_directories = {
        path.name for path in launcher.iterdir() if path.is_dir()
    }
    expected_directories = {"schema"} if schema_relative == "launcher/schema" else set()
    expected_files = set(REMOTE_LAUNCHER_REQUIRED_FILES)
    postflights_by_hand: dict[str, tuple[Path, ...]] = {}
    for hand in ("left", "right"):
        expected_files.update(
            {
                f"schema-{hand}-gpu-monitor.txt",
                f"schema-{hand}-owned-process-group.txt",
                f"schema-{hand}-preflight.txt",
            }
        )
        observed_postflight_names = sorted(
            name
            for name in direct_files
            if re.fullmatch(rf"schema-{hand}-postflight-[0-9]{{2}}\.txt", name)
        )
        expected_postflight_names = [
            f"schema-{hand}-postflight-{index:02d}.txt"
            for index in range(1, len(observed_postflight_names) + 1)
        ]
        if (
            not 1 <= len(observed_postflight_names) <= 3
            or observed_postflight_names != expected_postflight_names
        ):
            raise WaveformT1FinalizationError(
                f"OVPhysX schema-{hand} postflight evidence is not consecutive 01..03"
            )
        expected_files.update(expected_postflight_names)
        postflights_by_hand[hand] = tuple(
            launcher / name for name in expected_postflight_names
        )
    all_direct_entries = tuple(launcher.iterdir())
    if any(path.is_symlink() for path in all_direct_entries):
        raise WaveformT1FinalizationError("OVPhysX launcher contains a symbolic link")
    if direct_directories != expected_directories or direct_files != expected_files:
        raise WaveformT1FinalizationError(
            "OVPhysX launcher direct evidence topology is not exact: "
            f"missing={sorted(expected_files - direct_files)}, "
            f"unknown={sorted(direct_files - expected_files)}, "
            f"directories={sorted(direct_directories)}"
        )
    expected_nested_directories = (
        {"schema"} if schema_relative == "launcher/schema" else set()
    )
    observed_nested_directories = {
        path.relative_to(launcher).as_posix()
        for path in launcher.rglob("*")
        if path.is_dir()
    }
    if observed_nested_directories != expected_nested_directories:
        raise WaveformT1FinalizationError(
            "OVPhysX launcher contains an unknown nested directory"
        )
    if any(
        "foreign-gpu-process" in path.relative_to(launcher).as_posix()
        for path in launcher.rglob("*")
        if path.is_file()
    ):
        raise WaveformT1FinalizationError(
            "OVPhysX launcher recorded a foreign GPU process"
        )
    return postflights_by_hand


def _validate_gpu_monitor(
    path: Path,
    *,
    label: str,
    owner_pgid: int,
    gpu_index: str,
    gpu_uuid: str,
    gpu_name: str,
    driver_version: str,
) -> None:
    """Validate every retained monitor envelope; an empty fast run is valid."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WaveformT1FinalizationError(f"cannot read {label}") from exc
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].startswith("recorded_at_utc="):
            raise WaveformT1FinalizationError(f"{label} has an invalid record start")
        if cursor + 4 >= len(lines):
            raise WaveformT1FinalizationError(f"{label} contains a partial record")
        gpu_line = lines[cursor + 1]
        owner_pid_line = lines[cursor + 2]
        owner_pgid_line = lines[cursor + 3]
        if not gpu_line.startswith("gpu="):
            raise WaveformT1FinalizationError(f"{label} lacks GPU identity")
        gpu_fields = [field.strip() for field in gpu_line[4:].split(",")]
        if (
            len(gpu_fields) != 6
            or gpu_fields[0] != gpu_index
            or gpu_fields[1] != gpu_uuid
            or gpu_fields[2] != gpu_name
            or gpu_fields[3] != driver_version
            or re.fullmatch(r"[0-9]+", gpu_fields[4]) is None
            or re.fullmatch(r"[0-9]+", gpu_fields[5]) is None
        ):
            raise WaveformT1FinalizationError(f"{label} changed GPU identity")
        expected_owner = f"owned_case_pid={owner_pgid}"
        expected_group = f"owned_case_pgid={owner_pgid}"
        if owner_pid_line != expected_owner or owner_pgid_line != expected_group:
            raise WaveformT1FinalizationError(
                f"{label} is not bound to its owned process group"
            )
        if lines[cursor + 4] != "compute_processes_begin":
            raise WaveformT1FinalizationError(
                f"{label} lacks the compute-process envelope"
            )
        cursor += 5
        # The original frozen writer serialized an empty nvidia-smi query as
        # exactly one empty row.  Accept only that legacy empty-envelope form;
        # blank rows mixed with process records remain non-canonical.
        if (
            cursor + 1 < len(lines)
            and lines[cursor] == ""
            and lines[cursor + 1] == "compute_processes_end"
        ):
            cursor += 1
        while cursor < len(lines) and lines[cursor] != "compute_processes_end":
            fields = [field.strip() for field in lines[cursor].split(",")]
            if (
                len(fields) != 3
                or fields[0] != gpu_uuid
                or re.fullmatch(r"[1-9][0-9]*", fields[1]) is None
                or re.fullmatch(r"[0-9]+", fields[2]) is None
            ):
                raise WaveformT1FinalizationError(
                    f"{label} contains an unclassifiable compute process"
                )
            cursor += 1
        if cursor >= len(lines) or lines[cursor] != "compute_processes_end":
            raise WaveformT1FinalizationError(
                f"{label} has an unterminated compute-process envelope"
            )
        cursor += 1


def _validate_schema_evidence(
    remote_root: Path,
    manifest: ParityManifest,
    *,
    manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
    project_root: Path,
) -> tuple[str, tuple[dict[str, object], ...]]:
    schema_dir, relative = _schema_directory(remote_root)
    entries = tuple(schema_dir.iterdir())
    if {entry.name for entry in entries} != REMOTE_SCHEMA_FILES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise WaveformT1FinalizationError(
            "OVPhysX schema subtree must contain exactly two complete probes"
        )
    _verify_evidence_inventory(
        schema_dir, schema_dir / "evidence.sha256", "OVPhysX schema/evidence.sha256"
    )
    contract = _load_gate0_schema_contract(project_root)
    schema_probe_hash = _sha256_file(project_root / "scripts" / "probe_ovphysx_schema.py")
    runtime_log_records: list[dict[str, object]] = []
    for hand in ("left", "right"):
        for suffix, stream in (("stdout", "stdout"), ("stderr", "stderr")):
            runtime_log_records.append(
                _audit_runtime_log_file(
                    schema_dir / f"{hand}.{suffix}",
                    label=f"ovphysx/schema/{hand}.{suffix}",
                    scope="ovphysx_schema",
                    stream=stream,
                )
            )
        if (schema_dir / f"{hand}.exitcode").read_text(encoding="utf-8") != "0\n":
            raise WaveformT1FinalizationError(f"{hand} schema probe exit code is nonzero")
        try:
            contract._validate_schema_probe(
                _strict_json_file(schema_dir / f"{hand}.json", f"{hand} schema probe"),
                hand_side=hand,
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                session_id=session_id,
                source_revision=source_revision,
                schema_probe_script_sha256=schema_probe_hash,
            )
        except Exception as exc:
            raise WaveformT1FinalizationError(
                f"{hand} resolved-USD schema evidence is invalid: {exc}"
            ) from exc
    return relative, tuple(runtime_log_records)


def _legacy_gpu_contract(project_root: Path) -> object:
    return _load_gate0_schema_contract(project_root)


def _validate_remote_case_topology(
    case_dir: Path,
    case: ScenarioCase,
    manifest: ParityManifest,
) -> tuple[Path, ...]:
    required = {
        f"{case.case_id}{RUN_FILE_SUFFIX}",
        f"{case.case_id}{WORKER_RESULT_SUFFIX}",
        f"{case.case_id}{WORKER_STDOUT_SUFFIX}",
        f"{case.case_id}{WORKER_STDERR_SUFFIX}",
        f"{case.case_id}{WORKER_EXITCODE_SUFFIX}",
        "evidence.sha256",
        "gpu-monitor.txt",
        "launcher.exit-code.txt",
        "launcher.stderr.txt",
        "launcher.stdout.txt",
        "owned-process-group.txt",
        "preflight.txt",
        "private-fresh-process-identity.json",
    }
    observed = {
        path.relative_to(case_dir).as_posix()
        for path in case_dir.rglob("*")
        if path.is_file()
    }
    manifest_input = f"worker-inputs/{manifest_sha256(manifest)}.manifest.json"
    postflight_names = sorted(
        name for name in observed if re.fullmatch(r"postflight-[0-9]{2}\.txt", name)
    )
    expected_postflights = [
        f"postflight-{index:02d}.txt"
        for index in range(1, len(postflight_names) + 1)
    ]
    expected_files = required | {manifest_input, *expected_postflights}
    observed_directories = {
        path.relative_to(case_dir).as_posix()
        for path in case_dir.rglob("*")
        if path.is_dir()
    }
    if (
        not 1 <= len(postflight_names) <= 3
        or postflight_names != expected_postflights
        or observed != expected_files
        or observed_directories != {"worker-inputs"}
    ):
        unknown = sorted(observed - expected_files)
        missing = sorted(expected_files - observed)
        raise WaveformT1FinalizationError(
            f"{case.case_id} remote evidence topology is not exact: "
            f"missing={missing}, unknown={unknown}"
        )
    if load_manifest(case_dir / manifest_input) != manifest:
        raise WaveformT1FinalizationError(
            f"{case.case_id} worker manifest snapshot is not canonical"
        )
    return tuple(case_dir / name for name in postflight_names)


def _validate_remote_gpu_elapsed_s(status: Mapping[str, object]) -> int:
    value = status.get("gpu_elapsed_s")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WaveformT1FinalizationError("OVPhysX status.gpu_elapsed_s is invalid")
    if value > _MAX_REMOTE_GPU_ELAPSED_S:
        raise WaveformT1FinalizationError(
            "OVPhysX status.gpu_elapsed_s exceeds the two-hour hard ceiling"
        )
    return value


def _validate_remote_evidence(
    root: Path,
    manifest: ParityManifest,
    *,
    manifest_file_sha256: str,
    source_revision: str,
    source_tree: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str,
    project_root: Path,
) -> tuple[tuple[CollectedRun, ...], dict[str, object], str]:
    _verify_evidence_inventory(
        root,
        root / "launcher" / "evidence.sha256",
        "OVPhysX launcher/evidence.sha256",
    )
    top_entries = tuple(root.iterdir())
    allowed_top = {"launcher", "cases"}
    if {entry.name for entry in top_entries} != allowed_top or any(
        entry.is_symlink() or not entry.is_dir() for entry in top_entries
    ):
        raise WaveformT1FinalizationError("OVPhysX evidence root topology is not exact")
    launcher = root / "launcher"
    _schema_path, schema_relative_hint = _schema_directory(root)
    schema_postflights = _validate_remote_launcher_topology(
        root, schema_relative_hint
    )

    provenance = _parse_key_value_file(launcher / "provenance.txt", "OVPhysX provenance")
    status = _mapping(
        _strict_json_file(launcher / "status.json", "OVPhysX status"), "OVPhysX status"
    )
    session_id = provenance.get("session_id", "")
    run_id = provenance.get("run_id", "")
    if _SESSION_RE.fullmatch(session_id) is None or _RUN_ID_RE.fullmatch(run_id) is None:
        raise WaveformT1FinalizationError("OVPhysX session/run identity is invalid")
    required_provenance = {
        "schema_version": "1",
        "campaign": "waveform_t1",
        "manifest_id": EXPECTED_MANIFEST_ID,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "archive_sha256": source_archive_sha256,
        "snapshot_sha256": remote_snapshot_sha256,
        "project_path": f"project/{source_tree}",
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_lf_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "isaaclab_commit": _EXPECTED_ISAACLAB_COMMIT,
        "driver_version": _EXPECTED_DRIVER_VERSION,
        "formal_schema_probe_count": "2",
        **_EXPECTED_REMOTE_ENVIRONMENT,
    }
    mismatches = [
        key for key, value in required_provenance.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise WaveformT1FinalizationError(
            "OVPhysX provenance has invalid pinned fields: "
            + ", ".join(sorted(mismatches))
        )
    source_hashes = {
        "launcher_sha256": _sha256_file(project_root / "scripts" / "run_waveform_t1_remote.sh"),
        "worker_sha256": _sha256_file(project_root / "scripts" / "probe_ovphysx_runtime.py"),
        "schema_probe_sha256": _sha256_file(project_root / "scripts" / "probe_ovphysx_schema.py"),
        "schema_validator_sha256": _sha256_file(project_root / "scripts" / "finalize_gate0.py"),
    }
    if any(provenance.get(key) != value for key, value in source_hashes.items()):
        raise WaveformT1FinalizationError("OVPhysX launcher source hashes changed")
    gpu_index = provenance.get("gpu_index", "")
    gpu_uuid = provenance.get("gpu_uuid", "")
    gpu_name = provenance.get("gpu_name", "")
    if (
        re.fullmatch(r"[0-7]", gpu_index) is None
        or _GPU_UUID_RE.fullmatch(gpu_uuid) is None
        or _EXPECTED_GPU_NAME_FRAGMENT not in gpu_name
    ):
        raise WaveformT1FinalizationError("OVPhysX GPU identity is invalid")
    status_expected = {
        "schema_version": 1,
        "campaign": "waveform_t1",
        "status": "completed",
        "scientific_verdict": "pending_local_compare",
        "exit_code": 0,
        "run_id": run_id,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "snapshot_sha256": remote_snapshot_sha256,
        "selected_gpu_index": int(gpu_index),
        "selected_gpu_uuid": gpu_uuid,
        "completed_case_count": 16,
        "completed_schema_probe_count": 2,
        "verified_fresh_process_count": 16,
        "last_case_id": None,
    }
    if any(status.get(key) != value for key, value in status_expected.items()):
        raise WaveformT1FinalizationError("OVPhysX terminal status is not canonical")
    for key in ("elapsed_s", "approved_root_bytes"):
        value = status.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WaveformT1FinalizationError(f"OVPhysX status.{key} is invalid")
    _validate_remote_gpu_elapsed_s(status)
    if not isinstance(status.get("ended_at_utc"), str) or not status["ended_at_utc"]:
        raise WaveformT1FinalizationError("OVPhysX terminal timestamp is invalid")
    if set(status) != {
        *status_expected,
        "elapsed_s",
        "gpu_elapsed_s",
        "approved_root_bytes",
        "ended_at_utc",
    }:
        raise WaveformT1FinalizationError("OVPhysX terminal status fields are not exact")

    schema_relative, schema_log_records = _validate_schema_evidence(
        root,
        manifest,
        manifest_file_sha256=manifest_file_sha256,
        session_id=session_id,
        source_revision=source_revision,
        project_root=project_root,
    )
    if schema_relative != schema_relative_hint:
        raise WaveformT1FinalizationError("OVPhysX schema evidence location changed")
    case_lines = (launcher / "cases.txt").read_text(encoding="utf-8").splitlines()
    expected_cases = _expected_backend_cases(manifest, Simulator.OVPHYSX)
    expected_case_ids = [case.case_id for case in expected_cases]
    if case_lines != [f"manifest_sha256={manifest_sha256(manifest)}", *expected_case_ids]:
        raise WaveformT1FinalizationError("OVPhysX cases.txt is not the canonical matrix")

    gpu_contract = _legacy_gpu_contract(project_root)
    idle_kwargs = {
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": _EXPECTED_DRIVER_VERSION,
    }
    try:
        gpu_contract._validate_idle_gpu_state(
            launcher / "gpu-selection.txt", label="Waveform T1 GPU selection", **idle_kwargs
        )
    except Exception as exc:
        raise WaveformT1FinalizationError(f"OVPhysX GPU selection is invalid: {exc}") from exc

    schema_process_groups: list[int] = []
    for hand in ("left", "right"):
        pgid_text = (
            launcher / f"schema-{hand}-owned-process-group.txt"
        ).read_text(encoding="ascii").strip()
        if re.fullmatch(r"[1-9][0-9]*", pgid_text) is None:
            raise WaveformT1FinalizationError(
                f"OVPhysX schema-{hand} process-group identity is invalid"
            )
        pgid = int(pgid_text)
        schema_process_groups.append(pgid)
        postflights = schema_postflights[hand]
        try:
            gpu_contract._validate_idle_gpu_state(
                launcher / f"schema-{hand}-preflight.txt",
                label=f"schema-{hand} preflight",
                **idle_kwargs,
            )
            for path in postflights[:-1]:
                fields, _processes = gpu_contract._parse_gpu_state(
                    path, f"schema-{hand} {path.name}"
                )
                if fields.get("gpu_uuid") != gpu_uuid:
                    raise WaveformT1FinalizationError(
                        f"schema-{hand} postflight changed GPU UUID"
                    )
            gpu_contract._validate_idle_gpu_state(
                postflights[-1],
                label=f"schema-{hand} final postflight",
                **idle_kwargs,
            )
        except Exception as exc:
            raise WaveformT1FinalizationError(
                f"OVPhysX schema-{hand} GPU pre/postflight is invalid: {exc}"
            ) from exc
        _validate_gpu_monitor(
            launcher / f"schema-{hand}-gpu-monitor.txt",
            label=f"schema-{hand} GPU monitor",
            owner_pgid=pgid,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version=_EXPECTED_DRIVER_VERSION,
        )
    if len(set(schema_process_groups)) != 2:
        raise WaveformT1FinalizationError(
            "OVPhysX schema probes did not use two unique owned process groups"
        )

    runs = _load_exact_backend_runs(root, manifest, Simulator.OVPHYSX)
    run_by_id = {run.case.case_id: run for run in runs}
    worker_pids: list[int] = []
    process_groups: list[int] = []
    fresh_process_ids: list[str] = []
    runtime_log_records = list(schema_log_records)
    for case in expected_cases:
        case_dir = root / "cases" / case.case_id
        _verify_evidence_inventory(
            case_dir, case_dir / "evidence.sha256", f"{case.case_id}/evidence.sha256"
        )
        for filename, stream in (
            ("launcher.stdout.txt", "stdout"),
            ("launcher.stderr.txt", "stderr"),
            (f"{case.case_id}{WORKER_STDOUT_SUFFIX}", "stdout"),
            (f"{case.case_id}{WORKER_STDERR_SUFFIX}", "stderr"),
        ):
            runtime_log_records.append(
                _audit_runtime_log_file(
                    case_dir / filename,
                    label=f"ovphysx/cases/{case.case_id}/{filename}",
                    scope="ovphysx_case",
                    stream=stream,
                )
            )
        postflights = _validate_remote_case_topology(case_dir, case, manifest)
        for exit_name in (f"{case.case_id}{WORKER_EXITCODE_SUFFIX}", "launcher.exit-code.txt"):
            if (case_dir / exit_name).read_text(encoding="utf-8") != "0\n":
                raise WaveformT1FinalizationError(f"{case.case_id} remote exit code is nonzero")
        pgid_text = (case_dir / "owned-process-group.txt").read_text(encoding="ascii").strip()
        if re.fullmatch(r"[1-9][0-9]*", pgid_text) is None:
            raise WaveformT1FinalizationError(f"{case.case_id} process-group identity is invalid")
        process_groups.append(int(pgid_text))
        _validate_gpu_monitor(
            case_dir / "gpu-monitor.txt",
            label=f"{case.case_id} GPU monitor",
            owner_pgid=int(pgid_text),
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version=_EXPECTED_DRIVER_VERSION,
        )
        try:
            gpu_contract._validate_idle_gpu_state(
                case_dir / "preflight.txt", label=f"{case.case_id} preflight", **idle_kwargs
            )
            for path in postflights[:-1]:
                fields, _processes = gpu_contract._parse_gpu_state(
                    path, f"{case.case_id} {path.name}"
                )
                if fields.get("gpu_uuid") != gpu_uuid:
                    raise WaveformT1FinalizationError(
                        f"{case.case_id} postflight changed GPU UUID"
                    )
            gpu_contract._validate_idle_gpu_state(
                postflights[-1], label=f"{case.case_id} final postflight", **idle_kwargs
            )
        except Exception as exc:
            raise WaveformT1FinalizationError(
                f"{case.case_id} GPU pre/postflight is invalid: {exc}"
            ) from exc
        run = run_by_id[case.case_id]
        if not run.result.completed:
            raise WaveformT1FinalizationError(f"{case.case_id} did not complete")
        _validate_common_run_provenance(
            run, manifest, session_id=session_id, source_revision=source_revision
        )
        worker_pids.append(
            _positive_int(run.result.provenance.get("worker_pid"), f"{case.case_id} worker_pid")
        )
        fresh_process_ids.append(
            _validate_remote_fresh_process_record(
                _strict_json_file(
                    case_dir / "private-fresh-process-identity.json",
                    f"{case.case_id} private fresh-process identity",
                ),
                case=case,
                run=run,
                source_tree=source_tree,
            )
        )
    if (
        len(set(process_groups)) != 16
        or len(set(worker_pids)) != 16
        or len(set(fresh_process_ids)) != 16
        or status.get("verified_fresh_process_count") != len(fresh_process_ids)
    ):
        raise WaveformT1FinalizationError("OVPhysX evidence does not prove 16 fresh processes")
    return runs, {
        "run_id": run_id,
        "session_id": session_id,
        "source_tree": source_tree,
        "archive_sha256": source_archive_sha256,
        "snapshot_sha256": remote_snapshot_sha256,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": _EXPECTED_DRIVER_VERSION,
        "fresh_process_count": len(worker_pids),
        "runtime_log_audit": _summarize_runtime_log_records(runtime_log_records),
    }, schema_relative


def _validate_and_load_evidence(
    local_root: Path,
    remote_root: Path,
    manifest: ParityManifest,
    *,
    manifest_file_sha256: str,
    source_revision: str,
    source_tree: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str,
    project_root: Path,
) -> _EvidenceSet:
    local_runs, local_identity = _validate_local_evidence(
        local_root,
        manifest,
        manifest_file_sha256=manifest_file_sha256,
        source_revision=source_revision,
        source_tree=source_tree,
    )
    remote_runs, remote_identity, schema_relative = _validate_remote_evidence(
        remote_root,
        manifest,
        manifest_file_sha256=manifest_file_sha256,
        source_revision=source_revision,
        source_tree=source_tree,
        source_archive_sha256=source_archive_sha256,
        remote_snapshot_sha256=remote_snapshot_sha256,
        project_root=project_root,
    )
    runs = (*local_runs, *remote_runs)
    expected_ids = {case.case_id for case in expand_scenario_cases(manifest)}
    if len(runs) != 32 or {run.case.case_id for run in runs} != expected_ids:
        raise WaveformT1FinalizationError("active evidence is not exactly 32 canonical cases")
    return _EvidenceSet(
        runs=tuple(runs),
        local_identity=local_identity,
        remote_identity=remote_identity,
        schema_relative_path=schema_relative,
    )


def _build_descriptive_metrics(
    manifest: ParityManifest,
    runs: Sequence[CollectedRun],
) -> dict[str, object]:
    by_key = {
        (
            run.case.simulator,
            run.case.hand,
            run.case.scenario_id,
            run.case.timestep_variant,
            run.case.repeat_index,
        ): run.result
        for run in runs
    }
    cells: list[dict[str, object]] = []
    for hand_side in EXPECTED_HANDS:
        hand = manifest.hand(hand_side)
        for scenario_id in EXPECTED_SCENARIOS:
            mujoco = by_key[
                (Simulator.MUJOCO, hand_side, scenario_id, TimestepVariant.BASE, 1)
            ]
            ovphysx = by_key[
                (Simulator.OVPHYSX, hand_side, scenario_id, TimestepVariant.BASE, 1)
            ]
            cell: dict[str, object] = {
                "hand": hand_side.value,
                "scenario_id": scenario_id,
                "selection": "base_dt_repeat_01_preregistered_descriptive_view",
                "crosssim": state_state_metrics(
                    mujoco,
                    ovphysx,
                    joint_names=hand.joint_names,
                    frame_names=hand.distal_frame_names,
                ),
                "one_step_tracking": {
                    "pairing": "target[k]_to_state[k+1]",
                    "mujoco": one_step_tracking_metrics(
                        mujoco,
                        joint_names=hand.joint_names,
                        frame_names=hand.distal_frame_names,
                    ),
                    "ovphysx": one_step_tracking_metrics(
                        ovphysx,
                        joint_names=hand.joint_names,
                        frame_names=hand.distal_frame_names,
                    ),
                },
            }
            if scenario_id == "offset_sine":
                cell["waveform_response"] = {
                    "kind": "sine_ols",
                    "mujoco": sine_response_metrics(
                        mujoco,
                        joint_names=hand.joint_names,
                        frame_names=hand.distal_frame_names,
                    ),
                    "ovphysx": sine_response_metrics(
                        ovphysx,
                        joint_names=hand.joint_names,
                        frame_names=hand.distal_frame_names,
                    ),
                }
            else:
                cell["waveform_response"] = {
                    "kind": "chirp_fixed_frequency_bands",
                    "metrics": chirp_band_metrics(
                        mujoco,
                        ovphysx,
                        joint_names=hand.joint_names,
                        frame_names=hand.distal_frame_names,
                    ),
                }
            cells.append(cell)
    result: dict[str, object] = {
        "schema_version": 1,
        "decision_use": "descriptive_only_no_additional_thresholds",
        "timing_contract": (
            "target[k] applies over [t_k,t_{k+1}); tracking pairs target[k] "
            "with state[k+1]"
        ),
        "cells": cells,
    }
    json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return result


def _verify_crosssim_covers_all_dt_variants(
    manifest: ParityManifest,
    runs: Sequence[CollectedRun],
    comparison: Gate0Comparison,
) -> None:
    """Prove the reported cross-simulator maximum includes base and halved dt.

    Repeatability and dt-halving are prerequisite gates, but the frozen T1
    protocol did not narrow the scientific cross-simulator threshold to the
    base timestep.  This independent audit prevents a future comparator from
    silently dropping either registered timestep variant.
    """

    by_key = {
        (
            run.case.simulator,
            run.case.hand,
            run.case.scenario_id,
            run.case.timestep_variant,
            run.case.repeat_index,
        ): run.result
        for run in runs
    }
    result_by_cell = {
        (result.hand, result.scenario_id): result for result in comparison.results
    }
    for hand_side in EXPECTED_HANDS:
        hand = manifest.hand(hand_side)
        for scenario_id in EXPECTED_SCENARIOS:
            result = result_by_cell.get((hand_side, scenario_id))
            if result is None:
                # Structurally invalid executions have no scientific metric to
                # audit and remain INVALID/INCONCLUSIVE in the public report.
                continue
            observed = result.comparison.metrics
            if not observed:
                if (
                    result.comparison.comparison_status
                    is ComparisonStatus.INCONCLUSIVE
                ):
                    # The comparator intentionally emits no scientific metrics
                    # when execution validation failed.  Later finalization
                    # gates retain the INVALID/INCONCLUSIVE outcome.
                    continue
                raise WaveformT1FinalizationError(
                    "cross-simulator comparison must include metrics unless "
                    f"it is inconclusive: {hand_side.value}/{scenario_id}"
                )
            expected = [0.0, 0.0, 0.0]
            for variant in (TimestepVariant.BASE, TimestepVariant.HALVED):
                for repeat_index in range(1, manifest.run_policy.repeat_count + 1):
                    delta = trace_delta(
                        by_key[
                            (
                                Simulator.MUJOCO,
                                hand_side,
                                scenario_id,
                                variant,
                                repeat_index,
                            )
                        ],
                        by_key[
                            (
                                Simulator.OVPHYSX,
                                hand_side,
                                scenario_id,
                                variant,
                                repeat_index,
                            )
                        ],
                        hand,
                    )
                    expected = [max(left, right) for left, right in zip(expected, delta)]
            fields = (
                "crosssim_joint_max_abs_rad",
                "crosssim_frame_position_max_m",
                "crosssim_frame_orientation_max_rad",
            )
            missing = [field for field in fields if field not in observed]
            if missing:
                raise WaveformT1FinalizationError(
                    "cross-simulator comparison metrics are missing "
                    f"{', '.join(missing)} for "
                    f"{hand_side.value}/{scenario_id}"
                )
            mismatches = [
                field
                for field, value in zip(fields, expected)
                if not math.isclose(
                    float(observed.get(field, math.nan)),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ]
            if mismatches:
                raise WaveformT1FinalizationError(
                    "comparison cross-simulator metrics do not cover both base and "
                    f"halved dt for {hand_side.value}/{scenario_id}: "
                    + ", ".join(mismatches)
                )


def _payload_paths(root: Path) -> tuple[str, ...]:
    manifest_path = root / DEFAULT_MANIFEST_PATH
    paths: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WaveformT1FinalizationError(
                f"final bundle contains a symbolic link: {path.relative_to(root).as_posix()}"
            )
        if path.is_file() and path != manifest_path:
            paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def _regular_file_total_bytes(root: Path) -> int:
    _validate_regular_tree(root, "private bundle size input")
    return sum(
        path.stat(follow_symlinks=False).st_size
        for path in root.rglob("*")
        if path.is_file()
    )


def _seal_private_bundle_with_size(
    root: Path,
    *,
    finalization_path: Path,
    finalization_payload: Mapping[str, object],
    created_at_utc: str | None,
) -> int:
    """Write a self-consistent size record and exact bundle manifest."""

    recorded_size = 0
    for _attempt in range(8):
        payload = dict(finalization_payload)
        payload["private_bundle_size"] = {
            "regular_file_bytes": recorded_size,
            "hard_limit_bytes": MAX_PRIVATE_BUNDLE_BYTES,
            "within_hard_limit": True,
            "accounting_scope": "all_regular_files_including_bundle_manifest",
        }
        write_json_atomic(finalization_path, payload)
        write_bundle_manifest(
            root,
            _payload_paths(root),
            created_at_utc=created_at_utc,
        )
        actual_size = _regular_file_total_bytes(root)
        if actual_size > MAX_PRIVATE_BUNDLE_BYTES:
            raise WaveformT1FinalizationError(
                "private Waveform T1 bundle exceeds the 2 GiB hard limit: "
                f"{actual_size} > {MAX_PRIVATE_BUNDLE_BYTES}"
            )
        if actual_size == recorded_size:
            return actual_size
        recorded_size = actual_size
    raise WaveformT1FinalizationError(
        "private bundle size record did not converge to its exact sealed size"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise WaveformT1FinalizationError("text output parent does not exist")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_public_tree(
    root: Path,
    summary: Mapping[str, Any],
    report: str,
) -> None:
    _validate_regular_tree(root, "public Waveform T1 output")
    files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if files != {"summary.json", "report.md"}:
        raise WaveformT1FinalizationError(
            "public output must contain exactly summary.json and report.md"
        )
    if (root / "summary.json").read_text(encoding="utf-8") != canonical_summary_json(summary):
        raise WaveformT1FinalizationError("public summary is not canonical")
    if (root / "report.md").read_text(encoding="utf-8") != report:
        raise WaveformT1FinalizationError("public report is not deterministic")
    if render_public_report(summary) != report:
        raise WaveformT1FinalizationError("public report does not match its summary")
    combined = (root / "summary.json").read_text(encoding="utf-8") + report
    forbidden = (
        "session_id",
        "run_id",
        "worker_pid",
        "launcher_pid",
        "transport_pid",
        "pgid",
        "process_group",
        "fresh_process_id",
        "gpu_uuid",
        "private_plan",
        "/data/home/",
        "/mnt/ceph2",
    )
    lowered = combined.lower()
    if any(token in lowered for token in forbidden):
        raise WaveformT1FinalizationError("public output contains private execution evidence")
    if (
        re.search(
            r"(?:^|[\s\"'(<])(?:[A-Za-z]:[\\/]|\\\\)",
            combined,
            flags=re.MULTILINE,
        )
        is not None
    ):
        raise WaveformT1FinalizationError("public output contains an absolute Windows path")
    if _GPU_UUID_RE.search(combined) is not None:
        raise WaveformT1FinalizationError("public output contains a GPU UUID")


def _comparison_payload(comparison: Gate0Comparison | Mapping[str, Any]) -> Mapping[str, Any]:
    return comparison.to_dict() if isinstance(comparison, Gate0Comparison) else comparison


def finalize_waveform_t1(
    *,
    manifest_path: str | Path,
    local_evidence_root: str | Path,
    remote_evidence_root: str | Path,
    output_dir: str | Path,
    source_revision: str,
    source_archive_sha256: str,
    remote_snapshot_sha256: str,
    public_output_dir: str | Path | None = None,
    created_at_utc: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Create a private exact bundle and optional two-file public export."""

    if _REVISION_RE.fullmatch(source_revision) is None:
        raise WaveformT1FinalizationError("source_revision must be a lowercase 40-digit commit")
    archive_sha = _require_sha256(source_archive_sha256, "source_archive_sha256")
    snapshot_sha = _require_sha256(remote_snapshot_sha256, "remote_snapshot_sha256")
    project_root = Path(__file__).resolve(strict=True).parents[1]
    finalizer_runtime_identity = _local_runtime_identity()
    source_tree = _validate_source_identity(project_root, source_revision)
    reproduced_source = _validate_remote_source_snapshot(
        project_root,
        source_revision=source_revision,
        source_tree=source_tree,
        remote_archive_sha256=archive_sha,
        remote_snapshot_sha256=snapshot_sha,
    )
    archive_sha = reproduced_source["source_archive_sha256"]
    snapshot_sha = reproduced_source["source_snapshot_sha256"]
    source_hashes = {
        "finalizer_script_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "bundle_module_sha256": _sha256_file(
            project_root / "src/wave_asset_qa/parity/bundle.py"
        ),
        "compare_module_sha256": _sha256_file(
            project_root / "src/wave_asset_qa/parity/compare.py"
        ),
        "metrics_module_sha256": _sha256_file(
            project_root / "src/wave_asset_qa/parity/waveform_metrics.py"
        ),
        "report_module_sha256": _sha256_file(
            project_root / "src/wave_asset_qa/parity/waveform_report.py"
        ),
    }

    manifest_source = _resolved_file(manifest_path, "Waveform T1 manifest")
    local_source = _resolved_directory(local_evidence_root, "MuJoCo evidence root")
    remote_source = _resolved_directory(remote_evidence_root, "OVPhysX evidence root")
    output_parent, destination = _new_destination(output_dir, "private output")
    public_parent: Path | None = None
    public_destination: Path | None = None
    if public_output_dir is not None:
        public_parent, public_destination = _new_destination(public_output_dir, "public output")
    named_paths = [
        (local_source, "MuJoCo evidence"),
        (remote_source, "OVPhysX evidence"),
        (destination, "private output"),
    ]
    if public_destination is not None:
        named_paths.append((public_destination, "public output"))
    _validate_separate_trees(named_paths)

    manifest = load_manifest(manifest_source)
    _validate_manifest_contract(manifest)
    manifest_file_hash = _sha256_file(manifest_source)
    local_source_records = _tree_records(local_source)
    remote_source_records = _tree_records(remote_source)
    source_copy_budget = _validate_source_copy_budget(
        local_source_records,
        remote_source_records,
        manifest_size_bytes=manifest_source.stat(follow_symlinks=False).st_size,
    )
    private_staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial.", dir=output_parent)
    ).resolve(strict=True)
    public_staging: Path | None = None
    if public_parent is not None and public_destination is not None:
        public_staging = Path(
            tempfile.mkdtemp(prefix=f".{public_destination.name}.partial.", dir=public_parent)
        ).resolve(strict=True)
    try:
        manifest_dir = private_staging / "manifest"
        private_dir = private_staging / "private"
        evidence_dir = private_dir / "evidence"
        public_dir = private_staging / "public"
        manifest_dir.mkdir()
        private_dir.mkdir()
        evidence_dir.mkdir()
        public_dir.mkdir()
        _copy_tree_strict(local_source, evidence_dir / "mujoco", "MuJoCo evidence")
        _copy_tree_strict(remote_source, evidence_dir / "ovphysx", "OVPhysX evidence")
        if _tree_records(evidence_dir / "mujoco") != local_source_records:
            raise WaveformT1FinalizationError("MuJoCo evidence changed during copy")
        if _tree_records(evidence_dir / "ovphysx") != remote_source_records:
            raise WaveformT1FinalizationError("OVPhysX evidence changed during copy")

        manifest_snapshot = manifest_dir / "waveform_t1.json"
        with manifest_source.open("rb") as reader, manifest_snapshot.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if (
            _sha256_file(manifest_snapshot) != manifest_file_hash
            or load_manifest(manifest_snapshot) != manifest
        ):
            raise WaveformT1FinalizationError("manifest changed during finalization")

        evidence = _validate_and_load_evidence(
            evidence_dir / "mujoco",
            evidence_dir / "ovphysx",
            manifest,
            manifest_file_sha256=manifest_file_hash,
            source_revision=source_revision,
            source_tree=source_tree,
            source_archive_sha256=archive_sha,
            remote_snapshot_sha256=snapshot_sha,
            project_root=project_root,
        )
        (
            ovphysx_effort_clipping_observation,
            ovphysx_effort_clipping_audit,
        ) = _aggregate_ovphysx_effort_clipping(manifest, evidence.runs)
        thresholds = ComparisonThresholds(minimum_completion_fraction=0.99)
        comparison = compare_gate0_runs(manifest, evidence.runs, thresholds=thresholds)
        _verify_crosssim_covers_all_dt_variants(manifest, evidence.runs, comparison)
        descriptive_metrics = _build_descriptive_metrics(manifest, evidence.runs)
        runtime_log_audit = _combine_runtime_log_audits(
            (
                _mapping(
                    evidence.local_identity.get("runtime_log_audit"),
                    "MuJoCo runtime log audit",
                ),
                _mapping(
                    evidence.remote_identity.get("runtime_log_audit"),
                    "OVPhysX runtime log audit",
                ),
            )
        )
        local_fingerprint = _tree_fingerprint(evidence_dir / "mujoco")
        remote_fingerprint = _tree_fingerprint(evidence_dir / "ovphysx")
        sanitized_provenance = {
            "active_case_count": 32,
            "crosssim_threshold_timestep_variants": ["base", "halved"],
            "fresh_process_case_count": 32,
            "local_evidence_root_sha256": local_fingerprint["root_sha256"],
            "manifest_file_sha256": manifest_file_hash,
            "ovphysx_effort_clipping_observation": (
                ovphysx_effort_clipping_observation
            ),
            "remote_evidence_root_sha256": remote_fingerprint["root_sha256"],
            "remote_snapshot_sha256": snapshot_sha,
            "resolved_usd_schema_probe_count": 2,
            "source_archive_sha256": archive_sha,
            "source_revision": source_revision,
            "source_tree": source_tree,
            "runtime_log_gate_schema_version": runtime_log_audit["schema_version"],
            "runtime_log_file_count": runtime_log_audit["file_count"],
            "runtime_log_total_bytes": runtime_log_audit["total_bytes"],
            "runtime_log_total_lines": runtime_log_audit["total_lines"],
            "runtime_warning_line_count": runtime_log_audit["warning_line_count"],
            "runtime_known_kitless_diagnostic_line_count": runtime_log_audit[
                "known_kitless_diagnostic_line_count"
            ],
            "runtime_unclassified_warning_line_count": runtime_log_audit[
                "unclassified_warning_line_count"
            ],
            "runtime_fatal_match_count": runtime_log_audit["fatal_match_count"],
            "runtime_known_kitless_counts": runtime_log_audit[
                "known_kitless_counts"
            ],
        }
        public_summary, public_report = build_public_artifacts(
            comparison, descriptive_metrics, sanitized_provenance
        )
        write_json_atomic(private_dir / "comparison.json", _comparison_payload(comparison))
        write_json_atomic(private_dir / "descriptive_metrics.json", descriptive_metrics)
        finalization_payload: dict[str, object] = {
                "schema_version": 1,
                "protocol_id": "wavesimparity-waveform-t1-v1",
                "source_revision": source_revision,
                "source_tree": source_tree,
                "source_archive_sha256": archive_sha,
                "remote_snapshot_sha256": snapshot_sha,
                "manifest_file_sha256": manifest_file_hash,
                "manifest_canonical_sha256": manifest_sha256(manifest),
                "active_case_count": len(evidence.runs),
                "case_ids": sorted(run.case.case_id for run in evidence.runs),
                "local_evidence": local_fingerprint,
                "remote_evidence": remote_fingerprint,
                "source_copy_budget": source_copy_budget,
                "local_execution_identity": dict(evidence.local_identity),
                "remote_execution_identity": dict(evidence.remote_identity),
                "runtime_log_audit": runtime_log_audit,
                "ovphysx_effort_clipping_audit": (
                    ovphysx_effort_clipping_audit
                ),
                "resolved_usd_schema_evidence": {
                    "relative_path": evidence.schema_relative_path,
                    "probe_count": 2,
                    "physics_step_claimed": False,
                },
                "comparison": {
                    "execution_status": comparison.execution_status.value,
                    "scientific_label": comparison.comparison_status.value,
                    "thresholds": thresholds.to_dict(),
                    "crosssim_threshold_timestep_variants": ["base", "halved"],
                    "repeatability_and_dt_halving_are_prerequisite_gates": True,
                },
                "formal_gate0": {
                    "comparison_status": "divergent",
                    "pass_ready": False,
                    "unchanged": True,
                },
                "source_file_hashes": source_hashes,
                "publication_boundary": {
                    "private": ["private/**"],
                    "public": ["public/summary.json", "public/report.md"],
                },
            }
        write_json_atomic(private_dir / "finalization.json", finalization_payload)
        _write_text_atomic(public_dir / "summary.json", canonical_summary_json(public_summary))
        _write_text_atomic(public_dir / "report.md", public_report)
        _validate_public_tree(public_dir, public_summary, public_report)
        if public_staging is not None:
            _write_text_atomic(
                public_staging / "summary.json",
                canonical_summary_json(public_summary),
            )
            _write_text_atomic(public_staging / "report.md", public_report)
            _validate_public_tree(public_staging, public_summary, public_report)

        if _validate_source_identity(project_root, source_revision) != source_tree:
            raise WaveformT1FinalizationError("finalizer source identity changed")
        if _local_runtime_identity() != finalizer_runtime_identity:
            raise WaveformT1FinalizationError(
                "finalizer Python or dependency identity changed"
            )
        current_source_paths = {
            "finalizer_script_sha256": Path(__file__).resolve(strict=True),
            "bundle_module_sha256": project_root / "src/wave_asset_qa/parity/bundle.py",
            "compare_module_sha256": project_root / "src/wave_asset_qa/parity/compare.py",
            "metrics_module_sha256": project_root / "src/wave_asset_qa/parity/waveform_metrics.py",
            "report_module_sha256": project_root / "src/wave_asset_qa/parity/waveform_report.py",
        }
        if any(
            _sha256_file(current_source_paths[name]) != digest
            for name, digest in source_hashes.items()
        ):
            raise WaveformT1FinalizationError("finalizer source files changed during finalization")

        private_bundle_bytes = _seal_private_bundle_with_size(
            private_staging,
            finalization_path=private_dir / "finalization.json",
            finalization_payload=finalization_payload,
            created_at_utc=created_at_utc,
        )
        verification: dict[str, object] = dict(verify_bundle(private_staging))
        manifest_payload = _mapping(
            _strict_json_file(private_staging / DEFAULT_MANIFEST_PATH, "private bundle manifest"),
            "private bundle manifest",
        )
        listed = {record["path"] for record in manifest_payload["files"]}
        actual = {
            path.relative_to(private_staging).as_posix()
            for path in private_staging.rglob("*")
            if path.is_file() and path.name != DEFAULT_MANIFEST_PATH
        }
        if listed != actual:
            raise WaveformT1FinalizationError("private bundle inventory is not exact")
        verification.update(
            {
                "protocol_id": "wavesimparity-waveform-t1-v1",
                "active_case_count": 32,
                "validation_status": public_summary["validation_status"],
                "scientific_label": public_summary["scientific_label"],
                "formal_gate0_comparison_status": "DIVERGENT",
                "formal_gate0_pass_ready": False,
                "public_file_count": 2,
                "private_bundle_regular_file_bytes": private_bundle_bytes,
                "private_bundle_hard_limit_bytes": MAX_PRIVATE_BUNDLE_BYTES,
                "private_bundle_within_hard_limit": True,
                "runtime_log_file_count": runtime_log_audit["file_count"],
                "runtime_log_fatal_match_count": runtime_log_audit[
                    "fatal_match_count"
                ],
                "runtime_known_kitless_diagnostic_line_count": runtime_log_audit[
                    "known_kitless_diagnostic_line_count"
                ],
                "ovphysx_effort_observation_count": (
                    ovphysx_effort_clipping_observation[
                        "effort_observation_count"
                    ]
                ),
                "ovphysx_effort_clip_count": (
                    ovphysx_effort_clipping_observation["effort_clip_count"]
                ),
            }
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("private output appeared during finalization")
        if public_destination is not None and (
            public_destination.exists() or public_destination.is_symlink()
        ):
            raise FileExistsError("public output appeared during finalization")
        private_staging.replace(destination)
        if public_staging is not None and public_destination is not None:
            public_staging.replace(public_destination)
        return destination.resolve(strict=True), verification
    except BaseException:
        if private_staging.exists() and _is_within(private_staging, output_parent):
            shutil.rmtree(private_staging)
        if (
            public_staging is not None
            and public_parent is not None
            and public_staging.exists()
            and _is_within(public_staging, public_parent)
        ):
            shutil.rmtree(public_staging)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--local-evidence-root", required=True, type=Path)
    parser.add_argument("--remote-evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-output-dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--remote-snapshot-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    destination, verification = finalize_waveform_t1(
        manifest_path=args.manifest,
        local_evidence_root=args.local_evidence_root,
        remote_evidence_root=args.remote_evidence_root,
        output_dir=args.output_dir,
        public_output_dir=args.public_output_dir,
        source_revision=args.source_revision,
        source_archive_sha256=args.source_archive_sha256,
        remote_snapshot_sha256=args.remote_snapshot_sha256,
    )
    print(
        json.dumps(
            {
                "bundle": destination.name,
                "root_sha256": verification["root_sha256"],
                "validation_status": verification["validation_status"],
                "scientific_label": verification["scientific_label"],
                "formal_gate0": {"comparison_status": "DIVERGENT", "pass_ready": False},
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
