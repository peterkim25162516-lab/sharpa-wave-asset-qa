#!/usr/bin/env python3
"""Run the 16 canonical MuJoCo Waveform T1 cases in fresh processes.

Every case receives an exclusive evidence directory, one child process,
sanitized command evidence, raw stdout/stderr, exit code, parent/child PID
evidence, one canonical run payload, and a SHA-256 inventory.  This launcher
never mixes the optional small-step source bridge into the scientific matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from typing import BinaryIO, Mapping, Sequence


class LocalWaveformT1Error(RuntimeError):
    """Raised when local Waveform T1 execution or evidence is noncanonical."""


def _source_bytecode_artifacts(project_root: Path) -> tuple[Path, ...]:
    """List ignored Python bytecode below the executable source surfaces."""

    artifacts: list[Path] = []
    try:
        for source_root_name in ("src", "scripts"):
            source_root = project_root / source_root_name
            if not source_root.is_dir() or source_root.is_symlink():
                raise LocalWaveformT1Error(
                    f"project {source_root_name}/ must be a regular directory"
                )
            for path in source_root.rglob("*"):
                if path.name == "__pycache__" or path.suffix.lower() in {
                    ".pyc",
                    ".pyo",
                }:
                    artifacts.append(path)
    except OSError as exc:
        raise LocalWaveformT1Error(
            "cannot audit src/ and scripts/ for ignored Python bytecode"
        ) from exc
    return tuple(sorted(artifacts, key=lambda path: path.as_posix()))


def _reject_source_bytecode(project_root: Path) -> None:
    artifacts = _source_bytecode_artifacts(project_root)
    if artifacts:
        first = artifacts[0].relative_to(project_root).as_posix()
        raise LocalWaveformT1Error(
            "src/ and scripts/ must contain no .pyc, .pyo, or __pycache__ "
            f"before formal execution; first artifact: {first}"
        )


if __name__ == "__main__":
    # This guard is deliberately before any project or simulator package import.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    _reject_source_bytecode(Path(__file__).resolve(strict=True).parents[1])


import mujoco
import numpy
import wave_asset_qa

from wave_asset_qa.parity.contracts import ParityManifest, Simulator
from wave_asset_qa.parity.process_identity import (
    ProcessIdentityError,
    current_os_process_identity,
    os_process_identity_sha256,
    terminate_posix_process_identity,
    terminate_windows_process_identity,
    validate_os_process_identity,
    wait_for_os_process_exit,
)
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    canonical_manifest_json,
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPECTED_CASE_COUNT = 16
EXPECTED_MANIFEST_ID = "wavesimparity-waveform-t1"
EXPECTED_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_RUN_FILE_SUFFIX = ".run.json"
MAX_PRIVATE_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
TARGET_DIGEST_ENCODING = "utf8_json_lines_float_hex_v1"
TARGET_DIGEST_PROJECTION = "ieee754_binary32_roundtrip"
SCHEDULED_TARGET_SEMANTICS = (
    "q[0..N]; target q[k] is recorded at t_k and applies to "
    "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TREE_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_LOCAL_PYTHON_VERSION = "3.12.6"
EXPECTED_LOCAL_MUJOCO_VERSION = "3.12.0"
EXPECTED_LOCAL_NUMPY_VERSION = "2.5.2"
EXPECTED_LOCAL_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_LOCAL_PLATFORM = "Windows-11-10.0.26200-SP0"
EXPECTED_LOCAL_MACHINE = "AMD64"
EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256 = (
    "3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238"
)


class _ArtifactBudgetExceeded(RuntimeError):
    """Raised internally when the formal evidence tree exceeds its hard cap."""


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    returncode: int
    timed_out: bool
    transport_pid: int | None
    worker_process: Mapping[str, object] | None


def _resolved_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise LocalWaveformT1Error(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalWaveformT1Error(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise LocalWaveformT1Error(f"{label} must be a directory: {resolved}")
    return resolved


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise LocalWaveformT1Error(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalWaveformT1Error(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise LocalWaveformT1Error(f"{label} must be a regular file: {resolved}")
    return resolved


def _create_output_root(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output path already exists; refusing overwrite: {requested}")
    if requested.name in {"", ".", ".."}:
        raise LocalWaveformT1Error("output path must name a new directory")
    parent = _resolved_directory(requested.parent, "output parent")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"output path already exists; refusing overwrite: {destination}"
        )
    destination.mkdir(mode=0o700)
    return destination.resolve(strict=True)


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _write_text_exclusive(path, serialized)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_runtime_identity(python_executable: Path) -> dict[str, object]:
    """Validate and hash the exact interpreter and frozen local packages."""

    current_python = Path(sys.executable).resolve(strict=True)
    candidate = python_executable.resolve(strict=True)
    try:
        same_interpreter = os.path.samefile(candidate, current_python)
    except OSError as exc:
        raise LocalWaveformT1Error("cannot compare the Python executable") from exc
    if not same_interpreter:
        raise LocalWaveformT1Error(
            "--python must identify this launcher's own sys.executable"
        )
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
        raise LocalWaveformT1Error(
            "local Waveform T1 dependency version drift: "
            + ", ".join(sorted(drift))
        )
    executable_sha256 = _sha256_file(current_python)
    if executable_sha256 != EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256:
        raise LocalWaveformT1Error(
            "local Waveform T1 Python executable SHA-256 drifted from the freeze"
        )
    return {
        "schema_version": 1,
        **versions,
        "python_executable_sha256": executable_sha256,
    }


def _artifact_size_bytes(root: Path) -> int:
    """Return regular-file bytes while rejecting links and special entries.

    Case workers publish some files with atomic renames.  A directory entry can
    therefore disappear between enumeration and ``lstat``.  Retry a complete
    snapshot instead of leaking ``FileNotFoundError`` past the owned-worker
    termination path; persistent churn fails closed.
    """

    for attempt in range(3):
        total = 0
        try:
            for path in root.rglob("*"):
                entry = path.lstat()
                if stat.S_ISLNK(entry.st_mode):
                    raise LocalWaveformT1Error(
                        f"local evidence gained a symbolic link: {path}"
                    )
                if stat.S_ISDIR(entry.st_mode):
                    continue
                if not stat.S_ISREG(entry.st_mode):
                    raise LocalWaveformT1Error(
                        f"local evidence gained a non-regular entry: {path}"
                    )
                total += entry.st_size
        except FileNotFoundError as exc:
            if attempt < 2:
                continue
            raise LocalWaveformT1Error(
                "local evidence tree remained unstable during artifact-budget "
                "measurement"
            ) from exc
        except OSError as exc:
            raise LocalWaveformT1Error(
                f"cannot measure the local evidence artifact budget: {exc}"
            ) from exc
        return total
    raise AssertionError("unreachable artifact-size retry state")


def _enforce_artifact_budget(
    root: Path,
    *,
    maximum_bytes: int = MAX_PRIVATE_ARTIFACT_BYTES,
) -> int:
    observed = _artifact_size_bytes(root)
    if observed > maximum_bytes:
        raise _ArtifactBudgetExceeded(
            "Waveform T1 local evidence exceeded the 2 GiB hard artifact limit: "
            f"{observed} > {maximum_bytes} bytes"
        )
    return observed


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
        raise LocalWaveformT1Error(f"cannot verify the local Git source: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise LocalWaveformT1Error(
            "cannot verify the local Git source"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    """Bind execution to the exact clean, committed checkout."""

    root = project_root.resolve(strict=True)
    _reject_source_bytecode(root)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != root:
        raise LocalWaveformT1Error(
            f"launcher project root is not the Git top level: {root} != {top_level}"
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise LocalWaveformT1Error(
            f"source_revision does not match local HEAD: {source_revision} != {head}"
        )
    source_tree = _git_text(
        root, "rev-parse", "--verify", f"{source_revision}^{{tree}}"
    )
    if _TREE_RE.fullmatch(source_tree) is None:
        raise LocalWaveformT1Error("local source commit did not resolve to a Git tree")
    worktree_changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if worktree_changes:
        raise LocalWaveformT1Error(
            "worktree/index, including untracked files, must be clean before a "
            "formal local run"
        )
    launcher_relative = Path(__file__).resolve(strict=True).relative_to(root).as_posix()
    _git_text(root, "ls-files", "--error-unmatch", "--", launcher_relative)
    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise LocalWaveformT1Error(
            "wave_asset_qa was imported from a source tree other than this checkout"
        )
    return source_tree


def canonical_mujoco_cases(manifest: ParityManifest) -> tuple[ScenarioCase, ...]:
    if manifest.schema_version != EXPECTED_MANIFEST_SCHEMA_VERSION:
        raise LocalWaveformT1Error("Waveform T1 requires manifest schema_version=2")
    if manifest.manifest_id != EXPECTED_MANIFEST_ID:
        raise LocalWaveformT1Error(
            f"Waveform T1 manifest_id must be {EXPECTED_MANIFEST_ID!r}"
        )
    cases = tuple(
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.MUJOCO
    )
    case_ids = tuple(case.case_id for case in cases)
    if len(cases) != EXPECTED_CASE_COUNT or len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise LocalWaveformT1Error(
            f"Waveform T1 manifest must expand to exactly {EXPECTED_CASE_COUNT} unique "
            f"MuJoCo cases; found {len(cases)}"
        )
    return cases


def _validated_worker_pid(provenance: Mapping[str, object]) -> int:
    worker_pid = provenance.get("worker_pid")
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
    ):
        raise LocalWaveformT1Error("completed run has invalid worker_pid provenance")
    return worker_pid


def _validate_run_provenance(
    provenance: Mapping[str, object],
    *,
    manifest: ParityManifest,
    case: ScenarioCase,
    canonical_manifest_sha256: str,
    session_id: str,
    source_revision: str,
    observed_worker_pid: int,
    runtime_identity: Mapping[str, object],
) -> int:
    scenario = manifest.scenario(case.scenario_id)
    joint_names = manifest.hand(case.hand).joint_names
    requested_steps = round(scenario.duration_s / case.dt_s)
    scheduled_target_sha256 = canonical_target_sequence_sha256(
        scenario,
        joint_names,
        dt_s=case.dt_s,
        include_terminal=True,
    )
    expected: dict[str, object] = {
        "manifest_sha256": canonical_manifest_sha256,
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "mapping_schema_version": 1,
        "backend_version": runtime_identity["mujoco_version"],
        "python_version": runtime_identity["python_version"],
        "platform": runtime_identity["platform"],
        "local_runtime_identity": dict(runtime_identity),
        "target_sequence_digest_schema_version": 1,
        "target_sequence_digest_encoding": TARGET_DIGEST_ENCODING,
        "target_sequence_digest_projection": TARGET_DIGEST_PROJECTION,
        "target_sequence_canonical_joint_names": list(joint_names),
        "scheduled_target_sequence_semantics": SCHEDULED_TARGET_SEMANTICS,
        "scheduled_target_sequence_count": requested_steps + 1,
        "scheduled_target_sequence_sha256": scheduled_target_sha256,
    }
    mismatches = [
        key for key, value in expected.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise LocalWaveformT1Error(
            "completed run has invalid provenance field(s): "
            + ", ".join(sorted(mismatches))
        )
    worker_pid = _validated_worker_pid(provenance)
    if worker_pid != observed_worker_pid:
        raise LocalWaveformT1Error(
            "completed run worker_pid does not match the actual Python worker"
        )
    return worker_pid


def _validate_case_output(
    case_dir: Path,
    case: ScenarioCase,
    *,
    manifest: ParityManifest,
    canonical_manifest_sha256: str,
    session_id: str,
    source_revision: str,
    observed_worker_pid: int,
    runtime_identity: Mapping[str, object],
) -> int:
    # Imported only after the worker has durably recorded its process identity;
    # the internal worker path likewise defers runner/adapter module loading.
    from wave_asset_qa.parity.runner import RUN_FILE_SUFFIX, load_collected_runs

    if RUN_FILE_SUFFIX != EXPECTED_RUN_FILE_SUFFIX:
        raise LocalWaveformT1Error("run payload suffix changed unexpectedly")
    expected_file = case_dir / f"{case.case_id}{EXPECTED_RUN_FILE_SUFFIX}"
    run_files = tuple(sorted(case_dir.glob(f"*{EXPECTED_RUN_FILE_SUFFIX}")))
    if run_files != (expected_file,):
        raise LocalWaveformT1Error(
            f"{case.case_id} must produce exactly {expected_file.name}; found "
            + ", ".join(path.name for path in run_files)
        )
    runs = load_collected_runs(expected_file)
    if len(runs) != 1 or runs[0].case != case:
        raise LocalWaveformT1Error(
            f"{case.case_id} output is not its canonical collected run"
        )
    run = runs[0]
    if not run.result.completed:
        raise LocalWaveformT1Error(
            f"{case.case_id} did not complete: {run.result.message}"
        )
    if run.bundle_root_sha256 is not None:
        raise LocalWaveformT1Error(
            f"{case.case_id} raw run unexpectedly references an old bundle"
        )
    return _validate_run_provenance(
        run.result.provenance,
        manifest=manifest,
        case=case,
        canonical_manifest_sha256=canonical_manifest_sha256,
        session_id=session_id,
        source_revision=source_revision,
        observed_worker_pid=observed_worker_pid,
        runtime_identity=runtime_identity,
    )


def _write_evidence_hashes(root: Path) -> None:
    hash_file = root / "evidence.sha256"
    members = tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path != hash_file
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in members
    ]
    _write_text_exclusive(hash_file, "\n".join(lines) + "\n")


def _validated_worker_process_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "worker_pid",
        "os_process_identity",
        "fresh_process_id",
    }:
        raise LocalWaveformT1Error("worker process record fields are not exact")
    if value.get("schema_version") != 1:
        raise LocalWaveformT1Error("worker process record schema is invalid")
    try:
        identity = validate_os_process_identity(value.get("os_process_identity"))
    except ProcessIdentityError as exc:
        raise LocalWaveformT1Error(str(exc)) from exc
    worker_pid = value.get("worker_pid")
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid != identity["pid"]
    ):
        raise LocalWaveformT1Error(
            "worker process identity is not bound to worker_pid"
        )
    fresh_process_id = value.get("fresh_process_id")
    try:
        expected_fresh_process_id = os_process_identity_sha256(identity)
    except ProcessIdentityError as exc:  # pragma: no cover - identity validated above
        raise LocalWaveformT1Error(str(exc)) from exc
    if fresh_process_id != expected_fresh_process_id:
        raise LocalWaveformT1Error("worker fresh-process hash is invalid")
    return {
        "schema_version": 1,
        "worker_pid": worker_pid,
        "os_process_identity": identity,
        "fresh_process_id": fresh_process_id,
    }


def _read_worker_process_record(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise LocalWaveformT1Error("worker process record must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalWaveformT1Error("worker process record is not valid JSON") from exc
    return _validated_worker_process_record(value)


def _terminate_owned_worker(
    process: subprocess.Popen[bytes],
    worker_process: Mapping[str, object] | None,
) -> None:
    termination_error: BaseException | None = None
    if worker_process is not None:
        record = _validated_worker_process_record(worker_process)
        identity = record["os_process_identity"]
        assert isinstance(identity, Mapping)
        try:
            if identity["platform"] == "windows":
                terminate_windows_process_identity(identity, timeout_s=15.0)
            else:
                terminate_posix_process_identity(identity, timeout_s=15.0)
        except (ProcessIdentityError, OSError) as exc:
            termination_error = exc
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=15.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            termination_error = termination_error or exc
    if termination_error is not None:
        raise LocalWaveformT1Error(
            "cannot prove timed-out Waveform T1 worker termination"
        ) from termination_error


def _run_case_process(
    command: Sequence[str],
    *,
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    environment: Mapping[str, str],
    cwd: Path,
    worker_process_path: Path,
    timeout_s: float,
    evidence_root: Path | None = None,
    maximum_evidence_bytes: int = MAX_PRIVATE_ARTIFACT_BYTES,
) -> _ProcessOutcome:
    process: subprocess.Popen[bytes]
    try:
        process = subprocess.Popen(
            list(command),
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=dict(environment),
            cwd=cwd,
        )
    except OSError as exc:
        stderr_handle.write(f"launcher error: {type(exc).__name__}: {exc}\n".encode())
        return _ProcessOutcome(
            returncode=127,
            timed_out=False,
            transport_pid=None,
            worker_process=None,
        )
    transport_pid = process.pid
    deadline = time.monotonic() + timeout_s
    worker_process: dict[str, object] | None = None
    last_record_error: BaseException | None = None
    last_budget_check = 0.0

    def check_budget() -> None:
        nonlocal last_budget_check
        if evidence_root is None:
            return
        now = time.monotonic()
        if now - last_budget_check < 0.25:
            return
        _enforce_artifact_budget(
            evidence_root,
            maximum_bytes=maximum_evidence_bytes,
        )
        last_budget_check = now

    try:
        while worker_process is None:
            check_budget()
            if worker_process_path.is_file() or worker_process_path.is_symlink():
                try:
                    worker_process = _read_worker_process_record(worker_process_path)
                except LocalWaveformT1Error as exc:
                    # Exclusive creation becomes visible before the durable writer
                    # closes it, so a partial read is retried while the process lives.
                    last_record_error = exc
            if worker_process is not None:
                break
            if process.poll() is not None:
                if worker_process_path.is_file() and not worker_process_path.is_symlink():
                    try:
                        worker_process = _read_worker_process_record(
                            worker_process_path
                        )
                    except LocalWaveformT1Error as exc:
                        last_record_error = exc
                    else:
                        break
                if last_record_error is not None:
                    raise LocalWaveformT1Error(
                        "worker exited without a valid process-identity record"
                    ) from last_record_error
                raise LocalWaveformT1Error(
                    "worker exited before its process-identity handshake"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            time.sleep(min(0.01, remaining))

        while process.poll() is None:
            check_budget()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            time.sleep(min(0.05, remaining))
        check_budget()
        returncode = process.wait(timeout=0.0)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise subprocess.TimeoutExpired(command, timeout_s)
        assert worker_process is not None
        identity = worker_process["os_process_identity"]
        assert isinstance(identity, Mapping)
        try:
            wait_for_os_process_exit(identity, timeout_s=remaining)
        except ProcessIdentityError as exc:
            raise LocalWaveformT1Error(
                "cannot prove actual Python worker exit"
            ) from exc
        return _ProcessOutcome(
            returncode=returncode,
            timed_out=False,
            transport_pid=transport_pid,
            worker_process=worker_process,
        )
    except subprocess.TimeoutExpired:
        _terminate_owned_worker(process, worker_process)
        return _ProcessOutcome(
            returncode=124,
            timed_out=True,
            transport_pid=transport_pid,
            worker_process=worker_process,
        )
    except (_ArtifactBudgetExceeded, LocalWaveformT1Error) as exc:
        stderr_handle.write((str(exc) + "\n").encode("utf-8"))
        stderr_handle.flush()
        _terminate_owned_worker(process, worker_process)
        return _ProcessOutcome(
            returncode=125,
            timed_out=False,
            transport_pid=transport_pid,
            worker_process=worker_process,
        )


def run_local_waveform_t1(
    *,
    asset_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
    python_executable: str | Path = sys.executable,
    case_timeout_s: float = 600.0,
) -> Path:
    """Execute and validate the exact 16-case local MuJoCo T1 matrix."""

    if _SESSION_RE.fullmatch(session_id) is None:
        raise LocalWaveformT1Error("session_id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise LocalWaveformT1Error(
            "source_revision must be a lowercase 40-character project commit"
        )
    if (
        isinstance(case_timeout_s, bool)
        or not isinstance(case_timeout_s, (int, float))
        or not math.isfinite(float(case_timeout_s))
        or case_timeout_s <= 0.0
    ):
        raise LocalWaveformT1Error("case_timeout_s must be a positive finite number")

    assets = _resolved_directory(asset_root, "asset root")
    source_manifest = _resolved_file(manifest_path, "manifest")
    python = _resolved_file(python_executable, "Python executable")
    runtime_identity = _local_runtime_identity(python)
    manifest = load_manifest(source_manifest)
    cases = canonical_mujoco_cases(manifest)
    canonical_digest = manifest_sha256(manifest)
    project_root = Path(__file__).resolve(strict=True).parents[1]
    source_tree = _validate_source_identity(project_root, source_revision)
    launcher_script_sha256 = _sha256_file(Path(__file__).resolve(strict=True))
    manifest_file_sha256 = _sha256_file(source_manifest)
    launcher_pid = os.getpid()

    def validate_static_inputs() -> None:
        observed_tree = _validate_source_identity(project_root, source_revision)
        if observed_tree != source_tree:
            raise LocalWaveformT1Error("local Git source tree changed during the run")
        if _local_runtime_identity(python) != runtime_identity:
            raise LocalWaveformT1Error(
                "local Python or dependency identity changed during the run"
            )
        if _sha256_file(Path(__file__).resolve(strict=True)) != launcher_script_sha256:
            raise LocalWaveformT1Error("Waveform T1 launcher changed during the run")
        if _sha256_file(source_manifest) != manifest_file_sha256:
            raise LocalWaveformT1Error("Waveform T1 manifest changed during the run")

    root = _create_output_root(output_dir)
    launcher_dir = root / "launcher"
    cases_dir = root / "cases"
    launcher_dir.mkdir()
    cases_dir.mkdir()
    manifest_snapshot = launcher_dir / "waveform_t1.manifest.json"
    _write_text_exclusive(manifest_snapshot, canonical_manifest_json(manifest) + "\n")
    manifest_snapshot_sha256 = _sha256_file(manifest_snapshot)
    launcher_identity: dict[str, object] = {
        "schema_version": 1,
        "campaign": "waveform_t1",
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "launcher_pid": launcher_pid,
        "launcher_script_sha256": launcher_script_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": canonical_digest,
        "manifest_snapshot_sha256": manifest_snapshot_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification_required": "scanned_path_size_bytes_per_case",
        "local_runtime_identity": runtime_identity,
        "source_bridge_included": False,
    }
    _write_json_exclusive(launcher_dir / "provenance.json", launcher_identity)
    _write_json_exclusive(
        launcher_dir / "matrix.json",
        {
            **launcher_identity,
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
        },
    )

    source_dir = project_root / "src"
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(source_dir)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WAVEQA_SESSION_ID"] = session_id
    environment["WAVEQA_SOURCE_TREE"] = source_revision

    completed_ids: list[str] = []
    worker_pids: list[int] = []
    fresh_process_ids: list[str] = []
    for case in cases:
        validate_static_inputs()
        case_dir = cases_dir / case.case_id
        case_dir.mkdir()
        worker_process_path = case_dir / "worker-process.json"
        command = [
            str(python),
            "-P",
            str(Path(__file__).resolve(strict=True)),
            "_worker",
            "--identity-output",
            str(worker_process_path),
            "--backend",
            Simulator.MUJOCO.value,
            "--asset-root",
            str(assets),
            "--manifest",
            str(manifest_snapshot),
            "--output-dir",
            str(case_dir),
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--case-id",
            case.case_id,
        ]
        recorded_command = [
            "<python>",
            "-P",
            "scripts/run_waveform_t1_local.py",
            "_worker",
            "--identity-output",
            f"cases/{case.case_id}/worker-process.json",
            "--backend",
            Simulator.MUJOCO.value,
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
        _write_json_exclusive(
            case_dir / "launcher.command.json",
            {
                "argv": recorded_command,
                "case_id": case.case_id,
                "schema_version": 1,
            },
        )
        stdout_path = case_dir / "launcher.stdout.log"
        stderr_path = case_dir / "launcher.stderr.log"
        with stdout_path.open("xb") as stdout_handle, stderr_path.open(
            "xb"
        ) as stderr_handle:
            outcome = _run_case_process(
                command,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                environment=environment,
                cwd=project_root,
                worker_process_path=worker_process_path,
                timeout_s=float(case_timeout_s),
                evidence_root=root,
            )
        if outcome.timed_out:
            with stderr_path.open("ab") as stderr_handle:
                stderr_handle.write(
                    f"launcher timeout after {float(case_timeout_s):.6g}s\n".encode()
                )
        _write_text_exclusive(
            case_dir / "launcher.exitcode.txt", f"{outcome.returncode}\n"
        )
        _write_json_exclusive(
            case_dir / "launcher.process.json",
            {
                "schema_version": 1,
                "launcher_pid": launcher_pid,
                "transport_pid": outcome.transport_pid,
                "worker_process": (
                    dict(outcome.worker_process)
                    if outcome.worker_process is not None
                    else None
                ),
                "returncode": outcome.returncode,
                "timed_out": outcome.timed_out,
            },
        )
        if outcome.returncode != 0:
            _write_evidence_hashes(case_dir)
            raise LocalWaveformT1Error(
                f"{case.case_id} runner exited with code {outcome.returncode}; "
                f"evidence retained at {case_dir}"
            )
        if (
            outcome.transport_pid is None
            or outcome.transport_pid <= 0
            or outcome.transport_pid == launcher_pid
        ):
            _write_evidence_hashes(case_dir)
            raise LocalWaveformT1Error(
                f"{case.case_id} has invalid child process identity"
            )
        if outcome.worker_process is None:
            _write_evidence_hashes(case_dir)
            raise LocalWaveformT1Error(
                f"{case.case_id} has no actual Python worker identity"
            )
        worker_process = _validated_worker_process_record(outcome.worker_process)
        if _read_worker_process_record(worker_process_path) != worker_process:
            _write_evidence_hashes(case_dir)
            raise LocalWaveformT1Error(
                f"{case.case_id} worker identity changed after process exit"
            )
        observed_worker_pid = int(worker_process["worker_pid"])
        fresh_process_id = str(worker_process["fresh_process_id"])
        try:
            validate_static_inputs()
            worker_pid = _validate_case_output(
                case_dir,
                case,
                manifest=manifest,
                canonical_manifest_sha256=canonical_digest,
                session_id=session_id,
                source_revision=source_revision,
                observed_worker_pid=observed_worker_pid,
                runtime_identity=runtime_identity,
            )
            if worker_pid in worker_pids:
                raise LocalWaveformT1Error(
                    f"{case.case_id} reused a prior worker PID"
                )
            if fresh_process_id in fresh_process_ids:
                raise LocalWaveformT1Error(
                    f"{case.case_id} reused a prior worker process identity"
                )
        except Exception:
            _write_evidence_hashes(case_dir)
            raise
        worker_pids.append(worker_pid)
        fresh_process_ids.append(fresh_process_id)
        _write_evidence_hashes(case_dir)
        _enforce_artifact_budget(root)
        completed_ids.append(case.case_id)

    observed_dirs = {
        path.name
        for path in cases_dir.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_ids = {case.case_id for case in cases}
    if observed_dirs != expected_ids or len(completed_ids) != EXPECTED_CASE_COUNT:
        raise LocalWaveformT1Error(
            "local Waveform T1 evidence is not the exact 16-case matrix"
        )
    all_run_files = tuple(cases_dir.rglob(f"*{EXPECTED_RUN_FILE_SUFFIX}"))
    if len(all_run_files) != EXPECTED_CASE_COUNT:
        raise LocalWaveformT1Error(
            "local Waveform T1 evidence does not contain exactly 16 run files"
        )
    if len(worker_pids) != EXPECTED_CASE_COUNT or len(set(worker_pids)) != len(
        worker_pids
    ):
        raise LocalWaveformT1Error(
            "local Waveform T1 evidence does not prove 16 fresh worker processes"
        )
    if len(fresh_process_ids) != EXPECTED_CASE_COUNT or len(
        set(fresh_process_ids)
    ) != len(fresh_process_ids):
        raise LocalWaveformT1Error(
            "local Waveform T1 evidence has non-unique OS process identities"
        )
    validate_static_inputs()
    _write_json_exclusive(
        launcher_dir / "completed.json",
        {
            **launcher_identity,
            "completed_case_count": len(completed_ids),
            "case_ids": completed_ids,
            "worker_pids": worker_pids,
            "fresh_worker_process_count": len(worker_pids),
            "fresh_process_ids": fresh_process_ids,
        },
    )
    _write_evidence_hashes(launcher_dir)
    _enforce_artifact_budget(root)
    return root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--python", dest="python_executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--case-timeout-s", type=float, default=600.0)
    return parser


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal fresh-process Waveform T1 worker."
    )
    parser.add_argument("--identity-output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("mujoco",))
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--case-id", required=True)
    return parser


def _bind_worker_run_to_runtime(
    output_dir: Path,
    case_id: str,
    runtime_identity: Mapping[str, object],
) -> None:
    """Atomically add the worker-observed runtime identity to its run payload."""

    from wave_asset_qa.parity.runner import (
        collected_run_to_dict,
        load_collected_runs,
    )

    run_path = output_dir / f"{case_id}{EXPECTED_RUN_FILE_SUFFIX}"
    runs = load_collected_runs(run_path)
    if len(runs) != 1 or runs[0].case.case_id != case_id:
        raise LocalWaveformT1Error(
            "worker output is not the one canonical run before runtime binding"
        )
    run = runs[0]
    provenance = dict(run.result.provenance)
    if (
        provenance.get("python_version") != runtime_identity["python_version"]
        or provenance.get("backend_version")
        != runtime_identity["mujoco_version"]
        or provenance.get("platform") != runtime_identity["platform"]
        or "local_runtime_identity" in provenance
    ):
        raise LocalWaveformT1Error(
            "MuJoCo worker provenance contradicts the frozen local runtime"
        )
    provenance["local_runtime_identity"] = dict(runtime_identity)
    updated = replace(run, result=replace(run.result, provenance=provenance))
    payload = collected_run_to_dict(updated)
    temporary = run_path.with_name(f".{run_path.name}.runtime.{os.getpid()}.tmp")
    try:
        _write_json_exclusive(temporary, payload)
        os.replace(temporary, run_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _worker_main(argv: Sequence[str]) -> int:
    args = _build_worker_parser().parse_args(argv)
    output_dir = _resolved_directory(args.output_dir, "worker output directory")
    identity_output = Path(os.path.abspath(args.identity_output.expanduser()))
    if identity_output != output_dir / "worker-process.json":
        raise LocalWaveformT1Error(
            "worker identity must be the case-local worker-process.json"
        )
    try:
        identity = current_os_process_identity()
        fresh_process_id = os_process_identity_sha256(identity)
    except ProcessIdentityError as exc:
        raise LocalWaveformT1Error(
            "cannot establish actual Python worker identity"
        ) from exc
    _write_json_exclusive(
        identity_output,
        {
            "schema_version": 1,
            "worker_pid": os.getpid(),
            "os_process_identity": identity,
            "fresh_process_id": fresh_process_id,
        },
    )
    runtime_identity = _local_runtime_identity(Path(sys.executable))
    # Keep the runner import behind the durable identity handshake so the
    # parent can identify the real interpreter before simulator construction.
    from wave_asset_qa.parity.runner import main as runner_main

    returncode = runner_main(
        [
            "--backend",
            args.backend,
            "--asset-root",
            str(args.asset_root),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(output_dir),
            "--session-id",
            args.session_id,
            "--source-revision",
            args.source_revision,
            "--case-id",
            args.case_id,
        ]
    )
    if returncode == 0:
        _bind_worker_run_to_runtime(output_dir, args.case_id, runtime_identity)
    return returncode


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["_worker"]:
        try:
            return _worker_main(arguments[1:])
        except (LocalWaveformT1Error, OSError, ValueError) as exc:
            print(f"Waveform T1 worker failed: {exc}", file=sys.stderr)
            return 2
    args = _build_parser().parse_args(arguments)
    try:
        root = run_local_waveform_t1(
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
            python_executable=args.python_executable,
            case_timeout_s=args.case_timeout_s,
        )
    except (LocalWaveformT1Error, FileExistsError, OSError, ValueError) as exc:
        print(f"local Waveform T1 failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "backend": Simulator.MUJOCO.value,
                "case_count": EXPECTED_CASE_COUNT,
                "output_dir": str(root),
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
