#!/usr/bin/env python3
"""Run the frozen 16-case Contact Gate C0 MuJoCo campaign.

The launcher binds every case to one clean committed source tree, one pinned
local runtime, one verified asset subtree, and one fresh OS process identity.
All output is private evidence.  Existing paths are never replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


class LocalContactC0Error(RuntimeError):
    """Raised when local C0 execution or evidence is noncanonical."""


def _source_bytecode_artifacts(project_root: Path) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    try:
        for name in ("src", "scripts"):
            source_root = project_root / name
            if not source_root.is_dir() or source_root.is_symlink():
                raise LocalContactC0Error(
                    f"project {name}/ must be a regular directory"
                )
            for path in source_root.rglob("*"):
                if path.name == "__pycache__" or path.suffix.lower() in {
                    ".pyc",
                    ".pyo",
                }:
                    artifacts.append(path)
    except OSError as error:
        raise LocalContactC0Error(
            "cannot audit src/ and scripts/ for ignored Python bytecode"
        ) from error
    return tuple(sorted(artifacts, key=lambda path: path.as_posix()))


def _reject_source_bytecode(project_root: Path) -> None:
    artifacts = _source_bytecode_artifacts(project_root)
    if artifacts:
        first = artifacts[0].relative_to(project_root).as_posix()
        raise LocalContactC0Error(
            "src/ and scripts/ must contain no .pyc, .pyo, or __pycache__ "
            f"before formal execution; first artifact: {first}"
        )


if __name__ == "__main__":
    # This guard must run before project, MuJoCo, or NumPy imports.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    _reject_source_bytecode(Path(__file__).resolve(strict=True).parents[1])


import mujoco
import numpy
import wave_asset_qa

from wave_asset_qa.contact.bundle import (
    ContactEvidenceError,
    canonical_json_sha256,
    inventory_root_sha256,
    read_json_strict,
    regular_tree_records,
    sha256_file,
    verify_adapter_private_evidence,
    write_json_exclusive,
)
from wave_asset_qa.contact.records import canonical_contact_run_json
from wave_asset_qa.contact.runner import load_and_validate_contact_run
from wave_asset_qa.contact.scenarios import (
    ContactCase,
    canonical_contact_manifest_json,
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.diagnostics import asset_tree_sha256, is_link_like
from wave_asset_qa.parity.process_identity import (
    ProcessIdentityError,
    os_process_identity_sha256,
    terminate_posix_process_identity,
    terminate_windows_process_identity,
    validate_os_process_identity,
    wait_for_os_process_exit,
)


EXPECTED_CASE_COUNT = 16
EXPECTED_MANIFEST_ID = "wavesimparity-contact-c0"
EXPECTED_MANIFEST_SCHEMA_VERSION = 1
EXPECTED_RUN_FILE_SUFFIX = ".run.json"
MAX_PRIVATE_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_FINAL_METADATA_RESERVE_BYTES = 16 * 1024 * 1024
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_TREE_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

EXPECTED_LOCAL_PYTHON_VERSION = "3.12.6"
EXPECTED_LOCAL_MUJOCO_VERSION = "3.12.0"
EXPECTED_LOCAL_NUMPY_VERSION = "2.5.2"
EXPECTED_LOCAL_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_LOCAL_PLATFORM = "Windows-11-10.0.26200-SP0"
EXPECTED_LOCAL_MACHINE = "AMD64"
EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256 = (
    "3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238"
)

_PRIVATE_PROCESS_KEYS = {
    "schema_version",
    "visibility",
    "case_id",
    "backend",
    "session_id",
    "source_revision",
    "source_tree",
    "manifest_file_sha256",
    "manifest_semantic_sha256",
    "asset_tree_sha256",
    "python_version",
    "python_executable_realpath",
    "worker_module_realpath",
    "worker_module_sha256",
    "os_process_identity",
    "fresh_process_identity_sha256",
}


class _ArtifactBudgetExceeded(RuntimeError):
    """Raised internally when the private campaign exceeds its hard cap."""


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    returncode: int
    timed_out: bool
    transport_pid: int | None
    worker_process: Mapping[str, object] | None


def _reject_link_like_ancestors(path: str | Path, label: str) -> Path:
    """Return an absolute lexical path only when every existing ancestor is real.

    ``Path.resolve`` follows Windows directory junctions as well as symbolic
    links. Calling it before this audit would erase the evidence that an input
    or output was routed through a reparse point.
    """

    candidate = Path(os.path.abspath(Path(path).expanduser()))
    for ancestor in (candidate, *candidate.parents):
        if is_link_like(ancestor):
            raise LocalContactC0Error(
                f"{label} has a symbolic-link or junction ancestor: {ancestor}"
            )
    return candidate


def _resolved_directory(path: str | Path, label: str) -> Path:
    candidate = _reject_link_like_ancestors(path, label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise LocalContactC0Error(f"{label} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise LocalContactC0Error(f"{label} must be a directory: {resolved}")
    return resolved


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = _reject_link_like_ancestors(path, label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise LocalContactC0Error(f"{label} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise LocalContactC0Error(f"{label} must be a regular file: {resolved}")
    return resolved


def _new_output_path(path: str | Path) -> Path:
    requested = _reject_link_like_ancestors(path, "output path")
    if requested.exists() or is_link_like(requested):
        raise FileExistsError(f"output path already exists; refusing overwrite: {requested}")
    if requested.name in {"", ".", ".."}:
        raise LocalContactC0Error("output path must name a new directory")
    parent = _resolved_directory(requested.parent, "output parent")
    destination = parent / requested.name
    _reject_link_like_ancestors(destination, "output path")
    if destination.exists() or is_link_like(destination):
        raise FileExistsError(
            f"output path already exists; refusing overwrite: {destination}"
        )
    return destination


def _write_text_exclusive(path: Path, text: str) -> None:
    destination = _reject_link_like_ancestors(path, "evidence destination")
    if destination.exists() or is_link_like(destination):
        raise FileExistsError(f"refusing to overwrite evidence: {destination}")
    parent = _resolved_directory(destination.parent, "evidence parent")
    if not parent.is_dir():
        raise LocalContactC0Error("evidence parent must be a regular directory")
    with (parent / destination.name).open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _local_runtime_identity(python_executable: Path) -> dict[str, object]:
    current = Path(sys.executable).resolve(strict=True)
    candidate = python_executable.resolve(strict=True)
    try:
        same = os.path.samefile(current, candidate)
    except OSError as error:
        raise LocalContactC0Error("cannot compare the Python executable") from error
    if not same:
        raise LocalContactC0Error("--python must identify this launcher's sys.executable")
    observed: dict[str, object] = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "mujoco_version": str(getattr(mujoco, "__version__", "")),
        "numpy_version": str(getattr(numpy, "__version__", "")),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable_sha256": _sha256_file(current),
    }
    expected = {
        "python_version": EXPECTED_LOCAL_PYTHON_VERSION,
        "mujoco_version": EXPECTED_LOCAL_MUJOCO_VERSION,
        "numpy_version": EXPECTED_LOCAL_NUMPY_VERSION,
        "python_implementation": EXPECTED_LOCAL_PYTHON_IMPLEMENTATION,
        "platform": EXPECTED_LOCAL_PLATFORM,
        "machine": EXPECTED_LOCAL_MACHINE,
        "python_executable_sha256": EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256,
    }
    drift = sorted(name for name, value in expected.items() if observed[name] != value)
    if drift:
        raise LocalContactC0Error(
            "local Contact C0 runtime drift: " + ", ".join(drift)
        )
    return observed


def _artifact_size_bytes(root: Path) -> int:
    evidence_root = _reject_link_like_ancestors(root, "local evidence root")
    if not evidence_root.is_dir():
        raise LocalContactC0Error("local evidence root must be a regular directory")
    for attempt in range(3):
        total = 0
        try:
            for path in evidence_root.rglob("*"):
                entry = path.lstat()
                if stat.S_ISLNK(entry.st_mode) or is_link_like(path):
                    raise LocalContactC0Error(
                        f"local evidence contains a symbolic link or junction: {path}"
                    )
                if stat.S_ISDIR(entry.st_mode):
                    continue
                if not stat.S_ISREG(entry.st_mode):
                    raise LocalContactC0Error(
                        f"local evidence contains a special entry: {path}"
                    )
                total += entry.st_size
        except FileNotFoundError as error:
            if attempt < 2:
                continue
            raise LocalContactC0Error(
                "local evidence remained unstable during budget measurement"
            ) from error
        except OSError as error:
            raise LocalContactC0Error(
                f"cannot measure local evidence size: {error}"
            ) from error
        return total
    raise AssertionError("unreachable artifact-size retry state")


def _enforce_artifact_budget(
    root: Path, *, maximum_bytes: int = MAX_PRIVATE_ARTIFACT_BYTES
) -> int:
    observed = _artifact_size_bytes(root)
    if observed > maximum_bytes:
        raise _ArtifactBudgetExceeded(
            "Contact C0 local evidence exceeded the 2 GiB hard artifact limit: "
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
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LocalContactC0Error(f"cannot verify local Git source: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise LocalContactC0Error(
            "cannot verify local Git source" + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    root = project_root.resolve(strict=True)
    _reject_source_bytecode(root)
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise LocalContactC0Error("launcher project root is not the Git top level")
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise LocalContactC0Error("source_revision does not match local HEAD")
    tree = _git_text(root, "rev-parse", "--verify", f"{source_revision}^{{tree}}")
    if _TREE_RE.fullmatch(tree) is None:
        raise LocalContactC0Error("source commit did not resolve to one Git tree")
    changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if changes:
        raise LocalContactC0Error(
            "worktree/index, including untracked files, must be clean before a formal run"
        )
    for relative in (
        "scripts/run_contact_c0_local.py",
        "scripts/run_contact_c0_mujoco_worker.py",
        "src/wave_asset_qa/contact/worker.py",
        "configs/parity/contact_c0.json",
    ):
        _git_text(root, "ls-files", "--error-unmatch", "--", relative)
    imported = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported != (root / "src" / "wave_asset_qa").resolve(strict=True):
        raise LocalContactC0Error(
            "wave_asset_qa was imported from a different source tree"
        )
    return tree


def canonical_mujoco_cases(manifest: object) -> tuple[ContactCase, ...]:
    if getattr(manifest, "schema_version", None) != EXPECTED_MANIFEST_SCHEMA_VERSION:
        raise LocalContactC0Error("Contact C0 requires manifest schema_version=1")
    if getattr(manifest, "manifest_id", None) != EXPECTED_MANIFEST_ID:
        raise LocalContactC0Error(
            f"Contact C0 manifest_id must be {EXPECTED_MANIFEST_ID!r}"
        )
    cases = tuple(
        case
        for case in expand_contact_cases(manifest)  # type: ignore[arg-type]
        if case.simulator is Simulator.MUJOCO
    )
    case_ids = tuple(case.case_id for case in cases)
    if len(cases) != EXPECTED_CASE_COUNT or len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise LocalContactC0Error(
            f"manifest must expand to exactly {EXPECTED_CASE_COUNT} unique MuJoCo cases"
        )
    return cases


def _asset_subtree(assets: Path, manifest: object) -> Path:
    relative = Path(str(manifest.provenance.asset_root))  # type: ignore[attr-defined]
    if relative.is_absolute() or ".." in relative.parts:
        raise LocalContactC0Error("manifest asset_root is not a safe relative path")
    subtree = _resolved_directory(assets / relative, "manifest asset subtree")
    try:
        subtree.relative_to(assets)
    except ValueError as error:
        raise LocalContactC0Error("manifest asset subtree escapes asset root") from error
    if is_link_like(subtree) or not subtree.is_dir():
        raise LocalContactC0Error("manifest asset subtree must be a regular directory")
    return subtree


def _validate_asset_tree(assets: Path, manifest: object) -> tuple[Path, str]:
    subtree = _asset_subtree(assets, manifest)
    observed = asset_tree_sha256(subtree)
    expected = str(manifest.provenance.canonical_lf_asset_tree_sha256)  # type: ignore[attr-defined]
    if observed != expected:
        raise LocalContactC0Error(
            f"materialized asset subtree SHA-256 differs: {observed} != {expected}"
        )
    return subtree, observed


def _validate_private_process_record(
    value: object,
    *,
    expected: Mapping[str, object] | None = None,
    experimental_campaign: str | None = None,
) -> dict[str, object]:
    keys = set(_PRIVATE_PROCESS_KEYS)
    if experimental_campaign is not None:
        if experimental_campaign != 'contact-c0-async-v1':
            raise LocalContactC0Error('unknown candidate campaign')
        keys.add('experimental_campaign')
    if not isinstance(value, Mapping) or set(value) != keys:
        raise LocalContactC0Error("private process record fields are not exact")
    if experimental_campaign is not None and value.get('experimental_campaign') != experimental_campaign:
        raise LocalContactC0Error('candidate process marker mismatch')
    if value.get("schema_version") != 1:
        raise LocalContactC0Error("private process schema_version is invalid")
    try:
        identity = validate_os_process_identity(value.get("os_process_identity"))
        identity_hash = os_process_identity_sha256(identity)
    except ProcessIdentityError as error:
        raise LocalContactC0Error(str(error)) from error
    if value.get("fresh_process_identity_sha256") != identity_hash:
        raise LocalContactC0Error("private process identity hash is invalid")
    if expected is not None:
        mismatches = sorted(
            name for name, wanted in expected.items() if value.get(name) != wanted
        )
        if mismatches:
            raise LocalContactC0Error(
                "private process record differs: " + ", ".join(mismatches)
            )
    return {**dict(value), "os_process_identity": identity}


def _read_private_process(path: Path) -> dict[str, object]:
    try:
        value = read_json_strict(path)
    except ContactEvidenceError as error:
        raise LocalContactC0Error("private process record is not valid JSON") from error
    # This reader binds/terminates only an owned worker. Final acceptance below
    # still requires an explicitly selected campaign and its exact artifacts.
    campaign = value.get('experimental_campaign') if isinstance(value, Mapping) else None
    return _validate_private_process_record(value, experimental_campaign=campaign)


def _windows_parent_process_id(pid: int) -> int:
    import ctypes
    from ctypes import wintypes

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = (
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        )

    create_snapshot = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot.CreateToolhelp32Snapshot.argtypes = (
        wintypes.DWORD,
        wintypes.DWORD,
    )
    create_snapshot.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    create_snapshot.Process32FirstW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    )
    create_snapshot.Process32FirstW.restype = wintypes.BOOL
    create_snapshot.Process32NextW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    )
    create_snapshot.Process32NextW.restype = wintypes.BOOL
    create_snapshot.CloseHandle.argtypes = (wintypes.HANDLE,)
    create_snapshot.CloseHandle.restype = wintypes.BOOL

    snapshot = create_snapshot.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    entry = _PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        if not create_snapshot.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise OSError(ctypes.get_last_error(), "Process32FirstW failed")
        while True:
            if int(entry.th32ProcessID) == pid:
                return int(entry.th32ParentProcessID)
            if not create_snapshot.Process32NextW(snapshot, ctypes.byref(entry)):
                error_code = ctypes.get_last_error()
                if error_code == 18:
                    break
                raise OSError(error_code, "Process32NextW failed")
    finally:
        create_snapshot.CloseHandle(snapshot)
    raise ProcessLookupError(pid)


def _bind_private_process_record(
    value: object,
    *,
    transport_pid: int,
) -> dict[str, object]:
    campaign = value.get('experimental_campaign') if isinstance(value, Mapping) else None
    record = _validate_private_process_record(value, experimental_campaign=campaign)
    identity = record["os_process_identity"]
    assert isinstance(identity, Mapping)
    worker_pid = int(identity["pid"])
    belongs_to_transport = worker_pid == transport_pid
    if not belongs_to_transport and identity["platform"] == "windows":
        try:
            belongs_to_transport = (
                _windows_parent_process_id(worker_pid) == transport_pid
            )
        except (OSError, ProcessLookupError) as error:
            raise LocalContactC0Error(
                "cannot establish Windows worker ownership"
            ) from error
    if not belongs_to_transport:
        raise LocalContactC0Error(
            "worker process identity does not belong to the owned transport process"
        )
    return record


def _read_owned_private_process(
    path: Path,
    *,
    transport_pid: int,
) -> dict[str, object]:
    return _bind_private_process_record(
        _read_private_process(path),
        transport_pid=transport_pid,
    )


def _terminate_owned_worker(
    process: subprocess.Popen[bytes],
    worker_process: Mapping[str, object] | None,
) -> None:
    termination_error: BaseException | None = None
    if worker_process is not None:
        try:
            record = _bind_private_process_record(
                worker_process,
                transport_pid=process.pid,
            )
            identity = record["os_process_identity"]
            assert isinstance(identity, Mapping)
            if identity["platform"] == "windows":
                terminate_windows_process_identity(identity, timeout_s=15.0)
            else:
                terminate_posix_process_identity(identity, timeout_s=15.0)
        except (LocalContactC0Error, ProcessIdentityError, OSError) as error:
            termination_error = error
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=15.0)
        except (OSError, subprocess.TimeoutExpired) as error:
            termination_error = termination_error or error
    if termination_error is not None:
        raise LocalContactC0Error("cannot prove Contact C0 worker termination") from termination_error


def _run_case_process(
    command: Sequence[str],
    *,
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    environment: Mapping[str, str],
    cwd: Path,
    private_process_path: Path,
    timeout_s: float,
    evidence_root: Path | None = None,
    maximum_evidence_bytes: int = MAX_PRIVATE_ARTIFACT_BYTES,
) -> _ProcessOutcome:
    try:
        process = subprocess.Popen(
            list(command),
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=dict(environment),
            cwd=cwd,
        )
    except OSError as error:
        stderr_handle.write(f"launcher error: {type(error).__name__}: {error}\n".encode())
        return _ProcessOutcome(127, False, None, None)

    deadline = time.monotonic() + timeout_s
    worker_process: dict[str, object] | None = None
    last_record_error: BaseException | None = None
    last_budget_check = 0.0

    def check_budget() -> None:
        nonlocal last_budget_check
        if evidence_root is None:
            return
        now = time.monotonic()
        if now - last_budget_check >= 0.25:
            _enforce_artifact_budget(
                evidence_root, maximum_bytes=maximum_evidence_bytes
            )
            last_budget_check = now

    try:
        while worker_process is None:
            check_budget()
            if private_process_path.is_file() or is_link_like(private_process_path):
                try:
                    worker_process = _read_owned_private_process(
                        private_process_path,
                        transport_pid=process.pid,
                    )
                except LocalContactC0Error as error:
                    # Exclusive creation is visible before the durable writer closes.
                    last_record_error = error
            if worker_process is not None:
                break
            if process.poll() is not None:
                if private_process_path.is_file() and not is_link_like(
                    private_process_path
                ):
                    try:
                        worker_process = _read_owned_private_process(
                            private_process_path,
                            transport_pid=process.pid,
                        )
                    except LocalContactC0Error as error:
                        last_record_error = error
                    else:
                        break
                if last_record_error is not None:
                    raise LocalContactC0Error(
                        "worker exited without a valid private process record"
                    ) from last_record_error
                raise LocalContactC0Error(
                    "worker exited before its process-identity handshake"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            time.sleep(min(0.01, remaining))

        assert worker_process is not None
        identity = worker_process["os_process_identity"]
        assert isinstance(identity, Mapping)
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
        try:
            wait_for_os_process_exit(identity, timeout_s=remaining)
        except ProcessIdentityError as error:
            raise LocalContactC0Error("cannot prove actual worker exit") from error
        return _ProcessOutcome(returncode, False, process.pid, worker_process)
    except subprocess.TimeoutExpired:
        _terminate_owned_worker(process, worker_process)
        return _ProcessOutcome(124, True, process.pid, worker_process)
    except (_ArtifactBudgetExceeded, LocalContactC0Error) as error:
        stderr_handle.write((str(error) + "\n").encode("utf-8"))
        stderr_handle.flush()
        _terminate_owned_worker(process, worker_process)
        return _ProcessOutcome(125, False, process.pid, worker_process)


def _same_file(recorded: object, expected: Path, label: str) -> None:
    if not isinstance(recorded, str) or not recorded:
        raise LocalContactC0Error(f"{label} is missing")
    try:
        same = os.path.samefile(recorded, expected)
    except OSError as error:
        raise LocalContactC0Error(f"cannot validate {label}") from error
    if not same:
        raise LocalContactC0Error(f"{label} differs from the frozen file")


def _validate_case_artifacts(
    case_dir: Path,
    case: ContactCase,
    *,
    manifest: object,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    asset_tree_sha256_value: str,
    session_id: str,
    source_revision: str,
    source_tree: str,
    runtime_identity: Mapping[str, object],
    project_root: Path,
    outcome: _ProcessOutcome,
    experimental_campaign: str | None = None,
) -> str:
    expected_names = {
        f"{case.case_id}{EXPECTED_RUN_FILE_SUFFIX}",
        "private-process.json",
        "private-adapter-evidence.json",
        "stdout.txt",
        "stderr.txt",
        "exit-code.txt",
    }
    observed_names: set[str] = set()
    if experimental_campaign is not None:
        expected_names.add('private-campaign.json')
    for path in case_dir.iterdir():
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise LocalContactC0Error(
                f"{case.case_id} contains a non-regular artifact: {path.name}"
            )
        observed_names.add(path.name)
    if observed_names != expected_names:
        raise LocalContactC0Error(
            f"{case.case_id} artifact names are not exact: "
            + ", ".join(sorted(observed_names))
        )
    process_path = case_dir / "private-process.json"
    expected_process: dict[str, object] = {
        "visibility": "private_not_for_publication",
        "case_id": case.case_id,
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "asset_tree_sha256": asset_tree_sha256_value,
        "python_version": runtime_identity["python_version"],
    }
    process_record = _validate_private_process_record(
        read_json_strict(process_path), expected=expected_process,
        experimental_campaign=experimental_campaign,
    )
    if outcome.worker_process is None or dict(outcome.worker_process) != process_record:
        raise LocalContactC0Error("private process record changed after worker exit")
    identity = process_record["os_process_identity"]
    assert isinstance(identity, Mapping)
    worker_pid = identity["pid"]
    if (
        outcome.transport_pid is None
        or outcome.transport_pid <= 0
        or outcome.transport_pid == os.getpid()
        or worker_pid == os.getpid()
        or (
            identity["platform"] != "windows"
            and worker_pid != outcome.transport_pid
        )
    ):
        raise LocalContactC0Error("worker or transport process identity is invalid")
    python_path = Path(sys.executable).resolve(strict=True)
    _same_file(
        process_record["python_executable_realpath"],
        python_path,
        "worker Python executable",
    )
    worker_module = (project_root / "src/wave_asset_qa/contact/worker.py").resolve(
        strict=True
    )
    _same_file(
        process_record["worker_module_realpath"],
        worker_module,
        "worker module",
    )
    if process_record["worker_module_sha256"] != _sha256_file(worker_module):
        raise LocalContactC0Error("worker module SHA-256 differs")

    fresh_hash = str(process_record["fresh_process_identity_sha256"])
    run_path = case_dir / f"{case.case_id}{EXPECTED_RUN_FILE_SUFFIX}"
    run = load_and_validate_contact_run(
        run_path,
        manifest=manifest,  # type: ignore[arg-type]
        case=case,
        source_revision=source_revision,
        source_tree=source_tree,
        fresh_process_identity_sha256=fresh_hash,
    )
    canonical_bytes = (canonical_contact_run_json(run) + "\n").encode("utf-8")
    if run_path.read_bytes() != canonical_bytes:
        raise LocalContactC0Error("run JSON is strict but not canonical")

    adapter_value = read_json_strict(case_dir / "private-adapter-evidence.json")
    if not isinstance(adapter_value, Mapping) or set(adapter_value) != {
        "fixture_overlay",
        "runtime_fingerprint",
        "unexpected_pair_observations",
    }:
        raise LocalContactC0Error("MuJoCo adapter private evidence fields are not exact")
    if adapter_value.get("unexpected_pair_observations") != []:
        raise LocalContactC0Error("MuJoCo adapter observed an unexpected contact pair")
    adapter_evidence = verify_adapter_private_evidence(adapter_value, run)
    fixture = adapter_evidence.get("fixture_overlay")
    runtime = adapter_evidence.get("runtime_fingerprint")
    if not isinstance(fixture, Mapping) or not isinstance(runtime, Mapping):
        raise LocalContactC0Error("adapter evidence preimages are missing")
    collision_preimage = fixture.get("collision_inventory_hash_preimage")
    if canonical_json_sha256(collision_preimage) != run.fixture_readback.get(
        "collision_inventory_sha256"
    ):
        raise LocalContactC0Error("collision inventory preimage hash mismatch")
    expected_runtime = {
        "backend": Simulator.MUJOCO.value,
        "backend_version": runtime_identity["mujoco_version"],
        "python_version": runtime_identity["python_version"],
        "platform": runtime_identity["platform"],
        "device": "cpu",
    }
    runtime_mismatches = sorted(
        name for name, expected in expected_runtime.items() if runtime.get(name) != expected
    )
    if runtime_mismatches:
        raise LocalContactC0Error(
            "adapter runtime fingerprint differs: " + ", ".join(runtime_mismatches)
        )
    if set(runtime) != {
        "backend",
        "backend_version",
        "python_version",
        "platform",
        "device",
        "solver",
        "integrator",
        "control_path",
        "contact_path",
    }:
        raise LocalContactC0Error("adapter runtime fingerprint fields are not exact")
    for name in ("solver", "integrator", "control_path", "contact_path"):
        if not isinstance(runtime.get(name), str) or not runtime[name]:
            raise LocalContactC0Error(f"adapter runtime fingerprint {name} is invalid")
    if _sha256_file(manifest_path) != manifest_file_sha256:
        raise LocalContactC0Error("manifest changed during case validation")
    if experimental_campaign is not None:
        from wave_asset_qa.contact.async_campaign import load_candidate_case
        load_candidate_case(case_dir,manifest=manifest,case=case,
                            source_revision=source_revision,source_tree=source_tree)
    return fresh_hash


def _write_status(
    path: Path,
    *,
    status: str,
    message: str | None,
    session_id: str,
    source_revision: str,
    source_tree: str,
    completed_case_count: int,
    unique_process_count: int,
    failed_case_id: str | None,
    experimental_campaign: str | None = None,
) -> None:
    write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "campaign": experimental_campaign or "contact_c0",
            "backend": Simulator.MUJOCO.value,
            "status": status,
            "message": message,
            "session_id": session_id,
            "source_revision": source_revision,
            "source_tree": source_tree,
            "expected_case_count": EXPECTED_CASE_COUNT,
            "completed_case_count": completed_case_count,
            "unique_process_count": unique_process_count,
            "failed_case_id": failed_case_id,
        },
    )


def _write_evidence_manifest(root: Path, *, experimental_campaign: str | None = None) -> Path:
    destination = root / "evidence-manifest.json"
    records = regular_tree_records(root, exclude=("evidence-manifest.json",))
    payload = {
        "schema_version": 1,
        "scope": (experimental_campaign+'-mujoco' if experimental_campaign else "contact-c0-mujoco"),
        "records": list(records),
        "root_sha256": inventory_root_sha256(records),
    }
    return write_json_exclusive(destination, payload)


def run_local_contact_c0(
    *,
    asset_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
    python_executable: str | Path = sys.executable,
    case_timeout_s: float = 900.0,
    experimental_campaign: str | None = None,
) -> Path:
    """Execute and independently validate all 16 local MuJoCo C0 cases."""

    if experimental_campaign not in (None, 'contact-c0-async-v1'):
        raise LocalContactC0Error('unrecognized experimental campaign')
    if _SESSION_RE.fullmatch(session_id) is None:
        raise LocalContactC0Error("session_id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise LocalContactC0Error(
            "source_revision must be a lowercase 40-character Git commit"
        )
    if (
        isinstance(case_timeout_s, bool)
        or not isinstance(case_timeout_s, (int, float))
        or not math.isfinite(float(case_timeout_s))
        or float(case_timeout_s) <= 0.0
    ):
        raise LocalContactC0Error("case_timeout_s must be positive and finite")

    project_root = Path(__file__).resolve(strict=True).parents[1]
    assets = _resolved_directory(asset_root, "asset root")
    manifest_file = _resolved_file(manifest_path, "manifest")
    expected_manifest_file = (
        project_root / "configs/parity/contact_c0.json"
    ).resolve(strict=True)
    if manifest_file != expected_manifest_file:
        raise LocalContactC0Error("formal local C0 requires the checked-in manifest")
    python = _resolved_file(python_executable, "Python executable")
    manifest = load_contact_manifest(manifest_file)
    cases = canonical_mujoco_cases(manifest)
    semantic_manifest_sha = contact_manifest_sha256(manifest)
    raw_manifest_sha = _sha256_file(manifest_file)
    runtime_identity = _local_runtime_identity(python)
    source_tree = _validate_source_identity(project_root, source_revision)
    asset_subtree, observed_asset_sha = _validate_asset_tree(assets, manifest)
    if _SHA256_RE.fullmatch(observed_asset_sha) is None:
        raise LocalContactC0Error("asset subtree hash is malformed")

    launcher_script = Path(__file__).resolve(strict=True)
    worker_relative = ('scripts/run_contact_async_worker.py' if experimental_campaign
                       else 'scripts/run_contact_c0_mujoco_worker.py')
    worker_arguments = ['--backend','mujoco'] if experimental_campaign else []
    worker_script = (project_root / worker_relative).resolve(
        strict=True
    )
    worker_module = (project_root / "src/wave_asset_qa/contact/worker.py").resolve(
        strict=True
    )
    frozen_hashes = {
        "launcher_script_sha256": _sha256_file(launcher_script),
        "worker_script_sha256": _sha256_file(worker_script),
        "worker_module_sha256": _sha256_file(worker_module),
        "manifest_file_sha256": raw_manifest_sha,
    }

    def validate_static_inputs() -> None:
        if _validate_source_identity(project_root, source_revision) != source_tree:
            raise LocalContactC0Error("source tree changed during the campaign")
        if _local_runtime_identity(python) != runtime_identity:
            raise LocalContactC0Error("local runtime changed during the campaign")
        _subtree, current_asset_sha = _validate_asset_tree(assets, manifest)
        if _subtree != asset_subtree or current_asset_sha != observed_asset_sha:
            raise LocalContactC0Error("asset subtree changed during the campaign")
        current_hashes = {
            "launcher_script_sha256": _sha256_file(launcher_script),
            "worker_script_sha256": _sha256_file(worker_script),
            "worker_module_sha256": _sha256_file(worker_module),
            "manifest_file_sha256": _sha256_file(manifest_file),
        }
        if current_hashes != frozen_hashes:
            raise LocalContactC0Error("formal executable input changed during campaign")

    destination = _new_output_path(output_dir)
    try:
        destination.relative_to(assets)
    except ValueError:
        pass
    else:
        raise LocalContactC0Error("output directory must not be inside asset root")
    destination.mkdir(mode=0o700)
    root = destination.resolve(strict=True)
    launcher_dir = root / "launcher"
    cases_dir = root / "cases"
    launcher_dir.mkdir()
    cases_dir.mkdir()

    launcher_identity: dict[str, object] = {
        "schema_version": 1,
        "campaign": experimental_campaign or "contact_c0",
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "manifest_id": manifest.manifest_id,
        "manifest_file_sha256": raw_manifest_sha,
        "manifest_semantic_sha256": semantic_manifest_sha,
        "manifest_canonical_json_sha256": sha256(
            canonical_contact_manifest_json(manifest).encode("utf-8")
        ).hexdigest(),
        "asset_repository": manifest.provenance.repository,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_subtree_relative_path": manifest.provenance.asset_root,
        "asset_tree_sha256": observed_asset_sha,
        "asset_tree_verification": "scanned_path_size_bytes",
        "local_runtime_identity": runtime_identity,
        **frozen_hashes,
        "fresh_process_per_case": True,
        "expected_case_count": EXPECTED_CASE_COUNT,
        "case_ids": [case.case_id for case in cases],
        "worker_command_template": [
            "<python>",
            "-P",
            worker_relative,
            *worker_arguments,
            "--asset-root",
            "<asset-root>",
            "--manifest",
            "configs/parity/contact_c0.json",
            "--case-id",
            "<case-id>",
            "--output-dir",
            "cases/<case-id>",
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--source-tree",
            source_tree,
            "--asset-tree-sha256",
            observed_asset_sha,
        ],
    }
    write_json_exclusive(launcher_dir / "provenance.json", launcher_identity)

    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    completed_ids: list[str] = []
    fresh_process_hashes: list[str] = []
    active_case_id: str | None = None
    try:
        for case in cases:
            active_case_id = case.case_id
            validate_static_inputs()
            case_dir = cases_dir / case.case_id
            case_dir.mkdir()
            command = [
                str(python),
                "-P",
                str(worker_script),
                *worker_arguments,
                "--asset-root",
                str(assets),
                "--manifest",
                str(manifest_file),
                "--case-id",
                case.case_id,
                "--output-dir",
                str(case_dir),
                "--session-id",
                session_id,
                "--source-revision",
                source_revision,
                "--source-tree",
                source_tree,
                "--asset-tree-sha256",
                observed_asset_sha,
            ]
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                outcome = _run_case_process(
                    command,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    environment=environment,
                    cwd=project_root,
                    private_process_path=case_dir / "private-process.json",
                    timeout_s=float(case_timeout_s),
                    evidence_root=root,
                    maximum_evidence_bytes=(
                        MAX_PRIVATE_ARTIFACT_BYTES - _FINAL_METADATA_RESERVE_BYTES
                    ),
                )
            if outcome.timed_out:
                with stderr_path.open("ab") as handle:
                    handle.write(
                        f"launcher timeout after {float(case_timeout_s):.6g}s\n".encode()
                    )
            _write_text_exclusive(
                case_dir / "exit-code.txt", f"{outcome.returncode}\n"
            )
            if outcome.returncode != 0:
                raise LocalContactC0Error(
                    f"{case.case_id} worker exited with code {outcome.returncode}; "
                    f"evidence retained at {case_dir}"
                )
            validate_static_inputs()
            fresh_hash = _validate_case_artifacts(
                case_dir,
                case,
                manifest=manifest,
                manifest_path=manifest_file,
                manifest_file_sha256=raw_manifest_sha,
                manifest_semantic_sha256=semantic_manifest_sha,
                asset_tree_sha256_value=observed_asset_sha,
                session_id=session_id,
                source_revision=source_revision,
                source_tree=source_tree,
                runtime_identity=runtime_identity,
                project_root=project_root,
                outcome=outcome,
                experimental_campaign=experimental_campaign,
            )
            if fresh_hash in fresh_process_hashes:
                raise LocalContactC0Error(
                    f"{case.case_id} reused a prior fresh-process identity hash"
                )
            fresh_process_hashes.append(fresh_hash)
            completed_ids.append(case.case_id)
            _enforce_artifact_budget(
                root,
                maximum_bytes=MAX_PRIVATE_ARTIFACT_BYTES
                - _FINAL_METADATA_RESERVE_BYTES,
            )

        active_case_id = None
        expected_ids = {case.case_id for case in cases}
        observed_dirs = {
            path.name
            for path in cases_dir.iterdir()
            if path.is_dir() and not is_link_like(path)
        }
        if observed_dirs != expected_ids or completed_ids != [
            case.case_id for case in cases
        ]:
            raise LocalContactC0Error("local evidence is not the exact 16-case matrix")
        if len(set(fresh_process_hashes)) != EXPECTED_CASE_COUNT:
            raise LocalContactC0Error("local evidence does not prove 16 fresh processes")
        validate_static_inputs()
        _enforce_artifact_budget(
            root,
            maximum_bytes=MAX_PRIVATE_ARTIFACT_BYTES - _FINAL_METADATA_RESERVE_BYTES,
        )
        _write_status(
            launcher_dir / "status.json",
            status="completed",
            message=None,
            session_id=session_id,
            source_revision=source_revision,
            source_tree=source_tree,
            completed_case_count=len(completed_ids),
            unique_process_count=len(set(fresh_process_hashes)),
            failed_case_id=None,
            experimental_campaign=experimental_campaign,
        )
        _write_evidence_manifest(root,experimental_campaign=experimental_campaign)
        _enforce_artifact_budget(root)
        return root
    except BaseException as error:
        status_path = launcher_dir / "status.json"
        if not status_path.exists() and not is_link_like(status_path):
            try:
                _write_status(
                    status_path,
                    status="error",
                    message=f"{type(error).__name__}: {error}",
                    session_id=session_id,
                    source_revision=source_revision,
                    source_tree=source_tree,
                    completed_case_count=len(completed_ids),
                    unique_process_count=len(set(fresh_process_hashes)),
                    failed_case_id=active_case_id,
                    experimental_campaign=experimental_campaign,
                )
            except BaseException:
                pass
        manifest_out = root / "evidence-manifest.json"
        if not manifest_out.exists() and not is_link_like(manifest_out):
            try:
                _write_evidence_manifest(root,experimental_campaign=experimental_campaign)
            except BaseException:
                pass
        raise


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
    parser.add_argument("--case-timeout-s", type=float, default=900.0)
    parser.add_argument('--experimental-campaign',choices=['contact-c0-async-v1'])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        root = run_local_contact_c0(
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
            python_executable=args.python_executable,
            case_timeout_s=args.case_timeout_s,
            experimental_campaign=args.experimental_campaign,
        )
    except (LocalContactC0Error, FileExistsError, OSError, ValueError) as error:
        print(f"local Contact C0 failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "backend": Simulator.MUJOCO.value,
                "case_count": EXPECTED_CASE_COUNT,
                "output_dir": str(root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
