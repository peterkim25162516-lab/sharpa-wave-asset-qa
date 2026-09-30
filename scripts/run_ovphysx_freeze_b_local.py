#!/usr/bin/env python3
"""Run the eight unchanged MuJoCo controls frozen by the Freeze B plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, BinaryIO, Mapping, Sequence

from wave_asset_qa.parity.bundle import sha256_file
from wave_asset_qa.parity.diagnostics import (
    create_staging_root,
    is_link_like,
    promote_staging_root,
)
from wave_asset_qa.parity.process_identity import (
    ProcessIdentityError,
    os_process_identity_sha256,
    terminate_posix_process_identity,
    terminate_windows_process_identity,
    validate_os_process_identity,
    wait_for_os_process_exit,
)
from wave_asset_qa.parity.runner import RUN_FILE_SUFFIX, load_collected_runs
from wave_asset_qa.parity.scenarios import (
    canonical_manifest_json,
    load_manifest,
    manifest_sha256,
)
from wave_asset_qa.parity.sensitivity import (
    load_private_plan,
    private_plan_sha256,
)


_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FRESH_PROCESS_RECORD_KEYS = {
    "schema_version",
    "experiment_case_id",
    "canonical_case_id",
    "os_process_identity",
    "fresh_process_id",
}
_WORKER_PROCESS_RECORD_KEYS = {
    "schema_version",
    "worker_pid",
    "os_process_identity",
    "fresh_process_id",
}
_ATTEMPT_OWNERSHIP_KEYS = {
    "schema_version",
    "ownership_token",
    "session_id",
    "source_revision",
}
_WORKER_WRAPPER_RELATIVE_PATH = Path("scripts/run_freeze_b_mujoco_worker.py")
_PROCESS_IDENTITY_RELATIVE_PATH = Path(
    "src/wave_asset_qa/parity/process_identity.py"
)
_PUBLIC_PROTOCOL_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b.json"
)
_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_corrigendum.json"
)
_ADMISSION_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_admission_corrigendum.json"
)
_ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH = Path(
    "configs/parity/ovphysx_legacy_friction_freeze_b_asset_preflight_corrigendum.json"
)
_PREREGISTRATION_REVISION = "608f83f1f93042d713346bd102de09b0b999b588"
_PREREGISTRATION_TREE = "1d1467d9f156d1115dbcb8ff54d522f0189454cc"
_PUBLIC_PROTOCOL_BLOB_OID = "c716ce9a4496d7e26384aa94b25437b8d5f0ce08"
_PUBLIC_PROTOCOL_FILE_SHA256 = (
    "9967098e45e063e2a4ca6e3db8644219f55dab808e99723c892fa72d53dc79c7"
)
_PUBLIC_PROTOCOL_CANONICAL_SHA256 = (
    "de2a97ac4d73d95f564f730a5655ddf26e9b01426eefff30c73135d932cdf66b"
)
_PRIVATE_PLAN_SHA256 = (
    "b4416b6869bdae90743ace253212ef43f7e592e30a93093f470c648261bab1ea"
)
_PRIVATE_PLAN_FILE_SHA256 = (
    "6017453e9e13a40f64cc53a4ad933944868b8da4540f646b2986e7138ae5c519"
)
_IMPLEMENTATION_TO_PREREGISTRATION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b.json"
]
_PREREGISTRATION_TO_CORRECTION_DIFF = [
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "A\tscripts/run_freeze_b_mujoco_worker.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "A\tsrc/wave_asset_qa/parity/process_identity.py",
    "M\ttests/test_freeze_b_finalizer.py",
    "A\ttests/test_freeze_b_mujoco_worker.py",
    "M\ttests/test_freeze_b_preparation_local.py",
    "A\ttests/test_process_identity.py",
]
_CORRECTION_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_corrigendum.json"
]
_FIRST_EXECUTION_REVISION = "8bb84691c82ab0582254d21953630725d64e9395"
_FIRST_EXECUTION_TREE = "6052d16ddb4f66c93046a170f44a575dfe4e385e"
_FIRST_CORRIGENDUM_BLOB_OID = "0ec18c5738aee6f0654ccbfcdb5105aeae24e4ed"
_FIRST_CORRIGENDUM_FILE_SHA256 = (
    "41eba7988668c7396b042d5785dfe1b627709f5df46ac77aa94eda77d0b5595b"
)
_FIRST_CORRIGENDUM_CANONICAL_SHA256 = (
    "ccf01aebaca218f50f9adce9250f150d7d6e00e88bca8bd42db694c55808eff5"
)
_FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF = [
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "M\tscripts/run_ovphysx_freeze_b_remote.sh",
    "M\ttests/test_freeze_b_finalizer.py",
    "M\ttests/test_freeze_b_preparation_local.py",
    "M\ttests/test_freeze_b_remote.py",
]
_ADMISSION_CORRECTION_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_admission_corrigendum.json"
]
_ADMISSION_CORRECTION_REVISION = "4156bd4cc0df996b6fab1be0e48fdbb04a65f393"
_ADMISSION_CORRECTION_TREE = "c5966f06ef32672f490e9c40746c15e92fa96761"
_SECOND_EXECUTION_REVISION = "554e59a45a0b9596ca70ad677ba2aba858002536"
_SECOND_EXECUTION_TREE = "0335c0e5046025fae27311b0aa1b84003d811939"
_ADMISSION_CORRIGENDUM_BLOB_OID = "a7e50ece608fb1aa3eadb2f88bc472204eed5d9f"
_ADMISSION_CORRIGENDUM_FILE_SHA256 = (
    "6651af497d3ff1451153e3f6f3b1a03c977ded9879e5fc6389fc1b3a41c7172e"
)
_ADMISSION_CORRIGENDUM_CANONICAL_SHA256 = (
    "fccba33068aaa6d06e8269be80a3c507b95a94384272803e59e61111293237ee"
)
_INCIDENT_SOURCE_TO_EXECUTION_DIFF = [
    "A\tconfigs/parity/ovphysx_legacy_friction_freeze_b_asset_preflight_corrigendum.json",
    "M\tscripts/finalize_ovphysx_freeze_b.py",
    "M\tscripts/run_ovphysx_freeze_b_local.py",
    "M\ttests/test_freeze_b_finalizer.py",
    "M\ttests/test_freeze_b_preparation_local.py",
]
_ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES = {
    "local-a1": {
        "classification": "invalid_local_worker_identity_collection",
        "entry_count": 13,
        "file_count": 10,
        "total_bytes": 1_282_435,
        "root_sha256": "fb661580d635ae84fd6bf5d7eaa33e838c96e13b8d1b5c242ea9fedc1be10498",
        "reuse_allowed": False,
    },
    "local-a2": {
        "classification": "excluded_structurally_valid_local_campaign_evidence",
        "entry_count": 82,
        "file_count": 72,
        "total_bytes": 14_852_439,
        "root_sha256": "998a75cf133b0baac0f1d5c2baf53469fa2d6d955bb3c277c7c621712accec1b",
        "reuse_allowed": False,
    },
    "remote-a2": {
        "classification": "invalid_remote_payload_admission_collection",
        "entry_count": 20,
        "file_count": 17,
        "total_bytes": 1_245_586,
        "root_sha256": "155fdebc6515aa49a7f5e0aafbacb629cf348829028b2da4fedcc5a1603d3abe",
        "reuse_allowed": False,
    },
}
_EXCLUDED_EVIDENCE_IDENTITIES = {
    **_ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES,
    "local-a3": {
        "classification": "failed_pre_simulation_asset_tree_preflight",
        "entry_count": 16,
        "file_count": 13,
        "total_bytes": 36_508,
        "root_sha256": "bff7ab75459f9ed7b8ab6b0bfd44997c7f544ec72f3ce8ba5644c0996a30eb79",
        "reuse_allowed": False,
    },
}


class FreezeBLocalError(RuntimeError):
    """Raised when local control evidence is incomplete or mutable."""


class FreezeBWorkerTerminationError(FreezeBLocalError):
    """Raised when a launched worker cannot be proven stopped."""


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeBLocalError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> object:
    raise FreezeBLocalError(f"non-finite JSON number is forbidden: {token}")


def _strict_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except FreezeBLocalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeBLocalError(f"cannot read worker process record: {error}") from error


def _attempt_ownership_record(
    *, ownership_token: str, session_id: str, source_revision: str
) -> dict[str, object]:
    if _HEX64.fullmatch(ownership_token) is None:
        raise FreezeBLocalError("attempt ownership token is invalid")
    return {
        "schema_version": 1,
        "ownership_token": ownership_token,
        "session_id": session_id,
        "source_revision": source_revision,
    }


def _attempt_root_is_owned(
    root: Path,
    *,
    ownership_token: str,
    session_id: str,
    source_revision: str,
) -> bool:
    if not root.is_dir() or is_link_like(root):
        return False
    ownership_path = root / "launcher" / "attempt-ownership.json"
    if not ownership_path.is_file() or is_link_like(ownership_path):
        return False
    try:
        value = _strict_json(ownership_path)
    except FreezeBLocalError:
        return False
    return (
        isinstance(value, Mapping)
        and set(value) == _ATTEMPT_OWNERSHIP_KEYS
        and value
        == _attempt_ownership_record(
            ownership_token=ownership_token,
            session_id=session_id,
            source_revision=source_revision,
        )
    )


def _remove_owned_initialization_staging(
    staging: Path,
    *,
    output: Path,
    ownership_token: str,
    session_id: str,
    source_revision: str,
) -> None:
    if not staging.exists():
        return
    expected_parent = output.parent.resolve(strict=True)
    if (
        staging.parent.resolve(strict=True) != expected_parent
        or not staging.name.startswith(f".{output.name}.incomplete-")
        or is_link_like(staging)
        or not _attempt_root_is_owned(
            staging,
            ownership_token=ownership_token,
            session_id=session_id,
            source_revision=source_revision,
        )
    ):
        raise FreezeBLocalError("refusing unsafe initialization staging cleanup")
    shutil.rmtree(staging)


def _validated_os_process_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    try:
        return validate_os_process_identity(identity)
    except ProcessIdentityError as error:
        raise FreezeBLocalError(str(error)) from error


def _fresh_process_id(identity: Mapping[str, object]) -> str:
    try:
        return os_process_identity_sha256(identity)
    except ProcessIdentityError as error:
        raise FreezeBLocalError(str(error)) from error


def _validated_worker_process_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _WORKER_PROCESS_RECORD_KEYS:
        raise FreezeBLocalError("worker process record fields are not exact")
    if value.get("schema_version") != 1:
        raise FreezeBLocalError("worker process record schema is invalid")
    identity = value.get("os_process_identity")
    if not isinstance(identity, Mapping):
        raise FreezeBLocalError("worker OS process identity is missing")
    validated_identity = _validated_os_process_identity(identity)
    worker_pid = value.get("worker_pid")
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid != validated_identity["pid"]
    ):
        raise FreezeBLocalError("worker process identity is not bound to worker_pid")
    fresh_id = value.get("fresh_process_id")
    if fresh_id != _fresh_process_id(validated_identity):
        raise FreezeBLocalError("worker fresh process hash is invalid")
    return {
        "schema_version": 1,
        "worker_pid": worker_pid,
        "os_process_identity": validated_identity,
        "fresh_process_id": fresh_id,
    }


def _await_worker_process_record(
    path: Path,
    child: subprocess.Popen[bytes],
    *,
    timeout_s: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while True:
        if path.is_symlink():
            raise FreezeBLocalError("worker process record must not be a symlink")
        if path.is_file():
            try:
                return _validated_worker_process_record(_strict_json(path))
            except FreezeBLocalError as error:
                # The exclusive worker file becomes visible before its fsync;
                # retry until the writer closes it or the handshake expires.
                last_error = error
        if child.poll() is not None:
            raise FreezeBLocalError(
                "actual Python worker exited before a valid identity handshake"
            ) from last_error
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise FreezeBLocalError(
                "timed out waiting for actual Python worker identity handshake"
            ) from last_error
        time.sleep(min(0.01, remaining))


def _terminate_owned_child(
    child: subprocess.Popen[bytes],
    *,
    worker_process_path: Path,
    platform_name: str | None = None,
) -> None:
    worker_process: dict[str, object] | None = None
    if worker_process_path.is_file() and not worker_process_path.is_symlink():
        try:
            worker_process = _validated_worker_process_record(
                _strict_json(worker_process_path)
            )
        except FreezeBLocalError:
            worker_process = None
    selected_platform = os.name if platform_name is None else platform_name
    try:
        if selected_platform == "nt" and worker_process is not None:
            identity = worker_process["os_process_identity"]
            terminate_windows_process_identity(identity, timeout_s=30.0)
            wait_for_os_process_exit(identity, timeout_s=30.0)
        elif selected_platform == "nt":
            # A Windows venv redirector PID is neither the actual worker
            # identity nor a safe tree handle.  Killing a bare PID after the
            # redirector exits can target an unrelated reused PID.  Without a
            # durable actual-worker identity, fail closed and leave the
            # attempt unsealed for manual process inspection.
            raise FreezeBWorkerTerminationError(
                "actual Windows worker identity handshake is unavailable; "
                "refusing PID-only termination"
            )
        elif selected_platform == "posix":
            if worker_process is not None:
                identity = worker_process["os_process_identity"]
                if identity.get("platform") != "posix":
                    raise FreezeBWorkerTerminationError(
                        "worker identity platform does not match local platform"
                    )
                terminate_posix_process_identity(identity, timeout_s=30.0)
            else:
                raise FreezeBWorkerTerminationError(
                    "actual POSIX worker identity handshake is unavailable; "
                    "refusing PID-only termination"
                )
        else:
            raise FreezeBWorkerTerminationError(
                f"unsupported local process platform: {selected_platform}"
            )
        if child.poll() is None:
            try:
                child.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                if selected_platform == "posix":
                    raise FreezeBWorkerTerminationError(
                        "POSIX transport did not exit after exact pidfd termination"
                    )
                child.kill()
                child.wait(timeout=30.0)
        if child.poll() is None:
            raise FreezeBWorkerTerminationError(
                "launcher transport process did not terminate"
            )
    except FreezeBWorkerTerminationError:
        raise
    except BaseException as error:
        raise FreezeBWorkerTerminationError(
            "cannot prove launched worker termination"
        ) from error


def _wait_for_recorded_worker_exit(
    worker_process: Mapping[str, object], *, timeout_s: float
) -> None:
    validated = _validated_worker_process_record(worker_process)
    try:
        wait_for_os_process_exit(
            validated["os_process_identity"], timeout_s=timeout_s
        )
    except ProcessIdentityError as error:
        raise FreezeBLocalError(
            "cannot prove actual Python worker exited before evidence admission"
        ) from error


def _run_case_worker(
    command: Sequence[str],
    *,
    project_root: Path,
    environment: Mapping[str, str],
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    worker_process_path: Path,
    case_timeout_s: float,
) -> tuple[int, dict[str, object] | None]:
    """Run one wrapper and prove the actual interpreter stopped before return."""

    child: subprocess.Popen[bytes] | None = None
    case_started = time.monotonic()
    try:
        child = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except BaseException as error:
        # Popen can fail after the OS process exists but before returning a
        # child object.  Exception type cannot prove that CreateProcess/exec
        # never succeeded, so every constructor failure is ownership-unknown.
        raise FreezeBWorkerTerminationError(
            "process creation failed before child ownership was proven"
        ) from error

    worker_process: dict[str, object] | None = None
    try:
        remaining_timeout = case_timeout_s - (time.monotonic() - case_started)
        if remaining_timeout <= 0.0:
            raise subprocess.TimeoutExpired(command, case_timeout_s)
        worker_process = _await_worker_process_record(
            worker_process_path,
            child,
            timeout_s=min(remaining_timeout, 30.0),
        )
        remaining_timeout = case_timeout_s - (time.monotonic() - case_started)
        if remaining_timeout <= 0.0:
            raise subprocess.TimeoutExpired(command, case_timeout_s)
        returncode = child.wait(timeout=remaining_timeout)
        remaining_timeout = case_timeout_s - (time.monotonic() - case_started)
        if remaining_timeout <= 0.0:
            raise subprocess.TimeoutExpired(command, case_timeout_s)
        _wait_for_recorded_worker_exit(
            worker_process, timeout_s=remaining_timeout
        )
        return returncode, worker_process
    except subprocess.TimeoutExpired:
        _terminate_owned_child(child, worker_process_path=worker_process_path)
        return 124, worker_process
    except BaseException:
        _terminate_owned_child(child, worker_process_path=worker_process_path)
        raise


def _fresh_process_record(
    *,
    experiment_case_id: str,
    canonical_case_id: str,
    worker_process: Mapping[str, object],
) -> dict[str, object]:
    validated_worker = _validated_worker_process_record(worker_process)
    return {
        "schema_version": 1,
        "experiment_case_id": experiment_case_id,
        "canonical_case_id": canonical_case_id,
        "os_process_identity": validated_worker["os_process_identity"],
        "fresh_process_id": validated_worker["fresh_process_id"],
    }


def _append_unique_fresh_process_record(
    records: list[dict[str, object]], record: Mapping[str, object]
) -> None:
    if set(record) != _FRESH_PROCESS_RECORD_KEYS:
        raise FreezeBLocalError("fresh process record fields are not exact")
    fresh_id = record["fresh_process_id"]
    if not isinstance(fresh_id, str) or _HEX64.fullmatch(fresh_id) is None:
        raise FreezeBLocalError("fresh process ID is invalid")
    if any(item["fresh_process_id"] == fresh_id for item in records):
        raise FreezeBLocalError("a local child OS process identity was reused")
    records.append(dict(record))


def _write_text_exclusive(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise FreezeBLocalError(f"refusing to overwrite evidence: {path.name}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _hash_tree(root: Path) -> None:
    destination = root / "evidence.sha256"
    members = sorted(
        (path for path in root.rglob("*") if path.is_file() and path != destination),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    _write_text_exclusive(
        destination,
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in members
        ),
    )


def _hash_tree_once(root: Path) -> None:
    if not (root / "evidence.sha256").exists():
        _hash_tree(root)


def _seal_failed_attempt(
    *,
    launcher_dir: Path,
    cases_dir: Path,
    identity: Mapping[str, object],
    active_experiment_case_id: str | None,
    completed_before_failure: Sequence[str],
    error: BaseException,
) -> None:
    for case_dir in sorted(cases_dir.iterdir(), key=lambda path: path.name):
        if case_dir.is_dir():
            _hash_tree_once(case_dir)
    _write_json_exclusive(
        launcher_dir / "error.json",
        {
            **identity,
            "terminal_status": "error",
            "attempt_admitted_case_count": 0,
            "completed_before_failure_count": len(completed_before_failure),
            "completed_before_failure_experiment_case_ids": list(
                completed_before_failure
            ),
            "active_experiment_case_id": active_experiment_case_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )
    _hash_tree_once(launcher_dir)


def _git_identity(project_root: Path, revision: str) -> str:
    if _HEX40.fullmatch(revision) is None:
        raise FreezeBLocalError("source revision must be a 40-character commit")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != revision:
        raise FreezeBLocalError("source revision is not HEAD")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise FreezeBLocalError("source worktree must be clean")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _HEX40.fullmatch(tree) is None:
        raise FreezeBLocalError("source tree identity is invalid")
    return tree


def _exact_mapping(
    value: object, expected_keys: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise FreezeBLocalError(f"{label} fields are not exact")
    return value


def _corrigendum_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
) -> dict[str, object]:
    data = _exact_mapping(
        value,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_preregistration",
            "incident",
            "correction",
            "correction_source",
            "source_transition_policy",
        },
        "corrigendum",
    )
    literals = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-evidence-corrigendum-v1"
        ),
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "state": (
            "POST_PREREGISTRATION_EVIDENCE_CORRECTION_BEFORE_ANY_ADMITTED_"
            "FREEZE_B_EVIDENCE"
        ),
    }
    if any(data.get(key) != expected for key, expected in literals.items()):
        raise FreezeBLocalError("corrigendum identity literals drifted")
    frozen = _exact_mapping(
        data.get("frozen_preregistration"),
        {
            "implementation_revision",
            "implementation_tree",
            "private_plan_sha256",
            "private_plan_file_sha256",
            "preregistration_revision",
            "preregistration_tree",
            "public_protocol_path",
            "public_protocol_blob_oid",
            "public_protocol_file_sha256",
            "public_protocol_canonical_sha256",
            "public_protocol_unchanged",
        },
        "corrigendum frozen_preregistration",
    )
    expected_frozen = {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "private_plan_sha256": _PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": _PRIVATE_PLAN_FILE_SHA256,
        "preregistration_revision": _PREREGISTRATION_REVISION,
        "preregistration_tree": _PREREGISTRATION_TREE,
        "public_protocol_path": _PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
        "public_protocol_blob_oid": _PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": _PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": _PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
    }
    if dict(frozen) != expected_frozen:
        raise FreezeBLocalError("corrigendum frozen preregistration drifted")
    incident = _exact_mapping(
        data.get("incident"),
        {
            "attempt_label",
            "attempt_classification",
            "raw_local_run_file_count",
            "admitted_evidence_case_count",
            "planned_evidence_case_count",
            "ovphysx_case_count_executed",
            "remote_or_gpu_execution_occurred",
            "terminal_completion_ledger_written",
            "failure_class",
            "detected_by",
            "trajectory_file_deserialized_for_structural_validation",
            "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction",
            "scientific_metrics_computed",
            "scientific_label_assigned",
            "reuse_allowed",
        },
        "corrigendum incident",
    )
    expected_incident = {
        "attempt_label": "local-a1",
        "attempt_classification": "invalid_collection_attempt",
        "raw_local_run_file_count": 1,
        "admitted_evidence_case_count": 0,
        "planned_evidence_case_count": 24,
        "ovphysx_case_count_executed": 0,
        "remote_or_gpu_execution_occurred": False,
        "terminal_completion_ledger_written": False,
        "failure_class": (
            "windows_venv_launcher_pid_was_not_actual_python_worker_pid"
        ),
        "detected_by": "fresh_process_worker_pid_binding_gate",
        "trajectory_file_deserialized_for_structural_validation": True,
        "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction": False,
        "scientific_metrics_computed": False,
        "scientific_label_assigned": False,
        "reuse_allowed": False,
    }
    if dict(incident) != expected_incident:
        raise FreezeBLocalError("corrigendum incident facts drifted")
    correction = _exact_mapping(
        data.get("correction"),
        {
            "scope",
            "worker_identity_authority",
            "worker_identity_capture",
            "launcher_record_relation",
            "finalizer_relation",
            "full_matrix_rerun_required",
            "failed_attempt_artifacts_must_remain_unmodified",
            "scientific_protocol_fields_changed",
            "adapter_changed",
            "scenario_or_manifest_changed",
            "control_or_target_changed",
            "dt_or_case_order_changed",
            "metrics_or_thresholds_changed",
            "formal_gate0_status_unchanged",
            "formal_gate0_pass_ready_unchanged",
        },
        "corrigendum correction",
    )
    expected_correction = {
        "scope": "evidence_plumbing_only",
        "worker_identity_authority": "actual_python_worker_kernel_identity_v1",
        "worker_identity_capture": "same_python_process_before_runner_main",
        "launcher_record_relation": (
            "exact_copy_of_worker_identity_with_recomputed_hash"
        ),
        "finalizer_relation": (
            "worker_sidecar_fresh_record_run_worker_pid_three_way_binding"
        ),
        "full_matrix_rerun_required": True,
        "failed_attempt_artifacts_must_remain_unmodified": True,
        "scientific_protocol_fields_changed": [],
        "adapter_changed": False,
        "scenario_or_manifest_changed": False,
        "control_or_target_changed": False,
        "dt_or_case_order_changed": False,
        "metrics_or_thresholds_changed": False,
        "formal_gate0_status_unchanged": "DIVERGENT",
        "formal_gate0_pass_ready_unchanged": False,
    }
    if dict(correction) != expected_correction:
        raise FreezeBLocalError("corrigendum correction scope drifted")
    correction_source = _exact_mapping(
        data.get("correction_source"),
        {"revision", "tree"},
        "corrigendum correction_source",
    )
    if any(
        not isinstance(correction_source.get(key), str)
        or _HEX40.fullmatch(str(correction_source[key])) is None
        for key in ("revision", "tree")
    ):
        raise FreezeBLocalError("corrigendum correction source is invalid")
    policy = _exact_mapping(
        data.get("source_transition_policy"),
        {
            "implementation_to_preregistration_name_status",
            "preregistration_to_correction_name_status",
            "correction_to_execution_name_status",
            "renames_allowed",
            "other_paths_allowed",
        },
        "corrigendum source_transition_policy",
    )
    expected_policy = {
        "implementation_to_preregistration_name_status": (
            _IMPLEMENTATION_TO_PREREGISTRATION_DIFF
        ),
        "preregistration_to_correction_name_status": (
            _PREREGISTRATION_TO_CORRECTION_DIFF
        ),
        "correction_to_execution_name_status": _CORRECTION_TO_EXECUTION_DIFF,
        "renames_allowed": False,
        "other_paths_allowed": False,
    }
    if dict(policy) != expected_policy:
        raise FreezeBLocalError("corrigendum source transition policy drifted")
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_admission_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
) -> dict[str, object]:
    data = _exact_mapping(
        value,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_history",
            "incident",
            "correction",
            "excluded_evidence",
            "correction_source",
            "source_transition_policy",
        },
        "admission corrigendum",
    )
    correction_source = _exact_mapping(
        data.get("correction_source"),
        {"revision", "tree"},
        "admission correction source",
    )
    if any(
        not isinstance(correction_source.get(key), str)
        or _HEX40.fullmatch(str(correction_source[key])) is None
        for key in ("revision", "tree")
    ):
        raise FreezeBLocalError("admission correction source is invalid")
    expected = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-remote-admission-corrigendum-v1"
        ),
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "state": (
            "POST_PREREGISTRATION_REMOTE_ADMISSION_CORRECTION_AFTER_ABORTED_A2_"
            "COLLECTION_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
        "frozen_history": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "preregistration_revision": _PREREGISTRATION_REVISION,
            "preregistration_tree": _PREREGISTRATION_TREE,
            "first_correction_revision": first_correction_revision,
            "first_correction_tree": first_correction_tree,
            "first_execution_revision": _FIRST_EXECUTION_REVISION,
            "first_execution_tree": _FIRST_EXECUTION_TREE,
            "private_plan_sha256": _PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": _PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": _PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": _PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": _PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": _PUBLIC_PROTOCOL_CANONICAL_SHA256,
            "public_protocol_unchanged": True,
            "first_corrigendum_path": _CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": _FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": _FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": _FIRST_CORRIGENDUM_CANONICAL_SHA256,
            "first_corrigendum_unchanged": True,
        },
        "incident": {
            "campaign_label": "a2",
            "attempt_classification": "aborted_mixed_local_remote_collection_campaign",
            "local_attempt_label": "local-a2",
            "remote_attempt_label": "remote-a2",
            "local_mujoco_case_count_executed": 8,
            "local_mujoco_case_count_automated_admission_validated": 8,
            "remote_ovphysx_case_count_executed": 1,
            "remote_worker_completed_case_count": 1,
            "remote_launcher_admitted_case_count": 0,
            "final_analysis_admitted_case_count": 0,
            "remote_or_gpu_execution_occurred": True,
            "terminal_success_completion_ledger_written": False,
            "failure_class": "post_worker_payload_validator_disabled_pinned_site_packages",
            "detected_by": "remote_payload_admission_gate",
            "root_cause": "python_no_site_flag_blocked_pinned_numpy_import",
            "automated_structural_or_numerical_validity_validation_occurred": True,
            "trajectory_sample_values_may_have_been_reviewed": True,
            "trajectory_sample_values_used_to_select_or_design_correction": False,
            "scientific_metrics_computed": False,
            "scientific_label_assigned": False,
            "reuse_allowed": False,
        },
        "correction": {
            "scientific_execution_change_scope": (
                "remote_post_worker_payload_admission_only"
            ),
            "supporting_provenance_and_finalizer_plumbing_changed": True,
            "validator_python_flags_before": ["-P", "-S", "-"],
            "validator_python_flags_after": ["-P", "-"],
            "pinned_environment_site_packages_enabled_after": True,
            "user_site_packages_remain_disabled": True,
            "worker_execution_command_changed": False,
            "worker_or_adapter_changed": False,
            "scenario_or_manifest_changed": False,
            "control_or_target_changed": False,
            "dt_or_case_order_changed": False,
            "metrics_or_thresholds_changed": False,
            "scientific_protocol_fields_changed": [],
            "full_24_case_matrix_rerun_required": True,
            "failed_attempt_artifacts_must_remain_unmodified": True,
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
        },
        "excluded_evidence": _ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES,
        "correction_source": dict(correction_source),
        "source_transition_policy": {
            "first_execution_to_admission_correction_name_status": (
                _FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF
            ),
            "admission_correction_to_execution_name_status": (
                _ADMISSION_CORRECTION_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }
    if dict(data) != expected:
        raise FreezeBLocalError("admission corrigendum facts or scope drifted")
    return json.loads(json.dumps(data, allow_nan=False))


def _validate_asset_preflight_corrigendum(
    value: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    """Validate the cumulative, pre-simulation local-a3 incident ledger."""

    data = _exact_mapping(
        value,
        {
            "schema_version",
            "corrigendum_id",
            "protocol_id",
            "state",
            "frozen_history",
            "incident",
            "correction",
            "excluded_evidence",
            "incident_source",
            "source_transition_policy",
        },
        "asset preflight corrigendum",
    )
    expected = {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-asset-preflight-"
            "corrigendum-v1"
        ),
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "state": (
            "POST_PREREGISTRATION_OPERATOR_ASSET_PREFLIGHT_INCIDENT_AFTER_"
            "ABORTED_LOCAL_A3_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
        "frozen_history": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "preregistration_revision": _PREREGISTRATION_REVISION,
            "preregistration_tree": _PREREGISTRATION_TREE,
            "first_correction_revision": first_correction_revision,
            "first_correction_tree": first_correction_tree,
            "first_execution_revision": _FIRST_EXECUTION_REVISION,
            "first_execution_tree": _FIRST_EXECUTION_TREE,
            "admission_correction_revision": admission_correction_revision,
            "admission_correction_tree": admission_correction_tree,
            "private_plan_sha256": _PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": _PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": _PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": _PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": _PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": _PUBLIC_PROTOCOL_CANONICAL_SHA256,
            "public_protocol_unchanged": True,
            "first_corrigendum_path": _CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": _FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": _FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": (
                _FIRST_CORRIGENDUM_CANONICAL_SHA256
            ),
            "first_corrigendum_unchanged": True,
            "admission_corrigendum_path": (
                _ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
            ),
            "admission_corrigendum_blob_oid": _ADMISSION_CORRIGENDUM_BLOB_OID,
            "admission_corrigendum_file_sha256": (
                _ADMISSION_CORRIGENDUM_FILE_SHA256
            ),
            "admission_corrigendum_canonical_sha256": (
                _ADMISSION_CORRIGENDUM_CANONICAL_SHA256
            ),
            "admission_corrigendum_unchanged": True,
        },
        "incident": {
            "attempt_label": "local-a3",
            "attempt_classification": "aborted_local_pre_simulation_attempt",
            "worker_process_count_started": 1,
            "worker_process_exit_verified": True,
            "runner_asset_preflight_entered": True,
            "adapter_instance_created": False,
            "mujoco_runtime_imported": False,
            "model_created": False,
            "trajectory_case_count_executed": 0,
            "simulator_step_count": 0,
            "run_payload_count": 0,
            "launcher_admitted_case_count": 0,
            "final_analysis_admitted_case_count": 0,
            "remote_or_gpu_execution_occurred": False,
            "terminal_success_completion_ledger_written": False,
            "failure_class": (
                "materialized_asset_tree_did_not_match_frozen_canonical_lf_hash"
            ),
            "detected_by": "runner_canonical_lf_asset_tree_sha256_preflight",
            "expected_canonical_lf_asset_tree_sha256": (
                "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
            ),
            "operator_disclosed_noncanonical_asset_materialization": True,
            "actual_materialized_asset_root_recorded_in_evidence": False,
            "actual_materialized_asset_tree_sha256_recorded_in_evidence": False,
            "trajectory_sample_values_existed": False,
            "trajectory_sample_values_reviewed_or_used_for_metrics_or_correction": (
                False
            ),
            "scientific_metrics_computed": False,
            "scientific_label_assigned": False,
            "reuse_allowed": False,
        },
        "correction": {
            "scientific_execution_change_scope": (
                "operator_asset_materialization_compliance_only"
            ),
            "operator_invocation_changed_to_match_frozen_asset_identity": True,
            "supporting_provenance_finalizer_and_local_launcher_plumbing_changed": (
                True
            ),
            "adapter_or_runner_scientific_behavior_changed": False,
            "scenario_or_manifest_changed": False,
            "control_or_target_changed": False,
            "dt_or_case_order_changed": False,
            "metrics_or_thresholds_changed": False,
            "scientific_protocol_fields_changed": [],
            "full_24_case_matrix_rerun_required": True,
            "failed_attempt_artifacts_must_remain_unmodified": True,
            "formal_gate0_status_unchanged": "DIVERGENT",
            "formal_gate0_pass_ready_unchanged": False,
        },
        "excluded_evidence": _EXCLUDED_EVIDENCE_IDENTITIES,
        "incident_source": {
            "revision": _SECOND_EXECUTION_REVISION,
            "tree": _SECOND_EXECUTION_TREE,
        },
        "source_transition_policy": {
            "incident_source_to_execution_name_status": (
                _INCIDENT_SOURCE_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }
    if dict(data) != expected:
        raise FreezeBLocalError("asset preflight corrigendum facts or scope drifted")
    return json.loads(json.dumps(data, allow_nan=False))


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise FreezeBLocalError(
            f"Git source-transition check failed: {' '.join(arguments)}"
        ) from error


def _git_name_status(
    project_root: Path, older: str, newer: str
) -> list[str]:
    output = _git_output(
        project_root,
        "diff",
        "--name-status",
        "--no-renames",
        older,
        newer,
        "--",
    )
    return output.splitlines() if output else []


def _verify_source_transition(
    project_root: Path,
    *,
    implementation_revision: str,
    implementation_tree: str,
    execution_revision: str,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
) -> dict[str, object]:
    """Prove exact A -> B -> C -> D -> E -> F -> G transitions."""

    if _HEX40.fullmatch(implementation_revision) is None or _HEX40.fullmatch(
        implementation_tree
    ) is None:
        raise FreezeBLocalError("private plan implementation identity is invalid")
    if implementation_revision == _PREREGISTRATION_REVISION:
        raise FreezeBLocalError("implementation and preregistration must be distinct")
    if corrigendum_path.relative_to(project_root) != _CORRIGENDUM_RELATIVE_PATH:
        raise FreezeBLocalError("corrigendum path is not canonical")
    if (
        admission_corrigendum_path.relative_to(project_root)
        != _ADMISSION_CORRIGENDUM_RELATIVE_PATH
    ):
        raise FreezeBLocalError("admission corrigendum path is not canonical")
    if (
        asset_preflight_corrigendum_path.relative_to(project_root)
        != _ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH
    ):
        raise FreezeBLocalError(
            "asset preflight corrigendum path is not canonical"
        )
    corrigendum = _validate_corrigendum(
        _strict_json(corrigendum_path),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
    )
    correction = corrigendum["correction_source"]
    assert isinstance(correction, Mapping)
    correction_revision = str(correction["revision"])
    correction_tree = str(correction["tree"])
    admission_corrigendum = _validate_admission_corrigendum(
        _strict_json(admission_corrigendum_path),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
    )
    admission_correction = admission_corrigendum["correction_source"]
    assert isinstance(admission_correction, Mapping)
    admission_correction_revision = str(admission_correction["revision"])
    admission_correction_tree = str(admission_correction["tree"])
    asset_preflight_corrigendum = _validate_asset_preflight_corrigendum(
        _strict_json(asset_preflight_corrigendum_path),
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        first_correction_revision=correction_revision,
        first_correction_tree=correction_tree,
        admission_correction_revision=admission_correction_revision,
        admission_correction_tree=admission_correction_tree,
    )
    identities = (
        implementation_revision,
        _PREREGISTRATION_REVISION,
        correction_revision,
        _FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        _SECOND_EXECUTION_REVISION,
        execution_revision,
    )
    if len(set(identities)) != 7:
        raise FreezeBLocalError("A, B, C, D, E, F, and G must be distinct")
    expected_parents = {
        _PREREGISTRATION_REVISION: implementation_revision,
        correction_revision: _PREREGISTRATION_REVISION,
        _FIRST_EXECUTION_REVISION: correction_revision,
        admission_correction_revision: _FIRST_EXECUTION_REVISION,
        _SECOND_EXECUTION_REVISION: admission_correction_revision,
        execution_revision: _SECOND_EXECUTION_REVISION,
    }
    for revision, expected_parent in expected_parents.items():
        ancestry_line = _git_output(
            project_root, "rev-list", "--parents", "-n", "1", revision
        ).split()
        if ancestry_line != [revision, expected_parent]:
            raise FreezeBLocalError(
                "A, B, C, D, E, F, and G must be direct single-parent commits"
            )
    expected_trees = {
        implementation_revision: implementation_tree,
        _PREREGISTRATION_REVISION: _PREREGISTRATION_TREE,
        correction_revision: correction_tree,
        _FIRST_EXECUTION_REVISION: _FIRST_EXECUTION_TREE,
        admission_correction_revision: admission_correction_tree,
        _SECOND_EXECUTION_REVISION: _SECOND_EXECUTION_TREE,
    }
    for revision, expected_tree in expected_trees.items():
        if _git_output(project_root, "rev-parse", f"{revision}^{{tree}}") != expected_tree:
            raise FreezeBLocalError("source commit/tree pair drifted")
    execution_tree = _git_output(project_root, "rev-parse", f"{execution_revision}^{{tree}}")
    if _HEX40.fullmatch(execution_tree) is None:
        raise FreezeBLocalError("execution tree is invalid")
    if expected_trees[implementation_revision] != implementation_tree:
        raise FreezeBLocalError("private plan implementation tree mismatch")
    for older, newer in zip(identities, identities[1:]):
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=project_root,
            check=False,
            capture_output=True,
        )
        if ancestry.returncode != 0:
            raise FreezeBLocalError("source transition ancestry is invalid")
    observed_diffs = (
        _git_name_status(
            project_root, implementation_revision, _PREREGISTRATION_REVISION
        ),
        _git_name_status(
            project_root, _PREREGISTRATION_REVISION, correction_revision
        ),
        _git_name_status(
            project_root, correction_revision, _FIRST_EXECUTION_REVISION
        ),
        _git_name_status(
            project_root, _FIRST_EXECUTION_REVISION, admission_correction_revision
        ),
        _git_name_status(
            project_root, admission_correction_revision, _SECOND_EXECUTION_REVISION
        ),
        _git_name_status(
            project_root, _SECOND_EXECUTION_REVISION, execution_revision
        ),
    )
    expected_diffs = (
        _IMPLEMENTATION_TO_PREREGISTRATION_DIFF,
        _PREREGISTRATION_TO_CORRECTION_DIFF,
        _CORRECTION_TO_EXECUTION_DIFF,
        _FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF,
        _ADMISSION_CORRECTION_TO_EXECUTION_DIFF,
        _INCIDENT_SOURCE_TO_EXECUTION_DIFF,
    )
    if observed_diffs != expected_diffs:
        raise FreezeBLocalError("source transition differs from the exact corrigendum")
    protocol_path = _PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix()
    for revision in (
        _PREREGISTRATION_REVISION,
        correction_revision,
        _FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        _SECOND_EXECUTION_REVISION,
        execution_revision,
    ):
        if _git_output(project_root, "rev-parse", f"{revision}:{protocol_path}") != _PUBLIC_PROTOCOL_BLOB_OID:
            raise FreezeBLocalError("frozen public protocol blob changed after B")
    if sha256_file(project_root / _PUBLIC_PROTOCOL_RELATIVE_PATH) != _PUBLIC_PROTOCOL_FILE_SHA256:
        raise FreezeBLocalError("working-tree public protocol bytes drifted")
    corrigendum_relative = _CORRIGENDUM_RELATIVE_PATH.as_posix()
    for revision in (
        _FIRST_EXECUTION_REVISION,
        admission_correction_revision,
        _SECOND_EXECUTION_REVISION,
        execution_revision,
    ):
        if (
            _git_output(
                project_root, "rev-parse", f"{revision}:{corrigendum_relative}"
            )
            != _FIRST_CORRIGENDUM_BLOB_OID
        ):
            raise FreezeBLocalError("first corrigendum changed after D")
    if (
        _git_output(project_root, "hash-object", corrigendum_relative)
        != _FIRST_CORRIGENDUM_BLOB_OID
    ):
        raise FreezeBLocalError("corrigendum bytes differ from execution commit")
    admission_relative = _ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix()
    for revision in (_SECOND_EXECUTION_REVISION, execution_revision):
        if (
            _git_output(
                project_root, "rev-parse", f"{revision}:{admission_relative}"
            )
            != _ADMISSION_CORRIGENDUM_BLOB_OID
        ):
            raise FreezeBLocalError("admission corrigendum changed after F")
    if (
        _git_output(project_root, "hash-object", admission_relative)
        != _ADMISSION_CORRIGENDUM_BLOB_OID
    ):
        raise FreezeBLocalError(
            "admission corrigendum bytes differ from execution commit"
        )
    asset_preflight_relative = (
        _ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH.as_posix()
    )
    if _git_output(
        project_root,
        "rev-parse",
        f"{execution_revision}:{asset_preflight_relative}",
    ) != _git_output(project_root, "hash-object", asset_preflight_relative):
        raise FreezeBLocalError(
            "asset preflight corrigendum bytes differ from execution commit"
        )
    return {
        "preregistration_revision": _PREREGISTRATION_REVISION,
        "preregistration_tree": _PREREGISTRATION_TREE,
        "correction_revision": correction_revision,
        "correction_tree": correction_tree,
        "first_execution_revision": _FIRST_EXECUTION_REVISION,
        "first_execution_tree": _FIRST_EXECUTION_TREE,
        "admission_correction_revision": admission_correction_revision,
        "admission_correction_tree": admission_correction_tree,
        "incident_source_revision": _SECOND_EXECUTION_REVISION,
        "incident_source_tree": _SECOND_EXECUTION_TREE,
        "execution_revision": execution_revision,
        "execution_tree": execution_tree,
        "corrigendum_file_sha256": sha256_file(corrigendum_path),
        "corrigendum_canonical_sha256": _corrigendum_sha256(corrigendum),
        "admission_corrigendum_file_sha256": sha256_file(
            admission_corrigendum_path
        ),
        "admission_corrigendum_canonical_sha256": _corrigendum_sha256(
            admission_corrigendum
        ),
        "asset_preflight_corrigendum_file_sha256": sha256_file(
            asset_preflight_corrigendum_path
        ),
        "asset_preflight_corrigendum_canonical_sha256": _corrigendum_sha256(
            asset_preflight_corrigendum
        ),
    }


def _direct_results_child(project_root: Path, output: Path) -> Path:
    results = (project_root / "results").resolve(strict=True)
    candidate = Path(os.path.abspath(output.expanduser()))
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise FreezeBLocalError(
            "output must be a direct child of project results/"
        ) from error
    if parent != results:
        raise FreezeBLocalError("output must be a direct child of project results/")
    if candidate.exists() or candidate.is_symlink():
        raise FreezeBLocalError("output already exists")
    return candidate


def _canonical_case_ids(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = plan.get("mujoco_cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise FreezeBLocalError("private plan must contain exactly eight MuJoCo cases")
    return cases


def _validate_run_file(
    case_dir: Path,
    row: Mapping[str, Any],
    process_record: Mapping[str, object],
) -> None:
    canonical_id = str(row["canonical_case_id"])
    expected = case_dir / f"{canonical_id}{RUN_FILE_SUFFIX}"
    files = tuple(sorted(case_dir.glob(f"*{RUN_FILE_SUFFIX}")))
    if files != (expected,):
        raise FreezeBLocalError(f"{row['experiment_case_id']} produced an invalid run inventory")
    runs = load_collected_runs(expected)
    if len(runs) != 1 or runs[0].case.case_id != canonical_id:
        raise FreezeBLocalError(f"{row['experiment_case_id']} run identity mismatch")
    run = runs[0]
    if not run.result.completed:
        raise FreezeBLocalError(f"{row['experiment_case_id']} did not complete")
    if run.bundle_root_sha256 is not None:
        raise FreezeBLocalError("raw local run must not reference a parent bundle")
    if run.result.backend != "mujoco" or run.result.scenario_id != "small_step":
        raise FreezeBLocalError("local Freeze B run used an unexpected backend/scenario")
    if not math.isclose(run.result.dt, float(row["dt_s"]), rel_tol=0.0, abs_tol=1e-12):
        raise FreezeBLocalError("local Freeze B run used an unexpected dt")
    process_identity = process_record.get("os_process_identity")
    if not isinstance(process_identity, Mapping):
        raise FreezeBLocalError("local child OS process identity is missing")
    process_pid = process_identity.get("pid")
    worker_pid = run.result.provenance.get("worker_pid")
    if (
        isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid <= 0
        or isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid != process_pid
    ):
        raise FreezeBLocalError(
            "local child OS process identity is not bound to run worker_pid"
        )


def run_local_controls(
    *,
    project_root: Path,
    asset_root: Path,
    manifest_path: Path,
    private_plan_path: Path,
    corrigendum_path: Path,
    admission_corrigendum_path: Path,
    asset_preflight_corrigendum_path: Path,
    expected_plan_sha256: str,
    output_dir: Path,
    session_id: str,
    source_revision: str,
    python_executable: Path,
    case_timeout_s: float,
) -> Path:
    if _SESSION_RE.fullmatch(session_id) is None:
        raise FreezeBLocalError("session_id is invalid")
    if _HEX64.fullmatch(expected_plan_sha256) is None:
        raise FreezeBLocalError("private plan SHA-256 is invalid")
    if not math.isfinite(case_timeout_s) or case_timeout_s <= 0.0:
        raise FreezeBLocalError("case timeout must be positive and finite")
    project_root = project_root.resolve(strict=True)
    asset_root = asset_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    private_plan_path = private_plan_path.resolve(strict=True)
    corrigendum_path = corrigendum_path.resolve(strict=True)
    admission_corrigendum_path = admission_corrigendum_path.resolve(strict=True)
    asset_preflight_corrigendum_path = (
        asset_preflight_corrigendum_path.resolve(strict=True)
    )
    python_executable = python_executable.resolve(strict=True)
    if not asset_root.is_dir() or not python_executable.is_file():
        raise FreezeBLocalError("asset root or Python executable is unavailable")
    output = _direct_results_child(project_root, output_dir)
    source_tree = _git_identity(project_root, source_revision)
    plan = load_private_plan(private_plan_path)
    if private_plan_sha256(plan) != expected_plan_sha256:
        raise FreezeBLocalError("private plan hash mismatch")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise FreezeBLocalError("private plan inputs are invalid")
    implementation_revision = str(inputs.get("freeze_b_source_revision", ""))
    implementation_tree = str(inputs.get("freeze_b_source_tree", ""))
    source_transition = _verify_source_transition(
        project_root,
        implementation_revision=implementation_revision,
        implementation_tree=implementation_tree,
        execution_revision=source_revision,
        corrigendum_path=corrigendum_path,
        admission_corrigendum_path=admission_corrigendum_path,
        asset_preflight_corrigendum_path=asset_preflight_corrigendum_path,
    )
    if source_transition["execution_tree"] != source_tree:
        raise FreezeBLocalError("execution source tree changed during validation")
    manifest = load_manifest(manifest_path)
    if inputs.get("gate0_manifest_file_sha256") != sha256_file(manifest_path):
        raise FreezeBLocalError("private plan manifest file hash mismatch")
    if inputs.get("gate0_manifest_semantic_sha256") != manifest_sha256(manifest):
        raise FreezeBLocalError("private plan manifest semantic hash mismatch")

    launcher_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    worker_wrapper_path = (project_root / _WORKER_WRAPPER_RELATIVE_PATH).resolve(
        strict=True
    )
    process_identity_path = (
        project_root / _PROCESS_IDENTITY_RELATIVE_PATH
    ).resolve(strict=True)
    worker_wrapper_sha256 = sha256_file(worker_wrapper_path)
    process_identity_source_sha256 = sha256_file(process_identity_path)
    plan_file_sha256 = sha256_file(private_plan_path)
    cases = _canonical_case_ids(plan)
    identity = {
        "schema_version": 1,
        "protocol_id": plan["protocol_id"],
        "backend": "mujoco",
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "implementation_source_revision": implementation_revision,
        "implementation_source_tree": implementation_tree,
        "preregistration_source_revision": source_transition[
            "preregistration_revision"
        ],
        "preregistration_source_tree": source_transition["preregistration_tree"],
        "correction_source_revision": source_transition["correction_revision"],
        "correction_source_tree": source_transition["correction_tree"],
        "first_execution_source_revision": source_transition[
            "first_execution_revision"
        ],
        "first_execution_source_tree": source_transition["first_execution_tree"],
        "admission_correction_source_revision": source_transition[
            "admission_correction_revision"
        ],
        "admission_correction_source_tree": source_transition[
            "admission_correction_tree"
        ],
        "asset_preflight_incident_source_revision": source_transition[
            "incident_source_revision"
        ],
        "asset_preflight_incident_source_tree": source_transition[
            "incident_source_tree"
        ],
        "corrigendum_file_sha256": source_transition["corrigendum_file_sha256"],
        "corrigendum_canonical_sha256": source_transition[
            "corrigendum_canonical_sha256"
        ],
        "admission_corrigendum_file_sha256": source_transition[
            "admission_corrigendum_file_sha256"
        ],
        "admission_corrigendum_canonical_sha256": source_transition[
            "admission_corrigendum_canonical_sha256"
        ],
        "asset_preflight_corrigendum_file_sha256": source_transition[
            "asset_preflight_corrigendum_file_sha256"
        ],
        "asset_preflight_corrigendum_canonical_sha256": source_transition[
            "asset_preflight_corrigendum_canonical_sha256"
        ],
        "launcher_sha256": launcher_sha256,
        "worker_wrapper_sha256": worker_wrapper_sha256,
        "process_identity_source_sha256": process_identity_source_sha256,
        "fresh_process_identity_authority": (
            "actual_python_worker_kernel_identity_v1"
        ),
        "private_plan_sha256": expected_plan_sha256,
        "private_plan_file_sha256": plan_file_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_semantic_sha256": manifest_sha256(manifest),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
    }
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
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WAVEQA_SESSION_ID"] = session_id
    environment["WAVEQA_SOURCE_TREE"] = source_revision

    ownership_token = secrets.token_hex(32)
    initialization_staging: Path | None = None
    completed: list[str] = []
    fresh_process_records: list[dict[str, object]] = []
    active_experiment_id: str | None = None
    try:
        # Build the complete no-trajectory attempt skeleton under a unique
        # sibling, then atomically claim the fixed run ID.  The durable random
        # ownership record lets exception handling distinguish our promoted
        # directory even if interruption lands inside/after os.rename.
        initialization_staging = create_staging_root(output)
        staged_launcher = initialization_staging / "launcher"
        staged_cases = initialization_staging / "cases"
        staged_launcher.mkdir(mode=0o700)
        _write_json_exclusive(
            staged_launcher / "attempt-ownership.json",
            _attempt_ownership_record(
                ownership_token=ownership_token,
                session_id=session_id,
                source_revision=source_revision,
            ),
        )
        staged_cases.mkdir(mode=0o700)
        _write_text_exclusive(
            staged_launcher / "gate0.manifest.json",
            canonical_manifest_json(manifest) + "\n",
        )
        # This output is ignored/private.  Keeping the exact plan beside the
        # runs lets finalization verify the matrix byte-for-byte.
        _write_text_exclusive(
            staged_launcher / "freeze_b.private-plan.json",
            private_plan_path.read_text(encoding="utf-8"),
        )
        _write_json_exclusive(staged_launcher / "provenance.json", identity)
        _write_json_exclusive(
            staged_launcher / "matrix.json",
            {
                **identity,
                "case_count": 8,
                "experiment_case_ids": [
                    str(row["experiment_case_id"]) for row in cases
                ],
                "canonical_case_ids": [
                    str(row["canonical_case_id"]) for row in cases
                ],
            },
        )
        promoted = promote_staging_root(initialization_staging, output)
        if promoted != output:
            raise FreezeBLocalError("attempt output promotion escaped the fixed run ID")
        launcher_dir = output / "launcher"
        cases_dir = output / "cases"
        manifest_snapshot = launcher_dir / "gate0.manifest.json"
        plan_snapshot = launcher_dir / "freeze_b.private-plan.json"
        initialization_staging = None

        for row in cases:
            if _git_identity(project_root, source_revision) != source_tree:
                raise FreezeBLocalError("source identity changed during local controls")
            experiment_id = str(row["experiment_case_id"])
            active_experiment_id = experiment_id
            canonical_id = str(row["canonical_case_id"])
            case_dir = cases_dir / experiment_id
            case_dir.mkdir(mode=0o700)
            worker_process_path = case_dir / "worker-process.json"
            command = [
                str(python_executable),
                "-P",
                str(worker_wrapper_path),
                "--identity-output",
                str(worker_process_path),
                "--asset-root",
                str(asset_root),
                "--manifest",
                str(manifest_snapshot),
                "--output-dir",
                str(case_dir),
                "--session-id",
                session_id,
                "--source-revision",
                source_revision,
                "--case-id",
                canonical_id,
            ]
            _write_json_exclusive(
                case_dir / "launcher.command.json",
                {
                    "schema_version": 1,
                    "experiment_case_id": experiment_id,
                    "canonical_case_id": canonical_id,
                    "argv": [
                        "<python>",
                        "-P",
                        _WORKER_WRAPPER_RELATIVE_PATH.as_posix(),
                        "--identity-output",
                        f"cases/{experiment_id}/worker-process.json",
                        "--asset-root",
                        "<asset-root>",
                        "--manifest",
                        "launcher/gate0.manifest.json",
                        "--output-dir",
                        f"cases/{experiment_id}",
                        "--session-id",
                        session_id,
                        "--source-revision",
                        source_revision,
                        "--case-id",
                        canonical_id,
                    ],
                },
            )
            stdout_path = case_dir / "launcher.stdout.log"
            stderr_path = case_dir / "launcher.stderr.log"
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                returncode, worker_process = _run_case_worker(
                    command,
                    project_root=project_root,
                    environment=environment,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    worker_process_path=worker_process_path,
                    case_timeout_s=case_timeout_s,
                )
            _write_text_exclusive(
                case_dir / "launcher.exitcode.txt", f"{returncode}\n"
            )
            if returncode != 0:
                raise FreezeBLocalError(
                    f"{experiment_id} exited with {returncode}; evidence retained"
                )
            if worker_process is None:
                raise FreezeBLocalError("worker identity handshake was not retained")
            process_record = _fresh_process_record(
                experiment_case_id=experiment_id,
                canonical_case_id=canonical_id,
                worker_process=worker_process,
            )
            _append_unique_fresh_process_record(
                fresh_process_records, process_record
            )
            _write_json_exclusive(
                case_dir / "fresh-process.json", process_record
            )
            _validate_run_file(case_dir, row, process_record)
            _hash_tree(case_dir)
            completed.append(experiment_id)
            active_experiment_id = None

        if completed != [str(row["experiment_case_id"]) for row in cases]:
            raise FreezeBLocalError("local completed case order is not the frozen matrix")
        if len(fresh_process_records) != 8:
            raise FreezeBLocalError("local fresh-process evidence must contain eight records")
        fresh_process_ids = [
            str(record["fresh_process_id"]) for record in fresh_process_records
        ]
        _write_json_exclusive(
            launcher_dir / "fresh-process-ledger.json",
            {
                "schema_version": 1,
                "case_count": 8,
                "records": fresh_process_records,
            },
        )
        _write_json_exclusive(
            launcher_dir / "completed.json",
            {
                **identity,
                "completed_case_count": 8,
                "experiment_case_ids": completed,
                "fresh_process_ids": fresh_process_ids,
            },
        )
        _hash_tree(launcher_dir)
    except BaseException as error:
        if initialization_staging is not None and initialization_staging.exists():
            try:
                _remove_owned_initialization_staging(
                    initialization_staging,
                    output=output,
                    ownership_token=ownership_token,
                    session_id=session_id,
                    source_revision=source_revision,
                )
            except BaseException as cleanup_error:
                error_class = (
                    FreezeBWorkerTerminationError
                    if isinstance(error, FreezeBWorkerTerminationError)
                    else FreezeBLocalError
                )
                raise error_class(
                    "local attempt failed and initialization staging cleanup "
                    f"also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                ) from error
        if isinstance(error, FreezeBWorkerTerminationError):
            # Do not publish a terminal hash inventory while a worker may
            # still be mutating the attempt directory.
            raise
        owned_output = _attempt_root_is_owned(
            output,
            ownership_token=ownership_token,
            session_id=session_id,
            source_revision=source_revision,
        )
        if not owned_output:
            if output.exists() or is_link_like(output):
                raise FreezeBLocalError(
                    "fixed output appeared without this attempt's ownership proof; "
                    "refusing to modify it"
                ) from error
            # No fixed run ID was claimed and no trajectory could have started;
            # the caller may safely retry the same ID after fixing the input.
            raise
        launcher_dir = output / "launcher"
        cases_dir = output / "cases"
        try:
            _seal_failed_attempt(
                launcher_dir=launcher_dir,
                cases_dir=cases_dir,
                identity=identity,
                active_experiment_case_id=active_experiment_id,
                completed_before_failure=completed,
                error=error,
            )
        except BaseException as sealing_error:
            raise FreezeBLocalError(
                "local attempt failed and terminal evidence sealing also failed: "
                f"{type(sealing_error).__name__}: {sealing_error}"
            ) from error
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--corrigendum", type=Path, required=True)
    parser.add_argument("--admission-corrigendum", type=Path, required=True)
    parser.add_argument("--asset-preflight-corrigendum", type=Path, required=True)
    parser.add_argument("--private-plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--case-timeout-s", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = run_local_controls(
            project_root=args.project_root,
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            private_plan_path=args.private_plan,
            corrigendum_path=args.corrigendum,
            admission_corrigendum_path=args.admission_corrigendum,
            asset_preflight_corrigendum_path=args.asset_preflight_corrigendum,
            expected_plan_sha256=args.private_plan_sha256,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
            python_executable=args.python_executable,
            case_timeout_s=args.case_timeout_s,
        )
    except Exception as error:
        print(f"Freeze B local controls failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "completed", "backend": "mujoco", "case_count": 8, "output_dir": root.name},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
