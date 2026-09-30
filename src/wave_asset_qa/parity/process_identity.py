"""Exact, private OS-process identities for fresh-process evidence.

These identities are deliberately kernel-derived and machine-private.  They
must be removed from public summaries and reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import select
import signal
import time


_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class ProcessIdentityError(RuntimeError):
    """Raised when a kernel process identity cannot be established exactly."""


def _windows_current_process_creation_filetime() -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
    return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)


def _posix_process_start_ticks(stat_text: str) -> int:
    _prefix, separator, suffix = stat_text.rpartition(")")
    fields_after_comm = suffix.strip().split()
    if not separator or len(fields_after_comm) <= 19:
        raise ValueError("malformed /proc process stat")
    return int(fields_after_comm[19])


def validate_os_process_identity(value: object) -> dict[str, object]:
    """Return one exact JSON-safe identity or fail closed."""

    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ProcessIdentityError("OS process identity must be an object")
    platform_name = value.get("platform")
    expected = (
        {"platform", "pid", "process_creation_filetime"}
        if platform_name == "windows"
        else {"platform", "boot_id", "pid", "process_start_ticks"}
        if platform_name == "posix"
        else set()
    )
    if not expected or set(value) != expected:
        raise ProcessIdentityError("OS process identity fields are not exact")
    pid = value.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessIdentityError("OS process identity PID is invalid")
    if platform_name == "windows":
        creation = value.get("process_creation_filetime")
        if isinstance(creation, bool) or not isinstance(creation, int) or creation <= 0:
            raise ProcessIdentityError("Windows process creation FILETIME is invalid")
    else:
        boot_id = value.get("boot_id")
        ticks = value.get("process_start_ticks")
        if (
            not isinstance(boot_id, str)
            or _BOOT_ID_RE.fullmatch(boot_id) is None
            or isinstance(ticks, bool)
            or not isinstance(ticks, int)
            or ticks <= 0
        ):
            raise ProcessIdentityError("POSIX process identity is invalid")
    return dict(value)


def current_os_process_identity(
    *,
    platform_name: str | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, object]:
    """Read the current interpreter's kernel identity, never its launcher PID."""

    selected = os.name if platform_name is None else platform_name
    pid = os.getpid()
    try:
        if selected == "nt":
            identity: dict[str, object] = {
                "platform": "windows",
                "pid": pid,
                "process_creation_filetime": (
                    _windows_current_process_creation_filetime()
                ),
            }
        elif selected == "posix":
            identity = {
                "platform": "posix",
                "boot_id": (
                    proc_root / "sys/kernel/random/boot_id"
                ).read_text(encoding="ascii").strip(),
                "pid": pid,
                "process_start_ticks": _posix_process_start_ticks(
                    (proc_root / "self/stat").read_text(encoding="ascii")
                ),
            }
        else:
            raise ProcessIdentityError(
                f"unsupported process identity platform: {selected}"
            )
        return validate_os_process_identity(identity)
    except ProcessIdentityError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise ProcessIdentityError(
            "cannot establish current OS process identity"
        ) from error


def os_process_identity_sha256(value: object) -> str:
    """Hash the exact validated identity using canonical portable JSON."""

    identity = validate_os_process_identity(value)
    return sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _windows_process_identity_is_alive(identity: Mapping[str, object]) -> bool:
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    process_query_limited_information = 0x1000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        synchronize | process_query_limited_information,
        False,
        int(identity["pid"]),
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code in {87, 1168}:
            return False
        raise OSError(error_code, "OpenProcess failed")
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    try:
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        observed_creation = (
            (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        )
        if observed_creation != identity["process_creation_filetime"]:
            return False
        wait_result = int(kernel32.WaitForSingleObject(handle, 0))
        if wait_result == wait_object_0:
            return False
        if wait_result == wait_timeout:
            return True
        raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
    finally:
        kernel32.CloseHandle(handle)


def _posix_process_identity_is_alive(
    identity: Mapping[str, object], *, proc_root: Path
) -> bool:
    try:
        current_boot = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        if current_boot != identity["boot_id"]:
            return False
        stat_text = (proc_root / str(identity["pid"]) / "stat").read_text(
            encoding="ascii"
        )
    except FileNotFoundError:
        return False
    _prefix, separator, suffix = stat_text.rpartition(")")
    fields_after_comm = suffix.strip().split()
    if not separator or not fields_after_comm:
        raise ValueError("malformed /proc process stat")
    # A zombie has already exited; it merely awaits parent reaping.  Treating
    # state Z/X as alive would deadlock exact termination before Popen.wait().
    if fields_after_comm[0] in {"Z", "X", "x"}:
        return False
    return _posix_process_start_ticks(stat_text) == identity["process_start_ticks"]


def _os_process_identity_is_alive(
    identity: Mapping[str, object], *, proc_root: Path
) -> bool:
    if identity["platform"] == "windows":
        return _windows_process_identity_is_alive(identity)
    return _posix_process_identity_is_alive(identity, proc_root=proc_root)


def wait_for_os_process_exit(
    value: object,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.01,
    proc_root: Path = Path("/proc"),
) -> None:
    """Wait until the exact process exits; PID reuse counts as the old exit."""

    identity = validate_os_process_identity(value)
    if timeout_s <= 0.0 or poll_interval_s <= 0.0:
        raise ProcessIdentityError("process-exit wait bounds must be positive")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            alive = _os_process_identity_is_alive(identity, proc_root=proc_root)
        except (OSError, UnicodeError, ValueError) as error:
            raise ProcessIdentityError("cannot verify OS process exit") from error
        if not alive:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ProcessIdentityError("timed out waiting for OS process exit")
        time.sleep(min(poll_interval_s, remaining))


def terminate_windows_process_identity(
    value: object, *, timeout_s: float = 30.0
) -> None:
    """Terminate exactly one creation-time-bound Windows process."""

    identity = validate_os_process_identity(value)
    if identity["platform"] != "windows":
        raise ProcessIdentityError("exact Windows termination requires a Windows identity")
    if timeout_s <= 0.0:
        raise ProcessIdentityError("termination timeout must be positive")
    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    synchronize = 0x00100000
    process_query_limited_information = 0x1000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_terminate | synchronize | process_query_limited_information,
        False,
        int(identity["pid"]),
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code in {87, 1168}:
            return
        raise ProcessIdentityError(
            f"cannot open exact Windows worker for termination: {error_code}"
        )
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    try:
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ProcessIdentityError(
                "cannot read exact Windows worker creation time"
            )
        observed_creation = (
            (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        )
        if observed_creation != identity["process_creation_filetime"]:
            return
        if int(kernel32.WaitForSingleObject(handle, 0)) == wait_object_0:
            return
        if not kernel32.TerminateProcess(handle, 124):
            if int(kernel32.WaitForSingleObject(handle, 0)) == wait_object_0:
                return
            raise ProcessIdentityError("cannot terminate exact Windows worker")
        wait_result = int(
            kernel32.WaitForSingleObject(handle, max(1, round(timeout_s * 1000.0)))
        )
        if wait_result == wait_object_0:
            return
        if wait_result == wait_timeout:
            raise ProcessIdentityError(
                "timed out terminating exact Windows worker"
            )
        raise ProcessIdentityError("cannot wait for exact Windows worker termination")
    finally:
        kernel32.CloseHandle(handle)


def terminate_posix_process_identity(
    value: object, *, timeout_s: float = 30.0, proc_root: Path = Path("/proc")
) -> None:
    """Terminate one exact Linux process through a stable pidfd handle.

    Opening a pidfd first prevents a PID-reuse race.  The kernel identity is
    then checked against the process bound to that PID before any signal is
    sent.  A mismatch means the recorded worker has already exited and the
    current PID owner must not be touched.
    """

    identity = validate_os_process_identity(value)
    if identity["platform"] != "posix":
        raise ProcessIdentityError("exact POSIX termination requires a POSIX identity")
    if timeout_s <= 0.0:
        raise ProcessIdentityError("termination timeout must be positive")
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    sigkill = getattr(signal, "SIGKILL", 9)
    if pidfd_open is None or pidfd_send_signal is None:
        raise ProcessIdentityError(
            "safe POSIX worker termination requires pidfd support"
        )
    try:
        descriptor = pidfd_open(int(identity["pid"]), 0)
    except ProcessLookupError:
        return
    except OSError as error:
        raise ProcessIdentityError("cannot open exact POSIX worker pidfd") from error
    try:
        try:
            identity_is_alive = _posix_process_identity_is_alive(
                identity, proc_root=proc_root
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ProcessIdentityError(
                "cannot verify POSIX worker identity before termination"
            ) from error
        if not identity_is_alive:
            return
        try:
            pidfd_send_signal(descriptor, sigkill, None, 0)
        except ProcessLookupError:
            return
        except OSError as error:
            raise ProcessIdentityError(
                "cannot signal exact POSIX worker pidfd"
            ) from error
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        if not poller.poll(max(1, round(timeout_s * 1000.0))):
            raise ProcessIdentityError(
                "timed out terminating exact POSIX worker"
            )
    finally:
        os.close(descriptor)
    # pidfd poll readiness is the kernel's exact, reuse-safe exit proof.  The
    # parent launcher reaps its transport/direct child after this returns.
