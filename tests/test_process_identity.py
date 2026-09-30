from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from wave_asset_qa.parity import process_identity


WINDOWS_IDENTITY = {
    "platform": "windows",
    "pid": 42,
    "process_creation_filetime": 123456789,
}
POSIX_IDENTITY = {
    "platform": "posix",
    "boot_id": "12345678-1234-1234-1234-123456789abc",
    "pid": 42,
    "process_start_ticks": 98765,
}


@pytest.mark.parametrize("identity", [WINDOWS_IDENTITY, POSIX_IDENTITY])
def test_exact_identity_validation_and_canonical_hash(
    identity: dict[str, object],
) -> None:
    validated = process_identity.validate_os_process_identity(identity)
    expected = sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert validated == identity
    assert validated is not identity
    assert process_identity.os_process_identity_sha256(identity) == expected


@pytest.mark.parametrize(
    ("identity", "match"),
    [
        ({**WINDOWS_IDENTITY, "extra": 1}, "fields"),
        ({**WINDOWS_IDENTITY, "pid": True}, "PID"),
        ({**WINDOWS_IDENTITY, "pid": 0}, "PID"),
        ({**WINDOWS_IDENTITY, "process_creation_filetime": False}, "FILETIME"),
        ({**POSIX_IDENTITY, "boot_id": "not-a-boot-id"}, "POSIX"),
        ({**POSIX_IDENTITY, "process_start_ticks": 0}, "POSIX"),
    ],
)
def test_malformed_identity_is_rejected(
    identity: dict[str, object], match: str
) -> None:
    with pytest.raises(process_identity.ProcessIdentityError, match=match):
        process_identity.validate_os_process_identity(identity)


def test_current_posix_identity_parses_comm_with_spaces_and_parentheses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boot = tmp_path / "sys" / "kernel" / "random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text(
        str(POSIX_IDENTITY["boot_id"]) + "\n", encoding="ascii"
    )
    (tmp_path / "self").mkdir()
    fields = ["S", *(["0"] * 18), str(POSIX_IDENTITY["process_start_ticks"])]
    (tmp_path / "self" / "stat").write_text(
        "42 (python worker ) name) " + " ".join(fields) + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 42)

    assert process_identity.current_os_process_identity(
        platform_name="posix", proc_root=tmp_path
    ) == POSIX_IDENTITY


def test_current_windows_identity_uses_current_interpreter_kernel_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 42)
    monkeypatch.setattr(
        process_identity,
        "_windows_current_process_creation_filetime",
        lambda: WINDOWS_IDENTITY["process_creation_filetime"],
    )

    assert process_identity.current_os_process_identity(
        platform_name="nt"
    ) == WINDOWS_IDENTITY


def test_identity_validation_does_not_mutate_input() -> None:
    identity = deepcopy(POSIX_IDENTITY)
    before = deepcopy(identity)
    process_identity.validate_os_process_identity(identity)
    assert identity == before


def test_wait_for_exact_process_exit_rechecks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter([True, False])
    checked: list[dict[str, object]] = []

    def is_alive(identity: dict[str, object], *, proc_root: Path) -> bool:
        checked.append(dict(identity))
        return next(observations)

    monkeypatch.setattr(process_identity, "_os_process_identity_is_alive", is_alive)
    monkeypatch.setattr(process_identity.time, "sleep", lambda _seconds: None)

    process_identity.wait_for_os_process_exit(
        WINDOWS_IDENTITY, timeout_s=1.0, poll_interval_s=0.01
    )

    assert checked == [WINDOWS_IDENTITY, WINDOWS_IDENTITY]


def test_wait_for_exact_process_exit_fails_closed_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(
        process_identity,
        "_os_process_identity_is_alive",
        lambda _identity, *, proc_root: True,
    )
    monkeypatch.setattr(process_identity.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(process_identity.time, "sleep", lambda _seconds: None)

    with pytest.raises(process_identity.ProcessIdentityError, match="timed out"):
        process_identity.wait_for_os_process_exit(
            WINDOWS_IDENTITY, timeout_s=1.0, poll_interval_s=0.01
        )


def test_posix_pid_reuse_counts_as_original_worker_exit(tmp_path: Path) -> None:
    boot = tmp_path / "sys" / "kernel" / "random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text(
        str(POSIX_IDENTITY["boot_id"]) + "\n", encoding="ascii"
    )
    pid_dir = tmp_path / str(POSIX_IDENTITY["pid"])
    pid_dir.mkdir()
    fields = ["S", *(["0"] * 18), "99999"]
    (pid_dir / "stat").write_text(
        "42 (reused pid) " + " ".join(fields) + "\n", encoding="ascii"
    )

    assert (
        process_identity._posix_process_identity_is_alive(
            POSIX_IDENTITY, proc_root=tmp_path
        )
        is False
    )


def test_posix_zombie_is_an_exited_worker_before_parent_reaps(tmp_path: Path) -> None:
    boot = tmp_path / "sys" / "kernel" / "random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text(
        str(POSIX_IDENTITY["boot_id"]) + "\n", encoding="ascii"
    )
    pid_dir = tmp_path / str(POSIX_IDENTITY["pid"])
    pid_dir.mkdir()
    fields = [
        "Z",
        *(["0"] * 18),
        str(POSIX_IDENTITY["process_start_ticks"]),
    ]
    (pid_dir / "stat").write_text(
        "42 (exited worker) " + " ".join(fields) + "\n", encoding="ascii"
    )

    assert not process_identity._posix_process_identity_is_alive(
        POSIX_IDENTITY, proc_root=tmp_path
    )
    process_identity.wait_for_os_process_exit(
        POSIX_IDENTITY, timeout_s=0.1, proc_root=tmp_path
    )


def test_posix_termination_uses_identity_checked_pidfd_not_bare_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[tuple[int, int, object, int]] = []
    closed: list[int] = []
    monkeypatch.setattr(process_identity.select, "POLLIN", 1, raising=False)

    class ReadyPidfd:
        def register(self, descriptor: int, event: int) -> None:
            assert (descriptor, event) == (99, process_identity.select.POLLIN)

        def poll(self, timeout_ms: int) -> list[tuple[int, int]]:
            assert timeout_ms == 1000
            return [(99, process_identity.select.POLLIN)]

    monkeypatch.setattr(process_identity.os, "pidfd_open", lambda pid, flags: 99, raising=False)
    monkeypatch.setattr(
        process_identity.signal,
        "pidfd_send_signal",
        lambda descriptor, sig, siginfo, flags: signaled.append(
            (descriptor, sig, siginfo, flags)
        ),
        raising=False,
    )
    monkeypatch.setattr(process_identity.select, "poll", ReadyPidfd, raising=False)
    monkeypatch.setattr(process_identity.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        process_identity,
        "_posix_process_identity_is_alive",
        lambda identity, *, proc_root: True,
    )
    monkeypatch.setattr(
        process_identity,
        "wait_for_os_process_exit",
        lambda *_args, **_kwargs: pytest.fail(
            "pidfd poll is the exact exit proof before parent reaping"
        ),
    )

    process_identity.terminate_posix_process_identity(
        POSIX_IDENTITY, timeout_s=1.0
    )

    assert signaled == [(99, 9, None, 0)]
    assert closed == [99]


def test_posix_termination_never_signals_reused_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(process_identity.os, "pidfd_open", lambda pid, flags: 77, raising=False)
    monkeypatch.setattr(
        process_identity.signal,
        "pidfd_send_signal",
        lambda *_args: pytest.fail("reused PID owner must not be signaled"),
        raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        process_identity,
        "_posix_process_identity_is_alive",
        lambda identity, *, proc_root: False,
    )

    process_identity.terminate_posix_process_identity(POSIX_IDENTITY)

    assert closed == [77]
