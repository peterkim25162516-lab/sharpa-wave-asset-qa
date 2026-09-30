from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from wave_asset_qa.parity.process_identity import os_process_identity_sha256


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_IDENTITY = {
    "platform": "windows",
    "pid": 4242,
    "process_creation_filetime": 123456789,
}


def _module() -> object:
    path = ROOT / "scripts" / "run_freeze_b_mujoco_worker.py"
    spec = importlib.util.spec_from_file_location("freeze_b_mujoco_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(output: Path, identity_output: Path) -> list[str]:
    return [
        "--identity-output",
        str(identity_output),
        "--asset-root",
        str(output.parent / "assets"),
        "--manifest",
        str(output.parent / "manifest.json"),
        "--output-dir",
        str(output),
        "--session-id",
        "fixture-session",
        "--source-revision",
        "a" * 40,
        "--case-id",
        "mujoco.left.small_step.base.r01",
    ]


def test_worker_records_actual_interpreter_identity_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "case"
    output.mkdir()
    identity_output = output / "worker-process.json"
    captured: list[list[str]] = []
    monkeypatch.setattr(module, "current_os_process_identity", lambda: WINDOWS_IDENTITY)
    monkeypatch.setattr(module.os, "getpid", lambda: 4242)
    monkeypatch.setattr(
        module,
        "_run_runner",
        lambda argv: captured.append(list(argv)) or 0,
    )

    assert module.main(_argv(output, identity_output)) == 0

    payload = json.loads(identity_output.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "worker_pid": 4242,
        "os_process_identity": WINDOWS_IDENTITY,
        "fresh_process_id": os_process_identity_sha256(WINDOWS_IDENTITY),
    }
    assert captured == [
        [
            "--backend",
            "mujoco",
            "--asset-root",
            str(tmp_path / "assets"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(output.resolve()),
            "--session-id",
            "fixture-session",
            "--source-revision",
            "a" * 40,
            "--case-id",
            "mujoco.left.small_step.base.r01",
        ]
    ]


def test_worker_rejects_identity_outside_case_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "case"
    output.mkdir()
    called = False

    def fake_runner(argv: list[str]) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(module, "_run_runner", fake_runner)
    with pytest.raises(module.FreezeBMuJoCoWorkerError, match="case output"):
        module.main(_argv(output, tmp_path / "elsewhere.json"))
    assert called is False


def test_worker_refuses_to_overwrite_identity_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "case"
    output.mkdir()
    identity_output = output / "worker-process.json"
    identity_output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(module, "current_os_process_identity", lambda: WINDOWS_IDENTITY)
    monkeypatch.setattr(module.os, "getpid", lambda: 4242)
    monkeypatch.setattr(
        module,
        "_run_runner",
        lambda argv: pytest.fail("runner must not start after identity overwrite refusal"),
    )

    with pytest.raises(module.FreezeBMuJoCoWorkerError, match="overwrite"):
        module.main(_argv(output, identity_output))
    assert identity_output.read_text(encoding="utf-8") == "preserve"
