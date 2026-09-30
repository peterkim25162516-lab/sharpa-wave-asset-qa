from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.runner import write_collected_run
from wave_asset_qa.parity.scenarios import (
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "waveform_t1.json"
GATE0_MANIFEST = ROOT / "configs" / "parity" / "gate0.json"
SOURCE_TREE = "f" * 40


def _load_launcher():
    path = ROOT / "scripts" / "run_waveform_t1_local.py"
    spec = importlib.util.spec_from_file_location("test_run_waveform_t1_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _assert_hash_inventory(root: Path) -> None:
    hash_file = root / "evidence.sha256"
    lines = hash_file.read_text(encoding="utf-8").splitlines()
    recorded: dict[str, str] = {}
    for line in lines:
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    expected = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path != hash_file
    }
    assert recorded == expected


def _install_success_fakes(
    launcher: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str,
    source_revision: str,
    wrong_digest: bool = False,
    reuse_pid: bool = False,
) -> tuple[list[list[str]], list[str]]:
    calls: list[list[str]] = []
    source_checks: list[str] = []
    next_pid = 20_000
    runtime_identity = launcher._local_runtime_identity(Path(sys.executable))

    def source_identity(_root: Path, observed_revision: str) -> str:
        source_checks.append(observed_revision)
        return SOURCE_TREE

    def fake_process(command: list[str], **kwargs: object):
        nonlocal next_pid
        assert kwargs["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
        calls.append(list(command))
        case_id = _option(command, "--case-id")
        output_dir = Path(_option(command, "--output-dir"))
        manifest = load_manifest(_option(command, "--manifest"))
        case = next(
            item for item in expand_scenario_cases(manifest) if item.case_id == case_id
        )
        scenario = manifest.scenario(case.scenario_id)
        hand = manifest.hand(case.hand)
        requested_steps = round(scenario.duration_s / case.dt_s)
        child_pid = 20_000 if reuse_pid else next_pid
        next_pid += 1
        worker_identity = {
            "platform": "windows",
            "pid": child_pid,
            "process_creation_filetime": 10_000_000 + next_pid,
        }
        worker_process = {
            "schema_version": 1,
            "worker_pid": child_pid,
            "os_process_identity": worker_identity,
            "fresh_process_id": launcher.os_process_identity_sha256(
                worker_identity
            ),
        }
        launcher._write_json_exclusive(
            kwargs["worker_process_path"], worker_process
        )
        scheduled_digest = canonical_target_sequence_sha256(
            scenario,
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=True,
        )
        if wrong_digest:
            scheduled_digest = "0" * 64
        provenance = {
            "manifest_sha256": manifest_sha256(manifest),
            "session_id": session_id,
            "source_revision": source_revision,
            "worker_pid": child_pid,
            "asset_tree_sha256": (
                manifest.provenance.canonical_lf_asset_tree_sha256
            ),
            "asset_tree_verification": "scanned_path_size_bytes",
            "asset_commit": manifest.provenance.commit,
            "asset_git_tree": manifest.provenance.asset_git_tree,
            "mapping_schema_version": 1,
            "backend_version": runtime_identity["mujoco_version"],
            "python_version": runtime_identity["python_version"],
            "platform": runtime_identity["platform"],
            "local_runtime_identity": dict(runtime_identity),
            "target_sequence_digest_schema_version": 1,
            "target_sequence_digest_encoding": launcher.TARGET_DIGEST_ENCODING,
            "target_sequence_digest_projection": launcher.TARGET_DIGEST_PROJECTION,
            "target_sequence_canonical_joint_names": list(hand.joint_names),
            "scheduled_target_sequence_semantics": (
                launcher.SCHEDULED_TARGET_SEMANTICS
            ),
            "scheduled_target_sequence_count": requested_steps + 1,
            "scheduled_target_sequence_sha256": scheduled_digest,
        }
        result = AdapterRunResult(
            backend="mujoco",
            scenario_id=case.scenario_id,
            status="completed",
            message="synthetic Waveform T1 launcher fixture",
            dt=case.dt_s,
            requested_steps=requested_steps,
            completed_steps=requested_steps,
            joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
            samples=(),
            provenance=provenance,
        )
        write_collected_run(output_dir, CollectedRun(case=case, result=result))
        stdout_handle = kwargs["stdout_handle"]
        stderr_handle = kwargs["stderr_handle"]
        stdout_handle.write(f"case={case_id}\n".encode("utf-8"))
        stderr_handle.write(b"")
        return launcher._ProcessOutcome(
            returncode=0,
            timed_out=False,
            transport_pid=30_000 + len(calls),
            worker_process=worker_process,
        )

    monkeypatch.setattr(launcher, "_validate_source_identity", source_identity)
    monkeypatch.setattr(launcher, "_run_case_process", fake_process)
    return calls, source_checks


def test_local_launcher_runs_exact_16_fresh_cases_and_keeps_hashed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    runtime_identity = launcher._local_runtime_identity(Path(sys.executable))
    revision = "1" * 40
    session = "waveform-t1-local-test"
    calls, source_checks = _install_success_fakes(
        launcher,
        monkeypatch,
        session_id=session,
        source_revision=revision,
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "waveform-t1-evidence"

    returned = launcher.run_local_waveform_t1(
        asset_root=assets,
        manifest_path=MANIFEST,
        output_dir=output,
        session_id=session,
        source_revision=revision,
        python_executable=sys.executable,
        case_timeout_s=10.0,
    )

    assert returned == output.resolve()
    assert len(calls) == 16
    assert source_checks == [revision] * 34
    case_ids = [_option(command, "--case-id") for command in calls]
    assert len(set(case_ids)) == 16
    assert all(case_id.startswith("mujoco.") for case_id in case_ids)
    assert all("small_step" not in case_id for case_id in case_ids)
    assert all(command.count("--case-id") == 1 for command in calls)
    assert len({_option(command, "--manifest") for command in calls}) == 1
    assert len({_option(command, "--session-id") for command in calls}) == 1
    assert len({_option(command, "--source-revision") for command in calls}) == 1
    for case_id in case_ids:
        case_dir = output / "cases" / case_id
        assert {path.name for path in case_dir.iterdir()} == {
            f"{case_id}.run.json",
            "evidence.sha256",
            "launcher.command.json",
            "launcher.exitcode.txt",
            "launcher.process.json",
            "launcher.stderr.log",
            "launcher.stdout.log",
            "worker-process.json",
        }
        assert (case_dir / "launcher.exitcode.txt").read_text().strip() == "0"
        process_record = json.loads(
            (case_dir / "launcher.process.json").read_text(encoding="utf-8")
        )
        assert process_record["transport_pid"] > 0
        assert process_record["transport_pid"] != process_record["launcher_pid"]
        assert process_record["worker_process"]["worker_pid"] > 0
        command_record = json.loads(
            (case_dir / "launcher.command.json").read_text(encoding="utf-8")
        )
        assert command_record["argv"][0] == "<python>"
        assert "<asset-root>" in command_record["argv"]
        assert "launcher/waveform_t1.manifest.json" in command_record["argv"]
        assert str(tmp_path) not in json.dumps(command_record)
        run_payload = json.loads(
            (case_dir / f"{case_id}.run.json").read_text(encoding="utf-8")
        )
        assert (
            run_payload["result"]["provenance"]["local_runtime_identity"]
            == runtime_identity
        )
        _assert_hash_inventory(case_dir)

    launcher_dir = output / "launcher"
    assert {path.name for path in launcher_dir.iterdir()} == {
        "completed.json",
        "evidence.sha256",
        "matrix.json",
        "provenance.json",
        "waveform_t1.manifest.json",
    }
    completed = json.loads(
        (launcher_dir / "completed.json").read_text(encoding="utf-8")
    )
    assert completed["campaign"] == "waveform_t1"
    assert completed["completed_case_count"] == 16
    assert completed["fresh_worker_process_count"] == 16
    assert len(set(completed["worker_pids"])) == 16
    assert completed["local_runtime_identity"] == {
        "schema_version": 1,
        "python_version": "3.12.6",
        "mujoco_version": "3.12.0",
        "numpy_version": "2.5.2",
        "python_implementation": "CPython",
        "platform": "Windows-11-10.0.26200-SP0",
        "machine": "AMD64",
        "python_executable_sha256": launcher.EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256,
    }
    assert completed["source_bridge_included"] is False
    _assert_hash_inventory(launcher_dir)

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session,
            source_revision=revision,
            python_executable=sys.executable,
        )


def test_local_launcher_enforces_two_gib_evidence_budget(tmp_path: Path) -> None:
    launcher = _load_launcher()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "payload.bin").write_bytes(b"0123456789")

    assert launcher._enforce_artifact_budget(evidence, maximum_bytes=10) == 10
    with pytest.raises(launcher._ArtifactBudgetExceeded, match="2 GiB"):
        launcher._enforce_artifact_budget(evidence, maximum_bytes=9)


def test_local_launcher_stops_on_nonzero_and_retains_raw_process_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher, "_validate_source_identity", lambda _root, _revision: SOURCE_TREE
    )

    def fail(_command: list[str], **kwargs: object):
        kwargs["stdout_handle"].write(b"partial stdout\n")
        kwargs["stderr_handle"].write(b"worker failed\n")
        return launcher._ProcessOutcome(
            returncode=9,
            timed_out=False,
            transport_pid=23_456,
            worker_process=None,
        )

    monkeypatch.setattr(launcher, "_run_case_process", fail)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "failed-waveform-t1"
    with pytest.raises(launcher.LocalWaveformT1Error, match="exited with code 9"):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="failure-session",
            source_revision="2" * 40,
            python_executable=sys.executable,
        )

    case_dirs = list((output / "cases").iterdir())
    assert len(case_dirs) == 1
    case_dir = case_dirs[0]
    assert (case_dir / "launcher.stdout.log").read_text() == "partial stdout\n"
    assert (case_dir / "launcher.stderr.log").read_text() == "worker failed\n"
    assert (case_dir / "launcher.exitcode.txt").read_text().strip() == "9"
    _assert_hash_inventory(case_dir)


@pytest.mark.parametrize(
    ("wrong_digest", "reuse_pid", "match"),
    [
        (True, False, "scheduled_target_sequence_sha256"),
        (False, True, "reused a prior worker PID"),
    ],
)
def test_local_launcher_rejects_noncanonical_or_nonfresh_completed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_digest: bool,
    reuse_pid: bool,
    match: str,
) -> None:
    launcher = _load_launcher()
    revision = "3" * 40
    session = "invalid-completed-evidence"
    _install_success_fakes(
        launcher,
        monkeypatch,
        session_id=session,
        source_revision=revision,
        wrong_digest=wrong_digest,
        reuse_pid=reuse_pid,
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "rejected-waveform-t1"

    with pytest.raises(launcher.LocalWaveformT1Error, match=match):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session,
            source_revision=revision,
            python_executable=sys.executable,
        )

    case_dirs = list((output / "cases").iterdir())
    assert len(case_dirs) == (1 if wrong_digest else 2)
    for case_dir in case_dirs:
        _assert_hash_inventory(case_dir)
    assert not (output / "launcher" / "completed.json").exists()


def test_local_launcher_rejects_source_or_wrong_manifest_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    assets = tmp_path / "assets"
    assets.mkdir()

    dirty_output = tmp_path / "never-created-dirty"

    def reject(_root: Path, _revision: str) -> str:
        raise launcher.LocalWaveformT1Error("tracked worktree/index must be clean")

    monkeypatch.setattr(launcher, "_validate_source_identity", reject)
    with pytest.raises(launcher.LocalWaveformT1Error, match="tracked worktree/index"):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=dirty_output,
            session_id="source-check",
            source_revision="4" * 40,
            python_executable=sys.executable,
        )
    assert not dirty_output.exists()

    wrong_manifest_output = tmp_path / "never-created-wrong-manifest"
    with pytest.raises(launcher.LocalWaveformT1Error, match="schema_version=2"):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=GATE0_MANIFEST,
            output_dir=wrong_manifest_output,
            session_id="wrong-manifest",
            source_revision="4" * 40,
            python_executable=sys.executable,
        )
    assert not wrong_manifest_output.exists()


def test_source_identity_rejects_untracked_source_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    revision = "4" * 40

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(ROOT)
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return revision
        if arguments == ("rev-parse", "--verify", f"{revision}^{{tree}}"):
            return SOURCE_TREE
        if arguments[:2] == ("status", "--porcelain=v1"):
            assert "--untracked-files=all" in arguments
            return "?? src/sitecustomize.py"
        raise AssertionError(f"unexpected Git query: {arguments!r}")

    monkeypatch.setattr(launcher, "_git_text", fake_git)
    monkeypatch.setattr(launcher, "_reject_source_bytecode", lambda _root: None)
    with pytest.raises(
        launcher.LocalWaveformT1Error,
        match="including untracked files",
    ):
        launcher._validate_source_identity(ROOT, revision)


@pytest.mark.parametrize(
    "relative",
    (
        "src/package/ignored.pyc",
        "scripts/ignored.pyo",
        "src/package/__pycache__",
    ),
)
def test_source_bytecode_artifacts_fail_closed(
    tmp_path: Path, relative: str
) -> None:
    launcher = _load_launcher()
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    artifact = tmp_path / relative
    if artifact.name == "__pycache__":
        artifact.mkdir(parents=True)
    else:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"ignored bytecode")

    with pytest.raises(launcher.LocalWaveformT1Error, match=r"no \.pyc"):
        launcher._reject_source_bytecode(tmp_path)


def test_bytecode_guard_precedes_project_and_dependency_imports() -> None:
    source = (ROOT / "scripts" / "run_waveform_t1_local.py").read_text(
        encoding="utf-8"
    )
    bootstrap = source.index('if __name__ == "__main__":')

    assert bootstrap < source.index("\nimport mujoco\n")
    assert bootstrap < source.index("\nimport wave_asset_qa\n")
    assert source.index('os.environ["PYTHONDONTWRITEBYTECODE"] = "1"') < source.index(
        "\nimport mujoco\n"
    )


def test_local_runtime_rejects_arbitrary_python_and_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    arbitrary_python = tmp_path / "python.exe"
    arbitrary_python.write_bytes(b"not this process")
    with pytest.raises(launcher.LocalWaveformT1Error, match="own sys.executable"):
        launcher._local_runtime_identity(arbitrary_python)

    monkeypatch.setattr(launcher.numpy, "__version__", "2.5.3")
    with pytest.raises(launcher.LocalWaveformT1Error, match="numpy_version"):
        launcher._local_runtime_identity(Path(sys.executable))


def test_local_runtime_rejects_python_and_mujoco_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher.platform, "python_version", lambda: "3.12.7")
    with pytest.raises(launcher.LocalWaveformT1Error, match="python_version"):
        launcher._local_runtime_identity(Path(sys.executable))

    monkeypatch.undo()
    launcher = _load_launcher()
    monkeypatch.setattr(launcher.mujoco, "__version__", "3.12.1")
    with pytest.raises(launcher.LocalWaveformT1Error, match="mujoco_version"):
        launcher._local_runtime_identity(Path(sys.executable))


def test_local_runtime_rejects_interpreter_sha256_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher, "EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256", "0" * 64
    )
    with pytest.raises(launcher.LocalWaveformT1Error, match="SHA-256 drifted"):
        launcher._local_runtime_identity(Path(sys.executable))


def test_artifact_size_retries_atomic_rename_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"12345")
    original_lstat = Path.lstat
    failed_once = False

    def flaky_lstat(path: Path):
        nonlocal failed_once
        if path == payload and not failed_once:
            failed_once = True
            raise FileNotFoundError(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)
    assert launcher._artifact_size_bytes(tmp_path) == 5
    assert failed_once


def test_budget_measurement_failure_terminates_owned_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    stdout_path = tmp_path / "budget-stdout.log"
    stderr_path = tmp_path / "budget-stderr.log"
    terminated: list[int] = []
    original_terminate = launcher._terminate_owned_worker

    def fail_measurement(_root: Path, *, maximum_bytes: int) -> int:
        assert maximum_bytes == launcher.MAX_PRIVATE_ARTIFACT_BYTES
        raise launcher.LocalWaveformT1Error("unstable artifact tree")

    def record_termination(process, worker_process) -> None:
        terminated.append(process.pid)
        original_terminate(process, worker_process)

    monkeypatch.setattr(launcher, "_enforce_artifact_budget", fail_measurement)
    monkeypatch.setattr(launcher, "_terminate_owned_worker", record_termination)
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        outcome = launcher._run_case_process(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            environment=os.environ,
            cwd=ROOT,
            worker_process_path=tmp_path / "never-created-worker-process.json",
            timeout_s=10.0,
            evidence_root=tmp_path,
        )

    assert outcome.returncode == 125
    assert outcome.timed_out is False
    assert terminated == [outcome.transport_pid]
    assert "unstable artifact tree" in stderr_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("session_id", "revision", "match"),
    [
        ("bad session", "5" * 40, "session_id"),
        ("valid-session", "ABC", "source_revision"),
    ],
)
def test_local_launcher_rejects_ambiguous_identity_before_output(
    tmp_path: Path,
    session_id: str,
    revision: str,
    match: str,
) -> None:
    launcher = _load_launcher()
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "never-created-identity"
    with pytest.raises(launcher.LocalWaveformT1Error, match=match):
        launcher.run_local_waveform_t1(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session_id,
            source_revision=revision,
            python_executable=sys.executable,
        )
    assert not output.exists()


def test_process_helper_uses_a_real_child_and_captures_raw_streams(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    command = [
        sys.executable,
        "-c",
        (
            "import json,os,sys; from pathlib import Path; "
            "from wave_asset_qa.parity.process_identity import "
            "current_os_process_identity,os_process_identity_sha256; "
            "identity=current_os_process_identity(); "
            "record={'schema_version':1,'worker_pid':os.getpid(),"
            "'os_process_identity':identity,"
            "'fresh_process_id':os_process_identity_sha256(identity)}; "
            "Path(sys.argv[1]).write_text(json.dumps(record),encoding='utf-8'); "
            "print(os.getpid()); "
            "print('fake-worker-stderr', file=sys.stderr)"
        ),
        str(tmp_path / "worker-process.json"),
    ]
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        outcome = launcher._run_case_process(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            environment=os.environ,
            cwd=ROOT,
            worker_process_path=tmp_path / "worker-process.json",
            timeout_s=10.0,
        )

    assert outcome.returncode == 0
    assert outcome.timed_out is False
    assert outcome.transport_pid is not None
    assert outcome.transport_pid != os.getpid()
    assert outcome.worker_process is not None
    worker_pid = outcome.worker_process["worker_pid"]
    assert worker_pid != os.getpid()
    assert int(stdout_path.read_text().strip()) == worker_pid
    assert stderr_path.read_text().strip() == "fake-worker-stderr"
