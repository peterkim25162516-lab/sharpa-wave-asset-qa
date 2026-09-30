from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wave_asset_qa.contact.bundle import (
    canonical_json_sha256,
    inventory_root_sha256,
    regular_tree_records,
    write_json_exclusive,
)
from wave_asset_qa.contact.scenarios import (
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)
from wave_asset_qa.parity.contracts import Simulator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "contact_c0.json"
SOURCE_TREE = "f" * 40


def _load_launcher():
    path = ROOT / "scripts" / "run_contact_c0_local.py"
    spec = importlib.util.spec_from_file_location("test_run_contact_c0_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _runtime() -> dict[str, object]:
    return {
        "schema_version": 1,
        "python_version": "3.12.6",
        "mujoco_version": "3.12.0",
        "numpy_version": "2.5.2",
        "python_implementation": "CPython",
        "platform": "Windows-11-10.0.26200-SP0",
        "machine": "AMD64",
        "python_executable_sha256": sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
    }


def _assert_exact_evidence_manifest(root: Path, scope: str) -> None:
    path = root / "evidence-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "scope", "records", "root_sha256"}
    assert payload["schema_version"] == 1
    assert payload["scope"] == scope
    records = regular_tree_records(root, exclude=("evidence-manifest.json",))
    assert payload["records"] == list(records)
    assert payload["root_sha256"] == inventory_root_sha256(records)


def _install_campaign_fakes(
    launcher: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_identity: bool = False,
) -> tuple[list[list[str]], list[str]]:
    calls: list[list[str]] = []
    source_checks: list[str] = []
    manifest = load_contact_manifest(MANIFEST)
    asset_hash = manifest.provenance.canonical_lf_asset_tree_sha256

    def source_identity(_root: Path, revision: str) -> str:
        source_checks.append(revision)
        return SOURCE_TREE

    monkeypatch.setattr(launcher, "_validate_source_identity", source_identity)
    monkeypatch.setattr(launcher, "_local_runtime_identity", lambda _python: _runtime())
    monkeypatch.setattr(
        launcher,
        "_validate_asset_tree",
        lambda assets, _manifest: (assets, asset_hash),
    )

    def fake_process(command: list[str], **kwargs: object):
        calls.append(list(command))
        case_id = _option(command, "--case-id")
        case_dir = Path(_option(command, "--output-dir"))
        for name in (
            "private-process.json",
            "private-adapter-evidence.json",
            f"{case_id}.run.json",
        ):
            (case_dir / name).write_text("{}\n", encoding="utf-8")
        if "--backend" in command:
            (case_dir / "private-campaign.json").write_text("{}\n", encoding="utf-8")
        kwargs["stdout_handle"].write(f"case={case_id}\n".encode())
        identity_number = 1 if duplicate_identity else len(calls)
        return launcher._ProcessOutcome(
            returncode=0,
            timed_out=False,
            transport_pid=20_000 + len(calls),
            worker_process={"fake": identity_number},
        )

    def validate_case(_case_dir: Path, case: object, **_kwargs: object) -> str:
        return f"{(1 if duplicate_identity else len(calls)):064x}"

    monkeypatch.setattr(launcher, "_run_case_process", fake_process)
    monkeypatch.setattr(launcher, "_validate_case_artifacts", validate_case)
    return calls, source_checks


def test_async_candidate_launcher_has_separate_identity_and_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    calls, _ = _install_campaign_fakes(launcher, monkeypatch)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "candidate"
    campaign = "contact-c0-async-v1"
    launcher.run_local_contact_c0(
        asset_root=assets, manifest_path=MANIFEST, output_dir=output,
        session_id="candidate-test", source_revision="1" * 40,
        python_executable=sys.executable, case_timeout_s=10.0,
        experimental_campaign=campaign,
    )
    assert len(calls) == 16
    assert all(_option(command, "--backend") == "mujoco" for command in calls)
    assert all(any(Path(arg).name == "run_contact_async_worker.py" for arg in command)
               for command in calls)
    for name in ("status.json", "provenance.json"):
        payload = json.loads((output / "launcher" / name).read_text(encoding="utf-8"))
        assert payload["campaign"] == campaign
    _assert_exact_evidence_manifest(output, campaign + "-mujoco")


def test_launcher_rejects_unknown_campaign_before_creating_output(tmp_path: Path) -> None:
    launcher = _load_launcher()
    output = tmp_path / "unknown"
    with pytest.raises(launcher.LocalContactC0Error, match="unrecognized experimental campaign"):
        launcher.run_local_contact_c0(
            asset_root=tmp_path, manifest_path=MANIFEST, output_dir=output,
            session_id="unknown-test", source_revision="1" * 40,
            python_executable=sys.executable, experimental_campaign="unknown",
        )
    assert not output.exists()


def test_local_launcher_runs_exact_16_cases_with_fixed_topology_and_root_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    calls, source_checks = _install_campaign_fakes(launcher, monkeypatch)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "contact-c0-local"
    revision = "1" * 40

    returned = launcher.run_local_contact_c0(
        asset_root=assets,
        manifest_path=MANIFEST,
        output_dir=output,
        session_id="contact-c0-local-test",
        source_revision=revision,
        python_executable=sys.executable,
        case_timeout_s=10.0,
    )

    assert returned == output.resolve()
    assert len(calls) == 16
    assert source_checks == [revision] * 34
    case_ids = [_option(command, "--case-id") for command in calls]
    assert len(case_ids) == len(set(case_ids)) == 16
    assert all(case_id.startswith("mujoco.") for case_id in case_ids)
    assert all(_option(command, "--manifest") == str(MANIFEST) for command in calls)
    assert all(_option(command, "--source-tree") == SOURCE_TREE for command in calls)
    assert set(path.name for path in output.iterdir()) == {
        "launcher",
        "cases",
        "evidence-manifest.json",
    }
    assert set(path.name for path in (output / "launcher").iterdir()) == {
        "provenance.json",
        "status.json",
    }
    for case_id in case_ids:
        assert set(path.name for path in (output / "cases" / case_id).iterdir()) == {
            f"{case_id}.run.json",
            "private-process.json",
            "private-adapter-evidence.json",
            "stdout.txt",
            "stderr.txt",
            "exit-code.txt",
        }
    status = json.loads(
        (output / "launcher" / "status.json").read_text(encoding="utf-8")
    )
    assert status == {
        "schema_version": 1,
        "campaign": "contact_c0",
        "backend": "mujoco",
        "status": "completed",
        "message": None,
        "session_id": "contact-c0-local-test",
        "source_revision": revision,
        "source_tree": SOURCE_TREE,
        "expected_case_count": 16,
        "completed_case_count": 16,
        "unique_process_count": 16,
        "failed_case_id": None,
    }
    provenance = json.loads(
        (output / "launcher" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["local_runtime_identity"] == _runtime()
    assert provenance["fresh_process_per_case"] is True
    assert provenance["expected_case_count"] == 16
    assert len(provenance["case_ids"]) == 16
    assert str(tmp_path) not in json.dumps(provenance)
    _assert_exact_evidence_manifest(output, "contact-c0-mujoco")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        launcher.run_local_contact_c0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="contact-c0-local-test",
            source_revision=revision,
            python_executable=sys.executable,
        )


@pytest.mark.parametrize("campaign", [None, "contact-c0-async-v1"])
def test_failure_retains_streams_status_and_exact_partial_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: str | None
) -> None:
    launcher = _load_launcher()
    manifest = load_contact_manifest(MANIFEST)
    monkeypatch.setattr(
        launcher, "_validate_source_identity", lambda _root, _revision: SOURCE_TREE
    )
    monkeypatch.setattr(launcher, "_local_runtime_identity", lambda _python: _runtime())
    monkeypatch.setattr(
        launcher,
        "_validate_asset_tree",
        lambda assets, _manifest: (
            assets,
            manifest.provenance.canonical_lf_asset_tree_sha256,
        ),
    )

    def fail(_command: list[str], **kwargs: object):
        kwargs["stdout_handle"].write(b"partial stdout\n")
        kwargs["stderr_handle"].write(b"worker failed\n")
        return launcher._ProcessOutcome(9, False, 23_456, None)

    monkeypatch.setattr(launcher, "_run_case_process", fail)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "failed-contact-c0"
    with pytest.raises(launcher.LocalContactC0Error, match="exited with code 9"):
        launcher.run_local_contact_c0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="failure-test",
            source_revision="2" * 40,
            python_executable=sys.executable,
            experimental_campaign=campaign,
        )

    case_dirs = list((output / "cases").iterdir())
    assert len(case_dirs) == 1
    assert (case_dirs[0] / "stdout.txt").read_text() == "partial stdout\n"
    assert (case_dirs[0] / "stderr.txt").read_text() == "worker failed\n"
    assert (case_dirs[0] / "exit-code.txt").read_text().strip() == "9"
    status = json.loads(
        (output / "launcher" / "status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "error"
    assert status["campaign"] == (campaign or "contact_c0")
    assert status["completed_case_count"] == 0
    assert status["unique_process_count"] == 0
    assert status["failed_case_id"].startswith("mujoco.")
    _assert_exact_evidence_manifest(output, (campaign or "contact-c0") + "-mujoco")


def test_launcher_rejects_duplicate_fresh_process_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    _install_campaign_fakes(launcher, monkeypatch, duplicate_identity=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "duplicate-contact-c0"
    with pytest.raises(launcher.LocalContactC0Error, match="reused a prior"):
        launcher.run_local_contact_c0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="duplicate-test",
            source_revision="3" * 40,
            python_executable=sys.executable,
        )
    status = json.loads(
        (output / "launcher" / "status.json").read_text(encoding="utf-8")
    )
    assert status["completed_case_count"] == 1
    assert status["unique_process_count"] == 1
    _assert_exact_evidence_manifest(output, "contact-c0-mujoco")


def test_case_validation_checks_process_adapter_and_canonical_run_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    manifest = load_contact_manifest(MANIFEST)
    case = next(
        case
        for case in expand_contact_cases(manifest)
        if case.simulator is Simulator.MUJOCO
    )
    case_dir = tmp_path / case.case_id
    case_dir.mkdir()
    runtime = _runtime()
    revision = "4" * 40
    identity = {
        "platform": "windows",
        "pid": 24_680,
        "process_creation_filetime": 13_400_000_000_000_001,
    }
    fresh_hash = launcher.os_process_identity_sha256(identity)
    worker_module = ROOT / "src" / "wave_asset_qa" / "contact" / "worker.py"
    process = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": case.case_id,
        "backend": "mujoco",
        "session_id": "exact-test",
        "source_revision": revision,
        "source_tree": SOURCE_TREE,
        "manifest_file_sha256": launcher._sha256_file(MANIFEST),
        "manifest_semantic_sha256": contact_manifest_sha256(manifest),
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "python_version": runtime["python_version"],
        "python_executable_realpath": os.path.realpath(sys.executable),
        "worker_module_realpath": os.path.realpath(worker_module),
        "worker_module_sha256": launcher._sha256_file(worker_module),
        "os_process_identity": identity,
        "fresh_process_identity_sha256": fresh_hash,
    }
    write_json_exclusive(case_dir / "private-process.json", process)
    fixture = {"collision_inventory_hash_preimage": {"items": []}}
    runtime_fingerprint = {
        "backend": "mujoco",
        "backend_version": runtime["mujoco_version"],
        "python_version": runtime["python_version"],
        "platform": runtime["platform"],
        "device": "cpu",
        "solver": "2",
        "integrator": "0",
        "control_path": "MjData.ctrl -> mj_step",
        "contact_path": "explicit pair -> mj_contactForce",
    }
    adapter = {
        "fixture_overlay": fixture,
        "runtime_fingerprint": runtime_fingerprint,
        "unexpected_pair_observations": [],
    }
    write_json_exclusive(case_dir / "private-adapter-evidence.json", adapter)
    run = SimpleNamespace(
        provenance={
            "fixture_overlay_sha256": canonical_json_sha256(fixture),
            "runtime_fingerprint_sha256": canonical_json_sha256(runtime_fingerprint),
        },
        fixture_readback={
            "collision_inventory_sha256": canonical_json_sha256(
                fixture["collision_inventory_hash_preimage"]
            )
        },
    )
    monkeypatch.setattr(
        launcher, "load_and_validate_contact_run", lambda *_args, **_kwargs: run
    )
    monkeypatch.setattr(launcher, "canonical_contact_run_json", lambda _run: '{"ok":true}')
    (case_dir / f"{case.case_id}.run.json").write_text(
        '{"ok":true}\n', encoding="utf-8", newline="\n"
    )
    (case_dir / "stdout.txt").write_bytes(b"ok\n")
    (case_dir / "stderr.txt").write_bytes(b"")
    (case_dir / "exit-code.txt").write_text("0\n", encoding="utf-8")
    outcome = launcher._ProcessOutcome(0, False, identity["pid"], process)

    assert launcher._validate_case_artifacts(
        case_dir,
        case,
        manifest=manifest,
        manifest_path=MANIFEST,
        manifest_file_sha256=launcher._sha256_file(MANIFEST),
        manifest_semantic_sha256=contact_manifest_sha256(manifest),
        asset_tree_sha256_value=manifest.provenance.canonical_lf_asset_tree_sha256,
        session_id="exact-test",
        source_revision=revision,
        source_tree=SOURCE_TREE,
        runtime_identity=runtime,
        project_root=ROOT,
        outcome=outcome,
    ) == fresh_hash

    adapter["extra"] = True
    (case_dir / "private-adapter-evidence.json").unlink()
    write_json_exclusive(case_dir / "private-adapter-evidence.json", adapter)
    with pytest.raises(launcher.LocalContactC0Error, match="fields are not exact"):
        launcher._validate_case_artifacts(
            case_dir,
            case,
            manifest=manifest,
            manifest_path=MANIFEST,
            manifest_file_sha256=launcher._sha256_file(MANIFEST),
            manifest_semantic_sha256=contact_manifest_sha256(manifest),
            asset_tree_sha256_value=manifest.provenance.canonical_lf_asset_tree_sha256,
            session_id="exact-test",
            source_revision=revision,
            source_tree=SOURCE_TREE,
            runtime_identity=runtime,
            project_root=ROOT,
            outcome=outcome,
        )


def test_private_process_schema_and_hash_fail_closed() -> None:
    launcher = _load_launcher()
    identity = {
        "platform": "windows",
        "pid": 123,
        "process_creation_filetime": 456,
    }
    record = {name: None for name in launcher._PRIVATE_PROCESS_KEYS}
    record.update(
        {
            "schema_version": 1,
            "os_process_identity": identity,
            "fresh_process_identity_sha256": launcher.os_process_identity_sha256(
                identity
            ),
        }
    )
    assert launcher._validate_private_process_record(record)[
        "fresh_process_identity_sha256"
    ] == record["fresh_process_identity_sha256"]
    with pytest.raises(launcher.LocalContactC0Error, match="fields are not exact"):
        launcher._validate_private_process_record({**record, "extra": True})
    with pytest.raises(launcher.LocalContactC0Error, match="hash is invalid"):
        launcher._validate_private_process_record(
            {**record, "fresh_process_identity_sha256": "0" * 64}
        )


@pytest.mark.parametrize("campaign", [None, "contact-c0-async-v1"])
def test_private_process_must_belong_to_owned_transport(
    monkeypatch: pytest.MonkeyPatch, campaign: str | None,
) -> None:
    launcher = _load_launcher()
    transport_pid = 123
    foreign_pid = 456
    identity = {
        "platform": "windows",
        "pid": foreign_pid,
        "process_creation_filetime": 789,
    }
    record = {name: None for name in launcher._PRIVATE_PROCESS_KEYS}
    record.update(
        {
            "schema_version": 1,
            "os_process_identity": identity,
            "fresh_process_identity_sha256": launcher.os_process_identity_sha256(
                identity
            ),
        }
    )
    if campaign:
        record['experimental_campaign'] = campaign
    monkeypatch.setattr(launcher, '_windows_parent_process_id', lambda pid: transport_pid)
    assert launcher._bind_private_process_record(record, transport_pid=transport_pid) == record
    with pytest.raises(
        launcher.LocalContactC0Error,
        match="does not belong to the owned transport process",
    ):
        monkeypatch.setattr(
            launcher,
            "_windows_parent_process_id",
            lambda pid: 999,
        )
        launcher._bind_private_process_record(record, transport_pid=transport_pid)

    identity_terminations: list[object] = []
    monkeypatch.setattr(
        launcher,
        "terminate_windows_process_identity",
        lambda *args, **kwargs: identity_terminations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        launcher,
        "terminate_posix_process_identity",
        lambda *args, **kwargs: identity_terminations.append((args, kwargs)),
    )

    class _OwnedTransport:
        pid = transport_pid

        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        def poll(self) -> int | None:
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            return -9

    process = _OwnedTransport()
    with pytest.raises(
        launcher.LocalContactC0Error,
        match="cannot prove Contact C0 worker termination",
    ):
        launcher._terminate_owned_worker(process, record)
    assert identity_terminations == []
    assert process.killed is True
    assert process.waited is True


def test_process_helper_uses_direct_fresh_worker_and_captures_streams(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    process_path = tmp_path / "private-process.json"
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    code = (
        "import json,os,sys,time; from pathlib import Path; "
        "from wave_asset_qa.parity.process_identity import "
        "current_os_process_identity,os_process_identity_sha256; "
        "i=current_os_process_identity(); "
        "keys=" + repr(sorted(launcher._PRIVATE_PROCESS_KEYS)) + "; "
        "r={k:None for k in keys}; "
        "r.update({'schema_version':1,'os_process_identity':i,"
        "'fresh_process_identity_sha256':os_process_identity_sha256(i)}); "
        "Path(sys.argv[1]).write_text(json.dumps(r),encoding='utf-8'); "
        "print(os.getpid(),os.getppid(),flush=True); "
        "print('worker-stderr',file=sys.stderr); time.sleep(0.2)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        outcome = launcher._run_case_process(
            [sys.executable, "-P", "-c", code, str(process_path)],
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            environment=environment,
            cwd=ROOT,
            private_process_path=process_path,
            timeout_s=10.0,
        )
    assert outcome.returncode == 0
    assert outcome.timed_out is False
    worker_pid, worker_parent_pid = map(int, stdout_path.read_text().split())
    assert outcome.transport_pid is not None
    assert outcome.transport_pid != os.getpid()
    assert outcome.worker_process is not None
    assert outcome.worker_process["os_process_identity"]["pid"] == worker_pid
    if os.name == "nt":
        assert outcome.transport_pid in {worker_pid, worker_parent_pid}
    else:
        assert outcome.transport_pid == worker_pid
    assert worker_pid != os.getpid()
    assert stderr_path.read_text().strip() == "worker-stderr"


def test_manifest_matrix_runtime_budget_and_bytecode_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    manifest = load_contact_manifest(MANIFEST)
    cases = launcher.canonical_mujoco_cases(manifest)
    assert len(cases) == len({case.case_id for case in cases}) == 16
    assert all(case.simulator is Simulator.MUJOCO for case in cases)

    evidence = tmp_path / "budget"
    evidence.mkdir()
    (evidence / "ten.bin").write_bytes(b"0123456789")
    assert launcher._enforce_artifact_budget(evidence, maximum_bytes=10) == 10
    with pytest.raises(launcher._ArtifactBudgetExceeded, match="2 GiB"):
        launcher._enforce_artifact_budget(evidence, maximum_bytes=9)

    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src" / "x.pyc").write_bytes(b"bytecode")
    with pytest.raises(launcher.LocalContactC0Error, match=r"no \.pyc"):
        launcher._reject_source_bytecode(source)

    monkeypatch.setattr(launcher.numpy, "__version__", "2.5.3")
    with pytest.raises(launcher.LocalContactC0Error, match="numpy_version"):
        launcher._local_runtime_identity(Path(sys.executable))


def test_source_guard_includes_untracked_files_and_precedes_heavy_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    revision = "5" * 40

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(ROOT)
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return revision
        if arguments == ("rev-parse", "--verify", f"{revision}^{{tree}}"):
            return SOURCE_TREE
        if arguments[:2] == ("status", "--porcelain=v1"):
            assert "--untracked-files=all" in arguments
            return "?? src/untracked.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(launcher, "_git_text", fake_git)
    monkeypatch.setattr(launcher, "_reject_source_bytecode", lambda _root: None)
    with pytest.raises(launcher.LocalContactC0Error, match="including untracked files"):
        launcher._validate_source_identity(ROOT, revision)

    source = (ROOT / "scripts" / "run_contact_c0_local.py").read_text(
        encoding="utf-8"
    )
    guard = source.index('if __name__ == "__main__":')
    assert guard < source.index("\nimport mujoco\n")
    assert guard < source.index("\nimport wave_asset_qa\n")
    assert source.index('os.environ["PYTHONDONTWRITEBYTECODE"] = "1"') < source.index(
        "\nimport mujoco\n"
    )


@pytest.mark.parametrize(
    ("session", "revision", "match"),
    [
        ("bad session", "6" * 40, "session_id"),
        ("valid-session", "ABC", "source_revision"),
    ],
)
def test_ambiguous_identity_rejected_before_output(
    tmp_path: Path, session: str, revision: str, match: str
) -> None:
    launcher = _load_launcher()
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "never-created"
    with pytest.raises(launcher.LocalContactC0Error, match=match):
        launcher.run_local_contact_c0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session,
            source_revision=revision,
            python_executable=sys.executable,
        )
    assert not output.exists()


def test_path_guards_accept_real_ancestors_and_exclusive_new_files(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    real = tmp_path / "real"
    real.mkdir()
    source = real / "input.txt"
    source.write_text("input\n", encoding="utf-8")

    assert launcher._resolved_directory(real, "real directory") == real.resolve()
    assert launcher._resolved_file(source, "real file") == source.resolve()
    output = launcher._new_output_path(real / "new-output")
    assert output == real.resolve() / "new-output"
    evidence = real / "new-evidence.txt"
    launcher._write_text_exclusive(evidence, "evidence\n")
    assert evidence.read_text(encoding="utf-8") == "evidence\n"


def test_all_path_guards_reject_a_link_like_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    linked_ancestor = tmp_path / "simulated-junction"
    linked_ancestor.mkdir()
    source = linked_ancestor / "input.txt"
    source.write_text("input\n", encoding="utf-8")
    original = launcher.is_link_like
    monkeypatch.setattr(
        launcher,
        "is_link_like",
        lambda path: Path(path) == linked_ancestor or original(Path(path)),
    )

    operations = (
        lambda: launcher._resolved_directory(linked_ancestor, "directory input"),
        lambda: launcher._resolved_file(source, "file input"),
        lambda: launcher._new_output_path(linked_ancestor / "new-output"),
        lambda: launcher._write_text_exclusive(
            linked_ancestor / "new-evidence.txt", "evidence\n"
        ),
        lambda: launcher._artifact_size_bytes(linked_ancestor),
    )
    for operation in operations:
        with pytest.raises(
            launcher.LocalContactC0Error,
            match="symbolic-link or junction ancestor",
        ):
            operation()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics only")
def test_windows_directory_junction_is_detected_before_resolve(tmp_path: Path) -> None:
    launcher = _load_launcher()
    target = tmp_path / "junction-target"
    target.mkdir()
    source = target / "input.txt"
    source.write_text("input\n", encoding="utf-8")
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        pytest.skip("this Windows host cannot create a test junction")
    try:
        assert junction.is_junction()
        operations = (
            lambda: launcher._resolved_directory(junction, "junction directory"),
            lambda: launcher._resolved_file(junction / source.name, "junction file"),
            lambda: launcher._new_output_path(junction / "new-output"),
            lambda: launcher._write_text_exclusive(
                junction / "new-evidence.txt", "evidence\n"
            ),
            lambda: launcher._artifact_size_bytes(junction),
        )
        for operation in operations:
            with pytest.raises(
                launcher.LocalContactC0Error,
                match="symbolic-link or junction ancestor",
            ):
                operation()
    finally:
        os.rmdir(junction)
