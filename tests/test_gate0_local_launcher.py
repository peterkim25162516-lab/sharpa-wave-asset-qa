from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.runner import write_collected_run
from wave_asset_qa.parity.scenarios import expand_scenario_cases, load_manifest, manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "gate0.json"
SOURCE_TREE = "f" * 40


def _load_launcher():
    path = ROOT / "scripts" / "run_gate0_local.py"
    spec = importlib.util.spec_from_file_location("test_run_gate0_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_local_launcher_uses_one_fresh_process_per_exact_case_and_keeps_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    revision = "1" * 40
    session = "gate0-test-session"
    calls: list[tuple[list[str], dict[str, object]]] = []
    source_checks: list[str] = []

    def source_identity(_root: Path, observed_revision: str) -> str:
        source_checks.append(observed_revision)
        return SOURCE_TREE

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((list(command), dict(kwargs)))
        case_id = _option(command, "--case-id")
        output_dir = Path(_option(command, "--output-dir"))
        manifest = load_manifest(_option(command, "--manifest"))
        case = next(item for item in expand_scenario_cases(manifest) if item.case_id == case_id)
        hand = manifest.hand(case.hand)
        requested_steps = round(
            manifest.scenario(case.scenario_id).duration_s / case.dt_s
        )
        result = AdapterRunResult(
            backend="mujoco",
            scenario_id=case.scenario_id,
            status="completed",
            message="synthetic launcher fixture",
            dt=case.dt_s,
            requested_steps=requested_steps,
            completed_steps=requested_steps,
            joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
            samples=(),
            provenance={
                "manifest_sha256": manifest_sha256(manifest),
                "session_id": session,
                "source_revision": revision,
                "asset_tree_sha256": (
                    manifest.provenance.canonical_lf_asset_tree_sha256
                ),
                "asset_tree_verification": "scanned_path_size_bytes",
                "asset_commit": manifest.provenance.commit,
                "asset_git_tree": manifest.provenance.asset_git_tree,
                "mapping_schema_version": 1,
            },
        )
        write_collected_run(output_dir, CollectedRun(case=case, result=result))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher, "_validate_source_identity", source_identity)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "local-evidence"
    returned = launcher.run_local_gate0(
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
    case_ids = [_option(command, "--case-id") for command, _ in calls]
    assert len(set(case_ids)) == 16
    assert all(command.count("--case-id") == 1 for command, _ in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert len({
        _option(command, "--manifest") for command, _ in calls
    }) == 1
    assert len({_option(command, "--session-id") for command, _ in calls}) == 1
    assert len({_option(command, "--source-revision") for command, _ in calls}) == 1
    for case_id in case_ids:
        case_dir = output / "cases" / case_id
        assert (case_dir / f"{case_id}.run.json").is_file()
        assert (case_dir / "launcher.stdout.log").is_file()
        assert (case_dir / "launcher.stderr.log").is_file()
        assert (case_dir / "launcher.exitcode.txt").read_text().strip() == "0"
        command_record = json.loads(
            (case_dir / "launcher.command.json").read_text(encoding="utf-8")
        )
        assert command_record["argv"][0] == "<python>"
        assert "<asset-root>" in command_record["argv"]
        assert str(tmp_path) not in json.dumps(command_record)
        assert (case_dir / "evidence.sha256").is_file()
    launcher_dir = output / "launcher"
    assert {path.name for path in launcher_dir.iterdir()} == {
        "completed.json",
        "evidence.sha256",
        "gate0.manifest.json",
        "matrix.json",
        "provenance.json",
    }

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        launcher.run_local_gate0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session,
            source_revision=revision,
            python_executable=sys.executable,
        )


def test_local_launcher_stops_on_nonzero_and_retains_raw_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher, "_validate_source_identity", lambda _root, _revision: SOURCE_TREE
    )

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "failed-evidence"
    with pytest.raises(launcher.LocalGate0Error, match="exited with code 9"):
        launcher.run_local_gate0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="failure-session",
            source_revision="2" * 40,
            python_executable=sys.executable,
        )

    case_dirs = list((output / "cases").iterdir())
    assert len(case_dirs) == 1
    assert (case_dirs[0] / "launcher.stdout.log").is_file()
    assert (case_dirs[0] / "launcher.stderr.log").is_file()
    assert (case_dirs[0] / "launcher.exitcode.txt").read_text().strip() == "9"
    assert (case_dirs[0] / "evidence.sha256").is_file()


def test_local_launcher_rejects_wrong_or_dirty_source_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "never-created-source"

    def reject(_root: Path, _revision: str) -> str:
        raise launcher.LocalGate0Error("tracked worktree/index must be clean")

    monkeypatch.setattr(launcher, "_validate_source_identity", reject)
    with pytest.raises(launcher.LocalGate0Error, match="tracked worktree/index"):
        launcher.run_local_gate0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id="source-check",
            source_revision="4" * 40,
            python_executable=sys.executable,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("session_id", "revision", "match"),
    [
        ("bad session", "3" * 40, "session_id"),
        ("valid-session", "ABC", "source_revision"),
    ],
)
def test_local_launcher_rejects_ambiguous_identity_before_creating_output(
    tmp_path: Path, session_id: str, revision: str, match: str
) -> None:
    launcher = _load_launcher()
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "never-created"
    with pytest.raises(launcher.LocalGate0Error, match=match):
        launcher.run_local_gate0(
            asset_root=assets,
            manifest_path=MANIFEST,
            output_dir=output,
            session_id=session_id,
            source_revision=revision,
            python_executable=sys.executable,
        )
    assert not output.exists()
