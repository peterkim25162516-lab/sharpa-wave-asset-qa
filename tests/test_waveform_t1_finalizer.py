from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample
from wave_asset_qa.parity.bundle import verify_bundle
from wave_asset_qa.parity.compare import (
    CollectedRun,
    ComparisonThresholds,
    Gate0Comparison,
)
from wave_asset_qa.parity.contracts import (
    ComparisonRecord,
    ComparisonStatus,
    ExecutionRecord,
    ExecutionStatus,
    HandSide,
    ParityResult,
    Simulator,
)
from wave_asset_qa.parity.scenarios import (
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "waveform_t1.json"
REVISION = "1" * 40
SOURCE_TREE = "2" * 40
ARCHIVE_SHA256 = "3" * 64
SNAPSHOT_SHA256 = "4" * 64


def _load_finalizer():
    path = ROOT / "scripts" / "finalize_waveform_t1.py"
    spec = importlib.util.spec_from_file_location("test_finalize_waveform_t1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inventory(root: Path, inventory: Path | None = None) -> Path:
    target = inventory or root / "evidence.sha256"
    members = sorted(
        (path for path in root.rglob("*") if path.is_file() and path != target),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in members
        ),
        encoding="utf-8",
    )
    return target


def _metric_values(crosssim_joint: float = 0.03) -> dict[str, float]:
    return {
        "crosssim_joint_max_abs_rad": crosssim_joint,
        "crosssim_frame_position_max_m": 0.0,
        "crosssim_frame_orientation_max_rad": 0.0,
        "repeat_joint_max_abs_rad": 0.0,
        "repeat_frame_position_max_m": 0.0,
        "repeat_frame_orientation_max_rad": 0.0,
        "dt_halving_joint_max_abs_rad": 0.0,
        "dt_halving_frame_position_max_m": 0.0,
        "dt_halving_frame_orientation_max_rad": 0.0,
    }


def _ovphysx_effort_provenance(
    observation_count: int,
    *,
    clip_count: int = 0,
    saturated: bool = False,
) -> dict[str, object]:
    computed = [0.5] * 22
    applied = [0.5] * 22
    if saturated:
        computed[0] = 3.0
        applied[0] = 2.0
    return {
        "controller_dof_effort_limit": [2.0] * 22,
        "computed_effort_peak_abs_nm": computed,
        "applied_effort_peak_abs_nm": applied,
        "effort_observation_count": observation_count,
        "effort_clip_count": clip_count,
        "effort_command_source": (
            "articulation_data_computed_and_applied_torque_torch"
        ),
    }


def _comparison(
    *,
    crosssim_joint: float = 0.03,
    status: ComparisonStatus = ComparisonStatus.DIVERGENT,
) -> Gate0Comparison:
    manifest = load_manifest(MANIFEST_PATH)
    executions = tuple(
        ExecutionRecord(
            simulator=backend,
            execution_status=ExecutionStatus.COMPLETED,
            completed_repeats=2,
            requested_repeats=2,
            finite=True,
        )
        for backend in (Simulator.MUJOCO, Simulator.OVPHYSX)
    )
    results = tuple(
        ParityResult(
            schema_version=1,
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest_sha256(manifest),
            hand=hand,
            scenario_id=scenario,
            executions=executions,  # type: ignore[arg-type]
            comparison=ComparisonRecord(
                comparison_status=status,
                metrics=_metric_values(crosssim_joint),
                message="synthetic threshold observation",
            ),
        )
        for hand in (HandSide.LEFT, HandSide.RIGHT)
        for scenario in ("offset_sine", "offset_linear_chirp")
    )
    return Gate0Comparison(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest_sha256(manifest),
        upstream_repository=manifest.provenance.repository,
        upstream_commit=manifest.provenance.commit,
        asset_git_tree=manifest.provenance.asset_git_tree,
        canonical_lf_asset_tree_sha256=(
            manifest.provenance.canonical_lf_asset_tree_sha256
        ),
        expected_joint_mapping_count=44,
        expected_distal_frame_mapping_count=10,
        observed_joint_mapping_count=44,
        observed_distal_frame_mapping_count=10,
        expected_run_count=32,
        received_run_count=32,
        completed_run_count=32,
        completion_fraction=1.0,
        step_completion_fraction=1.0,
        nonfinite_run_count=0,
        execution_status=ExecutionStatus.COMPLETED,
        comparison_status=status,
        thresholds=ComparisonThresholds(minimum_completion_fraction=0.99),
        results=results,
    )


def _minimal_runs() -> tuple[CollectedRun, ...]:
    manifest = load_manifest(MANIFEST_PATH)
    runs: list[CollectedRun] = []
    for case in expand_scenario_cases(manifest):
        hand = manifest.hand(case.hand)
        requested_steps = round(
            manifest.scenario(case.scenario_id).duration_s / case.dt_s
        )
        provenance = (
            _ovphysx_effort_provenance(requested_steps + 1)
            if case.simulator is Simulator.OVPHYSX
            else {}
        )
        runs.append(
            CollectedRun(
                case=case,
                result=AdapterRunResult(
                    backend=case.simulator.value,
                    scenario_id=case.scenario_id,
                    status="completed",
                    message="small synthetic finalizer fixture",
                    dt=case.dt_s,
                    requested_steps=requested_steps,
                    completed_steps=requested_steps,
                    joint_names=hand.joint_names,
                    frame_names=hand.distal_frame_names,
                    samples=(),
                    provenance=provenance,
                ),
            )
        )
    return tuple(runs)


def _local_runtime_run(
    finalizer: object,
    runtime_identity: dict[str, object],
) -> CollectedRun:
    manifest = load_manifest(MANIFEST_PATH)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.MUJOCO
    )
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    requested_steps = round(scenario.duration_s / case.dt_s)
    provenance = {
        "manifest_sha256": manifest_sha256(manifest),
        "session_id": "local-runtime-fixture",
        "source_revision": REVISION,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "mapping_schema_version": 1,
        "backend_version": runtime_identity["mujoco_version"],
        "python_version": runtime_identity["python_version"],
        "platform": runtime_identity["platform"],
        "local_runtime_identity": dict(runtime_identity),
        "target_sequence_digest_schema_version": 1,
        "target_sequence_digest_encoding": finalizer.TARGET_DIGEST_ENCODING,
        "target_sequence_digest_projection": finalizer.TARGET_DIGEST_PROJECTION,
        "target_sequence_canonical_joint_names": list(hand.joint_names),
        "scheduled_target_sequence_semantics": finalizer.SCHEDULED_TARGET_SEMANTICS,
        "scheduled_target_sequence_count": requested_steps + 1,
        "scheduled_target_sequence_sha256": canonical_target_sequence_sha256(
            scenario,
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=True,
        ),
    }
    return CollectedRun(
        case=case,
        result=AdapterRunResult(
            backend="mujoco",
            scenario_id=case.scenario_id,
            status="completed",
            message="local runtime identity fixture",
            dt=case.dt_s,
            requested_steps=requested_steps,
            completed_steps=requested_steps,
            joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
            samples=(),
            provenance=provenance,
        ),
    )


def _audit_runs() -> tuple[CollectedRun, ...]:
    """Tiny traces whose halved-dt discrepancy is the registered maximum."""

    manifest = load_manifest(MANIFEST_PATH)
    runs: list[CollectedRun] = []
    for case in expand_scenario_cases(manifest):
        hand = manifest.hand(case.hand)
        value = (
            0.5
            if case.simulator is Simulator.OVPHYSX
            and case.timestep_variant.value == "halved"
            else 0.1
            if case.simulator is Simulator.OVPHYSX
            else 0.0
        )
        positions = {name: value for name in hand.joint_names}
        targets = {name: 0.0 for name in hand.joint_names}
        poses = {
            name: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            for name in hand.distal_frame_names
        }
        sample = TraceSample(
            step=0,
            time_s=0.0,
            qpos=tuple(positions.values()),
            qvel=(0.0,) * len(hand.joint_names),
            joint_positions=positions,
            frame_poses=poses,
            position_targets=targets,
            contact_count=0,
        )
        runs.append(
            CollectedRun(
                case=case,
                result=AdapterRunResult(
                    backend=case.simulator.value,
                    scenario_id=case.scenario_id,
                    status="completed",
                    message="small crosssim coverage fixture",
                    dt=case.dt_s,
                    requested_steps=0,
                    completed_steps=0,
                    joint_names=hand.joint_names,
                    frame_names=hand.distal_frame_names,
                    samples=(sample,),
                    provenance={},
                ),
            )
        )
    return tuple(runs)


def test_manifest_contract_is_exact_32_case_t1() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)

    cases = finalizer._validate_manifest_contract(manifest)

    assert len(cases) == 32
    assert len({case.case_id for case in cases}) == 32
    assert sum(case.simulator is Simulator.MUJOCO for case in cases) == 16
    assert sum(case.simulator is Simulator.OVPHYSX for case in cases) == 16


def test_ovphysx_effort_clipping_is_aggregated_from_all_16_runs() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = list(_minimal_runs())
    index = next(
        position
        for position, run in enumerate(runs)
        if run.case.simulator is Simulator.OVPHYSX
    )
    run = runs[index]
    provenance = dict(run.result.provenance)
    provenance.update(
        _ovphysx_effort_provenance(
            int(provenance["effort_observation_count"]),
            clip_count=7,
            saturated=True,
        )
    )
    runs[index] = replace(
        run,
        result=replace(run.result, provenance=provenance),
    )

    public, private = finalizer._aggregate_ovphysx_effort_clipping(
        manifest, runs
    )

    assert public == {
        "schema_version": 1,
        "backend_case_count": 16,
        "joint_count_per_observation": 22,
        "effort_observation_count": 48_016,
        "effort_clip_count": 7,
    }
    assert private["case_count_with_effort_clipping"] == 1
    assert len(private["cases"]) == 16
    clipped = [
        record for record in private["cases"] if record["effort_clip_count"] > 0
    ]
    assert len(clipped) == 1
    assert clipped[0]["peak_saturated_joint_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_count", "non-negative integer"),
        ("boolean_count", "non-negative integer"),
        ("wrong_observation_count", "coverage is not exact"),
        ("impossible_count", "exceeds observed"),
        ("self_reported_clip", "contradicts observed torque peaks"),
        ("hidden_peak_clip", "contradicts observed torque peaks"),
        ("forged_applied_peak", "contradicts clipping"),
    ),
)
def test_ovphysx_effort_clipping_provenance_fails_closed(
    mutation: str,
    match: str,
) -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = list(_minimal_runs())
    index = next(
        position
        for position, run in enumerate(runs)
        if run.case.simulator is Simulator.OVPHYSX
    )
    run = runs[index]
    provenance = dict(run.result.provenance)
    if mutation == "missing_count":
        del provenance["effort_clip_count"]
    elif mutation == "boolean_count":
        provenance["effort_clip_count"] = True
    elif mutation == "wrong_observation_count":
        provenance["effort_observation_count"] = int(
            provenance["effort_observation_count"]
        ) - 1
    elif mutation == "impossible_count":
        provenance["effort_clip_count"] = (
            int(provenance["effort_observation_count"]) * 22 + 1
        )
    elif mutation == "self_reported_clip":
        provenance["effort_clip_count"] = 1
    elif mutation == "hidden_peak_clip":
        provenance.update(
            _ovphysx_effort_provenance(
                int(provenance["effort_observation_count"]),
                clip_count=0,
                saturated=True,
            )
        )
    else:
        provenance["applied_effort_peak_abs_nm"] = [0.25] * 22
    runs[index] = replace(
        run,
        result=replace(run.result, provenance=provenance),
    )

    with pytest.raises(finalizer.WaveformT1FinalizationError, match=match):
        finalizer._aggregate_ovphysx_effort_clipping(manifest, runs)


def test_exact_inventory_rejects_extra_or_tampered_members(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "evidence"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "nested" / "b.txt").write_text("b\n", encoding="utf-8")
    inventory = _write_inventory(root)

    verified = finalizer._verify_evidence_inventory(root, inventory, "fixture")

    assert verified["file_count"] == 2
    assert len(verified["root_sha256"]) == 64

    (root / "extra.txt").write_text("unlisted\n", encoding="utf-8")
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="not exact"):
        finalizer._verify_evidence_inventory(root, inventory, "fixture")
    (root / "extra.txt").unlink()
    (root / "a.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="digest mismatch"):
        finalizer._verify_evidence_inventory(root, inventory, "fixture")


def test_clean_source_rejects_every_nonignored_untracked_file(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> None:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout

    git("init")
    git("config", "user.name", "Waveform T1 test")
    git("config", "user.email", "waveform-t1@example.invalid")
    (repository / ".gitignore").write_text(
        ".venv/\nexternal/\nresults/\n", encoding="utf-8"
    )
    (repository / "README.md").write_text("committed\n", encoding="utf-8")
    git("add", ".gitignore", "README.md")
    git("commit", "-m", "fixture")

    for relative in (
        ".venv/injected.py",
        "external/injected.py",
        "results/private/run.log",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored\n", encoding="utf-8")
    finalizer._validate_clean_committed_worktree(repository)

    for relative in (
        "src/injected.py",
        "scripts/injected.py",
        "configs/injected.json",
        "schemas/injected.json",
        "untracked-at-root.txt",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("untracked\n", encoding="utf-8")
        with pytest.raises(
            finalizer.WaveformT1FinalizationError,
            match="untracked non-ignored",
        ):
            finalizer._validate_clean_committed_worktree(repository)
        path.unlink()

    finalizer._validate_clean_committed_worktree(repository)


@pytest.mark.parametrize(
    "relative",
    (
        "src/package/ignored.pyc",
        "scripts/ignored.pyo",
        "scripts/__pycache__",
    ),
)
def test_finalizer_rejects_ignored_source_bytecode(
    tmp_path: Path, relative: str
) -> None:
    finalizer = _load_finalizer()
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    artifact = tmp_path / relative
    if artifact.name == "__pycache__":
        artifact.mkdir(parents=True)
    else:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"ignored bytecode")

    with pytest.raises(finalizer.WaveformT1FinalizationError, match=r"no \.pyc"):
        finalizer._reject_source_bytecode(tmp_path)


def test_finalizer_bytecode_guard_precedes_package_imports() -> None:
    source = (ROOT / "scripts" / "finalize_waveform_t1.py").read_text(
        encoding="utf-8"
    )
    bootstrap = source.index('if __name__ == "__main__":')

    assert bootstrap < source.index("\nimport mujoco\n")
    assert bootstrap < source.index("\nimport wave_asset_qa\n")
    assert source.index('os.environ["PYTHONDONTWRITEBYTECODE"] = "1"') < source.index(
        "\nimport mujoco\n"
    )


def test_finalizer_runtime_is_frozen_and_run_hash_forgery_is_rejected() -> None:
    finalizer = _load_finalizer()
    runtime_identity = finalizer._local_runtime_identity()
    assert runtime_identity == {
        "schema_version": 1,
        "python_version": "3.12.6",
        "mujoco_version": "3.12.0",
        "numpy_version": "2.5.2",
        "python_implementation": "CPython",
        "platform": "Windows-11-10.0.26200-SP0",
        "machine": "AMD64",
        "python_executable_sha256": (
            "3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238"
        ),
    }
    run = _local_runtime_run(finalizer, runtime_identity)
    manifest = load_manifest(MANIFEST_PATH)
    finalizer._validate_common_run_provenance(
        run,
        manifest,
        session_id="local-runtime-fixture",
        source_revision=REVISION,
        local_runtime_identity=runtime_identity,
    )

    forged_runtime = dict(runtime_identity)
    forged_runtime["python_executable_sha256"] = "0" * 64
    forged_provenance = dict(run.result.provenance)
    forged_provenance["local_runtime_identity"] = forged_runtime
    forged_run = replace(
        run,
        result=replace(run.result, provenance=forged_provenance),
    )
    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="local_runtime_identity",
    ):
        finalizer._validate_common_run_provenance(
            forged_run,
            manifest,
            session_id="local-runtime-fixture",
            source_revision=REVISION,
            local_runtime_identity=runtime_identity,
        )

    forged_platform = dict(run.result.provenance)
    forged_platform["platform"] = "Windows-forged"
    platform_run = replace(
        run,
        result=replace(run.result, provenance=forged_platform),
    )
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="platform"):
        finalizer._validate_common_run_provenance(
            platform_run,
            manifest,
            session_id="local-runtime-fixture",
            source_revision=REVISION,
            local_runtime_identity=runtime_identity,
        )


def test_finalizer_rejects_dependency_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(finalizer.mujoco, "__version__", "3.12.1")
    with pytest.raises(
        finalizer.WaveformT1FinalizationError, match="mujoco_version"
    ):
        finalizer._local_runtime_identity()


def test_finalizer_rejects_interpreter_sha256_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(
        finalizer, "EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256", "0" * 64
    )
    with pytest.raises(
        finalizer.WaveformT1FinalizationError, match="SHA-256 drifted"
    ):
        finalizer._local_runtime_identity()


def test_known_kitless_logs_are_retained_and_counted(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "evidence"
    root.mkdir()
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    stdout.write_text(
        "[Warning] [carb] getPluginDesc: Failed to find a plugin with a name: "
        "omni.physx.plugin.\n"
        "[Warning] [omni.physx.cooking.plugin] Failed to create a valid UJITSO "
        "Cooking Compute Service\n"
        "[Warning] [omni_physx_sdk] [UNLOAD #1] retained shutdown warning\n"
        "Warning: direct MaterialBindingAPI use is retained\n"
        "Status: OmniUsdResolver initialized\n"
        "[Error] [omni.physx.cooking.plugin] registry is false.\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")
    inventory = _write_inventory(root)
    finalizer._verify_evidence_inventory(root, inventory, "known-warning fixture")

    summary = finalizer._summarize_runtime_log_records(
        [
            finalizer._audit_runtime_log_file(
                stdout,
                label="known/stdout",
                scope="ovphysx_case",
                stream="stdout",
            ),
            finalizer._audit_runtime_log_file(
                stderr,
                label="known/stderr",
                scope="ovphysx_case",
                stream="stderr",
            ),
        ]
    )

    assert summary["fatal_match_count"] == 0
    assert summary["warning_line_count"] == 4
    assert summary["known_kitless_diagnostic_line_count"] == 6
    assert summary["unclassified_warning_line_count"] == 0
    assert summary["known_kitless_counts"] == {
        "missing_omni_physx_plugin": 1,
        "ujitso_cooking_service": 1,
        "ovphysx_sdk_unload_warning": 1,
        "material_binding_api_warning": 1,
        "usd_resolver_status": 1,
        "cooking_registry_diagnostic": 1,
    }


@pytest.mark.parametrize(
    "fatal_text",
    (
        "tRaCe_BaCk (most recent call last):\n",
        "CUDA_ERROR_LAUNCH_FAILED\n",
        "CUDA runtime failure: device unavailable\n",
        "allocator reports Out-Of-Memory\n",
        "worker terminated: OOM\n",
    ),
)
def test_fatal_log_tamper_remains_fatal_after_inventory_resign(
    tmp_path: Path,
    fatal_text: str,
) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "evidence"
    root.mkdir()
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    stdout.write_text("completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    inventory = _write_inventory(root)
    finalizer._verify_evidence_inventory(root, inventory, "clean fixture")

    stderr.write_text(fatal_text, encoding="utf-8")
    _write_inventory(root, inventory)
    # Re-signing makes the exact-inventory gate pass; the frozen semantic log
    # gate must still reject the fatal execution evidence.
    finalizer._verify_evidence_inventory(root, inventory, "re-signed fixture")
    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="fatal runtime log diagnostic",
    ):
        finalizer._audit_runtime_log_file(
            stderr,
            label="tampered/stderr",
            scope="ovphysx_case",
            stream="stderr",
        )


def test_runtime_log_gate_is_wired_to_every_frozen_execution_log() -> None:
    source = (ROOT / "scripts" / "finalize_waveform_t1.py").read_text(
        encoding="utf-8"
    )

    assert 'scope="mujoco_case"' in source
    assert '("launcher.stdout.log", "stdout")' in source
    assert '("launcher.stderr.log", "stderr")' in source
    assert 'scope="ovphysx_schema"' in source
    assert '("stdout", "stdout")' in source
    assert '("stderr", "stderr")' in source
    assert 'scope="ovphysx_case"' in source
    assert '("launcher.stdout.txt", "stdout")' in source
    assert '("launcher.stderr.txt", "stderr")' in source
    assert 'WORKER_STDOUT_SUFFIX' in source
    assert 'WORKER_STDERR_SUFFIX' in source


def test_remote_root_inventory_can_live_below_launcher(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "remote"
    (root / "launcher").mkdir(parents=True)
    (root / "cases" / "one").mkdir(parents=True)
    (root / "launcher" / "status.json").write_text("{}\n", encoding="utf-8")
    (root / "cases" / "one" / "run.txt").write_text("run\n", encoding="utf-8")
    inventory = _write_inventory(root, root / "launcher" / "evidence.sha256")

    verified = finalizer._verify_evidence_inventory(root, inventory, "remote root")

    assert verified["file_count"] == 2


def test_worker_process_identity_is_kernel_bound_and_tamper_evident() -> None:
    finalizer = _load_finalizer()
    identity = {
        "platform": "windows",
        "pid": 1234,
        "process_creation_filetime": 5678,
    }
    record = {
        "schema_version": 1,
        "worker_pid": 1234,
        "os_process_identity": identity,
        "fresh_process_id": finalizer.os_process_identity_sha256(identity),
    }

    assert finalizer._validated_worker_process_record(record, "worker") == record

    forged = dict(record)
    forged["fresh_process_id"] = "0" * 64
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="not bound"):
        finalizer._validated_worker_process_record(forged, "worker")


def test_remote_source_archive_and_virtual_snapshot_are_locally_reproduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as bundle:
        payload = b"print('pinned')\n"
        member = tarfile.TarInfo("scripts/example.py")
        member.size = len(payload)
        member.mode = 0o755
        bundle.addfile(member, io.BytesIO(payload))
    archive = stream.getvalue()
    archive_sha = sha256(archive).hexdigest()
    snapshot_sha = finalizer._virtual_snapshot_sha256(
        archive,
        source_revision=REVISION,
        source_tree=SOURCE_TREE,
        archive_sha256=archive_sha,
    )
    monkeypatch.setattr(finalizer, "_git_archive_bytes", lambda *_args: archive)

    assert finalizer._validate_remote_source_snapshot(
        ROOT,
        source_revision=REVISION,
        source_tree=SOURCE_TREE,
        remote_archive_sha256=archive_sha,
        remote_snapshot_sha256=snapshot_sha,
    ) == {
        "source_archive_sha256": archive_sha,
        "source_snapshot_sha256": snapshot_sha,
    }

    with pytest.raises(
        finalizer.WaveformT1FinalizationError, match="exact local execution-commit archive"
    ):
        finalizer._validate_remote_source_snapshot(
            ROOT,
            source_revision=REVISION,
            source_tree=SOURCE_TREE,
            remote_archive_sha256="f" * 64,
            remote_snapshot_sha256=snapshot_sha,
        )
    with pytest.raises(
        finalizer.WaveformT1FinalizationError, match="locally reproduced installer snapshot"
    ):
        finalizer._validate_remote_source_snapshot(
            ROOT,
            source_revision=REVISION,
            source_tree=SOURCE_TREE,
            remote_archive_sha256=archive_sha,
            remote_snapshot_sha256="e" * 64,
        )


def test_remote_fresh_process_identity_binds_kernel_pid_hash_and_case() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.OVPHYSX
    )
    identity = {
        "platform": "posix",
        "boot_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "pid": 4321,
        "process_start_ticks": 987654,
    }
    run = CollectedRun(
        case=case,
        result=AdapterRunResult(
            backend="ovphysx",
            scenario_id=case.scenario_id,
            status="completed",
            message="identity fixture",
            dt=case.dt_s,
            requested_steps=0,
            completed_steps=0,
            provenance={"worker_pid": 4321},
        ),
    )
    record = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": case.case_id,
        "worker_command_identity": {
            "python_realpath": (
                "/data/home/exampleuser/sharpa-wave-asset-qa-gate0/"
                "toolchain/python/bin/python3.12"
            ),
            "script_realpath": (
                "/data/home/exampleuser/sharpa-wave-asset-qa-gate0/project/"
                f"{SOURCE_TREE}/scripts/probe_ovphysx_runtime.py"
            ),
        },
        "os_process_identity": identity,
        "fresh_process_id": finalizer.os_process_identity_sha256(identity),
    }

    assert finalizer._validate_remote_fresh_process_record(
        record, case=case, run=run, source_tree=SOURCE_TREE
    ) == record["fresh_process_id"]

    forged = dict(record)
    forged["fresh_process_id"] = "0" * 64
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="hash is invalid"):
        finalizer._validate_remote_fresh_process_record(
            forged, case=case, run=run, source_tree=SOURCE_TREE
        )


def test_remote_case_topology_rejects_unknown_or_nonsequential_artifacts(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    case = next(
        item
        for item in expand_scenario_cases(manifest)
        if item.simulator is Simulator.OVPHYSX
    )
    case_dir = tmp_path / case.case_id
    worker_inputs = case_dir / "worker-inputs"
    worker_inputs.mkdir(parents=True)
    required = {
        f"{case.case_id}.run.json",
        f"{case.case_id}.worker.result.json",
        f"{case.case_id}.worker.stdout.log",
        f"{case.case_id}.worker.stderr.log",
        f"{case.case_id}.worker.exitcode.txt",
        "evidence.sha256",
        "gpu-monitor.txt",
        "launcher.exit-code.txt",
        "launcher.stderr.txt",
        "launcher.stdout.txt",
        "owned-process-group.txt",
        "preflight.txt",
        "private-fresh-process-identity.json",
        "postflight-01.txt",
    }
    for name in required:
        (case_dir / name).write_text("fixture\n", encoding="utf-8")
    (worker_inputs / f"{manifest_sha256(manifest)}.manifest.json").write_text(
        MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    assert [path.name for path in finalizer._validate_remote_case_topology(
        case_dir, case, manifest
    )] == ["postflight-01.txt"]

    (case_dir / "unknown.tmp").write_text("forged\n", encoding="utf-8")
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="unknown"):
        finalizer._validate_remote_case_topology(case_dir, case, manifest)
    (case_dir / "unknown.tmp").unlink()
    (case_dir / "postflight-03.txt").write_text("gap\n", encoding="utf-8")
    with pytest.raises(
        finalizer.WaveformT1FinalizationError, match="topology is not exact"
    ):
        finalizer._validate_remote_case_topology(case_dir, case, manifest)


def test_remote_launcher_topology_is_exact_for_both_schema_probes(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    remote = tmp_path / "remote"
    launcher = remote / "launcher"
    (launcher / "schema").mkdir(parents=True)
    for name in finalizer.REMOTE_LAUNCHER_REQUIRED_FILES:
        (launcher / name).write_text("fixture\n", encoding="utf-8")
    for hand in ("left", "right"):
        for name in (
            f"schema-{hand}-gpu-monitor.txt",
            f"schema-{hand}-owned-process-group.txt",
            f"schema-{hand}-preflight.txt",
            f"schema-{hand}-postflight-01.txt",
        ):
            (launcher / name).write_text("fixture\n", encoding="utf-8")

    observed = finalizer._validate_remote_launcher_topology(
        remote, "launcher/schema"
    )

    assert [path.name for path in observed["left"]] == [
        "schema-left-postflight-01.txt"
    ]
    assert [path.name for path in observed["right"]] == [
        "schema-right-postflight-01.txt"
    ]

    (launcher / "unknown.txt").write_text("forged\n", encoding="utf-8")
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="unknown"):
        finalizer._validate_remote_launcher_topology(remote, "launcher/schema")
    (launcher / "unknown.txt").unlink()
    (launcher / "schema-left-postflight-03.txt").write_text(
        "gap\n", encoding="utf-8"
    )
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="consecutive"):
        finalizer._validate_remote_launcher_topology(remote, "launcher/schema")


def test_gpu_monitor_records_are_bound_to_owned_group_and_gpu(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    monitor = tmp_path / "gpu-monitor.txt"
    gpu_uuid = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
    gpu_name = "NVIDIA A800-SXM4-40GB"
    monitor.write_text(
        "recorded_at_utc=2026-08-30T00:00:00Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 100, 2\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        f"{gpu_uuid}, 9876, 50\n"
        "compute_processes_end\n",
        encoding="utf-8",
    )

    finalizer._validate_gpu_monitor(
        monitor,
        label="fixture monitor",
        owner_pgid=4321,
        gpu_index="0",
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version="570.158.01",
    )

    monitor.write_text(
        monitor.read_text(encoding="utf-8").replace(gpu_uuid, "GPU-forged", 1),
        encoding="utf-8",
    )
    with pytest.raises(finalizer.WaveformT1FinalizationError, match="GPU identity"):
        finalizer._validate_gpu_monitor(
            monitor,
            label="fixture monitor",
            owner_pgid=4321,
            gpu_index="0",
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version="570.158.01",
        )


def test_gpu_monitor_accepts_canonical_empty_compute_process_envelope(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    monitor = tmp_path / "gpu-monitor.txt"
    gpu_uuid = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
    gpu_name = "NVIDIA A800-SXM4-40GB"
    monitor.write_text(
        "recorded_at_utc=2026-08-30T00:00:00Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 4, 0\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        "compute_processes_end\n",
        encoding="utf-8",
    )

    finalizer._validate_gpu_monitor(
        monitor,
        label="empty fixture monitor",
        owner_pgid=4321,
        gpu_index="0",
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version="570.158.01",
    )


def test_gpu_monitor_accepts_legacy_single_blank_empty_process_row(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    monitor = tmp_path / "gpu-monitor.txt"
    gpu_uuid = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
    gpu_name = "NVIDIA A800-SXM4-40GB"
    monitor.write_text(
        "recorded_at_utc=2026-08-30T00:00:00Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 4, 0\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        "\n"
        "compute_processes_end\n",
        encoding="utf-8",
    )

    finalizer._validate_gpu_monitor(
        monitor,
        label="legacy empty-row fixture monitor",
        owner_pgid=4321,
        gpu_index="0",
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version="570.158.01",
    )


def test_gpu_monitor_legacy_empty_record_does_not_desynchronize_next_record(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    monitor = tmp_path / "gpu-monitor.txt"
    gpu_uuid = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
    gpu_name = "NVIDIA A800-SXM4-40GB"
    monitor.write_text(
        "recorded_at_utc=2026-08-30T00:00:00Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 4, 0\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        "\n"
        "compute_processes_end\n"
        "recorded_at_utc=2026-08-30T00:00:01Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 12, 1\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        f"{gpu_uuid}, 4321, 8\n"
        "compute_processes_end\n",
        encoding="utf-8",
    )

    finalizer._validate_gpu_monitor(
        monitor,
        label="two-record fixture monitor",
        owner_pgid=4321,
        gpu_index="0",
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        driver_version="570.158.01",
    )


def test_gpu_monitor_rejects_whitespace_only_compute_process_row(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    monitor = tmp_path / "gpu-monitor.txt"
    gpu_uuid = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
    gpu_name = "NVIDIA A800-SXM4-40GB"
    monitor.write_text(
        "recorded_at_utc=2026-08-30T00:00:00Z\n"
        f"gpu=0, {gpu_uuid}, {gpu_name}, 570.158.01, 4, 0\n"
        "owned_case_pid=4321\n"
        "owned_case_pgid=4321\n"
        "compute_processes_begin\n"
        "   \n"
        "compute_processes_end\n",
        encoding="utf-8",
    )

    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="unclassifiable compute process",
    ):
        finalizer._validate_gpu_monitor(
            monitor,
            label="whitespace-row fixture monitor",
            owner_pgid=4321,
            gpu_index="0",
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version="570.158.01",
        )


@pytest.mark.parametrize("gpu_elapsed_s", (0, 1, 7199, 7200))
def test_remote_gpu_elapsed_accepts_only_values_within_two_hours(
    gpu_elapsed_s: int,
) -> None:
    finalizer = _load_finalizer()

    assert finalizer._validate_remote_gpu_elapsed_s(
        {"gpu_elapsed_s": gpu_elapsed_s}
    ) == gpu_elapsed_s


@pytest.mark.parametrize("gpu_elapsed_s", (-1, 7201, True, 1.5, "7200", None))
def test_remote_gpu_elapsed_rejects_invalid_or_over_budget_values(
    gpu_elapsed_s: object,
) -> None:
    finalizer = _load_finalizer()

    with pytest.raises(finalizer.WaveformT1FinalizationError, match="gpu_elapsed_s"):
        finalizer._validate_remote_gpu_elapsed_s({"gpu_elapsed_s": gpu_elapsed_s})


def test_crosssim_audit_requires_base_and_halved_dt_maxima() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = _audit_runs()
    all_variant_comparison = _comparison(crosssim_joint=0.5)

    finalizer._verify_crosssim_covers_all_dt_variants(
        manifest, runs, all_variant_comparison
    )

    base_only_results = tuple(
        replace(
            result,
            comparison=replace(
                result.comparison,
                metrics=_metric_values(crosssim_joint=0.1),
            ),
        )
        for result in all_variant_comparison.results
    )
    base_only = replace(all_variant_comparison, results=base_only_results)
    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="do not cover both base and halved dt",
    ):
        finalizer._verify_crosssim_covers_all_dt_variants(manifest, runs, base_only)


def test_crosssim_audit_skips_empty_metrics_only_when_inconclusive() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = _audit_runs()
    comparison = _comparison(status=ComparisonStatus.INCONCLUSIVE)
    inconclusive_results = tuple(
        replace(
            result,
            comparison=replace(result.comparison, metrics={}),
        )
        for result in comparison.results
    )

    finalizer._verify_crosssim_covers_all_dt_variants(
        manifest,
        runs,
        replace(comparison, results=inconclusive_results),
    )

    divergent = _comparison(status=ComparisonStatus.DIVERGENT)
    divergent_results = tuple(
        replace(
            result,
            comparison=replace(result.comparison, metrics={}),
        )
        for result in divergent.results
    )
    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="must include metrics unless it is inconclusive",
    ):
        finalizer._verify_crosssim_covers_all_dt_variants(
            manifest,
            runs,
            replace(divergent, results=divergent_results),
        )


def test_crosssim_audit_rejects_partial_metrics() -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = _audit_runs()
    comparison = _comparison()
    partial_results = tuple(
        replace(
            result,
            comparison=replace(
                result.comparison,
                metrics={"crosssim_joint_max_abs_rad": 0.5},
            ),
        )
        for result in comparison.results
    )

    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="comparison metrics are missing",
    ):
        finalizer._verify_crosssim_covers_all_dt_variants(
            manifest,
            runs,
            replace(comparison, results=partial_results),
        )


def test_finalizer_builds_deterministic_private_bundle_and_two_public_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    manifest = load_manifest(MANIFEST_PATH)
    runs = _minimal_runs()
    local_runtime_log = tmp_path / "mock-local-runtime.log"
    remote_runtime_log = tmp_path / "mock-remote-runtime.log"
    local_runtime_log.write_text("completed\n", encoding="utf-8")
    remote_runtime_log.write_text(
        "[Warning] [carb] getPluginDesc: Failed to find a plugin with a name: "
        "omni.physx.plugin.\n",
        encoding="utf-8",
    )
    local_runtime_log_audit = finalizer._summarize_runtime_log_records(
        [
            finalizer._audit_runtime_log_file(
                local_runtime_log,
                label="mujoco/mock/stdout",
                scope="mujoco_case",
                stream="stdout",
            )
        ]
    )
    remote_runtime_log_audit = finalizer._summarize_runtime_log_records(
        [
            finalizer._audit_runtime_log_file(
                remote_runtime_log,
                label="ovphysx/mock/stderr",
                scope="ovphysx_case",
                stream="stderr",
            )
        ]
    )
    evidence = finalizer._EvidenceSet(
        runs=runs,
        local_identity={
            "session_id": "private-local-session",
            "launcher_pid": 111,
            "fresh_process_count": 16,
            "runtime_log_audit": local_runtime_log_audit,
        },
        remote_identity={
            "run_id": "private-remote-run",
            "session_id": "private-remote-session",
            "gpu_uuid": "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625",
            "fresh_process_count": 16,
            "runtime_log_audit": remote_runtime_log_audit,
        },
        schema_relative_path="launcher/schema",
    )
    comparison = _comparison()
    compare_calls: list[ComparisonThresholds] = []

    monkeypatch.setattr(
        finalizer,
        "_validate_source_identity",
        lambda _root, observed_revision: (
            SOURCE_TREE if observed_revision == REVISION else pytest.fail("wrong revision")
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_remote_source_snapshot",
        lambda *_args, **_kwargs: {
            "source_archive_sha256": ARCHIVE_SHA256,
            "source_snapshot_sha256": SNAPSHOT_SHA256,
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_and_load_evidence",
        lambda *_args, **_kwargs: evidence,
    )

    def fake_compare(_manifest, observed_runs, *, thresholds):
        assert _manifest == manifest
        assert tuple(observed_runs) == runs
        compare_calls.append(thresholds)
        return comparison

    monkeypatch.setattr(finalizer, "compare_gate0_runs", fake_compare)
    monkeypatch.setattr(
        finalizer,
        "_verify_crosssim_covers_all_dt_variants",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        finalizer,
        "_build_descriptive_metrics",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "decision_use": "descriptive_only_no_additional_thresholds",
            "cells": [{"joint_rmse_rad": 0.01}],
        },
    )

    local = tmp_path / "local-evidence"
    remote = tmp_path / "remote-evidence"
    local.mkdir()
    remote.mkdir()
    (local / "fixture.txt").write_text("local private evidence\n", encoding="utf-8")
    (remote / "fixture.txt").write_text("remote private evidence\n", encoding="utf-8")

    roots: list[str] = []
    public_bytes: list[tuple[bytes, bytes]] = []
    for suffix in ("a", "b"):
        destination, verification = finalizer.finalize_waveform_t1(
            manifest_path=MANIFEST_PATH,
            local_evidence_root=local,
            remote_evidence_root=remote,
            output_dir=tmp_path / f"private-{suffix}",
            public_output_dir=tmp_path / f"public-{suffix}",
            source_revision=REVISION,
            source_archive_sha256=ARCHIVE_SHA256,
            remote_snapshot_sha256=SNAPSHOT_SHA256,
            created_at_utc="2026-08-30T00:00:00Z",
        )
        bundle = verify_bundle(destination)
        roots.append(str(bundle["root_sha256"]))
        assert verification["validation_status"] == "VALID"
        assert verification["scientific_label"] == "DIVERGENT"
        assert verification["formal_gate0_comparison_status"] == "DIVERGENT"
        assert verification["formal_gate0_pass_ready"] is False
        assert verification["runtime_log_file_count"] == 2
        assert verification["runtime_log_fatal_match_count"] == 0
        assert verification["runtime_known_kitless_diagnostic_line_count"] == 1
        assert verification["ovphysx_effort_observation_count"] == 48_016
        assert verification["ovphysx_effort_clip_count"] == 0
        public = tmp_path / f"public-{suffix}"
        assert {path.name for path in public.iterdir()} == {"summary.json", "report.md"}
        summary = json.loads((public / "summary.json").read_text(encoding="utf-8"))
        assert summary["validation_status"] == "VALID"
        assert summary["scientific_label"] == "DIVERGENT"
        assert summary["formal_gate0"] == {
            "comparison_status": "DIVERGENT",
            "pass_ready": False,
            "unchanged": True,
        }
        sanitized = summary["provenance"]["sanitized_execution"]
        assert sanitized["runtime_log_file_count"] == 2
        assert sanitized["runtime_fatal_match_count"] == 0
        assert sanitized["runtime_known_kitless_diagnostic_line_count"] == 1
        assert sanitized["runtime_known_kitless_counts"] == {
            "cooking_registry_diagnostic": 0,
            "material_binding_api_warning": 0,
            "missing_omni_physx_plugin": 1,
            "ovphysx_sdk_unload_warning": 0,
            "ujitso_cooking_service": 0,
            "usd_resolver_status": 0,
        }
        assert sanitized["ovphysx_effort_clipping_observation"] == {
            "schema_version": 1,
            "backend_case_count": 16,
            "joint_count_per_observation": 22,
            "effort_observation_count": 48_016,
            "effort_clip_count": 0,
        }
        assert summary["ovphysx_effort_saturation"] == {
            "backend": "kit-less ovphysx",
            "backend_case_count": 16,
            "effort_observation_count": 48_016,
            "joint_effort_value_count": 1_056_352,
            "observed_effort_clip_count": 0,
            "saturation_observed": False,
            "response_regime": "NO_EFFORT_CLIPPING_OBSERVED",
            "linear_transfer_function_claimed": False,
        }
        finalization = json.loads(
            (destination / "private" / "finalization.json").read_text(
                encoding="utf-8"
            )
        )
        assert finalization["runtime_log_audit"]["file_count"] == 2
        assert finalization["runtime_log_audit"]["fatal_match_count"] == 0
        assert finalization["ovphysx_effort_clipping_audit"][
            "effort_observation_count"
        ] == 48_016
        assert finalization["ovphysx_effort_clipping_audit"][
            "effort_clip_count"
        ] == 0
        rendered = (public / "summary.json").read_text(encoding="utf-8") + (
            public / "report.md"
        ).read_text(encoding="utf-8")
        for private_token in (
            "private-local-session",
            "private-remote-session",
            "private-remote-run",
            "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625",
            str(tmp_path),
        ):
            assert private_token not in rendered
        public_bytes.append(
            ((public / "summary.json").read_bytes(), (public / "report.md").read_bytes())
        )

    assert roots[0] == roots[1]
    assert public_bytes[0] == public_bytes[1]
    assert len(compare_calls) == 2
    assert all(item.minimum_completion_fraction == 0.99 for item in compare_calls)


def test_private_bundle_size_hard_limit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    root = tmp_path / "private-staging"
    private = root / "private"
    private.mkdir(parents=True)
    (root / "payload.bin").write_bytes(b"x" * 256)
    finalization = private / "finalization.json"
    monkeypatch.setattr(finalizer, "MAX_PRIVATE_BUNDLE_BYTES", 128)

    with pytest.raises(finalizer.WaveformT1FinalizationError, match="hard limit"):
        finalizer._seal_private_bundle_with_size(
            root,
            finalization_path=finalization,
            finalization_payload={"schema_version": 1},
            created_at_utc="2026-08-30T00:00:00Z",
        )


def test_source_copy_budget_fails_before_oversized_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer = _load_finalizer()
    monkeypatch.setattr(finalizer, "MAX_PRIVATE_BUNDLE_BYTES", 1_000)
    monkeypatch.setattr(finalizer, "PRIVATE_BUNDLE_DERIVED_HEADROOM_BYTES", 100)

    accepted = finalizer._validate_source_copy_budget(
        ({"size": 400},),
        ({"size": 450},),
        manifest_size_bytes=50,
    )
    assert accepted == {
        "local_evidence_bytes": 400,
        "remote_evidence_bytes": 450,
        "manifest_bytes": 50,
        "source_payload_bytes": 900,
        "derived_headroom_bytes": 100,
        "hard_limit_bytes": 1_000,
    }

    with pytest.raises(
        finalizer.WaveformT1FinalizationError,
        match="cannot fit the 2 GiB private-bundle hard limit",
    ):
        finalizer._validate_source_copy_budget(
            ({"size": 401},),
            ({"size": 450},),
            manifest_size_bytes=50,
        )


def test_public_tree_rejects_any_extra_file(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    summary, report = finalizer.build_public_artifacts(
        _comparison(),
        {"schema_version": 1, "cells": [{"joint_rmse_rad": 0.0}]},
        {
            "source_revision": REVISION,
            "ovphysx_effort_clipping_observation": {
                "schema_version": 1,
                "backend_case_count": 16,
                "joint_count_per_observation": 22,
                "effort_observation_count": 48_016,
                "effort_clip_count": 0,
            },
        },
    )
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text(
        finalizer.canonical_summary_json(summary), encoding="utf-8"
    )
    (public / "report.md").write_text(report, encoding="utf-8")
    (public / "private.txt").write_text("must not publish\n", encoding="utf-8")

    with pytest.raises(finalizer.WaveformT1FinalizationError, match="exactly"):
        finalizer._validate_public_tree(public, summary, report)
