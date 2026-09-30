from __future__ import annotations

from dataclasses import replace
import importlib.util
from hashlib import sha256
import json
from pathlib import Path

import pytest

from wave_asset_qa.adapters.base import AdapterRunResult
from wave_asset_qa.parity.bundle import verify_bundle
from wave_asset_qa.parity.compare import CollectedRun
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.runner import write_collected_run
from wave_asset_qa.parity.scenarios import (
    canonical_manifest_json,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "gate0.json"
SESSION = "gate0-finalize-test"
REVISION = "a" * 40
SOURCE_TREE = "b" * 40


def _load_finalizer():
    path = ROOT / "scripts" / "finalize_gate0.py"
    spec = importlib.util.spec_from_file_location("test_finalize_gate0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._validate_source_identity = lambda _root, _revision: SOURCE_TREE
    return module


def _write_evidence(root: Path, simulator: Simulator) -> None:
    manifest = load_manifest(MANIFEST)
    (root / "launcher").mkdir(parents=True)
    if simulator is Simulator.OVPHYSX:
        (root / "launcher" / "raw-launcher.log").write_text(
            f"raw {simulator.value} evidence\n", encoding="utf-8"
        )
    for case in expand_scenario_cases(manifest):
        if case.simulator is not simulator:
            continue
        case_dir = root / "cases" / case.case_id
        case_dir.mkdir(parents=True)
        hand = manifest.hand(case.hand)
        requested_steps = round(
            manifest.scenario(case.scenario_id).duration_s / case.dt_s
        )
        # Empty traces are intentionally report-level execution errors.  The
        # finalizer's job is to preserve and report scientifically valid raw
        # evidence topology, not rewrite a finding into a packaging failure.
        provenance = {
            "manifest_sha256": manifest_sha256(manifest),
            "session_id": SESSION,
            "source_revision": REVISION,
            "asset_tree_sha256": (
                manifest.provenance.canonical_lf_asset_tree_sha256
            ),
            "asset_commit": manifest.provenance.commit,
            "asset_git_tree": manifest.provenance.asset_git_tree,
            "mapping_schema_version": 1,
        }
        if simulator is Simulator.OVPHYSX:
            scenario = manifest.scenario(case.scenario_id)
            target = (
                scenario.target_position_rad
                if scenario.scenario_id == "small_step"
                else 0.0
            )
            provenance.update(
                {
                    "simulation_configuration": {
                        "verified": True,
                        "requested_dt_s": case.dt_s,
                        "cfg_dt_s": case.dt_s,
                        "backend_dt_s": case.dt_s,
                        "requested_gravity_m_s2": list(scenario.gravity_m_s2),
                        "cfg_gravity_m_s2": list(scenario.gravity_m_s2),
                        "physics_scene_gravity_m_s2": list(
                            scenario.gravity_m_s2
                        ),
                        "physics_prim_path": "/physicsScene",
                    },
                    "actuation_contract_version": 2,
                    "actuator_model": "IdealPDActuator",
                    "control_path": "explicit_pd_effort",
                    "controller_dof_stiffness": [1.0] * 22,
                    "controller_dof_damping": [0.1] * 22,
                    "controller_dof_effort_limit": [2.0] * 22,
                    "controller_dof_effort_limit_sim": [3.0] * 22,
                    "controller_parameter_source": "ideal_pd_actuator_tensor",
                    "backend_dof_stiffness": [0.0] * 22,
                    "backend_dof_damping": [0.0] * 22,
                    "backend_dof_drive_readback_source": "root_view_cpu_numpy_binding",
                    "position_target_readback_verified": True,
                    "position_target_readback_source": (
                        "articulation_data_joint_pos_target_torch"
                    ),
                    "zero_velocity_target_verified": True,
                    "zero_feedforward_effort_target_verified": True,
                    "position_target_readback_count": requested_steps + 2,
                    "position_target_readback_max_abs_error_rad": 0.0,
                    "position_target_readback_values_rad": [target] * 22,
                    "position_target_nonzero_readback_observed": (
                        scenario.scenario_id == "small_step"
                    ),
                    "computed_effort_peak_abs_nm": [0.5] * 22,
                    "applied_effort_peak_abs_nm": [0.5] * 22,
                    "effort_observation_count": requested_steps + 1,
                    "effort_formula_max_abs_error_nm": 0.0,
                    "effort_clip_max_abs_error_nm": 0.0,
                    "effort_clip_count": 0,
                    "effort_command_source": (
                        "articulation_data_computed_and_applied_torque_torch"
                    ),
                }
            )
        result = AdapterRunResult(
            backend=simulator.value,
            scenario_id=case.scenario_id,
            status="completed",
            message="synthetic finalizer fixture",
            dt=case.dt_s,
            requested_steps=requested_steps,
            completed_steps=requested_steps,
            joint_names=hand.joint_names,
            frame_names=hand.distal_frame_names,
            samples=(),
            provenance=provenance,
        )
        write_collected_run(case_dir, CollectedRun(case=case, result=result))
        if simulator is Simulator.MUJOCO:
            (case_dir / "launcher.command.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            "<python>",
                            "-P",
                            "-m",
                            "wave_asset_qa.parity.runner",
                            "--backend",
                            "mujoco",
                            "--asset-root",
                            "<asset-root>",
                            "--manifest",
                            "launcher/gate0.manifest.json",
                            "--output-dir",
                            f"cases/{case.case_id}",
                            "--session-id",
                            SESSION,
                            "--source-revision",
                            REVISION,
                            "--case-id",
                            case.case_id,
                        ],
                        "case_id": case.case_id,
                        "schema_version": 1,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (case_dir / "launcher.stdout.log").write_text(
                "runner summary\n", encoding="utf-8"
            )
            (case_dir / "launcher.stderr.log").write_text(
                "raw warning\n", encoding="utf-8"
            )
            (case_dir / "launcher.exitcode.txt").write_text("0\n", encoding="utf-8")
            _rehash_case(case_dir)
        else:
            (case_dir / "raw.stderr.log").write_text(
                "raw warning\n", encoding="utf-8"
            )
            (case_dir / "raw.exitcode.txt").write_text("0\n", encoding="utf-8")
            (case_dir / f"{case.case_id}.worker.result.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (case_dir / f"{case.case_id}.worker.stdout.log").write_text(
                "worker stdout\n", encoding="utf-8"
            )
            (case_dir / f"{case.case_id}.worker.stderr.log").write_text(
                "worker warning\n", encoding="utf-8"
            )
            (case_dir / f"{case.case_id}.worker.exitcode.txt").write_text(
                "0\n", encoding="utf-8"
            )
            (case_dir / "launcher.stdout.txt").write_text(
                "launcher stdout\n", encoding="utf-8"
            )
            (case_dir / "launcher.stderr.txt").write_text(
                "launcher warning\n", encoding="utf-8"
            )
            (case_dir / "launcher.exit-code.txt").write_text(
                "0\n", encoding="utf-8"
            )
            gpu_state = _gpu_state_text()
            (case_dir / "preflight.txt").write_text(gpu_state, encoding="utf-8")
            (case_dir / "postflight-01.txt").write_text(gpu_state, encoding="utf-8")
            _rehash_case(case_dir)
    if simulator is Simulator.MUJOCO:
        _write_local_launcher_evidence(root, manifest)
    else:
        _write_schema_evidence(root, manifest)
        _write_remote_launcher_evidence(root, manifest)
        _rehash_remote_root(root)


GPU_UUID = "GPU-f4c114dd-f889-5df3-9aed-0bc7436185a6"
GPU_NAME = "NVIDIA A800-SXM4-40GB"


def _gpu_state_text() -> str:
    return (
        "recorded_at_utc=2026-08-26T00:00:00Z\n"
        "gpu_index=5\n"
        f"gpu_uuid={GPU_UUID}\n"
        f"gpu_name={GPU_NAME}\n"
        "driver_version=570.158.01\n"
        "memory_used_mib=0\n"
        "utilization_percent=0\n"
        "compute_processes_begin\n"
        "\n"
        "compute_processes_end\n"
    )


def _rehash_case(case_dir: Path) -> None:
    hash_path = case_dir / "evidence.sha256"
    files = sorted(
        (
            path
            for path in case_dir.rglob("*")
            if path.is_file() and path != hash_path
        ),
        key=lambda path: path.relative_to(case_dir).as_posix(),
    )
    hash_path.write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"./{path.relative_to(case_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _write_local_launcher_evidence(root: Path, manifest: object) -> None:
    launcher = root / "launcher"
    manifest_snapshot = launcher / "gate0.manifest.json"
    manifest_snapshot.write_text(
        canonical_manifest_json(manifest) + "\n", encoding="utf-8", newline="\n"
    )
    cases = [
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.MUJOCO
    ]
    common = {
        "schema_version": 1,
        "backend": "mujoco",
        "session_id": SESSION,
        "source_revision": REVISION,
        "source_tree": SOURCE_TREE,
        "launcher_script_sha256": sha256(
            (ROOT / "scripts" / "run_gate0_local.py").read_bytes()
        ).hexdigest(),
        "manifest_file_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "manifest_snapshot_sha256": sha256(manifest_snapshot.read_bytes()).hexdigest(),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
    }
    payloads = {
        "provenance.json": common,
        "matrix.json": {
            **common,
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
        },
        "completed.json": {
            **common,
            "completed_case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
        },
    }
    for name, payload in payloads.items():
        (launcher / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _rehash_case(launcher)


def _write_remote_launcher_evidence(root: Path, manifest: object) -> None:
    launcher = root / "launcher"
    source_tree = SOURCE_TREE
    run_id = "remote-test"
    project_root = ROOT
    provenance = {
        "schema_version": "1",
        "run_id": run_id,
        "session_id": SESSION,
        "source_revision": REVISION,
        "source_tree": source_tree,
        "archive_sha256": "c" * 64,
        "project_path": f"project/{source_tree}",
        "launcher_sha256": sha256(
            (project_root / "scripts" / "run_gate0_remote.sh").read_bytes()
        ).hexdigest(),
        "worker_sha256": sha256(
            (project_root / "scripts" / "probe_ovphysx_runtime.py").read_bytes()
        ).hexdigest(),
        "manifest_file_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_canonical_sha256": manifest_sha256(manifest),
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_lf_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "isaaclab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
        "gpu_index": "5",
        "gpu_uuid": GPU_UUID,
        "gpu_name": GPU_NAME,
        "driver_version": "570.158.01",
        "python": "3.12.14",
        "isaaclab": "6.1.14",
        "isaaclab-ovphysx": "3.0.2",
        "ovphysx": "0.4.13",
        "torch": "2.10.0+cu128",
        "usd-core": "25.11",
        "kitless": "true",
    }
    (launcher / "provenance.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in provenance.items()),
        encoding="utf-8",
    )
    status = {
        "schema_version": 1,
        "status": "completed",
        "scientific_verdict": "pending_local_compare",
        "exit_code": 0,
        "run_id": run_id,
        "session_id": SESSION,
        "source_revision": REVISION,
        "source_tree": source_tree,
        "selected_gpu_index": 5,
        "selected_gpu_uuid": GPU_UUID,
        "completed_case_count": 16,
        "completed_schema_probe_count": 2,
        "last_case_id": None,
        "elapsed_s": 1,
        "gate0_root_bytes": 1,
        "ended_at_utc": "2026-08-26T00:00:01Z",
    }
    (launcher / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (launcher / "gpu-selection.txt").write_text(
        _gpu_state_text(), encoding="utf-8"
    )


def _rehash_remote_root(root: Path) -> None:
    hash_path = root / "launcher" / "evidence.sha256"
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != hash_path
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    hash_path.write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"./{path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _rehash_schema(schema_dir: Path) -> None:
    names = [
        f"{hand}.{suffix}"
        for hand in ("left", "right")
        for suffix in ("json", "stdout", "stderr", "exitcode")
    ]
    (schema_dir / "evidence.sha256").write_text(
        "".join(
            f"{sha256((schema_dir / name).read_bytes()).hexdigest()}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
    )


def _write_schema_evidence(root: Path, manifest: object) -> None:
    schema_dir = root / "launcher" / "schema"
    schema_dir.mkdir()
    manifest_digest = manifest_sha256(manifest)
    for hand in getattr(manifest, "hands"):
        side = hand.side.value
        inspection = {
            "source_path": hand.model_paths.ovphysx,
            "source_sha256": ("1" if side == "left" else "2") * 64,
            "stage_load_policy": "load_all",
            "prim_count": 100,
            "revolute_joint_count": 22,
            "revolute_joint_names": list(hand.joint_names),
            "angular_drive_count": 22,
            "angular_drive_joint_names": list(hand.joint_names),
            "angular_drive_records": [
                {
                    "joint_name": name,
                    "type": "force",
                    "stiffness": 1.0,
                    "damping": 0.1,
                    "max_force": 2.0,
                    "target_position": 0.0,
                }
                for name in hand.joint_names
            ],
            "physx_velocity_joint_count": 22,
            "physx_velocity_joint_names": list(hand.joint_names),
            "physx_velocity_records": [
                {"joint_name": name, "max_joint_velocity": 5.0}
                for name in hand.joint_names
            ],
            "distal_frame_count": 5,
            "distal_frame_names": list(hand.distal_frame_names),
            "articulation_root_paths": [f"/{side}_hand/root_joint"],
        }
        capabilities = {
            "kitless_contract": True,
            "renderer_disabled": True,
            "camera_disabled": True,
            "pinned_ovphysx_wheel_version": True,
            "isaaclab_import": True,
            "simulation_cfg_symbol": True,
            "build_simulation_context_symbol": True,
            "ovphysx_cfg_symbol": True,
            "forbidden_modules_absent": True,
            "resolved_usd_open": True,
            "stage_fully_composed": True,
            "articulation_root_schema": True,
            "canonical_joint_count": True,
            "position_drive_schema": True,
            "position_drive_values_finite": True,
            "position_drive_stiffness_positive": True,
            "physx_properties_authored": True,
            "physx_max_joint_velocity_values_finite_positive": True,
            "canonical_distal_frame_count": True,
            "physics_step": False,
        }
        payload = {
            "backend": "ovphysx",
            "mode": "resolved_usd_headless",
            "status": "available",
            "message": "synthetic formal schema fixture",
            "capabilities": capabilities,
            "provenance": {
                "hand": side,
                "model_path": hand.model_paths.ovphysx,
                "session_id": SESSION,
                "source_revision": REVISION,
                "manifest_sha256": manifest_digest,
                "manifest_file_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
                "asset_tree_sha256": (
                    manifest.provenance.canonical_lf_asset_tree_sha256
                ),
                "asset_commit": manifest.provenance.commit,
                "asset_git_tree": manifest.provenance.asset_git_tree,
                "schema_probe_script_sha256": sha256(
                    (ROOT / "scripts" / "probe_ovphysx_schema.py").read_bytes()
                ).hexdigest(),
                "imported_modules": {
                    "isaaclab.sim": {"version": "fixture", "symbols": {}}
                },
                "usd_inspection": inspection,
            },
            "error": None,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (schema_dir / f"{side}.json").write_text(rendered, encoding="utf-8")
        (schema_dir / f"{side}.stdout").write_text(rendered, encoding="utf-8")
        (schema_dir / f"{side}.stderr").write_text("", encoding="utf-8")
        (schema_dir / f"{side}.exitcode").write_text("0\n", encoding="utf-8")
    _rehash_schema(schema_dir)


def _matrix(tmp_path: Path) -> tuple[Path, Path]:
    mujoco = tmp_path / "mujoco"
    ovphysx = tmp_path / "ovphysx"
    _write_evidence(mujoco, Simulator.MUJOCO)
    _write_evidence(ovphysx, Simulator.OVPHYSX)
    return mujoco, ovphysx


def test_finalizer_copies_all_evidence_reports_and_content_hashes_exact_32_runs(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    output = tmp_path / "bundle"

    returned, verification = finalizer.finalize_gate0(
        manifest_path=MANIFEST,
        mujoco_evidence_root=mujoco,
        ovphysx_evidence_root=ovphysx,
        output_dir=output,
        session_id=SESSION,
        source_revision=REVISION,
    )

    assert returned == output.resolve()
    assert verification["valid"] is True
    bundle_check = verify_bundle(output)
    assert all(verification[key] == value for key, value in bundle_check.items())
    assert verification["gate0_execution_status"] == "error"
    assert verification["gate0_comparison_status"] == "inconclusive"
    assert verification["execution_complete"] is False
    assert verification["comparison_conclusive"] is False
    assert verification["successful"] is False
    assert verification["pass_ready"] is False
    assert "use pass_ready" in verification["successful_semantics"]
    assert (output / "manifest" / "gate0.json").read_bytes() == MANIFEST.read_bytes()
    assert (output / "evidence" / "mujoco" / "launcher" / "provenance.json").is_file()
    assert (output / "evidence" / "ovphysx" / "launcher" / "raw-launcher.log").is_file()
    assert (output / "evidence" / "ovphysx" / "launcher" / "schema" / "left.json").is_file()
    assert len(list((output / "evidence").rglob("*.run.json"))) == 32
    assert (output / "results" / "comparison.json").is_file()
    assert (output / "results" / "report.json").is_file()
    assert (output / "results" / "report.md").is_file()
    metadata = json.loads(
        (output / "results" / "finalization.json").read_text(encoding="utf-8")
    )
    assert metadata["expected_case_count"] == 32
    assert metadata["received_case_count"] == 32
    assert len(metadata["case_ids"]) == len(set(metadata["case_ids"])) == 32

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=output,
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_rejects_missing_and_extra_run_files(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    missing = next((mujoco / "cases").rglob("*.run.json"))
    missing.unlink()
    extra = ovphysx / "launcher" / "duplicate.run.json"
    extra.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        finalizer.Gate0FinalizationError,
        match="local evidence topology|exactly its 16",
    ):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "not-created",
            session_id=SESSION,
            source_revision=REVISION,
        )
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("field", ["session_id", "source_revision", "asset_commit"])
def test_finalizer_rejects_cross_run_or_wrong_provenance(
    tmp_path: Path, field: str
) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    run_file = next((ovphysx / "cases").rglob("*.run.json"))
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    payload["result"]["provenance"][field] = "wrong"
    run_file.write_text(json.dumps(payload), encoding="utf-8")
    _rehash_case(run_file.parent)

    with pytest.raises(finalizer.Gate0FinalizationError, match="invalid provenance"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected",
            session_id=SESSION,
            source_revision=REVISION,
        )
    assert not (tmp_path / "rejected").exists()


def test_finalizer_rejects_nested_evidence_roots_before_copy(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    with pytest.raises(finalizer.Gate0FinalizationError, match="non-nested"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=outer,
            ovphysx_evidence_root=inner,
            output_dir=tmp_path / "rejected",
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_requires_hashed_available_left_and_right_schema_probes(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    schema_dir = ovphysx / "launcher" / "schema"
    payload_path = schema_dir / "right.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["capabilities"]["physx_properties_authored"] = False
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _rehash_schema(schema_dir)

    with pytest.raises(finalizer.Gate0FinalizationError, match="failed capabilities"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-schema",
            session_id=SESSION,
            source_revision=REVISION,
        )
    assert not (tmp_path / "rejected-schema").exists()

    payload["capabilities"]["physx_properties_authored"] = True
    payload["provenance"]["usd_inspection"]["angular_drive_records"][0][
        "stiffness"
    ] = -1.0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _rehash_schema(schema_dir)
    with pytest.raises(finalizer.Gate0FinalizationError, match="stiffness must be finite and positive"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-values",
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_rejects_wrong_schema_probe_script_version(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    schema_dir = ovphysx / "launcher" / "schema"
    payload_path = schema_dir / "left.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["provenance"]["schema_probe_script_sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    _rehash_schema(schema_dir)

    with pytest.raises(finalizer.Gate0FinalizationError, match="schema_probe_script_sha256"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-schema-script",
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_verifies_local_case_and_launcher_hashes(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    case_dir = next(path for path in (mujoco / "cases").iterdir() if path.is_dir())
    raw_log = case_dir / "launcher.stderr.log"
    raw_log.write_text("tampered after hashing\n", encoding="utf-8")
    with pytest.raises(finalizer.Gate0FinalizationError, match="digest mismatch"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-local-case",
            session_id=SESSION,
            source_revision=REVISION,
        )

    _rehash_case(case_dir)
    launcher = mujoco / "launcher"
    provenance_path = launcher / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["launcher_script_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    _rehash_case(launcher)
    with pytest.raises(finalizer.Gate0FinalizationError, match="pinned run identity"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-local-launcher",
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_verifies_remote_top_level_inventory(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)
    (ovphysx / "launcher" / "raw-launcher.log").write_text(
        "tampered after the top-level inventory\n", encoding="utf-8"
    )

    with pytest.raises(finalizer.Gate0FinalizationError, match="digest mismatch"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=tmp_path / "rejected-remote-inventory",
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_rejects_unverified_local_source_before_output(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    mujoco, ovphysx = _matrix(tmp_path)

    def reject(_root: Path, _revision: str) -> str:
        raise finalizer.Gate0FinalizationError("tracked worktree/index must be clean")

    finalizer._validate_source_identity = reject
    output = tmp_path / "never-created-dirty-source"
    with pytest.raises(finalizer.Gate0FinalizationError, match="tracked worktree/index"):
        finalizer.finalize_gate0(
            manifest_path=MANIFEST,
            mujoco_evidence_root=mujoco,
            ovphysx_evidence_root=ovphysx,
            output_dir=output,
            session_id=SESSION,
            source_revision=REVISION,
        )
    assert not output.exists()


def test_remote_hard_gate_checks_status_pins_case_hashes_and_idle_gpu(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    _mujoco, ovphysx = _matrix(tmp_path)
    manifest = load_manifest(MANIFEST)
    status_path = ovphysx / "launcher" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["completed_case_count"] = 15
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(finalizer.Gate0FinalizationError, match="completed 16-case"):
        finalizer._validate_remote_launcher_evidence(
            ovphysx,
            manifest,
            expected_source_tree=SOURCE_TREE,
            manifest_file_sha256=sha256(MANIFEST.read_bytes()).hexdigest(),
            session_id=SESSION,
            source_revision=REVISION,
        )

    status["completed_case_count"] = 16
    status_path.write_text(json.dumps(status), encoding="utf-8")
    identity = finalizer._validate_remote_launcher_evidence(
        ovphysx,
        manifest,
        expected_source_tree=SOURCE_TREE,
        manifest_file_sha256=sha256(MANIFEST.read_bytes()).hexdigest(),
        session_id=SESSION,
        source_revision=REVISION,
    )
    provenance_path = ovphysx / "launcher" / "provenance.txt"
    provenance_text = provenance_path.read_text(encoding="utf-8")
    provenance_path.write_text(
        provenance_text.replace("driver_version=570.158.01", "driver_version=0"),
        encoding="utf-8",
    )
    with pytest.raises(finalizer.Gate0FinalizationError, match="pinned field"):
        finalizer._validate_remote_launcher_evidence(
            ovphysx,
            manifest,
            expected_source_tree=SOURCE_TREE,
            manifest_file_sha256=sha256(MANIFEST.read_bytes()).hexdigest(),
            session_id=SESSION,
            source_revision=REVISION,
        )
    provenance_path.write_text(provenance_text, encoding="utf-8")

    case_dir = next(path for path in (ovphysx / "cases").iterdir() if path.is_dir())
    raw_log = case_dir / "launcher.stderr.txt"
    raw_log.write_text("tampered after hashing\n", encoding="utf-8")
    with pytest.raises(finalizer.Gate0FinalizationError, match="digest mismatch"):
        finalizer._validate_remote_case_evidence(ovphysx, manifest, identity)

    _rehash_case(case_dir)
    preflight = case_dir / "preflight.txt"
    preflight.write_text(
        _gpu_state_text().replace("memory_used_mib=0", "memory_used_mib=1"),
        encoding="utf-8",
    )
    _rehash_case(case_dir)
    finalizer._validate_remote_case_evidence(ovphysx, manifest, identity)

    preflight.write_text(
        _gpu_state_text().replace("memory_used_mib=0", "memory_used_mib=33"),
        encoding="utf-8",
    )
    _rehash_case(case_dir)
    with pytest.raises(finalizer.Gate0FinalizationError, match="idle pinned GPU"):
        finalizer._validate_remote_case_evidence(ovphysx, manifest, identity)

    run_file = next(case_dir.glob("*.run.json"))
    run_payload = json.loads(run_file.read_text(encoding="utf-8"))
    run_payload["result"]["provenance"]["simulation_configuration"][
        "verified"
    ] = False
    run_file.write_text(json.dumps(run_payload), encoding="utf-8")
    run = finalizer.load_collected_runs(run_file)[0]
    with pytest.raises(finalizer.Gate0FinalizationError, match="simulation_configuration"):
        finalizer._validate_provenance(
            (run,),
            manifest=manifest,
            session_id=SESSION,
            source_revision=REVISION,
        )


def test_finalizer_rejects_invalid_ideal_pd_runtime_provenance(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    ovphysx = tmp_path / "ovphysx"
    _write_evidence(ovphysx, Simulator.OVPHYSX)
    manifest = load_manifest(MANIFEST)
    run_file = next((ovphysx / "cases").glob("*/*.run.json"))
    run = finalizer.load_collected_runs(run_file)[0]
    mutations = (
        ("actuation_contract_version", 1),
        ("actuation_contract_version", 2.0),
        ("actuator_model", "ImplicitActuator"),
        ("control_path", "implicit_physx_drive"),
        ("controller_dof_stiffness", [0.0] * 22),
        ("controller_dof_damping", [-0.1] * 22),
        ("controller_dof_effort_limit", [0.0] * 22),
        ("controller_dof_effort_limit_sim", [0.0] * 22),
        ("controller_parameter_source", "usd_drive_schema"),
        ("backend_dof_stiffness", [1e-7] * 22),
        ("backend_dof_damping", [-1e-7] * 22),
        ("backend_dof_drive_readback_source", "cached_tensor"),
        ("position_target_readback_source", "root_view_cuda_warp_binding"),
        ("zero_velocity_target_verified", False),
        ("zero_feedforward_effort_target_verified", False),
        ("computed_effort_peak_abs_nm", [-0.1] * 22),
        ("applied_effort_peak_abs_nm", [0.1] * 21),
        ("effort_observation_count", 0),
        ("effort_formula_max_abs_error_nm", 1.1e-5),
        ("effort_clip_max_abs_error_nm", 1.1e-6),
        ("effort_clip_count", True),
        ("effort_command_source", "actuator_cache"),
    )

    for field, invalid_value in mutations:
        provenance = dict(run.result.provenance)
        provenance[field] = invalid_value
        mutated = replace(
            run,
            result=replace(run.result, provenance=provenance),
        )
        with pytest.raises(finalizer.Gate0FinalizationError):
            finalizer._validate_ovphysx_runtime_provenance(mutated, manifest)


def test_finalizer_cli_returns_nonzero_for_preserved_inconclusive_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = _load_finalizer()
    bundle = tmp_path / "preserved-bundle"
    bundle.mkdir()
    monkeypatch.setattr(
        finalizer,
        "finalize_gate0",
        lambda **_kwargs: (
            bundle,
            {
                "valid": True,
                "successful": False,
                "pass_ready": False,
                "gate0_execution_status": "error",
                "gate0_comparison_status": "inconclusive",
            },
        ),
    )
    assert (
        finalizer.main(
            [
                "--manifest",
                "manifest.json",
                "--mujoco-evidence-root",
                "mujoco",
                "--ovphysx-evidence-root",
                "ovphysx",
                "--output-dir",
                "output",
                "--session-id",
                SESSION,
                "--source-revision",
                REVISION,
            ]
        )
        == 3
    )
