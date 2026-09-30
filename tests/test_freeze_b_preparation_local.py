from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from wave_asset_qa.parity import sensitivity
from wave_asset_qa.parity.contracts import Simulator


ROOT = Path(__file__).resolve().parents[1]


class _FakeChild:
    def __init__(
        self,
        *,
        pid: int = 111,
        returncode: int | None = None,
        wait_error: BaseException | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_error = wait_error
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _windows_worker_record(module: object, *, pid: int = 4242) -> dict[str, object]:
    identity = {
        "platform": "windows",
        "pid": pid,
        "process_creation_filetime": 123456789,
    }
    return {
        "schema_version": 1,
        "worker_pid": pid,
        "os_process_identity": identity,
        "fresh_process_id": module._fresh_process_id(identity),
    }


def _posix_worker_record(module: object, *, pid: int = 4242) -> dict[str, object]:
    identity = {
        "platform": "posix",
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "pid": pid,
        "process_start_ticks": 987654,
    }
    return {
        "schema_version": 1,
        "worker_pid": pid,
        "os_process_identity": identity,
        "fresh_process_id": module._fresh_process_id(identity),
    }


def _script(name: str) -> object:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corrigendum(
    module: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    correction_revision: str,
    correction_tree: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "corrigendum_id": (
            "ovphysx-legacy-joint-friction-freeze-b-evidence-corrigendum-v1"
        ),
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "state": (
            "POST_PREREGISTRATION_EVIDENCE_CORRECTION_BEFORE_ANY_ADMITTED_"
            "FREEZE_B_EVIDENCE"
        ),
        "frozen_preregistration": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "private_plan_sha256": module._PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": module._PRIVATE_PLAN_FILE_SHA256,
            "preregistration_revision": module._PREREGISTRATION_REVISION,
            "preregistration_tree": module._PREREGISTRATION_TREE,
            "public_protocol_path": module._PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": module._PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": module._PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": (
                module._PUBLIC_PROTOCOL_CANONICAL_SHA256
            ),
            "public_protocol_unchanged": True,
        },
        "incident": {
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
        },
        "correction": {
            "scope": "evidence_plumbing_only",
            "worker_identity_authority": (
                "actual_python_worker_kernel_identity_v1"
            ),
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
        },
        "correction_source": {
            "revision": correction_revision,
            "tree": correction_tree,
        },
        "source_transition_policy": {
            "implementation_to_preregistration_name_status": (
                module._IMPLEMENTATION_TO_PREREGISTRATION_DIFF
            ),
            "preregistration_to_correction_name_status": (
                module._PREREGISTRATION_TO_CORRECTION_DIFF
            ),
            "correction_to_execution_name_status": (
                module._CORRECTION_TO_EXECUTION_DIFF
            ),
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }


def _admission_corrigendum(
    module: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "corrigendum_id": "ovphysx-legacy-joint-friction-freeze-b-remote-admission-corrigendum-v1",
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "state": (
            "POST_PREREGISTRATION_REMOTE_ADMISSION_CORRECTION_AFTER_ABORTED_A2_"
            "COLLECTION_BEFORE_ANY_COMBINED_METRICS_OR_SCIENTIFIC_LABEL"
        ),
        "frozen_history": {
            "implementation_revision": implementation_revision,
            "implementation_tree": implementation_tree,
            "preregistration_revision": module._PREREGISTRATION_REVISION,
            "preregistration_tree": module._PREREGISTRATION_TREE,
            "first_correction_revision": first_correction_revision,
            "first_correction_tree": first_correction_tree,
            "first_execution_revision": module._FIRST_EXECUTION_REVISION,
            "first_execution_tree": module._FIRST_EXECUTION_TREE,
            "private_plan_sha256": module._PRIVATE_PLAN_SHA256,
            "private_plan_file_sha256": module._PRIVATE_PLAN_FILE_SHA256,
            "public_protocol_path": module._PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
            "public_protocol_blob_oid": module._PUBLIC_PROTOCOL_BLOB_OID,
            "public_protocol_file_sha256": module._PUBLIC_PROTOCOL_FILE_SHA256,
            "public_protocol_canonical_sha256": module._PUBLIC_PROTOCOL_CANONICAL_SHA256,
            "public_protocol_unchanged": True,
            "first_corrigendum_path": module._CORRIGENDUM_RELATIVE_PATH.as_posix(),
            "first_corrigendum_blob_oid": module._FIRST_CORRIGENDUM_BLOB_OID,
            "first_corrigendum_file_sha256": module._FIRST_CORRIGENDUM_FILE_SHA256,
            "first_corrigendum_canonical_sha256": module._FIRST_CORRIGENDUM_CANONICAL_SHA256,
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
        "excluded_evidence": module._ADMISSION_EXCLUDED_EVIDENCE_IDENTITIES,
        "correction_source": {
            "revision": admission_correction_revision,
            "tree": admission_correction_tree,
        },
        "source_transition_policy": {
            "first_execution_to_admission_correction_name_status": module._FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF,
            "admission_correction_to_execution_name_status": module._ADMISSION_CORRECTION_TO_EXECUTION_DIFF,
            "renames_allowed": False,
            "other_paths_allowed": False,
        },
    }


def _asset_preflight_corrigendum(
    module: object,
    *,
    implementation_revision: str,
    implementation_tree: str,
    first_correction_revision: str,
    first_correction_tree: str,
    admission_correction_revision: str,
    admission_correction_tree: str,
) -> dict[str, object]:
    value = json.loads(
        (ROOT / module._ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    value["frozen_history"] = {
        "implementation_revision": implementation_revision,
        "implementation_tree": implementation_tree,
        "preregistration_revision": module._PREREGISTRATION_REVISION,
        "preregistration_tree": module._PREREGISTRATION_TREE,
        "first_correction_revision": first_correction_revision,
        "first_correction_tree": first_correction_tree,
        "first_execution_revision": module._FIRST_EXECUTION_REVISION,
        "first_execution_tree": module._FIRST_EXECUTION_TREE,
        "admission_correction_revision": admission_correction_revision,
        "admission_correction_tree": admission_correction_tree,
        "private_plan_sha256": module._PRIVATE_PLAN_SHA256,
        "private_plan_file_sha256": module._PRIVATE_PLAN_FILE_SHA256,
        "public_protocol_path": module._PUBLIC_PROTOCOL_RELATIVE_PATH.as_posix(),
        "public_protocol_blob_oid": module._PUBLIC_PROTOCOL_BLOB_OID,
        "public_protocol_file_sha256": module._PUBLIC_PROTOCOL_FILE_SHA256,
        "public_protocol_canonical_sha256": module._PUBLIC_PROTOCOL_CANONICAL_SHA256,
        "public_protocol_unchanged": True,
        "first_corrigendum_path": module._CORRIGENDUM_RELATIVE_PATH.as_posix(),
        "first_corrigendum_blob_oid": module._FIRST_CORRIGENDUM_BLOB_OID,
        "first_corrigendum_file_sha256": module._FIRST_CORRIGENDUM_FILE_SHA256,
        "first_corrigendum_canonical_sha256": module._FIRST_CORRIGENDUM_CANONICAL_SHA256,
        "first_corrigendum_unchanged": True,
        "admission_corrigendum_path": module._ADMISSION_CORRIGENDUM_RELATIVE_PATH.as_posix(),
        "admission_corrigendum_blob_oid": module._ADMISSION_CORRIGENDUM_BLOB_OID,
        "admission_corrigendum_file_sha256": module._ADMISSION_CORRIGENDUM_FILE_SHA256,
        "admission_corrigendum_canonical_sha256": module._ADMISSION_CORRIGENDUM_CANONICAL_SHA256,
        "admission_corrigendum_unchanged": True,
    }
    value["excluded_evidence"] = module._EXCLUDED_EVIDENCE_IDENTITIES
    value["incident_source"] = {
        "revision": module._SECOND_EXECUTION_REVISION,
        "tree": module._SECOND_EXECUTION_TREE,
    }
    value["source_transition_policy"] = {
        "incident_source_to_execution_name_status": (
            module._INCIDENT_SOURCE_TO_EXECUTION_DIFF
        ),
        "renames_allowed": False,
        "other_paths_allowed": False,
    }
    return value


def test_preparer_case_inventory_matches_the_contract() -> None:
    module = _script("prepare_ovphysx_freeze_b.py")

    assert module.READBACK_SOURCE_REVISION == (
        "44571ea60201f77ca93d49c9c14cb2d28152c30f"
    )
    assert module.READBACK_SOURCE_TREE == (
        "d2a03e8ecceb46971e5c75debdb738baea63a4a3"
    )

    ovphysx, mujoco, order = module._cases()

    assert ovphysx == list(sensitivity.expected_case_matrix(Simulator.OVPHYSX))
    assert mujoco == list(sensitivity.expected_case_matrix(Simulator.MUJOCO))
    assert order == list(sensitivity.expected_run_order())
    assert order[:4] == [
        "freeze_b.ovphysx.left.small_step.base.sham.r01",
        "freeze_b.ovphysx.left.small_step.base.zero.r01",
        "freeze_b.ovphysx.left.small_step.base.zero.r02",
        "freeze_b.ovphysx.left.small_step.base.sham.r02",
    ]


def test_preparer_accepts_only_the_exact_r1_runtime_friction_schema() -> None:
    module = _script("prepare_ovphysx_freeze_b.py")
    runtime = {
        "binding_controller_exact_match": {
            "overall": True,
            "static_friction": True,
            "dynamic_friction": True,
            "viscous_friction": True,
        },
        "dof_friction_properties_binding": {
            "static": 0.0,
            "dynamic": 0.0,
            "viscous": 0.0,
        },
        "idealpd_controller_buffers": {
            "static": 0.0,
            "dynamic": 0.0,
            "viscous": 0.0,
        },
    }

    module._validate_r1_runtime_friction(runtime, "joint_fixture")
    legacy_array = deepcopy(runtime)
    legacy_array["dof_friction_properties_binding"] = [0.0, 0.0, 0.0]
    with pytest.raises(module.FreezeBPreparationError, match="must be an object"):
        module._validate_r1_runtime_friction(legacy_array, "joint_fixture")
    nonzero = deepcopy(runtime)
    nonzero["idealpd_controller_buffers"]["viscous"] = 0.25
    with pytest.raises(module.FreezeBPreparationError, match="positive zero"):
        module._validate_r1_runtime_friction(nonzero, "joint_fixture")


def test_preparer_window_metric_is_elementwise_and_inclusive() -> None:
    module = _script("prepare_ovphysx_freeze_b.py")
    samples_a = []
    samples_b = []
    for index, time_s in enumerate((0.0, 0.1, 0.5)):
        del index
        samples_a.append(
            SimpleNamespace(
                time_s=time_s,
                joint_positions={"joint": 0.0},
                frame_poses={"tip": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)},
            )
        )
        samples_b.append(
            SimpleNamespace(
                time_s=time_s,
                joint_positions={"joint": 9.0 if time_s < 0.1 else 2.0},
                frame_poses={
                    "tip": (
                        9.0 if time_s < 0.1 else 1.0,
                        9.0 if time_s < 0.1 else 2.0,
                        9.0 if time_s < 0.1 else 2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                },
            )
        )
    left = SimpleNamespace(result=SimpleNamespace(samples=tuple(samples_a)))
    right = SimpleNamespace(result=SimpleNamespace(samples=tuple(samples_b)))

    assert module._window_rmse(left, right, frame_position=False) == pytest.approx(2.0)
    assert module._window_rmse(left, right, frame_position=True) == pytest.approx(
        (3.0**0.5)
    )


def test_preparer_float32_and_output_boundaries_fail_closed(tmp_path: Path) -> None:
    module = _script("prepare_ovphysx_freeze_b.py")
    value, bits = module._float32(0.125, "fixture")
    assert value == 0.125 and bits.hex() == "3e000000"
    with pytest.raises(module.FreezeBPreparationError, match="finite"):
        module._float32(float("nan"), "fixture")

    output = tmp_path / "private.json"
    module._write_exclusive(output, {"safe": True})
    assert output.is_file()
    with pytest.raises(module.FreezeBPreparationError, match="overwrite"):
        module._write_exclusive(output, {"safe": True})


def test_local_launcher_accepts_only_direct_new_results_child(tmp_path: Path) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)

    accepted = module._direct_results_child(project, results / "freeze-b-private")
    assert accepted == results / "freeze-b-private"
    with pytest.raises(module.FreezeBLocalError, match="direct child"):
        module._direct_results_child(project, results / "nested" / "bad")
    existing = results / "existing"
    existing.mkdir()
    with pytest.raises(module.FreezeBLocalError, match="already exists"):
        module._direct_results_child(project, existing)


def test_local_launcher_plan_inventory_is_exact() -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    plan = {"mujoco_cases": list(sensitivity.expected_case_matrix("mujoco"))}
    assert len(module._canonical_case_ids(plan)) == 8
    with pytest.raises(module.FreezeBLocalError, match="exactly eight"):
        module._canonical_case_ids({"mujoco_cases": []})


def test_local_worker_identity_and_hash_exclude_case_and_session() -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    posix_identity = {
        "platform": "posix",
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "pid": 4242,
        "process_start_ticks": 987654,
    }
    worker_process = {
        "schema_version": 1,
        "worker_pid": 4242,
        "os_process_identity": posix_identity,
        "fresh_process_id": module._fresh_process_id(posix_identity),
    }

    first = module._fresh_process_record(
        experiment_case_id="freeze_b.mujoco.left.small_step.base.control.r01",
        canonical_case_id="mujoco.left.small_step.base.r01",
        worker_process=worker_process,
    )
    second = module._fresh_process_record(
        experiment_case_id="freeze_b.mujoco.right.small_step.halved.control.r02",
        canonical_case_id="mujoco.right.small_step.halved.r02",
        worker_process=worker_process,
    )
    assert first["fresh_process_id"] == second["fresh_process_id"]
    assert "case_id" not in posix_identity and "session_id" not in posix_identity
    with pytest.raises(module.FreezeBLocalError, match="fields are not exact"):
        module._fresh_process_id({**posix_identity, "case_id": "forbidden"})
    records: list[dict[str, object]] = []
    module._append_unique_fresh_process_record(records, first)
    with pytest.raises(module.FreezeBLocalError, match="was reused"):
        module._append_unique_fresh_process_record(records, second)

    mismatched = deepcopy(worker_process)
    mismatched["worker_pid"] = 9999
    with pytest.raises(module.FreezeBLocalError, match="worker_pid"):
        module._validated_worker_process_record(mismatched)
    forged = deepcopy(worker_process)
    forged["fresh_process_id"] = "0" * 64
    with pytest.raises(module.FreezeBLocalError, match="hash"):
        module._validated_worker_process_record(forged)


def test_local_failed_attempt_gets_terminal_error_and_exact_inventories(
    tmp_path: Path,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    launcher = tmp_path / "launcher"
    cases = tmp_path / "cases"
    case = cases / "freeze_b.mujoco.left.small_step.base.control.r01"
    launcher.mkdir()
    case.mkdir(parents=True)
    (launcher / "provenance.json").write_text("{}\n", encoding="utf-8")
    (case / "launcher.exitcode.txt").write_text("0\n", encoding="ascii")

    module._seal_failed_attempt(
        launcher_dir=launcher,
        cases_dir=cases,
        identity={"schema_version": 1, "protocol_id": "fixture"},
        active_experiment_case_id=case.name,
        completed_before_failure=[],
        error=module.FreezeBLocalError("fixture binding rejection"),
    )

    terminal = json.loads((launcher / "error.json").read_text(encoding="utf-8"))
    assert terminal["terminal_status"] == "error"
    assert terminal["attempt_admitted_case_count"] == 0
    assert terminal["active_experiment_case_id"] == case.name
    assert (launcher / "evidence.sha256").is_file()
    assert (case / "evidence.sha256").is_file()


def test_worker_handshake_uses_actual_worker_even_when_transport_pid_differs(
    tmp_path: Path,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    worker_record = _windows_worker_record(module, pid=4242)
    sidecar = tmp_path / "worker-process.json"
    sidecar.write_text(json.dumps(worker_record) + "\n", encoding="utf-8")
    transport = _FakeChild(pid=111)

    observed = module._await_worker_process_record(
        sidecar, transport, timeout_s=0.1
    )

    assert observed == worker_record
    assert observed["worker_pid"] != transport.pid


def test_windows_cleanup_terminates_sidecar_identity_not_redirector_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    worker_record = _windows_worker_record(module, pid=4242)
    sidecar = tmp_path / "worker-process.json"
    sidecar.write_text(json.dumps(worker_record) + "\n", encoding="utf-8")
    transport = _FakeChild(pid=111)
    terminated: list[object] = []
    waited: list[object] = []
    monkeypatch.setattr(
        module,
        "terminate_windows_process_identity",
        lambda identity, *, timeout_s: terminated.append(identity),
    )
    monkeypatch.setattr(
        module,
        "wait_for_os_process_exit",
        lambda identity, *, timeout_s: waited.append(identity),
    )

    module._terminate_owned_child(
        transport, worker_process_path=sidecar, platform_name="nt"
    )

    assert terminated == [worker_record["os_process_identity"]]
    assert waited == [worker_record["os_process_identity"]]
    assert transport.wait_timeouts == [30.0]
    assert transport.killed is False


def test_windows_pre_handshake_cleanup_refuses_any_pid_only_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("bare-PID taskkill must never run"),
    )
    for transport in (_FakeChild(pid=111), _FakeChild(pid=222, returncode=1)):
        with pytest.raises(
            module.FreezeBWorkerTerminationError,
            match="refusing PID-only termination",
        ):
            module._terminate_owned_child(
                transport,
                worker_process_path=tmp_path / "missing.json",
                platform_name="nt",
            )
        assert transport.killed is False
        assert transport.wait_timeouts == []


def test_posix_cleanup_uses_exact_pidfd_identity_and_never_child_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    worker_record = _posix_worker_record(module)
    sidecar = tmp_path / "worker-process.json"
    sidecar.write_text(json.dumps(worker_record) + "\n", encoding="utf-8")
    transport = _FakeChild(pid=4242)
    terminated: list[object] = []
    monkeypatch.setattr(
        module,
        "terminate_posix_process_identity",
        lambda identity, *, timeout_s: terminated.append(identity),
    )
    monkeypatch.setattr(
        module,
        "wait_for_os_process_exit",
        lambda *_args, **_kwargs: pytest.fail(
            "POSIX exact termination must reap before any /proc exit wait"
        ),
    )

    module._terminate_owned_child(
        transport, worker_process_path=sidecar, platform_name="posix"
    )

    assert terminated == [worker_record["os_process_identity"]]
    assert transport.killed is False
    assert transport.wait_timeouts == [30.0]


def test_posix_cleanup_without_identity_is_unsealable(
    tmp_path: Path,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    transport = _FakeChild(pid=4242)

    with pytest.raises(
        module.FreezeBWorkerTerminationError,
        match="refusing PID-only termination",
    ):
        module._terminate_owned_child(
            transport,
            worker_process_path=tmp_path / "missing.json",
            platform_name="posix",
        )
    assert transport.killed is False
    assert transport.wait_timeouts == []


def test_posix_transport_timeout_never_falls_back_to_bare_pid_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    worker_record = _posix_worker_record(module)
    sidecar = tmp_path / "worker-process.json"
    sidecar.write_text(json.dumps(worker_record) + "\n", encoding="utf-8")
    transport = _FakeChild(
        pid=4242,
        wait_error=subprocess.TimeoutExpired(["python", "worker.py"], 30.0),
    )
    monkeypatch.setattr(
        module, "terminate_posix_process_identity", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "wait_for_os_process_exit", lambda *_args, **_kwargs: None
    )

    with pytest.raises(
        module.FreezeBWorkerTerminationError,
        match="transport did not exit",
    ):
        module._terminate_owned_child(
            transport, worker_process_path=sidecar, platform_name="posix"
        )
    assert transport.killed is False


def test_case_worker_popen_oserror_is_ownership_unknown_and_unsealable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )
    monkeypatch.setattr(
        module,
        "_terminate_owned_child",
        lambda *_args, **_kwargs: pytest.fail("no child existed to terminate"),
    )
    with pytest.raises(module.FreezeBWorkerTerminationError, match="ownership"):
        module._run_case_worker(
            ["python", "worker.py"],
            project_root=tmp_path,
            environment={},
            stdout_handle=BytesIO(),
            stderr_handle=BytesIO(),
            worker_process_path=tmp_path / "worker-process.json",
            case_timeout_s=1.0,
        )


def test_promoted_attempt_is_recoverably_owned_across_initialization_interrupt(
    tmp_path: Path,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    output = tmp_path / "fixed-run-id"
    staging = tmp_path / ".fixed-run-id.incomplete-fixture"
    launcher = staging / "launcher"
    launcher.mkdir(parents=True)
    (staging / "cases").mkdir()
    token = "a" * 64
    module._write_json_exclusive(
        launcher / "attempt-ownership.json",
        module._attempt_ownership_record(
            ownership_token=token,
            session_id="fixture-session",
            source_revision="b" * 40,
        ),
    )

    staging.rename(output)

    assert module._attempt_root_is_owned(
        output,
        ownership_token=token,
        session_id="fixture-session",
        source_revision="b" * 40,
    )
    assert not module._attempt_root_is_owned(
        output,
        ownership_token="c" * 64,
        session_id="fixture-session",
        source_revision="b" * 40,
    )


def test_initialization_staging_cleanup_requires_exact_ownership_token(
    tmp_path: Path,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    output = tmp_path / "fixed-run-id"
    token = "a" * 64

    def make_staging(suffix: str, stored_token: str) -> Path:
        staging = tmp_path / f".fixed-run-id.incomplete-{suffix}"
        launcher = staging / "launcher"
        launcher.mkdir(parents=True)
        module._write_json_exclusive(
            launcher / "attempt-ownership.json",
            module._attempt_ownership_record(
                ownership_token=stored_token,
                session_id="fixture-session",
                source_revision="b" * 40,
            ),
        )
        return staging

    owned = make_staging("owned", token)
    module._remove_owned_initialization_staging(
        owned,
        output=output,
        ownership_token=token,
        session_id="fixture-session",
        source_revision="b" * 40,
    )
    assert not owned.exists()

    foreign = make_staging("foreign", "c" * 64)
    with pytest.raises(module.FreezeBLocalError, match="unsafe"):
        module._remove_owned_initialization_staging(
            foreign,
            output=output,
            ownership_token=token,
            session_id="fixture-session",
            source_revision="b" * 40,
        )
    assert foreign.is_dir()


def test_case_worker_interrupted_inside_popen_refuses_terminal_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(module.FreezeBWorkerTerminationError, match="ownership"):
        module._run_case_worker(
            ["python", "worker.py"],
            project_root=tmp_path,
            environment={},
            stdout_handle=BytesIO(),
            stderr_handle=BytesIO(),
            worker_process_path=tmp_path / "worker-process.json",
            case_timeout_s=1.0,
        )


@pytest.mark.parametrize("failure", [OSError("post-spawn"), KeyboardInterrupt()])
def test_case_worker_post_spawn_failure_terminates_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    transport = _FakeChild()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: transport)
    monkeypatch.setattr(
        module,
        "_await_worker_process_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    terminated: list[object] = []
    monkeypatch.setattr(
        module,
        "_terminate_owned_child",
        lambda child, **_kwargs: terminated.append(child),
    )

    with pytest.raises(type(failure), match=str(failure) or None):
        module._run_case_worker(
            ["python", "worker.py"],
            project_root=tmp_path,
            environment={},
            stdout_handle=BytesIO(),
            stderr_handle=BytesIO(),
            worker_process_path=tmp_path / "worker-process.json",
            case_timeout_s=1.0,
        )
    assert terminated == [transport]


def test_case_worker_proves_recorded_worker_exit_after_transport_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    transport = _FakeChild(pid=111)
    worker_record = _windows_worker_record(module, pid=4242)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: transport)
    monkeypatch.setattr(
        module, "_await_worker_process_record", lambda *_args, **_kwargs: worker_record
    )
    exit_proofs: list[object] = []
    monkeypatch.setattr(
        module,
        "_wait_for_recorded_worker_exit",
        lambda worker, *, timeout_s: exit_proofs.append(worker),
    )

    returncode, observed = module._run_case_worker(
        ["python", "worker.py"],
        project_root=tmp_path,
        environment={},
        stdout_handle=BytesIO(),
        stderr_handle=BytesIO(),
        worker_process_path=tmp_path / "worker-process.json",
        case_timeout_s=1.0,
    )

    assert returncode == 0
    assert observed == worker_record
    assert exit_proofs == [worker_record]


def test_case_worker_exit_proof_failure_terminates_before_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    transport = _FakeChild(pid=111)
    worker_record = _windows_worker_record(module, pid=4242)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: transport)
    monkeypatch.setattr(
        module, "_await_worker_process_record", lambda *_args, **_kwargs: worker_record
    )
    monkeypatch.setattr(
        module,
        "_wait_for_recorded_worker_exit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.FreezeBLocalError("worker remains alive")
        ),
    )
    terminated: list[object] = []
    monkeypatch.setattr(
        module,
        "_terminate_owned_child",
        lambda child, **_kwargs: terminated.append(child),
    )

    with pytest.raises(module.FreezeBLocalError, match="remains alive"):
        module._run_case_worker(
            ["python", "worker.py"],
            project_root=tmp_path,
            environment={},
            stdout_handle=BytesIO(),
            stderr_handle=BytesIO(),
            worker_process_path=tmp_path / "worker-process.json",
            case_timeout_s=1.0,
        )
    assert terminated == [transport]


def test_case_worker_timeout_is_admitted_only_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    transport = _FakeChild(
        wait_error=subprocess.TimeoutExpired(["python", "worker.py"], 1.0)
    )
    worker_record = _windows_worker_record(module)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: transport)
    monkeypatch.setattr(
        module, "_await_worker_process_record", lambda *_args, **_kwargs: worker_record
    )
    terminated: list[object] = []
    monkeypatch.setattr(
        module,
        "_terminate_owned_child",
        lambda child, **_kwargs: terminated.append(child),
    )

    result = module._run_case_worker(
        ["python", "worker.py"],
        project_root=tmp_path,
        environment={},
        stdout_handle=BytesIO(),
        stderr_handle=BytesIO(),
        worker_process_path=tmp_path / "worker-process.json",
        case_timeout_s=1.0,
    )

    assert result == (124, worker_record)
    assert terminated == [transport]


def test_local_launcher_uses_popen_and_writes_ordered_process_ledgers() -> None:
    source = (ROOT / "scripts/run_ovphysx_freeze_b_local.py").read_text(
        encoding="utf-8"
    )

    assert "child = subprocess.Popen(" in source
    assert "output.mkdir(mode=0o700)" not in source
    assert "initialization_staging = create_staging_root(output)" in source
    assert "promote_staging_root(initialization_staging, output)" in source
    assert 'staged_launcher / "attempt-ownership.json"' in source
    assert '"refusing PID-only termination"' in source
    assert '"taskkill"' not in source
    assert "scripts/run_freeze_b_mujoco_worker.py" in source
    assert 'case_dir / "worker-process.json"' in source
    assert 'case_dir / "fresh-process.json"' in source
    assert 'launcher_dir / "fresh-process-ledger.json"' in source
    assert '"fresh_process_ids": fresh_process_ids' in source
    assert "child.wait(timeout=remaining_timeout)" in source
    assert "_wait_for_recorded_worker_exit(" in source
    assert "worker_pid != process_pid" in source
    assert "_child_os_process_identity" not in source
    assert "_validate_run_file(case_dir, row, process_record)" in source


def test_local_run_file_is_bound_to_the_recorded_child_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    canonical_id = "mujoco.left.small_step.base.r01"
    experiment_id = "freeze_b.mujoco.left.small_step.base.control.r01"
    run_path = tmp_path / f"{canonical_id}.run.json"
    run_path.write_text("{}\n", encoding="utf-8")
    fake_run = SimpleNamespace(
        case=SimpleNamespace(case_id=canonical_id),
        result=SimpleNamespace(
            completed=True,
            backend="mujoco",
            scenario_id="small_step",
            dt=0.002,
            provenance={"worker_pid": 4242},
        ),
        bundle_root_sha256=None,
    )
    monkeypatch.setattr(module, "load_collected_runs", lambda _path: [fake_run])
    row = {
        "experiment_case_id": experiment_id,
        "canonical_case_id": canonical_id,
        "dt_s": 0.002,
    }
    record = {
        "os_process_identity": {
            "platform": "windows",
            "pid": 4242,
            "process_creation_filetime": 123456789,
        }
    }

    module._validate_run_file(tmp_path, row, record)
    fake_run.result.provenance["worker_pid"] = 4243
    with pytest.raises(module.FreezeBLocalError, match="worker_pid"):
        module._validate_run_file(tmp_path, row, record)


def test_local_launcher_requires_exact_a_b_c_d_e_f_g_source_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("run_ovphysx_freeze_b_local.py")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "freeze-b@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Freeze B Fixture"],
        cwd=repo,
        check=True,
    )
    (repo / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    for entry in module._PREREGISTRATION_TO_CORRECTION_DIFF:
        status, relative = entry.split("\t", 1)
        if status == "M":
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("before correction\n", encoding="utf-8")
    for entry in module._FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF:
        status, relative = entry.split("\t", 1)
        assert status == "M"
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("before admission correction\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "implementation"], cwd=repo, check=True, capture_output=True)
    implementation = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    implementation_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    config = repo / "configs" / "parity" / "ovphysx_legacy_friction_freeze_b.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", config.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "protocol"], cwd=repo, check=True, capture_output=True)
    preregistration = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    preregistration_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{config.relative_to(repo).as_posix()}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(module, "_PREREGISTRATION_REVISION", preregistration)
    monkeypatch.setattr(module, "_PREREGISTRATION_TREE", preregistration_tree)
    monkeypatch.setattr(module, "_PUBLIC_PROTOCOL_BLOB_OID", protocol_blob)
    monkeypatch.setattr(
        module, "_PUBLIC_PROTOCOL_FILE_SHA256", module.sha256_file(config)
    )
    monkeypatch.setattr(
        module, "_PUBLIC_PROTOCOL_CANONICAL_SHA256", sha256(b"{}").hexdigest()
    )
    monkeypatch.setattr(module, "_PRIVATE_PLAN_SHA256", "1" * 64)
    monkeypatch.setattr(module, "_PRIVATE_PLAN_FILE_SHA256", "2" * 64)

    for entry in module._PREREGISTRATION_TO_CORRECTION_DIFF:
        status, relative = entry.split("\t", 1)
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("after correction\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "evidence correction"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    correction = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    correction_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corrigendum_path = repo / module._CORRIGENDUM_RELATIVE_PATH
    corrigendum_path.parent.mkdir(parents=True, exist_ok=True)
    corrigendum_path.write_text(
        json.dumps(
            _corrigendum(
                module,
                implementation_revision=implementation,
                implementation_tree=implementation_tree,
                correction_revision=correction,
                correction_tree=correction_tree,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "public corrigendum"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    first_execution = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first_execution_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first_corrigendum_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"HEAD:{corrigendum_path.relative_to(repo).as_posix()}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(module, "_FIRST_EXECUTION_REVISION", first_execution)
    monkeypatch.setattr(module, "_FIRST_EXECUTION_TREE", first_execution_tree)
    monkeypatch.setattr(module, "_FIRST_CORRIGENDUM_BLOB_OID", first_corrigendum_blob)
    monkeypatch.setattr(
        module, "_FIRST_CORRIGENDUM_FILE_SHA256", module.sha256_file(corrigendum_path)
    )
    monkeypatch.setattr(
        module,
        "_FIRST_CORRIGENDUM_CANONICAL_SHA256",
        module._corrigendum_sha256(module._strict_json(corrigendum_path)),
    )

    for entry in module._FIRST_EXECUTION_TO_ADMISSION_CORRECTION_DIFF:
        _status, relative = entry.split("\t", 1)
        (repo / relative).write_text("after admission correction\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "admission correction"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    admission_correction = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    admission_correction_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    admission_corrigendum_path = repo / module._ADMISSION_CORRIGENDUM_RELATIVE_PATH
    admission_corrigendum_path.write_text(
        json.dumps(
            _admission_corrigendum(
                module,
                implementation_revision=implementation,
                implementation_tree=implementation_tree,
                first_correction_revision=correction,
                first_correction_tree=correction_tree,
                admission_correction_revision=admission_correction,
                admission_correction_tree=admission_correction_tree,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "admission corrigendum"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    second_execution = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    second_execution_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    admission_corrigendum_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"HEAD:{admission_corrigendum_path.relative_to(repo).as_posix()}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(module, "_ADMISSION_CORRECTION_REVISION", admission_correction)
    monkeypatch.setattr(module, "_ADMISSION_CORRECTION_TREE", admission_correction_tree)
    monkeypatch.setattr(module, "_SECOND_EXECUTION_REVISION", second_execution)
    monkeypatch.setattr(module, "_SECOND_EXECUTION_TREE", second_execution_tree)
    monkeypatch.setattr(
        module, "_ADMISSION_CORRIGENDUM_BLOB_OID", admission_corrigendum_blob
    )
    monkeypatch.setattr(
        module,
        "_ADMISSION_CORRIGENDUM_FILE_SHA256",
        module.sha256_file(admission_corrigendum_path),
    )
    monkeypatch.setattr(
        module,
        "_ADMISSION_CORRIGENDUM_CANONICAL_SHA256",
        module._corrigendum_sha256(module._strict_json(admission_corrigendum_path)),
    )

    asset_preflight_corrigendum_path = (
        repo / module._ASSET_PREFLIGHT_CORRIGENDUM_RELATIVE_PATH
    )
    asset_preflight_corrigendum_path.write_text(
        json.dumps(
            _asset_preflight_corrigendum(
                module,
                implementation_revision=implementation,
                implementation_tree=implementation_tree,
                first_correction_revision=correction,
                first_correction_tree=correction_tree,
                admission_correction_revision=admission_correction,
                admission_correction_tree=admission_correction_tree,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for entry in module._INCIDENT_SOURCE_TO_EXECUTION_DIFF:
        status, relative = entry.split("\t", 1)
        if status == "A":
            assert repo / relative == asset_preflight_corrigendum_path
            continue
        assert status == "M"
        (repo / relative).write_text(
            "after asset preflight incident\n", encoding="utf-8"
        )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "asset preflight incident"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    execution = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    transition = module._verify_source_transition(
        repo,
        implementation_revision=implementation,
        implementation_tree=implementation_tree,
        execution_revision=execution,
        corrigendum_path=corrigendum_path,
        admission_corrigendum_path=admission_corrigendum_path,
        asset_preflight_corrigendum_path=asset_preflight_corrigendum_path,
    )
    assert transition["correction_revision"] == correction
    assert transition["first_execution_revision"] == first_execution
    assert transition["admission_correction_revision"] == admission_correction
    assert transition["incident_source_revision"] == second_execution
    assert transition["execution_revision"] == execution

    (repo / "unallowed.py").write_text("drift\n", encoding="utf-8")
    subprocess.run(["git", "add", "unallowed.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "drift"], cwd=repo, check=True, capture_output=True)
    drifted = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(
        module.FreezeBLocalError,
        match="direct single-parent|exact corrigendum",
    ):
        module._verify_source_transition(
            repo,
            implementation_revision=implementation,
            implementation_tree=implementation_tree,
            execution_revision=drifted,
            corrigendum_path=corrigendum_path,
            admission_corrigendum_path=admission_corrigendum_path,
            asset_preflight_corrigendum_path=asset_preflight_corrigendum_path,
        )
