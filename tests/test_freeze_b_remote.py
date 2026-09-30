from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import symtable

import pytest

from wave_asset_qa.parity.contracts import HandSide, Simulator
from wave_asset_qa.parity.sensitivity import (
    expected_case_matrix,
    expected_pre_float32_sha256,
    expected_run_order,
)
from wave_asset_qa.parity.scenarios import load_manifest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "probe_ovphysx_freeze_b.py"
LAUNCHER_PATH = ROOT / "scripts" / "run_ovphysx_freeze_b_remote.sh"


def _worker_module() -> object:
    spec = importlib.util.spec_from_file_location("probe_ovphysx_freeze_b", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    return LAUNCHER_PATH.read_text(encoding="utf-8")


def _joint_names(side: str) -> tuple[str, ...]:
    from wave_asset_qa.parity.sensitivity import canonical_joint_names

    return canonical_joint_names(HandSide(side))


def _hand_stub(side: str) -> object:
    class Side:
        value = side

    class Hand:
        pass

    hand = Hand()
    hand.side = Side()
    hand.joint_names = _joint_names(side)
    return hand


def _hand_plan(side: str) -> dict[str, object]:
    names = _joint_names(side)
    values = {name: 0.25 + index / 100.0 for index, name in enumerate(names)}
    return {
        "hand": side,
        "joint_names": list(names),
        "joint_prim_paths": {
            name: f"/World/Env_0/Robot/joints/{name}" for name in names
        },
        "expected_pre_values": values,
        "sham_write_values": dict(values),
        "zero_write_values": {name: 0.0 for name in names},
        "expected_pre_float32_sha256": expected_pre_float32_sha256(names, values),
    }


def _plan() -> dict[str, object]:
    digest = "a" * 64
    oid = "b" * 40
    readback_ids = (
        "ovphysx.left.effective_readback.r01",
        "ovphysx.left.effective_readback.r02",
        "ovphysx.right.effective_readback.r01",
        "ovphysx.right.effective_readback.r02",
    )
    inputs = {
        "freeze_a_config_sha256": digest,
        "readback_bundle_root_sha256": digest,
        "readback_summary_sha256": digest,
        "readback_source_revision": oid,
        "readback_source_tree": oid,
        "readback_case_sha256": {case_id: digest for case_id in readback_ids},
        "formal_gate0_bundle_root_sha256": digest,
        "formal_gate0_source_revision": oid,
        "gate0_manifest_file_sha256": digest,
        "gate0_manifest_semantic_sha256": digest,
        "asset_commit": oid,
        "asset_git_tree": oid,
        "canonical_lf_asset_tree_sha256": digest,
        "freeze_b_source_revision": oid,
        "freeze_b_source_tree": oid,
        "formal_crosssim_window_rmse": {
            side: {
                variant: {
                    "joint_rmse_rad": 0.1,
                    "frame_position_rmse_m": 0.01,
                }
                for variant in ("base", "halved")
            }
            for side in ("left", "right")
        },
    }
    return {
        "schema_version": 1,
        "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
        "inputs": inputs,
        "hand_plans": [_hand_plan("left"), _hand_plan("right")],
        "ovphysx_cases": list(expected_case_matrix(Simulator.OVPHYSX)),
        "mujoco_cases": list(expected_case_matrix(Simulator.MUJOCO)),
        "run_order": list(expected_run_order()),
    }


def _intervention_evidence(
    module: object, *, side: str, role: str, plan_hash: str
) -> dict[str, object]:
    hand_plan = _hand_plan(side)
    names = tuple(hand_plan["joint_names"])
    expected = hand_plan["expected_pre_values"]
    writes = hand_plan[f"{role}_write_values"]
    records = []
    for name in names:
        prim = hand_plan["joint_prim_paths"][name]
        expected_hex = module._float32_hex(expected[name], name)
        write_hex = module._float32_hex(writes[name], name)
        records.append(
            {
                "canonical_id": name,
                "prim_path": prim,
                "property_path": f"{prim}.physxJoint:jointFriction",
                "observed_pre_value": expected[name],
                "observed_pre_float32_hex": expected_hex,
                "expected_pre_value": expected[name],
                "expected_pre_float32_hex": expected_hex,
                "write_value": writes[name],
                "write_float32_hex": write_hex,
                "post_write_value": writes[name],
                "post_write_float32_hex": write_hex,
                "post_reset_value": writes[name],
                "post_reset_float32_hex": write_hex,
                "post_cleanup_float32_hex": expected_hex,
            }
        )
    return {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": plan_hash,
        "attribute": "physxJoint:jointFriction",
        "edit_strategy": "anonymous_overlay_as_strongest_session_sublayer",
        "source_asset_sha256_before": "c" * 64,
        "source_asset_sha256_after_cleanup": "c" * 64,
        "source_asset_sha256_unchanged": True,
        "joint_count": 22,
        "joint_order": list(names),
        "exact_manifest_coverage_verified": True,
        "all_properties_prevalidated_before_write": True,
        "session_layer_anonymous": True,
        "original_edit_layer_anonymous": True,
        "override_layer_anonymous": True,
        "source_asset_was_edit_target": False,
        "articulation_uninitialized_before_write": True,
        "articulation_initialized_after_reset": True,
        "applied_before_first_simulation_reset": True,
        "post_write_verified": True,
        "post_reset_verified": True,
        "cleanup_status": "restored_pre_values",
        "records": records,
    }


def test_worker_accepts_only_the_counterbalanced_16_case_matrix() -> None:
    module = _worker_module()
    cases = module._validated_plan_matrix(_plan())

    assert len(cases) == 16
    assert [item["experiment_case_id"] for item in cases] == list(expected_run_order())
    assert cases[0]["experiment_case_id"].endswith("base.sham.r01")
    assert cases[2]["experiment_case_id"].endswith("base.zero.r02")

    changed = _plan()
    changed["run_order"][0], changed["run_order"][1] = (
        changed["run_order"][1],
        changed["run_order"][0],
    )
    with pytest.raises(module.FreezeBWorkerError, match="counterbalanced"):
        module._validated_plan_matrix(changed)


def test_worker_resolves_manifest_hands_with_the_handside_enum() -> None:
    module = _worker_module()
    manifest = load_manifest(ROOT / "configs" / "parity" / "gate0.json")

    observed = module._validated_hand_plan_map(
        [_hand_plan("left"), _hand_plan("right")], manifest=manifest
    )

    assert set(observed) == {"left", "right"}
    assert observed["left"]["joint_names"] == list(manifest.hand(HandSide.LEFT).joint_names)


@pytest.mark.parametrize("role", ("sham", "zero"))
def test_worker_requires_exact_float32_hand_plan(role: str) -> None:
    module = _worker_module()
    hand = _hand_stub("left")
    validated = module._validate_hand_plan(_hand_plan("left"), hand=hand)

    assert tuple(validated["joint_names"]) == hand.joint_names
    if role == "zero":
        assert set(validated["zero_write_values"].values()) == {0.0}

    changed = _hand_plan("left")
    first = hand.joint_names[0]
    changed["sham_write_values"][first] += 0.01
    with pytest.raises(module.FreezeBWorkerError, match="sham writes"):
        module._validate_hand_plan(changed, hand=hand)


@pytest.mark.parametrize("role", ("sham", "zero"))
def test_worker_validates_complete_cleanup_evidence(role: str) -> None:
    module = _worker_module()
    digest = "d" * 64
    hand_plan = module._validate_hand_plan(
        _hand_plan("right"), hand=_hand_stub("right")
    )
    provenance = {
        "legacy_joint_friction_intervention": _intervention_evidence(
            module, side="right", role=role, plan_hash=digest
        )
    }
    module._validate_intervention_evidence(
        provenance,
        role=role,
        private_plan_sha256=digest,
        hand_plan=hand_plan,
    )

    provenance["legacy_joint_friction_intervention"]["source_asset_sha256_unchanged"] = False
    with pytest.raises(module.FreezeBWorkerError, match="source asset"):
        module._validate_intervention_evidence(
            provenance,
            role=role,
            private_plan_sha256=digest,
            hand_plan=hand_plan,
        )

    provenance["legacy_joint_friction_intervention"] = _intervention_evidence(
        module, side="right", role=role, plan_hash=digest
    )
    provenance["legacy_joint_friction_intervention"][
        "articulation_uninitialized_before_write"
    ] = False
    with pytest.raises(
        module.FreezeBWorkerError,
        match="articulation_uninitialized_before_write",
    ):
        module._validate_intervention_evidence(
            provenance,
            role=role,
            private_plan_sha256=digest,
            hand_plan=hand_plan,
        )


def test_worker_keeps_dt_claim_at_python_configuration_scope() -> None:
    module = _worker_module()
    provenance = {
        "simulation_configuration_v2": {
            "schema_version": 2,
            "dt_claim_scope": "python_configuration_only_not_compiled_runtime",
            "requested_dt_s": 0.001,
            "simulation_cfg_dt_s": 0.001,
            "simulation_context_config_accessor_dt_s": 0.001,
            "python_configuration_dt_exact_match": True,
            "runtime_effective_dt_s": None,
            "runtime_effective_dt_status": "not_exposed_by_pinned_kitless_ovphysx",
            "runtime_effective_dt_verified": False,
            "requested_gravity_m_s2": [0.0, 0.0, 0.0],
            "simulation_cfg_gravity_m_s2": [0.0, 0.0, 0.0],
            "physics_scene_gravity_m_s2": [0.0, 0.0, 0.0],
            "physics_prim_path": "/physicsScene",
        }
    }
    module._validate_simulation_configuration_v2(provenance, dt_s=0.001)

    provenance["simulation_configuration_v2"]["runtime_effective_dt_s"] = 0.001
    with pytest.raises(module.FreezeBWorkerError, match="runtime_effective_dt_s"):
        module._validate_simulation_configuration_v2(provenance, dt_s=0.001)


def test_worker_fresh_process_id_hashes_only_linux_os_identity() -> None:
    module = _worker_module()
    identity = {
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "hostname": "example-gpu-node-6",
        "pid": 12345,
        "process_start_ticks": 987654,
    }

    first_case = module._fresh_process_id_from_os_identity(identity)
    second_case = module._fresh_process_id_from_os_identity(dict(identity))

    assert first_case == second_case
    assert len(first_case) == 64
    assert "case_id" not in identity and "session_id" not in identity
    changed = dict(identity)
    changed["process_start_ticks"] += 1
    assert module._fresh_process_id_from_os_identity(changed) != first_case
    extra = {**identity, "case_id": "forbidden"}
    with pytest.raises(module.FreezeBWorkerError, match="fields are not exact"):
        module._fresh_process_id_from_os_identity(extra)

    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    assert '"os_process_identity": os_process_identity' in worker_source
    assert 'provenance.get("forbidden_modules") != []' in worker_source


def test_launcher_is_server6_only_and_never_uses_ceph_sudo_or_kit() -> None:
    source = _source()

    assert 'readonly GATE0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"' in source
    assert 'readonly EXPECTED_HOSTNAME="example-gpu-node-6"' in source
    assert 'readonly EXPECTED_DRIVER_VERSION="570.158.01"' in source
    assert 'readonly EXPECTED_GPU_NAME_FRAGMENT="A800-SXM4-40GB"' in source
    assert '"$PRIVATE_PLAN_INPUT" != /mnt/ceph2' in source
    assert "sudo" not in source
    assert " docker " not in source
    assert 'print("kit=false")' in source
    assert 'print("renderer=false")' in source
    assert 'print("camera=false")' in source
    assert "omni.kit" in source and "omni.renderer" in source


def test_launcher_checks_canonical_plan_snapshot_asset_and_source_hashes() -> None:
    source = _source()

    assert "load_private_plan(" in source
    assert "public_protocol=public" in source
    assert "expected_sha256=expected_hash" in source
    assert 'readonly PUBLIC_PROTOCOL="$PROJECT_DIR/configs/parity/ovphysx_legacy_friction_freeze_b.json"' in source
    assert 'PUBLIC_PROTOCOL_FILE_SHA256="$(sha256sum -- "$PUBLIC_PROTOCOL"' in source
    assert '"public_protocol_canonical_sha256"' in source
    assert '"implementation_source_revision"' in source
    assert '"deployment_source_revision"' in source
    assert 'PRIVATE_PLAN_FILE_SHA256="$(sha256sum -- "$PRIVATE_PLAN"' in source
    assert '"$GATE0_ROOT/inputs/freeze-b/$PRIVATE_PLAN_SHA256.json"' in source
    assert 'find "$PRIVATE_PLAN" -perm /077' in source
    assert 'verify_plan' in source
    assert source.count('verify_snapshot "') >= 4
    assert 'hashlib.sha256(b"waveqa-source-snapshot-v1\\0")' in source
    assert 'sha256sum -- "$SOURCE_ARCHIVE"' in source
    assert 'git -C "$ASSET_ROOT" rev-parse HEAD' in source
    assert '[[ "$actual_asset_hash" == "$ASSET_LF_SHA256" ]]' in source
    post_matrix = source.index('verify_snapshot "post-matrix"')
    assert source.index("verify_pinned_repositories", post_matrix) > post_matrix
    assert source.index("check_root_budget", post_matrix) > post_matrix
    assert '[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_ovphysx_freeze_b_remote.sh" ]]' in source

    matrix_start = source.index("mapfile -t CASE_RECORDS")
    heredoc_start = source.index("<<'PY'\n", matrix_start) + len("<<'PY'\n")
    heredoc_end = source.index("\nPY\n)", heredoc_start)
    matrix_tree = ast.parse(source[heredoc_start:heredoc_end])
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "hashlib"
        and any(alias.name == "sha256" for alias in node.names)
        for node in ast.walk(matrix_tree)
    )


def test_all_remote_python_heredocs_compile_without_undefined_globals() -> None:
    source = _source()
    blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", source, flags=re.DOTALL)

    assert len(blocks) == 9
    builtin_names = set(dir(builtins))
    for index, block in enumerate(blocks, start=1):
        compile(block, f"<remote-heredoc-{index}>", "exec")
        table = symtable.symtable(block, f"<remote-heredoc-{index}>", "exec")
        undefined = sorted(
            name
            for name in table.get_identifiers()
            if (symbol := table.lookup(name)).is_referenced()
            and not (
                symbol.is_assigned()
                or symbol.is_imported()
                or symbol.is_parameter()
            )
            and name not in builtin_names
        )
        assert undefined == [], f"heredoc {index} undefined globals: {undefined}"


def test_launcher_runs_16_fresh_plan_ordered_workers_with_exact_contract() -> None:
    source = _source()

    assert '[[ "${#CASE_RECORDS[@]}" -eq 16 ]]' in source
    assert 'for record in "${CASE_RECORDS[@]}"; do' in source
    assert 'os.setsid()' in source
    assert 'os.execvpe(command[0], command, os.environ.copy())' in source
    assert '[[ "$COMPLETED_CASES" -eq 16 && "${#FRESH_PROCESS_IDS[@]}" -eq 16 ]]' in source
    assert "worker reused a fresh process identity" in source
    assert "printf '%s|%s\\n' \"$case_id\" \"$fresh_id\"" in source
    assert 'WAVEQA_SOURCE_TREE="$SOURCE_REVISION"' in source
    assert '"gpu_uuid": selected_gpu_uuid' in source
    assert '"gpu_name": selected_gpu_name' in source
    assert '"driver_version": selected_driver_version' in source
    assert 'if p.get("constraints") != expected_constraints:' in source
    assert 'process_identity = p.get("os_process_identity")' in source
    assert 'p.get("worker_pid") != pid' in source
    assert 'observed_fresh = sha256(' in source
    assert 'p.get("forbidden_modules") != []' in source
    assert '"maximum_gpu_hours": 0.5' in source
    for fragment in (
        '--approved-root "$GATE0_ROOT"',
        '--asset-root "$ASSET_ROOT"',
        '--manifest "$GATE0_MANIFEST"',
        '--public-protocol "$PUBLIC_PROTOCOL"',
        '--public-protocol-file-sha256 "$PUBLIC_PROTOCOL_FILE_SHA256"',
        '--public-protocol-canonical-sha256 "$PUBLIC_PROTOCOL_CANONICAL_SHA256"',
        '--private-plan "$PRIVATE_PLAN"',
        '--private-plan-sha256 "$PRIVATE_PLAN_SHA256"',
        '--case-id "$case_id"',
        '--output "$payload_path"',
        '--device cuda:0',
        '--session-id "$SESSION_ID"',
        '--source-revision "$SOURCE_REVISION"',
        '--asset-tree-sha256 "$ASSET_LF_SHA256"',
    ):
        assert fragment in source


def test_launcher_payload_validation_keeps_pinned_site_packages_available() -> None:
    source = _source()
    start = source.index("validate_case_payload() {")
    end = source.index("\n}\n", start)
    validator = source[start:end]

    assert '"$WORKER_PYTHON" -P - \\' in validator
    assert '"$WORKER_PYTHON" -P -S -' not in validator

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-",
        ],
        cwd=ROOT,
        env=environment,
        input=(
            "from wave_asset_qa.parity.runner "
            "import adapter_run_result_from_dict\n"
            "print(adapter_run_result_from_dict.__name__)\n"
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "adapter_run_result_from_dict"


def test_launcher_never_uses_a_busy_gpu_or_signals_foreign_processes() -> None:
    source = _source()

    assert '[[ "$GPU_INDEX" =~ ^[0-7]$ ]]' in source
    assert '(( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB ))' in source
    assert '[[ "$LAST_GPU_UTILIZATION" == 0 && -z "$processes" ]]' in source
    assert 'process_pgid" != "$owner_pgid"' in source
    assert 'terminate_owned_case "$owner_pid" "$owner_pgid"' in source
    assert 'kill -TERM -- "-$owner_pgid"' in source
    assert 'kill -KILL -- "-$owner_pgid"' in source
    assert 'kill -TERM -- "-$process_pid"' not in source
    assert "only the owned case process group was terminated" in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in source


def test_launcher_enforces_half_gpu_hour_and_500_mib_evidence_budgets() -> None:
    source = _source()

    assert 'readonly GLOBAL_GPU_TIMEOUT_S=$((30 * 60))' in source
    assert 'readonly MAX_RUN_BYTES=$((500 * 1024 * 1024))' in source
    assert 'readonly MAX_ARTIFACT_BYTES=$((500 * 1024 * 1024))' in source
    assert "ulimit -f" not in source
    assert 'timeout --foreground --signal=TERM --kill-after=30s "${envelope}s"' in source
    assert 'global 0.5 GPU-hour budget is exhausted' in source
    assert source.count("check_run_budget") >= 3
    assert source.count("check_root_budget") >= 3


def test_launcher_never_overwrites_and_records_hash_inventories_and_status() -> None:
    source = _source()

    readonly_preflight = source.index(
        "# Do not import project Python, create a lock, or write any managed path"
    )
    readonly_call = source.index("if ! read_gpu_state; then", readonly_preflight)
    first_project_python = source.index('environment_summary="$("$WORKER_PYTHON"')
    lock = source.index('flock -n 9 || die "another Freeze B launcher holds the run lock"')
    locked_recheck = source.index(
        'verify_snapshot "locked-readonly-preflight"'
    )
    candidate = source.index('run_root_candidate="$GATE0_ROOT/results/$RUN_ID"')
    nonexistence = source.index(
        '[[ ! -e "$run_root_candidate" && ! -L "$run_root_candidate" ]]'
    )
    initialize = source.index(
        'run_initializing_candidate="$GATE0_ROOT/results/.${RUN_ID}.initializing.$$"'
    )
    mkdir_root = source.index('mkdir "$run_initializing_candidate"')
    own_initializing = source.index(
        'INITIALIZING_ROOT="$run_initializing_candidate"'
    )
    mkdir_children = source.index(
        'mkdir "$run_initializing_candidate/launcher" "$run_initializing_candidate/cases"'
    )
    promote = source.index(
        'mv -T -n -- "$run_initializing_candidate" "$run_root_candidate"'
    )
    release_initializing = source.index('INITIALIZING_ROOT=""', promote)
    assign_run_root = source.index('RUN_ROOT="$run_root_candidate"')
    redirect = source.index('exec 1>"$RUN_ROOT/launcher/stdout.txt"')
    persisted_gpu = source.index(
        'capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt"'
    )
    assert (
        readonly_preflight
        < readonly_call
        < first_project_python
        < lock
        < locked_recheck
        < candidate
        < nonexistence
        < initialize
        < mkdir_root
        < own_initializing
        < mkdir_children
        < promote
        < assign_run_root
        < release_initializing
        < redirect
        < persisted_gpu
    )
    assert 'RUN_ROOT=""' in source
    assert 'INITIALIZING_ROOT=""' in source
    assert 'INITIALIZING_DEVICE_INODE=""' in source
    assert 'cleanup_initializing_root' in source
    assert 'adopt_promoted_initializing_root' in source
    assert "INITIALIZING_DEVICE_INODE=\"$(stat -c '%d:%i'" in source
    assert 'observed="$(stat -c \'%d:%i\'' in source
    finish_start = source.index("finish() {")
    assert source.index("adopt_promoted_initializing_root", finish_start) < (
        source.index("cleanup_initializing_root", finish_start)
    )
    assert 'rmdir -- "$INITIALIZING_ROOT/launcher"' in source
    assert 'rmdir -- "$INITIALIZING_ROOT/cases"' in source
    assert 'rmdir -- "$INITIALIZING_ROOT"' in source
    assert "run initialization promotion did not complete exactly once" in source
    assert "export PYTHONDONTWRITEBYTECODE=1" in source
    assert '[[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0' in source
    assert "run directory already exists; refusing overwrite" in source
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in source
    assert 'write_case_inventory "$case_dir"' in source
    assert "write_top_level_inventory" in source
    assert 'launcher/evidence.sha256' in source
    assert '"formal_gate0_status_unchanged": "DIVERGENT"' in source
    assert '"formal_gate0_pass_ready_unchanged": False' in source
    assert 'RUN_STATUS="collected"' in source
    assert 'SCIENTIFIC_VERDICT="pending_local_validation"' in source


def test_worker_python_dict_literals_have_no_duplicate_keys() -> None:
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    duplicates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        duplicates.extend(
            (node.lineno, key) for key in sorted(set(keys)) if keys.count(key) > 1
        )
    assert duplicates == []


def test_launcher_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            bash = str(git_bash)
    if bash is None:
        pytest.skip("bash is unavailable")
    completed = subprocess.run(
        [bash, "-n", str(LAUNCHER_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_worker_is_python_syntax_valid() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(WORKER_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
