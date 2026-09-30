from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ovphysx_effective_readback_remote.sh"
FREEZE_CONFIG = ROOT / "configs" / "parity" / "ovphysx_effective_readback.json"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _stdout_validator_program() -> str:
    source = _source()
    function = source.split("validate_worker_stdout() {", 1)[1]
    return function.split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]


def test_launcher_is_bound_to_server6_and_the_approved_root() -> None:
    source = _source()

    assert 'readonly GATE0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"' in source
    assert 'readonly EXPECTED_HOSTNAME="example-gpu-node-6"' in source
    assert 'readonly EXPECTED_USER="exampleuser"' in source
    assert 'readonly EXPECTED_OS_VERSION="22.04"' in source
    assert 'readonly EXPECTED_DRIVER_VERSION="570.158.01"' in source
    assert 'readonly EXPECTED_GPU_NAME_FRAGMENT="A800-SXM4-40GB"' in source
    assert '[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2' in source
    assert '[[ "$PROJECT_DIR" == "$GATE0_ROOT/project/"* ]]' in source
    assert '[[ "$PROJECT_DIR" == "$GATE0_ROOT/project/$SOURCE_TREE" ]]' in source
    assert 'findmnt -n -o TARGET -T "$GATE0_ROOT"' in source
    assert 'findmnt -n -o FSTYPE -T "$GATE0_ROOT"' in source
    assert "sudo" not in source
    assert " docker " not in source


def test_launcher_requires_an_immutable_source_snapshot() -> None:
    source = _source()

    for marker in (
        ".source-revision",
        ".source-tree",
        ".archive-sha256",
        ".snapshot-sha256",
    ):
        assert marker in source
    assert '[[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]]' in source
    assert 'writable="$(find "$PROJECT_DIR" -perm /222 -print -quit)"' in source
    assert 'hashlib.sha256(b"waveqa-source-snapshot-v1\\0")' in source
    assert 'verify_snapshot "preflight"' in source
    assert 'verify_snapshot "before-$case_id"' in source
    assert 'verify_snapshot "post-matrix"' in source
    assert 'sha256sum -- "$SOURCE_ARCHIVE"' in source
    assert '[[ "$actual_archive_sha256" == "$ARCHIVE_SHA256" ]]' in source
    assert '[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_ovphysx_effective_readback_remote.sh" ]]' in source


def test_launcher_derives_exactly_the_four_frozen_instances() -> None:
    source = _source()
    config = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))

    expected = [
        ("left", 1, "ovphysx.left.effective_readback.r01"),
        ("left", 2, "ovphysx.left.effective_readback.r02"),
        ("right", 1, "ovphysx.right.effective_readback.r01"),
        ("right", 2, "ovphysx.right.effective_readback.r02"),
    ]
    observed = [
        (item["side"], item["instance_index"], item["case_id"])
        for item in config["expected_instances"]
    ]
    assert observed == expected
    for side, index, case_id in expected:
        assert f'("{side}", {index}, "{case_id}")' in source

    assert '[[ "${#matrix_lines[@]}" -eq 7 ]]' in source
    assert 'CASE_RECORDS=("${matrix_lines[@]:3}")' in source
    assert 'for case_record in "${CASE_RECORDS[@]}"; do' in source
    assert '[[ "$COMPLETED_CASES" -eq 4 ]]' in source


def test_launcher_rejects_a_changed_freeze_or_gate0_context() -> None:
    source = _source()

    for field in (
        '"fresh_instances_per_hand": 2',
        '"total_instance_count": 4',
        '"fresh_process_per_instance": True',
        '"reuse_of_articulation_or_physics_scene_forbidden": True',
        '"required_user_trace_simulation_step_call_count": 0',
        '"required_trajectory_sample_count": 0',
        '"required_trace_time_s": 0.0',
        '"renderer_allowed": False',
        '"camera_allowed": False',
    ):
        assert field in source
    assert 'freeze_context.get("gate0_manifest_file_sha256") != manifest_file_sha' in source
    assert 'freeze_context.get("gate0_manifest_semantic_sha256") != manifest_semantic_sha' in source
    assert '"asset_commit": "6eea427eb24189519f32b9f21674cd534d3f973c"' in source
    assert '"asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1"' in source
    assert '"canonical_lf_asset_tree_sha256": "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"' in source


def test_launcher_invokes_the_pinned_worker_contract_in_fresh_processes() -> None:
    source = _source()

    assert 'readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_effective_params.py"' in source
    for fragment in (
        '--approved-root "$GATE0_ROOT"',
        '--asset-root "$ASSET_ROOT"',
        '--gate0-manifest "$GATE0_MANIFEST"',
        '--freeze-config "$FREEZE_CONFIG"',
        '--case-id "$case_id"',
        '--side "$side"',
        '--instance-index "$instance_index"',
        '--output "$payload_path"',
        '--device cuda:0',
        '--session-id "$SESSION_ID"',
        '--source-revision "$SOURCE_REVISION"',
        '--source-tree "$SOURCE_TREE"',
        '--source-archive-sha256 "$ARCHIVE_SHA256"',
        '--asset-tree-sha256 "$ASSET_LF_SHA256"',
    ):
        assert fragment in source
    assert 'os.setsid()' in source
    assert 'os.execvpe(command[0], command, os.environ.copy())' in source
    assert 'case_owner_pid="$!"' in source
    assert 'ACTIVE_CASE_PID="$case_owner_pid"' in source
    assert 'ACTIVE_CASE_PGID="$case_pgid"' in source
    assert 'timeout --foreground --signal=TERM --kill-after=30s "${envelope}s"' in source


def test_launcher_uses_one_explicit_idle_gpu_and_never_kills_foreign_pids() -> None:
    source = _source()

    assert '[[ "$GPU_INDEX" =~ ^[0-7]$ ]]' in source
    assert 'CUDA_DEVICE_ORDER=PCI_BUS_ID' in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in source
    assert '--device cuda:0' in source
    assert 'nvidia-smi -i "$GPU_INDEX"' in source
    assert '(( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB ))' in source
    assert '[[ "$LAST_GPU_UTILIZATION" == "0" ]]' in source
    assert '[[ -z "$processes" ]]' in source
    assert 'process_pgid" != "$owner_pgid"' in source
    assert 'terminate_owned_case "$owner_pid" "$owner_pgid"' in source
    assert 'kill -TERM -- "-$owner_pgid"' in source
    assert 'kill -KILL -- "-$owner_pgid"' in source
    assert 'kill -TERM -- "-$process_pid"' not in source
    assert 'only the owned case process group was terminated' in source
    assert 'capture_gpu_state "$case_dir/preflight.txt" "$SELECTED_GPU_UUID"' in source
    assert "postflight_attempt in 1 2 3" in source


def test_launcher_is_headless_kitless_and_has_no_trajectory_runner() -> None:
    source = _source()

    assert '[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]' in source
    assert "unset DISPLAY WAYLAND_DISPLAY" in source
    assert "if has_kit():" in source
    assert 'name == "omni.kit"' in source
    assert 'name == "omni.renderer"' in source
    assert 'print("renderer=false")' in source
    assert 'print("camera=false")' in source
    assert 'export HOME=' not in source
    assert "wave_asset_qa.parity.runner" not in source
    assert "probe_ovphysx_runtime.py" not in source


def test_launcher_never_overwrites_results_and_records_complete_evidence() -> None:
    source = _source()

    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in source
    assert 'run directory already exists; refusing overwrite' in source
    assert '[[ ! -e "$case_dir" && ! -L "$case_dir" ]]' in source
    assert 'os.O_WRONLY | os.O_CREAT | os.O_EXCL' in source
    assert '> "$case_dir/worker.stdout.txt" 2> "$case_dir/worker.stderr.txt"' in source
    assert 'printf \'%s\\n\' "$worker_exit_code" > "$case_dir/worker.exit-code.txt"' in source
    assert 'case_pid_file="$case_dir/owned-process-group.txt"' in source
    assert 'payload_path="$case_dir/$case_id.json"' in source
    assert 'write_case_inventory "$case_dir"' in source
    assert 'local inventory="$directory/evidence.sha256"' in source
    assert 'write_top_level_inventory' in source
    assert 'RUN_ROOT/launcher/evidence.sha256' in source
    assert "find . -type f" in source
    assert "sha256sum \"$evidence_path\"" in source


def test_launcher_limits_retained_evidence_without_capping_worker_temp_files() -> None:
    source = _source()

    assert "ulimit -f" not in source
    assert 'readonly MAX_ARTIFACT_BYTES=$((50 * 1024 * 1024))' in source
    assert 'readonly MAX_RUN_BYTES=$((200 * 1024 * 1024))' in source
    assert 'readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))' in source
    assert '-size +"${MAX_ARTIFACT_BYTES}"c' in source
    assert '[[ "$bytes" -le "$MAX_RUN_BYTES" ]]' in source
    assert '[[ "$bytes" -le "$MAX_ROOT_BYTES" ]]' in source
    assert source.count("check_run_budget") >= 4
    assert source.count("check_root_budget") >= 3
    assert 'timeout --foreground --signal=TERM --kill-after=30s "${envelope}s"' in source


def test_launcher_status_and_provenance_keep_the_claim_boundary() -> None:
    source = _source()

    for field in (
        '"experiment_id"',
        '"source_revision"',
        '"source_tree"',
        '"source_archive_sha256"',
        '"snapshot_sha256"',
        '"freeze_a_config_sha256"',
        '"gate0_manifest_file_sha256"',
        '"gate0_manifest_semantic_sha256"',
        '"launcher_sha256"',
        '"probe_source_sha256"',
        '"selected_gpu_index"',
        '"selected_gpu_uuid"',
        '"completed_case_count"',
    ):
        assert field in source
    assert 'RUN_STATUS="collected"' in source
    assert 'SCIENTIFIC_VERDICT="pending_local_validation"' in source
    success_tail = source[source.index('RUN_STATUS="collected"') :]
    assert 'CURRENT_CASE=""' not in success_tail
    assert '"last_case_id": json.loads(last_case_literal)' in source
    assert "no causal or formal Gate 0 conclusion was evaluated" in source
    assert "Gate 0 remains" not in source


def test_launcher_validates_the_worker_stdout_basename_without_leaking_a_path() -> None:
    source = _source()

    assert (
        'validate_worker_stdout "$case_dir/worker.stdout.txt" "$case_id" '
        '"$case_id.json"'
    ) in source
    assert (
        'validate_worker_stdout "$case_dir/worker.stdout.txt" "$case_id" '
        '"$payload_path"'
    ) not in source
    assert 'expected = {"status": "collected", "case_id": expected_case, "output": expected_output}' in source


def test_launcher_checks_structured_outcome_before_raw_exit_and_payload() -> None:
    source = _source()

    postflight = source.rindex('[[ "$postflight_ok" == true ]]')
    stdout_validation = source.rindex(
        'validate_worker_stdout "$case_dir/worker.stdout.txt" "$case_id"'
    )
    raw_exit = source.rindex('[[ "$worker_exit_code" -eq 0 ]]')
    payload = source.rindex('[[ -f "$payload_path" && ! -L "$payload_path" ]]')
    assert postflight < stdout_validation < raw_exit < payload
    assert (
        'printf \'%s\\n\' "$worker_exit_code" > "$case_dir/worker.exit-code.txt"'
        in source
    )


def test_stdout_validator_accepts_native_warnings_plus_one_summary(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "worker.stdout.txt"
    stdout_path.write_text(
        "PhysX warning: native diagnostic before collection\n"
        '{"case_id":"ovphysx.left.effective_readback.r01",'
        '"output":"ovphysx.left.effective_readback.r01.json",'
        '"status":"collected"}\n'
        "native shutdown warning\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _stdout_validator_program(),
            str(stdout_path),
            "ovphysx.left.effective_readback.r01",
            "ovphysx.left.effective_readback.r01.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_stdout_validator_rejects_worker_error_even_if_os_exit_was_zero(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "worker.stdout.txt"
    stdout_path.write_text(
        "native warning\n"
        '{"case_id":"ovphysx.left.effective_readback.r01",'
        '"error":{"message":"schema attribute absent","type":"RuntimeError"},'
        '"status":"worker_error"}\n'
        "native teardown warning\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _stdout_validator_program(),
            str(stdout_path),
            "ovphysx.left.effective_readback.r01",
            "ovphysx.left.effective_readback.r01.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "worker stdout summary mismatch" in completed.stderr


def test_stdout_validator_rejects_a_second_valid_json_line(tmp_path: Path) -> None:
    stdout_path = tmp_path / "worker.stdout.txt"
    summary = (
        '{"case_id":"ovphysx.left.effective_readback.r01",'
        '"output":"ovphysx.left.effective_readback.r01.json",'
        '"status":"collected"}'
    )
    stdout_path.write_text(summary + "\n{}\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _stdout_validator_program(),
            str(stdout_path),
            "ovphysx.left.effective_readback.r01",
            "ovphysx.left.effective_readback.r01.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "exactly one valid JSON value" in completed.stderr


@pytest.mark.parametrize(
    "invalid_json",
    (
        '{"status":"collected","status":"collected"}',
        '{"status":"collected","value":NaN}',
    ),
)
def test_stdout_validator_rejects_non_strict_json(
    tmp_path: Path,
    invalid_json: str,
) -> None:
    stdout_path = tmp_path / "worker.stdout.txt"
    stdout_path.write_text(invalid_json + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _stdout_validator_program(),
            str(stdout_path),
            "ovphysx.left.effective_readback.r01",
            "ovphysx.left.effective_readback.r01.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "worker stdout contains invalid JSON" in completed.stderr


def test_launcher_fail_closed_validation_rejects_partial_case_payloads() -> None:
    source = _source()

    assert "parse_constant=reject_constant" in source
    assert 'raise ValueError(f"duplicate JSON key: {key}")' in source
    assert '"readback_only_after_initialization": True' in source
    assert '"experimental_trace_dynamics_advance_performed": False' in source
    assert '"experimental_control_command_applied": False' in source
    assert '"experimental_target_write_performed": False' in source
    assert '"user_trace_simulation_step_call_count": 0' in source
    assert '"trajectory_sample_count": 0' in source
    assert 'if not isinstance(dof_records, list) or len(dof_records) != 22' in source
    assert 'if canonical_names != expected_joint_names or len(set(canonical_names)) != 22' in source
    assert 'if len(set(backend_indices)) != 22' in source
    assert 'set(canonical_hashes) != required_hashes' in source
    assert 'if isinstance(value, float) and not math.isfinite(value)' in source
    assert 're.fullmatch(r"[0-9a-f]{64}", fresh_process_id)' in source
    assert 'worker stdout must contain exactly one valid JSON value' in source
    assert '"status": "collected"' in source
    assert 'declare -a FRESH_PROCESS_IDS=()' in source
    assert 'declare -a FRESH_INSTANCE_IDS=()' in source
    assert 'worker reused a fresh process identity' in source
    assert 'worker reused a fresh instance identity' in source


def test_launcher_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            bash = str(git_bash)
    if bash is None:
        pytest.skip("bash is not available for a syntax-only check")

    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
