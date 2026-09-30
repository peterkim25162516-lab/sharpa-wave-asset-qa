from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_waveform_t1_remote.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _bash() -> str | None:
    bash = shutil.which("bash")
    if bash is not None:
        return bash
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if git_bash.is_file() else None


def test_waveform_remote_launcher_pins_scope_inputs_and_hard_budgets() -> None:
    source = _source()

    assert source.startswith("#!/usr/bin/env bash\n\nset -euo pipefail\n")
    assert (
        'readonly WAVEFORM_T1_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"'
        in source
    )
    assert 'readonly EXPECTED_MANIFEST_ID="wavesimparity-waveform-t1"' in source
    assert 'readonly EXPECTED_OVPHYSX_CASE_COUNT=16' in source
    assert 'readonly MAX_RUN_BYTES=$((2 * 1024 * 1024 * 1024))' in source
    assert 'readonly MAX_ARTIFACT_BYTES=$((2 * 1024 * 1024 * 1024))' in source
    assert 'readonly GLOBAL_GPU_TIMEOUT_S=$((2 * 60 * 60))' in source
    assert 'readonly WORKER_TIMEOUT_S=900' in source
    assert 'readonly CASE_ENVELOPE_TIMEOUT_S=960' in source
    assert source.count('GPU_STARTED_EPOCH="$(date +%s)"') == 1
    assert source.index('GPU_STARTED_EPOCH="$(date +%s)"') < source.index(
        "for schema_hand in left right; do"
    )
    assert 'schema_envelope="$SCHEMA_PROBE_TIMEOUT_S"' in source
    assert '"${schema_envelope}s"' in source
    assert '[[ "$GPU_INDEX" =~ ^[0-7]$ ]]' in source
    assert '[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]' in source
    assert source.count("--project-dir)") == 1
    assert source.count("--run-id)") == 1
    assert source.count("--gpu-index)") == 1
    assert source.count("--session-id)") == 1
    assert source.count("--source-revision)") == 1
    assert 'die "unsupported option: $1"' in source
    assert not re.search(r"(?m)^\s*sudo(?:\s|$)", source)
    assert not re.search(r"(?m)^\s*(?:docker|conda)(?:\s|$)", source)


def test_waveform_remote_launcher_binds_immutable_project_and_exclusive_run() -> None:
    source = _source()

    assert 'read_marker "$PROJECT_DIR/.source-revision" MARKER_SOURCE_REVISION' in source
    assert 'read_marker "$PROJECT_DIR/.source-tree" SOURCE_TREE' in source
    assert 'read_marker "$PROJECT_DIR/.archive-sha256" ARCHIVE_SHA256' in source
    assert 'read_marker "$PROJECT_DIR/.snapshot-sha256" SNAPSHOT_SHA256' in source
    assert 'readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"' in source
    assert 'sha256sum -- "$SOURCE_ARCHIVE"' in source
    assert (
        '[[ "$PROJECT_DIR" == "$WAVEFORM_T1_ROOT/project/$SOURCE_TREE" ]]'
        in source
    )
    assert 'hashlib.sha256(b"waveqa-source-snapshot-v1\\0")' in source
    assert 'verify_snapshot "preflight"' in source
    assert 'verify_snapshot "before-$case_id"' in source
    assert 'verify_snapshot "post-matrix"' in source
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in source
    assert "run directory already exists; refusing overwrite" in source
    assert 'flock -n 9 || die "another Waveform T1 matrix launcher' in source
    assert not re.search(r"(?m)^\s*rm(?:\s|$)", source)


def test_waveform_remote_launcher_uses_one_strictly_idle_gpu_only() -> None:
    source = _source()

    assert 'readonly MAX_IDLE_MEMORY_MIB=32' in source
    assert "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu" in source
    assert "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in source
    assert '(( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB ))' in source
    assert '[[ "$LAST_GPU_UTILIZATION" == "0" ]]' in source
    assert '[[ -z "$processes" ]]' in source
    assert 'capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt"' in source
    assert '"$case_dir/preflight.txt" "$SELECTED_GPU_UUID"' in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in source
    assert "--device cuda:0" in source
    assert 'monitor_owned_case \\' in source
    assert "foreign GPU process pid=$process_pid" in source
    assert "os.setsid()" in source
    assert 'kill -TERM -- "-$owner_pgid"' in source
    assert "only the owned case process group was terminated" in source
    assert "pkill" not in source
    assert "killall" not in source
    assert 'kill -TERM "$schema_owner_pid"' not in source
    assert 'kill -TERM "$case_owner_pid"' not in source


def test_gpu_process_envelopes_omit_a_phantom_row_when_query_is_empty() -> None:
    source = _source()

    helper = source[
        source.index("write_compute_process_envelope() {") : source.index(
            "capture_gpu_state() {"
        )
    ]
    assert "if [[ -n \"$processes\" ]]; then" in helper
    assert "printf '%s\\n' \"$processes\"" in helper
    assert "printf 'compute_processes_begin\\n%s\\ncompute_processes_end\\n'" not in source
    assert source.count("write_compute_process_envelope \"$LAST_GPU_PROCESSES\"") == 1
    assert source.count("write_compute_process_envelope \"$processes\"") == 1
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available for the envelope behavior check")
    completed = subprocess.run(
        [bash],
        input=(
            helper
            + '\nwrite_compute_process_envelope ""\n'
            + 'write_compute_process_envelope "GPU-test, 1234, 8"\n'
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "compute_processes_begin\n"
        "compute_processes_end\n"
        "compute_processes_begin\n"
        "GPU-test, 1234, 8\n"
        "compute_processes_end\n"
    )


def test_idle_selection_is_query_only_before_lock_and_run_creation() -> None:
    source = _source()
    query_call = "if ! query_idle_gpu_state; then"
    lock_open = 'exec 9>"$WAVEFORM_T1_ROOT/.waveform-t1-run.lock"'
    run_create = 'mkdir "$RUN_ROOT"'
    recorded = (
        'capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt" '
        '"$SELECTED_GPU_UUID"'
    )

    assert source.index(query_call) < source.index(lock_open) < source.index(run_create)
    assert source.index(run_create) < source.index(recorded)
    query_body = source[source.index("query_idle_gpu_state() {") : source.index(
        "capture_gpu_state() {"
    )]
    assert "RUN_ROOT" not in query_body
    assert "mv --" not in query_body


def test_gpu_deadline_reserves_kill_grace_and_cleanup_margin() -> None:
    source = _source()

    assert "readonly GPU_KILL_GRACE_S=30" in source
    assert "readonly GPU_DEADLINE_SAFETY_MARGIN_S=30" in source
    reserve = (
        "usable=$((remaining - GPU_KILL_GRACE_S - "
        "GPU_DEADLINE_SAFETY_MARGIN_S))"
    )
    assert source.count(reserve) == 2
    assert source.count('--kill-after="${GPU_KILL_GRACE_S}s"') == 2
    assert '"gpu_elapsed_s": %s' in source


def test_owned_group_cleanup_survives_leader_exit_and_precedes_active_clear() -> None:
    source = _source()
    terminate = source[source.index("terminate_owned_case() {") : source.index(
        "write_launcher_inventory() {"
    )]

    assert '[[ -n "$observed_pgid" && "$observed_pgid" != "$owner_pgid" ]]' in terminate
    assert 'kill -0 -- "-$owner_pgid"' in terminate
    assert 'kill -KILL -- "-$owner_pgid"' in terminate
    assert "return 1" in terminate
    for owner in ("schema_owner_pid", "case_owner_pid"):
        cleanup = f'terminate_owned_case "${owner}"'
        start = source.index(cleanup, source.index(f'wait "${owner}"'))
        clear = source.index('ACTIVE_CASE_PID=""', start)
        assert start < clear
    assert (
        'foreign_reason="GPU process pid=$process_pid disappeared before '
        'owned-group identity could be proven"'
    ) in source


def test_waveform_remote_launcher_runs_two_probes_and_exact_fresh_case_matrix() -> None:
    source = _source()

    assert 'readonly MANIFEST="$PROJECT_DIR/configs/parity/waveform_t1.json"' in source
    assert 'manifest.schema_version != 2' in source
    assert 'manifest.manifest_id != "wavesimparity-waveform-t1"' in source
    assert '"offset_sine",' in source
    assert '"offset_linear_chirp",' in source
    assert 'if len(cases) != 16 or len({case.case_id for case in cases}) != 16:' in source
    assert 'for schema_hand in left right; do' in source
    assert '[[ "$COMPLETED_SCHEMA_PROBES" -eq 2 ]]' in source
    assert 'for case_id in "${CASE_IDS[@]}"; do' in source
    assert '[[ "$COMPLETED_CASES" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]]' in source
    assert 'readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_runtime.py"' in source
    for flag in (
        "--backend ovphysx",
        '--approved-root "$WAVEFORM_T1_ROOT"',
        '--worker-script "$WORKER_SCRIPT"',
        '--worker-python "$WORKER_PYTHON"',
        '--worker-timeout-s "$WORKER_TIMEOUT_S"',
        '--session-id "$SESSION_ID"',
        '--source-revision "$SOURCE_REVISION"',
        '--asset-root "$ASSET_ROOT"',
        '--manifest "$MANIFEST"',
        '--output-dir "$case_dir"',
        '--case-id "$case_id"',
    ):
        assert flag in source
    assert '[[ "$runner_exit_code" -eq 0 ]] || die' in source


def test_waveform_remote_launcher_validates_complete_target_sequences() -> None:
    source = _source()

    for field in (
        '"target_sequence_digest_schema_version"',
        '"target_sequence_digest_encoding"',
        '"target_sequence_digest_projection"',
        '"target_sequence_canonical_joint_names"',
        '"scheduled_target_sequence_count"',
        '"requested_target_sequence_count"',
        '"immediate_target_readback_sequence_count"',
        '"pre_step_applied_target_readback_sequence_count"',
        '"scheduled_target_sequence_sha256"',
        '"requested_target_sequence_sha256"',
        '"immediate_target_readback_sequence_sha256"',
        '"pre_step_applied_target_readback_sequence_sha256"',
        '"position_target_canonical_readback_max_abs_error_rad"',
        '"position_target_readback_values_rad"',
    ):
        assert field in source
    assert "canonical_target_sequence_sha256" in source
    assert "include_terminal=True" in source
    assert "include_terminal=False" in source
    assert '"ieee754_binary32_roundtrip"' in source
    assert "<= 1e-6" in source


def test_waveform_remote_launcher_proves_actual_workers_are_fresh() -> None:
    source = _source()

    assert "capture_actual_worker_identity()" in source
    assert 'Path("/proc/sys/kernel/random/boot_id")' in source
    assert '(process_dir / "stat").read_text' in source
    assert '"process_start_ticks": int(fields_after_comm[19])' in source
    assert 'int(fields_after_comm[2]) != owner_pgid' in source
    assert 'os.path.realpath(os.fsdecode(argv[0])) != worker_python' in source
    assert 'os.path.realpath(os.fsdecode(argv[1])) != worker_script' in source
    assert "validate_os_process_identity" in source
    assert "os_process_identity_sha256" in source
    assert '"visibility": "private_not_for_publication"' in source
    assert 'worker_identity_file="$case_dir/private-fresh-process-identity.json"' in source
    assert 'provenance.get("worker_pid") != worker_identity.get("pid")' in source
    assert 'declare -A FRESH_PROCESS_IDS=()' in source
    assert 'FRESH_PROCESS_IDS["$fresh_process_id"]="$case_id"' in source
    assert "actual Python worker identity was reused across canonical cases" in source
    assert '"${#FRESH_PROCESS_IDS[@]}" -eq "$EXPECTED_OVPHYSX_CASE_COUNT"' in source
    assert '"verified_fresh_process_count": %s' in source
    # The wrapper/session PID is retained only for scoped termination.  It is
    # never accepted as proof of the Python worker that produced the run JSON.
    assert 'worker_identity.get("pid")' in source
    assert 'case_owner_pid" == provenance.get("worker_pid")' not in source


def test_waveform_remote_launcher_keeps_kitless_and_write_boundaries() -> None:
    source = _source()

    assert '[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2' in source
    assert '[[ "$(findmnt -n -o TARGET -T "$WAVEFORM_T1_ROOT")" == "/data" ]]' in source
    assert '[[ "$(findmnt -n -o FSTYPE -T "$WAVEFORM_T1_ROOT")" == "xfs" ]]' in source
    assert '[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]' in source
    assert 'if has_kit():' in source
    assert 'name == "omni.kit"' in source
    assert 'name == "omni.renderer"' in source
    assert 'export HOME="$WAVEFORM_T1_ROOT/data"' in source
    assert "unset PYTHONPATH" in source
    assert 'require_writable_directory "$RUN_ROOT/launcher/schema"' in source
    assert 'require_writable_directory "$RUN_ROOT/cases"' in source
    assert 'require_writable_directory "$case_dir"' in source
    assert 'ulimit -f "$MAX_ARTIFACT_BLOCKS"' in source
    assert '[[ "$bytes" -le "$MAX_RUN_BYTES" ]]' in source
    assert 'run_bytes="$(du -sb "$RUN_ROOT" | awk \'{print $1}\')"' in source
    assert '[[ "$run_bytes" -gt "$MAX_RUN_BYTES" ]]' in source
    assert "Waveform T1 run evidence exceeded the 2 GiB hard limit" in source
    assert source.index('run_bytes="$(du -sb "$RUN_ROOT"') < source.index(
        'while IFS= read -r row; do'
    )
    assert 'RUN_STATUS="completed"' in source
    assert 'SCIENTIFIC_VERDICT="pending_local_compare"' in source
    assert "evidence retained" in source


def test_waveform_remote_launcher_has_valid_bash_syntax() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available for a syntax-only check")
    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_waveform_remote_launcher_fake_host_fails_before_external_state(
    tmp_path: Path,
) -> None:
    """A fake Server 6 identity must stop before any managed path is touched."""

    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available for the sandbox check")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    required = (
        "awk",
        "basename",
        "date",
        "df",
        "du",
        "env",
        "find",
        "findmnt",
        "flock",
        "git",
        "hostname",
        "id",
        "mkdir",
        "mv",
        "nvidia-smi",
        "ps",
        "readlink",
        "sha256sum",
        "sleep",
        "sort",
        "stat",
        "timeout",
    )
    for name in required:
        command = fake_bin / name
        body = "printf 'definitely-not-server-6\\n'\n" if name == "hostname" else ":\n"
        command.write_text("#!/usr/bin/env sh\n" + body, encoding="utf-8")
        command.chmod(0o700)
    env = dict(os.environ)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")

    completed = subprocess.run(
        [
            bash,
            str(SCRIPT),
            "--project-dir",
            "/data/home/exampleuser/sharpa-wave-asset-qa-gate0/project/" + "a" * 40,
            "--run-id",
            "waveform-t1-sandbox-a1",
            "--gpu-index",
            "0",
            "--session-id",
            "waveform-t1-sandbox-a1",
            "--source-revision",
            "b" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode == 2
    assert "expected Server 6 hostname example-gpu-node-6" in completed.stderr
    assert list(tmp_path.iterdir()) == [fake_bin]
