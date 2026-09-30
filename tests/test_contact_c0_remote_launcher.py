from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_contact_c0_remote.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _bash() -> str | None:
    bash = shutil.which("bash")
    if bash is not None:
        return bash
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if git_bash.is_file() else None


def _environment_preflight(source: str) -> str:
    start = source.index('environment_summary="$(')
    end = source.index('\nmatrix_output="$', start)
    return source[start:end]


def _assert_environment_preflight_boundary(source: str) -> None:
    preflight = _environment_preflight(source)
    for prefix in (
        "isaacsim",
        "omni.kit",
        "omni.renderer",
        "omni.replicator",
        "omni.syntheticdata",
    ):
        assert f'name == "{prefix}"' in preflight
        assert f'name.startswith("{prefix}.")' in preflight
    assert 'name == "isaaclab.sensors.camera"' not in preflight
    assert 'name.startswith("isaaclab.sensors.camera.")' not in preflight
    assert "passive transitive namespace import" in preflight
    assert "zero UsdGeom.Camera prims" in preflight
    assert "exact ContactSensor-only inventory" in preflight
    assert "forbidden Kit/renderer/camera modules loaded" in preflight
    assert '"kitless": not kit_runtime and not kit_modules' in preflight
    assert '"renderer": bool(renderer_modules)' in preflight
    assert '"camera": bool(camera_runtime_modules)' in preflight
    assert "**runtime_boundary" in preflight
    assert '"camera": False' not in preflight


def test_contact_remote_launcher_pins_scope_inputs_and_budgets() -> None:
    source = _source()

    assert source.startswith("#!/usr/bin/env bash\n\nset -euo pipefail\numask 077\n")
    assert (
        'readonly CONTACT_C0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"'
        in source
    )
    assert 'readonly EXPECTED_MANIFEST_ID="wavesimparity-contact-c0"' in source
    assert (
        'readonly OVPHYSX_SDK_ARCHIVE="/data/home/exampleuser/sharpa-wave-asset-qa-gate0/'
        'dependencies/ovphysx-0.4.13/ovphysx-linux-x86_64-0.4.13.tar.gz"'
        in source
    )
    assert (
        'readonly OVPHYSX_SDK_ARCHIVE_SHA256="191dcaff34980f6fdf94bb783c8faacbab'
        '058aa4c2671e31ac306fcd89cdb1e7"'
        in source
    )
    assert "readonly EXPECTED_OVPHYSX_CASE_COUNT=16" in source
    assert "readonly MAX_RUN_BYTES=$((2 * 1024 * 1024 * 1024))" in source
    assert "readonly MAX_ARTIFACT_BYTES=$((2 * 1024 * 1024 * 1024))" in source
    assert "readonly FINAL_METADATA_RESERVE_BYTES=$((4 * 1024 * 1024))" in source
    assert "readonly WORKER_TIMEOUT_S=1200" in source
    assert "readonly CASE_ENVELOPE_TIMEOUT_S=1260" in source
    assert "readonly GLOBAL_GPU_TIMEOUT_S=$((4 * 60 * 60))" in source
    assert '[[ "$GPU_INDEX" =~ ^[0-7]$ ]]' in source
    assert '[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]' in source
    assert '[[ "$SOURCE_TREE_INPUT" =~ ^[0-9a-f]{40}$ ]]' in source
    assert '[[ "$SNAPSHOT_SHA256_INPUT" =~ ^[0-9a-f]{64}$ ]]' in source
    for option in (
        "--project-dir)",
        "--run-id)",
        "--gpu-index)",
        "--session-id)",
        "--source-revision)",
        "--source-tree)",
        "--snapshot-sha256)",
    ):
        assert source.count(option) == 1
    assert 'die "unsupported option: $1"' in source
    assert not re.search(r"(?m)^\s*sudo(?:\s|$)", source)
    assert not re.search(r"(?m)^\s*(?:docker|conda)(?:\s|$)", source)
    assert not re.search(r"(?m)^\s*(?:curl|wget)(?:\s|$)", source)
    assert "/mnt/ceph2" in source


def test_contact_remote_launcher_binds_exact_immutable_snapshot() -> None:
    source = _source()

    for marker in (
        ".source-revision",
        ".source-tree",
        ".archive-sha256",
        ".snapshot-sha256",
    ):
        assert marker in source
    assert '[[ "$SOURCE_TREE" == "$SOURCE_TREE_INPUT" ]]' in source
    assert '[[ "$SNAPSHOT_SHA256" == "$SNAPSHOT_SHA256_INPUT" ]]' in source
    assert '[[ "$PROJECT_DIR" == "$CONTACT_C0_ROOT/project/$SOURCE_TREE" ]]' in source
    assert 'readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"' in source
    assert 'sha256sum -- "$SOURCE_ARCHIVE"' in source
    assert 'hashlib.sha256(b"waveqa-source-snapshot-v1\\0")' in source
    assert 'verify_snapshot "preflight"' in source
    assert 'verify_snapshot "before-$case_id"' in source
    assert 'verify_snapshot "post-matrix"' in source
    assert 'find "$PROJECT_DIR" -perm /222' in source
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in source
    assert "run directory already exists; refusing overwrite or automatic retry" in source
    assert "RUN_CREATED=false" in source
    assert source.index("RUN_CREATED=true") > source.index('mkdir "$RUN_ROOT/launcher" "$RUN_ROOT/cases"')
    assert 'if [[ "$RUN_CREATED" == true && -n "$RUN_ROOT"' in source
    assert 'flock -n 9 || die "another Contact C0 launcher' in source
    assert not re.search(r"(?m)^\s*rm(?:\s|$)", source)


def test_contact_remote_launcher_requires_three_strict_idle_checks_per_case() -> None:
    source = _source()

    assert "readonly MAX_IDLE_MEMORY_MIB=32" in source
    assert "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu" in source
    assert "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in source
    assert "(( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB ))" in source
    assert '[[ "$LAST_GPU_UTILIZATION" == "0" ]]' in source
    assert '[[ -z "$processes" ]]' in source
    assert source.count('--format=csv,noheader,nounits)" || return 1') == 2
    assert "compute-process query failed while owned case was active" in source
    for name in ("gpu-preflight-01.txt", "gpu-preflight-02.txt", "gpu-preflight-03.txt"):
        assert name in source
    final_check = source.index('gpu-preflight-03.txt')
    spawn = source.index('case_pid_file="$case_dir/owned-process-group.txt"')
    assert final_check < spawn
    assert 'CUDA_VISIBLE_DEVICES="$WORKER_CUDA_SELECTOR"' in source
    assert 'WORKER_CUDA_SELECTOR="$GPU_INDEX"' in source
    assert "--device cuda:0" in source
    assert '"worker_visible_device": "cuda:0"' in source


def test_idle_probe_precedes_lock_and_run_creation() -> None:
    source = _source()

    initial_probe = 'if ! query_idle_gpu_state; then'
    lock_open = 'exec 9>"$CONTACT_C0_ROOT/.contact-c0-run.lock"'
    run_create = 'mkdir "$RUN_ROOT"'
    assert source.index(initial_probe) < source.index(lock_open) < source.index(run_create)
    query = source[source.index("query_idle_gpu_state() {") : source.index(
        "capture_gpu_state() {"
    )]
    assert "RUN_ROOT" not in query
    assert "mv --" not in query


def test_empty_gpu_process_envelope_has_no_phantom_row() -> None:
    source = _source()
    helper = source[
        source.index("write_compute_process_envelope() {") : source.index(
            "query_idle_gpu_state() {"
        )
    ]
    assert 'if [[ -n "$processes" ]]; then' in helper
    assert "printf '%s\\n' \"$processes\"" in helper
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    completed = subprocess.run(
        [bash],
        input=(
            helper
            + '\nwrite_compute_process_envelope ""\n'
            + 'write_compute_process_envelope "GPU-test, 123, 8"\n'
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
        "GPU-test, 123, 8\n"
        "compute_processes_end\n"
    )


def test_contact_remote_launcher_owns_only_its_case_process_group() -> None:
    source = _source()

    assert "os.setsid()" in source
    assert 'kill -TERM -- "-$owner_pgid"' in source
    assert 'kill -KILL -- "-$owner_pgid"' in source
    assert 'monitor_owned_case \\' in source
    assert "foreign GPU process pid=%s pgid=%s detected" in source
    assert "only the owned case process group was terminated" in source
    assert "worker process identity escaped owned PGID" in source
    assert 'process_pgid="$(ps -o pgid= -p "$process_pid"' in source
    assert "pkill" not in source
    assert "killall" not in source
    assert not re.search(r'kill -(?:TERM|KILL) "\$[A-Za-z_][A-Za-z0-9_]*"', source)
    assert source.count('--kill-after="${GPU_KILL_GRACE_S}s"') == 2
    assert '"${WORKER_TIMEOUT_S}s"' in source
    assert '"${envelope}s"' in source
    assert "usable=$((remaining - GPU_KILL_GRACE_S - GPU_DEADLINE_SAFETY_MARGIN_S))" in source


def test_contact_remote_launcher_runs_exact_fresh_worker_matrix() -> None:
    source = _source()

    assert 'readonly MANIFEST="$PROJECT_DIR/configs/parity/contact_c0.json"' in source
    assert (
        'WORKER_SCRIPT="$PROJECT_DIR/scripts/run_contact_c0_ovphysx_worker.py"'
        in source
    )
    assert "if len(cases) != 16 or len({case.case_id for case in cases}) != 16:" in source
    assert 'for case_id in "${CASE_IDS[@]}"; do' in source
    assert '[[ "$COMPLETED_CASES" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]]' in source
    assert '[[ "${#FRESH_PROCESS_IDS[@]}" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]]' in source
    assert 'FRESH_PROCESS_IDS["$fresh_process_id"]="$case_id"' in source
    for flag in (
        '--asset-root "$ASSET_ROOT"',
        '--manifest "$MANIFEST"',
        '--case-id "$case_id"',
        '--output-dir "$case_dir"',
        '--session-id "$SESSION_ID"',
        '--source-revision "$SOURCE_REVISION"',
        '--source-tree "$SOURCE_TREE"',
        '--asset-tree-sha256 "$ASSET_LF_SHA256"',
        "--device cuda:0",
    ):
        assert flag in source
    assert 'private-process.json' in source
    assert 'private-adapter-evidence.json' in source
    assert 'worker-group-binding.json' in source
    assert "load_and_validate_contact_run" in source
    assert "verify_adapter_private_evidence" in source


def test_contact_remote_launcher_builds_and_binds_native_ccd_helper() -> None:
    source = _source()
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "*.cpp text eol=lf" in attributes
    assert 'readonly NATIVE_HELPER_SOURCE_RELATIVE="src/wave_asset_qa/contact/native/physx_ccd_readback.cpp"' in source
    assert 'readonly NATIVE_HELPER_FILENAME="libwaveqa_physx_ccd_readback.so"' in source
    assert 'tarfile.open(fileobj=archive_handle, mode="r:gz")' in source
    assert "OVPhysX SDK archive does not have one exact PhysX include root" in source
    assert 'candidate / "PxArticulationLink.h"' in source
    assert '"PxRigidBody.h"' in source
    assert '"foundation" / "PxPhysicsVersion.h"' in source
    assert 'NATIVE_COMPILER="$(readlink -f -- "$(command -v g++)")"' in source
    for flag in (
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
    ):
        assert flag in source
    assert '"$NATIVE_HELPER_SOURCE"' in source
    assert '"$NATIVE_HELPER_PATH"' in source
    assert '"native_physx_ccd_helper": {' in source
    assert 'WAVEQA_PHYSX_CCD_HELPER_PATH="$NATIVE_HELPER_PATH"' in source
    assert 'WAVEQA_PHYSX_CCD_HELPER_SHA256="$NATIVE_HELPER_SHA256"' in source
    assert '"WAVEQA_PHYSX_CCD_HELPER_PATH": os.environ[' in source
    assert '"WAVEQA_PHYSX_CCD_HELPER_SHA256": os.environ[' in source
    assert source.count("verify_native_helper") >= 4


def test_contact_remote_launcher_keeps_kitless_simulation_only_boundary() -> None:
    source = _source()

    assert '[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]' in source
    assert "kit_runtime = bool(has_kit())" in source
    _assert_environment_preflight_boundary(source)
    assert '"hardware": False' in source
    assert '"sim2real": False' in source
    assert 'export HOME=' not in source
    assert 'export XDG_DATA_HOME="$CONTACT_C0_ROOT/data"' in source
    assert "unset PYTHONPATH" in source
    assert 'ulimit -f "$MAX_ARTIFACT_BLOCKS"' in source
    assert '[[ "$bytes" -le "$MAX_RUN_BYTES" ]]' in source


@pytest.mark.parametrize(
    "prefix",
    (
        "isaacsim",
        "omni.kit",
        "omni.renderer",
        "omni.replicator",
        "omni.syntheticdata",
    ),
)
def test_contact_remote_launcher_preflight_camera_guard_rejects_tampering(
    prefix: str,
) -> None:
    source = _source()
    _assert_environment_preflight_boundary(source)
    tampered = source.replace(
        f'name == "{prefix}"',
        f'name == "tampered.{prefix}"',
        1,
    )
    with pytest.raises(AssertionError):
        _assert_environment_preflight_boundary(tampered)


def test_contact_remote_launcher_writes_exact_terminal_evidence_contract() -> None:
    source = _source()

    assert '(os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") or "contact-c0") + "-ovphysx"' in source
    assert '"records": records' in source
    assert '"root_sha256": inventory_root_sha256(records)' in source
    assert 'if relative == "evidence-manifest.json":' in source
    assert '"backend": "ovphysx"' in source
    assert '"status": os.environ["CONTACT_STATUS"]' in source
    assert '"session_id": os.environ["CONTACT_STATUS_SESSION"]' in source
    assert '"source_revision": os.environ["CONTACT_STATUS_REVISION"]' in source
    assert '"source_tree": os.environ["CONTACT_STATUS_TREE"]' in source
    assert '"snapshot_sha256": os.environ["CONTACT_STATUS_SNAPSHOT"]' in source
    assert '"completed_case_count": int(os.environ["CONTACT_STATUS_COMPLETED"])' in source
    assert '"unique_process_count": int(os.environ["CONTACT_STATUS_UNIQUE"])' in source
    assert 'RUN_STATUS="completed"' in source
    assert 'SCIENTIFIC_VERDICT="pending_local_compare"' in source
    inventory_write = source.index('write_evidence_manifest || exit_code=2')
    restore = source.index('exec 1>&3 2>&4', inventory_write)
    assert inventory_write < restore


def test_contact_remote_launcher_has_valid_bash_syntax() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_contact_remote_launcher_fake_host_fails_before_external_state(
    tmp_path: Path,
) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    required = (
        "awk",
        "basename",
        "date",
        "dirname",
        "du",
        "env",
        "find",
        "findmnt",
        "flock",
        "g++",
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
            "contact-c0-fake-a1",
            "--gpu-index",
            "0",
            "--session-id",
            "contact-c0-fake-a1",
            "--source-revision",
            "b" * 40,
            "--source-tree",
            "a" * 40,
            "--snapshot-sha256",
            "c" * 64,
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
