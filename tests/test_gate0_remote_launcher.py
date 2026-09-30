from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gate0_remote.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_launcher_has_fixed_boundaries_and_strict_inputs() -> None:
    source = _source()

    assert source.startswith("#!/usr/bin/env bash\n\nset -euo pipefail\n")
    assert 'readonly GATE0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"' in source
    assert 'readonly EXPECTED_HOSTNAME="example-gpu-node-6"' in source
    assert 'readonly EXPECTED_USER="exampleuser"' in source
    assert 'readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))' in source
    assert 'readonly MAX_RUN_BYTES=$((500 * 1024 * 1024))' in source
    assert 'readonly MAX_ARTIFACT_BYTES=$((500 * 1024 * 1024))' in source
    assert 'readonly WORKER_TIMEOUT_S=300' in source
    assert 'readonly GLOBAL_GPU_TIMEOUT_S=$((2 * 60 * 60))' in source
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


def test_remote_launcher_binds_an_immutable_source_snapshot() -> None:
    source = _source()

    assert 'read_marker "$PROJECT_DIR/.source-revision" MARKER_SOURCE_REVISION' in source
    assert 'read_marker "$PROJECT_DIR/.source-tree" SOURCE_TREE' in source
    assert 'read_marker "$PROJECT_DIR/.archive-sha256" ARCHIVE_SHA256' in source
    assert 'read_marker "$PROJECT_DIR/.snapshot-sha256" SNAPSHOT_SHA256' in source
    assert 'readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"' in source
    assert 'verify_source_archive' in source
    assert 'sha256sum -- "$SOURCE_ARCHIVE"' in source
    assert '[[ "$PROJECT_DIR" == "$GATE0_ROOT/project/$SOURCE_TREE" ]]' in source
    assert '[[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]]' in source
    assert 'find "$PROJECT_DIR" -perm /222 -print -quit' in source
    assert 'hashlib.sha256(b"waveqa-source-snapshot-v1\\0")' in source
    assert 'verify_snapshot "before-$case_id"' in source
    assert 'verify_snapshot "post-matrix"' in source
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in source
    assert "run directory already exists; refusing overwrite" in source
    assert not re.search(r"(?m)^\s*rm(?:\s|$)", source)


def test_remote_launcher_pins_and_rechecks_one_idle_gpu() -> None:
    source = _source()

    assert "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu" in source
    assert "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in source
    assert 'readonly MAX_IDLE_MEMORY_MIB=32' in source
    assert '(( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB ))' in source
    assert '[[ "$LAST_GPU_UTILIZATION" == "0" ]]' in source
    assert '[[ -z "$processes" ]]' in source
    assert '"$case_dir/preflight.txt" "$SELECTED_GPU_UUID"' in source
    assert "postflight-%02d.txt" in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in source
    assert "--device cuda:0" in source
    assert "--kill-after=30s" in source
    assert 'monitor_owned_case \\' in source
    assert 'foreign GPU process pid=$process_pid' in source
    assert 'os.setsid()' in source
    assert '[[ ! "$case_pgid" =~ ^[1-9][0-9]*$' in source
    assert 'kill -TERM -- "-$owner_pgid"' in source
    assert 'only the owned case process group was terminated' in source
    assert "pkill" not in source
    assert "killall" not in source


def test_remote_launcher_invokes_one_process_isolated_case_at_a_time() -> None:
    source = _source()

    required_runner_flags = (
        "--backend ovphysx",
        '--approved-root "$GATE0_ROOT"',
        '--worker-script "$WORKER_SCRIPT"',
        '--worker-python "$WORKER_PYTHON"',
        '--worker-timeout-s "$WORKER_TIMEOUT_S"',
        '--session-id "$SESSION_ID"',
        '--source-revision "$SOURCE_REVISION"',
        '--asset-root "$ASSET_ROOT"',
        '--manifest "$MANIFEST"',
        '--output-dir "$case_dir"',
        '--case-id "$case_id"',
        "--device cuda:0",
    )
    for flag in required_runner_flags:
        assert flag in source
    assert 'readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_runtime.py"' in source
    assert 'if len(cases) != 16 or len({case.case_id for case in cases}) != 16:' in source
    assert 'for case_id in "${CASE_IDS[@]}"; do' in source
    assert '[[ "$runner_exit_code" -eq 0 ]] || die' in source


def test_remote_launcher_validates_environment_and_retains_evidence() -> None:
    source = _source()

    for pinned_value in (
        "6eea427eb24189519f32b9f21674cd534d3f973c",
        "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
        "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad",
        "ffff603eafc6b74264a5261cc0183d6a65390d78",
        "cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522",
    ):
        assert pinned_value in source
    assert '[[ "$(findmnt -n -o TARGET -T "$GATE0_ROOT")" == "/data" ]]' in source
    assert '[[ "$(findmnt -n -o FSTYPE -T "$GATE0_ROOT")" == "xfs" ]]' in source
    assert "expected Python version" in source
    assert 'if has_kit():' in source
    assert 'name == "omni.kit"' in source
    assert 'name == "omni.renderer"' in source
    assert '[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]' in source
    assert 'cd "$RUN_ROOT"' in source
    assert 'export HOME="$GATE0_ROOT/data"' in source
    assert 'export PYTHONSAFEPATH=1' in source
    assert 'unset PYTHONPATH' in source
    assert '"$WORKER_PYTHON" -P -m wave_asset_qa.parity.runner' in source
    assert 'torch_cuda_build=' in source
    assert 'nvidia_cuda_runtime_cu12=' in source
    assert '1>"$RUN_ROOT/launcher/stdout.txt" 2>"$RUN_ROOT/launcher/stderr.txt"' in source
    assert 'launcher.stdout.txt"' in source
    assert 'launcher.stderr.txt"' in source
    assert 'evidence.sha256"' in source
    assert "evidence retained" in source
    assert 'write_launcher_inventory' in source
    assert 'RUN_STATUS="completed"' in source
    assert 'SCIENTIFIC_VERDICT="pending_local_compare"' in source
    assert 'printf \'  "schema_version": 1,\\n\'' in source
    assert 'status="error"' in source


def test_remote_launcher_confines_every_declared_write_directory() -> None:
    source = _source()

    assert 'require_writable_directory()' in source
    for path in (
        '"$GATE0_ROOT/results"',
        '"$GATE0_ROOT/cache/cuda"',
        '"$GATE0_ROOT/cache/pip"',
        '"$GATE0_ROOT/cache/pycache"',
        '"$GATE0_ROOT/cache/torch_extensions"',
        '"$GATE0_ROOT/cache/triton"',
        '"$GATE0_ROOT/cache/uv"',
        '"$GATE0_ROOT/cache/warp"',
        '"$GATE0_ROOT/cache/xdg"',
        '"$GATE0_ROOT/config"',
        '"$GATE0_ROOT/data"',
        '"$GATE0_ROOT/state"',
        '"$GATE0_ROOT/tmp"',
    ):
        assert path in source
    assert '[[ "$(readlink -f -- "$path")" == "$path" ]]' in source
    assert 'done < <(find "$path" -type l -print0)' in source
    assert '[[ "$target" == "$path" || "$target" == "$path/"* ]]' in source
    assert 'managed write symlink escapes its directory' in source
    assert 'findmnt -n -o TARGET -T "$path"' in source
    assert 'require_writable_directory "$RUN_ROOT/launcher/schema"' in source
    assert 'require_writable_directory "$RUN_ROOT/cases"' in source
    assert 'require_writable_directory "$case_dir"' in source


def test_remote_launcher_collects_two_formal_schema_probes_in_same_session() -> None:
    source = _source()

    assert 'readonly SCHEMA_PROBE_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_schema.py"' in source
    assert 'readonly FINALIZER_SCRIPT="$PROJECT_DIR/scripts/finalize_gate0.py"' in source
    assert 'for schema_hand in left right; do' in source
    assert 'schema_json="$RUN_ROOT/launcher/schema/$schema_hand.json"' in source
    assert '--approved-root "$GATE0_ROOT"' in source
    assert '--device cuda:0' in source
    assert '--session-id "$SESSION_ID"' in source
    assert '--source-revision "$SOURCE_REVISION"' in source
    assert '--asset-tree-sha256 "$ASSET_LF_SHA256"' in source
    assert 'validate_finalizer_schema_probe "$schema_json" "$schema_hand"' in source
    assert 'module._validate_schema_probe(' in source
    assert 'left.exitcode left.json left.stderr left.stdout' in source
    assert 'right.exitcode right.json right.stderr right.stdout' in source
    assert 'write_schema_inventory' in source
    assert '[[ "$COMPLETED_SCHEMA_PROBES" -eq 2 ]]' in source


def test_remote_launcher_enforces_per_file_and_total_evidence_limits() -> None:
    source = _source()

    assert 'ulimit -f "$MAX_ARTIFACT_BLOCKS"' in source
    assert '-size +"${MAX_ARTIFACT_BYTES}"c' in source
    assert '[[ "$bytes" -le "$MAX_RUN_BYTES" ]]' in source
    assert 'check_run_budget' in source


def test_remote_launcher_checks_completed_run_provenance_and_mapping() -> None:
    source = _source()

    for field in (
        '"manifest_sha256"',
        '"session_id"',
        '"source_revision"',
        '"asset_tree_sha256"',
        '"asset_commit"',
        '"asset_git_tree"',
        '"mapping_schema_version"',
        '"backend_joint_names"',
        '"backend_frame_names"',
        '"joint_mapping"',
        '"frame_mapping"',
    ):
        assert field in source
    assert 'record["backend_name"] != backend_names[index]' in source
    assert 'record["backend"] != "ovphysx" or record["scope"] != side' in source


def test_remote_launcher_topology_matches_the_local_finalizer_contract(
    tmp_path: Path,
) -> None:
    """The launcher's declared topology must pass the pinned local validators."""

    source = _source()
    assert 'mkdir "$RUN_ROOT/launcher" "$RUN_ROOT/cases" "$RUN_ROOT/launcher/schema"' in source
    assert 'schema_json="$RUN_ROOT/launcher/schema/$schema_hand.json"' in source
    assert 'schema_stdout="$RUN_ROOT/launcher/schema/$schema_hand.stdout"' in source
    assert 'schema_stderr="$RUN_ROOT/launcher/schema/$schema_hand.stderr"' in source
    assert 'schema_exitcode="$RUN_ROOT/launcher/schema/$schema_hand.exitcode"' in source

    fixture = _load_module(
        "gate0_finalizer_fixture_contract",
        ROOT / "tests" / "test_gate0_finalizer.py",
    )
    finalizer = fixture._load_finalizer()
    _mujoco, remote = fixture._matrix(tmp_path)
    manifest = fixture.load_manifest(fixture.MANIFEST)
    identity = finalizer._validate_remote_launcher_evidence(
        remote,
        manifest,
        expected_source_tree=fixture.SOURCE_TREE,
        manifest_file_sha256=sha256(fixture.MANIFEST.read_bytes()).hexdigest(),
        session_id=fixture.SESSION,
        source_revision=fixture.REVISION,
    )
    finalizer._validate_formal_schema_evidence(
        remote,
        manifest,
        manifest_file_sha256=sha256(fixture.MANIFEST.read_bytes()).hexdigest(),
        session_id=fixture.SESSION,
        source_revision=fixture.REVISION,
        schema_probe_script_sha256=sha256(
            (ROOT / "scripts" / "probe_ovphysx_schema.py").read_bytes()
        ).hexdigest(),
    )
    finalizer._validate_remote_case_evidence(remote, manifest, identity)


def test_remote_launcher_has_valid_bash_syntax() -> None:
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
