#!/usr/bin/env bash

set -euo pipefail
umask 077

# Server 6 launcher for the frozen, descriptive OVPhysX effective-parameter
# readback.  The launcher intentionally runs each declared instance in a fresh
# process, pins exactly one caller-selected GPU, and never overwrites evidence.
readonly GATE0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
readonly EXPECTED_HOSTNAME="example-gpu-node-6"
readonly EXPECTED_USER="exampleuser"
readonly EXPECTED_OS_ID="ubuntu"
readonly EXPECTED_OS_VERSION="22.04"
readonly EXPECTED_DRIVER_VERSION="570.158.01"
readonly EXPECTED_GPU_NAME_FRAGMENT="A800-SXM4-40GB"
readonly MAX_IDLE_MEMORY_MIB=32
readonly ASSET_URL="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
readonly ASSET_COMMIT="6eea427eb24189519f32b9f21674cd534d3f973c"
readonly ASSET_GIT_TREE="bb00a9d5527b8a76de576ce876ebece67d8ffde1"
readonly ASSET_LF_SHA256="b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
readonly ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
readonly UV_LOCK_SHA256="cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522"
readonly EXPECTED_EXPERIMENT_ID="ovphysx-both-hands-effective-readback-freeze-a-v1"
readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))
readonly MAX_RUN_BYTES=$((200 * 1024 * 1024))
readonly MAX_ARTIFACT_BYTES=$((50 * 1024 * 1024))
readonly WORKER_TIMEOUT_S=300
readonly GLOBAL_GPU_TIMEOUT_S=$((30 * 60))

PROJECT_DIR_INPUT=""
PROJECT_DIR=""
RUN_ID=""
GPU_INDEX=""
SESSION_ID=""
SOURCE_REVISION=""
SOURCE_TREE=""
ARCHIVE_SHA256=""
SNAPSHOT_SHA256=""
FREEZE_CONFIG_SHA256=""
GATE0_MANIFEST_FILE_SHA256=""
GATE0_MANIFEST_SEMANTIC_SHA256=""
SELECTED_GPU_UUID=""
RUN_ROOT=""
RUN_STATUS="error"
SCIENTIFIC_VERDICT="not_evaluated"
CURRENT_CASE=""
COMPLETED_CASES=0
LAUNCH_STARTED_EPOCH="$(date +%s)"
GPU_STARTED_EPOCH=""
ACTIVE_CASE_PID=""
ACTIVE_CASE_PGID=""

# Preserve the caller's descriptors.  After RUN_ROOT is created all launcher
# output is captured as evidence until the EXIT trap has finished.
exec 3>&1 4>&2

usage() {
  printf '%s\n' \
    "usage: run_ovphysx_effective_readback_remote.sh --project-dir PATH" \
    "       --run-id ID --gpu-index INDEX --session-id ID" \
    "       --source-revision 40_HEX_COMMIT" >&4
}

die() {
  printf 'Refusing to continue: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

refuse_symlink() {
  [[ ! -L "$1" ]] || die "managed path is a symbolic link: $1"
}

safe_identifier() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9_.-]{0,127}$ ]]
}

require_real_directory() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || die "$label directory is missing: $path"
  refuse_symlink "$path"
  [[ "$(readlink -f -- "$path")" == "$path" ]] \
    || die "$label directory is not canonical: $path"
  [[ "$path" == "$GATE0_ROOT" || "$path" == "$GATE0_ROOT/"* ]] \
    || die "$label directory escapes the approved root: $path"
  [[ "$(stat -c '%U' "$path")" == "$EXPECTED_USER" ]] \
    || die "$label directory owner changed: $path"
  [[ "$(findmnt -n -o TARGET -T "$path")" == "/data" ]] \
    || die "$label directory left /data: $path"
  [[ "$(findmnt -n -o FSTYPE -T "$path")" == "xfs" ]] \
    || die "$label directory is not on XFS: $path"
}

require_writable_directory() {
  local path="$1"
  local link target
  require_real_directory "$path" "writable"
  [[ -w "$path" ]] || die "managed write directory is not writable: $path"
  while IFS= read -r -d '' link; do
    target="$(readlink -f -- "$link")" \
      || die "managed write directory contains a dangling symbolic link: $link"
    [[ "$target" == "$path" || "$target" == "$path/"* ]] \
      || die "managed write symlink escapes its directory: $link -> $target"
  done < <(find "$path" -type l -print0)
}

terminate_owned_case() {
  local owner_pid="${1:-}"
  local owner_pgid="${2:-}"
  local observed_pgid=""
  [[ "$owner_pid" =~ ^[1-9][0-9]*$ && "$owner_pgid" == "$owner_pid" ]] || return 0
  observed_pgid="$(ps -o pgid= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ "$observed_pgid" == "$owner_pgid" ]] || return 0
  # Only the process group created for this case can be signalled.  A foreign
  # GPU PID is evidence and is never a termination target.
  kill -TERM -- "-$owner_pgid" 2>/dev/null || true
  for _termination_attempt in {1..30}; do
    kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$owner_pgid" 2>/dev/null || true
}

write_top_level_inventory() {
  local temporary oversized
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  [[ ! -e "$RUN_ROOT/launcher/evidence.sha256" ]] || return 1
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || {
    printf 'final artifact exceeds 50 MiB: %s\n' "$oversized" >&2
    return 1
  }
  temporary="$RUN_ROOT/launcher/evidence.sha256.tmp.$$"
  (
    cd "$RUN_ROOT"
    while IFS= read -r -d '' evidence_path; do
      sha256sum "$evidence_path"
    done < <(
      find . -type f \
        ! -path './launcher/evidence.sha256' \
        ! -path './launcher/evidence.sha256.tmp.*' \
        -print0 | sort -z
    )
  ) > "$temporary"
  mv -- "$temporary" "$RUN_ROOT/launcher/evidence.sha256"
}

write_final_status() {
  local exit_code="$1"
  local ended_epoch elapsed root_bytes status last_case_json
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  ended_epoch="$(date +%s)"
  elapsed=$((ended_epoch - LAUNCH_STARTED_EPOCH))
  root_bytes="$(du -sb "$GATE0_ROOT" 2>/dev/null | awk '{print $1}')"
  status="$RUN_STATUS"
  [[ "$exit_code" -eq 0 ]] || status="error"
  if [[ -n "$CURRENT_CASE" ]]; then
    last_case_json="\"$CURRENT_CASE\""
  else
    last_case_json="null"
  fi
  STATUS_DESTINATION="$RUN_ROOT/launcher/status.json.tmp.$$" \
  STATUS_FINAL="$RUN_ROOT/launcher/status.json" \
  STATUS_CODE="$exit_code" STATUS_VALUE="$status" \
  STATUS_VERDICT="$SCIENTIFIC_VERDICT" STATUS_LAST_CASE="$last_case_json" \
  STATUS_ELAPSED="$elapsed" STATUS_ROOT_BYTES="${root_bytes:-0}" \
  "$WORKER_PYTHON" -P -S - <<'PY'
import json
import os
from pathlib import Path

last_case_literal = os.environ["STATUS_LAST_CASE"]
payload = {
    "schema_version": 1,
    "experiment_id": "ovphysx-both-hands-effective-readback-freeze-a-v1",
    "status": os.environ["STATUS_VALUE"],
    "scientific_verdict": os.environ["STATUS_VERDICT"],
    "exit_code": int(os.environ["STATUS_CODE"]),
    "run_id": os.environ["RUN_ID"],
    "session_id": os.environ["SESSION_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_tree": os.environ["SOURCE_TREE"],
    "source_archive_sha256": os.environ["ARCHIVE_SHA256"],
    "snapshot_sha256": os.environ["SNAPSHOT_SHA256"],
    "freeze_a_config_sha256": os.environ["FREEZE_CONFIG_SHA256"],
    "selected_gpu_index": int(os.environ["GPU_INDEX"]),
    "selected_gpu_uuid": os.environ["SELECTED_GPU_UUID"],
    "completed_case_count": int(os.environ["COMPLETED_CASES"]),
    "expected_case_count": 4,
    "last_case_id": json.loads(last_case_literal),
    "elapsed_s": int(os.environ["STATUS_ELAPSED"]),
    "gate0_root_bytes": int(os.environ["STATUS_ROOT_BYTES"]),
    "ended_at_utc": os.environ["STATUS_ENDED_AT_UTC"],
}
destination = Path(os.environ["STATUS_DESTINATION"])
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
destination.replace(Path(os.environ["STATUS_FINAL"]))
PY
}

finish() {
  local exit_code="$?"
  trap - EXIT
  set +e
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  export RUN_ID SESSION_ID SOURCE_REVISION SOURCE_TREE ARCHIVE_SHA256 SNAPSHOT_SHA256
  export FREEZE_CONFIG_SHA256 GPU_INDEX SELECTED_GPU_UUID COMPLETED_CASES
  export STATUS_ENDED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  write_final_status "$exit_code"
  if ! write_top_level_inventory; then
    exit_code=2
    RUN_STATUS="error"
    SCIENTIFIC_VERDICT="not_evaluated"
    write_final_status "$exit_code"
    write_top_level_inventory || true
  fi
  exec 1>&3 2>&4
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'OVPhysX effective readback collected; local validation is still required; evidence retained under results/%s\n' "$RUN_ID"
  elif [[ -n "$RUN_ROOT" ]]; then
    printf 'OVPhysX effective readback stopped; evidence retained under results/%s\n' "$RUN_ID" >&2
  fi
  exit "$exit_code"
}

signal_exit() {
  local signal_name="$1"
  local exit_code="$2"
  printf 'launcher received %s while case=%s\n' "$signal_name" "${CURRENT_CASE:-none}" >&2
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"
  exit "$exit_code"
}

trap finish EXIT
trap 'signal_exit HUP 129' HUP
trap 'signal_exit INT 130' INT
trap 'signal_exit TERM 143' TERM

seen_project=false
seen_run=false
seen_gpu=false
seen_session=false
seen_revision=false
while [[ "$#" -gt 0 ]]; do
  [[ "$#" -ge 2 ]] || { usage; die "every option requires one value"; }
  case "$1" in
    --project-dir)
      [[ "$seen_project" == false ]] || die "--project-dir was supplied more than once"
      PROJECT_DIR_INPUT="$2"
      seen_project=true
      ;;
    --run-id)
      [[ "$seen_run" == false ]] || die "--run-id was supplied more than once"
      RUN_ID="$2"
      seen_run=true
      ;;
    --gpu-index)
      [[ "$seen_gpu" == false ]] || die "--gpu-index was supplied more than once"
      GPU_INDEX="$2"
      seen_gpu=true
      ;;
    --session-id)
      [[ "$seen_session" == false ]] || die "--session-id was supplied more than once"
      SESSION_ID="$2"
      seen_session=true
      ;;
    --source-revision)
      [[ "$seen_revision" == false ]] || die "--source-revision was supplied more than once"
      SOURCE_REVISION="$2"
      seen_revision=true
      ;;
    *)
      usage
      die "unsupported option: $1"
      ;;
  esac
  shift 2
done

[[ "$seen_project" == true && "$seen_run" == true && "$seen_gpu" == true \
  && "$seen_session" == true && "$seen_revision" == true ]] \
  || { usage; die "all five launcher options are required"; }
safe_identifier "$RUN_ID" || die "run ID must be a safe lowercase identifier"
safe_identifier "$SESSION_ID" || die "session ID must be a safe lowercase identifier"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || die "source revision must be a lowercase 40-character Git commit"
[[ "$GPU_INDEX" =~ ^[0-7]$ ]] || die "GPU index must be an integer from 0 through 7"

for command_name in \
  awk basename date du env find findmnt flock git hostname id mkdir mv \
  nvidia-smi ps readlink sha256sum sleep sort stat timeout; do
  require_command "$command_name"
done

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || die "expected Server 6 hostname $EXPECTED_HOSTNAME"
[[ "$(id -un)" == "$EXPECTED_USER" ]] || die "expected user $EXPECTED_USER"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "$EXPECTED_OS_ID" && "${VERSION_ID:-}" == "$EXPECTED_OS_VERSION" ]] \
  || die "expected Ubuntu 22.04"
[[ -d "$GATE0_ROOT" ]] || die "approved root does not exist"
refuse_symlink "$GATE0_ROOT"
[[ "$(readlink -f "$GATE0_ROOT")" == "$GATE0_ROOT" ]] || die "approved root is not canonical"
[[ "$(stat -c '%U' "$GATE0_ROOT")" == "$EXPECTED_USER" ]] || die "approved root owner changed"
[[ "$(findmnt -n -o TARGET -T "$GATE0_ROOT")" == "/data" ]] || die "approved root left /data"
[[ "$(findmnt -n -o FSTYPE -T "$GATE0_ROOT")" == "xfs" ]] || die "approved root is not on XFS"
[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2 && "$PROJECT_DIR_INPUT" != /mnt/ceph2/* ]] \
  || die "the Ceph mount is outside the approved boundary"

PROJECT_DIR="$(readlink -f -- "$PROJECT_DIR_INPUT")"
[[ "$PROJECT_DIR" == "$PROJECT_DIR_INPUT" ]] || die "project directory must be an absolute canonical path"
[[ "$PROJECT_DIR" == "$GATE0_ROOT/project/"* ]] || die "project directory escapes the approved project root"
[[ -d "$PROJECT_DIR" ]] || die "project directory does not exist"
refuse_symlink "$PROJECT_DIR"
[[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] || die "project snapshot contains a symbolic link"

read_marker() {
  local path="$1"
  local destination_name="$2"
  local -a lines=()
  [[ -f "$path" ]] || die "missing source marker: $(basename "$path")"
  refuse_symlink "$path"
  mapfile -t lines < "$path"
  [[ "${#lines[@]}" -eq 1 && -n "${lines[0]}" ]] \
    || die "source marker must contain exactly one non-empty line: $(basename "$path")"
  printf -v "$destination_name" '%s' "${lines[0]}"
}

MARKER_SOURCE_REVISION=""
read_marker "$PROJECT_DIR/.source-revision" MARKER_SOURCE_REVISION
read_marker "$PROJECT_DIR/.source-tree" SOURCE_TREE
read_marker "$PROJECT_DIR/.archive-sha256" ARCHIVE_SHA256
read_marker "$PROJECT_DIR/.snapshot-sha256" SNAPSHOT_SHA256
[[ "$MARKER_SOURCE_REVISION" == "$SOURCE_REVISION" ]] \
  || die "source revision does not match the deployed snapshot marker"
[[ "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ ]] || die "source tree marker is not a Git SHA-1"
[[ "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "archive marker is not a SHA-256"
[[ "$SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "snapshot marker is not a SHA-256"
[[ "$PROJECT_DIR" == "$GATE0_ROOT/project/$SOURCE_TREE" ]] \
  || die "project directory must exactly match project/source-tree"

readonly ASSET_ROOT="$GATE0_ROOT/assets"
readonly ISAACLAB_ROOT="$GATE0_ROOT/IsaacLab"
readonly DOWNLOADS_ROOT="$GATE0_ROOT/downloads"
readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"
readonly WORKER_PYTHON="$GATE0_ROOT/env/bin/python"
readonly GATE0_MANIFEST="$PROJECT_DIR/configs/parity/gate0.json"
readonly FREEZE_CONFIG="$PROJECT_DIR/configs/parity/ovphysx_effective_readback.json"
readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_effective_params.py"
require_real_directory "$GATE0_ROOT/project" "project"
require_real_directory "$PROJECT_DIR" "project snapshot"
require_real_directory "$DOWNLOADS_ROOT" "downloads"
require_real_directory "$ASSET_ROOT" "asset"
require_real_directory "$ISAACLAB_ROOT" "IsaacLab"
[[ -f "$SOURCE_ARCHIVE" && ! -L "$SOURCE_ARCHIVE" ]] \
  || die "deployed source archive is missing or is a symlink"
[[ "$(readlink -f -- "$SOURCE_ARCHIVE")" == "$SOURCE_ARCHIVE" ]] \
  || die "deployed source archive path is not canonical"
for writable_path in \
  "$GATE0_ROOT/results" \
  "$GATE0_ROOT/cache" \
  "$GATE0_ROOT/cache/cuda" \
  "$GATE0_ROOT/cache/pip" \
  "$GATE0_ROOT/cache/pycache" \
  "$GATE0_ROOT/cache/torch_extensions" \
  "$GATE0_ROOT/cache/triton" \
  "$GATE0_ROOT/cache/uv" \
  "$GATE0_ROOT/cache/warp" \
  "$GATE0_ROOT/cache/xdg" \
  "$GATE0_ROOT/config" \
  "$GATE0_ROOT/data" \
  "$GATE0_ROOT/state" \
  "$GATE0_ROOT/tmp"; do
  require_writable_directory "$writable_path"
done
[[ -x "$WORKER_PYTHON" ]] || die "Gate 0 Python is missing"
worker_python_real="$(readlink -f "$WORKER_PYTHON")"
[[ "$worker_python_real" == "$GATE0_ROOT/env/"* \
  || "$worker_python_real" == "$GATE0_ROOT/toolchain/python/"* ]] \
  || die "Gate 0 Python resolves outside the approved root"
for source_file in "$GATE0_MANIFEST" "$FREEZE_CONFIG" "$WORKER_SCRIPT"; do
  [[ -f "$source_file" ]] || die "required source file is missing: $(basename "$source_file")"
  refuse_symlink "$source_file"
done
[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_ovphysx_effective_readback_remote.sh" ]] \
  || die "launcher must execute from the selected immutable project snapshot"

RUN_ROOT="$GATE0_ROOT/results/$RUN_ID"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run directory already exists; refusing overwrite"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/launcher" "$RUN_ROOT/cases"
for run_write_path in "$RUN_ROOT" "$RUN_ROOT/launcher" "$RUN_ROOT/cases"; do
  require_writable_directory "$run_write_path"
done
exec 1>"$RUN_ROOT/launcher/stdout.txt" 2>"$RUN_ROOT/launcher/stderr.txt"
printf 'OVPhysX effective-readback launcher started at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

refuse_symlink "$GATE0_ROOT/.ovphysx-effective-readback-run.lock"
exec 9>"$GATE0_ROOT/.ovphysx-effective-readback-run.lock"
flock -n 9 || die "another OVPhysX effective-readback launcher holds the run lock"
cd "$RUN_ROOT"

[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] \
  || die "display variables must be unset for kit-less readback"
unset DISPLAY WAYLAND_DISPLAY
unset PYTHONBREAKPOINT PYTHONCASEOK PYTHONCOERCECLOCALE PYTHONDEBUG PYTHONDEVMODE
unset PYTHONDONTWRITEBYTECODE PYTHONEXECUTABLE PYTHONFAULTHANDLER PYTHONHOME
unset PYTHONINSPECT PYTHONINTMAXSTRDIGITS PYTHONMALLOC PYTHONNODEBUGRANGES
unset PYTHONPATH PYTHONOPTIMIZE PYTHONPLATLIBDIR PYTHONPROFILEIMPORTTIME
unset PYTHONSTARTUP PYTHONTRACEMALLOC PYTHONUSERBASE PYTHONWARNDEFAULTENCODING
unset PYTHONWARNINGS
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$GATE0_ROOT/cache/pycache"
export PYTHONSAFEPATH=1
export PYTHONUTF8=1
export CUDA_CACHE_PATH="$GATE0_ROOT/cache/cuda"
export PIP_CACHE_DIR="$GATE0_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$GATE0_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$GATE0_ROOT/cache/triton"
export TMPDIR="$GATE0_ROOT/tmp"
export UV_CACHE_DIR="$GATE0_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$GATE0_ROOT/env"
export WARP_CACHE_PATH="$GATE0_ROOT/cache/warp"
export XDG_CACHE_HOME="$GATE0_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$GATE0_ROOT/config"
export XDG_DATA_HOME="$GATE0_ROOT/data"
export XDG_STATE_HOME="$GATE0_ROOT/state"

check_root_budget() {
  local bytes
  bytes="$(du -sb "$GATE0_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_ROOT_BYTES" ]] || die "approved root exceeds the 30 GiB stop limit"
}

check_run_budget() {
  local bytes oversized
  bytes="$(du -sb "$RUN_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_RUN_BYTES" ]] || die "run evidence exceeds the 200 MiB stop limit"
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || die "artifact exceeds the 50 MiB stop limit: $oversized"
}

verify_snapshot() {
  local checkpoint="$1"
  local actual writable
  [[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] \
    || die "project snapshot gained a symbolic link before $checkpoint"
  writable="$(find "$PROJECT_DIR" -perm /222 -print -quit)"
  [[ -z "$writable" ]] || die "project snapshot is writable before $checkpoint: $writable"
  actual="$(PROJECT_SNAPSHOT="$PROJECT_DIR" "$WORKER_PYTHON" -P -S - <<'PY'
import hashlib
import os
from pathlib import Path
import stat

root = Path(os.environ["PROJECT_SNAPSHOT"]).resolve(strict=True)
digest = hashlib.sha256(b"waveqa-source-snapshot-v1\0")
entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
for path in entries:
    relative = path.relative_to(root).as_posix()
    if relative == ".snapshot-sha256":
        continue
    if path.is_symlink():
        raise SystemExit(f"snapshot contains a symbolic link: {relative}")
    mode = path.stat(follow_symlinks=False).st_mode
    encoded = relative.encode("utf-8")
    if stat.S_ISDIR(mode):
        kind = b"D"
        executable = 0
    elif stat.S_ISREG(mode):
        kind = b"F"
        executable = int(bool(mode & 0o111))
    else:
        raise SystemExit(f"snapshot contains a non-regular entry: {relative}")
    digest.update(kind)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(executable.to_bytes(1, "big"))
    if kind == b"F":
        size = path.stat(follow_symlinks=False).st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
print(digest.hexdigest())
PY
)"
  [[ "$actual" == "$SNAPSHOT_SHA256" ]] \
    || die "project snapshot hash changed before $checkpoint"
  printf '%s checkpoint=%s snapshot_sha256=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checkpoint" "$actual" \
    >> "$RUN_ROOT/launcher/snapshot-verifications.txt"
}

verify_repository() {
  local path="$1"
  local expected_url="$2"
  local allowed_untracked="${3:-}"
  local status_line
  [[ -d "$path/.git" ]] || die "managed checkout is not a Git repository: $path"
  [[ "$(git -C "$path" remote get-url origin)" == "$expected_url" ]] \
    || die "unexpected origin URL in managed checkout"
  while IFS= read -r status_line; do
    [[ -z "$status_line" ]] && continue
    if [[ -n "$allowed_untracked" && "$status_line" == "?? $allowed_untracked" ]]; then
      continue
    fi
    die "managed checkout is dirty: $status_line"
  done < <(git -C "$path" status --porcelain --untracked-files=all)
}

verify_source_archive() {
  local actual_archive_sha256
  actual_archive_sha256="$(sha256sum -- "$SOURCE_ARCHIVE" | awk '{print $1}')"
  [[ "$actual_archive_sha256" == "$ARCHIVE_SHA256" ]] \
    || die "deployed source archive SHA-256 no longer matches its snapshot marker"
}

check_root_budget
verify_source_archive
verify_repository "$ASSET_ROOT" "$ASSET_URL"
[[ "$(git -C "$ASSET_ROOT" rev-parse HEAD)" == "$ASSET_COMMIT" ]] \
  || die "Sharpa asset commit mismatch"
[[ "$(git -C "$ASSET_ROOT" rev-parse 'HEAD:wave_01')" == "$ASSET_GIT_TREE" ]] \
  || die "Sharpa asset Git tree mismatch"
verify_repository "$ISAACLAB_ROOT" "$ISAACLAB_URL" "uv.lock"
[[ "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" == "$ISAACLAB_COMMIT" ]] \
  || die "Isaac Lab commit mismatch"
printf '%s  %s\n' "$UV_LOCK_SHA256" "$ISAACLAB_ROOT/uv.lock" \
  | sha256sum --check --status || die "Isaac Lab lock SHA-256 mismatch"
verify_snapshot "preflight"

actual_asset_lf_sha256="$(ASSET_TREE="$ASSET_ROOT/wave_01" "$WORKER_PYTHON" -P -S - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["ASSET_TREE"])
digest = hashlib.sha256()
files = sorted(
    (path for path in root.rglob("*") if path.is_file() and ".git" not in path.relative_to(root).parts),
    key=lambda path: path.relative_to(root).as_posix(),
)
for path in files:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
print(digest.hexdigest())
PY
)"
[[ "$actual_asset_lf_sha256" == "$ASSET_LF_SHA256" ]] \
  || die "canonical LF asset-tree SHA-256 mismatch"

environment_summary="$("$WORKER_PYTHON" -P - <<'PY'
from importlib import metadata
import sys

expected = {
    "isaaclab": "6.1.14",
    "isaaclab-ovphysx": "3.0.2",
    "ovphysx": "0.4.13",
    "torch": "2.10.0+cu128",
    "usd-core": "25.11",
}
if sys.version_info[:3] != (3, 12, 14):
    raise SystemExit(f"unexpected Python version: {sys.version.split()[0]}")
for package, version in expected.items():
    if metadata.version(package) != version:
        raise SystemExit(f"unexpected {package} version: {metadata.version(package)}")
try:
    metadata.version("usd-exchange")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("usd-exchange is present after the runtime overlay")

from isaaclab.utils.version import has_kit
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: F401
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: F401
import ovphysx  # noqa: F401
import torch

if has_kit():
    raise SystemExit("Isaac Lab reports a Kit runtime")
forbidden = [
    name
    for name in sys.modules
    if name == "isaacsim"
    or name.startswith("isaacsim.")
    or name == "omni.kit"
    or name.startswith("omni.kit.")
    or name == "omni.renderer"
    or name.startswith("omni.renderer.")
]
if forbidden:
    raise SystemExit(f"forbidden Kit/render modules loaded: {forbidden[:10]}")
print("python=3.12.14")
for package, version in expected.items():
    print(f"{package}={version}")
print(f"torch_cuda_build={torch.version.cuda}")
print("kit=false")
print("renderer=false")
print("camera=false")
PY
)"

matrix_file="$RUN_ROOT/launcher/cases.txt.tmp"
PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
  "$FREEZE_CONFIG" "$GATE0_MANIFEST" <<'PY' > "$matrix_file"
from hashlib import sha256
import json
from pathlib import Path
import sys

from wave_asset_qa.parity.scenarios import load_manifest, manifest_sha256

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_constant(token):
    raise ValueError(f"non-finite JSON number: {token}")

freeze_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
freeze = json.loads(
    freeze_path.read_text(encoding="utf-8"),
    object_pairs_hook=no_duplicates,
    parse_constant=reject_constant,
)
if freeze.get("schema_version") != 1:
    raise SystemExit("Freeze A schema version mismatch")
if freeze.get("experiment_id") != "ovphysx-both-hands-effective-readback-freeze-a-v1":
    raise SystemExit("Freeze A experiment ID mismatch")
if freeze.get("classification") != "descriptive_effective_parameter_readback_not_formal_gate0":
    raise SystemExit("Freeze A classification mismatch")
if freeze.get("plan_stage") != "freeze_a_before_any_new_ovphysx_parameter_value_is_observed":
    raise SystemExit("Freeze A plan stage mismatch")
scope = freeze.get("scope")
if not isinstance(scope, dict) or scope.get("backend") != "ovphysx":
    raise SystemExit("Freeze A must target OVPhysX")
if scope.get("hands") != ["left", "right"] or scope.get("readback_only") is not True:
    raise SystemExit("Freeze A must cover both hands in readback-only mode")
if scope.get("trajectory_in_scope") is not False:
    raise SystemExit("Freeze A unexpectedly authorizes a trajectory")
execution = freeze.get("execution_contract")
if not isinstance(execution, dict):
    raise SystemExit("Freeze A execution contract is missing")
required_execution = {
    "fresh_instances_per_hand": 2,
    "total_instance_count": 4,
    "fresh_process_per_instance": True,
    "reuse_of_articulation_or_physics_scene_forbidden": True,
    "user_or_experimental_trace_dynamics_advance_allowed": False,
    "required_user_trace_simulation_step_call_count": 0,
    "required_trajectory_sample_count": 0,
    "required_trace_time_s": 0.0,
    "experimental_control_command_allowed": False,
    "experimental_target_write_allowed": False,
    "renderer_allowed": False,
    "camera_allowed": False,
}
for key, expected in required_execution.items():
    if execution.get(key) != expected:
        raise SystemExit(f"Freeze A execution field changed: {key}")

expected_instances = [
    ("left", 1, "ovphysx.left.effective_readback.r01"),
    ("left", 2, "ovphysx.left.effective_readback.r02"),
    ("right", 1, "ovphysx.right.effective_readback.r01"),
    ("right", 2, "ovphysx.right.effective_readback.r02"),
]
observed = []
for item in freeze.get("expected_instances", []):
    if not isinstance(item, dict):
        raise SystemExit("Freeze A instance must be an object")
    if isinstance(item.get("instance_index"), bool) or not isinstance(item.get("instance_index"), int):
        raise SystemExit("Freeze A instance index must be an integer")
    observed.append((item.get("side"), item.get("instance_index"), item.get("case_id")))
if observed != expected_instances:
    raise SystemExit("Freeze A expected instance matrix changed")

freeze_context = freeze.get("frozen_context")
if not isinstance(freeze_context, dict):
    raise SystemExit("Freeze A frozen context is missing")
manifest_file_sha = sha256(manifest_path.read_bytes()).hexdigest()
if freeze_context.get("gate0_manifest_file_sha256") != manifest_file_sha:
    raise SystemExit("Freeze A Gate 0 manifest file hash mismatch")
manifest_semantic_sha = manifest_sha256(load_manifest(manifest_path))
if freeze_context.get("gate0_manifest_semantic_sha256") != manifest_semantic_sha:
    raise SystemExit("Freeze A Gate 0 manifest semantic hash mismatch")
expected_context = {
    "asset_commit": "6eea427eb24189519f32b9f21674cd534d3f973c",
    "asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
    "canonical_lf_asset_tree_sha256": "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad",
}
for key, expected in expected_context.items():
    if freeze_context.get(key) != expected:
        raise SystemExit(f"Freeze A frozen context changed: {key}")

print("freeze_config_sha256=" + sha256(freeze_path.read_bytes()).hexdigest())
print("gate0_manifest_file_sha256=" + manifest_file_sha)
print("gate0_manifest_semantic_sha256=" + manifest_semantic_sha)
for side, instance_index, case_id in expected_instances:
    print(f"{case_id}|{side}|{instance_index}")
PY
mv -- "$matrix_file" "$RUN_ROOT/launcher/cases.txt"
mapfile -t matrix_lines < "$RUN_ROOT/launcher/cases.txt"
[[ "${#matrix_lines[@]}" -eq 7 ]] \
  || die "readback matrix must contain three hashes and four instances"
FREEZE_CONFIG_SHA256="${matrix_lines[0]#freeze_config_sha256=}"
GATE0_MANIFEST_FILE_SHA256="${matrix_lines[1]#gate0_manifest_file_sha256=}"
GATE0_MANIFEST_SEMANTIC_SHA256="${matrix_lines[2]#gate0_manifest_semantic_sha256=}"
[[ "$FREEZE_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "Freeze A hash is invalid"
[[ "$GATE0_MANIFEST_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "manifest file hash is invalid"
[[ "$GATE0_MANIFEST_SEMANTIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "manifest semantic hash is invalid"
CASE_RECORDS=("${matrix_lines[@]:3}")
for case_record in "${CASE_RECORDS[@]}"; do
  IFS='|' read -r case_id side instance_index extra <<< "$case_record"
  [[ -z "${extra:-}" ]] || die "readback matrix record has unexpected fields"
  safe_identifier "$case_id" || die "unsafe readback case ID: $case_id"
  [[ "$case_id" == "ovphysx.$side.effective_readback.r0$instance_index" ]] \
    || die "readback case identity is inconsistent"
  [[ "$side" == "left" || "$side" == "right" ]] || die "invalid readback hand"
  [[ "$instance_index" == "1" || "$instance_index" == "2" ]] \
    || die "invalid readback instance index"
done

write_case_inventory() {
  local directory="$1"
  local inventory="$directory/evidence.sha256"
  local temporary="$inventory.tmp.$$"
  [[ ! -e "$inventory" && ! -L "$inventory" ]] \
    || die "case evidence inventory already exists: $inventory"
  check_run_budget
  (
    cd "$directory"
    while IFS= read -r -d '' evidence_path; do
      sha256sum "$evidence_path"
    done < <(
      find . -type f ! -name evidence.sha256 ! -name 'evidence.sha256.tmp.*' -print0 | sort -z
    )
  ) > "$temporary"
  mv -- "$temporary" "$inventory"
}

LAST_GPU_UUID=""
LAST_GPU_NAME=""
LAST_GPU_DRIVER=""
LAST_GPU_MEMORY=""
LAST_GPU_UTILIZATION=""
capture_gpu_state() {
  local destination="$1"
  local expected_uuid="${2:-}"
  local temporary gpu_line processes actual_index
  temporary="$destination.tmp.$$"
  gpu_line="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"
  [[ "$gpu_line" != *$'\n'* ]] || die "GPU query returned more than one device"
  actual_index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$gpu_line")"
  LAST_GPU_UUID="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
  LAST_GPU_NAME="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
  LAST_GPU_DRIVER="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
  LAST_GPU_MEMORY="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}' <<< "$gpu_line")"
  LAST_GPU_UTILIZATION="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}' <<< "$gpu_line")"
  processes="$(nvidia-smi -i "$GPU_INDEX" \
    --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader,nounits)"
  {
    printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu_index=%s\n' "$actual_index"
    printf 'gpu_uuid=%s\n' "$LAST_GPU_UUID"
    printf 'gpu_name=%s\n' "$LAST_GPU_NAME"
    printf 'driver_version=%s\n' "$LAST_GPU_DRIVER"
    printf 'memory_used_mib=%s\n' "$LAST_GPU_MEMORY"
    printf 'utilization_percent=%s\n' "$LAST_GPU_UTILIZATION"
    printf 'compute_processes_begin\n%s\ncompute_processes_end\n' "$processes"
  } > "$temporary"
  mv -- "$temporary" "$destination"

  [[ "$actual_index" == "$GPU_INDEX" ]] || return 1
  [[ "$LAST_GPU_UUID" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || return 1
  [[ -z "$expected_uuid" || "$LAST_GPU_UUID" == "$expected_uuid" ]] || return 1
  [[ "$LAST_GPU_NAME" == *"$EXPECTED_GPU_NAME_FRAGMENT"* ]] || return 1
  [[ "$LAST_GPU_DRIVER" == "$EXPECTED_DRIVER_VERSION" ]] || return 1
  [[ "$LAST_GPU_MEMORY" =~ ^[0-9]+$ ]] || return 1
  (( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB )) || return 1
  [[ "$LAST_GPU_UTILIZATION" == "0" ]] || return 1
  [[ -z "$processes" ]] || return 1
}

trim_csv_field() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

case_process_running() {
  local owner_pid="$1"
  local state
  kill -0 "$owner_pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ -n "$state" && "$state" != Z* ]]
}

monitor_owned_case() {
  local owner_pid="$1"
  local owner_pgid="$2"
  local destination="$3"
  local foreign_marker="$4"
  local gpu_line processes observed_uuid observed_name observed_driver
  local row process_uuid process_pid process_memory process_pgid foreign_reason
  [[ ! -e "$destination" && ! -L "$destination" ]] || return 1
  : > "$destination"
  while case_process_running "$owner_pid"; do
    gpu_line="$(nvidia-smi -i "$GPU_INDEX" \
      --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
      --format=csv,noheader,nounits)"
    processes="$(nvidia-smi -i "$GPU_INDEX" \
      --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
      --format=csv,noheader,nounits)"
    observed_uuid="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
    observed_name="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
    observed_driver="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
    {
      printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'gpu=%s\n' "$gpu_line"
      printf 'owned_case_pid=%s\nowned_case_pgid=%s\n' "$owner_pid" "$owner_pgid"
      printf 'compute_processes_begin\n%s\ncompute_processes_end\n' "$processes"
    } >> "$destination"

    foreign_reason=""
    if [[ "$observed_uuid" != "$SELECTED_GPU_UUID" \
      || "$observed_name" != *"$EXPECTED_GPU_NAME_FRAGMENT"* \
      || "$observed_driver" != "$EXPECTED_DRIVER_VERSION" ]]; then
      foreign_reason="selected GPU identity changed during the case"
    fi
    while IFS= read -r row; do
      [[ -n "$row" ]] || continue
      IFS=',' read -r process_uuid process_pid process_memory <<< "$row"
      process_uuid="$(trim_csv_field "$process_uuid")"
      process_pid="$(trim_csv_field "$process_pid")"
      process_memory="$(trim_csv_field "$process_memory")"
      if [[ "$process_uuid" != "$SELECTED_GPU_UUID" || ! "$process_pid" =~ ^[1-9][0-9]*$ ]]; then
        foreign_reason="unclassifiable GPU compute process: $row"
        break
      fi
      if [[ ! -e "/proc/$process_pid" ]]; then
        continue
      fi
      process_pgid="$(ps -o pgid= -p "$process_pid" 2>/dev/null | awk '{$1=$1; print}')"
      if [[ -z "$process_pgid" || "$process_pgid" != "$owner_pgid" ]]; then
        foreign_reason="foreign GPU process pid=$process_pid pgid=${process_pgid:-unknown} memory_mib=$process_memory"
        break
      fi
    done <<< "$processes"
    if [[ -n "$foreign_reason" ]]; then
      printf '%s\n' "$foreign_reason" > "$foreign_marker"
      terminate_owned_case "$owner_pid" "$owner_pgid"
      return 1
    fi
    sleep 1
  done
  return 0
}

validate_case_payload() {
  local payload="$1"
  local expected_case="$2"
  local expected_side="$3"
  local expected_instance="$4"
  "$WORKER_PYTHON" -P -S - \
    "$payload" "$expected_case" "$expected_side" "$expected_instance" \
    "$SESSION_ID" "$SOURCE_REVISION" "$SOURCE_TREE" "$ARCHIVE_SHA256" \
    "$FREEZE_CONFIG_SHA256" "$GATE0_MANIFEST_FILE_SHA256" \
    "$GATE0_MANIFEST_SEMANTIC_SHA256" "$ASSET_LF_SHA256" \
    "$PROBE_SOURCE_SHA256" "$GATE0_MANIFEST" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_constant(token):
    raise ValueError(f"non-finite JSON number: {token}")

(
    payload_path,
    expected_case,
    expected_side,
    expected_instance,
    session_id,
    source_revision,
    source_tree,
    source_archive_sha,
    freeze_sha,
    manifest_file_sha,
    manifest_semantic_sha,
    asset_tree_sha,
    probe_source_sha,
    gate0_manifest_path,
) = sys.argv[1:]
payload = json.loads(
    Path(payload_path).read_text(encoding="utf-8"),
    object_pairs_hook=no_duplicates,
    parse_constant=reject_constant,
)
if not isinstance(payload, dict):
    raise SystemExit("worker payload must be a JSON object")
required_top_level = {
    "schema_version",
    "experiment_id",
    "case_id",
    "side",
    "instance_index",
    "fresh_instance_id",
    "fresh_process_id",
    "readback_only_after_initialization",
    "initialization_state_write_performed",
    "experimental_trace_dynamics_advance_performed",
    "experimental_control_command_applied",
    "experimental_target_write_performed",
    "user_trace_simulation_step_call_count",
    "trajectory_sample_count",
    "trace_time_s",
    "gpu_warmup_outside_trace",
    "initialization_internal_advance_exposure",
    "provenance",
    "solver_runtime",
    "dof_records",
    "canonical_vector_sha256",
}
if set(payload) != required_top_level:
    raise SystemExit("worker payload top-level fields do not match Freeze A")
identity = {
    "schema_version": 1,
    "case_id": expected_case,
    "side": expected_side,
    "instance_index": int(expected_instance),
    "readback_only_after_initialization": True,
    "initialization_state_write_performed": True,
    "experimental_trace_dynamics_advance_performed": False,
    "experimental_control_command_applied": False,
    "experimental_target_write_performed": False,
    "user_trace_simulation_step_call_count": 0,
    "trajectory_sample_count": 0,
    "trace_time_s": 0.0,
}
for key, expected in identity.items():
    if payload.get(key) != expected:
        raise SystemExit(f"worker payload identity/scope mismatch: {key}")
if isinstance(payload.get("instance_index"), bool):
    raise SystemExit("worker instance index must be an integer")
if payload.get("experiment_id") != "ovphysx-both-hands-effective-readback-freeze-a-v1":
    raise SystemExit("worker payload experiment ID mismatch")
fresh_process_id = payload.get("fresh_process_id")
fresh_instance_id = payload.get("fresh_instance_id")
if not isinstance(fresh_process_id, str) or re.fullmatch(r"[0-9a-f]{64}", fresh_process_id) is None:
    raise SystemExit("worker fresh process ID must be a lowercase SHA-256")
if not isinstance(fresh_instance_id, str) or not fresh_instance_id or "|" in fresh_instance_id:
    raise SystemExit("worker fresh instance ID is invalid")
provenance = payload.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit("worker payload provenance is missing")
expected_provenance = {
    "session_id": session_id,
    "source_revision": source_revision,
    "source_tree": source_tree,
    "source_archive_sha256": source_archive_sha,
    "freeze_a_config_sha256": freeze_sha,
    "gate0_manifest_file_sha256": manifest_file_sha,
    "gate0_manifest_semantic_sha256": manifest_semantic_sha,
    "canonical_lf_asset_tree_sha256": asset_tree_sha,
    "probe_source_sha256": probe_source_sha,
    "asset_commit": "6eea427eb24189519f32b9f21674cd534d3f973c",
    "asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
}
for key, expected in expected_provenance.items():
    if provenance.get(key) != expected:
        raise SystemExit(f"worker payload provenance mismatch: {key}")
dof_records = payload.get("dof_records")
if not isinstance(dof_records, list) or len(dof_records) != 22:
    raise SystemExit("worker payload must contain exactly 22 DOF records")
manifest = json.loads(
    Path(gate0_manifest_path).read_text(encoding="utf-8"),
    object_pairs_hook=no_duplicates,
    parse_constant=reject_constant,
)
hand_specs = [item for item in manifest.get("hands", []) if item.get("side") == expected_side]
if len(hand_specs) != 1:
    raise SystemExit("Gate 0 manifest hand lookup is ambiguous")
expected_joint_names = hand_specs[0].get("joint_names")
if not isinstance(expected_joint_names, list) or len(expected_joint_names) != 22:
    raise SystemExit("Gate 0 manifest canonical joint list is invalid")
canonical_names = []
backend_indices = []
for record in dof_records:
    if not isinstance(record, dict):
        raise SystemExit("worker DOF record must be an object")
    if not {"mapping", "friction", "armature", "control_drive"}.issubset(record):
        raise SystemExit("worker DOF record lacks a required field family")
    mapping = record.get("mapping")
    if not isinstance(mapping, dict):
        raise SystemExit("worker DOF mapping is missing")
    if mapping.get("side") != expected_side:
        raise SystemExit("worker DOF mapping hand mismatch")
    canonical_names.append(mapping.get("canonical_joint_name"))
    backend_indices.append(mapping.get("backend_joint_index"))
if any(not isinstance(name, str) or not name for name in canonical_names):
    raise SystemExit("worker canonical joint names are invalid")
if canonical_names != expected_joint_names or len(set(canonical_names)) != 22:
    raise SystemExit("worker canonical joint coverage/order does not match Gate 0")
if any(isinstance(index, bool) or not isinstance(index, int) for index in backend_indices):
    raise SystemExit("worker backend joint indices are invalid")
if len(set(backend_indices)) != 22:
    raise SystemExit("worker backend joint indices are not unique")
canonical_hashes = payload.get("canonical_vector_sha256")
required_hashes = {
    "friction_runtime_effective",
    "armature_runtime_effective",
    "kp_runtime_effective",
    "kd_runtime_effective",
    "backend_drive_stiffness_runtime_effective",
    "backend_drive_damping_runtime_effective",
    "passive_joint_damping_runtime_effective",
    "gear_or_transmission_runtime_effective",
    "effort_limit_runtime_effective",
    "velocity_limit_runtime_effective",
}
if not isinstance(canonical_hashes, dict) or set(canonical_hashes) != required_hashes:
    raise SystemExit("worker canonical vector hash coverage is invalid")
if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in canonical_hashes.values()):
    raise SystemExit("worker canonical vector hash is invalid")

def visit(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise SystemExit("worker payload contains a non-finite numeric value")
    if isinstance(value, dict):
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)

visit(payload)
print(fresh_process_id + "|" + fresh_instance_id)
PY
}

validate_worker_stdout() {
  local stdout_path="$1"
  local expected_case="$2"
  local expected_output="$3"
  "$WORKER_PYTHON" -P -S - "$stdout_path" "$expected_case" "$expected_output" <<'PY'
import json
from pathlib import Path
import sys

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_constant(token):
    raise ValueError(f"non-finite JSON number: {token}")

path, expected_case, expected_output = sys.argv[1:]
lines = Path(path).read_text(encoding="utf-8").splitlines()
json_values = []
for line in lines:
    if not line.strip():
        continue
    try:
        value = json.loads(
            line,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError:
        continue
    except ValueError as exc:
        raise SystemExit(f"worker stdout contains invalid JSON: {exc}") from exc
    json_values.append(value)
if len(json_values) != 1:
    raise SystemExit("worker stdout must contain exactly one valid JSON value")
payload = json_values[0]
expected = {"status": "collected", "case_id": expected_case, "output": expected_output}
if payload != expected:
    raise SystemExit("worker stdout summary mismatch")
PY
}

if ! capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt"; then
  die "selected GPU is not the pinned, completely idle Server 6 device"
fi
SELECTED_GPU_UUID="$LAST_GPU_UUID"

PROVENANCE_TEMP="$RUN_ROOT/launcher/provenance.json.tmp.$$"
export PROJECT_DIR RUN_ID SESSION_ID SOURCE_REVISION SOURCE_TREE ARCHIVE_SHA256 SNAPSHOT_SHA256
export FREEZE_CONFIG_SHA256 GATE0_MANIFEST_FILE_SHA256 GATE0_MANIFEST_SEMANTIC_SHA256
export GPU_INDEX SELECTED_GPU_UUID LAST_GPU_NAME LAST_GPU_DRIVER
export PROVENANCE_TEMP
export LAUNCHER_SHA256="$(sha256sum -- "$PROJECT_DIR/scripts/run_ovphysx_effective_readback_remote.sh" | awk '{print $1}')"
export PROBE_SOURCE_SHA256="$(sha256sum -- "$WORKER_SCRIPT" | awk '{print $1}')"
export ENVIRONMENT_SUMMARY="$environment_summary"
"$WORKER_PYTHON" -P -S - <<'PY'
import json
import os
from pathlib import Path

environment = {}
for line in os.environ["ENVIRONMENT_SUMMARY"].splitlines():
    key, value = line.split("=", 1)
    environment[key] = value
payload = {
    "schema_version": 1,
    "experiment_id": "ovphysx-both-hands-effective-readback-freeze-a-v1",
    "run_id": os.environ["RUN_ID"],
    "session_id": os.environ["SESSION_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_tree": os.environ["SOURCE_TREE"],
    "source_archive_sha256": os.environ["ARCHIVE_SHA256"],
    "snapshot_sha256": os.environ["SNAPSHOT_SHA256"],
    "project_path": "project/" + os.environ["SOURCE_TREE"],
    "launcher_sha256": os.environ["LAUNCHER_SHA256"],
    "probe_source_sha256": os.environ["PROBE_SOURCE_SHA256"],
    "freeze_a_config_sha256": os.environ["FREEZE_CONFIG_SHA256"],
    "gate0_manifest_file_sha256": os.environ["GATE0_MANIFEST_FILE_SHA256"],
    "gate0_manifest_semantic_sha256": os.environ["GATE0_MANIFEST_SEMANTIC_SHA256"],
    "asset_commit": "6eea427eb24189519f32b9f21674cd534d3f973c",
    "asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
    "canonical_lf_asset_tree_sha256": "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad",
    "isaaclab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
    "selected_gpu_index": int(os.environ["GPU_INDEX"]),
    "selected_gpu_uuid": os.environ["SELECTED_GPU_UUID"],
    "selected_gpu_name": os.environ["LAST_GPU_NAME"],
    "driver_version": os.environ["LAST_GPU_DRIVER"],
    "expected_case_ids": [
        "ovphysx.left.effective_readback.r01",
        "ovphysx.left.effective_readback.r02",
        "ovphysx.right.effective_readback.r01",
        "ovphysx.right.effective_readback.r02",
    ],
    "constraints": {"headless": True, "kit": False, "renderer": False, "camera": False},
    "environment": environment,
    "recorded_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROVENANCE_TEMP"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY
mv -- "$PROVENANCE_TEMP" "$RUN_ROOT/launcher/provenance.json"

GPU_STARTED_EPOCH="$(date +%s)"
declare -a FRESH_PROCESS_IDS=()
declare -a FRESH_INSTANCE_IDS=()
: > "$RUN_ROOT/launcher/fresh-identities.txt"
for case_record in "${CASE_RECORDS[@]}"; do
  IFS='|' read -r case_id side instance_index <<< "$case_record"
  CURRENT_CASE="$case_id"
  verify_snapshot "before-$case_id"
  now_epoch="$(date +%s)"
  gpu_elapsed=$((now_epoch - GPU_STARTED_EPOCH))
    remaining=$((GLOBAL_GPU_TIMEOUT_S - gpu_elapsed))
    [[ "$remaining" -gt 0 ]] || die "global 30-minute GPU budget is exhausted"
  envelope="$WORKER_TIMEOUT_S"
  if [[ "$remaining" -lt "$envelope" ]]; then
    envelope="$remaining"
  fi

  case_dir="$RUN_ROOT/cases/$case_id"
  require_writable_directory "$RUN_ROOT/cases"
  [[ ! -e "$case_dir" && ! -L "$case_dir" ]] \
    || die "case evidence directory already exists"
  mkdir "$case_dir"
  require_writable_directory "$case_dir"
  if ! capture_gpu_state "$case_dir/preflight.txt" "$SELECTED_GPU_UUID"; then
    die "selected GPU changed or is no longer completely idle before $case_id"
  fi

  printf 'starting %s at %s\n' "$case_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case_pid_file="$case_dir/owned-process-group.txt"
  payload_path="$case_dir/$case_id.json"
  case_command=(
    timeout --foreground --signal=TERM --kill-after=30s "${envelope}s"
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    PYTHONPATH="$PROJECT_DIR/src"
    "$WORKER_PYTHON" -P "$WORKER_SCRIPT"
    --approved-root "$GATE0_ROOT"
    --asset-root "$ASSET_ROOT"
    --gate0-manifest "$GATE0_MANIFEST"
    --freeze-config "$FREEZE_CONFIG"
    --case-id "$case_id"
    --side "$side"
    --instance-index "$instance_index"
    --output "$payload_path"
    --device cuda:0
    --session-id "$SESSION_ID"
    --source-revision "$SOURCE_REVISION"
    --source-tree "$SOURCE_TREE"
    --source-archive-sha256 "$ARCHIVE_SHA256"
    --asset-tree-sha256 "$ASSET_LF_SHA256"
  )
  (
    # A process-level file-size ulimit also caps OVPhysX files below TMPDIR
    # (including scene.usda and Warp PCH files). Retained evidence is bounded
    # separately by check_run_budget and the per-artifact scan.
    exec "$WORKER_PYTHON" -P - "$case_pid_file" "${case_command[@]}"
  ) > "$case_dir/worker.stdout.txt" 2> "$case_dir/worker.stderr.txt" <<'PY' &
import os
from pathlib import Path
import sys

pid_path = Path(sys.argv[1])
command = sys.argv[2:]
if not command:
    raise SystemExit("owned readback wrapper received no command")
os.setsid()
descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execvpe(command[0], command, os.environ.copy())
PY
  case_owner_pid="$!"
  for _pid_wait_attempt in {1..50}; do
    [[ -s "$case_pid_file" ]] && break
    case_process_running "$case_owner_pid" || break
    sleep 0.1
  done
  if [[ ! -s "$case_pid_file" ]]; then
    set +e
    wait "$case_owner_pid"
    worker_exit_code="$?"
    set -e
    printf '%s\n' "$worker_exit_code" > "$case_dir/worker.exit-code.txt"
    die "owned process group was not established for $case_id"
  fi
  IFS= read -r case_pgid < "$case_pid_file"
  observed_case_pgid="$(ps -o pgid= -p "$case_owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ ! "$case_pgid" =~ ^[1-9][0-9]*$ \
    || "$case_pgid" != "$case_owner_pid" \
    || "$observed_case_pgid" != "$case_pgid" ]]; then
    kill -TERM "$case_owner_pid" 2>/dev/null || true
    set +e
    wait "$case_owner_pid"
    worker_exit_code="$?"
    set -e
    printf '%s\n' "$worker_exit_code" > "$case_dir/worker.exit-code.txt"
    die "case process-group identity is unsafe for $case_id"
  fi
  ACTIVE_CASE_PID="$case_owner_pid"
  ACTIVE_CASE_PGID="$case_pgid"
  foreign_gpu_process=false
  if ! monitor_owned_case \
    "$case_owner_pid" "$case_pgid" \
    "$case_dir/gpu-monitor.txt" "$case_dir/foreign-gpu-process.txt"; then
    foreign_gpu_process=true
  fi
  set +e
  wait "$case_owner_pid"
  worker_exit_code="$?"
  set -e
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  printf '%s\n' "$worker_exit_code" > "$case_dir/worker.exit-code.txt"

  if [[ "$foreign_gpu_process" == true ]]; then
    capture_gpu_state "$case_dir/postflight-foreign.txt" "$SELECTED_GPU_UUID" || true
    IFS= read -r foreign_reason < "$case_dir/foreign-gpu-process.txt" \
      || foreign_reason="foreign GPU process detected"
    die "$foreign_reason; only the owned case process group was terminated"
  fi

  postflight_ok=false
  for postflight_attempt in 1 2 3; do
    postflight_path="$(printf '%s/postflight-%02d.txt' "$case_dir" "$postflight_attempt")"
    if capture_gpu_state "$postflight_path" "$SELECTED_GPU_UUID"; then
      postflight_ok=true
      break
    fi
    [[ "$postflight_attempt" -eq 3 ]] || sleep 5
  done
  [[ "$postflight_ok" == true ]] || die "selected GPU did not return to idle after $case_id"
  # The pinned OVPhysX teardown can mask Python's SystemExit status. Treat the
  # strict structured summary as an independent outcome signal while retaining
  # the raw operating-system exit code unchanged as evidence.
  validate_worker_stdout "$case_dir/worker.stdout.txt" "$case_id" "$case_id.json"
  [[ "$worker_exit_code" -eq 0 ]] || die "readback worker returned $worker_exit_code for $case_id"
  [[ -f "$payload_path" && ! -L "$payload_path" ]] \
    || die "readback worker did not produce the expected case payload"
  identity_record="$(validate_case_payload "$payload_path" "$case_id" "$side" "$instance_index")"
  IFS='|' read -r fresh_process_id fresh_instance_id extra <<< "$identity_record"
  [[ -n "$fresh_process_id" && -n "$fresh_instance_id" && -z "${extra:-}" ]] \
    || die "worker fresh identity record is invalid"
  for seen_id in "${FRESH_PROCESS_IDS[@]}"; do
    [[ "$seen_id" != "$fresh_process_id" ]] || die "worker reused a fresh process identity"
  done
  for seen_id in "${FRESH_INSTANCE_IDS[@]}"; do
    [[ "$seen_id" != "$fresh_instance_id" ]] || die "worker reused a fresh instance identity"
  done
  FRESH_PROCESS_IDS+=("$fresh_process_id")
  FRESH_INSTANCE_IDS+=("$fresh_instance_id")
  printf '%s|%s|%s\n' "$case_id" "$fresh_process_id" "$fresh_instance_id" \
    >> "$RUN_ROOT/launcher/fresh-identities.txt"
  check_run_budget
  write_case_inventory "$case_dir"
  COMPLETED_CASES=$((COMPLETED_CASES + 1))
  check_root_budget
done

[[ "$COMPLETED_CASES" -eq 4 ]] || die "readback matrix did not complete all four instances"
[[ "${#FRESH_PROCESS_IDS[@]}" -eq 4 && "${#FRESH_INSTANCE_IDS[@]}" -eq 4 ]] \
  || die "readback matrix did not produce four unique fresh identities"
verify_snapshot "post-matrix"
check_run_budget
RUN_STATUS="collected"
SCIENTIFIC_VERDICT="pending_local_validation"
printf 'collected all four readback instances; no causal or formal Gate 0 conclusion was evaluated at %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
