#!/usr/bin/env bash

set -euo pipefail
umask 077

# Server 6 Waveform T1 matrix launcher.  This program is intentionally fail-closed:
# it runs one OVPhysX case per fresh process, never changes the selected GPU,
# and never removes or overwrites evidence from an earlier attempt.
readonly WAVEFORM_T1_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
readonly EXPECTED_HOSTNAME="example-gpu-node-6"
readonly EXPECTED_USER="exampleuser"
readonly EXPECTED_OS_ID="ubuntu"
readonly EXPECTED_OS_VERSION="22.04"
readonly EXPECTED_DRIVER_VERSION="570.158.01"
readonly EXPECTED_GPU_NAME_FRAGMENT="A800-SXM4-40GB"
readonly EXPECTED_MANIFEST_ID="wavesimparity-waveform-t1"
readonly EXPECTED_OVPHYSX_CASE_COUNT=16
readonly MAX_IDLE_MEMORY_MIB=32
readonly ASSET_URL="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
readonly ASSET_COMMIT="6eea427eb24189519f32b9f21674cd534d3f973c"
readonly ASSET_GIT_TREE="bb00a9d5527b8a76de576ce876ebece67d8ffde1"
readonly ASSET_LF_SHA256="b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
readonly ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
readonly UV_LOCK_SHA256="cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522"
readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))
readonly MAX_RUN_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_ARTIFACT_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_ARTIFACT_BLOCKS=$((MAX_ARTIFACT_BYTES / 1024))
readonly SCHEMA_PROBE_TIMEOUT_S=120
readonly WORKER_TIMEOUT_S=900
readonly CASE_ENVELOPE_TIMEOUT_S=960
readonly GLOBAL_GPU_TIMEOUT_S=$((2 * 60 * 60))
readonly GPU_KILL_GRACE_S=30
readonly GPU_DEADLINE_SAFETY_MARGIN_S=30

PROJECT_DIR_INPUT=""
RUN_ID=""
GPU_INDEX=""
SESSION_ID=""
SOURCE_REVISION=""
RUN_ROOT=""
RUN_STATUS="error"
SCIENTIFIC_VERDICT="not_evaluated"
CURRENT_CASE=""
COMPLETED_CASES=0
COMPLETED_SCHEMA_PROBES=0
COMPLETED_FRESH_PROCESSES=0
SELECTED_GPU_UUID=""
SOURCE_TREE=""
ARCHIVE_SHA256=""
SNAPSHOT_SHA256=""
MANIFEST_SHA256=""
LAUNCH_STARTED_EPOCH="$(date +%s)"
GPU_STARTED_EPOCH=""
ACTIVE_CASE_PID=""
ACTIVE_CASE_PGID=""
LAST_GPU_UUID=""
LAST_GPU_NAME=""
LAST_GPU_DRIVER=""
LAST_GPU_MEMORY=""
LAST_GPU_UTILIZATION=""
LAST_GPU_LINE=""
LAST_GPU_PROCESSES=""

# Preserve the caller's descriptors.  Once a new run directory exists, all
# launcher output is redirected into immutable evidence files.
exec 3>&1 4>&2

usage() {
  printf '%s\n' \
    "usage: run_waveform_t1_remote.sh --project-dir PATH --run-id ID --gpu-index INDEX" \
    "       --session-id ID --source-revision 40_HEX_COMMIT" >&4
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
  [[ "$(readlink -f -- "$path")" == "$path" ]] || die "$label directory is not canonical: $path"
  [[ "$path" == "$WAVEFORM_T1_ROOT" || "$path" == "$WAVEFORM_T1_ROOT/"* ]] \
    || die "$label directory escapes the approved root: $path"
  [[ "$(stat -c '%U' "$path")" == "$EXPECTED_USER" ]] || die "$label directory owner changed: $path"
  [[ "$(findmnt -n -o TARGET -T "$path")" == "/data" ]] || die "$label directory left /data: $path"
  [[ "$(findmnt -n -o FSTYPE -T "$path")" == "xfs" ]] || die "$label directory is not on XFS: $path"
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

query_idle_gpu_state() {
  local expected_uuid="${1:-}"
  local gpu_line processes actual_index
  gpu_line="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"
  [[ "$gpu_line" != *$'\n'* ]] || return 1
  actual_index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$gpu_line")"
  LAST_GPU_UUID="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
  LAST_GPU_NAME="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
  LAST_GPU_DRIVER="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
  LAST_GPU_MEMORY="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}' <<< "$gpu_line")"
  LAST_GPU_UTILIZATION="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}' <<< "$gpu_line")"
  processes="$(nvidia-smi -i "$GPU_INDEX" \
    --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader,nounits)"
  LAST_GPU_LINE="$gpu_line"
  LAST_GPU_PROCESSES="$processes"

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

write_compute_process_envelope() {
  local processes="${1:-}"
  printf 'compute_processes_begin\n'
  if [[ -n "$processes" ]]; then
    printf '%s\n' "$processes"
  fi
  printf 'compute_processes_end\n'
}

capture_gpu_state() {
  local destination="$1"
  local expected_uuid="${2:-}"
  local temporary actual_index query_status=0
  temporary="$destination.tmp.$$"
  query_idle_gpu_state "$expected_uuid" || query_status="$?"
  actual_index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$LAST_GPU_LINE")"
  {
    printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu_index=%s\n' "$actual_index"
    printf 'gpu_uuid=%s\n' "$LAST_GPU_UUID"
    printf 'gpu_name=%s\n' "$LAST_GPU_NAME"
    printf 'driver_version=%s\n' "$LAST_GPU_DRIVER"
    printf 'memory_used_mib=%s\n' "$LAST_GPU_MEMORY"
    printf 'utilization_percent=%s\n' "$LAST_GPU_UTILIZATION"
    write_compute_process_envelope "$LAST_GPU_PROCESSES"
  } > "$temporary"
  mv -- "$temporary" "$destination"
  return "$query_status"
}

terminate_owned_case() {
  local owner_pid="${1:-}"
  local owner_pgid="${2:-}"
  local observed_pgid=""
  [[ "$owner_pid" =~ ^[1-9][0-9]*$ && "$owner_pgid" == "$owner_pid" ]] || return 0
  observed_pgid="$(ps -o pgid= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ -n "$observed_pgid" && "$observed_pgid" != "$owner_pgid" ]]; then
    return 1
  fi
  # Once the leader/PGID pair has been verified at creation, a still-live
  # group with an exited leader consists of that owned group's descendants.
  # Do not require the leader to remain present before cleaning the group.
  kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
  # The negative id addresses only the fresh session/process group created for
  # this case.  Foreign GPU PIDs are observed and recorded, never signalled.
  kill -TERM -- "-$owner_pgid" 2>/dev/null || true
  for _termination_attempt in {1..30}; do
    kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$owner_pgid" 2>/dev/null || true
  for _termination_attempt in {1..30}; do
    kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

write_launcher_inventory() {
  local temporary oversized
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  [[ ! -e "$RUN_ROOT/launcher/evidence.sha256" ]] || return 1
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || {
    printf 'final artifact exceeds 2 GiB: %s\n' "$oversized" >&2
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
  local ended_epoch elapsed gpu_elapsed root_bytes status temporary
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  ended_epoch="$(date +%s)"
  elapsed=$((ended_epoch - LAUNCH_STARTED_EPOCH))
  gpu_elapsed=0
  if [[ -n "$GPU_STARTED_EPOCH" ]]; then
    gpu_elapsed=$((ended_epoch - GPU_STARTED_EPOCH))
  fi
  root_bytes="$(du -sb "$WAVEFORM_T1_ROOT" 2>/dev/null | awk '{print $1}')"
  status="$RUN_STATUS"
  [[ "$exit_code" -eq 0 ]] || status="error"
  temporary="$RUN_ROOT/launcher/status.json.tmp.$$"
  {
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "campaign": "waveform_t1",\n'
    printf '  "status": "%s",\n' "$status"
    printf '  "scientific_verdict": "%s",\n' "$SCIENTIFIC_VERDICT"
    printf '  "exit_code": %s,\n' "$exit_code"
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "session_id": "%s",\n' "$SESSION_ID"
    printf '  "source_revision": "%s",\n' "$SOURCE_REVISION"
    printf '  "source_tree": "%s",\n' "$SOURCE_TREE"
    printf '  "snapshot_sha256": "%s",\n' "$SNAPSHOT_SHA256"
    printf '  "selected_gpu_index": %s,\n' "$GPU_INDEX"
    printf '  "selected_gpu_uuid": "%s",\n' "$SELECTED_GPU_UUID"
    printf '  "completed_case_count": %s,\n' "$COMPLETED_CASES"
    printf '  "completed_schema_probe_count": %s,\n' "$COMPLETED_SCHEMA_PROBES"
    printf '  "verified_fresh_process_count": %s,\n' "$COMPLETED_FRESH_PROCESSES"
    if [[ -n "$CURRENT_CASE" ]]; then
      printf '  "last_case_id": "%s",\n' "$CURRENT_CASE"
    else
      printf '  "last_case_id": null,\n'
    fi
    printf '  "elapsed_s": %s,\n' "$elapsed"
    printf '  "gpu_elapsed_s": %s,\n' "$gpu_elapsed"
    printf '  "approved_root_bytes": %s,\n' "${root_bytes:-0}"
    printf '  "ended_at_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
  } > "$temporary"
  mv -- "$temporary" "$RUN_ROOT/launcher/status.json"
}

finish() {
  local exit_code="$?"
  trap - EXIT
  set +e
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  write_final_status "$exit_code"
  if ! write_launcher_inventory; then
    exit_code=2
    RUN_STATUS="error"
    SCIENTIFIC_VERDICT="not_evaluated"
    write_final_status "$exit_code"
    write_launcher_inventory || true
  fi
  exec 1>&3 2>&4
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'Waveform T1 collection completed; scientific verdict awaits local compare; evidence retained under results/%s\n' "$RUN_ID"
  elif [[ -n "$RUN_ROOT" ]]; then
    printf 'Waveform T1 remote matrix stopped; evidence retained under results/%s\n' "$RUN_ID" >&2
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
  awk basename date df du env find findmnt flock git hostname id mkdir mv \
  nvidia-smi ps readlink sha256sum sleep sort stat timeout; do
  require_command "$command_name"
done

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || die "expected Server 6 hostname $EXPECTED_HOSTNAME"
[[ "$(id -un)" == "$EXPECTED_USER" ]] || die "expected user $EXPECTED_USER"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "$EXPECTED_OS_ID" && "${VERSION_ID:-}" == "$EXPECTED_OS_VERSION" ]] \
  || die "expected Ubuntu 22.04"
[[ -d "$WAVEFORM_T1_ROOT" ]] || die "approved root does not exist"
refuse_symlink "$WAVEFORM_T1_ROOT"
[[ "$(readlink -f "$WAVEFORM_T1_ROOT")" == "$WAVEFORM_T1_ROOT" ]] || die "approved root is not canonical"
[[ "$(stat -c '%U' "$WAVEFORM_T1_ROOT")" == "$EXPECTED_USER" ]] || die "approved root owner changed"
[[ "$(findmnt -n -o TARGET -T "$WAVEFORM_T1_ROOT")" == "/data" ]] || die "approved root left /data"
[[ "$(findmnt -n -o FSTYPE -T "$WAVEFORM_T1_ROOT")" == "xfs" ]] || die "approved root is not on XFS"
[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2 && "$PROJECT_DIR_INPUT" != /mnt/ceph2/* ]] \
  || die "the Ceph mount is outside the approved boundary"

PROJECT_DIR="$(readlink -f -- "$PROJECT_DIR_INPUT")"
[[ "$PROJECT_DIR" == "$PROJECT_DIR_INPUT" ]] || die "project directory must be an absolute canonical path"
[[ "$PROJECT_DIR" == "$WAVEFORM_T1_ROOT/project/"* ]] || die "project directory escapes the approved project root"
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
[[ "$PROJECT_DIR" == "$WAVEFORM_T1_ROOT/project/$SOURCE_TREE" ]] \
  || die "project directory must exactly match project/source-tree"

readonly ASSET_ROOT="$WAVEFORM_T1_ROOT/assets"
readonly ISAACLAB_ROOT="$WAVEFORM_T1_ROOT/IsaacLab"
readonly DOWNLOADS_ROOT="$WAVEFORM_T1_ROOT/downloads"
readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"
readonly WORKER_PYTHON="$WAVEFORM_T1_ROOT/env/bin/python"
readonly MANIFEST="$PROJECT_DIR/configs/parity/waveform_t1.json"
readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_runtime.py"
readonly SCHEMA_PROBE_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_schema.py"
readonly SCHEMA_VALIDATOR_SCRIPT="$PROJECT_DIR/scripts/finalize_gate0.py"
require_real_directory "$WAVEFORM_T1_ROOT/project" "project"
require_real_directory "$PROJECT_DIR" "project snapshot"
require_real_directory "$DOWNLOADS_ROOT" "downloads"
require_real_directory "$ASSET_ROOT" "asset"
require_real_directory "$ISAACLAB_ROOT" "IsaacLab"
[[ -f "$SOURCE_ARCHIVE" ]] || die "deployed source archive is missing: $SOURCE_ARCHIVE"
refuse_symlink "$SOURCE_ARCHIVE"
[[ "$(readlink -f -- "$SOURCE_ARCHIVE")" == "$SOURCE_ARCHIVE" ]] \
  || die "deployed source archive path is not canonical"
for writable_path in \
  "$WAVEFORM_T1_ROOT/results" \
  "$WAVEFORM_T1_ROOT/cache" \
  "$WAVEFORM_T1_ROOT/cache/cuda" \
  "$WAVEFORM_T1_ROOT/cache/pip" \
  "$WAVEFORM_T1_ROOT/cache/pycache" \
  "$WAVEFORM_T1_ROOT/cache/torch_extensions" \
  "$WAVEFORM_T1_ROOT/cache/triton" \
  "$WAVEFORM_T1_ROOT/cache/uv" \
  "$WAVEFORM_T1_ROOT/cache/warp" \
  "$WAVEFORM_T1_ROOT/cache/xdg" \
  "$WAVEFORM_T1_ROOT/config" \
  "$WAVEFORM_T1_ROOT/data" \
  "$WAVEFORM_T1_ROOT/state" \
  "$WAVEFORM_T1_ROOT/tmp"; do
  require_writable_directory "$writable_path"
done
[[ -x "$WORKER_PYTHON" ]] || die "Waveform T1 Python is missing"
worker_python_real="$(readlink -f "$WORKER_PYTHON")"
[[ "$worker_python_real" == "$WAVEFORM_T1_ROOT/env/"* \
  || "$worker_python_real" == "$WAVEFORM_T1_ROOT/toolchain/python/"* ]] \
  || die "Waveform T1 Python resolves outside the approved root"
for source_file in \
  "$MANIFEST" "$WORKER_SCRIPT" "$SCHEMA_PROBE_SCRIPT" "$SCHEMA_VALIDATOR_SCRIPT"; do
  [[ -f "$source_file" ]] || die "required source file is missing: $(basename "$source_file")"
  refuse_symlink "$source_file"
done
[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_waveform_t1_remote.sh" ]] \
  || die "launcher must execute from the selected immutable project snapshot"

RUN_ROOT="$WAVEFORM_T1_ROOT/results/$RUN_ID"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run directory already exists; refusing overwrite"

# This first strict-idle probe is intentionally query-only.  A busy GPU leaves
# no managed write or mtime change.  The lock is then acquired before run-dir
# creation, and the recorded check closes the selection/publication race.
if ! query_idle_gpu_state; then
  die "selected GPU is not the pinned, completely idle Server 6 device"
fi
SELECTED_GPU_UUID="$LAST_GPU_UUID"

refuse_symlink "$WAVEFORM_T1_ROOT/.waveform-t1-run.lock"
exec 9>"$WAVEFORM_T1_ROOT/.waveform-t1-run.lock"
flock -n 9 || die "another Waveform T1 matrix launcher holds the run lock"

mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/launcher" "$RUN_ROOT/cases" "$RUN_ROOT/launcher/schema"
for run_write_path in \
  "$RUN_ROOT" "$RUN_ROOT/launcher" "$RUN_ROOT/cases" "$RUN_ROOT/launcher/schema"; do
  require_writable_directory "$run_write_path"
done
exec 1>"$RUN_ROOT/launcher/stdout.txt" 2>"$RUN_ROOT/launcher/stderr.txt"
GPU_STARTED_EPOCH="$(date +%s)"
if ! capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt" "$SELECTED_GPU_UUID"; then
  die "selected GPU changed or ceased to be completely idle after run creation"
fi
printf 'Waveform T1 launcher started at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$RUN_ROOT"

[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] \
  || die "display variables must be unset for kit-less Waveform T1"
unset DISPLAY WAYLAND_DISPLAY
unset PYTHONBREAKPOINT PYTHONCASEOK PYTHONCOERCECLOCALE PYTHONDEBUG PYTHONDEVMODE
unset PYTHONDONTWRITEBYTECODE PYTHONEXECUTABLE PYTHONFAULTHANDLER PYTHONHOME
unset PYTHONINSPECT PYTHONINTMAXSTRDIGITS PYTHONMALLOC PYTHONNODEBUGRANGES
unset PYTHONPATH
unset PYTHONOPTIMIZE PYTHONPLATLIBDIR PYTHONPROFILEIMPORTTIME
unset PYTHONSTARTUP PYTHONTRACEMALLOC PYTHONUSERBASE PYTHONWARNDEFAULTENCODING
unset PYTHONWARNINGS
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$WAVEFORM_T1_ROOT/cache/pycache"
export PYTHONSAFEPATH=1
export PYTHONUTF8=1
export HOME="$WAVEFORM_T1_ROOT/data"
export CUDA_CACHE_PATH="$WAVEFORM_T1_ROOT/cache/cuda"
export PIP_CACHE_DIR="$WAVEFORM_T1_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$WAVEFORM_T1_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$WAVEFORM_T1_ROOT/cache/triton"
export TMPDIR="$WAVEFORM_T1_ROOT/tmp"
export UV_CACHE_DIR="$WAVEFORM_T1_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$WAVEFORM_T1_ROOT/env"
export WARP_CACHE_PATH="$WAVEFORM_T1_ROOT/cache/warp"
export XDG_CACHE_HOME="$WAVEFORM_T1_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$WAVEFORM_T1_ROOT/config"
export XDG_DATA_HOME="$WAVEFORM_T1_ROOT/data"
export XDG_STATE_HOME="$WAVEFORM_T1_ROOT/state"

root_bytes() {
  du -sb "$WAVEFORM_T1_ROOT" | awk '{print $1}'
}

check_root_budget() {
  local bytes
  bytes="$(root_bytes)"
  [[ "$bytes" -le "$MAX_ROOT_BYTES" ]] || die "approved root exceeds the 30 GiB stop limit"
}

check_run_budget() {
  local bytes oversized
  bytes="$(du -sb "$RUN_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_RUN_BYTES" ]] || die "run evidence exceeds the 2 GiB stop limit"
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || die "artifact exceeds the 2 GiB stop limit: $oversized"
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
  [[ -f "$SOURCE_ARCHIVE" && ! -L "$SOURCE_ARCHIVE" ]] \
    || die "deployed source archive is no longer a regular file"
  [[ "$(readlink -f -- "$SOURCE_ARCHIVE")" == "$SOURCE_ARCHIVE" ]] \
    || die "deployed source archive path changed"
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
    actual = metadata.version(package)
    if actual != version:
        raise SystemExit(f"unexpected {package} version: {actual}")
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
try:
    cuda_runtime_wheel = metadata.version("nvidia-cuda-runtime-cu12")
except metadata.PackageNotFoundError:
    cuda_runtime_wheel = "not-installed"
print(f"nvidia_cuda_runtime_cu12={cuda_runtime_wheel}")
print("kitless=true")
PY
)"

matrix_file="$RUN_ROOT/launcher/cases.txt.tmp"
PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - "$MANIFEST" <<'PY' > "$matrix_file"
import sys

from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.scenarios import expand_scenario_cases, load_manifest, manifest_sha256

manifest = load_manifest(sys.argv[1])
if manifest.schema_version != 2:
    raise SystemExit("Waveform T1 manifest schema_version must be 2")
if manifest.manifest_id != "wavesimparity-waveform-t1":
    raise SystemExit("Waveform T1 manifest_id mismatch")
if tuple(scenario.scenario_id for scenario in manifest.scenarios) != (
    "offset_sine",
    "offset_linear_chirp",
):
    raise SystemExit("Waveform T1 scenario order mismatch")
if manifest.provenance.commit != "6eea427eb24189519f32b9f21674cd534d3f973c":
    raise SystemExit("manifest asset commit mismatch")
if manifest.provenance.asset_git_tree != "bb00a9d5527b8a76de576ce876ebece67d8ffde1":
    raise SystemExit("manifest asset Git tree mismatch")
if manifest.provenance.canonical_lf_asset_tree_sha256 != "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad":
    raise SystemExit("manifest canonical LF hash mismatch")
cases = [case for case in expand_scenario_cases(manifest) if case.simulator is Simulator.OVPHYSX]
if len(cases) != 16 or len({case.case_id for case in cases}) != 16:
    raise SystemExit(f"expected exactly 16 unique OVPhysX cases, found {len(cases)}")
print("manifest_sha256=" + manifest_sha256(manifest))
for case in cases:
    print(case.case_id)
PY
mv -- "$matrix_file" "$RUN_ROOT/launcher/cases.txt"
mapfile -t matrix_lines < "$RUN_ROOT/launcher/cases.txt"
[[ "${#matrix_lines[@]}" -eq $((EXPECTED_OVPHYSX_CASE_COUNT + 1)) ]] \
  || die "canonical matrix file must contain one hash and 16 cases"
MANIFEST_SHA256="${matrix_lines[0]#manifest_sha256=}"
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "canonical manifest SHA-256 is invalid"
CASE_IDS=("${matrix_lines[@]:1}")
declare -A FRESH_PROCESS_IDS=()
for case_id in "${CASE_IDS[@]}"; do
  safe_identifier "$case_id" || die "unsafe canonical case ID: $case_id"
  [[ "$case_id" == ovphysx.* ]] || die "remote launcher received a non-OVPhysX case"
done

write_directory_inventory() {
  local directory="$1"
  local inventory="$directory/evidence.sha256"
  local temporary="$inventory.tmp.$$"
  [[ ! -e "$inventory" && ! -L "$inventory" ]] || die "evidence inventory already exists: $inventory"
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

validate_schema_probe() {
  local raw_probe="$1"
  local hand="$2"
  local record="$3"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
    "$raw_probe" "$hand" "$record" "$SESSION_ID" "$SOURCE_REVISION" \
    "$SNAPSHOT_SHA256" "$MANIFEST_SHA256" "$ASSET_LF_SHA256" \
    "$ASSET_COMMIT" "$ASSET_GIT_TREE" "$MANIFEST" "$ASSET_ROOT" \
    "$SCHEMA_PROBE_SCRIPT" <<'PY'
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

(
    raw_path,
    hand,
    record_path,
    session_id,
    source_revision,
    snapshot_sha,
    manifest_sha,
    asset_sha,
    asset_commit,
    asset_git_tree,
    manifest_path,
    asset_root_path,
    probe_script_path,
) = sys.argv[1:]

def object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_constant(token):
    raise ValueError(f"non-finite JSON number: {token}")

def load_strict(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=object_without_duplicates,
        parse_constant=reject_constant,
    )

payload = load_strict(raw_path)
manifest = load_strict(manifest_path)
if not isinstance(payload, dict) or payload.get("backend") != "ovphysx":
    raise SystemExit("schema probe backend is not ovphysx")
if payload.get("mode") != "resolved_usd_headless" or payload.get("status") != "available":
    raise SystemExit("resolved-USD schema probe did not report available")
if payload.get("error") is not None:
    raise SystemExit("available schema probe unexpectedly contains an error")

capabilities = payload.get("capabilities")
if not isinstance(capabilities, dict):
    raise SystemExit("schema probe capabilities are missing")
required_true = {
    "kitless_contract",
    "renderer_disabled",
    "camera_disabled",
    "pinned_ovphysx_wheel_version",
    "resolved_usd_open",
    "stage_fully_composed",
    "articulation_root_schema",
    "canonical_joint_count",
    "position_drive_schema",
    "physx_properties_authored",
    "canonical_distal_frame_count",
}
missing = sorted(name for name in required_true if capabilities.get(name) is not True)
if missing:
    raise SystemExit("schema probe lacks required capabilities: " + ", ".join(missing))
if capabilities.get("physics_step") is not False:
    raise SystemExit("schema-only probe must not claim a physics step")

provenance = payload.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit("schema probe provenance is missing")
constraints = provenance.get("constraints")
if constraints != {"headless": True, "kit": False, "renderer": False, "camera": False}:
    raise SystemExit("schema probe violated the kit-less constraint record")
packages = provenance.get("packages")
expected_packages = {
    "isaaclab": "6.1.14",
    "isaaclab-ovphysx": "3.0.2",
    "ovphysx": "0.4.13",
    "torch": "2.10.0+cu128",
    "usd-core": "25.11",
}
if not isinstance(packages, dict) or any(packages.get(k) != v for k, v in expected_packages.items()):
    raise SystemExit("schema probe package provenance differs from the pinned environment")
inspection = provenance.get("usd_inspection")
if not isinstance(inspection, dict):
    raise SystemExit("schema probe lacks resolved USD observations")

hands = [item for item in manifest.get("hands", []) if item.get("side") == hand]
if len(hands) != 1:
    raise SystemExit(f"manifest does not contain exactly one {hand} hand")
hand_spec = hands[0]
relative_source = hand_spec["model_paths"]["ovphysx"]
asset_root = Path(asset_root_path).resolve(strict=True)
expected_source = (asset_root / relative_source).resolve(strict=True)
expected_source.relative_to(asset_root)
observed_source = Path(inspection.get("source_path", "")).resolve(strict=True)
if observed_source != expected_source:
    raise SystemExit("schema probe inspected the wrong resolved USD")
if inspection.get("source_sha256") != sha256(expected_source.read_bytes()).hexdigest():
    raise SystemExit("schema probe source SHA-256 does not match the resolved USD")

expected_joints = set(hand_spec["joint_names"])
expected_frames = set(hand_spec["distal_frame_names"])
list_expectations = {
    "revolute_joint_names": (22, expected_joints),
    "angular_drive_joint_names": (22, expected_joints),
    "physx_velocity_joint_names": (22, expected_joints),
    "distal_frame_names": (5, expected_frames),
}
for field, (count, expected_names) in list_expectations.items():
    values = inspection.get(field)
    if not isinstance(values, list) or len(values) != count or set(values) != expected_names:
        raise SystemExit(f"schema observation {field} does not match the canonical hand")
count_expectations = {
    "revolute_joint_count": 22,
    "angular_drive_count": 22,
    "physx_velocity_joint_count": 22,
    "distal_frame_count": 5,
}
for field, expected in count_expectations.items():
    value = inspection.get(field)
    if isinstance(value, bool) or value != expected:
        raise SystemExit(f"schema observation {field} must equal {expected}")
roots = inspection.get("articulation_root_paths")
if not isinstance(roots, list) or len(roots) != 1:
    raise SystemExit("schema probe must observe one articulation root")

portable_inspection = dict(inspection)
portable_inspection["source_path"] = relative_source
record = {
    "schema_version": 1,
    "backend": "ovphysx",
    "mode": "resolved_usd_headless",
    "status": "available",
    "hand": hand,
    "session_id": session_id,
    "source_revision": source_revision,
    "snapshot_sha256": snapshot_sha,
    "manifest_sha256": manifest_sha,
    "asset_tree_sha256": asset_sha,
    "asset_commit": asset_commit,
    "asset_git_tree": asset_git_tree,
    "schema_probe_script_sha256": sha256(Path(probe_script_path).read_bytes()).hexdigest(),
    "raw_probe_sha256": sha256(Path(raw_path).read_bytes()).hexdigest(),
    "packages": expected_packages,
    "capabilities": {name: capabilities[name] for name in sorted(required_true | {"physics_step"})},
    "usd_inspection": portable_inspection,
    "scientific_scope": "resolved_usd_schema_only_no_physics_step",
}
rendered = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
destination = Path(record_path)
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(rendered)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

validate_finalizer_schema_probe() {
  local payload="$1"
  local hand="$2"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
    "$SCHEMA_VALIDATOR_SCRIPT" "$SCHEMA_PROBE_SCRIPT" "$payload" "$hand" "$MANIFEST" \
    "$SESSION_ID" "$SOURCE_REVISION" <<'PY'
import importlib.util
from hashlib import sha256
from pathlib import Path
import sys

(
    finalizer_path,
    schema_probe_path,
    payload_path,
    hand,
    manifest_path,
    session_id,
    source_revision,
) = sys.argv[1:]
spec = importlib.util.spec_from_file_location("waveform_t1_remote_contract", finalizer_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load the pinned resolved-schema validator contract")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = module.load_manifest(Path(manifest_path))
payload = module._strict_json_file(Path(payload_path))
module._validate_schema_probe(
    payload,
    hand_side=hand,
    manifest=manifest,
    manifest_file_sha256=sha256(Path(manifest_path).read_bytes()).hexdigest(),
    session_id=session_id,
    source_revision=source_revision,
    schema_probe_script_sha256=sha256(Path(schema_probe_path).read_bytes()).hexdigest(),
)
PY
}

write_schema_inventory() {
  local schema_dir="$RUN_ROOT/launcher/schema"
  local inventory="$schema_dir/evidence.sha256"
  local temporary="$inventory.tmp.$$"
  local evidence_name digest
  local -a expected_names=(
    left.exitcode left.json left.stderr left.stdout
    right.exitcode right.json right.stderr right.stdout
  )
  [[ ! -e "$inventory" && ! -L "$inventory" ]] \
    || die "formal schema evidence inventory already exists"
  for evidence_name in "${expected_names[@]}"; do
    [[ -f "$schema_dir/$evidence_name" && ! -L "$schema_dir/$evidence_name" ]] \
      || die "formal schema evidence is incomplete: $evidence_name"
    digest="$(sha256sum -- "$schema_dir/$evidence_name" | awk '{print $1}')"
    printf '%s  %s\n' "$digest" "$evidence_name"
  done > "$temporary"
  [[ "$(find "$schema_dir" -mindepth 1 -maxdepth 1 -type f \
    ! -name evidence.sha256 ! -name 'evidence.sha256.tmp.*' | wc -l)" -eq 8 ]] \
    || die "formal schema directory contains unexpected files"
  mv -- "$temporary" "$inventory"
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

capture_actual_worker_identity() {
  local owner_pgid="$1"
  local case_id="$2"
  local destination="$3"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
    "$owner_pgid" "$case_id" "$WORKER_PYTHON" "$WORKER_SCRIPT" \
    "$destination" <<'PY'
import json
import os
from pathlib import Path
import sys

from wave_asset_qa.parity.process_identity import (
    os_process_identity_sha256,
    validate_os_process_identity,
)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


owner_pgid = int(sys.argv[1])
case_id = sys.argv[2]
worker_python = os.path.realpath(sys.argv[3])
worker_script = os.path.realpath(sys.argv[4])
destination = Path(sys.argv[5])
expected_record_keys = {
    "schema_version",
    "visibility",
    "case_id",
    "worker_command_identity",
    "os_process_identity",
    "fresh_process_id",
}

if destination.is_symlink():
    fail("fresh-process identity destination is a symbolic link")

existing = None
if destination.exists():
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read retained fresh-process identity: {type(error).__name__}")
    if not isinstance(existing, dict) or set(existing) != expected_record_keys:
        fail("retained fresh-process identity has an invalid record shape")
    if (
        existing.get("schema_version") != 1
        or existing.get("visibility") != "private_not_for_publication"
        or existing.get("case_id") != case_id
        or existing.get("worker_command_identity")
        != {
            "python_realpath": worker_python,
            "script_realpath": worker_script,
        }
    ):
        fail("retained fresh-process identity metadata mismatch")
    try:
        retained_identity = validate_os_process_identity(
            existing.get("os_process_identity")
        )
    except Exception as error:
        fail(f"retained fresh-process identity is invalid: {type(error).__name__}")
    if existing.get("fresh_process_id") != os_process_identity_sha256(
        retained_identity
    ):
        fail("retained fresh-process identity hash mismatch")

try:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
except (OSError, UnicodeError) as error:
    fail(f"cannot read Linux boot identity: {type(error).__name__}")

candidates: list[dict[str, object]] = []
for process_dir in Path("/proc").iterdir():
    if not process_dir.name.isdigit():
        continue
    try:
        stat_before = (process_dir / "stat").read_text(encoding="ascii")
        _prefix, separator, suffix = stat_before.rpartition(")")
        fields_after_comm = suffix.strip().split()
        if (
            not separator
            or len(fields_after_comm) <= 19
            or fields_after_comm[0] in {"Z", "X", "x"}
            or int(fields_after_comm[2]) != owner_pgid
        ):
            continue
        argv = (process_dir / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if len(argv) < 2:
            continue
        if (
            os.path.realpath(os.fsdecode(argv[0])) != worker_python
            or os.path.realpath(os.fsdecode(argv[1])) != worker_script
        ):
            continue
        stat_after = (process_dir / "stat").read_text(encoding="ascii")
        if stat_after != stat_before:
            continue
        identity = validate_os_process_identity(
            {
                "platform": "posix",
                "boot_id": boot_id,
                "pid": int(process_dir.name),
                "process_start_ticks": int(fields_after_comm[19]),
            }
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError, ValueError):
        continue
    candidates.append(identity)

if len(candidates) > 1:
    fail("more than one actual Python worker exists in the owned process group")
if existing is not None:
    if candidates and candidates[0] != retained_identity:
        fail("a second actual Python worker appeared in one canonical case")
    raise SystemExit(0)
if not candidates:
    raise SystemExit(1)

identity = candidates[0]
record = {
    "schema_version": 1,
    "visibility": "private_not_for_publication",
    "case_id": case_id,
    "worker_command_identity": {
        "python_realpath": worker_python,
        "script_realpath": worker_script,
    },
    "os_process_identity": identity,
    "fresh_process_id": os_process_identity_sha256(identity),
}
temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
try:
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
except Exception:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    raise
PY
}

monitor_owned_case() {
  local owner_pid="$1"
  local owner_pgid="$2"
  local destination="$3"
  local foreign_marker="$4"
  local gpu_line processes observed_uuid observed_name observed_driver
  local worker_identity_path="${5:-}"
  local worker_case_id="${6:-}"
  local row process_uuid process_pid process_memory process_pgid foreign_reason run_bytes
  local identity_capture_status
  [[ ! -e "$destination" && ! -L "$destination" ]] || return 1
  : > "$destination"
  while case_process_running "$owner_pid"; do
    if [[ -n "$worker_identity_path" ]]; then
      if capture_actual_worker_identity \
        "$owner_pgid" "$worker_case_id" "$worker_identity_path"; then
        :
      else
        identity_capture_status="$?"
        if [[ "$identity_capture_status" -ne 1 ]]; then
          printf '%s\n' \
            "actual Python worker identity capture failed closed" > "$foreign_marker"
          terminate_owned_case "$owner_pid" "$owner_pgid"
          return 1
        fi
      fi
    fi
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
      write_compute_process_envelope "$processes"
    } >> "$destination"

    foreign_reason=""
    run_bytes="$(du -sb "$RUN_ROOT" | awk '{print $1}')"
    if [[ ! "$run_bytes" =~ ^[0-9]+$ ]]; then
      foreign_reason="cannot measure the Waveform T1 run evidence tree"
    elif [[ "$run_bytes" -gt "$MAX_RUN_BYTES" ]]; then
      foreign_reason="Waveform T1 run evidence exceeded the 2 GiB hard limit"
    fi
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
        foreign_reason="GPU process pid=$process_pid disappeared before owned-group identity could be proven"
        break
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

for schema_hand in left right; do
  CURRENT_CASE="schema.$schema_hand"
  require_writable_directory "$RUN_ROOT/launcher/schema"
  verify_snapshot "schema-$schema_hand"
  now_epoch="$(date +%s)"
  gpu_elapsed=$((now_epoch - GPU_STARTED_EPOCH))
  remaining=$((GLOBAL_GPU_TIMEOUT_S - gpu_elapsed))
  usable=$((remaining - GPU_KILL_GRACE_S - GPU_DEADLINE_SAFETY_MARGIN_S))
  [[ "$usable" -gt 0 ]] || die "global two-hour GPU budget cannot fit kill grace and safety margin"
  schema_envelope="$SCHEMA_PROBE_TIMEOUT_S"
  if [[ "$usable" -lt "$schema_envelope" ]]; then
    schema_envelope="$usable"
  fi
  schema_json="$RUN_ROOT/launcher/schema/$schema_hand.json"
  schema_stdout="$RUN_ROOT/launcher/schema/$schema_hand.stdout"
  schema_stderr="$RUN_ROOT/launcher/schema/$schema_hand.stderr"
  schema_exitcode="$RUN_ROOT/launcher/schema/$schema_hand.exitcode"
  schema_pid_file="$RUN_ROOT/launcher/schema-$schema_hand-owned-process-group.txt"
  if ! capture_gpu_state \
    "$RUN_ROOT/launcher/schema-$schema_hand-preflight.txt" "$SELECTED_GPU_UUID"; then
    die "selected GPU is not completely idle before the $schema_hand schema probe"
  fi
  schema_command=(
    timeout --foreground --signal=TERM --kill-after="${GPU_KILL_GRACE_S}s" "${schema_envelope}s"
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    PYTHONPATH="$PROJECT_DIR/src"
    "$WORKER_PYTHON" -P "$SCHEMA_PROBE_SCRIPT"
    --approved-root "$WAVEFORM_T1_ROOT"
    --asset-root "$ASSET_ROOT"
    --manifest "$MANIFEST"
    --hand "$schema_hand"
    --output "$schema_json"
    --device cuda:0
    --session-id "$SESSION_ID"
    --source-revision "$SOURCE_REVISION"
    --asset-tree-sha256 "$ASSET_LF_SHA256"
  )
  (
    ulimit -f "$MAX_ARTIFACT_BLOCKS"
    exec "$WORKER_PYTHON" -P - "$schema_pid_file" "${schema_command[@]}"
  ) > "$schema_stdout" 2> "$schema_stderr" <<'PY' &
import os
from pathlib import Path
import sys

pid_path = Path(sys.argv[1])
command = sys.argv[2:]
if not command:
    raise SystemExit("owned schema-probe wrapper received no command")
os.setsid()
descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execvpe(command[0], command, os.environ.copy())
PY
  schema_owner_pid="$!"
  for _schema_pid_wait_attempt in {1..50}; do
    [[ -s "$schema_pid_file" ]] && break
    case_process_running "$schema_owner_pid" || break
    sleep 0.1
  done
  if [[ ! -s "$schema_pid_file" ]]; then
    set +e
    wait "$schema_owner_pid"
    schema_exit_code="$?"
    set -e
    printf '%s\n' "$schema_exit_code" > "$schema_exitcode"
    die "owned process group was not established for the $schema_hand schema probe"
  fi
  IFS= read -r schema_pgid < "$schema_pid_file"
  observed_schema_pgid="$(ps -o pgid= -p "$schema_owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ ! "$schema_pgid" =~ ^[1-9][0-9]*$ \
    || "$schema_pgid" != "$schema_owner_pid" \
    || "$observed_schema_pgid" != "$schema_pgid" ]]; then
    # The positive PID is not a safe signal target until the fresh PGID
    # handshake is proven.  The owned timeout wrapper is bounded, so retain
    # evidence and wait without risking a signal to a reused foreign PID.
    set +e
    wait "$schema_owner_pid"
    schema_exit_code="$?"
    set -e
    printf '%s\n' "$schema_exit_code" > "$schema_exitcode"
    die "schema-probe process-group identity is unsafe"
  fi
  ACTIVE_CASE_PID="$schema_owner_pid"
  ACTIVE_CASE_PGID="$schema_pgid"
  foreign_gpu_process=false
  if ! monitor_owned_case \
    "$schema_owner_pid" "$schema_pgid" \
    "$RUN_ROOT/launcher/schema-$schema_hand-gpu-monitor.txt" \
    "$RUN_ROOT/launcher/schema-$schema_hand-foreign-gpu-process.txt"; then
    foreign_gpu_process=true
  fi
  set +e
  wait "$schema_owner_pid"
  schema_exit_code="$?"
  set -e
  if ! terminate_owned_case "$schema_owner_pid" "$schema_pgid"; then
    die "owned schema-probe process group did not terminate"
  fi
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  printf '%s\n' "$schema_exit_code" > "$schema_exitcode"
  if [[ "$foreign_gpu_process" == true ]]; then
    IFS= read -r foreign_reason \
      < "$RUN_ROOT/launcher/schema-$schema_hand-foreign-gpu-process.txt" \
      || foreign_reason="foreign GPU process detected"
    die "$foreign_reason; only the owned schema-probe process group was terminated"
  fi
  schema_postflight_ok=false
  for schema_postflight_attempt in 1 2 3; do
    schema_postflight_path="$(printf '%s/launcher/schema-%s-postflight-%02d.txt' \
      "$RUN_ROOT" "$schema_hand" "$schema_postflight_attempt")"
    if capture_gpu_state "$schema_postflight_path" "$SELECTED_GPU_UUID"; then
      schema_postflight_ok=true
      break
    fi
    [[ "$schema_postflight_attempt" -eq 3 ]] || sleep 5
  done
  [[ "$schema_postflight_ok" == true ]] \
    || die "selected GPU did not return to idle after the $schema_hand schema probe"
  [[ "$schema_exit_code" -eq 0 ]] \
    || die "resolved-USD schema probe failed for $schema_hand"
  [[ -f "$schema_json" && ! -L "$schema_json" ]] \
    || die "schema probe did not produce structured JSON for $schema_hand"
  validate_finalizer_schema_probe "$schema_json" "$schema_hand"
  COMPLETED_SCHEMA_PROBES=$((COMPLETED_SCHEMA_PROBES + 1))
  check_run_budget
done
[[ "$COMPLETED_SCHEMA_PROBES" -eq 2 ]] || die "both formal schema probes are required"
write_schema_inventory
CURRENT_CASE=""

{
  printf 'schema_version=1\n'
  printf 'campaign=waveform_t1\n'
  printf 'manifest_id=%s\n' "$EXPECTED_MANIFEST_ID"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'session_id=%s\n' "$SESSION_ID"
  printf 'source_revision=%s\n' "$SOURCE_REVISION"
  printf 'source_tree=%s\n' "$SOURCE_TREE"
  printf 'archive_sha256=%s\n' "$ARCHIVE_SHA256"
  printf 'snapshot_sha256=%s\n' "$SNAPSHOT_SHA256"
  printf 'project_path=project/%s\n' "$SOURCE_TREE"
  printf 'launcher_sha256='; sha256sum "$PROJECT_DIR/scripts/run_waveform_t1_remote.sh" | awk '{print $1}'
  printf 'worker_sha256='; sha256sum "$WORKER_SCRIPT" | awk '{print $1}'
  printf 'schema_probe_sha256='; sha256sum "$SCHEMA_PROBE_SCRIPT" | awk '{print $1}'
  printf 'schema_validator_sha256='; sha256sum "$SCHEMA_VALIDATOR_SCRIPT" | awk '{print $1}'
  printf 'formal_schema_probe_count=%s\n' "$COMPLETED_SCHEMA_PROBES"
  printf 'manifest_file_sha256='; sha256sum "$MANIFEST" | awk '{print $1}'
  printf 'manifest_canonical_sha256=%s\n' "$MANIFEST_SHA256"
  printf 'asset_commit=%s\n' "$ASSET_COMMIT"
  printf 'asset_git_tree=%s\n' "$ASSET_GIT_TREE"
  printf 'asset_lf_sha256=%s\n' "$ASSET_LF_SHA256"
  printf 'isaaclab_commit=%s\n' "$ISAACLAB_COMMIT"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'gpu_uuid=%s\n' "$SELECTED_GPU_UUID"
  printf 'gpu_name=%s\n' "$LAST_GPU_NAME"
  printf 'driver_version=%s\n' "$LAST_GPU_DRIVER"
  printf '%s\n' "$environment_summary"
} > "$RUN_ROOT/launcher/provenance.txt"

validate_collected_run() {
  local run_file="$1"
  local case_id="$2"
  local worker_identity_file="$3"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
    "$run_file" "$case_id" "$SESSION_ID" "$SOURCE_REVISION" \
    "$MANIFEST_SHA256" "$ASSET_LF_SHA256" "$ASSET_COMMIT" "$ASSET_GIT_TREE" \
    "$MANIFEST" "$worker_identity_file" "$WORKER_PYTHON" "$WORKER_SCRIPT" <<'PY'
import json
import math
import os
import sys

from wave_asset_qa.parity.process_identity import (
    os_process_identity_sha256,
    validate_os_process_identity,
)
from wave_asset_qa.parity.runner import load_collected_runs
from wave_asset_qa.parity.scenarios import (
    canonical_target_sequence_sha256,
    load_manifest,
)

(
    path,
    expected_case,
    session_id,
    source_revision,
    manifest_sha,
    asset_sha,
    asset_commit,
    asset_git_tree,
    manifest_path,
    worker_identity_path,
    expected_worker_python,
    expected_worker_script,
) = sys.argv[1:]
manifest = load_manifest(manifest_path)
runs = load_collected_runs(path)
if len(runs) != 1:
    raise SystemExit("case output must contain exactly one collected run")
run = runs[0]
if run.case.case_id != expected_case:
    raise SystemExit("collected case ID mismatch")
if not run.result.completed:
    raise SystemExit("collected OVPhysX run is not completed")
provenance = dict(run.result.provenance)
expected_values = {
    "manifest_sha256": manifest_sha,
    "session_id": session_id,
    "source_revision": source_revision,
    "asset_tree_sha256": asset_sha,
    "asset_commit": asset_commit,
    "asset_git_tree": asset_git_tree,
    "mapping_schema_version": 1,
}
for key, expected in expected_values.items():
    if provenance.get(key) != expected:
        raise SystemExit(f"invalid completed-run provenance field: {key}")

scenario = manifest.scenario(run.case.scenario_id)
hand = manifest.hand(run.case.hand)
if run.result.joint_names != hand.joint_names:
    raise SystemExit("completed run canonical joint order mismatch")
if run.result.frame_names != hand.distal_frame_names:
    raise SystemExit("completed run canonical frame order mismatch")
expected_steps = round(scenario.duration_s / run.case.dt_s)
if run.result.requested_steps != expected_steps or run.result.completed_steps != expected_steps:
    raise SystemExit("completed run step count mismatch")

full_target_sha = canonical_target_sequence_sha256(
    scenario,
    hand.joint_names,
    dt_s=run.case.dt_s,
    include_terminal=True,
)
pre_step_target_sha = canonical_target_sequence_sha256(
    scenario,
    hand.joint_names,
    dt_s=run.case.dt_s,
    include_terminal=False,
)
target_fixed = {
    "target_sequence_digest_schema_version": 1,
    "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
    "target_sequence_digest_projection": "ieee754_binary32_roundtrip",
    "target_sequence_canonical_joint_names": list(hand.joint_names),
    "scheduled_target_sequence_count": expected_steps + 1,
    "requested_target_sequence_count": expected_steps + 1,
    "immediate_target_readback_sequence_count": expected_steps + 1,
    "pre_step_applied_target_readback_sequence_count": expected_steps,
    "scheduled_target_sequence_sha256": full_target_sha,
    "requested_target_sequence_sha256": full_target_sha,
    "immediate_target_readback_sequence_sha256": full_target_sha,
    "pre_step_applied_target_readback_sequence_sha256": pre_step_target_sha,
}
for key, expected in target_fixed.items():
    if provenance.get(key) != expected:
        raise SystemExit(f"invalid completed-run target-sequence field: {key}")
canonical_readback_error = provenance.get(
    "position_target_canonical_readback_max_abs_error_rad"
)
if (
    isinstance(canonical_readback_error, bool)
    or not isinstance(canonical_readback_error, (int, float))
    or not math.isfinite(float(canonical_readback_error))
    or not 0.0 <= float(canonical_readback_error) <= 1e-6
):
    raise SystemExit("canonical position-target readback error is invalid")
final_readback = provenance.get("position_target_readback_values_rad")
if not isinstance(final_readback, list) or len(final_readback) != 22:
    raise SystemExit("final position-target readback vector is invalid")
if any(
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not math.isfinite(float(value))
    for value in final_readback
):
    raise SystemExit("final position-target readback vector is non-finite")

with open(worker_identity_path, encoding="utf-8") as handle:
    worker_record = json.load(handle)
expected_worker_record_keys = {
    "schema_version",
    "visibility",
    "case_id",
    "worker_command_identity",
    "os_process_identity",
    "fresh_process_id",
}
if not isinstance(worker_record, dict) or set(worker_record) != expected_worker_record_keys:
    raise SystemExit("actual Python worker identity record shape mismatch")
if (
    worker_record.get("schema_version") != 1
    or worker_record.get("visibility") != "private_not_for_publication"
    or worker_record.get("case_id") != expected_case
):
    raise SystemExit("actual Python worker identity metadata mismatch")
worker_command = worker_record.get("worker_command_identity")
if not isinstance(worker_command, dict) or set(worker_command) != {
    "python_realpath",
    "script_realpath",
}:
    raise SystemExit("actual Python worker command identity mismatch")
if worker_command != {
    "python_realpath": os.path.realpath(expected_worker_python),
    "script_realpath": os.path.realpath(expected_worker_script),
}:
    raise SystemExit("actual Python worker command does not match the pinned worker")
worker_identity = validate_os_process_identity(
    worker_record.get("os_process_identity")
)
if worker_identity.get("platform") != "posix":
    raise SystemExit("actual Python worker is not bound to a POSIX identity")
if provenance.get("worker_pid") != worker_identity.get("pid"):
    raise SystemExit("actual Python worker identity is not bound to run worker_pid")
fresh_process_id = os_process_identity_sha256(worker_identity)
if worker_record.get("fresh_process_id") != fresh_process_id:
    raise SystemExit("actual Python worker fresh-process hash mismatch")

backend_joints = provenance.get("backend_joint_names")
backend_frames = provenance.get("backend_frame_names")
if not isinstance(backend_joints, list) or not backend_joints or len(set(backend_joints)) != len(backend_joints):
    raise SystemExit("backend_joint_names must be a non-empty unique list")
if not isinstance(backend_frames, list) or not backend_frames or len(set(backend_frames)) != len(backend_frames):
    raise SystemExit("backend_frame_names must be a non-empty unique list")

record_keys = {"canonical_id", "backend", "scope", "backend_name", "index", "sign", "offset", "unit"}
side = run.case.hand.value

def validate_mapping(name, expected_ids, backend_names):
    records = provenance.get(name)
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise SystemExit(f"{name} coverage mismatch")
    if {record.get("canonical_id") for record in records if isinstance(record, dict)} != set(expected_ids):
        raise SystemExit(f"{name} canonical IDs mismatch")
    for record in records:
        if not isinstance(record, dict) or set(record) != record_keys:
            raise SystemExit(f"{name} has an invalid record shape")
        index = record["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(backend_names):
            raise SystemExit(f"{name} has an invalid backend index")
        if record["backend"] != "ovphysx" or record["scope"] != side:
            raise SystemExit(f"{name} backend/scope mismatch")
        if record["backend_name"] != backend_names[index]:
            raise SystemExit(f"{name} index does not match backend_name")
        if isinstance(record["sign"], bool) or float(record["sign"]) not in (-1.0, 1.0):
            raise SystemExit(f"{name} has an invalid sign")
        if not math.isfinite(float(record["offset"])):
            raise SystemExit(f"{name} has a non-finite offset")
        if not isinstance(record["unit"], str) or not record["unit"]:
            raise SystemExit(f"{name} has an invalid unit")

validate_mapping("joint_mapping", run.result.joint_names, backend_joints)
validate_mapping("frame_mapping", run.result.frame_names, backend_frames)
print(fresh_process_id)
PY
}

for case_id in "${CASE_IDS[@]}"; do
  CURRENT_CASE="$case_id"
  verify_snapshot "before-$case_id"
  now_epoch="$(date +%s)"
  gpu_elapsed=$((now_epoch - GPU_STARTED_EPOCH))
  remaining=$((GLOBAL_GPU_TIMEOUT_S - gpu_elapsed))
  usable=$((remaining - GPU_KILL_GRACE_S - GPU_DEADLINE_SAFETY_MARGIN_S))
  [[ "$usable" -gt 0 ]] || die "global two-hour GPU budget cannot fit kill grace and safety margin"
  envelope="$CASE_ENVELOPE_TIMEOUT_S"
  if [[ "$usable" -lt "$envelope" ]]; then
    envelope="$usable"
  fi

  case_dir="$RUN_ROOT/cases/$case_id"
  require_writable_directory "$RUN_ROOT/cases"
  [[ ! -e "$case_dir" && ! -L "$case_dir" ]] || die "case evidence directory already exists"
  mkdir "$case_dir"
  require_writable_directory "$case_dir"
  if ! capture_gpu_state "$case_dir/preflight.txt" "$SELECTED_GPU_UUID"; then
    die "selected GPU changed or is no longer completely idle before $case_id"
  fi

  printf 'starting %s at %s\n' "$case_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case_pid_file="$case_dir/owned-process-group.txt"
  worker_identity_file="$case_dir/private-fresh-process-identity.json"
  case_command=(
    timeout --foreground --signal=TERM --kill-after="${GPU_KILL_GRACE_S}s" "${envelope}s"
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    PYTHONPATH="$PROJECT_DIR/src"
    "$WORKER_PYTHON" -P -m wave_asset_qa.parity.runner
    --backend ovphysx
    --approved-root "$WAVEFORM_T1_ROOT"
    --worker-script "$WORKER_SCRIPT"
    --worker-python "$WORKER_PYTHON"
    --worker-timeout-s "$WORKER_TIMEOUT_S"
    --session-id "$SESSION_ID"
    --source-revision "$SOURCE_REVISION"
    --asset-root "$ASSET_ROOT"
    --manifest "$MANIFEST"
    --output-dir "$case_dir"
    --case-id "$case_id"
    --device cuda:0
  )
  (
    ulimit -f "$MAX_ARTIFACT_BLOCKS"
    exec "$WORKER_PYTHON" -P - "$case_pid_file" "${case_command[@]}"
  ) > "$case_dir/launcher.stdout.txt" 2> "$case_dir/launcher.stderr.txt" <<'PY' &
import os
from pathlib import Path
import sys

pid_path = Path(sys.argv[1])
command = sys.argv[2:]
if not command:
    raise SystemExit("owned case wrapper received no command")
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
    runner_exit_code="$?"
    set -e
    printf '%s\n' "$runner_exit_code" > "$case_dir/launcher.exit-code.txt"
    die "owned case process group was not established for $case_id"
  fi
  IFS= read -r case_pgid < "$case_pid_file"
  observed_case_pgid="$(ps -o pgid= -p "$case_owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ ! "$case_pgid" =~ ^[1-9][0-9]*$ \
    || "$case_pgid" != "$case_owner_pid" \
    || "$observed_case_pgid" != "$case_pgid" ]]; then
    # Do not signal an unverified positive PID: it may already have exited and
    # been reused.  The owned timeout wrapper provides the bounded wait.
    set +e
    wait "$case_owner_pid"
    runner_exit_code="$?"
    set -e
    printf '%s\n' "$runner_exit_code" > "$case_dir/launcher.exit-code.txt"
    die "case process-group identity is unsafe for $case_id"
  fi
  ACTIVE_CASE_PID="$case_owner_pid"
  ACTIVE_CASE_PGID="$case_pgid"
  foreign_gpu_process=false
  if ! monitor_owned_case \
    "$case_owner_pid" "$case_pgid" \
    "$case_dir/gpu-monitor.txt" "$case_dir/foreign-gpu-process.txt" \
    "$worker_identity_file" "$case_id"; then
    foreign_gpu_process=true
  fi
  set +e
  wait "$case_owner_pid"
  runner_exit_code="$?"
  set -e
  if ! terminate_owned_case "$case_owner_pid" "$case_pgid"; then
    die "owned case process group did not terminate for $case_id"
  fi
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  printf '%s\n' "$runner_exit_code" > "$case_dir/launcher.exit-code.txt"

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
  [[ "$runner_exit_code" -eq 0 ]] || die "runner returned $runner_exit_code for $case_id"

  run_file="$case_dir/$case_id.run.json"
  [[ -f "$run_file" && ! -L "$run_file" ]] || die "runner did not produce the expected case payload"
  [[ -f "$worker_identity_file" && ! -L "$worker_identity_file" ]] \
    || die "actual Python worker identity was not captured for $case_id"
  fresh_process_id="$(
    validate_collected_run "$run_file" "$case_id" "$worker_identity_file"
  )"
  [[ "$fresh_process_id" =~ ^[0-9a-f]{64}$ ]] \
    || die "actual Python worker fresh-process ID is invalid for $case_id"
  [[ -z "${FRESH_PROCESS_IDS[$fresh_process_id]+present}" ]] \
    || die "actual Python worker identity was reused across canonical cases"
  FRESH_PROCESS_IDS["$fresh_process_id"]="$case_id"
  COMPLETED_FRESH_PROCESSES=$((COMPLETED_FRESH_PROCESSES + 1))
  check_run_budget
  write_directory_inventory "$case_dir"
  COMPLETED_CASES=$((COMPLETED_CASES + 1))
  check_root_budget
done

[[ "$COMPLETED_CASES" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]] \
  || die "remote matrix did not complete all 16 OVPhysX cases"
[[ "$COMPLETED_FRESH_PROCESSES" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" \
  && "${#FRESH_PROCESS_IDS[@]}" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]] \
  || die "remote matrix did not prove 16 unique actual Python worker processes"
verify_snapshot "post-matrix"
check_run_budget
RUN_STATUS="completed"
SCIENTIFIC_VERDICT="pending_local_compare"
CURRENT_CASE=""
printf 'collected all 16 OVPhysX cases; scientific verdict awaits local compare at %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
