#!/usr/bin/env bash

set -euo pipefail
umask 077

# Server 6 only.  One private-plan case is executed per fresh process.  This
# launcher never selects a busy GPU, overwrites evidence, signals a foreign
# process, mounts Ceph, invokes Kit/rendering, or evaluates the scientific result.
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
readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))
readonly MAX_RUN_BYTES=$((500 * 1024 * 1024))
readonly MAX_ARTIFACT_BYTES=$((500 * 1024 * 1024))
readonly WORKER_TIMEOUT_S=90
readonly GLOBAL_GPU_TIMEOUT_S=$((30 * 60))

PROJECT_DIR_INPUT=""
PRIVATE_PLAN_INPUT=""
PRIVATE_PLAN_SHA256=""
PRIVATE_PLAN_FILE_SHA256=""
PUBLIC_PROTOCOL_FILE_SHA256=""
PUBLIC_PROTOCOL_CANONICAL_SHA256=""
IMPLEMENTATION_REVISION=""
IMPLEMENTATION_TREE=""
RUN_ID=""
GPU_INDEX=""
SESSION_ID=""
SOURCE_REVISION=""
PROJECT_DIR=""
PRIVATE_PLAN=""
RUN_ROOT=""
INITIALIZING_ROOT=""
INITIALIZING_DEVICE_INODE=""
RUN_STATUS="error"
SCIENTIFIC_VERDICT="not_evaluated"
SOURCE_TREE=""
ARCHIVE_SHA256=""
SNAPSHOT_SHA256=""
SELECTED_GPU_UUID=""
SELECTED_GPU_NAME=""
SELECTED_DRIVER_VERSION=""
CURRENT_CASE=""
COMPLETED_CASES=0
ACTIVE_CASE_PID=""
ACTIVE_CASE_PGID=""
LAUNCH_STARTED_EPOCH="$(date +%s)"
GPU_STARTED_EPOCH=""
LAST_GPU_UUID=""
LAST_GPU_NAME=""
LAST_GPU_DRIVER=""
LAST_GPU_MEMORY=""
LAST_GPU_UTILIZATION=""
LAST_GPU_LINE=""
LAST_GPU_PROCESSES=""

exec 3>&1 4>&2

usage() {
  printf '%s\n' \
    "usage: run_ovphysx_freeze_b_remote.sh --project-dir PATH --private-plan PATH" \
    "       --private-plan-sha256 SHA256 --run-id ID --gpu-index 0..7" \
    "       --session-id ID --source-revision 40_HEX_COMMIT" >&4
}

die() {
  printf 'Refusing to continue: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

read_gpu_state() {
  local expected_uuid="${1:-}" gpu_line processes index
  gpu_line="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"
  [[ "$gpu_line" != *$'\n'* ]] || return 1
  index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$gpu_line")"
  LAST_GPU_UUID="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
  LAST_GPU_NAME="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
  LAST_GPU_DRIVER="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
  LAST_GPU_MEMORY="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}' <<< "$gpu_line")"
  LAST_GPU_UTILIZATION="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}' <<< "$gpu_line")"
  processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits)"
  [[ "$index" == "$GPU_INDEX" && "$LAST_GPU_UUID" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || return 1
  [[ -z "$expected_uuid" || "$LAST_GPU_UUID" == "$expected_uuid" ]] || return 1
  [[ "$LAST_GPU_NAME" == *"$EXPECTED_GPU_NAME_FRAGMENT"* ]] || return 1
  [[ "$LAST_GPU_DRIVER" == "$EXPECTED_DRIVER_VERSION" ]] || return 1
  [[ "$LAST_GPU_MEMORY" =~ ^[0-9]+$ ]] || return 1
  (( LAST_GPU_MEMORY <= MAX_IDLE_MEMORY_MIB )) || return 1
  [[ "$LAST_GPU_UTILIZATION" == 0 && -z "$processes" ]] || return 1
  LAST_GPU_LINE="$gpu_line"
  LAST_GPU_PROCESSES="$processes"
}

safe_identifier() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9_.-]{0,127}$ ]]
}

refuse_symlink() {
  [[ ! -L "$1" ]] || die "managed path is a symbolic link: $1"
}

require_real_directory() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || die "$label directory is missing: $path"
  refuse_symlink "$path"
  [[ "$(readlink -f -- "$path")" == "$path" ]] \
    || die "$label directory is not canonical: $path"
  [[ "$path" == "$GATE0_ROOT" || "$path" == "$GATE0_ROOT/"* ]] \
    || die "$label directory escapes the approved root"
  [[ "$(stat -c '%U' "$path")" == "$EXPECTED_USER" ]] \
    || die "$label directory owner changed"
  [[ "$(findmnt -n -o TARGET -T "$path")" == "/data" ]] \
    || die "$label directory left /data"
  [[ "$(findmnt -n -o FSTYPE -T "$path")" == "xfs" ]] \
    || die "$label directory is not on XFS"
}

require_writable_directory() {
  local path="$1"
  local link target
  require_real_directory "$path" "writable"
  [[ -w "$path" ]] || die "managed directory is not writable: $path"
  while IFS= read -r -d '' link; do
    target="$(readlink -f -- "$link")" || die "dangling managed symlink: $link"
    [[ "$target" == "$path" || "$target" == "$path/"* ]] \
      || die "managed symlink escapes its directory: $link"
  done < <(find "$path" -type l -print0)
}

cleanup_initializing_root() {
  local expected
  [[ -n "$INITIALIZING_ROOT" ]] || return 0
  expected="$GATE0_ROOT/results/.${RUN_ID}.initializing.$$"
  [[ "$INITIALIZING_ROOT" == "$expected" && -d "$INITIALIZING_ROOT" \
    && ! -L "$INITIALIZING_ROOT" ]] || return 1
  rmdir -- "$INITIALIZING_ROOT/launcher" 2>/dev/null || true
  rmdir -- "$INITIALIZING_ROOT/cases" 2>/dev/null || true
  rmdir -- "$INITIALIZING_ROOT" 2>/dev/null || true
  if [[ ! -e "$INITIALIZING_ROOT" && ! -L "$INITIALIZING_ROOT" ]]; then
    INITIALIZING_ROOT=""
    INITIALIZING_DEVICE_INODE=""
    return 0
  fi
  return 1
}

adopt_promoted_initializing_root() {
  local expected candidate observed
  [[ -z "$RUN_ROOT" && -n "$INITIALIZING_ROOT" \
    && "$INITIALIZING_DEVICE_INODE" =~ ^[0-9]+:[0-9]+$ ]] || return 0
  expected="$GATE0_ROOT/results/.${RUN_ID}.initializing.$$"
  candidate="$GATE0_ROOT/results/$RUN_ID"
  [[ "$INITIALIZING_ROOT" == "$expected" \
    && ! -e "$INITIALIZING_ROOT" && ! -L "$INITIALIZING_ROOT" \
    && -d "$candidate" && ! -L "$candidate" ]] || return 0
  observed="$(stat -c '%d:%i' -- "$candidate" 2>/dev/null)" || return 0
  if [[ "$observed" == "$INITIALIZING_DEVICE_INODE" ]]; then
    RUN_ROOT="$candidate"
    INITIALIZING_ROOT=""
    INITIALIZING_DEVICE_INODE=""
  fi
}

case_process_running() {
  local owner_pid="$1"
  local state
  kill -0 "$owner_pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ -n "$state" && "$state" != Z* ]]
}

terminate_owned_case() {
  local owner_pid="${1:-}"
  local owner_pgid="${2:-}"
  local observed_pgid
  [[ "$owner_pid" =~ ^[1-9][0-9]*$ && "$owner_pgid" == "$owner_pid" ]] || return 0
  observed_pgid="$(ps -o pgid= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ "$observed_pgid" == "$owner_pgid" ]] || return 0
  kill -TERM -- "-$owner_pgid" 2>/dev/null || true
  for _attempt in {1..30}; do
    kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$owner_pgid" 2>/dev/null || true
}

write_top_level_inventory() {
  local inventory temporary oversized
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  inventory="$RUN_ROOT/launcher/evidence.sha256"
  [[ ! -e "$inventory" ]] || return 1
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || return 1
  temporary="$inventory.tmp.$$"
  (
    cd "$RUN_ROOT"
    while IFS= read -r -d '' path; do
      sha256sum "$path"
    done < <(
      find . -type f \
        ! -path './launcher/evidence.sha256' \
        ! -path './launcher/evidence.sha256.tmp.*' -print0 | sort -z
    )
  ) > "$temporary"
  mv -- "$temporary" "$inventory"
}

write_final_status() {
  local exit_code="$1"
  local ended elapsed root_bytes status temporary
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] || return 0
  ended="$(date +%s)"
  elapsed=$((ended - LAUNCH_STARTED_EPOCH))
  root_bytes="$(du -sb "$GATE0_ROOT" 2>/dev/null | awk '{print $1}')"
  status="$RUN_STATUS"
  [[ "$exit_code" -eq 0 ]] || status="error"
  temporary="$RUN_ROOT/launcher/status.json.tmp.$$"
  STATUS_VALUE="$status" STATUS_EXIT="$exit_code" STATUS_ELAPSED="$elapsed" \
  STATUS_ROOT_BYTES="${root_bytes:-0}" \
  RUN_ID="$RUN_ID" SESSION_ID="$SESSION_ID" SOURCE_REVISION="$SOURCE_REVISION" \
  SOURCE_TREE="$SOURCE_TREE" PRIVATE_PLAN_SHA256="$PRIVATE_PLAN_SHA256" \
  GPU_INDEX="$GPU_INDEX" SELECTED_GPU_UUID="$SELECTED_GPU_UUID" \
  COMPLETED_CASES="$COMPLETED_CASES" CURRENT_CASE="$CURRENT_CASE" \
  "$WORKER_PYTHON" -P -S - "$temporary" <<'PY'
import json
import os
from pathlib import Path
import sys

last_case = os.environ["CURRENT_CASE"] or None
payload = {
    "schema_version": 1,
    "status": os.environ["STATUS_VALUE"],
    "scientific_verdict": "pending_local_validation" if os.environ["STATUS_VALUE"] == "collected" else "not_evaluated",
    "formal_gate0_status_unchanged": "DIVERGENT",
    "formal_gate0_pass_ready_unchanged": False,
    "exit_code": int(os.environ["STATUS_EXIT"]),
    "run_id": os.environ["RUN_ID"],
    "session_id": os.environ["SESSION_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_tree": os.environ["SOURCE_TREE"],
    "private_plan_sha256": os.environ["PRIVATE_PLAN_SHA256"],
    "selected_gpu_index": int(os.environ["GPU_INDEX"]),
    "selected_gpu_uuid": os.environ["SELECTED_GPU_UUID"] or None,
    "completed_case_count": int(os.environ["COMPLETED_CASES"]),
    "last_case_id": last_case,
    "elapsed_s": int(os.environ["STATUS_ELAPSED"]),
    "approved_root_bytes": int(os.environ["STATUS_ROOT_BYTES"]),
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY
  mv -- "$temporary" "$RUN_ROOT/launcher/status.json"
}

finish() {
  local exit_code="$?"
  trap - EXIT
  set +e
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  adopt_promoted_initializing_root
  cleanup_initializing_root
  write_final_status "$exit_code"
  if ! write_top_level_inventory; then
    exit_code=2
    RUN_STATUS="error"
    write_final_status "$exit_code"
    write_top_level_inventory || true
  fi
  exec 1>&3 2>&4
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'Freeze B remote collection completed; local validation is still required; evidence retained under results/%s\n' "$RUN_ID"
  elif [[ -n "$RUN_ROOT" ]]; then
    printf 'Freeze B remote collection stopped; evidence retained under results/%s\n' "$RUN_ID" >&2
  fi
  exit "$exit_code"
}

signal_exit() {
  local name="$1"
  local code="$2"
  printf 'launcher received %s while case=%s\n' "$name" "${CURRENT_CASE:-none}" >&2
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"
  exit "$code"
}

trap finish EXIT
trap 'signal_exit HUP 129' HUP
trap 'signal_exit INT 130' INT
trap 'signal_exit TERM 143' TERM

seen_project=false
seen_plan=false
seen_plan_hash=false
seen_run=false
seen_gpu=false
seen_session=false
seen_revision=false
while [[ "$#" -gt 0 ]]; do
  [[ "$#" -ge 2 ]] || { usage; die "every option requires one value"; }
  case "$1" in
    --project-dir) [[ "$seen_project" == false ]] || die "duplicate --project-dir"; PROJECT_DIR_INPUT="$2"; seen_project=true ;;
    --private-plan) [[ "$seen_plan" == false ]] || die "duplicate --private-plan"; PRIVATE_PLAN_INPUT="$2"; seen_plan=true ;;
    --private-plan-sha256) [[ "$seen_plan_hash" == false ]] || die "duplicate --private-plan-sha256"; PRIVATE_PLAN_SHA256="$2"; seen_plan_hash=true ;;
    --run-id) [[ "$seen_run" == false ]] || die "duplicate --run-id"; RUN_ID="$2"; seen_run=true ;;
    --gpu-index) [[ "$seen_gpu" == false ]] || die "duplicate --gpu-index"; GPU_INDEX="$2"; seen_gpu=true ;;
    --session-id) [[ "$seen_session" == false ]] || die "duplicate --session-id"; SESSION_ID="$2"; seen_session=true ;;
    --source-revision) [[ "$seen_revision" == false ]] || die "duplicate --source-revision"; SOURCE_REVISION="$2"; seen_revision=true ;;
    *) usage; die "unsupported option: $1" ;;
  esac
  shift 2
done
[[ "$seen_project" == true && "$seen_plan" == true && "$seen_plan_hash" == true \
  && "$seen_run" == true && "$seen_gpu" == true && "$seen_session" == true \
  && "$seen_revision" == true ]] || { usage; die "all launcher options are required"; }
safe_identifier "$RUN_ID" || die "run ID must be a safe lowercase identifier"
safe_identifier "$SESSION_ID" || die "session ID must be a safe lowercase identifier"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "source revision must be lowercase 40-hex"
[[ "$PRIVATE_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "private plan hash must be lowercase SHA-256"
[[ "$GPU_INDEX" =~ ^[0-7]$ ]] || die "GPU index must be 0 through 7"

for command_name in awk date du env find findmnt flock git hostname id mkdir mv \
  nvidia-smi ps readlink rmdir sha256sum sleep sort stat timeout; do
  require_command "$command_name"
done

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || die "expected Server 6 hostname $EXPECTED_HOSTNAME"
[[ "$(id -un)" == "$EXPECTED_USER" ]] || die "expected Server 6 user $EXPECTED_USER"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "$EXPECTED_OS_ID" && "${VERSION_ID:-}" == "$EXPECTED_OS_VERSION" ]] \
  || die "expected Ubuntu 22.04"
[[ -d "$GATE0_ROOT" ]] || die "approved root is missing"
refuse_symlink "$GATE0_ROOT"
[[ "$(readlink -f -- "$GATE0_ROOT")" == "$GATE0_ROOT" ]] || die "approved root is not canonical"
[[ "$(stat -c '%U' "$GATE0_ROOT")" == "$EXPECTED_USER" ]] || die "approved root owner changed"
[[ "$(findmnt -n -o TARGET -T "$GATE0_ROOT")" == "/data" ]] || die "approved root left /data"
[[ "$(findmnt -n -o FSTYPE -T "$GATE0_ROOT")" == "xfs" ]] || die "approved root is not XFS"
[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2 && "$PROJECT_DIR_INPUT" != /mnt/ceph2/* \
  && "$PRIVATE_PLAN_INPUT" != /mnt/ceph2 && "$PRIVATE_PLAN_INPUT" != /mnt/ceph2/* ]] \
  || die "/mnt/ceph2 is outside the approved boundary"

# Do not import project Python, create a lock, or write any managed path until
# the requested GPU has passed the strict idle predicate.
if ! read_gpu_state; then
  die "selected GPU is not strictly idle (memory<=32 MiB, utilization=0, no compute process)"
fi
SELECTED_GPU_UUID="$LAST_GPU_UUID"
SELECTED_GPU_NAME="$LAST_GPU_NAME"
SELECTED_DRIVER_VERSION="$LAST_GPU_DRIVER"
export PYTHONDONTWRITEBYTECODE=1

PROJECT_DIR="$(readlink -f -- "$PROJECT_DIR_INPUT")"
PRIVATE_PLAN="$(readlink -f -- "$PRIVATE_PLAN_INPUT")"
[[ "$PROJECT_DIR" == "$PROJECT_DIR_INPUT" && "$PROJECT_DIR" == "$GATE0_ROOT/project/"* ]] \
  || die "project directory is not an approved canonical snapshot path"
[[ "$PRIVATE_PLAN" == "$PRIVATE_PLAN_INPUT" && "$PRIVATE_PLAN" == "$GATE0_ROOT/"* ]] \
  || die "private plan is not an approved canonical path"
[[ "$PRIVATE_PLAN" == "$GATE0_ROOT/inputs/freeze-b/$PRIVATE_PLAN_SHA256.json" ]] \
  || die "private plan must use its exact hash-addressed input path"
[[ -d "$PROJECT_DIR" && -f "$PRIVATE_PLAN" ]] || die "project or private plan is missing"
refuse_symlink "$PROJECT_DIR"
refuse_symlink "$PRIVATE_PLAN"
[[ "$(stat -c '%U' "$PRIVATE_PLAN")" == "$EXPECTED_USER" ]] \
  || die "private plan owner must be exampleuser"
[[ -z "$(find "$PRIVATE_PLAN" -perm /077 -print -quit)" ]] \
  || die "private plan must grant no group/other permissions"
[[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] || die "project snapshot contains a symbolic link"
PRIVATE_PLAN_FILE_SHA256="$(sha256sum -- "$PRIVATE_PLAN" | awk '{print $1}')"
[[ "$PRIVATE_PLAN_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "private plan file SHA-256 is malformed"

read_marker() {
  local path="$1"
  local destination="$2"
  local -a lines=()
  [[ -f "$path" ]] || die "missing source marker: $path"
  refuse_symlink "$path"
  mapfile -t lines < "$path"
  [[ "${#lines[@]}" -eq 1 && -n "${lines[0]}" ]] || die "malformed source marker"
  printf -v "$destination" '%s' "${lines[0]}"
}

MARKER_REVISION=""
read_marker "$PROJECT_DIR/.source-revision" MARKER_REVISION
read_marker "$PROJECT_DIR/.source-tree" SOURCE_TREE
read_marker "$PROJECT_DIR/.archive-sha256" ARCHIVE_SHA256
read_marker "$PROJECT_DIR/.snapshot-sha256" SNAPSHOT_SHA256
[[ "$MARKER_REVISION" == "$SOURCE_REVISION" ]] || die "source revision marker mismatch"
[[ "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ && "$PROJECT_DIR" == "$GATE0_ROOT/project/$SOURCE_TREE" ]] \
  || die "source tree identity mismatch"
[[ "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ && "$SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "source archive/snapshot marker is malformed"

readonly ASSET_ROOT="$GATE0_ROOT/assets"
readonly ISAACLAB_ROOT="$GATE0_ROOT/IsaacLab"
readonly DOWNLOADS_ROOT="$GATE0_ROOT/downloads"
readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"
readonly WORKER_PYTHON="$GATE0_ROOT/env/bin/python"
readonly GATE0_MANIFEST="$PROJECT_DIR/configs/parity/gate0.json"
readonly PUBLIC_PROTOCOL="$PROJECT_DIR/configs/parity/ovphysx_legacy_friction_freeze_b.json"
readonly WORKER_SCRIPT="$PROJECT_DIR/scripts/probe_ovphysx_freeze_b.py"
for directory in "$GATE0_ROOT/project" "$PROJECT_DIR" "$ASSET_ROOT" \
  "$ISAACLAB_ROOT" "$DOWNLOADS_ROOT" "$GATE0_ROOT/inputs" \
  "$GATE0_ROOT/inputs/freeze-b"; do
  require_real_directory "$directory" "managed"
done
for writable in "$GATE0_ROOT/results" "$GATE0_ROOT/cache" \
  "$GATE0_ROOT/cache/cuda" "$GATE0_ROOT/cache/pip" \
  "$GATE0_ROOT/cache/pycache" "$GATE0_ROOT/cache/torch_extensions" \
  "$GATE0_ROOT/cache/triton" "$GATE0_ROOT/cache/uv" "$GATE0_ROOT/cache/warp" \
  "$GATE0_ROOT/cache/xdg" "$GATE0_ROOT/config" "$GATE0_ROOT/data" \
  "$GATE0_ROOT/state" "$GATE0_ROOT/tmp"; do
  require_writable_directory "$writable"
done
[[ -x "$WORKER_PYTHON" ]] || die "pinned Python environment is missing"
[[ "$(readlink -f -- "$WORKER_PYTHON")" == "$GATE0_ROOT/env/"* \
  || "$(readlink -f -- "$WORKER_PYTHON")" == "$GATE0_ROOT/toolchain/python/"* ]] \
  || die "worker Python escapes the approved root"
for file in "$SOURCE_ARCHIVE" "$GATE0_MANIFEST" "$PUBLIC_PROTOCOL" "$WORKER_SCRIPT"; do
  [[ -f "$file" ]] || die "required input is missing: $file"
  refuse_symlink "$file"
done
[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_ovphysx_freeze_b_remote.sh" ]] \
  || die "launcher is not executing from the immutable selected snapshot"

check_root_budget() {
  local bytes
  bytes="$(du -sb "$GATE0_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_ROOT_BYTES" ]] || die "approved root exceeds 30 GiB"
}

check_run_budget() {
  local bytes oversized
  bytes="$(du -sb "$RUN_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_RUN_BYTES" ]] || die "run evidence exceeds 500 MiB"
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || die "one retained artifact exceeds 500 MiB"
}

verify_plan() {
  [[ -f "$PRIVATE_PLAN" && ! -L "$PRIVATE_PLAN" ]] || die "private plan disappeared or became linked"
  [[ "$(sha256sum -- "$PRIVATE_PLAN" | awk '{print $1}')" == "$PRIVATE_PLAN_FILE_SHA256" ]] \
    || die "private plan file changed"
}

verify_snapshot() {
  local checkpoint="$1"
  local actual writable
  [[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] || die "snapshot gained a symlink"
  writable="$(find "$PROJECT_DIR" -perm /222 -print -quit)"
  [[ -z "$writable" ]] || die "snapshot is writable before $checkpoint: $writable"
  actual="$(PROJECT_SNAPSHOT="$PROJECT_DIR" "$WORKER_PYTHON" -P -S - <<'PY'
import hashlib
import os
from pathlib import Path
import stat

root = Path(os.environ["PROJECT_SNAPSHOT"]).resolve(strict=True)
digest = hashlib.sha256(b"waveqa-source-snapshot-v1\0")
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    if relative == ".snapshot-sha256":
        continue
    if path.is_symlink():
        raise SystemExit("snapshot contains a symlink")
    mode = path.stat(follow_symlinks=False).st_mode
    encoded = relative.encode("utf-8")
    if stat.S_ISDIR(mode):
        kind, executable = b"D", 0
    elif stat.S_ISREG(mode):
        kind, executable = b"F", int(bool(mode & 0o111))
    else:
        raise SystemExit("snapshot contains a non-regular entry")
    digest.update(kind)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(executable.to_bytes(1, "big"))
    if kind == b"F":
        digest.update(path.stat(follow_symlinks=False).st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
print(digest.hexdigest())
PY
)"
  [[ "$actual" == "$SNAPSHOT_SHA256" ]] || die "snapshot hash changed before $checkpoint"
  if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]]; then
    printf '%s checkpoint=%s snapshot_sha256=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checkpoint" "$actual" \
      >> "$RUN_ROOT/launcher/snapshot-verifications.txt"
  fi
}

verify_repository() {
  local path="$1" expected_url="$2" allowed_untracked="${3:-}" line
  [[ -d "$path/.git" ]] || die "managed checkout is not Git: $path"
  [[ "$(git -C "$path" remote get-url origin)" == "$expected_url" ]] \
    || die "managed checkout origin changed"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ -n "$allowed_untracked" && "$line" == "?? $allowed_untracked" ]] && continue
    die "managed checkout is dirty: $line"
  done < <(git -C "$path" status --porcelain --untracked-files=all)
}

verify_pinned_repositories() {
  local actual_asset_hash
  [[ "$(sha256sum -- "$SOURCE_ARCHIVE" | awk '{print $1}')" == "$ARCHIVE_SHA256" ]] \
    || die "source archive hash mismatch"
  verify_repository "$ASSET_ROOT" "$ASSET_URL"
  [[ "$(git -C "$ASSET_ROOT" rev-parse HEAD)" == "$ASSET_COMMIT" ]] \
    || die "asset commit mismatch"
  [[ "$(git -C "$ASSET_ROOT" rev-parse 'HEAD:wave_01')" == "$ASSET_GIT_TREE" ]] \
    || die "asset tree mismatch"
  verify_repository "$ISAACLAB_ROOT" "$ISAACLAB_URL" "uv.lock"
  [[ "$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)" == "$ISAACLAB_COMMIT" ]] \
    || die "Isaac Lab commit mismatch"
  printf '%s  %s\n' "$UV_LOCK_SHA256" "$ISAACLAB_ROOT/uv.lock" \
    | sha256sum --check --status || die "Isaac Lab lock hash mismatch"
  actual_asset_hash="$(ASSET_TREE="$ASSET_ROOT/wave_01" "$WORKER_PYTHON" -P -S - <<'PY'
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
  [[ "$actual_asset_hash" == "$ASSET_LF_SHA256" ]] \
    || die "canonical LF asset hash mismatch"
}

verify_pinned_repositories
check_root_budget

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
    raise SystemExit("unexpected Python version")
for package, version in expected.items():
    if metadata.version(package) != version:
        raise SystemExit(f"unexpected {package} version")
from isaaclab.utils.version import has_kit
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: F401
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: F401
import ovphysx  # noqa: F401
if has_kit():
    raise SystemExit("Kit is present")
forbidden = [
    name for name in sys.modules
    if name == "isaacsim" or name.startswith("isaacsim.")
    or name == "omni.kit" or name.startswith("omni.kit.")
    or name == "omni.renderer" or name.startswith("omni.renderer.")
]
if forbidden:
    raise SystemExit("forbidden Kit/render module imported")
print("python=3.12.14")
for package, version in expected.items():
    print(f"{package}={version}")
print("kit=false")
print("renderer=false")
print("camera=false")
PY
)"

PUBLIC_PROTOCOL_FILE_SHA256="$(sha256sum -- "$PUBLIC_PROTOCOL" | awk '{print $1}')"
protocol_identity="$(PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -S - \
  "$PUBLIC_PROTOCOL" "$PRIVATE_PLAN" "$PRIVATE_PLAN_SHA256" <<'PY'
from pathlib import Path
import sys

from wave_asset_qa.parity.sensitivity import (
    canonical_json_sha256,
    load_private_plan,
    load_public_protocol,
)

public_path, private_path, expected_private_hash = sys.argv[1:]
public = load_public_protocol(public_path)
private = load_private_plan(
    private_path,
    public_protocol=public,
    expected_sha256=expected_private_hash,
)
print("|".join((
    canonical_json_sha256(public),
    str(private["inputs"]["freeze_b_source_revision"]),
    str(private["inputs"]["freeze_b_source_tree"]),
)))
PY
)"
IFS='|' read -r PUBLIC_PROTOCOL_CANONICAL_SHA256 IMPLEMENTATION_REVISION IMPLEMENTATION_TREE extra \
  <<< "$protocol_identity"
[[ -z "${extra:-}" && "$PUBLIC_PROTOCOL_FILE_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$PUBLIC_PROTOCOL_CANONICAL_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$IMPLEMENTATION_REVISION" =~ ^[0-9a-f]{40}$ \
  && "$IMPLEMENTATION_TREE" =~ ^[0-9a-f]{40}$ ]] \
  || die "public/private protocol identity bridge is malformed"
[[ "$IMPLEMENTATION_REVISION" != "$SOURCE_REVISION" \
  && "$IMPLEMENTATION_TREE" != "$SOURCE_TREE" ]] \
  || die "implementation A and preregistration deployment B must remain distinct"

mapfile -t CASE_RECORDS < <(
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -S - \
    "$PRIVATE_PLAN" "$PRIVATE_PLAN_SHA256" "$PUBLIC_PROTOCOL" \
    "$GATE0_MANIFEST" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

from wave_asset_qa.parity.scenarios import load_manifest, manifest_sha256
from wave_asset_qa.parity.sensitivity import load_private_plan, load_public_protocol

plan_path, expected_hash, public_path, manifest_path = sys.argv[1:]
public = load_public_protocol(public_path)
plan = load_private_plan(
    plan_path,
    public_protocol=public,
    expected_sha256=expected_hash,
)
if plan["protocol_id"] != "ovphysx-legacy-joint-friction-freeze-b-v1":
    raise SystemExit("private plan protocol mismatch")
inputs = plan.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("private plan inputs missing")
manifest_file_hash = sha256(Path(manifest_path).read_bytes()).hexdigest()
manifest = load_manifest(manifest_path)
expected_inputs = {
    "gate0_manifest_file_sha256": manifest_file_hash,
    "gate0_manifest_semantic_sha256": manifest_sha256(manifest),
    "asset_commit": manifest.provenance.commit,
    "asset_git_tree": manifest.provenance.asset_git_tree,
    "canonical_lf_asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
}
for key, expected in expected_inputs.items():
    if inputs.get(key) != expected:
        raise SystemExit(f"private plan input drifted: {key}")
cases = plan.get("ovphysx_cases")
order = plan.get("run_order")
if not isinstance(cases, list) or len(cases) != 16 or not isinstance(order, list):
    raise SystemExit("private plan must contain 16 OVPhysX cases")
by_id = {case.get("experiment_case_id"): case for case in cases if isinstance(case, dict)}
if len(by_id) != 16 or len(order) != 16 or len(set(order)) != 16 or set(order) != set(by_id):
    raise SystemExit("private plan run order is not exact")
for case_id in order:
    case = by_id[case_id]
    required = {"experiment_case_id", "canonical_case_id", "backend", "role", "hand", "timestep_variant", "dt_s", "repeat_index"}
    if set(case) != required or case["backend"] != "ovphysx" or case["role"] not in {"sham", "zero"}:
        raise SystemExit("private plan case contract mismatch")
    print("|".join(str(case[key]) for key in (
        "experiment_case_id", "canonical_case_id", "role", "hand",
        "timestep_variant", "dt_s", "repeat_index",
    )))
PY
)
[[ "${#CASE_RECORDS[@]}" -eq 16 ]] || die "private plan did not yield 16 ordered cases"
for record in "${CASE_RECORDS[@]}"; do
  IFS='|' read -r case_id canonical_id role hand variant dt repeat extra <<< "$record"
  [[ -z "${extra:-}" ]] || die "case record has extra fields"
  safe_identifier "$case_id" || die "unsafe experiment case ID"
  [[ "$case_id" == "freeze_b.ovphysx.$hand.small_step.$variant.$role.r0$repeat" ]] \
    || die "experiment case identity mismatch"
  [[ "$canonical_id" == "ovphysx.$hand.small_step.$variant.r0$repeat" ]] \
    || die "canonical case identity mismatch"
  [[ "$hand" == left || "$hand" == right ]] || die "invalid case hand"
  [[ "$variant" == base || "$variant" == halved ]] || die "invalid timestep variant"
  [[ "$role" == sham || "$role" == zero ]] || die "invalid case role"
  [[ "$repeat" == 1 || "$repeat" == 2 ]] || die "invalid repeat"
  [[ ( "$variant" == base && "$dt" == 0.002 ) || ( "$variant" == halved && "$dt" == 0.001 ) ]] \
    || die "case timestep mismatch"
done

capture_gpu_state() {
  local destination="$1" expected_uuid="${2:-}" temporary
  read_gpu_state "$expected_uuid" || return 1
  temporary="$destination.tmp.$$"
  {
    printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu=%s\ncompute_processes_begin\n%s\ncompute_processes_end\n' \
      "$LAST_GPU_LINE" "$LAST_GPU_PROCESSES"
  } > "$temporary"
  mv -- "$temporary" "$destination"
}

trim_field() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

monitor_owned_case() {
  local owner_pid="$1" owner_pgid="$2" destination="$3" foreign_marker="$4"
  local gpu_line processes uuid name driver row process_uuid process_pid memory process_pgid reason
  : > "$destination"
  while case_process_running "$owner_pid"; do
    gpu_line="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits)"
    uuid="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
    name="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
    driver="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
    printf 'recorded_at_utc=%s\ngpu=%s\nowned_pid=%s\nowned_pgid=%s\nprocesses=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu_line" "$owner_pid" "$owner_pgid" "$processes" >> "$destination"
    reason=""
    [[ "$uuid" == "$SELECTED_GPU_UUID" && "$name" == *"$EXPECTED_GPU_NAME_FRAGMENT"* \
      && "$driver" == "$EXPECTED_DRIVER_VERSION" ]] || reason="selected GPU identity changed"
    while IFS= read -r row; do
      [[ -n "$row" ]] || continue
      IFS=',' read -r process_uuid process_pid memory <<< "$row"
      process_uuid="$(trim_field "$process_uuid")"
      process_pid="$(trim_field "$process_pid")"
      memory="$(trim_field "$memory")"
      if [[ "$process_uuid" != "$SELECTED_GPU_UUID" || ! "$process_pid" =~ ^[1-9][0-9]*$ ]]; then
        reason="unclassifiable GPU process"
        break
      fi
      [[ -e "/proc/$process_pid" ]] || continue
      process_pgid="$(ps -o pgid= -p "$process_pid" 2>/dev/null | awk '{$1=$1; print}')"
      if [[ "$process_pgid" != "$owner_pgid" ]]; then
        reason="foreign GPU process pid=$process_pid pgid=${process_pgid:-unknown} memory_mib=$memory"
        break
      fi
    done <<< "$processes"
    if [[ -n "$reason" ]]; then
      printf '%s\n' "$reason" > "$foreign_marker"
      terminate_owned_case "$owner_pid" "$owner_pgid"
      return 1
    fi
    sleep 1
  done
}

validate_worker_stdout() {
  local path="$1" expected_case="$2" expected_output="$3"
  "$WORKER_PYTHON" -P -S - "$path" "$expected_case" "$expected_output" <<'PY'
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
    raise ValueError(f"non-finite JSON: {token}")

path, expected_case, expected_output = sys.argv[1:]
values = []
for line in Path(path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        values.append(json.loads(line, object_pairs_hook=no_duplicates, parse_constant=reject_constant))
    except json.JSONDecodeError:
        continue
if len(values) != 1:
    raise SystemExit("worker stdout must contain exactly one valid JSON value")
value = values[0]
if not isinstance(value, dict) or value.get("status") != "completed" \
        or value.get("case_id") != expected_case or value.get("output") != expected_output:
    raise SystemExit("worker stdout summary mismatch")
if value.get("requested_steps") != value.get("completed_steps"):
    raise SystemExit("worker stdout reports incomplete execution")
PY
}

validate_case_payload() {
  local payload="$1" case_id="$2" canonical_id="$3" role="$4" hand="$5" variant="$6" dt="$7" repeat="$8"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P - \
    "$payload" "$case_id" "$canonical_id" "$role" "$hand" "$variant" "$dt" "$repeat" \
    "$PRIVATE_PLAN_SHA256" "$SESSION_ID" "$SOURCE_REVISION" "$SOURCE_TREE" \
    "$ASSET_LF_SHA256" "$PUBLIC_PROTOCOL_FILE_SHA256" \
    "$PUBLIC_PROTOCOL_CANONICAL_SHA256" "$IMPLEMENTATION_REVISION" \
    "$IMPLEMENTATION_TREE" "$SELECTED_GPU_UUID" "$SELECTED_GPU_NAME" \
    "$SELECTED_DRIVER_VERSION" "$EXPECTED_HOSTNAME" <<'PY'
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

from wave_asset_qa.parity.runner import adapter_run_result_from_dict

(path, case_id, canonical_id, role, hand, variant, dt_text, repeat_text,
 plan_hash, session_id, source_revision, source_tree, asset_hash,
 public_file_hash, public_canonical_hash, implementation_revision,
 implementation_tree, selected_gpu_uuid, selected_gpu_name,
 selected_driver_version, expected_hostname) = sys.argv[1:]
raw = json.loads(Path(path).read_text(encoding="utf-8"))
result = adapter_run_result_from_dict(raw)
dt = float(dt_text)
if not result.completed or result.backend != "ovphysx" or result.scenario_id != "small_step":
    raise SystemExit("worker result is not a completed OVPhysX small_step")
if not math.isclose(result.dt, dt, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("worker result dt mismatch")
p = result.provenance
expected = {
    "experiment_case_id": case_id,
    "canonical_case_id": canonical_id,
    "freeze_b_role": role,
    "hand": hand,
    "timestep_variant": variant,
    "repeat_index": int(repeat_text),
    "private_plan_sha256": plan_hash,
    "session_id": session_id,
    "source_revision": source_revision,
    "source_tree": source_tree,
    "deployment_source_revision": source_revision,
    "deployment_source_tree": source_tree,
    "implementation_source_revision": implementation_revision,
    "implementation_source_tree": implementation_tree,
    "asset_tree_sha256": asset_hash,
    "public_protocol_file_sha256": public_file_hash,
    "public_protocol_canonical_sha256": public_canonical_hash,
    "gpu_uuid": selected_gpu_uuid,
    "gpu_name": selected_gpu_name,
    "driver_version": selected_driver_version,
}
for key, value in expected.items():
    if p.get(key) != value:
        raise SystemExit(f"worker provenance mismatch: {key}")
fresh = p.get("fresh_process_id")
if not isinstance(fresh, str) or re.fullmatch(r"[0-9a-f]{64}", fresh) is None:
    raise SystemExit("fresh process identity is invalid")
process_identity = p.get("os_process_identity")
if not isinstance(process_identity, dict) or set(process_identity) != {
    "boot_id", "hostname", "pid", "process_start_ticks"
}:
    raise SystemExit("worker OS process identity fields are invalid")
boot_id = process_identity.get("boot_id")
hostname = process_identity.get("hostname")
pid = process_identity.get("pid")
start_ticks = process_identity.get("process_start_ticks")
if not isinstance(boot_id, str) \
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        ) is None \
        or hostname != expected_hostname \
        or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 \
        or isinstance(start_ticks, bool) or not isinstance(start_ticks, int) \
        or start_ticks < 0:
    raise SystemExit("worker OS process identity values are invalid")
if p.get("worker_pid") != pid:
    raise SystemExit("worker OS process identity is not bound to worker_pid")
observed_fresh = sha256(
    json.dumps(
        process_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if observed_fresh != fresh:
    raise SystemExit("fresh process hash differs from worker OS identity")
expected_constraints = {
    "kitless": True,
    "headless": True,
    "renderer": False,
    "camera": False,
    "fresh_process": True,
    "scenario_id": "small_step",
}
if p.get("constraints") != expected_constraints:
    raise SystemExit("worker execution constraints are invalid")
if p.get("forbidden_modules") != []:
    raise SystemExit("worker loaded forbidden Kit, renderer, or Isaac Sim modules")
intervention = p.get("legacy_joint_friction_intervention")
if not isinstance(intervention, dict) or intervention.get("role") != role \
        or intervention.get("private_plan_sha256") != plan_hash \
        or intervention.get("joint_count") != 22 \
        or intervention.get("source_asset_sha256_unchanged") is not True \
        or intervention.get("cleanup_status") != "restored_pre_values" \
        or not isinstance(intervention.get("records"), list) \
        or len(intervention["records"]) != 22:
    raise SystemExit("intervention evidence is incomplete")
config = p.get("simulation_configuration_v2")
if not isinstance(config, dict) or config.get("schema_version") != 2 \
        or config.get("runtime_effective_dt_s") is not None \
        or config.get("runtime_effective_dt_verified") is not False:
    raise SystemExit("simulation dt claim boundary is invalid")
if len(result.joint_names) != 22 or len(result.frame_names) != 5 \
        or len(result.samples) != result.requested_steps + 1:
    raise SystemExit("worker trace inventory is incomplete")
print(fresh)
PY
}

write_case_inventory() {
  local directory="$1" inventory="$1/evidence.sha256" temporary="$1/evidence.sha256.tmp.$$"
  [[ ! -e "$inventory" ]] || die "case inventory already exists"
  (
    cd "$directory"
    while IFS= read -r -d '' path; do sha256sum "$path"; done \
      < <(find . -type f ! -name evidence.sha256 ! -name 'evidence.sha256.tmp.*' -print0 | sort -z)
  ) > "$temporary"
  mv -- "$temporary" "$inventory"
}

[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] \
  || die "display variables must be unset"
unset DISPLAY WAYLAND_DISPLAY
unset PYTHONBREAKPOINT PYTHONCASEOK PYTHONCOERCECLOCALE PYTHONDEBUG PYTHONDEVMODE
unset PYTHONEXECUTABLE PYTHONFAULTHANDLER PYTHONHOME
unset PYTHONINSPECT PYTHONINTMAXSTRDIGITS PYTHONMALLOC PYTHONNODEBUGRANGES
unset PYTHONPATH PYTHONOPTIMIZE PYTHONPLATLIBDIR PYTHONPROFILEIMPORTTIME
unset PYTHONSTARTUP PYTHONTRACEMALLOC PYTHONUSERBASE PYTHONWARNDEFAULTENCODING
unset PYTHONWARNINGS
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONUTF8=1
export PYTHONPYCACHEPREFIX="$GATE0_ROOT/cache/pycache"
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

# This second GPU check remains read-only and happens immediately before lock.
if ! read_gpu_state "$SELECTED_GPU_UUID"; then
  die "selected GPU ceased to be strictly idle before acquiring the run lock"
fi

# The global lock precedes even computing/checking the candidate run path.
# Therefore a losing launcher leaves RUN_ROOT empty and its EXIT trap cannot
# write status or inventory files into the winning launcher's directory.
refuse_symlink "$GATE0_ROOT/.ovphysx-freeze-b-run.lock"
exec 9>"$GATE0_ROOT/.ovphysx-freeze-b-run.lock"
flock -n 9 || die "another Freeze B launcher holds the run lock"

# Recheck all mutable inputs and GPU availability inside the lock, still
# without creating any run evidence.
if ! read_gpu_state "$SELECTED_GPU_UUID"; then
  die "selected GPU ceased to be strictly idle while acquiring the run lock"
fi
verify_plan
verify_snapshot "locked-readonly-preflight"
verify_pinned_repositories
check_root_budget

run_root_candidate="$GATE0_ROOT/results/$RUN_ID"
[[ ! -e "$run_root_candidate" && ! -L "$run_root_candidate" ]] \
  || die "run directory already exists; refusing overwrite"
run_initializing_candidate="$GATE0_ROOT/results/.${RUN_ID}.initializing.$$"
[[ ! -e "$run_initializing_candidate" && ! -L "$run_initializing_candidate" ]] \
  || die "run initialization path already exists"
mkdir "$run_initializing_candidate"
INITIALIZING_ROOT="$run_initializing_candidate"
INITIALIZING_DEVICE_INODE="$(stat -c '%d:%i' -- "$run_initializing_candidate")"
[[ "$INITIALIZING_DEVICE_INODE" =~ ^[0-9]+:[0-9]+$ ]] \
  || die "run initialization inode identity is invalid"
mkdir "$run_initializing_candidate/launcher" "$run_initializing_candidate/cases"
require_writable_directory "$run_initializing_candidate"
require_writable_directory "$run_initializing_candidate/launcher"
require_writable_directory "$run_initializing_candidate/cases"
mv -T -n -- "$run_initializing_candidate" "$run_root_candidate"
[[ ! -e "$run_initializing_candidate" && ! -L "$run_initializing_candidate" \
  && -d "$run_root_candidate" && ! -L "$run_root_candidate" ]] \
  || die "run initialization promotion did not complete exactly once"
RUN_ROOT="$run_root_candidate"
INITIALIZING_ROOT=""
INITIALIZING_DEVICE_INODE=""
exec 1>"$RUN_ROOT/launcher/stdout.txt" 2>"$RUN_ROOT/launcher/stderr.txt"
printf 'Freeze B launcher started at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "${CASE_RECORDS[@]}" > "$RUN_ROOT/launcher/cases.txt"

if ! capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt" "$SELECTED_GPU_UUID"; then
  die "selected GPU is no longer strictly idle after run evidence creation"
fi
verify_plan
verify_snapshot "preflight"

LAUNCHER_SHA256="$(sha256sum -- "$PROJECT_DIR/scripts/run_ovphysx_freeze_b_remote.sh" | awk '{print $1}')"
WORKER_SHA256="$(sha256sum -- "$WORKER_SCRIPT" | awk '{print $1}')"
PROVENANCE_TEMP="$RUN_ROOT/launcher/provenance.json.tmp.$$"
RUN_ID="$RUN_ID" SESSION_ID="$SESSION_ID" SOURCE_REVISION="$SOURCE_REVISION" SOURCE_TREE="$SOURCE_TREE" \
ARCHIVE_SHA256="$ARCHIVE_SHA256" SNAPSHOT_SHA256="$SNAPSHOT_SHA256" PRIVATE_PLAN_SHA256="$PRIVATE_PLAN_SHA256" \
PRIVATE_PLAN_FILE_SHA256="$PRIVATE_PLAN_FILE_SHA256" \
PUBLIC_PROTOCOL_FILE_SHA256="$PUBLIC_PROTOCOL_FILE_SHA256" \
PUBLIC_PROTOCOL_CANONICAL_SHA256="$PUBLIC_PROTOCOL_CANONICAL_SHA256" \
IMPLEMENTATION_REVISION="$IMPLEMENTATION_REVISION" IMPLEMENTATION_TREE="$IMPLEMENTATION_TREE" \
LAUNCHER_SHA256="$LAUNCHER_SHA256" WORKER_SHA256="$WORKER_SHA256" GPU_INDEX="$GPU_INDEX" \
SELECTED_GPU_UUID="$SELECTED_GPU_UUID" GPU_NAME="$SELECTED_GPU_NAME" DRIVER="$SELECTED_DRIVER_VERSION" \
ENVIRONMENT_SUMMARY="$environment_summary" "$WORKER_PYTHON" -P -S - "$PROVENANCE_TEMP" <<'PY'
import json
import os
from pathlib import Path
import sys

payload = {
    "schema_version": 1,
    "protocol_id": "ovphysx-legacy-joint-friction-freeze-b-v1",
    "run_id": os.environ["RUN_ID"],
    "session_id": os.environ["SESSION_ID"],
    "source_revision": os.environ["SOURCE_REVISION"],
    "source_tree": os.environ["SOURCE_TREE"],
    "source_archive_sha256": os.environ["ARCHIVE_SHA256"],
    "source_snapshot_sha256": os.environ["SNAPSHOT_SHA256"],
    "private_plan_sha256": os.environ["PRIVATE_PLAN_SHA256"],
    "private_plan_file_sha256": os.environ["PRIVATE_PLAN_FILE_SHA256"],
    "public_protocol_file_sha256": os.environ["PUBLIC_PROTOCOL_FILE_SHA256"],
    "public_protocol_canonical_sha256": os.environ["PUBLIC_PROTOCOL_CANONICAL_SHA256"],
    "implementation_source_revision": os.environ["IMPLEMENTATION_REVISION"],
    "implementation_source_tree": os.environ["IMPLEMENTATION_TREE"],
    "deployment_source_revision": os.environ["SOURCE_REVISION"],
    "deployment_source_tree": os.environ["SOURCE_TREE"],
    "implementation_to_deployment_relation_scope": "locally_audited_before_gitless_remote_deployment_not_recomputed_here",
    "launcher_sha256": os.environ["LAUNCHER_SHA256"],
    "worker_sha256": os.environ["WORKER_SHA256"],
    "selected_gpu_index": int(os.environ["GPU_INDEX"]),
    "selected_gpu_uuid": os.environ["SELECTED_GPU_UUID"],
    "selected_gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER"],
    "case_count": 16,
    "maximum_gpu_hours": 0.5,
    "maximum_retained_bytes": 500 * 1024 * 1024,
    "headless": True,
    "kitless": True,
    "renderer": False,
    "camera": False,
    "environment": dict(line.split("=", 1) for line in os.environ["ENVIRONMENT_SUMMARY"].splitlines()),
    "claim_boundary": "collection_only_no_scientific_or_formal_gate0_verdict",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv -- "$PROVENANCE_TEMP" "$RUN_ROOT/launcher/provenance.json"

GPU_STARTED_EPOCH="$(date +%s)"
declare -a FRESH_PROCESS_IDS=()
: > "$RUN_ROOT/launcher/fresh-process-ids.txt"
for record in "${CASE_RECORDS[@]}"; do
  IFS='|' read -r case_id canonical_id role hand variant dt repeat <<< "$record"
  CURRENT_CASE="$case_id"
  verify_plan
  verify_snapshot "before-$case_id"
  elapsed=$(( $(date +%s) - GPU_STARTED_EPOCH ))
  remaining=$((GLOBAL_GPU_TIMEOUT_S - elapsed))
  [[ "$remaining" -gt 0 ]] || die "global 0.5 GPU-hour budget is exhausted"
  envelope="$WORKER_TIMEOUT_S"
  [[ "$remaining" -ge "$envelope" ]] || envelope="$remaining"
  case_dir="$RUN_ROOT/cases/$case_id"
  [[ ! -e "$case_dir" && ! -L "$case_dir" ]] || die "case directory already exists"
  mkdir "$case_dir"
  require_writable_directory "$case_dir"
  if ! capture_gpu_state "$case_dir/preflight.txt" "$SELECTED_GPU_UUID"; then
    die "selected GPU is no longer strictly idle before $case_id"
  fi
  payload_path="$case_dir/$case_id.json"
  pid_file="$case_dir/owned-process-group.txt"
  command=(
    timeout --foreground --signal=TERM --kill-after=30s "${envelope}s"
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    PYTHONPATH="$PROJECT_DIR/src"
    WAVEQA_SOURCE_TREE="$SOURCE_REVISION"
    "$WORKER_PYTHON" -P "$WORKER_SCRIPT"
    --approved-root "$GATE0_ROOT"
    --asset-root "$ASSET_ROOT"
    --manifest "$GATE0_MANIFEST"
    --public-protocol "$PUBLIC_PROTOCOL"
    --public-protocol-file-sha256 "$PUBLIC_PROTOCOL_FILE_SHA256"
    --public-protocol-canonical-sha256 "$PUBLIC_PROTOCOL_CANONICAL_SHA256"
    --private-plan "$PRIVATE_PLAN"
    --private-plan-sha256 "$PRIVATE_PLAN_SHA256"
    --case-id "$case_id"
    --output "$payload_path"
    --device cuda:0
    --session-id "$SESSION_ID"
    --source-revision "$SOURCE_REVISION"
    --asset-tree-sha256 "$ASSET_LF_SHA256"
  )
  (
    exec "$WORKER_PYTHON" -P - "$pid_file" "${command[@]}"
  ) > "$case_dir/worker.stdout.txt" 2> "$case_dir/worker.stderr.txt" <<'PY' &
import os
from pathlib import Path
import sys

pid_path = Path(sys.argv[1])
command = sys.argv[2:]
os.setsid()
descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execvpe(command[0], command, os.environ.copy())
PY
  owner_pid="$!"
  for _wait in {1..50}; do
    [[ -s "$pid_file" ]] && break
    case_process_running "$owner_pid" || break
    sleep 0.1
  done
  [[ -s "$pid_file" ]] || { wait "$owner_pid" || true; die "fresh process group was not established"; }
  IFS= read -r owner_pgid < "$pid_file"
  observed_pgid="$(ps -o pgid= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ "$owner_pgid" == "$owner_pid" && "$observed_pgid" == "$owner_pgid" ]] \
    || { kill -TERM "$owner_pid" 2>/dev/null || true; wait "$owner_pid" || true; die "unsafe process group identity"; }
  ACTIVE_CASE_PID="$owner_pid"
  ACTIVE_CASE_PGID="$owner_pgid"
  foreign=false
  monitor_owned_case "$owner_pid" "$owner_pgid" "$case_dir/gpu-monitor.txt" "$case_dir/foreign-gpu-process.txt" || foreign=true
  set +e
  wait "$owner_pid"
  worker_exit="$?"
  set -e
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  printf '%s\n' "$worker_exit" > "$case_dir/worker.exit-code.txt"
  if [[ "$foreign" == true ]]; then
    capture_gpu_state "$case_dir/postflight-foreign.txt" "$SELECTED_GPU_UUID" || true
    die "foreign GPU process detected; only the owned case process group was terminated"
  fi
  postflight_ok=false
  for attempt in 1 2 3; do
    if capture_gpu_state "$(printf '%s/postflight-%02d.txt' "$case_dir" "$attempt")" "$SELECTED_GPU_UUID"; then
      postflight_ok=true
      break
    fi
    [[ "$attempt" -eq 3 ]] || sleep 5
  done
  [[ "$postflight_ok" == true ]] || die "GPU did not return to strict idle after $case_id"
  validate_worker_stdout "$case_dir/worker.stdout.txt" "$case_id" "$case_id.json"
  [[ "$worker_exit" -eq 0 ]] || die "worker returned $worker_exit for $case_id"
  [[ -f "$payload_path" && ! -L "$payload_path" ]] || die "worker payload is missing"
  fresh_id="$(validate_case_payload "$payload_path" "$case_id" "$canonical_id" "$role" "$hand" "$variant" "$dt" "$repeat")"
  [[ "$fresh_id" =~ ^[0-9a-f]{64}$ ]] || die "fresh process ID is invalid"
  for seen in "${FRESH_PROCESS_IDS[@]}"; do
    [[ "$seen" != "$fresh_id" ]] || die "worker reused a fresh process identity"
  done
  FRESH_PROCESS_IDS+=("$fresh_id")
  printf '%s|%s\n' "$case_id" "$fresh_id" >> "$RUN_ROOT/launcher/fresh-process-ids.txt"
  check_run_budget
  write_case_inventory "$case_dir"
  COMPLETED_CASES=$((COMPLETED_CASES + 1))
  check_root_budget
done

[[ "$COMPLETED_CASES" -eq 16 && "${#FRESH_PROCESS_IDS[@]}" -eq 16 ]] \
  || die "Freeze B did not complete 16 unique fresh-process cases"
verify_plan
verify_snapshot "post-matrix"
verify_pinned_repositories
check_root_budget
check_run_budget
RUN_STATUS="collected"
SCIENTIFIC_VERDICT="pending_local_validation"
printf 'collected 16 private Freeze B cases; no scientific verdict was evaluated at %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
