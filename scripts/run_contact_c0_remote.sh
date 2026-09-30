#!/usr/bin/env bash

set -euo pipefail
umask 077

# Formal Server 6 collector for the kit-less OVPhysX half of Contact Gate C0.
# The launcher is deliberately fail-closed: it never overwrites an attempt,
# uses one already-idle physical GPU, and runs every canonical case in a fresh
# Linux process group.  A foreign GPU process is recorded but never signalled.
readonly CONTACT_C0_ROOT="/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
readonly EXPECTED_HOSTNAME="example-gpu-node-6"
readonly EXPECTED_USER="exampleuser"
readonly EXPECTED_OS_ID="ubuntu"
readonly EXPECTED_OS_VERSION="22.04"
readonly EXPECTED_DRIVER_VERSION="570.158.01"
readonly EXPECTED_GPU_NAME_FRAGMENT="A800-SXM4-40GB"
readonly EXPECTED_MANIFEST_ID="wavesimparity-contact-c0"
readonly EXPECTED_OVPHYSX_CASE_COUNT=16
readonly MAX_IDLE_MEMORY_MIB=32
readonly ASSET_URL="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
readonly ASSET_COMMIT="6eea427eb24189519f32b9f21674cd534d3f973c"
readonly ASSET_GIT_TREE="bb00a9d5527b8a76de576ce876ebece67d8ffde1"
readonly ASSET_LF_SHA256="b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
readonly ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
readonly UV_LOCK_SHA256="cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522"
readonly OVPHYSX_SDK_ARCHIVE="/data/home/exampleuser/sharpa-wave-asset-qa-gate0/dependencies/ovphysx-0.4.13/ovphysx-linux-x86_64-0.4.13.tar.gz"
readonly OVPHYSX_SDK_ARCHIVE_SHA256="191dcaff34980f6fdf94bb783c8faacbab058aa4c2671e31ac306fcd89cdb1e7"
readonly NATIVE_HELPER_SOURCE_RELATIVE="src/wave_asset_qa/contact/native/physx_ccd_readback.cpp"
readonly NATIVE_HELPER_FILENAME="libwaveqa_physx_ccd_readback.so"
readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))
readonly MAX_RUN_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_ARTIFACT_BYTES=$((2 * 1024 * 1024 * 1024))
readonly MAX_ARTIFACT_BLOCKS=$((MAX_ARTIFACT_BYTES / 1024))
readonly FINAL_METADATA_RESERVE_BYTES=$((4 * 1024 * 1024))
readonly WORKER_TIMEOUT_S=1200
readonly CASE_ENVELOPE_TIMEOUT_S=1260
readonly GLOBAL_GPU_TIMEOUT_S=$((4 * 60 * 60))
readonly GPU_KILL_GRACE_S=30
readonly GPU_DEADLINE_SAFETY_MARGIN_S=30

PROJECT_DIR_INPUT=""
RUN_ID=""
GPU_INDEX=""
CONTACT_EXPERIMENTAL_CAMPAIGN=""
CONTACT_SLURM_BINDING=""
WORKER_EXTRA=()
WORKER_CUDA_SELECTOR=""
SESSION_ID=""
SOURCE_REVISION=""
SOURCE_TREE_INPUT=""
SNAPSHOT_SHA256_INPUT=""
RUN_ROOT=""
RUN_CREATED=false
RUN_STATUS="error"
SCIENTIFIC_VERDICT="not_evaluated"
CURRENT_CASE=""
COMPLETED_CASES=0
SELECTED_GPU_UUID=""
SOURCE_TREE=""
ARCHIVE_SHA256=""
SNAPSHOT_SHA256=""
MANIFEST_SHA256=""
NATIVE_HELPER_SHA256=""
NATIVE_HEADERS_MANIFEST_SHA256=""
NATIVE_BUILD_PROVENANCE_SHA256=""
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
declare -A FRESH_PROCESS_IDS=()

# Keep the caller's descriptors.  Once the exclusive run exists, launcher
# output becomes private evidence and the terminal receives only a final line.
exec 3>&1 4>&2

usage() {
  printf '%s\n' \
    "usage: run_contact_c0_remote.sh --project-dir PATH --run-id ID --gpu-index INDEX" \
    "       --session-id ID --source-revision 40_HEX --source-tree 40_HEX" \
    "       --snapshot-sha256 64_HEX" >&4
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
  [[ "$path" == "$CONTACT_C0_ROOT" || "$path" == "$CONTACT_C0_ROOT/"* ]] \
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
      || die "managed write directory contains a dangling link: $link"
    [[ "$target" == "$path" || "$target" == "$path/"* ]] \
      || die "managed write symlink escapes its directory: $link -> $target"
  done < <(find "$path" -type l -print0)
}

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

write_compute_process_envelope() {
  local processes="${1:-}"
  printf 'compute_processes_begin\n'
  if [[ -n "$processes" ]]; then
    printf '%s\n' "$processes"
  fi
  printf 'compute_processes_end\n'
}

query_idle_gpu_state() {
  local expected_uuid="${1:-}"
  local gpu_line processes actual_index
  gpu_line="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)" || return 1
  [[ "$gpu_line" != *$'\n'* ]] || return 1
  actual_index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$gpu_line")"
  LAST_GPU_UUID="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$gpu_line")"
  LAST_GPU_NAME="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}' <<< "$gpu_line")"
  LAST_GPU_DRIVER="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<< "$gpu_line")"
  LAST_GPU_MEMORY="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}' <<< "$gpu_line")"
  LAST_GPU_UTILIZATION="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}' <<< "$gpu_line")"
  processes="$(nvidia-smi -i "$GPU_INDEX" \
    --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader,nounits)" || return 1
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

capture_gpu_state() {
  local destination="$1"
  local expected_uuid="${2:-}"
  local temporary actual_index query_status=0
  temporary="$destination.tmp.$$"
  query_idle_gpu_state "$expected_uuid" || query_status="$?"
  actual_index="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$LAST_GPU_LINE")"
  {
    printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'query_status=%s\n' "$query_status"
    printf 'gpu_index=%s\n' "$actual_index"
    printf 'gpu_uuid=%s\n' "$LAST_GPU_UUID"
    printf 'gpu_name=%s\n' "$LAST_GPU_NAME"
    printf 'driver_version=%s\n' "$LAST_GPU_DRIVER"
    printf 'memory_used_mib=%s\n' "$LAST_GPU_MEMORY"
    printf 'utilization_percent=%s\n' "$LAST_GPU_UTILIZATION"
    write_compute_process_envelope "$LAST_GPU_PROCESSES"
  } > "$temporary" || return 1
  mv -- "$temporary" "$destination" || return 1
  return "$query_status"
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
  local observed_pgid=""
  [[ "$owner_pid" =~ ^[1-9][0-9]*$ && "$owner_pgid" == "$owner_pid" ]] || return 0
  observed_pgid="$(ps -o pgid= -p "$owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ -n "$observed_pgid" && "$observed_pgid" != "$owner_pgid" ]]; then
    return 1
  fi
  kill -0 -- "-$owner_pgid" 2>/dev/null || return 0
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

capture_worker_group_binding() {
  local owner_pgid="$1"
  local case_id="$2"
  local process_file="$3"
  local destination="$4"
  local identity_line worker_pid worker_hash observed_pgid
  [[ -f "$process_file" && ! -L "$process_file" ]] || return 1
  identity_line="$(PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -B - \
    "$process_file" "$case_id" <<'PY'
import sys
from wave_asset_qa.contact.bundle import read_json_strict
from wave_asset_qa.parity.process_identity import (
    os_process_identity_sha256,
    validate_os_process_identity,
)

record = read_json_strict(sys.argv[1])
if not isinstance(record, dict) or record.get("case_id") != sys.argv[2]:
    raise SystemExit(2)
identity = validate_os_process_identity(record.get("os_process_identity"))
digest = os_process_identity_sha256(identity)
if record.get("fresh_process_identity_sha256") != digest:
    raise SystemExit(2)
print(f"{identity['pid']} {digest}")
PY
)" || return 1
  read -r worker_pid worker_hash <<< "$identity_line"
  [[ "$worker_pid" =~ ^[1-9][0-9]*$ && "$worker_hash" =~ ^[0-9a-f]{64}$ ]] || return 2
  observed_pgid="$(ps -o pgid= -p "$worker_pid" 2>/dev/null | awk '{$1=$1; print}')"
  [[ -n "$observed_pgid" ]] || return 1
  [[ "$observed_pgid" == "$owner_pgid" ]] || return 2
  CONTACT_BINDING_CASE="$case_id" CONTACT_BINDING_PID="$worker_pid" \
    CONTACT_BINDING_PGID="$observed_pgid" CONTACT_BINDING_HASH="$worker_hash" \
    "$WORKER_PYTHON" -P -B -S - "$destination" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
record = {
    "schema_version": 1,
    "visibility": "private_not_for_publication",
    "case_id": os.environ["CONTACT_BINDING_CASE"],
    "worker_pid": int(os.environ["CONTACT_BINDING_PID"]),
    "owned_process_group_id": int(os.environ["CONTACT_BINDING_PGID"]),
    "fresh_process_identity_sha256": os.environ["CONTACT_BINDING_HASH"],
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

monitor_owned_case() {
  local owner_pid="$1"
  local owner_pgid="$2"
  local case_id="$3"
  local process_file="$4"
  local binding_file="$5"
  local monitor_file="$6"
  local foreign_file="$7"
  local processes row process_uuid process_pid process_pgid binding_status
  : > "$monitor_file" || return 1
  while case_process_running "$owner_pid"; do
    if [[ ! -e "$binding_file" && -f "$process_file" ]]; then
      set +e
      capture_worker_group_binding \
        "$owner_pgid" "$case_id" "$process_file" "$binding_file"
      binding_status="$?"
      set -e
      if [[ "$binding_status" -eq 2 ]]; then
        printf 'worker process identity escaped owned PGID %s\n' "$owner_pgid" \
          > "$foreign_file"
        terminate_owned_case "$owner_pid" "$owner_pgid" || true
        return 1
      fi
    fi
    if ! processes="$(nvidia-smi -i "$GPU_INDEX" \
      --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
      --format=csv,noheader,nounits)"; then
      printf 'compute-process query failed while owned case was active\n' > "$foreign_file"
      terminate_owned_case "$owner_pid" "$owner_pgid" || true
      return 1
    fi
    {
      printf 'sample_at_utc=%s owner_pgid=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$owner_pgid"
      write_compute_process_envelope "$processes"
    } >> "$monitor_file" || {
      printf 'GPU monitor evidence write failed\n' > "$foreign_file" || true
      terminate_owned_case "$owner_pid" "$owner_pgid" || true
      return 1
    }
    if [[ -n "$processes" ]]; then
      while IFS= read -r row; do
        [[ -n "$row" ]] || continue
        process_uuid="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' <<< "$row")"
        process_pid="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' <<< "$row")"
        if [[ "$process_uuid" != "$SELECTED_GPU_UUID" || ! "$process_pid" =~ ^[1-9][0-9]*$ ]]; then
          printf 'malformed or wrong-device GPU process row: %s\n' "$row" > "$foreign_file"
          terminate_owned_case "$owner_pid" "$owner_pgid" || true
          return 1
        fi
        process_pgid="$(ps -o pgid= -p "$process_pid" 2>/dev/null | awk '{$1=$1; print}')"
        if [[ -z "$process_pgid" ]]; then
          printf 'GPU process pid=%s disappeared before owned-group identity could be proven\n' \
            "$process_pid" > "$foreign_file"
          terminate_owned_case "$owner_pid" "$owner_pgid" || true
          return 1
        fi
        if [[ "$process_pgid" != "$owner_pgid" ]]; then
          printf 'foreign GPU process pid=%s pgid=%s detected; expected owned pgid=%s\n' \
            "$process_pid" "$process_pgid" "$owner_pgid" > "$foreign_file"
          terminate_owned_case "$owner_pid" "$owner_pgid" || true
          return 1
        fi
      done <<< "$processes"
    fi
    sleep 1
  done
  [[ -f "$binding_file" && ! -L "$binding_file" ]] || {
    printf 'actual Python worker-to-PGID binding was not observed\n' > "$foreign_file"
    return 1
  }
}

root_bytes() {
  du -sb "$CONTACT_C0_ROOT" | awk '{print $1}'
}

check_root_budget() {
  local bytes
  bytes="$(root_bytes)"
  [[ "$bytes" -le "$MAX_ROOT_BYTES" ]] \
    || die "approved root exceeds the 30 GiB stop limit"
}

check_run_budget() {
  local bytes oversized
  [[ -n "$RUN_ROOT" && -d "$RUN_ROOT" ]] || return 0
  bytes="$(du -sb "$RUN_ROOT" | awk '{print $1}')"
  [[ "$bytes" -le "$MAX_RUN_BYTES" ]] \
    || die "Contact C0 run evidence exceeds the 2 GiB hard limit"
  oversized="$(find "$RUN_ROOT" -type f -size +"${MAX_ARTIFACT_BYTES}"c -print -quit)"
  [[ -z "$oversized" ]] || die "artifact exceeds the 2 GiB hard limit: $oversized"
}

compute_snapshot_sha256() {
  PROJECT_SNAPSHOT="$PROJECT_DIR" "$WORKER_PYTHON" -P -B -S - <<'PY'
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
}

verify_snapshot() {
  local checkpoint="$1"
  local actual writable
  [[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] \
    || die "project snapshot gained a symbolic link before $checkpoint"
  writable="$(find "$PROJECT_DIR" -perm /222 -print -quit)"
  [[ -z "$writable" ]] \
    || die "project snapshot is writable before $checkpoint: $writable"
  actual="$(compute_snapshot_sha256)"
  [[ "$actual" == "$SNAPSHOT_SHA256" ]] \
    || die "project snapshot hash changed before $checkpoint"
  if [[ -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]]; then
    printf '%s checkpoint=%s snapshot_sha256=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checkpoint" "$actual" \
      >> "$RUN_ROOT/launcher/snapshot-verifications.txt"
  fi
}

verify_repository() {
  local path="$1"
  local expected_url="$2"
  local allowed_untracked="${3:-}"
  local status_line
  [[ -d "$path/.git" ]] || die "managed checkout is not a Git repository: $path"
  [[ "$(git -C "$path" remote get-url origin)" == "$expected_url" ]] \
    || die "unexpected origin URL in managed checkout: $path"
  while IFS= read -r status_line; do
    [[ -z "$status_line" ]] && continue
    if [[ -n "$allowed_untracked" && "$status_line" == "?? $allowed_untracked" ]]; then
      continue
    fi
    die "managed checkout is dirty: $status_line"
  done < <(git -C "$path" status --porcelain --untracked-files=all)
}

verify_source_archive() {
  local actual
  [[ -f "$SOURCE_ARCHIVE" && ! -L "$SOURCE_ARCHIVE" ]] \
    || die "deployed source archive is no longer a regular file"
  [[ "$(readlink -f -- "$SOURCE_ARCHIVE")" == "$SOURCE_ARCHIVE" ]] \
    || die "deployed source archive path changed"
  actual="$(sha256sum -- "$SOURCE_ARCHIVE" | awk '{print $1}')"
  [[ "$actual" == "$ARCHIVE_SHA256" ]] \
    || die "deployed source archive SHA-256 differs from its marker"
}

verify_physx_sdk_archive() {
  local actual archive_root
  archive_root="$(dirname -- "$OVPHYSX_SDK_ARCHIVE")"
  require_real_directory "$archive_root" "OVPhysX SDK archive parent"
  [[ -f "$OVPHYSX_SDK_ARCHIVE" && ! -L "$OVPHYSX_SDK_ARCHIVE" ]] \
    || die "pinned OVPhysX SDK archive is not a regular file"
  [[ "$(readlink -f -- "$OVPHYSX_SDK_ARCHIVE")" == "$OVPHYSX_SDK_ARCHIVE" ]] \
    || die "pinned OVPhysX SDK archive path is not canonical"
  [[ "$(stat -c '%U' "$OVPHYSX_SDK_ARCHIVE")" == "$EXPECTED_USER" ]] \
    || die "pinned OVPhysX SDK archive owner changed"
  [[ "$(findmnt -n -o TARGET -T "$OVPHYSX_SDK_ARCHIVE")" == "/data" ]] \
    || die "pinned OVPhysX SDK archive left /data"
  [[ "$(findmnt -n -o FSTYPE -T "$OVPHYSX_SDK_ARCHIVE")" == "xfs" ]] \
    || die "pinned OVPhysX SDK archive is not on XFS"
  actual="$(sha256sum -- "$OVPHYSX_SDK_ARCHIVE" | awk '{print $1}')"
  [[ "$actual" == "$OVPHYSX_SDK_ARCHIVE_SHA256" ]] \
    || die "pinned OVPhysX SDK archive SHA-256 differs"
}

verify_native_helper() {
  local actual
  [[ -f "$NATIVE_HELPER_PATH" && ! -L "$NATIVE_HELPER_PATH" ]] \
    || die "native PhysX CCD helper is not a regular file"
  [[ "$(readlink -f -- "$NATIVE_HELPER_PATH")" == "$NATIVE_HELPER_PATH" ]] \
    || die "native PhysX CCD helper path changed"
  actual="$(sha256sum -- "$NATIVE_HELPER_PATH" | awk '{print $1}')"
  [[ "$actual" == "$NATIVE_HELPER_SHA256" ]] \
    || die "native PhysX CCD helper SHA-256 changed"
  actual="$(sha256sum -- "$NATIVE_HEADERS_MANIFEST" | awk '{print $1}')"
  [[ "$actual" == "$NATIVE_HEADERS_MANIFEST_SHA256" ]] \
    || die "native PhysX header manifest SHA-256 changed"
  actual="$(sha256sum -- "$NATIVE_BUILD_PROVENANCE" | awk '{print $1}')"
  [[ "$actual" == "$NATIVE_BUILD_PROVENANCE_SHA256" ]] \
    || die "native PhysX helper build provenance SHA-256 changed"
}

write_status() {
  local exit_code="$1"
  local ended_epoch elapsed gpu_elapsed status unique_count
  [[ "$RUN_CREATED" == true && -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]] \
    || return 0
  ended_epoch="$(date +%s)"
  elapsed=$((ended_epoch - LAUNCH_STARTED_EPOCH))
  gpu_elapsed=0
  [[ -z "$GPU_STARTED_EPOCH" ]] || gpu_elapsed=$((ended_epoch - GPU_STARTED_EPOCH))
  status="$RUN_STATUS"
  [[ "$exit_code" -eq 0 ]] || status="error"
  unique_count="${#FRESH_PROCESS_IDS[@]}"
  CONTACT_STATUS="$status" CONTACT_STATUS_EXIT="$exit_code" \
    CONTACT_STATUS_RUN="$RUN_ID" CONTACT_STATUS_SESSION="$SESSION_ID" \
    CONTACT_STATUS_REVISION="$SOURCE_REVISION" CONTACT_STATUS_TREE="$SOURCE_TREE" \
    CONTACT_STATUS_SNAPSHOT="$SNAPSHOT_SHA256" CONTACT_STATUS_ARCHIVE="$ARCHIVE_SHA256" \
    CONTACT_STATUS_GPU_INDEX="$GPU_INDEX" CONTACT_STATUS_GPU_UUID="$SELECTED_GPU_UUID" \
    CONTACT_STATUS_COMPLETED="$COMPLETED_CASES" CONTACT_STATUS_UNIQUE="$unique_count" \
    CONTACT_STATUS_LAST_CASE="$CURRENT_CASE" CONTACT_STATUS_ELAPSED="$elapsed" \
    CONTACT_STATUS_GPU_ELAPSED="$gpu_elapsed" CONTACT_STATUS_VERDICT="$SCIENTIFIC_VERDICT" \
    "$WORKER_PYTHON" -P -B -S - "$RUN_ROOT/launcher/status.json" <<'PY'
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

path = Path(sys.argv[1])
record = {
    "schema_version": 1,
    "campaign": os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") or "contact_c0",
    "backend": "ovphysx",
    "status": os.environ["CONTACT_STATUS"],
    "scientific_verdict": os.environ["CONTACT_STATUS_VERDICT"],
    "exit_code": int(os.environ["CONTACT_STATUS_EXIT"]),
    "run_id": os.environ["CONTACT_STATUS_RUN"],
    "session_id": os.environ["CONTACT_STATUS_SESSION"],
    "source_revision": os.environ["CONTACT_STATUS_REVISION"],
    "source_tree": os.environ["CONTACT_STATUS_TREE"],
    "source_archive_sha256": os.environ["CONTACT_STATUS_ARCHIVE"],
    "snapshot_sha256": os.environ["CONTACT_STATUS_SNAPSHOT"],
    "selected_gpu_index": int(os.environ["CONTACT_STATUS_GPU_INDEX"]),
    "selected_gpu_uuid": os.environ["CONTACT_STATUS_GPU_UUID"],
    "completed_case_count": int(os.environ["CONTACT_STATUS_COMPLETED"]),
    "unique_process_count": int(os.environ["CONTACT_STATUS_UNIQUE"]),
    "last_case_id": os.environ["CONTACT_STATUS_LAST_CASE"] or None,
    "elapsed_s": int(os.environ["CONTACT_STATUS_ELAPSED"]),
    "gpu_elapsed_s": int(os.environ["CONTACT_STATUS_GPU_ELAPSED"]),
    "ended_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

write_evidence_manifest() {
  [[ "$RUN_CREATED" == true && -n "$RUN_ROOT" && -d "$RUN_ROOT" ]] || return 0
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -B - \
    "$RUN_ROOT" "$RUN_ROOT/evidence-manifest.json" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys

from wave_asset_qa.contact.bundle import inventory_root_sha256, sha256_file

root = Path(sys.argv[1]).resolve(strict=True)
destination = Path(sys.argv[2])
if destination.exists() or destination.is_symlink():
    raise SystemExit("root evidence manifest already exists")
records = []
for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"evidence contains a link: {relative}")
    if stat.S_ISDIR(mode):
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"evidence contains a special entry: {relative}")
    if relative == "evidence-manifest.json":
        continue
    records.append({
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    })
payload = {
    "schema_version": 1,
    "scope": (os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") or "contact-c0") + "-ovphysx",
    "records": records,
    "root_sha256": inventory_root_sha256(records),
}
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

finish() {
  local exit_code="$?"
  local final_run_bytes="0"
  trap - EXIT
  set +e
  if ! terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID"; then
    exit_code=2
    RUN_STATUS="error"
    SCIENTIFIC_VERDICT="not_evaluated"
  fi
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  if [[ "$RUN_CREATED" == true && -n "$RUN_ROOT" && -d "$RUN_ROOT/launcher" ]]; then
    [[ "$exit_code" -eq 0 && "$RUN_STATUS" == "completed" ]] || {
      exit_code=2
      RUN_STATUS="error"
      SCIENTIFIC_VERDICT="not_evaluated"
    }
    final_run_bytes="$(du -sb "$RUN_ROOT" 2>/dev/null | awk '{print $1}')"
    if [[ ! "$final_run_bytes" =~ ^[0-9]+$ \
      || "$final_run_bytes" -gt $((MAX_RUN_BYTES - FINAL_METADATA_RESERVE_BYTES)) ]]; then
      exit_code=2
      RUN_STATUS="error"
      SCIENTIFIC_VERDICT="not_evaluated"
    fi
    write_status "$exit_code" || exit_code=2
    write_evidence_manifest || exit_code=2
  fi
  exec 1>&3 2>&4
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'Contact C0 OVPhysX collection completed; evidence retained under results/%s\n' "$RUN_ID"
  elif [[ "$RUN_CREATED" == true && -n "$RUN_ROOT" ]]; then
    printf 'Contact C0 OVPhysX collection stopped; evidence retained under results/%s\n' "$RUN_ID" >&2
  fi
  exit "$exit_code"
}

signal_exit() {
  local signal_name="$1"
  local exit_code="$2"
  printf 'launcher received %s while case=%s\n' "$signal_name" "${CURRENT_CASE:-none}" >&2
  terminate_owned_case "$ACTIVE_CASE_PID" "$ACTIVE_CASE_PGID" || true
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
seen_tree=false
seen_snapshot=false
while [[ "$#" -gt 0 ]]; do
  [[ "$#" -ge 2 ]] || { usage; die "every option requires one value"; }
  case "$1" in
    --project-dir)
      [[ "$seen_project" == false ]] || die "--project-dir was supplied more than once"
      PROJECT_DIR_INPUT="$2"; seen_project=true ;;
    --run-id)
      [[ "$seen_run" == false ]] || die "--run-id was supplied more than once"
      RUN_ID="$2"; seen_run=true ;;
    --gpu-index)
      [[ "$seen_gpu" == false ]] || die "--gpu-index was supplied more than once"
      GPU_INDEX="$2"; seen_gpu=true ;;
    --session-id)
      [[ "$seen_session" == false ]] || die "--session-id was supplied more than once"
      SESSION_ID="$2"; seen_session=true ;;
    --source-revision)
      [[ "$seen_revision" == false ]] || die "--source-revision was supplied more than once"
      SOURCE_REVISION="$2"; seen_revision=true ;;
    --source-tree)
      [[ "$seen_tree" == false ]] || die "--source-tree was supplied more than once"
      SOURCE_TREE_INPUT="$2"; seen_tree=true ;;
    --snapshot-sha256)
      [[ "$seen_snapshot" == false ]] || die "--snapshot-sha256 was supplied more than once"
      SNAPSHOT_SHA256_INPUT="$2"; seen_snapshot=true ;;
    --experimental-campaign)
      [[ -z "$CONTACT_EXPERIMENTAL_CAMPAIGN" && "$2" == contact-c0-async-v1 ]] \
        || die "unknown or duplicate experimental campaign"
      CONTACT_EXPERIMENTAL_CAMPAIGN="$2" ;;
    *)
      usage
      die "unsupported option: $1" ;;
  esac
  shift 2
done

[[ "$seen_project" == true && "$seen_run" == true && "$seen_gpu" == true \
  && "$seen_session" == true && "$seen_revision" == true \
  && "$seen_tree" == true && "$seen_snapshot" == true ]] \
  || { usage; die "all seven launcher options are required"; }
safe_identifier "$RUN_ID" || die "run ID must be a safe lowercase identifier"
safe_identifier "$SESSION_ID" || die "session ID must be a safe lowercase identifier"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || die "source revision must be a lowercase 40-character Git commit"
[[ "$SOURCE_TREE_INPUT" =~ ^[0-9a-f]{40}$ ]] \
  || die "source tree must be a lowercase 40-character Git tree"
[[ "$SNAPSHOT_SHA256_INPUT" =~ ^[0-9a-f]{64}$ ]] \
  || die "snapshot SHA-256 must be lowercase 64-character hex"
[[ "$GPU_INDEX" =~ ^[0-7]$ ]] || die "GPU index must be an integer from 0 through 7"

for command_name in \
  awk basename date dirname du env find findmnt flock g++ git hostname id mkdir mv \
  nvidia-smi ps readlink sha256sum sleep sort stat timeout; do
  require_command "$command_name"
done

# The false-host check deliberately precedes every managed filesystem write.
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || die "expected Server 6 hostname $EXPECTED_HOSTNAME"
[[ "$(id -un)" == "$EXPECTED_USER" ]] || die "expected user $EXPECTED_USER"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "$EXPECTED_OS_ID" && "${VERSION_ID:-}" == "$EXPECTED_OS_VERSION" ]] \
  || die "expected Ubuntu 22.04"
[[ -d "$CONTACT_C0_ROOT" ]] || die "approved root does not exist"
refuse_symlink "$CONTACT_C0_ROOT"
[[ "$(readlink -f "$CONTACT_C0_ROOT")" == "$CONTACT_C0_ROOT" ]] \
  || die "approved root is not canonical"
[[ "$(stat -c '%U' "$CONTACT_C0_ROOT")" == "$EXPECTED_USER" ]] \
  || die "approved root owner changed"
[[ "$(findmnt -n -o TARGET -T "$CONTACT_C0_ROOT")" == "/data" ]] \
  || die "approved root left /data"
[[ "$(findmnt -n -o FSTYPE -T "$CONTACT_C0_ROOT")" == "xfs" ]] \
  || die "approved root is not on XFS"
[[ "$PROJECT_DIR_INPUT" != /mnt/ceph2 && "$PROJECT_DIR_INPUT" != /mnt/ceph2/* ]] \
  || die "the Ceph mount is outside the approved boundary"

PROJECT_DIR="$(readlink -f -- "$PROJECT_DIR_INPUT")"
[[ "$PROJECT_DIR" == "$PROJECT_DIR_INPUT" ]] \
  || die "project directory must be an absolute canonical path"
[[ "$PROJECT_DIR" == "$CONTACT_C0_ROOT/project/"* ]] \
  || die "project directory escapes the approved project root"
[[ -d "$PROJECT_DIR" ]] || die "project directory does not exist"
refuse_symlink "$PROJECT_DIR"
[[ -z "$(find "$PROJECT_DIR" -type l -print -quit)" ]] \
  || die "project snapshot contains a symbolic link"

MARKER_SOURCE_REVISION=""
read_marker "$PROJECT_DIR/.source-revision" MARKER_SOURCE_REVISION
read_marker "$PROJECT_DIR/.source-tree" SOURCE_TREE
read_marker "$PROJECT_DIR/.archive-sha256" ARCHIVE_SHA256
read_marker "$PROJECT_DIR/.snapshot-sha256" SNAPSHOT_SHA256
[[ "$MARKER_SOURCE_REVISION" == "$SOURCE_REVISION" ]] \
  || die "source revision differs from the deployed marker"
[[ "$SOURCE_TREE" == "$SOURCE_TREE_INPUT" ]] \
  || die "source tree differs from the explicit launcher input"
[[ "$SNAPSHOT_SHA256" == "$SNAPSHOT_SHA256_INPUT" ]] \
  || die "snapshot SHA-256 differs from the explicit launcher input"
[[ "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "archive marker is not SHA-256"
[[ "$PROJECT_DIR" == "$CONTACT_C0_ROOT/project/$SOURCE_TREE" ]] \
  || die "project path must exactly equal project/source-tree"

readonly ASSET_ROOT="$CONTACT_C0_ROOT/assets"
readonly ISAACLAB_ROOT="$CONTACT_C0_ROOT/IsaacLab"
readonly DOWNLOADS_ROOT="$CONTACT_C0_ROOT/downloads"
readonly SOURCE_ARCHIVE="$DOWNLOADS_ROOT/$ARCHIVE_SHA256.tar"
readonly WORKER_PYTHON="$CONTACT_C0_ROOT/env/bin/python"
readonly MANIFEST="$PROJECT_DIR/configs/parity/contact_c0.json"
WORKER_SCRIPT="$PROJECT_DIR/scripts/run_contact_c0_ovphysx_worker.py"
WORKER_CUDA_SELECTOR="$GPU_INDEX"
if [[ -n "$CONTACT_EXPERIMENTAL_CAMPAIGN" ]]; then
  [[ "$GPU_INDEX" == 0 ]] || die "candidate requires Slurm query index 0"
  CONTACT_SLURM_BINDING="$(PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -B -c \
    'import json; from wave_asset_qa.contact.slurm_binding import capture_binding; print(json.dumps(capture_binding(),sort_keys=True))')"
  WORKER_SCRIPT="$PROJECT_DIR/scripts/run_contact_async_worker.py"
  WORKER_EXTRA=(--backend ovphysx)
  # Keep the pinned adapter's numeric selector contract. The single-device
  # Slurm inventory above binds logical 0 to the expected physical UUID.
  WORKER_CUDA_SELECTOR="0"
fi
export CONTACT_EXPERIMENTAL_CAMPAIGN CONTACT_SLURM_BINDING
readonly NATIVE_HELPER_SOURCE="$PROJECT_DIR/$NATIVE_HELPER_SOURCE_RELATIVE"
require_real_directory "$CONTACT_C0_ROOT/project" "project"
require_real_directory "$PROJECT_DIR" "project snapshot"
require_real_directory "$DOWNLOADS_ROOT" "downloads"
require_real_directory "$ASSET_ROOT" "asset"
require_real_directory "$ISAACLAB_ROOT" "IsaacLab"
[[ -x "$WORKER_PYTHON" ]] || die "pinned Python environment is missing"
worker_python_real="$(readlink -f "$WORKER_PYTHON")"
[[ "$worker_python_real" == "$CONTACT_C0_ROOT/env/"* \
  || "$worker_python_real" == "$CONTACT_C0_ROOT/toolchain/python/"* ]] \
  || die "worker Python resolves outside the approved root"
for source_file in "$MANIFEST" "$WORKER_SCRIPT" "$NATIVE_HELPER_SOURCE"; do
  [[ -f "$source_file" && ! -L "$source_file" ]] \
    || die "required immutable source file is missing: $(basename "$source_file")"
done
[[ "$(readlink -f -- "$0")" == "$PROJECT_DIR/scripts/run_contact_c0_remote.sh" ]] \
  || die "launcher must execute from the selected immutable project snapshot"

readonly RUN_RESULTS_ROOT="$CONTACT_C0_ROOT/results"
for writable_path in \
  "$RUN_RESULTS_ROOT" \
  "$CONTACT_C0_ROOT/cache" \
  "$CONTACT_C0_ROOT/cache/cuda" \
  "$CONTACT_C0_ROOT/cache/pip" \
  "$CONTACT_C0_ROOT/cache/pycache" \
  "$CONTACT_C0_ROOT/cache/torch_extensions" \
  "$CONTACT_C0_ROOT/cache/triton" \
  "$CONTACT_C0_ROOT/cache/uv" \
  "$CONTACT_C0_ROOT/cache/warp" \
  "$CONTACT_C0_ROOT/cache/xdg" \
  "$CONTACT_C0_ROOT/config" \
  "$CONTACT_C0_ROOT/data" \
  "$CONTACT_C0_ROOT/state" \
  "$CONTACT_C0_ROOT/tmp"; do
  require_writable_directory "$writable_path"
done

verify_source_archive
verify_physx_sdk_archive
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

actual_asset_lf_sha256="$(ASSET_TREE="$ASSET_ROOT/wave_01" \
  "$WORKER_PYTHON" -P -B -S - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["ASSET_TREE"])
digest = hashlib.sha256()
files = sorted(
    (p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts),
    key=lambda p: p.relative_to(root).as_posix(),
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

RUN_ROOT="$RUN_RESULTS_ROOT/$RUN_ID"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] \
  || die "run directory already exists; refusing overwrite or automatic retry"

# A busy card must leave no run directory or lock-file mtime change.
if ! query_idle_gpu_state; then
  die "selected GPU is not the pinned, strictly idle Server 6 device"
fi
SELECTED_GPU_UUID="$LAST_GPU_UUID"

refuse_symlink "$CONTACT_C0_ROOT/.contact-c0-run.lock"
exec 9>"$CONTACT_C0_ROOT/.contact-c0-run.lock"
flock -n 9 || die "another Contact C0 launcher holds the exclusive run lock"

mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/launcher" "$RUN_ROOT/cases"
RUN_CREATED=true
NATIVE_ROOT="$RUN_ROOT/launcher/native"
NATIVE_INCLUDE_ROOT="$NATIVE_ROOT/include"
NATIVE_HELPER_PATH="$NATIVE_ROOT/$NATIVE_HELPER_FILENAME"
NATIVE_HEADERS_MANIFEST="$NATIVE_ROOT/headers-manifest.json"
NATIVE_BUILD_PROVENANCE="$NATIVE_ROOT/build-provenance.json"
mkdir "$NATIVE_ROOT" "$NATIVE_INCLUDE_ROOT"
for run_write_path in \
  "$RUN_ROOT" "$RUN_ROOT/launcher" "$RUN_ROOT/cases" \
  "$NATIVE_ROOT" "$NATIVE_INCLUDE_ROOT"; do
  require_writable_directory "$run_write_path"
done
exec 1>"$RUN_ROOT/launcher/stdout.txt" 2>"$RUN_ROOT/launcher/stderr.txt"
GPU_STARTED_EPOCH="$(date +%s)"
if ! capture_gpu_state "$RUN_ROOT/launcher/gpu-selection.txt" "$SELECTED_GPU_UUID"; then
  die "selected GPU changed or ceased to be strictly idle after run creation"
fi

[[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] \
  || die "display variables must be unset for kit-less Contact C0"
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
export PYTHONPYCACHEPREFIX="$CONTACT_C0_ROOT/cache/pycache"
export PYTHONSAFEPATH=1
export PYTHONUTF8=1
# Keep the account HOME unchanged; all writable caches are explicitly scoped below.
export CUDA_CACHE_PATH="$CONTACT_C0_ROOT/cache/cuda"
export PIP_CACHE_DIR="$CONTACT_C0_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$CONTACT_C0_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$CONTACT_C0_ROOT/cache/triton"
export TMPDIR="$CONTACT_C0_ROOT/tmp"
export UV_CACHE_DIR="$CONTACT_C0_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$CONTACT_C0_ROOT/env"
export WARP_CACHE_PATH="$CONTACT_C0_ROOT/cache/warp"
export XDG_CACHE_HOME="$CONTACT_C0_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$CONTACT_C0_ROOT/config"
export XDG_DATA_HOME="$CONTACT_C0_ROOT/data"
export XDG_STATE_HOME="$CONTACT_C0_ROOT/state"
cd "$RUN_ROOT"

# The native readback bridge is built only from a fixed, pre-positioned SDK
# archive.  The archive is never downloaded or populated by this launcher.
"$WORKER_PYTHON" -P -B -S - \
  "$OVPHYSX_SDK_ARCHIVE" "$OVPHYSX_SDK_ARCHIVE_SHA256" \
  "$NATIVE_INCLUDE_ROOT" "$NATIVE_HEADERS_MANIFEST" <<'PY'
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

archive_path = Path(sys.argv[1]).resolve(strict=True)
expected_archive_sha256 = sys.argv[2]
include_root = Path(sys.argv[3]).resolve(strict=True)
manifest_path = Path(sys.argv[4])
if manifest_path.exists() or manifest_path.is_symlink():
    raise SystemExit("native header manifest destination already exists")
if any(include_root.iterdir()):
    raise SystemExit("native header extraction destination is not empty")


def file_sha256(handle) -> str:
    digest = sha256()
    handle.seek(0)
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def normalized_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise SystemExit("OVPhysX SDK archive contains an unsafe member name")
    raw_parts = name.split("/")
    if ".." in raw_parts:
        raise SystemExit("OVPhysX SDK archive contains parent traversal")
    parts = tuple(part for part in raw_parts if part not in ("", "."))
    if not parts:
        raise SystemExit("OVPhysX SDK archive contains an empty member path")
    return PurePosixPath(*parts)


before = archive_path.stat()
with archive_path.open("rb") as archive_handle:
    if file_sha256(archive_handle) != expected_archive_sha256:
        raise SystemExit("OVPhysX SDK archive hash changed before extraction")
    archive_handle.seek(0)
    with tarfile.open(fileobj=archive_handle, mode="r:gz") as sdk:
        members = sdk.getmembers()
        if not 1 <= len(members) <= 100000:
            raise SystemExit("OVPhysX SDK archive member count is unsafe")
        paths: dict[PurePosixPath, tarfile.TarInfo] = {}
        for member in members:
            path = normalized_member_path(member.name)
            if path in paths:
                raise SystemExit("OVPhysX SDK archive has duplicate normalized paths")
            paths[path] = member
        candidates = []
        for path, member in paths.items():
            if path.name != "PxScene.h" or not member.isfile():
                continue
            candidate = path.parent
            required = (
                candidate / "PxArticulationLink.h",
                candidate / "PxRigidBody.h",
                candidate / "foundation" / "PxPhysicsVersion.h",
            )
            if all(item in paths and paths[item].isfile() for item in required):
                candidates.append(candidate)
        if len(candidates) != 1:
            raise SystemExit("OVPhysX SDK archive does not have one exact PhysX include root")
        archive_include_root = candidates[0]
        selected: list[tuple[PurePosixPath, tarfile.TarInfo]] = []
        for path, member in paths.items():
            try:
                relative = path.relative_to(archive_include_root)
            except ValueError:
                continue
            if relative.suffix.lower() not in {".h", ".hpp", ".inl"}:
                continue
            if not member.isfile():
                raise SystemExit("OVPhysX SDK archive header is not a regular file")
            selected.append((relative, member))
        selected.sort(key=lambda item: item[0].as_posix())
        if not 3 <= len(selected) <= 20000:
            raise SystemExit("OVPhysX SDK header count is unsafe")
        total_bytes = sum(member.size for _, member in selected)
        if total_bytes <= 0 or total_bytes > 256 * 1024 * 1024:
            raise SystemExit("OVPhysX SDK header bytes exceed the extraction cap")
        records = []
        for relative, member in selected:
            if member.size < 0 or member.size > 16 * 1024 * 1024:
                raise SystemExit("OVPhysX SDK archive has an oversized header")
            destination = include_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = sdk.extractfile(member)
            if source is None:
                raise SystemExit("OVPhysX SDK header cannot be read")
            digest = sha256()
            remaining = member.size
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with source, os.fdopen(descriptor, "wb") as output:
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise SystemExit("OVPhysX SDK header is truncated")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                if source.read(1):
                    raise SystemExit("OVPhysX SDK header exceeds its declared size")
                output.flush()
                os.fsync(output.fileno())
            records.append({
                "path": relative.as_posix(),
                "size": member.size,
                "sha256": digest.hexdigest(),
            })
    if file_sha256(archive_handle) != expected_archive_sha256:
        raise SystemExit("OVPhysX SDK archive hash changed during extraction")
after = archive_path.stat()
if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
    after.st_dev,
    after.st_ino,
    after.st_size,
    after.st_mtime_ns,
):
    raise SystemExit("OVPhysX SDK archive identity changed during extraction")

root_sha256 = sha256(
    json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
payload = {
    "schema_version": 1,
    "sdk_archive_path": archive_path.as_posix(),
    "sdk_archive_sha256": expected_archive_sha256,
    "include_root_in_archive": archive_include_root.as_posix(),
    "file_count": len(records),
    "total_bytes": total_bytes,
    "records": records,
    "root_sha256": root_sha256,
}
descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
    json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
PY
verify_physx_sdk_archive

NATIVE_COMPILER="$(readlink -f -- "$(command -v g++)")"
[[ "$NATIVE_COMPILER" == /usr/bin/* && -x "$NATIVE_COMPILER" && ! -L "$NATIVE_COMPILER" ]] \
  || die "g++ does not resolve to a regular executable under /usr/bin"
NATIVE_COMPILER_SHA256="$(sha256sum -- "$NATIVE_COMPILER" | awk '{print $1}')"
NATIVE_COMPILER_VERSION="$("$NATIVE_COMPILER" --version)"
[[ -n "$NATIVE_COMPILER_VERSION" ]] || die "g++ version evidence is empty"
NATIVE_SOURCE_SHA256="$(sha256sum -- "$NATIVE_HELPER_SOURCE" | awk '{print $1}')"
native_build_command=(
  "$NATIVE_COMPILER"
  -std=c++17
  -O2
  -fPIC
  -shared
  -Wall
  -Wextra
  -Werror
  -I "$NATIVE_INCLUDE_ROOT"
  "$NATIVE_HELPER_SOURCE"
  -o "$NATIVE_HELPER_PATH"
)
"${native_build_command[@]}"
[[ -f "$NATIVE_HELPER_PATH" && ! -L "$NATIVE_HELPER_PATH" ]] \
  || die "g++ did not create a regular native PhysX CCD helper"
NATIVE_HELPER_SHA256="$(sha256sum -- "$NATIVE_HELPER_PATH" | awk '{print $1}')"
NATIVE_HEADERS_MANIFEST_SHA256="$(sha256sum -- "$NATIVE_HEADERS_MANIFEST" | awk '{print $1}')"

CONTACT_NATIVE_COMPILER_VERSION="$NATIVE_COMPILER_VERSION" \
  "$WORKER_PYTHON" -P -B -S - \
  "$NATIVE_BUILD_PROVENANCE" "$NATIVE_HEADERS_MANIFEST" \
  "$OVPHYSX_SDK_ARCHIVE" "$OVPHYSX_SDK_ARCHIVE_SHA256" \
  "$NATIVE_INCLUDE_ROOT" "$NATIVE_HEADERS_MANIFEST_SHA256" \
  "$NATIVE_HELPER_SOURCE_RELATIVE" "$NATIVE_HELPER_SOURCE" "$NATIVE_SOURCE_SHA256" \
  "$NATIVE_COMPILER" "$NATIVE_COMPILER_SHA256" "$NATIVE_HELPER_PATH" \
  "$NATIVE_HELPER_SHA256" <<'PY'
import json
import os
from pathlib import Path
import sys

(
    destination_arg,
    headers_manifest_arg,
    archive_arg,
    archive_sha256,
    include_arg,
    headers_manifest_sha256,
    source_relative,
    source_arg,
    source_sha256,
    compiler_arg,
    compiler_sha256,
    output_arg,
    output_sha256,
) = sys.argv[1:]
destination = Path(destination_arg)
headers_manifest = json.loads(Path(headers_manifest_arg).read_text(encoding="utf-8"))
build_argv = [
    compiler_arg,
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-I",
    include_arg,
    source_arg,
    "-o",
    output_arg,
]
payload = {
    "schema_version": 1,
    "helper_id": "waveqa_physx_ccd_readback",
    "sdk_archive": {
        "package": "ovphysx",
        "version": "0.4.13",
        "path": archive_arg,
        "sha256": archive_sha256,
    },
    "headers": {
        "directory": include_arg,
        "manifest_path": headers_manifest_arg,
        "manifest_sha256": headers_manifest_sha256,
        "include_root_in_archive": headers_manifest["include_root_in_archive"],
        "file_count": headers_manifest["file_count"],
        "total_bytes": headers_manifest["total_bytes"],
        "root_sha256": headers_manifest["root_sha256"],
    },
    "source": {
        "relative_path": source_relative,
        "path": source_arg,
        "sha256": source_sha256,
    },
    "compiler": {
        "path": compiler_arg,
        "sha256": compiler_sha256,
        "version": os.environ["CONTACT_NATIVE_COMPILER_VERSION"],
    },
    "build": {
        "argv": build_argv,
        "output_path": output_arg,
        "output_sha256": output_sha256,
    },
}
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
    json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
PY
NATIVE_BUILD_PROVENANCE_SHA256="$(sha256sum -- "$NATIVE_BUILD_PROVENANCE" | awk '{print $1}')"
verify_native_helper

environment_summary="$("$WORKER_PYTHON" -P -B - <<'PY'
from importlib import metadata
import json
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
        raise SystemExit(f"unexpected {package} version")
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

kit_runtime = bool(has_kit())
if kit_runtime:
    raise SystemExit("Isaac Lab reports a Kit runtime")
kit_modules = sorted(
    name for name in sys.modules
    if name == "isaacsim" or name.startswith("isaacsim.")
    or name == "omni.kit" or name.startswith("omni.kit.")
)
renderer_modules = sorted(
    name for name in sys.modules
    if name == "omni.renderer" or name.startswith("omni.renderer.")
)
# ``isaaclab.sensors.camera`` is a passive transitive namespace import in the
# pinned Isaac Lab build.  The launcher-level camera-runtime boundary is the
# absence of Replicator and synthetic-data modules; each formal worker also
# proves zero UsdGeom.Camera prims and an exact ContactSensor-only inventory.
camera_runtime_modules = sorted(
    name for name in sys.modules
    if name == "omni.replicator" or name.startswith("omni.replicator.")
    or name == "omni.syntheticdata" or name.startswith("omni.syntheticdata.")
)
forbidden = sorted({*kit_modules, *renderer_modules, *camera_runtime_modules})
if forbidden:
    raise SystemExit(
        f"forbidden Kit/renderer/camera modules loaded: {forbidden[:10]}"
    )
runtime_boundary = {
    "kitless": not kit_runtime and not kit_modules,
    "renderer": bool(renderer_modules),
    "camera": bool(camera_runtime_modules),
}
print(json.dumps({
    "python": sys.version.split()[0],
    "packages": expected,
    "torch_cuda_build": torch.version.cuda,
    **runtime_boundary,
}, sort_keys=True, separators=(",", ":")))
PY
)"

matrix_output="$(PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -B - \
  "$MANIFEST" "$RUN_ROOT/launcher/matrix.json" <<'PY'
import json
import os
from pathlib import Path
import sys

from wave_asset_qa.contact.bundle import sha256_file
from wave_asset_qa.contact.scenarios import (
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)
from wave_asset_qa.parity.contracts import Simulator

manifest_path = Path(sys.argv[1]).resolve(strict=True)
destination = Path(sys.argv[2])
manifest = load_contact_manifest(manifest_path)
if manifest.schema_version != 1 or manifest.manifest_id != "wavesimparity-contact-c0":
    raise SystemExit("Contact C0 manifest identity mismatch")
if manifest.provenance.commit != "6eea427eb24189519f32b9f21674cd534d3f973c":
    raise SystemExit("manifest asset commit mismatch")
if manifest.provenance.asset_git_tree != "bb00a9d5527b8a76de576ce876ebece67d8ffde1":
    raise SystemExit("manifest asset Git tree mismatch")
if manifest.provenance.canonical_lf_asset_tree_sha256 != "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad":
    raise SystemExit("manifest asset LF tree mismatch")
cases = [case for case in expand_contact_cases(manifest) if case.simulator is Simulator.OVPHYSX]
if len(cases) != 16 or len({case.case_id for case in cases}) != 16:
    raise SystemExit(f"expected exactly 16 OVPhysX cases, found {len(cases)}")
semantic_sha = contact_manifest_sha256(manifest)
payload = {
    "schema_version": 1,
    "manifest_id": manifest.manifest_id,
    "manifest_file_sha256": sha256_file(manifest_path),
    "manifest_semantic_sha256": semantic_sha,
    "backend": "ovphysx",
    "case_ids": [case.case_id for case in cases],
}
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("manifest_sha256=" + semantic_sha)
for case in cases:
    print(case.case_id)
PY
)"
mapfile -t matrix_lines <<< "$matrix_output"
[[ "${#matrix_lines[@]}" -eq $((EXPECTED_OVPHYSX_CASE_COUNT + 1)) ]] \
  || die "canonical matrix must contain one hash and 16 cases"
MANIFEST_SHA256="${matrix_lines[0]#manifest_sha256=}"
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "manifest semantic SHA-256 is invalid"
CASE_IDS=("${matrix_lines[@]:1}")
for case_id in "${CASE_IDS[@]}"; do
  safe_identifier "$case_id" || die "unsafe canonical case ID: $case_id"
  [[ "$case_id" == ovphysx.* ]] || die "remote matrix contains a non-OVPhysX case"
done

CONTACT_PROVENANCE_RUN="$RUN_ID" CONTACT_PROVENANCE_SESSION="$SESSION_ID" \
  CONTACT_PROVENANCE_REVISION="$SOURCE_REVISION" CONTACT_PROVENANCE_TREE="$SOURCE_TREE" \
  CONTACT_PROVENANCE_ARCHIVE="$ARCHIVE_SHA256" CONTACT_PROVENANCE_SNAPSHOT="$SNAPSHOT_SHA256" \
  CONTACT_PROVENANCE_MANIFEST="$MANIFEST_SHA256" CONTACT_PROVENANCE_GPU_INDEX="$GPU_INDEX" \
  CONTACT_PROVENANCE_GPU_UUID="$SELECTED_GPU_UUID" CONTACT_PROVENANCE_GPU_NAME="$LAST_GPU_NAME" \
  CONTACT_PROVENANCE_DRIVER="$LAST_GPU_DRIVER" CONTACT_PROVENANCE_ENV="$environment_summary" \
  CONTACT_PROVENANCE_NATIVE_BUILD="$NATIVE_BUILD_PROVENANCE_SHA256" \
  CONTACT_PROVENANCE_NATIVE_PATH="$NATIVE_HELPER_PATH" \
  CONTACT_PROVENANCE_NATIVE_SHA="$NATIVE_HELPER_SHA256" \
  "$WORKER_PYTHON" -P -B -S - "$RUN_ROOT/launcher/provenance.json" <<'PY'
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "campaign": os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") or "contact_c0",
    "backend": "ovphysx",
    "run_id": os.environ["CONTACT_PROVENANCE_RUN"],
    "session_id": os.environ["CONTACT_PROVENANCE_SESSION"],
    "source_revision": os.environ["CONTACT_PROVENANCE_REVISION"],
    "source_tree": os.environ["CONTACT_PROVENANCE_TREE"],
    "source_archive_sha256": os.environ["CONTACT_PROVENANCE_ARCHIVE"],
    "snapshot_sha256": os.environ["CONTACT_PROVENANCE_SNAPSHOT"],
    "manifest_id": "wavesimparity-contact-c0",
    "manifest_sha256": os.environ["CONTACT_PROVENANCE_MANIFEST"],
    "asset_commit": "6eea427eb24189519f32b9f21674cd534d3f973c",
    "asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
    "asset_tree_sha256": "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad",
    "isaaclab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
    "selected_gpu": {
        ("query_index" if os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") else "physical_index"): int(os.environ["CONTACT_PROVENANCE_GPU_INDEX"]),
        "uuid": os.environ["CONTACT_PROVENANCE_GPU_UUID"],
        "name": os.environ["CONTACT_PROVENANCE_GPU_NAME"],
        "driver_version": os.environ["CONTACT_PROVENANCE_DRIVER"],
        "worker_visible_device": "cuda:0",
    },
    "environment": json.loads(os.environ["CONTACT_PROVENANCE_ENV"]),
    "native_physx_ccd_helper": {
        "build_provenance_path": (
            "/data/home/exampleuser/sharpa-wave-asset-qa-gate0/results/"
            + os.environ["CONTACT_PROVENANCE_RUN"]
            + "/launcher/native/build-provenance.json"
        ),
        "build_provenance_sha256": os.environ["CONTACT_PROVENANCE_NATIVE_BUILD"],
        "helper_path": os.environ["CONTACT_PROVENANCE_NATIVE_PATH"],
        "helper_sha256": os.environ["CONTACT_PROVENANCE_NATIVE_SHA"],
    },
    "claim_boundary": {
        "synthetic_fixture_only": True,
        "kitless_ovphysx": True,
        "full_isaac_sim": False,
        "renderer": False,
        "camera": False,
        "dual_hand": False,
        "floating_base": False,
        "hardware": False,
        "sim2real": False,
    },
    "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
if os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN"):
    payload["slurm_binding"] = json.loads(os.environ["CONTACT_SLURM_BINDING"])
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

validate_collected_case() {
  local case_dir="$1"
  local case_id="$2"
  PYTHONPATH="$PROJECT_DIR/src" "$WORKER_PYTHON" -P -B - \
    "$case_dir" "$case_id" "$MANIFEST" "$SESSION_ID" \
    "$SOURCE_REVISION" "$SOURCE_TREE" "$ASSET_LF_SHA256" <<'PY'
import os
import platform
import sys
from pathlib import Path
import re
import stat

from wave_asset_qa.contact.bundle import (
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
    verify_adapter_private_evidence,
)
from wave_asset_qa.contact import worker as contact_worker_module
from wave_asset_qa.contact.records import canonical_contact_run_json
from wave_asset_qa.contact.runner import load_and_validate_contact_run, select_contact_case
from wave_asset_qa.contact.scenarios import contact_manifest_sha256, load_contact_manifest
from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.process_identity import (
    os_process_identity_sha256,
    validate_os_process_identity,
)

case_dir = Path(sys.argv[1]).resolve(strict=True)
case_id, manifest_arg, session_id, source_revision, source_tree, asset_sha = sys.argv[2:]
fixed_names = {
    f"{case_id}.run.json",
    "private-process.json",
    "private-adapter-evidence.json",
    "stdout.txt",
    "stderr.txt",
    "exit-code.txt",
    "command.json",
    "owned-process-group.txt",
    "process.json",
    "worker-group-binding.json",
    "gpu-monitor.txt",
    "gpu-preflight-01.txt",
    "gpu-preflight-02.txt",
    "gpu-preflight-03.txt",
}
candidate = os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN")
if candidate:
    fixed_names.add("private-campaign.json")
observed_names = set()
for path in case_dir.iterdir():
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit(f"case evidence contains a non-regular entry: {path.name}")
    observed_names.add(path.name)
postflight_names = sorted(
    name for name in observed_names
    if re.fullmatch(r"gpu-postflight-0[123]\.txt", name)
)
expected_postflight = [
    f"gpu-postflight-{index:02d}.txt"
    for index in range(1, len(postflight_names) + 1)
]
if not postflight_names or postflight_names != expected_postflight:
    raise SystemExit("GPU postflight evidence is missing or noncontiguous")
if observed_names != fixed_names | set(postflight_names):
    raise SystemExit("completed case evidence file names are not exact")
if (case_dir / "exit-code.txt").read_text(encoding="ascii") != "0\n":
    raise SystemExit("completed case exit code is not exactly zero")
manifest = load_contact_manifest(manifest_arg)
case = select_contact_case(manifest, case_id, simulator=Simulator.OVPHYSX)
run_path = case_dir / f"{case_id}.run.json"
process_path = case_dir / "private-process.json"
adapter_path = case_dir / "private-adapter-evidence.json"
binding_path = case_dir / "worker-group-binding.json"
group_path = case_dir / "process.json"
owned_group_path = case_dir / "owned-process-group.txt"
for path in (run_path, process_path, adapter_path, binding_path, group_path, owned_group_path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"required case evidence is absent: {path.name}")
process = read_json_strict(process_path)
expected_process = {
    "schema_version", "visibility", "case_id", "backend", "session_id",
    "source_revision", "source_tree", "manifest_file_sha256",
    "manifest_semantic_sha256", "asset_tree_sha256", "python_version",
    "python_executable_realpath", "worker_module_realpath",
    "worker_module_sha256", "os_process_identity",
    "fresh_process_identity_sha256",
}
if candidate:
    expected_process.add("experimental_campaign")
if not isinstance(process, dict) or set(process) != expected_process:
    raise SystemExit("private process record fields are not exact")
expected_values = {
    "schema_version": 1,
    "visibility": "private_not_for_publication",
    "case_id": case_id,
    "backend": "ovphysx",
    "session_id": session_id,
    "source_revision": source_revision,
    "source_tree": source_tree,
    "manifest_file_sha256": sha256_file(manifest_arg),
    "manifest_semantic_sha256": contact_manifest_sha256(manifest),
    "asset_tree_sha256": asset_sha,
    "python_version": platform.python_version(),
    "python_executable_realpath": os.path.realpath(sys.executable),
    "worker_module_realpath": os.path.realpath(contact_worker_module.__file__),
    "worker_module_sha256": sha256_file(contact_worker_module.__file__),
}
if candidate:
    expected_values["experimental_campaign"] = candidate
for key, expected in expected_values.items():
    if process.get(key) != expected:
        raise SystemExit(f"private process identity mismatch: {key}")
identity = validate_os_process_identity(process.get("os_process_identity"))
if identity.get("platform") != "posix":
    raise SystemExit("remote worker identity is not POSIX")
fresh_hash = os_process_identity_sha256(identity)
if process.get("fresh_process_identity_sha256") != fresh_hash:
    raise SystemExit("private fresh-process hash mismatch")
binding = read_json_strict(binding_path)
group = read_json_strict(group_path)
if not isinstance(group, dict) or set(group) != {
    "schema_version", "visibility", "case_id", "transport_pid",
    "owned_process_group_id",
}:
    raise SystemExit("owned process-group record fields are not exact")
transport_pid = group.get("transport_pid")
owned_pgid = group.get("owned_process_group_id")
if (
    group.get("schema_version") != 1
    or group.get("visibility") != "private_not_for_publication"
    or group.get("case_id") != case_id
    or isinstance(transport_pid, bool)
    or not isinstance(transport_pid, int)
    or transport_pid <= 0
    or owned_pgid != transport_pid
):
    raise SystemExit("owned process-group record is invalid")
if owned_group_path.read_text(encoding="ascii") != f"{owned_pgid}\n":
    raise SystemExit("owned process-group handshake differs from its record")
if not isinstance(binding, dict) or binding != {
    "schema_version": 1,
    "visibility": "private_not_for_publication",
    "case_id": case_id,
    "worker_pid": identity["pid"],
    "owned_process_group_id": owned_pgid,
    "fresh_process_identity_sha256": fresh_hash,
}:
    raise SystemExit("worker-to-process-group binding mismatch")
pgid = binding.get("owned_process_group_id")
if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0:
    raise SystemExit("owned process-group ID is invalid")
run = load_and_validate_contact_run(
    run_path,
    manifest=manifest,
    case=case,
    source_revision=source_revision,
    source_tree=source_tree,
    fresh_process_identity_sha256=fresh_hash,
)
if run_path.read_text(encoding="utf-8") != canonical_contact_run_json(run) + "\n":
    raise SystemExit("completed run payload is not canonical JSON")
adapter = read_json_strict(adapter_path)
verify_adapter_private_evidence(adapter, run)
fixture = adapter.get("fixture_overlay")
if not isinstance(fixture, dict):
    raise SystemExit("fixture preimage is absent")
if canonical_json_sha256(fixture.get("collision_inventory_hash_preimage")) != run.fixture_readback.get("collision_inventory_sha256"):
    raise SystemExit("collision inventory preimage hash mismatch")
runtime = adapter.get("runtime_fingerprint")
if not isinstance(runtime, dict):
    raise SystemExit("runtime fingerprint preimage is absent")
for key, expected in {
    "backend": "ovphysx", "device": "cuda:0", "kitless": True,
    "renderer": False, "camera": False,
}.items():
    if runtime.get(key) != expected:
        raise SystemExit(f"runtime boundary mismatch: {key}")
unexpected = adapter.get("unexpected_pair_observation")
if not isinstance(unexpected, dict) or unexpected.get("unexpected_pairs") != []:
    raise SystemExit("exclusive synthetic pair evidence is incomplete")
if candidate:
    from wave_asset_qa.contact.async_campaign import load_candidate_case
    load_candidate_case(case_dir, manifest=manifest, case=case,
                        source_revision=source_revision, source_tree=source_tree)
print(fresh_hash)
PY
}

check_root_budget
for case_id in "${CASE_IDS[@]}"; do
  CURRENT_CASE="$case_id"
  verify_snapshot "before-$case_id"
  now_epoch="$(date +%s)"
  gpu_elapsed=$((now_epoch - GPU_STARTED_EPOCH))
  remaining=$((GLOBAL_GPU_TIMEOUT_S - gpu_elapsed))
  usable=$((remaining - GPU_KILL_GRACE_S - GPU_DEADLINE_SAFETY_MARGIN_S))
  [[ "$usable" -gt 0 ]] \
    || die "global four-hour GPU budget cannot fit cleanup and safety margins"
  envelope="$CASE_ENVELOPE_TIMEOUT_S"
  [[ "$usable" -ge "$envelope" ]] || envelope="$usable"

  case_dir="$RUN_ROOT/cases/$case_id"
  [[ ! -e "$case_dir" && ! -L "$case_dir" ]] \
    || die "case evidence directory already exists"
  mkdir "$case_dir"
  require_writable_directory "$case_dir"

  # Both checks must be strictly idle and observe the originally selected UUID.
  if ! capture_gpu_state "$case_dir/gpu-preflight-01.txt" "$SELECTED_GPU_UUID"; then
    die "GPU is not strictly idle at first preflight for $case_id"
  fi
  if ! capture_gpu_state "$case_dir/gpu-preflight-02.txt" "$SELECTED_GPU_UUID"; then
    die "GPU is not strictly idle at immediate second preflight for $case_id"
  fi
  verify_native_helper

  command_path="$case_dir/command.json"
  CONTACT_COMMAND_CASE="$case_id" CONTACT_COMMAND_SESSION="$SESSION_ID" \
    CONTACT_COMMAND_REVISION="$SOURCE_REVISION" CONTACT_COMMAND_TREE="$SOURCE_TREE" \
    CONTACT_COMMAND_ASSET_SHA="$ASSET_LF_SHA256" CONTACT_COMMAND_MANIFEST="$MANIFEST" \
    CONTACT_COMMAND_OUTPUT="$case_dir" CONTACT_COMMAND_PYTHON="$WORKER_PYTHON" \
    CONTACT_COMMAND_SCRIPT="$WORKER_SCRIPT" CONTACT_COMMAND_GPU="$WORKER_CUDA_SELECTOR" \
    CONTACT_COMMAND_NATIVE_PATH="$NATIVE_HELPER_PATH" \
    CONTACT_COMMAND_NATIVE_SHA="$NATIVE_HELPER_SHA256" \
    "$WORKER_PYTHON" -P -B -S - "$command_path" <<'PY'
import json
import os
from pathlib import Path
import sys

argv = [
    os.environ["CONTACT_COMMAND_PYTHON"], "-P", os.environ["CONTACT_COMMAND_SCRIPT"],
    *(["--backend", "ovphysx"] if os.environ.get("CONTACT_EXPERIMENTAL_CAMPAIGN") else []),
    "--asset-root", "/data/home/exampleuser/sharpa-wave-asset-qa-gate0/assets",
    "--manifest", os.environ["CONTACT_COMMAND_MANIFEST"],
    "--case-id", os.environ["CONTACT_COMMAND_CASE"],
    "--output-dir", os.environ["CONTACT_COMMAND_OUTPUT"],
    "--session-id", os.environ["CONTACT_COMMAND_SESSION"],
    "--source-revision", os.environ["CONTACT_COMMAND_REVISION"],
    "--source-tree", os.environ["CONTACT_COMMAND_TREE"],
    "--asset-tree-sha256", os.environ["CONTACT_COMMAND_ASSET_SHA"],
    "--device", "cuda:0",
]
payload = {
    "schema_version": 1,
    "case_id": os.environ["CONTACT_COMMAND_CASE"],
    "argv": argv,
    "environment": {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": os.environ["CONTACT_COMMAND_GPU"],
        "WAVEQA_PHYSX_CCD_HELPER_PATH": os.environ["CONTACT_COMMAND_NATIVE_PATH"],
        "WAVEQA_PHYSX_CCD_HELPER_SHA256": os.environ["CONTACT_COMMAND_NATIVE_SHA"],
        "worker_device": "cuda:0",
        "DISPLAY": None,
        "WAYLAND_DISPLAY": None,
    },
}
path = Path(sys.argv[1])
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

  # Close the command-recording race with another strict check immediately
  # before the owned process group is created.
  if ! capture_gpu_state "$case_dir/gpu-preflight-03.txt" "$SELECTED_GPU_UUID"; then
    die "GPU is not strictly idle at final launch preflight for $case_id"
  fi
  verify_native_helper

  case_pid_file="$case_dir/owned-process-group.txt"
  case_command=(
    timeout --foreground --signal=TERM --kill-after="${GPU_KILL_GRACE_S}s" "${envelope}s"
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="$WORKER_CUDA_SELECTOR"
    WAVEQA_PHYSX_CCD_HELPER_PATH="$NATIVE_HELPER_PATH"
    WAVEQA_PHYSX_CCD_HELPER_SHA256="$NATIVE_HELPER_SHA256"
    PYTHONPATH="$PROJECT_DIR/src"
    timeout --foreground --signal=TERM --kill-after="${GPU_KILL_GRACE_S}s" "${WORKER_TIMEOUT_S}s"
    "$WORKER_PYTHON" -P "$WORKER_SCRIPT" "${WORKER_EXTRA[@]}"
    --asset-root "$ASSET_ROOT"
    --manifest "$MANIFEST"
    --case-id "$case_id"
    --output-dir "$case_dir"
    --session-id "$SESSION_ID"
    --source-revision "$SOURCE_REVISION"
    --source-tree "$SOURCE_TREE"
    --asset-tree-sha256 "$ASSET_LF_SHA256"
    --device cuda:0
  )
  (
    ulimit -f "$MAX_ARTIFACT_BLOCKS"
    exec "$WORKER_PYTHON" -P -B -S - "$case_pid_file" "${case_command[@]}"
  ) > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt" <<'PY' &
import os
from pathlib import Path
import sys

pid_path = Path(sys.argv[1])
command = sys.argv[2:]
if not command:
    raise SystemExit("owned Contact C0 wrapper received no command")
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
    printf '%s\n' "$runner_exit_code" > "$case_dir/exit-code.txt"
    die "owned process group was not established for $case_id"
  fi
  IFS= read -r case_pgid < "$case_pid_file"
  observed_case_pgid="$(ps -o pgid= -p "$case_owner_pid" 2>/dev/null | awk '{$1=$1; print}')"
  if [[ ! "$case_pgid" =~ ^[1-9][0-9]*$ \
    || "$case_pgid" != "$case_owner_pid" \
    || "$observed_case_pgid" != "$case_pgid" ]]; then
    set +e
    wait "$case_owner_pid"
    runner_exit_code="$?"
    set -e
    printf '%s\n' "$runner_exit_code" > "$case_dir/exit-code.txt"
    die "case process-group identity is unsafe for $case_id"
  fi
  CONTACT_GROUP_CASE="$case_id" CONTACT_GROUP_PID="$case_owner_pid" \
    CONTACT_GROUP_PGID="$case_pgid" "$WORKER_PYTHON" -P -B -S - \
    "$case_dir/process.json" <<'PY'
import json
import os
from pathlib import Path
import sys

payload = {
    "schema_version": 1,
    "visibility": "private_not_for_publication",
    "case_id": os.environ["CONTACT_GROUP_CASE"],
    "transport_pid": int(os.environ["CONTACT_GROUP_PID"]),
    "owned_process_group_id": int(os.environ["CONTACT_GROUP_PGID"]),
}
path = Path(sys.argv[1])
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

  ACTIVE_CASE_PID="$case_owner_pid"
  ACTIVE_CASE_PGID="$case_pgid"
  foreign_gpu_process=false
  if ! monitor_owned_case \
    "$case_owner_pid" "$case_pgid" "$case_id" \
    "$case_dir/private-process.json" "$case_dir/worker-group-binding.json" \
    "$case_dir/gpu-monitor.txt" "$case_dir/foreign-gpu-process.txt"; then
    foreign_gpu_process=true
  fi
  set +e
  wait "$case_owner_pid"
  runner_exit_code="$?"
  set -e
  if ! terminate_owned_case "$case_owner_pid" "$case_pgid"; then
    die "owned process group did not terminate for $case_id"
  fi
  ACTIVE_CASE_PID=""
  ACTIVE_CASE_PGID=""
  printf '%s\n' "$runner_exit_code" > "$case_dir/exit-code.txt"

  if [[ "$foreign_gpu_process" == true ]]; then
    capture_gpu_state "$case_dir/gpu-postflight-foreign.txt" "$SELECTED_GPU_UUID" || true
    IFS= read -r foreign_reason < "$case_dir/foreign-gpu-process.txt" \
      || foreign_reason="foreign GPU process detected"
    die "$foreign_reason; only the owned case process group was terminated"
  fi

  postflight_ok=false
  for postflight_attempt in 1 2 3; do
    postflight_path="$(printf '%s/gpu-postflight-%02d.txt' "$case_dir" "$postflight_attempt")"
    if capture_gpu_state "$postflight_path" "$SELECTED_GPU_UUID"; then
      postflight_ok=true
      break
    fi
    [[ "$postflight_attempt" -eq 3 ]] || sleep 5
  done
  [[ "$postflight_ok" == true ]] \
    || die "GPU did not return to strictly idle after $case_id"
  [[ "$runner_exit_code" -eq 0 ]] \
    || die "worker returned $runner_exit_code for $case_id"

  fresh_process_id="$(validate_collected_case "$case_dir" "$case_id")"
  [[ "$fresh_process_id" =~ ^[0-9a-f]{64}$ ]] \
    || die "worker fresh-process identity is invalid for $case_id"
  [[ -z "${FRESH_PROCESS_IDS[$fresh_process_id]+present}" ]] \
    || die "a Python worker identity was reused across canonical cases"
  FRESH_PROCESS_IDS["$fresh_process_id"]="$case_id"
  COMPLETED_CASES=$((COMPLETED_CASES + 1))
  check_run_budget
  check_root_budget
done

[[ "$COMPLETED_CASES" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]] \
  || die "remote matrix did not complete all 16 OVPhysX cases"
[[ "${#FRESH_PROCESS_IDS[@]}" -eq "$EXPECTED_OVPHYSX_CASE_COUNT" ]] \
  || die "remote matrix did not prove 16 unique Python processes"
verify_native_helper
verify_snapshot "post-matrix"
check_run_budget
RUN_STATUS="completed"
SCIENTIFIC_VERDICT="pending_local_compare"
CURRENT_CASE=""
printf 'collected all 16 Contact C0 OVPhysX cases at %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
