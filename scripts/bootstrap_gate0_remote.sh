#!/usr/bin/env bash

set -euo pipefail
umask 077

# This script is meant to run only on the approved Server 6 account. It keeps
# every project-owned write below one explicit root and never uses sudo.
readonly GATE0_PARENT="/data/home/exampleuser"
readonly GATE0_ROOT="$GATE0_PARENT/sharpa-wave-asset-qa-gate0"
readonly UV_VERSION="0.12.5"
readonly UV_TARGET="x86_64-unknown-linux-gnu"
readonly UV_ARCHIVE_SHA256="68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2"
readonly PYTHON_VERSION="3.12.14"
readonly UV_LOCK_FILENAME="isaaclab-v3.0.0-beta2.patch1-ov.uv.lock"
readonly UV_LOCK_SHA256="cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522"
readonly USD_CORE_VERSION="25.11"
readonly USD_EXCHANGE_VERSION="3.0.0"
readonly ISAACLAB_TAG="v3.0.0-beta2.patch1"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
readonly ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
readonly ASSET_URL="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
readonly ASSET_COMMIT="6eea427eb24189519f32b9f21674cd534d3f973c"
readonly ASSET_GIT_TREE="bb00a9d5527b8a76de576ce876ebece67d8ffde1"
readonly V01_WINDOWS_ASSET_TREE_SHA256="9c2d71aec9f8fe77aeb03660f735eb55bc53b93ba595f6cbfe5cc4e8fbb67de4"
readonly LINUX_ASSET_TREE_SHA256="b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
readonly MAX_ROOT_BYTES=$((30 * 1024 * 1024 * 1024))

die() {
  echo "Refusing to continue: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

refuse_symlink() {
  [[ ! -L "$1" ]] || die "managed path is a symbolic link: $1"
}

verify_repository() {
  local path="$1"
  local expected_url="$2"
  local allowed_untracked="${3:-}"
  local status_line
  [[ -d "$path/.git" ]] || die "managed checkout is not a Git repository: $path"
  [[ "$(git -C "$path" remote get-url origin)" == "$expected_url" ]] \
    || die "unexpected origin URL in $path"
  while IFS= read -r status_line; do
    [[ -z "$status_line" ]] && continue
    if [[ -n "$allowed_untracked" && "$status_line" == "?? $allowed_untracked" ]]; then
      continue
    fi
    die "managed checkout is dirty: $path ($status_line)"
  done < <(git -C "$path" status --porcelain --untracked-files=all)
}

[[ "$(hostname)" == "example-gpu-node-6" ]] || die "expected Server 6 hostname example-gpu-node-6"
[[ "$(id -un)" == "exampleuser" ]] || die "expected user exampleuser"
[[ -d "$GATE0_PARENT" ]] || die "expected $GATE0_PARENT"

# Gate 0 is pinned to the observed Server 6 base image. Stop if the machine was
# re-imaged instead of silently installing against a different platform.
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] \
  || die "expected Ubuntu 22.04"
for command_name in awk curl df du findmnt flock git nvidia-smi readlink sha256sum stat tar; do
  require_command "$command_name"
done
[[ "$(findmnt -n -o TARGET -T "$GATE0_PARENT")" == "/data" ]] \
  || die "expected $GATE0_PARENT to live on the /data mount"
[[ "$(findmnt -n -o FSTYPE -T "$GATE0_PARENT")" == "xfs" ]] \
  || die "expected the /data mount to use XFS"
available_bytes="$(df -B1 --output=avail "$GATE0_PARENT" | awk 'NR == 2 {print $1}')"
[[ "$available_bytes" -ge $((50 * 1024 * 1024 * 1024)) ]] \
  || die "less than 50 GiB is available on /data"

refuse_symlink "$GATE0_ROOT"
if [[ -e "$GATE0_ROOT" && ! -d "$GATE0_ROOT" ]]; then
  die "$GATE0_ROOT exists but is not a directory"
fi
mkdir -p "$GATE0_ROOT"
[[ "$(readlink -f "$GATE0_ROOT")" == "$GATE0_ROOT" ]] \
  || die "approved root resolves outside its literal path"
[[ "$(stat -c '%U' "$GATE0_ROOT")" == "exampleuser" ]] \
  || die "approved root is not owned by exampleuser"
[[ -w "$GATE0_ROOT" ]] || die "approved root is not writable"

refuse_symlink "$GATE0_ROOT/.bootstrap.lock"
exec 9>"$GATE0_ROOT/.bootstrap.lock"
flock -n 9 || die "another bootstrap process holds the Gate 0 lock"

managed_dirs=(
  "$GATE0_ROOT/bootstrap"
  "$GATE0_ROOT/cache"
  "$GATE0_ROOT/cache/cuda"
  "$GATE0_ROOT/cache/pip"
  "$GATE0_ROOT/cache/pycache"
  "$GATE0_ROOT/cache/torch_extensions"
  "$GATE0_ROOT/cache/triton"
  "$GATE0_ROOT/cache/uv"
  "$GATE0_ROOT/cache/warp"
  "$GATE0_ROOT/cache/xdg"
  "$GATE0_ROOT/config"
  "$GATE0_ROOT/data"
  "$GATE0_ROOT/downloads"
  "$GATE0_ROOT/logs"
  "$GATE0_ROOT/project"
  "$GATE0_ROOT/results"
  "$GATE0_ROOT/state"
  "$GATE0_ROOT/tmp"
  "$GATE0_ROOT/toolchain"
  "$GATE0_ROOT/toolchain/bin"
)
for managed_dir in "${managed_dirs[@]}"; do
  refuse_symlink "$managed_dir"
  mkdir -p "$managed_dir"
  [[ "$(readlink -f "$managed_dir")" == "$managed_dir" ]] \
    || die "managed directory resolves outside its literal path: $managed_dir"
done

export CUDA_CACHE_PATH="$GATE0_ROOT/cache/cuda"
export PIP_CACHE_DIR="$GATE0_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$GATE0_ROOT/cache/pycache"
export TORCH_EXTENSIONS_DIR="$GATE0_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$GATE0_ROOT/cache/triton"
export TMPDIR="$GATE0_ROOT/tmp"
export UV_CACHE_DIR="$GATE0_ROOT/cache/uv"
export UV_MANAGED_PYTHON=1
export UV_NO_MODIFY_PATH=1
export UV_PYTHON_BIN_DIR="$GATE0_ROOT/toolchain/bin"
export UV_PYTHON_INSTALL_DIR="$GATE0_ROOT/toolchain/python"
export UV_PROJECT_ENVIRONMENT="$GATE0_ROOT/env"
export WARP_CACHE_PATH="$GATE0_ROOT/cache/warp"
export XDG_CACHE_HOME="$GATE0_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$GATE0_ROOT/config"
export XDG_DATA_HOME="$GATE0_ROOT/data"
export XDG_STATE_HOME="$GATE0_ROOT/state"

readonly UV_BIN="$GATE0_ROOT/toolchain/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  readonly UV_ARCHIVE="$GATE0_ROOT/downloads/uv-${UV_VERSION}-${UV_TARGET}.tar.gz"
  refuse_symlink "$UV_ARCHIVE"
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz" \
    -o "$UV_ARCHIVE"
  printf '%s  %s\n' "$UV_ARCHIVE_SHA256" "$UV_ARCHIVE" | sha256sum --check --status \
    || die "uv archive SHA-256 mismatch"
  tar -xzf "$UV_ARCHIVE" --strip-components=1 -C "$GATE0_ROOT/toolchain/bin"
fi
[[ "$("$UV_BIN" --version | awk '{print $2}')" == "$UV_VERSION" ]] \
  || die "unexpected uv version: $("$UV_BIN" --version)"

"$UV_BIN" python install "$PYTHON_VERSION"
refuse_symlink "$GATE0_ROOT/env"
if [[ ! -e "$GATE0_ROOT/env" ]]; then
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$GATE0_ROOT/env"
elif [[ ! -x "$GATE0_ROOT/env/bin/python" ]]; then
  die "existing environment does not contain an executable Python"
fi
GATE0_ENV="$GATE0_ROOT/env" "$GATE0_ROOT/env/bin/python" - <<'PY'
import os
from pathlib import Path
import sys

expected = Path(os.environ["GATE0_ENV"]).resolve()
actual = Path(sys.prefix).resolve()
if actual != expected or sys.version_info[:3] != (3, 12, 14):
    raise SystemExit(
        f"unexpected Gate 0 Python environment: prefix={actual}, version={sys.version.split()[0]}"
    )
PY

if [[ ! -d "$GATE0_ROOT/IsaacLab/.git" ]]; then
  [[ ! -e "$GATE0_ROOT/IsaacLab" ]] || die "IsaacLab path exists but is not a checkout"
  git clone --filter=blob:none --branch "$ISAACLAB_TAG" --depth 1 \
    "$ISAACLAB_URL" "$GATE0_ROOT/IsaacLab"
fi
refuse_symlink "$GATE0_ROOT/IsaacLab"
verify_repository "$GATE0_ROOT/IsaacLab" "$ISAACLAB_URL" "uv.lock"
git -C "$GATE0_ROOT/IsaacLab" fetch --depth 1 origin "$ISAACLAB_COMMIT"
git -C "$GATE0_ROOT/IsaacLab" checkout --detach "$ISAACLAB_COMMIT"
[[ "$(git -C "$GATE0_ROOT/IsaacLab" rev-parse HEAD)" == "$ISAACLAB_COMMIT" ]] \
  || die "Isaac Lab commit mismatch"
verify_repository "$GATE0_ROOT/IsaacLab" "$ISAACLAB_URL" "uv.lock"

asset_checkout_created=false
if [[ ! -d "$GATE0_ROOT/assets/.git" ]]; then
  [[ ! -e "$GATE0_ROOT/assets" ]] || die "assets path exists but is not a checkout"
  git clone --filter=blob:none --no-checkout "$ASSET_URL" "$GATE0_ROOT/assets"
  asset_checkout_created=true
fi
refuse_symlink "$GATE0_ROOT/assets"
[[ "$(git -C "$GATE0_ROOT/assets" remote get-url origin)" == "$ASSET_URL" ]] \
  || die "unexpected origin URL in $GATE0_ROOT/assets"
if [[ "$asset_checkout_created" == false ]]; then
  verify_repository "$GATE0_ROOT/assets" "$ASSET_URL"
fi
# The v0.1 hash records a Windows CRLF worktree. Gate 0 keeps it as reference,
# but verifies a clean LF checkout plus the platform-independent Git subtree.
git -C "$GATE0_ROOT/assets" config core.autocrlf false
git -C "$GATE0_ROOT/assets" config core.eol lf
git -C "$GATE0_ROOT/assets" sparse-checkout init --cone
git -C "$GATE0_ROOT/assets" sparse-checkout set \
  wave_01/dual_sharpa_wave \
  wave_01/left_sharpa_wave \
  wave_01/right_sharpa_wave \
  wave_01/sharpa_wave_float_base_urdf_usd
git -C "$GATE0_ROOT/assets" fetch --depth 1 origin "$ASSET_COMMIT"
git -C "$GATE0_ROOT/assets" checkout --detach "$ASSET_COMMIT"
git -C "$GATE0_ROOT/assets" checkout-index --all --force
[[ "$(git -C "$GATE0_ROOT/assets" rev-parse HEAD)" == "$ASSET_COMMIT" ]] \
  || die "Sharpa asset commit mismatch"
[[ "$(git -C "$GATE0_ROOT/assets" rev-parse "HEAD:wave_01")" == "$ASSET_GIT_TREE" ]] \
  || die "Sharpa platform-independent Git tree mismatch"
verify_repository "$GATE0_ROOT/assets" "$ASSET_URL"

actual_asset_tree_sha256="$(ASSET_TREE="$GATE0_ROOT/assets/wave_01" "$GATE0_ROOT/env/bin/python" - <<'PY'
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
[[ "$actual_asset_tree_sha256" == "$LINUX_ASSET_TREE_SHA256" ]] \
  || die "Sharpa asset-tree SHA-256 mismatch: $actual_asset_tree_sha256"

cd "$GATE0_ROOT/IsaacLab"
refuse_symlink "$GATE0_ROOT/IsaacLab/uv.lock"
if [[ ! -f "$GATE0_ROOT/IsaacLab/uv.lock" ]]; then
  readonly VENDORED_UV_LOCK="$GATE0_ROOT/bootstrap/$UV_LOCK_FILENAME"
  [[ -f "$VENDORED_UV_LOCK" ]] \
    || die "missing vendored Isaac Lab lock: $VENDORED_UV_LOCK"
  refuse_symlink "$VENDORED_UV_LOCK"
  printf '%s  %s\n' "$UV_LOCK_SHA256" "$VENDORED_UV_LOCK" \
    | sha256sum --check --status \
    || die "vendored Isaac Lab lock SHA-256 mismatch"
  cp -- "$VENDORED_UV_LOCK" "$GATE0_ROOT/IsaacLab/uv.lock"
fi
printf '%s  %s\n' "$UV_LOCK_SHA256" "$GATE0_ROOT/IsaacLab/uv.lock" \
  | sha256sum --check --status \
  || die "Isaac Lab uv.lock SHA-256 mismatch"
"$UV_BIN" sync --extra ov --locked
"$UV_BIN" pip check --python "$GATE0_ROOT/env/bin/python"

# Isaac Lab at the pinned commit declares both usd-core and usd-exchange on
# Linux x86_64. Both distributions install overlapping, ABI-incompatible pxr
# modules. Preserve the original lock, then apply the narrow, reversible
# kit-less overlay documented in configs/parity/ovphysx-runtime-overlay.json.
actual_usd_exchange_version="$("$GATE0_ROOT/env/bin/python" - <<'PY'
from importlib import metadata
print(metadata.version("usd-exchange"))
PY
)"
[[ "$actual_usd_exchange_version" == "$USD_EXCHANGE_VERSION" ]] \
  || die "unexpected usd-exchange version: $actual_usd_exchange_version"
"$UV_BIN" pip uninstall --python "$GATE0_ROOT/env/bin/python" usd-exchange
"$UV_BIN" pip install --python "$GATE0_ROOT/env/bin/python" --reinstall \
  "usd-core==$USD_CORE_VERSION"
"$GATE0_ROOT/env/bin/python" - <<'PY'
from importlib import metadata
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: F401
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: F401
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: F401
import ovphysx  # noqa: F401

try:
    metadata.version("usd-exchange")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("usd-exchange remains installed after the runtime overlay")
if metadata.version("usd-core") != "25.11":
    raise SystemExit("unexpected usd-core version after the runtime overlay")
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
print("kit-less OVPhysX/OpenUSD imports: OK")
PY

root_bytes="$(du -sb "$GATE0_ROOT" | awk '{print $1}')"
[[ "$root_bytes" -le "$MAX_ROOT_BYTES" ]] \
  || die "Gate 0 root exceeds the 30 GiB stop limit"

readonly PROVENANCE_PATH="$GATE0_ROOT/results/bootstrap-provenance-$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  printf 'bootstrapped_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'uv='; "$UV_BIN" --version
  printf 'python='; "$GATE0_ROOT/env/bin/python" --version
  printf 'isaaclab_tag=%s\n' "$ISAACLAB_TAG"
  printf 'isaaclab_commit='; git -C "$GATE0_ROOT/IsaacLab" rev-parse HEAD
  printf 'asset_commit='; git -C "$GATE0_ROOT/assets" rev-parse HEAD
  printf 'asset_git_tree='; git -C "$GATE0_ROOT/assets" rev-parse 'HEAD:wave_01'
  printf 'v0_1_windows_asset_tree_sha256=%s\n' "$V01_WINDOWS_ASSET_TREE_SHA256"
  printf 'linux_asset_tree_sha256=%s\n' "$actual_asset_tree_sha256"
  printf 'uv_lock_sha256='; sha256sum "$GATE0_ROOT/IsaacLab/uv.lock" | awk '{print $1}'
  printf 'runtime_overlay=remove-usd-exchange-%s,reinstall-usd-core-%s\n' \
    "$USD_EXCHANGE_VERSION" "$USD_CORE_VERSION"
  printf 'root_bytes=%s\n' "$root_bytes"
} > "$PROVENANCE_PATH"

echo "Gate 0 environment prepared below $GATE0_ROOT"
