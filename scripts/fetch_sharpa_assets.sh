#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${script_dir}/../external/sharpa-urdf-usd-xml}"
repository="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git"
commit="6eea427eb24189519f32b9f21674cd534d3f973c"

if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
  echo "Destination exists but is not a Git checkout: ${destination}" >&2
  exit 2
fi

if [[ ! -d "${destination}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
fi

git -C "${destination}" sparse-checkout init --cone
git -C "${destination}" sparse-checkout set \
  wave_01/left_sharpa_wave \
  wave_01/right_sharpa_wave \
  wave_01/dual_sharpa_wave \
  wave_01/sharpa_wave_float_base_urdf_usd
git -C "${destination}" fetch origin "${commit}" --depth 1
git -C "${destination}" checkout --detach "${commit}"

actual_commit="$(git -C "${destination}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${commit}" ]]; then
  echo "Pinned revision mismatch: expected ${commit}, got ${actual_commit}" >&2
  exit 3
fi

echo "Sharpa assets ready at ${destination}"
echo "Pinned commit: ${actual_commit}"
