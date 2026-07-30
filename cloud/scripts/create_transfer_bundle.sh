#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gzip
require_command sha256sum
require_command tar
(( $# >= 2 )) || die "usage: $0 OUTPUT.tar.gz RELATIVE_PATH [RELATIVE_PATH ...]"

output="$1"
shift
[[ "${output}" == *.tar.gz ]] || die "output name must end in .tar.gz"
[[ ! -e "${output}" && ! -e "${output}.sha256" ]] || die "output or checksum already exists"

cd -- "${REPORL_ROOT}"
items=()
for item in "$@"; do
  [[ -n "${item}" && "${item}" != /* && "${item}" != *".."* ]] || \
    die "bundle paths must be repository-relative and must not contain '..': ${item}"
  [[ -e "${item}" ]] || die "bundle input does not exist: ${item}"
  if find -- "${item}" ! -type f ! -type d -print -quit | grep -q .; then
    die "bundle inputs may contain only regular files and directories: ${item}"
  fi
  items+=("${item}")
done

output_parent="$(dirname -- "${output}")"
mkdir -p -- "${output_parent}"
output_parent="$(cd -- "${output_parent}" && pwd)"
output_abs="${output_parent}/$(basename -- "${output}")"
for item in "${items[@]}"; do
  if [[ -d "${item}" ]]; then
    item_abs="$(cd -- "${item}" && pwd)"
    case "${output_abs}" in
      "${item_abs}"/*) die "bundle output must not be created inside an input directory" ;;
    esac
  fi
done
tmp_file="$(mktemp "${output_parent}/.reporl-bundle.XXXXXX")"
trap 'rm -f -- "${tmp_file}"' EXIT

tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -cf - \
  -- "${items[@]}" | gzip -n >"${tmp_file}"
mv -- "${tmp_file}" "${output_abs}"
trap - EXIT

(
  cd -- "${output_parent}"
  sha256sum -- "$(basename -- "${output_abs}")" >"$(basename -- "${output_abs}").sha256"
)
note "Created ${output_abs}"
note "Created ${output_abs}.sha256"
