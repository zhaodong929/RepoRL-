#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command sha256sum
require_command tar
(( $# == 1 )) || die "usage: $0 BUNDLE.tar.gz"
bundle="$1"
require_file "${bundle}"
require_file "${bundle}.sha256"

bundle_dir="$(cd -- "$(dirname -- "${bundle}")" && pwd)"
bundle_name="$(basename -- "${bundle}")"
(
  cd -- "${bundle_dir}"
  checksum_record="$(<"${bundle_name}.sha256")"
  recorded_name="${checksum_record#*  }"
  [[ "${recorded_name}" == "${bundle_name}" ]] || \
    die "checksum file refers to an unexpected bundle: ${recorded_name}"
  sha256sum --check --strict -- "${bundle_name}.sha256"
)

while IFS= read -r member; do
  [[ -n "${member}" ]] || continue
  [[ "${member}" != /* ]] || die "archive contains an absolute path: ${member}"
  case "/${member}/" in
    */../*) die "archive contains a parent traversal: ${member}" ;;
  esac
done < <(tar -tzf "${bundle}")

if tar -tvzf "${bundle}" | awk '$1 !~ /^[-d]/ {found=1} END {exit !found}'; then
  die "archive contains a link or special filesystem entry"
fi
note "Bundle checksum and member paths are valid: ${bundle}"
