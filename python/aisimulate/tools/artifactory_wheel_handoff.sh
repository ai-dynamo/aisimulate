#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 upload|download DIRECTORY" >&2
    exit 2
}

[[ "$#" -eq 2 ]] || usage
mode="$1"
directory="$2"
[[ "${mode}" == "upload" || "${mode}" == "download" ]] || usage

: "${ARTIFACTORY_URL:?ARTIFACTORY_URL is not configured}"
: "${ARTIFACTORY_TOKEN:?ARTIFACTORY_TOKEN is not configured}"
: "${ARTIFACTORY_PYPI_REPO_NAME:?ARTIFACTORY_PYPI_REPO_NAME is not configured}"
: "${ARTIFACTORY_SUBPATH:?ARTIFACTORY_SUBPATH is not configured}"

[[ "${ARTIFACTORY_URL}" == https://* ]] || {
    echo "::error::ARTIFACTORY_URL must use HTTPS"
    exit 1
}
case "${ARTIFACTORY_PYPI_REPO_NAME}" in
    ""|*/*|*..*|*[!A-Za-z0-9._-]*)
        echo "::error::invalid ARTIFACTORY_PYPI_REPO_NAME"
        exit 1
        ;;
esac
case "${ARTIFACTORY_SUBPATH}" in
    ""|/*|*..*|*//*|*$'\n'*|*[!A-Za-z0-9._/-]*)
        echo "::error::ARTIFACTORY_SUBPATH must be a relative normalized path"
        exit 1
        ;;
esac

mkdir -p "${directory}"
manifest="${directory}/_WHEEL.json"
base_url="${ARTIFACTORY_URL%/}/${ARTIFACTORY_PYPI_REPO_NAME}/${ARTIFACTORY_SUBPATH}"
headers="$(mktemp "${RUNNER_TEMP:-/tmp}/artifactory-wheel-headers.XXXXXX")"
download="$(mktemp "${RUNNER_TEMP:-/tmp}/artifactory-wheel-download.XXXXXX")"
trap 'rm -f "${headers}" "${download}"' EXIT

remote_sha256() {
    local url="$1" status
    : > "${headers}"
    status="$(
        curl -q --silent --show-error --head \
            --connect-timeout 30 --max-time 120 --retry 3 \
            --output /dev/null --dump-header "${headers}" \
            --write-out '%{http_code}' \
            --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
            "${url}"
    )"
    case "${status}" in
        200)
            awk 'tolower($1) == "x-checksum-sha256:" {gsub("\\r", "", $2); print tolower($2)}' \
                "${headers}" | tail -n 1
            ;;
        404)
            return 1
            ;;
        *)
            echo "::error::Artifactory probe failed with HTTP ${status}" >&2
            return 2
            ;;
    esac
}

upload_file() {
    local file="$1" remote_name="$2" sha url existing status
    sha="$(sha256sum "${file}" | cut -d' ' -f1)"
    url="${base_url}/${remote_name}"
    if existing="$(remote_sha256 "${url}")"; then
        [[ -n "${existing}" ]] || {
            echo "::error::${remote_name} has no Artifactory SHA-256 header"
            return 1
        }
        [[ "${existing}" == "${sha}" ]] || {
            echo "::error::${remote_name} already exists with a different checksum"
            return 1
        }
        echo "Already staged ${remote_name}"
        return 0
    else
        status="$?"
        [[ "${status}" -eq 1 ]] || return "${status}"
    fi

    curl -q --fail --silent --show-error --retry 3 \
        --connect-timeout 30 --max-time 900 \
        --output /dev/null \
        --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
        --header "X-Checksum-Sha256: ${sha}" \
        --upload-file "${file}" \
        "${url}"
    existing="$(remote_sha256 "${url}")"
    [[ "${existing}" == "${sha}" ]] || {
        echo "::error::uploaded ${remote_name} failed checksum read-back"
        return 1
    }
    echo "Staged ${remote_name}"
}

if [[ "${mode}" == "upload" ]]; then
    shopt -s nullglob
    wheels=("${directory}"/aisimulate-*.whl)
    [[ "${#wheels[@]}" -eq 1 ]] || {
        echo "::error::expected exactly one AISimulate wheel, found ${#wheels[@]}"
        exit 1
    }
    wheel="${wheels[0]}"
    filename="$(basename "${wheel}")"
    sha="$(sha256sum "${wheel}" | cut -d' ' -f1)"
    size="$(wc -c < "${wheel}" | tr -d ' ')"
    python3 - "${manifest}" "${filename}" "${sha}" "${size}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

path, filename, sha256, size = sys.argv[1:]
source_sha = os.environ.get("WHEEL_SOURCE_SHA") or os.environ.get("GITHUB_SHA", "")
run_id = os.environ.get("GITHUB_RUN_ID", "")
if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
    raise SystemExit("WHEEL_SOURCE_SHA or GITHUB_SHA must be a full lowercase commit SHA")
if not run_id.isdigit():
    raise SystemExit("GITHUB_RUN_ID must be numeric")
payload = {
    "schemaVersion": "aisimulate-wheel-handoff/1.0.0",
    "filename": filename,
    "sha256": sha256,
    "size": int(size),
    "source_sha": source_sha,
    "run_id": run_id,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    # The manifest is the completion marker, so publish it only after the wheel
    # has been uploaded and checksum-verified.
    upload_file "${wheel}" "${filename}"
    upload_file "${manifest}" "_WHEEL.json"
    echo "Wheel handoff ready at ${ARTIFACTORY_SUBPATH}/"
    exit 0
fi

curl -q --fail --silent --show-error --retry 3 \
    --connect-timeout 30 --max-time 120 \
    --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
    --output "${download}" \
    "${base_url}/_WHEEL.json"
mv "${download}" "${manifest}"

wheel_metadata="$(
    python3 - "${manifest}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("schemaVersion") != "aisimulate-wheel-handoff/1.0.0":
    raise SystemExit("unsupported wheel handoff manifest schema")
filename = payload.get("filename")
sha256 = payload.get("sha256")
size = payload.get("size")
source_sha = payload.get("source_sha")
run_id = payload.get("run_id")
if not isinstance(filename, str) or not re.fullmatch(r"aisimulate-[A-Za-z0-9_.+!-]+\.whl", filename):
    raise SystemExit("invalid wheel filename in handoff manifest")
if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
    raise SystemExit("invalid wheel SHA-256 in handoff manifest")
if not isinstance(size, int) or size <= 0:
    raise SystemExit("invalid wheel size in handoff manifest")
expected_sha = os.environ.get("EXPECTED_WHEEL_SOURCE_SHA", "")
if expected_sha and source_sha != expected_sha:
    raise SystemExit(f"wheel source mismatch: expected {expected_sha}, got {source_sha}")
expected_run_id = os.environ.get("EXPECTED_WHEEL_RUN_ID") or os.environ.get("GITHUB_RUN_ID", "")
if expected_run_id and run_id != expected_run_id:
    raise SystemExit(f"wheel run mismatch: expected {expected_run_id}, got {run_id}")
print(f"{filename}\t{sha256}\t{size}")
PY
)"
IFS=$'\t' read -r filename expected_sha256 expected_size <<< "${wheel_metadata}"
[[ -n "${filename}" && -n "${expected_sha256}" && -n "${expected_size}" ]] || {
    echo "::error::invalid wheel handoff manifest"
    exit 1
}
wheel="${directory}/${filename}"

curl -q --fail --silent --show-error --retry 3 \
    --connect-timeout 30 --max-time 900 \
    --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
    --output "${download}" \
    "${base_url}/${filename}"
actual_sha256="$(sha256sum "${download}" | cut -d' ' -f1)"
actual_size="$(wc -c < "${download}" | tr -d ' ')"
[[ "${actual_sha256}" == "${expected_sha256}" ]] || {
    echo "::error::downloaded wheel checksum mismatch"
    exit 1
}
[[ "${actual_size}" == "${expected_size}" ]] || {
    echo "::error::downloaded wheel size mismatch"
    exit 1
}
mv "${download}" "${wheel}"
echo "Fetched ${filename} from ${ARTIFACTORY_SUBPATH}/"
