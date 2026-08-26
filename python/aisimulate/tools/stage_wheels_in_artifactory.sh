#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${ARTIFACTORY_URL:?ARTIFACTORY_URL is not configured}"
: "${ARTIFACTORY_TOKEN:?ARTIFACTORY_TOKEN is not configured}"
: "${ARTIFACTORY_PYPI_REPO_NAME:?ARTIFACTORY_PYPI_REPO_NAME is not configured}"
: "${ARTIFACTORY_SUBPATH:?ARTIFACTORY_SUBPATH is not configured}"

if [[ "${ARTIFACTORY_URL}" != https://* ]]; then
    echo "::error::ARTIFACTORY_URL must use HTTPS"
    exit 1
fi

if [[ "${ARTIFACTORY_SUBPATH}" == /* || "${ARTIFACTORY_SUBPATH}" == *..* ]]; then
    echo "::error::ARTIFACTORY_SUBPATH must be a relative path without parent traversal"
    exit 1
fi

shopt -s nullglob
wheel_files=(wheelhouse/*.whl)
if [[ "${#wheel_files[@]}" -ne 1 ]]; then
    echo "::error::Expected exactly one wheel, found ${#wheel_files[@]}"
    exit 1
fi

wheel="${wheel_files[0]}"
filename="$(basename "${wheel}")"
if [[ "${filename}" != aisimulate-*.whl ]]; then
    echo "::error::Expected an aisimulate wheel, found ${filename}"
    exit 1
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Artifactory wheels"
        echo
        echo "- Path: \`${ARTIFACTORY_SUBPATH}/\`"
    } >> "${GITHUB_STEP_SUMMARY}"
fi

target="${ARTIFACTORY_URL%/}/${ARTIFACTORY_PYPI_REPO_NAME}/${ARTIFACTORY_SUBPATH}/${filename}"
curl --fail-with-body --show-error --silent \
    --connect-timeout 30 --max-time 900 \
    --retry 3 --retry-delay 5 --retry-all-errors \
    --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
    --upload-file "${wheel}" \
    "${target}"
echo "Uploaded ${filename} to ${ARTIFACTORY_SUBPATH}/"
