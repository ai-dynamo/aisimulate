#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
base_image="${AISIM_BASE_IMAGE_BY_DIGEST:-}"
if [[ ! "${base_image}" =~ ^[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[0-9a-f]{64}$ ]]; then
  echo "AISIM_BASE_IMAGE_BY_DIGEST must be an image@sha256:<64 lowercase hex digits> reference." >&2
  exit 2
fi
: "${AISIM_BUILD_IMAGE_TAG:?Set the authorized destination image tag}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec docker buildx build --platform linux/amd64,linux/arm64 \
  --build-arg "BASE_IMAGE=${base_image}" \
  --file "${repo_root}/.github/ci-image/Dockerfile" \
  --tag "${AISIM_BUILD_IMAGE_TAG}" --push "${repo_root}"
