#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Prepared images already contain these tools. Avoid a network dependency on
# every job while retaining a bounded bootstrap for existing runner images.
if command -v cc >/dev/null && command -v c++ >/dev/null && command -v make >/dev/null; then
  echo "Native build tools are already installed."
  exit 0
fi

as_root() {
  if [[ "$(id -u)" == 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}
for attempt in 1 2 3; do
  if as_root apt-get -o Acquire::Retries=3 update &&
     as_root apt-get -o Acquire::Retries=3 install --yes --no-install-recommends build-essential; then
    command -v cc >/dev/null
    command -v c++ >/dev/null
    command -v make >/dev/null
    exit 0
  fi
  if [[ "${attempt}" == 3 ]]; then
    echo "::error::Native build-tool installation failed after three attempts."
    exit 1
  fi
  # Mirror index publication can race package downloads. Refetch the indexes;
  # never disable signature or checksum verification to recover from that race.
  as_root rm -rf /var/lib/apt/lists/*
  sleep "$((attempt * 5))"
done
