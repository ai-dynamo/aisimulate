#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Idempotent Cloud Agent bootstrap: toolchain pins + uv sync for AISim dev.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="/usr/local/bin:${HOME}/.cargo/bin:/usr/local/cargo/bin:${PATH}"

has_compatible_uv() {
  local sync_help
  sync_help=$(uv sync --help 2>/dev/null) || return 1
  [[ "$sync_help" == *"--project"* && "$sync_help" == *"--extra"* ]]
}

if ! has_compatible_uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv
  hash -r
fi

need_rust_196() {
  command -v rustc >/dev/null || return 0
  rustc --version 2>/dev/null | grep -q '1\.96\.' && return 1 || return 0
}

if need_rust_196; then
  if ! command -v rustup >/dev/null; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.0 --profile minimal
  else
    rustup toolchain install 1.96.0 --profile minimal
    rustup default 1.96.0
  fi
  sudo ln -sf "${HOME}/.cargo/bin/cargo" /usr/local/bin/cargo 2>/dev/null || true
  sudo ln -sf "${HOME}/.cargo/bin/rustc" /usr/local/bin/rustc 2>/dev/null || true
  export RUSTUP_TOOLCHAIN=1.96.0
fi

uv sync --project python/aisim --extra dev
