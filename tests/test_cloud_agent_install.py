# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the bootstrap's shell control flow without installing host tools."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / ".cursor" / "cloud-agent-install.sh"

STUBS = r"""
uv() {
  if [[ "$*" == 'sync --help' ]]; then
    if [[ -f "$BOOTSTRAP_STATE/uv-ready" || "$UV_MODE" == compatible ]]; then
      printf '%s\n' '--project PATH --extra EXTRA'
    elif [[ "$UV_MODE" == project-only ]]; then
      printf '%s\n' '--project PATH'
    elif [[ "$UV_MODE" == extra-only ]]; then
      printf '%s\n' '--extra EXTRA'
    else
      return 2
    fi
  else
    printf 'sync %s toolchain=%s\n' "$*" "${RUSTUP_TOOLCHAIN-unset}" >> "$COMMAND_LOG"
    [[ -f "$BOOTSTRAP_STATE/uv-ready" || "$UV_MODE" == compatible ]] || return 2
    [[ "$*" == 'sync --project python/aisim --extra dev' ]] || return 3
    [[ -f python/aisim/pyproject.toml ]] || return 4
    [[ "$(rustc --version)" == 'rustc 1.96.'* ]] || return 5
  fi
}
rustc() {
  local version="${RUSTUP_TOOLCHAIN-}"
  if [[ -z "$version" ]]; then
    if [[ "$DIRECTORY_OVERRIDE" == true ]]; then
      version=1.84.0
    elif [[ -f "$BOOTSTRAP_STATE/rust-ready" ]]; then
      version=1.96.0
    else
      version="$INITIAL_RUST"
    fi
  fi
  printf 'rustc %s\n' "$version"
}
rustup() {
  printf 'rustup %s\n' "$*" >> "$COMMAND_LOG"
  if [[ "$*" == 'default 1.96.0' ]]; then
    touch "$BOOTSTRAP_STATE/rust-ready"
  fi
}
curl() {
  printf 'curl %s\n' "$*" >> "$COMMAND_LOG"
  [[ "$*" == '-LsSf https://astral.sh/uv/install.sh' ]] || return 6
  printf 'touch "$BOOTSTRAP_STATE/uv-ready"\n'
}
sudo() {
  printf 'sudo %s\n' "$*" >> "$COMMAND_LOG"
}
"""


def _bootstrap(tmp_path, *, uv="compatible", rust="1.96.1", toolchain=None, directory_override=False):
    stubs = tmp_path / "stubs.sh"
    stubs.write_text(STUBS)
    log = tmp_path / "commands.log"
    env = {
        **os.environ,
        "BASH_ENV": str(stubs),
        "BOOTSTRAP_STATE": str(tmp_path),
        "COMMAND_LOG": str(log),
        "UV_MODE": uv,
        "INITIAL_RUST": rust,
        "DIRECTORY_OVERRIDE": str(directory_override).lower(),
    }
    env.pop("RUSTUP_TOOLCHAIN", None)
    if toolchain is not None:
        env["RUSTUP_TOOLCHAIN"] = toolchain
    for _ in range(2):
        subprocess.run(["bash", str(INSTALL)], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    return log.read_text().splitlines()


@pytest.mark.parametrize("uv", ["legacy", "missing", "project-only", "extra-only"])
def test_bootstrap_replaces_incompatible_uv_and_is_repeatable(tmp_path, uv):
    commands = _bootstrap(tmp_path, uv=uv)
    assert commands.count("curl -LsSf https://astral.sh/uv/install.sh") == 1
    assert len([line for line in commands if line.startswith("sync ")]) == 2


@pytest.mark.parametrize("directory_override", [False, True])
def test_bootstrap_selects_installed_rust_over_existing_overrides(tmp_path, directory_override):
    commands = _bootstrap(
        tmp_path,
        rust="1.84.0",
        toolchain=None if directory_override else "1.84.0",
        directory_override=directory_override,
    )
    assert "rustup toolchain install 1.96.0 --profile minimal" in commands
    assert "rustup default 1.96.0" in commands
    assert [line for line in commands if line.startswith("sync ")] == [
        "sync sync --project python/aisim --extra dev toolchain=1.96.0"
    ] * 2


def test_bootstrap_preserves_compatible_tools_and_toolchain(tmp_path):
    commands = _bootstrap(tmp_path, toolchain="1.96.1")
    assert commands == ["sync sync --project python/aisim --extra dev toolchain=1.96.1"] * 2
