# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import subprocess as sp
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _get_exp_yaml_files():
    """Dynamically discover all YAML files in the exps directory."""
    exps_dir = pkg_resources.files("aisimulate") / "legacy_cli" / "exps"
    return sorted([str(yaml_file) for yaml_file in exps_dir.iterdir() if yaml_file.suffix == ".yaml"])


_ALL_EXP_YAMLS = _get_exp_yaml_files()

# Keep the pre-merge build subset small, stable, and representative. The full
# experiment matrix remains available to explicit e2e invocations; some
# large-EP searches intentionally exceed the per-test CI timeout.
_BUILD_EXP_FILENAMES = {
    "qwen3_32b_request_latency.yaml",
}


def _parametrize_exp_yamls(yaml_paths: list[str]) -> list:
    params: list = []
    for yaml_path in yaml_paths:
        name = Path(yaml_path).name
        if name in _BUILD_EXP_FILENAMES:
            params.append(pytest.param(yaml_path, id=name, marks=[pytest.mark.build]))
        else:
            params.append(pytest.param(yaml_path, id=name))
    return params


EXP_YAMLS_TO_TEST = _parametrize_exp_yamls(_ALL_EXP_YAMLS)


class TestExps:
    """Test aiconfigurator CLI with various exps."""

    @pytest.mark.parametrize("exp_yaml", EXP_YAMLS_TO_TEST)
    def test_exps(
        self,
        exp_yaml,
    ):
        cmd = ["aiconfigurator", "cli", "exp", "--yaml-path", exp_yaml]
        sp.run(cmd, check=True)
