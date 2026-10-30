# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

pytestmark = pytest.mark.unit

_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "aisimulate" / "generator" / "config" / "backend_templates" / "vllm"
)
_SPECULATIVE_TEMPLATES = [
    "cli_args.j2",
    "cli_args.0.11.0.j2",
    "cli_args.0.12.0.j2",
    "cli_args.0.14.1.j2",
    "cli_args.0.16.0.j2",
    "cli_args.0.19.0.j2",
    "cli_args.0.20.1.j2",
]


@pytest.mark.parametrize("template_name", _SPECULATIVE_TEMPLATES)
@pytest.mark.parametrize("nextn", [0, 1, 3])
def test_vllm_cli_template_preserves_speculative_config(template_name, nextn):
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)
    speculative = {"method": "mtp", "num_speculative_tokens": nextn} if nextn else {}
    args = shlex.split(
        env.get_template(template_name).render(
            vllm={"tensor-parallel-size": 2},
            speculative_config=speculative,
        )
    )

    assert args[:2] == ["--tensor-parallel-size", "2"]
    if nextn:
        assert args.count("--speculative-config") == 1
        assert json.loads(args[args.index("--speculative-config") + 1]) == speculative
    else:
        assert "--speculative-config" not in args
