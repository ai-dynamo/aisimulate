# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level guard for how the SGLang collectors read the installed SGLang version.

Current upstream SGLang images ship setuptools without ``pkg_resources``, so a collector that
imports it fails there. These collectors read the version through ``importlib.metadata`` instead,
as ``collect_moe.py`` already does. Importing them needs sglang and a GPU, so the test reads
their source.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

MODULES = [
    "collector.sglang.collect_attn",
    "collector.sglang.collect_attn_encoder",
    "collector.sglang.collect_computescale",
    "collector.sglang.collect_mla",
    "collector.sglang.collect_mla_bmm",
]

COLLECTOR = Path(__file__).resolve().parents[3] / "collector" / "sglang"


@pytest.mark.parametrize("name", MODULES)
def test_module_source_does_not_use_pkg_resources(name):
    source = (COLLECTOR / (name.rsplit(".", 1)[1] + ".py")).read_text(encoding="utf-8")
    assert "pkg_resources" not in source
    assert "from importlib.metadata import" in source
