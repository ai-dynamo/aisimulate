# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4 FMLA sidecar ops ``dsv4_hca_attn_module`` / ``dsv4_csa_attn_module``
were removed end to end on 2026-10-15: collector registries and case plan, ``PerfFile``,
the SDK ``PerfDataFilename``, the op catalog, the capability floors and the shipped
tables. Their SDK readers had been retired placeholders with no query path (csa_attn
never shipped data), and as registry-only ops every sm100 shard plan generated from
the registry since 2026-10-05 carried them and failed one or two shards per cell.
This guard keeps them out of every layer a shard plan or the SDK reads."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from collector.capabilities import _load_capabilities
from collector.op_catalog import load_family_map
from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit

RETIRED_OPS = frozenset({"dsv4_hca_attn_module", "dsv4_csa_attn_module"})
PKG_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("backend", ("vllm", "sglang", "trtllm"))
def test_retired_sidecar_ops_are_not_registered(backend):
    registered = {entry.op for entry in importlib.import_module(f"collector.{backend}.registry").REGISTRY}
    assert not registered & RETIRED_OPS


def test_retired_sidecar_tables_have_no_perf_file_catalog_entry_or_floor():
    assert not {member.name for member in PerfFile} & {"DSV4_HCA_ATTN_MODULE", "DSV4_CSA_ATTN_MODULE"}
    family_map = load_family_map()
    assert family_map is not None
    assert not {f"{op}_perf" for op in RETIRED_OPS} & set(family_map)
    _, op_min_sm = _load_capabilities()
    assert not RETIRED_OPS & set(op_min_sm)


def test_retired_sidecar_tables_have_no_sdk_filename_or_shipped_data():
    from aisimulate_core.sdk.common import PerfDataFilename

    assert not {member.name for member in PerfDataFilename} & RETIRED_OPS
    data_root = PKG_ROOT / "src" / "aisimulate_core" / "systems" / "data"
    shipped = sorted(str(p.relative_to(data_root)) for op in RETIRED_OPS for p in data_root.rglob(f"{op}_perf.parquet"))
    assert shipped == []
