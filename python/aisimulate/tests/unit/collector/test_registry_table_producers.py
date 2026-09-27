# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every staged perf table needs a declared checkpoint producer.

collect.py's fail-closed finalize binds each staged ``*_perf.txt`` to the
registry entries whose ``perf_filename`` names it (several ops may feed one
table, e.g. mla_bmm_gen_pre/post); a collector that writes a second table
its op does not declare cannot be finalized (first full vllm
compute_scale run, 2026-09-27: ``scale_matrix_perf.txt has no selected
producer``). compute_scale's second table therefore has its own registry op,
``scale_matrix``, sharing the module and case grid.
"""
import importlib

import pytest

from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit

BACKENDS = ("vllm", "sglang", "trtllm")


def _registry(backend):
    return importlib.import_module(f"collector.{backend}.registry").REGISTRY


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_scale_tables_have_one_producer_each(backend):
    entries = {entry.op: entry for entry in _registry(backend)}
    cs, sm = entries["compute_scale"], entries["scale_matrix"]
    assert cs.perf_filename == PerfFile.COMPUTESCALE
    assert sm.perf_filename == PerfFile.SCALE_MATRIX
    # same module, same case grid, distinct run functions -> distinct checkpoints
    assert cs.module == sm.module == f"collector.{backend}.collect_computescale"
    assert cs.get_func == sm.get_func
    assert cs.run_func == "run_computescale" and sm.run_func == "run_scale_matrix"


@pytest.mark.parametrize("backend", BACKENDS)
def test_scale_matrix_collector_exposes_run_func(backend):
    src = importlib.util.find_spec(f"collector.{backend}.collect_computescale").origin
    text = open(src).read()
    assert "def run_scale_matrix(m, k, *, perf_filename" in text
    assert "def run_computescale(m, k, *, perf_filename" in text
    # the hard-coded second table is gone: each run function writes only the
    # table the executor hands it
    assert 'perf_filename="scale_matrix_perf.txt"' not in text
