# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint producer owns every table its registry entry declares.

collect.py's fail-closed finalize binds each staged ``*_perf.txt`` to the
registry entries that declare it. A collector that writes a second table
its op did not declare cannot be finalized (first full vllm compute_scale
run, 2026-09-27: ``scale_matrix_perf.txt has no selected producer``; the
PerfFile.SCALE_MATRIX enum alone never fixed that). Owner decision
2026-09-28: the executor supports multi-table producers — ``OpEntry.
extra_perf_filenames`` — instead of splitting the op.
"""
import importlib
import sys

import pytest

from collector.registry_types import OpEntry, PerfFile

pytestmark = pytest.mark.unit

BACKENDS = ("vllm", "sglang", "trtllm")


def _registry(backend):
    return importlib.import_module(f"collector.{backend}.registry").REGISTRY


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_scale_declares_both_tables_on_one_producer(backend):
    entries = {entry.op: entry for entry in _registry(backend)}
    assert "scale_matrix" not in entries  # one measurement, one op
    cs = entries["compute_scale"]
    assert cs.perf_filename == PerfFile.COMPUTESCALE
    assert cs.extra_perf_filenames == (PerfFile.SCALE_MATRIX,)
    assert cs.perf_filenames == (PerfFile.COMPUTESCALE, PerfFile.SCALE_MATRIX)


def test_op_entry_rejects_duplicate_tables():
    with pytest.raises(ValueError, match="duplicate perf tables"):
        OpEntry(op="x", module="m", get_func="g", run_func="r", perf_filename=PerfFile.GEMM,
                extra_perf_filenames=(PerfFile.GEMM,))


@pytest.mark.parametrize("backend", BACKENDS)
def test_provenance_collections_carry_extra_tables(backend):
    from collector.version_resolver import build_collections

    version = {"vllm": "0.30.0", "sglang": "0.5.14", "trtllm": "1.3.0rc23"}[backend]
    (cs,) = [c for c in build_collections(_registry(backend), backend, version, ops=["compute_scale"])
             if c["type"] == "compute_scale"]
    assert cs["perf_filename"] == PerfFile.COMPUTESCALE
    assert tuple(cs["extra_perf_filenames"]) == (PerfFile.SCALE_MATRIX,)


def test_executor_binds_every_declared_table_to_the_producer():
    if "torch" not in sys.modules:  # collect.py imports torch at module level (see test_collect_provenance_writer)
        from unittest.mock import MagicMock

        _torch = MagicMock()
        _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
        sys.modules["torch"] = _torch
    import collect

    collection = {"name": "vllm", "type": "compute_scale", "module": "collector.vllm.collect_computescale",
                  "get_func": "get_computescale_test_cases", "run_func": "run_computescale",
                  "perf_filename": PerfFile.COMPUTESCALE, "extra_perf_filenames": (PerfFile.SCALE_MATRIX,)}
    assert collect._collection_perf_filenames(collection) == ["computescale_perf.txt", "scale_matrix_perf.txt"]
    assert collect._collection_tables(collection) == ("computescale_perf", "scale_matrix_perf")
    identity = {"schema": collect.RESUME_SCHEMA_VERSION, "backend": "vllm", "module": "vllm.compute_scale",
                "run_func": "run_computescale", "framework_version": "0.30.0", "sm_version": 90}
    assert set(identity) == set(collect._CHECKPOINT_IDENTITY_FIELDS)
    assert collect._registered_checkpoint_tables(identity, backend="vllm") == {"computescale_perf", "scale_matrix_perf"}
    gemm_identity = {**identity, "module": "vllm.gemm", "run_func": "run_gemm"}
    assert collect._registered_checkpoint_tables(gemm_identity, backend="vllm") == {"gemm_perf"}
    with pytest.raises(RuntimeError, match="no unambiguous registered table set"):
        collect._registered_checkpoint_tables({**identity, "module": "vllm.nope"}, backend="vllm")


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_scale_collector_writes_the_handed_in_tables(backend):
    src = importlib.util.find_spec(f"collector.{backend}.collect_computescale").origin
    text = open(src).read()
    assert "def run_computescale(m, k, *, perf_filename, extra_perf_filenames" in text
    assert "def run_scale_matrix" not in text
    # no table name is hard-coded any more: the executor hands both in
    assert 'perf_filename="scale_matrix_perf.txt"' not in text
    assert "perf_filename=scale_matrix_filename" in text
