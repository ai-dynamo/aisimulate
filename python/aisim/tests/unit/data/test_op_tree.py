# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-tree op-latency loading and the parity harness checks (AISIM-2063)."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.loaders import LoaderError
from aisim_core.data.op_tree import load_op_latency_tree, op_latency_entries

from .test_op_publish import _build, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "op_database_parity.py"
_spec = importlib.util.spec_from_file_location("op_database_parity", _TOOL)
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)


@pytest.fixture
def published(tree, tmp_path):  # noqa: F811
    out = tmp_path / "published"
    _build(tree, out)
    return tree, out


def test_load_reproduces_packaged_rows_and_lookup_files(published, tmp_path):
    systems, out = published
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert loaded.entries == 2 and set(loaded.lookup_files) == {
        "perf_data_reuse_manifest.yaml",
        "query_versions.yaml",
    }
    report = parity.compare_files(systems, loaded.root, out, loaded.runtime_paths)
    assert report["ok"], report["problems"]
    assert report["row_equal_tables"] == 2 and report["skipped_placeholders"] == 1
    assert report["not_runtime_published"] == ["data/h200_sxm/README.md"]
    assert parity.compare_lookup(systems, loaded.root)["ok"]


def test_load_copies_packaged_specs_but_never_packaged_tables(published, tmp_path):
    systems, out = published
    (systems / "h200_sxm.yaml").write_text("gpu: {}\nnode: {}\n")
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=systems)
    assert (loaded.root / "h200_sxm.yaml").is_file()
    assert (loaded.root / "perf_data_reuse_manifest.yaml").read_bytes() == (
        systems / "perf_data_reuse_manifest.yaml"
    ).read_bytes()


def test_selection_filters(published):
    _, out = published
    assert len(op_latency_entries(out)) == 2
    assert op_latency_entries(out, hardware=["l4"]) == []
    assert len(op_latency_entries(out, framework=["vllm"], hardware=["h200_sxm"])) == 2


def test_tampered_tree_is_rejected_and_leaves_no_root(published, tmp_path):
    _, out = published
    table = next((out / "data/op-latency/h200_sxm/gemm/vllm/1.0").glob("*.parquet"))
    pq.write_table(pa.table({"b": [1]}), table)
    with pytest.raises(ValueError, match="mismatch"):
        load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert not (tmp_path / "root").exists()


def test_existing_destination_and_empty_selection_are_refused(published, tmp_path):
    _, out = published
    (tmp_path / "root").mkdir()
    with pytest.raises(FileExistsError):
        load_op_latency_tree(out, tmp_path / "root")
    with pytest.raises(LoaderError, match="no op-latency entries"):
        load_op_latency_tree(out, tmp_path / "other", hardware=["l4"])
    assert not (tmp_path / "other").exists()


def test_file_comparison_detects_changed_missing_and_extra(published, tmp_path):
    systems, out = published
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    root = loaded.root
    table = next((root / "data/h200_sxm/gemm/vllm/1.0").glob("*.parquet"))
    pq.write_table(pa.table({"b": [1]}), table)
    (root / "data/h200_sxm/gemm/vllm/1.0/collection_meta.yaml").unlink()
    (root / "data/h200_sxm/stray.txt").write_text("x")
    report = parity.compare_files(systems, root, out, loaded.runtime_paths)
    assert not report["ok"]
    joined = "\n".join(report["problems"])
    assert "rows differ" in joined and "collection_meta.yaml" in joined and "unaccounted for" in joined


def test_prediction_comparison_is_exact_and_flags_any_drift():
    def run(total, source="measured"):
        scenario = {"model": "m", "system": "s", "backend": "b"}
        case = {"total_ms": total, "ops": [["gemm", total, 0.0, source, []]]}
        return {"scenarios": [{"scenario": scenario, "error": None, "cases": {"prefill:b1:l1": case}}]}

    same = parity.compare_predictions(run(1.5), run(1.5))
    assert same["ok"] and same["compared_values"] == 2 and same["max_abs_diff_ms"] == 0.0
    drift = parity.compare_predictions(run(1.5), run(1.5000001))
    assert not drift["ok"] and drift["max_abs_diff_ms"] > 0
    assert not parity.compare_predictions(run(1.5), run(1.5, source="sol"))["ok"]
