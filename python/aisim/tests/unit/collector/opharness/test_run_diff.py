# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run_diff lays a new table next to the previous release: a slow node shows as compute-bound
rows slowing while launch-bound rows hold (the H100 gemm tables of the 2026-11 upgrade passed
every data check and ran 1.3-4x slow), and a key the new release dropped shows as a lost bucket."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture(scope="module")
def run_diff():
    spec = importlib.util.spec_from_file_location("run_diff_t", COMPONENTS / "run_diff.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_diff_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(scale=lambda latency: latency, dtypes=("bfloat16", "fp8")):
    rows = [
        {"framework": "vllm", "version": "v", "kernel_source": "k", "gemm_type": dtype, "m": m, "latency": m * 0.01}
        for dtype in dtypes
        for m in range(1, 101)
    ]
    frame = pd.DataFrame(rows)
    frame["latency"] = frame["latency"].map(scale)
    return frame


def test_a_slow_node_slows_the_large_cases_only(run_diff):
    old = _table()
    slow = _table(lambda latency: latency * (2.0 if latency > 0.5 else 1.0))
    result = run_diff.compare(slow, old)
    assert (result["shared"], result["lost"], result["gained"]) == (200, 0, 0)
    assert (result["small"], result["large"]) == (1.0, 2.0) and run_diff.lead(result) == "large-slower"
    same = run_diff.compare(_table(lambda latency: latency * 1.03), old)
    assert run_diff.lead(same) == ""
    # a change of the timed region moves every case: a shift to explain, not a slow node
    assert run_diff.lead(run_diff.compare(_table(lambda latency: latency * 1.4), old)) == "shifted"


def test_a_dropped_key_is_a_lost_bucket(run_diff):
    result = run_diff.compare(_table(dtypes=("bfloat16",)), _table())
    assert result["buckets"] == ["gemm_type"] and result["keys"] == ["gemm_type", "m"]
    assert result["buckets_lost"] == [("fp8",)] and result["lost"] == 100
    assert run_diff.lead(result) == "buckets-lost"
    # fewer points of a swept column lose cases, not a bucket
    thinner = _table()
    thinner = thinner[thinner.m % 2 == 0]
    result = run_diff.compare(thinner, _table())
    assert result["lost"] == 100 and result["buckets_lost"] == []


def test_the_new_table_is_the_manifest_pin_and_its_previous_the_greatest_older_version(run_diff, tmp_path, monkeypatch):
    monkeypatch.setattr(run_diff, "pinned_version", lambda backend, family: "1.4.0rc2")
    for root, system, version, scale in (
        ("base", "rtx_pro_6000_server", "1.3.0rc20", 3.0),
        ("base", "rtx_pro_6000_server", "1.3.0rc29", 1.0),
        ("new", "rtxpro6000_blackwell_server", "1.4.0rc2", 1.0),
        ("new", "rtxpro6000_blackwell_server", "1.3.0rc29", 9.0),  # not the pin: never the new table
    ):
        d = tmp_path / root / system / "gemm" / "trtllm" / version
        d.mkdir(parents=True)
        _table(lambda latency, scale=scale: latency * scale).to_parquet(d / "gemm_perf.parquet")
    alias = {"rtxpro6000_blackwell_server": "rtx_pro_6000_server"}
    (row,) = run_diff.tables(tmp_path / "new", tmp_path / "base", alias=alias)
    assert (row["version"], row["previous"], row["large"], row["lead"]) == ("1.4.0rc2", "1.3.0rc29", 1.0, "")
    (row,) = run_diff.tables(tmp_path / "new", tmp_path / "base")  # the system is named apart: nothing to compare
    assert row["previous"] is None and "large" not in row


def test_the_manifest_names_the_release_of_a_data_family(run_diff):
    import yaml

    spec = yaml.safe_load((COMPONENTS.parents[1] / "framework_manifest.yaml").read_text())["frameworks"]["vllm"]
    family, pin = next(iter(spec["families"].items()))
    assert run_diff.pinned_version("vllm", family) == str(pin["version"])
    assert run_diff.pinned_version("vllm", "no_such_family") == str(spec["default"]["version"])
    assert run_diff.pinned_version("no_such_backend", family) is None


def test_failures_count_failed_tasks_per_shard(run_diff, tmp_path):
    shard = tmp_path / "result" / "gemm__fp8"
    shard.mkdir(parents=True)
    _table().to_parquet(shard / "gemm_perf.parquet")
    errors = [
        {
            "error_type": "RuntimeError",
            "error_message": "K must be a multiple of 128, (K=7168)\ntraceback",
            "classification": "unexpected",
        },
        {
            "error_type": "RuntimeError",
            "error_message": "K must be a multiple of 128, (K=5120)",
            "classification": "unexpected",
        },
        {
            "error_type": "RuntimeError",
            "error_message": "FIXME(kernel-limit): does not fit",
            "classification": "kernel_limit",
        },
        {"error_type": "UnresolvedFailures", "error_message": "3 tasks failed"},
    ]
    (shard / "errors_vllm.gemm.json").write_text(json.dumps({"errors": errors}))
    (row,) = run_diff.failures([tmp_path / "result"])
    assert (row["rows"], row["failed"]) == (200, 3)
    assert row["classes"] == {"unexpected": 2, "kernel_limit": 1}
    assert row["messages"][0] == ("K must be a multiple of N, (K=N)", 2)
