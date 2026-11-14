# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run_diff lays a new table next to the previous release: a slow node shows as compute-bound
rows slowing while launch-bound rows hold (the H100 gemm tables of the 2026-11 upgrade passed
every data check and ran 1.3-4x slow), and a key the new release dropped shows as a lost bucket.
The same reading against the table a recollection replaces and against a sibling system, and the
failed tasks of a run against those of the run before."""

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


def _put(root, system, version, scale=None):
    d = root / system / "gemm" / "trtllm" / version
    d.mkdir(parents=True, exist_ok=True)
    _table(scale or (lambda latency: latency)).to_parquet(d / "gemm_perf.parquet")


def _slow(latency):
    return latency * (1.5 if latency > 0.5 else 1.0)


def test_a_recollection_is_read_next_to_the_table_it_replaces_and_a_sibling(run_diff, tmp_path, monkeypatch, capsys):
    """A node that slows the compute-bound cases stands above the sibling system where the base
    table stood with it; the table of the same version under the base root is the one replaced."""
    monkeypatch.setattr(run_diff, "pinned_version", lambda backend, family: "2.0")
    base, new = tmp_path / "base", tmp_path / "new"
    for system, version in (("sys_a", "1.0"), ("sys_a", "2.0"), ("sys_b", "1.0"), ("sys_b", "2.0")):
        _put(base, system, version)
    _put(new, "sys_a", "2.0", _slow)
    (row,) = run_diff.tables(new, base, siblings=["sys_b", "sys_none"])
    assert (row["previous"], row["lead"], row["spikes"]) == ("1.0", "large-slower", 0)
    replaces = row["replaces"]
    assert (replaces["shared"], replaces["small"], replaces["large"]) == (200, 1.0, 1.5)
    assert (replaces["over2"], replaces["under2"], replaces["lead"]) == (0, 0, "large-slower")
    assert list(row["siblings"]) == ["sys_b"]  # a system without the table is left out
    view = row["siblings"]["sys_b"]
    assert (view["new"]["large"], view["base"]["large"], view["base_version"]) == (1.5, 1.0, "2.0")
    assert view["lead"] == "slow host?"
    monkeypatch.setattr(sys, "argv", ["run_diff.py", "tables", str(new), "--base", str(base), "--sibling", "sys_b"])
    assert run_diff.main() == 0
    out = capsys.readouterr().out
    assert "    replaces: rows=200" in out and "    sys_b new: " in out and out.count("slow host?") == 1
    # a first collection replaces nothing: the base table is the previous release, against the
    # sibling at that release
    (base / "sys_a" / "gemm" / "trtllm" / "2.0" / "gemm_perf.parquet").unlink()
    (row,) = run_diff.tables(new, base, siblings=["sys_b"])
    assert "replaces" not in row and row["siblings"]["sys_b"]["base_version"] == "1.0"
    assert row["siblings"]["sys_b"]["lead"] == "slow host?"
    # a system that stood above its sibling before is no slow host
    _put(base, "sys_a", "1.0", _slow)
    (row,) = run_diff.tables(new, base, siblings=["sys_b"])
    assert (row["siblings"]["sys_b"]["base"]["large"], row["siblings"]["sys_b"]["lead"]) == (1.5, "")


def test_cases_more_than_twice_apart_and_isolated_spikes_are_counted(run_diff):
    old = _table()
    new = _table(lambda latency: latency * (3.0 if latency > 0.97 else 1.0))  # m = 98, 99, 100 of both dtypes
    assert (run_diff.compare(new, old)["over2"], run_diff.compare(old, new)["under2"]) == (6, 6)
    assert run_diff.spikes(old) == 0
    spiked = _table()
    spiked.loc[(spiked.gemm_type == "fp8") & (spiked.m == 50), "latency"] *= 10
    assert run_diff.spikes(spiked) == 1
    # a steep sweep and its last point are a trend, not spikes
    steep = pd.DataFrame({"gemm_type": "fp8", "m": [1, 2, 4, 8, 16], "latency": [1.0, 8.0, 64.0, 512.0, 4096.0]})
    assert run_diff.spikes(steep) == 0


def test_failures_against_the_previous_run_and_tasks_that_still_wrote_rows(run_diff, tmp_path, monkeypatch, capsys):
    """A task that times several points and loses some is recorded failed with the count; the
    others were written, so it is no whole-task failure. Tasks match across runs by their id."""
    refused = "FIXME(kernel-limit): 3/40 point(s) refused: b=1 s=65536 prefix=0: does not fit"
    every_point = "FIXME(kernel-limit): 40/40 point(s) of the module series failed: s=1 prefix=0: does not fit"
    runs = {
        "before": [("t1", "boom"), ("t2", refused)],
        "after": [("t2", refused), ("t3", every_point), ("t4", "boom")],
    }
    for run, tasks in runs.items():
        shard = tmp_path / run / "sys_a" / "moe__nvfp4"
        shard.mkdir(parents=True)
        _table().to_parquet(shard / "moe_perf.parquet")
        errors = [{"task_id": task, "error_message": message} for task, message in tasks]
        (shard / "errors_vllm.moe.json").write_text(json.dumps(errors))
    (row,) = run_diff.failures([tmp_path / "after"], [tmp_path / "before"])
    assert (row["new"], row["gone"], row["unchanged"], row["previous"]) == (["t3", "t4"], ["t1"], 1, True)
    assert (row["failed"], row["partial"]) == (3, ["t2"])
    (row,) = run_diff.failures([tmp_path / "after"])
    assert "new" not in row and row["partial"] == ["t2"]
    (row,) = run_diff.failures([tmp_path / "after"], [tmp_path / "no_such_run"])
    assert row["previous"] is False and row["new"] == ["t2", "t3", "t4"]
    argv = ["run_diff.py", "failures", str(tmp_path / "after"), "--base", str(tmp_path / "before")]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_diff.main() == 0
    out = capsys.readouterr().out
    assert "failed=3 (1.5%) partial=1 new=2 gone=1 unchanged=1" in out and "    new: t3" in out
