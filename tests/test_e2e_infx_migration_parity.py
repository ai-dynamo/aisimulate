# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohort parity between the current InferenceX cache path and its aisim-e2e/v1 migration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "aisim" / "src"))

from aisim_core.data.migrate_infx import InfxCache, migrate_infx_cache
from scripts.e2e_accuracy import hf_measurements
from scripts.e2e_accuracy import migrate_infx_parity as parity

RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/{n}/attempts/1"
RUNS = {
    RUN.format(n=n): {
        "github_run_id": str(n),
        "run_attempt": 1,
        "html_url": RUN.format(n=n),
        "head_sha": "a" * 40,
        "head_branch": "main",
        "run_started_at": f"2026-10-0{n}T00:00:00Z",
        "status": "completed",
        "conclusion": "success",
    }
    for n in (1, 2)
}


def _record(rid, **over):
    rec = dict(
        id=str(rid),
        model="dsr1",
        hardware="b200",
        framework="sglang",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=4,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=0,
        num_prefill_gpu=4,
        num_decode_gpu=8,
        offload_mode="off",
        workers=None,
        benchmark_type="single_turn",
        isl=1024,
        osl=128,
        conc=4,
        date="2026-10-01",
        image="img:1",
        recipe_fingerprint="f" * 64,
        run_url=RUN.format(n=1),
        metrics={"mean_ttft": 0.1, "mean_tpot": 0.01, "mean_e2el": 1.5, "tput_per_gpu": 10.0, "avg_power_w": 5.0},
    )
    rec.update(over)
    return rec


def _dataset(tmp_path):
    records = [
        _record(1),
        _record(2, conc=8, date="2026-10-02", image="img:2", run_url=RUN.format(n=2)),  # newer image
        _record(3, conc=16),  # older image of the same configuration: superseded by image coherence
        _record(4, spec_method="mtp", decode_dp_attention=True),
    ]
    partition = {"source_model": "dsr1", "hardware": "b200", "framework": "sglang", "display_model": "D"}
    cache = InfxCache(
        root=tmp_path,
        index={"partitions": [partition], "schema_version": "e2e-source-index/v1"},
        snapshot={
            "snapshot_id": "c" * 64,
            "schema_version": "inferencex-input-snapshot/v1",
            "source_scope": {"kind": "api-current-model-view"},
            "selection_policy": "archive-all-returned-records-v1",
            "acquisitions": [],
        },
        records=records,
        observation_rows=[],
        metric_rows=[],
        partition_of={"inferencex:" + r["id"]: ("dsr1", "b200", "sglang") for r in records},
    )
    migrate_infx_cache(cache, tmp_path / "ds", runs=RUNS)
    return records, tmp_path / "ds"


def test_raw_backed_view_reproduces_the_current_path_exactly(tmp_path):
    records, root = _dataset(tmp_path)
    old = hf_measurements.to_tables(records, RUNS)
    backed = parity.tables_from_dataset(root, raw_backed=True)
    assert parity.compare_tables(old, backed)["configs"]["field_differences"] == {}
    report = parity.selection_report(old, backed, "raw_backed")
    assert report["cohort_input_digest_equal"] and report["decision_differences"] == {}
    assert report["decision_counts"]["old"] == report["decision_counts"]["new"]


def test_typed_view_names_every_field_it_cannot_represent(tmp_path):
    records, root = _dataset(tmp_path)
    old = hf_measurements.to_tables(records, RUNS)
    typed = parity.compare_tables(old, parity.tables_from_dataset(root, raw_backed=False))
    configs = typed["configs"]["field_differences"]
    assert {"prefill_tp", "prefill_ep", "num_prefill_gpu", "prefill_num_workers"} <= set(configs)
    benches = typed["benchmark_results"]["field_differences"]
    assert benches["recipe_fingerprint"] == 4
    assert benches["metrics.avg_power_w:missing_in_new"] == 4
    # The selection itself does not depend on those fields.
    report = parity.selection_report(old, parity.tables_from_dataset(root, raw_backed=False), "typed")
    assert report["decision_differences"] == {} and report["point_ids_only_in_old"] == 0
    assert report["cohort_input_digest_equal"] is False
