# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
import math

import pytest

from aiconfigurator_core.sdk import fpm_learned

pytestmark = pytest.mark.unit


def _decode_fpm(nd: int, kv: int, wall: float, *, rank: int = 0, counter: int = 0, var: float = 0.0) -> dict:
    return {
        "version": 1,
        "worker_id": "w0",
        "dp_rank": rank,
        "counter_id": counter,
        "wall_time": wall,
        "scheduled_requests": {
            "num_decode_requests": nd,
            "sum_decode_kv_tokens": kv,
            "var_decode_kv_tokens": var,
        },
        "queued_requests": {},
    }


def _prefill_fpm(np_: int, ptok: int, pkv: int, wall: float) -> dict:
    return {
        "version": 1,
        "wall_time": wall,
        "scheduled_requests": {
            "num_prefill_requests": np_,
            "sum_prefill_tokens": ptok,
            "sum_prefill_kv_tokens": pkv,
            "var_prefill_length": 0.0,
        },
    }


def test_feature_names_match_rust_abi() -> None:
    """The Rust side hard-codes the same tables; guard against drift."""
    rust_src = fpm_learned.__file__.rsplit("/python/", 1)[0] + "/crates/core/src/perfmodel/fpm/learned.rs"
    try:
        with open(rust_src, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        pytest.skip("Rust sources not available in this checkout layout")

    def rust_array(const: str) -> tuple[str, ...]:
        block = text.split(f"pub const {const}", 1)[1].split("];", 1)[0]
        return tuple(line.strip().strip(",").strip('"') for line in block.splitlines() if line.strip().startswith('"'))

    assert rust_array("AGGREGATE_FEATURE_NAMES") == fpm_learned.AGGREGATE_FEATURE_NAMES
    assert rust_array("REQUEST_FEATURE_NAMES") == fpm_learned.REQUEST_FEATURE_NAMES
    slot_count = int(text.split("pub const SLOT_COUNT: usize = ", 1)[1].split(";", 1)[0])
    assert slot_count == fpm_learned.SLOT_COUNT
    assert fpm_learned.FEATURE_NAMES[-1] == f"slot{slot_count - 1}_extend"


def test_per_request_features_and_slots() -> None:
    it = [
        {
            "version": 1,
            "wall_time": 0.1,
            "scheduled_requests": {
                "num_decode_requests": 3,
                "sum_decode_kv_tokens": 600,
                "extend_lengths": [1, 1, 1],
                "past_kv_lengths": [100, 300, 200],
            },
        }
    ]
    f = fpm_learned.compute_features(it)
    assert f["req_batch_size"] == 3.0
    assert f["req_max_past"] == 300.0 and f["req_min_past"] == 100.0
    assert f["req_sum_extend_x_past"] == 600.0
    assert f["req_sum_attn_flops"] == pytest.approx(600.0 + 1.5)
    assert f["req_is_decode"] == 1.0 and f["req_is_prefill"] == 0.0
    # slots sorted by past descending
    assert (f["slot0_present"], f["slot0_past"], f["slot0_extend"]) == (1.0, 300.0, 1.0)
    assert (f["slot2_present"], f["slot2_past"]) == (1.0, 100.0)
    assert f["slot3_present"] == 0.0
    # aggregates-only producer -> NaN request features, empty slots
    g = fpm_learned.compute_features([_decode_fpm(3, 600, 0.1)])
    assert math.isnan(g["req_batch_size"]) and g["slot0_present"] == 0.0


def test_compute_features_named_formulas() -> None:
    iteration = [
        _prefill_fpm(2, 200, 1000, 0.1),
        _decode_fpm(4, 400, 0.1, var=100.0),
        {"version": 1, "wall_time": 0.0, "scheduled_requests": {}},
    ]
    f = fpm_learned.compute_features(iteration)
    assert f["num_active_ranks"] == 2.0
    assert f["mean_prefill_chunk"] == 100.0
    assert f["mean_prefill_kv"] == 500.0
    assert f["prefill_attention_pairs"] == pytest.approx(110_100.0)
    assert f["sum_decode_kv_squared"] == pytest.approx(40_400.0)
    assert f["max_rank_decode_kv_tokens"] == 400.0
    assert f["log1p_sum_decode_kv_tokens"] == pytest.approx(math.log1p(400.0))
    assert set(f) == set(fpm_learned.FEATURE_NAMES)


def test_classify_workload_rules() -> None:
    assert fpm_learned.classify_workload([_decode_fpm(1, 10, 0.1)], "decode") == "pure_decode"
    assert fpm_learned.classify_workload([_prefill_fpm(1, 10, 0, 0.1)], "prefill") == "pure_prefill"
    agg = [_prefill_fpm(1, 10, 0, 0.1), _decode_fpm(1, 10, 0.1)]
    assert fpm_learned.classify_workload(agg, "aggregated") == "cross_rank_aggregated"
    mixed = {
        "version": 1,
        "wall_time": 0.1,
        "scheduled_requests": {"sum_prefill_tokens": 5, "num_prefill_requests": 1, "num_decode_requests": 2},
    }
    assert fpm_learned.classify_workload([mixed], "aggregated") == "contains_locally_mixed"
    assert fpm_learned.classify_workload([{"scheduled_requests": {}}], "aggregated") is None
    with pytest.raises(ValueError, match="decode regression worker"):
        fpm_learned.classify_workload([_prefill_fpm(1, 10, 0, 0.1)], "decode")


def test_iter_records_and_group_by_counter(tmp_path) -> None:
    path = tmp_path / "fpm.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        # flat sink record, Dynamo trace envelope, heartbeat, junk line
        handle.write(json.dumps(_decode_fpm(2, 100, 0.01, rank=0, counter=7)) + "\n")
        handle.write(
            json.dumps(
                {
                    "event": {
                        "observed_at_unix_ms": 1_789_743_132_982,
                        "fpm": _decode_fpm(3, 200, 0.012, rank=1, counter=7),
                    }
                }
            )
            + "\n"
        )
        handle.write(json.dumps(_decode_fpm(0, 0, 0.0, rank=0, counter=8)) + "\n")
        handle.write("not json\n")
    records = list(fpm_learned.iter_fpm_records([path]))
    assert len(records) == 3
    # the trace sink keeps the wall-clock stamp on the envelope; it must survive unwrapping
    assert records[1]["observed_at_unix_ms"] == 1_789_743_132_982
    single = fpm_learned.group_iterations(records, "none")
    assert len(single) == 2  # heartbeat dropped
    joined = fpm_learned.group_iterations(records, "counter")
    assert len(joined) == 1
    assert [m["dp_rank"] for m in joined[0]] == [0, 1]
    assert fpm_learned.iteration_wall_ms(joined[0]) == pytest.approx(12.0)


def _stump_artifact() -> dict:
    return {
        "schema": fpm_learned.SCHEMA_NAME,
        "schema_version": fpm_learned.SCHEMA_VERSION,
        "worker_type": "decode",
        "target": "log_ms",
        "features": ["num_decode_requests", "sum_decode_kv_tokens"],
        "stores": {
            "pure_decode": {
                "baseline": 0.0,
                "trees": [
                    {
                        "left": [1, -1, -1],
                        "right": [2, -1, -1],
                        "feature": [0, -1, -1],
                        "threshold": [4.5, 0.0, 0.0],
                        "value": [0.0, math.log(10.0), math.log(20.0)],
                        "missing_left": [True, True, True],
                    }
                ],
            }
        },
        "metadata": {},
    }


def test_reference_predict_and_evaluate() -> None:
    artifact = _stump_artifact()
    assert fpm_learned.predict_ms(artifact, [_decode_fpm(2, 100, 0.0)]) == pytest.approx(10.0)
    assert fpm_learned.predict_ms(artifact, [_decode_fpm(8, 100, 0.0)]) == pytest.approx(20.0)
    assert fpm_learned.predict_ms(artifact, [{"scheduled_requests": {}}]) == 0.0
    report = fpm_learned.evaluate(
        artifact,
        [[_decode_fpm(2, 100, 0.010)], [_decode_fpm(8, 100, 0.025)]],
    )
    assert report["pure_decode"]["n"] == 2
    assert report["pure_decode"]["mape_pct"] == pytest.approx(10.0)  # (0% + 20%) / 2


def test_train_export_roundtrip_matches_sklearn() -> None:
    sklearn = pytest.importorskip("sklearn")
    del sklearn
    rng_iterations = []
    for i in range(400):
        nd = 1 + i % 16
        kv = 1000 * (1 + (i * 7) % 50)
        wall = (0.005 + 0.0004 * nd + 0.00000002 * kv) * (1.0 + 0.01 * ((i * 13) % 7 - 3))
        fpm = _decode_fpm(nd, kv, wall, counter=i)
        fpm["scheduled_requests"]["extend_lengths"] = [1] * nd
        fpm["scheduled_requests"]["past_kv_lengths"] = [kv // nd] * nd
        rng_iterations.append([fpm])
    # aggregate-only records cannot feed the default per-request feature set
    with pytest.raises(ValueError, match="per-request features requested"):
        fpm_learned.train([[_decode_fpm(2, 100, 0.01)]] * 30, "decode", max_iter=5, min_store_rows=10)
    artifact = fpm_learned.train(rng_iterations, "decode", max_iter=50, min_store_rows=10)
    assert artifact["schema"] == fpm_learned.SCHEMA_NAME
    assert set(artifact["stores"]) == {"pure_decode"}
    assert artifact["features"] == list(fpm_learned.REQUEST_FEATURE_NAMES)
    # Exported trees reproduce the in-distribution fit closely.
    report = fpm_learned.evaluate(artifact, rng_iterations)
    assert report["pure_decode"]["mape_pct"] < 3.0
    # Structural checks the Rust loader enforces too.
    for tree in artifact["stores"]["pure_decode"]["trees"]:
        n = len(tree["value"])
        assert all(len(tree[k]) == n for k in ("left", "right", "feature", "threshold", "missing_left"))
        for i in range(n):
            if tree["left"][i] >= 0:
                assert tree["left"][i] > i and tree["right"][i] > i


def test_rust_model_matches_reference_predictor() -> None:
    try:
        from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

        model = RustForwardPassPerfModel.from_learned(_stump_artifact())
    except Exception as exc:  # compiled extension may be stale or missing
        pytest.skip(f"compiled learned model unavailable: {exc}")
    artifact = _stump_artifact()
    for nd, kv in ((1, 50), (4, 800), (5, 800), (16, 20_000)):
        iteration = [_decode_fpm(nd, kv, 0.0)]
        assert model.estimate_forward_pass_time_ms(iteration) == pytest.approx(
            fpm_learned.predict_ms(artifact, iteration)
        )
    assert model.learned_feature_names() == artifact["features"]
    assert model.diagnostics()["source"] == "learned"
