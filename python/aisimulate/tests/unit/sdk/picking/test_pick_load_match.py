# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparison-score contracts for direct SDK load matching."""

import pandas as pd
import pytest

from aisimulate.sdk.picking import pick_load_match

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize(
    "target, gpus_per_replica, gpu_cap, expected_score, expected_load_pct",
    [
        (1.0, 2, 2, 0.5, None),  # Exact-budget fits still minimize GPUs.
        (1.0, 2, 4, 0.5, None),
        (10.0, 2, 4, 100.0, 20.0),
        (10.0, 3, 4, 75.0, 10.0),  # A capped replica can leave unused GPUs.
    ],
)
def test_load_match_comparison_score_uses_actual_fallback(
    load_target, target, gpus_per_replica, gpu_cap, expected_score, expected_load_pct
):
    frame = pd.DataFrame(
        {
            "parallel": ["single"],
            "tpot": [25.0],
            "tokens/s/user": [40.0],
            "tokens/s/gpu": [100.0],
            "seq/s": [1.0],
            "concurrency": [1.0],
            "num_total_gpus": [gpus_per_replica],
        }
    )

    result = pick_load_match(
        frame, serving_mode="agg", target_tpot=30.0, max_total_gpus=gpu_cap, **{load_target: target}
    )

    assert result["best_throughput"] == expected_score
    row = result["best_config_df"].iloc[0]
    if expected_load_pct is None:
        assert "load_served_pct" not in row.index
        assert row["total_gpus_needed"] == gpus_per_replica
    else:
        assert row["load_served_pct"] == expected_load_pct
        assert row["tokens/s/gpu_cluster"] == expected_score


def _load_frame(tpots, gpus):
    return pd.DataFrame(
        {
            "parallel": [f"config-{i}" for i in range(len(tpots))],
            "tpot": tpots,
            "request_latency": tpots,
            "tokens/s/user": [40.0] * len(tpots),
            "tokens/s/gpu": [100.0] * len(tpots),
            "seq/s": [1.0] * len(tpots),
            "concurrency": [1.0] * len(tpots),
            "num_total_gpus": gpus,
        }
    )


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize("sla_target", ["target_tpot", "target_request_latency"])
def test_load_match_sla_fallback_respects_gpu_cap(load_target, sla_target):
    result = pick_load_match(
        _load_frame([40.0, 50.0], [2, 3]),
        serving_mode="agg",
        max_total_gpus=4,
        **{load_target: 10.0, sla_target: 30.0},
    )

    configs = result["best_config_df"]
    assert (configs["total_gpus_needed"] <= 4).all()
    assert configs["tpot"].tolist() == [40.0, 50.0]
    assert not configs["sla_satisfied"].any()
    assert not configs["load_satisfied"].any()
    assert configs["gpu_budget_satisfied"].all()
    assert result["best_throughput"] == 100.0


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize("tpot", [25.0, 40.0])
@pytest.mark.parametrize("gpu_cap", [0, 1])
def test_load_match_rejects_single_replica_over_budget(load_target, tpot, gpu_cap):
    result = pick_load_match(
        _load_frame([tpot], [2]),
        serving_mode="agg",
        target_tpot=30.0,
        max_total_gpus=gpu_cap,
        **{load_target: 10.0},
    )

    assert result["best_config_df"].empty
    assert result["best_throughput"] == 0.0


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
def test_load_match_keeps_budget_feasible_sla_diagnostic(load_target):
    result = pick_load_match(
        _load_frame([25.0, 40.0], [8, 2]),
        serving_mode="agg",
        target_tpot=30.0,
        max_total_gpus=4,
        **{load_target: 1.0},
    )

    row = result["best_config_df"].iloc[0]
    assert row["total_gpus_needed"] == 2
    assert not row["sla_satisfied"]
    assert row["load_satisfied"]
    assert row["gpu_budget_satisfied"]
    assert result["best_throughput"] == 100.0


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize("tpot", [25.0, 40.0])
def test_load_match_without_budget_preserves_sizing(load_target, tpot):
    result = pick_load_match(_load_frame([tpot], [2]), serving_mode="agg", target_tpot=30.0, **{load_target: 10.0})

    row = result["best_config_df"].iloc[0]
    assert row["total_gpus_needed"] == 20
    assert result["best_throughput"] == 0.05
    assert row["sla_satisfied"] == (tpot <= 30.0)
    assert row["load_satisfied"]
    assert row["gpu_budget_satisfied"]


@pytest.mark.filterwarnings("error:The inplace keyword in DataFrame.reset_index is deprecated:DeprecationWarning")
@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize(
    "gpu_cap, loads, gpus, tpots, expected",
    [
        pytest.param(
            None,
            [3.0, 10.0, 5.0],
            [2, 3, 1],
            [25.0, 20.0, 15.0],
            (["B", "A-fast"], [2, 1], [2, 3], None, 0.5),
            id="unbudgeted",
        ),
        pytest.param(
            4,
            [3.0, 10.0, 5.0],
            [2, 3, 1],
            [25.0, 20.0, 15.0],
            (["B", "A-fast"], [2, 1], [2, 3], None, 0.5),
            id="budget-filters-one-candidate",
        ),
        pytest.param(
            4,
            [1.0, 2.0, 3.0],
            [2, 3, 2],
            [25.0, 20.0, 15.0],
            (["A-fast", "B"], [1, 2], [3, 4], [20.0, 50.0], 150.0),
            id="gpu-capped",
        ),
        pytest.param(
            None,
            [3.0, 10.0, 5.0],
            [2, 3, 1],
            [50.0, 40.0, 45.0],
            (["A-fast", "B"], [1, 2], [3, 2], None, 1.0 / 3),
            id="closest-sla",
        ),
    ],
)
def test_load_match_handles_duplicate_sweep_indices(load_target, gpu_cap, loads, gpus, tpots, expected):
    """Concatenating sweeps must preserve row identity through filtering and grouping."""
    frame = _load_frame(tpots, gpus)
    frame["parallel"] = ["A", "A", "B"]
    frame["config_id"] = ["A-slow", "A-fast", "B"]
    frame["tokens/s/gpu"] = [100.0, 200.0, 120.0]
    frame["seq/s"] = 100.0
    frame["concurrency"] = 100.0
    frame["seq/s" if load_target == "target_request_rate" else "concurrency"] = loads
    frame["load_served_pct"] = 99.0  # A stale marker must be removed or recomputed.
    frame = pd.concat([frame.iloc[[i]].reset_index(drop=True) for i in range(len(frame))])
    original = frame.copy(deep=True)

    result = pick_load_match(
        frame,
        serving_mode="agg",
        target_tpot=30.0,
        max_total_gpus=gpu_cap,
        **{load_target: 10.0},
    )

    configs = result["best_config_df"]
    names, replicas, total_gpus, load_pct, score = expected
    assert configs["config_id"].tolist() == names
    assert configs["replicas_needed"].tolist() == replicas
    assert configs["total_gpus_needed"].tolist() == total_gpus
    assert configs["sla_satisfied"].tolist() == [max(tpots) <= 30.0] * len(names)
    assert configs["load_satisfied"].tolist() == [load_pct is None] * len(names)
    assert configs["gpu_budget_satisfied"].all()
    assert result["best_throughput"] == pytest.approx(score)
    expected_columns = (set(frame.columns) - {"load_served_pct"}) | {
        "replicas_needed",
        "total_gpus_needed",
        "tokens/s/gpu_cluster",
        "sla_satisfied",
        "load_satisfied",
        "gpu_budget_satisfied",
    }
    if load_pct is not None:
        assert configs["load_served_pct"].tolist() == load_pct
        expected_columns.add("load_served_pct")
    assert set(configs.columns) == expected_columns
    pd.testing.assert_index_equal(configs.index, pd.RangeIndex(len(names)))
    pd.testing.assert_frame_equal(frame, original)
