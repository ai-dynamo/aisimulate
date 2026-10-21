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
