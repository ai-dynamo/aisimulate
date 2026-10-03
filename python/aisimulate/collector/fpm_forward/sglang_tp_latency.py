# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang TP>1 forward-latency reduction for GLM same-request FPM points.

The native SGLang producer records one device-timer duration
(``native_forward_ms``) per TP rank and forward; there are no per-rank start or
end timestamps. Rank 0's duration also includes waiting for later-arriving
ranks inside the collectives, which inflates small decode points. Each
measured repetition therefore takes the fastest TP rank's duration. With a
common collective end, that rank's interval equals common end minus latest
start. The point latency is the median over the measured repetitions.

The producer keeps publishing the rank-0 median, and the reader keeps checking
it against the raw traces. The row ``measurement_policy`` label stays
``sglang_native_real_hybrid_median_v1`` because the installed consumer admits
only that label; this module's name records the reduction in validation
evidence.
"""

from __future__ import annotations

import statistics

from .sglang_artifact import MEASUREMENTS, WARMUPS

POLICY = "sglang_tp_fastest_rank_duration_median_v1"


def fastest_rank_median_seconds(observations: dict, benchmark_id: int) -> float:
    """Median over measured repetitions of the per-repetition fastest TP rank, in seconds."""

    ranks = sorted(observations)
    if not ranks or ranks != list(range(len(ranks))):
        raise ValueError("SGLang TP rank observations must be contiguous from rank zero")
    repetitions = [
        min(float(observations[rank][benchmark_id, repetition]["native_forward_ms"]) for rank in ranks)
        for repetition in range(WARMUPS, WARMUPS + MEASUREMENTS)
    ]
    return statistics.median(repetitions) / 1000
