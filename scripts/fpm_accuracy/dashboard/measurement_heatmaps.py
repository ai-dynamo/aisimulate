# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98.
# See ../README.md for upstream paths and modifications.
"""Build compact workload heatmaps directly from measured HF observations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from fpm_accuracy.dashboard.data import Heatmap, HeatmapBin, HeatmapCell
from fpm_accuracy.hf.models import MeasurementObservation

AXES: dict[str, tuple[str, str]] = {
    "prefill": ("num_prefill_requests", "sum_prefill_tokens"),
    "decode": ("num_decode_requests", "sum_decode_kv_tokens"),
    "mixed": ("kv_read", "new_kv"),
}

_STREAM_COUNT_FIELDS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)


def measurement_stream_entry(row: Mapping[str, object]) -> bytes:
    """Return the canonical bytes for one dashboard measurement row."""

    sequence_index = row.get("sequence_index")
    if isinstance(sequence_index, bool) or not isinstance(sequence_index, int) or sequence_index < 0:
        raise ValueError("measurement stream sequence_index must be a non-negative integer")
    observation_id = row.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("measurement stream observation_id is required")
    workload = row.get("workload_kind")
    if workload not in AXES:
        raise ValueError(f"unsupported measurement workload {workload!r}")
    actual_raw = row.get("actual_ms")
    if isinstance(actual_raw, bool) or not isinstance(actual_raw, (int, float)):
        raise ValueError("measurement stream actual_ms must be positive and finite")
    actual_ms = float(actual_raw)
    if not math.isfinite(actual_ms) or actual_ms <= 0:
        raise ValueError("measurement stream actual_ms must be positive and finite")
    counts: list[int] = []
    for field in _STREAM_COUNT_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"measurement stream {field} must be a non-negative integer")
        counts.append(value)
    payload = json.dumps(
        (sequence_index, observation_id, actual_ms.hex(), workload, *counts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(payload).to_bytes(8, "big") + payload


def measurement_stream_digest(observations: Sequence[MeasurementObservation]) -> str:
    """Hash the measured fields that must survive unchanged in prediction Parquet."""

    digest = hashlib.sha256()
    for sequence_index, observation in enumerate(observations):
        scheduled = observation.scheduled
        digest.update(
            measurement_stream_entry(
                {
                    "sequence_index": sequence_index,
                    "observation_id": observation.observation_id,
                    "actual_ms": observation.actual_ms,
                    "workload_kind": str(observation.workload_kind),
                    "num_prefill_requests": scheduled.num_prefill_requests,
                    "sum_prefill_tokens": scheduled.sum_prefill_tokens,
                    "sum_prefill_kv_tokens": scheduled.sum_prefill_kv_tokens,
                    "num_decode_requests": scheduled.num_decode_requests,
                    "sum_decode_kv_tokens": scheduled.sum_decode_kv_tokens,
                }
            )
        )
    return digest.hexdigest()


def _axis_values(observation: MeasurementObservation) -> tuple[float, float]:
    scheduled = observation.scheduled
    workload = str(observation.workload_kind)
    if workload == "prefill":
        return float(scheduled.num_prefill_requests), float(scheduled.sum_prefill_tokens)
    if workload == "decode":
        return float(scheduled.num_decode_requests), float(scheduled.sum_decode_kv_tokens)
    if workload == "mixed":
        return (
            float(scheduled.sum_prefill_kv_tokens + scheduled.sum_decode_kv_tokens),
            float(scheduled.sum_prefill_tokens + scheduled.num_decode_requests),
        )
    raise ValueError(f"unsupported measured workload {workload!r}")


def _bins(values: Iterable[float], *, limit: int = 8) -> tuple[HeatmapBin, ...]:
    ordered = sorted(set(values))
    if not ordered:
        return ()
    if len(ordered) <= limit:
        return tuple(HeatmapBin(label=f"{value:g}", lower=value, upper=value) for value in ordered)
    bounds = [ordered[math.floor(index * len(ordered) / limit)] for index in range(limit)]
    result: list[HeatmapBin] = []
    for index, lower in enumerate(bounds):
        upper = bounds[index + 1] if index + 1 < len(bounds) else ordered[-1]
        if index + 1 < len(bounds):
            upper = math.nextafter(upper, -math.inf)
        result.append(HeatmapBin(label=f"{lower:g}-{upper:g}", lower=lower, upper=upper))
    return tuple(result)


def _bin_index(value: float, bins: Sequence[HeatmapBin]) -> int:
    for index, bin_ in enumerate(bins):
        if bin_.lower <= value <= bin_.upper:
            return index
    raise ValueError(f"value {value} did not fit any heatmap bin")


def measurement_workload_heatmaps(
    observations: Sequence[MeasurementObservation],
) -> dict[str, Heatmap]:
    """Summarize measured scheduled work without predictor or FPM rows."""

    coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for observation in observations:
        workload = str(observation.workload_kind)
        coordinates[workload].append(_axis_values(observation))

    result: dict[str, Heatmap] = {}
    for workload, values in coordinates.items():
        x_bins = _bins(value[0] for value in values)
        y_bins = _bins(value[1] for value in values)
        cells: dict[tuple[int, int], int] = defaultdict(int)
        for x_value, y_value in values:
            cells[(_bin_index(x_value, x_bins), _bin_index(y_value, y_bins))] += 1
        x_label, y_label = AXES[workload]
        result[workload] = Heatmap(
            x_label=x_label,
            y_label=y_label,
            x_bins=x_bins,
            y_bins=y_bins,
            cells=tuple(
                HeatmapCell(
                    x_index=x_index,
                    y_index=y_index,
                    measured_count=count,
                    predicted_count=0,
                )
                for (x_index, y_index), count in sorted(cells.items())
            ),
        )
    return result
