# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-k delta correction of synthetic_kv rows at publish time (declared in topk_correction.yaml).

A synthetic_kv cached-prefill / decode row ran the serving kernels on bounded random cache contents; where the
A/B against real_kv rows showed a bias, the declared per-(backend, component, phase) batch factor multiplies
the latency and is stored in ``kv_seed_correction`` (1.0 for every other row). Rows are corrected once: a row
that already carries a factor other than 1.0 is refused.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

CORRECTION_PATH = Path(__file__).with_name("topk_correction.yaml")


def load_corrections(path: Path = CORRECTION_PATH) -> dict:
    data = yaml.safe_load(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    return data


def batch_factor(spec: dict, batch: int) -> float:
    factors = {int(k): float(v) for k, v in spec["batch_factors"].items()}
    if any(not math.isfinite(v) or v < 1.0 for v in factors.values()):
        raise ValueError("batch_factors are finite and >= 1.0")
    if batch in factors:
        return factors[batch]
    sizes = sorted(factors)
    if batch > sizes[-1]:
        if spec.get("beyond_largest_batch") != "hold":
            raise ValueError(f"no factor declared for batch {batch}")
        return factors[sizes[-1]]
    if batch < sizes[0]:
        return factors[sizes[0]]
    lo = max(b for b in sizes if b < batch)
    hi = min(b for b in sizes if b > batch)
    t = (math.log2(batch) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return factors[lo] + t * (factors[hi] - factors[lo])


def apply_topk_correction(rows: list[dict], backend: str, corrections: dict | None = None) -> dict:
    """Multiply the declared synthetic_kv rows in place; returns counts per (component, phase)."""
    corrections = load_corrections() if corrections is None else corrections
    table = corrections.get(backend) or {}
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("kv_seed_regime") != "synthetic_kv":
            continue
        if row.get("kv_seed_correction", 1.0) != 1.0:
            raise ValueError("row already carries a kv_seed_correction")
        spec = (table.get(row["component"]) or {}).get(row["phase"])
        if spec is None:
            continue
        factor = batch_factor(spec, int(row["batch_size"]))
        row["latency"] = float(row["latency"]) * factor
        row["kv_seed_correction"] = factor
        key = f"{row['component']}/{row['phase']}"
        counts[key] = counts.get(key, 0) + 1
    return counts
