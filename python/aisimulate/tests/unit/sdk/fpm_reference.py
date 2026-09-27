# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python mirror of the learned forward-pass tree walk, for parity tests only.

Production and evaluation use the compiled Rust model; this module exists so the
exported artifact can be checked against an independent walk of the same trees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from aisimulate_core.sdk import fpm_learned as fl


def _tree_leaf_value(tree: dict[str, Any], x: Sequence[float]) -> float:
    left, right, feature, threshold, value = (tree[key] for key in fl._TREE_ARRAYS[:-1])
    missing_left = tree.get("missing_left") or []
    node = 0
    for _ in range(len(value)):  # a valid tree strictly descends, so at most n hops
        left_child, right_child = left[node], right[node]
        if left_child < 0 and right_child < 0:
            return float(value[node])
        if left_child < 0 or right_child < 0 or left_child <= node or right_child <= node:
            raise ValueError(f"malformed tree node {node}: children {left_child}/{right_child}")
        v = x[feature[node]]
        if math.isnan(v):
            go_left = bool(missing_left[node]) if missing_left else True
        else:
            go_left = v <= threshold[node]
        node = left_child if go_left else right_child
    raise ValueError("malformed tree: traversal did not reach a leaf")


def reference_predict_ms(artifact: dict[str, Any], metrics_by_rank: Sequence[dict[str, Any]]) -> float | None:
    """Pure-Python mirror of the Rust prediction for one iteration (``None`` when no store applies).

    Test tooling only: production and evaluation go through ``RustForwardPassPerfModel``.
    """
    kind = fl.classify_workload(metrics_by_rank, artifact["worker_type"])
    if kind is None:
        return 0.0
    store = artifact["stores"].get(kind)
    if store is None:
        return None
    named = fl.compute_features(metrics_by_rank)
    x = [named[name] for name in artifact["features"]]
    raw = float(store["baseline"]) + sum(_tree_leaf_value(tree, x) for tree in store["trees"])
    ms = math.exp(raw) if artifact["target"] == "log_ms" else raw
    if not math.isfinite(ms):
        return None
    return max(ms, fl.MIN_POSITIVE_PREDICTION_MS)
