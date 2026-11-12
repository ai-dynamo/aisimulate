# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize the public accuracy summary without changing saved evidence identities."""

from __future__ import annotations

from copy import deepcopy

SUMMARY_VERSION = 2
SNAPSHOT_FIELDS = ("completed_at", "method", "packages", "sot_sha256")


def normalize_summary(value: dict, *, target_version: int = SUMMARY_VERSION) -> dict:
    """Read v1/v2 summaries in the reader's chosen version; reject mixed fields.

    Only public series and snapshot field names change. Distribution names,
    URLs, campaign envelopes, identifiers, and numeric results remain intact.
    The caller must verify any source-byte checksum before normalization.
    """
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("unsupported accuracy summary schema")
    version = value["schema_version"]
    if (
        version not in (1, SUMMARY_VERSION)
        or type(target_version) is not int
        or target_version not in (1, SUMMARY_VERSION)
    ):
        raise ValueError("unsupported accuracy summary schema")
    result = deepcopy(value)
    old, other = ("aisimulate", "aisim") if version == 1 else ("aisim", "aisimulate")
    new = "aisimulate" if target_version == 1 else "aisim"

    def field(item, source, destination, unexpected):
        if not isinstance(item, dict):
            return
        if unexpected in item:
            raise ValueError(f"summary v{version} contains unexpected field {unexpected}")
        if version != target_version and source in item:
            renamed = [(destination if key == source else key, value) for key, value in item.items()]
            item.clear()
            item.update(renamed)

    def series(item):
        field(item, old, new, other)

    def children(item, key):
        values = item.get(key, []) if isinstance(item, dict) else []
        return values if isinstance(values, list) else []

    snapshot = result.get("snapshot")
    for suffix in SNAPSHOT_FIELDS:
        field(snapshot, f"{old}_{suffix}", f"{new}_{suffix}", f"{other}_{suffix}")
    totals = result.get("totals")
    series(totals)
    groups = totals.get("by_configuration_quality", {}) if isinstance(totals, dict) else {}
    if isinstance(groups, dict):
        for group in groups.values():
            series(group)
    for model in children(result, "models"):
        series(model)
        for workload in children(model, "workloads"):
            series(workload)
            for gpu in children(workload, "gpus"):
                series(gpu)
                for topology in children(gpu, "topologies"):
                    series(topology)
                    for point in children(topology, "points"):
                        series(point)
    result["schema_version"] = target_version
    return result
