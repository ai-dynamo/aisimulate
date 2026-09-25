# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit analysis groups retaining every original native SG graph policy.

Compatibility is derived from checked native snapshots, original allocated
layouts and named call ownership. Process-local receipt hashes are never
rewritten. Native source/API attribution remains in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from pathlib import Path

from collector.glm53flash_contract import canonical_json, sha256_json

CONTRACT = "sglang_named_graph_group_v1"
COLUMNS = ("graph_group", "graph_group_sha256", "graph_member_id")
POLICY_FIELDS = (
    "schema_version",
    "backend",
    "backend_version",
    "backend_revision",
    "checkpoint_format",
    "checkpoint_revision",
    "tp_size",
    "phase",
    "runtime_mode",
    "capture_sizes",
    "disable_padding",
    "captured_req_width",
    "native_flags",
    "source_pins",
    "source_sha256",
    "config_sha256",
    "runtime_digest",
)


def enabled(run):
    value = run["spec"].get("ops_graph_group_contract")
    if value not in (None, CONTRACT):
        raise ValueError("unknown native graph analysis group contract")
    if value is not None and (
        run["key"][0] != "sglang" or run["spec"].get("ops_execution_mode") != "native_full_graph"
    ):
        raise ValueError("graph group analysis requires native SGLang FULL decode")
    return value == CONTRACT


def compatibility(snapshot, execution_policy, layouts, captures):
    """Called only after the reader rederives each original capture/forward.

    Keep all tensor dimensions/strides/capacities in this first bounded rule.
    Actual UUID stays in its original hash-bound layout; each GPU is validated
    independently. No native input or receipt is altered by this comparison.
    """
    from collector.glm53flash_validation import _state_layout

    state = {}
    for rank, layout in sorted(layouts.items()):
        _state_layout(layout, "sglang")
        if layout.get("tp_rank") != rank:
            raise ValueError("graph group layout belongs to another native rank")
        hardware = layout["hardware"]
        state[str(rank)] = {
            **{
                key: layout[key]
                for key in (
                    "admitted",
                    "groups",
                    "logical_kv_dtype",
                    "physical_kv_dtype",
                    "pooled_index_layout",
                    "pool_class",
                    "full_pool_class",
                    "tp_rank",
                )
            },
            "hardware_class": {
                key: hardware[key]
                for key in (
                    "schema",
                    "compute_capability",
                    "cuda_device_index",
                    "name",
                    "total_memory_bytes",
                )
            },
        }
    ownership = {}
    for rank, registry in sorted(captures.items()):
        if not registry:
            continue
        ownership[str(rank)] = [
            {
                "native_shape_key": capture["native_shape_key"],
                "capture_scope": capture["capture_scope"],
                "operations": capture["operations"],
                "native_api_libraries": {
                    name: {
                        key: capture["native_api_libraries"][name][key]
                        for key in (("sha256", "runtime_version", "abi") if name == "cudart" else ("sha256",))
                    }
                    for name in ("cudart", "cupti")
                },
                "calls": [
                    {key: call[key] for key in ("name", "source", "index", "completed")} for call in capture["calls"]
                ],
                "setup_ownership": "validated_native_capture_node_complement_v1",
            }
            for _, capture in sorted(registry.items())
        ]
    return {
        "execution_policy": execution_policy,
        "native_snapshot_sha256": sha256_json(snapshot),
        "state_layout_sha256": sha256_json(state),
        "source_ownership_sha256": sha256_json(ownership) if ownership else None,
    }


def native_policy(native):
    """Use one validated member's common declarations, never an aggregate policy."""
    if "_children" in native and native.get("graph_group_contract") == CONTRACT:
        children = native["_children"]
        if not children:
            raise ValueError("graph group has no original native members")
        return children[sorted(children)[0]]["graph_policy"]
    return native["graph_policy"]


def same_members(left, right):
    from collector.fpm_forward.glm53flash_validation import _same_sglang_policy

    _same_sglang_policy(left, right, "explicit native graph group")
    a, b = left.get("graph_group_compatibility"), right.get("graph_group_compatibility")
    if not isinstance(a, dict) or not a or a != b:
        raise ValueError("graph group changed actual native snapshot, allocated layout or named ownership")
    lp, rp = native_policy(left), native_policy(right)
    if lp.get("schema_version") == 1 and rp.get("schema_version") == 1:
        if any(lp[key] != rp[key] for key in POLICY_FIELDS):
            raise ValueError("graph group changed native execution declarations")
    elif set(lp) == set(rp) == {"native_snapshot", "provenance"}:
        if lp["native_snapshot"] != rp["native_snapshot"] or any(
            lp["provenance"][key] != rp["provenance"][key]
            for key in ("source_sha256", "config_sha256", "runtime_digest", "checkpoint_revision")
        ):
            raise ValueError("graph group changed independent native holdout declarations")
    else:
        raise ValueError("graph group mixes calibration and independent native truth")


def make_group(parent, members, compatibility_value):
    if not enabled(parent) or not compatibility_value.get("source_ownership_sha256"):
        raise ValueError("graph group requires explicit opt-in and original named capture ownership")
    return {
        "contract": CONTRACT,
        "source_plan_sha256": parent["plan"]["sha256"],
        "shard_manifest_sha256": sha256_json(parent["shard_manifest"]),
        "corpus_sha256": parent["corpus"],
        "compatibility": compatibility_value,
        "members": dict(sorted(members.items())),
    }


def annotate(rows, group):
    encoded, digest = canonical_json(group), sha256_json(group)
    for row in rows:
        row.update(graph_group=encoded, graph_group_sha256=digest)


def validate_audit(audit, group):
    """Bind actual Rust-selected endpoints to their original member records."""
    if audit.get("group_contract") != CONTRACT or audit.get("graph_group_sha256") != sha256_json(group):
        raise ValueError("graph endpoint audit changed its analysis group binding")
    if "graph_policy_sha256" in audit:
        raise ValueError("graph group audit invents one aggregate native policy")
    for operation in audit.get("operations", []):
        for endpoint in operation.get("endpoints", []):
            member = group["members"].get(endpoint.get("graph_member_id"))
            if member is None or any(
                endpoint.get(key) != member[key]
                for key in (
                    "graph_policy_sha256",
                    "evidence_sha256",
                    "rank_selection_sha256",
                    "native_runtime_run_id",
                )
            ):
                raise ValueError("graph endpoint audit changed original member policy/evidence")
            point = {
                key: endpoint.get(key)
                for key in (
                    "batch_size",
                    "prefix",
                    "padded_batch_size",
                    "native_benchmark_id",
                    "original_point_id",
                )
            }
            if point not in member["points"]:
                raise ValueError("graph endpoint audit changed original point ownership")


def validate_independent_holdout(native, binding, holdout):
    """A held-out native attempt cannot reuse a calibration or control identity."""
    attempts = [
        {
            "evidence_root": child["evidence_root"],
            "native_runtime_run_id": child["runtime_run_id"],
            "request_ids": child["request_ids"],
        }
        for child in native["_children"].values()
    ] + [receipt["control"] for receipt in binding["children"]]
    for original in attempts:
        if (
            holdout["runtime_run_id"] == original["native_runtime_run_id"]
            or Path(holdout["evidence_root"]).resolve() == Path(original["evidence_root"]).resolve()
            or set(holdout["request_ids"]) & set(original["request_ids"])
        ):
            raise ValueError("graph group holdout reused a calibration/control native attempt")
