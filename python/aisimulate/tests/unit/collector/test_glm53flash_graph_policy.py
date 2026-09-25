# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY native policy fixtures; no synthetic runtime/performance evidence."""

import copy

import pytest
from collector.glm53flash_contract import BACKENDS, CHECKPOINTS
from collector.glm53flash_graph_policy import (
    DIRECT_FLAGS,
    EXTRA_FLAGS,
    NATIVE_SOURCE_SHA256,
    SOURCE_PINS,
    build_policy,
    padded_batch,
    validate_snapshot,
)

pytestmark = pytest.mark.unit


def snapshot(rank=0, sizes=None):
    sizes = [1, 2, 4] if sizes is None else sizes
    return {
        "backend": "sglang",
        "backend_version": BACKENDS["sglang"][0],
        "backend_revision": BACKENDS["sglang"][1],
        "source_pins": dict(SOURCE_PINS),
        "native_flags": dict.fromkeys(DIRECT_FLAGS + EXTRA_FLAGS, False),
        "capture_sizes": sizes,
        "max_bs": sizes[-1],
        "captured_req_width": 1,
        "disable_padding": False,
        "captured_keys": [
            {"size": size, "stream_idx": None, "variant_label": None, "attention_variant": None} for size in sizes
        ],
        "tp_rank": rank,
    }


def test_actual_capture_list_controls_prediction_and_formal_differs_from_pilot():
    assert padded_batch(snapshot(), 3) == 4
    assert padded_batch(snapshot(sizes=list(range(1, 33))), 3) == 3
    with pytest.raises(ValueError, match="outside"):
        padded_batch(snapshot(), 5)
    exact = snapshot()
    exact["disable_padding"] = True
    with pytest.raises(ValueError, match="outside"):
        padded_batch(exact, 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_bs", True),
        ("capture_sizes", [1, 4, 2]),
        ("captured_req_width", 2),
        ("tp_rank", False),
        ("disable_padding", 0),
        ("backend_version", "0.5.19"),
    ],
)
def test_unknown_or_inexact_native_identity_rejected(field, value):
    invalid = snapshot()
    invalid[field] = value
    with pytest.raises(ValueError):
        validate_snapshot(invalid)


def test_predicates_and_captured_variants_are_not_boolean_admission_claims():
    for field in DIRECT_FLAGS + EXTRA_FLAGS:
        invalid = snapshot()
        invalid["native_flags"][field] = True
        with pytest.raises(ValueError, match="predicates"):
            validate_snapshot(invalid)
    invalid = snapshot()
    invalid["captured_keys"][1]["attention_variant"] = "unreviewed"
    with pytest.raises(ValueError, match="keys"):
        validate_snapshot(invalid)
    invalid = snapshot()
    invalid["source_pins"][next(iter(SOURCE_PINS))] = "a" * 64
    with pytest.raises(ValueError, match="source"):
        validate_snapshot(invalid)


def test_policy_requires_same_actual_policy_across_all_tp_ranks():
    snapshots = {0: snapshot(), 1: snapshot(1)}
    args = dict(
        checkpoint_format="fp8",
        tp_size=2,
        provenance={
            "checkpoint_revision": CHECKPOINTS["fp8"][1],
            "source_sha256": NATIVE_SOURCE_SHA256,
            "config_sha256": "a" * 64,
            "runtime_digest": "sha256:" + "b" * 64,
        },
        resolved_config_sha256="c" * 64,
        state_layout_sha256={0: "d" * 64, 1: "e" * 64},
        capture_registry_sha256={0: ["f" * 64] * 3, 1: ["a" * 64] * 3},
    )
    policy = build_policy(snapshots, **args)
    assert policy["capture_sizes"] == [1, 2, 4]
    assert policy["tp_size"] == 2 and "holdout" not in policy
    bad = copy.deepcopy(snapshots)
    bad[1] = snapshot(1, sizes=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="different"):
        build_policy(bad, **args)
    with pytest.raises(ValueError, match="all TP"):
        build_policy({0: snapshot()}, **args)


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "f" * 64), ("runtime_digest", "b" * 64), ("config_sha256", "")]
)
def test_policy_rejects_unbound_model_source_and_runtime(field, value):
    provenance = {
        "checkpoint_revision": CHECKPOINTS["fp8"][1],
        "source_sha256": NATIVE_SOURCE_SHA256,
        "config_sha256": "a" * 64,
        "runtime_digest": "sha256:" + "b" * 64,
    }
    provenance[field] = value
    with pytest.raises(ValueError, match="source/config/runtime"):
        build_policy(
            {0: snapshot(), 1: snapshot(1)},
            checkpoint_format="fp8",
            tp_size=2,
            provenance=provenance,
            resolved_config_sha256="c" * 64,
            state_layout_sha256={0: "d" * 64, 1: "e" * 64},
            capture_registry_sha256={0: ["f" * 64] * 3, 1: ["a" * 64] * 3},
        )
