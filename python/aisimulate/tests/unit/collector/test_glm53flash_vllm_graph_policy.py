# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY initialized dispatch fixtures; no GPU or measured data claim."""

import copy

import pytest

from collector.glm53flash_contract import BACKENDS
from collector.glm53flash_vllm_graph_policy import FLAGS, SOURCE_PINS, select_descriptor, validate_snapshot

pytestmark = pytest.mark.unit


def snapshot():
    full = [
        {
            "cg_mode": "FULL",
            "num_tokens": n,
            "num_reqs": n,
            "uniform_token_count": 1,
            "max_query_len": None,
            "num_active_loras": 0,
            "num_ubatches": 1,
        }
        for n in (1, 2, 4)
    ]
    pw = [{**row, "cg_mode": "PIECEWISE", "num_reqs": None, "uniform_token_count": None} for row in full]
    return {
        "backend": "vllm",
        "backend_version": BACKENDS["vllm"][0],
        "backend_revision": BACKENDS["vllm"][1],
        "source_pins": dict(SOURCE_PINS),
        "native_flags": dict.fromkeys(FLAGS, False),
        "capture_sizes": [1, 2, 4],
        "max_num_reqs": 4,
        "max_capture_tokens": 4,
        "decode_query_len": 1,
        "graphs_captured": True,
        "lora_capture_cases": [0],
        "dp_size": 1,
        "tp_size": 4,
        "tp_rank": 0,
        "resolved_mode": "FULL_AND_PIECEWISE",
        "use_breakable_cg": True,
        "capture_descriptors": {"FULL": list(reversed(full)), "PIECEWISE": list(reversed(pw))},
        "full_graphs": full,
        "candidates": [
            {"num_tokens": n, "num_active_loras": 0, "descriptors": [full[i], pw[i]]}
            for n, i in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 2))
        ],
        "piecewise_entries": [
            {
                "num_tokens": n,
                "num_reqs": None,
                "uniform": False,
                "has_lora": False,
                "num_active_loras": 0,
                "completed": True,
                "num_graphs": 46,
                "num_eager_breaks": 45,
            }
            for n in (1, 2, 4)
        ],
    }


def test_native_decode_and_prefill_same_token_count_keep_different_execution_identity():
    value = snapshot()
    decode = select_descriptor(value, batch=3, query=1, is_context=False)
    prefill = select_descriptor(value, batch=3, query=1, is_context=True)
    assert decode["cg_mode"] == "FULL" and decode["num_tokens"] == decode["num_reqs"] == 4
    assert prefill["cg_mode"] == "PIECEWISE" and prefill["num_tokens"] == 4 and prefill["num_reqs"] is None
    assert select_descriptor(value, batch=1, query=8192, is_context=True)["cg_mode"] == "NONE"


def test_none_prefill_is_native_policy_without_piecewise_not_a_collector_fallback():
    value = snapshot()
    value.update(resolved_mode="FULL_DECODE_ONLY", use_breakable_cg=False, piecewise_entries=[])
    del value["capture_descriptors"]["PIECEWISE"]
    for row in value["candidates"]:
        row["descriptors"] = row["descriptors"][:1]
    assert select_descriptor(value, batch=3, query=1, is_context=False)["cg_mode"] == "FULL"
    assert select_descriptor(value, batch=3, query=1, is_context=True)["cg_mode"] == "NONE"


@pytest.mark.parametrize("flag", FLAGS)
def test_actual_extra_native_predicate_cannot_be_ignored(flag):
    value = snapshot()
    value["native_flags"][flag] = True
    with pytest.raises(ValueError, match="predicates"):
        validate_snapshot(value)


@pytest.mark.parametrize(
    "defect",
    [
        "uncaptured",
        "priority",
        "bool_pad",
        "foreign_source",
        "missing_bucket",
        "incomplete_pw",
        "uniform_pw",
        "unknown_version",
    ],
)
def test_source_and_complete_initialized_capture_inventory_are_required(defect):
    value = copy.deepcopy(snapshot())
    if defect == "uncaptured":
        value["graphs_captured"] = False
    elif defect == "priority":
        value["candidates"][3]["descriptors"].reverse()
    elif defect == "bool_pad":
        value["full_graphs"][0]["num_tokens"] = True
    elif defect == "foreign_source":
        value["source_pins"][next(iter(SOURCE_PINS))] = "0" * 64
    elif defect == "missing_bucket":
        value["full_graphs"].pop()
    elif defect == "incomplete_pw":
        value["piecewise_entries"][0]["completed"] = False
    elif defect == "uniform_pw":
        value["piecewise_entries"][0]["uniform"] = True
    elif defect == "unknown_version":
        value["backend_version"] += ".unqualified"
    with pytest.raises(ValueError):
        validate_snapshot(value)


@pytest.mark.parametrize(
    "batch,query,context", [(True, 1, False), (0, 1, False), (5, 1, False), (1, 2, False), (1, 0, True)]
)
def test_missing_native_eligibility_is_not_inferred_from_token_total(batch, query, context):
    with pytest.raises(ValueError, match="geometry"):
        select_descriptor(snapshot(), batch=batch, query=query, is_context=context)
