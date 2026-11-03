# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from audit_geometry import qualify_rows


def records():
    result = []
    for context in (8192, 393216):
        prefill = 8192 if context == 8192 else 64
        for phase, tokens, graph in [("step[EXTEND bs=1]", prefill, False),
                                      ("step[DRAFT bs=1]", prefill, False)] + [
                                          item for _ in range(11) for item in [
                                              ("draft", 1, True), ("step[VERIFY bs=1]", 4, False),
                                              ("draft_extend", 4, True)]]:
            result.append({"schema": "aisimulate.glm-rsi.forward-geometry.v1", "ordinal": len(result),
                           "device": 0, "batch_size": 1, "begin_ns": "1", "end_ns": "2",
                           "seq_lens": [context], "input_token_count": tokens, "stage": phase,
                           "num_tokens_per_req": tokens, "graph_entry": graph,
                           "positions_bounded": list(range(context - tokens, context)) if tokens <= 64 else None,
                           "dsa_topk_seed_present": False})
    return result


def test_geometry_keeps_all_executions_and_rejects_incomplete_or_wrong_width():
    rows = records()
    result = qualify_rows(rows, 0)
    assert result["8192"]["counts"]["target_verify"] == 11
    assert sum(len(value["forwards"]) for value in result.values()) == len(rows)
    changed = copy.deepcopy(rows)
    changed[3]["input_token_count"] = 3
    with pytest.raises(ValueError, match="token width"):
        qualify_rows(changed, 0)
    changed = copy.deepcopy(rows)
    changed[3]["positions_bounded"] = [1, 2, 2, 4]
    with pytest.raises(ValueError, match="consecutive"):
        qualify_rows(changed, 0)
    with pytest.raises(ValueError, match="rank"):
        qualify_rows(rows, 1)
    with pytest.raises(ValueError, match="Incomplete numerical profile"):
        qualify_rows(rows[:35], 0)
