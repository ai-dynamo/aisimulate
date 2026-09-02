# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _BF16_CALL_PATCHED,
    _MXFP4_CALL_TAIL_PATCHED,
    _MXFP4_SETUP_PATCHED,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RETURN_ORIGINAL as _REPLAY_RETURN_ORIGINAL,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RETURN_PATCHED as _REPLAY_RETURN_PATCHED,
)
from collector.expert_popularity.patch_sglang_flashinfer_replay_recorder import (
    _RUNNER_PATCHED as _REPLAY_RUNNER_PATCHED,
)
from collector.expert_popularity.patch_sglang_hash_topk_capturer import (
    _CAPTURE_PATCHED as _HASH_CAPTURE_PATCHED,
)

pytestmark = pytest.mark.unit


def test_flashinfer_replay_bridge_records_kernel_selected_ids_without_rerouting():
    assert "routing_replay_out = torch.empty" in _REPLAY_RETURN_PATCHED
    assert 'kwargs["routing_replay_out"] = routing_replay_out' in _REPLAY_RETURN_PATCHED
    assert "recorder.on_select_experts(topk_ids=routing_replay_out)" in _REPLAY_RETURN_PATCHED
    assert "to_standard" not in _REPLAY_RETURN_PATCHED
    assert _REPLAY_RETURN_ORIGINAL.strip() == "return trtllm_fp8_block_scale_moe(**kwargs)"


def test_flashinfer_replay_bridge_observes_mxfp4_internal_routing_without_rerouting():
    assert "routing_replay_out = torch.empty" in _MXFP4_SETUP_PATCHED
    assert "routing_replay_out=routing_replay_out" in _MXFP4_CALL_TAIL_PATCHED
    assert "recorder.on_select_experts(topk_ids=routing_replay_out)" in _MXFP4_CALL_TAIL_PATCHED
    assert "trtllm_fp4_block_scale_routed_moe" not in _MXFP4_CALL_TAIL_PATCHED
    assert "to_standard" not in _MXFP4_CALL_TAIL_PATCHED


def test_flashinfer_replay_bridge_observes_bf16_internal_routing_without_rerouting():
    assert "routing_replay_out = torch.empty" in _BF16_CALL_PATCHED
    assert "routing_replay_out=routing_replay_out" in _BF16_CALL_PATCHED
    assert "recorder.on_select_experts(topk_ids=routing_replay_out)" in _BF16_CALL_PATCHED
    assert "trtllm_bf16_routed_moe" not in _BF16_CALL_PATCHED
    assert "to_standard" not in _BF16_CALL_PATCHED


def test_flashinfer_replay_bridge_keeps_hash_topk_on_routed_kernel():
    assert "TopKOutputChecker.format_is_standard" in _REPLAY_RUNNER_PATCHED
    assert "use_routed_topk = use_routed_topk or" in _REPLAY_RUNNER_PATCHED
    assert "hash-routed layers" in _REPLAY_RUNNER_PATCHED


def test_hash_topk_response_bridge_captures_produced_logical_ids_before_placement():
    capture = _HASH_CAPTURE_PATCHED.index("capturer.capture")
    placement = _HASH_CAPTURE_PATCHED.index("topk_ids_logical_to_physical")
    assert capture < placement
    assert "topk_indices=topk_ids" in _HASH_CAPTURE_PATCHED
    assert "self.layer_id is None" in _HASH_CAPTURE_PATCHED
