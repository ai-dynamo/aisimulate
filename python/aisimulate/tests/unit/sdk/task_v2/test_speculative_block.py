# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/task_v2/test_speculative_block.py

"""Task-level `speculative:` block — the scheme-based generalization of the
legacy nextn/nextn_accepted pair."""

from __future__ import annotations

import pytest

from aisimulate.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

DRAFT_CONFIG = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}


def _task(**kw) -> Task:
    # b200_sxm: the task-level native-FP4-on-Hopper gate predates the Hopper
    # w4a16 serving identities (now with native data) — relaxing it is a
    # separate upstream item; these tests target block parsing/profile math.
    return Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V4-Flash",
        system_name="b200_sxm",
        backend_name="sglang",
        **kw,
    )


def _dspark_block(**over) -> dict:
    block = {
        "method": "dspark",
        "params": {"num_draft_tokens": 7},
        "draft_config": dict(DRAFT_CONFIG),
        "accepted_tokens": 4.09,
    }
    block.update(over)
    return block


class TestSpeculativeBlock:
    def test_dspark_block_resolves_scheme_config(self):
        t = _task(speculative=_dspark_block())
        cfg = t.build_model_config(role="agg")
        assert cfg.speculation is not None
        assert cfg.speculation.kind == "dspark"
        assert cfg.nextn == 0  # dspark never triggers mtp scaling
        profile = t.build_speculative_profile()
        assert profile.tokens_per_iteration == pytest.approx(5.09)

    def test_mtp_method_desugars_to_legacy_pair(self):
        t = _task(speculative={"method": "mtp", "params": {"depth": 2}, "accepted_tokens": 0.8})
        assert t.nextn == 2
        assert t.nextn_accepted == 0.8
        legacy = _task(nextn=2, nextn_accepted=0.8)
        assert t.build_speculative_profile() == legacy.build_speculative_profile()

    def test_conflict_with_legacy_nextn_raises(self):
        with pytest.raises(ValueError):
            _task(nextn=2, nextn_accepted=0.5, speculative=_dspark_block())

    def test_accepted_tokens_required(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(accepted_tokens=None))

    def test_accepted_bound_is_scheme_derived(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(accepted_tokens=7.5))  # > N drafted

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(typo_key=1))

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            _task(speculative={"method": "warpdrive", "accepted_tokens": 1.0})

    def test_none_method_is_noop(self):
        t = _task(speculative={"method": "none"})
        assert t.build_model_config(role="agg").speculation is None
        assert t.build_speculative_profile().tokens_per_iteration == 1.0


@pytest.mark.parametrize("serving_mode,enable_epd", [("disagg", False), ("afd", False), ("agg", True)])
def test_unsupported_task_modes_rejected_before_model_resolution(monkeypatch, serving_mode, enable_epd):
    def unexpected_model_resolution(_self):
        pytest.fail("unsupported speculative task reached model identity resolution")

    monkeypatch.setattr(Task, "_resolve_model_identity", unexpected_model_resolution)
    with pytest.raises(ValueError, match="aggregated serving without EPD"):
        Task(
            serving_mode=serving_mode,
            enable_epd=enable_epd,
            speculative={"method": "ngram", "params": {"num_speculative_tokens": 3}, "accepted_tokens": 1.8},
        )


@pytest.mark.parametrize(
    "block,error_type,match",
    [
        ({}, ValueError, "method"),
        ([], TypeError, "mapping"),
        ({"method": "mtp", "params": {"deph": 2}}, ValueError, "Unknown"),
        ({"method": "mtp", "params": {"depth": 1.5}, "accepted_tokens": 0.5}, ValueError, "integer"),
        ({"method": "mtp", "draft_model_path": "unused", "params": {"depth": 1}}, ValueError, "draft checkpoint"),
        ({"method": "none", "params": {"depth": 2}}, ValueError, "does not accept"),
        ({"method": "ngram", "params": {"typo": 3}, "accepted_tokens": 1}, ValueError, "Unknown"),
        ({"method": "ngram", "params": [], "accepted_tokens": 1}, TypeError, "mapping"),
    ],
)
def test_invalid_speculative_blocks_fail_at_task_boundary(block, error_type, match):
    with pytest.raises(error_type, match=match):
        Task(speculative=block)


def test_speculative_block_rejects_conflicting_legacy_acceptance():
    with pytest.raises(ValueError, match="Conflicting"):
        Task(
            nextn=1,
            nextn_accepted=0.5,
            speculative={"method": "mtp", "params": {"depth": 1}, "accepted_tokens": 0.8},
        )
    with pytest.raises(ValueError, match="nextn_accepted is MTP-only"):
        Task(nextn_accepted=0.5, speculative={"method": "ngram", "accepted_tokens": 0.8})


def test_task_yaml_scheme_reaches_real_agg_consumer():
    def run(accepted_tokens):
        task = Task.from_yaml(
            {
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-8B",
                "system_name": "h100_sxm",
                "backend_name": "vllm",
                "backend_version": "0.24.0",
                "gemm_quant_mode": "bfloat16",
                "kvcache_quant_mode": "bfloat16",
                "fmha_quant_mode": "bfloat16",
                "isl": 64,
                "osl": 261,
                "speculative": {
                    "method": "ngram",
                    "params": {"num_speculative_tokens": 3},
                    "accepted_tokens": accepted_tokens,
                },
            }
        )
        return task.run_single_agg(tp=1, batch_size=8)

    zero = run(0)
    accepted = run(1.8)
    assert 0 < accepted["tpot"] < zero["tpot"]
    assert accepted["tokens/s"] > zero["tokens/s"]


def test_pre_speculation_positional_task_call_keeps_attention_backend():
    # Literal call shape from the pre-migration Task signature through its
    # thirtieth positional argument, attention_backend (baseline 7dbd110f).
    task = Task(
        "agg",
        4000,
        1000,
        0,
        0,
        0,
        1,
        True,
        1000.0,
        50.0,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "op_level",
        "",
        "",
        "trtllm",
        None,
        False,
        False,
        False,
        0,
        None,
        None,
        "fa3",
    )
    assert task.attention_backend == "fa3"
    assert task.moe_backend is None
    assert task.speculative is None


@pytest.mark.parametrize("nextn", [1, "auto"])
def test_none_block_rejects_legacy_mtp(nextn):
    with pytest.raises(ValueError, match="Conflicting speculative inputs"):
        _task(nextn=nextn, speculative={"method": "none"})


def test_mtp_block_without_depth_preserves_checkpoint_auto():
    legacy = _task(nextn="auto", nextn_accepted=0.5)
    explicit = _task(nextn="auto", speculative={"method": "mtp", "accepted_tokens": 0.5})
    assert explicit.nextn == legacy.nextn > 0
    assert explicit.nextn_accepted == legacy.nextn_accepted == 0.5
    assert explicit.build_speculative_profile() == legacy.build_speculative_profile()


def test_auto_mtp_block_validates_acceptance_after_checkpoint_resolution():
    with pytest.raises(ValueError, match="resolved to nextn"):
        _task(nextn="auto", speculative={"method": "mtp", "accepted_tokens": 999})
