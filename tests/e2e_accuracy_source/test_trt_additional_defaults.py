# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from e2e_accuracy_source import trt_additional_defaults as defaults


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(defaults, "_verified_sources", lambda version: defaults.SOURCES[version])


def test_exact_dsr1_post2_serve_defaults_keep_explicit_token_budget():
    args, evidence = defaults.apply_additional_trt_defaults(
        {"backend": "pytorch", "max_num_tokens": 1216, "max_seq_len": 2048},
        "1.1.0rc2.post2",
        aggregated=True,
        hardware="b200",
        checkpoint={"architectures": ["DeepseekV3ForCausalLM"], "kv_lora_rank": 512, "qk_rope_head_dim": 64},
    )
    assert args["max_num_seqs"] == 2048
    assert args["max_num_batched_tokens"] == 1216
    assert args["block_size"] == 32
    assert args["enable_chunked_prefill"] is False
    assert all(x["version"] == "1.1.0rc2.post2" for x in evidence)


def test_exact_gptoss_post1_preserves_explicit_batch():
    args, _ = defaults.apply_additional_trt_defaults(
        {"max_batch_size": 512, "max_num_tokens": 20000, "max_seq_len": 2048},
        "1.2.0rc0.post1",
        aggregated=True,
        hardware="b200",
        checkpoint={"architectures": ["GptOssForCausalLM"]},
    )
    assert args["max_num_seqs"] == 512
    assert args["max_num_batched_tokens"] == 20000
    assert args["block_size"] == 32
    assert args["enable_chunked_prefill"] is False


@pytest.mark.parametrize(
    "version,aggregated,hardware,architecture",
    [
        ("1.1.0rc2", True, "b200", "DeepseekV3ForCausalLM"),
        ("1.2.0rc0", True, "b200", "GptOssForCausalLM"),
        (None, True, "gb200", "GlmMoeDsaForCausalLM"),
        ("1.2.0rc0.post1", False, "b200", "GptOssForCausalLM"),
        ("1.2.0rc0.post1", True, "b200", "UnknownModel"),
    ],
)
def test_unknown_or_different_context_does_not_borrow_defaults(version, aggregated, hardware, architecture):
    original = {"max_batch_size": 8}
    assert defaults.apply_additional_trt_defaults(
        original, version, aggregated=aggregated, hardware=hardware, checkpoint={"architectures": [architecture]}
    ) == (original, [])


def test_hopper_flash_mla_forces_64_even_when_block_size_was_supplied():
    args, _ = defaults.apply_additional_trt_defaults(
        {"block_size": 32},
        "1.1.0rc2.post2",
        aggregated=True,
        hardware="h200",
        checkpoint={"architectures": ["DeepseekV3ForCausalLM"], "kv_lora_rank": 512, "qk_rope_head_dim": 64},
    )
    assert args["block_size"] == 64


def test_token_budget_model_capacity_cap():
    args, _ = defaults.apply_additional_trt_defaults(
        {"max_batch_size": 2, "max_num_tokens": 8192, "max_seq_len": 2048},
        "1.2.0rc0.post1",
        aggregated=True,
        hardware="b200",
        checkpoint={"architectures": ["GptOssForCausalLM"]},
    )
    assert args["max_num_batched_tokens"] == 4096
