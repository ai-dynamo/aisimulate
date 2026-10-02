# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact post-release TRT serve defaults for the September legacy recipes.

The scopes below intentionally exclude Dynamo workers: their image version is
not their TRT version, and their startup path is different from trtllm-serve.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import requests

from e2e_accuracy_source.inferencex_recipe import InferenceXRecipeError

SOURCES = json.loads(Path(__file__).with_name("trt_additional_default_sources.json").read_text())
_ARCHITECTURES = {
    "1.1.0rc2.post2": {"DeepseekV3ForCausalLM"},
    "1.2.0rc0.post1": {"GptOssForCausalLM"},
}
_BLACKWELL = {"b200", "b300", "gb200", "gb300"}
_HOPPER = {"h100", "h200"}


@cache
def _verified_sources(version: str) -> list[dict]:
    records = SOURCES[version]
    try:
        for record in records:
            response = requests.get(record["url"], timeout=30)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != record["sha256"]:
                raise InferenceXRecipeError(f"reviewed TRT source changed: {record['url']}")
    except requests.RequestException as error:
        raise InferenceXRecipeError(f"cannot verify TRT post-release defaults: {error}") from error
    return records


def apply_additional_trt_defaults(
    args: dict[str, Any],
    version: str | None,
    *,
    aggregated: bool,
    hardware: str,
    checkpoint: dict | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Resolve reviewed serve controls; preserve unknown versions and contexts."""
    result = dict(args)
    config = checkpoint or {}
    architecture = next(iter(config.get("architectures") or []), None)
    if (
        version not in SOURCES
        or not aggregated
        or architecture not in _ARCHITECTURES.get(version, set())
        or hardware not in _BLACKWELL | _HOPPER
        or args.get("backend", "pytorch") != "pytorch"
        or args.get("speculative_config")
        or args.get("sparse_attention_config")
        or args.get("attn_backend", "TRTLLM") != "TRTLLM"
    ):
        return result, []
    records = _verified_sources(version)
    evidence = []

    def put(knob: str, value: Any, rule: str, force: bool = False):
        previous = result.get(knob)
        if (previous is None or force) and previous != value:
            result[knob] = value
            evidence.append(
                dict(
                    knob=knob,
                    version=version,
                    value=value,
                    previous=previous,
                    kind="runtime_override" if previous is not None else "verified_default",
                    rule=rule,
                    sources=records,
                )
            )

    for source, target in (("max_batch_size", "max_num_seqs"), ("max_num_tokens", "max_num_batched_tokens")):
        if args.get(source) is not None:
            put(target, args[source], f"explicit trtllm-serve {source}", force=True)
    put("max_num_seqs", 2048, "trtllm-serve -> BuildConfig.max_batch_size -> TorchLlmArgs -> executor")
    put("max_num_batched_tokens", 8192, "trtllm-serve -> BuildConfig.max_num_tokens -> model engine")
    put("enable_chunked_prefill", False, "BaseLlmArgs.enable_chunked_prefill -> executor enable_chunked_context")
    maximum = args.get("max_seq_len", args.get("max_model_len"))
    batch = result.get("max_num_seqs")
    tokens = result.get("max_num_batched_tokens")
    if all(isinstance(x, int) and x > 0 for x in (maximum, batch, tokens)):
        put("max_num_batched_tokens", min(tokens, maximum * batch), "model_engine._init_max_num_tokens cap", force=True)
    kv = args.get("kv_cache_config") or {}
    if version == "1.2.0rc0.post1" and kv.get("tokens_per_block") is not None:
        put("block_size", kv["tokens_per_block"], "explicit KvCacheConfig.tokens_per_block", force=True)
    text = config.get("text_config") or config
    mla = bool(text.get("kv_lora_rank") and text.get("qk_rope_head_dim"))
    flash_mla = mla and text["kv_lora_rank"] + text["qk_rope_head_dim"] == 576 and hardware in _HOPPER
    if flash_mla:
        put("block_size", 64, "SM90 MLA head_dim=576 -> ModelConfig.enable_flash_mla -> executor override", force=True)
    else:
        rule = (
            "update_executor_config -> default BuildConfig.PluginConfig.tokens_per_block"
            if version == "1.1.0rc2.post2"
            else "KvCacheConfig.tokens_per_block -> py_executor_creator"
        )
        put("block_size", 32, rule)
    if mla and hardware in {"b300", "gb300"}:
        # These exact releases only permit SM90/SM100 MLA chunking.
        put("enable_chunked_prefill", False, "executor disables MLA chunking outside SM90/SM100", force=True)
    # KV dtype, cache reuse and memory fraction are explicit in these archived
    # families. Leave omissions unresolved rather than importing newer rules.
    return result, evidence
