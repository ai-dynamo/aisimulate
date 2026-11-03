#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare private coding-chat primitives; never claim measured replay history."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from aiperf.common import random_generator as rng
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ModelEndpointInfo, RequestInfo
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset.content import PromptConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.workers.session_manager import UserSession


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("agentx-harness") != "1.0.12":
        raise ValueError("Requires the qualified AgentX 1.0.12 harness")
    if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision):
        raise ValueError("Pin the model to a full immutable revision")
    profile_bytes = args.profile.read_bytes()
    cfg = BenchmarkConfig.model_validate(json.loads(profile_bytes)["input_config"])
    dataset = cfg.get_default_dataset()
    if dataset.dataset != "semianalysis_cc_traces_weka_062126" or cfg.get_prompt_corpus() not in (None, "coding"):
        raise ValueError("Requires the original coding-corpus configuration")
    if dataset.random_seed != 42 or cfg.endpoint.type != "chat":
        raise ValueError("Requires the recorded seed and chat endpoint")
    args.output.mkdir(exist_ok=False, parents=True)
    rng.init(42)
    tokenizer = Tokenizer.from_pretrained(
        cfg.tokenizer.name, trust_remote_code=cfg.tokenizer.trust_remote_code,
        revision=args.model_revision, resolve_alias=False,
    )
    generator = CodingContentGenerator(config=PromptConfig(), tokenizer=tokenizer)
    run = BenchmarkRun(benchmark_id="rsi-source", cfg=cfg, artifact_dir=args.output, random_seed=42)
    with args.trace.open() as stream:
        raw_trace = next(stream).encode()
    trace = WekaTrace.model_validate_json(raw_trace)
    loader = WekaTraceLoader(run=run, prompt_generator=generator)
    conversations = loader.convert_to_conversations({trace.id: [trace]})
    conversation = next(c for c in conversations if c.is_root)
    if conversation.context_mode is None:
        raise ValueError("Loader did not resolve context mode")
    session = UserSession(
        x_correlation_id="rsi-source", num_turns=len(conversation.turns), conversation=conversation,
        context_mode=conversation.context_mode,
    )
    turns = [(i, t) for i, t in enumerate(conversation.turns)
             if t.source_outer_idx is not None and t.source_inner_idx is None]
    targets = {
        size: min(turns, key=lambda it: abs(trace.requests[it[1].source_outer_idx].input_length - size))[0]
        for size in (8192, 131072, 393216)
    }
    endpoint_info = ModelEndpointInfo.from_run(run)
    endpoint = ChatEndpoint(endpoint_info)
    rows = []
    for i, turn in enumerate(conversation.turns):
        session.advance_turn(i)
        sizes = [size for size, index in targets.items() if index == i]
        if not sizes:
            continue
        info = RequestInfo(
            credit_num=i, credit_phase=CreditPhase.PROFILING, conversation_id=conversation.session_id,
            turn_index=i, x_request_id=f"rsi-{i}", x_correlation_id="rsi-source", model_endpoint=endpoint_info,
            turns=session.turn_list, system_message=conversation.system_message,
            user_context_message=conversation.user_context_message,
        )
        payload = endpoint.format_payload(info)
        ids = tokenizer._tokenizer.apply_chat_template(
            payload["messages"], tools=payload.get("tools"), tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        raw = json.dumps({"payload": payload, "input_ids": ids}, ensure_ascii=False, separators=(",", ":")).encode()
        name = f"turn-{i}.json"
        (args.output / name).write_bytes(raw)
        rows.append({
            "target_sizes": sizes, "turn_index": i, "source_outer_idx": turn.source_outer_idx,
            "source_input_tokens": trace.requests[turn.source_outer_idx].input_length,
            "actual_chat_input_tokens": len(ids), "message_count": len(payload["messages"]),
            "tools": len(payload.get("tools") or []), "payload_bytes": len(json.dumps(payload).encode()),
            "private_artifact": name, "bytes": len(raw), "sha256": digest(raw),
            "token_sha256": digest(json.dumps(ids, separators=(",", ":")).encode()),
        })
    receipt = {
        "scope": "one whole-play reconstruction; representative frontend primitives, not measured history or latency",
        "harness_version": "1.0.12", "model_revision": args.model_revision, "seed": 42,
        "profile_sha256": digest(profile_bytes), "first_trace_row_sha256": digest(raw_trace),
        "source_sha256": digest(Path(__file__).read_bytes()), "pool_tokens": len(generator._tool_pool),
        "context_mode": conversation.context_mode.value, "rows": rows,
        "worker_markers_applied": False,
        "historical_equivalence": "unqualified: AgentX scenario adds first_turn_prefix markers in the worker",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"prepared": len(rows), "actual_input_tokens": [r["actual_chat_input_tokens"] for r in rows]}))


if __name__ == "__main__":
    main()
