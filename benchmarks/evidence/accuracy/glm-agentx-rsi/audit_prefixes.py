#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct adjacent first-play root prefixes; never infer resident silicon KV."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import struct
from pathlib import Path
from types import SimpleNamespace

from aiperf.common import random_generator as rng
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models import ModelEndpointInfo, RequestInfo
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset.content import PromptConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker
from aiperf.workers.session_manager import UserSession
from aiperf.workers.worker import _apply_cache_bust
from audit_inputs import checksum, longest_prefix, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output already exists")
    if importlib.metadata.version("agentx-harness") != "1.0.12":
        raise ValueError("Requires the pinned AgentX harness")
    if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision):
        raise ValueError("Requires a full immutable model revision")
    profile = read(args.profile)
    cfg = BenchmarkConfig.model_validate(profile["input_config"])
    if cfg.get_prompt_corpus() not in (None, "coding") or cfg.get_default_dataset().random_seed != 42:
        raise ValueError("Requires the qualified coding corpus and seed")
    with args.trace.open("rb") as stream:
        raw = next(stream)
    trace = WekaTrace.model_validate_json(raw)
    if trace.block_size != 64:
        raise ValueError("Requires the recorded 64-token blocks")
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    selected = [row for row in records if row["metadata"]["source_trace_id"] == trace.id
                and row["metadata"]["source_kind"] == "weka_main"
                and row["metadata"]["benchmark_phase"] == "profiling"]
    by_turn = {row["metadata"]["turn_index"]: row for row in selected}
    if len(by_turn) != len(selected) or not selected:
        raise ValueError("Ambiguous or empty recorded first-play root cohort")
    rng.init(42)
    tokenizer = Tokenizer.from_pretrained(cfg.tokenizer.name, trust_remote_code=cfg.tokenizer.trust_remote_code,
                                         revision=args.model_revision, resolve_alias=False)
    generator = CodingContentGenerator(config=PromptConfig(), tokenizer=tokenizer)
    run = BenchmarkRun(benchmark_id="rsi-prefix-audit", cfg=cfg, artifact_dir=args.output.parent, random_seed=42)
    loader = WekaTraceLoader(run=run, prompt_generator=generator)
    conversations = loader.convert_to_conversations({trace.id: [trace]})
    conversation = next(row for row in conversations if row.is_root)
    if conversation.context_mode is None:
        raise ValueError("Unresolved context mode")
    session = UserSession(x_correlation_id="rsi-prefix", num_turns=len(conversation.turns),
                          conversation=conversation, context_mode=conversation.context_mode)
    marker = build_cache_bust_marker(profile["benchmark_id"], 0, 0, trace.id, target=CacheBustTarget.FIRST_TURN_PREFIX)
    credit = SimpleNamespace(cache_bust_marker=marker, cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX)
    endpoint_info = ModelEndpointInfo.from_run(run)
    endpoint = ChatEndpoint(endpoint_info)
    rows, previous, previous_source = [], None, None
    for index, turn in enumerate(conversation.turns):
        session.advance_turn(index)
        system = _apply_cache_bust(session, credit, conversation.system_message)
        if index not in by_turn:
            continue
        measured = by_turn[index]
        if measured["metadata"]["source_outer_idx"] != turn.source_outer_idx:
            raise ValueError("Recorded turn identity differs from reconstruction")
        info = RequestInfo(credit_num=index, credit_phase=CreditPhase.PROFILING,
                           conversation_id=conversation.session_id, turn_index=index,
                           x_request_id=f"rsi-prefix-{index}", x_correlation_id="rsi-prefix",
                           model_endpoint=endpoint_info, turns=session.turn_list, system_message=system,
                           user_context_message=conversation.user_context_message)
        payload = endpoint.format_payload(info)
        ids = tokenizer._tokenizer.apply_chat_template(payload["messages"], tools=payload.get("tools"),
                                                      tokenize=True, add_generation_prompt=True, return_dict=False)
        source = trace.requests[turn.source_outer_idx]
        usage = measured["metrics"].get("usage_prompt_tokens")
        observed = usage["value"] if isinstance(usage, dict) else usage
        row = {"turn_index": index, "source_outer_idx": turn.source_outer_idx,
               "input_tokens": len(ids), "measured_input_tokens": observed,
               "input_count_agrees": None if observed is None else len(ids) == observed,
               "token_sha256_u32_le": hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()}
        if previous is not None:
            actual_lcp = longest_prefix(previous, ids)
            source_blocks = min(longest_prefix(previous_source.hash_ids, source.hash_ids),
                                previous_source.input_length // 64, source.input_length // 64)
            row.update(reconstructed_lcp_tokens=actual_lcp, reconstructed_full_prefix_tokens=actual_lcp // 64 * 64,
                       source_full_prefix_tokens=source_blocks * 64,
                       full_prefix_delta_tokens=(actual_lcp // 64 - source_blocks) * 64)
        rows.append(row)
        previous, previous_source = ids, source
    if len(rows) != len(selected):
        raise ValueError("Recorded root request accounting is incomplete")
    deltas = [row["full_prefix_delta_tokens"] for row in rows if "full_prefix_delta_tokens" in row]
    result = {"schema": "aisimulate.glm-rsi.reconstructed-prefix-audit.v1",
              "scope": "Original first-play lane0/recycle0 root profile requests only; literal prefix geometry, "
                       "not original wire-byte identity, resident silicon cache receipts or a new accuracy cohort.",
              "harness": "1.0.12", "model_revision": args.model_revision,
              "source_sha256": checksum(Path(__file__)), "profile_sha256": checksum(args.profile),
              "records_sha256": checksum(args.records), "first_trace_row_sha256": hashlib.sha256(raw).hexdigest(),
              "rows": rows, "summary": {"recorded": len(selected), "reconstructed": len(rows),
              "count_agrees": sum(row["input_count_agrees"] is True for row in rows),
              "count_unavailable": sum(row["input_count_agrees"] is None for row in rows),
              "count_mismatch": sum(row["input_count_agrees"] is False for row in rows),
              "adjacent_pairs": len(deltas), "prefix_agrees": sum(delta == 0 for delta in deltas),
              "mean_signed_prefix_delta_tokens": statistics.mean(deltas) if deltas else None,
              "mean_absolute_prefix_delta_tokens": statistics.mean(map(abs, deltas)) if deltas else None,
              "maximum_absolute_prefix_delta_tokens": max(map(abs, deltas)) if deltas else None}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
