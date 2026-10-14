# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pair op-cost decomposition with SSE-gap observations, without accuracy claims."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playground", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for k in [3, 5, 7]:
        model = RustForwardPassPerfModel.best_available(
            {
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "system": "b200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "worker_type": "aggregated",
                "tp": 4,
                "moe_tp_size": 1,
                "moe_ep_size": 4,
                "kvcache_quant_mode": "fp8",
                "estimation_mode": "op_level",
                "speculation": {
                    "kind": "dspark",
                    "params": {
                        "num_speculative_tokens": k,
                        "adaptive_verification": False,
                        "draft_config": {"dspark_block_size": 5, "dspark_target_layer_ids": [40, 41, 42]},
                    },
                },
            }
        )
        path = args.playground / f"measurements/static_batch/wave_dspark{k}.jsonl"
        observations = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        observations = [row for row in observations if not row["warmup"]]
        for batch in sorted({row["c"] for row in observations}):
            all_group = [row for row in observations if row["c"] == batch]
            group = [row for row in all_group if "round_ms_p50" in row]
            counter_rounds = [row["round_ms_counter"] for row in all_group if "round_ms_counter" in row]
            observed = statistics.median(row["round_ms_p50"] for row in group)
            # These prompts are approximately 64 tokens with 200 forced outputs.
            # Show the cost range across their decode span, not a fitted KV point.
            endpoints = []
            for context in [64, 264]:
                ops = model.static_phase_diagnostics(batch_size=batch, context_length=context, prefill=False)
                draft = sum(op["latency_ms"] for op in ops if op["name"].startswith("draft_"))
                target = sum(op["latency_ms"] for op in ops if not op["name"].startswith("draft_"))
                assert draft > 0 and target > 0
                endpoints.append(
                    {"context": context, "draft_ms": draft, "verify_ms": target, "total_ms": draft + target}
                )
            drafts = sum(row["counters"].get("vllm:spec_decode_num_drafts_total", 0) for row in group)
            accepted = sum(row["counters"].get("vllm:spec_decode_num_accepted_tokens_total", 0) for row in group)
            rows.append(
                {
                    "k": k,
                    "batch": batch,
                    "observation_count": len(group),
                    "source": str(path.relative_to(args.playground)),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "observed_sse_gap_median_ms": observed,
                    "observed_counter_round_median_ms": statistics.median(counter_rounds) if counter_rounds else None,
                    "counter_observation_count": len(counter_rounds),
                    "measured_accepted_drafts_per_round": accepted / drafts if drafts else None,
                    "predictions": endpoints,
                    "diagnostic_gap_percent": [100 * (p["total_ms"] - observed) / observed for p in endpoints],
                    "eligible_for_accuracy": False,
                    "exclusions": [
                        "SSE chunk gap is not a measured operator round",
                        "per-round KV, graph path and adaptive verification are not established",
                        "runtime 0.25.1-dev and op database 0.24.0 are different identities",
                        "draft/verify split is predicted, not independently measured",
                    ],
                    "modeling_assumptions": model.diagnostics()["provenance"]["modeling_assumptions"],
                }
            )
    args.output.write_text(
        json.dumps({"accuracy_mape": None, "eligible_points": 0, "diagnostic_points": rows}, indent=2) + "\n"
    )
    print(f"wrote {len(rows)} diagnostic points; accuracy MAPE unavailable")


if __name__ == "__main__":
    main()
