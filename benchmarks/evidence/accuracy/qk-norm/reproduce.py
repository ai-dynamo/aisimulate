# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only H200/vLLM Qwen3 decode diagnostic for AISimulate issue #434.

Run with the matching native extension built. The nearest table row is a
reference kernel timing, not a measurement of the complete forward pass.
"""

import json
from importlib.resources import files

import pyarrow.parquet as pq

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

path = files("aisimulate_core") / "systems/data/h200_sxm/attention/vllm/0.24.0/generation_attention_perf.parquet"
table = pq.read_table(path).to_pandas()
for model, layers, heads in [("Qwen/Qwen3-8B", 36, 32), ("Qwen/Qwen3-0.6B", 28, 16)]:
    perf = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=model,
            system="h200_sxm",
            backend="vllm",
            backend_version="0.24.0",
            worker_type="aggregated",
            estimation_mode="op_level",
        )
    )
    for batch in [1, 8]:
        for kv in [64, 576, 4096]:
            ops = perf.static_phase_diagnostics(batch_size=batch, context_length=kv, prefix=0, prefill=False)
            attention = next(op for op in ops if op["name"] == "generation_attention")
            candidates = table[
                (table.num_heads == heads)
                & (table.num_key_value_heads == 8)
                & (table.head_dim == 128)
                & (table.batch_size == batch)
                & (table.window_size == 0)
                & (table.kv_cache_dtype == "bfloat16")
                & (table.step >= kv - 1)
            ].sort_values("step")
            row = candidates.iloc[0]
            latency = perf.estimate_forward_pass_time_ms(
                {"scheduled_requests": {"num_decode_requests": batch, "sum_decode_kv_tokens": batch * kv}}
            )
            print(
                json.dumps(
                    dict(
                        model=model,
                        layers=layers,
                        batch=batch,
                        kv=kv,
                        attention_ms=attention["latency_ms"],
                        attention_source=attention["source"],
                        nearest_table_step=int(row.step),
                        nearest_kernel_ms=float(row.latency) * layers,
                        decode_step_ms=latency,
                    )
                )
            )
