#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E049: native wide-query versus flat paged-indexer controls, never perf rows."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from functools import partial
from pathlib import Path

import deep_gemm
import torch
from sglang.kernels.ops.attention import dsa_metadata
from sglang.kernels.ops.attention.dsa import (
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
    paged_mqa_logits,
)
from sglang.srt.layers.attention.dsa import dsa_indexer

PINS = {
    "test": "f24c94d50ce3c892293ee718c1e07d2cacbfc93ad0b08776cd1c1b1dbff7f686",
    "indexer": "2f9634d275bde01c6eaffa0fabc339d592d2de67fa397b2e97af673f78ffbbfc",
    "wrapper": "51591504130457f6e423f8930749ab8744e4bece16ca0cd19c79d2196634807e",
    "metadata": "91ed5448286fae496e559dc868155713ad0de4a70015a29223929a9bcc7fa5f2",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_trials(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(50):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    trials = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        trials.append(start.elapsed_time(end) / 500)
    del graph
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Check imports and source identities without GPUs")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    result = {"schema": "aisimulate.glm-rsi.paged-indexer-probe.v1", "status": "running",
              "planned_arms": 27, "attempted_arms": 0, "qualified_arms": 0, "unattempted_arms": 27,
              "cases": [], "scope": "Synthetic paged-indexer kernel only; no perf rows, model fix or request MAPE."}

    def save():
        result["unattempted_arms"] = result["planned_arms"] - result["attempted_arms"]
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        files = {"test": args.upstream_test, "indexer": Path(dsa_indexer.__file__),
                 "wrapper": Path(paged_mqa_logits.__file__), "metadata": Path(dsa_metadata.__file__)}
        result["source_sha256"] = {name: digest(path) for name, path in files.items()}
        if result["source_sha256"] != PINS:
            raise ValueError("Exact historical source gate failed")
        if importlib.metadata.version("sglang") != "0.5.20":
            raise ValueError("Requires SGLang0.5.20")
        spec = importlib.util.spec_from_file_location("rsi_pinned_upstream_test", args.upstream_test)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if args.preflight:
            result["status"] = "cpu_source_imports_qualified"
            return
        if torch.cuda.device_count() != 1:
            raise ValueError("Requires exactly one visible GPU")
        device = torch.cuda.get_device_properties(0)
        if (device.major, device.minor) != (10, 3) or "B300" not in device.name:
            raise ValueError("Requires B300 SM103")
        result["runtime"] = {"sglang": importlib.metadata.version("sglang"), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "python": platform.python_version(),
                             "gpu": device.name, "sm": [device.major, device.minor],
                             "sms": device.multi_processor_count, "deep_gemm_file_sha256": digest(deep_gemm.__file__)}
        for lanes in (1, 8, 14):
            for context in (8192, 131072, 393216):
                case = {"lanes": lanes, "context_label": context, "arms": []}
                result["cases"].append(case)
                save()
                # Reuse the pinned upstream generator, preserving its random
                # context population instead of claiming uniform lengths.
                data = module._generate_test_data(lanes, 4, 32, context, ((context * 13 // 10 + 63) // 64) * 64)
                lengths = data["context_lens"]
                case["context_lengths"] = lengths.cpu().tolist()
                maximum = data["block_table"].shape[1] * 64
                # Exact fused TARGET_VERIFY metadata writes base_seq+4 into
                # every query column; the downstream top-k causal mask is not
                # part of this isolated indexer timing.
                wide_lengths = lengths[:, None].expand(-1, 4).contiguous()
                flat_lengths = wide_lengths.reshape(-1, 1).contiguous()
                tables = data["block_table"].repeat_interleave(4, dim=0)
                q = data["q_fp8"].reshape(lanes * 4, 32, 128)
                weights = data["weights"]
                fused = data["kv_fused"]
                copies = fused.repeat(4, *([1] * (fused.ndim - 1)))
                copy_offsets = torch.arange(4, device="cuda", dtype=tables.dtype).repeat(lanes)[:, None]
                copied_tables = tables + copy_offsets * fused.shape[0]
                wide_schedule = deep_gemm.get_paged_mqa_logits_metadata(wide_lengths, 64, device.multi_processor_count)
                flat_schedule = deep_gemm.get_paged_mqa_logits_metadata(flat_lengths, 64, device.multi_processor_count)

                wide = partial(deepgemm_paged_mqa_logits_native,
                               deep_gemm.fp8_paged_mqa_logits, q, fused, weights, wide_lengths,
                               tables, wide_schedule, maximum, q_offset=lanes * 4, B=lanes, next_n=4)
                shared = partial(deepgemm_paged_mqa_logits_split,
                                 deep_gemm.fp8_paged_mqa_logits, q, fused, weights, flat_lengths,
                                 tables, flat_schedule, maximum, q_offset=lanes * 4)
                disjoint = partial(deepgemm_paged_mqa_logits_split,
                                   deep_gemm.fp8_paged_mqa_logits, q, copies, weights, flat_lengths,
                                   copied_tables, flat_schedule, maximum, q_offset=lanes * 4)

                mask = torch.arange(maximum, device="cuda")[None, :] < flat_lengths
                reference = None
                for name, fn in (("wide_native", wide), ("flat_shared_pages", shared), ("flat_copied_pages", disjoint)):
                    arm = {"name": name, "status": "running"}
                    case["arms"].append(arm)
                    result["attempted_arms"] += 1
                    save()
                    logits = fn().float().masked_fill(~mask, 0)
                    if not torch.isfinite(logits).all().item():
                        raise ValueError("Nonfinite valid logits")
                    if reference is None:
                        reference = logits.clone()
                    torch.testing.assert_close(logits, reference, atol=5e-5, rtol=1e-5)
                    arm["max_absolute_logit_difference"] = (logits - reference).abs().max().item()
                    arm["graph_trials_ms"] = graph_trials(fn)
                    arm["status"] = "qualified"
                    result["qualified_arms"] += 1
                    save()
                del wide, shared, disjoint, fn
                del data, copies, fused, q, weights, tables, copied_tables, reference, logits, mask
                del wide_lengths, flat_lengths, wide_schedule, flat_schedule
                gc.collect()
                torch.cuda.empty_cache()
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
