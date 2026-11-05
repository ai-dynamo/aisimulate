# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Native K3.1 Gated DSA, using the pinned serving model and batch builders.

Serving reference: SGLang's Apache-2.0 ``benchmark/one_batch.py`` and
``srt/models/kimi_k3.py`` in the official day-zero image, source commit
afcd7d6b487a9cbe73e25941971b4cc107e6f9ae. Both files and configs/kimi_linear.py
are byte-identical to the public source at
https://github.com/sgl-project/sglang/blob/e086da8aa1b675a61e08cc09c9aa940a7f66f490/python/sglang/srt/models/kimi_k3.py
Modified lifecycle: one native sparse layer, timed module hooks, physical-key
logging, immutable runtime/config checks, and a clean subprocess per case.

Boundary: native self_attn (Q/K/V and indexer projections, sparse attention,
g_proj, sigmoid gate and o_proj). The decoder's attention normalization and
TP reduction live outside this boundary. No DeepSeek attention is substituted.

Deviation: this initial lane explicitly configures eager serving. Its rows
carry an ``_eager`` kernel_source suffix. Native graph replay/overlap can change
latency in either direction; qualification must compare graph serving before
removing this note or claiming default-serving prediction accuracy. The
registry remains unverified until a real GPU campaign qualifies this boundary.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from collector.helper import get_sm_version, log_perf
from collector.op_plans import ModuleCase, plan_cases

__compat__ = "sglang==0.0.0.dev1+g41208373c.d20261104"

MODEL = "moonshotai/Kimi-K3.1-Pro"
MODEL_REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
ARCHITECTURE = "KimiK31ForConditionalGeneration"
RUNTIME_VERSION = "0.0.0.dev1+g41208373c.d20261104"
SOURCE_COMMIT = "afcd7d6b487a9cbe73e25941971b4cc107e6f9ae"
NATIVE_HEADS = 96


def get_kimi_k31_dsa_context_test_cases():
    return plan_cases("kimi_k31_dsa_context_module", "sglang", get_sm_version())


def get_kimi_k31_dsa_generation_test_cases():
    return plan_cases("kimi_k31_dsa_generation_module", "sglang", get_sm_version())


def validate_case(case: ModuleCase) -> int:
    if case.model_path != MODEL or case.num_heads not in (96, 48, 24, 12):
        raise ValueError(f"unsupported native Kimi K3.1 case: {case}")
    if case.compute_dtype != "bfloat16" or case.gemm_type != "bfloat16":
        raise ValueError("Kimi K3.1's released attention projections are BF16")
    if case.kv_cache_dtype not in ("bfloat16", "fp8"):
        raise ValueError(f"unsupported KV precision {case.kv_cache_dtype}")
    if min(case.seq_len, case.batch_size) <= 0 or case.prefix_len < 0:
        raise ValueError(f"invalid workload: {case}")
    if case.seq_len + case.prefix_len > 1048576:
        raise ValueError("Kimi K3.1 workload exceeds the native context limit")
    return NATIVE_HEADS // case.num_heads


def single_layer_config(config: dict) -> dict:
    """Keep native attention geometry while omitting untimed model layers.

    KimiLinearConfig._init_dsa and KimiK3DecoderLayer select each attention
    from the matching one-based dsa/full/KDA lists. Global heads stay 96;
    SGLang's real TP group shards projections, caches and dispatch metadata.
    """
    result = copy.deepcopy(config)
    text = result.get("text_config", {})
    expected = {
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_output_gate": True,
    }
    if result.get("architectures") != [ARCHITECTURE]:
        raise ValueError("upstream Kimi K3.1 architecture differs from the collector contract")
    for name, value in expected.items():
        if text.get(name) != value:
            raise ValueError(f"upstream Kimi K3.1 {name}={text.get(name)!r}, expected {value!r}")
    dsa = text.get("dsa_config", {})
    for name, value in {
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "index_topk_freq": 1,
        "index_skip_topk_offset": 1,
    }.items():
        if dsa.get(name) != value:
            raise ValueError(f"upstream Kimi K3.1 DSA {name} differs from the collector contract")
    text["num_hidden_layers"] = 1
    text["num_nextn_predict_layers"] = 0
    text["linear_attn_config"]["full_attn_layers"] = [1]
    text["linear_attn_config"]["kda_layers"] = []
    dsa["dsa_layers"] = [1]
    # Native framework config classes resolve this architecture. The collector
    # does not execute checkpoint-supplied Python or redistribute its config.
    result.pop("auto_map", None)
    return result


def perf_item(case: ModuleCase, phase: str, latency: float) -> dict:
    """Existing DSA consumer key: generation uses isl + step as total KV length."""
    return {
        "model": case.model_path,
        "architecture": ARCHITECTURE,
        "mla_dtype": case.compute_dtype,
        "kv_cache_dtype": case.kv_cache_dtype,
        "gemm_type": case.gemm_type,
        "num_heads": case.num_heads,
        "batch_size": case.batch_size,
        "isl": case.seq_len if phase == "context" else 1,
        "tp_size": validate_case(case),
        "step": case.prefix_len if phase == "context" else case.seq_len - 1,
        "latency": f"{latency:.4f}",
    }


def executed_kernel_source(backend, phase: str) -> str:
    # HybridLinearAttnBackend delegates full attention to this native member.
    backend = getattr(backend, "full_attn_backend", backend)
    if phase == "context" and backend.use_mha:
        leaf = "trtllm_ragged" if backend.device_sm_major >= 10 else "fa3"
        return f"sglang_dsa_dense_mha_{leaf}_eager"
    leaf = backend.dsa_prefill_impl if phase == "context" else backend.dsa_decode_impl
    if not isinstance(leaf, str) or not leaf:
        raise RuntimeError("native DSA backend did not report its executed implementation")
    return f"sglang_dsa_indexer_{leaf}_eager"


def run_kimi_k31_dsa_worker(*args, device="cuda:0", perf_filename=None):
    case = ModuleCase(*args)
    validate_case(case)
    if perf_filename is None:
        raise ValueError("native Kimi K3.1 collection requires an output filename")
    phase = "context" if "context" in str(perf_filename) else "generation"
    # The generic executor's worker must not create children using fork after
    # CUDA initialization. A clean subprocess owns SGLang's spawned TP ranks.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.sglang.collect_kimi_k31_dsa",
            "--case",
            json.dumps(list(case)),
            "--phase",
            phase,
            "--output",
            str(Path(perf_filename).resolve()),
            "--device",
            str(device),
        ],
        check=True,
    )


def _assert_runtime() -> None:
    version = importlib.metadata.version("sglang")
    if version != RUNTIME_VERSION:
        raise RuntimeError(f"native Kimi K3.1 requires exactly {RUNTIME_VERSION}; found {version}")
    import sglang

    source_root = Path(sglang.__file__).resolve().parents[2]
    marker = source_root / ".k31-source-commit"
    if not marker.is_file() or marker.read_text().strip() != SOURCE_COMMIT:
        raise RuntimeError("native Kimi K3.1 source identity differs from the pinned official image")


def _prepare_forward(bench, case: ModuleCase, phase: str):
    """Let native one_batch build every tensor, KV slot and attention metadata.

    Prefix ownership follows one_batch's correctness-test Req protocol; a real
    native prefix forward creates both latent and index-K history. Decode
    begins at total length minus one, then native prepare_for_decode adds one.
    """
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    runner = bench.torch_runner
    captured = []
    original = runner.forward

    def capture(batch, *args, **kwargs):
        captured[:] = [batch]
        return original(batch, *args, **kwargs)

    runner.forward = capture
    try:
        bench.clear()
        if phase == "generation":
            reqs = prepare_synthetic_inputs_for_latency_test(case.batch_size, case.seq_len - 1)
            tokens, _, batch = bench.extend(reqs)
            bench.decode(tokens, batch)
        else:
            reqs = prepare_synthetic_inputs_for_latency_test(case.batch_size, case.prefix_len + case.seq_len)
            if case.prefix_len:
                for req in reqs:
                    req.extend_end = case.prefix_len
                bench.extend(reqs)
                for req in reqs:
                    req.prefix_len = case.prefix_len
                    req.extend_end = case.prefix_len + case.seq_len
            bench.extend(reqs)
    finally:
        runner.forward = original
    if len(captured) != 1:
        raise RuntimeError("native batch builder did not produce one forward")
    return captured[0]


def _rank_main(rank, server_args, port_args, case_values, phase, output, first_gpu):
    import torch
    from sglang.benchmark.one_batch import _init_process_global_configs, load_model
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank

    case = ModuleCase(*case_values)
    gpu = first_gpu + rank
    torch.cuda.set_device(gpu)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=rank, pp_rank=0),
            gpu_id=gpu,
        ),
    )
    _init_process_global_configs()
    bench, _tokenizer = load_model(server_args, port_args, gpu, rank)
    runner = bench.torch_runner
    modules = [module for name, module in runner.model.named_modules() if name.endswith("layers.0.self_attn")]
    if len(modules) != 1:
        raise RuntimeError("native Kimi K3.1 model did not build one attention observation boundary")
    attention = modules[0]
    if type(attention).__name__ != "KimiK3MLAAttention" or attention.indexer is None:
        raise RuntimeError("native model builder did not select Kimi Gated DSA")
    if not attention.use_output_gate or attention.num_local_heads != case.num_heads:
        raise RuntimeError("native gate or local TP head geometry differs from the case")
    batch = _prepare_forward(bench, case, phase)
    expected_tokens = case.batch_size * (case.seq_len if phase == "context" else 1)
    if batch.input_ids.numel() != expected_tokens:
        raise RuntimeError("native forward token count differs from the persisted case")

    # KimiK3DecoderLayer._run_self_attn calls the Module normally. Events
    # observe that boundary without replacing its implementation or inputs.
    starts, ends = [], []

    def before(_module, _args):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        starts.append(event)

    def after(_module, _args, _output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        ends.append(event)

    handles = [attention.register_forward_pre_hook(before), attention.register_forward_hook(after)]
    try:
        with torch.inference_mode():
            for _ in range(3):
                runner.forward(batch)
            torch.cuda.synchronize()
            starts.clear()
            ends.clear()
            for _ in range(10):
                runner.forward(batch)
            torch.cuda.synchronize()
        if len(starts) != 10 or len(ends) != 10:
            raise RuntimeError("native serving bypassed the declared attention measurement boundary")
        latency = statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
        source = executed_kernel_source(runner.attn_backend, phase)
        # Every rank executes, while the physical key is one local shard. Use
        # the slowest rank's local module duration; TP reduction stays outside.
        duration = torch.tensor(latency, device=f"cuda:{gpu}")
        torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX)
        if rank == 0:
            log_perf(
                [perf_item(case, phase, duration.item())],
                "SGLang",
                RUNTIME_VERSION,
                torch.cuda.get_device_name(gpu),
                f"dsa_{phase}_module",
                source,
                output,
            )
    finally:
        for handle in handles:
            handle.remove()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--phase", choices=("context", "generation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    case = ModuleCase(*json.loads(args.case))
    tp = validate_case(case)
    _assert_runtime()
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    first_gpu = int(args.device.rsplit(":", 1)[-1])
    if first_gpu + tp > torch.cuda.device_count():
        raise RuntimeError(f"FIXME(kernel-limit): does not fit: native TP{tp} requires {tp} available GPUs")
    config_path = hf_hub_download(MODEL, "config.json", revision=MODEL_REVISION)
    config = single_layer_config(json.loads(Path(config_path).read_text()))
    tokenizer_path = snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        allow_patterns=["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"],
    )
    with tempfile.TemporaryDirectory(prefix="kimi-k31-native-") as directory:
        Path(directory, "config.json").write_text(json.dumps(config))
        full_length = case.seq_len + case.prefix_len
        # Native allocation includes the complete history, never top-k alone.
        total_tokens = case.batch_size * (full_length + 64)
        server_args = ServerArgs(
            model_path=directory,
            tokenizer_path=tokenizer_path,
            load_format="dummy",
            dtype="bfloat16",
            tp_size=tp,
            trust_remote_code=False,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_overlap_schedule=True,
            max_running_requests=max(2 * case.batch_size, 8),
            max_total_tokens=total_tokens,
            chunked_prefill_size=max(16384, case.batch_size * full_length),
            context_length=max(full_length, 4096),
            mem_fraction_static=0.8,
            kv_cache_dtype="fp8_e4m3" if case.kv_cache_dtype == "fp8" else "bf16",
        )
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        # Allocate the rendezvous port once before spawning, as one_batch.main
        # does. Workers never recheck an already-started shared port.
        ports = PortArgs.init_new(server_args)
        torch.multiprocessing.spawn(
            _rank_main,
            args=(server_args, ports, list(case), args.phase, args.output, first_gpu),
            nprocs=tp,
            join=True,
        )


if __name__ == "__main__":
    main()
