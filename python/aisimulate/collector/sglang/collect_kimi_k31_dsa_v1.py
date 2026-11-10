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
registry admits SM103 after native B300 TP1/2/4 boundary validation; other
targeted SMs remain unverified.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from collector.framework_manifest import platform_runtime_version
from collector.helper import get_sm_version, log_perf
from collector.op_plans import ModuleCase, plan_cases

MODEL = "moonshotai/Kimi-K3.1-Pro"
MODEL_REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
ARCHITECTURE = "KimiK31ForConditionalGeneration"
# Exact package builds of the same audited official source/image index. Keep
# this startup guard independent of externally supplied runtime manifests.
RUNTIME_VERSIONS = {
    "x86_64": "0.0.0.dev1+g41208373c.d20261104",
    "aarch64": "0.0.0.dev1+ge79de62d2.d20261104",
}
RUNTIME_VERSION = platform_runtime_version({"platform_versions": RUNTIME_VERSIONS})
__compat__ = f"sglang=={RUNTIME_VERSION}"
SOURCE_COMMIT = "afcd7d6b487a9cbe73e25941971b4cc107e6f9ae"
NATIVE_HEADS = 96
HISTORY_CHUNK_TOKENS = 8192


def get_kimi_k31_dsa_context_test_cases():
    return plan_cases("kimi_k31_dsa_context_module", "sglang", get_sm_version())


def get_kimi_k31_dsa_generation_test_cases():
    return plan_cases("kimi_k31_dsa_generation_module", "sglang", get_sm_version())


def validate_case(case: ModuleCase) -> int:
    if case.model_path != MODEL or case.num_heads not in (96, 48, 24, 12):
        raise ValueError(f"unsupported native Kimi K3.1 case: {case}")
    if case.compute_dtype != "bfloat16" or case.gemm_type != "bfloat16":
        raise ValueError("Kimi K3.1's released attention projections are BF16")
    # afcd arg_groups/overrides.py::_kimi_k31_kv_cache_dtype rejects BF16.
    if case.kv_cache_dtype != "fp8":
        raise ValueError(f"native Kimi K3.1 requires FP8 KV; got {case.kv_cache_dtype}")
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
    # does not execute checkpoint-supplied model configuration Python or
    # redistribute its config. The native tokenizer is loaded separately.
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


def executed_kernel_source(backend, pool, phase: str) -> str:
    # HybridLinearAttnBackend delegates full attention to this native member.
    backend = getattr(backend, "full_attn_backend", backend)
    pool = getattr(pool, "full_kv_pool", pool)
    packed = pool.dsa_kv_cache_store_fp8
    cast_on_write = pool.fp8_cast_on_write
    if packed and pool.kv_cache_dim == 656 and not cast_on_write:
        layout = "kv_flashmla_packed"
    elif not packed and pool.kv_cache_dim == 576 and cast_on_write:
        layout = "kv_raw_fp8"
    else:
        raise RuntimeError("native K31 KV pool does not match a supported physical layout")
    # Dense-prefill kernel choice does not identify the KV write format. Keep
    # the actual pool identity even when short prefill bypasses sparse reads.
    if phase == "context" and backend.use_mha:
        leaf = "trtllm_ragged" if backend.device_sm_major >= 10 else "fa3"
        return f"sglang_dsa_dense_mha_{leaf}_{layout}_eager"
    leaf = backend.dsa_prefill_impl if phase == "context" else backend.dsa_decode_impl
    if not isinstance(leaf, str) or not leaf:
        raise RuntimeError("native DSA backend did not report its executed implementation")
    return f"sglang_dsa_indexer_{leaf}_{layout}_eager"


def run_kimi_k31_dsa_worker(*args, device="cuda:0", perf_filename=None):
    case = ModuleCase(*args)
    if validate_case(case) != 1:
        raise RuntimeError(
            "native K31 TP>1 needs exclusive ownership of its GPU group; "
            "use the standalone module on a dedicated allocation, not the per-GPU collector worker"
        )
    _run_native_case(case, device, perf_filename)


def run_kimi_k31_dsa_fullnode(*, perf_filename, limit=None, cases=None):
    """Own the visible TP group and report every selected case to fullnode.py.

    Cases execute sequentially in fresh subprocesses. The common full-node
    orchestrator owns case selection, checkpoint identity and finalization;
    this adapter neither drops cases nor alters native timing boundaries.
    """
    if cases is None:
        getter = (
            get_kimi_k31_dsa_context_test_cases
            if "context" in str(perf_filename)
            else get_kimi_k31_dsa_generation_test_cases
        )
        cases = getter()
    selected = list(cases)[:limit] if limit is not None else list(cases)
    failed = []
    for case in selected:
        try:
            _run_native_case(case, "cuda:0", perf_filename)
        except Exception:
            logging.getLogger(__name__).exception("Native Kimi K3.1 full-node case failed: %s", case)
            failed.append(case)
    return {"succeeded": len(selected) - len(failed), "failed": failed}


def _run_native_case(case, device, perf_filename):
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
            "collector.sglang.collect_kimi_k31_dsa_v1",
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


def _extend_history(bench, reqs, history_length: int, chunk_tokens: int = HISTORY_CHUNK_TOKENS):
    """Build complete decode history through native request continuation.

    afcd one_batch.prepare_extend_inputs_for_correctness_test and
    mem_cache/common.py::checkpoint_kv_cache advance Req.prefix_len to the
    previous extend_end. ScheduleBatch.prepare_for_extend and
    allocation.py::_prefix_kv_indices reuse its existing cache row. Each
    chunk executes the native model, including latent KV and index-K writes.
    This untimed setup avoids one_batch's monolithic-prefill scratch limits;
    it never replaces history with top-k tokens or changes timed decode.
    """
    if not reqs or history_length <= 0 or chunk_tokens < len(reqs):
        raise ValueError("native history requires positive length and a chunk budget covering every request")
    per_request = chunk_tokens // len(reqs)
    prefix = 0
    slots = None
    chunks = 0
    while prefix < history_length:
        end = min(history_length, prefix + per_request)
        for req in reqs:
            req.prefix_len = prefix
            req.extend_end = end
        tokens, _, batch = bench.extend(reqs)
        if (
            batch.seq_lens_cpu.tolist() != [end] * len(reqs)
            or batch.prefix_lens != [prefix] * len(reqs)
            or batch.extend_lens != [end - prefix] * len(reqs)
        ):
            raise RuntimeError("native history chunk differs from the requested continuation")
        current_slots = tuple(req.kv.req_pool_idx for req in reqs)
        if None in current_slots or len(set(current_slots)) != len(reqs):
            raise RuntimeError("native history requests lack distinct allocated cache rows")
        if slots is not None and current_slots != slots:
            raise RuntimeError("native history continuation changed request cache ownership")
        slots = current_slots
        prefix = end
        chunks += 1
    return (
        tokens,
        batch,
        {
            "protocol": "native-req-continuation-v1",
            "chunk_tokens": chunk_tokens,
            "chunks": chunks,
            "history_per_request": history_length,
            "request_slots_preserved": True,
        },
    )


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
    history = None
    try:
        bench.clear()
        if phase == "generation":
            reqs = prepare_synthetic_inputs_for_latency_test(case.batch_size, case.seq_len - 1)
            tokens, batch, history = _extend_history(
                bench, reqs, case.seq_len - 1, max(HISTORY_CHUNK_TOKENS, case.batch_size)
            )
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
    return captured[0], history


def validate_forward_batch(batch, case: ModuleCase, phase: str) -> dict:
    """Verify native scheduler history against the persisted DSA coordinates."""
    expected_tokens = case.batch_size * (case.seq_len if phase == "context" else 1)
    if batch.input_ids.numel() != expected_tokens:
        raise RuntimeError("native forward token count differs from the persisted case")
    seq_lens = batch.seq_lens.tolist()
    expected_length = case.seq_len + case.prefix_len
    if seq_lens != [expected_length] * case.batch_size:
        raise RuntimeError(f"native KV history {seq_lens} differs from persisted length {expected_length}")
    receipt = {"seq_lens": seq_lens}
    if phase == "context":
        prefixes = batch.extend_prefix_lens_cpu
        lengths = batch.extend_seq_lens_cpu
        if prefixes != [case.prefix_len] * case.batch_size or lengths != [case.seq_len] * case.batch_size:
            raise RuntimeError("native cached-prefix/extension lengths differ from the persisted case")
        receipt.update(extend_prefix_lens=prefixes, extend_seq_lens=lengths)
    return receipt


def local_tp_size(tp: int, nnodes: int, node_rank: int) -> int:
    """Match native one_batch.main's contiguous, equal per-node TP groups."""
    if nnodes < 1 or tp % nnodes or not 0 <= node_rank < nnodes:
        raise ValueError("native TP must divide evenly across valid node ranks")
    return tp // nnodes


def _rank_main(local_rank, server_args, port_args, case_values, phase, output, first_gpu):
    import torch
    from sglang.benchmark.one_batch import _init_process_global_configs, load_model
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank

    case = ModuleCase(*case_values)
    # Pinned one_batch.main maps each node's local GPU ordinal separately from
    # its global TP rank. NVL systems expose four GPUs per physical host.
    local_size = local_tp_size(validate_case(case), server_args.nnodes, server_args.node_rank)
    rank = server_args.node_rank * local_size + local_rank
    gpu = first_gpu + local_rank
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
    batch, history = _prepare_forward(bench, case, phase)
    batch_receipt = validate_forward_batch(batch, case, phase)

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
        source = executed_kernel_source(runner.attn_backend, runner.token_to_kv_pool, phase)
        # Every rank executes, while the physical key is one local shard. Use
        # the slowest rank's local module duration; TP reduction stays outside.
        duration = torch.tensor(latency, device=f"cuda:{gpu}")
        torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "native_module": type(attention).__name__,
                        "native_indexer": type(attention.indexer).__name__,
                        "local_heads": attention.num_local_heads,
                        "output_gate": attention.use_output_gate,
                        "case": list(case),
                        "phase": phase,
                        "forward_batch": batch_receipt,
                        "history_preparation": history,
                        "kernel_source": source,
                        "observations": len(starts),
                        "latency_ms": duration.item(),
                    }
                ),
                flush=True,
            )
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
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--dist-init-addr")
    args = parser.parse_args()
    case = ModuleCase(*json.loads(args.case))
    tp = validate_case(case)
    local_size = local_tp_size(tp, args.nnodes, args.node_rank)
    if args.nnodes > 1 and not args.dist_init_addr:
        raise ValueError("multi-node collection requires the same --dist-init-addr on every node")
    _assert_runtime()
    import torch
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import hf_hub_download, snapshot_download
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    first_gpu = int(args.device.rsplit(":", 1)[-1])
    if first_gpu < 0 or first_gpu + local_size > torch.cuda.device_count():
        raise RuntimeError(
            f"FIXME(kernel-limit): does not fit: native TP{tp} across {args.nnodes} nodes "
            f"requires {local_size} available GPUs per node"
        )
    config_path = hf_hub_download(MODEL, "config.json", revision=MODEL_REVISION)
    config = single_layer_config(json.loads(Path(config_path).read_text()))
    tokenizer_path = snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        # Explicit cache-only lookup avoids repository-tree enumeration in
        # offline campaigns with a preverified immutable checkpoint snapshot.
        local_files_only=hf_constants.HF_HUB_OFFLINE,
        # Native one_batch loads this checkpoint's TikTokenTokenizer even
        # for synthetic token IDs. Its only local Python dependency is
        # encoding_k3.py; both were inspected at this immutable revision.
        allow_patterns=["tokenizer_config.json", "tokenization_kimi.py", "encoding_k3.py", "tiktoken.model"],
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
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            dist_init_addr=args.dist_init_addr,
            trust_remote_code=True,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_overlap_schedule=True,
            max_running_requests=max(2 * case.batch_size, 8),
            max_total_tokens=total_tokens,
            chunked_prefill_size=max(16384, case.batch_size * full_length),
            context_length=max(full_length, 4096),
            mem_fraction_static=0.8,
            kv_cache_dtype="fp8_e4m3",
        )
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        # Allocate the rendezvous port once before spawning, as one_batch.main
        # does. Workers never recheck an already-started shared port.
        ports = PortArgs.init_new(server_args)
        torch.multiprocessing.spawn(
            _rank_main,
            args=(server_args, ports, list(case), args.phase, args.output, first_gpu),
            nprocs=local_size,
            join=True,
        )


if __name__ == "__main__":
    main()
