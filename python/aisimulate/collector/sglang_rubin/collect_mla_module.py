# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 NVFP4 DSA modules for the frozen Rubin SGLang image.

Image-specific fork of this project's ``sglang/collect_mla_module.py``.
It preserves the DSA case tuples, perf filenames, and BF16 projection labels.
It does not advertise general SGLang compatibility.

Serving API audit: NVIDIA SGLang 02c5a855aceb968c310e6fbc6632270e26edc84b:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b
Line references below are relative to ``python/sglang`` at that revision.
No framework implementation is vendored here.
"""

from __future__ import annotations

__compat__ = "sglang==0.5.18+nvinternal.rubin.0.8full.66997102"

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from array import array
from contextlib import nullcontext
from importlib.metadata import version as get_version
from itertools import pairwise
from pathlib import Path

from collector.case_generator import get_mla_module_model_specs, get_mla_module_sweep_spec
from collector.sglang.runtime_limits import alloc_prefix_indices, required_kv_alloc_tokens
from collector.sglang_rubin.runtime import REQUIRED_SERVER_ARGS

MODEL_PATH = "nvidia/GLM-5.2-NVFP4"
ARCHITECTURE = "GlmMoeDsaForCausalLM"
_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
_DSA_CEILING_MAX_POSITIONS = (163840, 202752, 1048576)


def _check_model(model_path: str) -> None:
    if model_path != MODEL_PATH:
        raise ValueError(f"Rubin pilot collectors require {MODEL_PATH}; received {model_path}")


def _model_config(model_path: str) -> dict:
    _check_model(model_path)
    return json.loads((_MODEL_CONFIG_DIR / f"{model_path.replace('/', '--')}_config.json").read_text())


def _model_max_position_embeddings(model_path: str) -> int:
    return int(_model_config(model_path)["max_position_embeddings"])


def _model_shares_dsa_index(model_path: str) -> bool:
    return int(_model_config(model_path)["index_topk_freq"]) > 1


def _pilot_model_spec():
    requested = os.environ.get("COLLECTOR_MODEL_PATH")
    if requested:
        _check_model(requested)
    specs = get_mla_module_model_specs(attention_type="dsa", apply_model_filter=False)
    return next(spec for spec in specs if spec.model_path == MODEL_PATH)


def _build_module_test_cases(attn_type: str, mode: str):
    if attn_type != "dsa" or mode not in {"context", "generation"}:
        raise ValueError("The Rubin pilot supports DSA context and generation modules only")
    model = _pilot_model_spec()
    sweep = get_mla_module_sweep_spec("sglang")
    # NVFP4 applies to experts, while the cached checkpoint excludes self_attn
    # from quantization. FP8 KV is independent of these BF16 projections.
    # Leave the backend argument unset: the pinned framework resolves it.
    batches = sweep.context_batch_sizes if mode == "context" else [0]
    return [
        [0, bs, model.native_num_heads // tp, "fp8", "bfloat16", "bfloat16", MODEL_PATH, "dsa", None, tp, None]
        for tp in sweep.module_tp_sizes
        if model.native_num_heads % tp == 0 and model.native_num_heads // tp in sweep.inner_sweep_head_counts
        for bs in batches
    ]


def get_dsa_context_module_test_cases():
    return _build_module_test_cases("dsa", "context")


def get_dsa_generation_module_test_cases():
    return _build_module_test_cases("dsa", "generation")


def get_dsa_context_module_skip_indexer_test_cases():
    if not _model_shares_dsa_index(MODEL_PATH):
        raise ValueError("The pilot checkpoint no longer declares shared DSA indices")
    return get_dsa_context_module_test_cases()


def get_dsa_generation_module_skip_indexer_test_cases():
    if not _model_shares_dsa_index(MODEL_PATH):
        raise ValueError("The pilot checkpoint no longer declares shared DSA indices")
    return get_dsa_generation_module_test_cases()


def _env_shape_filter(shapes, *, is_prefill: bool):
    phase = "CONTEXT" if is_prefill else "GENERATION"
    filters = {}
    for dimension, suffix in ((0, "PREFIX_LENS"), (1, "SEQ_LENS"), (2, "BATCH_SIZES")):
        raw = os.environ.get(f"AIC_DSA_{phase}_{suffix}")
        if raw:
            filters[dimension] = {int(item) for item in raw.split(",")}
    return [shape for shape in shapes if all(shape[index] in values for index, values in filters.items())]


def _dsa_context_derived_shapes(model_path):
    """Reuse the declared SGLang input grid and existing context ceilings."""
    sweep = get_mla_module_sweep_spec("sglang")
    max_position = _model_max_position_embeddings(model_path)
    shapes = []
    for bs in sweep.context_batch_sizes:
        for isl in sweep.context_sequence_lengths:
            if isl <= 1 or bs * isl > sweep.context_max_tokens:
                continue
            if isl >= sweep.context_large_sequence_min and bs > sweep.context_large_sequence_max_batch_size:
                continue
            prefixes = list(sweep.context_prefix_lengths) + [cover - isl for cover in _DSA_CEILING_MAX_POSITIONS]
            shapes.extend((prefix, isl, bs) for prefix in prefixes if prefix >= 0 and prefix + isl <= max_position)
    return _env_shape_filter(list(dict.fromkeys(shapes)), is_prefill=True)


def _dsa_generation_derived_shapes(model_path):
    sweep = get_mla_module_sweep_spec("sglang")
    max_position = _model_max_position_embeddings(model_path)
    shapes = []
    for bs in sweep.generation_batch_sizes:
        for past in sweep.generation_sequence_lengths:
            if bs * past > sweep.generation_max_tokens or past >= max_position:
                continue
            if past >= sweep.generation_large_sequence_min and bs > sweep.generation_large_sequence_max_batch_size:
                continue
            shapes.append((past, 1, bs))
    shapes.extend((cover - 1, 1, 1) for cover in _DSA_CEILING_MAX_POSITIONS if cover <= max_position)
    return _env_shape_filter(list(dict.fromkeys(shapes)), is_prefill=False)


def _validate_pilot_case(model_path, head_num, target_tp_size, kv_cache_dtype, compute_dtype, gemm_type):
    _check_model(model_path)
    if (kv_cache_dtype, compute_dtype, gemm_type) != ("fp8", "bfloat16", "bfloat16"):
        raise ValueError("The NVFP4 pilot requires FP8 KV and BF16 attention projections")
    native_heads = int(_model_config(model_path)["num_attention_heads"])
    if target_tp_size <= 0 or head_num * target_tp_size != native_heads:
        raise ValueError(f"Local heads {head_num} x TP {target_tp_size} must equal {native_heads}")


def _validate_runtime():
    from collector.sglang_rubin.collect_gemm import _require_runtime

    _require_runtime()


def _local_model_config(model_path, head_num):
    """Retain quantization exclusions and native GLM index-sharing metadata."""
    config = _model_config(model_path)
    config.pop("auto_map", None)
    # Same cached-config normalization used by the stock AISimulate collector;
    # this descriptive field does not select SGLang's DSA implementation.
    if "layer_types" in config:
        config["layer_types"] = [
            "compressed_sparse_attention" if value == "deepseek_sparse_attention" else value
            for value in config["layer_types"]
        ]
    # GLM-5.2 declares an offset of 3: layers 0, 1, 2 produce their own
    # indices, and layer 3 is the first shared layer. Preserve that offset;
    # do not reinterpret layer 1 as shared by changing index_topk_freq.
    num_layers = config["indexer_types"].index("shared") + 1
    config["num_hidden_layers"] = num_layers
    config["num_attention_heads"] = head_num
    config["num_key_value_heads"] = head_num
    for name in ("layer_types", "mlp_layer_types", "indexer_types"):
        if isinstance(config.get(name), list):
            config[name] = config[name][:num_layers]
    directory = tempfile.mkdtemp(prefix="aisimulate_rubin_glm52_")
    Path(directory, "config.json").write_text(json.dumps(config))
    shutil.copyfile(
        _MODEL_CONFIG_DIR / f"{model_path.replace('/', '--')}_hf_quant_config.json",
        Path(directory, "hf_quant_config.json"),
    )
    return directory


def _validate_module_shapes(attn, head_num):
    # deepseek_v2.py:1740-1809 and 1889-1921 create these projections from
    # num_attention_heads / attn_tp_size. Verify the simulated local shard.
    expected = {
        "q_b_proj": ("output_size_per_partition", head_num * (attn.qk_nope_head_dim + attn.qk_rope_head_dim)),
        "o_proj": ("input_size_per_partition", head_num * attn.v_head_dim),
    }
    if attn.num_local_heads != head_num:
        raise RuntimeError("DSA local head count differs from the requested TP shard")
    for name, (attribute, size) in expected.items():
        module = getattr(attn, name)
        if getattr(module, attribute) != size:
            raise RuntimeError(f"DSA {name}.{attribute} differs from the requested TP shard")
        # ModelOpt self-attention exclusions must have survived dummy loading.
        if module.weight.dtype.__str__() != "torch.bfloat16":
            raise RuntimeError(f"DSA {name} is not the checkpoint's BF16 projection")


def load_model_runner(
    model_path,
    head_num,
    kv_cache_dtype="fp8",
    attention_backend=None,
    dsa_prefill_backend=None,
    device="cuda:0",
    tp_rank=0,
    gemm_type="bfloat16",
    target_tp_size=1,
    enable_piecewise_cuda_graph=False,
    max_total_tokens=None,
):
    import torch
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

    _validate_pilot_case(model_path, head_num, target_tp_size, kv_cache_dtype, "bfloat16", gemm_type)
    if attention_backend not in {None, "dsa"} or dsa_prefill_backend is not None or enable_piecewise_cuda_graph:
        raise ValueError("Rubin DSA collectors require framework-selected backends and module timing")
    gpu_id = torch.device(device).index or 0
    directory = _local_model_config(model_path, head_num)
    try:
        # All knobs precede resolution. This fork resolves DSA defaults in
        # arg_groups/overrides.py:1718-1809; do not mutate ServerArgs afterward.
        args = ServerArgs(
            model_path=directory,
            dtype="bfloat16",
            device="cuda",
            load_format="dummy",
            tp_size=1,
            trust_remote_code=False,
            disable_radix_cache=True,
            kv_cache_dtype="fp8_e4m3",
            max_total_tokens=max_total_tokens,
            quantization="modelopt_fp4",
            **REQUIRED_SERVER_ARGS,
        )
        if args.disable_prefill_cuda_graph is not True:
            raise RuntimeError("Rubin DSA pilot requires --disable-prefill-cuda-graph in serving and collection")
        _set_envs_and_config(args)
        initialize_moe_config(args)
        initialize_fp8_gemm_config(args)
        initialize_fp4_gemm_config(args)
        # managers/scheduler.py:901-904 initializes all three GEMM selectors
        # before model loading; BF16 auto resolves to CuteDSL on SM10x.
        initialize_bf16_gemm_config(args)
        # benchmark/one_batch.py:299-360 and ParallelState.trivial:29-56.
        runner = ModelRunner(
            model_config=ModelConfig.from_server_args(args),
            mem_fraction_static=args.mem_fraction_static,
            gpu_id=gpu_id,
            ps=ParallelState.trivial(gpu_id=gpu_id),
            nccl_port=20000 + os.getpid() % 30000,
            server_args=args,
        )
        if args.is_startup_weight_load_overlap:
            runner.start_startup_weight_load()
        runner.alloc_memory_pool()
        runner.init_attention_backends()
        if args.is_startup_weight_load_overlap:
            runner.finalize_startup_weight_load()
        if runner.model_config.hf_config.architectures != [ARCHITECTURE]:
            raise RuntimeError("Loaded model architecture differs from the GLM-5.2 DSA pilot")
        for layer in runner.model.model.layers:
            _validate_module_shapes(layer.self_attn, head_num)
        return runner
    finally:
        shutil.rmtree(directory)


def _prepare_batch(runner, prefix, isl, batch_size, *, is_prefill):
    """Use the serving scheduler to construct all attention/indexer metadata."""
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    # For decode, allocate a cached prefix and extend its last token, then use
    # ScheduleBatch.prepare_for_decode for the next token. This avoids a giant
    # artificial prefill allocation. The framework owns all page-table fields.
    cached = prefix if is_prefill else max(0, prefix - 1)
    extend = isl if is_prefill else 1
    prefixes = alloc_prefix_indices(runner, batch_size, cached)
    reqs = []
    for index in range(batch_size):
        ids = array("q", [1]) * (cached + extend)
        req = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=ids,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        # benchmark/one_batch.py:457-464; schedule_batch.py:1265-1269,
        # 2369-2401. Req.fill_len / set_extend_input_len no longer exist.
        req.prefix_indices = prefixes[index]
        req.full_untruncated_fill_ids = ids
        req.set_extend_range(cached, cached + extend)
        req.logprob_start_len = -1
        reqs.append(req)
    cache = ChunkCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.page_size,
        )
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=cache,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    # benchmark/one_batch.py:505-519: the scheduler now defers input H2D.
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    if not is_prefill:
        # benchmark/one_batch.py:525-534; schedule_batch.py:3038-3085.
        batch.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=runner.device)
        for req in batch.reqs:
            req.output_ids.append(0)
        batch.prepare_for_decode()
    forward = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    runner.attn_backend.init_forward_metadata(forward)
    return forward


def _module_pair(runner, skip_indexer):
    # Native layer sharing: deepseek_v2.py:1813-1831; forward_mla.py:129-146,
    # 889-897. A reuse layer has no indexer weights; do not patch a full layer.
    # configs/model_config.py:200-228 applies index_skip_topk_offset, not
    # simply layer_id % index_topk_freq. Read the instantiated flags.
    layers = [layer.self_attn for layer in runner.model.model.layers]
    pairs = [(left, right) for left, right in pairwise(layers) if left.next_skip_topk and right.skip_topk]
    if not pairs:
        raise RuntimeError("No native GLM-5.2 producer/shared layer pair was loaded")
    producer, shared = pairs[0]
    consumer = shared if skip_indexer else producer
    if producer.skip_topk or not producer.next_skip_topk or producer.indexer is None:
        raise RuntimeError("GLM-5.2 producer is not an index-producing layer")
    if skip_indexer and (not consumer.skip_topk or consumer.indexer is not None):
        raise RuntimeError("GLM-5.2 consumer is not a native index-sharing layer")
    return producer, consumer


def _make_module_call(runner, forward, attention, hidden, *, previous_topk=None, use_cuda_graph=False):
    import torch
    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.runner import model_capture_mode
    from sglang.srt.utils import BumpAllocator

    allocator = BumpAllocator(buffer_size=2048, dtype=torch.float32, device=hidden.device)

    def call():
        # ModelRunner._forward_raw:1648-1652; DecodeCudaGraphRunner uses
        # model_capture_mode to select the production dual-stream path.
        with (
            model_capture_mode() if use_cuda_graph else nullcontext(),
            forward_context(ForwardContext(attn_backend=runner.attn_backend)),
        ):
            # utils/common.py:3629: serving owns a new bump allocator per
            # forward. Reuse its zero buffer, resetting only our cursor.
            allocator._pointer = 0
            # communicator.py:733-737 creates inputs per layer invocation.
            # Fresh inputs keep the cached latent projection inside timing;
            # deepseek_v2.py:2207-2224 selects its native GEMM implementation.
            inputs = AttentionInputs(hidden, forward, attention.prepare_qkv_latent)
            get_attn_tp_context().set_attn_inputs(inputs)
            return attention(
                positions=forward.positions,
                hidden_states=hidden,
                forward_batch=forward,
                zero_allocator=allocator,
                prev_topk_indices=previous_topk,
            )

    return call


def _generation_cuda_graph_enabled_for_tokens(runner, num_tokens):
    config = runner.server_args.cuda_graph_config.decode
    if config.backend == "disabled":
        return False
    return num_tokens in config.bs if config.bs else 0 < num_tokens <= (config.max_bs or 256)


def _measure_module(runner, forward, *, skip_indexer, is_prefill, device):
    import torch
    from sglang.srt.model_executor.runner.flashinfer_autotune import (
        flashinfer_autotune_context,
        should_run_flashinfer_autotune,
    )

    from collector.helper import benchmark_with_power

    producer, attention = _module_pair(runner, skip_indexer)
    hidden = torch.randn(
        forward.input_ids.numel(), runner.model.config.hidden_size, dtype=torch.bfloat16, device=device
    )
    graph = not is_prefill and _generation_cuda_graph_enabled_for_tokens(runner, forward.batch_size)
    previous = None
    if skip_indexer:
        output = _make_module_call(runner, forward, producer, hidden, use_cuda_graph=graph)()
        # Dense MHA returns only a Tensor (forward_mha.py:256-266,334-350).
        # deepseek_v2.py:2472-2475 carries no topk for that result; sparse MLA
        # returns the producer's (output, topk) pair (forward_mla.py:889-896).
        if isinstance(output, tuple) and len(output) == 2:
            previous = output[1]
        elif not (runner.attn_backend.use_mha and isinstance(output, torch.Tensor)):
            raise RuntimeError("Index producer did not return the native dense Tensor or (output, topk) contract")
        if previous is None and not runner.attn_backend.use_mha:
            raise RuntimeError("Index producer returned no topk indices for sparse attention")
    call = _make_module_call(runner, forward, attention, hidden, previous_topk=previous, use_cuda_graph=graph)
    # Warm compilation and autotuning before capture. No eager fallback: graph
    # failures must reach the executor's failure record rather than change data.
    # runner/flashinfer_autotune.py:50-132,171-215 chooses whether this
    # serving stack tunes FlashInfer and applies its skip policy. Tune only
    # the measured module; full-model startup tuning is not required here.
    tuning = (
        flashinfer_autotune_context(runner, run_lm_head=False)
        if should_run_flashinfer_autotune(runner)
        else nullcontext()
    )
    with tuning:
        for _ in range(8):
            call()
    torch.cuda.synchronize(device)
    with benchmark_with_power(
        device=device, kernel_func=call, num_warmups=2, num_runs=10, repeat_n=1, use_cuda_graph=graph
    ) as results:
        pass
    backend = runner.attn_backend.dsa_prefill_impl if is_prefill else runner.attn_backend.dsa_decode_impl
    indexer = "skip_indexer" if skip_indexer else "indexer"
    density = "dense" if runner.attn_backend.use_mha else "sparse"
    method = "cuda_graph" if graph else "eager"
    return results, f"sglang_dsa_{indexer}_{backend}_{density}_{method}"


def run_mla_module(
    attn_type,
    head_num,
    model_path,
    kv_cache_dtype,
    compute_dtype,
    gemm_type,
    is_prefill,
    gpu_id,
    output_path=None,
    attention_backend=None,
    batch_size_filter=None,
    target_tp_size=1,
    dsa_prefill_backend=None,
    skip_indexer=False,
):
    import torch

    from collector.helper import log_perf

    _validate_runtime()
    _validate_pilot_case(model_path, head_num, target_tp_size, kv_cache_dtype, compute_dtype, gemm_type)
    if attn_type != "dsa":
        raise ValueError("Only GLM-5.2 DSA is supported by the Rubin pilot")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    shapes = (_dsa_context_derived_shapes if is_prefill else _dsa_generation_derived_shapes)(model_path)
    if batch_size_filter:
        shapes = [shape for shape in shapes if shape[2] == batch_size_filter]
    if not shapes:
        raise RuntimeError("Queued DSA case resolved no shapes")
    # The same optional smoke sampling as the existing sparse module collector.
    if "--smoke" in sys.argv and len(shapes) > 8:
        shapes = [shapes[index] for index in sorted({round(i * (len(shapes) - 1) / 7) for i in range(8)})]
    max_tokens = max(
        required_kv_alloc_tokens(
            bs, isl if is_prefill else prefix, prefix if is_prefill else 0, 64, is_prefill=is_prefill
        )
        for prefix, isl, bs in shapes
    )
    runner = load_model_runner(
        model_path,
        head_num,
        kv_cache_dtype,
        attention_backend,
        dsa_prefill_backend,
        device=device,
        gemm_type=gemm_type,
        target_tp_size=target_tp_size,
        max_total_tokens=max_tokens + 1024,
    )
    phase = "context" if is_prefill else "generation"
    op_name = f"dsa_{phase}_module" + ("_skip_indexer" if skip_indexer else "")
    filename = str(Path(output_path or os.getcwd()) / f"dsa_{phase}_module_perf.txt")
    with torch.inference_mode():
        for prefix, isl, bs in shapes:
            try:
                forward = _prepare_batch(runner, prefix, isl, bs, is_prefill=is_prefill)
                results, source = _measure_module(
                    runner, forward, skip_indexer=skip_indexer, is_prefill=is_prefill, device=device
                )
                if not log_perf(
                    item_list=[
                        {
                            "model": model_path,
                            "architecture": ARCHITECTURE,
                            "mla_dtype": compute_dtype,
                            "kv_cache_dtype": kv_cache_dtype,
                            "gemm_type": gemm_type,
                            "num_heads": head_num,
                            "batch_size": bs,
                            "isl": isl,
                            "tp_size": target_tp_size,
                            "step": prefix,
                            "latency": f"{results['latency_ms']:.4f}",
                        }
                    ],
                    framework="SGLang",
                    version=get_version("sglang"),
                    device_name=torch.cuda.get_device_name(device),
                    op_name=op_name,
                    kernel_source=source,
                    perf_filename=filename,
                    power_stats=results["power_stats"],
                ):
                    raise RuntimeError(f"Failed to persist {op_name} row to {filename}")
            except Exception as error:
                raise RuntimeError(f"{op_name} failed at batch={bs}, isl={isl}, past_kv={prefix}: {error}") from error


def run_mla_module_worker(
    seq_len,
    batch_size,
    num_heads,
    kv_cache_dtype,
    compute_dtype,
    gemm_type,
    model_path,
    attn_type,
    attention_backend=None,
    target_tp_size=1,
    dsa_prefill_backend=None,
    *,
    perf_filename,
    device="cuda:0",
):
    """Preserve the stock worker tuple; isolate framework state in a subprocess."""
    is_prefill = "context" in Path(perf_filename).name
    arguments = dict(
        attn_type=attn_type,
        head_num=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=0,
        output_path=str(Path(perf_filename).resolve().parent),
        attention_backend=attention_backend,
        batch_size_filter=batch_size if is_prefill and batch_size > 0 else None,
        target_tp_size=target_tp_size,
        dsa_prefill_backend=dsa_prefill_backend,
        skip_indexer="skip_indexer" in Path(perf_filename).name,
    )
    _run_subprocess("collector.sglang_rubin.collect_mla_module", arguments, device)


def _run_subprocess(module_name, arguments, device):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
    # Static module entrypoint + JSON argv: model/output strings cannot become code.
    command = [sys.executable, "-m", module_name, "--payload", json.dumps(arguments)]
    if "--smoke" in sys.argv:
        command.append("--smoke")
    timeout = os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC")
    subprocess.run(command, env=env, check=True, timeout=int(timeout) if timeout else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_mla_module(**json.loads(args.payload))


if __name__ == "__main__":
    main()
