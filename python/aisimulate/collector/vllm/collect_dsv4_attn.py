# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 module-level attention collector for vLLM.

Full CSA/HCA attention constructs one ``DeepseekV4Attention`` layer, binds the
DSV4 main/SWA/indexer/compressor metadata and KV caches, then benchmarks the
full attention wrapper forward.  Sparse HCA is also isolated from that module.
Paged MQA logits is collected as a kernel-level benchmark with directly
constructed vLLM DeepGEMM inputs, matching the sparse correction model.

The attention path uses dummy weights. With dummy FP8 block weights, vLLM's
DeepGEMM path may require real checkpoint layouts/scales to be representative;
the collector does not override vLLM's backend selection at import time.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from collector.case_generator import (
    _DSV4_DEFAULT_MODELS,
    _DSV4_MODULE_TP_SIZES,
    DSV4_ATTN_KINDS,
    get_dsv4_sweep,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.registry_types import PerfFile
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    enable_engine_fused_ops,
    setup_distributed,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

# B200 0.25.0 qualification (installed vLLM dd10e03f9), job 1968046:
# CSA/HCA context/generation: 32/32; auxiliary sparse kernels: 4/4. The native framework
# builders/selectors remain authoritative; no kernel fallback is introduced.
# The campaign manifest still selects one exact release per run.
# Versions are upgraded IN PLACE: this file always
# targets the manifest pin; prior-version code is `git log -- <this file>`
# away and prior-version DATA stays permanent under its version key. Serving
# citations are @v0.32.0 unless stated otherwise.
__compat__ = "vllm==0.32.0"


DEFAULT_MODEL = _DSV4_DEFAULT_MODELS[0]
ARCHITECTURE = "DeepseekV4ForCausalLM"
ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}
# paged_mqa_logits (the CSA indexer scoring GEMM) is the only standalone sparse
# sub-kernel; the hca_attn FMLA sidecar was removed (retired SDK
# placeholder).
SPARSE_KERNEL_TO_ATTN_KIND = {"paged_mqa_logits": "csa"}
SPARSE_KERNEL_TO_OP_NAME = {"paged_mqa_logits": "dsv4_paged_mqa_logits_module"}
SPARSE_KERNEL_TO_PERF_FILE = {"paged_mqa_logits": PerfFile.DSV4_PAGED_MQA_LOGITS_MODULE}
MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
SUPPORTED_GEMM_TYPES = {"fp8_block"}


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    if filename is None:
        raise ValueError("filename is required")
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _read_model_config(model_id: str) -> dict:
    if os.path.isdir(model_id):
        with open(os.path.join(model_id, "config.json"), encoding="utf-8") as f:
            return json.load(f)

    config_file = MODEL_CONFIGS_DIR / f"{model_id.replace('/', '--')}_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: {config_file}")
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def _patched_config_dir(model_id: str, *, compress_ratio: int):
    config = dict(_read_model_config(model_id))
    config.pop("auto_map", None)

    # The converted sgl-project configs use ``deepseek_ref``. vLLM has a native
    # DeepseekV4Config, so use the production model type rather than routing the
    # layer through a DeepSeek-V3 config compatibility hack.
    config["model_type"] = "deepseek_v4"
    config["architectures"] = [ARCHITECTURE]
    config["num_hidden_layers"] = 1
    config["num_key_value_heads"] = 1
    config["compress_ratios"] = [compress_ratio]
    # The converted SGLang FP8 artifacts omit this field, while their canonical
    # DeepSeek-V4 configs and vLLM's native attention module require it.
    config["rms_norm_eps"] = 1e-6

    with tempfile.TemporaryDirectory(prefix=f"aic_vllm_dsv4_{compress_ratio}_{os.getpid()}_") as tmp_dir:
        with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        yield tmp_dir


def _init_cuda(device: str) -> None:
    setup_distributed(device)
    torch.cuda.set_device(device)
    enable_engine_fused_ops()
    init_workspace_manager(torch.device(device))


@contextmanager
def _tp_simulation(tp_size: int):
    if tp_size == 1:
        yield
        return

    import vllm.model_executor.layers.linear as linear_mod
    import vllm.models.deepseek_v4.attention as dsv4_attn_mod

    def world_size() -> int:
        return tp_size

    def rank() -> int:
        return 0

    def identity_collective(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    patches = [
        (linear_mod, "get_tensor_model_parallel_world_size", world_size),
        (linear_mod, "get_tensor_model_parallel_rank", rank),
        (linear_mod, "tensor_model_parallel_all_reduce", identity_collective),
        (linear_mod, "tensor_model_parallel_all_gather", identity_collective),
        (dsv4_attn_mod, "get_tensor_model_parallel_world_size", world_size),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, replacement in patches:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def _init_dummy_module_tensors(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.fill_(1.0)
            else:
                tensor.fill_(0.01)


def _process_quantized_weights(module: torch.nn.Module, vllm_config) -> None:
    with set_current_vllm_config(vllm_config):
        for _, child in module.named_modules():
            quant_method = getattr(child, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(child)


@contextmanager
def _create_dsv4_attention_module(
    *,
    model_path: str,
    attn_kind: str,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
):
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    with _patched_config_dir(model_path, compress_ratio=compress_ratio) as local_model:
        # Serving's default max_model_len is the checkpoint's max_position_embeddings
        # (config/model.py:2592-2602,2621-2634: YaRN keeps it unscaled), 1M for the
        # DeepSeek-V4 checkpoints, and DSV4 plans its prefill with it: the FlashMLA
        # prefill splits a step into chunks whose gather workspace area is
        # PREFILL_CHUNK_SIZE * (max_model_len / compress_ratio + window + tokens)
        # (v1/attention/backends/mla/sparse_swa.py:250-329, models/deepseek_v4/attention.py:155,
        # nvidia/flashmla.py:310), and the indexer's prefill buffer, its chunking and its
        # decode-logits width follow it too (models/deepseek_v4/attention.py:995-1002,
        # config/vllm.py:1790-1798). A max_model_len of the point's own length splits
        # small-isl, large-batch prefills into chunks serving never runs, so build at
        # serving's default.
        max_position = int(_read_model_config(model_path).get("max_position_embeddings") or 0)
        max_model_len = max(seq_len, max_position or 4096)
        if query_len is None:
            query_len = seq_len if is_context else 1
        query_tokens = batch_size * query_len
        max_num_batched_tokens = max(query_tokens, 2048)
        block_size = 256
        cache_blocks = _cache_blocks(batch_size, seq_len)

        vllm_config = create_vllm_config(
            model_name=local_model,
            tensor_parallel_size=tp_size,
            # The collector simulates TP shard shapes in one process instead of
            # launching one vLLM executor rank per shard.
            distributed_executor_backend="mp",
            max_model_len=max_model_len,
            block_size=block_size,
            num_gpu_blocks=cache_blocks,
            max_num_seqs=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            use_fp8_kv_cache=True,
            trust_remote_code=True,
        )
        hf_config = vllm_config.model_config.hf_config
        hf_config.num_hidden_layers = 1
        hf_config.compress_ratios = [compress_ratio]
        hf_config.num_key_value_heads = 1

        attn_cls = _select_dsv4_attn_cls(vllm_config)
        capability = current_platform.get_device_capability()
        if capability is not None and capability.to_int() == 89:
            raise RuntimeError(
                "vLLM DeepSeek-V4 attention is unsupported on SM89 (collector/skips.py skips it below SM90): "
                "the production selector falls back to FlashMLA, whose DSV4 "
                "backend supports SM90 and SM10x only."
            )
        if capability is not None and not attn_cls.backend_cls.supports_compute_capability(capability):
            raise RuntimeError(
                f"{attn_cls.__name__} backend {attn_cls.backend_cls.get_name()} "
                f"does not support SM{capability.to_int()}"
            )

        # Keep production selection for SM100/SM103 and SM120 (_select_dsv4_attn_cls,
        # models/deepseek_v4/nvidia/model.py:1013-1044: FlashInfer on SM12x, FlashMLA
        # elsewhere). B200 was qualified (header); SM103 and SM120 are source-derived.
        # DeepSeek rotary construction creates CPU tensors internally; setting the
        # default device only for module construction keeps those tensors aligned.
        torch.set_default_device(device)
        try:
            with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
                topk_indices_buffer = torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    hf_config.index_topk,
                    dtype=torch.int32,
                    device=device,
                )
                aux_streams = [torch.cuda.Stream() for _ in range(3)]
                attn_module = attn_cls(
                    vllm_config,
                    prefix="model.layers.0.attn",
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=aux_streams,
                )
        finally:
            torch.set_default_device("cpu")

        if any(p.is_meta for p in attn_module.parameters()):
            attn_module = attn_module.to_empty(device=torch.device(device))
        else:
            attn_module = attn_module.to(device)
        attn_module.eval()
        attn_module.requires_grad_(False)
        _init_dummy_module_tensors(attn_module)
        _process_quantized_weights(attn_module, vllm_config)
        yield attn_module, vllm_config


def _cache_blocks_for_block_size(batch_size: int, seq_len: int, block_size: int) -> int:
    logical_blocks = batch_size * max(1, math.ceil(seq_len / block_size))
    return max(256, logical_blocks + 64)


def _cache_blocks(batch_size: int, seq_len: int) -> int:
    return _cache_blocks_for_block_size(batch_size, seq_len, 64)


def _make_common_metadata(
    *,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    device: str,
    query_len: int | None = None,
):
    if query_len is None:
        query_len = seq_len if is_context else 1
    batch_spec = BatchSpec(
        seq_lens=[seq_len] * batch_size,
        query_lens=[query_len] * batch_size,
    )
    common = create_common_attn_metadata(
        batch_spec,
        block_size=64,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(common, "_seq_lens_cpu", None) is not None:
        common.seq_lens_cpu_upper_bound = common._seq_lens_cpu
    common.positions = _positions(batch_size, seq_len, is_context, query_len=query_len, device=device)

    context_len = seq_len - query_len
    slot_values: list[int] = []
    for req_idx in range(batch_size):
        for q_idx in range(query_len):
            pos = context_len + q_idx
            block_id = int(common.block_table_tensor[req_idx, pos // 64].item())
            slot_values.append(block_id * 64 + pos % 64)
    common.slot_mapping.copy_(torch.tensor(slot_values, dtype=torch.int64, device=device))
    return common


def _remap_common_metadata(common, *, block_size: int, device: str):
    seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common.seq_lens_cpu_upper_bound
    seq_lens = [int(x) for x in seq_lens_cpu.tolist()]
    query_start_loc_cpu = common.query_start_loc_cpu
    query_lens = [int((query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]).item()) for i in range(len(seq_lens))]
    remapped = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=block_size,
        device=torch.device(device),
        arange_block_indices=True,
    )
    if getattr(remapped, "_seq_lens_cpu", None) is not None:
        remapped.seq_lens_cpu_upper_bound = remapped._seq_lens_cpu
    remapped.positions = common.positions
    return remapped


def _allocate_attention_kv_cache(backend, spec, num_blocks: int, cache_dtype: str, *, device: str) -> torch.Tensor:
    # The shape contract lives in the generic spec pipeline. Serving
    # allocates flat bytes and views each layer as the 4D logical [B,H,N,C]
    # page (compute_layer_kv_cache_shape_bytes, v1/kv_cache_interface.py:319-334;
    # per-layer element view = bytes.view(spec.dtype), create_kv_cache_views :377).
    # The dsv4 layout facts (584B fp8_ds_mla state content, 576B alignment,
    # tokens_per_state=compress_ratio) are spec fields
    # (models/deepseek_v4/attention.py:865-878), so the helper output IS the
    # serving shape. Page padding: mirror create_kv_cache_views' block stride
    # (page_size_bytes per block) via empty_strided when the spec pads pages.
    from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes

    shape_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
    dtype_size = torch.empty((), dtype=spec.dtype).element_size()
    shape = (*shape_bytes[:-1], shape_bytes[-1] // dtype_size)
    if spec.page_size_padded is None:
        return torch.zeros(shape, dtype=spec.dtype, device=device)

    strides = list(torch.empty(shape).stride())
    strides[0] = spec.page_size_bytes // dtype_size
    return torch.empty_strided(shape, tuple(strides), dtype=spec.dtype, device=device).zero_()


def _make_builder(backend, spec, prefix, vllm_config, common, *, device: str):
    """Construct the layer's metadata builder exactly as serving does.

    Builders may demand kw-only block_table_width (DeepseekV32IndexerMetadataBuilder,
    mla/indexer.py:988,1000); serving computes it via get_block_table_width and
    ALLOCATES the block table at that width (v1/worker/utils.py:308-316 +
    v1/worker/block_table.py:29-41), so builders assume
    table.shape[1] == width — the family-100 decode path copies the table into
    a width-sized expanded buffer (indexer.py:1091-1095,1197-1248) and crashes on
    narrower tables (found on B300 in the mla 029 lane; same contract here).
    Zero-pad our ceil(seq/block)-column table to the width. All @v0.32.0.
    """
    builder_cls = backend.get_builder_cls()
    kwargs = {}
    if getattr(builder_cls, "requires_block_table_width", False):
        from vllm.v1.worker.block_table import get_block_table_width

        max_num_blocks = spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len)
        width = get_block_table_width(max_num_blocks, spec.block_size)
        kwargs["block_table_width"] = width
        _bt = common.block_table_tensor
        if _bt.shape[1] < width:
            _pad = torch.zeros((_bt.shape[0], width - _bt.shape[1]), dtype=_bt.dtype, device=_bt.device)
            common.block_table_tensor = torch.cat([_bt, _pad], dim=1)
    return builder_cls(spec, [prefix], vllm_config, torch.device(device), **kwargs)


def _build_metadata_and_bind_caches(attn_module: DeepseekV4Attention, vllm_config, common, *, device: str):
    cache_blocks = _cache_blocks(int(common.num_reqs), int(common.max_seq_len))
    metadata = {}
    static_ctx = vllm_config.compilation_config.static_forward_context

    cache_layers = [attn_module, attn_module.swa_cache_layer]
    if attn_module.indexer is not None:
        cache_layers.append(attn_module.indexer.k_cache)
    for layer in cache_layers:
        registered_layer = static_ctx[layer.prefix]
        spec = registered_layer.get_kv_cache_spec(vllm_config)
        if spec is None:
            continue
        backend = registered_layer.get_attn_backend()
        metadata[layer.prefix] = _make_builder(backend, spec, layer.prefix, vllm_config, common, device=device).build(
            0, common
        )
        cache_dtype = getattr(spec, "cache_dtype_str", None) or "auto"
        # serving binds through each layer's bind_kv_cache
        # (v1/worker/utils.py:811-880) — layer overrides may reshape the
        # raw view (e.g. squeeze the H=1 dim), so never assign .kv_cache raw
        registered_layer.bind_kv_cache(
            _allocate_attention_kv_cache(
                backend,
                spec,
                cache_blocks,
                cache_dtype,
                device=device,
            )
        )

    compressors = [attn_module.compressor]
    if attn_module.indexer is not None:
        compressors.append(attn_module.indexer.compressor)
    for compressor in filter(None, compressors):
        state_cache = static_ctx[compressor.state_cache.prefix]
        spec = state_cache.get_kv_cache_spec(vllm_config)
        backend = state_cache.get_attn_backend()
        compressor_common = _remap_common_metadata(common, block_size=spec.block_size, device=device)
        metadata[state_cache.prefix] = _make_builder(
            backend, spec, state_cache.prefix, vllm_config, compressor_common, device=device
        ).build(0, compressor_common)
        state_cache_blocks = _cache_blocks_for_block_size(
            int(common.num_reqs),
            int(common.max_seq_len),
            spec.block_size,
        )
        state_cache.bind_kv_cache(
            _allocate_attention_kv_cache(
                backend,
                spec,
                state_cache_blocks,
                getattr(spec, "cache_dtype_str", None) or "auto",
                device=device,
            )
        )

    return metadata


def _positions(
    batch_size: int,
    seq_len: int,
    is_context: bool,
    *,
    device: str,
    query_len: int | None = None,
) -> torch.Tensor:
    if query_len is None:
        query_len = seq_len if is_context else 1
    start_pos = seq_len - query_len
    return (start_pos + torch.arange(query_len, device=device, dtype=torch.long)).repeat(batch_size)


def _bench_attention_shape(
    *,
    model_path: str,
    attn_kind: str,
    mode: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    tp_size: int,
    gemm_type: str,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    is_context = mode == "context"
    # Module-generation ``seq_len`` is the decode step / past KV length used
    # as the perf DB key.  vLLM metadata expects the sequence length including
    # the current decode token, so construct with +1 while logging the original
    # step.  This matches the SGLang collector's generation convention.
    metadata_seq_len = prefix_len + seq_len if is_context else seq_len + 1
    query_len = seq_len if is_context else 1
    with (
        _tp_simulation(tp_size),
        _create_dsv4_attention_module(
            model_path=model_path,
            attn_kind=attn_kind,
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            tp_size=tp_size,
            is_context=is_context,
            device=device,
            query_len=query_len,
        ) as (attn_module, vllm_config),
    ):
        common = _make_common_metadata(
            batch_size=batch_size,
            seq_len=metadata_seq_len,
            is_context=is_context,
            device=device,
            query_len=query_len,
        )
        metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, device=device)

        hf_config = vllm_config.model_config.hf_config
        concrete_layer = vllm_config.compilation_config.static_forward_context[attn_module.prefix]
        backend_name = concrete_layer.get_attn_backend().get_name()
        cache_spec = concrete_layer.get_kv_cache_spec(vllm_config)
        if cache_spec is None:
            raise RuntimeError(f"DSV4 {attn_kind} layer did not register a KV-cache spec")
        architecture = hf_config.architectures[0] if hf_config.architectures else ARCHITECTURE
        # Persisted ``num_heads`` is rank-LOCAL (unified convention);
        # consumers derive native as ``num_heads * tp_size``. vllm's own
        # ``n_local_heads`` is that count — cross-check it against the config
        # so a TP-simulation regression cannot mislabel rows.
        local_num_heads = int(attn_module.n_local_heads)
        expected_local = max(1, int(hf_config.num_attention_heads) // tp_size)
        if local_num_heads != expected_local:
            raise RuntimeError(
                f"DSV4 attention head geometry mismatch: module n_local_heads={local_num_heads} != "
                f"config num_attention_heads // tp_size = {expected_local} (tp_size={tp_size})"
            )
        num_tokens = batch_size * seq_len if is_context else batch_size
        hidden_states = torch.full(
            (num_tokens, hf_config.hidden_size),
            0.01,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = common.positions

        with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), torch.inference_mode():
            attn_module(positions, hidden_states, None)
            torch.cuda.synchronize()

            def kernel_func(
                attn_module=attn_module,
                positions=positions,
                hidden_states=hidden_states,
            ):
                attn_module(positions, hidden_states, None)

            # Deviations from serving, which runs DeepseekV4ForCausalLM on breakable CUDA
            # graphs (config/vllm.py:74-102,812-840) where this times one graph per step:
            # - CSA: serving runs a step above its graph ceiling (1024 tokens on the SM100
            #   family, else 512; config/vllm.py:2341-2446) eagerly, and there the indexer's
            #   short-context branch (max_seq_len // 4 <= index_topk) skips the indexer query
            #   projection, MQA logits and top-k (models/deepseek_v4/attention.py:1082-1107,
            #   keyed on torch.cuda.is_current_stream_capturing()). The graph captured here
            #   always runs them, so such rows overstate serving: about 1.1k (SM100) / 1.4k
            #   (SM90) of the 9,880 CSA context points, and SM90 generation above 512
            #   requests. Steps up to the ceiling are captured in serving too and match.
            #   Retired by timing eager-regime steps eagerly (the pending measurement-method
            #   decision) or by pinning that capture check as collect_mla_module does for DSA.
            # - SM120: serving autotunes the FlashInfer sparse-MLA decode at warm-up
            #   (model_executor/warmup/flashinfer_sparse_mla_warmup.py:225-289,
            #   kernel_warmup.py:234-235); these rows time FlashInfer's untuned tactic.
            with benchmark_with_power(
                device=torch.device(device),
                kernel_func=kernel_func,
                num_warmups=warming_up,
                num_runs=test_ite,
                repeat_n=1,
                use_cuda_graph=True,
            ) as result:
                pass

    latency = float(result["latency_ms"])
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gemm_type": gemm_type,
                "num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": seq_len if is_context else 1,
                "tp_size": tp_size,
                "step": prefix_len if is_context else seq_len,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"dsv4_{attn_kind}_{mode}_module",
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(
        f"[vllm-dsv4] {attn_kind} {mode} b={batch_size} s={seq_len} prefix={prefix_len} "
        f"heads={local_num_heads} backend={backend_name} latency={latency:.4f} ms"
    )
    del attn_module, vllm_config, hidden_states, positions
    torch.cuda.empty_cache()
    gc.collect()
    return latency


def _kv_cache_cast_to_fp8_indexer(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = x_amax / 448.0
    x_scaled = (x * (1.0 / scale)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    out[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    out[:, block_size * head_dim :] = scale.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def _bench_paged_mqa_logits_kernel(
    *,
    hf_config,
    num_query_rows: int,
    past_kv: int,
    device: str,
    warming_up: int,
    test_ite: int,
) -> dict:
    """Benchmark paged MQA logits with packed M rows across the batch."""
    m = num_query_rows
    full_s = m + past_kv
    full_c4 = max(1, full_s // 4)
    block_kv = 64
    n_heads = int(hf_config.index_n_heads)
    head_dim = int(hf_config.index_head_dim)

    q_bf16 = torch.randn(m, 1, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_quant = q_bf16.to(torch.float8_e4m3fn)

    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(
        blocks_per_req,
        block_kv,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(m, n_heads, dtype=torch.float32, device=device)
    causal_seq = torch.arange(
        past_kv + 1,
        past_kv + m + 1,
        dtype=torch.int32,
        device=device,
    )
    context_lens = (causal_seq // 4).clamp(min=1).view(m, 1)

    block_table = torch.arange(blocks_per_req, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(m, blocks_per_req).contiguous()
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_compute_units(),
    )

    def kernel_func():
        return fp8_fp4_paged_mqa_logits(
            (q_quant, None),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_model_len=full_c4,
            clean_logits=False,
        )

    with torch.inference_mode():
        kernel_func()
        torch.cuda.synchronize()
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=True,
        ) as result:
            pass
    if not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    return result


def _bench_sparse_kernel_shape(
    *,
    model_path: str,
    kernel: str,
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    device: str,
    perf_filename: str,
    warming_up: int,
    test_ite: int,
) -> float | None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    full_seq_len = past_kv + isl
    if full_seq_len <= 0:
        raise ValueError(f"invalid sparse sequence length: isl={isl}, past_kv={past_kv}")

    attn_kind = SPARSE_KERNEL_TO_ATTN_KIND[kernel]
    num_tokens = batch_size * isl

    hf_config = SimpleNamespace(**_read_model_config(model_path))
    architecture = hf_config.architectures[0] if hf_config.architectures else ARCHITECTURE
    native_num_heads = int(hf_config.num_attention_heads)
    local_num_heads = native_num_heads // tp_size
    backend_name = "vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits"
    result = _bench_paged_mqa_logits_kernel(
        hf_config=hf_config,
        num_query_rows=num_tokens,
        past_kv=past_kv,
        device=device,
        warming_up=warming_up,
        test_ite=test_ite,
    )

    latency = float(result["latency_ms"])

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "fp8_e4m3",
                "kv_cache_dtype": "fp8",
                "gemm_type": "fp8_block",
                # Sparse-op consumers key ``num_heads`` by the native model
                # count. Keep that contract and record the TP-local count too.
                "num_heads": native_num_heads,
                "local_num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": past_kv,
                "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                "latency": f"{latency:.4f}",
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=SPARSE_KERNEL_TO_OP_NAME[kernel],
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=result.get("power_stats"),
    )
    print(
        f"[vllm-dsv4] {kernel} b={batch_size} isl={isl} past_kv={past_kv} "
        f"tp={tp_size} local_heads={local_num_heads} backend={backend_name} "
        f"latency={latency:.4f} ms"
    )
    torch.cuda.empty_cache()
    gc.collect()
    return latency


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    attention_backend: str | None = None,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    if attn_kind not in DSV4_ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    if kv_cache_dtype != "fp8":
        raise ValueError(f"unsupported vLLM DSV4 kv_cache_dtype={kv_cache_dtype}; expected fp8")
    if compute_dtype != "bfloat16":
        raise ValueError(f"unsupported vLLM DSV4 compute_dtype={compute_dtype}; expected bfloat16")
    if gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM DSV4 gemm_type={gemm_type}; supported={sorted(SUPPORTED_GEMM_TYPES)}")
    if attention_backend is not None:
        raise ValueError(
            f"vLLM DSV4 attention_backend must be unset because vLLM selects it internally; got {attention_backend!r}"
        )

    mode = "context" if "context" in os.path.basename(perf_filename) else "generation"
    _init_cuda(device)
    try:
        _bench_attention_shape(
            model_path=model_path,
            attn_kind=attn_kind,
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            prefix_len=prefix_len,
            tp_size=tp_size,
            gemm_type=gemm_type,
            device=device,
            perf_filename=perf_filename,
            warming_up=5,
            test_ite=10,
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"[vllm-dsv4] OOM: {attn_kind} {mode} b={batch_size} s={seq_len} prefix={prefix_len}")
        torch.cuda.empty_cache()
        raise
    except Exception:
        traceback.print_exc()
        raise


def run_dsv4_sparse_kernel_worker(
    batch_size: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    kernel: str,
    model_path: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    if kernel not in SPARSE_KERNEL_TO_ATTN_KIND:
        raise ValueError(f"unknown sparse kernel={kernel}")
    if tp_size not in _DSV4_MODULE_TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}")
    full_s = isl + past_kv
    max_full_s = get_dsv4_sweep().sparse_max_full_sequence_length
    if full_s > max_full_s:
        raise ValueError(
            f"{kernel} b={batch_size} isl={isl} past_kv={past_kv} has "
            f"full_s={full_s} > max_position_embeddings={max_full_s}"
        )
    _init_cuda(device)
    try:
        _bench_sparse_kernel_shape(
            model_path=model_path,
            kernel=kernel,
            batch_size=batch_size,
            isl=isl,
            past_kv=past_kv,
            tp_size=tp_size,
            device=device,
            perf_filename=perf_filename,
            warming_up=5,
            test_ite=10,
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"[vllm-dsv4] OOM: {kernel} b={batch_size} isl={isl} past_kv={past_kv} tp={tp_size}")
        torch.cuda.empty_cache()
        raise
    except Exception:
        traceback.print_exc()
        raise


def get_dsv4_csa_context_test_cases():
    return plan_cases("dsv4_csa_context_module", "vllm", get_sm_version())


def get_dsv4_hca_context_test_cases():
    return plan_cases("dsv4_hca_context_module", "vllm", get_sm_version())


def get_dsv4_csa_generation_test_cases():
    return plan_cases("dsv4_csa_generation_module", "vllm", get_sm_version())


def get_dsv4_hca_generation_test_cases():
    return plan_cases("dsv4_hca_generation_module", "vllm", get_sm_version())


def get_dsv4_paged_mqa_logits_test_cases():
    return plan_cases("dsv4_paged_mqa_logits_module", "vllm", get_sm_version())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM DeepSeek-V4 attention module latency.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--attn-kind", choices=list(DSV4_ATTN_KINDS), default="csa")
    parser.add_argument("--sparse-kernel", choices=list(SPARSE_KERNEL_TO_ATTN_KIND), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--past-kv", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gemm-type", default="fp8_block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    filename = {
        ("csa", "context"): PerfFile.DSV4_CSA_CONTEXT_MODULE.value,
        ("hca", "context"): PerfFile.DSV4_HCA_CONTEXT_MODULE.value,
        ("csa", "generation"): PerfFile.DSV4_CSA_GENERATION_MODULE.value,
        ("hca", "generation"): PerfFile.DSV4_HCA_GENERATION_MODULE.value,
    }[(args.attn_kind, args.mode)]
    if args.gemm_type not in SUPPORTED_GEMM_TYPES:
        raise ValueError(f"unsupported vLLM DSV4 gemm_type={args.gemm_type}; supported={sorted(SUPPORTED_GEMM_TYPES)}")
    _init_cuda(args.device)
    if args.sparse_kernel is not None:
        sparse_filename = SPARSE_KERNEL_TO_PERF_FILE[args.sparse_kernel].value
        perf_filename = _resolve_perf_path(args.output_path, sparse_filename)
        _bench_sparse_kernel_shape(
            model_path=args.model_path,
            kernel=args.sparse_kernel,
            batch_size=args.batch_size,
            isl=args.seq_len,
            past_kv=args.past_kv,
            tp_size=args.tp_size,
            device=args.device,
            perf_filename=perf_filename,
            warming_up=3,
            test_ite=3,
        )
        return

    perf_filename = _resolve_perf_path(args.output_path, filename)
    _bench_attention_shape(
        model_path=args.model_path,
        attn_kind=args.attn_kind,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        prefix_len=args.past_kv if args.mode == "context" else 0,
        tp_size=args.tp_size,
        gemm_type=args.gemm_type,
        device=args.device,
        perf_filename=perf_filename,
        warming_up=3,
        test_ite=3,
    )


if __name__ == "__main__":
    main()
