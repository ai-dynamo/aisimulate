# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Versions are upgraded IN PLACE: this file always
# targets the manifest pin; prior-version code is `git log -- <this file>`
# away and prior-version DATA stays permanent under its version key. The kv
# dtype FORKS the SM90 DSA backend (bf16 -> FLASH_ATTN_MLA_SPARSE, fp8 ->
# FLASHMLA_SPARSE; platforms/cuda.py:137-153 @v0.32.0) — both keys are
# collected.
#
# Portions adapted from vLLM dd10e03f95f94edbea1975c67ace3a35ec9a8a40.
# Copyright contributors to the vLLM project. Modified for collector execution.
__compat__ = "vllm==0.32.0"

"""
MLA Module Collector for vLLM — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass (projections + attention +
output), not just the bare attention kernel. Uses vLLM's own modeling code to
construct a single `DeepseekV2MLAAttention` module with dummy weights, then
benchmarks its forward.

MLA vs DSA is determined by the presence of `index_topk` in the HF config.
Op names and data schema are aligned with TRT-LLM's collect_mla_module.py
so that queries can be reused across frameworks.

Supported models, attention types, and micro-sweeps are defined in collector v2
YAML and loaded through collector.case_generator. The collector reads a real HF
config, overrides the layer-local shape fields in-memory, and then instantiates
just the attention module.

Case layout: one case per (batch, heads, kv dtype, compute dtype, gemm type,
model, attention type) carrying its points: seq_lens, or [seq_len, prefix_len]
pairs for DSA context; run_mla_module builds the module once per case and
measures every point on it, each with its own KV cache and metadata (the module
build was most of the per-point cost in the 2026-10 campaign: vllm
dsa_context_module alone was 59% of its GPU time).

Usage:
    # MLA context phase (DeepSeek-V3 style)
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3

    # DSA generation phase (DeepSeek-V3.2 style)
    python collect_mla_module.py --mode generation --model deepseek-ai/DeepSeek-V3.2

    # The op plans' references (collector/op_plans.py), context phase
    python collect_mla_module.py --mode context

    # Quick single-point test
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3 --quick --batch-size 4 --seq-len 2048
"""

import argparse
import gc
import json
import math
import os
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
from collector.case_generator import get_mla_module_model_specs
from collector.helper import benchmark_with_power as _benchmark_with_power
from collector.helper import flashinfer_autotuned, flashinfer_tuning_bucket, get_sm_version, log_perf
from collector.op_plans import plan, plan_cases
from collector.registry_types import PerfFile
from collector.vllm.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_vllm_config,
    enable_engine_fused_ops,
    setup_distributed,
    with_exit_stack,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — vLLM 0.24.0 registers the GlmMoeDsaForCausalLM
# model class but omits the config-type mapping for "glm_moe_dsa", so
# AutoConfig.from_pretrained() fails.  The config layout is identical to
# DeepSeek-V3 (GlmMoeDsaForCausalLM inherits DeepseekV2ForCausalLM), so
# reusing DeepseekV3Config is safe.
# ═══════════════════════════════════════════════════════════════════════
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

# Keep graph lifetime handling local to the MLA/DSA module collector. Other
# collectors retain benchmark_with_power's default cleanup and synchronization.
benchmark_with_power = partial(_benchmark_with_power, explicit_graph_cleanup=True)

if "glm_moe_dsa" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"


# ═══════════════════════════════════════════════════════════════════════
# Local model config resolution — avoid HuggingFace Hub downloads
# ═══════════════════════════════════════════════════════════════════════

# Pre-cached HF configs ship with AISim as "<org>--<model>_config.json".
# The directory is helper.py's constant — never recomputed here, so a
# package rename cannot silently strand this lane (the aiconfigurator ->
# aisimulate consolidation did exactly that to a hand-built copy of it).
# vLLM's ModelConfig accepts a local directory containing config.json, so
# we create a temp dir with a symlink when the cached file exists.
from collector.helper import _AIC_MODEL_CONFIG_DIR as _AIS_MODEL_CONFIG_DIR

_MODEL_CONFIGS_DIR = Path(_AIS_MODEL_CONFIG_DIR)

# Cache of model_name -> temp dir path (created once per process).
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local directory path for *model_name* if a cached config exists, else return model_name as-is."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]

    config_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_config.json"
    if not config_file.exists():
        return model_name

    # Create a temp directory with config.json so vLLM's ModelConfig
    # loads from disk instead of downloading from HuggingFace Hub.
    tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
    os.symlink(config_file, os.path.join(tmp_dir, "config.json"))
    # Strip auto_map if present.  Some models (e.g. DeepSeek-V3) ship
    # config.json with auto_map pointing to a custom Python config class
    # (configuration_deepseek.py).  HuggingFace's AutoConfig.from_pretrained()
    # — called by vLLM's ModelConfig — unconditionally tries to import that
    # module from the model directory where it doesn't exist; vLLM natively
    # supports these architectures and only needs the JSON fields.
    with open(config_file) as f:
        config_data = json.load(f)
    if "auto_map" in config_data:
        config_data.pop("auto_map")
        os.remove(os.path.join(tmp_dir, "config.json"))
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_data, f)

    # Also symlink hf_quant_config.json if present (used by quantized models).
    quant_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_hf_quant_config.json"
    if quant_file.exists():
        os.symlink(quant_file, os.path.join(tmp_dir, "hf_quant_config.json"))

    _local_config_cache[model_name] = tmp_dir
    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════


def get_mla_context_module_test_cases():
    return plan_cases("mla_context_module", "vllm", get_sm_version())


def get_mla_generation_module_test_cases():
    return plan_cases("mla_generation_module", "vllm", get_sm_version())


def get_dsa_context_module_test_cases():
    return plan_cases("dsa_context_module", "vllm", get_sm_version())


def get_dsa_generation_module_test_cases():
    return plan_cases("dsa_generation_module", "vllm", get_sm_version())


# MLA cache layout of every model this module collects (DeepSeek-family
# checkpoints: kv_lora_rank 512 + qk_rope_head_dim 64). Used only as a lower
# bound when refusing generation points the device cannot hold.
_MLA_KV_ENTRY_ELEMS = 512 + 64

_MEMORY_BUDGET_SAFETY_FACTOR = 0.9


def _generation_kv_footprint_bytes(total_tokens: int, kv_cache_dtype: str) -> int:
    """Lower bound of a generation case's peak device footprint.

    run_mla_module materializes the per-request context KV inputs and the
    paged cache they are copied into (utils.create_and_prepopulate_kv_cache_mla),
    i.e. two allocations of ~total_tokens x 576 elements each. fp8 layouts use
    at least one byte per element (the packed fp8_ds_mla entry is 656 B/token,
    still above this bound). Module weights and workspace are deliberately
    excluded so the estimate stays a provable lower bound: a case refused on it
    cannot fit on the device, on any platform.
    """
    bytes_per_elem = 1 if "fp8" in kv_cache_dtype else 2
    return 2 * total_tokens * _MLA_KV_ENTRY_ELEMS * bytes_per_elem


def _kv_does_not_fit(total_tokens: int, kv_cache_dtype: str, total_memory: int) -> str | None:
    """Why a generation point's KV cache cannot fit the device, or None when it can.

    The module plans do not depend on the device, so the run refuses such a
    point before allocating it. The footprint is a lower bound, so a refused point
    could not have run.
    """
    needed = _generation_kv_footprint_bytes(total_tokens, kv_cache_dtype)
    allowed = int(total_memory * _MEMORY_BUDGET_SAFETY_FACTOR)
    if needed <= allowed:
        return None
    return (
        f"FIXME(kernel-limit): does not fit: the KV cache needs {needed / 2**30:.1f} GiB, "
        f"the device allows {allowed / 2**30:.1f} GiB ({_MEMORY_BUDGET_SAFETY_FACTOR:g} of its memory)"
    )


def _strided_throwaway_copy(query):
    """A scratch tensor with the query's shape AND strides — never a contiguous clone.

    vLLM's own rope kernel derives ONE head stride from the query and applies it to the
    key as well (csrc/pos_encoding_kernels.cu:57,68,164-165 @v0.32.0). The q_pe that
    models/deepseek_v32/attention.py:558-562 @v0.32.0 ropes alone is a last-dim slice of
    q, so its head stride is qk_head_dim (256 for GLM-5.2, 192 for DeepSeek-V3.2) while a
    contiguous clone's is 64: the kernel then addresses the clone at
    token*heads*64 + head*256, i.e. up to 192*(heads-1) elements PAST its end, and ropes
    in place — a CUDA illegal memory access when the overrun leaves the allocation,
    silent corruption of the neighbouring tensor otherwise. That was the GLM-5.2 fp8-KV
    dsa_context (isl <= index_topk) illegal-address / hang family on b200, b300, gb200
    and gb300. FlashInfer's rope takes per-tensor strides (csrc/rope.cu
    q_stride_h / k_stride_h) and is safe with either layout; matching the
    query's strides is correct for both kernels.
    """
    shape = tuple(query.shape)
    stride = tuple(query.stride())
    if any(s == 0 for s in shape):
        return query.clone()
    storage_numel = 1 + sum((s - 1) * st for s, st in zip(shape, stride, strict=True))
    key = query.new_empty(storage_numel).as_strided(shape, stride)
    key.copy_(query)
    return key


def _rope_tolerating_missing_key(forward):
    """Wrap a rotary module's forward so `forward(positions, query)` (key=None) works.

    vLLM's DeepseekV32 attention ropes q_pe alone on its dense/masked-MHA prefill path
    when the forward quantized the MQA query to FP8 (models/deepseek_v32/attention.py:
    556-562 @v0.32.0), but both DeepseekScalingRotaryEmbedding implementations require a
    key (forward_native asserts, deepseek_scaling_rope.py:115-123; the FlashInfer op calls
    key.view, flashinfer 0.7.0.post1 rope.py:1278). Rope the query against a throwaway
    copy that shares the query's strides (see _strided_throwaway_copy) and drop it.
    """
    import functools

    if getattr(forward, "_aisim_rope_shim", False):
        # Idempotent: run_mla_module installs the shim per case on a module that may be reused,
        # and named_modules() can visit a shared rotary twice. Re-wrapping stacked one closure
        # per case until "maximum recursion depth exceeded" (b200/gb200 pipelines 72387095 /
        # 72387091 at e598d380: 2,286 cases each).
        return forward

    @functools.wraps(forward)
    def wrapped(positions, query, key=None, *args, **kwargs):
        # Forward only what the caller passed: the base RotaryEmbedding.forward_cuda takes
        # (positions, query, key) — no `offsets` (rotary_embedding/base.py:221-226 @v0.32.0) —
        # so a positional None for offsets raised
        # "takes from 3 to 4 positional arguments but 5 were given" (508 GLM-5 cases, same runs).
        if key is None:
            out = forward(positions, query, _strided_throwaway_copy(query), *args, **kwargs)
            q = out[0] if isinstance(out, tuple) else out
            return q, None
        return forward(positions, query, key, *args, **kwargs)

    wrapped._aisim_rope_shim = True
    return wrapped


def _rope_module_rejects_missing_key(module) -> bool:
    """True only for rotary classes whose forward cannot take key=None.

    Serving calls ``self.rotary_emb(positions, q_pe)`` with no key
    (models/deepseek_v32/attention.py:558-562 @v0.32.0). vLLM's base RotaryEmbedding and
    every subclass that reaches ops.rotary_embedding accept that (the kernel gets a null
    key pointer and never touches key memory; _custom_ops.py:210-219 key: Tensor | None),
    so GLM-5.x (rope_type "default" -> RotaryEmbedding) must run exactly serving's call —
    wrapping it fed vLLM's kernel a contiguous key whose head stride differs from the
    query's (see _strided_throwaway_copy). Only DeepseekScalingRotaryEmbedding's own
    forward_cuda rejects None (deepseek_scaling_rope.py:203-221: FlashInfer op
    key.view / forward_native assert key is not None); its DeepseekV4 subclass overrides
    forward_cuda onto ops.rotary_embedding and accepts None again. All @v0.32.0.
    """
    mro = type(module).__mro__
    if not any(c.__name__ == "DeepseekScalingRotaryEmbedding" for c in mro):
        return False
    owner = next((c for c in mro if "forward_cuda" in vars(c)), None)
    return owner is not None and owner.__name__ == "DeepseekScalingRotaryEmbedding"


def _install_rope_missing_key_shims(attn_module) -> tuple[list[str], list[str]]:
    """Shim only the rotary submodules that reject key=None.

    Returns the class names that were wrapped and the rotary class names left on
    serving's own path.
    """
    patched: list[str] = []
    left: list[str] = []
    modules = attn_module.named_modules() if hasattr(attn_module, "named_modules") else ()
    for _name, module in modules:
        if not (type(module).__name__.endswith("RotaryEmbedding") and hasattr(module, "forward")):
            continue
        if _rope_module_rejects_missing_key(module):
            module.forward = _rope_tolerating_missing_key(module.forward)
            patched.append(type(module).__name__)
        else:
            left.append(type(module).__name__)
    return patched, left


def _model_max_position_embeddings(model_id: str) -> int | None:
    """The model's max context length from its cached config (same file the
    sglang module collector reads); None when unavailable (no cap applied)."""
    config_file = os.path.join(_AIS_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file) as f:
            value = json.load(f).get("max_position_embeddings")
        return int(value) if value else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Module Construction
# ═══════════════════════════════════════════════════════════════════════


def _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata):
    """Ground-truth backend for the perf row.

    MLA prefill runs mla_layer.prefill_backend; everything else — DSA, decode,
    and "context" batches that vLLM classified entirely as decodes
    (num_prefills == 0, e.g. s=1) — runs mla_layer.attn_backend.

    ``num_prefills`` must only be read on the MLA-context branch: sparse DSA
    metadata (FlashMLASparseMetadata / FlashInferMLASparseMetadata)
    does not carry that attribute at the top level, so an eager read breaks
    every DSA row.
    """
    if attn_type == "dsa" or not is_context or attn_metadata.num_prefills == 0:
        return mla_layer.attn_backend.get_name()
    return mla_layer.prefill_backend.get_name()


def _create_gemm_quant_config(gemm_type: str):
    """Create the vLLM QuantizationConfig for a given gemm_type.

    Returns None for bfloat16 (unquantised GEMMs).
    Quantized modes use their serialized native weight format; synthetic
    parameters are populated before native ``process_weights_after_loading``.
    """
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        # vLLM requires is_checkpoint_fp8_serialized=True for block-scaled
        # FP8 (fp8.py raises ValueError otherwise).  This routes through
        # Fp8LinearMethod (block_quant=True) → W8A8BlockFp8LinearOp →
        # DeepGEMM on SM≥89.
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    if gemm_type == "nvfp4":
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
    raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


def _move_module_preserving_buffers(module, device):
    """Materialize meta weights without discarding initialized runtime buffers.

    Two construction paths reach here. The explicit-class path builds the
    module on the host, so only serialized block-scaled FP8 weights are meta
    and every buffer already holds its constructor value. The serving-parity
    path instantiates the framework's whole model under torch.device("meta")
    (_create_attention_module), so its buffers are meta as well, and each is
    rebuilt from its owner's own constructor recipe:

    - rotary cos/sin tables: `rotary_emb` and `indexer_rope_emb` on
      DeepseekV32Attention (models/deepseek_v32/attention.py:272-284 @v0.32.0),
      registered non-persistent by rotary_embedding/base.py:58-63
      and deepseek_scaling_rope.py:241-242 from a device-free recipe
      (_compute_cos_sin_cache: torch.arange / einsum on the default device),
      cast to the dtype the constructor chose (the meta tensor carries it);
    - KV-quant scales `_k_scale/_v_scale/_q_scale/_prob_scale` that
      MLAAttention registers through _init_kv_cache_quant ->
      set_default_quant_scales(register_buffer=True) as 1.0 fp32
      (layers/attention/attention.py:133-139,188 @v0.32.0), together with
      the `<name>_cpu` host copies the same call keeps beside them.

    Any other meta buffer is refused rather than filled: a filled table
    silently changes what the kernels see (the pre-#404 collector filled RoPE
    with 0.01 and these scales with 0.5).
    """
    if not any(parameter.is_meta for parameter in module.parameters()):
        return module.to(device)
    buffers = {}
    for name, buffer in module.named_buffers():
        if buffer.is_meta:
            parent_name, _, local_name = name.rpartition(".")
            owner = module.get_submodule(parent_name) if parent_name else module
            recompute = getattr(owner, "_compute_cos_sin_cache", None)
            if local_name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
                buffer = torch.ones(buffer.shape, dtype=buffer.dtype)
                host_name = f"{local_name}_cpu"
                if getattr(getattr(owner, host_name, None), "is_meta", False):
                    setattr(owner, host_name, torch.ones((), dtype=torch.float32))
            elif local_name.startswith("cos_sin_cache") and callable(recompute):
                table = recompute()
                if table.is_meta or tuple(table.shape) != tuple(buffer.shape):
                    raise RuntimeError(
                        f"Rotary table rebuilt for {name} is {tuple(table.shape)} on {table.device}; "
                        f"the constructor registered {tuple(buffer.shape)}"
                    )
                buffer = table.to(dtype=buffer.dtype)
            else:
                raise RuntimeError(f"Cannot preserve an uninitialized meta buffer in the attention module: {name}")
        buffers[name] = buffer
    module = module.to_empty(device=torch.device(device))
    for name, buffer in buffers.items():
        parent_name, _, local_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, local_name, buffer.to(device))
    return module


def _initialize_synthetic_parameters(module):
    """Keep synthetic FP8 projections nonzero without using CUDA RNG."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, tensor in module.named_parameters():
            if tensor.is_meta:
                continue
            is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if is_fp8 and name.rsplit(".", 1)[-1] != "weight":
                # NVFP4's weight_scale also uses FP8 storage (KNvfp4Static,
                # modelopt.py:2019-2035 @v0.32.0). It is a scale, not a
                # projection weight; retain its previous policy.
                tensor.zero_()
                continue
            if is_fp8 or (name.endswith("wq_b.weight") and torch.is_floating_point(tensor)):
                # Serialized FP8 weights feed the real indexer query projection
                # (DeepseekV32Indexer.wq_b, models/deepseek_v32/attention.py:71-77
                # @v0.32.0). Zero weights
                # make every query/logit zero even with valid historical keys;
                # top-k latency depends on that distribution. An unquantized wq_b
                # takes the same draw: under the 0.01 fill every index head had the
                # same query, so a key scored relu(q.k) times the summed head weights,
                # exactly 0.0 for about half the history, and while a row had fewer
                # than about twice index_topk candidates the top-k threshold fell
                # inside that tie group. Generate on CPU
                # because CUDA RNG can be unavailable after module construction.
                flat = tensor.view(-1)
                for start in range(0, flat.numel(), 1048576):
                    end = min(start + 1048576, flat.numel())
                    values = torch.randn(end - start, generator=generator, dtype=torch.float32, device="cpu") * 0.02
                    flat[start:end].copy_(values.to(device=tensor.device, dtype=tensor.dtype))
            elif tensor.dtype == torch.uint8:
                tensor.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.fill_(0.5)
            else:
                tensor.fill_(0.01)


def _initialize_nvfp4_parameters(module):
    """Populate ModelOpt NVFP4 linears through the native quantizer.

    This is synthetic format qualification, not checkpoint calibration. The
    CPU RNG is local, temporary tensors are bounded, and framework buffers are
    untouched. vLLM subsequently performs its own layout conversion/dispatch.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

    # Native API contracts at vLLM v0.32.0: every ModelOpt linear is the
    # generic ModelOptLinearMethod carrying a (weight, activation) QuantSpec
    # and a CkptCtx (modelopt.py:2522-2560, CkptCtx :1923). The NVFP4 weight
    # scheme KNvfp4Static stores two E2M1 values per uint8, one E4M3 scale per
    # ctx.group_size inputs (created as a NaN sentinel that process() refuses)
    # and a per-partition FP32 global scale `weight_scale_2`
    # (modelopt.py:1987-2064); the W4A4 activation scheme KNvfp4Dynamic adds
    # the per-partition `input_scale` (modelopt.py:2067-2101) while W4A16 has
    # none. Both global scales are ModelOpt's dequantizing amax/(6*448)
    # values, raw-max'ed at process_weights_after_loading. _custom_ops.py:
    # 1513-1532 accepts the inverse global scale and can return unswizzled
    # block scales.
    generator = torch.Generator(device="cpu").manual_seed(0)
    initialized = 0
    with torch.no_grad():
        for name, layer in module.named_modules():
            quant_method = getattr(layer, "quant_method", None)
            spec = getattr(quant_method, "spec", None)
            if spec is None or getattr(spec, "weight", None) != kNvfp4Static:
                continue
            weight, scales = layer.weight, layer.weight_scale
            if weight.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Unexpected native NVFP4 storage for {name}")
            if weight.ndim != 2 or tuple(scales.shape) != (weight.shape[0], weight.shape[1] // 8):
                raise ValueError(f"Unexpected native NVFP4 block layout for {name}")
            group_size = getattr(getattr(quant_method, "ctx", None), "group_size", None)
            if group_size != 16 or weight.shape[1] % 8:
                raise ValueError(f"Native NVFP4 quantization requires 16-input blocks for {name}")
            input_size = weight.shape[1] * 2
            rows_per_chunk = max(1, 1048576 // input_size)
            # Bounded N(0,.02) weights, clipped to +/- .08, and a unit-range
            # synthetic activation calibration. E2M1 max=6, E4M3 max=448.
            # These fixed source ranges are unrelated to timing/accuracy targets.
            weight_global_scale = 0.08 / (6.0 * 448.0)
            layer.weight_scale_2.fill_(weight_global_scale)
            input_scale = getattr(layer, "input_scale", None)  # W4A4 only (KNvfp4Dynamic)
            if input_scale is not None:
                input_scale.fill_(1.0 / (6.0 * 448.0))
            inverse_scale = torch.tensor(1.0 / weight_global_scale, dtype=torch.float32, device=weight.device)
            for start in range(0, weight.shape[0], rows_per_chunk):
                end = min(start + rows_per_chunk, weight.shape[0])
                values = torch.randn((end - start, input_size), generator=generator, dtype=torch.float32, device="cpu")
                values.clamp_(-4.0, 4.0).mul_(0.02)
                packed, block_scales = ops.scaled_fp4_quant(
                    values.to(device=weight.device, dtype=torch.bfloat16),
                    inverse_scale,
                    is_sf_swizzled_layout=False,
                )
                if packed.shape != weight[start:end].shape or block_scales.shape != scales[start:end].shape:
                    raise ValueError(f"Native NVFP4 quantizer returned an incompatible shape for {name}")
                if not bool(torch.isfinite(block_scales.float()).all()) or not bool((block_scales.float() > 0).all()):
                    raise ValueError(f"Native NVFP4 quantizer returned invalid block scales for {name}")
                weight[start:end].copy_(packed)
                scales[start:end].copy_(block_scales)
            initialized += 1
    if not initialized:
        raise RuntimeError("NVFP4 was requested but no native ModelOpt NVFP4 linear was constructed")


def _create_attention_module(
    model_path: str,
    attn_type: str,
    num_heads: int,
    use_fp8_kv_cache: bool,
    max_seq_len: int,
    max_batch_size: int,
    max_query_len: int,
    use_prefill_fp8: bool = False,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
    is_context: bool = True,
    max_model_len: int | None = None,
):
    """
    Create a DeepseekV2MLAAttention module from vLLM's own modeling code.

    Loads a real HF config from model_path, overrides the layer-local attention
    dimensions we want to benchmark in-memory, and then constructs the module
    with dummy weights. The module includes all projections + attention +
    output.

    Args:
        model_path: HuggingFace model path (e.g. "deepseek-ai/DeepSeek-V3.2").
        attn_type: Attention type ("mla" or "dsa").
        max_seq_len: Total context length, including the cached prefix.
        max_model_len: The engine's max_model_len (default: max_seq_len, at least 4096).
            Indexer buffers scale with it, so a series passes the model's
            max_position to time every point at the size serving runs.
        max_query_len: New tokens per request for a context forward pass.
        use_prefill_fp8: When True, opt in to FP8 prefill query compute via
            ``attention_config.use_prefill_query_quantization`` plus an
            explicit ``attention_config.mla_prefill_backend`` pin. vLLM honors
            the quantization flag only with an FP8 KV cache on the SM100
            family with a FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA prefill
            backend (determine_prefill_query_data_type +
            backend_supports_prefill_query_quantization,
            mla_attention.py:1877-1903,2320-2360 @v0.32.0), and the SM100
            auto selector picks FLASH_ATTN for DeepSeek-V3 dims instead
            (platforms/interface.py:462-480 get_mla_prefill_backend_cls
            @v0.32.0); the metadata probe in
            _create_kv_cache_and_metadata fails closed if the decision
            does not match the case label.
        gemm_type: Precision for linear-layer GEMMs — "bfloat16",
            "fp8_block", or "nvfp4".
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

    local_model_path = _resolve_model_path(model_path)

    # The module is built at the framework default page; the real page is the one serving
    # derives from the backend the module selected (_create_kv_cache_and_metadata).
    block_size = None  # unconstrained selection; _create_kv_cache_and_metadata asks the selected backend for the page
    # seq_len includes the current token; generation caches only seq_len - 1.
    # Keep exact-limit models such as Kimi-K2-Instruct at their declared limit.
    max_model_len = max(max_seq_len, max_model_len or 4096)
    num_kv_cache_blocks = max(
        1
        + math.ceil((max_seq_len + 1) / 16)
        * max_batch_size,  # upper bound at the smallest page; recomputed per page below
        8192,
    )

    # FIXME(gap): the module is built at TP=1 with the per-rank head count, so a check that
    # multiplies by the TP world size sees the shard. The sparse-MLA masked-MHA prefill (SM100
    # family, FA4, unquantized KV) admits only the model-level dims of DeepSeek-V3.2 (128
    # heads) and GLM-5.x (64 heads) (_is_masked_mha_available, sparse_mla_attention.py:93-127,
    # called with num_heads * the TP world size at :695-702), and _use_masked_mha reads its
    # thresholds by tensor_parallel_size (mla_attention.py:1193-1218,1761-1872). So for a TP>1
    # shard, e.g. GLM-5.2 TP8 BF16 KV, serving runs masked MHA for pure prefills up to 64K
    # while this module times sparse MQA. Affects SM100 BF16-KV DSA context rows below the full
    # head count (FP8 KV never takes masked MHA; TP1 rows match). Not a gap: the indexer's TP
    # prefill row sharding (indexer.py:208-261,1577-1586) engages only for steps of at least
    # 16384 * TP prefill rows, above serving's default token budget (8192/16384,
    # engine/arg_utils.py:2923-2943). All @v0.32.0.
    # Determine kv cache dtype string for sparse MLA.
    # For DSA (DeepSeekV3.2), fp8 uses the custom ``fp8_ds_mla`` 656-byte
    # cache format (512B quantized NoPE + 16B scales + 128B RoPE).
    # For dense MLA, standard fp8 (fp8_e4m3) is used.
    is_dsa = attn_type == "dsa"

    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        # The case's batch, not serving's scheduler bound (1024 on 70 GB+ GPUs,
        # collect_attn.SERVING_MAX_NUM_SEQS). Of the dense-MLA decode builders only
        # FLASH_ATTN_MLA (SM90, BF16 KV) times differently with it: it caps FA3 split-KV
        # at flash_attn_max_num_splits_for_cuda_graph (32) for decode batches of up to
        # max_cudagraph_capture_size tokens (v1/attention/backends/mla/flashattn_mla.py:
        # 151-176,225-236), and that size is min(2 * max_num_seqs, 512 on SM90,
        # max_num_batched_tokens) (config/vllm.py:2354-2364,2445-2446, all @v0.32.0).
        # Generation batches reach serving's decision either way. Context batches with
        # s <= 512 also take the decode path (reorder_batch_threshold 512,
        # flashattn_mla.py:130); for b*s in (2b, 512] serving caps at 32 splits where these
        # rows run FA3's heuristic, but with no cached prefix their KV is at most 512 tokens,
        # fewer key blocks than the cap. TRITON_MLA (SM89, SM120) reserves its decode
        # workspace at max_num_seqs x heads x splits (triton_mla.py:67-92): at serving's
        # 1024 that is 70-95 GiB for 128 heads, which those GPUs cannot hold.
        max_num_seqs=max_batch_size,
        # The serving scheduler budgets NEW tokens, excluding computed prefix
        # tokens (v1/core/sched/scheduler.py:472-480). DeepseekV2Model
        # sizes its indexer output as [max_num_batched_tokens, index_topk]
        # (model_executor/models/deepseek_v2.py:1442-1446 @v0.32.0). Keep the
        # existing sweep floor, but do not allocate one output row per cached
        # token. max_model_len and KV capacity still include the entire prefix.
        max_num_batched_tokens=max(max_batch_size * max_query_len, 131072) if is_context else max_batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
        trust_remote_code=True,
        # Forward the test sweep's per-case head counts so that
        # ModelConfig.model_arch_config (built once at __init__ from hf_config) stays in
        # sync. Without this, the V1 FA3 builder reads the model's natural head count
        # via get_num_attention_heads(), the AOT scheduler precomputes
        # scheduler_metadata for that shape, and impl.forward then runs with the test's
        # actual head count — _vllm_fa3_C.fwd's shape check rejects the mismatch.
        # MLA collapses KV to 1 head via get_num_kv_heads (use_mla=True), so the
        # num_kv_heads override is a no-op in that path; we pass it for parity with
        # the attention collector.
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    # Override quant_config to control linear-layer GEMM precision.
    # DeepSeek-V3.2 ships with FP8 quantisation by default, so we
    # must always set quant_config explicitly: None for bf16,
    # Fp8Config for fp8/fp8_block, ModelOptNvFp4Config for nvfp4.
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    # Opt in to FP8 prefill query compute before the module (and later the
    # metadata builder) reads the attention config.
    if use_prefill_fp8:
        from vllm.platforms import current_platform

        vllm_config.attention_config.use_prefill_query_quantization = True
        if current_platform.is_device_capability_family(100):
            from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

            # The quantization flag alone never engages on SM100: for
            # DeepSeek-V3 dims the auto selector ranks FLASH_ATTN first
            # (platforms/interface.py:474-480 @v0.32.0) and FLASH_ATTN is
            # always eligible there, but FP8 query compute requires a
            # FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA prefill backend
            # (backend_supports_prefill_query_quantization,
            # mla_attention.py:1877-1903 @v0.32.0). The FP8-prefill serving
            # configuration therefore also pins attention_config
            # .mla_prefill_backend; TRTLLM_RAGGED is the framework's own
            # highest-priority SM100 backend among the supporting set
            # (interface.py:474-480). get_mla_prefill_backend validates the
            # explicit selection and raises when it is invalid
            # (prefill/selector.py:84-107 @v0.32.0), and the
            # metadata probe below still fails closed on the resolved
            # q_data_type.
            vllm_config.attention_config.mla_prefill_backend = MLAPrefillBackendEnum.TRTLLM_RAGGED
        # Outside the SM100 family the pin has no source proof: vLLM gates
        # FP8 prefill query quantization to that family outright
        # (backend_supports_prefill_query_quantization returns False when
        # not is_device_capability_family(100), mla_attention.py:1891-1892
        # @v0.32.0),
        # and TRTLLM_RAGGED fails its own capability validation
        # (e.g. "compute capability 12.0 not supported" on SM120). Leave the
        # auto selector in charge so the q_data_type probe below raises the
        # accurate classified error for the unsupported fp8 label.

    # backend_supports_prefill_query_quantization() is a zero-argument
    # functools.cache that reads the *current* vllm config on first call
    # (mla_attention.py:1877-1898 @v0.32.0). Serving processes hold one config, but
    # collector workers build many per process, so a value cached from a
    # previous case (e.g. bf16-prefill, FLASH_ATTN) would silently redirect
    # this case's q_data_type decision. Clear it so vLLM re-decides from this
    # case's config, exactly as a fresh serving process would.
    from vllm.model_executor.layers.attention.mla_attention import (
        backend_supports_prefill_query_quantization,
    )

    backend_supports_prefill_query_quantization.cache_clear()

    # Override just the layer-local dimensions we sweep in the collector.
    hf_config = vllm_config.model_config.hf_text_config
    hf_config.num_hidden_layers = 1
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_heads

    # Create topk_indices_buffer for DSA
    topk_indices_buffer = None
    if is_dsa and hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(
            max_tokens,
            hf_config.index_topk,
            dtype=torch.int32,
            device=device,
        )

    # Build the attention module inside set_current_vllm_config() context.
    # FP8 quantized Linear layers (QuantFP8 / CustomOp) call
    # get_current_vllm_config() during __init__, so the config must be set.
    # set_default_torch_dtype is required because MLAAttention.__init__
    # calls torch.get_default_dtype() to select the attention backend
    # (MLA backends only support bfloat16, not float32).
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        if attn_type == "dsa":
            # GENERIC serving-parity construction (owner rule
            # config in -> the module SERVING would build comes out; never a
            # per-model class import — that is how the prelude-fusion drift
            # was born). Resolve the model class through the framework's own
            # registry (exactly serving's routing, registry.resolve_model_cls),
            # instantiate the WHOLE model on the meta device (free), and
            # extract layer 0's attention submodule. Attention layers register
            # themselves into static_forward_context during init, which is
            # what _create_kv_cache_and_metadata reads — same as serving.
            from vllm.model_executor.models.registry import ModelRegistry

            model_cls, _resolved = ModelRegistry.resolve_model_cls(hf_config.architectures, vllm_config.model_config)
            with torch.device("meta"):
                _meta_model = model_cls(vllm_config=vllm_config, prefix="")
            attn_module = None
            for _n, _m in _meta_model.named_modules():
                if _n.endswith("layers.0.self_attn"):
                    attn_module = _m
                    break
            if attn_module is None:  # no fallback: raise, never substitute
                raise RuntimeError(
                    f"framework model {model_cls.__name__} exposes no "
                    "layers.0.self_attn — extend the extraction, do not pin a class"
                )
            # Every reader of the shared top-k buffer (this layer, its indexer and
            # SparseAttnIndexer; the impl reads it through the indexer) takes the
            # collector's buffer of the same shape. DeepseekV32Model allocates its
            # own with an explicit device, so that one is real memory even under
            # the meta device, and the index group built over it keeps it alive
            # unread (models/deepseek_v32/nvidia/model.py:188-201,
            # sparse_mla_attention.py:616-620,672-686 @v0.32.0).
            for _n, _m in attn_module.named_modules():
                if getattr(_m, "topk_indices_buffer", None) is not None:
                    _m.topk_indices_buffer = topk_indices_buffer
            if getattr(attn_module, "topk_indices_buffer", None) is not None:
                attn_module.topk_indices_buffer = topk_indices_buffer
        else:
            attn_module = DeepseekV2MLAAttention(
                vllm_config=vllm_config,
                config=hf_config,
                hidden_size=hf_config.hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=hf_config.qk_nope_head_dim,
                qk_rope_head_dim=hf_config.qk_rope_head_dim,
                v_head_dim=hf_config.v_head_dim,
                q_lora_rank=hf_config.q_lora_rank if hasattr(hf_config, "q_lora_rank") else None,
                kv_lora_rank=hf_config.kv_lora_rank,
                max_position_embeddings=hf_config.max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix="model.layers.0.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )

    # Serialized block-scaled FP8 creates weight params on meta device;
    # to_empty() also discards already-initialized buffers. Preserve their
    # constructor values, including nonpersistent RoPE tables, across it. On
    # the serving-parity path above the whole module is meta, so the RoPE
    # tables and KV-quant scales are rebuilt from their owners' constructor
    # recipe instead of preserved.
    attn_module = _move_module_preserving_buffers(attn_module, device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # FP8 weights use finite, diverse values; other dtypes initially retain the
    # previous policy. NVFP4 below then populates its packed weights and scales
    # through the native quantizer. Buffers (including RoPE) are untouched.
    # These remain synthetic inputs, not a claim of
    # checkpoint-activation timing parity; CUDA RNG is deliberately avoided
    # (vllm-project/vllm#39371).
    _initialize_synthetic_parameters(attn_module)
    if gemm_type == "nvfp4":
        _initialize_nvfp4_parameters(attn_module)

    return attn_module, vllm_config


def _process_module_weights(attn_module, vllm_config, device):
    """Process weights after loading, mimicking vLLM's model loader.

    This must be called after module construction to:
      1. Run FP8 quantization on linear layer weights.
      2. Create W_UK_T and W_UV matrices in MLAAttention that are
         required for the forward pass.
    """
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        # 1. Process quantized linear layers (FP8 weight conversion).
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)

        # 2. Process MLAAttention layers (creates W_UK_T, W_UV).
        for _, module in attn_module.named_modules():
            if isinstance(module, MLAAttention):
                module.process_weights_after_loading(vllm_config.model_config.dtype)


def _serving_autotuned_linears(attn_module) -> list:
    """The module's linears whose FlashInfer GEMM serving runs on an autotuned tactic.

    Serving's warm-up autotunes the FlashInfer ops it reaches on SM90 and newer
    (model_executor/warmup/kernel_warmup.py:261-268; enable_flashinfer_autotune is on
    from O1) in a dummy run of max_num_batched_tokens tokens over at most max_num_seqs
    requests (:334-381; v1/worker/gpu/model_runner.py:841-871). Every default bound
    gives that batch more tokens than requests (engine/arg_utils.py:2923-2952), so
    MLA attention runs it as prefill and the dense decode op (flashinfer
    trtllm_batch_decode_mla, tuner-owned in flashinfer 0.7.0.post1 mla/_core.py:
    4293-4298) is never tuned; this module's MLA prefill kernels are not tuner ops.
    That leaves the NVFP4 linears' fp4_gemm, which serving skips when any linear runs
    the SM100/SM103 cute-dsl kernel (_flashinfer_autotune_skip_ops, :295-311): only
    FlashInferCutlassNvFp4LinearKernel (mm_fp4 "cutlass"), vLLM's NVFP4 linear on
    SM120 (kernels/linear/__init__.py:574-591; nvfp4/flashinfer.py:119-126,194-207),
    is tuned. All vLLM citations @v0.32.0.
    """
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer

    if not (has_flashinfer() and current_platform.has_device_capability(90)):
        return []
    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
    )

    kernels = [
        (module, getattr(getattr(module, holder, None), "kernel", None))
        for _, module in attn_module.named_modules()
        for holder in ("quant_method", "scheme")
    ]
    if any(isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel) for _, kernel in kernels):
        return []
    return [module for module, kernel in kernels if isinstance(kernel, FlashInferCutlassNvFp4LinearKernel)]


def _run_linears(linears, num_tokens: int, device: str) -> None:
    """One forward of each linear on a num_tokens-row activation: the ops a tuning pass profiles."""
    for layer in linears:
        layer(torch.full((num_tokens, layer.input_size_per_partition), 0.01, dtype=torch.bfloat16, device=device))


def _create_context_kv_inputs(batch_spec: BatchSpec, kv_lora_rank: int, qk_rope_head_dim: int, device: str):
    """
    Create cached KV tensors for the tokens that already exist in the paged KV
    cache before the current forward() call.

    Context phase: cache is empty because all tokens are processed in the same
    prefill step.
    Generation phase: cache holds seq_len - 1 historical tokens, and the
    current forward processes exactly 1 new token per request.
    """
    kv_c_contexts = []
    k_pe_contexts = []
    for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens, strict=True):
        context_len = max(0, int(seq_len) - int(query_len))
        kv_c_contexts.append(torch.full((context_len, kv_lora_rank), 0.01, dtype=torch.bfloat16, device=device))
        k_pe_contexts.append(torch.full((context_len, 1, qk_rope_head_dim), 0.01, dtype=torch.bfloat16, device=device))
    return kv_c_contexts, k_pe_contexts


def _populate_indexer_kv_cache(
    indexer_kv_cache: torch.Tensor,
    common_attn_metadata,
    context_lens: list[int],
    indexer,
) -> None:
    """Populate historical keys through the serving indexer's cache producer."""
    from vllm import _custom_ops as ops

    block_table = common_attn_metadata.block_table_tensor
    block_size = indexer_kv_cache.shape[1]
    device = indexer_kv_cache.device
    head_dim = indexer.head_dim
    quant_block_size = indexer.quant_block_size
    scale_fmt = indexer.scale_fmt
    expected_entry_dim = head_dim + head_dim // quant_block_size * 4
    if indexer_kv_cache.dtype != torch.uint8 or indexer_kv_cache.shape[2] != expected_entry_dim:
        raise ValueError("Indexer cache does not match the native FP8 values and FP32 scales layout")

    # Generate on CPU: CUDA RNG can be unavailable after module construction.
    # Bound temporary storage independently of the length of a cached prefix.
    generator = torch.Generator(device="cpu").manual_seed(1234)
    chunk_tokens = 8192

    for i, context_len in enumerate(context_lens):
        for start in range(0, context_len, chunk_tokens):
            end = min(start + chunk_tokens, context_len)
            keys = torch.randn((end - start, head_dim), generator=generator, device="cpu", dtype=torch.float32).to(
                device=device, dtype=torch.bfloat16
            )
            token_offsets = torch.arange(start, end, dtype=torch.long, device=device)
            block_ids = block_table[i, token_offsets // block_size]
            slots = block_ids.to(torch.long) * block_size + token_offsets % block_size
            # This writes both FP8 values and their valid per-group scales;
            # values/scales are block-planar, not token-interleaved. It is the
            # producer of serving's unfused path (sparse_attn_indexer.py:523-541);
            # the fused DSV3.2 path writes the same layout for the indexer cache
            # it reads back (fused_norm_rope -> _fp8_quant_and_cache_write,
            # models/deepseek_v32/common/kernels.py:77-117,509-564); layout and
            # ue8m0 rounding are in csrc/cache_kernels.cu:624-700,1917-1940. This
            # fill precedes the timed region. All @v0.32.0.
            # The physical slot is the page-table block ID plus token offset.
            ops.indexer_k_quant_and_cache(keys, indexer_kv_cache, slots, quant_block_size, scale_fmt)


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


# Per-case phase timings. run_mla_module prints one "[... module timing]" line
# per case so a run log shows where the module collectors spend their time
# (framework setup per case vs. the timed forwards); _create_kv_cache_and_metadata
# records the indexer-cache fill here because its return tuple is a contract.
_PHASE_TIMINGS: dict[str, float] = {}


def _serving_kv_block_size(vllm_config, attn_layer) -> int:
    """The KV page serving runs the layer's selected backend with.

    After loading the model, serving sizes the cache-manager block with the platform's
    own update_block_size_for_backend (v1/executor/uniproc_executor.py:75,
    multiproc_executor.py:701): without --block-size, the smallest size every non-SSM
    attention backend of the model accepts (platforms/interface.py:750-781 ->
    _find_non_ssm_backends :679-697, _preferred_block_size_for_backends :700-748). The
    kernel page is select_common_block_size of that block, the layer's backend and its
    spec (v1/worker/utils.py:515-561, called per KV-cache group at :672-674). Dense MLA
    has one backend, so the block is its get_preferred_block_size(16)
    (v1/attention/backend.py:152-160). DSA adds the indexer cache, whose CUDA backend
    accepts only 64 (v1/attention/backends/mla/indexer.py:284-285), so
    FLASHINFER_MLA_SPARSE (MultipleOf(32)) runs at 64. All @v0.32.0.
    """
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import select_common_block_size

    current_platform.update_block_size_for_backend(vllm_config)
    return int(
        select_common_block_size(
            vllm_config.cache_config.block_size,
            [attn_layer.get_attn_backend()],
            [attn_layer.get_kv_cache_spec(vllm_config)],
        )
    )


def _create_kv_cache_and_metadata(
    vllm_config,
    attn_type: str,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    compute_dtype: str = "bfloat16",
    device: str = "cuda:0",
    indexer=None,
):
    """Create KV cache and attention metadata for benchmarking."""
    _PHASE_TIMINGS["indexer_fill"] = 0.0
    hf_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    _attn_layer = vllm_config.compilation_config.static_forward_context["model.layers.0.self_attn.attn"]
    block_size = _serving_kv_block_size(vllm_config, _attn_layer)
    vllm_config.cache_config.block_size = block_size
    is_dsa = attn_type == "dsa"

    if is_dsa and indexer is None:
        raise ValueError("DSA cache population requires the constructed Indexer and its quantization contract")

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(
            seq_lens=[prefix_len + seq_len] * batch_size,
            query_lens=[seq_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    num_kv_cache_blocks = max(
        1 + math.ceil((prefix_len + seq_len + 1) / block_size) * batch_size,
        8192,
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, torch.device(device), arange_block_indices=True
    )

    # TRT-LLM Gen MLA (and cutlass_mla) require the page table's block
    # dimension to be a multiple of ``128 / block_size``; for block_size=64
    # that means at least two blocks even when a request only fills one.
    # Short sequences (seq_len <= block_size) otherwise produce block_num=1
    # and the kernel raises ``Expected block_num % (128 / block_size) == 0``.
    # Pad with zero-indexed blocks (they are not read for absent tokens).
    required_divisor = max(1, 128 // block_size)
    current_block_num = common_attn_metadata.block_table_tensor.shape[1]
    if current_block_num % required_divisor != 0:
        padded_block_num = ((current_block_num + required_divisor - 1) // required_divisor) * required_divisor
        padding = torch.zeros(
            (common_attn_metadata.block_table_tensor.shape[0], padded_block_num - current_block_num),
            dtype=common_attn_metadata.block_table_tensor.dtype,
            device=common_attn_metadata.block_table_tensor.device,
        )
        common_attn_metadata.block_table_tensor = torch.cat([common_attn_metadata.block_table_tensor, padding], dim=1)

    # Use the cache format chosen by the concrete production layer. In 0.24.0
    # this matters on both SM10x (FlashMLA vs FlashInfer sparse) and SM120,
    # where sparse MLA canonicalizes auto/fp8 to the packed fp8_ds_mla layout.
    attn_layer_name = "model.layers.0.self_attn.attn"
    attn_layer = vllm_config.compilation_config.static_forward_context[attn_layer_name]
    backend_cls = attn_layer.get_attn_backend()
    kv_cache_spec = attn_layer.get_kv_cache_spec(vllm_config)
    cache_dtype = kv_cache_spec.dtype
    kv_cache_dtype_str = kv_cache_spec.cache_dtype_str

    # Populate KV cache with the tokens that exist before this forward.
    kv_c_contexts, k_pe_contexts = _create_context_kv_inputs(
        batch_spec=batch_spec,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        device=device,
    )

    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=kv_cache_spec.head_size,
        dtype=cache_dtype,
        device=torch.device(device),
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype_str,
        scale=attn_layer._k_scale,
    )

    builder_cls = backend_cls.get_builder_cls()

    layer_names = [attn_layer_name]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, torch.device(device))

    # Fail closed on the framework's own prefill query dtype decision:
    # vLLM silently falls back to the model dtype when FP8 prefill query
    # quantization is requested but unsupported (fp8 KV cache + SM100 family
    # + FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA prefill backend required —
    # determine_prefill_query_data_type, mla_attention.py:2320-2360 @v0.32.0).
    # A silently-bf16 run must not be recorded as an mla_dtype=fp8 row.
    if is_context:
        from vllm.platforms import current_platform

        model_dtype = vllm_config.model_config.dtype
        expected_q_dtype = current_platform.fp8_dtype() if compute_dtype == "fp8" else model_dtype
        actual_q_dtype = getattr(builder, "q_data_type", model_dtype)
        if actual_q_dtype != expected_q_dtype:
            raise RuntimeError(
                f"vLLM selected prefill query dtype {actual_q_dtype} but the case is labeled "
                f"compute_dtype={compute_dtype!r}; refusing to record a mislabeled row "
                "(see determine_prefill_query_data_type for the support conditions)"
            )

    attn_metadata = builder.build(
        common_prefix_len=prefix_len,
        common_attn_metadata=common_attn_metadata,
    )

    # For DSA, the Indexer has its own KV cache and metadata builder.
    indexer_kv_cache = None
    indexer_metadata = None
    if is_dsa:
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_layer = vllm_config.compilation_config.static_forward_context[indexer_layer_name]
        indexer_spec = indexer_layer.get_kv_cache_spec(vllm_config)
        indexer_kv_cache = torch.zeros(
            num_kv_cache_blocks,
            block_size,
            indexer_spec.head_size,
            dtype=indexer_spec.dtype,
            device=device,
        )
        indexer_builder_cls = indexer_layer.get_attn_backend().get_builder_cls()
        # The indexer builder demands block_table_width (mla/indexer.py:988,1000);
        # mirror serving's population exactly (v1/worker/utils.py:308-316)
        # using vllm's own width helper (v1/worker/block_table.py:29-41), all @v0.32.0.
        builder_kwargs = {}
        if getattr(indexer_builder_cls, "requires_block_table_width", False):
            from vllm.v1.worker.block_table import get_block_table_width

            max_num_blocks = indexer_spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len)
            builder_kwargs["block_table_width"] = get_block_table_width(max_num_blocks, indexer_spec.block_size)
            # Serving allocates the block table AT block_table_width columns
            # (v1/worker/block_table.py), so builders may assume
            # table.shape[1] == width. The family-100 decode path acts on
            # that assumption: it copies the metadata's table into
            # expanded_block_table_buffer sized by the width
            # (v1/attention/backends/mla/indexer.py:1091-1095,1197-1248) and
            # crashes on our ceil(seq/block)-column table ("expanded size
            # (64) must match existing size (32)"). Zero-pad to the width —
            # padding blocks are never dereferenced for absent tokens. SM90
            # paths never read the expanded buffer, which is why this stayed
            # latent on H20 (found on B300).
            _width = builder_kwargs["block_table_width"]
            _bt = common_attn_metadata.block_table_tensor
            if _bt.shape[1] < _width:
                _pad = torch.zeros(
                    (_bt.shape[0], _width - _bt.shape[1]),
                    dtype=_bt.dtype,
                    device=_bt.device,
                )
                common_attn_metadata.block_table_tensor = torch.cat([_bt, _pad], dim=1)
        indexer_builder = indexer_builder_cls(
            indexer_spec, [indexer_layer_name], vllm_config, torch.device(device), **builder_kwargs
        )
        indexer_metadata = indexer_builder.build(
            common_prefix_len=prefix_len,
            common_attn_metadata=common_attn_metadata,
        )
        if os.environ.get("AIS_DEBUG_IDXMETA") or os.environ.get(
            "AIC_DEBUG_IDXMETA"
        ):  # diagnostic dump, no behavior change
            _pre = getattr(indexer_metadata, "prefill", None)
            print(
                f"[idxmeta] type={type(indexer_metadata).__name__} "
                f"prefill={type(_pre).__name__ if _pre is not None else None} "
                f"decode={getattr(indexer_metadata, 'decode', None) is not None}"
            )
            if _pre is not None:
                _ch = getattr(_pre, "chunks", None)
                print(f"[idxmeta] chunks={None if _ch is None else len(_ch)}")
                for _c in (_ch or [])[:1]:
                    for _f in dir(_c):
                        _v = getattr(_c, _f, None)
                        if not _f.startswith("_") and isinstance(_v, (int, bool, float)):
                            print(f"[idxmeta]   chunk.{_f} = {_v}")
        _fill_start = time.perf_counter()
        _populate_indexer_kv_cache(
            indexer_kv_cache=indexer_kv_cache,
            common_attn_metadata=common_attn_metadata,
            context_lens=[tensor.shape[0] for tensor in kv_c_contexts],
            indexer=indexer,
        )
        _PHASE_TIMINGS["indexer_fill"] = time.perf_counter() - _fill_start

    return kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


def _series_points(seq_len, prefix_len: int = 0) -> list[tuple[int, int]]:
    """The (seq_len, prefix_len) points a case measures on one module, largest total
    context first: the module and every per-point buffer are sized by the largest
    point, so a capacity failure shows up on the first point.

    ``seq_len`` is one point (int), the lengths of a series behind one ``prefix_len``,
    or the series' [seq_len, prefix_len] points (a DSA context row of the plan).
    """
    if isinstance(seq_len, bool) or isinstance(prefix_len, bool):
        raise TypeError("seq_len and prefix_len must be ints or a list of points")
    values = [seq_len] if isinstance(seq_len, int) else list(seq_len)
    if not values:
        raise ValueError("a module series needs at least one point")
    points = {
        (int(value[0]), int(value[1])) if isinstance(value, (list, tuple)) else (int(value), int(prefix_len))
        for value in values
    }
    return sorted(points, key=lambda sp: (sp[0] + sp[1], sp[0]), reverse=True)


def _is_fatal_cuda_error(exc: BaseException) -> bool:
    """A sticky CUDA failure: nothing measured on this context afterwards is trustworthy."""
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        return True
    text = str(exc).lower()
    markers = ("cuda error", "illegal memory access", "cudaerrorillegaladdress", "device-side assert")
    return any(marker in text for marker in markers)


def _serving_cudagraph_ceiling() -> int:
    """The largest step, in tokens, serving replays as a CUDA graph; larger steps run eagerly.

    Serving's default max_cudagraph_capture_size is min(2 * max_num_seqs, 1024 on the
    SM100 family else 512, max_num_batched_tokens) (config/vllm.py:2341-2446), and the
    dispatcher runs any step above it with no graph (v1/worker/gpu/cudagraph_utils.py:
    518-548). Serving's scheduler defaults on the GPUs this op runs on (max_num_seqs 1024 or
    256, max_num_batched_tokens 2048 or more; engine/arg_utils.py:2923-2952) never bind
    below the platform constant. All @v0.32.0.
    """
    from vllm.platforms import current_platform

    return 1024 if current_platform.is_device_capability_family(100) else 512


@contextmanager
def _dsa_serving_capture_parity(attn_module, *, serving_captures_forward: bool):
    """Make DeepseekV32Attention's capture-dependent choices the ones serving makes for this step.

    Serving runs the DSA architectures on breakable CUDA graphs with torch.compile off
    (DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES and _maybe_enable_breakable_cudagraph,
    config/vllm.py:74-102,812-840): a prefill step of up to _serving_cudagraph_ceiling()
    tokens replays the forward as graph segments around eager breaks, a larger one runs
    eagerly, and a uniform decode step replays one FULL graph (breakable_cudagraph.py:318-343,
    v1/worker/gpu/cudagraph_utils.py:189-194,571-576). _sparse_indexer_and_attn is such a
    break (@eager_break_during_capture, models/deepseek_v32/attention.py:465-466): add_eager
    ends the segment before running it (breakable_cudagraph.py:67-120,198-211), so serving
    never runs it on a capturing stream outside a FULL graph. This collector times every
    step as one CUDA graph, so two choices vLLM keys on torch.cuda.is_current_stream_capturing()
    would see a capture serving does not have:

    1. _should_prepare_mqa_query (attention.py:286-296, applied at :336-338,422-425,446): a
       captured forward always runs the W_UK BMM and, on an FP8-query backend, the FP8
       MQA-query quantization; an eager forward skips both on the dense/masked-MHA prefill
       path. Here the capture term is whether serving captures a step of this size.
    2. sparse_attn_indexer's dense-MHA early return (sparse_attn_indexer.py:542-558): on a
       pure prefill whose MLA metadata takes dense MHA, serving's eager break returns before
       any indexer kernel (no cp_gather, MQA logits or top-k; skip_k_cache_insert is set on
       this path, attention.py:501). Here the same predicate skips the indexer call through
       the layer's own skip_topk, which _sparse_indexer_and_attn reads only at attention.py:481.

    Both are serving's own conditions with only the capture state replaced, all @v0.32.0;
    re-check them at every pin bump. Raises when the pinned class no longer has these hooks.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    cls = type(attn_module)
    sparse_indexer_and_attn = getattr(cls, "_sparse_indexer_and_attn", None)
    hooks = ("_should_prepare_mqa_query", "_use_sparse_mha")
    if not (
        callable(sparse_indexer_and_attn)
        and all(callable(getattr(cls, hook, None)) for hook in hooks)
        and hasattr(attn_module, "skip_topk")
        and hasattr(attn_module, "_dense_mha_metadata_layer_name")
    ):
        raise RuntimeError(
            f"{cls.__name__} lacks the capture-dependent DSA choices this collector pins to serving "
            "(_should_prepare_mqa_query, _sparse_indexer_and_attn, skip_topk); re-derive them from the pin"
        )

    def should_prepare_mqa_query(attn_metadata, cudagraph_runtime_mode):
        return (
            attn_metadata is None
            or not attn_module._use_sparse_mha(attn_metadata)
            or cudagraph_runtime_mode == CUDAGraphMode.FULL
            or serving_captures_forward
        )

    def serving_skips_indexer_scoring() -> bool:
        forward_context = get_forward_context()
        layer_name = attn_module._dense_mha_metadata_layer_name
        attn_metadata = forward_context.attn_metadata
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL or not layer_name:
            return False
        mla_metadata = attn_metadata.get(layer_name) if isinstance(attn_metadata, dict) else None
        prefill = getattr(mla_metadata, "prefill", None)
        return bool(getattr(prefill, "use_dense_mha", False)) and getattr(mla_metadata, "num_decode_tokens", -1) == 0

    def sparse_indexer_and_attn_as_serving(*args, **kwargs):
        if attn_module.skip_topk or not serving_skips_indexer_scoring():
            return sparse_indexer_and_attn(attn_module, *args, **kwargs)
        attn_module.skip_topk = True
        try:
            return sparse_indexer_and_attn(attn_module, *args, **kwargs)
        finally:
            attn_module.skip_topk = False

    attn_module._should_prepare_mqa_query = should_prepare_mqa_query
    attn_module._sparse_indexer_and_attn = sparse_indexer_and_attn_as_serving
    try:
        yield
    finally:
        del attn_module._should_prepare_mqa_query
        del attn_module._sparse_indexer_and_attn


@with_exit_stack
def run_mla_module(
    exit_stack,
    seq_len: int | list,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    prefix_len: int = 0,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run one MLA / DSA module series: one module, one measured point per (``seq_len``, prefix).

    ``seq_len`` is the query length (context) or the KV length (generation) of a
    point, the list of them that share (batch, heads, dtypes, gemm, model) behind
    one cached ``prefix_len``, or a DSA context series' [seq_len, prefix_len]
    points (see _series_points). The module (vLLM config, synthetic/quantized
    weights, rope shims) is built once for the largest point; the KV cache, the indexer cache and the
    attention metadata are still built per point exactly as a single-point run
    does, so every row keeps its serving-parity construction. A point that fails
    for a non-fatal reason (OOM, a kernel limit) is recorded and the remaining
    points are still measured; the failure is raised after the series so
    collect.py classifies the case (the point's own exception when it is the
    only one). A sticky CUDA failure aborts the series at once.

    Returns the full-indexer latencies of the measured points, largest first.
    """
    if attn_type not in {"mla", "dsa"}:
        raise ValueError(f"unsupported vLLM attention type: {attn_type!r}")
    if kv_cache_dtype not in {"bfloat16", "fp8"}:
        raise ValueError(f"unsupported vLLM MLA KV-cache dtype: {kv_cache_dtype!r}")
    if compute_dtype not in {"bfloat16", "fp8"}:
        raise ValueError(f"unsupported vLLM MLA query compute dtype: {compute_dtype!r}")
    if compute_dtype == "fp8" and kv_cache_dtype != "fp8":
        raise ValueError(
            "vLLM FP8 MLA prefill query compute requires an FP8 KV cache "
            "(determine_prefill_query_data_type, mla_attention.py:2329-2333 "
            f"@0.32.0); got kv_cache_dtype={kv_cache_dtype!r}"
        )
    is_context = "context" in perf_filename
    points = _series_points(seq_len, prefix_len if is_context else 0)

    _t_series = _t_last = time.perf_counter()
    _setup_phase: dict[str, float] = {}

    def _lap(phases: dict[str, float], name: str) -> None:
        nonlocal _t_last
        now = time.perf_counter()
        phases[name] = now - _t_last
        _t_last = now

    setup_distributed(device)
    torch.cuda.set_device(device)
    enable_engine_fused_ops()

    # Registered before the vLLM contexts below, so they restore their globals
    # before workspace tensors are released, on success and every failure path.
    # On success this also runs after the function's local module/KV references
    # have gone out of scope instead of emptying the cache while they are live.
    exit_stack.callback(_cleanup)

    # DSA's sparse_attn_indexer requires a WorkspaceManager.
    init_workspace_manager(torch.device(device))

    use_fp8_kv_cache = kv_cache_dtype == "fp8"
    use_prefill_fp8 = compute_dtype == "fp8"
    seq_lens = sorted({s for s, _ in points}, reverse=True)
    prefix_lens = sorted({p for _, p in points}, reverse=True)
    prefix_desc = prefix_lens[0] if len(prefix_lens) == 1 else prefix_lens
    phase = "context" if is_context else "generation"
    variant = attn_type.upper()
    # Generation points whose KV cache cannot fit the device are refused below, before
    # anything is allocated for them; the module is sized for the points that fit.
    does_not_fit: dict[tuple[int, int], str] = {}
    if not is_context:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        for point in points:
            reason = _kv_does_not_fit(batch_size * point[0], kv_cache_dtype, total_memory)
            if reason is not None:
                does_not_fit[point] = reason
        if len(does_not_fit) == len(points):
            # Not even the smallest point fits: refuse the case with its reason.
            raise RuntimeError(does_not_fit[points[-1]])
    fitting = [point for point in points if point not in does_not_fit]
    max_seq_len = max(s for s, _ in fitting)
    max_total_len = max(s + p for s, p in fitting)
    print(
        f"\n[{variant} module] {phase} b={batch_size}, s={seq_lens if len(seq_lens) > 1 else seq_lens[0]}, "
        f"prefix={prefix_desc}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention module, sized for the series' largest point and at the
    #    model's max_position_embeddings as max_model_len — serving's default, and
    #    independent of the batch: the DSA indexer's decode logits
    #    [rows, max_model_len] (sparse_attn_indexer.py:825-833) and its prefill
    #    buffer, max_model_len * 8 * sparse_mla_prefill_workspace_factor (5 up to
    #    256K, else 2; mla/indexer.py:918-929, config/vllm.py:1790-1798), scale
    #    with it, all @v0.32.0, so a series-max size would bend batch scaling
    #    within one table. The other sizes
    #    only reach capacity declarations (max_num_batched_tokens, the top-k index
    #    buffer); the backend selection is dtype/SM driven and every point's KV
    #    cache + metadata is built below for that point's own shape.
    attn_module, vllm_config = _create_attention_module(
        model_path=model_path,
        attn_type=attn_type,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        use_prefill_fp8=use_prefill_fp8,
        max_seq_len=max_total_len,
        max_batch_size=batch_size,
        max_query_len=max_seq_len,
        gemm_type=gemm_type,
        device=device,
        is_context=is_context,
        max_model_len=_model_max_position_embeddings(model_path),
    )
    _lap(_setup_phase, "module")
    if attn_type == "dsa" and use_fp8_kv_cache:
        # DeepseekV32 attention on the dense/masked-MHA prefill path calls
        # `self.rotary_emb(positions, q_pe)[0]` — key=None — when the forward quantized the MQA
        # query to FP8 (`mqa_q.dtype != q_pe.dtype`, models/deepseek_v32/attention.py:556-562
        # @v0.32.0). That happens on the SM100 family with an FP8 KV cache (_fp8_query = FP8 KV
        # and FlashInferMLASparseImpl.supports_quant_query_input, attention.py:238-239 /
        # flashinfer_mla_sparse.py:493) for steps serving captures (see
        # _dsa_serving_capture_parity); an eager step keeps q_pe in bf16 and makes no call. The
        # yarn rope of DeepSeek-V3.2 (DeepseekScalingRotaryEmbedding) rejects key=None —
        # FlashInfer op `key.view(...)` ("'NoneType' object has no attribute 'view'", 3.9k b200
        # + ~3k b300 + ~3k gb200 cases at 0.30) / forward_native `assert key is not None`
        # (deepseek_scaling_rope.py:123; the ed9109ec attempt to run native moved every case
        # onto that assert: gb200 pipeline 72343556), so serving fails those steps too — for
        # THAT class the call's intent, "rope q_pe only", is given to the module: rope the
        # query against a throwaway copy and return (query, None). Deviation: the copy and the
        # rope of the throwaway key are extra work in those rows. GLM-5.x's default rope
        # (RotaryEmbedding) accepts key=None natively and is left on serving's exact call:
        # wrapping it too handed vLLM's own rope kernel a contiguous key with the wrong head
        # stride -> out-of-bounds in-place writes, the GLM-5.2 fp8-KV illegal-address / stall
        # family (see _strided_throwaway_copy). Drop the shim once vLLM's call or
        # DeepseekScalingRotaryEmbedding accepts key=None.
        _patched, _left = _install_rope_missing_key_shims(attn_module)
        if _patched:
            print(
                f"[vllm-mla-module] dsa fp8: rope shim on {sorted(set(_patched))} (reject key=None; "
                f"attention.py:556-562 ropes q_pe alone); on serving's own call: {sorted(set(_left))}"
            )
    # The step size up to which serving replays a DSA forward as a CUDA graph.
    capture_ceiling = _serving_cudagraph_ceiling() if attn_type == "dsa" else None

    # 1b. Process weights (FP8 quantization + create W_UK_T / W_UV for MLA)
    _process_module_weights(attn_module, vllm_config, device)
    _lap(_setup_phase, "weights")
    # 1c. The linears serving runs on FlashInfer-autotuned tactics (SM120 NVFP4 only).
    tuned_linears = _serving_autotuned_linears(attn_module)

    # set_current_vllm_config — needed by quantised layers and RoPE for every
    # forward of the series; entered once, after _cleanup was registered, so it
    # unwinds before the workspace is released.
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    attn_layer_name = "model.layers.0.self_attn.attn"
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    hidden_size = vllm_config.model_config.hf_text_config.hidden_size

    # Index reuse (GLM-5.x index_topk_freq / index_topk_pattern): the full dry run below
    # populates the shared top-k buffer; serving builds its reuse layers without an
    # indexer, keeping the MLA projections and sparse attention while skipping all indexer
    # work, and skip_topk=True on this indexer layer takes exactly those branches
    # (vllm/models/deepseek_v32/attention.py:166-200,226,309,339,427,481 @v0.32.0; the
    # generic DeepseekV2 wrapper keeps the flag on mla_attn, layers/mla.py:109,147,232).
    # Emit both op_name variants for checkpoints declaring reuse; never a model-averaged
    # latency in a full-indexer row.
    hf_text_config = vllm_config.model_config.hf_text_config
    reuse_pattern = getattr(hf_text_config, "index_topk_pattern", None)
    has_reuse = attn_type == "dsa" and (
        "S" in reuse_pattern if reuse_pattern is not None else (getattr(hf_text_config, "index_topk_freq", 1) or 1) > 1
    )
    # DeepseekV32Attention IS the MLA layer and owns skip_topk; the legacy generic
    # DeepseekV2MLAAttention nests it under mla_attn.
    skip_owner = next(
        (c for c in (attn_module, getattr(attn_module, "mla_attn", None)) if c is not None and hasattr(c, "skip_topk")),
        None,
    )
    if has_reuse:
        if skip_owner is None:
            raise RuntimeError("vLLM DSA reuse requires the native skip_topk execution path")
        exit_stack.callback(setattr, skip_owner, "skip_topk", skip_owner.skip_topk)

    # Record architecture to distinguish different DSA models in the perf CSV.
    # perf_database uses this as a dict key when loading data.
    # Aligns with sdk/models.py which uses architectures[0] throughout.
    hf_cfg = vllm_config.model_config.hf_config
    architecture = getattr(hf_cfg, "architectures", [getattr(hf_cfg, "model_type", "unknown")])[0]
    # legacy generic module nests the layer (module.mla_attn.mla_attn);
    # DeepseekV32Attention IS the MLAAttention layer itself
    mla_layer = attn_module if not hasattr(attn_module, "mla_attn") else attn_module.mla_attn.mla_attn
    actual_kv_cache_dtype = "fp8" if mla_layer.kv_cache_dtype.startswith("fp8") else "bfloat16"

    def _run_point(point_stack, seq_len: int, prefix_len: int, point_phase: dict[str, float]) -> float:
        # 2. Create KV cache + metadata for this point
        kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata = (
            _create_kv_cache_and_metadata(
                vllm_config=vllm_config,
                attn_type=attn_type,
                batch_size=batch_size,
                seq_len=seq_len,
                is_context=is_context,
                prefix_len=prefix_len,
                compute_dtype=compute_dtype,
                device=device,
                indexer=getattr(attn_module, "indexer", None),
            )
        )
        _lap(point_phase, "kv_metadata")

        # 2b. Bind the KV cache to the attention layer, as MLAAttention.bind_kv_cache does
        # for serving's [B, 1, N, C] view (mla_attention.py:734-736 @v0.32.0); this cache is
        # allocated as [B, N, C] already.
        forward_ctx[attn_layer_name].kv_cache = kv_cache

        # For DSA, also bind the indexer's KV cache.
        if indexer_kv_cache is not None and indexer_layer_name in forward_ctx:
            forward_ctx[indexer_layer_name].kv_cache = indexer_kv_cache

        # 3. Input tensors
        if is_context:
            num_tokens = seq_len * batch_size
            positions = (
                torch.arange(prefix_len, prefix_len + seq_len, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
                .contiguous()
            )
        else:
            num_tokens = batch_size
            positions = torch.full(
                (batch_size,),
                seq_len - 1,
                device=device,
                dtype=torch.long,
            )

        hidden_states = torch.full(
            (num_tokens, hidden_size),
            0.01,
            dtype=torch.bfloat16,
            device=device,
        )

        # 4. Dry run
        #    set_forward_context — provides attn_metadata + kv_cache to the
        #    MLAAttention.forward() path (it calls get_forward_context()).
        # FIXME(kernel-limit): on SM100, FlashInfer's trtllm-gen sparse-MLA decode
        # ships kernels only for tileSizeQ >= 8. DSA cases whose per-rank head
        # count is 1/2/4 (tileSizeQ = heads for q_len 1) raise "Missing TRTLLM-GEN
        # kernel (decode)" (flashinfer csrc/trtllm_fmha_kernel_launcher.cu:382 at
        # 0.7.0.post1, vLLM 0.32.0's pin; measured at 0.6.12) whenever vLLM selects
        # FLASHINFER_MLA_SPARSE — always for FP8 KV, and for BF16 KV at heads <= 16
        # (platforms/cuda.py:96-118 @v0.32.0). Heads 8..128 pass; boundary measured
        # on B200. Serving fails identically; collector/skips.py leaves them out.
        #
        # FIXME(kernel-limit): on SM120 and SM89, vLLM's dense-MLA decode backend
        # is TRITON_MLA (SM120: platforms/cuda.py:132-136; SM89: the else-branch
        # priority list at cuda.py:137-153, where FLASH_ATTN_MLA/FLASHMLA/
        # FLASHINFER_MLA all reject major != 9/10 and TRITON_MLA is the only
        # eligible backend; @v0.32.0). Two measured limits (RTX PRO 6000
        # Blackwell; serving fails identically; the kernel and its launch
        # are unchanged at v0.32.0, re-verify on the next vLLM bump):
        # 1. FP8 KV + q-heads >= 2 routes to the grouped decode kernel, which
        #    requests 102400B shared memory vs SM120's 101376B limit ->
        #    OutOfResources (heads == 1 takes the non-grouped kernel and
        #    passes; bf16 KV passes). vLLM's overflow guard only drops
        #    num_stages at BLOCK_DMODEL >= 1024, which the MLA Lk=576 path
        #    (BLOCK_DMODEL=512) never reaches (triton_decode_attention.py:
        #    489-492,519-531 @v0.32.0). Upstream fix: vllm#46728 (num_stages=1
        #    fallback for exactly this tile, fixes vllm#46721), not in v0.32.0.
        # 2. Decode batches whose total cached tokens exceed ~2^31/576 raise
        #    a deterministic illegal memory access (reproduced in isolation
        #    on a clean GPU): largest passing batch*seq = 2.10M tokens,
        #    smallest failing = 4.19M, bracketing 2^31 / 576 elements = 3.73M
        #    — consistent with int32 offset overflow in the Triton decode
        #    kernel's KV indexing. Affects both KV dtypes (fp8 only at
        #    heads == 1, where limit 1 does not fire first). Same family on
        #    SM89 (L40S): 33/400 sampled generation cases, smallest failing
        #    batch*seq again 4.19M tokens (bf16 KV; fp8-KV combos are not
        #    declared below SM90).
        # FIXME(re-verify): an earlier vLLM pin failed every DSA case on SM120: the
        # CUDA sparse attention indexer hard-requires DeepGEMM, whose fp8 MQA-logits
        # kernels then shipped for SM90/SM100 only and asserted "Unsupported
        # architecture" (deepgemm csrc/apis/attention.hpp:184); cases with fp8_block
        # linears died even earlier in DeepGEMM's scale-factor layout transform
        # ("Unknown SF transformation", layout.hpp:59). That premise no longer holds
        # in source: vLLM 0.32.0 (as 0.30.0) builds the vllm-project DeepGEMM fork,
        # "upstream 2.8.0 plus the SM120 port", for 12.0a/12.0f
        # (cmake/external_projects/deepgemm.cmake:32-37,80-85), sizes its paged MQA
        # logits pages for SM120 (utils/deep_gemm.py:33-36), and serves SM120 DSA
        # on FLASHINFER_MLA_SPARSE_SM120 (platforms/cuda.py:132-136). A targeted
        # SM120 run decides whether these cases now pass.
        #
        attn_metadata_dict = {attn_layer_name: attn_metadata}
        if indexer_metadata is not None:
            attn_metadata_dict[indexer_layer_name] = indexer_metadata
        # DeepseekV32Attention reads per-layer slot mappings from
        # forward_context.slot_mapping (attention.py:323-331 @v0.32.0 — a dict
        # keyed by layer name); mirror serving's population with the SAME slot
        # tensor the metadata was built from.
        _slot = getattr(common_attn_metadata, "slot_mapping", None)
        _slot_map = dict.fromkeys(attn_metadata_dict, _slot) if _slot is not None else None
        point_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config, slot_mapping=_slot_map))
        if capture_ceiling is not None:
            # The dry run and the timed graph make the capture-dependent choices serving
            # makes for a step of num_tokens tokens: graph-replayed up to the ceiling,
            # eager above it (see _dsa_serving_capture_parity).
            point_stack.enter_context(
                _dsa_serving_capture_parity(attn_module, serving_captures_forward=num_tokens <= capture_ceiling)
            )
        try:
            with torch.inference_mode():
                _fwd_args = (
                    (positions, hidden_states)
                    if type(attn_module).__name__ == "DeepseekV32Attention"
                    else (positions, hidden_states, None)
                )
                attn_module.forward(*_fwd_args)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  Dry run OOM: {e}")
            # Let collect.py record the capacity failure. Returning normally would
            # mark a task done even though it emitted no performance row.
            raise
        except Exception as e:
            print(f"  Dry run failed: {e}")
            traceback.print_exc()
            # Propagate to collect.py's worker so the failure is recorded in the
            # error queue. Swallowing here lets a whole op complete with zero
            # rows while the pipeline reports success.
            raise

        _lap(point_phase, "dry_run")

        # 5. Benchmark
        def kernel_func():
            attn_module.forward(*_fwd_args)

        def measure_and_log(skip_indexer):
            if has_reuse:
                skip_owner.skip_topk = skip_indexer

            # Graph timing for every MLA/DSA case. DSA context used to run eagerly to dodge a
            # private-pool retention (~18 GiB flashmla-sparse scratch per capture, saturating
            # at ~146 GiB per worker); that bought small cases a ~0.6 ms launch floor — 0.30
            # rows at isl <= 1024 measured 1.6-8x the graph-timed values of the same cells
            # (b200/l40s vs #404's B200 tables). benchmark_with_power
            # (explicit_graph_cleanup) resets each graph and the ExitStack releases vLLM's
            # workspace after the forward contexts unwind. FIXME(re-verify): confirm on 0.32
            # with a small->large DSA context sweep that the retention is gone before a campaign.
            # Deviation (DSA): serving runs a prefill step above _serving_cudagraph_ceiling()
            # eagerly and _sparse_indexer_and_attn as an eager break below it, so those
            # rows exclude the host launch gaps serving pays there; the kernels match
            # (_dsa_serving_capture_parity). Retired by the pending measurement-method decision.
            # The linears serving autotunes run on the tactic tuned at the bucket serving maps
            # this token count to (mm_fp4, flashinfer gemm/gemm_base.py:7887-7915 @0.7.0.post1);
            # the tuning pass runs only those linears, so no other FlashInfer op is tuned.
            tune_tokens = flashinfer_tuning_bucket(num_tokens) if tuned_linears else None
            autotuned = (
                nullcontext()
                if tune_tokens is None
                else flashinfer_autotuned(
                    tune_tokens,
                    partial(_run_linears, tuned_linears, num_tokens, device),
                    label="FlashInferCutlassNvFp4LinearKernel",
                )
            )
            with (
                autotuned,
                benchmark_with_power(
                    device=torch.device(device),
                    kernel_func=kernel_func,
                    num_warmups=warming_up,
                    # The point's dry run already executed this forward; three eager
                    # passes cover lazily initialised paths (eager_warmup was ~40% of a
                    # measurement's wall time at ten).
                    num_eager_warmups=3,
                    num_runs=test_ite,
                    repeat_n=1,
                    use_cuda_graph=True,
                    allow_graph_fail=False,
                ) as results,
            ):
                pass
            _lap(point_phase, "measure_skip" if skip_indexer else "measure_full")
            # Protocol phases of this measurement (eager_warmup, capture, graph_warmup,
            # timed, cleanup) on the point's timing line, prefixed by the measurement.
            _bench_prefix = "skip" if skip_indexer else "full"
            for _phase_name, _phase_secs in (results.get("phase_seconds") or {}).items():
                point_phase[f"{_bench_prefix}_{_phase_name}"] = _phase_secs

            if results.get("used_cuda_graph") is not True:
                raise RuntimeError("vLLM MLA/DSA module requires CUDA graph timing; refusing to publish eager timing")
            latency = results["latency_ms"]

            # 6. Log results — schema aligned with TRT-LLM
            if is_context:
                isl = seq_len
                step = prefix_len
            else:
                isl = 1
                # BatchSpec.seq_lens includes this forward's one new token.
                # Both module loaders reconstruct total KV length as isl + step.
                step = seq_len - 1

            op_name = f"{attn_type}_{phase}_module" + ("_skip_indexer" if skip_indexer else "")
            backend_name = _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata)

            log_perf(
                item_list=[
                    {
                        "model": model_path,
                        "architecture": architecture,
                        "mla_dtype": "bfloat16" if compute_dtype == "bfloat16" else compute_dtype,
                        "kv_cache_dtype": actual_kv_cache_dtype,
                        "gemm_type": "bfloat16" if gemm_type == "bfloat16" else gemm_type,
                        "num_heads": num_heads,
                        "batch_size": batch_size,
                        "isl": isl,
                        "tp_size": 1,
                        "step": step,
                        "latency": f"{latency:.4f}",
                    }
                ],
                framework="VLLM",
                version=vllm_version,
                device_name=torch.cuda.get_device_name(device),
                op_name=op_name,
                kernel_source=backend_name,
                perf_filename=perf_filename,
                power_stats=results["power_stats"],
            )

            print(
                f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
                f"prefix={prefix_len}, gemm={gemm_type}, compute={compute_dtype}, "
                f"kv={kv_cache_dtype}, backend={backend_name}: {latency:.4f} ms"
            )

            return latency

        full_latency = measure_and_log(False)
        if has_reuse:
            measure_and_log(True)
        return full_latency

    latencies: list[float] = []
    failures: list[tuple[tuple[int, int], BaseException]] = []
    for index, (point_seq_len, point_prefix) in enumerate(points):
        _t_point = _t_last = time.perf_counter()
        point_phase: dict[str, float] = dict(_setup_phase) if index == 0 else {}
        try:
            if (point_seq_len, point_prefix) in does_not_fit:
                raise RuntimeError(does_not_fit[(point_seq_len, point_prefix)])
            with ExitStack() as point_stack:
                latencies.append(_run_point(point_stack, point_seq_len, point_prefix, point_phase))
        except Exception as exc:
            if _is_fatal_cuda_error(exc) or len(points) == 1:
                raise
            # The traceback holds _run_point's frame (KV cache, indexer cache, metadata,
            # activations): clear it, or the largest-first series keeps that memory
            # pinned and the smaller points that would fit fail too.
            traceback.clear_frames(exc.__traceback__)
            failures.append(((point_seq_len, point_prefix), exc))
            print(
                f"  [{variant} module] {phase} b={batch_size}, s={point_seq_len}, prefix={point_prefix}: "
                f"point failed ({type(exc).__name__}: {str(exc)[:200]}); the series continues"
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue
        point_phase["indexer_fill"] = globals().get("_PHASE_TIMINGS", {}).get("indexer_fill", 0.0)
        print(
            f"  [{variant} module timing] {phase} b={batch_size}, s={point_seq_len}, prefix={point_prefix}, "
            f"heads={num_heads}, gemm={gemm_type}, kv={kv_cache_dtype}: "
            + " ".join(f"{name}={seconds:.2f}s" for name, seconds in point_phase.items())
            + f" total={time.perf_counter() - _t_point:.2f}s"
        )
    if len(points) > 1:
        print(
            f"  [{variant} module series] {phase} b={batch_size}, prefix={prefix_desc}, heads={num_heads}, "
            f"gemm={gemm_type}, kv={kv_cache_dtype}: {len(latencies)}/{len(points)} points in "
            f"{time.perf_counter() - _t_series:.1f}s"
        )
    if failures:
        if len(failures) == 1:
            raise failures[0][1]
        summary = "; ".join(f"s={s} prefix={p}: {type(exc).__name__}: {str(exc)[:200]}" for (s, p), exc in failures)
        # Keep the kernel-limit classification when every point was refused on a limit.
        kernel_limit = "FIXME(kernel-limit)"
        prefix = f"{kernel_limit}: " if all(str(exc).startswith(kernel_limit) for _, exc in failures) else ""
        raise RuntimeError(f"{prefix}{len(failures)}/{len(points)} point(s) of the module series failed: {summary}")
    return latencies


def run_mla_module_worker(
    seq_len: int | list,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py.

    ``seq_len`` is one point (the CLI's --quick) or the case's list of points
    measured on one module (the registry plan, see _series_points).
    """
    # Serving executes model forward under inference mode (execute_model of the
    # default Model Runner V2, v1/worker/gpu/model_runner.py:1718-1719 @v0.32.0;
    # config/vllm.py:703-738 picks it). Keep module initialization,
    # dry runs, graph warmup and timing in the SAME mode: FlashInfer 0.6.13
    # mutates a cached workspace that may have been created by the dry run.
    with torch.inference_mode():
        return run_mla_module(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            prefix_len=prefix_len,
            perf_filename=perf_filename,
            model_path=model_path,
            attn_type=attn_type,
            device=device,
        )


def _cleanup():
    # Release vLLM's WorkspaceManager singleton scratch buffers before
    # returning to the worker loop. ``_ensure_workspace_size`` only grows —
    # once a task demands N bytes, the manager pins N bytes for the worker's
    # lifetime, so a single large task turns into a permanent allocator
    # reservation that starves subsequent small tasks into OOM.
    #
    # The module-level ``_manager`` is private; vLLM offers no public
    # teardown API. Nulling it drops the Python reference to the old
    # manager (and its workspace tensors), and the next task's
    # ``init_workspace_manager()`` call creates a fresh instance on demand.
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    # gc.collect() must run BEFORE empty_cache() — ``empty_cache`` only
    # releases blocks with zero live allocations, so any Python-reachable
    # tensors need to be dropped first or the pass is a no-op.
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _supported_model_map(*, apply_model_filter: bool = True) -> dict[str, str]:
    return {
        spec.model_path: spec.attention_type
        for spec in get_mla_module_model_specs(backend="vllm", apply_model_filter=apply_model_filter)
        if spec.attention_type in {"mla", "dsa"}
    }


def _perf_file_for(attn_type: str, mode: str) -> str:
    if mode == "context":
        return PerfFile.MLA_CONTEXT_MODULE if attn_type == "mla" else PerfFile.DSA_CONTEXT_MODULE
    return PerfFile.MLA_GENERATION_MODULE if attn_type == "mla" else PerfFile.DSA_GENERATION_MODULE


def main():
    supported_models = _supported_model_map(apply_model_filter=False)
    model_names = list(supported_models.keys())

    parser = argparse.ArgumentParser(
        description="MLA/DSA module-level collector for vLLM",
    )
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=model_names,
        help="Model to benchmark. If omitted, run one declared canonical reference per consumer identity.",
    )
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by number of heads")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size (for --quick)")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq len (for --quick)")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8 when GPU supports it)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="Compute dtype for attention (default: auto based on phase and GPU)",
    )
    parser.add_argument(
        "--gemm-type",
        type=str,
        choices=["bfloat16", "fp8_block", "nvfp4"],
        default=None,
        help="GEMM quantisation type for linear layers (default: run all supported by GPU)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Quick single-point test")
    args = parser.parse_args()

    # Select models to run
    if args.model:
        models_to_run = {args.model: supported_models[args.model]}
    else:
        models_to_run = _supported_model_map()

    if not args.quick:
        # The full sweep is the op plan's rows; --model targets that checkpoint, as collect.py --model-path does.
        if args.model:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model
        for attn_type in sorted(set(models_to_run.values())):
            perf_filename = _perf_file_for(attn_type, args.mode)
            rows = plan(f"{attn_type}_{args.mode}_module", "vllm", get_sm_version()).rows
            for points, b, h, kv_dtype, compute, gemm, model_path, _attn in rows:
                if args.num_heads is not None and h != args.num_heads:
                    continue
                if args.kv_cache_dtype is not None and kv_dtype != args.kv_cache_dtype:
                    continue
                if args.compute_dtype is not None and compute != args.compute_dtype:
                    continue
                if args.gemm_type is not None and gemm != args.gemm_type:
                    continue
                try:
                    run_mla_module_worker(
                        points,
                        b,
                        h,
                        kv_dtype,
                        compute,
                        gemm,
                        model_path,
                        attn_type,
                        perf_filename=perf_filename,
                        device=args.device,
                    )
                except Exception as e:
                    print(f"  FAILED: b={b}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}: {e}")
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    gc.collect()
        return

    for model_path, attn_type in models_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path}  |  Attention: {attn_type.upper()}")
        print(f"{'=' * 60}")
        run_mla_module_worker(
            seq_len=args.seq_len or 2048,
            batch_size=args.batch_size or 4,
            num_heads=args.num_heads or 128,
            kv_cache_dtype=args.kv_cache_dtype or "bfloat16",
            compute_dtype=args.compute_dtype or "bfloat16",
            gemm_type=args.gemm_type or "bfloat16",
            perf_filename=_perf_file_for(attn_type, args.mode),
            model_path=model_path,
            attn_type=attn_type,
            device=args.device,
        )


if __name__ == "__main__":
    main()
