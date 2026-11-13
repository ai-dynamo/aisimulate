# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 support landed in SGLang v0.5.16 (models/minimax_m3.py +
# layers/attention/minimax_sparse_backend.py + minimax_sparse_ops/); the msa
# family runs on 0.5.23. One release per file; earlier code is
# `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

"""
MSA Module Collector for SGLang — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass (the
fused qkv+index projection, grouped Gemma QK norm + partial RoPE, index
attention + top-k block selection, sparse GQA and the output projection) at
the ModelRunner level, through SGLang's own ServerArgs → ModelRunner →
ScheduleBatch → ForwardBatch pipeline with dummy weights.

Construction follows serving for the sparse layers (layers 3+ of the real
checkpoint): ``MiniMaxM3DecoderLayer`` builds
``MiniMaxM3Attention(is_sparse_attention_layer=True, disable_index_value=True)``
for every layer whose ``sparse_attention_freq``/``sparse_disable_index_value``
entries are set (models/minimax_m3.py:1702-1721@v0.5.23); the collector
truncates those per-layer arrays so its benchmark layer takes that path.
Serving fuses qkv_proj and index_qkv_proj into one GEMM at the end of
``load_weights`` (models/minimax_m3.py:2244-2246, 1117-1158); the dummy loader
never calls ``load_weights`` (model_loader/loader.py:1493-1530), so
``load_model_runner`` runs that same step. Attention dispatch is SGLang's own:
``attn_backend_wrapper`` wraps the platform dense backend with
``MiniMaxHybridAttnBackend``/``MiniMaxSparseAttnBackend`` for every
``is_minimax_sparse`` model (layers/attention/attention_registry.py:375-391),
and the benchmarked layer id is verified to route to the sparse backend.

Per SM, SGLang's M3 server-args override
(arg_groups/model_overrides/minimax_m3.py@v0.5.23) resolves backend and page:
  * SM90: fa3 + page 128 (:178-192); the sparse layers run the Triton path.
  * SM100/103: fa4 + page 128 with a BF16 KV cache, trtllm_mha + page 128 with
    fp8_e4m3 (:147-177). The latter turns on the fp8 attention GEMMs: fp8 q,
    an fp8 index cache and fp8 sparse kernels (server_args.py:592-610
    m3_fp8_attn_gemm_enabled). The main sparse step runs fmha_sm100 (MSA)
    when the image ships it (minimax_sparse_ops/msa.py:68-82,
    minimax_sparse_backend.py:196-249), else Triton.
  * SM89, SM120/121: no override branch. The generic default is flashinfer
    (arg_groups/model_override_base.py:277-334) at page 1
    (arg_groups/overrides.py:1341-1362); the sparse layers run Triton.
The collector never pins a kernel: ``kernel_source`` records what ran.

Op names / perf schema are aligned with collector/trtllm/collect_msa_module.py
(msa_context_module / msa_generation_module, architecture
MiniMaxM3ForCausalLM) so perf_database queries work across frameworks.
Unless noted, SGLang citations in this file are @v0.5.23 and relative to
python/sglang/srt (kernel paths start at python/sglang/kernels).

Usage (inside the pinned SGLang image, collector/framework_manifest.yaml):
    python3 -m collector.sglang.collect_msa_module --mode context --quick \
        --batch-size 2 --seq-len 512 --num-heads 64
    python3 -m collector.sglang.collect_msa_module --mode generation --quick \
        --batch-size 4 --seq-len 2048 --num-heads 64
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import traceback
from array import array
from importlib.metadata import version as get_version

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.helper import (
    PhaseClock,
    _resolve_local_model_path,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    render_phase_totals,
    run_measurement_subprocess,
)
from collector.op_plans import plan, plan_cases

try:
    from collector.sglang.runtime_compat import resolved_arg
    from collector.sglang.runtime_limits import (
        kernel_limit_error,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        serving_memory_defaults,
        temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:  # direct-script import inside the subprocess
    from runtime_compat import resolved_arg
    from runtime_limits import (
        kernel_limit_error,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        serving_memory_defaults,
        temporarily_chunked_alloc_extend,
    )

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

MSA_ARCHITECTURE = "MiniMaxM3ForCausalLM"  # AIC-canonical persisted architecture
# SGLang's serving identity for the M3 text decoder. The official
# MiniMaxAI/MiniMax-M3 checkpoint is a VL repo whose text_config declares
# architectures=["MiniMaxM3SparseForCausalLM"] (HF config.json), and SGLang
# keys every M3 code path on that name: is_minimax_sparse
# (configs/model_config.py:426-431), the model EntryClass
# (models/minimax_m3.py:2274), the server-args override
# (arg_groups/model_overrides/minimax_m3.py:37-41), and the KV-pool /
# attention-backend wiring (mem_cache/kv_cache_configurator.py:1890-1896,
# 2552-2600; layers/attention/attention_registry.py:375-391).
SGLANG_MSA_ARCHITECTURE = "MiniMaxM3SparseForCausalLM"

# Perf-database dtype strings → SGLang ServerArgs kv_cache_dtype values
# (same mapping as collect_mla_module.py).
SGLANG_KV_DTYPE = {"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}

# MiniMax-M3 native GQA geometry (bundled MiniMaxAI--MiniMax-M3_config.json,
# identical to the HF checkpoint text_config): 64 query heads / 4 KV heads /
# 4 index heads. Serving TP sharding (models/minimax_m3.py):
#   q  heads: num_heads = total // attn_tp_size    (:749-751)
#   kv heads: max(1, total_kv // attn_tp_size)     (:752-762,
#             replicated when attn_tp_size > total_kv)
#   index heads: idx_head_tp_size = min(attn_tp_size, total_idx);
#             num_idx_heads = total_idx // idx_head_tp_size — i.e.
#             max(1, total_idx // attn_tp_size), replica ranks share one
#             head when attn_tp_size > total_idx (:784-796).
# All three are emulated on one GPU by shrinking the config head counts.

# Tokens of cached history written per store call (_initialize_msa_history):
# 64K tokens of K/V + index K stay below 200 MiB at TP 1.
_HISTORY_CHUNK_TOKENS = 1 << 16

# Per-subprocess sum of the point phase times printed by run_msa_module.
_POINT_PHASE_TOTALS: dict[str, float] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════════
# Test-case generation (collect.py entrypoints)
# ═══════════════════════════════════════════════════════════════════════


def get_msa_context_module_test_cases():
    return plan_cases("msa_context_module", "sglang", get_sm_version())


def get_msa_generation_module_test_cases():
    return plan_cases("msa_generation_module", "sglang", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def _register_minimax_m3_text_config():
    """Make transformers AutoConfig accept model_type="minimax_m3".

    The bundled text-only config keeps the checkpoint's raw schema
    (model_type "minimax_m3"), which neither transformers 5.12 nor SGLang's
    _CONFIG_REGISTRY knows. Serving materialises the M3 text config as a
    generic attribute bag: SGLang's own VL config coerces the checkpoint's
    text_config dict through CONFIG_MAPPING.get(model_type, PretrainedConfig)
    — an unknown model_type "falls back to PretrainedConfig so dict keys
    still become real attributes" (configs/minimax_vl.py:10-23), and
    the M3 modeling code reads plain attributes only. Registering a bare
    PretrainedConfig alias therefore changes HOW the config object is
    constructed, never which kernel runs (M3 dispatch keys on
    architectures[0] + sparse_attention_config, both preserved verbatim).
    Same shim precedent as collector/trtllm/collect_msa_module.py.
    """
    import transformers

    class _MiniMaxM3TextConfig(transformers.PretrainedConfig):
        model_type = "minimax_m3"

    try:
        transformers.AutoConfig.register("minimax_m3", _MiniMaxM3TextConfig)
    except ValueError as e:  # already registered by an earlier case in this proc
        if "already" not in str(e).lower():
            raise


def _resolve_sglang_model_dir(model_id: str) -> str:
    """Resolve the bundled M3 config and present it under SGLang's identity.

    ``collector.helper._resolve_local_model_path`` materialises the bundled
    AIC config (auto_map stripped). That config carries AIC's canonical
    ``architectures=["MiniMaxM3ForCausalLM"]``; SGLang's text-decoder entry
    class is ``MiniMaxM3SparseForCausalLM`` — exactly what the official
    checkpoint's own text_config.architectures declares (see
    SGLANG_MSA_ARCHITECTURE above). Rewrite only that field so SGLang loads
    the same model class serving uses; every other field stays verbatim.
    """
    base_dir = _resolve_local_model_path(model_id)
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)

    if config.get("architectures") not in (
        [MSA_ARCHITECTURE],
        [SGLANG_MSA_ARCHITECTURE],
    ):
        raise ValueError(
            f"bundled config for {model_id} declares architectures="
            f"{config.get('architectures')!r}; expected the MiniMax-M3 text decoder"
        )
    if not config.get("sparse_attention_config"):
        raise ValueError(
            f"model {model_id} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )
    config["architectures"] = [SGLANG_MSA_ARCHITECTURE]

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_msa_config_{model_id.replace('/', '_')}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    target = os.path.join(tmp_dir, "config.json")
    tmp_target = f"{target}.{os.getpid()}.tmp"
    with open(tmp_target, "w") as f:
        json.dump(config, f)
    os.replace(tmp_target, target)
    return tmp_dir


def _build_model_override_args(config: dict, num_heads: int, target_tp_size: int, num_layers: int) -> dict:
    """Single-GPU emulation of one target-TP rank + benchmark-layer surgery.

    Head shrinking mirrors SGLang's own TP sharding math (citations at the
    top of this file: models/minimax_m3.py:749-762 q/kv, :784-796 index
    heads). The per-layer pattern arrays are truncated so every loaded layer
    takes the sparse construction path of serving layers 3+
    (is_sparse_attention_layer=True, disable_index_value=True —
    models/minimax_m3.py:1702-1721 reading get_minimax_sparse_layer_ids /
    get_minimax_sparse_disable_value_layer_ids,
    configs/model_config.py:460-471). The MLP is made dense
    (moe_layer_freq=0 → MiniMaxM3MLP, models/minimax_m3.py:1725-1744,
    1791-1793) — it is outside the timed attention module; same single-layer
    surgery as the trtllm MSA collector. All sparse scalar fields stay
    checkpoint-verbatim.
    """
    native_q = int(config["num_attention_heads"])
    native_kv = int(config["num_key_value_heads"])
    if native_q % target_tp_size != 0 or native_q // target_tp_size != num_heads:
        raise ValueError(f"num_heads={num_heads} does not equal native {native_q} / target_tp {target_tp_size}")
    local_kv = max(1, native_kv // target_tp_size)

    sparse_cfg = dict(config["sparse_attention_config"])
    native_idx = int(sparse_cfg["sparse_num_index_heads"])
    local_idx = max(1, native_idx // target_tp_size)
    sparse_cfg["sparse_attention_freq"] = [1] * num_layers
    # Serving layers 3+ all disable the index value path
    # (sparse_disable_index_value[3:] == 1 in the checkpoint config); the
    # benchmark layer replicates that flavor.
    sparse_cfg["sparse_disable_index_value"] = [1] * num_layers
    sparse_cfg["sparse_num_index_heads"] = local_idx

    return {
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": local_kv,
        "moe_layer_freq": [0] * num_layers,
        "sparse_attention_config": sparse_cfg,
    }


def _expect(actual, expected, what: str):
    if actual != expected:
        raise RuntimeError(f"{what}={actual!r}, expected {expected!r}")


def _validate_msa_module(model_runner, attn, num_heads: int, target_tp_size: int, head_dim: int):
    """Verify the single-GPU module emulates one target-TP rank of serving.

    Shapes per models/minimax_m3.py: qkv_proj packs
    (num_heads + 2*num_kv_heads)*head_dim (:801-810), o_proj input
    num_heads*head_dim (:812-820), index_qkv_proj packs
    (num_idx_heads + 1)*idx_head_dim with v_head_size=0 when
    disable_index_value (:830-841), index_o_proj is None (:843-844). The
    projection serving runs is their concatenation (_FusedQKVIndexProj,
    :245-284, built by maybe_build_fused_qkv_index :1117-1158 for a CUDA sparse
    layer whose two projections share one quant method, as BF16 ones do).
    """
    local_kv = max(1, num_heads // 16)  # native 64q/4kv → GQA ratio 16
    local_idx = max(1, 4 // target_tp_size)
    if not getattr(attn, "is_sparse_attention_layer", False):
        raise RuntimeError("benchmark layer came out dense; sparse_attention_freq surgery did not take")
    if not getattr(attn, "disable_index_value", False):
        raise RuntimeError("benchmark layer keeps the index-value path; serving layers 3+ disable it")
    _expect(int(attn.num_heads), num_heads, "self_attn.num_heads")
    _expect(int(attn.num_kv_heads), local_kv, "self_attn.num_kv_heads")
    _expect(int(attn.num_idx_heads), local_idx, "self_attn.num_idx_heads")
    _expect(int(attn.head_dim), head_dim, "self_attn.head_dim")
    _expect(
        int(attn.qkv_proj.output_size_per_partition),
        (num_heads + 2 * local_kv) * head_dim,
        "qkv_proj.output_size_per_partition",
    )
    _expect(
        int(attn.o_proj.input_size_per_partition),
        num_heads * head_dim,
        "o_proj.input_size_per_partition",
    )
    _expect(
        int(attn.index_qkv_proj.output_size_per_partition),
        (local_idx + 1) * int(attn.idx_head_dim),
        "index_qkv_proj.output_size_per_partition",
    )
    if attn.index_o_proj is not None:
        raise RuntimeError("index_o_proj exists despite disable_index_value=True")
    fused = getattr(attn, "_fused_qkv_index", None)
    if fused is None:
        raise RuntimeError(
            "benchmark layer has no fused qkv+index projection; serving builds it after loading "
            "(models/minimax_m3.py:1117-1158, 2244-2246) and forward_prepare runs it (:1491-1551)"
        )
    _expect(
        int(fused.output_size_per_partition),
        (num_heads + 2 * local_kv) * head_dim + (local_idx + 1) * int(attn.idx_head_dim),
        "fused_qkv_index_proj.output_size_per_partition",
    )

    backend = model_runner.attn_backend
    sparse = getattr(backend, "sparse", None)
    if sparse is None or attn.attn.layer_id not in getattr(backend, "sparse_layer_ids", ()):
        raise RuntimeError(
            "benchmark layer does not route to MiniMaxSparseAttnBackend; "
            "attn_backend_wrapper did not wrap the runner (attention_registry.py:375-391)"
        )


def cleanup_distributed():
    """Clean up SGLang distributed process-group state between subprocesses."""
    import sglang.srt.distributed.parallel_state as parallel_state

    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass


def load_model_runner(
    model_path: str,
    num_heads: int,
    kv_cache_dtype: str,
    target_tp_size: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
    device: str = "cuda:0",
):
    """Load an SGLang ModelRunner hosting the M3 sparse module (dummy weights).

    Serving-selected defaults are deliberately NOT overridden: attention
    backend, page size and the fp8 attention-GEMM mode resolve exactly as for
    a server launched with only --kv-cache-dtype (the M3 override and the
    generic default, cited at the top of this file). The prefill CUDA graph
    config is left at serving's default too; this collector never captures a
    prefill graph (deviation note at _run_prefill_point).

    max_running_requests / max_total_tokens / mem_fraction_static are
    benchmark-harness resource knobs (user-tunable ServerArgs, not kernel
    dispatch). run_msa_module leaves chunked_prefill_size to SGLang, which
    derives it from device memory as serving does, and refuses a prefill
    beyond it; mem_fraction_static is pinned because SGLang's heuristic
    couples it to chunked_prefill_size (arg_groups/memory_hook.py
    _handle_gpu_memory_settings) and would go negative.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype)
    if sglang_kv_dtype is None:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r} for MSA")

    _register_minimax_m3_text_config()
    local_model_path = _resolve_sglang_model_dir(model_path)
    with open(os.path.join(local_model_path, "config.json")) as f:
        raw_config = json.load(f)
    override_args = _build_model_override_args(raw_config, num_heads, target_tp_size, num_layers)

    # attention_backend stays framework-selected (None): the collector measures
    # serving truth. SM89 and SM12x have no M3 override branch and take the
    # generic flashinfer + page-1 default. Through v0.5.16 that crashed at
    # backend init on the M3 KV pool (get_kv_cache_quant_method returned
    # None), and the l40s/rtx tables collected with --attention-backend triton
    # were withdrawn (review 4969690316 Standards-1). MiniMaxSparseKVPool now
    # forwards get_kv_cache_quant_method to its main pool
    # (mem_cache/memory_pool.py:6057-6060), so the default may initialize
    # there; whether it does is unverified on a GPU at this pin. Cases that
    # still fail stay classified failures.
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        kv_cache_dtype=sglang_kv_dtype,
        attention_backend=None,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=(chunked_prefill_size or 16384),
        mem_fraction_static=0.80,
        max_running_requests=max_running_requests,
        json_model_override_args=json.dumps(override_args),
    )

    _set_envs_and_config(server_args)

    model_config = ModelConfig.from_server_args(server_args)
    actual_architecture = (model_config.hf_config.architectures or [None])[0]
    if actual_architecture != SGLANG_MSA_ARCHITECTURE:
        raise RuntimeError(
            f"SGLang loaded architecture={actual_architecture!r} for {model_path}, expected {SGLANG_MSA_ARCHITECTURE!r}"
        )

    from collector.sglang.runtime_compat import free_local_port, init_runtime_config

    nccl_port = free_local_port()
    _runner_parallel_kwargs = init_runtime_config(server_args, gpu_id, nccl_port=nccl_port, model_config=model_config)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        **_runner_parallel_kwargs,
        server_args=server_args,
    )
    # Serving's load_weights ends by fusing each sparse layer's qkv_proj and
    # index_qkv_proj into one GEMM (models/minimax_m3.py:2244-2246), which
    # forward_prepare then runs with the grouped QK norm/RoPE kernel
    # (:1491-1551). The dummy loader never calls load_weights
    # (model_loader/loader.py:1493-1530; M3 has no post_load_weights hook,
    # :315-321), so without this step the module times two GEMMs and two
    # norm/RoPE kernels. The BF16 weights need no post-processing on CUDA
    # (layers/quantization/unquant.py:568-570), so fusing after the dummy
    # loader's pass gives serving's weights.
    from sglang.srt.models.minimax_m3 import build_minimax_fused_qkv_index

    build_minimax_fused_qkv_index(model_runner.model)
    model_runner.alloc_memory_pool()
    model_runner.init_attention_backends()

    head_dim = int(raw_config.get("head_dim", raw_config["hidden_size"] // raw_config["num_attention_heads"]))
    attn = model_runner.model.model.layers[0].self_attn
    _validate_msa_module(model_runner, attn, num_heads, target_tp_size, head_dim)

    sparse = model_runner.attn_backend.sparse
    print(
        f"SGLang resolved: dense={type(model_runner.attn_backend.dense).__name__}, "
        f"attention_backend={resolved_arg(server_args, 'attention_backend')}, page_size={sparse.page_size}, "
        f"kv={sparse.kv_pool.main_pool.dtype}, index_kv={getattr(sparse.kv_pool.index_k_pool, 'dtype', None)}, "
        f"fp8_attn_gemm={sparse.fp8_attn_gemm}, use_msa={sparse.use_msa}, msa_decode={sparse._use_msa_decode}, "
        f"chunked_prefill_size={resolved_arg(server_args, 'chunked_prefill_size')}, "
        f"max_total_tokens={max_total_tokens}, max_running_requests={max_running_requests}"
    )
    return model_runner


# ═══════════════════════════════════════════════════════════════════════
# ForwardBatch construction + benchmarking
# ═══════════════════════════════════════════════════════════════════════


class PerfLogWriteError(RuntimeError):
    """Fail the subprocess when a measured row was not durably persisted."""


def _forward_ctx(model_runner, forward_batch):
    """Publish SGLang's per-forward control context, exactly as serving does
    around every model forward (model_executor/model_runner.py:1973-1981).
    RadixAttention resolves the attention backend through it
    (layers/radix_attention.py:183, model_executor/forward_context.py:73-74)."""
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    return forward_context(ForwardContext(attn_backend=model_runner.attn_backend), forward_batch=forward_batch)


def _make_reqs(batch_size: int, full_length: int):
    """Requests with a full_length-token prompt, built as SGLang's offline batch
    builder builds them (benchmark/one_batch.py:450-476): fill ids are the
    prompt and this round extends [prefix_len=0, extend_end) (managers/
    schedule_batch.py:1103-1106, 1218-1220). The token ids are never embedded."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    reqs = []
    for i in range(batch_size):
        # A bulk copy: per-token Python ints cost seconds at 1M-token prompts.
        prompt = array("q")
        prompt.frombytes(torch.randint(0, 10000, (full_length,), dtype=torch.int64).numpy().tobytes())
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=prompt,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = array("q", prompt)
        req.logprob_start_len = 0
        req.extend_end = full_length
        reqs.append(req)
    return reqs


def _radix_disabled_tree_cache(model_runner):
    """The prefix cache serving runs with --disable-radix-cache (load_model_runner sets it).

    v0.5.23 deleted ChunkCache (sglang #43201). For a full-attention-only model a
    disabled cache is a UnifiedRadixCache with disable=True and the FULL component
    (mem_cache/registry.py:83-98 -> create_unified_radix_cache :149-191;
    CacheInitParams as mem_cache/kv_cache_builder.py:389-402 builds it, page_size
    the allocator's without DCP). It matches, inserts and evicts nothing. Same
    construction as collect_mla_module._radix_disabled_tree_cache.
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
            tree_components=(ComponentType.FULL,),
        )
    )


def _make_schedule_batch(model_runner, reqs, tree_cache):
    """A non-overlap, non-speculative batch, as the scheduler builds one (ScheduleBatch.init_new)."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _extend_batch(model_runner, reqs, prefix_len: int):
    """An extend batch over [prefix_len, extend_end) whose cached prefix is an earlier chunk.

    With the radix cache disabled serving matches nothing, so a prefix lives only in
    the request's own row from its previous chunk (mem_cache/allocation.py:363-371).
    The row is built as serving builds it: an extend over [0, prefix_len) allocates it
    (schedule_batch.py:2847-2919, allocation.py:374+), the scheduler's checkpoint
    advances prefix_len to its end (mem_cache/common.py:271-280), and the next round
    extends [prefix_len, extend_end). The KV pool's page size is whatever serving
    resolved: at page 1 both rounds take the token allocator, as serving does. The
    prefill ids reach the forward stream the way SGLang's offline builder hands them
    over (benchmark/one_batch.py:516-523), so the ForwardBatch counts its real tokens
    (model_executor/forward_batch_info.py:1364-1382).
    """
    from sglang.srt.mem_cache.common import checkpoint_kv_cache

    batch_size = len(reqs)
    full_length = reqs[0].extend_end
    tree_cache = _radix_disabled_tree_cache(model_runner)
    if prefix_len:
        for req in reqs:
            req.extend_end = prefix_len
        prefix_batch = _make_schedule_batch(model_runner, reqs, tree_cache)
        with temporarily_chunked_alloc_extend(model_runner, batch_size * prefix_len):
            prefix_batch.prepare_for_extend()
        for req in reqs:
            checkpoint_kv_cache(req, tree_cache)
            req.extend_end = full_length
    batch = _make_schedule_batch(model_runner, reqs, tree_cache)
    with temporarily_chunked_alloc_extend(model_runner, batch_size * (full_length - prefix_len)):
        batch.prepare_for_extend()
    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
    batch.prefill_input_ids_cpu = None
    return batch


def _initialize_msa_history(model_runner, attention_module, forward_batch, history_length: int) -> None:
    """Write the cached history the benchmark layer reads, through serving's own KV writer.

    The allocator only assigns slots. The pools start as zeros
    (mem_cache/memory_pool.py:2580-2600 MHATokenToKVPool._create_buffers_normal) and
    nothing here runs the forwards that would fill them, so every history block
    would score the same and the decode radix top-k would select among ties. Serving
    stores post-norm/RoPE K/V and index K with set_fused_kv_index_buffer
    (layers/attention/minimax_sparse_backend.py:1483-1494, 1724-1735; the pool's writer
    mem_cache/memory_pool.py:6211-6280); seeded N(0, 1) activations go through the same
    writer outside the measurement. They stand for no particular checkpoint distribution.
    Same treatment as the DSA modules (collect_mla_module._initialize_dsa_history).
    """
    if history_length <= 0:
        return
    layer = attention_module.attn
    pool = model_runner.token_to_kv_pool
    locations = (
        model_runner.req_to_token_pool.req_to_token[forward_batch.req_pool_indices.long(), :history_length]
        .flatten()
        .to(torch.int64)
        .contiguous()
    )
    expected = int(forward_batch.batch_size) * history_length
    if locations.numel() != expected:
        raise ValueError(f"MSA history has {locations.numel()} locations; expected {expected}")
    num_kv_heads, head_dim = int(attention_module.num_kv_heads), int(attention_module.head_dim)
    idx_head_dim = int(attention_module.idx_head_dim)
    generator = torch.Generator(device=locations.device)
    generator.manual_seed(0)

    def activations(n: int, heads: int, dim: int):
        return torch.randn((n, heads, dim), generator=generator, device=locations.device, dtype=torch.bfloat16)

    for start in range(0, expected, _HISTORY_CHUNK_TOKENS):
        slots = locations[start : start + _HISTORY_CHUNK_TOKENS]
        n = slots.numel()
        pool.set_fused_kv_index_buffer(
            layer,
            slots,
            activations(n, num_kv_heads, head_dim),
            activations(n, num_kv_heads, head_dim),
            activations(n, 1, idx_head_dim),
            None if attention_module.disable_index_value else activations(n, 1, idx_head_dim),
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )


def _serving_decode_graph_batch(model_runner, batch_size: int, target_tp_size: int) -> int | None:
    """The captured batch size serving replays this decode batch on, or None when it runs eagerly.

    DecodeCudaGraphRunner replays a batch no larger than its max_bs, padded up to
    the next captured size (model_executor/runner/decode_cuda_graph_runner.py:845-905,
    1615-1631). The sizes come from generate_decode_cuda_graph_batch_sizes over the
    decode max_bs (arg_groups/cuda_graph_hook.py:500-531), which
    _handle_gpu_memory_settings derives from device memory and the server's TP size
    (arg_groups/memory_hook.py:89-183). This runner resolved them for TP 1; the row
    stands for one rank of a TP-``target_tp_size`` server, which below 160 GiB
    captures up to 512 at TP >= 4 instead of 256 (runtime_limits.serving_memory_defaults).
    """
    decode_cfg = resolved_arg(model_runner.server_args, "cuda_graph_config").decode
    if decode_cfg.backend == "disabled":
        return None
    from sglang.srt.arg_groups.cuda_graph_hook import generate_decode_cuda_graph_batch_sizes

    capture_bs = generate_decode_cuda_graph_batch_sizes(
        model_runner.server_args, serving_memory_defaults(target_tp_size)[1]
    )
    return next((int(bs) for bs in sorted(capture_bs) if int(bs) >= batch_size), None)


def _minimax_sparse_ops():
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse

    return minimax_sparse


def _ran_msa(use_msa: bool) -> bool:
    """Whether the main sparse step ran fmha_sm100 since _run_*_point cleared the flag.

    minimax_sparse_prefill/decode fall back to the Triton kernel when MSA raises
    MSAUnavailableError and record it in _msa_fallback_warned
    (layers/attention/minimax_sparse_ops/minimax_sparse.py:28-40, 217-257, 446-447).
    """
    return use_msa and not _minimax_sparse_ops()._msa_fallback_warned


def _decode_topk_variant(sparse_backend, max_seqlen_k: int) -> str:
    """The decode top-k kernel the sparse indexer selects for this KV bound — the
    gate of flash_decode_with_topk_idx (kernels/ops/attention/minimax_sparse/decode/
    flash_with_topk_idx.py:25, 1040-1055): SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX,
    score blocks of min(max_seqlen, req_to_token width) <= _JIT_TOPK_MAX_NUM_BLOCKS
    (16384 on CUDA; 4096 at v0.5.21) and topk <= 32 select the JIT radix select,
    otherwise the 2-stage Triton top-k."""
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import _JIT_TOPK_MAX_NUM_BLOCKS
    from sglang.srt.environ import envs

    score_kv_len = min(int(max_seqlen_k), int(sparse_backend.req_to_token.shape[1]))
    score_blocks = _ceil_div(score_kv_len, sparse_backend.block_size_k)
    use_radix = (
        envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get()
        and score_blocks <= _JIT_TOPK_MAX_NUM_BLOCKS
        and sparse_backend.topk_blocks <= 32
    )
    return "topk_radix" if use_radix else "topk_split"


def _prefill_kernel_source(model_runner) -> str:
    """use_msa selects fmha_sm100 for the main step (minimax_sparse_backend.py:196-234)."""
    main = "msa_fmha_sm100" if _ran_msa(model_runner.attn_backend.sparse.use_msa) else "triton_sparse"
    return f"sglang_minimax_prefill_{main}"


def _decode_kernel_source(model_runner, max_seqlen_k: int) -> str:
    """``max_seqlen_k`` is the bound the decode kernels were launched with (see _run_decode_point)."""
    sparse = model_runner.attn_backend.sparse
    if not hasattr(sparse, "_use_msa_decode"):
        # kernel_source records ground truth; a silent False default would
        # mislabel every decode row as triton_sparse if SGLang renames the
        # flag (minimax_sparse_backend.py:289-291 sets it in __init__).
        raise RuntimeError(
            "MiniMaxSparseAttnBackend has no _use_msa_decode attribute "
            "(renamed upstream?); refusing to guess the decode kernel identity"
        )
    main = "msa_fmha_sm100" if _ran_msa(sparse._use_msa_decode) else "triton_sparse"
    return f"sglang_minimax_decode_{main}_{_decode_topk_variant(sparse, max_seqlen_k)}"


def _log_msa_row(
    *,
    perf_filename: str,
    model_path: str,
    compute_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    num_heads: int,
    batch_size: int,
    isl: int,
    step: int,
    target_tp_size: int,
    latency_ms: float,
    op_name: str,
    kernel_source: str,
    device_name: str,
    power_stats,
):
    if not log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": MSA_ARCHITECTURE,
                "mla_dtype": compute_dtype,
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": target_tp_size,
                "step": step,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=get_version("sglang"),
        device_name=device_name,
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise PerfLogWriteError(f"failed to persist MSA row to {perf_filename}")


def _run_prefill_point(
    model_runner,
    attention_module,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
):
    """Benchmark one context (prefill) point; returns (latency_ms, power_stats).

    The whole module is timed eagerly. Deviation (every CUDA SM): v0.5.23 serves
    M3 prefill under the breakable prefill CUDA graph (default_prefill_backend,
    model_executor/cuda_graph_config.py:119-123; M3 is on the BCG allow-list,
    configs/model_config.py:2501-2516, and no rule in
    arg_groups/cuda_graph_hook.py:170-245 turns it off). The sparse attention
    runs eagerly between the graph segments there too
    (RadixAttention._eager_attention, layers/radix_attention.py:133-185, 187-250),
    on the same backend code and metadata, so its kernels match. What differs:
    serving replays the fused qkv+index GEMM, the grouped QK norm/RoPE and o_proj
    from captured segments at the token count padded to the capture bucket, and
    copies the attention output into the segment's buffer; this collector
    launches them eagerly at the real token count. Launch overhead reads slower
    here for short prefills, bucket padding makes serving's GEMMs larger than
    timed here; neither is measured. The DSA prefill tables carry the same
    deviation (collect_mla_module.load_model_runner). Retire by timing prefill
    under the breakable runner once the measurement method for prefill graphs is
    decided.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    point_clock = PhaseClock()
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    batch = _extend_batch(model_runner, _make_reqs(batch_size, prefix_len + seq_len), prefix_len)
    point_clock.mark("batch")
    # return_hidden_states_before_norm=False mirrors the serving worker
    # (managers/tp_worker.py:274).
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    # Serving metadata init for the eager forward (base
    # AttentionBackend.init_forward_metadata → out_graph + in_graph;
    # MiniMaxHybrid fans out to sparse+dense — minimax_sparse_backend.py:1878-1881).
    with _forward_ctx(model_runner, forward_batch):
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    point_clock.mark("metadata")
    _initialize_msa_history(model_runner, attention_module, forward_batch, prefix_len)
    point_clock.mark("history")

    hidden_states = torch.randn(
        batch_size * seq_len,
        model_runner.model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    positions = (
        torch.arange(prefix_len, prefix_len + seq_len, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
        .flatten()
    )
    point_clock.mark("inputs")

    def kernel_func():
        with _forward_ctx(model_runner, forward_batch):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    _minimax_sparse_ops()._msa_fallback_warned = False
    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=False,
        ) as results:
            pass
        point_clock.mark("bench")
        point_clock.absorb(results.get("phase_seconds"), prefix="bench_")
        return results["latency_ms"], results["power_stats"]
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(f"  [MSA point timing] context b={batch_size}, s={seq_len}, prefix={prefix_len}: {point_clock.render()}")


def _run_decode_point(
    model_runner,
    attention_module,
    batch_size: int,
    kv_len: int,
    target_tp_size: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
):
    """Benchmark one generation (decode) point; returns (latency_ms, power_stats, kernel_source).

    Decode is timed under a captured CUDA graph exactly where serving replays one
    (_serving_decode_graph_batch). Serving's DecodeCudaGraphRunner allocates the
    backend's graph state (init_cuda_graph_state,
    model_executor/runner/decode_cuda_graph_runner.py:429), builds metadata at
    capture with init_forward_metadata_out_graph(in_capture=True) (:1400) and
    refills it before every replay with init_forward_metadata_out_graph (:1812).
    For the sparse backend the benchmark layer routes to
    (minimax_sparse_backend.py:1985-1988), that sequence reduces to the capture
    call: its init_cuda_graph_state is a no-op (:614-615); in_capture fixes the
    host-side _max_seqlen_k at max_context_len (:452-461), which sizes the score
    tensor and picks the top-k kernel baked into the graph; the refill only resets
    that host value to the live length (:463), which the captured kernels never
    read again, so replaying it before this capture would bake the live length in
    instead. Its decode kernels read req_to_token at full width in graph and eager
    mode alike (:1811-1852), so no page-table width differs from serving (unlike
    DSA's). Serving's MSA decode plan is built only when MSA owns decode, which a
    graph-enabled server never does (:289-291, 305-307, 469-470). The dense half
    of the hybrid backend is never read by the benchmark layer.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    point_clock = PhaseClock()
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    graph_batch = _serving_decode_graph_batch(model_runner, batch_size, target_tp_size)
    if graph_batch is not None and graph_batch != batch_size:
        # The plan's decode batches are capture sizes; an ad-hoc --quick batch may not be.
        raise RuntimeError(
            f"decode batch {batch_size} replays on serving's padded graph batch {graph_batch}; "
            "the collector times captured batch sizes only"
        )
    use_graph = graph_batch is not None

    batch = _extend_batch(model_runner, _make_reqs(batch_size, kv_len), 0)
    for req in batch.reqs:
        req.output_ids.append(0)
    # The sampled token of the previous step (benchmark/one_batch.py:536-538);
    # prepare_for_decode allocates its slot and advances seq_lens
    # (managers/schedule_batch.py:3780-3789).
    batch.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=batch.device)
    batch.prepare_for_decode()
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    point_clock.mark("batch")

    sparse = model_runner.attn_backend.sparse
    with _forward_ctx(model_runner, forward_batch):
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        if use_graph:
            sparse.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    # The KV bound every decode kernel of this point is launched with
    # (minimax_sparse_backend.py:1823 passes it to minimax_sparse_decode).
    max_seqlen_k = int(sparse._max_seqlen_k)
    point_clock.mark("metadata")
    _initialize_msa_history(model_runner, attention_module, forward_batch, kv_len)
    point_clock.mark("history")

    hidden_states = torch.randn(
        batch_size,
        model_runner.model.config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # The decoded token's position (seq_lens - 1 after prepare_for_decode).
    positions = torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int64)
    point_clock.mark("inputs")

    def kernel_func():
        with _forward_ctx(model_runner, forward_batch):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    _minimax_sparse_ops()._msa_fallback_warned = False
    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=use_graph,
        ) as results:
            pass
        point_clock.mark("bench")
        point_clock.absorb(results.get("phase_seconds"), prefix="bench_")
        return results["latency_ms"], results["power_stats"], _decode_kernel_source(model_runner, max_seqlen_k)
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(f"  [MSA point timing] generation b={batch_size}, kv={kv_len}: {point_clock.render()}")


# ═══════════════════════════════════════════════════════════════════════
# Subprocess orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_msa_module(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    batch_size_filter: int | None = None,
    target_tp_size: int = 1,
    quick_shape: tuple | None = None,
    inner_manifest=(),
):
    """Run the MSA module benchmark sweep — called inside a subprocess.

    ``quick_shape``: optional single (batch, seq, prefix) point for CLI
    smoke runs; replaces the sweep but keeps every construction path.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    if compute_dtype != "bfloat16":
        # SGLang derives the attention GEMM precision itself: bf16, or fp8 q with
        # an fp8_e4m3 KV cache on trtllm_mha + SM100 (server_args.py:592-610).
        raise ValueError(f"the MSA plan collects compute bfloat16 only; got compute={compute_dtype}")
    if gemm_type != "bfloat16":
        raise ValueError(
            f"MiniMax-M3 is a BF16 artifact; SGLang's quantized-M3 flows are MXFP8 and the "
            f"M3.1 NVFP4 stack (arg_groups/model_overrides/minimax_m3.py:18-34), "
            f"not {gemm_type!r} (skips.py leaves these cases out)"
        )
    if kv_cache_dtype not in SGLANG_KV_DTYPE:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r}")

    phase = "context" if is_prefill else "generation"

    if quick_shape is not None:
        b, s, prefix = quick_shape
        shapes = [(b, s, prefix)] if is_prefill else [(b, s, 0)]
    else:
        # The worker never re-derives the grid: it runs the points the plan's row lists.
        shapes = [tuple(shape) for shape in inner_manifest]

    if not shapes:
        raise RuntimeError(
            f"MSA module {phase} has no points: the plan queues no empty row; model={model_path}, "
            f"heads={num_heads}, batch_filter={batch_size_filter} (ad-hoc runs: --quick)"
        )

    # A point this device cannot run is refused with its reason before anything is
    # allocated for it, and the ModelRunner is sized for the points that remain.
    refused: dict[tuple, str] = {}
    if is_prefill:
        # SGLang derives its prefill chunk from device memory alone
        # (_handle_gpu_memory_settings, arg_groups/memory_hook.py:89-176); serving
        # chunks a longer prefill.
        chunk = int(serving_memory_defaults(tp_size=1)[0])
        refuse(shapes, refused, lambda shape: over_chunk(shape, chunk))
    runnable = [shape for shape in shapes if shape not in refused]
    if not runnable:
        raise RuntimeError(refusal_summary(refused, len(shapes)))

    # 128 is the M3 serving page on SM90/SM100 (arg_groups/model_overrides/
    # minimax_m3.py:163-168, 181-187), 1 elsewhere; page-128 rounding is an upper
    # bound for any smaller resolved page. The pool SGLang builds is checked
    # against the real page below.
    page_size_guess = 128
    max_total_tokens = max(
        required_kv_alloc_tokens(b, s, p, page_size_guess, is_prefill=is_prefill) for (b, s, p) in runnable
    )
    max_total_tokens += max(1024, max_total_tokens // 20)
    max_running_requests = max(b for (b, _s, _p) in runnable)

    print(
        f"\n{'=' * 60}\nMSA Module {phase}: model={model_path}, heads={num_heads}, "
        f"target_tp={target_tp_size}, kv={kv_cache_dtype}, gemm={gemm_type}, "
        f"shapes={len(shapes)}, GPU={gpu_id}\n{'=' * 60}"
    )

    cleanup_distributed()
    torch.cuda.empty_cache()

    ok = 0
    failures: list[str] = []
    try:
        model_runner = load_model_runner(
            model_path=model_path,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            target_tp_size=target_tp_size,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=None,
            max_running_requests=max_running_requests,
            device=device,
        )

        # Refuse what the built runner cannot run: a prefill beyond the chunk it
        # resolved, or a point beyond its KV pool.
        if is_prefill:
            runtime_chunk = runtime_chunk_size(model_runner)
            print(f"[MSA] SGLang runtime chunked_prefill_size={runtime_chunk}")
            refuse(shapes, refused, lambda shape: over_chunk(shape, runtime_chunk))
        capacity = kv_pool_capacity_tokens(model_runner)
        if capacity is not None:
            page_size = kv_pool_page_size(model_runner)
            refuse(shapes, refused, lambda shape: over_kv_pool(shape, page_size, capacity, is_prefill))
        runnable = [shape for shape in shapes if shape not in refused]
        if not runnable:
            raise RuntimeError(refusal_summary(refused, len(shapes)))

        attention_module = model_runner.model.model.layers[0].self_attn
        device_name = torch.cuda.get_device_name(device)
        perf_filename = _resolve_perf_path(output_path, f"msa_{phase}_module_perf.txt")
        op_name = f"msa_{phase}_module"

        for i, (b, s, p) in enumerate(runnable):
            label = f"b={b}, s={s}, prefix={p}" if is_prefill else f"b={b}, kv={s}"
            print(f"[{i + 1}/{len(runnable)}] {phase} {label}, heads={num_heads}")
            try:
                if is_prefill:
                    latency, power_stats = _run_prefill_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        p,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                    )
                    kernel_source = _prefill_kernel_source(model_runner)
                    isl, step = s, p
                else:
                    latency, power_stats, kernel_source = _run_decode_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        target_tp_size,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                    )
                    isl, step = 1, s
                _log_msa_row(
                    perf_filename=perf_filename,
                    model_path=model_path,
                    compute_dtype=compute_dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    gemm_type=gemm_type,
                    num_heads=num_heads,
                    batch_size=b,
                    isl=isl,
                    step=step,
                    target_tp_size=target_tp_size,
                    latency_ms=latency,
                    op_name=op_name,
                    kernel_source=kernel_source,
                    device_name=device_name,
                    power_stats=power_stats,
                )
                print(f"  {label}: {latency:.4f} ms [{kernel_source}]")
                ok += 1
            except PerfLogWriteError:
                raise
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                print(f"  OOM at {label}: {e}")
                failures.append(f"{label}: OOM: {e}")
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                error_str = str(e).lower()
                if "cuda" in error_str and "illegal" in error_str:
                    # CUDA context is poisoned; stop to preserve prior rows.
                    failures.append(f"{label}: CUDA illegal access: {e}")
                    raise RuntimeError(
                        f"CUDA illegal access at {label}; aborting subprocess. "
                        f"progress ok={ok} failed={len(failures)} total={len(runnable)}"
                    ) from e
                failures.append(f"{label}: {type(e).__name__}: {e}")

        summary = f"ok={ok} error={len(failures)} refused={len(refused)} total={len(shapes)}"
        print(f"[MSA] {phase} {summary}")
        print(f"[MSA timing] {phase}: {render_phase_totals(_POINT_PHASE_TOTALS)}")
        if ok == 0:
            raise RuntimeError(f"MSA module {phase} persisted no rows; {summary}")
        if failures:
            details = "\n- ".join(failures)
            raise RuntimeError(f"MSA module {phase} failed strict completeness: {summary}; failures:\n- {details}")
        if refused:
            raise RuntimeError(refusal_summary(refused, len(shapes)))
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        gc.collect()


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


def _run_msa_subprocess(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None,
    batch_size_filter: int | None,
    target_tp_size: int,
    inner_manifest=(),
):
    """Run one MSA sweep in a subprocess with CUDA_VISIBLE_DEVICES isolation
    (same pattern as collect_mla_module._run_mla_subprocess: SGLang's
    ModelRunner/NCCL state cannot be re-initialized in-process)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    batch_repr = "None" if batch_size_filter is None else str(batch_size_filter)
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_msa_module import run_msa_module\n"
        f'run_msa_module({num_heads}, "{model_path}", "{kv_cache_dtype}", '
        f'"{compute_dtype}", "{gemm_type}", {is_prefill}, 0, {output_repr}, '
        f"{batch_repr}, {target_tp_size}, None, {inner_manifest!r})\n"
    )

    _timeout_env = os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC")
    subprocess_timeout = int(_timeout_env) if _timeout_env else None
    try:
        # beats the stall watchdog while the child prints progress; the child dies with the worker
        returncode, stdout = run_measurement_subprocess(
            [sys.executable, "-c", code],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=subprocess_timeout,
        )
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        if exc.output:
            print(exc.output.decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"MSA module {phase} subprocess timed out after {subprocess_timeout}s "
            f"(heads={num_heads}, model={model_path}, kv={kv_cache_dtype})"
        ) from exc

    if returncode != 0:
        tail = ""
        lines = []
        if stdout:
            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-30:])
        # Keep the kernel-limit classification of points the device refused.
        error = kernel_limit_error(lines)
        if error is not None:
            raise RuntimeError(error)
        raise RuntimeError(
            f"MSA module {phase} subprocess failed (exit code {returncode})\n"
            f"--- subprocess output (last 30 lines) ---\n{tail}"
        )


def run_msa_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    target_tp_size: int = 1,
    inner_manifest=(),
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py.

    Positional prefix matches the trtllm MSA worker; seq_len is a 0
    placeholder. ``inner_manifest`` is the [batch, seq_len, prefix_len] points
    the plan's row lists (op_plans._sglang_msa_rows); the subprocess runs
    them, or refuses one with its reason, and never re-derives the grid.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename
    batch_size_filter = batch_size if is_prefill and batch_size > 0 else None

    print(f"\n{'=' * 60}")
    print(
        f"MSA Module {'Context' if is_prefill else 'Generation'}: model={model_path}, "
        f"heads={num_heads}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"batch_filter={batch_size_filter or 'all'}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    output_path = os.path.dirname(perf_filename) or os.getcwd()

    _run_msa_subprocess(
        num_heads=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
        batch_size_filter=batch_size_filter,
        target_tp_size=target_tp_size,
        inner_manifest=tuple(tuple(shape) for shape in inner_manifest),
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for SGLang")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default="MiniMaxAI/MiniMax-M3")
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--target-tp-size", type=int, default=None, help="defaults to 64 // num_heads")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8"], default="bfloat16")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="run a single (batch, seq[, prefix]) point in-process")
    args = parser.parse_args()

    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    target_tp = args.target_tp_size or max(1, 64 // args.num_heads)
    is_prefill = args.mode == "context"

    if args.quick:
        run_msa_module(
            num_heads=args.num_heads,
            model_path=args.model,
            kv_cache_dtype=args.kv_cache_dtype,
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=is_prefill,
            gpu_id=gpu_id,
            output_path=args.output_path,
            batch_size_filter=None,
            target_tp_size=target_tp,
            quick_shape=(
                args.batch_size or (2 if is_prefill else 4),
                args.seq_len or (512 if is_prefill else 2048),
                args.prefix_len,
            ),
        )
        return

    # Full sweeps run the plan's rows, so a CLI reproduction runs the exact planned points.
    for _0, batch, heads, kv_dtype, compute, gemm, model, target_tp, points in plan(
        f"msa_{args.mode}_module", "sglang", get_sm_version()
    ).rows:
        if model != args.model or heads != args.num_heads or kv_dtype != args.kv_cache_dtype:
            continue
        if args.batch_size is not None and batch != args.batch_size:
            continue
        run_msa_module(
            num_heads=heads,
            model_path=model,
            kv_cache_dtype=kv_dtype,
            compute_dtype=compute,
            gemm_type=gemm,
            is_prefill=is_prefill,
            gpu_id=gpu_id,
            output_path=args.output_path,
            batch_size_filter=batch or None,
            target_tp_size=target_tp,
            inner_manifest=points,
        )


if __name__ == "__main__":
    main()
