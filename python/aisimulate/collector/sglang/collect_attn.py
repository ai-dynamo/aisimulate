# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang dense attention collector.

Builds lightweight RadixAttention/ForwardBatch mocks to benchmark context and
generation attention without launching a full SGLang server. The cases and
the backend each runs on come from the shared attention plan
(collector/op_plans.py); this file checks that backend against the one SGLang
serving resolves for the row's architecture on the device, and owns SGLang
backend construction, KV-cache setup and perf logging for the SGLang runtime.
"""

__compat__ = "sglang==0.5.23"

import math
import os
from importlib.metadata import version as _dist_version  # setuptools/pkg_resources is absent from sglang>=0.5.21 images
from types import SimpleNamespace
from typing import NamedTuple

import torch
from collector.case_generator import get_attention_architecture_models, sglang_default_attention_backend
from collector.helper import _resolve_local_model_path, benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_buffer, get_parallel

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


class Timing(NamedTuple):
    mean: float


# Mock objects to satisfy RadixAttention dependencies. Citations are sglang v0.5.23.
class MockModelConfig:
    # model_config.py:1692; read by flashinfer_backend.py and triton_backend.py
    def get_max_num_attention_heads(self) -> int:
        return int(self.num_attention_heads)

    def __init__(
        self,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        v_head_dim,
        architecture,
        runtime_window_size,
        attention_chunk_size,
    ):
        self.is_encoder_decoder = False
        self.context_len = 32768
        self.attention_arch = AttentionArch.MHA
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.swa_head_dim = head_dim
        self.swa_v_head_dim = v_head_dim
        self.sliding_window_size = runtime_window_size if runtime_window_size >= 0 else None
        self.attention_chunk_size = attention_chunk_size
        self.is_hybrid_swa = runtime_window_size >= 0 or attention_chunk_size is not None
        self.swa_attention_layer_ids = [0] if self.is_hybrid_swa else []
        self.full_attention_layer_ids = [] if self.is_hybrid_swa else [0]
        self.is_multimodal = False
        self.hidden_size = num_attention_heads * head_dim
        self.is_local_attention_model = attention_chunk_size is not None
        # TritonAttnBackend asks mambaish_config(model_config) (triton_backend.py:298),
        # which reads model_config.linear_attn_registry_result (configs/hybrid_arch.py:35-36,
        # a ModelConfig property at model_config.py:1259) to detect hybrid linear-attention
        # archs; the dense-attention mock has none.
        self.linear_attn_registry_result = None

        class MockHFConfig:
            def __init__(self, *, num_attention_heads, num_key_value_heads, head_dim, v_head_dim, architecture):
                self.architectures = [architecture or "LlamaForCausalLM"]
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                self.head_dim = head_dim
                self.v_head_dim = v_head_dim
                self.swa_head_dim = head_dim
                self.swa_v_head_dim = v_head_dim
                self.hidden_size = num_attention_heads * head_dim
                self.attn_logit_softcapping = None

            def get_text_config(self):
                # HF PretrainedConfig API: hybrid_gdn_config unwraps multimodal configs
                # through it (configs/hybrid_arch.py:55-56, reached from
                # triton_backend.py:298); a text-only mock is its own text config.
                return self

        self.hf_config = MockHFConfig(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            architecture=architecture,
        )
        self.hf_text_config = self.hf_config
        self.dtype = torch.bfloat16

    # model_config.py:1700 adds dcp_size (DCP replicates KV); the collector runs no DCP
    def get_num_kv_heads(self, tp_size, dcp_size=1):
        return self.num_key_value_heads // tp_size


class MockServerArgs:
    # The only server-args field a dense attention backend reads off the runner:
    # FlashInferAttnBackend builds DllmConfig.from_server_args(model_runner.server_args)
    # (flashinfer_backend.py:348 -> dllm/config.py:43-45), None for every collected model.
    # Every other server-args value the backends read comes from the published runtime
    # context (ensure_offline_runtime_published: get_exec()/get_spec()/get_schedule()).
    dllm_algorithm = None


class MockModelRunner:
    """The ModelRunner fields the v0.5.23 dense backends read (flashattention_backend.py,
    flashinfer_backend.py, triton_backend.py, trtllm_mha_backend.py), each set where
    serving's ModelRunner sets it."""

    # model_runner.py:876-895: logits rows per decode slot, 1 without speculative decoding
    # (read at triton_backend.py:229,233)
    def decode_num_tokens_per_req(self, *, num_draft_tokens=None) -> int:
        return 1

    def __init__(
        self,
        device,
        kv_cache_dtype="auto",
        page_size: int = 64,
        num_heads=None,
        num_kv_heads=None,
        head_dim=None,
        v_head_dim=None,
        architecture=None,
        runtime_window_size=-1,
        attention_chunk_size=None,
        attn_backend_name=None,
    ):
        self.device = device
        # Set once the pools exist (ModelRunner.init_memory_pool / init_kv_index_translator).
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.token_to_kv_pool_allocator = None
        self.kv_index_translator = None
        self.server_args = MockServerArgs()
        # One lane = one backend for both phases (model_runner.py:1120 resolved.prefill /
        # .decode); TRTLLMHAAttnBackend (trtllm_mha_backend.py:218-225) and
        # FlashInferAttnBackend (flashinfer_backend.py:358-359) gate their kv-access
        # checks on it.
        self.prefill_attention_backend_str = attn_backend_name
        self.decode_attention_backend_str = attn_backend_name
        # trtllm_mha_backend.py:325-328 scans model_runner.model.modules() for ENCODER_ONLY
        # attention layers (expanded TARGET_VERIFY metadata); an empty Module declares none,
        # which is the dense-decoder answer.
        self.model = torch.nn.Module()
        # TRTLLMHAAttnBackend sizes its multi-CTA KV counter from it (trtllm_mha_backend.py:
        # 396-401, floored at 8192 requests by trtllm_mla_backend.py:113-119); 512 = the
        # generator's --max-running-requests default.
        self.max_running_requests = 512
        # The backends read spec_algorithm only on a draft worker (flashinfer_backend.py:601-602,
        # trtllm_mha_backend.py:561-562, triton_backend.py:253-256); the collector runs none.
        self.is_draft_worker = False
        self.spec_algorithm = None
        # model_runner.py:1353 -> load_model_utils.py:144-157 resolve_sliding_window_size: the
        # model's window, else a chunked-local model's (Llama 4) attention chunk, which
        # FlashAttentionBackend's local-attention builder also reads (flashattention_backend.py:
        # 386-398; runner.attention_chunk_size at model_runner.py:407).
        self.sliding_window_size = attention_chunk_size
        if self.sliding_window_size is None and runtime_window_size >= 0:
            self.sliding_window_size = runtime_window_size
        self.attention_chunk_size = attention_chunk_size
        self.model_config = MockModelConfig(
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            architecture,
            runtime_window_size,
            attention_chunk_size,
        )
        # The KV-cache torch dtype; the caller replaces the string with it.
        self.kv_cache_dtype = kv_cache_dtype
        # FlashAttentionBackend reads the server-args string next to the torch dtype
        # (flashattention_backend.py:191-192, drives the mxfp8 branch); serving sets it from
        # server_args.kv_cache_dtype (model_runner.py:1572). The collector's kv_cache_dtype is
        # that same string.
        self.kv_cache_dtype_str = kv_cache_dtype
        self.page_size = page_size
        self.dtype = torch.bfloat16  # model_runner.py:1362 (model dtype; trtllm/flashinfer q dtype)
        self.gpu_id = 0  # model_runner.py:366; triton_backend.py:315 reads its SM count


# The model a base-grid row (no architecture) is checked against. Qwen3ForCausalLM
# has no provider under arg_groups/model_overrides/ and no MODEL_OVERRIDES entry
# (arg_groups/model_override_base.py:18-22), no attention sinks and equal K and V
# widths, so serving resolves get_default_attn_backend's choice for a plain MHA
# model (model_override_base.py:277-334).
DEFAULT_BACKEND_MODEL = "Qwen/Qwen3-32B"

# architecture (None: the base grid) -> ((prefill, decode) or None, the model resolved), or the failure.
_SERVING_BACKENDS: dict = {}


def _architecture_model(architecture: str | None) -> str:
    """The first model of an architecture's attention profiles whose config resolves (bundled, else fetched)."""
    if architecture is None:
        return DEFAULT_BACKEND_MODEL
    models = get_attention_architecture_models(architecture)
    for model in models:
        try:
            _resolve_local_model_path(model)
        except FileNotFoundError:
            continue
        return model
    raise FileNotFoundError(f"no model of {architecture} has a config (attention profiles declare {models})")


def resolve_serving_attention_backends(model: str) -> tuple[str, str] | None:
    """The (prefill, decode) attention backends SGLang serving resolves for a model on this device.

    The model's ServerArgs go through serving's own resolution
    (ServerArgs.resolve_once, srt/server_args.py:244-300, as launch_server.py:29
    calls it): the model's override provider declares its backend
    (arg_groups/model_hook.py:233-237 collect_model_override_declarations) and
    the default fills what is left (arg_groups/attention_hook.py:120
    _attention_backend_default -> get_default_attn_backend). The pair is read as
    the pipeline reads it before publishing (model_override_base.py:138-153
    attention_backends_of; a runner takes the same pair, model_executor/
    model_runner_components/attention_backend_setup.py:184-204). The record is
    never published: the worker keeps its offline runtime.

    None for a model whose own attention is MLA (Kimi-K3): its dense rows time
    the speculative draft, which ``attention_backend`` does not describe.

    Needs the SGLang runtime and the device: the platform facts the providers
    read are probed from the GPU.
    """
    from sglang.srt.arg_groups.model_override_base import attention_backends_of, resolved_view, use_mla_backend
    from sglang.srt.server_args import ServerArgs

    # The resolved directory may hold config.json only: "dummy" is serving's load
    # format for a run without weights, as the module collectors use it.
    server_args = ServerArgs(model_path=_resolve_local_model_path(model), load_format="dummy", trust_remote_code=True)
    server_args.resolve_once()
    if use_mla_backend(server_args):
        return None
    prefill, decode = attention_backends_of(resolved_view(server_args))
    return prefill, decode


def serving_attention_backends(architecture: str | None) -> tuple[tuple[str, str] | None, str]:
    """An architecture's resolved (prefill, decode) backends and the model they were resolved for, once per process."""
    if architecture not in _SERVING_BACKENDS:
        try:
            model = _architecture_model(architecture)
            _SERVING_BACKENDS[architecture] = (resolve_serving_attention_backends(model), model)
        except Exception as exc:
            _SERVING_BACKENDS[architecture] = (
                f"cannot resolve the attention backend SGLang serves {architecture or 'a model without an override'} "
                f"on: {type(exc).__name__}: {exc}"
            )
    resolved = _SERVING_BACKENDS[architecture]
    if isinstance(resolved, str):
        raise RuntimeError(resolved)
    return resolved


def check_dispatch_is_servings(attn_backend_name: str, architecture: str | None, is_context_phase: bool) -> None:
    """Raise when the plan's dispatch is not the backend SGLang serving resolves for the row's architecture.

    The plan picks the backend from case_generator.SGLANG_ATTENTION_BACKENDS and
    the default table, which a person keeps in step with SGLang's overrides;
    this is the check against the framework. Context rows time prefill and
    generation rows decode, which serving may put on different backends.
    """
    backends, model = serving_attention_backends(architecture)
    if backends is None:
        return
    phase, served = ("prefill", backends[0]) if is_context_phase else ("decode", backends[1])
    if served != attn_backend_name:
        raise RuntimeError(
            f"the plan dispatches {architecture or 'base-grid'} {phase} attention on {attn_backend_name}, "
            f"but SGLang serving resolves {served} for {model} on this device"
        )


def create_req_to_token_pool(batch_size, total_len, page_size, torch_device, device_str):
    """Create req_to_token mapping consistent with test_flashattn_backend.py."""
    assert total_len > 0, "Total sequence length must be positive"
    pool = ReqToTokenPool(
        size=batch_size,
        max_context_len=total_len,
        device=device_str,
        enable_memory_saver=False,
    )
    req_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device).view(batch_size, 1)
    token_offsets = torch.arange(total_len, dtype=torch.int32, device=torch_device).view(1, total_len)
    token_matrix = (req_indices * total_len) + token_offsets + page_size
    pool.req_to_token[:batch_size, :total_len] = token_matrix
    return pool, token_matrix.contiguous()


def get_context_attention_test_cases():
    return plan_cases("attention_context", "sglang", get_sm_version())


def get_generation_attention_test_cases():
    return plan_cases("attention_generation", "sglang", get_sm_version())


_FLASHINFER_WORKSPACE_MIN_BYTES = 4 << 30


def _ensure_flashinfer_workspace(device) -> int:
    """Give this process SGLang's shared FlashInfer workspace at the collector's floor
    BEFORE the first FlashInfer-family backend (FlashInferAttnBackend, or
    TRTLLMHAAttnBackend, which subclasses it) is built. Returns the workspace size in bytes.

    sglang v0.5.23 sizes the shared decode/prefill plan buffer from
    ``envs.SGLANG_FLASHINFER_WORKSPACE_SIZE`` (EnvInt, default 384 MiB, environ.py:1221)
    and allocates it ONCE per process through ``get_buffer("flashinfer_workspace")``
    (flashinfer_backend.py:456-463; runtime_context.py:1014-1020 keeps the first
    allocation). The backend constructor first resets the knob to 512 MiB for
    Qwen2/Qwen3/MiMo/Qwen3-VL architectures (flashinfer_backend.py:421-431), and the
    collector's FlashInfer rows carry ``Qwen3ForCausalLM`` and ``MiMoForCausalLM``: raising
    the knob alone left the whole worker on 512 MiB whenever such a row built its first
    backend. So the buffer is created here, under serving's own name, and the knob is
    raised too (TRTLLMHAAttnBackend sizes its own workspace from the knob when it is set,
    trtllm_mha_backend.py:200-207).

    Why a floor at all: the decode sweep's wide cells need more than the default — l40s
    0.5.21: "Buffer overflow when allocating batch_prefill_tmp_v with size 536870912 ...
    only 402653184 bytes available", 671 attention_generation cases (heads 128 / kv 1 /
    head_dim 128, batch 32 x isl 255..4095). 1 GiB (5c0f54a5) cleared 447 of them
    (pipeline 72369799, job 475655072); the remaining 224 — bf16 and fp8 KV alike, batch x
    isl from 32x255 up to 4x16383 — asked the tensor-core decode planner for
    batch_prefill_tmp_v of 1.5-2.22 GiB. 4 GiB covers the largest observed request with
    margin. The buffer is one torch.empty per worker process, never touched beyond what
    the plan uses, and neither the FlashInfer planner's split decision (it sizes tmp_v
    from the split count) nor the trtllm-gen kernel choice (the workspace only carries
    the multi-CTA scratch, flashinfer csrc/trtllm_fmha_kernel_launcher.cu:304-370 @v0.7.2)
    depends on the buffer size, so the measured kernels are unchanged.
    """
    from sglang.srt.environ import envs

    knob = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
    floor = max(int(knob.get()), _FLASHINFER_WORKSPACE_MIN_BYTES)
    knob.set(floor)
    workspace = get_buffer(
        "flashinfer_workspace",
        lambda: torch.empty(floor, dtype=torch.uint8, device=device),
    )
    return int(workspace.numel())


@get_parallel().override(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
)
def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,
    head_dim,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    window_size=0,
    v_head_dim=None,
    runtime_window_size=None,
    attention_chunk_size=None,
    has_attention_sink=False,
    scaling=None,
    attn_backend_name=None,
    architecture=None,
    *,
    perf_filename,
    device="cuda:0",
    page_size: int | None = None,
):
    from collector.sglang.runtime_compat import ensure_offline_runtime_published

    ensure_offline_runtime_published()  # sglang>=0.5.20 backends read get_exec()/get_parallel()
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache, "If you want to use fp8 context fmha, kv cache must be fp8"
    kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16

    torch_device = torch.device(device)
    device_str = str(torch_device)

    explicit_v_head_dim = v_head_dim is not None
    if head_dim == 192 and (
        not explicit_v_head_dim or architecture != "MiMoV2FlashForCausalLM" or attn_backend_name is None
    ):
        raise ValueError("head_dim=192 requires the explicit MiMo-V2 SGLang attention contract")

    v_head_dim = head_dim if v_head_dim is None else v_head_dim
    if runtime_window_size is None:
        runtime_window_size = window_size or -1

    if attn_backend_name is None:
        # The plan names the backend in every row; a direct call takes SGLang's default.
        sm_version = get_sm_version()
        attn_backend_name = sglang_default_attention_backend(sm_version, has_attention_sink)
        if attn_backend_name is None:
            raise ValueError(f"No SGLang attention backend for SM{sm_version}")
    check_dispatch_is_servings(attn_backend_name, architecture, is_context_phase)
    if page_size is None:
        # The page size serving resolves for the backend: trtllm_mha forces 64 when unset
        # (arg_groups/overrides.py:1174-1187 _mla_backend_page_constraints), fa4 on SM100/103
        # forces 128 (:1273-1297 _fa4_page_constraint, get_platform().is_sm100 = capability
        # major 10); the others keep the default 1.
        if attn_backend_name == "trtllm_mha":
            page_size = 64
        elif attn_backend_name == "fa4" and get_sm_version() // 10 == 10:
            page_size = 128
        else:
            page_size = 1

    # Inputs are BF16 on every lane, as serving hands attention the model-dtype Q/K/V after
    # RoPE; the backend's own per-call FP8 work stays in the timed call. With an FP8 KV
    # cache, fa3 (head_dim <= 256) casts Q in prepare_paged_mha_query
    # (flashattention_backend.py:1426-1436) and set_kv_buffer casts K/V
    # (mem_cache/memory_pool.py:2926-2933); trtllm_mha quantizes Q and writes K/V in one
    # fused kernel (trtllm_mha_backend.py:1058-1100, called at :1611 and :1718). fa4 prefill
    # (flashattention_backend.py:1430), flashinfer and triton never quantize Q, so their
    # prefill computes BF16 against the FP8 cache. Rows are labeled with the compute dtype
    # the backend runs; the plan leaves out the labels a lane cannot produce
    # (collector/skips.py).

    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype="fp8_e4m3" if use_fp8_kv_cache else "auto",
        page_size=page_size,
        num_heads=num_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        architecture=architecture,
        runtime_window_size=runtime_window_size,
        attention_chunk_size=attention_chunk_size,
        attn_backend_name=attn_backend_name,
    )
    model_runner.kv_cache_dtype = kvtype

    total_len = input_len if is_context_phase else input_len + 1
    # TRTLLM MHA sizes its page table from context_len.
    model_runner.model_config.context_len = max(model_runner.model_config.context_len, total_len)
    req_to_token_pool, token_matrix = create_req_to_token_pool(
        batch_size=batch_size,
        total_len=total_len,
        page_size=model_runner.page_size,
        torch_device=torch_device,
        device_str=device_str,
    )
    model_runner.req_to_token_pool = req_to_token_pool

    total_tokens = batch_size * total_len
    kv_cache_size = max(
        model_runner.page_size,
        math.ceil(total_tokens / model_runner.page_size) * model_runner.page_size,
    )
    kv_pool = MHATokenToKVPool(
        size=kv_cache_size,
        page_size=model_runner.page_size,
        dtype=kvtype,
        head_num=num_key_value_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device=device_str,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool
    # The backends ask the allocator only for its pool (flashinfer_backend.py:596-606 and
    # trtllm_mha_backend.py:548-568 _resolve_swa_kv_pool); a static MHA pool is not an SWA pool.
    model_runner.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=model_runner.page_size,
        get_kvcache=lambda: model_runner.token_to_kv_pool,
    )
    # ModelRunner.init_kv_index_translator (model_runner.py:950-961): over a static pool the
    # translator translates nothing, and every backend reads its page tables through it.
    model_runner.kv_index_translator = KVIndexTranslator(
        req_to_token=req_to_token_pool.req_to_token,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        token_to_kv_pool=kv_pool,
        page_size=model_runner.page_size,
        device=device_str,
        is_draft_worker=model_runner.is_draft_worker,
    )

    # Each lane builds the backend class serving's factory builds for its name
    # (layers/attention/attention_registry.py:55-77, 206-214, 238-284).
    if attn_backend_name == "flashinfer":
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        _ensure_flashinfer_workspace(torch_device)
        attn_backend = FlashInferAttnBackend(model_runner)
    elif attn_backend_name == "trtllm_mha":
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        _ensure_flashinfer_workspace(torch_device)  # TRTLLMHAAttnBackend subclasses FlashInferAttnBackend
        attn_backend = TRTLLMHAAttnBackend(model_runner)
    elif attn_backend_name == "triton":
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        attn_backend = TritonAttnBackend(model_runner)
    elif attn_backend_name == "fa3":
        attn_backend = FlashAttentionBackend(model_runner)
    elif attn_backend_name == "fa4":
        # create_flashattention_v4_backend (attention_registry.py:264-274): the FA3 class on
        # the FA4 kernels (flashattention_backend.py:290-316).
        attn_backend = FlashAttentionBackend(model_runner, fa_impl_ver=4)
    else:
        raise ValueError(f"Unsupported SGLang attention backend: {attn_backend_name}")

    layer = RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5 if scaling is None else scaling,
        num_kv_heads=num_key_value_heads,
        layer_id=0,
        v_head_dim=v_head_dim,
        sliding_window_size=runtime_window_size,
        use_irope=attention_chunk_size is not None and window_size > 0,
    ).to(torch_device)

    sinks = None
    if has_attention_sink:
        sinks_dtype = torch.float32 if attn_backend_name == "trtllm_mha" else torch.bfloat16
        sinks = torch.randn(num_heads, device=torch_device, dtype=sinks_dtype)

    seqlen_q = input_len if is_context_phase else 1
    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=torch_device,
        dtype=torch.bfloat16,
    )

    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device)

    if is_context_phase:
        forward_mode = ForwardMode.EXTEND
        k = torch.randn(
            batch_size * input_len,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch_size * input_len,
            num_key_value_heads,
            v_head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )

        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=torch_device)
        seq_lens_cpu = seq_lens.cpu()
        prefix_lens = torch.zeros((batch_size,), dtype=torch.int32, device=torch_device)
        out_cache_loc = token_matrix.reshape(-1).to(torch.int32)

        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, input_len, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=seq_lens,
            extend_prefix_lens=prefix_lens,
            # Host lists, as ForwardBatch.init_new fills them (forward_batch_info.py:1240-1241);
            # trtllm_mha's mixed-prefill split walks and sums them (trtllm_mha_backend.py:1316-1327).
            extend_seq_lens_cpu=[input_len] * batch_size,
            extend_prefix_lens_cpu=[0] * batch_size,
            extend_num_tokens=int(seq_lens.sum().item()),
        )
    else:
        forward_mode = ForwardMode.DECODE
        history_len = input_len
        new_token_loc = token_matrix[:, history_len:].reshape(-1).contiguous()
        history_loc = token_matrix[:, :history_len].reshape(-1).contiguous() if history_len > 0 else None

        k = torch.randn(
            batch_size,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch_size,
            num_key_value_heads,
            v_head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )

        seq_lens = torch.full((batch_size,), total_len, dtype=torch.int32, device=torch_device)
        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, 1, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=new_token_loc.to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
        )

        if history_loc is not None and history_loc.numel() > 0:
            cache_k = torch.randn(
                history_loc.numel(),
                num_key_value_heads,
                head_dim,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            cache_v = torch.randn(
                history_loc.numel(),
                num_key_value_heads,
                v_head_dim,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            kv_pool.set_kv_buffer(
                layer,
                history_loc.to(torch.int64),
                cache_k,
                cache_v,
                layer.k_scale,
                layer.v_scale,
            )

    # ForwardBatch.init_new binds every forward to its iteration's KV-location plan
    # (forward_batch_info.py:1139-1141). TRTLLMHAAttnBackend.init_forward_metadata reads its
    # page table through the plan unconditionally (trtllm_mha_backend.py:1341-1343), and
    # KVIndexTranslator.read_table takes plan.req_pool_indices even over a static pool
    # (kv_index_translator.py:499-512): an unbound batch fails every trtllm_mha case.
    translator = model_runner.kv_index_translator
    translator.own_plan(forward_batch).bind(forward_batch, translator)

    with forward_context(ForwardContext(attn_backend=attn_backend)):
        attn_backend.init_forward_metadata(forward_batch)

        # Mirror the serving call contract: only sink-carrying models pass the
        # ``sinks`` kwarg (gpt_oss.py:499, mimo_v2.py:849), and
        # FlashInferAttnBackend.forward_extend/forward_decode do not accept it
        # (flashinfer_backend.py:1357, 1521) — passing ``sinks=None``
        # unconditionally breaks every flashinfer-routed case. A sink profile
        # routed to flashinfer still fails closed with the framework's own TypeError.
        layer_kwargs = {"sinks": sinks} if has_attention_sink else {}

        def run_iter():
            # FIXME(kernel-limit): 37826f10 observed an SM120 illegal-memory
            # access for large Q/O tensors on SGLang 0.5.10 Triton. That is not
            # proof for every 0.5.14 backend; keep the cases attempted until an
            # SM120 source/hardware audit can replace or remove this note.
            layer(q, k, v, forward_batch, **layer_kwargs)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=run_iter,
            num_warmups=3,
            num_runs=20,
            repeat_n=1,
        ) as results:
            pass

    latency = results["latency_ms"]

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    if not log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": "fp8" if use_fp8_context_fmha else "bfloat16",
                "kv_cache_dtype": "fp8" if use_fp8_kv_cache else "bfloat16",
                "step": step,
                "window_size": window_size,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=_dist_version("sglang"),
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=attn_backend_name,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist SGLang attention performance row to {perf_filename}")

    return Timing(latency * 1e-3)
