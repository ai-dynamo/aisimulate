# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang MLA collector.

Builds standalone RadixAttention/ForwardBatch mocks for MLA context and
generation kernels without starting a server. Shared MLA cases come from YAML;
this file owns SGLang MLA backend choice, paged KV-cache setup, runtime
dispatch, and perf logging. Citations are sglang v0.5.23 unless stated.
"""

__compat__ = "sglang==0.5.23"

import gc
import math
import random
from importlib.metadata import version as _dist_version  # setuptools/pkg_resources is absent from sglang>=0.5.21 images

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.registry_types import PerfFile
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel

# DeepSeek MLA dims (the only geometry the MLA plan admits, op_plans._MLA_GEOMETRY).
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
# Scaling follows production: 1 / sqrt(qk_nope + qk_rope)
MLA_SCALING = 1 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)


def _cuda_version_at_least(major: int, minor: int) -> bool:
    if torch.version.cuda is None:
        return False
    version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return version >= (major, minor)


def _select_default_mla_backend() -> str:
    """The MLA attention backend SGLang serving picks for DeepSeek V3 on this SM.

    get_default_attn_backend's MLA branch (arg_groups/model_override_base.py:336-351):
    fa3 on Hopper with CUDA >= 12.3, flashinfer on SM100/103 with CUDA >= 12.8
    (utils/common.py:292-313, capability major 10), triton elsewhere. The DeepSeek
    family override replaces SM100's flashinfer with trtllm_mla when no backend is
    given (arg_groups/model_overrides/deepseek_v2.py:177-188). SM80/86 have no
    mapping; collector/skips.py leaves them out of the plan.
    """
    sm_version = get_sm_version()
    if sm_version in {100, 103} and _cuda_version_at_least(12, 8):
        return "trtllm_mla"
    if sm_version == 90 and _cuda_version_at_least(12, 3):
        return "fa3"
    if sm_version in {89, 90, 100, 103, 120}:
        return "triton"
    raise ValueError(f"No SGLang MLA backend mapping for SM{sm_version}")


def _mla_attention_geometry(backend: str, is_context_phase: bool) -> tuple[int, int, bool]:
    """(qk head dim, v head dim, absorbed) of the attention call serving makes.

    Decode on every backend, and prefill on fa3 / triton, run the absorbed MLA
    method through ``attn_mqa`` (models/deepseek_v2.py:2183-2192: one 576-wide
    latent KV head, 512-wide output; forward_mla.py:770-799). Serving's prefill
    on fa3 and triton is captured by the default breakable prefill CUDA graph
    (model_executor/cuda_graph_config.py:119-123) for every pass up to
    min(8192, chunked_prefill_size) tokens (arg_groups/memory_hook.py:233-238),
    and a captured prefill pins the absorbed method
    (models/deepseek_common/attention_backend_handler.py:108-113, :233-235).

    Prefill on trtllm_mla is the non-absorbed MHA call: DeepseekV3ForCausalLM on
    trtllm_mla runs prefill eagerly (arg_groups/cuda_graph_hook.py:279-312, applied
    at pipeline.py:245), where handle_attention_trtllm_mla picks MHA_CHUNKED_KV for
    a prefix-free extend (attention_backend_handler.py:171-179) and the model calls
    ``attn_mha`` (deepseek_v2.py:2206-2215: H KV heads of qk_nope + qk_rope = 192,
    v_head_dim 128) with save_kv_cache=False (forward_mha.py:383-386).

    Deviation (fa3 / triton prefill above the breakable graph's token buckets):
    a pass larger than min(8192, chunked_prefill_size) tokens replays eagerly and
    takes MHA_ONE_SHOT / MHA (attention_backend_handler.py:125-138, :247-251), a
    192/128 MHA kernel this lane does not time. Serving forms such a pass only with
    --chunked-prefill-size above 8192 (its default is 4096 on 48 GB and 8192 on
    80-160 GB devices, memory_hook.py:118-153), so the context rows for
    batch x isl > 8192 measure the absorbed kernel, which is more expensive than
    the MHA one there. Retire by timing those cells on the MHA call.
    """
    if backend == "trtllm_mla" and is_context_phase:
        return QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM, V_HEAD_DIM, False
    if backend in {"trtllm_mla", "fa3", "triton"}:
        return KV_LORA_RANK + QK_ROPE_HEAD_DIM, KV_LORA_RANK, True
    raise ValueError(f"No SGLang MLA attention geometry for backend {backend!r}")


def _mla_compute_dtype(backend: str, kv_cache_dtype: torch.dtype) -> str:
    # trtllm_mla quantizes the prefill Q/K/V to FP8 when self.data_type (the KV
    # dtype) is FP8 (trtllm_mla_backend.py:1080-1081), and decode quantizes Q with
    # the fused RoPE (:1245-1277). BF16 input tensors therefore do not imply BF16
    # kernel compute. fa3's absorbed MLA has head_dim=576: prepare_paged_mha_query
    # casts Q to the KV dtype only for head_dim <= 256 (flashattention_backend.py:
    # 1426-1436), and the absorbed path casts the KV cache to Q's BF16
    # (:1944, :2350). Triton also computes in BF16.
    if backend == "trtllm_mla" and kv_cache_dtype == torch.float8_e4m3fn:
        return "fp8"
    if backend in {"trtllm_mla", "fa3", "triton"}:
        return "bfloat16"
    raise ValueError(f"No SGLang MLA compute-precision mapping for backend {backend!r}")


class MockModelConfig:
    """The ModelConfig fields the v0.5.23 MLA backends read, at serving's DeepSeek-V3
    values (configs/model_config.py:1380-1416 for the MLA branch)."""

    def get_max_num_attention_heads(self) -> int:  # model_config.py:1692 (triton_backend.py:266)
        return int(self.num_attention_heads)

    def __init__(self, num_attention_heads: int, scaling: float):
        self.attention_arch = AttentionArch.MLA
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = KV_LORA_RANK
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        # triton_backend.py:293-296 compares it with v_head_dim next to a sliding
        # window only; the MLA model has none.
        self.swa_v_head_dim = V_HEAD_DIM
        self.head_dim = 256  # model_config.py:1396
        self.scaling = scaling
        self.context_len = 32768
        self.is_encoder_decoder = False
        self.is_local_attention_model = False
        self.is_draft_model = False
        # TritonAttnBackend asks mambaish_config(model_config) (triton_backend.py:298),
        # which reads model_config.linear_attn_registry_result and hf_config
        # (configs/hybrid_arch.py:35-36, :145-155) to detect hybrid linear-attention
        # archs; an MLA module is none of them.
        self.linear_attn_registry_result = None

        class _MockHFConfig:
            architectures = ["DeepseekV3ForCausalLM"]  # noqa: RUF012
            model_type = "deepseek_v3"

            def get_text_config(self):
                return self

        self.hf_config = _MockHFConfig()
        # flashattention_backend.py:326-328 reads num_attention_heads off it.
        self.hf_text_config = self

    def get_num_kv_heads(self, tp_size: int, dcp_size: int = 1):  # model_config.py:1700: MLA has one
        return 1


class MockModelRunner:
    """The ModelRunner fields the v0.5.23 MLA backends read (trtllm_mla_backend.py,
    flashinfer_mla_backend.py, flashattention_backend.py, triton_backend.py), each
    set where serving's ModelRunner sets it."""

    # model_runner.py:876-899: logits rows per decode slot, 1 without speculative
    # decoding (read at triton_backend.py:229,233)
    def decode_num_tokens_per_req(self, *, num_draft_tokens=None) -> int:
        return 1

    def __init__(
        self,
        device: torch.device,
        kv_cache_dtype: torch.dtype,
        page_size: int,
        num_attention_heads: int,
        scaling: float,
    ):
        self.device = device
        self.gpu_id = device.index if device.index is not None else torch.cuda.current_device()
        self.kv_cache_dtype = kv_cache_dtype
        # model_runner.py:1572: the server-args spelling the FA3 backend reads
        # (flashattention_backend.py:191-192)
        self.kv_cache_dtype_str = "fp8_e4m3" if kv_cache_dtype == torch.float8_e4m3fn else "auto"
        self.dtype = torch.bfloat16  # model_runner.py:1362
        self.page_size = page_size
        self.model_config = MockModelConfig(num_attention_heads, scaling)
        # Set once the pools exist (ModelRunner.init_memory_pool / init_kv_index_translator).
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        # Serving's MLA allocator is a plain paged allocator; the only reader in these
        # lanes stores it (triton_backend.py:221), and KVIndexTranslator only asks
        # whether it is a unified one (kv_index_translator.py:135-140).
        self.token_to_kv_pool_allocator = None
        self.kv_index_translator = None
        self.sliding_window_size = None  # model_runner.py:1353: DeepSeek V3 has no window
        # TRTLLMMLABackend sizes its multi-CTA KV counter from it (trtllm_mla_backend.py:
        # 269-275); 512 = the generator's --max-running-requests default.
        self.max_running_requests = 512
        # The backends read spec_algorithm only on a draft worker; the collector runs none.
        self.is_draft_worker = False
        self.spec_algorithm = None


def create_req_to_token_pool(
    batch_size: int,
    total_len: int,
    page_size: int,
    torch_device: torch.device,
    device_str: str,
) -> tuple[ReqToTokenPool, torch.Tensor]:
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


def benchmark_layer(layer, attn_backend, forward_batch, q, k, v, **layer_kwargs):
    # CUDA-graph replay for every lane: a capture failure raises (helper default
    # allow_graph_fail=False) and an eager result is refused below, so a kernel
    # that stops being capturable fails the case instead of shipping its launch
    # floor (sglang mla_context on B200/GB200, data validation).
    device = q.device

    def kernel_func():
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            layer(q, k, v, forward_batch, **layer_kwargs)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass

    if results.get("used_cuda_graph") is not True:
        raise RuntimeError("SGLang MLA requires CUDA graph timing; refusing to publish eager timing")
    return results["latency_ms"], results["power_stats"]


def get_context_mla_test_cases():
    return plan_cases("mla_context", "sglang", get_sm_version())


def get_generation_mla_test_cases():
    return plan_cases("mla_generation", "sglang", get_sm_version())


@get_parallel().override(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
)
def run_mla(
    input_len,
    batch_size,
    output_len,
    kv_cache_dtype,
    num_heads,
    world_size,
    tp_size,
    tokens_per_block,
    warming_up,
    test_ite,
    is_context_phase,
    *,
    perf_filename,
    device="cuda:0",
):
    from collector.sglang.runtime_compat import ensure_offline_runtime_published

    ensure_offline_runtime_published()  # sglang>=0.5.20 backends read get_exec()/get_parallel()
    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    device_str = str(torch_device)
    random.seed(0)
    torch.manual_seed(0)
    del world_size, warming_up, test_ite, output_len

    assert kv_cache_dtype in [torch.bfloat16, torch.float8_e4m3fn], "Unsupported kv cache dtype"
    assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
    local_num_heads = num_heads // tp_size

    selected_backend = _select_default_mla_backend()
    # trtllm_mla forces page_size 64 when unset (arg_groups/overrides.py:1155-1163,
    # _mla_backend_page_constraints); fa3 and triton keep the default 1.
    expected_page_size = 64 if selected_backend == "trtllm_mla" else 1
    if tokens_per_block != expected_page_size:
        raise ValueError(f"SGLang {selected_backend} requires page_size={expected_page_size}, got {tokens_per_block}")
    # FIXME(kernel-limit): trtllm_mla DECODE on Blackwell with 64 < local heads < 128.
    # flashinfer's trtllm-gen MLA decode refuses that head range (flashinfer mla/_core.py:4402
    # @v0.7.2) and its auto backend falls back to the cute-dsl runner, but sglang's
    # trtllm_mla_backend passes multi_ctas_kv_counter_buffer on every trtllm-gen call
    # (trtllm_mla_backend.py:1036-1040) -> "multi_ctas_kv_counter_buffer is only supported
    # when a trtllm-gen runner is selected" (_core.py:4741-4744 @v0.7.2). Serving hits the
    # same raise (B200 identity probe: Kimi-K3 bf16 tp1, 96 heads; finding
    # sm100_probe_failures_recheck_2026_10_04 (i)), so the cell is serving-unreachable at this tp;
    # the b200_sxm shard run (GitLab job 469988017) failed exactly the 362 num_heads=96 decode cases
    # (bf16 + fp8 KV) and nothing else. Prefill (mla_context, 96 heads) is unaffected. Fail-closed:
    # recorded as a classified failure, never a predicted skip or a silent backend swap.
    if (
        not is_context_phase
        and selected_backend == "trtllm_mla"
        and get_sm_version() in {100, 103}
        and 64 < local_num_heads < 128
    ):
        raise RuntimeError(
            "FIXME(kernel-limit): sglang trtllm_mla decode on SM100/103 does not support "
            f"64 < local_num_heads < 128 (num_heads={num_heads}, tp_size={tp_size} -> {local_num_heads}); "
            "flashinfer trtllm-gen MLA decode rejects the head range and the cute-dsl fallback rejects "
            "multi_ctas_kv_counter_buffer (mla/_core.py:4402, :4741); serving fails identically (Kimi-K3 tp1)"
        )

    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype,
        tokens_per_block,
        num_attention_heads=num_heads,
        scaling=MLA_SCALING,
    )
    total_len = input_len if is_context_phase else input_len + 1
    req_to_token_pool, token_matrix = create_req_to_token_pool(
        batch_size=batch_size,
        total_len=total_len,
        page_size=tokens_per_block,
        torch_device=torch_device,
        device_str=device_str,
    )
    model_runner.req_to_token_pool = req_to_token_pool

    total_tokens = max(1, batch_size * total_len)
    kv_cache_size = max(
        tokens_per_block,
        math.ceil(total_tokens / tokens_per_block) * tokens_per_block,
    )
    kv_pool = MLATokenToKVPool(
        size=kv_cache_size,
        page_size=tokens_per_block,
        dtype=kv_cache_dtype,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=1,
        device=device_str,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool
    # ModelRunner.init_kv_index_translator (model_runner.py:950-961): over a static pool
    # the translator translates nothing, and every backend reads its page tables through it.
    model_runner.kv_index_translator = KVIndexTranslator(
        req_to_token=req_to_token_pool.req_to_token,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        token_to_kv_pool=kv_pool,
        page_size=tokens_per_block,
        device=device_str,
        is_draft_worker=model_runner.is_draft_worker,
    )

    # Each lane builds the backend class serving's factory builds for its name
    # (layers/attention/attention_registry.py).
    if selected_backend == "trtllm_mla":
        # TRTLLMMLABackend inherits FlashInferMLAAttnBackend, whose indices updaters hold
        # the backend (FlashInferMLAIndicesUpdaterDecode(model_runner, self)): collect
        # the previous case's cycle before building another.
        gc.collect()
        attn_backend = TRTLLMMLABackend(model_runner)
        kernel_source = "trtllm_mla"
    elif selected_backend == "triton":
        attn_backend = TritonAttnBackend(model_runner)
        kernel_source = "triton"
    else:
        # Deviation (SM90 fa3, FP8 KV cache): the absorbed path casts the layer's whole
        # KV buffer to BF16 on every call (`get_key_buffer(...).to(q.dtype)`,
        # flashattention_backend.py:1944 extend, :2350 decode). Serving's buffer is the
        # full KV pool (max_total_num_tokens); this pool holds only the case's
        # batch x (isl + 1) tokens, so FP8-KV rows under-measure serving by the cast of
        # the rest of its pool, a per-layer cost that grows with the serving pool, not
        # with the case. BF16-KV rows are unaffected (the cast is a no-op). Retire by
        # sizing the pool like serving, or when SGLang reads the FP8 cache in place.
        attn_backend = FlashAttentionBackend(model_runner)
        kernel_source = "flash_attention"

    qk_head_dim, v_head_dim, absorbed = _mla_attention_geometry(selected_backend, is_context_phase)
    layer = RadixAttention(
        num_heads=local_num_heads,
        head_dim=qk_head_dim,
        scaling=MLA_SCALING,
        num_kv_heads=1 if absorbed else local_num_heads,
        layer_id=0,
        v_head_dim=v_head_dim,
    ).to(torch_device)

    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device)
    # Serving's out_cache_loc is the allocator's int64 slots (mem_cache/allocator/paged.py:242,302).
    if is_context_phase:
        num_tokens = batch_size * input_len
        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=torch_device)
        prefix_lens = torch.zeros_like(seq_lens)
        positions = torch.cat([torch.arange(input_len, device=torch_device) for _ in range(batch_size)])
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix.reshape(-1).to(torch.int64),
            seq_lens_sum=num_tokens,
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=seq_lens,
            extend_prefix_lens=prefix_lens,
            # Host lists, as ForwardBatch.init_new fills them (forward_batch_info.py:1240-1241);
            # trtllm_mla reads max(extend_seq_lens_cpu) (trtllm_mla_backend.py:809).
            extend_seq_lens_cpu=[input_len] * batch_size,
            extend_prefix_lens_cpu=[0] * batch_size,
            extend_num_tokens=num_tokens,
            positions=positions,
        )
    else:
        history_len = input_len
        num_tokens = batch_size
        seq_lens = torch.full((batch_size,), history_len + 1, dtype=torch.int32, device=torch_device)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix[:, history_len:].reshape(-1).to(torch.int64),
            seq_lens_sum=int(batch_size * (history_len + 1)),
            seq_lens_cpu=seq_lens.cpu(),
            positions=torch.full((batch_size,), history_len, device=torch_device),
        )

    # ForwardBatch.init_new binds every forward to its iteration's KV-location plan
    # (forward_batch_info.py:1139-1142); the MLA backends hand it to the translator for
    # their page tables (trtllm_mla_backend.py:872, triton_backend.py:910,1025).
    translator = model_runner.kv_index_translator
    translator.own_plan(forward_batch).bind(forward_batch, translator)

    if not absorbed:
        # trtllm_mla prefill, as forward_normal_prepare hands it to attn_mha
        # (forward_mha.py:214-293): q is q_b_proj's (T, H, 192) view, k the concatenated
        # (T, H, 192) key, and v the strided value half of kv_b_proj's (T, H, 256) output.
        q = torch.randn(num_tokens, local_num_heads, qk_head_dim, device=torch_device, dtype=torch.bfloat16)
        k = torch.randn(num_tokens, local_num_heads, qk_head_dim, device=torch_device, dtype=torch.bfloat16)
        kv = torch.randn(
            num_tokens, local_num_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, device=torch_device, dtype=torch.bfloat16
        )
        v = kv[..., QK_NOPE_HEAD_DIM:]
        # forward_normal_chunked_kv_core (forward_mha.py:383-386): no prefix, no LSE, and
        # the latent cache was written in prepare, outside the attention call.
        forward_batch.mha_return_lse = False
        forward_batch.set_attn_attend_prefix_cache(False)
        layer_kwargs = {"save_kv_cache": False}
    else:
        if not is_context_phase and history_len > 0:
            history_loc = token_matrix[:, :history_len].reshape(-1).to(torch.int64)
            kv_pool.set_mla_kv_buffer(
                layer,
                history_loc,
                torch.randn(history_loc.numel(), 1, KV_LORA_RANK, device=torch_device, dtype=torch.bfloat16),
                torch.randn(history_loc.numel(), 1, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.bfloat16),
            )
        # forward_absorb_core (forward_mla.py:726-799): q_nope_out / q_pe and the new
        # token's k_nope / k_pe, saved to the cache inside the call.
        q_nope = torch.randn(num_tokens, local_num_heads, KV_LORA_RANK, device=torch_device, dtype=torch.bfloat16)
        q_rope = torch.randn(num_tokens, local_num_heads, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.bfloat16)
        k_nope = torch.randn(num_tokens, 1, KV_LORA_RANK, device=torch_device, dtype=torch.bfloat16)
        k_rope = torch.randn(num_tokens, 1, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.bfloat16)
        v = k_nope
        layer_kwargs = {}
        if selected_backend == "triton":
            # Backends outside FORWARD_ABSORB_CORE_ATTENTION_BACKENDS (models/deepseek_common/
            # utils.py:65-75) get the concatenated q / k (forward_mla.py:784-799).
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q = q_nope
            k = k_nope
            layer_kwargs.update(q_rope=q_rope, k_rope=k_rope)
        if selected_backend == "trtllm_mla" and not is_context_phase and kv_cache_dtype == torch.float8_e4m3fn:
            # FP8 trtllm_mla decode defers RoPE into the backend's fused rope + quantize
            # (forward_mla.py:964-987 _fuse_rope_for_trtllm_mla, :730-735). The cache is
            # the rotary embedding's float32 [max_position, rotary_dim] cos|sin table
            # (flashinfer.rope.mla_rope_quantize_fp8 requires float32).
            layer_kwargs["cos_sin_cache"] = torch.randn(
                total_len, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.float32
            )

    with forward_context(ForwardContext(attn_backend=attn_backend)):
        attn_backend.init_forward_metadata(forward_batch)

    latency, power_stats = benchmark_layer(layer, attn_backend, forward_batch, q, k, v, **layer_kwargs)

    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    str_type = "bfloat16" if kv_cache_dtype == torch.bfloat16 else "fp8"
    if not log_perf(
        item_list=[
            {
                "mla_dtype": _mla_compute_dtype(selected_backend, kv_cache_dtype),
                "kv_cache_dtype": str_type,
                "num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=_dist_version("sglang"),
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"Failed to persist SGLang MLA performance row to {perf_filename}")


if __name__ == "__main__":
    test_cases = get_context_mla_test_cases()
    for test_case in test_cases[0:10]:
        print(test_case)
        run_mla(*test_case, perf_filename=PerfFile.CONTEXT_MLA)

    test_cases = get_generation_mla_test_cases()
    for test_case in test_cases[0:10]:
        print(test_case)
        run_mla(*test_case, perf_filename=PerfFile.GENERATION_MLA)
