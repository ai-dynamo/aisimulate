# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang dense attention collector.

Builds lightweight RadixAttention/ForwardBatch mocks to benchmark context and
generation attention without launching a full SGLang server. Shared attention
shape intent should live in YAML; this file owns SGLang backend construction,
KV-cache setup, backend dispatch, and perf logging for the SGLang runtime.
"""

# 0.5.21 added 2026-10-01 (H20/sm90 collector port: op_smoke + path gates in the v0.5.21 image; findings hopper_sglang_collector_port_0514_to_0521_2026_10_01). Releases in between are unvalidated and excluded.
__compat__ = "sglang>=0.5.14,<=0.5.21,!=0.5.15,!=0.5.16,!=0.5.17,!=0.5.18,!=0.5.19,!=0.5.20"

import math
import os
from types import SimpleNamespace
from typing import NamedTuple

from importlib.metadata import version as _dist_version  # setuptools/pkg_resources is absent from sglang>=0.5.21 images
import torch
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


class Timing(NamedTuple):
    mean: float


# Mock objects to satisfy RadixAttention dependencies
class MockModelConfig:
    def get_max_num_attention_heads(self) -> int:  # model_config.py:1504 @0.5.21 (triton_backend.py:255)
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
        # sglang 0.5.16: configs/hybrid_arch.py:30-31 reads
        # model_config.linear_attn_registry_result (ModelConfig property,
        # model_config.py:707) to detect hybrid linear-attention archs; the
        # dense-attention mock has none.
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
                # HF PretrainedConfig API: sglang 0.5.16 unwraps multimodal
                # configs through it (configs/hybrid_arch.py:49,
                # linear_attn_model_registry.py:56); a text-only mock is its own
                # text config.
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

    def get_num_kv_heads(self, tp_size, dcp_size=1):  # model_config.py:1512 @0.5.21 adds dcp_size (DCP replicates KV)
        return self.num_key_value_heads // tp_size


class MockServerArgs:
    def __init__(self, page_size: int):
        self.enable_lora = False
        self.enable_deterministic_inference = False
        self.kv_cache_dtype = "auto"
        self.speculative_eagle_topk = 0
        self.speculative_num_draft_tokens = 0
        self.speculative_num_steps = None
        self.page_size = page_size
        self.multi_item_scoring_delimiter = None
        self.dllm_algorithm = None
        self.dllm_algorithm_config = None
        self.is_embedding = False
        self.disable_radix_cache = False
        self.enable_dp_attention = False
        self.model_path = None
        self.revision = None
        # Required by TritonAttnBackend
        self.triton_attention_num_kv_splits = 8
        self.triton_attention_split_tile_size = None
        self.disable_cuda_graph = False
        self.chunked_prefill_size = -1
        self.enable_mis = False
        self.enable_two_batch_overlap = False
        self.disable_attn_tp_gather = False
        self.moe_dense_tp_size = None
        # sglang 0.5.16: flashattention_backend.py:51 reads
        # server_args.enable_prefill_cp (prefill context parallelism) — off
        self.enable_prefill_cp = False


class MockModelRunner:
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
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.server_args = MockServerArgs(page_size=page_size)
        self.attn_cp_size = 1  # Context parallelism size; required by FlashAttentionBackend in sglang >=0.5.10
        # sglang 0.5.21 TRTLLMHAAttnBackend (the dense backend sglang picks on SM100/103) reads
        # model_runner.prefill_attention_backend_str / decode_attention_backend_str to decide its
        # prefill/decode split; fa3/flashinfer on Hopper never asked (B200 smoke 2026-10-04:
        # AttributeError 'MockModelRunner' object has no attribute 'prefill_attention_backend_str').
        self.prefill_attention_backend_str = attn_backend_name
        self.decode_attention_backend_str = attn_backend_name
        # trtllm_mha_backend.py:310@0.5.21 scans model_runner.model.modules() for ENCODER_ONLY attention
        # layers (expanded TARGET_VERIFY metadata); the collector has no model — an empty Module
        # declares none, which is the dense-decoder answer.
        self.model = torch.nn.Module()
        # TRTLLMHAAttnBackend also sizes its workspaces from model_runner.max_running_requests
        # (512 = the generator's --max-running-requests default).
        self.max_running_requests = 512
        # sglang 0.5.16 reads the parallel geometry from ``model_runner.ps``
        # (flashattention_backend.py:183 attn_cp_size, :271-274 tp_size; the
        # runner sets it at model_runner.py:262 from a ParallelState). Provide
        # the real trivial ParallelState when the wrapper exists, else a
        # namespace with the same fields (parallel_state_wrapper.py:6-24).
        try:
            from sglang.srt.distributed.parallel_state_wrapper import ParallelState

            self.ps = ParallelState.trivial(gpu_id=0)
        except ImportError:
            from types import SimpleNamespace

            self.ps = SimpleNamespace(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, dp_rank=0, dp_size=1,
                attn_tp_rank=0, attn_tp_size=1, attn_cp_rank=0, attn_cp_size=1,
                attn_dp_rank=0, attn_dp_size=1, moe_ep_rank=0, moe_ep_size=1,
                moe_dp_rank=0, moe_dp_size=1, dcp_size=1, gpu_id=0,
            )
        self.is_draft_worker = False
        # sglang>=0.5.21 backends ask the runner for logits rows per decode slot
        # (model_runner.py:796 decode_num_tokens_per_req; triton_backend.py:218,222);
        # the kernel collectors never run speculative decoding -> 1
        self.spec_algorithm = None
        self.model_is_mrope = False
        self.sliding_window_size = attention_chunk_size
        if self.sliding_window_size is None and runtime_window_size >= 0:
            self.sliding_window_size = runtime_window_size
        self.attention_chunk_size = attention_chunk_size
        # Initialized after the KV pool is created.
        self.token_to_kv_pool_allocator = None
        self.model_config = MockModelConfig(
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            architecture,
            runtime_window_size,
            attention_chunk_size,
        )
        self.kv_cache_dtype = kv_cache_dtype  # Default
        # sglang>=0.5.21 FlashAttentionBackend reads the server-args string next to the
        # torch dtype (flashattention_backend.py:192-194 kv_cache_dtype_str, drives the
        # mxfp8 branch); serving sets it from server_args.kv_cache_dtype
        # (model_runner.py:1481). The collector's kv_cache_dtype is that same string.
        self.kv_cache_dtype_str = kv_cache_dtype
        self.page_size = page_size
        self.tp_size = 1
        self.is_hybrid = False
        self.dtype = torch.bfloat16
        # Provide compatibility across sglang versions that expect this flag
        self.is_hybrid_swa = self.model_config.is_hybrid_swa
        self.server_args.kv_cache_dtype = kv_cache_dtype
        self.server_args.page_size = page_size
        # Required by TritonAttnBackend
        self.gpu_id = 0
        self.hybrid_gdn_config = None
        self.kimi_linear_config = None
        self.linear_attn_model_spec = None


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


def _int_list(values):
    return [int(value) for value in values]


def get_context_attention_test_cases():
    test_cases = []

    # FP8 KV-cache cases follow the FP8 hardware floor (SM89+, Ada — see
    # cases/capabilities.yaml dtype_min_sm.fp8). SGLang 0.5.14 puts no SM
    # gate on --kv-cache-dtype fp8_e4m3 (server_args.py:596-600) and the
    # flashinfer backend passes kv_cache_dtype straight into its plan calls
    # (flashinfer_backend.py:1151, 1377-1397); any backend-level rejection
    # surfaces as a classified runtime failure, not a generation-time skip.
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 89

    for shape_sweep in get_attention_context_shape_sweeps("sglang"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep["max_batch_size_self_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_config in get_attention_head_configs(
            shape_sweep,
            phase="context",
            backend="sglang",
            sm_version=sm_version,
        ):
            n = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            for s in sorted(sequence_lengths, reverse=True):
                for b in sorted(batch_sizes, reverse=True):
                    if num_kv_heads == n:
                        if b * s > max_tokens_self_attention or b > max_batch_size_self_attention:
                            continue
                    else:
                        if b * s > max_tokens_grouped_query_attention:
                            continue
                    if b * s * num_kv_heads * head_dim * 2 >= max_kv_elements:
                        continue
                    # the backend this head config runs on (per-model map, else the SM default table) —
                    # the same resolution run_attention_torch makes before refusing a combo
                    lane = head_config.kernel_source or default_attention_backend(sm_version, head_config.has_attention_sink)
                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        use_fp8_context_fmha = bool(precision_case["fp8_context_fmha"])
                        if skip_fp8 and use_fp8_kv_cache:
                            continue
                        if use_fp8_context_fmha and lane == "flashinfer":
                            # FlashInfer has no FP8 prefill compute path (BF16 Q reads the FP8 cache through
                            # descales), so run_attention_torch refuses this case by design; scheduling it on
                            # the flashinfer SMs (sm89 / sm120) only made a third of the op's tasks fail
                            # (sm89 sample 2026-10-04: 13 of 40). Not an SM gate: the fp8_kv_cache case stays.
                            continue
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                head_dim,
                                use_fp8_kv_cache,
                                use_fp8_context_fmha,
                                True,
                                window_size,
                                head_config.v_head_dim,
                                head_config.runtime_window_size,
                                head_config.attention_chunk_size,
                                head_config.has_attention_sink,
                                head_config.scaling,
                                head_config.kernel_source,
                                head_config.architecture,
                            ]
                        )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, head_dim, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens // s // num_heads * 128 // head_dim
        for b in sorted(batch_sizes):
            if b > max_b:
                break
            if s not in s_b_dict:
                s_b_dict[s] = {b}
            else:
                s_b_dict[s].add(b)
    for s, b_set in s_b_dict.items():
        if len(b_set) < int(shape_sweep["min_batch_options_per_sequence"]):
            continue
        for b in b_set:
            if b not in b_s_dict:
                b_s_dict[b] = {s - 1}
            b_s_dict[b].add(s - 1)
    return b_s_dict


def get_generation_attention_test_cases():
    test_cases = []

    # FP8 KV-cache cases follow the FP8 hardware floor (SM89+, Ada); see the
    # context getter for the framework citations.
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 89

    for shape_sweep in get_attention_generation_shape_sweeps("sglang"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])
        for head_config in get_attention_head_configs(
            shape_sweep,
            phase="generation",
            backend="sglang",
            sm_version=sm_version,
        ):
            n = head_config.num_heads
            n_kv = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            max_tokens = (
                int(shape_sweep["max_mha_tokens_per_step"])
                if n == n_kv
                else int(shape_sweep["max_xqa_tokens_per_step"])
            )
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                head_dim,
                max_tokens,
                shape_sweep,
            )
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= min_drop_batch:
                    target_s_list = target_s_list[:-1]
                for s in target_s_list:
                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        if skip_fp8 and use_fp8_kv_cache:
                            continue
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                head_dim,
                                use_fp8_kv_cache,
                                False,
                                False,
                                window_size,
                                head_config.v_head_dim,
                                head_config.runtime_window_size,
                                head_config.attention_chunk_size,
                                head_config.has_attention_sink,
                                head_config.scaling,
                                head_config.kernel_source,
                                head_config.architecture,
                            ]
                        )
    return test_cases


@get_parallel().override(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
)
def default_attention_backend(sm_version: int, has_attention_sink: bool) -> str | None:
    """SGLang's default dense-attention backend per SM (the table run_attention_torch documents
    above its call site); None for SMs outside the supported set."""
    return {
        89: "triton" if has_attention_sink else "flashinfer",
        90: "fa3",
        100: "trtllm_mha",
        103: "trtllm_mha",
        120: "triton" if has_attention_sink else "flashinfer",
    }.get(sm_version)


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
        sm_version = get_sm_version()
        # Mirrors SGLang 0.5.14 server_args._get_default_attn_backend (MHA),
        # python/sglang/srt/server_args.py:4407-4455 at image source 49e384ce:
        # SM90 Hopper+CUDA>=12.3 -> fa3 (line 4437); SM100/103 -> trtllm_mha
        # (is_sm100_supported() matches major 10, lines 4438-4446); other
        # supported CUDA SMs -> flashinfer unless the model has attention
        # sinks (FlashInfer rejects sinks, lines 4451-4454) -> triton.
        # Per-model deviations (Qwen3.5 hybrid-GDN -> triton on SM100,
        # server_args.py:4188-4211; NemotronH -> flashinfer,
        # arg_groups/nemotron_h_hook.py:60-62) arrive through the profile's
        # sglang_backends map as an explicit attn_backend_name, never through
        # this default table. SM80/86 are outside the
        # supported platform set {89, 90, 100, 103, 120} and fail closed below.
        attn_backend_name = default_attention_backend(sm_version, has_attention_sink)
        if attn_backend_name is None:
            raise ValueError(f"No SGLang 0.5.14 attention backend mapping for SM{sm_version}")
    if page_size is None:
        page_size = 64 if attn_backend_name == "trtllm_mha" else 1

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
    # sglang>=0.5.21 backends ask the runner which backend serves each phase
    # (model_runner.py:1020-1021 resolved.prefill/decode; flashinfer_backend.py:331-332
    # gate the kv-access checks on it). One lane = one backend for both phases. Hopper's
    # fa3 / Blackwell's trtllm_mha never read it, so only the sm89/sm120 flashinfer lane failed.
    model_runner.prefill_attention_backend_str = attn_backend_name
    model_runner.decode_attention_backend_str = attn_backend_name

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
    model_runner.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=model_runner.page_size,
        get_kvcache=lambda: model_runner.token_to_kv_pool,
    )
    from collector.sglang.runtime_compat import attach_kv_index_translator

    attach_kv_index_translator(model_runner)

    if attn_backend_name == "flashinfer":
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        attn_backend = FlashInferAttnBackend(model_runner)
    elif attn_backend_name == "trtllm_mha":
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        attn_backend = TRTLLMHAAttnBackend(model_runner)
    elif attn_backend_name == "triton":
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        attn_backend = TritonAttnBackend(model_runner)
    elif attn_backend_name == "fa3":
        attn_backend = FlashAttentionBackend(model_runner)
    else:
        raise ValueError(f"Unsupported SGLang attention backend: {attn_backend_name}")

    model_runner.attn_backend = attn_backend

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
            extend_seq_lens_cpu=seq_lens_cpu,
            extend_prefix_lens_cpu=prefix_lens.cpu(),
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

    forward_batch.req_to_token_pool = req_to_token_pool
    forward_batch.token_to_kv_pool = kv_pool

    with forward_context(ForwardContext(attn_backend=attn_backend)):
        attn_backend.init_forward_metadata(forward_batch)

        # Label prefill rows by the compute dtype the backend actually uses in
        # SGLang 0.5.14 — the input dtype is an API contract, not the compute
        # dtype:
        #   - fa3 (SM90): whenever the KV cache is FP8 and head_dim <= 256 it
        #     casts Q to the KV dtype itself (flashattention_backend.py:857-872),
        #     so FP8-KV prefill IS FP8 FMHA compute.
        #   - trtllm_mha (TRTLLM-GEN on SM100/103): requires BF16 inputs and
        #     quantizes Q internally via scaled_fp8_quant when the KV cache is
        #     FP8 (trtllm_mha_backend.py:154-158, 291-301). Feed BF16 and let
        #     the backend quantize; an external FP8 cast would break its input
        #     contract without changing the (already FP8) compute.
        #   - flashinfer: BF16 Q reads the FP8 KV cache with descales; there is
        #     no FP8 prefill compute path, so an fp8-labeled row would be a lie.
        # Consequence: on fa3/trtllm_mha a "BF16-compute prefill on FP8 KV"
        # combo does not exist — the backend force-quantizes Q — so collecting
        # it would record FP8 compute under a bfloat16 label. Fail closed.
        if is_context_phase:
            if use_fp8_context_fmha:
                if attn_backend_name == "flashinfer":
                    raise ValueError("SGLang flashinfer has no FP8 prefill compute path (plan-time skip missed this case)")
                if attn_backend_name != "trtllm_mha":
                    q = q.to(kvtype)
                    k = k.to(kvtype)
                    v = v.to(kvtype)
            elif use_fp8_kv_cache and (
                attn_backend_name == "trtllm_mha" or (attn_backend_name == "fa3" and head_dim <= 256)
            ):
                raise ValueError(
                    f"SGLang 0.5.14 {attn_backend_name} quantizes Q to FP8 internally when the KV cache "
                    "is FP8, so a BF16-compute prefill on an FP8 KV cache does not exist; this "
                    "combination is the fp8_context_fmha case"
                )

        # Mirror the serving call contract: only sink-carrying models pass the
        # ``sinks`` kwarg (e.g. gpt_oss.py), and SGLang 0.5.14
        # FlashInferAttnBackend.forward_extend/forward_decode do not accept it
        # at all — passing ``sinks=None`` unconditionally breaks every
        # flashinfer-routed case (first seen on SM100 NemotronH; flashinfer is
        # never the selected dense backend on SM90). A sink profile routed to
        # flashinfer still fails closed with the framework's own TypeError.
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
