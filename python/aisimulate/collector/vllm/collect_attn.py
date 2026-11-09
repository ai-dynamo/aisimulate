# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Versions are upgraded IN PLACE: this file always
# targets the manifest pin; prior-version code is `git log -- <this file>`
# away and prior-version DATA stays permanent under its version key.
#
# Audited against vLLM v0.32.0 (5df63a72, from v0.31.0 db9527a4); citations
# below are v0.32.0 lines. Every import, constructor argument and metadata
# field used here exists with the same signature. Backend selection is
# unchanged for this collector's BF16/FP8-KV, no-sink cases: the CUDA priority
# lists (platforms/cuda.py:154-179), FlashAttention's head-size, FP8-KV and
# FA-version rules (flash_attn.py:422-509, fa_utils.py:74-221) and FlashInfer's
# head and page sizes (flashinfer.py:459-479, :565-568) only gained XPU,
# NVFP4-KV and cascade-removal edits; cascade attention is gone and
# common_prefix_len is always 0 (v1/attention/backend.py:693-694).
# Kernel-side changes: vllm-flash-attention stays at 9cd61de3
# (cmake/external_projects/vllm_flash_attn.cmake:42) and FlashInfer at
# 0.7.0.post1; nvidia-cutlass-dsl, which compiles FA4 (SM90 head_dim 512),
# moves 4.7.1 -> 4.8.0 (requirements/cuda.txt); Triton unified attention
# scales P before its FP8 P@V dot (ops/triton_unified_attention.py:589-597) and
# uses 64 softmax segments on SM120 when 16 under-fill the SMs
# (triton_attn.py:146-186, :240-256); reshape_and_cache_flash moved out of
# csrc/libtorch_stable with include-only edits. FA2/FA3 and FlashInfer rows
# keep their kernels. FlashInfer's autotuner owns no dense-attention op
# (flashinfer 0.7.0.post1 decode.py, prefill.py and xqa.py never reach
# AutoTuner), so serving's warmup autotune (warmup/kernel_warmup.py:261-268)
# leaves these kernels on their fixed heuristics, as timed here.
__compat__ = "vllm==0.32.0"

import os

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_kv_cache_and_block_mappings,
    create_standard_kv_cache_spec,
    create_vllm_config,
    framework_kv_block_size,
    get_attention_backend,
    with_exit_stack,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.version import __version__ as vllm_version

# Serving's default scheduler bound on GPUs with 70 GB or more (H100 and newer,
# RTX PRO 6000); 256 below (engine/arg_utils.py:2924-2952). The CUDA-graph
# capture sizes derive from it: up to min(2 * max_num_seqs, 1024 on SM100/SM103,
# else 512) (config/vllm.py:2354-2364). 1024 reproduces serving's capture sizes
# on every target GPU: SM10x parts all have 160 GB or more, and the 48 GB L40S
# (256) still reaches the 512 cap.
SERVING_MAX_NUM_SEQS = 1024


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


# https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends
# support MHA GQA MQA bfloat16 tensor and bfloat16/fp8 kv cache


def kv_cache_in_resolved_layout(kv_cache: torch.Tensor, layer_view_order: tuple[int, ...]) -> torch.Tensor:
    """Return a tensor with the SAME logical [B, H, N, C] view as ``kv_cache``
    whose memory follows the resolved physical axis order ``layer_view_order``
    (vllm.v1.kv_cache_layout.KVCacheLayout.layer_view_order), i.e. exactly what
    ``kv_cache.permute(*layer_view_order)`` in the backends expects to be
    contiguous. Identity layout -> the input itself."""
    order = tuple(int(i) for i in layer_view_order)
    if order == tuple(range(kv_cache.dim())):
        return kv_cache.contiguous()
    inverse = [order.index(i) for i in range(len(order))]
    return kv_cache.permute(*order).contiguous().permute(*inverse)


def _dense_kernel_source(backend_name_str, impl, attn_metadata):
    """Ground-truth kernel_source for the dense attention row."""
    if backend_name_str == "FLASH_ATTN":
        fa_version = impl.vllm_flash_attn_version
        if fa_version is None:
            raise RuntimeError("vLLM selected FlashAttention without a concrete FA version")
        return f"vllm_flash_attn_fa{fa_version}"
    if backend_name_str == "FLASHINFER":
        # FlashInfer classifies requests by query length, not by the
        # collector's phase label: with query_len <= reorder_batch_threshold
        # (1, flashinfer.py:737, :928-941) an s=1 "context" batch is entirely
        # decodes (split_decodes_and_prefills, :1401-1411) and metadata.prefill
        # stays None (:1487-1501), so read the portion vLLM actually populated.
        if attn_metadata.num_prefills > 0:
            phase_metadata = attn_metadata.prefill
        else:
            phase_metadata = attn_metadata.decode
        if phase_metadata is None:
            raise RuntimeError("vLLM FlashInfer metadata has neither a prefill nor a decode portion")
        label = f"vllm_flashinfer_{type(phase_metadata).__name__}".lower()
        # XQA (SM90, SM12x) and trtllm-gen (SM10x) decode share ONE metadata
        # class, FlashInferTrtllmAPIDecode, which names the kernel in its
        # `kernel` field (flashinfer.py:629-633 FlashInferDecodeKernel,
        # :668-672, :1153-1160; decode_with_xqa / decode_with_trtllm_gen at
        # :2058-2059). The class name alone would label SM120's XQA row and
        # SM100's trtllm-gen row identically; kernel_source must name the
        # kernel that ran.
        kernel = getattr(phase_metadata, "kernel", None)
        if kernel is not None:
            label += "_" + str(getattr(kernel, "value", kernel)).lower().replace("-", "_")
        return label
    return f"vllm_{backend_name_str}".lower()


def match_serving_fa_aot_schedule(builder, window_size: int) -> None:
    """Put a FlashAttention builder in the AOT-scheduling state serving reaches.

    FA3's builder schedules ahead of time (get_scheduler_metadata in build())
    only when every FlashAttention layer of the model shares one sliding-window
    config. On its first build it collects the configs of all registered layers
    (flash_attn.py:563-578, :834-848): one config keeps AOT with that window,
    more than one turns AOT off for the whole model, and FA3 then schedules
    inside each forward. The collector registers no layer, so it sets the
    outcome itself. The sliding-window profiles in cases/models (GPT-OSS,
    Gemma 4, MiMo-V2-Flash, Step-3.7-Flash, Muse Glimmer) all belong to models
    that also have full-attention layers, so window_size > 0 rows run with AOT
    off, as serving runs those models; window_size 0 rows keep the pure
    full-attention state (AOT on, window (-1, -1)). Llama 4's window 8192 rows
    stay an approximation either way: vLLM serves its local layers as fixed
    chunks with no sliding window (AOT on), and the rows time a sliding window
    (cases/models/Llama4ForConditionalGeneration_cases.yaml). Builders without
    the field (not FlashAttention) are left alone; on FA2/FA4 aot_schedule is
    already off (:703).
    """
    if window_size <= 0 or not hasattr(builder, "aot_sliding_window"):
        return
    builder.aot_sliding_window = (-1, -1)
    builder.aot_schedule = False


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,  # keep same as num_heads for MHA
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    window_size=0,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    # Backend selection runs at the framework default page (serving does the same: the
    # model is built at CacheConfig.DEFAULT_BLOCK_SIZE, the page is re-derived from the
    # selected backend afterwards — utils.framework_kv_block_size).
    block_size = None  # unconstrained selection (no --block-size); the page is asked from the backend below

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[input_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            # vLLM seq_lens includes the current query token. ``input_len`` is
            # the persisted pre-query history (the raw ``step`` column).
            seq_lens=[input_len + 1] * batch_size,
            query_lens=[1] * batch_size,
        )

    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        # Serving's scheduler bound, not this case's batch: the builders read
        # the capture sizes it sets (SERVING_MAX_NUM_SEQS). FA3 caps split-KV at
        # flash_attn_max_num_splits_for_cuda_graph (32) for every batch of up
        # to the largest capture size in tokens and uses its heuristic above it
        # (flash_attn.py:742-767, :850-860); FlashInfer plans its CUDA-graph
        # decode wrapper up to it (flashinfer.py:787-801, :1777-1781); Triton
        # takes its 2D/3D decode threshold from the capture sizes
        # (triton_attn.py:115-144). With max_num_seqs=batch_size, FA3 prefills
        # of up to 512 tokens and more than 2 tokens per request ran the split
        # heuristic where serving caps at 32 splits. A batch above serving's
        # bound keeps its own size.
        max_num_seqs=max(batch_size, SERVING_MAX_NUM_SEQS),
        # One collector call represents the full prefill, including shapes
        # above a serving scheduler's new-token budget. Size native builder
        # buffers for that call (flashinfer.py:770-772, workspace :1107-1130).
        max_num_batched_tokens=max(8192, batch_spec.compute_num_tokens()),
        use_fp8_kv_cache=use_fp8_kv_cache,
        sliding_window=window_size if window_size > 0 else None,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # Serving's Attention layer selects through get_attn_backend
    # (model_executor/layers/attention/attention.py:350-360), which builds the
    # selector config from the current VllmConfig (v1/attention/selector.py:
    # 105-200; block_size stays None without --block-size, :131-135). This
    # collector times no attention sinks (op_plans._vllm_attention_rows).
    # Deviation (Gemma 4 shapes: head_dim 256 window 1024, head_dim 512; no
    # other model has them): serving's Gemma4Config (model_executor/models/
    # config.py:229-290) keeps per-layer selection only for FP8 KV on SM90,
    # forces FA4 on every FlashAttention layer elsewhere when FA4 is available,
    # and forces TRITON_ATTN when it is not. These rows time native per-shape
    # selection, so on SM90 with BF16 KV the head_dim 256 layer runs FA3 where
    # Gemma 4 runs FA4, and where FA4 is unavailable the rows run FA2 or
    # FlashInfer where Gemma 4 runs Triton. Direction unknown. Retiring it needs
    # a model-aware row for these shapes.
    backend_cls = get_attn_backend(
        head_dim,
        dtype,
        "fp8" if use_fp8_kv_cache else "auto",
        has_sliding_window=window_size > 0,
    )
    backend_name_str = backend_cls.get_name()
    backend_name = AttentionBackendEnum[backend_name_str]

    # Serving without --block-size pages the cache manager at the first
    # backend's get_preferred_block_size(CacheConfig.DEFAULT_BLOCK_SIZE=16)
    # (platforms/interface.py:767-781, :721-722; v1/attention/backend.py:151-160;
    # config/cache.py:72) and the kernels at select_common_block_size
    # (v1/worker/utils.py:515-557); utils.framework_kv_block_size asks the same.
    # FlashInfer accepts 16 (flashinfer.py:459-479), so the isolated Qwen
    # n=4/kv=1/d=256 case pages at 16. Hybrid-model page alignment
    # (platforms/interface.py:783-787) is outside one isolated layer.
    block_size = framework_kv_block_size(backend_cls)
    vllm_config.cache_config.block_size = block_size

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Ensure the KV cache has enough blocks for all sequences.
    required_blocks = 1 + sum((s_len + block_size - 1) // block_size for s_len in batch_spec.seq_lens)
    num_blocks = vllm_config.cache_config.num_gpu_blocks or required_blocks
    num_blocks = max(num_blocks, required_blocks)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence
        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and realistic history/query slot mappings.
    kv_cache, history_slot_mapping = create_kv_cache_and_block_mappings(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_spec.dtype,
        device=device,
        num_blocks=num_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
    )

    # Serving resolves one layout for the model from its backends' supported
    # sets (v1/attention/backends/utils.py:209-244, :259-350); reset
    # cache_config after the case so the next case re-resolves instead of
    # inheriting ("a layout already present on cache_config wins outright").
    _supported = get_supported_kv_cache_layouts([backend_cls])
    resolve_kv_cache_layout(vllm_config, [[m.name for m in _supported]], [kv_cache_spec])
    exit_stack.callback(lambda: setattr(vllm_config.cache_config, "kv_cache_layout", None))
    # The impls consume ONE logical tensor [num_blocks, H, block, 2*D] with K/V
    # split on the last dim (flash_attn.py:1251-1252, triton_attn.py:625-627,
    # flashinfer.py:2660-2663 "(B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H,
    # hs))"). Convert the populated helper layout [2, B, N, H, D] once,
    # value-equivalent.
    _k, _v = kv_cache[0], kv_cache[1]  # [B, N, H, D]
    kv_cache = torch.cat((_k.permute(0, 2, 1, 3), _v.permute(0, 2, 1, 3)), dim=-1).contiguous()  # [B, H, N, 2*D]
    # Serving allocates the per-layer cache in the RESOLVED physical layout and
    # backends read it through kv_cache.permute(*layout.layer_view_order)
    # (v1/kv_cache_layout.py:34-37; flashinfer.py:2135-2136). FlashInfer on
    # SM10x resolves LBHNC (head-major blocks, flashinfer.py:603-610);
    # FLASH_ATTN, TRITON_ATTN and FlashInfer elsewhere declare no layout and
    # resolve the default LBNHC (v1/attention/backends/utils.py:195-202). The
    # XQA and trtllm-gen decode paths assert "KV cache inner dims (block_size,
    # head_size) must be contiguous" (flashinfer.py:2488-2495): HND memory under
    # the NHD layout failed every fp8-KV GQA generation case on SM120,
    # 2026-10-01. Lay the memory out per the resolved layout and hand the
    # backend the same logical view.
    kv_cache = kv_cache_in_resolved_layout(
        kv_cache, vllm_config.cache_config.get_resolved_kv_cache_layout().layer_view_order
    )

    builder_cls, impl_cls = get_attention_backend(backend_name)
    layer_names = ["placeholder"]

    # FlashInfer's builder reads window, soft cap and scale from the registered
    # layers (get_per_layer_parameters, v1/attention/backends/utils.py:369-397,
    # called at flashinfer.py:943-948); the collector registers none, so hand
    # it what one layer of this shape gives: window_left = sliding_window - 1
    # (flashinfer.py:1867-1873), no soft cap (0.0 and None plan alike,
    # :2201, :2408) and scale 1/sqrt(head_dim).
    if backend_name_str == "FLASHINFER":
        import unittest.mock

        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            window_left = window_size - 1 if window_size > 0 else -1
            return {
                layer_name: PerLayerParameters(
                    window_left=window_left,
                    logits_soft_cap=0.0,
                    sm_scale=1.0 / (head_dim**0.5),
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters", mock_get_per_layer_parameters
        ):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        match_serving_fa_aot_schedule(builder, window_size)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
    )

    # Create mock layer
    mock_layer = MockAttentionLayer(device)

    # Populate persisted history through the selected backend's production
    # writer. In particular, vLLM stores FP8 KV as uint8 and the writer applies
    # the layer scales plus the backend-specific physical cache layout.
    if history_slot_mapping.numel() > 0:
        impl.do_kv_cache_update(
            mock_layer,
            torch.cat(k_contexts),
            torch.cat(v_contexts),
            kv_cache,
            history_slot_mapping,
        )

    # Run forward pass

    test_ite = 6
    warm_up = 3

    # The timed closure is serving's per-layer attention call. With an FP8 KV
    # cache, impls that take an FP8 query get it from the Attention layer's
    # QuantFP8 (model_executor/layers/attention/attention.py:467-486,
    # :509-521), a torch op that Inductor fuses into the op producing the query,
    # so it stays outside the timing; impls that quantize the query themselves
    # (FlashInfer prefill on SM90, maybe_quant_query, flashinfer.py:1984-2003)
    # keep it inside forward, as serving does. The KV-cache write is a separate
    # op per layer (unified_kv_cache_update, attention.py:540-570) and is timed:
    # the RoPE+KV-cache fusion passes are ROCm-only on this release
    # (config/compilation.py:282-293).
    needs_fp8_query = use_fp8_kv_cache and impl.supports_quant_query_input
    query_fwd = query_vllm.to(current_platform.fp8_dtype()) if needs_fp8_query else query_vllm
    # Output buffer is always BF16 — the impl dequantises internally.
    output = torch.empty_like(query_vllm)

    # head_dim 192 on SM100/SM103: FlashInfer does not take 192
    # (flashinfer.py:565-568), so FLASH_ATTN serves it (flash_attn.py:422-430),
    # and get_flash_attn_version moves FA4 head sizes above 128 other than 256
    # and (192, V=128) to FA2 (fa_utils.py:193-209). With V=192 these rows run
    # FA2 (kernel_source vllm_flash_attn_fa2); FA4's own (192, 192) rejection
    # (flash-attention 9cd61de3 flash_attn/cute/interface.py) is not reached.
    # MiMo-V2-Flash, whose head_dim 192 rows these are, serves V=128 on
    # FLASH_ATTN_DIFFKV or TRITON_ATTN_DIFFKV (model_executor/models/
    # mimo_v2.py:300-321); op_plans._vllm_attention_rows notes the V=192
    # deviation. No GPU run has qualified this at v0.32.0.
    def run():
        if not backend_cls.forward_includes_kv_cache_update:
            impl.do_kv_cache_update(
                mock_layer,
                key_vllm,
                value_vllm,
                kv_cache,
                common_attn_metadata.slot_mapping,
            )
        impl.forward(
            mock_layer,
            query_fwd,
            key_vllm,
            value_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )

    # Use benchmark_with_power context manager
    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=warm_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"attn latency: {latency}")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    kv_cache_dtype_str = "bfloat16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "bfloat16"
    kernel_source = _dense_kernel_source(backend_name_str, impl, attn_metadata)

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "window_size": window_size,
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def get_context_attention_test_cases():
    return plan_cases("attention_context", "vllm", get_sm_version())


def get_generation_attention_test_cases():
    return plan_cases("attention_generation", "vllm", get_sm_version())


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_context_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
