# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local NVFP4 MoE timing for the GLM-5.2 TP4/EP1 Rubin pilot.

Case and perf-row contracts follow this repository's sglang/collect_moe.py.
Only the supported runtime is imported at execution time. Framework references
are relative to python/sglang at immutable dl/sglang/sglang commit
02c5a855aceb968c310e6fbc6632270e26edc84b; no framework implementation is copied.
The campaign must select TP4/EP1 before queueing. Unsupported queued cases raise.
"""

from __future__ import annotations

import gc
import importlib.metadata
import tempfile
from contextlib import contextmanager
from pathlib import Path

from collector.sglang_rubin.collect_gemm import (
    PILOT_MODEL,
    SGLANG_DISTRIBUTION_VERSION,
    _require_model_scope,
    _require_runtime,
)

__compat__ = f"sglang=={SGLANG_DISTRIBUTION_VERSION}"
# Frozen serving uses --max-running-requests 32 --cuda-graph-max-bs-decode 32.
# This is a tuning ceiling, not a case filter or a limit on measured tokens.
PILOT_MAX_DECODE_TOKENS = 32


def get_moe_test_cases():
    """Expand canonical cases; topology narrowing belongs to the campaign."""
    from collector.case_generator import get_common_moe_test_cases, moe_model_allows_quantization

    _require_model_scope()
    cases = []
    seen = {}
    for case in get_common_moe_test_cases(backend="sglang"):
        if case.model_name != PILOT_MODEL or not moe_model_allows_quantization("sglang", case.model_name, "nvfp4"):
            raise ValueError(f"Rubin pilot case declaration is not the NVFP4 checkpoint: {case.model_name}")
        for tokens in case.num_tokens_list:
            # arg_groups/overrides.py:1826-1880 resolves DeepSeek-family
            # modelopt_fp4 + a2a=none + SM10x to flashinfer_trtllm. GLM DSA
            # is included in that architecture family (:1810).
            args = [
                "nvfp4",
                tokens,
                case.hidden_size,
                case.inter_size,
                case.topk,
                case.num_experts,
                case.tp,
                case.ep,
                case.model_name,
                case.token_expert_distribution,
                case.power_law_alpha,
                case.sglang_moe_swiglu_limit,
                "flashinfer_trtllm",
                case.sglang_moe_activation,
                case.sglang_moe_is_gated,
                case.sglang_moe_has_bias,
                case.sglang_moe_gemm1_alpha,
                case.sglang_moe_gemm1_clamp_limit,
                case.sglang_moe_scoring_func,
                case.sglang_moe_routing_method_type,
                case.sglang_moe_routed_scaling_factor,
                case.sglang_moe_renormalize,
                case.sglang_moe_has_correction_bias,
                case.sglang_moe_num_expert_group,
                case.sglang_moe_topk_group,
                case.sglang_moe_apply_router_weight_on_input,
                False,
            ]
            identity = tuple(args[:8] + args[9:11])
            execution = tuple(args[11:])
            if identity in seen and seen[identity] != execution:
                raise ValueError(f"Rubin MoE declarations have conflicting execution semantics for {identity!r}")
            if identity not in seen:
                seen[identity] = execution
                cases.append(args)
    return cases


def _kernel_source(layer, topk_output):
    """Confirm the constructed runner and routing leaf, not a requested label."""
    # fused_moe_triton/layer.py:387-391 resolves the launch environment into
    # layer state. Require finalized compute in both phases: our perf key has
    # no phase axis, and deepseek_v2.py:978-1023 otherwise uses deferred output
    # plus a separate native finalizer during captured serving forwards.
    if getattr(layer, "supports_deferred_finalize", None) is not False:
        raise RuntimeError("Rubin MoE requires a constructed layer with deferred finalization disabled")
    method = layer.quant_method
    runner = getattr(method, "runner", None)
    method_backend = getattr(getattr(method, "_moe_runner_backend", None), "value", None)
    runner_backend = getattr(getattr(runner, "runner_backend", None), "value", None)
    fused = getattr(runner, "fused_func", None)
    if (
        type(method).__name__ != "ModelOptNvFp4FusedMoEMethod"
        or method_backend != "flashinfer_trtllm"
        or runner_backend != method_backend
        or not getattr(method, "enable_flashinfer_trtllm_moe", False)
        or getattr(fused, "__name__", None) != "fused_experts_none_to_flashinfer_trtllm"
        or getattr(fused, "__module__", None) != "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
        or not hasattr(layer, "g1_scale_c")
        or type(topk_output).__name__ != "BypassedTopKOutput"
    ):
        raise RuntimeError(
            "Rubin NVFP4 MoE did not construct the expected fused TRTLLM path: "
            f"method={type(method).__name__}, method_backend={method_backend}, runner_backend={runner_backend}, "
            f"fused={getattr(fused, '__name__', None)}, topk={type(topk_output).__name__}"
        )
    # modelopt_quant.py:2794-2836 constructs FP4 quant_info; the registered
    # fused func at moe_runner/flashinfer_trtllm.py:1291-1301 selects its FP4
    # leaf. BypassedTopKOutput takes trtllm_fp4_block_scale_moe (:1100-1155),
    # rather than the separately routed kernel (:1069+).
    return "sglang_flashinfer_trtllm_moe"


@contextmanager
def _rank_local_context(device):
    """Real singleton communication context, with one TP4 weight shard.

    The perf contract times local MoE compute; TP reduction is a separate op.
    FusedMoE reads its weight partition from get_parallel() (layer.py:282-319).
    ParallelContext.override (runtime_context.py:158) is the framework's scoped
    injection API. A real singleton group satisfies framework allocation calls
    without fake get_tp_group bindings; no collective is timed. Its symmetric
    allocator is explicitly a no-op for world_size=1 (pynccl_allocator.py:325).
    """
    import torch
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    if torch.distributed.is_initialized():
        raise RuntimeError(
            "Rubin rank-local MoE requires a dedicated collector worker without an existing process group"
        )
    with (
        get_context().override_server_args(
            tp_size=4,
            ep_size=1,
            dp_size=1,
            enable_dp_attention=False,
            quantization="modelopt_fp4",
            moe_runner_backend="flashinfer_trtllm",
            moe_a2a_backend="none",
            disable_shared_experts_fusion=True,
        ),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
            a2a_backend=MoeA2ABackend.NONE,
            disable_shared_experts_fusion=True,
            quantization="modelopt_fp4",
        ),
        get_flags().dp.override(enabled=False),
        tempfile.TemporaryDirectory(prefix="aisim-rubin-moe-") as directory,
    ):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=torch.device(device).index or 0,
                distributed_init_method=(Path(directory) / "rendezvous").as_uri(),
                backend="nccl",
                timeout=60,
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
            with get_parallel().override(
                tp_size=4,
                tp_rank=0,
                moe_tp_size=4,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ):
                yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _measure_moe(kernel_func, device, *, tuning_func):
    import torch
    from flashinfer.autotuner import AutoTuner, autotune
    from sglang.srt.model_executor.runner.flashinfer_autotune import get_flashinfer_autotune_skip_ops

    from collector.helper import benchmark_with_power

    # base_runner.py:242-248 and decode_cuda_graph_runner.py:516-528 tune the
    # largest decode shape before capture. flashinfer_autotune.py:249-265 skips
    # extend tuning by default. Keep its native buckets and skip policy; full
    # prefill inputs must reach the native untuned fallback without being tuned.
    # _rank_local_context requires a dedicated worker. Reset its public native
    # cache per case so previously collected shapes cannot supply extra tactics.
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        raise RuntimeError("Rubin MoE requires a dedicated collector worker outside an autotune context")
    tuner.clear_cache()
    try:
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), autotune(True, skip_ops=get_flashinfer_autotune_skip_ops(None)):
            tuning_func()
        stream.synchronize()
        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=5,
            num_runs=10,
            repeat_n=1,
        ) as results:
            pass
        return results
    finally:
        tuner.clear_cache()


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0,
    swiglu_limit=None,
    moe_backend="triton",
    activation="silu",
    is_gated=True,
    has_bias=False,
    gemm1_alpha=None,
    gemm1_clamp_limit=None,
    scoring_func="softmax",
    routing_method_type=None,
    routed_scaling_factor=None,
    renormalize=True,
    has_correction_bias=False,
    num_expert_group=None,
    topk_group=None,
    apply_router_weight_on_input=False,
    is_fp4_experts=False,
    *,
    perf_filename,
    device="cuda:0",
):
    expected = (
        "nvfp4",
        6144,
        2048,
        8,
        256,
        4,
        1,
        PILOT_MODEL,
        None,
        "flashinfer_trtllm",
        "silu",
        True,
        False,
        None,
        None,
        "sigmoid",
        "DeepSeekV3",
        2.5,
        True,
        True,
        1,
        1,
        False,
        False,
    )
    requested = (
        moe_type,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        model_name,
        swiglu_limit,
        moe_backend,
        activation,
        is_gated,
        has_bias,
        gemm1_alpha,
        gemm1_clamp_limit,
        scoring_func,
        routing_method_type,
        routed_scaling_factor,
        renormalize,
        has_correction_bias,
        num_expert_group,
        topk_group,
        apply_router_weight_on_input,
        is_fp4_experts,
    )
    if requested != expected:
        raise ValueError(f"Unsupported Rubin MoE case; pilot requires GLM-5.2 NVFP4 TP4/EP1 routing: {requested!r}")
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
        raise ValueError("MoE num_tokens must be a positive integer")
    if distributed not in {"balanced", "power_law", "uniform"}:
        raise ValueError(f"Unsupported token distribution: {distributed!r}")
    _require_runtime()

    from sglang.srt.environ import envs

    # Native EnvBool(False) in srt/environ.py:919 accepts an unset variable.
    # Read its resolved value; never repair the caller's launch environment.
    if envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        raise RuntimeError("Rubin MoE pilot requires SGLANG_FLASHINFER_AUTOTUNE_EXTEND=false")

    import torch
    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

    from collector.helper import WORKER_RESTART, balanced_logits, log_perf, power_law_logits_v3

    torch.cuda.set_device(device)
    layer = hidden_states = topk_layer = probe = logits = None
    try:
        with _rank_local_context(device), torch.device(device):
            # The GLM DSA class inherits DeepseekV2ForCausalLM
            # (models/glm4_moe.py:1447); its routed expert constructor and
            # grouped router are models/deepseek_v2.py:646-695. Tensor shapes,
            # scale dtypes and load-time packing are owned by the selected
            # quant method (modelopt_quant.py:2215-2373), not recreated here.
            quant = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
            layer = get_moe_impl_class(quant)(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                layer_id=3,
                top_k=topk,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                quant_config=quant,
                prefix="model.layers.3.mlp.experts",
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=RoutingMethodType.DeepSeekV3,
            ).to(device)
            with torch.no_grad():
                for name, parameter in layer.named_parameters():
                    parameter.fill_(1) if "scale" in name or "alpha" in name else parameter.zero_()
            layer.quant_method.process_weights_after_loading(layer)
            topk_layer = TopK(
                top_k=topk,
                layer_id=3,
                use_grouped_topk=True,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                renormalize=renormalize,
                scoring_func=scoring_func,
                correction_bias=torch.zeros(num_experts, dtype=torch.float32, device=device),
                quant_config=quant,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=layer.should_fuse_routed_scaling_factor_in_topk,
            )
            hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            # Match the shared collector's routing distributions; these are
            # synthetic workload inputs, not checkpoint model weights.
            if distributed == "balanced":
                logits = [
                    balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)
                    for _ in range(5)
                ]
            elif distributed == "power_law":
                logits = [
                    power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(
                        device=device, dtype=torch.float32
                    )
                    for _ in range(5)
                ]
            else:
                logits = [torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(5)]
            probe = topk_layer(hidden_states, logits[0])
            kernel_source = _kernel_source(layer, probe)

            def kernel_func():
                for router_logits in logits:
                    layer(hidden_states, topk_layer(hidden_states, router_logits))

            def tuning_func():
                # Views preserve all five synthetic inputs. A smaller case only
                # needs its reachable decode buckets; no measured shape changes.
                hidden = hidden_states[:PILOT_MAX_DECODE_TOKENS]
                for router_logits in logits:
                    layer(hidden, topk_layer(hidden, router_logits[:PILOT_MAX_DECODE_TOKENS]))

            results = _measure_moe(kernel_func, device, tuning_func=tuning_func)
            if not log_perf(
                item_list=[
                    {
                        "moe_dtype": moe_type,
                        "num_tokens": num_tokens,
                        "hidden_size": hidden_size,
                        "inter_size": inter_size,
                        "topk": topk,
                        "num_experts": num_experts,
                        "moe_tp_size": moe_tp_size,
                        "moe_ep_size": moe_ep_size,
                        "distribution": "power_law_" + str(power_law_alpha)
                        if distributed == "power_law"
                        else distributed,
                        "latency": results["latency_ms"] / len(logits),
                    }
                ],
                framework="SGLang",
                version=importlib.metadata.version("sglang"),
                device_name=torch.cuda.get_device_name(device),
                op_name="moe",
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=results["power_stats"],
            ):
                raise RuntimeError("Failed to persist Rubin MoE performance row")
    finally:
        if layer is not None:
            for parameter in layer.parameters():
                parameter.__dict__.pop("weight_loader", None)
            parameter = None
        layer = hidden_states = topk_layer = probe = logits = None
        gc.collect()
        torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(device) > torch.cuda.get_device_properties(device).total_memory // 4:
        return WORKER_RESTART
