# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 and GLM-5.3-Flash (Glm5Next) mHC module collector for SGLang."""

# Requires an SGLang build with DeepSeek-V4 support. Stock lmsysorg/sglang:v*
# images may not include the required deepseek_v4 modules; use a DeepSeek-V4
# capable image or put a matching SGLang source tree on PYTHONPATH.
from __future__ import annotations

# The file-level range spans the two audited routes; each architecture is
# additionally gated to its own audited release by _ARCHITECTURE_COMPAT, so
# the DeepSeek-V4 path still runs only on 0.5.14 (its ModelRunner/ServerArgs
# API use is 0.5.14-specific) and GLM only on 0.5.20.
__compat__ = "sglang>=0.5.14,<=0.5.20"

import argparse
import copy
import gc
import json
import os
import random
import sys
import tempfile
from collections.abc import Sequence
from importlib.metadata import version as get_version

import torch

os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile
    from version_resolver import _check_compat

    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_mhc_test_cases
    from registry_types import PerfFile
    from version_resolver import _check_compat

    from helper import benchmark_with_power, log_perf


DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
PERF_FILENAME = PerfFile.MHC_MODULE.value

DSV4_ARCHITECTURE = "DeepseekV4ForCausalLM"
GLM5NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"

# Per-architecture audited SGLang releases; anything else raises
# MhcRuntimeNotAuditedError (a classified failure) instead of running.
_ARCHITECTURE_COMPAT = {
    DSV4_ARCHITECTURE: "sglang==0.5.14",
    GLM5NEXT_ARCHITECTURE: "sglang==0.5.20",
}

# GLM-5.3-Flash mHC call sites in SGLang 0.5.20 serving
# (lmsysorg/sglang arm64 sha256:b0d8718a..., python/sglang/srt/models/glm5_next.py
# sha256 12c5157b..., layers/communicator_mhc.py sha256 d1b8b711...,
# kernels/ops/layernorm/mhc.py sha256 37c79625...). Every decoder layer builds
# an MHCLayerCommunicator (glm5_next.py L694-705) whose MHCState runs, per layer:
#   attention pre:  hc_attn_pre with input_layernorm as out_norm
#                   (communicator_mhc.py L84-92, called from prepare_attn L474)
#   attention post: hc_post (L97) then FFN pre: hc_ffn_pre with
#                   post_attention_layernorm (L98-105, prepare_mlp)
#   FFN post:       hc_post (mlp_combine L107-108, postprocess_layer)
# plus hc_expand on the first layer (L472) and hc_contract on the last
# (L302/L345/L361). There is no fused post->pre call for this model
# (SGLANG_OPT_FUSE_MHC_POST_PRE is read only by deepseek_v4 code). Per forward
# with L=45 layers: pre=2L=90, post=2L=90, expand=1, contract=1. The decoder
# layer's _hc_pre (glm5_next.py L709-725) calls kernels/ops/layernorm/mhc.py
# hc_pre with post_mult_value=2.0 and out_norm_weight, and hc_pre dispatches on
# SGLANG_OPT_USE_TILELANG_MHC_PRE / SGLANG_OPT_DEEPGEMM_HC_PRENORM /
# SGLANG_OPT_USE_TILELANG_MHC_POST (mhc.py L1034, L1838, L1877; defaults True,
# environ.py L1468-1470). The only server-args hooks that change them are the
# DeepseekV4 SM120/HIP branches (arg_groups/model_hook.py L372-414); the
# Glm5Next branch (L187) leaves them at their defaults, so the collector
# publishes real ServerArgs and records the resolved values.
#
# Row convention (shared with the DeepSeek-V4 rows of this table): pre and
# post rows time both per-layer sites in one graph (num_sites=2), pre
# including its input/post-attention RMSNorm (fused or not, as dispatched);
# expand/contract occur once per forward (num_sites=1).
GLM5NEXT_SGLANG_OPS = ("pre", "post", "expand", "contract")
_GLM5NEXT_NUM_SITES = {"pre": 2, "post": 2, "expand": 1, "contract": 1}


class MhcRuntimeNotAuditedError(RuntimeError):
    """The installed SGLang release is not audited for this architecture's mHC dispatch."""


def _require_audited_runtime(architecture: str, runtime_version: str) -> None:
    compat = _ARCHITECTURE_COMPAT.get(architecture)
    if compat is None:
        raise MhcRuntimeNotAuditedError(f"no audited SGLang mHC dispatch for architecture {architecture!r}")
    if not _check_compat(compat, runtime_version):
        raise MhcRuntimeNotAuditedError(
            f"SGLang {runtime_version} is not an audited mHC runtime for {architecture} (audited: {compat})"
        )


# AIC's cached HuggingFace model configs — avoids HF downloads and local
# model directories. Under dummy load_format the collector never needs
# tokenizer files or weights, so the packaged config.json alone is enough.
_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aisimulate_core",
    "model_configs",
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _read_model_config(model_id: str) -> dict:
    """Load AIC's packaged ``model_configs/<id>_config.json`` for ``model_id``.

    Only AIC-cached configs are supported — local model directories and HF
    Hub downloads are intentionally not attempted. The dummy ``load_format``
    used by this collector does not need tokenizer or weight files.
    """
    cfg_fname = model_id.replace("/", "--") + "_config.json"
    config_file = os.path.join(_MODEL_CONFIG_DIR, cfg_fname)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: expected {config_file}")
    with open(config_file) as f:
        return json.load(f)


def _default_num_tokens(model_path: str) -> list[int]:
    """Return the default num_tokens sweep for ``model_path`` from case_generator.

    Falls back to the first registered mHC test case when ``model_path`` is not
    listed in ``case_generator.py`` (e.g. custom / local model ids). All
    registered models share the same sweep, so the fallback is equivalent.
    """
    cases = get_common_mhc_test_cases()
    for case in cases:
        if case.model_name == model_path:
            return case.num_tokens_list
    if cases:
        return cases[0].num_tokens_list
    raise RuntimeError("get_common_mhc_test_cases() returned no cases")


def get_mhc_module_test_cases() -> list[dict]:
    """Return one task per model/op; each worker sweeps all num_tokens internally.

    Loading the one-layer runner is expensive, so we pay it once per op
    and model instead of per (op, model, num_tokens) combo.
    """
    cases: list[dict] = []
    # Architecture is part of the invocation identity: DeepSeek-V4-Flash and
    # GLM-5.3-Flash share (hidden_size, hc_mult) but run different call sites.
    seen: set[tuple[str, str, int, int]] = set()
    glm5next_profiles: dict[tuple[int, int], str] = {}
    for case in get_common_mhc_test_cases():
        key = (case.architecture, case.phase, case.hidden_size, case.hc_mult)
        if key in seen:
            continue
        seen.add(key)
        if case.architecture == GLM5NEXT_ARCHITECTURE:
            # GLM call sites are a framework-dispatch fact (audit above), not
            # the generator's pre/post phase pair.
            glm5next_profiles.setdefault((case.hidden_size, case.hc_mult), case.model_name)
            continue
        model_id = case.model_name.replace("/", "_")
        cases.append(
            {
                "id": f"mhc_{case.phase}_hs{case.hidden_size}_hcm{case.hc_mult}_{model_id}",
                "params": [case.phase, case.model_name],
            }
        )
    for (hidden_size, hc_mult), model_path in glm5next_profiles.items():
        model_id = model_path.replace("/", "_")
        for op in GLM5NEXT_SGLANG_OPS:
            cases.append(
                {
                    "id": f"mhc_glm5next_{op}_hs{hidden_size}_hcm{hc_mult}_{model_id}",
                    "params": [op, model_path, GLM5NEXT_ARCHITECTURE],
                }
            )
    return cases


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    filename = filename or PERF_FILENAME
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _patched_model_dir(model_id: str) -> str:
    """Build a patched model dir with a minimal ``config.json`` from AIC cache.

    Steps:
    1. Read the original config from AIC's packaged ``model_configs/``.
    2. Write a patched ``config.json`` into a temp dir — weights and tokenizer
       are NOT needed because the collector always runs with
       ``load_format="dummy"``.
    3. Preset ``SGLANG_DSV4_FP4_EXPERTS`` from ``original_config.expert_dtype``,
       since SGLang would otherwise probe routed-expert dtype from safetensors.
       An explicit user-provided env var always wins.
    """
    original_config = _read_model_config(model_id)
    config = copy.deepcopy(original_config)

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    config["num_hidden_layers"] = num_layers  # shrink depth to speed up collector init
    if config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        config["architectures"] = ["DeepseekV4ForCausalLM"]
    # Match collect_dsv4_attn.py: current Transformers does not know a
    # native deepseek_v4 config, while SGLang selects the V4 model class from
    # the architectures field.
    config["model_type"] = "deepseek_v3"

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_mhc_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Preset FP4 experts env from the untouched original config.
    if "SGLANG_DSV4_FP4_EXPERTS" not in os.environ:
        expert_dtype = str(original_config.get("expert_dtype", "")).lower()
        fp4_value = "1" if expert_dtype == "fp4" else "0"
        os.environ["SGLANG_DSV4_FP4_EXPERTS"] = fp4_value
        print(f"[mhc-collector] auto-set SGLANG_DSV4_FP4_EXPERTS={fp4_value} (expert_dtype={expert_dtype or 'unset'})")

    print(f"[mhc-collector] patched_dir={tmp_dir} model_id={model_id}")
    return tmp_dir


def _load_one_layer_runner(
    model_path: str,
    device: str,
    mem_fraction_static: float,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)

    local_model_path = _patched_model_dir(model_path)
    gpu_id = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype="fp8_e4m3",
        max_total_tokens=4096,
        max_running_requests=16,
        max_prefill_tokens=4096,
    )
    server_args.attention_backend = "dsv4"

    print(f"[mhc-collector] model_path {model_path} -> {local_model_path}")

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return ModelRunner(
        model_config=model_config,
        mem_fraction_static=mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=29500 + random.randint(0, 10000),
        server_args=server_args,
    )


def _hidden_size(layer) -> int:
    return int(layer.config.hidden_size)


def _make_residual(layer, num_tokens: int, device: str) -> torch.Tensor:
    return torch.randn(
        num_tokens,
        layer.hc_mult,
        _hidden_size(layer),
        dtype=torch.bfloat16,
        device=device,
    )


def _mhc_call_args(layer):
    # A real DSV4 layer executes mHC once before attention and once before FFN.
    # This collector folds both calls into the reported pre/post op.
    return (
        (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, layer.input_layernorm),
        (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, layer.post_attention_layernorm),
    )


def _hc_pre_post_inputs(hc_pre_output):
    if len(hc_pre_output) == 3:
        return hc_pre_output
    if len(hc_pre_output) == 4:
        x, post, comb, _norm_fused = hc_pre_output
        return x, post, comb
    raise ValueError(f"unexpected hc_pre output arity: {len(hc_pre_output)}")


def _make_kernel(layer, op: str, residual: torch.Tensor):
    if op == "pre":
        call_args = _mhc_call_args(layer)

        def kernel():
            return [layer.hc_pre(residual, fn, scale, base, norm=norm) for fn, scale, base, norm in call_args]

        return kernel

    if op == "post":
        with torch.no_grad():
            post_inputs = [
                _hc_pre_post_inputs(layer.hc_pre(residual, fn, scale, base, norm=norm))
                for fn, scale, base, norm in _mhc_call_args(layer)
            ]
        torch.cuda.synchronize()

        def kernel():
            return [layer.hc_post(x, residual, post, comb) for x, post, comb, *_ in post_inputs]

        return kernel

    raise ValueError(f"unsupported mHC op: {op}")


def _benchmark_mhc_kernel(
    *,
    device: str,
    kernel_func,
    num_warmup: int,
    num_iterations: int,
) -> dict:
    def timed_kernel():
        with torch.no_grad():
            return kernel_func()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=True,
    ) as bench_result:
        pass

    if not bench_result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    return bench_result


def _log_result(
    *,
    output_path: str | None,
    perf_filename: str | None,
    op: str,
    num_tokens: int,
    num_sites: int,
    hc_mult: int,
    hidden_size: int,
    sinkhorn_iters: int,
    latency_ms: float,
    version: str,
    device_name: str,
    kernel_source: str,
    power_stats: dict | None,
    architecture: str = DSV4_ARCHITECTURE,
) -> None:
    if not log_perf(
        item_list=[
            {
                "architecture": architecture,
                "num_tokens": num_tokens,
                "num_sites": num_sites,
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
                "sinkhorn_iters": sinkhorn_iters,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        op_name=op,
        kernel_source=kernel_source,
        perf_filename=_resolve_perf_path(output_path, perf_filename),
        power_stats=power_stats,
    ):
        raise RuntimeError("Failed to persist SGLang mHC performance row")


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    model_path: str = DEFAULT_MODEL,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    perf_filename: str | None = None,
) -> list[dict[str, float]]:
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    _require_audited_runtime(DSV4_ARCHITECTURE, get_version("sglang"))

    token_cases = [int(num_tokens) for num_tokens in (num_tokens_cases or _default_num_tokens(model_path))]
    results: list[dict[str, float]] = []
    error_count = 0
    model_runner = None

    try:
        # Load inside the guarded region: a mid-init failure (e.g. after
        # SGLang created its TP/world groups but before finishing) must still
        # reach the teardown below, or the next task in this worker inherits
        # half-initialized module globals.
        model_runner = _load_one_layer_runner(
            model_path,
            device=device,
            mem_fraction_static=mem_fraction_static,
        )

        layer = model_runner.model.model.layers[0]
        hidden_size = _hidden_size(layer)
        version = get_version("sglang")
        device_name = torch.cuda.get_device_name(device)

        # Print the RESOLVED kernel-selection env values, not only the raw
        # process environment: MHC-PRENORM-ENV history shows the module-level
        # default and the central collect_sglang() setdefault can disagree, and
        # the H20 log that omitted the resolved prenorm value could not prove
        # which kernel won.
        from sglang.srt.environ import envs as _envs

        print(
            "[mhc-collector] "
            f"hc_mult={layer.hc_mult}, hidden_size={hidden_size}, "
            f"tilelang_pre={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_PRE', 'default')}, "
            f"tilelang_post={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_POST', 'default')}, "
            f"resolved_tilelang_pre={_envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()}, "
            f"resolved_tilelang_post={_envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()}, "
            f"resolved_deepgemm_hc_prenorm={_envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()}"
        )

        for op in ops:
            from sglang.srt.environ import envs

            if op == "pre":
                if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
                    kernel_source = "sglang_tilelang_mhc_pre"
                elif envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
                    kernel_source = "sglang_deepgemm_mhc_pre"
                else:
                    kernel_source = "sglang_torch_mhc_pre"
            elif op == "post":
                kernel_source = (
                    "sglang_tilelang_mhc_post"
                    if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
                    else "sglang_torch_mhc_post"
                )
            else:
                raise ValueError(f"unsupported mHC op: {op}")
            for num_tokens in token_cases:
                try:
                    residual = _make_residual(layer, num_tokens, device)
                    bench_result = _benchmark_mhc_kernel(
                        device=device,
                        kernel_func=_make_kernel(layer, op, residual),
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: op={op}, num_tokens={num_tokens}; skipping")
                    error_count += 1
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as err:
                    # Runtime-incompatible kernels (e.g. tilelang version mismatch,
                    # unsupported shapes) should skip the single case rather than
                    # abort the whole sweep.
                    print(f"  RuntimeError: op={op}, num_tokens={num_tokens}; skipping ({err})")
                    error_count += 1
                    torch.cuda.empty_cache()
                    continue

                latency_ms = float(bench_result["latency_ms"])
                _log_result(
                    output_path=output_path,
                    perf_filename=perf_filename,
                    op=op,
                    num_tokens=num_tokens,
                    num_sites=len(_mhc_call_args(layer)),
                    hc_mult=layer.hc_mult,
                    hidden_size=hidden_size,
                    sinkhorn_iters=int(getattr(layer.config, "hc_sinkhorn_iters", 20)),
                    latency_ms=latency_ms,
                    version=version,
                    device_name=device_name,
                    kernel_source=kernel_source,
                    power_stats=bench_result.get("power_stats"),
                )
                results.append(
                    {
                        "op": op,
                        "num_tokens": num_tokens,
                        "mean_ms": latency_ms,
                        "n": int(bench_result.get("num_runs_executed", num_iterations)),
                        "used_cuda_graph": True,
                        "throttled": bool(bench_result.get("throttled", False)),
                    }
                )
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        del model_runner
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.eplb import expert_location as _expert_location

        # Mirror SGLang 0.5.14 cleanup_dist_env_and_memory: destroying only the
        # torch process group leaves parallel_state._WORLD pointing at a dead
        # group, so the NEXT task in the same worker fails ModelRunner init
        # with "not initialized in the world group map". H20 never sequenced
        # two mHC tasks through one worker (4 tasks over 8 GPU workers); the
        # single-worker B200 smoke exposed it. Teardown errors still propagate
        # and fail the worker rather than hiding retained groups.
        destroy_model_parallel()
        destroy_distributed_environment()
        # SGLang has no public reset for this module global (its serving
        # process never re-creates a ModelRunner); set_global_... asserts None,
        # so a second in-worker task fails init unless it is returned to the
        # module's pre-init state here.
        _expert_location._global_expert_location_metadata = None
        torch.cuda.empty_cache()
        gc.collect()
    summary = f"ok={len(results)} error={error_count} skip=0 total={len(results) + error_count}"
    print(f"[mhc-collector] {summary}")
    if not results or error_count > 0:
        raise RuntimeError(f"mHC sweep failed: {summary}")
    return results


def _init_glm5next_runtime(model_path: str, *, device: str, mem_fraction_static: float):
    """Publish GLM ServerArgs and init the TP=1 groups like the serving scheduler.

    Mirrors the serving process start: run_scheduler_process publishes the
    ServerArgs before anything reads config (managers/scheduler.py L5807-5808,
    which also runs the model hooks), then ModelRunner initialises torch
    distributed (model_executor/model_runner.py L1144-1157 ->
    distributed/bootstrap.py L70-172) and the DeepGEMM config
    (model_runner.py L456-458). No model weights are loaded: the mHC kernels
    only need this process state plus the layer's own parameters.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)
    gpu_id = device_obj.index if device_obj.index is not None else torch.cuda.current_device()

    original_config = _read_model_config(model_path)
    if original_config.get("architectures") != [GLM5NEXT_ARCHITECTURE]:
        raise ValueError(f"{model_path!r} is not a {GLM5NEXT_ARCHITECTURE} config")
    tmp_dir = os.path.join(tempfile.gettempdir(), f"aic_mhc_{model_path.replace('/', '_')}_{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(original_config, f)

    server_args = ServerArgs(
        model_path=tmp_dir,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        skip_tokenizer_init=True,
    )
    _set_envs_and_config(server_args)
    publish(server_args, role="scheduler")
    model_config = ModelConfig.from_server_args(server_args)
    bootstrap.init_torch_distributed(
        server_args=server_args,
        model_config=model_config,
        device="cuda",
        ps=ParallelState.trivial(gpu_id=gpu_id),
        dist_port=29500 + random.randint(0, 10000),
        is_draft_worker=False,
        local_omp_cpuid=None,
    )
    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        deep_gemm_wrapper.update_deep_gemm_config(gpu_id)
    text_config = model_config.hf_text_config
    if not getattr(text_config, "mhc", False):
        raise ValueError(f"{model_path!r} text config has mhc disabled")
    return text_config


def _teardown_glm5next_runtime() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from sglang.srt.runtime_context import reset_context

    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()
    torch.cuda.empty_cache()
    gc.collect()


def _build_glm5next_sites(text_config, *, device: str):
    """One Glm5NextDecoderLayer's mHC state, without its attention/MLP.

    Binds the serving decoder layer's own ``_hc_pre`` / ``hc_attn_pre`` /
    ``hc_ffn_pre`` / ``hc_post`` (glm5_next.py L709-756), so every kernel
    argument is the one serving passes; parameters and norms mirror
    ``Glm5NextDecoderLayer.__init__`` (L657-679). The norms are bf16 like the
    served (bfloat16) checkpoint weights.
    """
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

    class _Glm5NextMhcSites(torch.nn.Module):
        _hc_pre = Glm5NextDecoderLayer._hc_pre
        hc_attn_pre = Glm5NextDecoderLayer.hc_attn_pre
        hc_ffn_pre = Glm5NextDecoderLayer.hc_ffn_pre
        hc_post = Glm5NextDecoderLayer.hc_post

        def __init__(self) -> None:
            super().__init__()
            self.config = text_config
            hc_mult = int(text_config.hc_mult)
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * int(text_config.hidden_size)
            for site in ("attn", "ffn"):
                setattr(self, f"hc_{site}_base", torch.nn.Parameter(torch.randn(mix_hc, dtype=torch.float32)))
                setattr(self, f"hc_{site}_scale", torch.nn.Parameter(torch.ones(3, dtype=torch.float32)))
                setattr(self, f"hc_{site}_fn", torch.nn.Parameter(torch.randn(mix_hc, hc_dim, dtype=torch.float32)))
            eps = text_config.rms_norm_eps
            self.input_layernorm = RMSNorm(text_config.hidden_size, eps=eps, weight_dtype=torch.bfloat16)
            self.post_attention_layernorm = RMSNorm(text_config.hidden_size, eps=eps, weight_dtype=torch.bfloat16)

    return _Glm5NextMhcSites().to(device)


def _glm5next_mhc_states(sites):
    """The two per-layer pre sites as the communicator's own MHCState objects.

    MHCState.attn_split (communicator_mhc.py L84-92) is the attention pre; the
    FFN pre inside attn_to_mlp (L98-105) is the same code with hc_ffn_pre and
    post_attention_layernorm, so a second MHCState bound to hc_ffn_pre runs it
    verbatim without the preceding hc_post.
    """
    from sglang.srt.layers.communicator_mhc import MHCState

    hc_mult = int(sites.config.hc_mult)
    return (
        (MHCState(hc_mult, sites.hc_attn_pre, sites.hc_ffn_pre, sites.hc_post), sites.input_layernorm),
        (MHCState(hc_mult, sites.hc_ffn_pre, sites.hc_ffn_pre, sites.hc_post), sites.post_attention_layernorm),
    )


def _glm5next_kernel(sites, op: str, num_tokens: int, *, device: str):
    from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand

    hc_mult = int(sites.config.hc_mult)
    hidden_size = int(sites.config.hidden_size)

    def stream():  # widened residual stream [T, hc_mult * H]
        return torch.randn(num_tokens, hc_mult * hidden_size, dtype=torch.bfloat16, device=device)

    def layer_out():  # sublayer output [T, H]
        return torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    if op == "expand":
        x = layer_out()
        return lambda: hc_expand(x, hc_mult)
    if op == "contract":
        x = stream()
        return lambda: hc_contract(x, hc_mult)

    states = _glm5next_mhc_states(sites)
    streams = [stream() for _ in states]
    if op == "pre":
        return lambda: [state.attn_split(x, out_norm=norm) for x, (state, norm) in zip(streams, states, strict=True)]
    if op == "post":
        with torch.no_grad():
            for x, (state, norm) in zip(streams, states, strict=True):
                state.attn_split(x, out_norm=norm)
        outs = [layer_out() for _ in states]
        torch.cuda.synchronize()
        return lambda: [
            sites.hc_post(x, residual, state.h_res, state.h_post)
            for x, residual, (state, _norm) in zip(outs, streams, states, strict=True)
        ]
    raise ValueError(f"unsupported GLM mHC op: {op}")


def _glm5next_kernel_source(sites, op: str, *, device: str) -> str:
    """Name what actually runs; the pre label records the observed norm fusion."""
    if op == "expand":
        return "sglang.kernels.ops.layernorm.mhc.hc_expand"
    if op == "contract":
        return "sglang.kernels.ops.layernorm.mhc.hc_contract"
    from sglang.srt.environ import envs

    if op == "post":
        return "sglang.hc_post[tilelang]" if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get() else "sglang.hc_post[torch]"
    if op == "pre":
        if not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            impl = "torch"
        elif envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            impl = "tilelang,prenorm_gemm=deepgemm"
        else:
            impl = "tilelang,prenorm_gemm=tilelang"
        hc_mult = int(sites.config.hc_mult)
        probe = torch.randn(1, hc_mult * int(sites.config.hidden_size), dtype=torch.bfloat16, device=device)
        norm = sites.input_layernorm
        with torch.no_grad():
            *_outputs, norm_fused = sites.hc_attn_pre(probe, norm.weight.data, norm.variance_epsilon)
        return f"sglang.hc_pre[{impl},norm={'fused' if norm_fused else 'separate'}]"
    raise ValueError(f"unsupported GLM mHC op: {op}")


def run_glm5next_mhc_module(
    *,
    ops: Sequence[str],
    model_path: str,
    num_tokens_cases: Sequence[int] | None = None,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    perf_filename: str | None = None,
) -> list[dict[str, float]]:
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    version = get_version("sglang")
    _require_audited_runtime(GLM5NEXT_ARCHITECTURE, version)
    for op in ops:
        if op not in GLM5NEXT_SGLANG_OPS:
            raise ValueError(f"unsupported GLM mHC op: {op}")

    token_cases = [int(num_tokens) for num_tokens in (num_tokens_cases or _default_num_tokens(model_path))]
    results: list[dict[str, float]] = []
    error_count = 0
    try:
        text_config = _init_glm5next_runtime(model_path, device=device, mem_fraction_static=mem_fraction_static)
        sites = _build_glm5next_sites(text_config, device=device)
        device_name = torch.cuda.get_device_name(device)

        from sglang.srt.environ import envs as _envs

        print(
            "[mhc-collector] "
            f"arch={GLM5NEXT_ARCHITECTURE}, hc_mult={text_config.hc_mult}, hidden_size={text_config.hidden_size}, "
            f"sinkhorn={text_config.hc_sinkhorn_iters}, hc_eps={text_config.hc_eps}, "
            f"rms_eps={text_config.rms_norm_eps}, "
            f"resolved_tilelang_pre={_envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()}, "
            f"resolved_tilelang_post={_envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()}, "
            f"resolved_deepgemm_hc_prenorm={_envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()}"
        )
        for op in ops:
            kernel_source = _glm5next_kernel_source(sites, op, device=device)
            for num_tokens in token_cases:
                try:
                    bench_result = _benchmark_mhc_kernel(
                        device=device,
                        kernel_func=_glm5next_kernel(sites, op, num_tokens, device=device),
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: op={op}, num_tokens={num_tokens}; recorded as failure")
                    error_count += 1
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as err:
                    print(f"  RuntimeError: op={op}, num_tokens={num_tokens}; recorded as failure ({err})")
                    error_count += 1
                    torch.cuda.empty_cache()
                    continue

                latency_ms = float(bench_result["latency_ms"])
                _log_result(
                    output_path=output_path,
                    perf_filename=perf_filename,
                    op=op,
                    num_tokens=num_tokens,
                    num_sites=_GLM5NEXT_NUM_SITES[op],
                    hc_mult=int(text_config.hc_mult),
                    hidden_size=int(text_config.hidden_size),
                    sinkhorn_iters=int(text_config.hc_sinkhorn_iters),
                    latency_ms=latency_ms,
                    version=version,
                    device_name=device_name,
                    kernel_source=kernel_source,
                    power_stats=bench_result.get("power_stats"),
                    architecture=GLM5NEXT_ARCHITECTURE,
                )
                print(f"[mhc-collector] op={op} tokens={num_tokens} latency={latency_ms:.4f} ms")
                results.append({"op": op, "num_tokens": num_tokens, "mean_ms": latency_ms})
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        _teardown_glm5next_runtime()
    summary = f"ok={len(results)} error={error_count} skip=0 total={len(results) + error_count}"
    print(f"[mhc-collector] {summary}")
    if not results or error_count > 0:
        raise RuntimeError(f"mHC sweep failed: {summary}")
    return results


def run_mhc_module_worker(
    op: str,
    model_path: str | None = None,
    architecture: str = DSV4_ARCHITECTURE,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """Worker-compatible wrapper used by collector/collect.py.

    Each call sweeps all num_tokens for a single model/op pair in one
    subprocess. Direct callers that pass only ``op`` still use
    ``COLLECTOR_MODEL_PATH`` (set by ``collect.py --model-path``) or the
    default Pro model. ``perf_filename`` and ``device`` are keyword-only args
    supplied by collect.py via functools.partial and the worker dispatch loop.
    """
    model_path = model_path or os.environ.get("COLLECTOR_MODEL_PATH") or DEFAULT_MODEL
    output_path = os.path.dirname(perf_filename) or os.getcwd()
    if architecture == GLM5NEXT_ARCHITECTURE:
        run_glm5next_mhc_module(
            ops=[op],
            model_path=model_path,
            device=device,
            output_path=output_path,
            perf_filename=os.path.basename(perf_filename),
        )
        return
    if architecture != DSV4_ARCHITECTURE:
        raise MhcRuntimeNotAuditedError(f"no audited SGLang mHC dispatch for architecture {architecture!r}")
    run_mhc_module(
        ops=[op],
        num_tokens_cases=None,
        model_path=model_path,
        device=device,
        output_path=output_path,
        perf_filename=os.path.basename(perf_filename),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DeepSeek-V4 / GLM-5.3-Flash mHC module latency on SGLang.")
    parser.add_argument("--architecture", choices=[DSV4_ARCHITECTURE, GLM5NEXT_ARCHITECTURE], default=DSV4_ARCHITECTURE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--op", choices=["pre", "post", "expand", "contract", "all"], default="all")
    parser.add_argument("--num-tokens", default=None)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = parser.parse_args()

    if args.architecture == GLM5NEXT_ARCHITECTURE:
        run_glm5next_mhc_module(
            ops=list(GLM5NEXT_SGLANG_OPS) if args.op == "all" else [args.op],
            model_path=args.model_path,
            num_tokens_cases=_parse_int_list(args.num_tokens) if args.num_tokens else None,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            device=args.device,
            output_path=args.output_path,
            mem_fraction_static=args.mem_fraction_static,
        )
        return
    if args.op not in {"pre", "post", "all"}:
        parser.error(f"--op {args.op} is GLM-only")
    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens) if args.num_tokens else None,
        model_path=args.model_path,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=args.device,
        output_path=args.output_path,
        mem_fraction_static=args.mem_fraction_static,
    )


if __name__ == "__main__":
    main()
