# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide sglang runtime config for the module collectors, across sglang generations.

sglang 0.5.21 (runtime_context "bags"): a worker process must ``publish(server_args, role=...,
ranks=SpawnRanks(...))`` first and the config initializers then take NO arguments
(``benchmark/one_batch.py:893-902`` @0.5.21: publish -> initialize_moe_config() ->
initialize_fp8_gemm_config() -> initialize_fp4_gemm_config()); ``ModelRunner.__init__`` no longer
accepts the parallel geometry (``model_executor/model_runner.py``: model_config, mem_fraction_static,
gpu_id, nccl_port, server_args, ...) — it reads ``get_parallel()``.
sglang 0.5.16: the initializers take ``server_args`` and the geometry travels as
``ps=ParallelState.trivial(gpu_id)`` (parallel_state_wrapper.py:6-24).
Older pins: initializers take ``server_args`` and ModelRunner takes tp_rank/tp_size/... keywords.

This only changes HOW the same engine state is published (layer_permissions.md API-compat shim
rule); the kernels the runner selects are untouched.
"""

from __future__ import annotations

import inspect


def init_runtime_config(server_args, gpu_id: int, *, nccl_port: int | None = None, model_config=None,
                        tp_rank: int = 0, tp_size: int | None = None) -> dict:
    """Run the framework's own config initializers and return the ModelRunner parallel kwargs.

    0.5.21 additionally needs the distributed world built BEFORE ModelRunner (model_runner.py
    init_torch_distributed -> bootstrap.measure_pre_model_load_memory reads get_world_group()):
    scheduler.py:525-530 and benchmark/one_batch.py load_model run bootstrap.init_parallel_runtime
    (publish -> groups) then bootstrap.init_layer_runtime(model_config). Pass nccl_port + model_config.
    """
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

    if inspect.signature(initialize_moe_config).parameters:  # <= 0.5.20: seeded from ServerArgs directly
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)
        try:
            from sglang.srt.distributed.parallel_state_wrapper import ParallelState

            return {"ps": ParallelState.trivial(gpu_id=gpu_id)}
        except ImportError:
            return {"tp_rank": tp_rank, "tp_size": tp_size or getattr(server_args, "tp_size", 1),
                    "pp_rank": 0, "pp_size": 1, "moe_ep_rank": 0, "moe_ep_size": 1}
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank

    publish(server_args, role="scheduler",
            ranks=SpawnRanks(world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    from sglang.srt.distributed import bootstrap

    if nccl_port is None or model_config is None:
        raise RuntimeError("sglang>=0.5.21 runner init needs nccl_port and model_config (bootstrap.init_parallel_runtime/init_layer_runtime)")
    try:
        from sglang.srt.runtime_context import get_device

        device = get_device().device
    except (ImportError, AttributeError):
        device = getattr(server_args, "device", "cuda")
    bootstrap.init_parallel_runtime(server_args=server_args, device=device, dist_port=nccl_port)
    bootstrap.init_layer_runtime(model_config=model_config)
    return {}


_OFFLINE_PUBLISHED = False


def ensure_offline_runtime_published(**server_args_fields) -> bool:
    """Publish a minimal process-wide runtime config for the kernel-level collectors
    that build attention backends on a mock runner (no engine, no model).

    sglang>=0.5.20 projects ``get_exec()`` / ``get_parallel()`` from a published
    ServerArgs; the backends read them (flashattention_backend.py:320,349 get_exec().
    deterministic; parallel_state world_group). sglang's own unit tests publish an
    offline record exactly this way (``ServerArgs(model_path="dummy", **fields)`` +
    ``publish(role=..., ranks=SpawnRanks(world_rank=0))``, test/test_utils.py:1990-2023
    @0.5.21; the resolution pipeline returns early on the dummy/absent-model path).
    Older sglang has no ``publish`` -> no-op. Returns True when a publish happened.
    """
    global _OFFLINE_PUBLISHED
    if _OFFLINE_PUBLISHED:
        return True
    try:
        from sglang.srt.runtime_context import SpawnRanks, publish
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return False
    publish(ServerArgs(model_path="dummy", **server_args_fields), role="scheduler",
            ranks=SpawnRanks(world_rank=0))
    _OFFLINE_PUBLISHED = True
    return True


def attach_kv_index_translator(mock_runner) -> None:
    """sglang>=0.5.21 attention backends take ``model_runner.kv_index_translator``
    (flashattention_backend.py:205; built by ModelRunner.init_kv_index_translator at
    model_runner.py:860-868 from req_to_token / allocator / pool / page_size / device).
    Build the real one when the mock runner carries those objects; otherwise a
    non-translating stand-in (the backends only consult ``is_translating`` on the
    non-DCP, non-hybrid paths these kernel collectors exercise). No-op on older sglang.
    """
    try:
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    except ImportError:
        return
    try:
        mock_runner.kv_index_translator = KVIndexTranslator(
            req_to_token=mock_runner.req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=mock_runner.token_to_kv_pool_allocator,
            token_to_kv_pool=mock_runner.token_to_kv_pool,
            page_size=getattr(mock_runner, "page_size", None) or getattr(mock_runner.server_args, "page_size", 1) or 1,
            device=str(mock_runner.device),
        )
    except Exception as exc:  # noqa: BLE001 - stand-in is recorded, never silent
        from types import SimpleNamespace

        print(f"[sglang-compat] KVIndexTranslator not constructible on the mock runner ({type(exc).__name__}: {exc}); "
              "using a non-translating stand-in")
        mock_runner.kv_index_translator = SimpleNamespace(is_translating=False, reads_are_translated=False)
