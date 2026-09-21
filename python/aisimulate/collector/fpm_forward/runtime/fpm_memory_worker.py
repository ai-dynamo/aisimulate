# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only worker hooks; the audited API contract is in fpm_memory_observer."""

import logging

from fpm_memory_observer import compilation_config, observe

try:
    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.v1.worker.gpu_worker import Worker
except ImportError as error:
    raise RuntimeError(
        "FPM memory observation requires a compatible vLLM V1 GPU-worker image; the audited runtime is vLLM 0.27.0"
    ) from error


class FpmResourceWorker(Worker):
    def determine_available_memory(self):
        # vLLM 0.27.0 profiles a minimal cache and resolves graph mode here,
        # before initialize_from_config. Retain the first snapshot on repeats.
        if not hasattr(self, "_fpm_initial_compilation_config"):
            try:
                self._fpm_initial_compilation_config = compilation_config(self.vllm_config)
            except (AttributeError, TypeError) as error:
                self._fpm_initial_compilation_config = None
                logging.getLogger(__name__).warning("FPM initial graph configuration is unavailable: %s", error)
        result = super().determine_available_memory()
        self._fpm_available_cache_bytes = result
        return result

    def initialize_from_config(self, kv_cache_config):
        result = super().initialize_from_config(kv_cache_config)
        self._fpm_cache_initialized = True
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        parallel = self.vllm_config.parallel_config
        dp_rank = getattr(parallel, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = parallel.data_parallel_rank
        observe(
            "worker", self, dp_rank=dp_rank, tp_rank=get_tp_group().rank_in_group, pp_rank=get_pp_group().rank_in_group
        )
        return result
