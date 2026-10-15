# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle hooks shared by the two source-audited adapters."""

import logging

from fpm_memory_observer import compilation_config
from vllm.distributed import get_pp_group, get_tp_group

from .observer import observe


class WorkerObservation:
    def determine_available_memory(self):
        if not hasattr(self, "_fpm_initial_compilation_config"):
            try:
                self._fpm_initial_compilation_config = compilation_config(self.vllm_config)
            except (AttributeError, TypeError) as error:
                logging.getLogger(__name__).warning("Initial graph configuration unavailable: %s", error)
        result = super().determine_available_memory()
        self._fpm_available_cache_bytes = result
        return result

    def initialize_from_config(self, kv_cache_config):
        result = super().initialize_from_config(kv_cache_config)
        self._fpm_cache_initialized = True
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._observation_warmup_completed = True
        parallel = self.vllm_config.parallel_config
        dp_rank = getattr(parallel, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = parallel.data_parallel_rank
        observe(
            "worker",
            self,
            version=self.observation_version,
            dp_rank=dp_rank,
            tp_rank=get_tp_group().rank_in_group,
            pp_rank=get_pp_group().rank_in_group,
        )
        return result


class SchedulerObservation:
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        observe(
            "scheduler", self, version=self.observation_version, dp_rank=self._fpm_dp_rank, cache_config=kv_cache_config
        )
