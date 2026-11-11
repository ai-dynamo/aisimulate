# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the existing Dynamo scheduler after normal initialization.

Constructor and DP rank API audited at Dynamo commit
41882ae9b07232eed4850fb1daf8c958abb2556a, components/src/dynamo/vllm/instrumented_scheduler.py.
https://github.com/ai-dynamo/dynamo/blob/41882ae9b07232eed4850fb1daf8c958abb2556a/components/src/dynamo/vllm/instrumented_scheduler.py
The same API was audited at Dynamo 1.7.0-rc.1, commit 4279877c5334e8de30f1cb64bd3a0b34c83891fc,
and on release/1.7.0, commit 9abd2ef0d908c4bb323e1b8956ea76a5772a1d41, where the file is identical:
__init__(vllm_config, kv_cache_config, ...) at lines 2220-2236, self._fpm_dp_rank at line 2243.
"""

from fpm_memory_observer import observe, observe_cpu

try:
    from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler
except ImportError as error:
    raise RuntimeError(
        "FPM memory observation requires an image with Dynamo's native InstrumentedScheduler; "
        "see the pinned API contract in this module"
    ) from error


class FpmExecutionInstrumentedScheduler(InstrumentedScheduler):
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        observe_cpu("scheduler", dp_rank=self._fpm_dp_rank)


class FpmResourceInstrumentedScheduler(FpmExecutionInstrumentedScheduler):
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        observe("scheduler", self, dp_rank=self._fpm_dp_rank, cache_config=kv_cache_config)
