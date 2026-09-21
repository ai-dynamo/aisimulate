# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the existing Dynamo scheduler after normal initialization.

Constructor and DP rank API audited at Dynamo commit
41882ae9b07232eed4850fb1daf8c958abb2556a, components/src/dynamo/vllm/instrumented_scheduler.py.
https://github.com/ai-dynamo/dynamo/blob/41882ae9b07232eed4850fb1daf8c958abb2556a/components/src/dynamo/vllm/instrumented_scheduler.py
"""

from fpm_memory_observer import observe

try:
    from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler
except ImportError as error:
    raise RuntimeError(
        "FPM memory observation requires an image with Dynamo's native InstrumentedScheduler; "
        "see the pinned API contract in this module"
    ) from error


class FpmResourceInstrumentedScheduler(InstrumentedScheduler):
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        observe("scheduler", self, dp_rank=self._fpm_dp_rank, cache_config=kv_cache_config)
