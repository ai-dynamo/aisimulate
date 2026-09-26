# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin shells for exact, latency-only measured prefill composites.

The native engine owns profile admission, complete table validation and all
latency arithmetic. Raw views expose the measured scope without adding energy.
"""

from typing import ClassVar

import aisimulate_core._native as _core
from aisimulate_core.sdk.operations.base import OpShellKit


class SglangPrefillAttentionSequence(_core.SglangPrefillAttentionSequence, OpShellKit):
    _data_cache: ClassVar[dict] = {}

    @classmethod
    def load_data(cls, database):
        from aisimulate_core.sdk.common import PerfDataFilename
        from aisimulate_core.sdk.engine_table_view import load_view

        key = (database.systems_root, database.system, database.backend, database.version)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_sglang_prefill_attention_sequence_data", PerfDataFilename.sglang_prefill_attention_sequence
            )
            cls._record_load()
        database._sglang_prefill_attention_sequence_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls):
        cls._data_cache.clear()


class SglangPrefillCommNormBoundary(_core.SglangPrefillCommNormBoundary, OpShellKit):
    _data_cache: ClassVar[dict] = {}

    @classmethod
    def load_data(cls, database):
        from aisimulate_core.sdk.common import PerfDataFilename
        from aisimulate_core.sdk.engine_table_view import load_view

        key = (database.systems_root, database.system, database.backend, database.version)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_sglang_prefill_comm_norm_boundary_data", PerfDataFilename.sglang_prefill_comm_norm_boundary
            )
            cls._record_load()
        database._sglang_prefill_comm_norm_boundary_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls):
        cls._data_cache.clear()
