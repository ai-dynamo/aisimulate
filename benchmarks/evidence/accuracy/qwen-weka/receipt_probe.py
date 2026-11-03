# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extend the retained geometry probe with the effective configuration, privately."""

from cache_probe import AlignmentCacheProbe, serializable


class CampaignCacheProbe(AlignmentCacheProbe):
    def alignment_cache_geometry(self):
        result = super().alignment_cache_geometry()
        result["resolved_vllm_config"] = serializable(self.vllm_config)
        return result
