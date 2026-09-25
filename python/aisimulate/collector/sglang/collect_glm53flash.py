# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned native GLM sglang operation collection route."""

from collector.collect_glm53flash import get_test_cases, run_native

__compat__ = "sglang==0.5.20"


def get_glm53flash_test_cases():
    return get_test_cases("sglang")


def run_glm53flash_worker(*args, **kwargs):
    return run_native(*args, **kwargs)
