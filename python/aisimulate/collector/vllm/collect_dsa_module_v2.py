# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31 sparse MLA route, using the shared native cache adapter."""

__compat__ = "vllm==0.31.0"

from collector.vllm import collect_mla_module, collect_mla_module_v2


def get_dsa_context_module_test_cases():
    return collect_mla_module.get_dsa_context_module_test_cases()


def get_dsa_generation_module_test_cases():
    return collect_mla_module.get_dsa_generation_module_test_cases()


def run_mla_module_worker(*args, **kwargs):
    return collect_mla_module_v2.run_mla_module_worker(*args, **kwargs)
