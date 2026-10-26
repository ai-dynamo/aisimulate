# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain the qualified 0.24/0.25 sparse MLA route."""

__compat__ = "vllm>=0.24.0,<=0.25.1"

from collector.vllm import collect_mla_module


def get_dsa_context_module_test_cases():
    return collect_mla_module.get_dsa_context_module_test_cases()


def get_dsa_generation_module_test_cases():
    return collect_mla_module.get_dsa_generation_module_test_cases()


def run_mla_module_worker(*args, **kwargs):
    return collect_mla_module.run_mla_module_worker(*args, **kwargs)
