# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from collector import case_generator

pytestmark = pytest.mark.unit


def test_pro_shared_expert_tp_shapes_are_exact_hits_without_duplicates(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    case_generator._load_model_cases_data.cache_clear()
    cases = case_generator.get_gemm_case_specs("vllm")
    keys = [(case.x, case.n, case.k) for case in cases]
    assert len(keys) == len(set(keys))
    shapes = set(keys)
    tokens = {case.x for case in cases}
    for tp in (1, 2, 4, 8):
        for m in tokens:
            assert (m, 2 * 3072 // tp, 7168) in shapes
            assert (m, 7168, 3072 // tp) in shapes
