# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31.0 unpadded MLA pre-BMM: token-major output storage.

Serving contract: vllm-project/vllm at db9527a46873454610df6dbedf79a36d6bf1a7f6,
vllm/model_executor/layers/attention/mla_attention.py:1080-1099.
The CUTLASS_MLA padded-head branch is outside this collector's scope.
"""

__compat__ = "vllm==0.31.0"

from collector.vllm import collect_mla_bmm_v1


def get_mla_gen_pre_test_cases():
    return collect_mla_bmm_v1.get_mla_gen_pre_test_cases()


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    return collect_mla_bmm_v1.run_mla_gen_pre(
        num_tokens,
        num_heads,
        dtype,
        num_warmups,
        num_runs,
        perf_filename=perf_filename,
        device=device,
        _token_major_output=True,
    )
