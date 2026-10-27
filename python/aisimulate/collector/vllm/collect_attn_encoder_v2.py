# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact vLLM 0.31 route for BF16 encoder attention through native wrappers.

API and input contracts rechecked at db9527a46873454610df6dbedf79a36d6bf1a7f6:
vllm/model_executor/layers/attention/mm_encoder_attention.py:352-367,568-590;
vllm/v1/attention/ops/vit_attn_wrappers.py:27-124;
vllm/platforms/cuda.py:539-595. Selector and wrapper signatures are unchanged.
The native wrapper's CUDA-graph output initialization is part of measured work;
this route requires fresh data, not reuse of the old wrapper's measurements.
"""

__compat__ = "vllm==0.31.0"

from collector.vllm import collect_attn_encoder_v1


def get_encoder_attention_test_cases():
    return collect_attn_encoder_v1.get_encoder_attention_test_cases()


def run_encoder_attention_torch(batch_size, seq_len, num_heads, head_dim, *, perf_filename, device="cuda:0"):
    return collect_attn_encoder_v1.run_encoder_attention_torch(
        batch_size, seq_len, num_heads, head_dim, perf_filename=perf_filename, device=device
    )
