# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Guard the singleton FP8 descale ABI without changing values or storage."""
import pytest
import torch


@pytest.mark.parametrize('batch,heads', [(1,1), (2,1), (2,4)])
def test_descale_view_preserves_storage_and_values(batch, heads):
    scale = torch.tensor(0.75)
    old = scale.expand(batch, heads)
    fixed = scale.view(-1).expand(batch, heads)
    assert fixed.data_ptr() == scale.data_ptr()
    torch.testing.assert_close(fixed, old, rtol=0, atol=0)
    if heads == 1:
        assert fixed.stride(-1) == 1
    if batch == heads == 1:
        assert old.contiguous().stride(-1) == 0


@pytest.mark.parametrize('batch', [1,2])
def test_real_fa4_descale_matches_dense_storage(batch):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import to_cute_tensor

    assert torch.cuda.get_device_capability()[0] == 10
    torch.manual_seed(42)
    tokens, heads, dim, page = 4, 16, 128, 128
    q = torch.randn(batch*tokens, heads, dim, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(batch, page, 1, dim, device='cuda').to(torch.float8_e4m3fn)
    v = torch.randn_like(k, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = [torch.tensor(value, device='cuda') for value in (0.5, 0.75, 1.25)]
    if batch == 1:
        with pytest.raises(RuntimeError, match='Expected strides'):
            to_cute_tensor(scales[0].expand(1,1), assumed_align=4, leading_dim=1)
        to_cute_tensor(scales[0].view(-1).expand(1,1), assumed_align=4, leading_dim=1)
    cu = torch.arange(batch+1, dtype=torch.int32, device='cuda')*tokens
    used = torch.full((batch,), tokens, dtype=torch.int32, device='cuda')
    blocks = torch.arange(batch, dtype=torch.int32, device='cuda').view(batch,1)
    def run(descales):
        out = torch.empty(q.shape, dtype=torch.bfloat16, device='cuda')
        flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
            max_seqlen_q=tokens, seqused_k=used, max_seqlen_k=tokens,
            block_table=blocks, causal=True, fa_version=4,
            q_descale=descales[0], k_descale=descales[1], v_descale=descales[2])
        return out
    fixed = run([scale.view(-1).expand(batch,1) for scale in scales])
    dense = run([scale.expand(batch,1).clone(memory_format=torch.contiguous_format) for scale in scales])
    torch.testing.assert_close(fixed, dense, rtol=0, atol=0)
