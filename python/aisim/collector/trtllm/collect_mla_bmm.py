# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM MLA generation BMM micro-collector.

Times the two absorption BMMs of serving's MLA.forward_absorption_generation
(_torch/attention/mla.py:1594-1818): q_nope x W_UK into the latent part of fused_q
(:1709-1745) and the attention output x W_UV (:1796-1814). Each call is the one
serving makes for the weight dtype, with serving's operand layouts: q_nope is the
nope slice of the q_b_proj output (q.view(T, H, 192).split, :1606-1608), the pre-BMM
writes into fused_q[..., :kv_lora_rank] of a [T, H, kv_lora_rank + rope] buffer
(:1649-1656, :1713, :1734), and the post-BMM reads the [T, H * kv_lora_rank]
attention output and writes the [T, H * v_head_dim] output (:1792-1794).

- bf16 weights: MLA._bmm_bf16_out (:1564-1592), torch.ops.trtllm.bmm_out. On
  SM120/121 it first makes the head-major LHS contiguous (:1584-1591), so that copy
  is timed there. The CuTe DSL bf16 BMM is off unless use_cute_dsl_bf16_bmm, which
  serving enables only with pipeline parallelism (llmapi/llm_args.py:7043-7049,
  7996-8003).
- fp8 (fp8_block checkpoint) weights: fp8_block_scaling_bmm_out (:195-227). SM89/90
  quantize the activation with fp8_batched_quantize_1x128_permute102, SM120 with
  fp8_utils.per_token_quant_and_transform(need_permute102=True); both run
  torch.ops.trtllm.fp8_block_scaling_bmm_out into a fresh buffer and copy it into the
  output view, and that copy is timed. SM120 weights are resmoothed to e8m0 scales at
  load (MLA.transform_weights, :2190-2213). The SM100 family (is_sm_100f,
  _utils.py:742-745) runs torch.bmm on weights dequantized to bf16 at load
  (:219-225; MLA.create_weights, :781-796), unless use_cute_dsl_blockscaling_bmm,
  which defaults to off (llm_args.py:7032-7035). Every other SM raises, as serving's
  wrapper does (:226-227).

kernel_source names the op that ran.
"""

__compat__ = "trtllm==1.4.0rc2"

import tensorrt_llm
import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases

# The DeepSeek-V3 MLA geometry every planned case has (op_plans._MLA_GEOMETRY).
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_KV_LORA_RANK = 512
_V_HEAD_DIM = 128


def _fp8_bmm_path() -> str:
    """The op serving's fp8_block_scaling_bmm_out runs on this SM (mla.py:195-227)."""
    sm = get_sm_version()
    if sm in (89, 90, 120):
        return "fp8_block_scaling_bmm_out"
    if 100 <= sm < 110:
        return "torch_bmm_dequant_bf16"
    raise RuntimeError(
        f"serving's fp8_block_scaling_bmm_out raises NotImplementedError on SM{sm} "
        "(_torch/attention/mla.py:226-227); no serving-true fp8 mla_bmm measurement exists on this platform"
    )


def _dequant_fp8_weight(weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 128x128-block-scaled fp8 weight to bf16, the load-time weight
    serving keeps for the SM100 family (k_b_proj_trans_dequant / v_b_proj_dequant)."""
    scale = weight_scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    # multiply in float32 (bf16 * f32 type-promotes to f32 anyway), then cast
    # back: torch.bmm requires out dtype == input dtype, and the serving path
    # stores the dequantized weight as bf16.
    dequant = weight_fp8.to(torch.float32) * scale[:, : weight_fp8.shape[1], : weight_fp8.shape[2]]
    return dequant.to(torch.bfloat16).contiguous()


def _prep_fp8_weight(weight_fp8: torch.Tensor, weight_scale: torch.Tensor):
    """Load-time weight-scale prep, mirroring serving per SM.

    SM120 serving resmooths block scales to e8m0 and transforms the layout at
    weight-load time (MLA.transform_weights -> resmooth_parameters,
    mla.py:2173-2213); SM89/90 consume the raw float32 128x128 block scales.
    """
    if get_sm_version() == 120:
        weight_fp8, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(weight_fp8, weight_scale)
        weight_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[1],
            k=weight_fp8.shape[2],
            recipe=(1, 128, 128),
            num_groups=weight_fp8.shape[0],
            is_sfa=False,
        )
    return weight_fp8, weight_scale


def _quantize_fp8_activation(x: torch.Tensor):
    """Per-forward activation quant of fp8_block_scaling_bmm_out (mla.py:204-213)."""
    if get_sm_version() == 120:
        return fp8_utils.per_token_quant_and_transform(x, need_permute102=True)
    return torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(x)


def _fp8_block_scaling_bmm_out(mat1, mat2_fp8, mat2_scale, out) -> None:
    """Serving's SM89/90/120 branch of fp8_block_scaling_bmm_out (mla.py:204-218):
    quantize, BMM into a fresh buffer, copy into the output view."""
    mat1_fp8, mat1_scale = _quantize_fp8_activation(mat1)
    output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
    torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output)
    out.copy_(output)


def _bmm_bf16_out(a, b_transposed, output) -> None:
    """Serving's default branch of MLA._bmm_bf16_out (mla.py:1583-1592)."""
    if get_sm_version() in (120, 121):
        a = a.contiguous()
    torch.ops.trtllm.bmm_out(a, b_transposed, output)


def _random_fp8_weight(shape, device):
    """An fp8 weight and its 128x128 block scales. The scales are positive: SM120's
    load-time resmooth to e8m0 works in the log2 domain and would NaN on negative ones."""
    weight = torch.randn(shape, dtype=torch.bfloat16, device=device).to(dtype=torch.float8_e4m3fn)
    scale = torch.rand([shape[0], shape[1] // 128, shape[2] // 128], dtype=torch.float32, device=device) + 0.5
    return weight, scale


def _gen_pre_operands(num_tokens, num_heads, device):
    """q_nope as the nope slice of the q_b_proj output and the fused_q latent view
    the BMM writes (mla.py:1606-1608, 1649-1656, 1713)."""
    qk_head_dim = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
    q = torch.randn([num_tokens, num_heads * qk_head_dim], dtype=torch.bfloat16, device=device)
    q_nope, _q_pe = q.view([-1, num_heads, qk_head_dim]).split([_QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM], dim=-1)
    fused_q = torch.empty(
        [num_tokens, num_heads, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM], dtype=torch.bfloat16, device=device
    )
    q_nope_out = fused_q[..., :_KV_LORA_RANK].transpose(0, 1)
    return q, fused_q, q_nope, q_nope_out


def _gen_post_operands(num_tokens, num_heads, device):
    """The attention output read as [T, H, kv_lora_rank] and the [T, H, v_head_dim]
    output the BMM writes (mla.py:1792-1794)."""
    attn_out_latent = torch.randn([num_tokens, num_heads * _KV_LORA_RANK], dtype=torch.bfloat16, device=device).view(
        [-1, num_heads, _KV_LORA_RANK]
    )
    output = torch.empty([num_tokens, num_heads * _V_HEAD_DIM], dtype=torch.bfloat16, device=device)
    attn_output = output.view([num_tokens, num_heads, _V_HEAD_DIM])
    return attn_out_latent, attn_output


def get_mla_gen_pre_test_cases():
    return plan_cases("mla_bmm_gen_pre", "trtllm", get_sm_version())


def get_mla_gen_post_test_cases():
    return plan_cases("mla_bmm_gen_post", "trtllm", get_sm_version())


def _time_and_log(
    kernel_func, op_name, kernel_source, num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device
):
    kernel_func()  # dry run
    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    """q_nope x W_UK into fused_q (mla.py:1709-1745); num_heads is already split by TP."""
    torch.cuda.set_device(device)

    if dtype == "bfloat16":
        # [num_heads, kv_lora_rank, qk_nope_head_dim], read through its transpose.
        k_b_proj_trans = torch.randn([num_heads, _KV_LORA_RANK, _QK_NOPE_HEAD_DIM], dtype=torch.bfloat16, device=device)
        _q, _fused_q, q_nope, q_nope_out = _gen_pre_operands(num_tokens, num_heads, device)
        kernel_source = "trtllm_bmm_out"

        def kernel_func():
            _bmm_bf16_out(q_nope.transpose(0, 1), k_b_proj_trans.transpose(1, 2), q_nope_out)

    elif dtype == "fp8":
        path = _fp8_bmm_path()
        k_b_proj_raw, k_b_proj_scale_raw = _random_fp8_weight([num_heads, _KV_LORA_RANK, _QK_NOPE_HEAD_DIM], device)
        _q, _fused_q, q_nope, q_nope_out = _gen_pre_operands(num_tokens, num_heads, device)
        if path == "fp8_block_scaling_bmm_out":
            k_b_proj_trans, k_b_proj_trans_scale = _prep_fp8_weight(k_b_proj_raw, k_b_proj_scale_raw)
            kernel_source = "trtllm_fp8_block_scaling_bmm_out"

            def kernel_func():
                _fp8_block_scaling_bmm_out(q_nope, k_b_proj_trans, k_b_proj_trans_scale, q_nope_out)

        else:
            k_b_proj_trans_dequant = _dequant_fp8_weight(k_b_proj_raw, k_b_proj_scale_raw)
            kernel_source = "torch_bmm_dequant_bf16"

            def kernel_func():
                torch.bmm(q_nope.transpose(0, 1), k_b_proj_trans_dequant.transpose(1, 2), out=q_nope_out)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    _time_and_log(
        kernel_func,
        "mla_gen_pre",
        kernel_source,
        num_tokens,
        num_heads,
        dtype,
        num_warmups,
        num_runs,
        perf_filename=perf_filename,
        device=device,
    )


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    """Attention output x W_UV (mla.py:1796-1814); num_heads is already split by TP."""
    torch.cuda.set_device(device)

    if dtype == "bfloat16":
        # [num_heads, v_head_dim, kv_lora_rank], read through its transpose.
        v_b_proj = torch.randn([num_heads, _V_HEAD_DIM, _KV_LORA_RANK], dtype=torch.bfloat16, device=device)
        attn_out_latent, attn_output = _gen_post_operands(num_tokens, num_heads, device)
        kernel_source = "trtllm_bmm_out"

        def kernel_func():
            _bmm_bf16_out(attn_out_latent.transpose(0, 1), v_b_proj.transpose(1, 2), attn_output.transpose(0, 1))

    elif dtype == "fp8":
        path = _fp8_bmm_path()
        v_b_proj_raw, v_b_proj_scale_raw = _random_fp8_weight([num_heads, _V_HEAD_DIM, _KV_LORA_RANK], device)
        attn_out_latent, attn_output = _gen_post_operands(num_tokens, num_heads, device)
        if path == "fp8_block_scaling_bmm_out":
            v_b_proj, v_b_proj_scale = _prep_fp8_weight(v_b_proj_raw, v_b_proj_scale_raw)
            kernel_source = "trtllm_fp8_block_scaling_bmm_out"

            def kernel_func():
                _fp8_block_scaling_bmm_out(attn_out_latent, v_b_proj, v_b_proj_scale, attn_output.transpose(0, 1))

        else:
            v_b_proj_dequant = _dequant_fp8_weight(v_b_proj_raw, v_b_proj_scale_raw)
            kernel_source = "torch_bmm_dequant_bf16"

            def kernel_func():
                torch.bmm(
                    attn_out_latent.transpose(0, 1), v_b_proj_dequant.transpose(1, 2), out=attn_output.transpose(0, 1)
                )

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    _time_and_log(
        kernel_func,
        "mla_gen_post",
        kernel_source,
        num_tokens,
        num_heads,
        dtype,
        num_warmups,
        num_runs,
        perf_filename=perf_filename,
        device=device,
    )


if __name__ == "__main__":
    from registry_types import PerfFile

    test_cases = get_mla_gen_pre_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_pre(*test_case, perf_filename=PerfFile.MLA_BMM)
    test_cases = get_mla_gen_post_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_post(*test_case, perf_filename=PerfFile.MLA_BMM)
