# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash prefix-cache KDA state checkpoint copy — stock vLLM 0.31.0.

With prefix caching on, vLLM switches the hybrid GLM-5.3-Flash to mamba
cache mode ``align``. Every step, the V2 model runner
(v1/worker/gpu/model_runner.py:1772-1782) calls
``MambaHybridModelState.preprocess_state``
(v1/worker/gpu/model_states/mamba_hybrid.py:179-226), which launches
``MambaSpecDecodeGPUContext.run_fused_precopy`` (v1/worker/mamba_utils.py:
1188-1232): one eager ``precopy_mamba_align_fused_kernel`` (:549-631) with
grid ``(num_reqs, num_states = 2 x 34, _TEMPORAL_TILES = 16)`` (:35). A
request copies its conv + recurrent state of every KDA layer from block
column ``src_col`` to ``dst_col`` only when it crossed a mamba block boundary
(``src_col != dst_col``); every other program exits early.

One row variant: ``vllm_precopy`` (op ``glm53_mamba_state_checkpoint_copy``,
kernel_source ``precopy_mamba_align_fused_kernel``, graph_mode ``eager``,
table ``glm53_mamba_state_copy_perf``). The collector builds the context
exactly as serving does (``MambaSpecDecodeGPUContext.create`` +
``initialize_from_forward_context``, mamba_hybrid.py:137-177) from a
synthetic KVCacheConfig and forward context, then times the stock
``run_fused_precopy`` launch with CUDA events. The first
``num_copy_requests`` rows copy (src_col 0 -> dst_col 1, token_bias 0: no
speculative decoding, so num_accepted - 1 = 0); the rest have
src_col == dst_col == 0 and early-exit. ``latency == gpu_time``.

Synthetic serving state (every field cites its population site at stock
vllm v0.31.0, tag commit db9527a4, no overlay):
- State shapes/dtypes from the GLM KDA layer's own calculators
  (models/glm5next/common/kda.py:158-180, moved from
  models/glm5next/nvidia/kda.py at 0.30.0 with identical content ->
  model_executor/layers/mamba/mamba_utils.py:303-326 kda_state_shape with
  num_spec 0, :133-149 kda_state_dtype with the bf16 model dtype and
  mamba_cache_dtype "auto": conv bf16 [3, conv_dim] (SD layout, the default
  of get_conv_state_layout :29-45), recurrent fp32 [H, 128, 128]).
- Per-layer state views: one int8 page per block, sliced into the conv and
  recurrent views by the stock ``MambaBase.bind_kv_cache``
  (model_executor/layers/mamba/abstract.py:28-42). Pages are unpadded
  (serving pads a mamba page up to the attention page,
  ``page_size_padded``, which only widens the gap between blocks; the
  bytes copied per block are the natural state sizes,
  v1/worker/mamba_utils.py:1034-1046,322,357).
- KV-cache groups: the 34 KDA layers split like
  v1/core/kv_cache_utils.py:1558-1592 (group size = the 11 attention
  layers, ``layers[i::4]``), i.e. 4 mamba groups with their own int32 block
  tables (v1/worker/gpu/block_table.py:58-80); README section 1 records the
  4 mamba groups of the GLM serving trace (0.30.0; the grouping functions
  get_kv_cache_groups / _get_kv_cache_groups_uniform_page_size are
  unchanged at 0.31.0 apart from docstrings and MTP-only eagle annotation).
- Copy functions: ``MambaStateCopyFuncCalculator.kda_state_copy_func()``
  (mamba_utils.py:446-448) keyed by the layer's mamba_type GDN_ATTN
  (model_executor/layers/mamba/gdn/base.py:49-51), as
  ``Glm5NextForCausalLM.get_mamba_state_copy_func`` +
  ``IsHybrid.get_mamba_state_copy_funcs`` return them
  (models/glm5next/common/model.py:1005-1012,
  model_executor/models/interfaces.py:1141-1147).
- ``idx_mapping``: batch-ordered int32 request-state indices
  (v1/worker/gpu/model_runner.py:1336-1338; int64 at 0.30.0, int32 since
  0.31.0 in both the real and the dummy batch, input_batch.py:140).

0.31.0 audit (2026-10-07, source-only: tag v0.31.0 == db9527a4 vs v0.30.0
ced6857a): every cited block above is byte-identical to its 0.30.0
counterpart (lines shifted; v1/worker/mamba_utils.py differs only in
docstrings), except block_table.py (num_blocks formatting only) and the
idx_mapping dtype (int64 -> int32, followed here). The earlier 0.30.0
audit targeted the 0.30.0+glm53tail.eb4704514fdf overlay, which did not
touch this op's path.

Timing method: see collector/glm53_mamba_state_copy_common.py.
"""

# Audited serving source: stock vllm v0.31.0 (tag commit db9527a4). Any other
# release raises MambaStateCopyRuntimeNotAuditedError inside the run function.
__compat__ = "vllm==0.31.0"

import gc
import os

try:
    from collector.case_generator import get_common_glm53_mamba_state_copy_test_cases
    from collector.glm53_mamba_state_copy_common import (
        L2Flusher,
        build_row,
        check_local_geometry,
        median_device_ms,
        persist_row,
        require_audited_runtime,
        validate_case,
    )
    from collector.helper import log_perf
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_glm53_mamba_state_copy_test_cases
    from glm53_mamba_state_copy_common import (
        L2Flusher,
        build_row,
        check_local_geometry,
        median_device_ms,
        persist_row,
        require_audited_runtime,
        validate_case,
    )

    from helper import log_perf

VLLM_VARIANTS = ("vllm_precopy",)

# v1/core/kv_cache_utils.py:1558-1592 @v0.31.0: hybrid layers are split into groups of
# min-layer-count (the 11 GLM attention layers) layers; the 34 KDA layers
# therefore form ceil(34 / 11) = 4 groups with layers[i::4].
NUM_MAMBA_GROUPS = 4
# MambaSpec.block_size steers WHEN serving crosses a block (expressed here
# directly through src_col/dst_col); run_fused_precopy and the precopy
# kernel never read it (mamba_utils.py:549-631,1188-1232). Recorded only to
# build a valid spec.
_UNUSED_MAMBA_BLOCK_SIZE = 1
_MODEL_DTYPE_NAME = "bfloat16"  # GLM-5.3-Flash config.json "dtype"


def get_glm53_mamba_state_copy_test_cases():
    """[variant, tp_size, num_layers, num_heads, head_dim, conv_kernel_size,
    batch_size, num_copy_requests, model_name] per vLLM variant."""
    common = get_common_glm53_mamba_state_copy_test_cases()
    return [
        [
            variant,
            case.tp_size,
            case.num_layers,
            case.num_heads,
            case.head_dim,
            case.conv_kernel_size,
            case.batch_size,
            case.num_copy_requests,
            case.model_name,
        ]
        for variant in VLLM_VARIANTS
        for case in common
    ]


def mamba_group_layer_names(num_layers: int, num_groups: int = NUM_MAMBA_GROUPS) -> list[list[str]]:
    """KDA layer names per mamba group, split like kv_cache_utils.py:1585-1592."""
    names = [f"language_model.model.layers.{index}.linear_attn" for index in range(num_layers)]
    return [names[group::num_groups] for group in range(num_groups)]


class _KdaStateLayer:
    """The GLM KDA layer surface the align context reads: its state
    shape/dtype hooks (models/glm5next/common/kda.py:158-180) and the
    ``kv_cache`` tuple bound by MambaBase.bind_kv_cache."""

    def __init__(self, shapes, dtypes):
        self._shapes = shapes
        self._dtypes = dtypes
        self.kv_cache = ()

    def get_state_shape(self):
        return self._shapes

    def get_state_dtype(self):
        return self._dtypes


def _build_context(torch, device, tp_size, num_layers, num_heads, head_dim, conv_kernel_size, batch_size):
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncCalculator,
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.utils.torch_utils import get_dtype_size
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    shapes = MambaStateShapeCalculator.kda_state_shape(
        tp_size,
        num_heads,
        head_dim,
        conv_kernel_size=conv_kernel_size,
        num_spec=0,
    )
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(getattr(torch, _MODEL_DTYPE_NAME), "auto")
    # MambaBase.get_kv_cache_spec (abstract.py:64-81) in align mode, no
    # speculative blocks.
    spec = MambaSpec(
        block_size=_UNUSED_MAMBA_BLOCK_SIZE,
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    groups = mamba_group_layer_names(num_layers)
    # Block 0 is vLLM's null block; request r owns blocks 1 + 2r (column 0)
    # and 2 + 2r (column 1) in every group.
    num_blocks = 2 * batch_size + 1
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec) for names in groups],
    )
    page_bytes = sum(
        int(torch.Size(shape).numel()) * get_dtype_size(dtype) for shape, dtype in zip(shapes, dtypes, strict=True)
    )
    forward_context = {}
    for names in groups:
        for name in names:
            layer = _KdaStateLayer(tuple(shapes), tuple(dtypes))
            pages = torch.zeros((num_blocks, 1, 1, page_bytes), dtype=torch.int8, device=device)
            MambaBase.bind_kv_cache(layer, pages)
            forward_context[name] = layer
    block_table = torch.zeros((batch_size, 2), dtype=torch.int32, device=device)
    block_table[:, 0] = torch.arange(1, 2 * batch_size, 2, dtype=torch.int32, device=device)
    block_table[:, 1] = torch.arange(2, 2 * batch_size + 1, 2, dtype=torch.int32, device=device)
    block_tables = [block_table.clone() for _ in groups]

    copy_funcs = {MambaAttentionBackendEnum.GDN_ATTN: MambaStateCopyFuncCalculator.kda_state_copy_func()}
    # mamba_hybrid.py:156-177
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=batch_size,
        kv_cache_config=kv_cache_config,
        copy_funcs=copy_funcs,
        device=device,
        make_buffer=lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=device),
    )
    ctx.initialize_from_forward_context(kv_cache_config, forward_context, copy_funcs, block_tables)
    if ctx.num_states != 2 * num_layers:
        raise RuntimeError(f"vLLM align context holds {ctx.num_states} states, expected {2 * num_layers}")
    first = next(iter(forward_context.values()))
    conv_state, ssm_state = first.kv_cache[:2]
    return ctx, forward_context, block_tables, conv_state, ssm_state


def _time_precopy(torch, device, ctx, batch_size, num_copy_requests, flusher):
    # Per-request-slot GPU state of mamba_hybrid.py:96-106 after
    # preprocess_mamba_align_fused_kernel: int32 dst column (state idx), src
    # column and token bias (src_off).
    dst_col = torch.zeros(batch_size, dtype=torch.int32, device=device)
    dst_col[:num_copy_requests] = 1
    src_col = torch.zeros(batch_size, dtype=torch.int32, device=device)
    token_bias = torch.zeros(batch_size, dtype=torch.int32, device=device)
    # Serving uploads idx_mapping as int32 (model_runner.py:1336-1338 @v0.31.0).
    idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device)

    def launch_precopy():
        # mamba_hybrid.py:220-226
        ctx.run_fused_precopy(batch_size, dst_col, src_col, token_bias, idx_mapping)

    gpu_ms = median_device_ms(torch, launch_precopy, flusher)
    return gpu_ms, gpu_ms


def run_glm53_mamba_state_copy(
    variant: str,
    tp_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    batch_size: int,
    num_copy_requests: int,
    model_name: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    from vllm.version import __version__ as vllm_version

    require_audited_runtime("vllm", vllm_version)
    validate_case("vllm", variant, tp_size, num_heads, batch_size, num_copy_requests)

    import torch

    torch.cuda.set_device(device)
    ctx = forward_context = block_tables = conv_state = ssm_state = flusher = None
    try:
        with torch.inference_mode():
            ctx, forward_context, block_tables, conv_state, ssm_state = _build_context(
                torch, device, tp_size, num_layers, num_heads, head_dim, conv_kernel_size, batch_size
            )
            conv_width, conv_dim = int(conv_state.shape[-2]), int(conv_state.shape[-1])
            local_heads, local_head_dim = int(ssm_state.shape[-3]), int(ssm_state.shape[-2])
            check_local_geometry(
                num_heads=num_heads,
                tp_size=tp_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                local_heads=local_heads,
                local_head_dim=local_head_dim,
                conv_width=conv_width,
                conv_dim=conv_dim,
            )
            flusher = L2Flusher(torch, device)
            latency_ms, gpu_ms = _time_precopy(torch, device, ctx, batch_size, num_copy_requests, flusher)
        row = build_row(
            variant=variant,
            tp_size=tp_size,
            num_layers=num_layers,
            num_heads=local_heads,
            head_dim=local_head_dim,
            conv_dim=conv_dim,
            conv_width=conv_width,
            ssm_bytes_per_req_layer=int(ssm_state[0].numel() * ssm_state.element_size()),
            conv_bytes_per_req_layer=int(conv_state[0].numel() * conv_state.element_size()),
            batch_size=batch_size,
            num_copy_requests=num_copy_requests,
            latency=latency_ms,
            gpu_time=gpu_ms,
        )
        persist_row(
            row,
            framework_label="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            perf_filename=perf_filename,
            log_perf=log_perf,
        )
    finally:
        del ctx, forward_context, block_tables, conv_state, ssm_state, flusher
        gc.collect()
        torch.cuda.empty_cache()
