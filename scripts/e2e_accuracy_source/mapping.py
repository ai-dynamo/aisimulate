# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

HARDWARE_TO_SYSTEM: dict[str, str] = {
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "b200": "b200_sxm",
    "b300": "b300_sxm",
    "gb200": "gb200",
    "gb300": "gb300",
}


MOE_MODELS: frozenset[str] = frozenset(
    {
        "minimaxm2.5",
        "minimaxm2.7",
        "dsr1",
        "kimik2.5",
        "kimik2.6",
        "kimik3",
        "qwen3.5",
        "gptoss120b",
        "dsv4",
        "glm5",
        "glm5.1",
        "glm5.2",
        "minimaxm3",
    }
)


@dataclass(frozen=True)
class _WorkerShape:
    """Resolved per-worker parallelism for one worker group."""

    tp_size: int
    pp_size: int
    attention_dp_size: int
    moe_ep_size: int | None
    moe_tp_size: int | None


def _resolve_worker_shape(
    *,
    num_gpu: int,
    num_workers: int,
    tp: int,
    ep: int,
    dp_attention: bool,
    is_moe: bool,
    backend: str,
    single_node_vllm: bool = False,
) -> _WorkerShape:
    """Apply the unified per-worker rule, with backend-aware MoE shape.

    Per docs → "Unified rule (works for agg and disagg, dense and MoE)" plus
    backend-specific branches:

    - **single-node aggregated vLLM**: InferenceX uses the declared `tp`
      as the whole process-group width. With DP attention disabled it launches
      TP=`tp`, DP=1; with DP attention enabled it launches TP=1, DP=`tp`.
      Expert parallelism reuses that same world and is not another
      multiplicative GPU axis. The exported `num_gpu` field may still contain
      a legacy `tp * ep` product, so this branch does not derive its world size
      from that field.
    - **other vLLM**: keep the per-worker GPU arithmetic used by disaggregated
      deployments. AISim's vLLM backend asserts `moe_tp_size == 1 OR
      moe_ep_size == 1`, so wide EP is represented with `moe_tp=1`.
    - **sglang / trtllm**: AISim's non-wideep backends assert
      `attention_tp_size == 1 OR attention_dp_size == 1`. We branch on
      silicon's `dp_attention` flag:
        * `dp_attention=true`  → `tp_size=1`,  `attention_dp_size=per_worker/pp`
        * `dp_attention=false` → `tp_size=per_worker/pp`, `attention_dp_size=1`
      MoE keeps silicon's literal `ep`; `moe_tp = per_worker/(pp*ep)`.

    Raises ValueError on any invariant break.
    """
    workers = max(1, num_workers)  # agg has prefill_num_workers == 0; treat as 1
    if num_gpu <= 0 or workers <= 0:
        raise ValueError(f"non-positive GPU/worker count: num_gpu={num_gpu}, workers={workers}")

    if backend == "vllm" and single_node_vllm:
        if tp <= 0 or ep <= 0:
            raise ValueError(f"non-positive vLLM parallelism: tp={tp}, ep={ep}")
        if ep > 1 and ep != tp:
            raise ValueError(f"vLLM effective topology is ambiguous without server logs: declared tp={tp}, ep={ep}")

        tp_size = 1 if dp_attention else tp
        adp = tp if dp_attention else 1
        if not is_moe:
            return _WorkerShape(
                tp_size=tp_size,
                pp_size=1,
                attention_dp_size=adp,
                moe_ep_size=None,
                moe_tp_size=None,
            )

        if ep > 1:
            moe_ep = tp
            moe_tp = 1
        else:
            moe_ep = 1
            moe_tp = tp
        return _WorkerShape(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=adp,
            moe_ep_size=moe_ep,
            moe_tp_size=moe_tp,
        )

    if num_gpu % workers != 0:
        raise ValueError(f"num_gpu ({num_gpu}) not divisible by workers ({workers})")

    per_worker = num_gpu // workers
    pp = 1  # InferenceX does not expose PP

    if per_worker % pp != 0:
        raise ValueError(f"per_worker_gpus ({per_worker}) not divisible by pp ({pp})")
    per_worker_no_pp = per_worker // pp

    if backend == "vllm":
        # vLLM: keep silicon's TP literal; derive attention DP.
        if per_worker_no_pp % tp != 0:
            raise ValueError(f"per_worker_gpus/pp ({per_worker_no_pp}) not divisible by tp ({tp})")
        tp_size = 1 if dp_attention else tp
        adp = per_worker_no_pp if dp_attention else per_worker_no_pp // tp
    else:
        # sglang, trtllm: route on dp_attention to avoid the TP+DP-on-attn assert.
        if dp_attention:
            tp_size = 1
            adp = per_worker_no_pp
        else:
            tp_size = per_worker_no_pp
            adp = 1

    if not is_moe:
        return _WorkerShape(
            tp_size=tp_size,
            pp_size=pp,
            attention_dp_size=adp,
            moe_ep_size=None,
            moe_tp_size=None,
        )

    width = tp_size * adp
    if backend == "vllm":
        # vLLM: at most one of (moe_tp, moe_ep) may be > 1.
        if ep > 1:
            moe_ep: int = width
            moe_tp = 1
        else:
            moe_ep = 1
            moe_tp = width
    else:
        # trtllm, sglang: keep silicon ep literally; derive moe_tp.
        if per_worker_no_pp % ep != 0:
            raise ValueError(f"per_worker_gpus/pp ({per_worker_no_pp}) not divisible by moe_ep ({ep})")
        moe_ep = ep
        moe_tp = per_worker_no_pp // ep

    if tp_size * adp != moe_tp * moe_ep:
        raise ValueError(
            f"MoE width invariant broken: tp*adp ({tp_size}*{adp}={tp_size * adp}) != "
            f"moe_tp*moe_ep ({moe_tp}*{moe_ep}={moe_tp * moe_ep})"
        )
    return _WorkerShape(
        tp_size=tp_size,
        pp_size=pp,
        attention_dp_size=adp,
        moe_ep_size=moe_ep,
        moe_tp_size=moe_tp,
    )
