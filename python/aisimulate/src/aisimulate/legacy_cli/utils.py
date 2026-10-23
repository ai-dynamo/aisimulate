# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from aisimulate.sdk.pareto_analysis import (
    get_best_configs_for_target_load,
    get_pareto_front,
)
from aisimulate.sdk.picking import pick_default, pick_load_match
from aisimulate.sdk.task_v2 import Task

logger = logging.getLogger(__name__)


def process_experiment_result(
    task: Task,
    result: dict[str, pd.DataFrame],
    top_n: int = 5,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    max_total_gpus: int | None = None,
    strict_sla: bool = False,
) -> tuple:
    """
    Process the result of a single experiment.

    This is a thin wrapper around :func:`picking.pick_default` and
    :func:`picking.pick_load_match` that extracts parameters from the
    ``Task``.

    Args:
        task: Task object for the experiment.
        result: Dictionary containing the pareto_df result of the experiment.
        top_n: Number of top configurations to return.
        target_request_rate: If set, activates load-match picking (minimize
            GPUs for the given request rate in req/s).
        target_concurrency: If set, activates load-match picking (minimize
            GPUs for the given number of concurrent requests).
        max_total_gpus: Optional upper bound on total GPUs for load-match.
        strict_sla: When ``True``, ``pareto_df`` is filtered to only
            SLA-compliant data points (TPOT or request-latency) *before*
            the Pareto frontier is computed.  TTFT is already enforced at
            sweep time.  The resulting frontier, plot, and ``pareto.csv``
            will only contain configs that meet the user's SLA targets.

    Returns:
        tuple:
            - best_config_df: Best configuration dataframe.
            - best_throughput: Comparison score: throughput, or inverse GPU
              count for a full-target load match; capped fallback uses throughput.
            - pareto_frontier_df: Pareto frontier dataframe.
            - x_axis_col: X-axis column name.
            - best_latencies: Dict with ``ttft``, ``tpot``, ``request_latency``
              (and load-match fields when applicable) from the rank-1 config.
    """
    load_match = target_request_rate is not None or target_concurrency is not None

    pareto_df = result["pareto_df"]
    target_tpot = task.tpot
    target_request_latency = task.request_latency
    use_request_latency = target_request_latency is not None and target_request_latency > 0
    serving_mode = task.serving_mode
    total_gpus = task.effective_total_gpus if serving_mode == "afd" else (task.total_gpus or 0)

    x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"

    if load_match:
        picking_result = pick_load_match(
            pareto_df=pareto_df,
            serving_mode=serving_mode,
            target_tpot=target_tpot,
            target_request_latency=target_request_latency,
            target_request_rate=target_request_rate,
            target_concurrency=target_concurrency,
            max_total_gpus=max_total_gpus,
            top_n=top_n,
        )
    else:
        picking_result = pick_default(
            pareto_df=pareto_df,
            total_gpus=total_gpus,
            serving_mode=serving_mode,
            target_tpot=target_tpot,
            target_request_latency=target_request_latency,
            top_n=top_n,
            strict_sla=strict_sla,
        )

    best_config_df = picking_result["best_config_df"]
    best_throughput = picking_result["best_throughput"]
    best_latencies = picking_result.get("best_latencies", {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0})
    pareto_frontier_df = picking_result.get("pareto_frontier_df", pd.DataFrame())

    return best_config_df, best_throughput, pareto_frontier_df, x_axis_col, best_latencies


def _merge_into_top_n(
    exps: list[str],
    tasks: dict[str, Task],
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    pareto_x_axis: dict[str, str],
    top_n: int = 5,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    max_total_gpus: int | None = None,
) -> tuple:
    """Merge the best configs and pareto fronts into top N."""
    best_configs_dfs = []
    pareto_dfs = []
    retained_exps: list[str] = []
    for exp_name in exps:
        if exp_name not in best_configs:
            continue
        retained_exps.append(exp_name)
        backend_name = tasks[exp_name].primary_backend_name
        df = best_configs[exp_name].copy()
        if not df.empty:
            df["backend"] = backend_name
            df["_task_key"] = exp_name
            best_configs_dfs.append(df)

        pf = pareto_fronts.get(exp_name)
        if pf is not None and not pf.empty:
            pf_copy = pf.copy()
            pf_copy["backend"] = backend_name
            pf_copy["_task_key"] = exp_name
            pareto_dfs.append(pf_copy)

    # Preserve the picking objective when comparing results across backends.
    load_match = target_request_rate is not None or target_concurrency is not None
    if best_configs_dfs:
        df_best_configs = pd.concat(best_configs_dfs, ignore_index=True)
        if load_match:
            ref_task = tasks[df_best_configs.iloc[0]["_task_key"]]
            use_request_latency = ref_task.request_latency is not None and ref_task.request_latency > 0
            df_best_configs = get_best_configs_for_target_load(
                pareto_df=df_best_configs,
                constraint_col="request_latency" if use_request_latency else "tpot",
                constraint_value=ref_task.request_latency if use_request_latency else ref_task.tpot,
                target_request_rate=target_request_rate,
                target_concurrency=target_concurrency,
                max_total_gpus=max_total_gpus,
                top_n=top_n,
            )
        else:
            df_best_configs = df_best_configs.sort_values("tokens/s/gpu_cluster", ascending=False).head(top_n)
    else:
        df_best_configs = pd.DataFrame()

    best_throughput = 0.0
    if not df_best_configs.empty:
        row = df_best_configs.iloc[0]
        full_match = load_match and row["load_satisfied"] and (row["sla_satisfied"] or max_total_gpus is None)
        if full_match:
            gpus = row["total_gpus_needed"]
            best_throughput = 1.0 / gpus if gpus > 0 else 0.0
        else:
            best_throughput = float(row["tokens/s/gpu_cluster"])

    df_merged_pareto_front = x_col = None
    # Merge pareto fronts for plotting and recompute Pareto frontier
    if pareto_dfs:
        df_combined_pareto = pd.concat(pareto_dfs, ignore_index=True)
        ref_exp = next((name for name in retained_exps if name in pareto_x_axis), None)
        x_col = pareto_x_axis.get(ref_exp, "tokens/s/user")
        df_merged_pareto_front = get_pareto_front(
            df_combined_pareto,
            x_col,
            "tokens/s/gpu_cluster",
            maximize_x=(x_col != "request_latency"),
            maximize_y=True,
        )

    return df_best_configs, best_throughput, df_merged_pareto_front, x_col


def merge_experiment_results_by_mode(
    tasks: dict[str, Task],
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    pareto_x_axis: dict[str, str],
    top_n: int = 5,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    max_total_gpus: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, pd.DataFrame], dict[str, str]]:
    """
    Merge results from multiple experiments into Top N agg and disagg.
    For example, when backend="auto", we have 6 experiments: agg_trtllm, agg_vllm, agg_sglang,
    disagg_trtllm, disagg_vllm, disagg_sglang. This function merges them into 2:
    agg (with top N from all backends) and disagg (with top N from all backends).

    Args:
        tasks: Dictionary containing the task configs of the experiments.
        best_configs: Dictionary containing the best configs of the experiments.
        pareto_fronts: Dictionary containing the pareto fronts of the experiments.
        pareto_x_axis: Dictionary containing the pareto x-axis of the experiments.
        top_n: Number of top configurations to return.
        target_request_rate: Target request rate for minimum-GPU load-match picking.
        target_concurrency: Target concurrency for minimum-GPU load-match picking.
        max_total_gpus: Optional GPU budget for load-match picking.

    Returns:
        tuple:
            - best_configs: Dictionary containing the best configs of the merged experiments.
            - best_throughputs: Dictionary of objective-specific comparison scores: inverse GPU count for
                full-target load matches, or cluster token throughput for GPU-capped fallbacks, SLA
                diagnostics under a GPU budget, and runs without a load target. Unbudgeted sizing
                retains inverse-GPU scores for diagnostic results.
            - pareto_fronts: Dictionary containing the pareto fronts of the merged experiments.
            - pareto_x_axis: Dictionary containing the pareto x-axis of the merged experiments.
    """
    agg_exps = [name for name, task in tasks.items() if task.serving_mode == "agg"]
    disagg_exps = [name for name, task in tasks.items() if task.serving_mode == "disagg"]
    afd_exps = [name for name, task in tasks.items() if task.serving_mode == "afd"]

    merged_best_configs = {}
    merged_best_throughputs = {}
    merged_pareto_fronts = {}
    merged_pareto_x_axis = {}

    # AFD results use their own ColumnsAFD schema; merge each serving mode
    # within its own bucket instead of concatenating across schemas.
    for mode_name, mode_exps in (("agg", agg_exps), ("disagg", disagg_exps), ("afd", afd_exps)):
        if not mode_exps:
            continue
        mode_merged = _merge_into_top_n(
            mode_exps,
            tasks,
            best_configs,
            pareto_fronts,
            pareto_x_axis,
            top_n,
            target_request_rate=target_request_rate,
            target_concurrency=target_concurrency,
            max_total_gpus=max_total_gpus,
        )
        merged_best_configs[mode_name] = mode_merged[0]
        merged_best_throughputs[mode_name] = mode_merged[1]
        merged_pareto_fronts[mode_name] = mode_merged[2]
        merged_pareto_x_axis[mode_name] = mode_merged[3]

    return merged_best_configs, merged_best_throughputs, merged_pareto_fronts, merged_pareto_x_axis
