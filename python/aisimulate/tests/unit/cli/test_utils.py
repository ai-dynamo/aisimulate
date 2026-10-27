# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI utility functions in utils.py.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aisimulate.legacy_cli.utils import (
    _merge_into_top_n,
    merge_experiment_results_by_mode,
    process_experiment_result,
)

pytestmark = pytest.mark.unit


class TestProcessExperimentResult:
    """Tests for process_experiment_result function."""

    def test_process_result_with_tpot_constraint(self):
        """Test processing result with TPOT constraint."""
        # Create mock task config
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        # Create sample pareto_df
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0],
                "tokens/s/user": [10.0, 12.0, 9.0],
                "num_total_gpus": [8, 8, 8],
                "tpot": [25.0, 28.0, 20.0],
                "parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4", "tp1_pp1_dp8"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, _ = process_experiment_result(
            mock_task_config, result, top_n=2
        )

        # Verify results
        assert not best_config_df.empty
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        assert best_throughput > 0.0
        assert x_axis_col == "tokens/s/user"
        assert not pareto_frontier_df.empty

    def test_process_result_with_request_latency_constraint(self):
        """Test processing result with request_latency constraint."""
        # Create mock task config
        mock_task_config = MagicMock()
        mock_task_config.tpot = None
        mock_task_config.request_latency = 1200.0
        mock_task_config.serving_mode = "disagg"
        mock_task_config.total_gpus = 32

        # Create sample pareto_df
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0],
                "tokens/s/user": [10.0, 12.0, 9.0],
                "num_total_gpus": [8, 8, 8],
                "request_latency": [1000.0, 1100.0, 900.0],
                "(d)parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4", "tp1_pp1_dp8"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, _ = process_experiment_result(
            mock_task_config, result, top_n=3
        )

        # Verify results
        assert not best_config_df.empty
        assert x_axis_col == "request_latency"
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        assert best_throughput > 0.0

    def test_process_result_with_empty_pareto_df(self):
        """Test processing result with empty pareto_df."""
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        result = {"pareto_df": pd.DataFrame()}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, _ = process_experiment_result(
            mock_task_config, result, top_n=5
        )

        # Verify results
        assert best_config_df.empty
        assert best_throughput == 0.0
        assert pareto_frontier_df.empty
        assert x_axis_col in ["tokens/s/user", "request_latency"]

    def test_process_result_with_none_pareto_df(self):
        """Test processing result with None pareto_df."""
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        result = {"pareto_df": None}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, _ = process_experiment_result(
            mock_task_config, result, top_n=5
        )

        # Verify results
        assert best_config_df.empty
        assert best_throughput == 0.0
        assert pareto_frontier_df.empty

    def test_process_result_disagg_mode_uses_correct_group_by(self):
        """Test that disagg mode uses (d)parallel as group_by key."""
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "disagg"
        mock_task_config.total_gpus = 32

        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0],
                "tokens/s/user": [10.0, 12.0],
                "num_total_gpus": [8, 8],
                "tpot": [25.0, 28.0],
                "(d)parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _, _ = process_experiment_result(mock_task_config, result, top_n=5)

        # Verify that results are returned (group_by should work correctly)
        assert not best_config_df.empty

    def test_process_result_top_n_limiting(self):
        """Test that top_n correctly limits the number of returned configs."""
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        # Create sample with more configs than top_n
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0, 110.0, 95.0, 105.0],
                "tokens/s/user": [10.0, 12.0, 9.0, 11.0, 9.5, 10.5],
                "num_total_gpus": [8, 8, 8, 8, 8, 8],
                "tpot": [25.0, 28.0, 20.0, 26.0, 22.0, 24.0],
                "parallel": [f"config_{i}" for i in range(6)],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _, _ = process_experiment_result(mock_task_config, result, top_n=3)

        # Should return at most 3 configs
        assert len(best_config_df) <= 3

    def test_process_result_computes_tokens_per_gpu_cluster(self):
        """Test that tokens/s/gpu_cluster is correctly computed."""
        mock_task_config = MagicMock()
        mock_task_config.tpot = 30.0
        mock_task_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0],
                "tokens/s/user": [10.0],
                "num_total_gpus": [8],
                "tpot": [25.0],
                "parallel": ["tp4_pp1_dp2"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _, _ = process_experiment_result(mock_task_config, result, top_n=5)

        # Verify tokens/s/gpu_cluster is computed
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        # Expected: 100 * (32 // 8) * 8 / 32 = 100 * 4 * 8 / 32 = 100
        assert best_config_df["tokens/s/gpu_cluster"].values[0] > 0


class TestMergeIntoTopN:
    """Tests for _merge_into_top_n helper function."""

    def test_merge_multiple_backends(self):
        """Test merging configs from multiple backends."""
        # Create mock task configs
        tasks = {}
        for backend in ["trtllm", "vllm", "sglang"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.primary_backend_name = backend
            tasks[f"agg_{backend}"] = mock_config

        # Create mock best configs
        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
        }

        pareto_fronts = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
        }

        pareto_x_axis = {"agg_trtllm": "tokens/s/user", "agg_vllm": "tokens/s/user", "agg_sglang": "tokens/s/user"}

        exps = ["agg_trtllm", "agg_vllm", "agg_sglang"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Verify results
        assert not merged_best.empty
        assert "backend" in merged_best.columns
        assert merged_throughput == 120.0  # Best from vllm
        assert not merged_pareto.empty
        assert "backend" in merged_pareto.columns
        assert x_col == "tokens/s/user"

    def test_merge_with_top_n_limiting(self):
        """Test that merging respects top_n limit."""
        tasks = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for i, backend in enumerate(["trtllm", "vllm", "sglang"]):
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.primary_backend_name = backend
            exp_name = f"agg_{backend}"
            tasks[exp_name] = mock_config

            # Create configs with different throughputs
            best_configs[exp_name] = pd.DataFrame(
                {
                    "tokens/s/gpu_cluster": [100.0 + i * 10, 95.0 + i * 10],
                    "tokens/s/user": [10.0 + i, 9.5 + i],
                }
            )
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        exps = list(tasks.keys())

        merged_best, _, _, _ = _merge_into_top_n(exps, tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=3)

        # Should return at most 3 configs
        assert len(merged_best) <= 3
        # Should be sorted by throughput descending
        assert merged_best["tokens/s/gpu_cluster"].is_monotonic_decreasing

    def test_merge_with_empty_dataframes(self):
        """Test merging when some configs are empty."""
        tasks = {
            "agg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm"),
            "agg_vllm": MagicMock(backend_name="vllm", primary_backend_name="vllm"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame(),  # Empty
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
        }

        pareto_fronts = {
            "agg_trtllm": pd.DataFrame(),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
        }

        pareto_x_axis = {"agg_trtllm": "tokens/s/user", "agg_vllm": "tokens/s/user"}

        exps = ["agg_trtllm", "agg_vllm"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should only contain vllm results
        assert not merged_best.empty
        assert len(merged_best) == 1
        assert merged_best["backend"].values[0] == "vllm"
        assert merged_throughput == 120.0

    def test_merge_with_missing_pareto_fronts(self):
        """Test merging when pareto fronts are missing."""
        tasks = {"agg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm")}

        best_configs = {"agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]})}

        pareto_fronts = {}  # No pareto fronts

        pareto_x_axis = {"agg_trtllm": "tokens/s/user"}

        exps = ["agg_trtllm"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should still merge best configs
        assert not merged_best.empty
        assert merged_throughput == 100.0
        # Pareto front should be None or empty
        assert merged_pareto is None or merged_pareto.empty

    def test_merge_recomputes_pareto_frontier(self):
        """Test that merge recomputes Pareto frontier from combined data."""
        tasks = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for backend in ["trtllm", "vllm"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.primary_backend_name = backend
            exp_name = f"agg_{backend}"
            tasks[exp_name] = mock_config

            # Create different pareto fronts
            if backend == "trtllm":
                pareto_fronts[exp_name] = pd.DataFrame(
                    {
                        "tokens/s/gpu_cluster": [100.0, 90.0],
                        "tokens/s/user": [10.0, 12.0],
                    }
                )
            else:
                pareto_fronts[exp_name] = pd.DataFrame(
                    {
                        "tokens/s/gpu_cluster": [110.0, 95.0],
                        "tokens/s/user": [11.0, 13.0],
                    }
                )

            best_configs[exp_name] = pareto_fronts[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        exps = ["agg_trtllm", "agg_vllm"]

        _, _, merged_pareto, _ = _merge_into_top_n(exps, tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5)

        # Should combine and recompute pareto frontier
        assert not merged_pareto.empty
        assert "backend" in merged_pareto.columns


class TestMergeExperimentResultsByMode:
    """Tests for merge_experiment_results_by_mode function."""

    def test_merge_six_backends_into_two_modes(self):
        """Test merging 6 backend experiments into agg and disagg."""
        # Create mock task configs for all 6 experiments
        tasks = {}
        for backend in ["trtllm", "vllm", "sglang"]:
            for mode in ["agg", "disagg"]:
                mock_config = MagicMock()
                mock_config.backend_name = backend
                mock_config.primary_backend_name = backend
                mock_config.serving_mode = mode
                tasks[f"{mode}_{backend}"] = mock_config

        # Create mock best configs with different throughputs
        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
            "disagg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [140.0], "tokens/s/user": [14.0]}),
            "disagg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [130.0], "tokens/s/user": [13.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        # Merge results
        merged_configs, merged_throughputs, merged_fronts, merged_axis = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should have only 2 merged results: agg and disagg
        assert len(merged_configs) == 2
        assert "agg" in merged_configs
        assert "disagg" in merged_configs

        # Merged configs should have "backend" column
        assert "backend" in merged_configs["agg"].columns
        assert "backend" in merged_configs["disagg"].columns

        # Best throughput should be from vllm for agg (120.0) and trtllm for disagg (150.0)
        assert merged_throughputs["agg"] == 120.0
        assert merged_throughputs["disagg"] == 150.0

        # Pareto fronts should be merged
        assert not merged_fronts["agg"].empty
        assert not merged_fronts["disagg"].empty

        # X-axis should be preserved
        assert merged_axis["agg"] == "tokens/s/user"
        assert merged_axis["disagg"] == "tokens/s/user"

    def test_merge_with_top_n_limiting(self):
        """Test that merge respects top_n limit across backends."""
        tasks = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for i, backend in enumerate(["trtllm", "vllm", "sglang"]):
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.primary_backend_name = backend
            mock_config.serving_mode = "agg"
            exp_name = f"agg_{backend}"
            tasks[exp_name] = mock_config

            # Each backend has 3 configs
            best_configs[exp_name] = pd.DataFrame(
                {
                    "tokens/s/gpu_cluster": [100.0 + i * 10 + j for j in range(3)],
                    "tokens/s/user": [10.0 + i + j * 0.1 for j in range(3)],
                }
            )
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        # Merge with top_n=5
        merged_configs, _, _, _ = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should return at most 5 configs for agg
        assert len(merged_configs["agg"]) <= 5

    def test_merge_with_mixed_agg_disagg(self):
        """Test merging with mix of agg and disagg experiments."""
        tasks = {
            "agg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm", serving_mode="agg"),
            "agg_vllm": MagicMock(backend_name="vllm", primary_backend_name="vllm", serving_mode="agg"),
            "disagg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm", serving_mode="disagg"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        merged_configs, merged_throughputs, _, _ = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should have 2 results: agg (from 2 backends) and disagg (from 1 backend)
        assert len(merged_configs) == 2
        assert not merged_configs["agg"].empty
        assert not merged_configs["disagg"].empty

        # Agg should have 2 configs, disagg should have 1
        assert len(merged_configs["agg"]) == 2
        assert len(merged_configs["disagg"]) == 1

    def test_merge_preserves_backend_information(self):
        """Test that backend information is preserved in merged results."""
        tasks = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for backend in ["trtllm", "vllm"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.primary_backend_name = backend
            mock_config.serving_mode = "agg"
            exp_name = f"agg_{backend}"
            tasks[exp_name] = mock_config

            best_configs[exp_name] = pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]})
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        merged_configs, _, _, _ = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Backend column should exist and contain correct values
        assert "backend" in merged_configs["agg"].columns
        backends = set(merged_configs["agg"]["backend"].values)
        assert backends.issubset({"trtllm", "vllm"})

    def test_merge_with_empty_experiments(self):
        """Test merging when some experiments have no results."""
        tasks = {
            "agg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm", serving_mode="agg"),
            "agg_vllm": MagicMock(backend_name="vllm", primary_backend_name="vllm", serving_mode="agg"),
            "disagg_trtllm": MagicMock(backend_name="trtllm", primary_backend_name="trtllm", serving_mode="disagg"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame(),  # Empty
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        merged_configs, merged_throughputs, _, _ = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should still produce results for both modes
        assert len(merged_configs) == 2
        # Agg should only have vllm result
        assert not merged_configs["agg"].empty
        assert len(merged_configs["agg"]) == 1
        assert merged_throughputs["agg"] == 120.0


@pytest.mark.parametrize("load_target", [{"target_request_rate": 1.0}, {"target_concurrency": 1.0}])
@pytest.mark.parametrize("gpu_cap", [None, 4])
@pytest.mark.parametrize("multiple_backends", [False, True])
def test_default_load_match_keeps_minimum_gpu_ranking(monkeypatch, load_target, gpu_cap, multiple_backends):
    from aisimulate.legacy_cli import main as cli_main

    monkeypatch.setattr(cli_main, "log_final_summary", lambda **kwargs: None)
    tasks = {}
    for name, mode, gpus, throughput, capacity in [
        ("cheap", "agg", 2, 100.0, 1.0),
        ("fast", "agg", 8, 120.0, 2.0),
        ("disagg", "disagg", 4, 110.0, 1.0),
    ]:
        if name == "fast" and not multiple_backends:
            continue
        task = MagicMock(
            primary_backend_name=name,
            serving_mode=mode,
            total_gpus=gpus,
            tpot=30.0,
            request_latency=None,
        )
        task.to_yaml.return_value = "test config"
        task.run.return_value = pd.DataFrame(
            {
                "parallel": [f"tp{gpus}"],
                "(d)parallel": [f"tp{gpus}"],
                "tpot": [25.0],
                "tokens/s/user": [40.0],
                "tokens/s/gpu": [throughput],
                "seq/s": [capacity],
                "concurrency": [capacity],
                "num_total_gpus": [gpus],
            }
        )
        tasks[name] = task

    chosen, configs, _, scores, _, _ = cli_main._execute_tasks(
        tasks, mode="default", top_n=1, max_total_gpus=gpu_cap, **load_target
    )

    # Both agg options meet the same load/SLA. One replica needs 2 GPUs versus
    # 8 GPUs, so a faster per-GPU rate cannot justify the more expensive one.
    agg_key = "agg" if multiple_backends else "cheap"
    assert configs[agg_key].iloc[0]["total_gpus_needed"] == 2
    if multiple_backends:
        assert configs[agg_key].iloc[0]["backend"] == "cheap"
    assert scores[agg_key] == 0.5
    # An exact-budget disagg fit must use the same inverse-GPU units as agg.
    assert scores["disagg"] == 0.25
    assert chosen == agg_key


def test_default_without_load_target_still_chooses_throughput(monkeypatch):
    from aisimulate.legacy_cli import main as cli_main

    monkeypatch.setattr(cli_main, "log_final_summary", lambda **kwargs: None)
    tasks = {}
    for mode, throughput in (("agg", 100.0), ("disagg", 200.0)):
        for backend in range(2):
            task = MagicMock(
                primary_backend_name=f"backend{backend}",
                serving_mode=mode,
                total_gpus=4,
                tpot=30.0,
                request_latency=None,
            )
            task.to_yaml.return_value = "test config"
            task.run.return_value = pd.DataFrame(
                {
                    "parallel": ["shared"],
                    "(d)parallel": ["shared"],
                    "tpot": [25.0],
                    "tokens/s/user": [40.0],
                    "tokens/s/gpu": [throughput - backend * 5.0],
                    "num_total_gpus": [2],
                }
            )
            tasks[f"{mode}_{backend}"] = task

    chosen, _, _, scores, _, _ = cli_main._execute_tasks(tasks, mode="default", top_n=1)

    assert scores == {"agg": 100.0, "disagg": 200.0}
    assert chosen == "disagg"


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize("multiple_backends", [False, True])
@pytest.mark.parametrize(
    "scenario, expected_chosen, expected_scores",
    [
        ("full_agg", "agg", {"agg": 0.5, "disagg": 75.0}),
        ("full_disagg", "disagg", {"agg": 75.0, "disagg": 0.5}),
        ("all_capped", "agg", {"agg": 100.0, "disagg": 75.0}),
        ("empty_agg", "disagg", {"agg": 0.0, "disagg": 75.0}),
        ("all_empty", "none", {"agg": 0.0, "disagg": 0.0}),
    ],
)
def test_load_match_cross_mode_prefers_full_target(
    monkeypatch, load_target, multiple_backends, scenario, expected_chosen, expected_scores
):
    from aisimulate.legacy_cli import main as cli_main

    if scenario != "all_empty":
        monkeypatch.setattr(cli_main, "log_final_summary", lambda **kwargs: None)
    tasks = {}
    for mode in ("agg", "disagg"):
        for backend in range(2 if multiple_backends else 1):
            full_target = scenario == f"full_{mode}"
            gpus = 2 if full_target or (scenario == "all_capped" and mode == "agg") else 3
            capacity = 1.0 if full_target else 0.1
            if backend and full_target:
                # A capped backend gives the merged full-target row a null
                # load_served_pct column; null still means no fallback.
                gpus, capacity = 3, 0.1
            columns = {
                "parallel": ["shared"],
                "(d)parallel": ["shared"],
                "tpot": [25.0],
                "tokens/s/user": [40.0],
                "tokens/s/gpu": [100.0 - backend * 10.0],
                "seq/s": [capacity],
                "concurrency": [capacity],
                "num_total_gpus": [gpus],
            }
            if scenario == "all_empty" or (scenario == "empty_agg" and mode == "agg"):
                # A successful sweep can still have no load-match candidate.
                del columns["seq/s" if load_target == "target_request_rate" else "concurrency"]
            task = MagicMock(
                primary_backend_name=f"backend{backend}",
                serving_mode=mode,
                total_gpus=4,
                tpot=30.0,
                request_latency=None,
            )
            task.to_yaml.return_value = "test config"
            task.run.return_value = pd.DataFrame(columns)
            name = f"{mode}_{backend}" if multiple_backends else mode
            tasks[name] = task

    chosen, configs, _, scores, _, _ = cli_main._execute_tasks(
        tasks, mode="default", top_n=1, max_total_gpus=4, **{load_target: 1.0}
    )

    # Scores retain their documented units; a full-target 0.5 must beat a
    # capped 75, while all-capped modes compare throughput (100 beats 75).
    assert scores == expected_scores
    if scenario == "all_empty":
        assert all(frame.empty for frame in configs.values())
    elif scenario == "empty_agg":
        assert configs["agg"].empty
    assert chosen == expected_chosen


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize(
    "target, gpu_cap, request_latency, expected_backends, expected_gpus, expected_score",
    [
        (1.0, None, None, ["cheap", "fast"], [2, 4], 0.5),
        (1.0, 3, None, ["cheap"], [2], 0.5),
        (1.0, 2, None, ["cheap"], [2], 0.5),
        (10.0, 4, None, ["fast", "cheap"], [4, 4], 120.0),
        (1.0, None, 700.0, ["cheap"], [2], 0.5),
    ],
)
def test_merge_load_match_preserves_constraints_and_backend_choices(
    mode, load_target, target, gpu_cap, request_latency, expected_backends, expected_gpus, expected_score
):
    tasks = {}
    best_configs = {}
    for backend, gpus, throughput, capacity, latency in [
        ("cheap", 2, 100.0, 1.0, 500.0),
        ("fast", 4, 120.0, 2.0, 1000.0),
    ]:
        task = SimpleNamespace(
            serving_mode=mode,
            primary_backend_name=backend,
            tpot=30.0,
            request_latency=request_latency,
        )
        tasks[backend] = task
        best_configs[backend] = pd.DataFrame(
            {
                # Identical parallel labels from different backends are distinct choices.
                "parallel": ["shared"],
                "(d)parallel": ["shared"],
                "tpot": [25.0],
                "request_latency": [latency],
                "tokens/s/gpu": [throughput],
                "seq/s": [capacity],
                "concurrency": [capacity],
                "num_total_gpus": [gpus],
            }
        )

    configs, scores, _, _ = merge_experiment_results_by_mode(
        tasks,
        best_configs,
        {},
        {},
        top_n=2,
        max_total_gpus=gpu_cap,
        **{load_target: target},
    )

    assert configs[mode]["backend"].tolist() == expected_backends
    assert configs[mode]["total_gpus_needed"].tolist() == expected_gpus
    assert scores[mode] == expected_score


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
@pytest.mark.parametrize("multiple_backends", [False, True])
@pytest.mark.parametrize("sla_fallback_fits_load", [False, True])
@pytest.mark.parametrize("gpu_cap", [0, 1, 4])
def test_load_match_prefers_sla_compliant_partial_load(
    monkeypatch, load_target, multiple_backends, sla_fallback_fits_load, gpu_cap
):
    from aisimulate.legacy_cli import main as cli_main

    monkeypatch.setattr(cli_main, "log_final_summary", lambda **kwargs: None)
    tasks = {}
    for mode in ("agg", "disagg"):
        for backend in range(2 if multiple_backends else 1):
            capacity = (10.0 if sla_fallback_fits_load else 1.0) if mode == "agg" else 3.0
            task = MagicMock(
                primary_backend_name=f"backend{backend}",
                serving_mode=mode,
                total_gpus=4,
                tpot=30.0,
                request_latency=None,
            )
            task.to_yaml.return_value = "test config"
            task.run.return_value = pd.DataFrame(
                {
                    "parallel": ["shared"],
                    "(d)parallel": ["shared"],
                    "tpot": [40.0 if mode == "agg" else 25.0],
                    "tokens/s/user": [40.0],
                    "tokens/s/gpu": [200.0 if mode == "agg" else 150.0],
                    "seq/s": [capacity],
                    "concurrency": [capacity],
                    "num_total_gpus": [2],
                }
            )
            tasks[f"{mode}_{backend}" if multiple_backends else mode] = task

    chosen, configs, _, scores, _, _ = cli_main._execute_tasks(
        tasks, mode="default", top_n=1, max_total_gpus=gpu_cap, **{load_target: 10.0}
    )

    if gpu_cap < 2:
        assert chosen == "none"
        assert all(frame.empty for frame in configs.values())
        return
    assert chosen == "disagg"
    assert all((frame["total_gpus_needed"] <= 4).all() for frame in configs.values())
    assert configs["disagg"].iloc[0]["sla_satisfied"]
    assert not configs["disagg"].iloc[0]["load_satisfied"]
    assert not configs["agg"].iloc[0]["sla_satisfied"]
    assert configs["agg"].iloc[0]["load_satisfied"] == sla_fallback_fits_load
    assert scores["agg"] == 200.0


@pytest.mark.parametrize("load_target", ["target_request_rate", "target_concurrency"])
def test_merge_load_match_recomputes_fallback_state(load_target):
    tasks = {
        "backend": SimpleNamespace(serving_mode="agg", primary_backend_name="backend", tpot=30.0, request_latency=None)
    }
    stale = pd.DataFrame(
        {
            "tpot": [25.0],
            "tokens/s/gpu": [100.0],
            "seq/s": [10.0],
            "concurrency": [10.0],
            "num_total_gpus": [2],
            "load_served_pct": [10.0],
            "sla_satisfied": [False],
            "load_satisfied": [False],
            "gpu_budget_satisfied": [False],
        }
    )

    configs, scores, _, _ = merge_experiment_results_by_mode(
        tasks, {"backend": stale}, {}, {}, top_n=1, max_total_gpus=4, **{load_target: 10.0}
    )

    row = configs["agg"].iloc[0]
    assert row["sla_satisfied"] and row["load_satisfied"] and row["gpu_budget_satisfied"]
    assert "load_served_pct" not in row.index
    assert scores["agg"] == 0.5
