# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check review-stage search spaces without running models or simulations."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import yaml
from dynamo.planner.simulation.config import PlannerRecommendationConfig
from dynamo.planner.simulation.load_predictor import LOAD_PREDICTOR_PRESETS
from dynamo.planner.simulation.presets import (
    FPM_SAMPLING,
    LOAD_SENSITIVITY,
    SCALING_POLICIES,
    throughput_intervals,
)
from dynamo.planner.simulation.provider import PlannerSearchSpace
from dynamo.router.simulation.config import (
    OVERLAP_SCORE_CREDIT_DEFAULTS,
    PREFILL_LOAD_SCALE_DEFAULTS,
    TEMPERATURE_DEFAULTS,
    RouterRecommendationConfig,
    RouterSearchSpace,
)

from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.config.common import split_config_sections

ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_expansion(values: list, defaults: list | tuple | dict, added: int) -> None:
    originals = set(defaults)
    named = {v for v in values if isinstance(v, (str, int, float))}
    require(originals <= named, f"Lost default candidates: {originals - named}")
    require(len(values) == len(originals) + added, "Unexpected extension size")
    require(
        len({json.dumps(v, sort_keys=True) for v in values}) == len(values),
        "Duplicate candidates",
    )


def main() -> None:
    contract = yaml.safe_load((ROOT / "experiment-contract.yaml").read_text())
    require(
        version("aisimulate") == contract["runtime"]["aisimulate_python"],
        "Use the pinned AISimulate environment",
    )
    paths = sorted(ROOT.glob("0[123]-*.yaml"))
    require(len(paths) == 3, "Expected exactly three scenario configurations")
    raw = [yaml.safe_load(path.read_text()) for path in paths]
    split = [split_config_sections(value, command="recommend") for value in raw]
    cores = [CoreRecommendationConfig.model_validate(core) for core, _ in split]
    require(
        split[0][0] == split[1][0] == split[2][0],
        "Core search spaces must be identical",
    )
    require(not split[0][1], "Scenario 1 must have no component adapters")
    require(set(split[1][1]) == {"router"}, "Scenario 2 must add only Router")
    require(
        set(split[2][1]) == {"router", "planner"},
        "Scenario 3 must add Router and Planner",
    )
    require(
        raw[1]["router"] == raw[2]["router"],
        "Scenario 3 must retain the full scenario 2 Router space",
    )
    for core in cores:
        require(
            core.optimizer.max_trials == 256 and core.optimizer.parallelism == 8,
            "Unexpected search budget",
        )
        require(
            core.optimization.target == "goodput_per_gpu"
            and not core.optimization.strict_sla,
            "Objective changed",
        )
        require(
            core.optimization.constraints.max_candidate_gpus == 256,
            "GPU ceiling changed",
        )
        require(
            core.optimization.constraints.min_candidate_gpus is None,
            "Unexpected GPU floor",
        )

    router = raw[1]["router"]
    RouterRecommendationConfig.model_validate(router)
    router_space = RouterSearchSpace.model_validate(router)
    require(
        router_space.mode == ["kv_router"], "Router scenario must enable KV routing"
    )
    require(
        set(router_space.prefill_load_model_type) == {"none", "ais"},
        "Default load-model choices lost",
    )
    for name, defaults in (
        ("overlap_score_credit", OVERLAP_SCORE_CREDIT_DEFAULTS),
        ("prefill_load_scale", PREFILL_LOAD_SCALE_DEFAULTS),
        ("temperature", TEMPERATURE_DEFAULTS),
    ):
        check_expansion(getattr(router_space, name), defaults, 3)

    planner = PlannerRecommendationConfig.model_validate(raw[2]["planner"])
    space = PlannerSearchSpace.model_validate(
        planner.model_dump(mode="python", exclude_none=True)
    )
    require(
        planner.policy == "enabled" and planner.max_num_gpus == 256,
        "Planner must stay enabled with 256-GPU ceiling",
    )
    enabled_defaults = {
        name: value for name, value in SCALING_POLICIES.items() if name != "disabled"
    }
    check_expansion(space.scaling_policy.preset, enabled_defaults, 3)
    check_expansion(space.fpm_sampling.preset, FPM_SAMPLING, 2)
    check_expansion(space.load_sensitivity.preset, LOAD_SENSITIVITY, 3)
    require(
        space.load_predictor.preset == list(LOAD_PREDICTOR_PRESETS),
        "Keep the entire default predictor menu",
    )
    require(
        "disabled" not in space.scaling_policy.preset,
        "Planner scenario must not silently disable scaling",
    )
    intervals = throughput_intervals(space.scaling_policy.preset)
    require(
        intervals
        == contract["traffic"]["planner_history"]["forecast_intervals_seconds"],
        "Warmup must cover every cadence",
    )
    require(
        contract["traffic"]["planner_history"]["source_window_seconds"] == [0, 345600],
        "Invalid history boundary",
    )
    require(
        contract["traffic"]["evaluation"]["source_window_seconds"] == [345600, 432000],
        "Invalid evaluation boundary",
    )
    require(
        contract["traffic"]["planner_history"]["evaluation_rows_allowed"] == 0,
        "History must exclude day 5",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "qualification": "schema_and_design_only_not_end_to_end_execution",
                "aisimulate": version("aisimulate"),
                "scenarios": [path.name for path in paths],
                "router_domains": {
                    name: len(getattr(router_space, name))
                    for name in (
                        "prefill_load_model_type",
                        "overlap_score_credit",
                        "prefill_load_scale",
                        "temperature",
                    )
                },
                "planner_domains": {
                    "enabled_scaling_policies": len(space.scaling_policy.preset),
                    "fpm_sampling": len(space.fpm_sampling.preset),
                    "load_sensitivity": len(space.load_sensitivity.preset),
                    "predictor_candidates_in_separate_presearch": len(
                        space.load_predictor.preset
                    ),
                },
                "history_intervals_seconds": intervals,
                "predictor_training_calls": 0,
                "simulation_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
