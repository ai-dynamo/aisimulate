# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native component adapters with an explicit, separate predictor history input."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import time

from aisimulate.config_adapter import RecommendationAdapterContext


class CampaignAdapter:
    api_version = 1

    def __init__(self, adapter, public_config, core_config, history_trace, output):
        self.adapter = adapter
        self.name = adapter.name
        self.section = adapter.section
        self.public_config = deepcopy(public_config)
        self.core_config = core_config
        self.history_trace = history_trace
        self.output = Path(output)

    def generate_search_space(self, search_spec, context):
        from run_sweep import event, write

        del search_spec
        traffic = self.core_config.traffic.model_dump(mode="json", exclude_none=True)
        if self.name == "dynamo.planner":
            if not self.history_trace:
                raise ValueError("Planner presearch requires explicit four-day history")
            workload = deepcopy(dict(context.workload))
            workload.update(trace_path=self.history_trace, trace_paths=[self.history_trace],
                            max_sim_time_ms=345600000.0)
            context = replace(context, workload=workload, show_progress=True)
            traffic["source"]["paths"] = [self.history_trace]
            traffic["stop"] = {"max_virtual_time_seconds": 345600.0}
        adapter_context = RecommendationAdapterContext(
            engine=self.core_config.engine.model_dump(mode="json", exclude_none=True),
            traffic=traffic,
            evaluation=self.core_config.evaluation.model_dump(mode="json", exclude_none=True),
            optimization=self.core_config.optimization.model_dump(mode="json", exclude_none=True),
            sweep=context,
        )
        start = time.perf_counter()
        event(self.output / "adapter-events.jsonl", {
            "event": "prepare_started", "adapter": self.name,
            "trace_paths": context.workload.get("trace_paths"),
        })
        plan = self.adapter.compile_recommendation(self.public_config, adapter_context)
        elapsed = time.perf_counter() - start
        write(self.output / (self.section + "-search-plan.json"), asdict(plan))
        event(self.output / "adapter-events.jsonl", {
            "event": "prepare_completed", "adapter": self.name,
            "seconds": elapsed, "diagnostics": plan.diagnostics,
        })
        return plan

    def materialize_replay(self, plan, selection, context):
        result = self.adapter.materialize_candidate(plan, selection, context)
        if self.name != "dynamo.planner":
            return result
        hooks = []
        for hook in result.runtime_hooks:
            config = deepcopy(hook.config)
            if hook.provider == "dynamo.planner":
                config["planner_config"]["load_predictor_warmup_trace"] = self.history_trace
                if config["planner_config"].get("max_gpu_budget") != 256:
                    raise ValueError("Planner budget changed during materialization")
            hooks.append(replace(hook, config=config))
        return replace(result, runtime_hooks=tuple(hooks))
