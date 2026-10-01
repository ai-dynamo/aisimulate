# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small unranked fixture to exercise the exact experiment runner on compute nodes."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import yaml
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config_adapter import PredictionAdapterContext
from aisimulate.sweeper.replay import canonical_json

from scenario_runner import ScenarioRunnerFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "01-static.yaml").read_text())
    trace = args.output / "tiny-evaluation.jsonl"
    trace.write_text("".join(json.dumps({"timestamp": i * 1000.0, "input_length": 128,
                                        "output_length": 8, "hash_ids": [1, i + 2]}) + "\n"
                             for i in range(8)))
    history = args.output / "tiny-history.jsonl"
    history.write_text("".join(json.dumps({"timestamp": i * 180000.0, "input_length": 128,
                                          "output_length": 8, "hash_ids": [1, i + 20]}) + "\n"
                               for i in range(12)))
    cases = []
    for backend, mode, dp in [("vllm", "aggregated", 1), ("vllm", "disaggregated", 8),
                              ("sglang", "aggregated", 8), ("sglang", "disaggregated", 1)]:
        engine = deepcopy(base["engine"])
        engine.update(mode=mode, backend=backend,
                      backend_version="0.24.0" if backend == "vllm" else "0.5.14")
        if mode == "aggregated":
            engine.pop("kv_transfer", None)
        roles = ["aggregated"] if mode == "aggregated" else ["prefill", "decode"]
        workers = {}
        for role in roles:
            worker = deepcopy(engine["workers"][role])
            worker["parallelism"] = {"replicas": 1, "tensor": 8 // dp, "attention_data": dp,
                                     "pipeline": 1, "moe_tensor": 8 // dp, "moe_expert": dp}
            worker["scheduler"] = {"max_batched_tokens": 16384, "max_sequences": 128}
            workers[role] = worker
        engine["workers"] = workers
        traffic = deepcopy(base["traffic"])
        traffic["source"]["paths"] = [str(trace)]
        core = CorePredictionConfig.model_validate({"engine": engine, "traffic": traffic,
                                                    "evaluation": base["evaluation"]})
        context = PredictionAdapterContext(
            engine=core.engine.model_dump(mode="json", exclude_none=True),
            traffic=core.traffic.model_dump(mode="json", exclude_none=True),
            evaluation=core.evaluation.model_dump(mode="json", exclude_none=True))
        adapters = {}
        if args.scenario >= 2:
            from dynamo.router.simulation.provider import create_provider
            provider = create_provider()
            adapters[provider.name] = provider.compile_prediction(
                {"policy": "kv_router", "prefill_load_model": {"type": "ais"},
                 "overlap_score_credit": 0.75, "prefill_load_scale": 2.0, "temperature": 0.0}, context)
        if args.scenario == 3:
            from dynamo.planner.simulation.provider import create_provider
            provider = create_provider()
            adapter = provider.compile_prediction(
                {"policy": "enabled", "target": "sla", "max_num_gpus": 256,
                 "min_workers": 1, "prefill_min_workers": 1, "decode_min_workers": 1,
                 "enable_throughput_scaling": True, "enable_load_scaling": True,
                 "load_predictor": "constant", "throughput_adjustment_interval_seconds": 180}, context)
            hooks = []
            for hook in adapter.runtime_hooks:
                payload = deepcopy(hook.config)
                payload["planner_config"]["load_predictor_warmup_trace"] = str(history)
                hooks.append(replace(hook, config=payload))
            adapters[provider.name] = replace(adapter, runtime_hooks=tuple(hooks))
        spec = prediction_to_replay_spec(core, adapter_specs=adapters)
        spec = replace(spec, goal={**spec.goal, "target": "goodput_per_gpu"})
        factory = ScenarioRunnerFactory(
            attempts_dir=str(args.output / "attempts"), scenario=args.scenario,
            trace_path=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            history_trace=str(history) if args.scenario == 3 else None,
            history_sha256=hashlib.sha256(history.read_bytes()).hexdigest() if args.scenario == 3 else None,
            dynamo_sha="c7241c2f153efba10b57c38c2144b70d82194a4d",
            expected_requests=8, expected_input_tokens=1024, expected_output_tokens=64)
        runner = factory.create(len(cases))
        started = time.perf_counter()
        try:
            report = runner.run(spec)
        finally:
            runner.close()
        entry = {"backend": backend, "mode": mode, "attention_dp": dp,
                 "seconds": time.perf_counter() - started,
                 "derived": report.metadata["scenario"]["derived"],
                 "planner_ticks": report.metrics.get("planner_total_ticks"),
                 "completed": report.metrics["completed_requests"]}
        assert entry["completed"] == 8
        cases.append(entry)
        print(json.dumps(entry), flush=True)
        from deserialize import replay_spec_from_dict
        assert canonical_json(replay_spec_from_dict(json.loads(canonical_json(spec)))) == canonical_json(spec)
    result = {"status": "PASS", "qualification": "SMOKE_ONLY_NOT_A_FORMAL_STUDY", "cases": cases}
    (args.output / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
