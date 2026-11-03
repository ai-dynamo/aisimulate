#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supplemental synthetic host costs: caller-managed initial fit, seeds and mixed prior.

Use one fresh worker process per case and trial. The shared host_costs
helper provides identity/hash/RSS validation. No silicon accuracy is implied.
--dry-run performs every assertion and public call without recording timings.
All timers include full public calls; input generation, assertions, diagnostics,
fixture creation, and offline seed preparation are reported separately as setup.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

CASES = ("cold2", "cold4", "seeded2", "seeded4", "fallback")
KINDS = ("pure_decode", "pure_prefill", "contains_locally_mixed", "cross_rank_aggregated")


def load_helper(path):
    spec = importlib.util.spec_from_file_location("gbdt_host_cost_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observations(stores, count=71):
    from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics

    result = {}
    for kind in KINDS[:stores]:
        rows = []
        for index in range(count):
            ranks = []
            rank_count = 2 if stores == 4 else 1
            batch = 1 + index % 5
            for rank in range(rank_count):
                prefill = (
                    kind == "pure_prefill"
                    or kind == "contains_locally_mixed"
                    or (kind == "cross_rank_aggregated" and rank == 0)
                )
                decode = (
                    kind == "pure_decode"
                    or kind == "contains_locally_mixed"
                    or (kind == "cross_rank_aggregated" and rank == 1)
                )
                ranks.append(
                    ForwardPassMetric(
                        fpm_id=index * 2 + rank,
                        configuration_id="synthetic-host-expanded-v1",
                        version=1,
                        worker_id="expanded-worker",
                        dp_rank=rank,
                        counter_id=index,
                        wall_time_s=(10 + KINDS.index(kind) + 0.4 * (index % 5) + (1 if index >= 20 else 0)) / 1000,
                        scheduled=RequestMetrics(
                            num_prefill_requests=batch if prefill else 0,
                            sum_prefill_tokens=32 * batch if prefill else 0,
                            num_decode_requests=batch if decode else 0,
                            sum_decode_kv_tokens=128 * batch if decode else 0,
                        ),
                        queued=RequestMetrics(),
                        capture_id="synthetic-host-expanded-v1",
                    )
                )
            rows.append(ForwardPassIteration(tuple(ranks)))
        result[kind] = rows
    return result


def mixed_setup(root, temporary):
    """Reuse each pinned checkout's real original-A fixture without pytest collection."""
    import pytest

    sys.path.insert(0, str(root / "python/aisimulate"))
    from tests.unit.sdk import test_fpm_mixed_composition_integration as mixed_fixture
    from tests.unit.sdk import test_fpm_phase_readiness as phases

    from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA
    from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics

    patch = pytest.MonkeyPatch()
    try:
        fixture = mixed_fixture.mixed_seed.__wrapped__(temporary, phases.phase_roots.__wrapped__(temporary, patch))
        config, _, original = fixture
        mixed = object.__new__(MixedOriginalA)
        mixed.scenario, mixed.manifests = "ac", {}
        mixed.provenance = json.loads(Path(original["manifest_path"]).read_text())["provenance"]
        mixed.prepared = SimpleNamespace(close=lambda: None)
        mixed.configure_request = lambda request: request

        def observed(kind, index, target):
            payload = mixed_fixture._iteration(kind, index=index, wall_ms=target)[0]
            return ForwardPassIteration.single_rank(
                ForwardPassMetric(
                    fpm_id=index,
                    configuration_id="mixed-fixture",
                    version=1,
                    worker_id="worker",
                    dp_rank=0,
                    counter_id=index,
                    wall_time_s=target / 1000,
                    scheduled=RequestMetrics(**payload["scheduled_requests"]),
                    queued=RequestMetrics(),
                    capture_id="fixture",
                )
            )

        training = [
            observed(kind, index, target) for kind, target in (("prefill", 2), ("decode", 3)) for index in range(20)
        ]
        feedback = [observed("mixed", 100 + index, 8) for index in range(71)]
        return config, mixed, training, {"contains_locally_mixed": feedback}, patch
    except BaseException:
        patch.undo()
        raise


def worker(args, helper):
    root = args.checkout.resolve()
    sys.path.insert(0, str(root))
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned
    from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
    from scripts.fpm_accuracy.scenario_contract import GBDT_ONLINE_RECIPE

    identity = helper.identity(root, args.native_sha256)
    identity["supplemental_harness_sha256"] = helper.sha(__file__)
    setup, stages, models = {}, {}, []
    snapshot_capabilities = []
    prediction_evidence = []
    checkpoint_sizes, snapshot_sizes = [], []
    training_payloads = []
    rss = {"start": helper.memory()}
    patch = predictor = None

    def measured(destination, name, function, count=1):
        start = None if args.dry_run else time.perf_counter_ns()
        value = function()
        destination[name] = {
            "operations": count,
            "elapsed_ns": None if start is None else time.perf_counter_ns() - start,
        }
        return value

    def stage(name, function, count=1):
        return measured(stages, name, function, count)

    def setup_stage(name, function):
        return measured(setup, name, function)

    with tempfile.TemporaryDirectory(prefix="gbdt-host-expanded-") as directory:
        directory = Path(directory)
        try:
            if args.case == "fallback":
                config, mixed, training, data, patch = setup_stage(
                    "real_original_a_fixture", lambda: mixed_setup(root, directory)
                )
                request = ForwardPassPerfModelConfig(**config)
                training_payloads = [row.tuning_payload() for row in training]

                def construct():
                    return LearnedScenarioPredictor(
                        "gbdt",
                        RustForwardPassPerfModel,
                        request,
                        fpm_learned,
                        training,
                        online=True,
                        mixed=mixed,
                    )

                predictor = stage("construction_including_offline_training", construct)
                queries = {kind: rows[-1].prediction_input() for kind, rows in data.items()}
                first = stage(
                    "first_prior_prediction", lambda: predictor.predict(next(iter(queries.values())))
                ).value_ms
                if abs(first - 4) > 1e-10:
                    raise AssertionError(f"Unexpected synthetic uncalibrated prior {first}")
                stage(
                    "warm_prior_predictions",
                    lambda: [predictor.predict(next(iter(queries.values()))) for _ in range(args.warm_queries)],
                    args.warm_queries,
                )
            else:
                stores = int(args.case[-1])
                data = setup_stage("generate_synthetic_inputs", lambda: observations(stores))
                queries = {kind: rows[-1].prediction_input() for kind, rows in data.items()}
                request = ForwardPassPerfModelConfig(
                    model="Qwen/Qwen3-8B",
                    system="h200_sxm",
                    worker_type="aggregated",
                    backend="vllm",
                    backend_version="0.25.1",
                    tp=2 if stores == 4 else 1,
                    attention_dp=2 if stores == 4 else 1,
                )
                if args.case.startswith("cold"):
                    predictor = stage(
                        "construction",
                        lambda: LearnedScenarioPredictor(
                            "gbdt", RustForwardPassPerfModel, request, fpm_learned, online=True, cold_start=True
                        ),
                    )
                    first = stage(
                        "first_cold_predictions",
                        lambda: [predictor.predict(query).value_ms for query in queries.values()],
                        len(queries),
                    )
                    if any(value is not None for value in first):
                        raise AssertionError("Cold store predicted before admission")
                else:
                    seed_rows = [row.tuning_payload() for rows in data.values() for row in rows[:20]]
                    training_payloads = seed_rows
                    checkpoint = directory / "seed"
                    controls = copy.deepcopy(GBDT_ONLINE_RECIPE["online"])
                    controls.pop("recent_sampling")
                    controls["adaptation"] = "online"
                    training_controls = {
                        key: GBDT_ONLINE_RECIPE["training"][key]
                        for key in (
                            "target",
                            "max_iter",
                            "max_leaf_nodes",
                            "learning_rate",
                            "min_samples_leaf",
                            "min_store_rows",
                            "random_state",
                            "early_stopping",
                        )
                    }
                    setup_stage(
                        "offline_seed_training",
                        lambda: fpm_learned.train(
                            seed_rows,
                            "aggregated",
                            identity=request.to_dict(),
                            features=fpm_learned.FEATURE_PRESETS["fpm5"],
                            online_checkpoint_dir=checkpoint,
                            **training_controls,
                        ),
                    )
                    config = request.to_dict()
                    config["estimation_mode"] = "fpm_learned"
                    config["estimator_config"]["correction"] = {"enabled": False}
                    config["estimator_config"]["fpm_learned"] = {**controls, "checkpoint_path": str(checkpoint)}
                    source_kind = "training_checkpoint"
                    native = stage("seeded_construction", lambda: RustForwardPassPerfModel.best_available(config))
                    models.append(native)
                    query_payloads = {kind: query.aic_payload() for kind, query in queries.items()}
                    first = stage(
                        "first_seeded_predictions",
                        lambda: [native.estimate_forward_pass_time_ms(query) for query in query_payloads.values()],
                        len(queries),
                    )
                    for value in first:
                        helper.valid_prediction(value)
            if predictor is not None:
                stage(
                    "collect_first_19_per_store",
                    lambda: [predictor.tune([rows[index]]) for index in range(19) for rows in data.values()],
                    19 * len(data),
                )
                rss["before_initial_fit"] = helper.memory()
                for kind, rows in data.items():
                    stage("initial_fit_20th_" + kind, lambda rows=rows: predictor.tune([rows[19]]))
                    helper.valid_prediction(predictor.predict(queries[kind]).value_ms)
                rss["after_initial_fit"] = helper.memory()
                stage(
                    "warm_wrapper_predictions",
                    lambda: [predictor.predict(query) for _ in range(args.warm_queries) for query in queries.values()],
                    args.warm_queries * len(queries),
                )
                if args.case == "fallback":
                    scope = next(iter(predictor._scopes.values()))
                    selected = [(scope.model, scope.config, list(queries.values()))]
                else:
                    selected = [
                        (scope.model, scope.config, [queries[key[1]]]) for key, scope in predictor._scopes.items()
                    ]
                stage(
                    "ordinary_updates_50_per_store",
                    lambda: [predictor.tune([rows[index]]) for index in range(20, 70) for rows in data.values()],
                    50 * len(data),
                )
                source_kind = "wrapper_owned"
            else:
                selected = [(native, config, list(queries.values()))]
                stage(
                    "ordinary_updates_50_per_store",
                    lambda: [
                        native.tune_with_fpms([rows[index].tuning_payload()])
                        for index in range(20, 70)
                        for rows in data.values()
                    ],
                    50 * len(data),
                )
            for index, (native, config, selected_queries) in enumerate(selected):
                payloads = [query.aic_payload() for query in selected_queries]
                expected = [helper.valid_prediction(native.estimate_forward_pass_time_ms(query)) for query in payloads]
                prediction_evidence.append(
                    {
                        "predictions_hex": [value.hex() for value in expected],
                        "forest_sha256": hashlib.sha256(
                            helper.encoded(native.export_learned_artifact()["stores"])
                        ).hexdigest(),
                    }
                )
                result = stage(
                    f"warm_sdk_predictions_scope_{index}",
                    lambda: [
                        native.estimate_forward_pass_time_ms(query)
                        for _ in range(args.warm_queries)
                        for query in payloads
                    ],
                    args.warm_queries * len(payloads),
                )
                if result != expected * args.warm_queries:
                    raise AssertionError("Inference changed without feedback")
                checkpoint = directory / f"checkpoint-{index}"
                stage(f"checkpoint_save_scope_{index}", lambda: native.save_online_checkpoint(checkpoint))
                checkpoint_sizes.append(helper.disk_bytes(checkpoint))
                checkpoint_config = helper.runtime_checkpoint_config(config, checkpoint)
                # A bare runtime checkpoint has no bundle prior. Reauthor
                # the exact existing original-A policy through the documented
                # compatibility input; never relabel its resolved source pin.
                fallback_pin = config["estimator_config"]["fpm_learned"].get("resolved_fallback")
                if fallback_pin is not None and fallback_pin.get("policy") is not None:
                    checkpoint_config["estimator_config"]["fpm_learned"].pop("fallback", None)
                    checkpoint_config["estimator_config"]["temporary_mixed_composition"] = copy.deepcopy(
                        fallback_pin["policy"]
                    )
                restored = stage(
                    f"checkpoint_restore_scope_{index}",
                    lambda: RustForwardPassPerfModel.best_available(checkpoint_config),
                )
                models.append(restored)
                if [restored.estimate_forward_pass_time_ms(query) for query in payloads] != expected:
                    raise AssertionError("Checkpoint changed prediction bits")
                snapshot = directory / f"snapshot-{index}"
                if args.case == "fallback":
                    # The baseline explicitly lacks portable mixed snapshots.
                    # Probe the capability, without comparing unequal work.
                    try:
                        started = None if args.dry_run else time.perf_counter_ns()
                        native.save_state_snapshot(snapshot)
                    except ValueError as error:
                        if "temporary_mixed_composition" not in str(error) or "portable state snapshots" not in str(
                            error
                        ):
                            raise
                        snapshot_capabilities.append({"supported": False, "reason": str(error)})
                    else:
                        saved_ns = None if started is None else time.perf_counter_ns() - started
                        started = None if args.dry_run else time.perf_counter_ns()
                        restored = RustForwardPassPerfModel.best_available(state_snapshot=snapshot)
                        restored_ns = None if started is None else time.perf_counter_ns() - started
                        models.append(restored)
                        if [restored.estimate_forward_pass_time_ms(query) for query in payloads] != expected:
                            raise AssertionError("Snapshot changed prediction bits")
                        snapshot_capabilities.append(
                            {"supported": True, "save_ns": saved_ns, "restore_ns": restored_ns}
                        )
                        snapshot_sizes.append(helper.disk_bytes(snapshot))
                else:
                    stage(f"snapshot_save_scope_{index}", lambda: native.save_state_snapshot(snapshot))
                    restored = stage(
                        f"snapshot_restore_scope_{index}",
                        lambda: RustForwardPassPerfModel.best_available(state_snapshot=snapshot),
                    )
                    models.append(restored)
                    if [restored.estimate_forward_pass_time_ms(query) for query in payloads] != expected:
                        raise AssertionError("Snapshot changed prediction bits")
                    snapshot_capabilities.append({"supported": True})
                    snapshot_sizes.append(helper.disk_bytes(snapshot))
            state = [
                {"shape": helper.forest_shape(model), "online": model.learned_online_diagnostics()}
                for model, _, _ in selected
            ]
            rss["after_updates_and_restores"] = helper.memory()
            payloads = {kind: [row.tuning_payload() for row in rows] for kind, rows in data.items()}
            result = {
                "schema_version": 1,
                "complete": True,
                "dry_run": args.dry_run,
                "case": args.case,
                "trial": args.trial,
                "identity": identity,
                "source_kind": source_kind,
                "snapshot_capabilities": snapshot_capabilities,
                "checkpoint_sizes": checkpoint_sizes,
                "snapshot_sizes": snapshot_sizes,
                "prediction_evidence": prediction_evidence,
                "checkpoint_prior_restore": "explicit_original_a_compatibility_policy"
                if args.case == "fallback"
                else None,
                "synthetic_input_sha256": hashlib.sha256(helper.encoded(payloads)).hexdigest(),
                "training_input_sha256": hashlib.sha256(helper.encoded(training_payloads)).hexdigest(),
                "query_input_sha256": hashlib.sha256(
                    helper.encoded({kind: query.aic_payload() for kind, query in queries.items()})
                ).hexdigest(),
                "work_contract": {
                    "stores": list(data),
                    "training_observations": len(training_payloads),
                    "native_model_count": len(selected),
                    "feedback_per_store": 70 if predictor is not None else 50,
                    "warm_queries_per_store": args.warm_queries,
                    "online_controls": GBDT_ONLINE_RECIPE["online"],
                },
                "setup": setup,
                "stages": stages,
                "rss": rss,
                "state": state,
                "accuracy_qualified": False,
                "scope": (
                    "Synthetic functional costs. Cold modes use the unchanged caller-managed Gym fit "
                    "per native workload. "
                    "Seeded modes use all stores in one SDK model. Fallback uses original-A pure seeds "
                    "plus one initially missing mixed store."
                ),
            }
        finally:
            for model in models:
                model.close()
            if predictor is not None:
                predictor.close()
            if patch is not None:
                patch.undo()
            gc.collect()
    final = helper.identity(root, args.native_sha256)
    for key in ("revision", "code_sha256", "native_sha256", "dependency_freeze_sha256"):
        if final[key] != identity[key]:
            raise ValueError(f"Build/environment changed during trial: {key}")
    helper.write(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--shared-harness", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-queries", type=int, default=1000)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    helper = load_helper(args.shared_harness)
    if args.warm_queries < 1 or any(os.environ.get(name) != "1" for name in helper.THREAD_ENV):
        parser.error("Positive warm queries and all four BLAS/OpenMP thread controls set to 1 are required")
    if args.output.exists():
        parser.error("Output must not exist")
    worker(args, helper)


if __name__ == "__main__":
    main()
