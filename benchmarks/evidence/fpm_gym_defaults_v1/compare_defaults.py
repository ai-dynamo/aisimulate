# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native API defaults with the local Gym's unchanged scenario scorer.

The Gym owns split selection, prediction-before-feedback, five-second admissions,
cold bootstrap, MAPE, and coverage. This adapter changes only configuration
construction and replaces training with verified, immutable existing seed stores.
All numerical work uses RustForwardPassPerfModel.best_available(). Each worker
loads one explicitly selected native library in a fresh Python process.

Example (paths are supplied by the caller, never inferred from an installed SDK)::

    python compare_defaults.py worker --gym-source GYM_CHECKOUT --sdk-source SRC \
      --prepared-root SWEEP_EVIDENCE --seed-root GYM_SEED_CHECKPOINTS \
      --native BASELINE_SO --revision 0c53fab4 --variant before \
      --mode online --cell glm_bc_50 --output before-online-glm_bc_50.json

Run each requested (revision, mode, cell) separately, then use ``summarize`` on
the resulting JSON files. No new serving measurements, offline retraining, or
network access are performed. Online fitting runs normally in the native model.
Only trusted, hash-verified local prepared-case pickles are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import statistics
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def matrix_hash(features, wall, source):
    """Bind all fitted feature values, targets, and ordered row provenance."""
    import numpy as np

    result = hashlib.sha256()
    for values in (features, wall):
        array = np.asarray(values, dtype="<f8", order="C")
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    result.update(json.dumps(source, separators=(",", ":")).encode())
    return result.hexdigest()


class ImmutableSeeds:
    """Resolve exact fitted rows to preserved PR460 forests without training."""

    def __init__(self, trainer, root, directories, recipe):
        import numpy as np
        import pyarrow.parquet as pq

        self.trainer, self.root = trainer, root
        self.recipe = recipe
        self.index, self.receipts = {}, {}
        for name in sorted(directories):
            directory = root / name
            manifest_path = directory / "manifest.json"
            if file_hash(manifest_path) != name:
                raise ValueError(f"Seed manifest checksum mismatch: {name}")
            manifest = json.loads(manifest_path.read_text())
            artifact_path = directory / manifest["artifact_file"]
            if file_hash(artifact_path) != manifest["artifact_sha256"]:
                raise ValueError(f"Seed forest checksum mismatch: {name}")
            artifact = json.loads(artifact_path.read_text())
            parameters = artifact["metadata"]["hyperparameters"]
            if any(
                parameters[key] != value
                for key, value in {"max_iter": 150, "max_leaf_nodes": 15, "learning_rate": 0.2}.items()
            ):
                raise ValueError("Seed does not use the fixed PR460 forest recipe")
            for kind, store in manifest["stores"].items():
                rows = directory / store["rows_file"]
                if file_hash(rows) != store["rows_sha256"]:
                    raise ValueError(f"Seed rows checksum mismatch: {name}/{kind}")
                table = pq.read_table(rows)
                columns = [table[f"x{i}"].to_numpy() for i in range(len(manifest["features"]))]
                key = (
                    kind,
                    matrix_hash(
                        np.column_stack(columns), table["wall_time_ms"].to_numpy(), table["source_json"].to_pylist()
                    ),
                )
                entry = (directory, manifest, artifact)
                previous = self.index.get(key)
                if previous and previous[2]["stores"][kind] != artifact["stores"][kind]:
                    raise ValueError("Identical training rows have different preserved forests")
                self.index[key] = entry

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def train(self, iterations, worker_type, *, online_checkpoint_dir=None, identity=None, **parameters):
        provenance = {}
        features = parameters["features"]
        if list(features) != self.recipe["features"]:
            raise ValueError("Requested feature ABI differs from the preserved seed recipe")
        if any(
            value != self.recipe["training"].get(field) for field, value in parameters.items() if field != "features"
        ):
            raise ValueError("Requested training controls differ from the preserved seed recipe")
        dataset = self.trainer._build_dataset(iterations, worker_type, features, provenance=provenance)
        selected = {}
        for kind, (values, wall) in dataset.items():
            if len(wall) < parameters["min_store_rows"]:
                continue
            key = (kind, matrix_hash(values, wall, [source for _, source in provenance[kind]]))
            if key not in self.index:
                raise ValueError(f"No immutable seed matches all training rows for {kind} ({len(wall)} rows)")
            directory, manifest, artifact = self.index[key]
            if list(features) != manifest["features"] or artifact["worker_type"] != worker_type:
                raise ValueError("Seed feature or worker identity mismatch")
            if parameters["target"] != artifact["target"] or artifact["target"] != "log_ms":
                raise ValueError("Seed target transform differs from the requested training recipe")
            if (
                identity is not None
                and self.trainer._checkpoint_identity(identity, worker_type) != manifest["identity"]
            ):
                raise ValueError("Seed canonical identity mismatch")
            for field, value in parameters.items():
                expected = artifact["metadata"]["hyperparameters"].get(field)
                if field in artifact["metadata"]["hyperparameters"] and value != expected:
                    raise ValueError(f"Seed training parameter mismatch: {field}")
            selected[kind] = (directory, manifest, artifact)
            self.receipts[f"{directory.name}/{kind}"] = {
                "checkpoint_manifest_sha256": directory.name,
                "artifact_sha256": manifest["artifact_sha256"],
                "rows_sha256": manifest["stores"][kind]["rows_sha256"],
                "training_input_sha256": key[1],
                "training_rows": len(wall),
                "store": kind,
            }
        if not selected:
            raise ValueError("every store fell below min_store_rows; nothing to export")
        # A single-store online adapter may use one store of a worker checkpoint.
        # Preserve its complete tree objects and bin metadata; only the manifest
        # and artifact store inventory changes. No fitting or rounding occurs.
        first = next(iter(selected.values()))
        artifact = deepcopy(first[2])
        artifact["stores"] = {kind: entry[2]["stores"][kind] for kind, entry in selected.items()}
        for field in ("train_rows", "trees_fitted"):
            if field in artifact["metadata"]:
                artifact["metadata"][field] = {
                    kind: entry[2]["metadata"][field][kind] for kind, entry in selected.items()
                }
        if online_checkpoint_dir is not None:
            destination = Path(online_checkpoint_dir)
            original = len({entry[0] for entry in selected.values()}) == 1 and set(selected) == set(first[1]["stores"])
            if original:
                destination.symlink_to(first[0], target_is_directory=True)
                return deepcopy(first[2])
            destination.mkdir()
            manifest = deepcopy(first[1])
            manifest["stores"] = {kind: entry[1]["stores"][kind] for kind, entry in selected.items()}
            (destination / "model.json").write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False))
            manifest["artifact_sha256"] = file_hash(destination / "model.json")
            for kind, (directory, _, _) in selected.items():
                filename = manifest["stores"][kind]["rows_file"]
                (destination / filename).symlink_to(directory / filename)
            (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        return artifact


class DefaultPredictor:
    """Transport Gym observations through the current public API only.

    Online GBDT uses the Gym's original worker/store partition. Selector and
    frozen hybrid retain one model per worker. Cold C is excluded because the
    public API intentionally has no mutable artifact-append API used by that
    Gym adapter. The full original scored streams are retained for A+C/B+C.
    """

    def __init__(self, context, trainer, configuration, mode, scenario, observations):
        from scripts.fpm_accuracy.scenario_contract import GBDT_ONLINE_RECIPE

        from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel
        from scripts.fpm_accuracy.models.aic_config import map_worker_config_to_aic

        if scenario == "c":
            raise ValueError("This comparison requires an offline seed (A+C or B+C); cold C is excluded")
        engine = map_worker_config_to_aic(context.worker, context.engine_config_overrides)
        self._request = ForwardPassPerfModelConfig.from_legacy_engine_config(engine, context.worker_role, {})
        self._perf_model = RustForwardPassPerfModel
        self.trainer, self.configuration = trainer, configuration
        self.mode, self.scenario = mode, scenario
        self.recipe = GBDT_ONLINE_RECIPE
        self.by_store = mode == "online"
        self.models, self.pending, self.stores = {}, {}, {}
        self.temporary = TemporaryDirectory(prefix="fpm-defaults-seeds-")
        grouped = defaultdict(list)
        for observation in observations:
            rows = observation.tuning_payload()
            grouped[self.key(rows)].append(rows)
        try:
            for key, rows in grouped.items():
                self.fit(key, rows)
        except Exception:
            self.close()
            raise

    def key(self, rows):
        worker = str(rows[0]["worker_id"])
        return (worker, self.trainer.classify_workload(rows, self._request.worker_type)) if self.by_store else worker

    def fit(self, key, rows):
        index = len(self.models)
        checkpoint = Path(self.temporary.name) / f"seed-{index}"
        settings = self.recipe["training"]
        parameters = {
            name: settings[name]
            for name in (
                "target",
                "max_iter",
                "learning_rate",
                "max_leaf_nodes",
                "min_samples_leaf",
                "min_store_rows",
                "random_state",
                "early_stopping",
            )
        }
        try:
            artifact = self.trainer.train(
                rows,
                self._request.worker_type,
                features=self.recipe["features"],
                online_checkpoint_dir=checkpoint if self.mode != "hybrid" else None,
                identity=self._request.to_dict(),
                **parameters,
            )
        except ValueError as exc:
            if str(exc) == "every store fell below min_store_rows; nothing to export":
                self.pending[key] = rows
                return
            raise
        if self.mode == "hybrid":
            source = Path(self.temporary.name) / f"seed-{index}.json"
            source.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False))
        else:
            source = checkpoint
        config = self.configuration(self, source)
        self.models[key] = self._perf_model.best_available(config)
        self.stores[key] = set(artifact["stores"])
        self.pending.pop(key, None)

    def predict(self, features):
        from scripts.fpm_accuracy.models.fpt_predictor import Prediction

        rows = features.aic_payload()
        if any("wall_time" in row for row in rows):
            raise ValueError("Prediction inputs contain a measured target")
        model = self.models.get(self.key(rows))
        return Prediction(None if model is None else model.estimate_forward_pass_time_ms(rows))

    def tune(self, observations):
        for observation in observations:
            rows = observation.tuning_payload()
            key = self.key(rows)
            model = self.models.get(key)
            kind = self.trainer.classify_workload(rows, self._request.worker_type)
            if model is not None:
                if kind in self.stores[key]:
                    model.tune_with_fpms([rows])
            elif self.mode == "online":
                pending = self.pending.setdefault(key, [])
                pending.append(rows)
                if len(pending) >= self.recipe["training"]["min_store_rows"]:
                    try:
                        self.fit(key, pending)
                    except Exception:
                        pending.pop()
                        raise

    def diagnostics(self):
        return {
            "initialized_models": len(self.models),
            "pending_rows": sum(map(len, self.pending.values())),
            "models": [
                {
                    "online": model.learned_online_diagnostics(),
                    "selector": model.learned_selector_diagnostics(),
                    "forest_sha256": digest(model.export_learned_artifact()) if self.mode != "selector" else None,
                }
                for model in self.models.values()
            ],
        }

    def close(self):
        for model in self.models.values():
            model.close()
        self.models.clear()
        self.temporary.cleanup()


def worker(args):
    started = time.monotonic()
    bound_paths = [Path(__file__), args.native, args.prepared_root / "code/runner.py"]
    bound_paths += list((args.gym_source / "scripts/fpm_accuracy").rglob("*.py"))
    bound_paths += [
        args.sdk_source / "aisimulate_core/sdk" / name for name in ("rust_engine_step.py", "fpm_learned.py")
    ]
    startup_hashes = {path: file_hash(path) for path in bound_paths}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path[:0] = [str(args.gym_source), str(args.sdk_source), str(args.prepared_root / "code")]
    specification = importlib.util.spec_from_file_location("aisimulate._runtime", args.native)
    native = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = native
    specification.loader.exec_module(native)

    from runner import Recorder
    from scripts.fpm_accuracy.scenarios import score_scenario, split_observations

    from scripts.fpm_accuracy.models import scenario_predictors

    manifest_path = args.prepared_root / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cell = manifest["cells"][args.cell]
    case_info = manifest["cases"][cell["case"]]
    case_path, bootstrap_path = (args.prepared_root / case_info[name] for name in ("path", "bootstrap_path"))
    for path, expected in ((case_path, case_info["sha256"]), (bootstrap_path, case_info["bootstrap_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("Prepared Gym input checksum mismatch")
    with case_path.open("rb") as stream:
        case = pickle.load(stream)
    with bootstrap_path.open("rb") as stream:
        bootstrap = pickle.load(stream)
    record_key = digest([case.configuration_id, cell["scenario"], cell["train_percent"]])
    source_record = json.loads((args.seed_root.parent / "checkpoints" / f"{record_key}.json").read_text())
    if source_record["cell"]["scoring_population_sha256"] != cell["scoring_population_sha256"]:
        raise ValueError("Saved Gym seed belongs to a different scoring population")
    seed_directories = [item["sha256"] for item in source_record["seed_checkpoints"]]
    trainer = ImmutableSeeds(
        scenario_predictors._import_gbdt_trainer(),
        args.seed_root,
        seed_directories,
        source_record["cell"]["online_model"]["recipe"],
    )
    scenario_predictors._import_gbdt_trainer = lambda: trainer
    configurations = {}

    def configuration(predictor, source):
        learned = {"artifact_path" if args.mode == "hybrid" else "checkpoint_path": str(source)}
        authored = {"fpm_learned": learned}
        if args.mode == "selector":
            learned["adaptation"] = "causal_selector"
        elif args.mode == "online" and args.variant == "before":
            learned["adaptation"] = "online"
            authored["correction"] = {"enabled": False}
        elif args.mode == "hybrid" and args.variant == "after":
            learned["adaptation"] = "frozen"
        request = replace(
            predictor._request, estimation_mode="fpm_learned", fallback_policy="deny", estimator_config=authored
        )
        result = predictor._perf_model.normalize_config(request)
        normalized = deepcopy(result["estimator_config"])
        for key in ("artifact_path", "checkpoint_path"):
            if normalized["fpm_learned"].get(key):
                normalized["fpm_learned"][key] = "<immutable-seed>"
        configurations[digest(normalized)] = normalized
        return result

    testing = split_observations(case.observations, cell["scenario"], train_percent=cell["train_percent"])[1]
    recorder = Recorder(testing, traced=False, check_native=False)

    def factory(method, context, **kwargs):
        return recorder.wrap(
            DefaultPredictor(
                context,
                trainer,
                configuration,
                args.mode,
                cell["scenario"],
                kwargs["training_iterations"],
            )
        )

    result = score_scenario(
        case,
        "gbdt" if args.mode == "online" else args.mode,
        cell["scenario"],
        train_percent=cell["train_percent"],
        factory=factory,
        bootstrap=lambda *_: bootstrap,
    )
    for index, workload in enumerate(("all", "prefill", "decode", "mixed")):
        result["metrics"][workload]["ape_sum_pct"] = recorder.selected_sums[index]
    # The scorer attaches the historical configured recipe; replace that label
    # because this run intentionally exercises the runtime's omitted defaults.
    result["native_diagnostics"] = result.pop("online_model", {}).get("diagnostics")
    result.pop("cost", None)  # Accuracy evidence, not a host performance benchmark.
    result["artifact"] = (
        None
        if result["artifact"] is None
        else {key: value for key, value in result["artifact"].items() if key not in {"path", "metadata_path"}}
    )
    result.update(
        {
            "schema": "fpm-gym-defaults-comparison-v1",
            "cell_id": args.cell,
            "mode": args.mode,
            "variant": args.variant,
            "replica": args.replica,
            "revision": args.revision,
            "native_sha256": file_hash(args.native),
            "native_loaded_from_requested_file": Path(native.__file__).resolve() == args.native.resolve(),
            "runner_sha256": file_hash(Path(__file__)),
            "prepared_manifest_sha256": file_hash(manifest_path),
            "case_sha256": case_info["sha256"],
            "bootstrap_sha256": case_info["bootstrap_sha256"],
            "gym_source_sha256": {
                str(path.relative_to(args.gym_source)): file_hash(path)
                for path in sorted((args.gym_source / "scripts/fpm_accuracy").rglob("*.py"))
            },
            "recorder_source_sha256": file_hash(args.prepared_root / "code/runner.py"),
            "sdk_source_sha256": {
                name: file_hash(args.sdk_source / "aisimulate_core/sdk" / name)
                for name in ("rust_engine_step.py", "fpm_learned.py")
            },
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow", "scikit-learn")},
            "python_version": sys.version.split()[0],
            "seed_source_record_sha256": file_hash(args.seed_root.parent / "checkpoints" / f"{record_key}.json"),
            "normalized_estimator_configurations": list(configurations.values()),
            "seed_receipts": sorted(
                trainer.receipts.values(), key=lambda item: (item["checkpoint_manifest_sha256"], item["store"])
            ),
            "coverage_sha256": recorder.coverage.hexdigest(),
            "scored_rows_sha256": recorder.successful.hexdigest(),
            "admissions_sha256": recorder.admissions.hexdigest(),
            "admission_count": recorder.admission_count,
            "prediction_error_count": recorder.prediction_errors,
            "tuning_error_count": recorder.tuning_errors,
            "failures": recorder.failures[:20],
            "scenario": cell["scenario"],
            "train_percent": cell["train_percent"],
            "observations_processed": recorder.index,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    if result["scoring_population_sha256"] != cell["scoring_population_sha256"]:
        raise ValueError("Scoring population differs from the immutable prepared manifest")
    if any(file_hash(path) != expected for path, expected in startup_hashes.items()):
        raise ValueError("A bound runner, native binary, or imported source changed during execution")
    save(args.output, result)
    print(
        json.dumps(
            {
                "cell": args.cell,
                "mode": args.mode,
                "variant": args.variant,
                "status": result["status"],
                "metrics": result["metrics"]["all"],
                "elapsed_seconds": result["elapsed_seconds"],
                "failures": result["failures"],
            }
        ),
        flush=True,
    )
    if result["status"] != "evaluated" or result["failures"] or recorder.index != len(testing):
        raise SystemExit("Comparison cell did not execute completely without predictor errors")


def summarize(args):
    results = [json.loads(path.read_text()) for path in args.inputs]
    paired = defaultdict(dict)
    for result in results:
        if (
            result["status"] != "evaluated"
            or result["failures"]
            or result["observations_processed"] != result["test_count"]
            or result["prediction_error_count"]
            or result["tuning_error_count"]
            or result["metrics"]["all"]["error_count"]
            or result["metrics"]["all"]["tuning_error_count"]
        ):
            raise ValueError("Incomplete or failed result cannot enter the comparison")
        key = (result["mode"], result["cell_id"], result["replica"])
        if result["variant"] in paired[key]:
            raise ValueError(f"Duplicate result: {key}/{result['variant']}")
        paired[key][result["variant"]] = result
    rows = []
    for (mode, cell, replica), pair in sorted(paired.items()):
        before, after = pair["before"], pair["after"]
        for field in (
            "case_sha256",
            "bootstrap_sha256",
            "scoring_population_sha256",
            "training_population_sha256",
            "seed_receipts",
            "admissions_sha256",
            "coverage_sha256",
            "scored_rows_sha256",
        ):
            if before[field] != after[field]:
                raise ValueError(f"Paired inputs or schedule differ: {mode}/{cell}/{field}")
        for workload in ("all", "prefill", "decode", "mixed"):
            old, new = before["metrics"][workload], after["metrics"][workload]
            rows.append(
                {
                    "mode": mode,
                    "cell_id": cell,
                    "replica": replica,
                    "workload": workload,
                    "before": old,
                    "after": new,
                    "delta_percentage_points": new["mape_pct"] - old["mape_pct"]
                    if new["mape_pct"] is not None and old["mape_pct"] is not None
                    else None,
                }
            )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["cell_id"], row["workload"])].append(row)
    summaries = []
    for (mode, cell, workload), repeats in sorted(groups.items()):
        summary = {"mode": mode, "cell_id": cell, "workload": workload, "repeats": len(repeats)}
        for variant in ("before", "after"):
            metrics = [item[variant] for item in repeats]
            values = [item["mape_pct"] for item in metrics if item["mape_pct"] is not None]
            fields = (
                "measured_count",
                "predicted_count",
                "unavailable_count",
                "error_count",
                "tuning_error_count",
                "coverage_pct",
            )
            if any(any(item[field] != metrics[0][field] for field in fields) for item in metrics):
                raise ValueError("Replicates differ in coverage or error counts")
            summary[variant] = {
                **{field: metrics[0][field] for field in fields},
                "mape_mean_pct": statistics.mean(values) if values else None,
                "mape_min_pct": min(values) if values else None,
                "mape_max_pct": max(values) if values else None,
            }
        old, new = summary["before"]["mape_mean_pct"], summary["after"]["mape_mean_pct"]
        summary["delta_mean_percentage_points"] = new - old if old is not None and new is not None else None
        summaries.append(summary)
    save(
        args.output,
        {
            "schema": "fpm-gym-defaults-summary-v1",
            "rows": rows,
            "replicate_summaries": summaries,
            "aggregation": (
                "Gym cell MAPE = sum(100*abs(predicted-actual)/actual)/successful_count; "
                "all pools successful workload observations. No cross-cell pooling or weighting is applied."
            ),
            "input_sha256": {path.name: file_hash(path) for path in args.inputs},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("worker")
    for name in ("gym-source", "sdk-source", "prepared-root", "seed-root", "native", "output"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--revision", required=True)
    command.add_argument("--variant", choices=("before", "after"), required=True)
    command.add_argument("--mode", choices=("online", "selector", "hybrid"), required=True)
    command.add_argument("--cell", required=True)
    command.add_argument("--replica", type=int, default=0)
    command = subparsers.add_parser("summarize")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    {"worker": worker, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
