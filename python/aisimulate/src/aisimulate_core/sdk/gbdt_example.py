# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic inputs and exercise a seeded B-to-B+C GBDT workflow.

Installed entry point: ``python -m aisimulate_core.sdk.gbdt_example``.
These formula targets are functional examples, never hardware measurements.
No downloaded data, model weights, external service, or GPU is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

DATASET_ID = "aisimulate-self-authored-synthetic-fpm-v1"
IDENTITY = {
    "model": "aisimulate/synthetic-fpm-lifecycle-v1",
    "system": "synthetic-functional-only",
    "backend": "vllm",
    "backend_version": "0.0.0-synthetic",
    "worker_type": "decode",
    "tp": 1,
    "pp": 1,
    "attention_dp": 1,
}
OFFLINE_RECIPE = {
    "target": "log_ms",
    "max_iter": 12,
    "learning_rate": 0.1,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 4,
    "min_store_rows": 20,
    "random_state": 0,
    "early_stopping": False,
}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, value):
    path.write_bytes(_bytes(value) + b"\n")


def _iteration(index, split):
    """A new arithmetic surface, authored for this example; units are explicit."""
    batch = 1 + index % 8
    context = 64 * (1 + (index // 8) % 6)
    scheduled = {
        "num_decode_requests": batch,
        "sum_decode_kv_tokens": batch * context,
    }
    target_microseconds = 700 + 35 * batch + batch * context
    return [
        {
            "version": 1,
            "worker_id": f"synthetic-{split}",
            "dp_rank": 0,
            "counter_id": index,
            "wall_time": target_microseconds / 1_000_000,
            "scheduled_requests": scheduled,
        }
    ]


def generate_inputs(output):
    """Write deterministic UTF-8 JSONL and a content-addressed provenance manifest."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    splits = (("training", 0, 48), ("evaluation", 101, 16), ("c_feedback", 211, 40))
    files = {}
    for name, offset, count in splits:
        path = output / f"{name}.jsonl"
        with path.open("wb") as stream:
            for index in range(offset, offset + count):
                stream.write(_bytes(_iteration(index, name)) + b"\n")
        files[name] = {
            "path": path.name,
            "sha256": _sha(path),
            "byte_length": path.stat().st_size,
            "iterations": count,
            "stores": {"pure_decode": count},
        }
    manifest = {
        "kind": "aisimulate_synthetic_example_inputs",
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "origin": "self_authored_arithmetic_formulas",
        "license": "Apache-2.0",
        "qualification": "functional_only_synthetic",
        "hardware_measured": False,
        "identity": IDENTITY,
        "files": files,
        "formula_units": "integer microseconds converted to wall_time seconds",
        "decode_formula": "700 + 35 * batch + batch * context",
        "generator": {"path": "gbdt_example.py", "sha256": _sha(__file__)},
    }
    _write(output / "inputs.json", manifest)
    return manifest


def _load(inputs, manifest, split):
    entry = manifest["files"][split]
    path = inputs / entry["path"]
    _require(path.name == entry["path"], "Synthetic input references must be plain filenames")
    _require(_sha(path) == entry["sha256"], f"Synthetic input changed: {path}")
    _require(path.stat().st_size == entry["byte_length"], f"Synthetic input length changed: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == entry["iterations"], f"Synthetic input iteration count changed: {path}")
    counts = {"pure_decode": 0}
    for row in rows:
        _require(isinstance(row, list) and len(row) == 1, "Expected one complete single-rank iteration")
        scheduled = row[0]["scheduled_requests"]
        decode = scheduled.get("num_decode_requests", 0) > 0
        prefill = scheduled.get("num_prefill_requests", 0) > 0
        _require(decode and not prefill, "Expected a pure-decode synthetic iteration")
        counts["pure_decode"] += 1
    _require(counts == entry["stores"], f"Synthetic input store counts changed: {path}")
    return rows


def _request(*, source_policy="explicit_only", systems_paths=None):
    request = {
        **IDENTITY,
        "estimation_mode": "fpm_learned",
        "fallback_policy": "deny",
        "estimator_config": {
            "correction": {"enabled": False},
            "fpm_learned": {
                "adaptation": "online",
                "source_policy": source_policy,
                "fallback": "disabled",
            },
        },
    }
    if systems_paths is not None:
        request["systems_paths"] = [str(path) for path in systems_paths]
    return request


def _query(iteration):
    return [{key: value for key, value in rank.items() if key != "wall_time"} for rank in iteration]


def _predictions(model, rows):
    values = [model.estimate_forward_pass_time_ms(_query(row)) for row in rows]
    if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        raise AssertionError("Seeded example failed to predict a covered synthetic query")
    return [value.hex() for value in values]


def _ready(model):
    report = model.readiness_report()
    _require(report["execution"]["ready"], f"The decode example requires complete execution readiness: {report}")
    return report


def _same_predictions(left, right, rows):
    if _predictions(left, rows) != _predictions(right, rows):
        raise AssertionError("Snapshot continuation changed predicted floating-point bits")


def run_example(output, input_dir=None):
    """Exercise installed public APIs; fail if an advertised behavior is missing."""
    import aisimulate._runtime as runtime
    from aisimulate_core.sdk import fpm_learned
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    inputs = Path(input_dir).resolve(strict=True) if input_dir is not None else output / "inputs"
    manifest = json.loads((inputs / "inputs.json").read_text()) if input_dir is not None else generate_inputs(inputs)
    _require(manifest["identity"] == IDENTITY, "Input identity differs from this synthetic decode example")
    _require(manifest["hardware_measured"] is False, "This example accepts its synthetic functional inputs only")
    training = _load(inputs, manifest, "training")
    evaluation = _load(inputs, manifest, "evaluation")
    feedback = _load(inputs, manifest, "c_feedback")
    provenance = {
        "anchor_kind": "b",
        "sources": [{"reference": f"{DATASET_ID}/training.jsonl", "sha256": manifest["files"]["training"]["sha256"]}],
        "evaluation_sources": [
            {"reference": f"{DATASET_ID}/evaluation.jsonl", "sha256": manifest["files"]["evaluation"]["sha256"]}
        ],
    }
    bundle = output / "trained-bundle"
    seed_config = fpm_learned.train_bundle(
        training,
        "decode",
        output_dir=bundle,
        identity=IDENTITY,
        provenance=provenance,
        config=_request(),
        **OFFLINE_RECIPE,
    )
    # The producer also supports already accepted checkpoints, without retraining.
    exported = output / "exported-bundle"
    fpm_learned.export_bundle(
        bundle / "seed",
        exported,
        provenance=provenance,
        recipe_version="synthetic_example_offline_hgb_v1",
        config=_request(),
    )
    systems = output / "systems"
    installed = fpm_learned.install_bundle(exported / "bundle.json", systems, config=_request())
    with ExitStack() as stack:

        def prepare(config=None, **kwargs):
            model = RustForwardPassPerfModel.best_available(config, **kwargs)
            stack.callback(model.close)
            return model

        seed = prepare(seed_config)
        pinned = prepare(installed)
        discovered = prepare(_request(source_policy="auto", systems_paths=[systems]))
        seed_forest = seed.export_learned_artifact()
        expected = _predictions(seed, evaluation)
        _require(_predictions(pinned, evaluation) == expected, "Installed seed predictions changed")
        _require(_predictions(discovered, evaluation) == expected, "Discovered seed predictions changed")
        _require(seed.export_learned_artifact() == seed_forest, "Prediction mutated the seed forest")
        source = discovered.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_learned"]
        _require(
            source["source_sha256"] == installed["estimator_config"]["fpm_learned"]["source_sha256"],
            "Catalogue discovery selected a different bundle",
        )
        _write(output / "seeded-config.json", installed)

        # This application owns the chronological C input and its consumption cursor.
        # Native tuning receives only complete, already selected rank groups.
        live = seed
        _ready(live)
        consumed = 0
        causal_predictions = []
        for row in feedback[:20]:
            causal_predictions.extend(_predictions(live, [row]))
            live.tune_with_fpms([row])
            consumed += 1
        live.flush_updates()
        _ready(live)
        _require(
            live.learned_online_diagnostics()["stores"]["pure_decode"]["accepted_observations"] == consumed,
            "The seeded model did not admit each selected C observation exactly once",
        )

        snapshot = output / "bc-snapshot"
        saved = live.save_state_snapshot(snapshot)
        feedback_state = {
            "kind": "example_application_feedback_cursor",
            "schema_version": 1,
            "feedback_sha256": manifest["files"]["c_feedback"]["sha256"],
            "next_iteration": consumed,
            "snapshot_sha256": saved["estimator_config"]["state_snapshot"]["sha256"],
        }
        _write(output / "feedback-state.json", feedback_state)
        moved = output / "relocated" / "bc-snapshot"
        moved.parent.mkdir()
        shutil.move(str(snapshot), moved)
        fixed = prepare(state_snapshot=moved)
        _ready(fixed)
        _same_predictions(live, fixed, evaluation)
        fixed_predictions = _predictions(fixed, evaluation)
        try:
            fixed.tune_with_fpms([feedback[consumed]])
        except ValueError as error:
            if "on_tune" not in str(error):
                raise
        else:
            raise AssertionError("Fixed snapshot accepted feedback")
        _require(_predictions(fixed, evaluation) == fixed_predictions, "Rejected feedback changed fixed state")

        # Snapshot state and the application's input cursor are separate artifacts.
        # Bind them before resuming; production applications need their own durable
        # commit/reconciliation protocol for these two records.
        cursor = json.loads((output / "feedback-state.json").read_text())
        continued_config = copy.deepcopy(fixed.diagnostics()["provenance"]["config"])
        pin = continued_config["estimator_config"]["state_snapshot"]
        _require(cursor["snapshot_sha256"] == pin["sha256"], "Application cursor belongs to another snapshot")
        _require(
            cursor["feedback_sha256"] == _sha(inputs / manifest["files"]["c_feedback"]["path"]),
            "Application feedback stream changed",
        )
        pin["on_tune"] = "continue"
        continued = prepare(continued_config)
        for row in feedback[cursor["next_iteration"] :]:
            for model in (live, continued):
                _predictions(model, [row])  # Predict before this observation's measured target is admitted.
                model.tune_with_fpms([row])
        for model in (live, continued):
            model.flush_updates()
            _ready(model)
        _same_predictions(live, continued, evaluation)
        _require(_predictions(fixed, evaluation) == fixed_predictions, "Fixed snapshot predictions changed")
        _require(
            continued.learned_online_diagnostics()["stores"]["pure_decode"]["accepted_observations"] == len(feedback),
            "Continued model lost or duplicated C observations",
        )
        final_config = continued.save_state_snapshot(output / "continued-bc-snapshot")
        _write(output / "continued-config.json", final_config)
        receipt = {
            "kind": "aisimulate_synthetic_lifecycle_example",
            "schema_version": 1,
            "qualification": "functional_only_synthetic",
            "hardware_measured": False,
            "dataset_manifest_sha256": _sha(inputs / "inputs.json"),
            "bundle_manifest_sha256": _sha(exported / "bundle.json"),
            "native": {"path": runtime.__file__, "sha256": _sha(runtime.__file__)},
            "python": sys.version,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("aisimulate", "scikit-learn", "numpy", "scipy", "pyarrow")
            },
            "sdk_origin": fpm_learned.__file__,
            "offline_recipe": OFFLINE_RECIPE,
            "feedback_cursor": feedback_state,
            "feedback_observations": len(feedback),
            "pre_admission_predictions": causal_predictions,
            "readiness": continued.readiness_report(),
            "checks": [
                "train_and_export_existing_checkpoint",
                "portable_install_and_catalogue_discovery",
                "seeded_decode_execution_ready",
                "prediction_does_not_fit",
                "seeded_b_receives_external_c_feedback",
                "application_cursor_bound_to_feedback_and_snapshot",
                "relocated_bc_snapshot_preserves_predictions",
                "fixed_snapshot_rejects_tuning",
                "ready_snapshot_matches_prediction_bits",
                "continued_feedback_matches_uninterrupted_bits",
            ],
            "coverage_exclusions": [
                "measured_hardware_accuracy",
                "serving_or_replay_resource_preflight",
                "mixed_and_cross_rank_stores",
                "original_a_prior",
                "causal_selector",
                "caller_managed_initial_training",
                "production_feedback_delivery_and_exactly_once_recovery",
                "performance_qualification",
            ],
        }
        _write(output / "receipt.json", receipt)
    return receipt


def verify_bundle(output, bundle_path=None):
    """Verify the packaged synthetic bundle through ordinary public source APIs.

    This action needs the native runtime, but performs no offline training.
    ``bundle_path`` permits the exact same check before wheel assembly.
    """
    import aisimulate._runtime as runtime
    import aisimulate_core
    from aisimulate_core.sdk import fpm_learned
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    if bundle_path is None:
        bundle_path = (
            Path(aisimulate_core.__file__).parent / "systems/fpm_learned/examples/synthetic-functional-v1/bundle.json"
        )
    bundle_path = Path(bundle_path).resolve(strict=True)
    source_digest = _sha(bundle_path)
    declaration = json.loads(bundle_path.read_text())
    inputs = bundle_path.parent / "inputs"
    manifest = json.loads((inputs / "inputs.json").read_text())
    if manifest["dataset_id"] != DATASET_ID or manifest["hardware_measured"] is not False:
        raise ValueError("Expected the self-authored synthetic example input manifest")
    _require(
        declaration["provenance"]["anchor_kind"] == "b" and declaration["provenance"]["c_admission_lineage"] is None,
        "The example bundle must contain a pristine synthetic B seed",
    )
    _require(manifest["identity"] == IDENTITY, "Packaged input identity differs from the synthetic example")
    _require(
        manifest["generator"]["sha256"] == _sha(__file__), "Packaged inputs were generated by different source bytes"
    )
    for split, field in (("training", "sources"), ("evaluation", "evaluation_sources")):
        expected = [{"reference": f"{DATASET_ID}/{split}.jsonl", "sha256": manifest["files"][split]["sha256"]}]
        _require(
            declaration["provenance"][field] == expected,
            f"Packaged bundle {split} lineage differs from packaged input bytes",
        )
    # Verify all published input bytes, even though evaluation alone is queried.
    for split in manifest["files"]:
        _load(inputs, manifest, split)
    evaluation = _load(inputs, manifest, "evaluation")
    selected = fpm_learned.inspect_bundle(bundle_path, config=_request())
    systems = output / "systems"
    installed = fpm_learned.install_bundle(bundle_path, systems, config=_request())
    with ExitStack() as stack:
        models = []
        for config in (selected, installed, _request(source_policy="auto", systems_paths=[systems])):
            model = RustForwardPassPerfModel.best_available(config)
            stack.callback(model.close)
            models.append(model)
        expected = _predictions(models[0], evaluation)
        _require(
            all(_predictions(model, evaluation) == expected for model in models[1:]),
            "Explicit, installed and discovered bundle predictions differ",
        )
        _require(
            all(set(model.export_learned_artifact()["stores"]) == {"pure_decode"} for model in models),
            "The example bundle must contain exactly the declared decode store",
        )
        readiness = [_ready(model) for model in models]
        _write(output / "resolved-config.json", models[-1].diagnostics()["provenance"]["config"])
        receipt = {
            "kind": "aisimulate_packaged_synthetic_bundle_verification",
            "schema_version": 1,
            "qualification": "functional_only_synthetic",
            "hardware_measured": False,
            "bundle_manifest_sha256": source_digest,
            "input_manifest_sha256": _sha(inputs / "inputs.json"),
            "evaluation_observations": len(evaluation),
            "readiness": readiness,
            "native": {"path": runtime.__file__, "sha256": _sha(runtime.__file__)},
            "sdk_origin": fpm_learned.__file__,
            "checks": [
                "all_input_hashes_lengths_and_counts",
                "training_and_evaluation_lineage",
                "generator_source_digest",
                "explicit_source",
                "private_install",
                "explicit_learned_catalogue_discovery",
                "identical_seed_prediction_bits",
            ],
            "offline_training_performed": False,
        }
    if _sha(bundle_path) != source_digest:
        raise AssertionError("Read/install operations mutated the packaged bundle")
    _write(output / "verification.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "run", "verify-bundle"))
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="A new directory; existing data is never replaced"
    )
    parser.add_argument("--bundle-path", type=Path, help="verify-bundle only: override the packaged synthetic manifest")
    parser.add_argument("--input-dir", type=Path, help="run only: externally managed directory from generate")
    args = parser.parse_args(argv)
    if args.bundle_path is not None and args.action != "verify-bundle":
        parser.error("--bundle-path belongs to verify-bundle")
    if args.input_dir is not None and args.action != "run":
        parser.error("--input-dir belongs to run")
    if args.action == "generate":
        result = generate_inputs(args.output_dir)
    elif args.action == "verify-bundle":
        result = verify_bundle(args.output_dir, args.bundle_path)
    else:
        result = run_example(args.output_dir, args.input_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
