# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual PR533 and candidate native readers in separate processes.

The fixture is produced by fpm_online_publication_v1/benchmark.py prepare.
Ordinary modes exercise normalization; regression and learned modes also use
best_available. Learned models predict, accept feedback, and export; adaptive
models also flush pending updates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def semantic_diagnostics(value):
    if isinstance(value, dict):
        return {
            key: semantic_diagnostics(item)
            for key, item in value.items()
            if key not in ("history_grouping_ms", "training_state_bytes")
        }
    if isinstance(value, list):
        return [semantic_diagnostics(item) for item in value]
    return value


def snapshot_files(path):
    return {
        str(item.relative_to(path)): digest_bytes(item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def runtime_capture(model, queries):
    artifact = json.loads(model.export_learned_artifact())
    return {
        "predictions": [model.estimate_forward_pass_time_ms(json.dumps(row)) for row in queries],
        "artifact_sha256": digest_json(artifact),
        "frozen_prefix_sha256": {kind: digest_json(store["trees"][:-1]) for kind, store in artifact["stores"].items()},
        "diagnostics": semantic_diagnostics(json.loads(model.learned_online_diagnostics())),
    }


def runtime_child(model_class, payload):
    config = copy.deepcopy(payload["config"])
    checkpoint = Path(payload["checkpoint"])
    if payload["action"] == "runtime_continue":
        config["estimator_config"]["fpm_learned"]["checkpoint_path"] = str(checkpoint)
    model = model_class.best_available(json.dumps(config))
    queries = payload["queries"]
    assert len(queries) == 256
    feedback = copy.deepcopy(queries)
    for row in feedback:
        for rank in row:
            rank["wall_time"] *= 1.2
    if payload["action"] == "runtime_create":
        initial = runtime_capture(model, queries)
        model.tune_with_fpms(json.dumps(feedback[:5]))
        saved = runtime_capture(model, queries)
        model.save_online_checkpoint(str(checkpoint))
        assert runtime_capture(model, queries) == saved, "saving changed live model state"
        assert initial["frozen_prefix_sha256"] == saved["frozen_prefix_sha256"]
    else:
        saved = runtime_capture(model, queries)
    state = saved["diagnostics"]["stores"]["pure_decode"]
    assert state["accepted_observations"] == 5 and state["publications"] == 1, state
    assert state["recent_rows"] == 4 and state["pending_rows"] == 1, state
    assert state["history_bin_grouping"] and state["mutable_tree_tail"] == 1, state
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    files_before = snapshot_files(checkpoint)
    model.tune_with_fpms(json.dumps(feedback[5:]))
    model.flush_updates()
    final = runtime_capture(model, queries)
    assert final["frozen_prefix_sha256"] == saved["frozen_prefix_sha256"]
    state = final["diagnostics"]["stores"]["pure_decode"]
    assert state["accepted_observations"] == 256 and state["pending_rows"] == 0, state
    assert state["recent_rows"] <= 7 and state["evicted_observations"] > 0, state
    assert state["update_failures"] == 0, state
    assert snapshot_files(checkpoint) == files_before, "loading or continuing mutated the saved snapshot"
    return {
        "snapshot_files": files_before,
        "checkpoint_schema": manifest["schema"],
        "checkpoint_schema_version": manifest["schema_version"],
        "loaded": saved,
        "final": final,
    }


def child(binary):
    spec = importlib.util.spec_from_file_location("_runtime", binary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    model_class = native.RustForwardPassPerfModel
    payload = json.load(sys.stdin)
    if payload.get("action", "").startswith("runtime_"):
        result = runtime_child(model_class, payload)
        result["binary_sha256"] = digest_bytes(Path(binary).read_bytes())
        print(json.dumps(result))
        return
    results = []
    for case in payload["cases"]:
        result = {"name": case["name"]}
        try:
            result["normalized"] = json.loads(model_class.normalize_config(json.dumps(case["config"])))
            if case["construct"]:
                model = model_class.best_available(json.dumps(result["normalized"]))
                result["constructed"] = True
                if case["learned"]:
                    queries = payload["queries"][:8]
                    initial = [model.estimate_forward_pass_time_ms(json.dumps(row)) for row in queries]
                    assert all(value is not None and value > 0 for value in initial)
                    feedback = copy.deepcopy(queries)
                    for row in feedback:
                        for rank in row:
                            rank["wall_time"] *= 1.2
                    model.tune_with_fpms(json.dumps(feedback))
                    if case["config"]["estimator_config"]["fpm_learned"]["adaptation"] != "frozen":
                        model.flush_updates()
                    result["predictions_sha256"] = digest_json(
                        initial + [model.estimate_forward_pass_time_ms(json.dumps(row)) for row in queries]
                    )
                    if not case["causal"]:
                        result["artifact_sha256"] = digest_json(json.loads(model.export_learned_artifact()))
            result["accepted"] = True
        except Exception as error:
            result.update(accepted=False, error=str(error))
        results.append(result)
    print(json.dumps({"binary_sha256": digest_bytes(Path(binary).read_bytes()), "cases": results}))


def run_child(binary, payload):
    completed = subprocess.run(
        [sys.executable, __file__, "--child", str(binary)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def query(binary, cases, queries):
    return run_child(binary, {"cases": cases, "queries": queries})


def compare_runtime_checkpoints(args, cases, queries):
    config = copy.deepcopy(next(case["config"] for case in cases if case["name"] == "checkpoint-online-omitted"))
    config["estimator_config"]["fpm_learned"].update(
        mutable_tree_tail=1,
        update_every=4,
        history_bin_grouping=True,
        recent_window=7,
        recent_max_age_observations=9,
    )
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="reader-comparison-", dir=args.runtime_dir))
    results = {}
    for origin, binary in (("pr533", args.baseline), ("pr534", args.candidate)):
        checkpoint = run_dir / f"{origin}-snapshot"
        payload = {"action": "runtime_create", "config": config, "queries": queries, "checkpoint": str(checkpoint)}
        created = run_child(binary, payload)
        continuations = {}
        for reader, reader_binary in (("pr533", args.baseline), ("pr534", args.candidate)):
            resumed = run_child(reader_binary, {**payload, "action": "runtime_continue"})
            for key in ("snapshot_files", "checkpoint_schema", "checkpoint_schema_version", "loaded", "final"):
                assert resumed[key] == created[key], (origin, reader, key)
            continuations[reader] = resumed["binary_sha256"]
        result = {key: value for key, value in created.items() if key not in ("loaded", "final")}
        for stage in ("loaded", "final"):
            result[stage] = {key: value for key, value in created[stage].items() if key != "predictions"}
            result[stage]["predictions_sha256"] = digest_json(created[stage]["predictions"])
            result[stage]["prediction_count"] = len(created[stage]["predictions"])
        result["checkpoint_path"] = str(checkpoint)
        result["matching_continuation_readers"] = continuations
        results[origin] = result
    assert results["pr533"]["loaded"] == results["pr534"]["loaded"]
    assert results["pr533"]["final"] == results["pr534"]["final"]
    return {
        "directions_passed": ["pr533_saved_to_pr534_resumed", "pr534_saved_to_pr533_resumed"],
        "comparison": (
            "Each origin saves after 5 labels (4 fitted, 1 pending), then continues uninterrupted. "
            "Both native binaries independently load that same snapshot, consume labels 6..256 and flush. "
            "All 256 predictions, exported artifacts and semantic diagnostics match the uninterrupted origin exactly. "
            "Only history_grouping_ms and capacity-based training_state_bytes are removed from diagnostics."
        ),
        "controls": config["estimator_config"]["fpm_learned"],
        "origins": results,
    }


def make_cases(fixture):
    manifest = json.loads((fixture / "seed/manifest.json").read_text())
    identity = manifest["identity"]
    defaults = {"alpha": 0.025, "warmup": 10, "margin": 0.025, "persistence": 1}
    cases = []

    def add(name, config, *, learned=False, causal=False):
        cases.append(
            {
                "name": name,
                "config": config,
                "construct": learned or config["estimation_mode"] == "fpm_regression",
                "learned": learned,
                "causal": causal,
            }
        )

    for mode in ("auto", "op_level", "fpm_interpolation", "fpm_regression"):
        for label, selector in (("omitted", None), ("empty", {}), ("default", defaults)):
            config = {**identity, "estimation_mode": mode, "estimator_config": {}}
            if selector is not None:
                config["estimator_config"]["selector"] = selector
            add(f"{mode}-{label}", config)
    for source, adaptation in (("artifact", "frozen"), ("checkpoint", "frozen"), ("checkpoint", "online")):
        for label, selector in (("omitted", None), ("empty", {}), ("default", defaults)):
            controls = {
                "adaptation": adaptation,
                f"{source}_path": str(fixture / ("seed/model.json" if source == "artifact" else "seed")),
                "update_mode": "leaf_only",
                "update_every": 3,
                "mutable_tree_tail": 10,
                "recent_max_age_observations": 5,
            }
            config = {
                **identity,
                "estimation_mode": "fpm_learned",
                "estimator_config": {"fpm_learned": controls, "correction": {"enabled": False}},
            }
            if selector is not None:
                config["estimator_config"]["selector"] = selector
            add(f"{source}-{adaptation}-{label}", config, learned=True)
    for label, selector in (
        ("default", defaults),
        ("alpha", {"alpha": 0.125}),
        ("warmup", {"warmup": 2}),
        ("margin", {"margin": 0.1}),
        ("persistence", {"persistence": 3}),
        ("all", {"alpha": 1.0, "warmup": 1, "margin": 0.0, "persistence": 2}),
    ):
        add(
            f"causal-{label}",
            {
                **identity,
                "estimation_mode": "fpm_learned",
                "estimator_config": {
                    "fpm_learned": {
                        "adaptation": "causal_selector",
                        "checkpoint_path": str(fixture / "seed"),
                        "update_mode": "leaf_only",
                        "update_every": 3,
                        "mutable_tree_tail": 10,
                    },
                    "correction": {"enabled": True, "mode": "hybrid"},
                    "selector": selector,
                },
            },
            learned=True,
            causal=True,
        )
    return cases


def compact(result):
    answer = copy.deepcopy(result)
    for case in answer["cases"]:
        if "normalized" in case:
            case["normalized_sha256"] = digest_json(case.pop("normalized"))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--baseline-validation", type=Path)
    parser.add_argument("--candidate-revision")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    required = ("candidate", "baseline", "fixture", "baseline_validation", "candidate_revision", "out")
    if any(getattr(args, name) is None for name in required):
        parser.error("all non-child arguments are required")
    if args.runtime_dir is None:
        args.runtime_dir = args.out.parent / "runtime-checkpoints"
    validation = json.loads(args.baseline_validation.read_text())
    assert digest_bytes(args.baseline.read_bytes()) == validation["native_module_sha256"]
    fixture = args.fixture.resolve()
    cases = make_cases(fixture)
    queries = json.loads((fixture / "queries.json").read_text())
    candidate = query(args.candidate, cases, queries)
    assert all(case["accepted"] for case in candidate["cases"]), candidate
    serialized = [
        {**case, "config": result["normalized"]} for case, result in zip(cases, candidate["cases"], strict=True)
    ]
    reloaded = query(args.candidate, serialized, queries)
    assert candidate == reloaded, "candidate serialization changed normalized config or model behavior"
    baseline = query(args.baseline, serialized, queries)
    for case, current, previous in zip(cases, candidate["cases"], baseline["cases"], strict=True):
        controls = current["normalized"]["estimator_config"]
        custom_selector = case["causal"] and case["name"] != "causal-default"
        assert ("selector" in controls) == custom_selector, case["name"]
        if not case["learned"]:
            assert "fpm_learned" not in controls, case["name"]
        if case["causal"]:
            if custom_selector:
                for key, value in case["config"]["estimator_config"]["selector"].items():
                    assert controls["selector"][key] == value, case["name"]
            assert not previous["accepted"], previous
            assert (
                "unknown variant `causal_selector`" in previous["error"]
                or "unknown field `selector`" in previous["error"]
            ), previous
        else:
            assert previous == current, (case["name"], previous, current)
    runtime = compare_runtime_checkpoints(args, cases, queries)
    receipt = {
        "schema": "pr534-native-reader-compatibility-v1",
        "summary": {
            "prior_reader_accepted_default_selector_configurations": 21,
            "candidate_causal_configurations_rejected_by_prior_reader": 6,
            "candidate_roundtrips": len(cases),
            "standalone_learned_behavior_matches_prior_reader": 9,
            "runtime_checkpoint_directions_passed": len(runtime["directions_passed"]),
        },
        "baseline_implementation_revision": validation["implementation_commit"],
        "baseline_validation_sha256": digest_bytes(args.baseline_validation.read_bytes()),
        "candidate_revision": args.candidate_revision,
        "script_sha256": digest_bytes(Path(__file__).read_bytes()),
        "fixture_files": {
            str(path.relative_to(fixture)): digest_bytes(path.read_bytes())
            for path in sorted(fixture.rglob("*"))
            if path.is_file()
        },
        "candidate": compact(candidate),
        "baseline": compact(baseline),
        "roundtrip": compact(reloaded),
        "runtime_checkpoint_continuation": runtime,
        "scope": (
            "Actual native readers and constructors in separate processes; ordinary modes normalize, "
            "regression also constructs, learned models predict before/after 8 labels; adaptive models flush. "
            "Default standalone learned models also export identical artifacts."
        ),
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
