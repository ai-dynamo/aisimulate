# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Produce real older-version artifacts, then verify them with a newer package.

Uses unchanged test fixtures from the explicitly supplied AISimulate checkout.
No model or transport mocks. Artifact directories must be disposable test data.
"""

import argparse
import copy
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import traceback
from contextlib import closing
from pathlib import Path

from aisimulate import _runtime
from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk import rust_engine_step


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {
        "python": sys.executable,
        "sdk": rust_engine_step.__file__,
        "native": _runtime.__file__,
        "native_sha256": digest(_runtime.__file__),
    }


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def produce(args, fixtures):
    root = args.artifacts
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for kind in ("error_threshold", "selected", "fpm5", "learned_frozen", "online", "causal_selector"):
        directory = root / kind
        directory.mkdir()
        build, worker, warm, following = fixtures._kind_case(directory, kind)
        model = build()
        queries = fixtures._queries(worker)
        case = {
            "kind": kind,
            "config": model.diagnostics()["provenance"]["config"],
            "initial_bits": fixtures._bits(model, queries),
            "queries": queries,
            "continuation": [following[i : i + 7] for i in range(0, len(following), 7)],
        }
        model.tune_with_fpms(warm)
        case["warm_bits"] = fixtures._bits(model, queries)
        case["snapshot_config"] = model.save_state_snapshot(directory / "snapshot")
        manifest = json.loads((directory / "snapshot/manifest.json").read_text())
        case["snapshot_schema"] = manifest["schema_version"]
        case["snapshot_layout"] = manifest["state_layout_version"]
        # The old API intentionally rejects selector-only online checkpoints;
        # its full selector state is exercised through the snapshot below.
        if kind == "online":
            model.save_online_checkpoint(directory / "runtime")
            runtime = json.loads((directory / "runtime/manifest.json").read_text())
            case["runtime_schema"] = runtime["schema_version"]
            case["runtime_config"] = {
                "model": "structural-test",
                "system": "test-gpu",
                "backend": "vllm",
                "worker_type": "decode",
                "estimation_mode": "fpm_learned",
                "estimator_config": {
                    "correction": {"enabled": kind != "online"},
                    "fpm_learned": {"checkpoint_path": str(directory / "runtime"), "adaptation": kind},
                },
            }
        for chunk in case["continuation"]:
            model.tune_with_fpms(chunk)
        case["continued_bits"] = fixtures._bits(model, queries)
        model.close()
        cases.append(case)
    record = {
        "producer": environment(),
        "fixture_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.fixture_repo, text=True
        ).strip(),
        "fixture_sha256": digest(fixtures.__file__),
        "files": {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
        "cases": cases,
    }
    write(root / "cases.json", record)
    write(args.report, record)
    print(f"Produced {len(cases)} cases with {record['producer']['native_sha256']}")


def verify(args, fixtures):
    record = json.loads((args.artifacts / "cases.json").read_text())
    for relative, expected in record["files"].items():
        assert digest(args.artifacts / relative) == expected, relative
    results = []
    for case in record["cases"]:
        result = {"kind": case["kind"], "checks": []}
        try:
            with closing(RustForwardPassPerfModel.best_available(case["config"])) as raw:
                assert fixtures._bits(raw, case["queries"]) == case["initial_bits"]
            result["checks"].append("old_resolved_config_and_seed_predict_identically")
            if "runtime_config" in case:
                with closing(RustForwardPassPerfModel.best_available(case["runtime_config"])) as runtime:
                    assert fixtures._bits(runtime, case["queries"]) == case["warm_bits"]
                    for chunk in case["continuation"]:
                        runtime.tune_with_fpms(chunk)
                    assert fixtures._bits(runtime, case["queries"]) == case["continued_bits"]
                result["checks"].append("old_runtime_checkpoint_continues_identically")
            old = args.artifacts / case["kind"]
            moved = args.artifacts / "relocated" / case["kind"]
            moved.parent.mkdir(exist_ok=True)
            shutil.move(str(old / "snapshot"), moved)
            for source in (old / "checkpoint", old / "forest.json"):
                if source.is_dir():
                    shutil.rmtree(source)
                elif source.is_file():
                    source.unlink()
            with closing(RustForwardPassPerfModel.best_available(state_snapshot=moved)) as fixed:
                assert fixtures._bits(fixed, case["queries"]) == case["warm_bits"]
                try:
                    fixed.tune_with_fpms(case["continuation"][0])
                except ValueError as error:
                    assert "on_tune" in str(error), str(error)
                else:
                    raise AssertionError("fixed legacy snapshot accepted updates")
            result["checks"].append("old_snapshot_relocates_without_original_seed_and_restores_fixed")
            continued = copy.deepcopy(case["snapshot_config"])
            continued["estimator_config"]["state_snapshot"].update(path=str(moved), on_tune="continue")
            with closing(RustForwardPassPerfModel.best_available(continued)) as resumed:
                for chunk in case["continuation"]:
                    resumed.tune_with_fpms(chunk)
                assert fixtures._bits(resumed, case["queries"]) == case["continued_bits"]
            result["checks"].append("old_snapshot_continuation_matches_uninterrupted_old_model_bits")
            result["passed"] = True
        except Exception:
            result.update(passed=False, traceback=traceback.format_exc())
        results.append(result)
        print(json.dumps(result), flush=True)
    write(args.report, {"consumer": environment(), "producer": record["producer"], "cases": results})
    return 0 if all(result["passed"] for result in results) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("produce", "verify"))
    parser.add_argument("--fixture-repo", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    # Import the package first (above), then use fixtures from the pinned checkout.
    sys.path.insert(0, str(args.fixture_repo / "python/aisimulate"))
    fixtures = importlib.import_module("tests.unit.sdk.test_fpm_state_snapshot")
    assert Path(fixtures.__file__).is_relative_to(args.fixture_repo)
    if args.phase == "produce":
        produce(args, fixtures)
        return 0
    return verify(args, fixtures)


if __name__ == "__main__":
    raise SystemExit(main())
