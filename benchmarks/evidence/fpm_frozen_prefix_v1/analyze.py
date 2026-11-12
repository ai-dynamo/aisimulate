# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate complete paired outputs, then export compact measurements and summaries."""

import argparse
import json
import math
import statistics
from pathlib import Path

from controller import digest, validate_results


def percentile(values, quantile):
    """Linear interpolation between adjacent sorted observations."""
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def retained(diagnostics):
    return {
        key: store["training_state_bytes"]
        for key, store in diagnostics["stores"].items()
    }


def semantic(value):
    if isinstance(value, dict):
        return {
            key: semantic(child)
            for key, child in value.items()
            if key not in ("history_grouping_ms", "training_state_bytes")
        }
    if isinstance(value, list):
        return [semantic(child) for child in value]
    return value


def main(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    gates = json.loads((root / "equality-gates.json").read_text())
    assert gates["status"] == "pass"
    assert gates["sources_binaries_and_inputs_unchanged_after_run"]
    rechecked = validate_results(root, manifest)
    for key, value in rechecked.items():
        assert value == gates[key] or (
            key == "equality_keys" and list(value) == gates[key]
        ), key
    for name in ("controller.py", "prepare_fixture.py"):
        key = (
            "controller_sha256"
            if name == "controller.py"
            else "fixture_preparation_script_sha256"
        )
        assert digest(Path(__file__).parent / name) == manifest[key]
    assert (
        digest(Path(__file__).parent / "probe.rs")
        == manifest["sources"]["baseline"]["probe_rs_sha256"]
    )
    assert (
        digest(Path(__file__).parent.parent / "fpm_online_publication_v1/benchmark.py")
        == manifest["fixture_preparation"]["generator_sha256"]
    )
    records, samples, initial_diagnostics = {}, [], {}
    for path in sorted((root / "results").glob("*.json")):
        case, phase, round_, arm = path.stem.split("__")
        record = json.loads(path.read_text())
        first = semantic(record["initial_diagnostics"])
        if case in initial_diagnostics:
            assert first == initial_diagnostics[case]
        else:
            initial_diagnostics[case] = first
        assert semantic(record["final_diagnostics"]) == record["semantic_diagnostics"]
        state = record["final_diagnostics"]["stores"]["pure_decode"]
        assert state["anchor_rows"] == manifest["fixtures"][case]["history_rows"]
        assert state["accepted_observations"] == manifest["updates_per_process"]
        assert state["publications"] == manifest["updates_per_process"]
        assert state["eager_updates"] == manifest["updates_per_process"]
        assert state["pending_rows"] == state["update_failures"] == 0
        records[(case, phase, int(round_[1:]), arm)] = record
        samples.append(
            {
                "file": path.name,
                "sha256": digest(path),
                "setup_ns": record["setup_ns"],
                "durations_ns": record["durations_ns"],
                "allocations": record["allocations"],
                "before_memory": record["before_memory"],
                "after_memory": record["after_memory"],
                "initial_retained_training_bytes": retained(
                    record["initial_diagnostics"]
                ),
                "final_retained_training_bytes": retained(record["final_diagnostics"]),
            }
        )
    summaries = {}
    for case in manifest["fixtures"]:
        trees, rows, _ = case.split("_")
        assert manifest["fixtures"][case]["trees"] == int(trees.removesuffix("trees"))
        assert manifest["fixtures"][case]["history_rows"] == int(
            rows.removesuffix("rows")
        )
        timing, allocation = {}, {}
        for arm in ("baseline", "candidate"):
            timed = [
                records[(case, "timing", index, arm)]
                for index in range(manifest["rounds"])
            ]
            counted = [
                records[(case, "allocation", index, arm)]
                for index in range(manifest["rounds"])
            ]
            timing[arm] = {
                "process_median_update_ns": distribution(
                    [statistics.median(r["durations_ns"]) for r in timed]
                ),
                "process_p95_update_ns": distribution(
                    [percentile(r["durations_ns"], 0.95) for r in timed]
                ),
                "setup_ns": distribution([r["setup_ns"] for r in timed]),
                "rss_kib_before_updates": distribution(
                    [r["before_memory"]["rss_kib"] for r in timed]
                ),
                "rss_kib_after_updates": distribution(
                    [r["after_memory"]["rss_kib"] for r in timed]
                ),
                "process_peak_rss_kib_before_updates": distribution(
                    [r["before_memory"]["peak_rss_kib"] for r in timed]
                ),
                "process_peak_rss_kib_after_updates": distribution(
                    [r["after_memory"]["peak_rss_kib"] for r in timed]
                ),
                "initial_retained_training_bytes": retained(
                    timed[0]["initial_diagnostics"]
                ),
                "final_retained_training_bytes": retained(
                    timed[0]["final_diagnostics"]
                ),
            }
            allocation[arm] = {}
            for key in (
                "requested_bytes",
                "allocations",
                "reallocations",
                "deallocations",
                "peak_additional_live_requested_bytes",
                "net_live_requested_bytes",
            ):
                for statistic, function in (
                    ("median", statistics.median),
                    ("mean", statistics.mean),
                    ("max", max),
                ):
                    allocation[arm][f"process_{statistic}_{key}"] = distribution(
                        [function([x[key] for x in r["allocations"]]) for r in counted]
                    )
        paired = []
        for index in range(manifest["rounds"]):
            baseline = statistics.median(
                records[(case, "timing", index, "baseline")]["durations_ns"]
            )
            candidate = statistics.median(
                records[(case, "timing", index, "candidate")]["durations_ns"]
            )
            paired.append(candidate / baseline)
        summaries[case] = {
            "timing": timing,
            "allocation": allocation,
            "paired_candidate_over_baseline_process_median": distribution(paired),
        }
    output.mkdir(exist_ok=True, parents=True)
    (output / "samples.json").write_text(
        json.dumps({"schema_version": 1, "processes": samples}, separators=(",", ":"))
        + "\n"
    )
    result = {
        "schema_version": 1,
        "manifest": manifest,
        "equality_gates": gates,
        "initial_semantic_diagnostics_equal_per_scenario": True,
        "all_64_updates_accepted_published_eager_without_failures": True,
        "fixture_generator_sha256_verified_against_repository": True,
        "statistical_unit": (
            "fresh process, paired by round; 64 updates are repeated measurements "
            "within one process"
        ),
        "percentile_method": "linear interpolation between sorted adjacent values",
        "summaries": summaries,
        "raw_samples_sha256": digest(output / "samples.json"),
        "analyzer_sha256": digest(__file__),
        "execution_log_sha256": {
            path.name: digest(path) for path in sorted(root.glob("*.log"))
        },
    }
    (output / "measurements.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "pass", "processes": len(samples)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.root.resolve(), arguments.output.resolve())
