# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read completed cost probes; emit compact measurements and verified comparisons.

Usage: python analyze.py /path/to/fpm-model-publication-v1 > measurements.json
This script never starts benchmarks or modifies its input directory. Large
prediction arrays are checked and compared before being replaced with hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

CRITICAL_SOURCES = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/core/Cargo.toml",
    "crates/core/src/perfmodel/fpm/config.rs",
    "crates/core/src/perfmodel/fpm/estimator.rs",
    "crates/core/src/perfmodel/fpm/model.rs",
    "crates/core/src/perfmodel/fpm/learned.rs",
    "crates/core/src/perfmodel/fpm/learned_online.rs",
    "crates/core/src/perfmodel/fpm/learned_online/checkpoint.rs",
    "crates/core/src/perfmodel/fpm/learned_online/fitting.rs",
    "crates/core/src/perfmodel/fpm/selector.rs",
    "crates/core/src/perfmodel/fpm/combined.rs",
    "crates/core/src/perfmodel/py.rs",
    "python/aisimulate/src/aisimulate_core/sdk/fpm_learned.py",
    "python/aisimulate/src/aisimulate_core/sdk/rust_engine_step.py",
)
METRICS = (
    "setup_ms",
    "cold_query_us",
    "warm_query_us_median",
    "update_us_median",
    "update_us_p95",
    "update_us_max",
    "process_peak_rss_kib",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def toolchain(root):
    result = {}
    for tool in ("rustc", "cargo"):
        path = Path.home() / ".cargo/bin" / tool
        probe = subprocess.run(
            [str(path), "--version"], text=True, capture_output=True, check=True
        )
        result[tool] = {"executable": str(path), "version": probe.stdout.strip()}
    python = root.parent / "fpm-beta-env/bin/python"
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import sys,json,importlib.metadata as m; "
            "print(json.dumps({'python':sys.version,'packages':"
            "{p:m.version(p) for p in ('numpy','scikit-learn','pyarrow')}}))",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result["benchmark_python"] = {"executable": str(python), **json.loads(probe.stdout)}
    return result


def source_snapshot(root):
    files = {}
    for directory, suffix in (("crates", ".rs"), ("python/aisimulate/src", ".py")):
        for path in sorted((root / directory).rglob(f"*{suffix}")):
            files[path.relative_to(root).as_posix()] = digest(path)
    for path in sorted(root.rglob("Cargo.toml")):
        files[path.relative_to(root).as_posix()] = digest(path)
    if (root / "Cargo.lock").exists():
        files["Cargo.lock"] = digest(root / "Cargo.lock")
    binaries = {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(
            (root / "python/aisimulate/src/aisimulate").glob("_runtime*.so")
        )
    }
    assert len(binaries) == 1, (root, binaries)
    return {
        "source_file_count": len(files),
        "source_inventory_sha256": canonical_digest(files),
        "source_inventory_rule": (
            "all crates/**/*.rs, python/aisimulate/src/**/*.py, **/Cargo.toml, Cargo.lock; "
            "sorted path-to-SHA256 JSON with compact separators"
        ),
        "critical_source_sha256": {
            key: files[key] for key in CRITICAL_SOURCES if key in files
        },
        "binary_sha256": binaries,
    }


def fixture(root):
    manifest = load(root / "seed/manifest.json")
    artifact = load(root / "seed" / manifest["artifact_file"])
    inputs = {
        "queries.json": digest(root / "queries.json"),
        "seed/manifest.json": digest(root / "seed/manifest.json"),
        "seed/" + manifest["artifact_file"]: digest(
            root / "seed" / manifest["artifact_file"]
        ),
    }
    assert inputs["seed/" + manifest["artifact_file"]] == manifest["artifact_sha256"]
    for store in manifest["stores"].values():
        name = "seed/" + store["rows_file"]
        inputs[name] = digest(root / name)
        assert inputs[name] == store["rows_sha256"]
    return {
        "input_sha256": inputs,
        "training_rows": sum(
            store["row_count"] for store in manifest["stores"].values()
        ),
        "query_count": len(load(root / "queries.json")),
        "feature_count": len(artifact["features"]),
        "features": artifact["features"],
        "identity": manifest["identity"],
        "sklearn_version": manifest.get("sklearn_version"),
        "target": artifact["target"],
        "trees_per_store": {
            kind: len(store["trees"]) for kind, store in artifact["stores"].items()
        },
        "max_leaf_nodes": {
            kind: store["max_leaf_nodes"] for kind, store in manifest["stores"].items()
        },
    }


def predictions(result, which):
    values = result[which + "_predictions_ms"]
    assert len(values) == 256
    assert all(
        isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values
    )
    actual = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    assert actual == result[which + "_prediction_sha256"]
    return values


def comparison(left, right):
    answer = {}
    for which in ("initial", "final"):
        a, b = predictions(left, which), predictions(right, which)
        differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
        ulps = [
            abs(
                struct.unpack(">Q", struct.pack(">d", x))[0]
                - struct.unpack(">Q", struct.pack(">d", y))[0]
            )
            for x, y in zip(a, b, strict=True)
        ]
        answer[which] = {
            "prediction_count": len(a),
            "exact_equal": a == b,
            "different_count": sum(x != y for x, y in zip(a, b, strict=True)),
            "max_abs_difference_ms": max(differences),
            "max_relative_difference": max(
                d / x for d, x in zip(differences, a, strict=True)
            ),
            "max_ulps": max(ulps),
            "left_sha256": left[which + "_prediction_sha256"],
            "right_sha256": right[which + "_prediction_sha256"],
        }
    return answer


def compact(result):
    result = json.loads(json.dumps(result))
    for which in ("initial", "final"):
        result.pop(which + "_predictions_ms")
    for store in (result.get("diagnostics") or {}).get("stores", {}).values():
        store.pop("retained_row_ids", None)
    return result


def median_metrics(records):
    answer = {
        name: statistics.median(record[name] for record in records) for name in METRICS
    }
    states = [
        record["diagnostics"]["training_state_bytes"]
        for record in records
        if record.get("diagnostics") and "training_state_bytes" in record["diagnostics"]
    ]
    answer["training_state_bytes"] = statistics.median(states) if states else None
    answer["process_count"] = len(records)
    return answer


def main(root):
    expected = {}
    for round_ in range(3):
        for source, arm in (
            ("base", "base"),
            ("online", "frozen"),
            ("online-before", "online"),
            ("online", "online"),
        ):
            expected[f"cost-results/{source}-{arm}-{round_}.json"] = (
                source,
                "fixture",
                f"initial/10000/{source}/{arm}",
                round_,
            )
        for source in ("online-before", "online"):
            expected[f"cost-results/small-{source}-{round_}.json"] = (
                source,
                "small-fixture",
                f"initial/256/{source}/online",
                round_,
            )
        for inputs, size in (("fixture", 10000), ("small-fixture", 256)):
            for source in ("online", "selector"):
                expected[f"selector-cost-results/{inputs}-{source}-{round_}.json"] = (
                    source,
                    inputs,
                    f"combined/{size}/{source}",
                    round_,
                )
    missing = [name for name in expected if not (root / name).exists()]
    missing += [
        name
        for name in (
            "cross-version-result.json",
            "forward-gate/comparison.json",
            "forward-gate/raw_results.json",
        )
        if not (root / name).exists()
    ]
    if missing:
        raise SystemExit("Benchmark campaign is incomplete: " + ", ".join(missing))

    sources = {
        arm: source_snapshot(root / arm)
        for arm in ("base", "online-before", "online", "selector")
    }
    fixtures = {name: fixture(root / name) for name in ("fixture", "small-fixture")}
    raw, measured, groups = {}, {}, defaultdict(list)
    for name, (source, inputs, group, round_) in expected.items():
        record = load(root / name)
        assert record["binary_sha256"] in sources[source]["binary_sha256"].values(), (
            name
        )
        assert (
            record["seed_manifest_sha256"]
            == fixtures[inputs]["input_sha256"]["seed/manifest.json"]
        ), name
        assert (
            record["queries_sha256"] == fixtures[inputs]["input_sha256"]["queries.json"]
        ), name
        for which in ("initial", "final"):
            predictions(record, which)
        for metric in METRICS:
            assert math.isfinite(record[metric]) and record[metric] > 0, (name, metric)
        if record["arm"] in ("online", "selector"):
            state = record["diagnostics"]["stores"]["pure_decode"]
            assert state["accepted_observations"] == state["publications"] == 256
            assert (
                state["recent_rows"] == 64
                and state["pending_rows"] == state["update_failures"] == 0
            )
        raw[name] = record
        measured[name] = {
            "raw_file_sha256": digest(root / name),
            "round": round_,
            **compact(record),
        }
        groups[group].append(record)

    reuse, frozen = {}, {}
    for round_ in range(3):
        for size, prefix in ((10000, ""), (256, "small-")):
            before = f"cost-results/{prefix}online-before{'-online' if not prefix else ''}-{round_}.json"
            after = f"cost-results/{prefix}online{'-online' if not prefix else ''}-{round_}.json"
            diff = comparison(raw[before], raw[after])
            assert all(item["exact_equal"] for item in diff.values()), (
                before,
                after,
                diff,
            )
            reuse[f"{size}/round-{round_}"] = diff
        frozen[f"round-{round_}"] = comparison(
            raw[f"cost-results/base-base-{round_}.json"],
            raw[f"cost-results/online-frozen-{round_}.json"],
        )

    cross = load(root / "cross-version-result.json")
    assert (
        cross["equal"] and cross["accepted"] == 7 and cross["prediction_count"] == 256
    )
    parent, child = (
        load(root / "cross-version-parent.json"),
        load(root / "cross-version-child.json"),
    )
    assert parent == child
    assert digest(root / "cross-version-parent.json") == cross["parent_output_sha256"]
    assert digest(root / "cross-version-child.json") == cross["child_output_sha256"]
    checkpoint_path = root / "cross-version-checkpoint"
    checkpoint_manifest = load(checkpoint_path / "manifest.json")
    checkpoint_inputs = {"manifest.json": digest(checkpoint_path / "manifest.json")}
    for field in ("artifact", "state"):
        name = checkpoint_manifest[field + "_file"]
        checkpoint_inputs[name] = digest(checkpoint_path / name)
        assert checkpoint_inputs[name] == checkpoint_manifest[field + "_sha256"]
    forward = load(root / "forward-gate/comparison.json")
    forward_raw = load(root / "forward-gate/raw_results.json")
    assert not forward["run_errors"] and forward["selected_case_count"] == 1
    assert all(len(point["rounds"]) == 5 for point in forward["points"])
    forward_compact = []
    for case in forward_raw["cases"]:
        rounds = []
        for record in case["rounds"]:
            row = {
                key: value
                for key, value in record.items()
                if key not in ("base", "head")
            }
            for arm in ("base", "head"):
                row[arm] = {
                    key: value
                    for key, value in record[arm].items()
                    if key not in ("case", "environment", "warm_samples_us")
                }
            rounds.append(row)
        forward_compact.append({"case": case["case"], "rounds": rounds})
    summaries = {
        name: median_metrics(records) for name, records in sorted(groups.items())
    }
    ratios = {}
    for size in (10000, 256):
        for label, before, after in (
            (
                "histogram_reuse",
                f"initial/{size}/online-before/online",
                f"initial/{size}/online/online",
            ),
            ("combined_vs_raw", f"combined/{size}/online", f"combined/{size}/selector"),
        ):
            ratios[f"{label}/{size}"] = {
                metric + "_ratio": summaries[after][metric] / summaries[before][metric]
                for metric in METRICS
            }
    protocol_files = (
        "benchmark.py",
        "selector-benchmark.py",
        "controller.py",
        "selector-controller.py",
        "cross-version.py",
    )
    return {
        "schema_version": 1,
        "complete": True,
        "measurement_count": len(measured),
        "workstation_input_root": str(root),
        "toolchain_observed_at_collection": toolchain(root),
        "source_labels": {
            "base": "PR493 baseline; forward gate reports 56f47543a15ae1f7842ecc5bcf653409c138d496",
            "online-before": (
                "Uncommitted standalone-online source snapshot before eager root histogram reuse; "
                "identify by source and binary hashes, not a Git revision"
            ),
            "online": (
                "Standalone-online candidate; forward gate reports 470898a39f526aaa0af479ba89092a92b5883ea8; "
                "source and binary hashes are authoritative"
            ),
            "selector": "Combined-selector source snapshot; identify by recorded source and binary hashes",
        },
        "sources": sources,
        "protocol_sha256": {name: digest(root / name) for name in protocol_files},
        "fixtures": fixtures,
        "measurements": measured,
        "summary_median_of_three_process_statistics": summaries,
        "ratio_of_medians": ratios,
        "histogram_reuse_output_parity": reuse,
        "pr493_vs_canonical_frozen_numerical_comparison": frozen,
        "cross_version_online_checkpoint": {
            "raw_file_sha256": digest(root / "cross-version-result.json"),
            "checkpoint_input_sha256": checkpoint_inputs,
            **cross,
        },
        "forward_gate": {
            "comparison_sha256": digest(root / "forward-gate/comparison.json"),
            "raw_results_sha256": digest(root / "forward-gate/raw_results.json"),
            "comparison": forward,
            "configuration": forward_raw["configuration"],
            "host": forward_raw["host"],
            "created_at": forward_raw["created_at"],
            "cases": forward_compact,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument(
        "--annotate",
        type=Path,
        help="Add maintained-protocol hashes to an already collected measurement JSON",
    )
    args = parser.parse_args()
    if (args.root is None) == (args.annotate is None):
        parser.error("provide a benchmark root or --annotate, but not both")
    report = load(args.annotate) if args.annotate else main(args.root.resolve())
    maintained = Path(__file__).resolve().parent / "protocol"
    if maintained.is_dir():
        report["maintained_protocol_sha256"] = {
            path.name: digest(path) for path in sorted(maintained.glob("*.py"))
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
