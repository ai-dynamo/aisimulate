# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run matched native probes only after both isolated release builds finish."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

BASELINE = "4a710f144930a40e1cfee5010b386b35319457a7"
CANDIDATE = "247e02da9c56da398391ebe77dad724d0c5ae654"
EQUALITY_KEYS = (
    "initial_predictions_sha256",
    "pre_update_trajectory_sha256",
    "final_predictions_sha256",
    "final_artifact_sha256",
    "semantic_diagnostics_sha256",
    "queries_sha256",
    "seed_manifest_sha256",
    "config",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(root):
    inventory = {
        path.relative_to(root).as_posix(): (
            "symlink:" + os.readlink(path) if path.is_symlink() else digest(path)
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    production = {
        key: value
        for key, value in inventory.items()
        if key != "crates/core/examples/fpm_prefix_probe.rs"
    }
    return {
        "file_count": len(inventory),
        "inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "fitting_rs_sha256": inventory[
            "crates/core/src/perfmodel/fpm/learned_online/fitting.rs"
        ],
        "cargo_lock_sha256": inventory["Cargo.lock"],
        "probe_rs_sha256": inventory["crates/core/examples/fpm_prefix_probe.rs"],
        "production_file_count": len(production),
        "production_inventory_sha256": hashlib.sha256(
            json.dumps(production, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_results(root, manifest):
    """Mandatory output, binary and input parity gates before cost summaries."""
    references = {}
    paths = sorted((root / "results").glob("*.json"))
    expected = len(manifest["fixtures"]) * manifest["rounds"] * 4
    assert len(paths) == expected, (len(paths), expected)
    for path in paths:
        name, phase, round_, arm = path.stem.split("__")
        assert phase in ("timing", "allocation")
        assert 0 <= int(round_[1:]) < manifest["rounds"]
        record = json.loads(path.read_text())
        assert record["binary_sha256"] == manifest["binaries"][arm]["sha256"]
        assert record["updates"] == manifest["updates_per_process"]
        assert record["phase"] == phase
        files = manifest["fixtures"][name]["file_sha256"]
        assert record["seed_manifest_sha256"] == files["seed/manifest.json"]
        assert record["queries_sha256"] == files["queries.json"]
        assert len(record["durations_ns"]) == manifest["updates_per_process"]
        assert len(record["allocations"]) == (
            manifest["updates_per_process"] if phase == "allocation" else 0
        )
        evidence = {key: record[key] for key in EQUALITY_KEYS}
        if name in references:
            for key in EQUALITY_KEYS:
                assert evidence[key] == references[name][key], (path.name, key)
        else:
            references[name] = evidence
    return {
        "status": "pass",
        "records": len(paths),
        "paired_comparisons": len(paths) // 2,
        "all_rounds_and_phases_equal_per_scenario": True,
        "equality_keys": EQUALITY_KEYS,
        "scenario_references": references,
    }


def fixture(path):
    manifest = json.loads((path / "seed/manifest.json").read_text())
    artifact = json.loads((path / "seed" / manifest["artifact_file"]).read_text())
    files = {
        "queries.json": digest(path / "queries.json"),
        "seed/manifest.json": digest(path / "seed/manifest.json"),
        "seed/" + manifest["artifact_file"]: digest(
            path / "seed" / manifest["artifact_file"]
        ),
    }
    assert files["seed/" + manifest["artifact_file"]] == manifest["artifact_sha256"]
    for store in manifest["stores"].values():
        key = "seed/" + store["rows_file"]
        files[key] = digest(path / key)
        assert files[key] == store["rows_sha256"]
    return {
        "path": str(path),
        "file_sha256": files,
        "history_rows": sum(x["row_count"] for x in manifest["stores"].values()),
        "trees": len(artifact["stores"]["pure_decode"]["trees"]),
        "feature_count": len(artifact["features"]),
        "sklearn_version": manifest.get("sklearn_version"),
    }


def main(root, rounds, updates, cpu):
    scenarios = {
        "150trees_256rows_tail1": (root.parent / "small-fixture", "1"),
        "150trees_10000rows_tail1": (root.parent / "fixture", "1"),
        "300trees_10000rows_tail1": (root / "fixture-300", "1"),
        "150trees_10000rows_all": (root.parent / "fixture", "all"),
    }
    outputs = root / "results"
    outputs.mkdir()
    binaries = {
        name: {
            "path": str(root / (name + "-probe")),
            "sha256": digest(root / (name + "-probe")),
        }
        for name in ("baseline", "candidate")
    }
    manifest = {
        "schema_version": 1,
        "baseline_revision": BASELINE,
        "candidate_revision": CANDIDATE,
        "rounds": rounds,
        "updates_per_process": updates,
        "cpu_affinity": cpu,
        "host": {
            "platform": platform.platform(),
            "node": platform.node(),
            "lscpu": subprocess.check_output(["lscpu"], text=True),
            "meminfo": Path("/proc/meminfo").read_text(),
            "loadavg_before": Path("/proc/loadavg").read_text().strip(),
        },
        "started_at_unix": time.time(),
        "timing_definition": "Rust allocator wrapper present, allocation counters disabled",
        "allocation_definition": (
            "Rust GlobalAlloc requested sizes; successful allocations and full new "
            "realloc requests; peak is maximum net live-byte increase during one update"
        ),
        "sources": {name: source(root / name) for name in binaries},
        "binaries": binaries,
        "fixtures": {name: fixture(path) for name, (path, _) in scenarios.items()},
        "source_archives": json.loads((root / "source-archives.json").read_text()),
        "controller_sha256": digest(__file__),
        "fixture_preparation": json.loads((root / "prepare-fixture.log").read_text()),
        "fixture_preparation_script_sha256": digest(root / "prepare_fixture.py"),
        "rustc": subprocess.check_output(
            [str(Path.home() / ".cargo/bin/rustc"), "--version"], text=True
        ).strip(),
    }
    assert binaries["baseline"]["sha256"] != binaries["candidate"]["sha256"]
    for key in ("cargo_lock_sha256", "probe_rs_sha256"):
        assert (
            manifest["sources"]["baseline"][key]
            == manifest["sources"]["candidate"][key]
        )
    for arm in binaries:
        actual, expected = manifest["sources"][arm], manifest["source_archives"][arm]
        assert actual["production_inventory_sha256"] == expected["inventory_sha256"]
        assert actual["production_file_count"] == expected["file_count"]
        assert actual["fitting_rs_sha256"] == expected["fitting_rs_sha256"]
        assert actual["cargo_lock_sha256"] == expected["cargo_lock_sha256"]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
    }
    for phase in ("timing", "allocation"):
        for round_ in range(rounds):
            cases = list(scenarios.items())
            arms = ["baseline", "candidate"]
            if round_ % 2:
                cases.reverse()
                arms.reverse()
            for name, (path, tail) in cases:
                for arm in arms:
                    output = outputs / f"{name}__{phase}__r{round_}__{arm}.json"
                    command = [
                        "taskset",
                        "-c",
                        str(cpu),
                        binaries[arm]["path"],
                        str(path / "seed"),
                        str(path / "queries.json"),
                        tail,
                        str(updates),
                        phase,
                    ]
                    print(time.strftime("%H:%M:%S"), output.name, flush=True)
                    with output.open("w") as stream:
                        subprocess.run(command, env=env, stdout=stream, check=True)
                    record = json.loads(output.read_text())
                    assert record["binary_sha256"] == binaries[arm]["sha256"]
                    assert record["updates"] == updates
                    assert (
                        record["queries_sha256"]
                        == manifest["fixtures"][name]["file_sha256"]["queries.json"]
                    )
                    assert (
                        record["seed_manifest_sha256"]
                        == manifest["fixtures"][name]["file_sha256"][
                            "seed/manifest.json"
                        ]
                    )
    gates = validate_results(root, manifest)
    for arm in binaries:
        assert digest(binaries[arm]["path"]) == binaries[arm]["sha256"]
        assert source(root / arm) == manifest["sources"][arm]
    for name, (path, _) in scenarios.items():
        assert fixture(path) == manifest["fixtures"][name]
    gates["sources_binaries_and_inputs_unchanged_after_run"] = True
    gates["completed_at_unix"] = time.time()
    gates["loadavg_after"] = Path("/proc/loadavg").read_text().strip()
    (root / "equality-gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True)
    )
    print("complete: mandatory equality and provenance gates passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    main(args.root.resolve(), args.rounds, args.updates, args.cpu)
