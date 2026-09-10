# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recompute predictions against frozen, qualified GB300 observations.

Run from the SILICON repository root with its rebuilt extension. The analysis
helpers come from an immutable AISimulate commit, extracted without importing
that checkout's prediction model. Private inputs remain outside this report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_REF = "29b45ac55df501b061be7d3aac4b059922059125"
NATIVE_REF = "09954c4453801a7252d2e25f1cac1290ddc94521"
TOOLS_REF = "be0c44dfd1e319139d16659f2d96bf9a88337c53"
NATIVE_SHA = "eb72fd7443f4e286bd9fc9d73cd9fc213b3acd29eb2420c0485a780714640704"
TOOLS_PREFIX = "data/experimental/deepseek-v41/verification-plan"
DATA_PREFIX = "data/experimental/deepseek-v41/gb300-silicon"
MODES = ("sol", "hybrid", "silicon")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("forward", "off", "on"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--off-qualified", type=Path)
    parser.add_argument("--off-raw", type=Path)
    parser.add_argument("--on-qualified", type=Path)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    import aisimulate._runtime as native

    if sha(native.__file__) != NATIVE_SHA:
        raise ValueError("reproduction requires the recorded corrected SILICON native extension")
    model_paths = [
        "python/aisimulate/src/aiconfigurator_core/sdk/models/deepseek_v41.py",
        "python/aisimulate/src/aiconfigurator_core/sdk/deepseek_v41.py",
        "python/aisimulate/src/aiconfigurator_core/sdk/engine.py",
        "python/aisimulate/src/aiconfigurator_core/sdk/rust_engine_step.py",
        "crates/core/src/perfmodel/operators/dsv41.rs",
    ]
    model_hashes = {}
    for path in model_paths:
        original = subprocess.check_output(["git", "show", f"{MODEL_REF}:{path}"])
        model_hashes[path] = hashlib.sha256(original).hexdigest()
        if sha(repo / path) != model_hashes[path]:
            raise ValueError("prediction source differs from frozen corrected model: " + path)
    output = args.output_dir.resolve() / args.scope
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("refusing to overwrite an existing report scope")
    with tempfile.TemporaryDirectory(prefix="dsv41-sol-review-v2-") as scratch:
        scratch = Path(scratch)
        tools = scratch / "tools"
        tools.mkdir()
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", TOOLS_REF, TOOLS_PREFIX], text=True)
        tool_hashes = {}
        for name in names.splitlines():
            if not name.endswith(".py"):
                continue
            data = subprocess.check_output(["git", "show", f"{TOOLS_REF}:{name}"])
            (tools / Path(name).name).write_bytes(data)
            tool_hashes[name] = hashlib.sha256(data).hexdigest()
        env = os.environ | {"PYTHONPATH": f"{repo}/python/aisimulate/src:{repo}/python/aisimulate"}
        jobs = []
        profiles = (
            ("full", "decoder_bounded")
            if args.scope == "forward"
            else ("full" if args.scope == "off" else "decoder_bounded",)
        )
        for profile in profiles:
            for mode in MODES:
                name = f"{profile}-{mode}" if args.scope == "forward" else mode
                config = read(repo / DATA_PREFIX / "report/prefix-refinement-v1" / f"{profile}-{mode}-config.json")
                config["systems_path"] = f"{DATA_PREFIX}/indexer-identity-v2/prefix-refinement/{profile}/systems"
                config["forward_model"] = "op_level"
                config_path = output / f"{name}-config.json"
                write_new(config_path, config)
                if args.scope == "forward":
                    inputs = {
                        "observations": repo
                        / DATA_PREFIX
                        / "prefix-refinement-v1"
                        / profile
                        / "heldout/forward-results.json",
                        "heldout-plan": repo / DATA_PREFIX / "prefix-refinement-v1/heldout-points.json",
                    }
                    jobs.append((name, "compare_forward.py", config_path, inputs))
                elif args.scope == "off":
                    if args.off_qualified is None or args.off_raw is None:
                        raise ValueError("OFF requires qualified audit and original raw lifecycle directories")
                    inputs = {
                        "audit": args.off_qualified / "audit.json",
                        "plan": args.off_raw / "strata/primary/main-plan.json",
                        "measurement": args.off_qualified / "measurement.json",
                    }
                    jobs.append((f"trace-{name}", "compare_trace.py", config_path, inputs))
                    jobs.append(
                        (
                            f"e2e-{name}",
                            "compare_e2e.py",
                            config_path,
                            inputs
                            | {
                                "client-summary": args.off_raw / "strata/primary/main-client/summary.json",
                                "scheduler-receipt": args.off_raw / "scheduler-receipt.json",
                            },
                        )
                    )
                else:
                    if args.on_qualified is None:
                        raise ValueError("ON requires the qualified closed-segment directory")
                    jobs.append(
                        (
                            name,
                            "compare_segments.py",
                            config_path,
                            {
                                "logical-plan": args.on_qualified / "logical-plan.json",
                                "segments-file": args.on_qualified / "segments.json",
                            },
                        )
                    )

        def execute(job):
            name, script, config, inputs = job
            result = scratch / f"{name}-results.json"
            cmd = [sys.executable, str(tools / script), "--prediction-config", str(config), "--output", str(result)]
            before = {key: sha(path) for key, path in inputs.items()}
            for key, path in inputs.items():
                cmd.extend(["--" + key, str(path.resolve())])
            log = scratch / f"{name}.log"
            with log.open("w") as stream:
                status = subprocess.run(cmd, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"{name} failed: " + log.read_text()[-3000:])
            if before != {key: sha(path) for key, path in inputs.items()}:
                raise ValueError("qualified inputs changed during replay")
            payload = read(result)
            suffix = ".json" if args.scope == "forward" else ".json.gz"
            destination = output / f"{name}-results{suffix}"
            data = result.read_bytes()
            with destination.open("xb") as stream:
                stream.write(gzip.compress(data, mtime=0) if suffix.endswith("gz") else data)
            print(json.dumps({"completed": name, "scope": args.scope}), flush=True)
            return {
                "name": name,
                "comparison_script": script,
                "input_files_sha256": before,
                "prediction_config": config.name,
                "prediction_config_sha256": sha(config),
                "output": destination.name,
                "output_sha256": sha(destination),
                "uncompressed_output_sha256": hashlib.sha256(data).hexdigest(),
                "schema": payload["schema"],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(execute, jobs))
        write_new(
            output / "replay-provenance.json",
            {
                "schema": "dsv41.sol-review.report-replay.v2",
                "scope": args.scope,
                "model_commit": MODEL_REF,
                "native_compiled_source_commit": NATIVE_REF,
                "native_sha256": NATIVE_SHA,
                "analysis_commit": TOOLS_REF,
                "model_source_sha256": model_hashes,
                "analysis_source_sha256": tool_hashes,
                "replay_program_sha256": sha(__file__),
                "indexer_derivation_sha256": sha(repo / DATA_PREFIX / "indexer-identity-v2/derivation.json"),
                "new_measurements": False,
                "correction_fitting": False,
                "jobs": results,
            },
        )


if __name__ == "__main__":
    main()
