# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay either separately qualified service corpus profile against corrected op predictions."""

import argparse
import concurrent.futures
import gzip
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from replay import MODES, NATIVE_SHA, TOOLS_REF, read, sha, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("off", "on"))
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = next(p for p in root.parents if (p / "Cargo.toml").is_file())
    import aisimulate._runtime as native

    if sha(native.__file__) != NATIVE_SHA:
        raise ValueError("corrected native binary differs")
    provenance = read(root / args.profile / "replay-provenance.json")
    for path, digest in provenance["model_source_sha256"].items():
        if sha(repo / path) != digest:
            raise ValueError("corrected model source differs")
    output = (args.output_dir or root) / f"service-{args.profile}"
    output.mkdir(exist_ok=False)
    inputs = {
        "audit": args.qualified / "audit.json",
        "measurement": args.qualified / "measurement.json",
        "plan": args.raw / "strata/service-records/main-plan.json",
    }
    scope_receipt = args.qualified / "supplemental-scope-receipt.json"
    scope = read(scope_receipt)
    if scope["corpus"] != "service-records" or read(inputs["plan"])["requested_trials"] != 30:
        raise ValueError("expected separately qualified 30-trial service corpus")
    measurement = read(inputs["measurement"])
    if measurement["decoder_replay"] != (args.profile == "on") or not measurement["complete_requested_study"]:
        raise ValueError("expected complete admitted service corpus and Decoder profile")
    for name, digest in scope["qualified_output_sha256"].items():
        if sha(args.qualified / name) != digest:
            raise ValueError("qualified output does not match its closed-lifetime admission")
    input_hashes = {key: sha(path) for key, path in inputs.items()}
    with tempfile.TemporaryDirectory(prefix="dsv41-service-report-") as temporary:
        temporary = Path(temporary)
        for path, digest in provenance["analysis_source_sha256"].items():
            data = subprocess.check_output(["git", "show", f"{TOOLS_REF}:{path}"], cwd=repo)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("comparison helper identity differs")
            (temporary / Path(path).name).write_bytes(data)
        jobs = []
        for mode in MODES:
            config = root / args.profile / f"{mode}-config.json"
            target = output / config.name
            write_new(target, read(config))
            for kind in ("trace", "e2e"):
                jobs.append((mode, kind, target))

        def execute(job):
            mode, kind, config = job
            name = f"{kind}-{mode}"
            paths = inputs.copy()
            if kind == "e2e":
                paths |= {
                    "client-summary": args.raw / "strata/service-records/main-client/summary.json",
                    "scheduler-receipt": args.raw / "scheduler-receipt.json",
                }
            hashes = {key: sha(path) for key, path in paths.items()}
            target = temporary / f"{name}.json"
            cmd = [
                sys.executable,
                str(temporary / f"compare_{kind}.py"),
                "--prediction-config",
                str(config),
                "--output",
                str(target),
            ]
            for key, path in paths.items():
                cmd.extend(["--" + key, str(path.resolve())])
            with (temporary / f"{name}.log").open("w") as stream:
                status = subprocess.run(cmd, cwd=repo, env=os.environ, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError((temporary / f"{name}.log").read_text()[-3000:])
            if hashes != {key: sha(path) for key, path in paths.items()}:
                raise ValueError("qualified service observations changed")
            data = target.read_bytes()
            destination = output / f"{name}-results.json.gz"
            destination.write_bytes(gzip.compress(data, mtime=0))
            print(name, flush=True)
            return {
                "output": destination.name,
                "output_sha256": sha(destination),
                "uncompressed_output_sha256": hashlib.sha256(data).hexdigest(),
                "input_sha256": hashes,
                "prediction_config": config.name,
                "prediction_config_sha256": sha(config),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(execute, jobs))
    provenance = {k: v for k, v in provenance.items() if k not in ("jobs", "scope", "replay_program_sha256")}
    provenance.update(
        scope=f"service-{args.profile}",
        replay_program_sha256=sha(__file__),
        jobs=results,
        scope_admission_sha256=sha(scope_receipt),
        qualified_input_sha256=input_hashes,
        trial_count=30,
        sampling_precision_note="Frozen supplementary window; no claim of 5% precision",
    )
    write_new(output / "replay-provenance.json", provenance)


if __name__ == "__main__":
    main()
