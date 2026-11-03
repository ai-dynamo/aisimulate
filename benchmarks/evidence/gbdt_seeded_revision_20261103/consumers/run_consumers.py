"""Run unchanged consumer sources against a selected installed SDK/native build.

All fixture measurements are synthetic functional checks, not host-cost evidence.
External Dynamo sources stay in /private/tmp and are pinned in source-manifest.json.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("baseline", "candidate"))
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--only", choices=("dynamo-real-native",))
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    candidate = evidence.parents[3]
    sdk = args.sdk_root.resolve()
    python = sdk / "python/aisimulate/.venv/bin/python"
    runtime = sdk / "python/aisimulate/src/aisimulate/_runtime.abi3.so"
    actual = hashlib.sha256(runtime.read_bytes()).hexdigest()
    if actual != args.native_sha256:
        raise ValueError(f"Native hash mismatch: {actual}")
    gym_source = Path("/private/tmp/gbdt-baseline-gym-a8b30ca6")
    dynamo_source = Path("/private/tmp/gbdt-dynamo-318ab")
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        FPM_COMPAT_EXPECTED_SDK_ROOT=str(sdk) + "/",
        FPM_COMPAT_EXPECTED_NATIVE_SHA=args.native_sha256,
        GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="commit.gpgsign", GIT_CONFIG_VALUE_0="false",
    )
    commands_path = evidence / f"commands-{args.label}.json"
    commands = json.loads(commands_path.read_text()) if args.only and commands_path.exists() else []

    def run(name, argv, cwd, extra=None):
        if args.only is not None and args.only != name:
            return
        run_env = {**env, **(extra or {})}
        log = evidence / f"{name}-{args.label}.log"
        with log.open("w") as output:
            result = subprocess.run([str(x) for x in argv], cwd=cwd, env=run_env,
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        record = {"name": name, "label": args.label, "argv": [str(x) for x in argv],
                  "cwd": str(cwd), "env": {k: v for k, v in run_env.items() if k in
                  {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "FPM_COMPAT_EXPECTED_SDK_ROOT",
                   "FPM_COMPAT_EXPECTED_NATIVE_SHA", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
                   "PYTHONPATH", "GYM_COMPAT_RECEIPTS", "DYNAMO_COMPAT_EVIDENCE", "DYNAMO_COMPAT_CONTROL",
                   "DYNAMO_COMPAT_CANDIDATE_BUNDLE", "DYNAMO_COMPAT_BUNDLE"}}, "exit_code": result.returncode,
                  "log": str(log.relative_to(evidence))}
        commands.append(record)
        commands_path.write_text(json.dumps(commands, indent=2) + "\n")
        print(f"{name}: exit={result.returncode}, log={log}", flush=True)

    pytest_flags = ["-c", "/dev/null", "-p", "no:timeout", "-p", "no:cacheprovider"]
    runner = evidence / "gym/gbdt-gym-compat-runner.py"
    run("gym-unchanged", [python, runner, args.label, evidence / "gym/unchanged-suite", *pytest_flags,
        f"--rootdir={gym_source}", gym_source / "tests/fpm_accuracy", "-ra"], candidate)
    run("gym-real-native", [python, runner, args.label, evidence / "gym/real-native", *pytest_flags,
        f"--rootdir={gym_source}", evidence / "gym/test_unchanged_gym_real_native.py", "-v", "-ra"],
        gym_source, {"GYM_COMPAT_RECEIPTS": str(evidence / "gym/real-native" / args.label)})
    dynamo_pythonpath = str(dynamo_source / "components/src") + ":" + str(sdk / "python/aisimulate/src")
    run("dynamo-upstream", [python, "-m", "pytest", *pytest_flags, f"--rootdir={dynamo_source}",
        dynamo_source / "components/src/dynamo/planner/tests/unit/test_ais_perf_adapter.py",
        dynamo_source / "components/src/dynamo/planner/tests/unit/test_engine_query.py", "-q"],
        dynamo_source, {"PYTHONPATH": dynamo_pythonpath})
    bundle = evidence / "dynamo/aggregated-fixture/bundle.json"
    dynamo_env = {"PYTHONPATH": dynamo_pythonpath, "DYNAMO_COMPAT_BUNDLE": str(bundle),
                  "DYNAMO_COMPAT_EVIDENCE": str(evidence / "dynamo" / args.label)}
    if args.label == "baseline":
        dynamo_env.update(DYNAMO_COMPAT_CONTROL="1", DYNAMO_COMPAT_CANDIDATE_BUNDLE=str(bundle))
    run("dynamo-real-native", [python, evidence / "dynamo/real_native_harness.py"], dynamo_source, dynamo_env)


if __name__ == "__main__":
    main()
