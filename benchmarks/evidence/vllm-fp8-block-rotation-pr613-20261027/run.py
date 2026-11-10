# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Jan Fostier
# SPDX-License-Identifier: Apache-2.0

"""Run exact Git revisions of the real collector; observe timing and publication.

GPU/framework imports stay inside workers. Each arm/start gets a fresh process
and an extracted, unmodified collector package from its pinned Git revision.
Only log_perf is redirected to evidence JSONL; its latency is not recomputed.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTOR = "python/aisimulate/collector"


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gpu_status():
    fields = "name,uuid,memory.total,memory.used,temperature.gpu,pstate,clocks.current.sm,clocks.current.memory"
    fields += ",power.limit"
    return command("nvidia-smi", f"--query-gpu={fields}", "--format=csv")


def worker(args, protocol):
    # Use all collector dependencies from the same revision, not just the
    # changed module. No source replacement or allocation-count override.
    revision = protocol[f"{args.worker}_revision"]
    archive = subprocess.check_output(["git", "archive", revision, COLLECTOR], cwd=args.repo)
    with tempfile.TemporaryDirectory(prefix="pr613-collector-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(temporary, filter="data")
        sys.path.insert(0, str(Path(temporary) / "python/aisimulate"))
        import torch
        import vllm
        from collector.vllm import collect_gemm

        properties = torch.cuda.get_device_properties(0)
        if properties.name != protocol["device_name"]:
            raise RuntimeError(f"Expected {protocol['device_name']}; found {properties.name}")
        if vllm.__version__ != protocol["framework"].split("==")[1] or torch.__version__ != protocol["torch"]:
            raise RuntimeError(f"Unexpected runtime: vLLM {vllm.__version__}, torch {torch.__version__}")
        prefix = f"{args.worker}-{args.replicate}"
        runtime_paths = [
            "model_executor/layers/quantization/fp8.py",
            "model_executor/layers/quantization/utils/fp8_utils.py",
            "model_executor/kernels/linear/scaled_mm/flashinfer.py",
            "model_executor/kernels/linear/scaled_mm/deep_gemm.py",
            "utils/deep_gemm.py",
        ]
        root = Path(vllm.__file__).parent
        environment = {
            "arm": args.worker,
            "replicate": args.replicate,
            "revision": revision,
            "protocol_sha256": digest(args.output / "protocol.json"),
            "runner_sha256": digest(__file__),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "packages": {
                d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]
            },
            "device": {
                "name": properties.name,
                "l2_bytes": properties.L2_cache_size,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            },
            "runtime_source_sha256": {p: digest(root / p) for p in runtime_paths},
            "collector_source_sha256": {
                p: digest(Path(temporary) / COLLECTOR / p)
                for p in ["vllm/collect_gemm.py", "vllm/utils.py", "helper.py"]
            },
            "environment": {
                name: os.environ.get(name)
                for name in [
                    "COLLECTOR_MEASURE_POWER",
                    "CUDA_HOME",
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "VLLM_USE_DEEP_GEMM",
                    "VLLM_USE_DEEP_GEMM_E8M0",
                    "VLLM_BATCH_INVARIANT",
                ]
            },
            "gpu_before": gpu_status(),
        }
        write_json(args.output / f"{prefix}-environment.json", environment)
        original_benchmark = collect_gemm.benchmark_with_power
        state = {}

        @contextlib.contextmanager
        def observe(**kwargs):
            expected = protocol["timing"]
            for key in ["num_warmups", "num_runs", "repeat_n", "use_cuda_graph", "allow_graph_fail"]:
                if kwargs[key] != expected[key]:
                    raise RuntimeError(f"Collector changed timing setting {key}: {kwargs[key]}")
            ops = inspect.getclosurevars(kwargs["kernel_func"]).nonlocals["op_list"]
            pointers = [op.weight.untyped_storage().data_ptr() for op in ops]
            copies = len(ops)
            wanted = (
                1
                if args.worker == "base"
                else max(
                    1, min(6, (2 * properties.L2_cache_size + state["n"] * state["k"] - 1) // (state["n"] * state["k"]))
                )
            )
            if copies != wanted or len(set(pointers)) != copies:
                raise RuntimeError(f"Invalid allocation rotation: {copies=}, {wanted=}")
            state.update(
                copies=copies,
                distinct_weight_storages=len(set(pointers)),
                weight_dtype=str(ops[0].weight.dtype),
                one_weight_storage_bytes=ops[0].weight.untyped_storage().nbytes(),
            )
            # Call the original helper with its original arguments. In
            # particular, do not lengthen graphs, warmup or timed repeats.
            with original_benchmark(**kwargs) as result:
                if result["used_cuda_graph"] is not True or result["num_runs_executed"] != 6:
                    raise RuntimeError(f"Unexpected timing mode: {result}")
                state.update(
                    loop_latency_ms=result["latency_ms"],
                    used_cuda_graph=result["used_cuda_graph"],
                    num_runs_executed=result["num_runs_executed"],
                )
                yield result

        raw_path = args.output / f"{prefix}-measurements.jsonl"
        with raw_path.open("x") as raw:

            def publish(**kwargs):
                if state.get("published"):
                    raise RuntimeError("More than one performance row from a collector call")
                (row,) = kwargs["item_list"]
                if [row[key] for key in ["m", "n", "k"]] != [state[key] for key in ["m", "n", "k"]]:
                    raise RuntimeError("Published shape does not match the requested case")
                latency = row["latency"]
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError(f"Invalid published latency: {latency}")
                if not math.isclose(latency * state["copies"], state["loop_latency_ms"], rel_tol=1e-12):
                    raise RuntimeError("Incorrect per-operation latency normalization")
                result = {
                    **state,
                    "arm": args.worker,
                    "replicate": args.replicate,
                    "revision": revision,
                    "published_latency_ms": latency,
                    "kernel_source": kwargs["kernel_source"],
                    "framework": kwargs["framework"],
                    "framework_version": kwargs["version"],
                    "device_name": kwargs["device_name"],
                }
                raw.write(json.dumps(result, allow_nan=False) + "\n")
                raw.flush()
                state["published"] = True
                return True

            collect_gemm.benchmark_with_power = observe
            collect_gemm.log_perf = publish
            order = list(enumerate(protocol["shapes_m_n_k"]))
            random.Random(protocol["seed"] + args.replicate).shuffle(order)
            for shape_index, (m, n, k) in order:
                for sample in range(protocol["samples_per_shape"]):
                    seed = protocol["seed"] + args.replicate * 10000 + shape_index * 100 + sample
                    state.clear()
                    state.update(m=m, n=n, k=k, shape_index=shape_index, sample=sample, seed=seed)
                    torch.manual_seed(seed)
                    with torch.inference_mode():
                        collect_gemm.run_gemm("fp8_block", m, n, k, perf_filename="evidence-only")
                    if not state.get("published"):
                        raise RuntimeError("Collector returned without a performance row")
                print(f"DONE {prefix} shape={m, n, k}", flush=True)
        write_json(args.output / f"{prefix}-gpu-after.json", {"gpu_after": gpu_status()})


def campaign(args, protocol):
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    for arm in ["base", "head"]:
        ref = protocol[f"{arm}_revision"]
        if command("git", "rev-parse", ref, cwd=args.repo) != ref:
            raise RuntimeError(f"Protocol requires an immutable full Git revision: {ref}")
    changed = command(
        "git",
        "diff",
        "--name-only",
        protocol["base_revision"],
        protocol["head_revision"],
        "--",
        COLLECTOR,
        cwd=args.repo,
    ).splitlines()
    if changed != [f"{COLLECTOR}/vllm/collect_gemm.py"]:
        raise RuntimeError(f"Unexpected collector changes: {changed}")
    active = command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
    if active:
        raise RuntimeError(f"Start with an idle GPU; compute processes are present: {active}")
    write_json(args.output / "protocol.json", protocol)
    environment = os.environ.copy()
    environment["COLLECTOR_MEASURE_POWER"] = "0"
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.pop("PYTHONPATH", None)
    jobs = []
    write_json(
        args.output / "invocation.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "runner_sha256": digest(__file__),
            "gpu_before": gpu_status(),
        },
    )
    for replicate in range(protocol["replicates"]):
        arms = ["base", "head"] if replicate % 2 == 0 else ["head", "base"]
        for arm in arms:
            prefix = f"{arm}-{replicate}"
            child = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                arm,
                "--replicate",
                str(replicate),
                "--repo",
                str(args.repo),
                "--output",
                str(args.output),
            ]
            started = time.time()
            print(f"START {prefix}", flush=True)
            with (args.output / f"{prefix}.log").open("x") as log:
                result = subprocess.run(child, cwd=args.repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
            jobs.append(
                {
                    "arm": arm,
                    "replicate": replicate,
                    "command": child,
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "exit_code": result.returncode,
                }
            )
            (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"{prefix} exited {result.returncode}; inspect {prefix}.log")
            print(f"FINISHED {prefix} ({time.time() - started:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--worker", choices=["base", "head"])
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    protocol_path = args.output / "protocol.json" if args.worker else HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if args.worker:
        worker(args, protocol)
    else:
        campaign(args, protocol)


if __name__ == "__main__":
    main()
