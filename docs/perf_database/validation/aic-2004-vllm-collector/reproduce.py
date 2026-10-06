# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute an attested, bounded matrix through the unchanged native collector API."""

import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
import traceback
from pathlib import Path

import torch
import vllm
from collector import helper, provenance
from collector.vllm import collect_mla_module as collector

out = Path("/output")
root = Path(__file__).parent
plan = json.loads((root / "plan.json").read_text())


def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


assert vllm.__version__ == "0.25.1", vllm.__version__
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability() == (10, 0)
assert "B200" in torch.cuda.get_device_name()
payload = Path(collector.__file__).resolve().parents[2]
closure = provenance.collector_hash(
    "collector.vllm.collect_mla_module", payload, provenance.load_closures(payload / "collector/hash_closures.yaml")
)
identity = dict(
    vllm=vllm.__version__,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(),
    device_properties=str(torch.cuda.get_device_properties(0)),
    collector_ref=plan["collector_ref"],
    collector_hash=closure,
    case_plan_hash=provenance.case_plan_hash([c["id"] for c in plan["cases"]]),
    plan_sha256=hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    hostname=os.uname().nodename,
    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    image_digest="sha256:4566da49d52bb305f971ea75411bb3f06f5de795387fc1682c8f211dcd309c1e",
    runtime_source_commit="752a3a504485790a2e8491cacbb35c137339ad34",
)
for package in ["flashinfer-python", "flashinfer-cubin", "transformers"]:
    identity[package] = importlib.metadata.version(package)
save("identity.json", identity)
native = Path(vllm.__file__).parent
sources = {}
for rel in [
    "model_executor/models/deepseek_v2.py",
    "model_executor/layers/mla.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "v1/worker/workspace.py",
]:
    sources[rel] = hashlib.sha256((native / rel).read_bytes()).hexdigest()
save("runtime-source-hashes.json", sources)
original_benchmark = collector.benchmark_with_power
records = []
current = {}


@contextlib.contextmanager
def observe(**kwargs):
    # Passive receipt: do not change kernel, graph flags, iterations, or measured results.
    module = inspect.getclosurevars(kwargs["kernel_func"]).nonlocals["attn_module"]
    before = dict(
        case_id=current["id"],
        skip_indexer=module.mla_attn.skip_topk,
        graph_requested=kwargs["use_cuda_graph"],
        allow_graph_fail=kwargs["allow_graph_fail"],
        warmups=kwargs["num_warmups"],
        iterations=kwargs["num_runs"],
    )
    with original_benchmark(**kwargs) as result:
        records.append(dict(before, **result))
        save("timing-receipts.json", records)
        yield result


collector.benchmark_with_power = observe
completed = []
failed = []
keys = [
    "seq_len",
    "batch_size",
    "num_heads",
    "kv_cache_dtype",
    "compute_dtype",
    "gemm_type",
    "prefix_len",
    "model_path",
    "attn_type",
]
for case in [*plan["cases"], *plan["repeat_controls"]]:
    current = case
    started = time.monotonic()
    dest = out / ("controls" if case.get("repeat_control") else "data")
    dest.mkdir(exist_ok=True)
    print("START", case["id"], flush=True)
    try:
        with torch.inference_mode():
            collector.run_mla_module(
                **{k: case[k] for k in keys},
                perf_filename=str(dest / f"dsa_{case['phase']}_module_perf.txt"),
                warming_up=plan["warmups"],
                test_ite=plan["iterations"],
            )
        torch.cuda.synchronize()
        completed.append(
            dict(
                case_id=case["id"],
                wall_seconds=time.monotonic() - started,
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            )
        )
        save("completed.json", completed)
        print("COMPLETE", case["id"], flush=True)
    except Exception as exc:
        failed.append(dict(case_id=case["id"], error=repr(exc), traceback=traceback.format_exc()))
        save("failures.json", failed)
        raise
for dest in [out / "data", out / "controls"]:
    if dest.exists():
        for csv in sorted(dest.glob("*_perf.txt")):
            helper.convert_perf_csv_to_parquet(csv, delete_source=False)
save(
    "result.json",
    dict(
        status="complete",
        planned_cases=len(plan["cases"]),
        completed_cases=len(completed),
        failed_cases=len(failed),
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ),
)
print("ALL_COMPLETE", flush=True)
