# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral GPU-side helpers for the dsv411 producers.

* identity: allocated-device witness, installed-source hashes, allocation-private JIT caches;
* timing: :class:`Intervals` (CUDA-event intervals around existing native calls, eager with a
  stream drain or recorded as *external* events inside a CUDA-graph capture) and
  :class:`GraphedCalls` (record a sub-call's arguments once, then capture and replay it as its
  own CUDA graph — the generation regime for the token-only components);
* raw evidence: :class:`RowStream` (one JSON row per rank, sample and physical key) and the
  per-rank receipt.

Nothing here knows a framework; adapters decide WHICH native methods are wrapped and prove
their geometry against the SDK manifest before installing any wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import types
from pathlib import Path

from .contract import RECEIPT_SCHEMA, canonical_json, make_row, regime_for, validate_row

JIT_CACHE_ENV = (
    ("TRITON_CACHE_DIR", "triton"),
    ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ("DEEP_GEMM_CACHE_DIR", "deep-gemm"),
    ("VLLM_CACHE_ROOT", "vllm"),
)


def dispatch_label(module, method: str) -> str:
    """Witness label: owning class + method + the quant method / kernel the framework selected."""
    if isinstance(module, types.ModuleType):
        return f"{module.__name__}.{method}"
    quant = getattr(module, "quant_method", None)
    kernel = getattr(quant, "kernel", None)
    suffix = (
        ""
        if quant is None
        else f"/{type(quant).__module__}.{type(quant).__name__}"
        + ("" if kernel is None else f"[{type(kernel).__name__}]")
    )
    return f"{type(module).__module__}.{type(module).__name__}.{method}{suffix}"


def prepare_private_caches(rank: int, receipt: dict, *, marker_prefix: str) -> None:
    """Bind JIT/compile caches to the allocation; run BEFORE importing the framework."""
    root = Path(os.environ["DSV411_PRIVATE_CACHE"])
    allocation = os.environ.get("SLURM_JOB_ID") or os.environ["DSV411_ALLOCATION_ID"]
    marker = root / "home-cache" / f"{marker_prefix}-{allocation}-{rank}"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("allocation-private\n")
    receipt["cache_bindings"] = []
    for target in (Path.home() / ".cache", Path("/root/.cache")):
        if not (target / marker.name).samefile(marker):
            raise RuntimeError("HOME/root cache does not share the allocation-private file")
        receipt["cache_bindings"].append({"path": str(target), "resolved": str(target.resolve())})
    for variable, leaf in JIT_CACHE_ENV:
        cache = root / f"rank-{rank}" / leaf
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)


def device_witness(local_rank: int, plan: dict) -> dict:
    import torch

    props = torch.cuda.get_device_properties(local_rank)
    sm = props.major * 10 + props.minor
    if (
        re.search(r"\b" + re.escape(plan["expected_gpu"]) + r"\b", props.name, re.IGNORECASE) is None
        or sm != plan["expected_sm"]
    ):
        raise RuntimeError(f"allocated GPU {props.name!r} sm{sm} differs from the plan")
    uuid = str(props.uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    command = [
        "nvidia-smi",
        "--id=" + uuid,
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20)
    witness = dict(
        name=props.name,
        sm=sm,
        cuda_local_rank=local_rank,
        gpu_uuid=uuid,
        memory=props.total_memory,
        argv=command,
        returncode=query.returncode,
        stdout=query.stdout,
        stderr=query.stderr,
    )
    if query.returncode != 0 or len(query.stdout.strip().splitlines()) != 1 or uuid not in query.stdout:
        raise RuntimeError("allocated CUDA UUID did not match the native driver witness")
    return witness


def source_hashes(package_root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }


def check_pins(plan: dict, sources: dict, model_path: Path, manifest: dict, runtime_digest: str) -> None:
    for path, digest in plan["source_pins"].items():
        if sources.get(path) != digest:
            raise RuntimeError("installed native source differs from the plan: " + path)
    for name, digest in plan["metadata_pins"].items():
        if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("checkpoint/tokenizer metadata changed: " + name)
    config = json.loads((model_path / "config.json").read_text())
    if hashlib.sha256(canonical_json(config).encode()).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("checkpoint config differs from the SDK manifest")
    if runtime_digest != plan["runtime_digest"] or os.environ.get("DSV411_LAUNCH_IMAGE_SHA256") != plan["image_sha256"]:
        raise RuntimeError("launch image differs from the frozen plan")


def finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


# --------------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------------
class Intervals:
    """CUDA-event intervals around existing native calls.

    ``active`` gates recording; ``bucket`` selects where events go (``None`` = the current eager
    forward; a hashable capture key while a framework captures a CUDA graph — events are then
    created *external* so their elapsed time is readable after every replay).
    """

    def __init__(self):
        import torch

        self.torch = torch
        self.active = False
        self.bucket = None
        self.buckets: dict = {None: []}
        self.originals = []
        self.violations: list[str] = []

    def _event(self):
        capturing = self.torch.cuda.is_current_stream_capturing()
        return self.torch.cuda.Event(enable_timing=True, external=capturing)

    def wrap(self, module, method: str, resolve, *, drain: bool = False, after=None, witness: str | None = None):
        """Wrap ``module.method``; ``resolve(*args, **kwargs) -> tag or None`` decides per call.

        ``drain`` synchronizes the device before the start event (eager regime only), so the
        interval is kernels + host enqueue of this call and never a backlog of earlier layers.
        """
        original = getattr(module, method)
        witness = witness or dispatch_label(module, method)
        intervals = self

        def timed(*args, **kwargs):
            tag = resolve(*args, **kwargs) if intervals.active else None
            if tag is None:
                result = original(*args, **kwargs)
                return after(result) if after is not None else result
            capturing = intervals.torch.cuda.is_current_stream_capturing()
            if drain and not capturing:
                intervals.torch.cuda.synchronize()
            start, end = intervals._event(), intervals._event()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            intervals.buckets.setdefault(intervals.bucket, []).append((tag, witness, start, end, capturing))
            return after(result) if after is not None else result

        self.originals.append((module, method, original))
        setattr(module, method, timed)

    def restore(self):
        for module, method, original in self.originals:
            setattr(module, method, original)
        self.originals.clear()

    def take(self, bucket=None, *, clear: bool = True) -> list[tuple]:
        """(tag, witness, elapsed_ms, captured) of one bucket; synchronizes first."""
        self.torch.cuda.synchronize()
        events = self.buckets.get(bucket, [])
        out = [(tag, witness, start.elapsed_time(end), captured) for tag, witness, start, end, captured in events]
        if clear:
            self.buckets[bucket] = []
        return out


def reduce_layer_intervals(
    records: list[tuple], representatives: dict, manifest_entries: list[dict], phase: str
) -> dict:
    """Turn (tag, witness, ms, captured) records of one forward into per-entry latencies.

    Tags are ``("layer", layer_id)`` for a whole attention layer interval and ``("indexer",
    layer_id)`` for the nested index-scoring interval. ``attention_core`` = layer minus indexer of the
    same layer; ``indexer`` = its own interval. Only representative layers produce rows.
    """
    layer_ms, indexer_ms, witnesses, captured_flags = {}, {}, {}, {}
    for (kind, layer), witness, ms, captured in records:
        target = layer_ms if kind == "layer" else indexer_ms
        target[layer] = target.get(layer, 0.0) + ms
        witnesses.setdefault((kind, layer), set()).add(witness)
        captured_flags.setdefault((kind, layer), set()).add(captured)
    out = {}
    for entry in manifest_entries:
        if entry["phase"] != phase or entry["component"] not in ("attention_core", "indexer"):
            continue
        layer = entry["layer"]
        if representatives[phase][entry["component"]].get(entry["structure_key"]) != layer:
            continue
        if entry["component"] == "indexer":
            if layer not in indexer_ms:
                raise RuntimeError(f"no indexer interval recorded at representative layer {layer}")
            out[(entry["component"], entry["structure_key"])] = (
                entry,
                indexer_ms[layer],
                witnesses[("indexer", layer)],
                captured_flags[("indexer", layer)],
            )
        else:
            if layer not in layer_ms:
                raise RuntimeError(f"no attention interval recorded at representative layer {layer}")
            ms = layer_ms[layer] - indexer_ms.get(layer, 0.0)
            if ms <= 0:
                raise RuntimeError(
                    f"attention_core interval at layer {layer} is not positive after removing the indexer"
                )
            label = set(witnesses[("layer", layer)])
            if layer in indexer_ms:
                label.add("minus:" + "+".join(sorted(witnesses[("indexer", layer)])))
            out[(entry["component"], entry["structure_key"])] = (entry, ms, label, captured_flags[("layer", layer)])
    return out


class GraphedCalls:
    """Generation regime for token-only components: record each wrapped sub-call once (eager),
    then capture ONE CUDA graph per sub-call and time its replay. Collectives between the
    sub-calls (Engram gather/reduce, shared-expert reduce) stay outside every graph."""

    def __init__(self):
        import torch

        self.torch = torch
        self.calls: list[tuple] = []
        self.originals = []
        self.recording = False

    def wrap(self, module, method: str, entry: dict):
        original = getattr(module, method)
        witness = dispatch_label(module, method)
        graphed = self

        def recorded(*args, **kwargs):
            if graphed.recording:
                graphed.calls.append((entry, witness, original, args, kwargs))
            return original(*args, **kwargs)

        self.originals.append((module, method, original))
        setattr(module, method, recorded)

    def restore(self):
        for module, method, original in self.originals:
            setattr(module, method, original)
        self.originals.clear()
        self.calls.clear()

    def record(self, call):
        """Run ``call`` once eagerly, remembering every wrapped sub-call's arguments."""
        self.calls.clear()
        self.recording = True
        try:
            return call()
        finally:
            self.recording = False

    def measure(self, warmup: int, iterations: int) -> dict:
        """Per (component, structure_key): (entry, [latency per iteration], witnesses)."""
        torch = self.torch
        results: dict = {}
        side = torch.cuda.Stream()
        for entry, witness, original, args, kwargs in self.calls:
            torch.cuda.synchronize()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    original(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                original(*args, **kwargs)
            torch.cuda.synchronize()
            samples = []
            for sample in range(warmup + iterations):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                if sample >= warmup:
                    samples.append(start.elapsed_time(end))
            key = (entry["component"], entry["structure_key"])
            slot = results.setdefault(key, (entry, [0.0] * iterations, set()))
            for i, ms in enumerate(samples):
                slot[1][i] += ms
            slot[2].add(witness)
            del graph
        return results


# --------------------------------------------------------------------------------------------
# raw evidence
# --------------------------------------------------------------------------------------------
def completed_cases(output: Path, plan: dict) -> set[str]:
    """Attention cases every rank of a preserved run finished (intersection of the progress files),
    restricted to the plan's cases. Resuming skips exactly these; everything else is re-measured."""
    planned = {c["case_id"] for c in plan["cases"] if c["kind"] == "attention"}
    done = None
    for progress in sorted(output.glob("progress-rank-*.jsonl")):
        ids = {json.loads(line)["case_id"] for line in progress.read_text().splitlines() if line.strip()}
        done = ids if done is None else done & ids
    return (done or set()) & planned


def mark_case(output: Path, rank: int, case_id: str | None) -> None:
    """Leave the id of the running attention case in ``current-rank-N.case`` so an attempt that dies
    without writing a receipt (SIGABRT after an illegal access, torchrun SIGTERM) still names it."""
    marker = output / f"current-rank-{rank}.case"
    if case_id is None:
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(case_id)


def failed_cases(output: Path) -> dict[str, str]:
    """case_id -> error of the attention cases that killed preserved attempts of this run: the
    ``failed_case`` of the archived ``rank-*.attempt-*.json`` receipts (unparseable receipts of hard-killed
    ranks are skipped) and the leftover ``current-rank-*.case`` markers. Resuming skips and records them."""
    failed: dict[str, str] = {}
    for receipt_path in sorted(output.glob("rank-*.attempt-*.json")):
        try:
            receipt = json.loads(receipt_path.read_text())
        except json.JSONDecodeError:
            continue  # the rank was killed while writing it; the marker below still names the case
        case = receipt.get("failed_case")
        if case:
            failed.setdefault(case, receipt.get("error", "unknown error"))
    for marker in sorted(output.glob("current-rank-*.case")):
        case = marker.read_text().strip()
        if case:
            failed.setdefault(case, f"attempt died without a receipt while running it ({marker.name})")
    return failed


class RowStream:
    def __init__(self, path: Path, *, plan: dict, provenance: dict, rank: int, keep_cases: set[str] | None = None):
        """``keep_cases`` (resume): rewrite an existing row file keeping only the rows of those case ids
        (partial rows of the interrupted case and token-component rows are re-measured), then append."""
        self.rows = 0
        if keep_cases is not None and path.exists():
            kept = [
                line
                for line in path.read_text().splitlines()
                if line.strip() and json.loads(line)["case_id"] in keep_cases
            ]
            path.write_text("".join(line + "\n" for line in kept))
            self.rows = len(kept)
            self.stream = path.open("a")
        else:
            self.stream = path.open("x")
        self.plan, self.provenance, self.rank = plan, provenance, rank
        self.output = path.parent

    def write(
        self, entry: dict, case: dict, *, latency: float, kernel_source: str, used_cuda_graph: bool, sample: int
    ) -> None:
        regime, expected_graph = regime_for(self.plan, entry["component"], case["phase"])
        if used_cuda_graph != expected_graph:
            raise RuntimeError(
                f"{entry['component']} {case['case_id']}: measured with used_cuda_graph={used_cuda_graph}, "
                f"plan regime {regime} requires {expected_graph}"
            )
        row = make_row(
            entry,
            case,
            tp_size=self.plan["tp_size"],
            latency=latency,
            kernel_source=kernel_source,
            regime=regime,
            used_cuda_graph=used_cuda_graph,
        )
        row.update(
            self.provenance,
            sample=sample,
            invocation=case["index"],
            tp_rank=self.rank,
            case_id=case["case_id"],
            collection_purpose=self.plan["purpose"],
        )
        validate_row(row)
        self.stream.write(json.dumps(row) + "\n")
        self.rows += 1

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def new_receipt(rank: int, backend: str) -> dict:
    return dict(
        schema=RECEIPT_SCHEMA,
        state="started",
        tp_rank=rank,
        framework=backend,
        started_unix_ns=time.time_ns(),
        checkpoint_weights_loaded=False,
        publication_permitted=False,
        regime_violations=[],
    )


def write_receipt(path: Path, receipt: dict) -> None:
    receipt["finished_unix_ns"] = time.time_ns()
    # atomic: a rank killed mid-write must not leave a truncated receipt behind
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)
