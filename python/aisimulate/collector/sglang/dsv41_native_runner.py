# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure native V4.1 components while executing real text requests.

Framework integration: sgl-project/sglang@1aa0e962b206102b7c439a4a0c4981cfec6e87bc,
python/sglang/benchmark/one_batch.py (Apache-2.0). This original adapter calls
the upstream loader and request lifecycle, rather than copying metadata logic.
See README.dsv41.md and the repository third-party notices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path


def _dispatch(module, method):
    quant = getattr(module, "quant_method", None)
    suffix = "" if quant is None else f"/{type(quant).__module__}.{type(quant).__name__}"
    return f"{type(module).__module__}.{type(module).__name__}.{method}{suffix}"


class ComponentRecorder:
    """CUDA event intervals exclude each separately modeled collective.

    mHC side-stream overlap is disabled explicitly for this eager leaf
    measurement, preserving the framework-selected scalar kernels. Summing
    the two mix/combine and post sites measures the native two-site op.
    """

    def __init__(self, runner, manifest):
        import torch
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        self.torch = torch
        self.manifest = manifest
        self.events = []
        self.phase = "context"
        self.active = False
        self.trace = False
        layers = [m for m in runner.model.modules() if type(m).__name__ == "DeepseekV4DecoderLayer"]
        if len(layers) != 40:
            raise RuntimeError(f"expected 40 actual text decoder layers, got {len(layers)}")
        self.selected = {}
        for phase, entries in manifest["phases"].items():
            for entry in entries:
                if entry["component"] == "mhc" and entry["layer"] < 2:
                    # First-layer attention copies stream zero because there
                    # is no predecessor pre-mix; normal layers combine it.
                    continue
                key = (entry["component"], entry["geometry"])
                self.selected.setdefault((phase, key), entry["layer"])

        def wrap(module, method, layer_id, component, selector=None, after=None):
            original = getattr(module, method)
            witness = _dispatch(module, method)

            def timed(*args, **kwargs):
                entries = [
                    e
                    for e in manifest["phases"][self.phase]
                    if e["layer"] == layer_id
                    and e["component"] == component
                    and (selector is None or selector in e["name"])
                    and self.selected[(self.phase, (component, e["geometry"]))] == layer_id
                ]
                record = self.active and bool(entries)
                if record:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                with (
                    torch.profiler.record_function(f"dsv41::{component}::{layer_id}::{method}")
                    if self.trace
                    else nullcontext()
                ):
                    result = original(*args, **kwargs)
                if record:
                    end.record()
                    self.events.append((entries[0], start, end, witness))
                return after(result) if after is not None else result

            setattr(module, method, timed)

        for layer_id, layer in enumerate(layers):
            # Upstream deepseek_v4.py:2649-2788 dispatches the same mHC
            # functions with optional stats_stream. None serializes local work.
            layer.hc_stats_stream = None
            wrap(layer, "_hc_mix_and_combine", layer_id, "mhc")
            wrap(layer, "hc_post", layer_id, "mhc")
            attn = layer.self_attn
            # Upstream deepseek_v4.py:726,781-793 exposes this exact boundary:
            # wo_b_reduce_results=False omits output reduction. We execute the
            # unchanged reduction immediately after the timed local interval.
            if manifest["tp_size"] > 1 and not getattr(attn.wo_b, "reduce_results", False):
                raise RuntimeError("attention output reduction ownership differs from pure TP contract")
            attn.wo_b.reduce_results = False
            wrap(
                attn,
                "forward",
                layer_id,
                "attention",
                after=tensor_model_parallel_all_reduce if manifest["tp_size"] > 1 else None,
            )
            shared = layer.mlp.shared_experts
            if shared is None or getattr(layer.mlp, "_shared_expert_tp1", False):
                raise RuntimeError("native shared-expert TP layout does not match sharded SOL graph")
            wrap(shared.gate_up_proj, "forward", layer_id, "linear", selector="gate_up")
            wrap(shared.down_proj, "forward", layer_id, "linear", selector="ffn2")
            engram = getattr(layer, "engram", None)
            if engram is not None:
                if engram.embed.host_table is not None or engram.embed._shared:
                    raise RuntimeError("Engram must use GPU TP-sharded storage")
                # Upstream engram.py:780-789 performs the all-reduce between
                # _owned_rows and wkv; it stays outside all three intervals.
                wrap(engram.embed, "_owned_rows", layer_id, "engram")
                wrap(engram.wkv, "forward", layer_id, "engram")
                wrap(engram, "apply_gate", layer_id, "engram")

    def finish(self, batch, query, prefix, real_kv):
        self.torch.cuda.synchronize()
        grouped = {}
        for entry, start, end, witness in self.events:
            key = (entry["component"], entry["geometry"])
            value = grouped.setdefault(key, [0.0, set()])
            value[0] += start.elapsed_time(end)
            value[1].add(witness)
        self.events.clear()
        rows = []
        for (component, geometry_json), (latency, sources) in grouped.items():
            geometry = json.loads(geometry_json)
            q, p = query, prefix
            if component == "attention" and geometry["is_context"] and geometry["bounded_prefill"]:
                q = min(query, geometry["window_size"])
                p += query - q
            rows.append(
                {
                    "component": component,
                    "geometry": geometry_json,
                    "batch_size": batch if component == "attention" else 1,
                    "prefix": p if component == "attention" and self.phase == "context" else 0,
                    "x": (q if self.phase == "context" else query + prefix)
                    if component == "attention"
                    else batch * query,
                    "latency": latency,
                    "kernel_source": "+".join(sorted(sources)),
                    "measurement_scope": "local_compute",
                    "used_cuda_graph": False,
                    "sample_count": 1,
                    "kv_seed_regime": "real_kv" if component == "attention" and (real_kv or p > 0) else "n/a",
                }
            )
        return rows


def run_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    import torch
    import torch.distributed as dist
    from sglang.benchmark import one_batch as bench

    options = bench_args.dsv41_options
    manifest = json.loads(Path(options.manifest).read_text())
    bench.publish(server_args, role="scheduler")
    bench.configure_logger(server_args, prefix=f" TP{tp_rank}")
    runner, tokenizer = bench.load_model(server_args, port_args, gpu_id, tp_rank)
    recorder = ComponentRecorder(runner.torch_runner, manifest)
    package_root = Path(bench.__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provenance = {
        "source_sha256": source_digest,
        "config_sha256": manifest["config_sha256"],
        "runtime_digest": options.runtime_digest,
        "execution_profile": manifest["execution_profile"],
    }
    config_path = Path(server_args.model_path) / "config.json"
    actual_config = hashlib.sha256(
        json.dumps(json.loads(config_path.read_text()), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_config != manifest["config_sha256"]:
        raise RuntimeError("checkpoint config differs from native graph manifest")
    text = Path(options.prompt_file).read_text()
    token_ids = tokenizer.encode(text)
    if len(token_ids) < max(options.lengths) + max(options.prefixes):
        raise RuntimeError("tokenized input corpus is too short for requested workloads")
    raw_path = Path(options.output) / f"rank-{tp_rank}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if tp_rank == 0:
        (raw_path.parent / "source_hashes.json").write_text(json.dumps(sources, sort_keys=True))
        (raw_path.parent / "input_provenance.json").write_text(
            json.dumps(
                {
                    "source": "tokenizer_text",
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                    "unique_token_count": len(set(token_ids)),
                    "token_count": len(token_ids),
                    **provenance,
                }
            )
        )

    invocation = 0

    def execute(call, phase, batch_size, query, prefix, real_kv, sample):
        nonlocal invocation
        invocation += 1
        recorder.phase, recorder.active = phase, sample >= options.warmup
        recorder.trace = options.profile_canary and invocation <= 2 and tp_rank == 0
        profile = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            if recorder.trace
            else None
        )
        with profile if profile is not None else nullcontext():
            result = call()
        if profile is not None:
            profile.export_chrome_trace(str(raw_path.parent / f"canary-{phase}.trace.json"))
            kernels = sorted(
                {event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
            )
            (raw_path.parent / f"canary-{phase}.kernels.json").write_text(json.dumps(kernels))
        rows = recorder.finish(batch_size, query, prefix, real_kv)
        logits = result[1]
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("native forward produced nonfinite logits")
        if recorder.active:
            with raw_path.open("a") as out:
                for row in rows:
                    out.write(
                        json.dumps(
                            {**row, **provenance, "sample": sample, "invocation": invocation, "tp_rank": tp_rank}
                        )
                        + "\n"
                    )
        return result

    for batch_size in options.batches:
        for prefix in options.prefixes:
            for length in options.lengths:
                for sample in range(options.warmup + options.iterations):
                    runner.clear()
                    recorder.active = False
                    initial = prefix or length
                    reqs = bench.prepare_synthetic_inputs_for_latency_test(
                        batch_size, initial, [token_ids[:initial] for _ in range(batch_size)]
                    )
                    if prefix:
                        recorder.phase = "context"
                        runner.extend(reqs)
                        temporary = argparse.Namespace(cut_len=prefix)
                        bench.prepare_extend_inputs_for_correctness_test(
                            temporary, [token_ids[: prefix + length] for _ in reqs], reqs, runner.torch_runner
                        )
                    next_ids, _, batch = execute(
                        lambda: runner.extend(reqs), "context", batch_size, length, prefix, bool(prefix), sample
                    )
                    for step in range(options.decode_steps):
                        next_ids, _ = execute(
                            lambda: runner.decode(next_ids, batch),
                            "generation",
                            batch_size,
                            1,
                            prefix + length + step,
                            True,
                            sample,
                        )
                    runner.cleanup(batch)
    if dist.is_initialized():
        dist.barrier()
    if tp_rank == 0:
        (raw_path.parent / "COMPLETE").write_text("native component collection completed\n")


def main():
    from sglang.benchmark import one_batch as bench

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 128, 129, 256])
    parser.add_argument("--prefixes", type=int, nargs="+", default=[0, 256])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--profile-canary", action="store_true")
    options, rest = parser.parse_known_args()
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists() or list(output.glob("rank-*.jsonl")):
        raise RuntimeError("output has prior raw records; use a fresh attempt directory")
    native = argparse.ArgumentParser()
    bench.ServerArgs.add_cli_args(native)
    bench.BenchArgs.add_cli_args(native)
    args = native.parse_args(rest)
    server_args, bench_args = bench.ServerArgs.from_cli_args(args), bench.BenchArgs.from_cli_args(args)
    bench_args.dsv41_options = options
    bench.latency_test = run_worker
    bench.main(server_args, bench_args)
    if not (Path(options.output) / "COMPLETE").exists():
        raise RuntimeError("native worker failed; inspect preserved rank logs")


if __name__ == "__main__":
    main()
