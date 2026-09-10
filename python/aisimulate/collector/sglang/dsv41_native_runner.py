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
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .dsv41_workloads import baseline_tokens, coordinates, freeze_workloads
else:
    from dsv41_workloads import baseline_tokens, coordinates, freeze_workloads


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
            expected = [entry for entry in manifest["phases"]["context"] if entry["layer"] == layer_id]
            attention_shape = json.loads(next(e["geometry"] for e in expected if e["component"] == "attention"))
            for field, attribute in (
                ("num_heads", "n_local_heads"),
                ("o_groups", "n_local_groups"),
                ("head_dim", "head_dim"),
                ("q_lora_rank", "q_lora_rank"),
                ("o_lora_rank", "o_lora_rank"),
                ("compress_ratio", "compress_ratio"),
            ):
                if int(getattr(layer.self_attn, attribute)) != attention_shape[field]:
                    raise RuntimeError(f"native attention {attribute} differs from graph at layer {layer_id}")
            if not layer.hc_pre_from_prev_sublayer:
                raise RuntimeError("native mHC must use predecessor single-pass mixing")
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
            for module, name, n_attr, k_attr in (
                (shared.gate_up_proj, "gate_up", "output_size_per_partition", "input_size"),
                (shared.down_proj, "ffn2", "output_size", "input_size_per_partition"),
            ):
                geometry = json.loads(
                    next(e["geometry"] for e in expected if e["component"] == "linear" and name in e["name"])
                )
                if (getattr(module, n_attr), getattr(module, k_attr)) != (geometry["n"], geometry["k"]):
                    raise RuntimeError(f"native shared projection geometry differs at layer {layer_id}")
                block = module.quant_method.quant_config.weight_block_size
                if tuple(block) != (32, 32):
                    raise RuntimeError("native shared projection must retain FP8 block-32 checkpoint contract")
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
            b, p, x = coordinates(component, geometry, self.phase, batch, query, prefix)
            rows.append(
                {
                    "component": component,
                    "geometry": geometry_json,
                    "batch_size": b,
                    "prefix": p,
                    "x": x,
                    "latency": latency,
                    "kernel_source": "+".join(sorted(sources)),
                    "measurement_scope": "local_compute",
                    "used_cuda_graph": False,
                    "sample_count": 1,
                    "kv_seed_regime": "real_kv" if component == "attention" and (real_kv or p > 0) else "n/a",
                }
            )
        return rows


def collect_native_baselines(runner, options, tp_rank, provenance):
    """Benchmark loaded native kernels with explicit uniform synthetic routing.

    This separate op sweep does not change real-request routing or KV state.
    All ranks use the same seeded input and expert IDs. Communication calls
    use the actual NCCL process group rather than framework custom all-reduce.
    """
    import torch
    import torch.distributed as dist
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    layers = [m for m in runner.model.modules() if type(m).__name__ == "DeepseekV4DecoderLayer"]
    experts = layers[2].mlp.experts
    gate = layers[2].mlp.gate
    lm_head = runner.model.lm_head
    if tuple(gate.weight.shape) != (384, 5120) or tuple(lm_head.weight.shape) != (32320, 5120):
        raise RuntimeError("native baseline GEMM physical padding differs from TP4 graph")
    if type(experts.quant_method).__name__ != "Mxfp4FlashinferTrtllmMoEMethod":
        raise RuntimeError("native baseline MoE requires verified MXFP4 TRTLLM dispatch")
    if experts.quant_method.flashinfer_mxfp4_moe_precision != "default":
        raise RuntimeError("native baseline MoE requires MXFP8 activations")
    if getattr(experts, "reduce_results", False):
        raise RuntimeError("native expert kernel unexpectedly owns a collective")
    recorder_path = Path(options.output) / f"baseline-rank-{tp_rank}.jsonl"
    token_counts = (
        baseline_tokens(options.workload_plan["cases"])
        if options.workload_plan is not None
        else sorted({1, 2, *(b * n for b in options.batches for n in options.lengths)})
    )
    generator = torch.Generator().manual_seed(20260910)
    with recorder_path.open("w") as stream:
        for tokens in token_counts:
            hidden = torch.randn(tokens, 5120, generator=generator, dtype=torch.bfloat16).cuda()
            logits = torch.rand(tokens, 384, generator=generator, dtype=torch.float32).cuda()
            ids = logits.topk(6, dim=-1).indices.to(torch.int32)
            weights = torch.full((tokens, 6), 1 / 6, dtype=torch.float32, device=hidden.device)
            topk = StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=logits)
            routing_histogram = torch.bincount(ids.flatten().long(), minlength=384).cpu().tolist()
            cases = [
                (
                    "gemm",
                    {"gemm_dtype": "bfloat16", "m": tokens, "n": 384, "k": 5120},
                    lambda: gate(hidden),
                    _dispatch(gate, "forward"),
                ),
                (
                    "gemm",
                    {"gemm_dtype": "bfloat16", "m": tokens, "n": 32320, "k": 5120},
                    lambda: lm_head.quant_method.apply(lm_head, hidden),
                    _dispatch(lm_head, "quant_method.apply"),
                ),
                (
                    "moe",
                    {
                        "moe_dtype": "w4a8_mxfp4_mxfp8",
                        "num_tokens": tokens,
                        "hidden_size": 5120,
                        "inter_size": 2304,
                        "topk": 6,
                        "num_experts": 384,
                        "moe_tp_size": 4,
                        "moe_ep_size": 1,
                        "distribution": "uniform",
                    },
                    lambda: experts(hidden, topk),
                    "sglang_mxfp4_flashinfer_trtllm_moe",
                ),
            ]
            for width in (5120, 6144):
                payload = torch.zeros(tokens, width, dtype=torch.bfloat16, device=hidden.device)
                cases.append(
                    (
                        "nccl",
                        {
                            "op_name": "all_reduce",
                            "nccl_dtype": "half",
                            "num_gpus": 4,
                            "message_size": 2 * tokens * width,
                        },
                        lambda payload=payload: dist.all_reduce(payload),
                        "torch.distributed.nccl.all_reduce",
                    )
                )
            for kind, key, call, dispatch in cases:
                for sample in range(2 + options.iterations):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    call()
                    end.record()
                    torch.cuda.synchronize()
                    if sample >= 2:
                        stream.write(
                            json.dumps(
                                {
                                    "kind": kind,
                                    **key,
                                    "latency": start.elapsed_time(end),
                                    "sample": sample,
                                    "tp_rank": tp_rank,
                                    "kernel_source": dispatch,
                                    "used_cuda_graph": False,
                                    "routing_seed": 20260910,
                                    "routing_histogram": routing_histogram if kind == "moe" else None,
                                    "physical_local_intermediate": int(experts.w2_weight.shape[-1]) * 2,
                                    **provenance,
                                }
                            )
                            + "\n"
                        )


def run_workload(runner, recorder, bench, token_ids, case, execute, *, decode_steps=0):
    """Create real state on one request and measure only its declared forward."""
    batch_size, query, prefix = case["batch_size"], case["query"], case["prefix"]
    runner.clear()
    recorder.active = False
    recorder.phase = "context"
    initial = prefix or query
    reqs = bench.prepare_synthetic_inputs_for_latency_test(
        batch_size, initial, [token_ids[:initial] for _ in range(batch_size)]
    )
    if case["phase"] == "generation":
        next_ids, _, batch = runner.extend(reqs)
        execute(lambda: runner.decode(next_ids, batch), "generation", batch_size, 1, prefix, True)
    else:
        if prefix:
            runner.extend(reqs)
            bench.prepare_extend_inputs_for_correctness_test(
                argparse.Namespace(cut_len=prefix),
                [token_ids[: prefix + query] for _ in reqs],
                reqs,
                runner.torch_runner,
            )
        next_ids, _, batch = execute(lambda: runner.extend(reqs), "context", batch_size, query, prefix, bool(prefix))
        for step in range(decode_steps):
            next_ids, _ = execute(
                lambda: runner.decode(next_ids, batch), "generation", batch_size, 1, prefix + query + step, True
            )
    runner.cleanup(batch)


def validate_forward_contract(runner, manifest):
    """Check the serving layout without altering any model or stream method."""
    layers = [m for m in runner.model.modules() if type(m).__name__ == "DeepseekV4DecoderLayer"]
    if len(layers) != 40:
        raise RuntimeError("forward-only benchmark requires the 40-layer text backbone")
    for layer_id, layer in enumerate(layers):
        expected = [e for e in manifest["phases"]["context"] if e["layer"] == layer_id]
        geometry = json.loads(next(e["geometry"] for e in expected if e["component"] == "attention"))
        for field, attribute in (
            ("num_heads", "n_local_heads"),
            ("o_groups", "n_local_groups"),
            ("head_dim", "head_dim"),
            ("q_lora_rank", "q_lora_rank"),
            ("o_lora_rank", "o_lora_rank"),
            ("compress_ratio", "compress_ratio"),
        ):
            if int(getattr(layer.self_attn, attribute)) != geometry[field]:
                raise RuntimeError(f"forward-only attention layout differs at layer {layer_id}")
        if not layer.hc_pre_from_prev_sublayer:
            raise RuntimeError("forward-only benchmark requires native predecessor mHC")
        shared = layer.mlp.shared_experts
        if shared is None or getattr(layer.mlp, "_shared_expert_tp1", False):
            raise RuntimeError("forward-only benchmark requires TP-sharded shared experts")
        for module in (shared.gate_up_proj, shared.down_proj):
            if tuple(module.quant_method.quant_config.weight_block_size) != (32, 32):
                raise RuntimeError("forward-only shared projection must retain FP8 block32")
        engram = getattr(layer, "engram", None)
        if engram is not None and (engram.embed.host_table is not None or engram.embed._shared):
            raise RuntimeError("forward-only Engram must use TP-sharded GPU tables")


def timed_native_forward(runner, call):
    """Use one_batch.py:793-799/827-843's native synchronized wall boundary."""
    runner.synchronize()
    start = time.perf_counter()
    result = call()
    runner.synchronize()
    return result, (time.perf_counter() - start) * 1000


def run_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    import torch
    import torch.distributed as dist
    from sglang.benchmark import one_batch as bench

    options = bench_args.dsv41_options
    manifest = json.loads(Path(options.manifest).read_text())
    config_path = Path(server_args.model_path) / "config.json"
    actual_config = hashlib.sha256(
        json.dumps(json.loads(config_path.read_text()), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_config != manifest["config_sha256"]:
        raise RuntimeError("checkpoint config differs from native graph manifest")
    bench.publish(server_args, role="scheduler")
    # Native benchmark latency_test at one_batch.py:896-899 seeds the
    # process-local dispatch flags after publication in every spawned rank.
    # Omitting these initializers makes the MoE accessor default to AUTO
    # even when the published configuration selects flashinfer_mxfp4.
    bench.initialize_moe_config()
    bench.initialize_fp8_gemm_config()
    bench.initialize_fp4_gemm_config()
    bench.configure_logger(server_args, prefix=f" TP{tp_rank}")
    from sglang.srt.layers.moe import get_moe_runner_backend
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    original_selector = Fp8Config.get_quant_method

    def observe_selector(quant_config, layer, prefix):
        method = original_selector(quant_config, layer, prefix)
        if prefix.endswith(".experts"):
            print(
                json.dumps(
                    {
                        "event": "native_expert_dispatch",
                        "tp_rank": tp_rank,
                        "prefix": prefix,
                        "is_fp4_experts": quant_config.is_fp4_experts,
                        "runner_backend": str(get_moe_runner_backend()),
                        "method": type(method).__module__ + "." + type(method).__name__,
                    }
                ),
                flush=True,
            )
        return method

    Fp8Config.get_quant_method = observe_selector
    runner, tokenizer = bench.load_model(server_args, port_args, gpu_id, tp_rank)
    Fp8Config.get_quant_method = original_selector
    if options.forward_only:
        validate_forward_contract(runner.torch_runner, manifest)
        recorder = SimpleNamespace(active=False, phase="context", trace=False)
    else:
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
    text = Path(options.prompt_file).read_text()
    token_ids = tokenizer.encode(text)
    required_tokens = (
        max(c["query"] + c["prefix"] for c in options.workload_plan["cases"])
        if options.workload_plan is not None
        else max(options.lengths) + max(options.prefixes)
    )
    if len(token_ids) < required_tokens:
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
        if options.forward_only:
            result, native_forward_ms = timed_native_forward(runner, call)
        else:
            forward_start, forward_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            forward_start.record()
            with profile if profile is not None else nullcontext():
                result = call()
            forward_end.record()
        if profile is not None:
            profile.export_chrome_trace(str(raw_path.parent / f"canary-{phase}.trace.json"))
            kernels = sorted(
                {event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
            )
            (raw_path.parent / f"canary-{phase}.kernels.json").write_text(json.dumps(kernels))
        rows = [] if options.forward_only else recorder.finish(batch_size, query, prefix, real_kv)
        logits = result[1]
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("native forward produced nonfinite logits")
        if recorder.active and options.forward_only:
            with (raw_path.parent / f"forward-rank-{tp_rank}.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "invocation": invocation,
                            "sample": sample,
                            "tp_rank": tp_rank,
                            "phase": phase,
                            "batch_size": batch_size,
                            "query": query,
                            "prefix": prefix,
                            "real_kv": real_kv,
                            **(
                                {"canonical_past_kv": prefix, "native_inclusive_kv": prefix + 1}
                                if phase == "generation"
                                else {}
                            ),
                            "finite_logits": True,
                            "native_benchmark_forward_ms": native_forward_ms,
                            "timing_boundary": "sglang_one_batch_synchronized_wall_including_prepare_forward_sample",
                            "component_recorder": False,
                            "used_cuda_graph": False,
                            **provenance,
                        }
                    )
                    + "\n"
                )
        elif recorder.active:
            with (raw_path.parent / f"invocations-rank-{tp_rank}.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "invocation": invocation,
                            "sample": sample,
                            "tp_rank": tp_rank,
                            "phase": phase,
                            "batch_size": batch_size,
                            "query": query,
                            "prefix": prefix,
                            "real_kv": real_kv,
                            **(
                                {"canonical_past_kv": prefix, "native_inclusive_kv": prefix + 1}
                                if phase == "generation"
                                else {}
                            ),
                            "finite_logits": True,
                            "instrumented_forward_ms": forward_start.elapsed_time(forward_end),
                            **provenance,
                        }
                    )
                    + "\n"
                )
            with raw_path.open("a") as out:
                for row in rows:
                    out.write(
                        json.dumps(
                            {**row, **provenance, "sample": sample, "invocation": invocation, "tp_rank": tp_rank}
                        )
                        + "\n"
                    )
        return result

    if options.workload_plan is not None:
        cases = options.workload_plan["cases"]
    else:
        cases = [
            {"phase": "context", "batch_size": b, "query": q, "prefix": p}
            for b in options.batches
            for p in options.prefixes
            for q in options.lengths
        ]
    for case_index, case in enumerate(cases):
        for sample in range(options.warmup + options.iterations):
            started = time.monotonic()
            progress = {
                **case,
                "case_index": case_index,
                "sample": sample,
                "tp_rank": tp_rank,
                "measured": sample >= options.warmup,
            }
            try:
                run_workload(
                    runner,
                    recorder,
                    bench,
                    token_ids,
                    case,
                    lambda call, phase, batch, query, prefix, real_kv: execute(
                        call, phase, batch, query, prefix, real_kv, sample
                    ),
                    decode_steps=options.decode_steps if options.workload_plan is None else 0,
                )
            except BaseException as error:
                progress.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            else:
                progress["status"] = "passed"
            finally:
                progress["elapsed_seconds"] = time.monotonic() - started
                with (raw_path.parent / f"workloads-rank-{tp_rank}.jsonl").open("a") as out:
                    out.write(json.dumps(progress) + "\n")
    if options.collect_baselines:
        recorder.active = False
        collect_native_baselines(runner.torch_runner, options, tp_rank, provenance)
    if dist.is_initialized():
        dist.barrier()
    if tp_rank == 0:
        (raw_path.parent / "COMPLETE").write_text("native workload collection completed\n")


def main():
    from sglang.benchmark import one_batch as bench

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--workload-plan", help="Frozen explicit module workload plan; replaces the legacy cartesian grid"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 128, 129, 256])
    parser.add_argument("--prefixes", type=int, nargs="+", default=[0, 256])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--profile-canary", action="store_true")
    parser.add_argument("--collect-baselines", action="store_true")
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Native synchronized benchmark wall timing, without component interception",
    )
    options, rest = parser.parse_known_args()
    if options.warmup < 0 or options.iterations < 1 or options.decode_steps < 0:
        raise ValueError("warmup/decode steps must be nonnegative and iterations positive")
    if options.workload_plan is not None:
        payload = json.loads(Path(options.workload_plan).read_text())
        if payload != freeze_workloads(payload["source_payload"]):
            raise ValueError("workload plan differs from its frozen source points")
        options.workload_plan = payload
    if options.forward_only and (options.workload_plan is None or options.collect_baselines or options.profile_canary):
        raise ValueError("forward-only requires an explicit plan and excludes component baselines/profiling")
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    owned_files = (
        "COMPLETE",
        "source_hashes.json",
        "input_provenance.json",
        "workload-plan.json",
        "execution-contract.json",
    )
    if (
        any((output / name).exists() for name in owned_files)
        or any(output.glob("*-rank-*.jsonl"))
        or any(output.glob("rank-*.jsonl"))
    ):
        raise RuntimeError("output has prior raw records; use a fresh attempt directory")
    if options.workload_plan is not None:
        (output / "workload-plan.json").write_text(json.dumps(options.workload_plan, sort_keys=True) + "\n")
    native = argparse.ArgumentParser()
    bench.ServerArgs.add_cli_args(native)
    bench.BenchArgs.add_cli_args(native)
    args = native.parse_args(rest)
    if options.forward_only and not all(
        getattr(args, name, False)
        for name in (
            "disable_custom_all_reduce",
            "enforce_disable_flashinfer_allreduce_fusion",
            "disable_shared_experts_fusion",
        )
    ):
        raise ValueError("forward-only requires unfused NCCL and sharded shared-expert execution flags")
    (output / "execution-contract.json").write_text(
        json.dumps(
            {
                "mode": "native_benchmark_forward" if options.forward_only else "local_components",
                "native_cli_args": rest,
                "component_recorder": not options.forward_only,
                "warmup": options.warmup,
                "iterations": options.iterations,
                "timing_boundary": "sglang_one_batch_synchronized_wall_including_prepare_forward_sample"
                if options.forward_only
                else "cuda_events_local_components",
            },
            sort_keys=True,
        )
        + "\n"
    )
    server_args, bench_args = bench.ServerArgs.from_cli_args(args), bench.BenchArgs.from_cli_args(args)
    bench_args.dsv41_options = options
    bench.latency_test = run_worker
    bench.main(server_args, bench_args)
    if not (Path(options.output) / "COMPLETE").exists():
        raise RuntimeError("native worker failed; inspect preserved rank logs")


if __name__ == "__main__":
    main()
