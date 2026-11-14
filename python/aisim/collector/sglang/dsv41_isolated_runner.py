# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect checkpoint-shaped native V4.1 modules with dummy weights.

Original API adapter for sgl-project/sglang at
1aa0e962b206102b7c439a4a0c4981cfec6e87bc: benchmark/one_batch.py,
srt/distributed/bootstrap.py, srt/model_loader/loader.py, models/deepseek_v2.py,
layers/engram.py and layers/vocab_parallel_embedding.py. No upstream
implementation is copied. This measures isolated operators, not whole-model
execution, logits, attention coverage, or FPM accuracy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from .dsv41_contract import canonical_json, validate_row
from .dsv41_native_runner import _dispatch, collect_native_kernel_baselines

FRAMEWORK_COMMIT = "1aa0e962b206102b7c439a4a0c4981cfec6e87bc"
WEIGHT_INITIALIZER = {
    "name": "native_random_chunks_v1",
    "chunk_elements": 8 * 1024 * 1024,
    "float_low": -1e-3,
    "float_high": 1e-3,
    "packed_byte_low": 0,
    "packed_byte_high": 256,
    "unit_scales": True,
}
REQUIRED_SOURCES = {
    "benchmark/one_batch.py",
    "srt/distributed/bootstrap.py",
    "srt/model_loader/loader.py",
    "srt/model_loader/weight_utils.py",
    "srt/models/deepseek_v2.py",
    "srt/models/deepseek_v4.py",
    "srt/layers/engram.py",
    "srt/model_executor/forward_batch_info.py",
    "srt/utils/hf_transformers_utils.py",
    "srt/layers/quantization/fp8.py",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan, manifest):
    """Validate declared ownership before importing any serving/GPU library."""
    if plan["schema"] != "dsv41.isolated-collection.v1" or plan["tp_size"] not in (2, 4):
        raise ValueError("unsupported isolated collection plan")
    if plan["purpose"] not in ("smoke", "calibration"):
        raise ValueError("collection purpose must be declared before measurements")
    if plan.get("weight_initializer") != WEIGHT_INITIALIZER:
        raise ValueError("prospectively declared native random-weight initializer required")
    if plan["tp_size"] != manifest["tp_size"] or plan["execution_profile"] != manifest["execution_profile"]:
        raise ValueError("isolated plan and consumer manifest differ")
    tokens = plan["token_counts"]
    if not tokens or tokens != sorted(set(tokens)) or any(type(n) is not int or not 1 <= n <= 8192 for n in tokens):
        raise ValueError("token counts must be distinct increasing integers in 1..8192")
    components = plan["components"]
    if (
        not components
        or len(set(components)) != len(components)
        or not set(components) <= {"baselines", "linear", "engram", "mhc"}
    ):
        raise ValueError("isolated plan contains unsupported or duplicate components")
    if plan["warmup"] < 2 or plan["iterations"] < 5:
        raise ValueError("isolated collection requires warmup and repeated measurements")
    if plan["moe_runner_backend"] not in ("flashinfer_mxfp4", "humming"):
        raise ValueError("explicit qualified native MoE backend required")
    expected_sm = {"H100": 90, "H200": 90, "B200": 100, "GB200": 100}
    if plan["expected_gpu"] not in expected_sm or plan["expected_sm"] != expected_sm[plan["expected_gpu"]]:
        raise ValueError("GPU and SM identity differ")
    if plan["framework_commit"] != FRAMEWORK_COMMIT:
        raise ValueError("serving APIs require the pinned framework commit")
    if re.fullmatch(r"[0-9a-f]{40}", plan.get("collector_revision", "")) is None:
        raise ValueError("immutable collector source revision required")
    if not plan["source_pins"].keys() >= REQUIRED_SOURCES:
        raise ValueError("missing native source identity pins")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= plan["metadata_pins"].keys():
        raise ValueError("missing checkpoint/tokenizer identity pins")
    digests = [plan.get("image_sha256", ""), *plan["source_pins"].values(), *plan["metadata_pins"].values()]
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
        raise ValueError("immutable image/source/metadata SHA-256 pins required")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", plan["runtime_digest"]) is None:
        raise ValueError("immutable runtime image digest required")
    if manifest["config_sha256"] != "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d":
        raise ValueError("only the unchanged V4.1 checkpoint architecture is qualified")


def aggregate_isolated_records(output, plan_path, manifest_path):
    """Require complete declared operator coverage before writing perf tables."""
    from .collect_dsv41_module import aggregate_baseline_records, aggregate_rank_records

    plan, manifest = json.loads(plan_path.read_text()), json.loads(manifest_path.read_text())
    validate_plan(plan, manifest)
    if plan["purpose"] != "calibration":
        raise ValueError("smoke measurements cannot be published as calibration data")
    tp = plan["tp_size"]
    expected_sources = None
    for rank in range(tp):
        receipt = json.loads((output / f"isolated-rank-{rank}.json").read_text())
        if (
            receipt.get("state") != "complete_pending_admission"
            or receipt.get("tp_rank") != rank
            or receipt.get("plan_sha256") != sha(plan_path)
            or receipt.get("manifest_sha256") != sha(manifest_path)
            or receipt.get("runtime_digest") != plan["runtime_digest"]
            or receipt.get("purpose") != "calibration"
            or receipt.get("full_model") is not False
            or receipt.get("checkpoint_weights_loaded") is not False
        ):
            raise ValueError("isolated run receipt differs from the declared measurement")
        sources = receipt.get("source_hashes", {})
        if any(sources.get(path) != digest for path, digest in plan["source_pins"].items()):
            raise ValueError("isolated native source receipt differs from the frozen plan")
        if expected_sources is not None and expected_sources != sources:
            raise ValueError("isolated native source identity differs between ranks")
        expected_sources = sources
        expected_provenance = dict(
            source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
            config_sha256=manifest["config_sha256"],
            runtime_digest=plan["runtime_digest"],
            execution_profile=plan["execution_profile"],
            case_plan_sha256=sha(plan_path),
            collection_purpose="calibration",
        )
        filenames = [f"rank-{rank}.jsonl"]
        if "baselines" in plan["components"]:
            filenames.append(f"baseline-rank-{rank}.jsonl")
        for filename in filenames:
            for line in (output / filename).read_text().splitlines():
                row = json.loads(line)
                if any(row.get(key) != value for key, value in expected_provenance.items()):
                    raise ValueError("isolated raw measurement differs from its frozen plan and source receipt")
    rows = aggregate_rank_records([output / f"rank-{rank}.jsonl" for rank in range(tp)], tp)
    expected = {
        (e["component"], e["geometry"], n)
        for e in manifest["phases"]["context"]
        if e["component"] in plan["components"]
        for n in plan["token_counts"]
    }
    if {(row["component"], row["geometry"], row["x"]) for row in rows} != expected or any(
        row["sample_count"] != plan["iterations"] for row in rows
    ):
        raise ValueError("isolated native component coverage is incomplete or changed")
    baselines = {}
    if "baselines" in plan["components"]:
        baselines = aggregate_baseline_records([output / f"baseline-rank-{rank}.jsonl" for rank in range(tp)], tp)
        for kind, per_token in (("gemm", 2), ("moe", 1), ("nccl", 2)):
            values = baselines.get(kind, [])
            if len(values) != per_token * len(plan["token_counts"]) or any(
                row["sample_count"] != plan["iterations"] for row in values
            ):
                raise ValueError("isolated native baseline coverage is incomplete or changed")
    for row in rows:
        for key in ("case_plan_sha256", "collection_purpose"):
            row.pop(key)
    return rows, baselines


def prepare_private_caches(rank, receipt):
    """Run before native imports; preserve HOME while checking actual binds."""
    root = Path(os.environ["DSV41_PRIVATE_CACHE"])
    marker = root / "home-cache" / f"isolated-{os.environ['SLURM_JOB_ID']}-{rank}"
    marker.write_text("allocation-private\n")
    receipt["cache_bindings"] = []
    for target in (Path.home() / ".cache", Path("/root/.cache")):
        if not (target / marker.name).samefile(marker):
            raise RuntimeError("HOME/root cache does not share the allocation-private file")
        receipt["cache_bindings"].append({"path": str(target), "resolved": str(target.resolve())})
    for variable, leaf in (
        ("TRITON_CACHE_DIR", "triton"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("DEEP_GEMM_CACHE_DIR", "deep-gemm"),
    ):
        cache = root / f"rank-{rank}" / leaf
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    if os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE") != "env://":
        raise RuntimeError("torchrun requires native bootstrap env:// override")


def initialize_dummy_weights(module, plan):
    """Call the native random initializer on bounded views of native storage.

    SGLang@1aa0e962, srt/model_loader/weight_utils.py:1647-1682 requires
    nonconstant random weights and exposes low/high/seed. Views bound the FP8
    conversion temporary; per-chunk seeds are reproducible, not bit-identical
    to the full-parameter RNG stream. No native implementation is copied.
    """
    import torch
    from sglang.srt.model_loader.weight_utils import initialize_dummy_weights as native_initialize

    records = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_contiguous():
                raise RuntimeError("native dummy-storage initializer requires contiguous parameters: " + name)
            flat = parameter.view(-1)
            leaf = name.rsplit(".", 1)[-1]
            scale = "scale" in name or "alpha" in name
            if scale or leaf in ("q_weight", "k_weight"):
                # Native scale/gate storage contracts: fp8.py:1354-1377;
                # mxfp4_flashinfer_cutlass_moe.py:74-81; engram.py:696-704,
                # 923-924. fp8.py:618-633 marks raw UE8M0 uint8 scales.
                # Engram's scale is torch.empty, so positive one there is our
                # declared synthetic choice, not its constructor's value.
                # E8M0 exponent127 encodes1; signed uniform values are invalid.
                if parameter.dtype == torch.float8_e8m0fnu or (
                    parameter.dtype == torch.uint8 and getattr(parameter, "format_ue8m0", False)
                ):
                    flat.view(torch.uint8).fill_(127)
                else:
                    flat.fill_(1)
                population, chunks = "positive_unit_scale_or_native_gate", 0
            else:
                population, chunks = "native_float_uniform", 0
                for start in range(0, flat.numel(), WEIGHT_INITIALIZER["chunk_elements"]):
                    view = flat[start : start + WEIGHT_INITIALIZER["chunk_elements"]]
                    seed = plan["seed"] + chunks
                    if parameter.dtype in (torch.int8, torch.uint8):
                        # fp8.py:169-188,196-213 defines all16 finite E2M1
                        # nibbles; :1266-1284 stores two per int8 byte. This is
                        # declared synthetic packed data, not checkpoint values.
                        if leaf not in ("w13_weight", "w2_weight"):
                            raise RuntimeError("unqualified integer parameter storage: " + name)
                        generator = torch.Generator(device=view.device).manual_seed(seed)
                        view.view(torch.uint8).random_(0, 256, generator=generator)
                        population = "uniform_packed_e2m1_bytes"
                    elif torch.is_floating_point(parameter):
                        holder = torch.nn.Module()
                        holder.register_parameter("weight", torch.nn.Parameter(view, requires_grad=False))
                        native_initialize(
                            holder,
                            low=WEIGHT_INITIALIZER["float_low"],
                            high=WEIGHT_INITIALIZER["float_high"],
                            seed=seed,
                        )
                        del holder
                    else:
                        raise RuntimeError("unqualified native parameter dtype: " + name)
                    chunks += 1
            sample = flat[: min(4096, flat.numel())].float().cpu()
            if not torch.isfinite(sample).all().item():
                raise RuntimeError("native initialized storage sample is non-finite: " + name)
            nonzero = int(torch.count_nonzero(sample).item())
            if flat.numel() >= 4096 and nonzero == 0:
                raise RuntimeError("native initialized storage sample is degenerate: " + name)
            records[name] = dict(
                population=population,
                first_chunk_seed=plan["seed"],
                chunk_seed_increment=1,
                chunks=chunks,
                elements=flat.numel(),
                dtype=str(parameter.dtype),
                sample_elements=sample.numel(),
                sample_nonzero=nonzero,
                sample_sha256=hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            )
        # Native post-load ordering: model_loader/loader.py:1601-1621.
        for child in list(module.modules()):
            method = getattr(child, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(child)
    return records


def describe_module(module):
    return {
        "class": type(module).__module__ + "." + type(module).__name__,
        "parameters": {
            name: dict(shape=list(p.shape), dtype=str(p.dtype), bytes=p.numel() * p.element_size())
            for name, p in module.named_parameters()
        },
    }


class LocalIntervals:
    """Observe existing local calls; collectives execute outside these sites."""

    def __init__(self):
        self.events = []
        self.originals = []

    def install(self, module, method, entry):
        import torch

        original = getattr(module, method)
        witness = _dispatch(module, method)

        def call(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events.append((entry, witness, start, end))
            return result

        self.originals.append((module, method, original))
        setattr(module, method, call)

    def restore(self):
        for module, method, original in self.originals:
            setattr(module, method, original)
        self.originals.clear()
        self.events.clear()

    def measure(self, call, tokens, plan, rank, provenance, stream, expected_calls):
        import torch

        for sample in range(plan["warmup"] + plan["iterations"]):
            self.events.clear()
            torch.cuda.synchronize()
            output = call()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all().item():
                raise RuntimeError("isolated native module returned non-finite output")
            if len(self.events) != expected_calls:
                raise RuntimeError("isolated native local-call coverage changed")
            if sample < plan["warmup"]:
                continue
            groups = {}
            for entry, witness, start, end in self.events:
                key = (entry["component"], entry["geometry"])
                value = groups.setdefault(key, [0.0, set()])
                value[0] += start.elapsed_time(end)
                value[1].add(witness)
            for (component, geometry), (latency, witnesses) in groups.items():
                row = dict(
                    component=component,
                    geometry=geometry,
                    batch_size=1,
                    prefix=0,
                    x=tokens,
                    latency=latency,
                    kernel_source="+".join(sorted(witnesses)),
                    measurement_scope="local_compute",
                    used_cuda_graph=False,
                    sample_count=1,
                    kv_seed_regime="n/a",
                    sample=sample,
                    invocation=tokens,
                    tp_rank=rank,
                    **provenance,
                )
                validate_row(row)
                stream.write(json.dumps(row) + "\n")


def collect_linear(moe, manifest, plan, rank, provenance, stream):
    import torch

    shared = moe.shared_experts
    if shared is None or getattr(moe, "_shared_expert_tp1", False) or shared.down_proj.reduce_results:
        raise RuntimeError("native shared expert is not the local sharded consumer layout")
    observer = LocalIntervals()
    try:
        for module, name, actual in (
            (shared.gate_up_proj, "gate_up", (shared.gate_up_proj.output_size_per_partition, 5120)),
            (shared.down_proj, "ffn2", (5120, shared.down_proj.input_size_per_partition)),
        ):
            entry = next(e for e in manifest["phases"]["context"] if e["layer"] == 2 and name in e["name"])
            geometry = json.loads(entry["geometry"])
            if actual != (geometry["n"], geometry["k"]) or tuple(
                module.quant_method.quant_config.weight_block_size
            ) != (32, 32):
                raise RuntimeError("native shared projection differs from consumer geometry/FP8 block32")
            observer.install(module, "forward", entry)
        for tokens in plan["token_counts"]:
            # DeepseekV2MLP.forward deepseek_v2.py:314-337 accepts [T,hidden];
            # these seeded BF16 vectors are synthetic operator inputs.
            generator = torch.Generator().manual_seed(plan["seed"] + tokens)
            hidden = torch.randn(tokens, 5120, generator=generator, dtype=torch.bfloat16).cuda()
            observer.measure(lambda: shared(hidden), tokens, plan, rank, provenance, stream, 2)
    finally:
        observer.restore()


def collect_engram(config, quant, manifest, plan, rank, provenance, stream, token_ids, receipt):
    import torch
    from sglang.srt.layers.engram import Engram, EngramHasher, build_engram_layout
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    layout = build_engram_layout(config)
    # EngramHasher.from_config/init_history (engram.py:232-268) use the same
    # serving tokenizer and initialize one legal request slot plus PAD history.
    hasher = EngramHasher.from_config(config, layout).cuda()
    hasher.init_history(1, torch.device("cuda"))
    receipt["engram_hash_inputs"] = []
    receipt["engram_hasher_buffers"] = {
        name: hashlib.sha256(getattr(hasher, name).cpu().numpy().tobytes()).hexdigest()
        for name in ("token_map", "multipliers", "primes", "offsets")
    }
    for entry in (e for e in manifest["phases"]["context"] if e["component"] == "engram"):
        layer_id = entry["layer"]
        with torch.device("cuda"):
            module = Engram(config, layer_id, layout, quant, prefix=f"model.layers.{layer_id}.engram")
        geometry = json.loads(entry["geometry"])
        if (
            module.embed.host_table is not None
            or module.embed._shared
            or module.embed.tp_size != manifest["tp_size"]
            or layout.num_embeddings[module.layer_hash_index] != geometry["num_embeddings"]
            or tuple(module.wkv.weight.shape) != (25600, 6144)
            or tuple(module.wkv.quant_method.quant_config.weight_block_size) != (32, 32)
        ):
            raise RuntimeError("native Engram GPU table/projection differs from consumer geometry")
        receipt.setdefault("weight_initialization", {})[f"engram.{layer_id}"] = initialize_dummy_weights(module, plan)
        receipt.setdefault("loaded_modules", {})[f"engram.{layer_id}"] = describe_module(module)
        ownership = [None] * plan["tp_size"]
        torch.distributed.all_gather_object(
            ownership, (module.embed.row_start, module.embed.row_start + module.embed.rows)
        )
        if (
            ownership[0][0] != 0
            or ownership[-1][1] != geometry["num_embeddings"]
            or any(left[1] != right[0] for left, right in itertools.pairwise(ownership))
        ):
            raise RuntimeError("native Engram rank ownership does not cover the table exactly")
        observer = LocalIntervals()
        observer.install(module.embed, "_owned_rows", entry)
        observer.install(module.wkv, "forward", entry)
        observer.install(module, "apply_gate", entry)
        try:
            for tokens in plan["token_counts"]:
                # Pinned schedule_batch.py:2553-2595 populates one no-prefix
                # EXTEND with input_ids[:T], seq_lens=[T], extend_lens=[T].
                # forward_batch_info.py:780-798 copies core fields; :903-928,
                # 1826-1840 produce positions=arange(T), extend_start_loc=[0].
                # Slot0 is our legal isolated history slot, NOT a claim about
                # scheduler allocation order (schedule_batch.py:2618,2770).
                ids = torch.tensor(token_ids[:tokens], dtype=torch.int64, device="cuda")
                hasher.history.zero_()
                lengths = torch.tensor([tokens], dtype=torch.int32, device="cuda")
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=ids,
                    req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
                    seq_lens=lengths.to(torch.int64),
                    # Hasher-only API: engram.py:346-356 explicitly allows
                    # None while committing real history. Zero KV slots would
                    # instead mark graph padding. No attention/KV claim here.
                    out_cache_loc=None,
                    seq_lens_sum=tokens,
                    positions=torch.arange(tokens, dtype=torch.int64, device="cuda"),
                    extend_seq_lens=lengths,
                    extend_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
                )
                # EngramHasher.forward:285-340 consumes these extend fields;
                # ngram_history=None reads our reset slot. Positions before a
                # predecessor are native PAD (engram.py:399-404).
                hashes = hasher(ids, batch)[:, module.layer_hash_index, :].contiguous()
                torch.cuda.synchronize()
                digest = hashlib.sha256(hashes.cpu().numpy().tobytes()).hexdigest()
                gathered = [None] * plan["tp_size"]
                torch.distributed.all_gather_object(gathered, digest)
                if len(set(gathered)) != 1:
                    raise RuntimeError("native Engram hashes differ between TP ranks")
                primes = hasher.primes[module.layer_hash_index].flatten()
                offsets = hasher.offsets[module.layer_hash_index]
                if (
                    hashes.shape != (tokens, geometry["hash_columns"])
                    or not ((hashes >= offsets) & (hashes < offsets + primes)).all().item()
                ):
                    raise RuntimeError("native Engram hashes exceed their actual buckets")
                receipt["engram_hash_inputs"].append(
                    dict(
                        layer=layer_id,
                        tokens=tokens,
                        ids_sha256=digest,
                        shape=list(hashes.shape),
                        unique_rows=int(hashes.unique().numel()),
                        per_rank_owned_ids=[
                            int(((hashes >= start) & (hashes < end)).sum().item()) for start, end in ownership
                        ],
                        tp_input_agreement=True,
                    )
                )
                generator = torch.Generator().manual_seed(plan["seed"] + tokens)
                # Engram.forward, engram.py:926-942: x[T,hc_mult,hidden] plus
                # this layer's real native hash columns. Values are synthetic.
                hidden = torch.randn(
                    tokens, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16
                ).cuda()
                observer.measure(
                    lambda module=module: module(hidden, hashes), tokens, plan, rank, provenance, stream, 3
                )
        finally:
            observer.restore()
        del module
        gc.collect()
        torch.cuda.empty_cache()


def collect_mhc(config, quant, manifest, plan, rank, provenance, stream, receipt):
    import torch
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    with torch.device("cuda"):
        layer = DeepseekV4DecoderLayer(config, 2, quant, prefix="model.layers.2", hc_stats_stream=None)
    if not layer.hc_pre_from_prev_sublayer or layer.use_fused_mhc_post_pre:
        raise RuntimeError("native mHC predecessor/serial dispatch differs")
    receipt.setdefault("weight_initialization", {})["mhc_layer2"] = initialize_dummy_weights(layer, plan)
    receipt.setdefault("loaded_modules", {})["mhc_layer2"] = describe_module(layer)
    entry = next(e for e in manifest["phases"]["context"] if e["component"] == "mhc" and e["layer"] == 2)
    if json.loads(entry["geometry"]) != dict(
        hidden_size=layer.hidden_size, hc_mult=layer.hc_mult, sinkhorn_iters=layer.hc_sinkhorn_iters
    ):
        raise RuntimeError("native mHC differs from consumer geometry")
    receipt["mhc_scope"] = dict(
        native_layer=2,
        stats_stream=None,
        sublayer_outputs="seeded_synthetic",
        predecessor_pre="native_untimed_ffn_mix",
        attention_executed=False,
        moe_executed=False,
        overlap_aware_latency=False,
    )
    observer = LocalIntervals()
    observer.install(layer, "_hc_mix_and_combine", entry)
    observer.install(layer, "hc_post", entry)
    try:
        for tokens in plan["token_counts"]:
            # deepseek_v4.py:2651-2661,2734-2785 defines residual[T,hc,hidden]
            # and sublayer outputs[T,hidden]. We generate the latter explicitly;
            # no attention/MLP output or overlap-aware block latency is claimed.
            generator = torch.Generator().manual_seed(plan["seed"] + tokens)
            hidden = torch.randn(
                tokens, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16
            ).cuda()
            attn_output = torch.randn(tokens, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
            ffn_output = torch.randn(tokens, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
            _, prev_pre, _, _ = layer._hc_mix_and_combine(
                hidden, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, apply_pre=None
            )
            torch.cuda.synchronize()

            def both_sites():
                # forward_hc_pre_from_prev:2751-2785: attention consumes the
                # predecessor FFN pre; FFN consumes attn_pre. Each post uses its
                # own coefficients and corresponding unsqueezed residual.
                _, attn_pre, attn_post, attn_comb = layer._hc_mix_and_combine(
                    hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, apply_pre=prev_pre
                )
                residual = layer.hc_post(attn_output, hidden, attn_post, attn_comb)
                _, _, ffn_post, ffn_comb = layer._hc_mix_and_combine(
                    residual, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, apply_pre=attn_pre
                )
                return layer.hc_post(ffn_output, residual, ffn_post, ffn_comb)

            observer.measure(both_sites, tokens, plan, rank, provenance, stream, 4)
    finally:
        observer.restore()


def run(args, receipt):
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    validate_plan(plan, manifest)
    rank, local_rank, tp = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if tp != plan["tp_size"]:
        raise RuntimeError("actual torchrun world differs from plan")
    prepare_private_caches(rank, receipt)
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != tp:
        raise RuntimeError("visible CUDA device count differs from allocated TP")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(plan["seed"])
    props = torch.cuda.get_device_properties(local_rank)
    if (
        re.search(r"\b" + re.escape(plan["expected_gpu"]) + r"\b", props.name, re.IGNORECASE) is None
        or props.major * 10 + props.minor != plan["expected_sm"]
    ):
        raise RuntimeError("allocated GPU identity differs from plan")
    receipt["device"] = dict(
        name=props.name, uuid=str(props.uuid), memory=props.total_memory, sm=props.major * 10 + props.minor
    )
    raw_uuid = str(props.uuid)
    if re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", raw_uuid) is None:
        raise RuntimeError("unexpected allocated CUDA UUID format")
    gpu_uuid = raw_uuid if raw_uuid.startswith("GPU-") else "GPU-" + raw_uuid
    command = [
        "nvidia-smi",
        "--id=" + gpu_uuid,
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20)
    receipt["allocated_device_witness"] = dict(
        cuda_local_rank=local_rank,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_uuid=gpu_uuid,
        argv=command,
        returncode=query.returncode,
        stdout=query.stdout,
        stderr=query.stderr,
    )
    if query.returncode != 0 or len(query.stdout.strip().splitlines()) != 1 or gpu_uuid not in query.stdout:
        raise RuntimeError("allocated CUDA UUID did not match native GPU driver witness")
    from sglang.benchmark import one_batch as bench
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.model_loader.loader import _get_quantization_config
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.runtime_context import get_model, get_serving
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    package = Path(bench.__file__).resolve().parents[1]
    sources = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    receipt["source_hashes"] = sources
    receipt["expected_source_hashes"] = plan["source_pins"]
    receipt["source_modifications"] = plan.get("source_modifications", [])
    for path, digest in plan["source_pins"].items():
        if sources.get(path) != digest:
            raise RuntimeError("installed source differs: " + path)
    config_json = json.loads((args.model_path / "config.json").read_text())
    if hashlib.sha256(canonical_json(config_json).encode()).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("unchanged checkpoint config required")
    if (
        args.runtime_digest != plan["runtime_digest"]
        or os.environ.get("DSV41_LAUNCH_IMAGE_SHA256") != plan["image_sha256"]
    ):
        raise RuntimeError("actual launch image differs from the frozen plan")
    for name, digest in plan["metadata_pins"].items():
        if sha(args.model_path / name) != digest:
            raise RuntimeError("checkpoint/tokenizer metadata changed: " + name)
    receipt.update(
        framework_version="dev-" + FRAMEWORK_COMMIT,
        collector_revision=plan["collector_revision"],
        raw_package_version=importlib.metadata.version("sglang"),
        plan_sha256=sha(args.plan),
        manifest_sha256=sha(args.manifest),
        source_hashes=sources,
        python_executable=sys.executable,
        runtime_digest=args.runtime_digest,
        metadata_hashes=plan["metadata_pins"],
        purpose=plan["purpose"],
        weight_initializer=plan["weight_initializer"],
        image=dict(path=os.environ.get("DSV41_LAUNCH_IMAGE_PATH"), sha256=plan["image_sha256"]),
    )
    server = bench.ServerArgs(
        model_path=str(args.model_path),
        trust_remote_code=True,
        tp_size=tp,
        moe_runner_backend=plan["moe_runner_backend"],
        moe_a2a_backend="none",
        disable_shared_experts_fusion=True,
        disable_custom_all_reduce=True,
        enforce_disable_flashinfer_allreduce_fusion=True,
        load_format="dummy",
        dtype="bfloat16",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )
    bench.publish(server, role="scheduler")
    bench.initialize_moe_config()
    bench.initialize_fp8_gemm_config()
    bench.initialize_fp4_gemm_config()
    model_config = ModelConfig.from_server_args(server)
    config = model_config.hf_text_config
    ps = ParallelState.trivial(tp_rank=rank, tp_size=tp, attn_tp_rank=rank, attn_tp_size=tp, gpu_id=local_rank)
    bootstrap.init_torch_distributed(
        server_args=server,
        model_config=model_config,
        device="cuda",
        ps=ps,
        dist_port=int(os.environ["MASTER_PORT"]),
        is_draft_worker=False,
        local_omp_cpuid=None,
    )
    quant = _get_quantization_config(model_config, LoadConfig(load_format="dummy"))
    if type(quant).__name__ != "Fp8Config" or quant.is_fp4_experts is not True or quant.dequant_fp4_to_fp8:
        raise RuntimeError("native checkpoint quantization selection differs")
    provenance = dict(
        source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
        config_sha256=manifest["config_sha256"],
        runtime_digest=args.runtime_digest,
        execution_profile=manifest["execution_profile"],
        case_plan_sha256=sha(args.plan),
        collection_purpose=plan["purpose"],
    )
    with (args.output / f"rank-{rank}.jsonl").open("x") as stream:
        if {"baselines", "linear"} & set(plan["components"]):
            with torch.device("cuda"):
                moe = DeepseekV2MoE(config, 2, quant, prefix="model.layers.2.mlp", is_deepseek_v4=True)
                head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant, prefix="lm_head")
            receipt.setdefault("weight_initialization", {}).update(
                moe=initialize_dummy_weights(moe, plan), lm_head=initialize_dummy_weights(head, plan)
            )
            receipt.setdefault("loaded_modules", {}).update(moe=describe_module(moe), lm_head=describe_module(head))
            receipt["baseline_warmup"] = 2
            receipt["native_expert_method"] = type(moe.experts.quant_method).__name__
            if "baselines" in plan["components"]:
                options = SimpleNamespace(
                    output=str(args.output),
                    iterations=plan["iterations"],
                    workload_plan={"cases": [{"batch_size": 1, "query": n} for n in plan["token_counts"]]},
                )
                collect_native_kernel_baselines(
                    moe.experts, moe.gate, head, tp, config.vocab_size, options, rank, provenance
                )
            if "linear" in plan["components"]:
                collect_linear(moe, manifest, plan, rank, provenance, stream)
            del moe, head
            gc.collect()
            torch.cuda.empty_cache()
        if "engram" in plan["components"]:
            tokenizer = get_tokenizer(
                get_serving().tokenizer_path,
                tokenizer_mode=get_serving().tokenizer_mode,
                trust_remote_code=get_model().trust_remote_code,
                revision=get_model().revision,
                tokenizer_backend="huggingface",
            )
            text = args.prompt_file.read_text()
            ids = tokenizer.encode(text)
            if len(ids) < max(plan["token_counts"]) or len(set(ids)) < 100:
                raise RuntimeError("native tokenizer corpus is too short or degenerate")
            receipt["input_provenance"] = dict(
                source="native_tokenizer_and_engram_hasher",
                text_sha256=sha(args.prompt_file),
                token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                token_count=len(ids),
                unique_tokens=len(set(ids)),
            )
            collect_engram(config, quant, manifest, plan, rank, provenance, stream, ids, receipt)
        if "mhc" in plan["components"]:
            collect_mhc(config, quant, manifest, plan, rank, provenance, stream, receipt)
    receipt.update(state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated())
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "manifest", "model-path", "prompt-file", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--runtime-digest", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    receipt = dict(
        schema="dsv41.isolated-run.v1",
        state="started",
        tp_rank=rank,
        started_unix_ns=time.time_ns(),
        full_model=False,
        attention_measured=False,
        checkpoint_weights_loaded=False,
        weight_source="native_random_chunk_views_packed_e2m1_random_positive_scales",
        publication_permitted=False,
    )
    path = args.output / f"isolated-rank-{rank}.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite prior isolated collection")
    try:
        run(args, receipt)
    except BaseException as error:
        receipt.update(
            state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
        raise
    finally:
        receipt["finished_unix_ns"] = time.time_ns()
        with path.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
