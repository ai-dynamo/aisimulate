# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 isolated operator producer for vLLM (torchrun, one module at a time).

Measures the DSV4.1 contract components the SGLang producer measures
(``collector/sglang/dsv41_contract.py``) on vLLM's native modules, with dummy weights and
real TP collectives outside every timed interval:

* ``linear``    shared-expert ``DeepseekV4MLP.{gate_up_proj,down_proj}`` (deepseek_v4/nvidia/model.py:101-158)
* ``engram``    ``ParallelEngramEmbedding.lookup`` + ``Engram.forward`` with the TP all-gather
                excluded (deepseek_v41/common/engram.py:645-1056); tables GPU-resident
                (``EngramConfig(cpu_offload=False)``), TP head-sharded as in serving
* ``mhc``       both ``mhc_shifted_post_pre`` sites of layer 2 (post + pre-mix + fused norm;
                deepseek_v41/nvidia/model.py:406-450, ops/mega_mhc.py:92-156)
* ``baselines`` router ``GateLinear``, ``ParallelLMHead``, routed experts
                (``RoutedExperts.forward_modular`` with uniform routing) and NCCL all-reduce

Pinned to vLLM v0.30.0 (image ``vllm/vllm-openai:v0.30.0``). Serving identity on sm90 at
this pin: FP8 block[32,32] linears run through ``ModelOptLinearMethod[MarlinMxfp8LinearKernel]``
(W8A16 Marlin; the MXFP8 FlashInfer/DeepGEMM kernels need SM100), ``wo_a`` through the
BF16 emulation kernel, experts through ``Mxfp4MoEMethod`` backend MARLIN (W4A16). The rows
carry the SDK's geometry keys (``gemm_quant_mode=fp8_block``) and record that identity in
``kernel_source``; the perf-database version key is ``0.30.0`` under backend ``vllm``.

Plan schema ``dsv41.isolated-collection.v1`` as the SGLang producer, with ``moe_backend``
(``marlin``) in place of ``moe_runner_backend``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from collector.sglang.dsv41_contract import canonical_json, validate_row

BACKEND = "vllm"
# vllm-project/vllm tag v0.30.0 (2026-09-21), image vllm/vllm-openai:v0.30.0
FRAMEWORK_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
FRAMEWORK_VERSION = "0.30.0"
EXPECTED_SM = {"H100": 90, "H200": 90, "H20": 90, "B200": 100, "GB200": 100}
WEIGHT_INITIALIZER = {
    "name": "vllm_dummy_loader_plus_packed_fill_v1",
    # floats: vllm.model_executor.model_loader.weight_utils.initialize_single_dummy_weight (uniform)
    "float_low": -1e-3,
    "float_high": 1e-3,
    # fp8 e4m3 linears: uniform(-1e-3, 1e-3) cast to float8_e4m3fn
    # e8m0 (uint8) block scales: 127 (= 2^0); packed MXFP4 experts and the fp8 engram tables:
    # random bytes with the 0x7F/0xFF NaN encodings zeroed
    "e8m0_scale": 127,
    "packed_bytes": "uniform_nan_free",
}
REQUIRED_SOURCES = {
    "models/deepseek_v41/attention.py",
    "models/deepseek_v41/compressor.py",
    "models/deepseek_v41/sparse_mla.py",
    "models/deepseek_v41/quant_config.py",
    "models/deepseek_v41/common/engram.py",
    "models/deepseek_v41/nvidia/model.py",
    "models/deepseek_v41/nvidia/engram.py",
    "models/deepseek_v41/nvidia/flashmla.py",
    "models/deepseek_v41/nvidia/ops/mega_mhc.py",
    "models/deepseek_v4/nvidia/model.py",
    "models/deepseek_v4/nvidia/ops/o_proj.py",
    "model_executor/layers/fused_moe/router/gate_linear.py",
    "model_executor/layers/fused_moe/routed_experts.py",
    "model_executor/layers/fused_moe/runner/moe_runner.py",
    "model_executor/layers/quantization/modelopt.py",
    "model_executor/layers/quantization/mxfp4.py",
    "model_executor/kernels/linear/mxfp8/marlin.py",
    "model_executor/kernels/mhc/tilelang.py",
    "model_executor/model_loader/weight_utils.py",
    "model_executor/layers/vocab_parallel_embedding.py",
    "config/engram.py",
    "distributed/parallel_state.py",
}
CONFIG_SHA256 = "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
HIDDEN, EXPERTS, TOPK, INTER = 5120, 384, 6, 2304
MOE_DTYPE = {"MARLIN": "w4a16_mxfp4_marlin"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan, manifest):
    """Validate declared ownership before importing any serving/GPU library."""
    if plan["schema"] != "dsv41.isolated-collection.v1" or plan["tp_size"] not in (2, 4):
        raise ValueError("unsupported isolated collection plan")
    if plan["purpose"] not in ("smoke", "calibration"):
        raise ValueError("collection purpose must be declared before measurements")
    if plan.get("weight_initializer") != WEIGHT_INITIALIZER:
        raise ValueError("prospectively declared dummy-weight initializer required")
    if plan["tp_size"] != manifest["tp_size"] or plan["execution_profile"] != manifest["execution_profile"]:
        raise ValueError("isolated plan and consumer manifest differ")
    if plan["execution_profile"] != "full":
        raise ValueError("vLLM serves DeepSeek-V4.1 with the full execution profile only")
    tokens = plan["token_counts"]
    if not tokens or tokens != sorted(set(tokens)) or any(type(n) is not int or not 1 <= n <= 8192 for n in tokens):
        raise ValueError("token counts must be distinct increasing integers in 1..8192")
    components = plan["components"]
    if not components or len(set(components)) != len(components) or not set(components) <= {"baselines", "linear", "engram", "mhc"}:
        raise ValueError("isolated plan contains unsupported or duplicate components")
    if plan["warmup"] < 2 or plan["iterations"] < 5:
        raise ValueError("isolated collection requires warmup and repeated measurements")
    if plan["moe_backend"] not in MOE_DTYPE:
        raise ValueError("explicit qualified native MoE backend required")
    if plan["expected_gpu"] not in EXPECTED_SM or plan["expected_sm"] != EXPECTED_SM[plan["expected_gpu"]]:
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
    if manifest["config_sha256"] != CONFIG_SHA256:
        raise ValueError("only the unchanged V4.1 checkpoint architecture is qualified")


def aggregate_isolated_records(output, plan_path, manifest_path):
    """Require complete declared operator coverage before writing perf tables (vLLM receipts)."""
    from collector.sglang.collect_dsv41_module import aggregate_baseline_records, aggregate_rank_records

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
            or receipt.get("framework") != BACKEND
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
        filenames = [f"rank-{rank}.jsonl"] + ([f"baseline-rank-{rank}.jsonl"] if "baselines" in plan["components"] else [])
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
            if len(baselines.get(kind, [])) != per_token * len(plan["token_counts"]):
                raise ValueError(f"native {kind} baseline coverage is incomplete")
    for row in rows:
        for key in ("case_plan_sha256", "collection_purpose"):
            row.pop(key)
    return rows, baselines


def prepare_private_caches(rank, receipt):
    """Run before native imports; JIT/compile caches stay allocation-private per rank."""
    root = Path(os.environ["DSV41_PRIVATE_CACHE"])
    allocation = os.environ.get("SLURM_JOB_ID") or os.environ["DSV41_ALLOCATION_ID"]
    marker = root / "home-cache" / f"vllm-isolated-{allocation}-{rank}"
    marker.write_text("allocation-private\n")
    receipt["cache_bindings"] = []
    for target in (Path.home() / ".cache", Path("/root/.cache")):
        if not (target / marker.name).samefile(marker):
            raise RuntimeError("HOME/root cache does not share the allocation-private file")
        receipt["cache_bindings"].append({"path": str(target), "resolved": str(target.resolve())})
    for variable, leaf in (
        ("TRITON_CACHE_DIR", "triton"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
    ):
        cache = root / f"rank-{rank}" / leaf
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise RuntimeError("the stateless engram hasher requires the V2 model runner contract (VLLM_USE_V2_MODEL_RUNNER=1)")


def _fill_bytes(tensor, generator, chunk=1 << 28):
    import torch

    flat = tensor.view(torch.uint8).view(-1)
    for i in range(0, flat.numel(), chunk):
        n = min(chunk, flat.numel() - i)
        values = torch.randint(0, 256, (n,), dtype=torch.uint8, device=tensor.device, generator=generator)
        values[(values & 0x7F) == 0x7F] = 0  # 0x7F / 0xFF encode NaN in float8_e4m3fn
        flat[i : i + n] = values


def initialize_dummy_weights(module, plan, vllm_config, device, big_types=()):
    """Native dummy init for floats; explicit fills for the packed/uint8 tensors the loader leaves
    uninitialized (weight_utils.py:1348-1354 @v0.30.0), then the native post-load processing."""
    import torch
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight

    generator = torch.Generator(device="cuda").manual_seed(plan["seed"])
    report = defaultdict(int)
    for sub in module.modules():
        if isinstance(sub, big_types):
            # giant fp8 tables: bytes in place (the loader's fp16 temporary of a 49 GiB table OOMs)
            for name, parameter in sub.named_parameters(recurse=False):
                if "scale" in name:
                    parameter.data.fill_(WEIGHT_INITIALIZER["e8m0_scale"])
                else:
                    _fill_bytes(parameter.data, generator)
                report[f"table:{parameter.dtype}"] += parameter.numel()
            continue
        for name, tensor in get_layer_tensors(sub).items():
            if name in sub._non_persistent_buffers_set:
                continue
            if tensor.dtype == torch.uint8:
                if "scale" in name:
                    tensor.fill_(WEIGHT_INITIALIZER["e8m0_scale"])
                    report["e8m0"] += tensor.numel()
                else:
                    _fill_bytes(tensor, generator)
                    report["packed"] += tensor.numel()
            elif tensor.dtype == torch.float8_e4m3fn:
                low, high = WEIGHT_INITIALIZER["float_low"], WEIGHT_INITIALIZER["float_high"]
                tensor.copy_((torch.rand(tensor.shape, device=tensor.device, generator=generator) * (high - low) + low).to(torch.float8_e4m3fn))
                report["fp8"] += tensor.numel()
            else:
                initialize_single_dummy_weight(tensor, WEIGHT_INITIALIZER["float_low"], WEIGHT_INITIALIZER["float_high"], plan["seed"])
                report[str(tensor.dtype)] += tensor.numel()
    process_weights_after_loading(module, vllm_config.model_config, device)
    return dict(report)


def describe_module(module):
    return {
        "class": type(module).__module__ + "." + type(module).__name__,
        "parameters": {
            name: dict(shape=list(p.shape), dtype=str(p.dtype), bytes=p.numel() * p.element_size())
            for name, p in module.named_parameters()
        },
    }


def _dispatch(module, method):
    """Witness label: module class + method + the quant method and kernel vLLM selected."""
    import types

    if isinstance(module, types.ModuleType):
        return f"{module.__name__}.{method}"
    quant = getattr(module, "quant_method", None)
    kernel = getattr(quant, "kernel", None)
    suffix = "" if quant is None else f"/{type(quant).__module__}.{type(quant).__name__}" + ("" if kernel is None else f"[{type(kernel).__name__}]")
    return f"{type(module).__module__}.{type(module).__name__}.{method}{suffix}"


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
                value = groups.setdefault((entry["component"], entry["geometry"]), [0.0, set()])
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


def build_vllm_config(model_path, tp):
    from vllm.config import CacheConfig, CompilationConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.config.engram import EngramConfig
    from vllm.config.load import LoadConfig

    model_config = ModelConfig(model=str(model_path), tokenizer=str(model_path), trust_remote_code=True, dtype="bfloat16", seed=0, max_model_len=8192)
    # SM90 serving page size for the DSV4.1 FlashMLA backend (sparse_mla.py:89-90); fp8 KV is
    # forced to fp8_ds_mla by the attention module (attention.py:136-152).
    cache_config = CacheConfig(block_size=64, cache_dtype="fp8")
    cache_config.num_gpu_blocks, cache_config.num_cpu_blocks = 512, 0
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(tensor_parallel_size=tp, disable_custom_all_reduce=True),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True, max_model_len=8192, is_encoder_decoder=False),
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        # serving default is host-resident tables (VLLM_PLE_CPU_OFFLOAD=1); the contract measures
        # the GPU-resident TP-sharded tables like the SGLang producer
        engram_config=EngramConfig(cpu_offload=False),
    )


def collect_baselines(layer, lm_head, plan, rank, provenance, stream, receipt):
    """Router GEMM, LM head, routed experts (uniform routing) and NCCL all-reduce — the SGLang case grid."""
    import torch
    import torch.distributed as dist

    tp = plan["tp_size"]
    if dist.get_world_size() != tp or dist.get_rank() != rank or dist.get_backend() != "nccl":
        raise RuntimeError("native baseline collectives require the actual NCCL process group of the TP ranks")
    ffn = layer.ffn
    gate, runner = ffn.gate, ffn.experts
    quant = runner._quant_method
    if type(quant).__name__ != "Mxfp4MoEMethod" or quant.mxfp4_backend.name != plan["moe_backend"].upper() or quant.is_monolithic:
        raise RuntimeError(f"native MoE dispatch differs from the plan: {type(quant).__name__}/{getattr(quant, 'mxfp4_backend', None)}")
    routed = runner.routed_experts
    moe_dtype = MOE_DTYPE[quant.mxfp4_backend.name]
    moe_kernel_source = f"vllm_{quant.mxfp4_backend.name.lower()}_mxfp4_moe/{quant.experts_cls.__name__}"
    local_vocab = lm_head.weight.shape[0]
    if tuple(gate.weight.shape) != (EXPERTS, HIDDEN) or lm_head.weight.shape[1] != HIDDEN or local_vocab * tp < 129280:
        raise RuntimeError("native baseline GEMM physical padding differs from the model TP graph")
    receipt["baseline_identity"] = dict(
        moe_quant_method=type(quant).__name__,
        mxfp4_backend=quant.mxfp4_backend.name,
        experts_cls=quant.experts_cls.__name__,
        physical_local_intermediate=int(routed.intermediate_size_per_partition),
        gate=_dispatch(gate, "forward"),
        lm_head=_dispatch(lm_head, "quant_method.apply"),
        lm_head_local_vocab=int(local_vocab),
    )
    generator = torch.Generator().manual_seed(20260910)
    for tokens in plan["token_counts"]:
        hidden = torch.randn(tokens, HIDDEN, generator=generator, dtype=torch.bfloat16).cuda()
        logits = torch.rand(tokens, EXPERTS, generator=generator, dtype=torch.float32).cuda()
        ids = logits.topk(TOPK, dim=-1).indices.to(torch.int32)
        weights = torch.full((tokens, TOPK), 1 / TOPK, dtype=torch.float32, device=hidden.device)
        histogram = torch.bincount(ids.flatten().long(), minlength=EXPERTS).cpu().tolist()
        cases = [
            ("gemm", {"gemm_dtype": "bfloat16", "m": tokens, "n": EXPERTS, "k": HIDDEN}, lambda: gate(hidden)[0], _dispatch(gate, "forward")),
            ("gemm", {"gemm_dtype": "bfloat16", "m": tokens, "n": int(local_vocab), "k": HIDDEN}, lambda: lm_head.quant_method.apply(lm_head, hidden), _dispatch(lm_head, "quant_method.apply")),
            (
                "moe",
                {"moe_dtype": moe_dtype, "num_tokens": tokens, "hidden_size": HIDDEN, "inter_size": INTER, "topk": TOPK, "num_experts": EXPERTS, "moe_tp_size": tp, "moe_ep_size": 1, "distribution": "uniform"},
                lambda: routed.forward_modular(hidden, weights, ids),
                moe_kernel_source,
            ),
        ]
        for width in (5120, 6144):
            payload = torch.zeros(tokens, width, dtype=torch.bfloat16, device=hidden.device)
            cases.append(("nccl", {"op_name": "all_reduce", "nccl_dtype": "half", "num_gpus": tp, "message_size": 2 * tokens * width}, lambda payload=payload: dist.all_reduce(payload), "torch.distributed.nccl.all_reduce"))
        for kind, key, call, dispatch in cases:
            for sample in range(2 + plan["iterations"]):
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = call()
                end.record()
                torch.cuda.synchronize()
                if out is not None and hasattr(out, "isfinite") and not torch.isfinite(out).all().item():
                    raise RuntimeError(f"native {kind} baseline returned non-finite output")
                if sample >= 2:
                    stream.write(
                        json.dumps(
                            {
                                "kind": kind,
                                **key,
                                "latency": start.elapsed_time(end),
                                "sample": sample,
                                "tp_rank": rank,
                                "kernel_source": dispatch,
                                "used_cuda_graph": False,
                                "routing_seed": 20260910,
                                "routing_histogram": histogram if kind == "moe" else None,
                                "physical_local_intermediate": int(routed.intermediate_size_per_partition),
                                **provenance,
                            }
                        )
                        + "\n"
                    )


def collect_linear(layer, manifest, plan, rank, provenance, stream, receipt):
    import torch

    shared = layer.ffn.shared_experts
    if shared is None or shared.down_proj.reduce_results:
        raise RuntimeError("native shared expert is not the local sharded consumer layout")
    observer = LocalIntervals()
    try:
        for module, name, actual in (
            (shared.gate_up_proj, "gate_up", (shared.gate_up_proj.output_size_per_partition, HIDDEN)),
            (shared.down_proj, "ffn2", (HIDDEN, shared.down_proj.input_size_per_partition)),
        ):
            entry = next(e for e in manifest["phases"]["context"] if e["layer"] == 2 and name in e["name"])
            geometry = json.loads(entry["geometry"])
            if actual != (geometry["n"], geometry["k"]) or geometry["quant_mode"] != "fp8_block":
                raise RuntimeError("native shared projection differs from consumer geometry")
            observer.install(module, "forward", entry)
        receipt["linear_identity"] = dict(
            gate_up=_dispatch(shared.gate_up_proj, "forward"),
            down=_dispatch(shared.down_proj, "forward"),
            weight_block_size=list(getattr(shared.gate_up_proj, "weight_block_size", []) or []),
        )
        for tokens in plan["token_counts"]:
            generator = torch.Generator().manual_seed(plan["seed"] + tokens)
            hidden = torch.randn(tokens, HIDDEN, generator=generator, dtype=torch.bfloat16).cuda()
            observer.measure(lambda: shared(hidden), tokens, plan, rank, provenance, stream, 2)
    finally:
        observer.restore()


def collect_engram(vllm_config, manifest, plan, rank, provenance, stream, token_ids, receipt):
    import torch
    from vllm.models.deepseek_v41.common.engram import EngramLayout, NgramHashState
    from vllm.models.deepseek_v41.nvidia.engram import Engram, ParallelEngramEmbedding

    config = vllm_config.model_config.hf_config
    layout = EngramLayout(config)
    # Stateless V2-runner hashing: the SWA cache module only supplies block_size; an unbound
    # (empty) kv_cache is the "profile run" shape the model itself tolerates (nvidia/model.py:633).
    hasher = NgramHashState(vllm_config, layout, SimpleNamespace(block_size=32, kv_cache=torch.empty(0, device="cuda"))).cuda()
    receipt["engram_hash_inputs"] = []
    receipt["engram_hasher_buffers"] = {name: hashlib.sha256(getattr(hasher, name).cpu().numpy().tobytes()).hexdigest() for name in ("token_map", "multipliers", "primes", "offsets")}
    device = torch.device("cuda", torch.cuda.current_device())
    for entry in (e for e in manifest["phases"]["context"] if e["component"] == "engram"):
        layer_id = entry["layer"]
        index = layout.layer_ids.index(layer_id)
        geometry = json.loads(entry["geometry"])
        with torch.device("cuda"):
            module = Engram(config, vllm_config.quant_config, layout, index, False, f"model.layers.{layer_id}.engram")
        receipt.setdefault("weight_initialization", {})[f"engram.{layer_id}"] = initialize_dummy_weights(module, plan, vllm_config, device, big_types=(ParallelEngramEmbedding,))
        receipt.setdefault("loaded_modules", {})[f"engram.{layer_id}"] = describe_module(module)
        embed = module.embed_tokens
        if embed.cpu_offload or getattr(embed, "dp_size", 1) != 1 or embed.tp_size != plan["tp_size"]:
            raise RuntimeError("native Engram table is not the GPU-resident TP-sharded layout")
        if layout.num_embeddings[index] != geometry["num_embeddings"] or layout.n_hash_cols != geometry["hash_columns"] or embed.dim != geometry["head_dim"]:
            raise RuntimeError("native Engram layout differs from consumer geometry")
        ownership = [None] * plan["tp_size"]
        torch.distributed.all_gather_object(ownership, (embed.head_start, embed.head_start + embed.part_n_hash_cols))
        if ownership[0][0] != 0 or ownership[-1][1] < layout.n_hash_cols or any(a[1] != b[0] for a, b in zip(ownership, ownership[1:])):
            raise RuntimeError("native Engram head ownership does not cover the hash columns exactly")
        receipt.setdefault("engram_identity", {})[str(layer_id)] = dict(
            sharding="tp_head_sharded_all_gather", heads_per_rank=int(embed.part_n_hash_cols), rows_per_rank=int(embed.part_num_embeddings),
            lookup=_dispatch(embed, "lookup"), wkv=_dispatch(module.wkv, "forward"), gate="vllm.models.deepseek_v41.common.engram._fused_engram_post_wkv_kernel",
        )
        observer = LocalIntervals()
        observer.install(embed, "lookup", entry)
        observer.install(module, "forward", entry)
        try:
            for tokens in plan["token_counts"]:
                ids = torch.tensor(token_ids[:tokens], dtype=torch.int64, device="cuda")
                hashes = hasher(
                    ids, torch.arange(tokens, device="cuda"), torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
                    torch.zeros(tokens, dtype=torch.bool, device="cuda"),
                    torch.full((1, layout.max_ngram_size - 1), -1, dtype=torch.int64, device="cuda"),
                    torch.zeros(1, layout.max_ngram_size - 1, dtype=torch.bool, device="cuda"), None, None,
                )[:, index].contiguous()
                torch.cuda.synchronize()
                digest = hashlib.sha256(hashes.cpu().numpy().tobytes()).hexdigest()
                gathered = [None] * plan["tp_size"]
                torch.distributed.all_gather_object(gathered, digest)
                if len(set(gathered)) != 1:
                    raise RuntimeError("native Engram hashes differ between TP ranks")
                primes, offsets = hasher.primes[index].flatten(), hasher.offsets[index]
                if hashes.shape != (tokens, geometry["hash_columns"]) or not ((hashes >= offsets) & (hashes < offsets + primes)).all().item():
                    raise RuntimeError("native Engram hashes exceed their actual buckets")
                receipt["engram_hash_inputs"].append(dict(layer=layer_id, tokens=tokens, ids_sha256=digest, shape=list(hashes.shape), unique_rows=int(hashes.unique().numel()), tp_input_agreement=True))
                generator = torch.Generator().manual_seed(plan["seed"] + tokens)
                hidden = torch.randn(tokens, config.hc_mult, config.hidden_size, generator=generator, dtype=torch.bfloat16).cuda()

                def call(module=module, hidden=hidden, hashes=hashes):
                    # serving order (nvidia/model.py:677-682, common/engram.py:995-1004): lookup into the
                    # staging rows, all-gather the TP head shards, then wkv + gate. The gather runs
                    # untimed between the two observed intervals.
                    module.prepare_embeddings(hashes)
                    rows = module.embed(hashes)
                    module.embed = lambda _hash_ids, rows=rows: rows
                    try:
                        return module(hidden, hashes)
                    finally:
                        del module.embed

                observer.measure(call, tokens, plan, rank, provenance, stream, 2)
        finally:
            observer.restore()
        del module
        gc.collect()
        torch.cuda.empty_cache()


def collect_mhc(layer, manifest, plan, rank, provenance, stream, receipt):
    import torch
    from vllm.models.deepseek_v41.nvidia.ops import mega_mhc as native_mhc

    entry = next(e for e in manifest["phases"]["context"] if e["component"] == "mhc" and e["layer"] == 2)
    if json.loads(entry["geometry"]) != dict(hidden_size=layer.hidden_size, hc_mult=layer.hc_mult, sinkhorn_iters=layer.hc_sinkhorn_iters):
        raise RuntimeError("native mHC differs from consumer geometry")
    receipt["mhc_scope"] = dict(
        native_layer=2,
        sites="mhc_shifted_post_pre x2 (attention site with predecessor FFN pre-mix, FFN site with attention pre-mix)",
        sublayer_outputs="seeded_synthetic",
        predecessor_pre="native_untimed_call",
        attention_executed=False,
        moe_executed=False,
        norm_fused_into_combine=True,
        kernel_path="deep_gemm mega_mhc on SM100, TileLang mhc_fused_post_pre_delayed_tilelang otherwise (ops/mega_mhc.py:92-156)",
    )
    hc, hidden_size = layer.hc_mult, layer.hidden_size
    observer = LocalIntervals()
    observer.install(native_mhc, "mhc_shifted_post_pre", entry)
    try:
        for tokens in plan["token_counts"]:
            generator = torch.Generator().manual_seed(plan["seed"] + tokens)
            residual = torch.randn(tokens, hc, hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
            attn_output = torch.randn(tokens, hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
            ffn_output = torch.randn(tokens, hidden_size, generator=generator, dtype=torch.bfloat16).cuda()
            post_mix = torch.ones(tokens, hc, 1, dtype=torch.float32, device="cuda")
            res_mix = torch.eye(hc, dtype=torch.float32, device="cuda").expand(tokens, hc, hc).contiguous()
            seed_pre = torch.full((tokens, hc), 1.0 / hc, dtype=torch.float32, device="cuda")
            # predecessor FFN pre-mix (what layer 1 hands to layer 2), untimed
            observer.events.clear()
            *_, prev_pre, _ = native_mhc.mhc_shifted_post_pre(
                ffn_output, residual, post_mix, res_mix, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                pre_mix=seed_pre, norm_weight=layer.ffn_norm.weight, norm_eps=layer.ffn_norm.variance_epsilon,
            )
            torch.cuda.synchronize()

            def both_sites():
                r1, p1, m1, _, attn_pre, _ = native_mhc.mhc_shifted_post_pre(
                    attn_output, residual, post_mix, res_mix, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                    layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                    pre_mix=prev_pre, norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon,
                )
                r2, *_ = native_mhc.mhc_shifted_post_pre(
                    ffn_output, r1, p1, m1, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                    layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters,
                    pre_mix=attn_pre, norm_weight=layer.ffn_norm.weight, norm_eps=layer.ffn_norm.variance_epsilon,
                )
                return r2

            observer.measure(both_sites, tokens, plan, rank, provenance, stream, 2)
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
    torch.manual_seed(plan["seed"])
    props = torch.cuda.get_device_properties(local_rank)
    if re.search(r"\b" + re.escape(plan["expected_gpu"]) + r"\b", props.name, re.IGNORECASE) is None or props.major * 10 + props.minor != plan["expected_sm"]:
        raise RuntimeError("allocated GPU identity differs from plan")
    receipt["device"] = dict(name=props.name, uuid=str(props.uuid), memory=props.total_memory, sm=props.major * 10 + props.minor)
    gpu_uuid = str(props.uuid)
    gpu_uuid = gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid
    command = ["nvidia-smi", "--id=" + gpu_uuid, "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv,noheader,nounits"]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20)
    receipt["allocated_device_witness"] = dict(cuda_local_rank=local_rank, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_uuid=gpu_uuid, argv=command, returncode=query.returncode, stdout=query.stdout, stderr=query.stderr)
    if query.returncode != 0 or len(query.stdout.strip().splitlines()) != 1 or gpu_uuid not in query.stdout:
        raise RuntimeError("allocated CUDA UUID did not match native GPU driver witness")
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    package = Path(vllm.__file__).resolve().parent
    sources = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    receipt["source_hashes"] = sources
    receipt["expected_source_hashes"] = plan["source_pins"]
    for path, digest in plan["source_pins"].items():
        if sources.get(path) != digest:
            raise RuntimeError("installed source differs: " + path)
    config_json = json.loads((args.model_path / "config.json").read_text())
    if hashlib.sha256(canonical_json(config_json).encode()).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("unchanged checkpoint config required")
    if args.runtime_digest != plan["runtime_digest"] or os.environ.get("DSV41_LAUNCH_IMAGE_SHA256") != plan["image_sha256"]:
        raise RuntimeError("actual launch image differs from the frozen plan")
    for name, digest in plan["metadata_pins"].items():
        if sha(args.model_path / name) != digest:
            raise RuntimeError("checkpoint/tokenizer metadata changed: " + name)
    receipt.update(
        framework=BACKEND,
        framework_version=FRAMEWORK_VERSION,
        collector_revision=plan["collector_revision"],
        raw_package_version=importlib.metadata.version("vllm"),
        plan_sha256=sha(args.plan),
        manifest_sha256=sha(args.manifest),
        python_executable=sys.executable,
        runtime_digest=args.runtime_digest,
        metadata_hashes=plan["metadata_pins"],
        purpose=plan["purpose"],
        weight_initializer=plan["weight_initializer"],
        image=dict(path=os.environ.get("DSV41_LAUNCH_IMAGE_PATH"), sha256=plan["image_sha256"]),
    )
    vllm_config = build_vllm_config(args.model_path, tp)
    init_distributed_environment(tp, rank, "env://", local_rank)
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(tp, 1)
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device)
    receipt["native_config"] = dict(
        quant_config=type(vllm_config.quant_config).__name__,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        block_size=vllm_config.cache_config.block_size,
        engram_cpu_offload=vllm_config.engram_config.cpu_offload,
        use_v2_model_runner=vllm_config.use_v2_model_runner,
        disable_custom_all_reduce=vllm_config.parallel_config.disable_custom_all_reduce,
    )
    provenance = dict(
        source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
        config_sha256=manifest["config_sha256"],
        runtime_digest=args.runtime_digest,
        execution_profile=manifest["execution_profile"],
        case_plan_sha256=sha(args.plan),
        collection_purpose=plan["purpose"],
    )
    config = vllm_config.model_config.hf_config
    with (args.output / f"rank-{rank}.jsonl").open("x") as stream, set_current_vllm_config(vllm_config), torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        if {"baselines", "linear", "mhc"} & set(plan["components"]):
            from vllm.forward_context import set_forward_context
            from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
            from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer

            # Layer 2 is the representative ordinary layer (kv/index source, ratio 2); its native
            # ffn carries the shared experts, router and routed experts the baselines measure.
            topk_buffer = torch.empty(8192, config.index_topk, dtype=torch.int32)
            streams = [torch.cuda.Stream() for _ in range(3)]
            layer = DeepseekV4DecoderLayer(vllm_config, "model.layers.2", topk_indices_buffer=topk_buffer, aux_stream_list=streams, candidate_block_buffer=None)
            receipt.setdefault("weight_initialization", {})["layer2"] = initialize_dummy_weights(layer, plan, vllm_config, device)
            receipt.setdefault("loaded_modules", {})["layer2.attn"] = dict(**describe_module(layer.attn), attention_class=type(layer.attn).__name__, kv_cache_dtype=layer.attn.kv_cache_dtype)
            if "baselines" in plan["components"]:
                lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
                receipt["weight_initialization"]["lm_head"] = initialize_dummy_weights(lm_head, plan, vllm_config, device)
                with (args.output / f"baseline-rank-{rank}.jsonl").open("x") as baselines, set_forward_context(None, vllm_config):
                    collect_baselines(layer, lm_head, plan, rank, provenance, baselines, receipt)
                del lm_head
            if "linear" in plan["components"]:
                collect_linear(layer, manifest, plan, rank, provenance, stream, receipt)
            if "mhc" in plan["components"]:
                collect_mhc(layer, manifest, plan, rank, provenance, stream, receipt)
            del layer
            gc.collect()
            torch.cuda.empty_cache()
        if "engram" in plan["components"]:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
            text = args.prompt_file.read_text()
            ids = tokenizer.encode(text)
            if len(ids) < max(plan["token_counts"]) or len(set(ids)) < 100:
                raise RuntimeError("native tokenizer corpus is too short or degenerate")
            receipt["input_provenance"] = dict(source="hf_tokenizer_and_native_ngram_hash_state", text_sha256=sha(args.prompt_file), token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), token_count=len(ids), unique_tokens=len(set(ids)))
            collect_engram(vllm_config, manifest, plan, rank, provenance, stream, ids, receipt)
    receipt.update(state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated())
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        weight_source="vllm_dummy_loader_plus_packed_fill",
        publication_permitted=False,
    )
    try:
        run(args, receipt)
    except Exception as error:  # preserve the failure receipt for admission
        receipt.update(state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["finished_unix_ns"] = time.time_ns()
        (args.output / f"isolated-rank-{rank}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
