# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native attention-only V4.1 collection with real request-owned KV.

Original integration adapter, informed by SGLang (Apache-2.0) at
https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc:
benchmark/one_batch.py, srt/models/deepseek_v4.py, model_loader/loader.py,
layers/attention/deepseek_v4_backend.py. See THIRD_PARTY_NOTICES.md.
This is not a CausalLM implementation: weights and hidden inputs are synthetic;
only native MQA intervals, never stub logits or whole-forward times, are data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

from .dsv41_contract import canonical_json, validate_attention_manifest, validate_row, write_parquet
from .dsv41_isolated_runner import FRAMEWORK_COMMIT, REQUIRED_SOURCES, sha
from .dsv41_native_runner import ComponentRecorder, _dispatch, run_workload
from .dsv41_workloads import freeze_workloads, projected_keys

WEIGHT_INITIALIZER = {"name": "native_uniform_attention_v1", "low": -1e-3, "high": 1e-3, "norm_weight": 1.0}
INPUT_METHOD = "seeded_bf16_hidden_native_rmsnorm_per_layer_real_native_kv_v1"
ATTENTION_SOURCES = REQUIRED_SOURCES | {
    "srt/model_executor/model_runner.py",
    "srt/layers/attention/deepseek_v4_backend.py",
    "srt/mem_cache/deepseek_v4_memory_pool.py",
    "srt/layers/attention/dsv4/dsv41_sparse.py",
    "srt/layers/attention/dsv4/compressor.py",
}
PHYSICAL_KEY = ("component", "geometry", "batch_size", "prefix", "x")


def attention_keys(manifest, case):
    return {key for key in projected_keys(manifest, case) if key[0] == "attention"}


def collision_owners(manifest, workloads):
    """Keep distinct invocations; equal physical keys alone do not authorize merge."""
    owners = defaultdict(list)
    for case in workloads["cases"]:
        for key in sorted(attention_keys(manifest, case)):
            owners[key].append(case["case_id"])
    return [{"key": list(key), "case_ids": ids} for key, ids in sorted(owners.items()) if len(ids) > 1]


def validate_plan(plan, manifest, workloads):
    if plan["schema"] != "dsv41.attention-collection.v1" or plan["purpose"] not in ("smoke", "calibration", "heldout"):
        raise ValueError("explicit attention collection purpose required")
    if type(plan["tp_size"]) is not int or plan["tp_size"] not in (2, 4):
        raise ValueError("attention requires its own TP2 or TP4 manifest")
    if (plan["tp_size"], plan["execution_profile"]) != (manifest["tp_size"], manifest["execution_profile"]):
        raise ValueError("attention plan and native graph manifest differ")
    if plan["execution_profile"] not in ("full", "decoder_bounded"):
        raise ValueError("unknown attention execution profile")
    if manifest["config_sha256"] != "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d":
        raise ValueError("unchanged forty-layer checkpoint architecture required")
    if freeze_workloads(workloads["source_payload"]) != workloads:
        raise ValueError("frozen workloads differ from their native source payload")
    if plan["weight_initializer"] != WEIGHT_INITIALIZER or plan["input_method"] != INPUT_METHOD:
        raise ValueError("prospectively declared random weights and synthetic native inputs required")
    for key, minimum in (("warmup", 2), ("iterations", 5), ("seed", 0)):
        if type(plan[key]) is not int or plan[key] < minimum:
            raise ValueError("invalid " + key)
    if (plan["context_length"], plan["max_total_tokens"], plan["max_requests"]) != (8192, 8192, 4):
        raise ValueError("attention pool limits differ from the qualified contract")
    for case in workloads["cases"]:
        length = case["query"] + case["prefix"]
        if case["batch_size"] > 4 or length > 8192 or case["batch_size"] * length > 8192:
            raise ValueError("workload exceeds declared native pool capacity")
    expected_sm = {"H100": 90, "H200": 90, "B200": 100, "GB200": 100}
    if expected_sm.get(plan["expected_gpu"]) != plan["expected_sm"]:
        raise ValueError("GPU/SM identity differs")
    if plan["moe_runner_backend"] not in ("flashinfer_mxfp4", "humming"):
        raise ValueError("explicit native quantization dispatch required")
    if plan["framework_commit"] != FRAMEWORK_COMMIT:
        raise ValueError("native attention APIs require the pinned framework commit")
    if not plan["source_pins"].keys() >= ATTENTION_SOURCES:
        raise ValueError("missing native source pins")
    if not plan["metadata_pins"].keys() >= {"config.json", "tokenizer.json", "tokenizer_config.json"}:
        raise ValueError("missing checkpoint/tokenizer pins")
    for name in plan["metadata_pins"]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("metadata pins must be snapshot-local filenames")
    digests = [
        plan["image_sha256"],
        plan["workloads_sha256"],
        plan["prompt_sha256"],
        *plan["source_pins"].values(),
        *plan["metadata_pins"].values(),
    ]
    if any(not isinstance(d, str) or re.fullmatch(r"[a-f0-9]{64}", d) is None for d in digests):
        raise ValueError("immutable source/image/workload/input digests required")
    if re.fullmatch(r"sha256:[a-f0-9]{64}", plan["runtime_digest"]) is None:
        raise ValueError("immutable OCI runtime digest required")
    if re.fullmatch(r"[a-f0-9]{40}", plan["collector_revision"]) is None:
        raise ValueError("immutable collector revision required")


class AttentionRecorder(ComponentRecorder):
    """Use the existing row/timer reduction, with attention-only native hooks."""

    def __init__(self, layers, manifest, *, torch_module=None, all_reduce=None):
        if torch_module is None:
            import torch as torch_module
        if all_reduce is None:
            from sglang.srt.distributed import tensor_model_parallel_all_reduce as all_reduce
        if len(layers) != 40:
            raise RuntimeError("all forty native attention layers are required for KV ownership")
        validate_attention_manifest(layers, manifest)
        self.torch, self.manifest = torch_module, manifest
        self.events, self.phase, self.active, self.trace = [], "context", False, False
        self.originals, self.selected = [], {}
        for phase, entries in manifest["phases"].items():
            for entry in entries:
                if entry["component"] == "attention":
                    self.selected.setdefault((phase, entry["geometry"]), entry)
        # Validate every reduction owner before mutating any method.
        for layer in layers:
            attn = layer.self_attn
            if attn.attn_tp_size != manifest["tp_size"] or attn.wo_b.reduce_results is not True:
                raise RuntimeError("attention requires unchanged pure-TP output reduction ownership")
        for index, layer in enumerate(layers):
            attn = layer.self_attn
            original = attn.forward
            self.originals.append((attn, original, attn.wo_b.reduce_results))
            attn.wo_b.reduce_results = False
            attn.forward = self._wrap(attn, original, index, all_reduce)

    def _wrap(self, attn, original, index, all_reduce):
        witness = _dispatch(attn, "forward")

        def timed(*args, **kwargs):
            if attn.attn_tp_size != self.manifest["tp_size"] or attn.wo_b.reduce_results is not False:
                raise RuntimeError("native attention TP/reduction changed during execution")
            entries = [e for (phase, _), e in self.selected.items() if phase == self.phase and e["layer"] == index]
            record = self.active and bool(entries)
            if record:
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                start.record()
            result = original(*args, **kwargs)
            if record:
                end.record()
                self.events.append((entries[0], start, end, witness))
            # deepseek_v4.py:2078-2082: the pure-TP wo_b reduction follows
            # local projection. Only its placement outside the interval changes.
            return all_reduce(result)

        return timed

    def restore(self):
        for attn, forward, reduce_results in self.originals:
            attn.forward, attn.wo_b.reduce_results = forward, reduce_results
        self.originals.clear()


def validate_attention_outputs(observations):
    """Validate actual MQA outputs from an untimed pass, never the logits stub."""
    if not observations or len(observations) % 40:
        raise RuntimeError("missing complete native attention output evidence")
    for start in range(0, len(observations), 40):
        group = observations[start : start + 40]
        if [value["layer"] for value in group] != list(range(40)):
            raise RuntimeError("native attention output layer coverage differs")
        if any(value["finite"] is not True or value["nonzero"] is not True or value["rows"] < 1 for value in group):
            raise RuntimeError("native attention output is nonfinite, empty or degenerate")


def build_native_runner(bench, server, model_config, ps, plan, receipt):
    import torch
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor import model_runner as native_runner
    from sglang.srt.model_loader.loader import DummyModelLoader, _get_quantization_config, _post_load_weights
    from sglang.srt.model_loader.weight_utils import initialize_dummy_weights
    from sglang.srt.models import deepseek_v4 as native_model
    from torch import nn

    cfg = model_config.hf_text_config
    if cfg.num_hidden_layers != 40 or cfg.model_type != "deepseek_v41":
        raise RuntimeError("unaltered native V4.1 config required")

    class LayerFacade(nn.Module):
        def __init__(self, quant, index):
            super().__init__()
            self.self_attn = native_model.MQALayer(cfg, index, quant, prefix=f"model.layers.{index}.self_attn")
            self.input_layernorm = native_model.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_attention_layernorm = native_model.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        def refresh_mhc_norm_weight_cache(self):
            native_model.DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(self)

    class AttentionStack(nn.Module):
        def __init__(self, quant):
            super().__init__()
            self.config, self.quant_config = cfg, quant
            self.start_layer, self.end_layer = 0, 40
            self.wo_a_fp8 = native_model.wo_a_fp8_gemm_enabled(quant)
            self.model = nn.Module()
            self.model.start_layer, self.model.end_layer = 0, 40
            self.model.layers = nn.ModuleList([LayerFacade(quant, i) for i in range(40)])
            native_model.get_attn_tp_context().init_context(cfg.q_lora_rank, is_dsa=True)
            self.late_layer_start = (
                max(cfg.kv_source_layer_ids) + 1 if plan["execution_profile"] == "decoder_bounded" else None
            )
            self.qualifying, self.observations = False, []

        def _setup_fp8_wo_a_scales(self, is_nextn):
            native_model.DeepseekV4ForCausalLM._setup_fp8_wo_a_scales(self, is_nextn)

        def post_load_weights(self):
            native_model.DeepseekV4ForCausalLM.post_load_weights(self)

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch, **kwargs):
            # Declared operator input, NOT model hidden states. Native serving
            # deepseek_v4.py:2743-2764 applies input_layernorm before MQA;
            # normalization/RNG are outside every recorded MQA interval.
            generator = torch.Generator(device=input_ids.device).manual_seed(plan["seed"])
            hidden = torch.randn(
                (input_ids.numel(), cfg.hidden_size), generator=generator, dtype=torch.bfloat16, device=input_ids.device
            )
            backend, saved, tail = native_model.get_attn_backend(), None, None
            if self.late_layer_start is not None and forward_batch.forward_mode.is_extend_without_speculative():
                tail = backend.tail_forward_metadata.late_layer_tail
            try:
                for index, layer in enumerate(self.model.layers):
                    # Native bounded-tail ownership, deepseek_v4.py:3542-3563;
                    # native pool/backend retain the actual source-layer KV.
                    if tail is not None and index == self.late_layer_start:
                        saved = backend.enter_late_layer_tail(forward_batch)
                        hidden, positions = tail.rows(hidden), tail.positions
                    value = layer.input_layernorm(hidden)
                    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
                        value = layer.self_attn(x=value, positions=positions, forward_batch=forward_batch, x_quant=None)
                    if self.qualifying:
                        # Only untimed qualification executes validation kernels.
                        # Retain GPU scalar flags; one host transfer follows the
                        # whole workload, with no per-layer sync/IO in timings.
                        self.observations.append(
                            (index, value.shape[0], torch.isfinite(value).all(), value.ne(0).any())
                        )
            finally:
                if saved is not None:
                    backend.exit_late_layer_tail(saved, forward_batch)
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(
                    (forward_batch.batch_size, cfg.vocab_size), dtype=torch.float32, device=input_ids.device
                )
            )

        def finish_qualification(self):
            flags = (
                torch.stack([flag for _, _, finite, nonzero in self.observations for flag in (finite, nonzero)])
                .cpu()
                .tolist()
            )
            observations = [
                dict(layer=index, rows=rows, finite=flags[2 * i], nonzero=flags[2 * i + 1])
                for i, (index, rows, _, _) in enumerate(self.observations)
            ]
            self.observations.clear()
            validate_attention_outputs(observations)
            return observations

    class AttentionRunner(native_runner.ModelRunner):
        def load_model(self):
            begin = time.monotonic()
            before = native_runner.get_available_gpu_memory(self.device, self.gpu_id)
            self.load_config = LoadConfig(load_format="dummy")
            quant = _get_quantization_config(self.model_config, self.load_config)
            with torch.device(f"cuda:{ps.gpu_id}"):
                self.model = AttentionStack(quant)
            # Native loader.py:1592-1621 order: random storage, model post-load,
            # then native quant-method processing. No fake model weights load.
            initialize_dummy_weights(
                self.model,
                low=plan["weight_initializer"]["low"],
                high=plan["weight_initializer"]["high"],
                seed=plan["seed"],
            )
            with torch.no_grad():
                for layer in self.model.model.layers:
                    layer.input_layernorm.weight.fill_(plan["weight_initializer"]["norm_weight"])
                    layer.post_attention_layernorm.weight.fill_(plan["weight_initializer"]["norm_weight"])
            _post_load_weights(self.model)
            for child in list(self.model.modules()):
                method = getattr(child, "quant_method", None)
                if method is not None:
                    method.process_weights_after_loading(child)
            receipt["native_wo_a_post_load"] = []
            for index, layer in enumerate(self.model.model.layers):
                weight = layer.self_attn.wo_a.weight
                scale = getattr(layer.self_attn.wo_a, "weight_scale_inv", None)
                if self.model.wo_a_fp8:
                    if (
                        weight.dtype != torch.float8_e4m3fn
                        or scale is None
                        or not isinstance(getattr(scale, "format_ue8m0", None), bool)
                    ):
                        raise RuntimeError("native grouped wo_a FP8 post-load contract is missing")
                elif weight.dtype != torch.bfloat16:
                    raise RuntimeError("native wo_a dtype differs")
                receipt["native_wo_a_post_load"].append(
                    dict(
                        layer=index,
                        shape=list(weight.shape),
                        dtype=str(weight.dtype),
                        scale_dtype=str(scale.dtype) if scale is not None else None,
                    )
                )
            self.model.eval()
            self.loader = DummyModelLoader(self.load_config)
            self.sliding_window_size = native_runner.resolve_sliding_window_size(self.model, self.model_config)
            self.prefill_aware_swa, self.dtype = False, self.model_config.dtype
            self.weight_load_mem_usage = before - native_runner.get_available_gpu_memory(self.device, self.gpu_id)
            self.weight_load_time = time.monotonic() - begin

    runner = AttentionRunner(
        model_config=model_config,
        mem_fraction_static=server.mem_fraction_static,
        gpu_id=ps.gpu_id,
        ps=ps,
        nccl_port=int(os.environ["MASTER_PORT"]),
        server_args=server,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    runner.init_cuda_graphs()
    receipt["native_pool"] = dict(
        type=type(runner.token_to_kv_pool).__name__,
        backend=type(runner.attn_backend).__name__,
        max_tokens=runner.max_total_num_tokens,
        low_ratios=sorted(runner.token_to_kv_pool.kv_pools),
    )
    if receipt["native_pool"]["type"] != "DeepSeekV4TokenToKVPool" or receipt["native_pool"]["low_ratios"] != [1, 2]:
        raise RuntimeError("native compressed KV pool ownership differs")
    return bench._TorchBenchRunner(runner)


def collect_workloads(
    runner, recorder, bench, token_ids, plan, workloads, stream, rank, provenance, *, qualifications=None
):
    stack = runner.torch_runner.model
    if qualifications is None:
        qualifications = []
    for case_index, case in enumerate(workloads["cases"]):
        # An entirely separate native replay checks actual outputs, including
        # real prefix/decode seeding; stub logits are deliberately ignored.
        recorder.active, stack.qualifying = False, True
        try:
            run_workload(runner, recorder, bench, token_ids, case, lambda call, *unused: call())
            observations = stack.finish_qualification()
            expected_forwards = 2 if case["prefix"] or case["phase"] == "generation" else 1
            if len(observations) != 40 * expected_forwards:
                raise RuntimeError("native seed/measured forward coverage differs: " + case["case_id"])
            qualifications.append(dict(case_id=case["case_id"], observations=observations))
        finally:
            stack.qualifying = False
        for sample in range(plan["warmup"] + plan["iterations"]):

            def execute(call, phase, batch, query, prefix, real_kv):
                recorder.phase, recorder.active = phase, sample >= plan["warmup"]
                result = call()
                rows = recorder.finish(batch, query, prefix, real_kv)
                for row in rows:
                    row.update(
                        **provenance,
                        tp_rank=rank,
                        invocation=case_index,
                        sample=sample,
                        case_id=case["case_id"],
                        input_method=INPUT_METHOD,
                        producer_kind="native_attention_isolated",
                        collection_purpose=plan["purpose"],
                    )
                    validate_row(row)
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                return result

            run_workload(runner, recorder, bench, token_ids, case, execute)
    return qualifications


def aggregate_attention_records(output, plan_path, manifest_path, workloads_path):
    """Check all ranks/cases/samples and prohibit silent whole-model collisions."""
    from .collect_dsv41_module import aggregate_rank_records

    plan, manifest, workloads = [json.loads(path.read_text()) for path in (plan_path, manifest_path, workloads_path)]
    validate_plan(plan, manifest, workloads)
    if plan["purpose"] != "calibration" or sha(workloads_path) != plan["workloads_sha256"]:
        raise ValueError("only the unchanged declared calibration plan may be admitted")
    common_sources = None
    for rank in range(plan["tp_size"]):
        receipt = json.loads((output / f"attention-rank-{rank}.json").read_text())
        expected_receipt = dict(
            state="complete_pending_admission",
            tp_rank=rank,
            plan_sha256=sha(plan_path),
            manifest_sha256=sha(manifest_path),
            workloads_sha256=sha(workloads_path),
            full_model=False,
            checkpoint_weights_loaded=False,
            input_method=INPUT_METHOD,
            purpose="calibration",
        )
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ValueError("attention receipt differs from frozen calibration")
        sources = receipt["source_hashes"]
        if any(sources.get(path) != digest for path, digest in plan["source_pins"].items()):
            raise ValueError("native source receipt differs from plan")
        if common_sources is not None and sources != common_sources:
            raise ValueError("installed source identities differ between ranks")
        common_sources = sources
        expected_provenance = dict(
            source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
            config_sha256=manifest["config_sha256"],
            runtime_digest=plan["runtime_digest"],
            execution_profile=plan["execution_profile"],
            case_plan_sha256=sha(plan_path),
            collection_purpose="calibration",
            producer_kind="native_attention_isolated",
            input_method=INPUT_METHOD,
            tp_rank=rank,
        )
        qualifications = receipt["qualifications"]
        if [item["case_id"] for item in qualifications] != [case["case_id"] for case in workloads["cases"]]:
            raise ValueError("attention qualification case coverage differs")
        for item, case in zip(qualifications, workloads["cases"], strict=True):
            validate_attention_outputs(item["observations"])
            expected_forwards = 2 if case["prefix"] or case["phase"] == "generation" else 1
            if len(item["observations"]) != 40 * expected_forwards:
                raise ValueError("missing real seed forward qualification")
        expected = {
            (index, sample, key)
            for index, case in enumerate(workloads["cases"])
            for sample in range(plan["warmup"], plan["warmup"] + plan["iterations"])
            for key in attention_keys(manifest, case)
        }
        actual = set()
        for line in (output / f"rank-{rank}.jsonl").read_text().splitlines():
            row = json.loads(line)
            validate_row(row)
            if any(row.get(key) != value for key, value in expected_provenance.items()):
                raise ValueError("whole-model or incompatible attention invocation in raw rows")
            if type(row["invocation"]) is not int or not 0 <= row["invocation"] < len(workloads["cases"]):
                raise ValueError("unplanned native attention invocation")
            if row["case_id"] != workloads["cases"][row["invocation"]]["case_id"]:
                raise ValueError("native attention case identity differs")
            key = (row["invocation"], row["sample"], tuple(row[k] for k in PHYSICAL_KEY))
            if key not in expected or key in actual or row["sample_count"] != 1:
                raise ValueError("duplicate or unplanned attention sample")
            actual.add(key)
        if actual != expected:
            raise ValueError("incomplete native attention coverage")
    collisions = collision_owners(manifest, workloads)
    if collisions:
        raise ValueError(
            "different invocations share attention keys; source equivalence audit required: "
            + canonical_json(collisions)
        )
    rows = aggregate_rank_records([output / f"rank-{rank}.jsonl" for rank in range(plan["tp_size"])], plan["tp_size"])
    return rows


def prepare_caches(rank, receipt):
    root = Path(os.environ["DSV41_PRIVATE_CACHE"])
    home_cache = root / "home-cache"
    targets = [Path.home() / ".cache", Path("/root/.cache")]
    if any(not target.samefile(home_cache) for target in targets):
        raise RuntimeError("HOME/root cache must already bind the private allocation cache")
    receipt["cache_bindings"] = [str(target) for target in targets]
    for variable, leaf in (
        ("TRITON_CACHE_DIR", "triton"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("DEEP_GEMM_CACHE_DIR", "deep-gemm"),
    ):
        destination = root / f"rank-{rank}" / leaf
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    if os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE") != "env://":
        raise RuntimeError("native torchrun bootstrap requires env://")


def run(args, receipt):
    plan, manifest, workloads = [json.loads(path.read_text()) for path in (args.plan, args.manifest, args.workloads)]
    validate_plan(plan, manifest, workloads)
    if sha(args.workloads) != plan["workloads_sha256"] or sha(args.prompt_file) != plan["prompt_sha256"]:
        raise RuntimeError("frozen workloads or tokenizer corpus changed")
    if (
        args.runtime_digest != plan["runtime_digest"]
        or os.environ.get("DSV41_LAUNCH_IMAGE_SHA256") != plan["image_sha256"]
    ):
        raise RuntimeError("launch image differs from frozen plan")
    for name, digest in plan["metadata_pins"].items():
        if sha(args.model_path / name) != digest:
            raise RuntimeError("checkpoint metadata changed: " + name)
    config = json.loads((args.model_path / "config.json").read_text())
    if hashlib.sha256(canonical_json(config).encode()).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("checkpoint config differs from consumer graph")
    rank, local_rank, tp = [int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    if tp != plan["tp_size"] or not 0 <= rank < tp or local_rank != rank:
        raise RuntimeError("single-node pure TP allocation differs from plan")
    prepare_caches(rank, receipt)
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != tp:
        raise RuntimeError("visible CUDA device count differs from allocated TP")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    props = torch.cuda.get_device_properties(local_rank)
    if (
        re.search(r"\b" + re.escape(plan["expected_gpu"]) + r"\b", props.name, re.IGNORECASE) is None
        or props.major * 10 + props.minor != plan["expected_sm"]
    ):
        raise RuntimeError("actual CUDA device differs from plan")
    gpu_uuid = str(props.uuid)
    if re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid) is None:
        raise RuntimeError("unexpected CUDA UUID")
    gpu_uuid = gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid
    query = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    receipt["allocated_device_witness"] = dict(
        name=props.name,
        uuid=gpu_uuid,
        sm=props.major * 10 + props.minor,
        returncode=query.returncode,
        stdout=query.stdout,
        stderr=query.stderr,
    )
    if query.returncode or len(query.stdout.strip().splitlines()) != 1 or gpu_uuid not in query.stdout:
        raise RuntimeError("allocated CUDA UUID differs from native driver witness")
    from sglang.benchmark import one_batch as bench
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    package = Path(bench.__file__).resolve().parents[1]
    # Same complete installed-source digest as the whole-model producer.
    sources = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*.py"))}
    receipt["source_hashes"] = sources
    if any(sources.get(path) != digest for path, digest in plan["source_pins"].items()):
        raise RuntimeError("installed native source differs from prospective pins")
    receipt.update(
        plan_sha256=sha(args.plan),
        manifest_sha256=sha(args.manifest),
        workloads_sha256=sha(args.workloads),
        framework_version="dev-" + FRAMEWORK_COMMIT,
        raw_package_version=importlib.metadata.version("sglang"),
        collector_revision=plan["collector_revision"],
        runtime_digest=plan["runtime_digest"],
        image_sha256=plan["image_sha256"],
        input_method=INPUT_METHOD,
        weight_initializer=WEIGHT_INITIALIZER,
        purpose=plan["purpose"],
        collision_owners=collision_owners(manifest, workloads),
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
        context_length=plan["context_length"],
        max_total_tokens=plan["max_total_tokens"],
        max_running_requests=plan["max_requests"],
        mem_fraction_static=0.6,
        enable_decoder_swa_bounded_replay=plan["execution_profile"] == "decoder_bounded",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )
    bench.publish(server, role="scheduler")
    bench.initialize_moe_config()
    bench.initialize_fp8_gemm_config()
    bench.initialize_fp4_gemm_config()
    ps = ParallelState.trivial(tp_rank=rank, tp_size=tp, attn_tp_rank=rank, attn_tp_size=tp, gpu_id=local_rank)
    runner = build_native_runner(bench, server, ModelConfig.from_server_args(server), ps, plan, receipt)
    tokenizer = get_tokenizer(str(args.model_path), trust_remote_code=True, tokenizer_backend="huggingface")
    ids = tokenizer.encode(args.prompt_file.read_text())
    if len(ids) < max(c["query"] + c["prefix"] for c in workloads["cases"]) or len(set(ids)) < 100:
        raise RuntimeError("native tokenizer corpus is too short or degenerate")
    receipt["input_provenance"] = dict(
        text_sha256=sha(args.prompt_file),
        token_count=len(ids),
        unique_tokens=len(set(ids)),
        token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        seed=plan["seed"],
        hidden_source="same seeded synthetic tensor per layer; native RMSNorm outside intervals",
        stub_output="zero logits only for native request lifecycle; never validated or timed",
    )
    provenance = dict(
        source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(),
        config_sha256=manifest["config_sha256"],
        runtime_digest=plan["runtime_digest"],
        execution_profile=manifest["execution_profile"],
        case_plan_sha256=sha(args.plan),
    )
    recorder = AttentionRecorder(runner.torch_runner.model.model.layers, manifest)
    try:
        with (args.output / f"rank-{rank}.jsonl").open("x") as stream:
            collect_workloads(
                runner,
                recorder,
                bench,
                ids,
                plan,
                workloads,
                stream,
                rank,
                provenance,
                qualifications=receipt.setdefault("qualifications", []),
            )
    finally:
        recorder.restore()
    receipt.update(state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated())
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "manifest", "workloads", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("model-path", "prompt-file"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--runtime-digest")
    parser.add_argument("--admit", type=Path, help="write a new table after strict calibration admission")
    args = parser.parse_args()
    if args.admit:
        rows = aggregate_attention_records(args.output, args.plan, args.manifest, args.workloads)
        # Exclusive creation rejects whole-model tables and accidental reruns.
        with args.admit.open("xb") as destination:
            write_parquet(rows, destination)
        return
    if any(getattr(args, key) is None for key in ("model_path", "prompt_file", "runtime_digest")):
        parser.error("collection requires model-path, prompt-file and runtime-digest")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    path = args.output / f"attention-rank-{rank}.json"
    receipt = dict(
        schema="dsv41.attention-run.v1",
        state="started",
        tp_rank=rank,
        started_unix_ns=time.time_ns(),
        full_model=False,
        checkpoint_weights_loaded=False,
        publication_permitted=False,
        accuracy_acceptance="NOT_EVALUATED",
    )
    with path.open("x") as stream:
        json.dump(receipt, stream)
    try:
        run(args, receipt)
    except BaseException as error:
        receipt.update(
            state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
        raise
    finally:
        receipt["finished_unix_ns"] = time.time_ns()
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
