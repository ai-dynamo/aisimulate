# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 attention-only producer for vLLM v0.30.0 (torchrun, all 40 native layers).

Builds the forty ``DeepseekV4FlashMLAAttention`` modules the way ``DeepseekV4Model`` does
(shared top-k / candidate buffers, aux streams; deepseek_v41/nvidia/model.py:455-560), binds
every native KV cache through each layer's own ``get_kv_cache_spec``/``bind_kv_cache`` and
metadata builder (compressed KV, indexer K, SWA, compressor state; the collector/vllm
DSV4 helpers), and times ``attn.forward`` per layer with the ``wo_b`` TP all-reduce moved
after the end event — the SGLang contract boundary (``collector/sglang/dsv41_attention_runner.py``).

Workloads come from ``dsv41_workloads.freeze_workloads``: uncached prefill (query, prefix 0),
cached prefill (a real prefix prefill into the same block tables, then the query extend) and
real-KV decode (prefix tokens seeded, then one decode token). Inputs are the same seeded
synthetic BF16 hidden tensor per layer through a native RMSNorm outside the timed interval;
the vLLM serving path feeds attention the mHC-fused norm output instead
(nvidia/model.py:406-423), so this isolates the attention module exactly as the SGLang
producer does. Only the ``full`` profile exists on vLLM.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from collector.sglang.dsv41_contract import canonical_json, validate_row, write_parquet
from collector.sglang.dsv41_workloads import coordinates, freeze_workloads, projected_keys
from collector.vllm.dsv41_isolated_runner import (
    BACKEND,
    CONFIG_SHA256,
    EXPECTED_SM,
    FRAMEWORK_COMMIT,
    FRAMEWORK_VERSION,
    REQUIRED_SOURCES,
    WEIGHT_INITIALIZER,
    _dispatch,
    build_vllm_config,
    describe_module,
    initialize_dummy_weights,
    prepare_private_caches,
    sha,
)

INPUT_METHOD = "seeded_bf16_hidden_native_rmsnorm_per_layer_collector_metadata_real_kv_v1"
ATTENTION_SOURCES = REQUIRED_SOURCES | {
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/utils.py",
    "v1/kv_cache_interface.py",
    "v1/worker/block_table.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "model_executor/layers/layernorm.py",
    "forward_context.py",
    "third_party/flashmla/flash_mla_interface.py",
}
PHYSICAL_KEY = ("component", "geometry", "batch_size", "prefix", "x")
POOL_LIMITS = dict(context_length=8192, max_total_tokens=8192, max_requests=4)


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
    if plan["execution_profile"] != "full":
        raise ValueError("vLLM serves DeepSeek-V4.1 with the full execution profile only")
    if manifest["config_sha256"] != CONFIG_SHA256:
        raise ValueError("unchanged forty-layer checkpoint architecture required")
    if freeze_workloads(workloads["source_payload"]) != workloads:
        raise ValueError("frozen workloads differ from their native source payload")
    if plan["weight_initializer"] != WEIGHT_INITIALIZER or plan["input_method"] != INPUT_METHOD:
        raise ValueError("prospectively declared dummy weights and synthetic inputs required")
    for key, minimum in (("warmup", 2), ("iterations", 5), ("seed", 0)):
        if type(plan[key]) is not int or plan[key] < minimum:
            raise ValueError("invalid " + key)
    if any(plan[k] != v for k, v in POOL_LIMITS.items()):
        raise ValueError("attention pool limits differ from the qualified contract")
    for case in workloads["cases"]:
        length = case["query"] + case["prefix"]
        if case["batch_size"] > POOL_LIMITS["max_requests"] or length > POOL_LIMITS["context_length"] or case["batch_size"] * length > POOL_LIMITS["max_total_tokens"]:
            raise ValueError("workload exceeds declared native pool capacity")
    if EXPECTED_SM.get(plan["expected_gpu"]) != plan["expected_sm"]:
        raise ValueError("GPU/SM identity differs")
    if plan["framework_commit"] != FRAMEWORK_COMMIT:
        raise ValueError("native attention APIs require the pinned framework commit")
    if not plan["source_pins"].keys() >= ATTENTION_SOURCES:
        raise ValueError("missing native source pins")
    if not plan["metadata_pins"].keys() >= {"config.json", "tokenizer.json", "tokenizer_config.json"}:
        raise ValueError("missing checkpoint/tokenizer pins")
    digests = [plan["image_sha256"], plan["workloads_sha256"], plan["prompt_sha256"], *plan["source_pins"].values(), *plan["metadata_pins"].values()]
    if any(not isinstance(d, str) or re.fullmatch(r"[a-f0-9]{64}", d) is None for d in digests):
        raise ValueError("immutable source/image/workload/input digests required")
    if re.fullmatch(r"sha256:[a-f0-9]{64}", plan["runtime_digest"]) is None:
        raise ValueError("immutable OCI runtime digest required")
    if re.fullmatch(r"[a-f0-9]{40}", plan["collector_revision"]) is None:
        raise ValueError("immutable collector revision required")


def layer_role(attn):
    if attn.compress_ratio == 0:
        return "swa"
    if attn.is_kv_source:
        return "full"
    if attn.is_index_source:
        return "reindex"
    return "reuse"


def validate_attention_geometry(attn, geometry, layer_id):
    """Reject a planned identity that differs from the loaded vLLM module.

    Field map (deepseek_v41/attention.py @v0.30.0): n_local_heads :239, n_local_groups :246,
    head_dim :242, q/o_lora_rank :240-241, compress_ratio :262-273, is_kv_source/is_index_source
    :285-286; the indexer replicates all heads per rank (ReplicatedLinear wq_b :1089-1095,
    weights_proj :1096-1102) exactly like the SGLang graph the geometry describes.
    """
    for field, attribute in (("num_heads", "n_local_heads"), ("o_groups", "n_local_groups"), ("head_dim", "head_dim"), ("q_lora_rank", "q_lora_rank"), ("o_lora_rank", "o_lora_rank"), ("compress_ratio", "compress_ratio")):
        if int(getattr(attn, attribute)) != geometry[field]:
            raise RuntimeError(f"native attention {attribute} differs from graph at layer {layer_id}")
    if layer_role(attn) != geometry["role"] or int(attn.window_size) != geometry["window_size"]:
        raise RuntimeError(f"native attention role/window differs from graph at layer {layer_id}")
    candidate_limit = attn.candidate_topk_blocks * attn.candidate_block_size if 0 <= attn.candidate_source_layer < layer_id and attn.compress_ratio > 0 else 0
    if candidate_limit != geometry["candidate_limit"] or (layer_id == attn.candidate_source_layer) != geometry["is_candidate_source"]:
        raise RuntimeError(f"native candidate topology differs from graph at layer {layer_id}")
    indexer = getattr(attn, "indexer", None)
    if (geometry["role"] in ("full", "reindex")) != (indexer is not None):
        raise RuntimeError(f"native indexer ownership differs from graph at layer {layer_id}")
    if indexer is None:
        return
    if (int(indexer.n_head), int(indexer.head_dim), int(indexer.topk_tokens)) != (geometry["index_n_heads"], geometry["index_head_dim"], geometry["index_topk"]):
        raise RuntimeError(f"native indexer dimensions differ from graph at layer {layer_id}")
    expected = {
        "wq_b": (geometry["index_n_heads"] * geometry["index_head_dim"], geometry["q_lora_rank"]),
        "weights_proj": (geometry["index_n_heads"], geometry["hidden_size"]),
    }
    for name, shape in expected.items():
        module = getattr(indexer, name)
        actual = (int(getattr(module, "output_size_per_partition", module.output_size)), int(getattr(module, "input_size_per_partition", module.input_size)))
        if actual != shape:  # ReplicatedLinear: full (replicated) sizes, exactly the SGLang graph's per-rank indexer
            raise RuntimeError(f"native indexer {name} partition differs from graph at layer {layer_id}")


def validate_attention_manifest(attns, manifest):
    indexer = next((a.indexer for a in attns if getattr(a, "indexer", None) is not None), None)
    if indexer is None:
        raise RuntimeError("V4.1 collection requires an actual indexer owner")
    for phase in ("context", "generation"):
        entries = manifest["phases"][phase]
        for layer_id, attn in enumerate(attns):
            shapes = [e["geometry"] for e in entries if e["component"] == "attention" and e["layer"] == layer_id]
            if len(shapes) != 1:
                raise RuntimeError(f"expected one {phase} attention geometry at layer {layer_id}")
            geometry = json.loads(shapes[0])
            validate_attention_geometry(attn, geometry, layer_id)
            if (geometry["index_n_heads"], geometry["index_head_dim"], geometry["index_topk"]) != (int(indexer.n_head), int(indexer.head_dim), int(indexer.topk_tokens)):
                raise RuntimeError(f"native indexer labels differ from graph at layer {layer_id}")


class AttentionRecorder:
    """Time each layer's native attention forward; the pure-TP wo_b reduction runs after the end event."""

    def __init__(self, attns, manifest):
        import torch
        from vllm.distributed import tensor_model_parallel_all_reduce

        self.torch, self.manifest, self.all_reduce = torch, manifest, tensor_model_parallel_all_reduce
        self.events, self.phase, self.active = [], "context", False
        self.selected = {}
        for phase, entries in manifest["phases"].items():
            for entry in entries:
                if entry["component"] == "attention":
                    self.selected.setdefault((phase, entry["layer"]), entry)
        self.originals = []
        for attn in attns:
            if attn.wo_b.reduce_results is not True or attn.wo_b.tp_size != manifest["tp_size"]:
                raise RuntimeError("attention requires unchanged pure-TP output reduction ownership")
        for index, attn in enumerate(attns):
            original = attn.forward
            self.originals.append((attn, original))
            attn.wo_b.reduce_results = False  # o_proj.py:87 returns wo_b(z); linear.py:1776 reduces when True
            attn.forward = self._wrap(attn, original, index)

    def _wrap(self, attn, original, index):
        witness = _dispatch(attn, "forward")

        def timed(*args, **kwargs):
            if attn.wo_b.reduce_results is not False:
                raise RuntimeError("native attention reduction changed during execution")
            entry = self.selected.get((self.phase, index))
            record = self.active and entry is not None
            if record:
                start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                start.record()
            result = original(*args, **kwargs)
            if record:
                end.record()
                self.events.append((entry, start, end, witness))
            return self.all_reduce(result)

        return timed

    def restore(self):
        for attn, forward in self.originals:
            attn.forward, attn.wo_b.reduce_results = forward, True
        self.originals.clear()

    def finish(self, batch, query, prefix, real_kv):
        self.torch.cuda.synchronize()
        grouped = {}
        for entry, start, end, witness in self.events:
            value = grouped.setdefault((entry["component"], entry["geometry"]), [0.0, set()])
            value[0] += start.elapsed_time(end)
            value[1].add(witness)
        self.events.clear()
        rows = []
        for (component, geometry_json), (latency, sources) in grouped.items():
            b, p, x = coordinates(component, json.loads(geometry_json), self.phase, batch, query, prefix)
            rows.append(dict(component=component, geometry=geometry_json, batch_size=b, prefix=p, x=x, latency=latency, kernel_source="+".join(sorted(sources)),
                             measurement_scope="local_compute", used_cuda_graph=False, sample_count=1, kv_seed_regime="real_kv" if (real_kv or p > 0) else "n/a"))
        return rows


class AttentionStack:
    """Forty native attention layers with their caches; collector-built metadata per forward."""

    def __init__(self, vllm_config, plan, device):
        import torch
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.models.deepseek_v41.nvidia.model import _select_dsv4_attn_cls

        self.torch, self.vllm_config, self.plan, self.device = torch, vllm_config, plan, device
        config = vllm_config.model_config.hf_config
        self.config = config
        if config.num_hidden_layers != 40 or config.model_type != "deepseek_v41":
            raise RuntimeError("unaltered native V4.1 config required")
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # DeepseekV4Model.__init__ (nvidia/model.py:482-504): shared top-k / candidate buffers, 3 aux streams
        self.topk_indices_buffer = torch.empty(max_tokens, config.index_topk, dtype=torch.int32, device=device)
        self.candidate_block_buffer = torch.empty(max_tokens, config.candidate_topk_blocks, dtype=torch.int32, device=device) if getattr(config, "candidate_source_layer_id", -1) >= 0 else None
        self.aux_streams = [torch.cuda.Stream() for _ in range(3)]
        cls = _select_dsv4_attn_cls(vllm_config)
        self.attns = torch.nn.ModuleList(
            [cls(vllm_config, prefix=f"model.layers.{i}.attn", topk_indices_buffer=self.topk_indices_buffer, aux_stream_list=self.aux_streams, candidate_block_buffer=self.candidate_block_buffer) for i in range(40)]
        )
        self.norms = torch.nn.ModuleList([RMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(40)])
        self.attention_class = cls.__name__
        self.bound_blocks = None

    def cache_layers(self, attn):
        """Every AttentionLayerBase this layer registered: compressed KV (kv sources), SWA, indexer K
        (index sources) and the compressor state cache (deepseek_v41/attention.py:463-490; V4.1's
        indexer owns no compressor of its own, unlike V4)."""
        layers = [attn, attn.swa_cache_layer]
        indexer = getattr(attn, "indexer", None)
        if indexer is not None and getattr(indexer, "k_cache", None) is not None:
            layers.append(indexer.k_cache)
        states = []
        for owner in (getattr(attn, "compressor", None), getattr(indexer, "compressor", None) if indexer is not None else None):
            state = getattr(owner, "state_cache", None) if owner is not None else None
            if state is not None:
                states.append(state)
        return layers, states

    def bind_caches(self, batch, seq_len):
        """Allocate and bind every layer's native caches for one case (sizes from the serving specs)."""
        from collector.vllm.collect_dsv4_attn import _allocate_attention_kv_cache, _cache_blocks_for_block_size

        static_ctx = self.vllm_config.compilation_config.static_forward_context
        bound = {}
        for attn in self.attns:
            layers, states = self.cache_layers(attn)
            for layer in layers + states:
                registered = static_ctx[layer.prefix]
                spec = registered.get_kv_cache_spec(self.vllm_config)
                if spec is None:
                    continue
                blocks = _cache_blocks_for_block_size(batch, seq_len, spec.block_size)
                registered.bind_kv_cache(_allocate_attention_kv_cache(registered.get_attn_backend(), spec, blocks, getattr(spec, "cache_dtype_str", None) or "auto", device=str(self.device)))
                bound[layer.prefix] = dict(spec=type(spec).__name__, block_size=spec.block_size, blocks=blocks, backend=registered.get_attn_backend().get_name())
        self.bound_blocks = bound
        return bound

    def metadata(self, common):
        """Per-prefix attention metadata for one forward, built by each layer's own builder."""
        from collector.vllm.collect_dsv4_attn import _make_builder, _remap_common_metadata

        static_ctx = self.vllm_config.compilation_config.static_forward_context
        metadata = {}
        remapped = {}
        for attn in self.attns:
            layers, states = self.cache_layers(attn)
            for layer in layers + states:
                registered = static_ctx[layer.prefix]
                spec = registered.get_kv_cache_spec(self.vllm_config)
                if spec is None:
                    continue
                if spec.block_size not in remapped:
                    remapped[spec.block_size] = common if spec.block_size == self.vllm_config.cache_config.block_size else _remap_common_metadata(common, block_size=spec.block_size, device=str(self.device))
                sub = remapped[spec.block_size]
                metadata[layer.prefix] = _make_builder(registered.get_attn_backend(), spec, layer.prefix, self.vllm_config, sub, device=str(self.device)).build(0, sub)
        return metadata

    def forward(self, common, recorder, qualify=False):
        """One native forward over all 40 attention layers under collector-built metadata."""
        import torch
        from vllm.forward_context import set_forward_context

        metadata = self.metadata(common)
        num_tokens = int(common.positions.numel())
        generator = torch.Generator(device="cuda").manual_seed(self.plan["seed"])
        hidden = torch.randn(num_tokens, self.config.hidden_size, generator=generator, dtype=torch.bfloat16, device=self.device)
        observations = []
        with set_forward_context(metadata, self.vllm_config), torch.inference_mode():
            for index, (attn, norm) in enumerate(zip(self.attns, self.norms)):
                value = norm(hidden)  # outside every recorded interval (declared operator input)
                value = attn(common.positions, value)
                if qualify:
                    observations.append((index, value.shape[0], torch.isfinite(value).all(), value.ne(0).any()))
        if qualify:
            flags = torch.stack([f for _, _, a, b in observations for f in (a, b)]).cpu().tolist()
            observations = [dict(layer=i, rows=r, finite=flags[2 * k], nonzero=flags[2 * k + 1]) for k, (i, r, _, _) in enumerate(observations)]
            if any(not o["finite"] or not o["nonzero"] for o in observations):
                raise RuntimeError("native attention produced non-finite or all-zero output: " + json.dumps([o for o in observations if not o["finite"] or not o["nonzero"]][:3]))
        return observations


def common_metadata(batch, seq_len, query_len, device, block_table=None):
    """Collector-built CommonAttentionMetadata (collect_dsv4_attn._make_common_metadata semantics)."""
    import torch
    from collector.vllm.utils import BatchSpec, create_common_attn_metadata

    common = create_common_attn_metadata(BatchSpec(seq_lens=[seq_len] * batch, query_lens=[query_len] * batch), block_size=64, device=torch.device(device), arange_block_indices=True)
    if block_table is not None:
        # same physical blocks as the forward that seeded the prefix (cached prefill / decode)
        width = common.block_table_tensor.shape[1]
        common.block_table_tensor = block_table[:, :width].contiguous() if block_table.shape[1] >= width else torch.cat([block_table, torch.zeros((batch, width - block_table.shape[1]), dtype=block_table.dtype, device=block_table.device)], dim=1)
    if getattr(common, "_seq_lens_cpu", None) is not None:
        common.seq_lens_cpu_upper_bound = common._seq_lens_cpu
    start = seq_len - query_len
    common.positions = (start + torch.arange(query_len, device=device, dtype=torch.long)).repeat(batch)
    slots = []
    table = common.block_table_tensor.cpu()
    for req in range(batch):
        for q in range(query_len):
            pos = start + q
            slots.append(int(table[req, pos // 64]) * 64 + pos % 64)
    common.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
    return common


def run_case(stack, recorder, case, execute, device):
    """Seed real KV for the case's prefix, then execute the declared forward (mirrors native_runner.run_workload)."""
    batch, query, prefix = case["batch_size"], case["query"], case["prefix"]
    total = prefix + query if case["phase"] == "context" else prefix + 1
    stack.bind_caches(batch, total)
    full = common_metadata(batch, total, query if case["phase"] == "context" else 1, device)
    if prefix:
        recorder.active = False
        seed = common_metadata(batch, prefix, prefix, device, block_table=full.block_table_tensor)
        stack.forward(seed, recorder)
    if case["phase"] == "generation":
        return execute(lambda: stack.forward(full, recorder), "generation", batch, 1, prefix, True)
    return execute(lambda: stack.forward(full, recorder), "context", batch, query, prefix, bool(prefix))


def collect_workloads(stack, recorder, plan, workloads, stream, rank, provenance, device):
    qualifications = []
    for case_index, case in enumerate(workloads["cases"]):
        recorder.active = False
        observations = run_case(stack, recorder, case, lambda call, *unused: call(), device)  # noqa: F841 (untimed replay)
        # separate untimed qualification: every layer finite and nonzero on the declared forward
        batch, query, prefix = case["batch_size"], case["query"], case["prefix"]
        total = prefix + query if case["phase"] == "context" else prefix + 1
        full = common_metadata(batch, total, query if case["phase"] == "context" else 1, device)
        qualifications.append(dict(case_id=case["case_id"], observations=stack.forward(full, recorder, qualify=True)))
        for sample in range(plan["warmup"] + plan["iterations"]):

            def execute(call, phase, batch, query, prefix, real_kv, sample=sample, case=case, case_index=case_index):
                recorder.phase, recorder.active = phase, sample >= plan["warmup"]
                result = call()
                for row in recorder.finish(batch, query, prefix, real_kv):
                    row.update(**provenance, tp_rank=rank, invocation=case_index, sample=sample, case_id=case["case_id"], input_method=INPUT_METHOD, producer_kind="native_attention_isolated", collection_purpose=plan["purpose"])
                    validate_row(row)
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                return result

            run_case(stack, recorder, case, execute, device)
    return qualifications


def aggregate_attention_records(output, plan_path, manifest_path, workloads_path):
    """Check all ranks/cases/samples; refuse collisions and smoke/heldout promotion."""
    from collector.sglang.collect_dsv41_module import aggregate_rank_records

    plan, manifest, workloads = [json.loads(path.read_text()) for path in (plan_path, manifest_path, workloads_path)]
    validate_plan(plan, manifest, workloads)
    if plan["purpose"] != "calibration" or sha(workloads_path) != plan["workloads_sha256"]:
        raise ValueError("only the unchanged declared calibration plan may be admitted")
    common_sources = None
    for rank in range(plan["tp_size"]):
        receipt = json.loads((output / f"attention-rank-{rank}.json").read_text())
        expected = dict(state="complete_pending_admission", tp_rank=rank, plan_sha256=sha(plan_path), manifest_sha256=sha(manifest_path), workloads_sha256=sha(workloads_path),
                        full_model=False, checkpoint_weights_loaded=False, input_method=INPUT_METHOD, purpose="calibration", runtime_digest=plan["runtime_digest"],
                        image_sha256=plan["image_sha256"], framework=BACKEND, framework_version=FRAMEWORK_VERSION, collector_revision=plan["collector_revision"], weight_initializer=WEIGHT_INITIALIZER)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("attention receipt differs from frozen calibration")
        sources = receipt["source_hashes"]
        if any(sources.get(p) != d for p, d in plan["source_pins"].items()) or (common_sources is not None and common_sources != sources):
            raise ValueError("attention native source identity differs from the frozen plan")
        common_sources = sources
        if len(receipt["qualifications"]) != len(workloads["cases"]) or any(len(q["observations"]) != 40 or not all(o["finite"] and o["nonzero"] for o in q["observations"]) for q in receipt["qualifications"]):
            raise ValueError("attention output qualification incomplete")
    provenance = dict(source_sha256=hashlib.sha256(canonical_json(common_sources).encode()).hexdigest(), config_sha256=manifest["config_sha256"], runtime_digest=plan["runtime_digest"], execution_profile=plan["execution_profile"])
    seen = set()
    for rank in range(plan["tp_size"]):
        for line in (output / f"rank-{rank}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if any(row.get(k) != v for k, v in provenance.items()) or row.get("producer_kind") != "native_attention_isolated" or row.get("collection_purpose") != "calibration":
                raise ValueError("whole-model or incompatible attention invocation in raw rows")
            if type(row["invocation"]) is not int or not 0 <= row["invocation"] < len(workloads["cases"]) or row["case_id"] != workloads["cases"][row["invocation"]]["case_id"]:
                raise ValueError("unplanned native attention invocation")
            seen.add((row["invocation"], row["sample"], row["tp_rank"], tuple(row[k] for k in PHYSICAL_KEY)))
    for case_index, case in enumerate(workloads["cases"]):
        for key in attention_keys(manifest, case):
            for sample in range(plan["warmup"], plan["warmup"] + plan["iterations"]):
                for rank in range(plan["tp_size"]):
                    if (case_index, sample, rank, key) not in seen:
                        raise ValueError(f"missing attention measurement for {case['case_id']} sample {sample} rank {rank}")
    if collision_owners(manifest, workloads):
        raise ValueError("different invocations share attention keys; source equivalence audit required: " + canonical_json(collision_owners(manifest, workloads)))
    rows = aggregate_rank_records([output / f"rank-{rank}.jsonl" for rank in range(plan["tp_size"])], plan["tp_size"])
    for row in rows:
        for key in ("case_id", "input_method", "producer_kind", "collection_purpose", "case_plan_sha256"):
            row.pop(key, None)
        if row["sample_count"] != plan["iterations"]:
            raise ValueError("attention sample coverage differs from plan")
    return rows


def run(args, receipt):
    plan, manifest, workloads = [json.loads(p.read_text()) for p in (args.plan, args.manifest, args.workloads)]
    validate_plan(plan, manifest, workloads)
    rank, local_rank, tp = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if tp != plan["tp_size"] or local_rank != rank:
        raise RuntimeError("single-node torchrun world must match the plan TP")
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
    gpu_uuid = str(props.uuid)
    gpu_uuid = gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid
    command = ["nvidia-smi", "--id=" + gpu_uuid, "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv,noheader,nounits"]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20)
    receipt["allocated_device_witness"] = dict(name=props.name, sm=props.major * 10 + props.minor, cuda_local_rank=local_rank, gpu_uuid=gpu_uuid, argv=command, returncode=query.returncode, stdout=query.stdout, stderr=query.stderr)
    if query.returncode or len(query.stdout.strip().splitlines()) != 1 or gpu_uuid not in query.stdout:
        raise RuntimeError("allocated CUDA UUID differs from native driver witness")
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    package = Path(vllm.__file__).resolve().parent
    sources = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    receipt["source_hashes"] = sources
    if any(sources.get(path) != digest for path, digest in plan["source_pins"].items()):
        raise RuntimeError("installed native source differs from prospective pins")
    if hashlib.sha256(canonical_json(json.loads((args.model_path / "config.json").read_text())).encode()).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("unchanged checkpoint config required")
    if args.runtime_digest != plan["runtime_digest"] or os.environ.get("DSV41_LAUNCH_IMAGE_SHA256") != plan["image_sha256"]:
        raise RuntimeError("actual launch image differs from the frozen plan")
    for name, digest in plan["metadata_pins"].items():
        if sha(args.model_path / name) != digest:
            raise RuntimeError("checkpoint/tokenizer metadata changed: " + name)
    if sha(args.prompt_file) != plan["prompt_sha256"] or sha(args.workloads) != plan["workloads_sha256"]:
        raise RuntimeError("workloads/prompt differ from the frozen plan")
    receipt.update(
        framework=BACKEND, framework_version=FRAMEWORK_VERSION, plan_sha256=sha(args.plan), manifest_sha256=sha(args.manifest), workloads_sha256=sha(args.workloads),
        raw_package_version=importlib.metadata.version("vllm"), collector_revision=plan["collector_revision"], runtime_digest=plan["runtime_digest"], image_sha256=plan["image_sha256"],
        input_method=INPUT_METHOD, weight_initializer=WEIGHT_INITIALIZER, purpose=plan["purpose"], collision_owners=collision_owners(manifest, workloads),
    )
    vllm_config = build_vllm_config(args.model_path, tp)
    init_distributed_environment(tp, rank, "env://", local_rank)
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(tp, 1)
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device)
    provenance = dict(source_sha256=hashlib.sha256(canonical_json(sources).encode()).hexdigest(), config_sha256=manifest["config_sha256"], runtime_digest=args.runtime_digest, execution_profile=manifest["execution_profile"], case_plan_sha256=sha(args.plan))
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        stack = AttentionStack(vllm_config, plan, device)
        receipt["weight_initialization"] = initialize_dummy_weights(stack.attns, plan, vllm_config, device)
        for norm in stack.norms:
            norm.weight.data.fill_(1.0)
        validate_attention_manifest(list(stack.attns), manifest)
        receipt["native_pool"] = dict(
            attention_class=stack.attention_class, backend=stack.attns[2].get_attn_backend().get_name(), kv_cache_dtype=stack.attns[2].kv_cache_dtype,
            block_size=vllm_config.cache_config.block_size, swa_block_size=stack.attns[0].swa_cache_layer.block_size, max_tokens=POOL_LIMITS["max_total_tokens"],
            kv_bytes_per_token=int(stack.attns[2].kv_bytes_per_token), kv_page_alignment=int(stack.attns[2].kv_page_alignment),
        )
        receipt["loaded_modules"] = {f"attn.{i}": describe_module(a) for i, a in enumerate(stack.attns) if i in (0, 1, 2, 3, 20, 21, 24)}
        recorder = AttentionRecorder(list(stack.attns), manifest)
        try:
            with (args.output / f"rank-{rank}.jsonl").open("x") as stream:
                receipt["qualifications"] = collect_workloads(stack, recorder, plan, workloads, stream, rank, provenance, device)
        finally:
            recorder.restore()
        receipt["bound_caches_last_case"] = stack.bound_blocks
    receipt.update(state="complete_pending_admission", memory_peak_allocated=torch.cuda.max_memory_allocated())
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ("plan", "manifest", "workloads", "model-path", "prompt-file", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--admit", type=Path, help="write the admitted calibration table here (no GPU)")
    args = parser.parse_args()
    if args.admit is not None:
        if args.admit.exists():
            raise SystemExit("refusing to overwrite an existing table")
        rows = aggregate_attention_records(args.output, args.plan, args.manifest, args.workloads)
        write_parquet(rows, args.admit)
        print(f"admitted {len(rows)} attention rows -> {args.admit}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    receipt = dict(schema="dsv41.attention-run.v1", state="started", tp_rank=rank, started_unix_ns=time.time_ns(), full_model=False, attention_measured=True, checkpoint_weights_loaded=False, publication_permitted=False, accuracy_acceptance="NOT_EVALUATED")
    try:
        run(args, receipt)
    except Exception as error:
        receipt.update(state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["finished_unix_ns"] = time.time_ns()
        (args.output / f"attention-rank-{rank}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
