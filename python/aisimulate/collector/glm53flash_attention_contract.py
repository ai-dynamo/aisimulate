# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract for GLM-5.3-Flash NoPE sparse-MLA + IndexPool attention rows.

The collectors (``collector/{sglang,vllm}/glm53flash_attention_runner.py``) load
one real checkpoint attention layer through the framework's own model builder
and time it standalone over real KV/IndexPool state. This module owns
everything that does not need a GPU: the physical key derived from the
checkpoint configuration, the workload plan, raw-row validation, per-sample
rank reduction and the strict Parquet writer that rejects duplicate keys and
mixed provenance. See ``collector/README.glm53flash_attention.md``.

The published table is the one consumed by the Rust reader
``crates/core/src/perfmodel/perf_database/glm53flash.rs`` (DeepSeek-V4.1 module
schema): ``geometry`` is the canonical sorted compact JSON of the model's
``Glm53Attention`` operator body without ``name``/``measured``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

BASENAME = "glm53_attention_module_perf.parquet"
FAMILY_DIR = "glm53_attention"
OP_NAME = "glm53flash_attention"
MEASUREMENT_SCOPE = "local_compute"
BACKENDS = ("vllm", "sglang")
CHECKPOINTS = {
    # checkpoint_format -> (HF model id, immutable revision)
    "fp8": ("zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"),
    "nvfp4": ("nvidia/GLM-5.3-Flash-NVFP4", "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"),
}
# Exact runtimes the collectors are pinned to: stock vLLM v0.31.0 (upstream
# tag commit db9527a46873454610df6dbedf79a36d6bf1a7f6, aarch64 image digest
# below) and SGLang 0.5.20. Stock v0.31.0 still corrupts the boundary pool of
# a prefill chunk whose start is not a multiple of index_kpool (4)
# (vllm/models/glm5next/nvidia/sparse_indexer.py:47-90 _kpool_compress_insert,
# "Assumes pool-aligned chunk starts"); every planned vLLM chunk therefore
# starts on a multiple of KPOOL_ALIGN (see build_plan and the vLLM runner).
RUNTIME_VERSIONS = {"vllm": "0.31.0", "sglang": "0.5.20"}
RUNTIME_IMAGES = {
    "vllm": "sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971",
    "sglang": "sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862",
}
KPOOL_ALIGN = 4
PHASES = ("context", "generation")
# Published columns, exactly the Rust reader's schema.
COLUMNS = (
    "component",
    "geometry",
    "batch_size",
    "prefix",
    "x",
    "latency",
    "kernel_source",
    "measurement_scope",
    "source_sha256",
    "config_sha256",
    "runtime_digest",
    "used_cuda_graph",
    "sample_count",
    "kv_seed_regime",
    "execution_profile",
)
INTEGER_COLUMNS = ("batch_size", "prefix", "x", "sample_count")
KEY_COLUMNS = ("geometry", "batch_size", "prefix", "x")
BODY_FIELDS = (
    "backend",
    "checkpoint_format",
    "conv_kernel",
    "gate_lower_bound",
    "head_dim",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_pool",
    "index_topk",
    "is_context",
    "kv_cache_dtype",
    "kv_lora_rank",
    "layer_kind",
    "num_heads",
    "projection_quant_mode",
    "q_lora_rank",
    "tp_size",
    "value_head_dim",
)
# GPU kernel time only (no host launch gaps): the module call is replayed under
# the framework's own serving CUDA graph mechanism (prefill: the breakable /
# piecewise prefill graph whose eager breaks launch from the host; decode: a
# full graph captured from the framework's decode-graph capture), and each
# repetition's latency is the union of the GPU-busy intervals of the CUPTI
# kernel, memcpy and memset activities attributed to that repetition
# (glm53flash_attention_runtime.KernelTimer). This matches the graph-mode
# kernel-time basis of the KDA and the other operator tables.
KERNEL_PREFILL = "cupti_gpu_busy_union_framework_breakable_module_graph_replay"
KERNEL_DECODE = "cupti_gpu_busy_union_captured_module_graph_replay"
TIMING_METHODS = {
    "context": {KERNEL_PREFILL: True},
    "generation": {KERNEL_DECODE: True},
}
# Documented properties of the published tables (collection_meta.yaml notes).
_COMMON_NOTES = [
    "IndexPool alignment: every prefill chunk starts on a multiple of 4 (the stock vLLM 0.31.0 "
    "defect's trigger); decode rows keep their true sequence length L (prompt L-1, whose last seeding "
    "chunk may end off the pool grid and completes in the tail as in serving).",
    "Regular rows are collected at the serving context limit 131079; long-context rows (prefill "
    "prefix >= 262144, decode x >= 262144, B=1) at the model limit 1048576 are table-extrapolation "
    "coverage. Each attempt's limit is its max_model_len.",
]
TABLE_NOTES = {
    "vllm": _COMMON_NOTES
    + [
        "Known bias: kernel-only rows read 2-12% (median 6%) above the per-layer MLA kernels of stock "
        "vllm serve under nsys (fp8-tp2, nvfp4-tp4; 7 points each). Per kernel, the cuBLAS (nvjet) "
        "projection GEMMs run 5-18% slower in the standalone module replay while fmha and fwht match; "
        "200 warmup repetitions do not change it. Same standalone-op method as the other tables.",
    ],
    "sglang": _COMMON_NOTES
    + [
        "Validation: kernel-only rows are 0.93-1.02x (one point 1.105x) the per-layer MLA kernels of "
        "the SGLang serving node traces at 8 geometries x 4 deployments.",
    ],
}
# Capacity-only attempt settings (memory pool sizes, allocator split); they
# never change a kernel, bucket or schedule and are frozen per manifest.
CAPACITY_KNOBS = ("vllm_gpu_memory_utilization", "sglang_mem_fraction_static", "allocator_max_split_size_mb")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def mla_layers(config: dict) -> list[int]:
    return [i for i, kind in enumerate(text_config(config)["layer_types"]) if kind == "deepseek_sparse_attention"]


def projection_quant_mode(backend: str, checkpoint_format: str) -> str:
    """Actual MLA projection precision in the pinned framework model builders.

    vLLM 0.31.0 ``models/glm5next/common/model.py:318-334`` builds
    ``Glm5NextMLAAttention(quant_config=None)`` and dequantizes FP8 q_a/kv_a/o
    weights to BF16 on load, so every vLLM projection is BF16. SGLang 0.5.20
    ``models/glm5_next.py`` passes the checkpoint quant_config to
    ``DeepseekV2AttentionMLA``: the FP8 checkpoint serves fused q_a/kv_a, q_b
    and o_proj as FP8 block-128 (kv_b_proj and the indexer are listed in
    ``modules_to_not_convert``), while the NVFP4 checkpoint excludes
    ``layers.*.self_attn*`` entirely. The runners verify this against the
    loaded modules' quant methods before timing.
    """
    if backend not in BACKENDS or checkpoint_format not in CHECKPOINTS:
        raise ValueError(f"unsupported GLM attention identity {backend!r}/{checkpoint_format!r}")
    return "fp8_block" if backend == "sglang" and checkpoint_format == "fp8" else "bfloat16"


def representative_layer_is_uniform(config: dict, layer_id: int, checkpoint_format: str) -> None:
    """Every NoPE sparse-MLA layer must be interchangeable with ``layer_id``.

    All 11 layers use the same module class, dimensions, own indexer
    (``indexer_types == "full"``, no top-k reuse) and the same per-layer
    quantization exclusions. A checkpoint where this fails would need one key
    per layer, so the representative sample is rejected instead.
    """
    text = text_config(config)
    layers = mla_layers(config)
    if layer_id not in layers:
        raise ValueError(f"layer {layer_id} is not a NoPE sparse-MLA layer")
    indexer_types = text.get("indexer_types")
    if indexer_types is not None and any(indexer_types[i] != "full" for i in layers):
        raise ValueError("a sparse-MLA layer reuses another layer's top-k indices")
    quant = config.get("quantization_config") or {}
    excluded = quant.get("modules_to_not_convert") or quant.get("ignore") or []
    if checkpoint_format == "nvfp4":
        excluded = (quant.get("quantization") or quant).get("exclude_modules", excluded)

    def signature(layer: int) -> list[str]:
        marker = re.compile(rf"layers\.{layer}\.self_attn")
        return sorted(marker.sub("layers.N.self_attn", name) for name in excluded if marker.search(name))

    reference = signature(layer_id)
    for layer in layers:
        if signature(layer) != reference:
            raise ValueError(f"layer {layer} quantization differs from representative layer {layer_id}")


def geometry(config: dict, backend: str, checkpoint_format: str, tp_size: int) -> dict:
    """Physical key fields for one (backend, checkpoint, TP) deployment."""
    text = text_config(config)
    heads = text["num_attention_heads"]
    if tp_size < 1 or heads % tp_size:
        raise ValueError(f"TP {tp_size} does not divide {heads} attention heads")
    if text.get("qk_rope_head_dim", 0) != 0 or not text.get("mla_use_nope", False):
        raise ValueError("GLM-5.3-Flash attention contract requires NoPE MLA")
    return {
        "backend": backend,
        "checkpoint_format": checkpoint_format,
        "projection_quant_mode": projection_quant_mode(backend, checkpoint_format),
        "kv_cache_dtype": "fp8",
        "tp_size": tp_size,
        "hidden_size": text["hidden_size"],
        "num_heads": heads // tp_size,
        "head_dim": text["qk_nope_head_dim"],
        "q_lora_rank": text["q_lora_rank"],
        "kv_lora_rank": text["kv_lora_rank"],
        "value_head_dim": text["v_head_dim"],
        "index_n_heads": text["index_n_heads"],
        "index_head_dim": text["index_head_dim"],
        "index_topk": text["index_topk"],
        "index_pool": text["index_kpool"],
        "conv_kernel": text["linear_attn_config"]["short_conv_kernel_size"],
        "gate_lower_bound": float(text["linear_attn_config"]["gate_lower_bound"]),
    }


def attention_body(flat: dict, is_context: bool) -> dict:
    """The model's ``Glm53Attention`` body without ``name``/``measured``.

    Matches PR #323 ``sdk/models/glm53flash.py`` ``attention()`` for a sparse
    MLA layer (KDA-only fields carry the checkpoint's values), which is the
    measured key of the Rust reader.
    """
    return {
        "backend": flat["backend"],
        "checkpoint_format": flat["checkpoint_format"],
        "conv_kernel": flat["conv_kernel"],
        "gate_lower_bound": flat["gate_lower_bound"],
        "head_dim": flat["head_dim"],
        "hidden_size": flat["hidden_size"],
        "index_head_dim": flat["index_head_dim"],
        "index_n_heads": flat["index_n_heads"],
        "index_pool": flat["index_pool"],
        "index_topk": flat["index_topk"],
        "is_context": bool(is_context),
        "kv_cache_dtype": flat["kv_cache_dtype"],
        "kv_lora_rank": flat["kv_lora_rank"],
        "layer_kind": "sparse_mla",
        "num_heads": flat["num_heads"],
        "projection_quant_mode": flat["projection_quant_mode"],
        "q_lora_rank": flat["q_lora_rank"],
        "tp_size": flat["tp_size"],
        "value_head_dim": flat["value_head_dim"],
    }


def geometry_key(body: dict) -> str:
    """serde_json of a sorted map: identical to the Rust ``geometry()``."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def indexer_regime(phase: str, prefix: int, x: int, index_topk: int) -> str:
    """Short-prefix regimes select every pool without MQA scoring.

    vLLM 0.31.0 skips scoring when the batch's maximum prefill sequence
    (prefix + query; models/glm5next/nvidia/sparse_indexer.py:250-271) or
    decode sequence length (common/sparse_indexer.py:116-129) is at most
    ``topk_tokens``; SGLang dsa_indexer_kpool.py skips extend logits when
    ``max_kv_len <= index_topk``. The key keeps both regimes apart so a curve
    never interpolates across that discontinuity.
    """
    if phase == "context":
        return "short" if prefix + x <= index_topk else "pooled"
    if phase == "generation":
        if prefix:
            raise ValueError("decode rows use absolute sequence length x with prefix=0")
        return "short" if x <= index_topk else "pooled"
    raise ValueError(f"unknown phase {phase!r}")


def seed_chunk(budget: int, batch: int) -> int:
    """Per-request seeding chunk: the batch's share of the step budget, floored
    to the IndexPool size so every seeding chunk of every request starts on a
    multiple of KPOOL_ALIGN (stock vLLM 0.31.0 corrupts the boundary pool of an
    unaligned chunk start; SGLang only seeds state with it)."""
    chunk = budget // batch // KPOOL_ALIGN * KPOOL_ALIGN
    if chunk < KPOOL_ALIGN:
        raise ValueError(f"batch {batch} leaves no aligned seeding chunk in a {budget}-token step")
    return chunk


def _prefix_grid(spec, batch: int, query: int) -> list[int]:
    """Prefix grid: one list for every (B, Q), or explicit lists per B (and Q)."""
    if not isinstance(spec, dict):
        return list(spec)
    per_batch = spec.get(str(batch), spec.get(batch))
    if per_batch is None:
        raise ValueError(f"no prefill prefix grid for batch {batch}")
    if isinstance(per_batch, dict):
        per_query = per_batch.get(str(query), per_batch.get(query))
        if per_query is None:
            raise ValueError(f"no prefill prefix grid for batch {batch} query {query}")
        return list(per_query)
    return list(per_batch)


def unaligned_targets(plan: dict) -> list[str]:
    """Prefill targets whose chunk start or length is off the IndexPool grid.

    Stock vLLM 0.31.0 must never run such a chunk (``_kpool_compress_insert``);
    SGLang is unaffected and keeps its own geometry.
    """
    bad = []
    for request_set in plan["sets"]:
        if request_set["seed_chunk"] % KPOOL_ALIGN:
            bad.append(f"{request_set['set_id']}:seed_chunk={request_set['seed_chunk']}")
        if request_set["phase"] == "context":
            bad += [
                f"{request_set['set_id']}:prefix={prefix}"
                for prefix in request_set["targets"]
                if prefix % KPOOL_ALIGN or request_set["query"] % KPOOL_ALIGN
            ]
    return bad


def _context_sets(section: dict, budget: int, max_context: int, max_model_len: int, label: str) -> list[dict]:
    """Request sets of one context class (regular serving or long context).

    Prefill sets keep B homogeneous requests in lockstep. The request grows
    through ``prefix_lengths`` with seeding chunks of ``seed_chunk(budget, B)``
    tokens per request (inside the serving chunked-prefill budget), and at each
    target prefix one step extends every request by exactly Q tokens. Decode
    sets seed B requests to L-1 tokens, then one real decode step reads L
    tokens (x=L). Every prompt stays below ``max_model_len`` (vLLM rejects a
    prompt of max_model_len tokens; a decode needs L <= max_model_len - 1).
    """
    sets = []
    prefix_id = "" if label == "regular" else f"{label}-"
    prefill = section["prefill"]
    for batch in prefill["batch_sizes"]:
        queries = prefill["query_lengths"].get(str(batch), prefill["query_lengths"].get(batch))
        if queries is None:
            raise ValueError(f"no prefill query grid for batch {batch}")
        for query in queries:
            if batch * query > budget:
                raise ValueError(f"prefill B={batch} Q={query} exceeds the {budget}-token step budget")
            prefixes = [p for p in _prefix_grid(prefill["prefix_lengths"], batch, query) if p + query <= max_context]
            if not prefixes:
                raise ValueError(f"prefill B={batch} Q={query} has no admissible prefix")
            if max(prefixes) + query >= max_model_len:
                raise ValueError(f"prefill B={batch} Q={query} prompt reaches max_model_len {max_model_len}")
            # A measured step advances every request by Q, so one request set
            # can only visit prefixes at least Q apart. Greedy chains keep the
            # long, expensive prefixes in the first chain; the remainder are
            # short prefixes that are cheap to seed again.
            chains: list[list[int]] = []
            for prefix in sorted(prefixes):
                for chain in chains:
                    if prefix >= chain[-1] + query:
                        chain.append(prefix)
                        break
                else:
                    chains.append([prefix])
            for index, chain in enumerate(chains):
                sets.append(
                    {
                        "set_id": f"{prefix_id}prefill-b{batch}-q{query}-c{index}",
                        "phase": "context",
                        "context_class": label,
                        "max_model_len": max_model_len,
                        "batch_size": batch,
                        "query": query,
                        "targets": chain,
                        "seed_chunk": seed_chunk(budget, batch),
                    }
                )
    decode = section["decode"]
    for batch in decode["batch_sizes"]:
        lengths = decode["sequence_lengths"].get(str(batch), decode["sequence_lengths"].get(batch))
        if lengths is None:
            raise ValueError(f"no decode length grid for batch {batch}")
        if any(length < 2 or length > max_context or length >= max_model_len for length in lengths):
            raise ValueError("decode lengths need one real past token and must fit the context")
        sets.append(
            {
                "set_id": f"{prefix_id}decode-b{batch}",
                "phase": "generation",
                "context_class": label,
                "max_model_len": max_model_len,
                "batch_size": batch,
                "targets": sorted(lengths),
                "seed_chunk": seed_chunk(budget, batch),
            }
        )
    return sets


def build_plan(sweep: dict) -> dict:
    """Expand the base-op sweep into request sets with explicit target steps.

    The regular class covers the serving context (``max_context``, served with
    ``max_model_len``); the optional ``long_context`` class adds B=1 rows up to
    the model's 1M positions, collected by attempts whose server limit is the
    long class's ``max_model_len`` (a capacity setting recorded per set). SGLang
    may continue a decode set to the next L (the decoded token remains part of
    the real sequence); vLLM cannot append prompt tokens after decoding and
    runs one request set per L.
    """
    budget = int(sweep["max_step_tokens"])
    if budget % KPOOL_ALIGN:
        raise ValueError("the step budget must be a multiple of the IndexPool size")
    max_context = int(sweep["max_context"])
    max_model_len = int(sweep["max_model_len"])
    sets = _context_sets(sweep, budget, max_context, max_model_len, "regular")
    plan = {
        "schema_version": 2,
        "op": OP_NAME,
        "layer_id": int(sweep["layer_id"]),
        "warmup": int(sweep["warmup"]),
        "iterations": int(sweep["iterations"]),
        "max_step_tokens": budget,
        "max_context": max_context,
        "max_model_len": max_model_len,
    }
    long_context = sweep.get("long_context")
    if long_context is not None:
        long_len = int(long_context["max_model_len"])
        long_max = int(long_context["max_context"])
        if long_max <= max_context:
            raise ValueError("long-context rows must extend beyond the regular context")
        sets += _context_sets(long_context, budget, long_max, long_len, "long")
        plan["long_context"] = {"max_context": long_max, "max_model_len": long_len}
    identities = [s["set_id"] for s in sets]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate request-set identity")
    plan["sets"] = sets
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def context_class_sets(plan: dict, context_class: str) -> list[str]:
    """Set identities of one context class (one server ``max_model_len``)."""
    return [s["set_id"] for s in plan["sets"] if s["context_class"] == context_class]


def selected_max_model_len(manifest: dict) -> int:
    """The single server context limit of an attempt's selected sets."""
    limits = {s["max_model_len"] for s in selected_plan(manifest)["sets"]}
    if len(limits) != 1:
        raise ValueError(f"an attempt must select sets of one context class, got max_model_len {sorted(limits)}")
    return limits.pop()


def target_keys(plan: dict) -> list[tuple[str, int, int, int]]:
    """(phase, batch, prefix, x) for every planned measurement."""
    keys = []
    for request_set in plan["sets"]:
        batch = request_set["batch_size"]
        if request_set["phase"] == "context":
            keys += [("context", batch, prefix, request_set["query"]) for prefix in request_set["targets"]]
        else:
            keys += [("generation", batch, 0, length) for length in request_set["targets"]]
    if len(set(keys)) != len(keys):
        raise ValueError("plan measures one physical key twice")
    return keys


def _is_uint32(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= 2**32 - 1


def validate_row(row: dict) -> dict:
    """Reject a row the Rust reader would reject; return its parsed body."""
    if tuple(row) != COLUMNS:
        raise ValueError(f"row columns {tuple(row)} differ from the published schema {COLUMNS}")
    for key in INTEGER_COLUMNS:
        if not _is_uint32(row[key]):
            raise ValueError(f"{key} must be an exact uint32")
    if row["component"] != "attention":
        raise ValueError("component must be attention")
    body = json.loads(row["geometry"])
    if geometry_key(body) != row["geometry"]:
        raise ValueError("geometry must be canonical sorted compact JSON")
    if set(body) != set(BODY_FIELDS) or body["layer_kind"] != "sparse_mla":
        raise ValueError("geometry must be a sparse_mla Glm53Attention body without name/measured")
    if body["backend"] not in BACKENDS or body["checkpoint_format"] not in CHECKPOINTS:
        raise ValueError("unknown backend or checkpoint")
    if body["projection_quant_mode"] != projection_quant_mode(body["backend"], body["checkpoint_format"]):
        raise ValueError("projection precision contradicts the pinned framework model builder")
    if body["kv_cache_dtype"] != "fp8" or body["tp_size"] * body["num_heads"] != 64:
        raise ValueError("GLM attention rows require FP8 KV and TP-sharded 64 MLA heads")
    if not row["batch_size"] or not row["x"] or not row["sample_count"]:
        raise ValueError("empty measurement")
    latency = row["latency"]
    if isinstance(latency, bool) or not isinstance(latency, float) or not math.isfinite(latency) or latency <= 0:
        raise ValueError("latency must be positive finite milliseconds")
    if not isinstance(row["used_cuda_graph"], bool):
        raise ValueError("used_cuda_graph must be boolean")
    phase = "context" if body["is_context"] else "generation"
    if row["used_cuda_graph"] not in TIMING_METHODS[phase].values():
        raise ValueError("CUDA graph use does not match a serving execution mode of this phase")
    if not body["is_context"] and row["prefix"]:
        raise ValueError("decode uses absolute sequence length x with prefix=0")
    if row["measurement_scope"] != MEASUREMENT_SCOPE or not row["kernel_source"].strip():
        raise ValueError("rows need a dispatch witness and local_compute scope (output all-reduce excluded)")
    if row["kv_seed_regime"] != "real_kv" or row["execution_profile"] != "full":
        raise ValueError("every GLM attention row uses real KV/IndexPool state and the full profile")
    for key in ("source_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"invalid {key}")
    if row["runtime_digest"] != RUNTIME_IMAGES[body["backend"]]:
        raise ValueError("runtime digest is not the pinned immutable image")
    return body


def physical_key(row: dict) -> tuple:
    return tuple(row[c] for c in KEY_COLUMNS)


def aggregate_rank_samples(records: list[dict], tp_size: int) -> tuple[list[dict], list[dict]]:
    """Median over repetitions of the per-repetition maximum across TP ranks.

    Each raw record is one rank's timing of one repetition of one target. A
    target is admitted only when every rank reported the same repetition set
    with identical invocation identity; duplicate ranks/repetitions fail.
    Returns the published rows and adjacent per-row evidence.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record["target_id"]].append(record)
    rows, evidence = [], []
    for target, items in sorted(groups.items()):
        identity_fields = ("key", "provenance", "kernel_source", "timing_method", "used_cuda_graph")
        identities = {canonical_json({k: item[k] for k in identity_fields}) for item in items}
        if len(identities) != 1:
            raise ValueError(f"target {target} mixes invocation identities across ranks")
        samples: dict[int, dict[int, float]] = defaultdict(dict)
        for item in items:
            rank, repetition = item["tp_rank"], item["repetition"]
            latency = item["latency_ms"]
            if not isinstance(latency, float) or not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"target {target} has an invalid latency")
            if rank in samples[repetition]:
                raise ValueError(f"target {target} repeats rank {rank} repetition {repetition}")
            samples[repetition][rank] = latency
        if any(set(ranks) != set(range(tp_size)) for ranks in samples.values()):
            raise ValueError(f"target {target} is missing a TP rank")
        maxima = [max(ranks.values()) for _, ranks in sorted(samples.items())]
        first = items[0]
        key, provenance = first["key"], first["provenance"]
        row = {
            "component": "attention",
            "geometry": key["geometry"],
            "batch_size": key["batch_size"],
            "prefix": key["prefix"],
            "x": key["x"],
            "latency": float(statistics.median(maxima)),
            "kernel_source": first["kernel_source"],
            "measurement_scope": MEASUREMENT_SCOPE,
            "source_sha256": provenance["source_sha256"],
            "config_sha256": provenance["config_sha256"],
            "runtime_digest": provenance["runtime_digest"],
            "used_cuda_graph": first["used_cuda_graph"],
            "sample_count": len(maxima),
            "kv_seed_regime": "real_kv",
            "execution_profile": "full",
        }
        body = validate_row(row)
        if not all(item["extra"].get("finite", False) for item in items):
            raise ValueError(f"target {target} produced nonfinite attention output")
        phase = "context" if body["is_context"] else "generation"
        if TIMING_METHODS[phase].get(first["timing_method"]) != row["used_cuda_graph"]:
            raise ValueError(f"target {target} used an unexpected timing method")
        rows.append(row)
        evidence.append(
            {
                "target_id": target,
                "geometry": row["geometry"],
                "batch_size": row["batch_size"],
                "prefix": row["prefix"],
                "x": row["x"],
                "indexer_regime": indexer_regime(phase, row["prefix"], row["x"], body["index_topk"]),
                "latency_min": float(min(maxima)),
                "latency_max": float(max(maxima)),
                "latency_cv": float(statistics.pstdev(maxima) / statistics.fmean(maxima)),
                "rank_max_ms": maxima,
                "timing_method": first["timing_method"],
                **{k: provenance[k] for k in ("framework_version", "checkpoint_revision", "layer_id")},
                "extra": sorted({canonical_json(item["extra"]) for item in items}),
            }
        )
    return rows, evidence


def validate_table(rows: list[dict]) -> None:
    """Duplicate keys and mixed provenance fail.

    ``source_sha256``/``runtime_digest`` are table-wide (one runtime per
    <backend>/<version> directory); ``config_sha256`` is uniform per
    checkpoint; ``used_cuda_graph`` per (checkpoint, TP, phase) (serving runs
    both phases under the frameworks' CUDA graphs). ``kernel_source`` is a
    per-row witness: the dispatched method changes with the IndexPool regime.
    """
    if not rows:
        raise ValueError("an empty GLM attention table cannot be published")
    keys = set()
    table = set()
    by_checkpoint = defaultdict(set)
    by_phase = defaultdict(set)
    for row in rows:
        body = validate_row(row)
        key = physical_key(row)
        if key in keys:
            raise ValueError(f"duplicate physical GLM attention key {key}")
        keys.add(key)
        table.add((body["backend"], row["source_sha256"], row["runtime_digest"]))
        by_checkpoint[body["checkpoint_format"]].add(row["config_sha256"])
        phase = (body["checkpoint_format"], body["tp_size"], body["is_context"])
        by_phase[phase].add(row["used_cuda_graph"])
    if len(table) != 1:
        raise ValueError("a table needs one backend/runtime/source identity")
    if any(len(v) != 1 for v in by_checkpoint.values()):
        raise ValueError("a checkpoint's rows mix configuration identities")
    if any(len(v) != 1 for v in by_phase.values()):
        raise ValueError("one deployment phase mixes CUDA graph identities")


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_table(rows)
    fields = []
    for column in COLUMNS:
        if column in INTEGER_COLUMNS:
            kind = pa.int64()
        elif column == "latency":
            kind = pa.float64()
        elif column == "used_cuda_graph":
            kind = pa.bool_()
        else:
            kind = pa.string()
        fields.append(pa.field(column, kind, nullable=False))
    ordered = sorted(rows, key=physical_key)
    table = pa.Table.from_pylist([{c: row[c] for c in COLUMNS} for row in ordered], schema=pa.schema(fields))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def selected_plan(manifest: dict) -> dict:
    """The plan restricted to a split attempt's set selection (if any)."""
    plan = manifest["plan"]
    only = manifest.get("only_sets")
    if only is None:
        return plan
    known = {s["set_id"] for s in plan["sets"]}
    if not only or not set(only) <= known:
        raise ValueError(f"split attempt selects unknown or no sets: {only}")
    return {**plan, "sets": [s for s in plan["sets"] if s["set_id"] in set(only)]}


def _target_context(request_set: dict, value: int) -> tuple[tuple, int]:
    batch = request_set["batch_size"]
    if request_set["phase"] == "context":
        return ("context", batch, value, request_set["query"]), value + request_set["query"]
    return ("generation", batch, 0, value), value


def memory_drops(plan: dict, budget: dict | None) -> list[dict]:
    """Generation-time memory-feasibility filter (layer_permissions.md).

    ``budget`` holds the deployment's measured capacity: ``kv_tokens`` (the
    framework's KV pool size at the attempt's memory setting) and
    ``transient_gib`` (device memory left for transient buffers). A target is
    never queued when its live KV (``batch * context`` tokens) exceeds the pool,
    or when the IndexPool MQA logits of one full step at its context
    (``step_tokens * batch * ceil(context / index_pool) * 4`` bytes: one fp32
    logit per pool for every query token of the step against every request's
    pooled keys) exceed the transient memory. Size vs capacity only.
    """
    if not budget:
        return []
    drops = []
    for request_set in plan["sets"]:
        batch = request_set["batch_size"]
        for value in request_set["targets"]:
            key, context = _target_context(request_set, value)
            kv = batch * context
            logits = plan["max_step_tokens"] * batch * -(-context // KPOOL_ALIGN) * 4
            reasons = []
            if kv > budget["kv_tokens"]:
                reasons.append(f"kv {kv} tokens > pool {budget['kv_tokens']}")
            if logits > budget["transient_gib"] * 2**30:
                reasons.append(f"mqa logits {logits / 2**30:.1f} GiB > {budget['transient_gib']} GiB")
            if reasons:
                drops.append({"set_id": request_set["set_id"], "key": list(key), "reason": "; ".join(reasons)})
    return drops


def queued_plan(manifest: dict) -> dict:
    """The attempt's selected sets without its memory-dropped targets."""
    plan = selected_plan(manifest)
    dropped = manifest.get("memory_drops") or []
    if dropped != memory_drops(plan, manifest.get("memory_budget")):
        raise ValueError("manifest memory drops differ from its memory budget")
    gone = {(d["set_id"], tuple(d["key"])) for d in dropped}
    sets = []
    for request_set in plan["sets"]:
        keep = [
            v for v in request_set["targets"] if (request_set["set_id"], _target_context(request_set, v)[0]) not in gone
        ]
        if keep:
            sets.append({**request_set, "targets": keep})
    return {**plan, "sets": sets}


def check_split_closure(attempts: list[tuple[dict, list[dict]]], classified: list[dict] | None = None) -> None:
    """Attempts of one deployment must cover its planned keys exactly once.

    Planned keys are excluded only by an attempt's memory drops or by an
    explicit classified failure (deployment, key, reason, evidence); a
    classified key that was measured is a contradiction.
    """
    by_deployment: dict[tuple, list[tuple[dict, list[dict]]]] = {}
    for manifest, rows in attempts:
        geometry = manifest["geometry"]
        by_deployment.setdefault((geometry["checkpoint_format"], geometry["tp_size"]), []).append((manifest, rows))
    for deployment, group in by_deployment.items():
        plans = {json.dumps(m["plan"], sort_keys=True) for m, _ in group}
        if len(plans) != 1:
            raise ValueError(f"{deployment} attempts disagree on the plan")
        seen: set[tuple] = set()
        for _, rows in group:
            keys = {(r["geometry"], r["batch_size"], r["prefix"], r["x"]) for r in rows}
            if keys & seen:
                raise ValueError(f"{deployment} attempts measure a key twice")
            seen |= keys
        dropped = {tuple(d["key"]) for m, _ in group for d in m.get("memory_drops") or []}
        name = f"{deployment[0]}-tp{deployment[1]}"
        failed = {tuple(c["key"]) for c in classified or [] if c["deployment"] == name}
        planned = set(target_keys(group[0][0]["plan"]))
        if not failed <= planned - dropped:
            raise ValueError(f"{deployment} classified failures are not queued planned keys")
        bodies = {geometry_key(attention_body(group[0][0]["geometry"], ph == "context")): ph for ph in PHASES}
        measured = {(bodies[k[0]], k[1], k[2], k[3]) for k in seen}
        if measured & failed:
            raise ValueError(f"{deployment} classified failures were measured: {sorted(measured & failed)[:3]}")
        wanted = [key for key in planned if key not in dropped and key not in failed]
        if len(seen) != len(wanted):
            raise ValueError(f"{deployment} attempts cover {len(seen)} of {len(wanted)} planned keys")


def load_attempt(attempt: Path, partial: bool = False) -> tuple[dict, list[dict], list[dict]]:
    """Admit one runner attempt: completion receipt, plan closure, every target.

    ``attempt`` holds the frozen ``manifest.json``; the runner's per-rank
    streams and completion receipt live in ``attempt/raw``. A ``partial``
    attempt (no receipt; its progress log records a failed set) contributes
    only the targets it measured completely on every rank; the returned
    manifest then lists ``admitted_sets`` (``set:n/m`` when partly measured)
    and ``failed_sets``, and the remaining keys must come from other attempts
    or be classified failures (finalize closure).
    """
    attempt = Path(attempt)
    raw = attempt / "raw"
    complete = (raw / "COMPLETE").is_file()
    if complete == partial:
        raise ValueError(f"{attempt} has {'a' if complete else 'no'} completion receipt; partial={partial}")
    manifest = json.loads((attempt / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest["manifest_sha256"] != sha256_json(body):
        raise ValueError(f"{attempt} manifest identity does not match its content")
    plan = manifest["plan"]
    if plan != build_plan(manifest["sweep"]):
        raise ValueError(f"{attempt} plan differs from its frozen sweep")
    if manifest["role"] != "full":
        raise ValueError(f"{attempt} is a {manifest['role']} attempt; only full attempts publish")
    tp_size = manifest["geometry"]["tp_size"]
    paths = sorted(raw.glob("rank-*.jsonl"))
    if {p.name for p in paths} != {f"rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError(f"{attempt} is missing TP rank files")
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("record") == "sample"]
    bodies = {geometry_key(attention_body(manifest["geometry"], phase == "context")): phase for phase in PHASES}
    for record in records:
        if record["key"]["geometry"] not in bodies:
            raise ValueError(f"{attempt} sample geometry differs from its deployment")
        if record["provenance"]["layer_id"] != manifest["layer_id"]:
            raise ValueError(f"{attempt} sample layer differs from the manifest")
    queued = queued_plan(manifest)
    if partial:
        failed = set()
        for path in raw.glob("progress*.jsonl"):
            for line in path.read_text().splitlines():
                entry = json.loads(line) if line.strip() else {}
                if entry.get("status") == "failed":
                    failed.add(entry["set_id"])
        if not failed:
            raise ValueError(f"{attempt} is partial but records no failed set")
        reps_by_rank: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        for record in records:
            reps_by_rank[record["target_id"]][record["tp_rank"]].add(record["repetition"])
        # A target is complete when every rank wrote the same repetitions.
        full = {
            t
            for t, ranks in reps_by_rank.items()
            if set(ranks) == set(range(tp_size)) and len({frozenset(v) for v in ranks.values()}) == 1
        }
        # Every target measured completely on every rank is admitted; the
        # rest of a failed set must come from other attempts or be classified.
        admitted, sets = [], []
        for request_set in queued["sets"]:
            keep = []
            for value in request_set["targets"]:
                phase, batch, prefix, x = _target_context(request_set, value)[0]
                if f"{phase}-b{batch}-p{prefix}-x{x}" in full:
                    keep.append(value)
            if keep:
                sets.append({**request_set, "targets": keep})
                admitted.append(
                    request_set["set_id"]
                    if len(keep) == len(request_set["targets"])
                    else f"{request_set['set_id']}:{len(keep)}/{len(request_set['targets'])}"
                )
        queued = {**queued, "sets": sets}
        keep_ids = {f"{k[0]}-b{k[1]}-p{k[2]}-x{k[3]}" for k in target_keys(queued)}
        records = [r for r in records if r["target_id"] in keep_ids]
        manifest = {
            **manifest,
            "admitted_sets": admitted,
            "failed_sets": sorted(failed),
        }
    rows, evidence = aggregate_rank_samples(records, tp_size)
    measured = {(bodies[r["geometry"]], r["batch_size"], r["prefix"], r["x"]) for r in rows}
    expected = set(target_keys(queued))
    if manifest["max_model_len"] != selected_max_model_len(manifest):
        raise ValueError(f"{attempt} server max_model_len differs from its selected context class")
    if measured != expected:
        missing = sorted(expected - measured)[:4]
        extra = sorted(measured - expected)[:4]
        raise ValueError(f"{attempt} measured keys differ from its plan: missing {missing}, unplanned {extra}")
    return manifest, rows, evidence


def _execution_modes(rows: list[dict]) -> dict[str, str]:
    modes = {}
    for row in rows:
        body = json.loads(row["geometry"])
        phase = "context" if body["is_context"] else "generation"
        key = f"{body['checkpoint_format']}-tp{body['tp_size']}-{phase}"
        modes[key] = "cuda_graph" if row["used_cuda_graph"] else "eager"
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="print the frozen plan for a sweep YAML")
    plan_parser.add_argument("--sweep", type=Path, required=True)
    finalize = sub.add_parser("finalize", help="merge admitted attempts into one backend/version table")
    finalize.add_argument("attempts", type=Path, nargs="*")
    finalize.add_argument(
        "--classified-failures", type=Path, help="JSON list of {deployment, key, reason, evidence} planned keys"
    )
    finalize.add_argument(
        "--partial", type=Path, action="append", default=[], help="failed attempt: admit its completed sets only"
    )
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--evidence", type=Path, required=True, help="per-row sample evidence JSON")
    args = parser.parse_args()
    if args.command == "plan":
        import yaml

        sweep = yaml.safe_load(args.sweep.read_text())["common_case_values"][OP_NAME]
        plan = build_plan(sweep)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "targets": len(target_keys(plan))}))
        return
    rows, evidence, manifests, loaded = [], [], [], []
    complete_keys: set[tuple] = set()
    for attempt, partial in [(a, False) for a in args.attempts] + [(a, True) for a in args.partial]:
        manifest, attempt_rows, attempt_evidence = load_attempt(attempt, partial=partial)
        superseded = 0
        if partial:
            # A key also measured by a complete (fresh-process) attempt, or by
            # a partial attempt listed earlier, is taken from that attempt.
            kept = [r for r in attempt_rows if physical_key(r) not in complete_keys]
            superseded = len(attempt_rows) - len(kept)
            keys = {physical_key(r) for r in kept}
            complete_keys |= keys
            attempt_rows = kept
            attempt_evidence = [
                e for e in attempt_evidence if (e["geometry"], e["batch_size"], e["prefix"], e["x"]) in keys
            ]
        else:
            complete_keys |= {physical_key(r) for r in attempt_rows}
        loaded.append((manifest, attempt_rows))
        rows += attempt_rows
        evidence += [{**e, "attempt": Path(attempt).name} for e in attempt_evidence]
        manifests.append(
            {
                "attempt": Path(attempt).name,
                "manifest_sha256": manifest["manifest_sha256"],
                "plan_sha256": manifest["plan"]["plan_sha256"],
                "deployment": f"{manifest['geometry']['checkpoint_format']}-tp{manifest['geometry']['tp_size']}",
                "source_commit": manifest["source_commit"],
                "max_model_len": manifest["max_model_len"],
                **{k: manifest[k] for k in CAPACITY_KNOBS if manifest.get(k) is not None},
                **({"only_sets": manifest["only_sets"]} if manifest.get("only_sets") is not None else {}),
                **(
                    {
                        "partial": {
                            "admitted_sets": manifest["admitted_sets"],
                            "failed_sets": manifest["failed_sets"],
                            "superseded_by_complete_attempts": superseded,
                        }
                    }
                    if partial
                    else {}
                ),
                **(
                    {"memory_budget": manifest["memory_budget"], "memory_drops": manifest["memory_drops"]}
                    if manifest.get("memory_drops")
                    else {}
                ),
            }
        )
    classified = json.loads(args.classified_failures.read_text()) if args.classified_failures else []
    for entry in classified:
        if set(entry) != {"deployment", "key", "reason", "evidence"} or not entry["reason"] or not entry["evidence"]:
            raise ValueError(f"classified failure needs deployment, key, reason and evidence: {entry}")
    check_split_closure(loaded, classified)
    # Request token provenance (glm53flash_attention_tokens) must be one spec.
    input_specs = {canonical_json(m.get("input_tokens")) for m, _ in loaded}
    if len(input_specs) != 1 or None in (m.get("input_tokens") for m, _ in loaded):
        raise ValueError("attempts must share one recorded input_tokens generator spec")
    input_tokens = {"source": "seeded_random_tokens", **json.loads(input_specs.pop())}
    backends = {json.loads(r["geometry"])["backend"] for r in rows}
    if len(backends) != 1:
        raise ValueError("one table holds one backend")
    backend = backends.pop()
    write_parquet(rows, args.output)
    Path(args.evidence).write_text(json.dumps({"attempts": manifests, "rows": evidence}, indent=1) + "\n")
    import yaml

    meta = {
        "schema_version": 1,
        "runtime": {
            "framework": backend,
            "version": RUNTIME_VERSIONS[backend],
            "image": {"vllm": "vllm/vllm-openai", "sglang": "lmsysorg/sglang"}[backend],
            "image_digest": RUNTIME_IMAGES[backend],
        },
        "tables": {
            Path(BASENAME).stem: {
                "status": "complete",
                "rows": len(rows),
                "data_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                "collector": f"collector.{backend}.glm53flash_attention_runner",
                "measurement": "one real sparse-MLA layer (3); output all-reduce excluded; GPU kernel time only",
                # Per phase: CUPTI GPU-busy union of the module's kernels,
                # memcpys and memsets per repetition under the serving graphs.
                "timing_method": {phase: sorted(methods) for phase, methods in TIMING_METHODS.items()},
                "notes": TABLE_NOTES[backend],
                "input_tokens": input_tokens,
                # One execution mode per geometry (checkpoint, TP, phase).
                "execution_mode": dict(sorted(_execution_modes(rows).items())),
                "attempts": manifests,
                **({"classified_failures": classified} if classified else {}),
            }
        },
    }
    (Path(args.output).parent / "collection_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
