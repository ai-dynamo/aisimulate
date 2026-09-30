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
# Exact runtimes the collectors are pinned to. vLLM runs with the reviewed
# retained-tail overlay because stock 0.30.0 corrupts unaligned pooled prefill
# (vllm/model_executor/layers/sparse_attn_indexer_kpool.py: _kpool_compress_insert
# borrows the request's retained tail; kv_cache_interface.py KpoolTailSpec
# uses_slot_mapping=False). Both overlay files change the IndexPool/KPool path
# this table measures, so the stock and tail identities are never mixed.
RUNTIME_VERSIONS = {"vllm": "0.30.0+glm53tail.eb4704514fdf", "sglang": "0.5.20"}
RUNTIME_IMAGES = {
    "vllm": "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56",
    "sglang": "sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862",
}
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
TIMING_METHODS = {
    # Prefill runs eagerly in both pinned serving configurations: SGLang's
    # prefill graph backend is "disabled" and vLLM only captures <=64-token
    # batches. Decode runs under full CUDA graphs in both.
    "context": ("cuda_events_eager_repeated_module_call", False),
    "generation": ("cuda_events_captured_module_graph_replay", True),
}


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

    vLLM 0.30.0 ``models/glm5next/nvidia/model.py`` builds
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

    vLLM sparse_attn_indexer_kpool.py skips scoring when the batch's maximum
    prefill sequence (prefix + query) or decode sequence length is at most
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


def build_plan(sweep: dict) -> dict:
    """Expand the base-op sweep into request sets with explicit target steps.

    Prefill sets keep B homogeneous requests in lockstep. The request grows
    through ``prefix_lengths`` with seeding chunks of at most
    ``max_step_tokens // B`` tokens per request (the serving chunked-prefill
    budget), and at each target prefix one step extends every request by
    exactly Q tokens. Decode sets seed B requests to L-1 tokens, then one real
    decode step reads L tokens (x=L). SGLang may continue a decode set to the
    next L (the decoded token remains part of the real sequence); vLLM cannot
    append prompt tokens after decoding and runs one set per L.
    """
    budget = int(sweep["max_step_tokens"])
    max_context = int(sweep["max_context"])
    prefill = sweep["prefill"]
    sets = []
    for batch in prefill["batch_sizes"]:
        queries = prefill["query_lengths"].get(str(batch), prefill["query_lengths"].get(batch))
        if queries is None:
            raise ValueError(f"no prefill query grid for batch {batch}")
        for query in queries:
            if batch * query > budget:
                raise ValueError(f"prefill B={batch} Q={query} exceeds the {budget}-token step budget")
            prefixes = [p for p in prefill["prefix_lengths"] if p + query <= max_context]
            if not prefixes:
                raise ValueError(f"prefill B={batch} Q={query} has no admissible prefix")
            if any(p % 4 or query % 4 for p in prefixes):
                raise ValueError("prefill targets must be IndexPool-aligned (multiples of 4)")
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
                        "set_id": f"prefill-b{batch}-q{query}-c{index}",
                        "phase": "context",
                        "batch_size": batch,
                        "query": query,
                        "targets": chain,
                        "seed_chunk": budget // batch,
                    }
                )
    decode = sweep["decode"]
    for batch in decode["batch_sizes"]:
        lengths = decode["sequence_lengths"].get(str(batch), decode["sequence_lengths"].get(batch))
        if lengths is None:
            raise ValueError(f"no decode length grid for batch {batch}")
        if any(length < 2 or length > max_context for length in lengths):
            raise ValueError("decode lengths need one real past token and must fit the context")
        sets.append(
            {
                "set_id": f"decode-b{batch}",
                "phase": "generation",
                "batch_size": batch,
                "targets": sorted(lengths),
                "seed_chunk": budget // batch,
            }
        )
    identities = [s["set_id"] for s in sets]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate request-set identity")
    plan = {
        "schema_version": 1,
        "op": OP_NAME,
        "layer_id": int(sweep["layer_id"]),
        "warmup": int(sweep["warmup"]),
        "iterations": int(sweep["iterations"]),
        "max_step_tokens": budget,
        "max_context": max_context,
        "sets": sets,
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


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
    if row["used_cuda_graph"] != TIMING_METHODS[phase][1]:
        raise ValueError("CUDA graph use does not match the serving execution mode of this phase")
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
        if first["timing_method"] != TIMING_METHODS[phase][0]:
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
    checkpoint; ``used_cuda_graph`` and the kernel witness per (checkpoint, TP,
    phase), because serving runs prefill eagerly and decode under CUDA graphs.
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
        by_phase[phase].add((row["used_cuda_graph"], row["kernel_source"]))
    if len(table) != 1:
        raise ValueError("a table needs one backend/runtime/source identity")
    if any(len(v) != 1 for v in by_checkpoint.values()):
        raise ValueError("a checkpoint's rows mix configuration identities")
    if any(len(v) != 1 for v in by_phase.values()):
        raise ValueError("one deployment phase mixes CUDA graph or kernel identities")


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


def load_attempt(attempt: Path) -> tuple[dict, list[dict], list[dict]]:
    """Admit one runner attempt: completion receipt, plan closure, every target.

    ``attempt`` holds the frozen ``manifest.json``; the runner's per-rank
    streams and completion receipt live in ``attempt/raw``.
    """
    attempt = Path(attempt)
    raw = attempt / "raw"
    if not (raw / "COMPLETE").is_file():
        raise ValueError(f"{attempt} has no completion receipt")
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
    rows, evidence = aggregate_rank_samples(records, tp_size)
    measured = {(bodies[r["geometry"]], r["batch_size"], r["prefix"], r["x"]) for r in rows}
    expected = set(target_keys(plan))
    if measured != expected:
        missing = sorted(expected - measured)[:4]
        extra = sorted(measured - expected)[:4]
        raise ValueError(f"{attempt} measured keys differ from its plan: missing {missing}, unplanned {extra}")
    return manifest, rows, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="print the frozen plan for a sweep YAML")
    plan_parser.add_argument("--sweep", type=Path, required=True)
    finalize = sub.add_parser("finalize", help="merge admitted attempts into one backend/version table")
    finalize.add_argument("attempts", type=Path, nargs="+")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--evidence", type=Path, required=True, help="per-row sample evidence JSON")
    args = parser.parse_args()
    if args.command == "plan":
        import yaml

        sweep = yaml.safe_load(args.sweep.read_text())["common_case_values"][OP_NAME]
        plan = build_plan(sweep)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "targets": len(target_keys(plan))}))
        return
    rows, evidence, manifests = [], [], []
    for attempt in args.attempts:
        manifest, attempt_rows, attempt_evidence = load_attempt(attempt)
        rows += attempt_rows
        evidence += [{**e, "attempt": str(attempt)} for e in attempt_evidence]
        manifests.append({"attempt": str(attempt), "manifest_sha256": manifest["manifest_sha256"]})
    write_parquet(rows, args.output)
    Path(args.evidence).write_text(json.dumps({"attempts": manifests, "rows": evidence}, indent=1) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
