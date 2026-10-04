# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dsv411 contract: the physical identities, grid, plan schema and admission rules shared by
every producer of ``dsv411_module_perf.parquet``.

CPU-only. The structural identity of every row is exported from the production SDK graph
(``DEEPSEEKV411``), never reconstructed from a framework; the producers only confirm that the
loaded native modules carry the same dimensions before timing them.

Consumer: ``crates/core/src/perfmodel/perf_database/dsv411.rs`` (``row_structure`` rebuilds the
operator structure key from the columns written here; ``load`` enforces the regime, seeding and
provenance rules mirrored in :func:`validate_row`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

TABLE = "dsv411_module_perf"
MANIFEST_SCHEMA = "dsv411.manifest.v1"
PLAN_SCHEMA = "dsv411.collection.v1"
RECEIPT_SCHEMA = "dsv411.run.v1"
GRID_PATH = Path(__file__).resolve().parents[1] / "cases" / "base_ops" / "dsv411_module.yaml"

COMPONENTS = {
    "Dsv411AttentionCore": "attention_core",
    "Dsv411Indexer": "indexer",
    "Dsv411Engram": "engram",
    "Dsv411Mhc": "mhc",
    "Dsv411SharedLinear": "shared_linear",
}
ATTENTION_COMPONENTS = ("attention_core", "indexer")
TOKEN_COMPONENTS = ("engram", "mhc", "shared_linear")
REGIME_CONTEXT = "eager_drained"
REGIME_GENERATION = "cuda_graph"
REGIME_EXCEPTION = "eager_exception"
# how the KV behind a cached-prefill / decode row got there (plan input, stored per row):
#   real_kv   - chunked prefills of corpus tokens through the model (serving-faithful content, hours of
#               seeding at 1M kv; with dummy weights the indexer's top-k is pseudo-random either way)
#   random_kv - the serving allocation bookkeeping (slots, windows, pages) without the forwards; the
#               caches hold bounded random values. Same kernels and shapes; top-k gathers over a uniform
#               selection (locality slightly pessimistic) - a top-k delta calibration corrects that if needed.
KV_SEED_REGIMES = ("real_kv", "random_kv")
PURPOSES = ("smoke", "calibration")

# Column order of the structure key per component (must match perf_database/dsv411.rs row_structure).
STRUCTURE_FIELDS = {
    "attention_core": (
        "role",
        "compress_ratio",
        "num_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "window_size",
        "index_topk",
        "quant_mode",
    ),
    "indexer": (
        "compress_ratio",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "is_candidate_source",
        "candidate_limit",
        "q_lora_rank",
        "quant_mode",
    ),
    "engram": ("num_embeddings", "head_dim", "hash_columns", "hc_mult", "sharding", "quant_mode"),
    "mhc": ("hidden_size", "hc_mult", "sinkhorn_iters"),
    "shared_linear": ("n", "k", "quant_mode"),
}
STRUCTURE_COLUMNS = tuple(
    dict.fromkeys(
        field for fields in STRUCTURE_FIELDS.values() for field in fields if field not in ("role", "compress_ratio")
    )
)
KEY_COLUMNS = ("component", "role", "compress_ratio", "phase", "tp_size", "batch_size", "query", "kv_len")
VALUE_COLUMNS = ("latency", "sample_count")
WITNESS_COLUMNS = ("kernel_source", "measurement_scope", "measurement_regime", "kv_seed_regime", "used_cuda_graph")
IDENTITY_COLUMNS = ("source_sha256", "config_sha256", "runtime_digest")
INTEGER_COLUMNS = frozenset(
    {
        "compress_ratio",
        "tp_size",
        "batch_size",
        "query",
        "kv_len",
        "sample_count",
        *(c for c in STRUCTURE_COLUMNS if c not in ("quant_mode", "sharding")),
    }
)
STRING_COLUMNS = frozenset(
    {"component", "role", "phase", "quant_mode", "sharding", *WITNESS_COLUMNS[:4], *IDENTITY_COLUMNS}
)
PHYSICAL_KEY = (*KEY_COLUMNS, *STRUCTURE_COLUMNS)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# structural identity (SDK graph -> row columns)
# --------------------------------------------------------------------------------------------
def structure_of(kind: str, body: dict) -> dict:
    """The structural columns of one native operator body, exactly the Rust ``row_structure`` fields."""
    component = COMPONENTS[kind]
    if component == "attention_core":
        values = {
            f: body[f]
            for f in (
                "role",
                "compress_ratio",
                "num_heads",
                "head_dim",
                "q_lora_rank",
                "o_lora_rank",
                "o_groups",
                "window_size",
                "index_topk",
            )
        }
        values["quant_mode"] = body["gemm_quant_mode"]
    elif component == "indexer":
        values = {
            f: body[f]
            for f in (
                "compress_ratio",
                "index_n_heads",
                "index_head_dim",
                "index_topk",
                "candidate_limit",
                "q_lora_rank",
            )
        }
        values["is_candidate_source"] = int(bool(body["is_candidate_source"]))
        values["quant_mode"] = body["gemm_quant_mode"]
    elif component == "engram":
        values = {f: body[f] for f in ("num_embeddings", "head_dim", "hash_columns", "hc_mult", "sharding")}
        values["quant_mode"] = body["gemm_quant_mode"]
    elif component == "mhc":
        values = {f: body[f] for f in ("hidden_size", "hc_mult", "sinkhorn_iters")}
    else:
        values = {f: body[f] for f in ("n", "k", "quant_mode")}
    return {field: values[field] for field in STRUCTURE_FIELDS[component]}


def structure_key(component: str, structure: dict) -> str:
    """``k=v|k=v`` in the Rust field order (``operators/dsv411.rs`` ``structure()``)."""
    parts = []
    for field in STRUCTURE_FIELDS[component]:
        value = structure[field]
        if field == "is_candidate_source":
            value = "true" if value else "false"
        parts.append(f"{field}={value}")
    return "|".join(parts)


def row_structure_columns(component: str, structure: dict) -> dict:
    """Row columns carrying the structure: ``role``/``compress_ratio`` always present."""
    columns = {"role": "", "compress_ratio": 0}
    columns.update(dict.fromkeys(STRUCTURE_COLUMNS))
    columns.update(structure)
    return columns


def build_manifest(tp_size: int, backend: str) -> dict:
    """Export the measured identities from the SDK ``DEEPSEEKV411`` graph (CPU, no perf data)."""
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH
    from aisimulate_core.sdk.models import get_model
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

    config = ModelConfig(tp_size=tp_size, pp_size=1, attention_dp_size=1, moe_tp_size=tp_size, moe_ep_size=1)
    config.dsv41_family = "dsv411"
    model = get_model(MODEL_PATH, config, backend)
    if model.model_family != "DEEPSEEKV411":
        raise RuntimeError("the dsv411 manifest requires the DEEPSEEKV411 graph")
    entries = []
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        for op in ops:
            spec = json.loads(op._spec_json())
            if "Dsv411Stage" not in spec:
                continue
            stage = spec["Dsv411Stage"]
            layer = int(stage["name"].rsplit("_", 1)[1])
            for child in stage["children"]:
                kind, body = next(iter(child.items()))
                if kind not in COMPONENTS:
                    continue
                component = COMPONENTS[kind]
                structure = structure_of(kind, body)
                entries.append(
                    dict(
                        phase=phase,
                        layer=layer,
                        component=component,
                        name=body["name"],
                        structure=structure,
                        structure_key=structure_key(component, structure),
                    )
                )
    representatives: dict = {}
    for entry in entries:
        representatives.setdefault(entry["phase"], {}).setdefault(entry["component"], {}).setdefault(
            entry["structure_key"], entry["layer"]
        )
    facts = model.runtime_facts
    return dict(
        schema=MANIFEST_SCHEMA,
        config_sha256=sha256_json(_load_pre_downloaded_hf_config(MODEL_PATH)),
        tp_size=tp_size,
        backend=backend,
        model_family=model.model_family,
        runtime_facts=dict(
            window_entry_bytes=facts.window_entry_bytes,
            main_entry_bytes=facts.main_entry_bytes,
            index_entry_bytes=facts.index_entry_bytes,
            fmha_quant_mode=facts.fmha_quant_mode,
            index_scoring_quant_mode=facts.index_scoring_quant_mode,
            index_skip_within_topk=facts.index_skip_within_topk,
            engram_sharding=facts.engram_sharding,
        ),
        layer_roles=[model.extra_params.layer_role(i) for i in range(model.extra_params.num_hidden_layers)],
        entries=entries,
        representatives=representatives,
    )


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("model_family") != "DEEPSEEKV411":
        raise ValueError("dsv411 manifest schema differs")
    if type(manifest["tp_size"]) is not int or manifest["tp_size"] < 1:
        raise ValueError("manifest tp_size must be a positive integer")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["config_sha256"]) is None:
        raise ValueError("manifest config_sha256 must be a SHA-256 hex digest")
    for entry in manifest["entries"]:
        if entry["component"] not in STRUCTURE_FIELDS or entry["phase"] not in ("context", "generation"):
            raise ValueError("manifest entry has an unknown component or phase")
        if structure_key(entry["component"], entry["structure"]) != entry["structure_key"]:
            raise ValueError("manifest structure key differs from its structure")


def representative_entries(manifest: dict, phase: str, components) -> list[dict]:
    """One entry per (component, structure) measured in ``phase``: the first layer that carries it."""
    selected = []
    for component, structures in manifest["representatives"][phase].items():
        if component not in components:
            continue
        for key, layer in structures.items():
            entry = next(
                e
                for e in manifest["entries"]
                if e["phase"] == phase
                and e["layer"] == layer
                and e["component"] == component
                and e["structure_key"] == key
            )
            selected.append(entry)
    return selected


# --------------------------------------------------------------------------------------------
# grid / cases
# --------------------------------------------------------------------------------------------
def load_grid(path=GRID_PATH) -> dict:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if data.get("op") != "dsv411_module":
        raise ValueError("dsv411 grid file differs")
    return data["common_case_values"]["dsv411_module"]


def coordinates(case: dict) -> tuple[int, int, int]:
    """(batch_size, query, kv_len) row coordinates of one case (the engine's query coordinates)."""
    if case["kind"] == "tokens":
        return 1, case["tokens"], 0
    if case["phase"] == "context":
        return case["batch_size"], case["query"], case["past_kv"]
    # decode reads the inclusive length: past kv + the token being generated
    return case["batch_size"], 1, case["past_kv"] + 1


def expand_cases(grid: dict, components, *, overrides: dict | None = None) -> tuple[list[dict], dict]:
    """Expand the declared grid into cases; drops are counted per budget reason, never silent."""
    overrides = overrides or {}
    wants_attention = any(c in ATTENTION_COMPONENTS for c in components)
    wants_tokens = any(c in TOKEN_COMPONENTS for c in components)
    cases, drops = [], defaultdict(int)

    def axis(section: str, name: str):
        return list(overrides.get(section, {}).get(name, grid[section][name]))

    if wants_attention:
        ctx = grid["context"]
        for batch in axis("context", "batch_sizes"):
            for query in axis("context", "query_lengths"):
                for past in axis("context", "past_kv_lengths"):
                    if batch * query > ctx["max_new_tokens"]:
                        drops["context.max_new_tokens"] += 1
                        continue
                    if past + query > ctx["max_sequence_length"]:
                        drops["context.max_sequence_length"] += 1
                        continue
                    if past >= ctx["long_kv_min"] and batch > ctx["long_kv_max_batch"]:
                        drops["context.long_kv_max_batch"] += 1
                        continue
                    cases.append(dict(kind="attention", phase="context", batch_size=batch, query=query, past_kv=past))
        gen = grid["generation"]
        for batch in axis("generation", "batch_sizes"):
            for past in axis("generation", "past_kv_lengths"):
                if batch * (past + 1) > gen["max_tokens"]:
                    drops["generation.max_tokens"] += 1
                    continue
                if past + 1 > gen["max_sequence_length"]:
                    drops["generation.max_sequence_length"] += 1
                    continue
                if any(past >= floor and batch > max_batch for floor, max_batch in gen["decode_batch_ladder"]):
                    drops["generation.decode_batch_ladder"] += 1
                    continue
                cases.append(dict(kind="attention", phase="generation", batch_size=batch, query=1, past_kv=past))
    if wants_tokens:
        for tokens in axis("tokens", "context_tokens"):
            cases.append(dict(kind="tokens", phase="context", tokens=tokens))
        for tokens in axis("tokens", "generation_tokens"):
            cases.append(dict(kind="tokens", phase="generation", tokens=tokens))
    for index, case in enumerate(cases):
        if case["kind"] == "attention":
            case["case_id"] = f"{case['phase']}-b{case['batch_size']}-q{case['query']}-kv{case['past_kv']}"
        else:
            case["case_id"] = f"tokens-{case['phase']}-t{case['tokens']}"
        case["index"] = index
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate dsv411 cases")
    return cases, dict(drops)


def seed_group(case: dict) -> tuple:
    """Cases that share one seeded KV prefix: the sglang producer seeds (phase, batch, past kv) once and
    measures every query length on it, so a shard must own whole groups."""
    if case["kind"] != "attention":
        return ("tokens", case["case_id"])
    return ("attention", case["phase"], case["batch_size"], case["past_kv"])


def shard_cases(cases: list[dict], shard: tuple[int, int] | None) -> list[dict]:
    """Round-robin over seed groups (in first-appearance order), never splitting a group."""
    if shard is None:
        return cases
    index, count = shard
    groups: dict[tuple, int] = {}
    for case in cases:
        groups.setdefault(seed_group(case), len(groups))
    return [case for case in cases if groups[seed_group(case)] % count == index]


# --------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------
def validate_plan(
    plan: dict, manifest: dict, *, framework_commit: str, framework_version: str, expected_sm: dict, required_sources
) -> None:
    validate_manifest(manifest)
    if plan.get("schema") != PLAN_SCHEMA or plan["purpose"] not in PURPOSES:
        raise ValueError("dsv411 plan schema/purpose differs")
    if plan["backend"] != manifest["backend"] or plan["tp_size"] != manifest["tp_size"]:
        raise ValueError("plan and manifest disagree on backend/tp")
    if plan["framework_commit"] != framework_commit or plan["framework_version"] != framework_version:
        raise ValueError("plan pins a different framework")
    components = plan["components"]
    if not components or len(set(components)) != len(components) or not set(components) <= set(STRUCTURE_FIELDS):
        raise ValueError("plan components unknown or duplicated")
    if ("attention_core" in components) != ("indexer" in components):
        raise ValueError("attention_core and indexer are measured in the same forward; declare both")
    for key, minimum in (("warmup", 2), ("iterations", 5), ("seed", 0)):
        if type(plan[key]) is not int or plan[key] < minimum:
            raise ValueError(f"invalid {key}")
    if plan["regimes"] != {"context": REGIME_CONTEXT, "generation": REGIME_GENERATION}:
        raise ValueError("dsv411 regimes are fixed: context eager_drained, generation cuda_graph")
    if kv_seed_of(plan) not in KV_SEED_REGIMES:
        raise ValueError(f"kv_seed_regime must be one of {KV_SEED_REGIMES}")
    for exception in plan.get("regime_exceptions", []):
        if (
            exception.get("component") not in STRUCTURE_FIELDS
            or exception.get("phase") != "generation"
            or not exception.get("reason")
        ):
            raise ValueError("regime exceptions name a component, the generation phase and a reason")
    if not plan["cases"]:
        raise ValueError("empty plan")
    for case in plan["cases"]:
        if case["kind"] not in ("attention", "tokens") or case["phase"] not in ("context", "generation"):
            raise ValueError("unknown case kind/phase")
        if case["kind"] == "attention" and ("attention_core" not in components):
            raise ValueError("attention cases without attention components")
        if case["kind"] == "tokens" and not set(components) & set(TOKEN_COMPONENTS):
            raise ValueError("token cases without token components")
    cap = (plan.get("pool") or {}).get("max_total_tokens")
    if cap is not None:
        for case in plan["cases"]:
            if case["kind"] != "attention":
                continue
            new_tokens = case["query"] if case["phase"] == "context" else 1
            resident = case["batch_size"] * (case["past_kv"] + new_tokens)
            if resident > cap:
                raise ValueError(
                    f"{case['case_id']}: {resident} resident KV tokens exceed the pool's max_total_tokens {cap}"
                )
    if re.fullmatch(r"[0-9a-f]{64}", plan["grid_sha256"]) is None:
        raise ValueError("plan must pin the grid file")
    if expected_sm.get(plan["expected_gpu"]) != plan["expected_sm"]:
        raise ValueError("GPU/SM identity differs")
    if not plan["source_pins"].keys() >= set(required_sources):
        raise ValueError("missing native source pins")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= plan["metadata_pins"].keys():
        raise ValueError("missing checkpoint/tokenizer pins")
    digests = [plan["image_sha256"], *plan["source_pins"].values(), *plan["metadata_pins"].values()]
    if any(not isinstance(d, str) or re.fullmatch(r"[0-9a-f]{64}", d) is None for d in digests):
        raise ValueError("immutable source/image digests required")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", plan["runtime_digest"]) is None:
        raise ValueError("immutable OCI runtime digest required")
    if re.fullmatch(r"[0-9a-f]{40}", plan["collector_revision"]) is None:
        raise ValueError("immutable collector revision required")
    shard = plan.get("shard")
    if shard is not None and (len(shard) != 2 or not 0 <= shard[0] < shard[1]):
        raise ValueError("shard must be [index, count]")


def regime_for(plan: dict, component: str, phase: str) -> tuple[str, bool]:
    """(measurement_regime, used_cuda_graph) the plan requires for a (component, phase)."""
    if phase == "context":
        return REGIME_CONTEXT, False
    if any(e["component"] == component for e in plan.get("regime_exceptions", [])):
        return REGIME_EXCEPTION, False
    return REGIME_GENERATION, True


# --------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------
def kv_seed_of(plan: dict) -> str:
    return plan.get("kv_seed_regime", "real_kv")


def make_row(
    entry: dict,
    case: dict,
    *,
    tp_size: int,
    latency: float,
    kernel_source: str,
    regime: str,
    used_cuda_graph: bool,
    kv_seed: str = "real_kv",
) -> dict:
    component = entry["component"]
    batch, query, kv_len = coordinates(case)
    attention_like = component in ATTENTION_COMPONENTS
    if not attention_like:
        batch, kv_len = 1, 0
    row = dict(component=component, phase=case["phase"], tp_size=tp_size, batch_size=batch, query=query, kv_len=kv_len)
    row.update(row_structure_columns(component, entry["structure"]))
    row.update(
        latency=latency,
        sample_count=1,
        kernel_source=kernel_source,
        measurement_scope="local_compute",
        measurement_regime=regime,
        kv_seed_regime=kv_seed if attention_like and (case["phase"] == "generation" or kv_len > 0) else "n/a",
        used_cuda_graph=used_cuda_graph,
    )
    return row


def validate_row(row: dict) -> None:
    component = row.get("component")
    if component not in STRUCTURE_FIELDS:
        raise ValueError("unknown dsv411 component")
    if row["phase"] not in ("context", "generation"):
        raise ValueError("unknown phase")
    for column in ("tp_size", "batch_size", "query", "kv_len", "compress_ratio", "sample_count"):
        value = row[column]
        if isinstance(value, bool) or type(value) is not int or not 0 <= value <= 2**32 - 1:
            raise ValueError(f"{column} must be an exact uint32")
    if not row["tp_size"] or not row["batch_size"] or not row["query"] or not row["sample_count"]:
        raise ValueError("empty measurement")
    if not math.isfinite(row["latency"]) or row["latency"] <= 0:
        raise ValueError("latency must be positive finite milliseconds")
    for field in STRUCTURE_FIELDS[component]:
        if row.get(field) is None:
            raise ValueError(f"{component} rows require {field}")
    if row["measurement_scope"] != "local_compute" or not row["kernel_source"]:
        raise ValueError("local compute dispatch witness required")
    if not isinstance(row["used_cuda_graph"], bool):
        raise ValueError("used_cuda_graph must be boolean")
    regime, graph = row["measurement_regime"], row["used_cuda_graph"]
    if row["phase"] == "context":
        if regime != REGIME_CONTEXT or graph:
            raise ValueError("context rows are eager_drained without CUDA graphs")
    elif not ((regime == REGIME_GENERATION and graph) or (regime == REGIME_EXCEPTION and not graph)):
        raise ValueError("generation rows are cuda_graph (or a declared eager_exception)")
    attention_like = component in ATTENTION_COMPONENTS
    if attention_like:
        if (row["phase"] == "generation" or row["kv_len"] > 0) and row["kv_seed_regime"] not in KV_SEED_REGIMES:
            raise ValueError("decode and cached-prefill rows carry a KV seeding regime (real_kv | random_kv)")
        if row["phase"] == "context" and row["kv_len"] == 0 and row["kv_seed_regime"] != "n/a":
            raise ValueError("prefix-free context rows seed no KV")
        if row["phase"] == "generation" and row["query"] != 1:
            raise ValueError("generation rows measure one query token")
    elif (row["batch_size"], row["kv_len"], row["kv_seed_regime"]) != (1, 0, "n/a"):
        raise ValueError("token-only components use batch=1, kv_len=0, kv_seed_regime=n/a")
    for key in ("source_sha256", "config_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
            raise ValueError(f"invalid {key}")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", row["runtime_digest"]) is None:
        raise ValueError("immutable runtime digest required")


def physical_key(row: dict) -> tuple:
    return tuple(row.get(column) for column in PHYSICAL_KEY)


def validate_table(rows: list[dict]) -> None:
    seen = set()
    identities = set()
    for row in rows:
        validate_row(row)
        key = physical_key(row)
        if key in seen:
            raise ValueError(f"duplicate dsv411 physical key: {key}")
        seen.add(key)
        identities.add(tuple(row[c] for c in IDENTITY_COLUMNS))
    if len(identities) != 1:
        raise ValueError("a dsv411 table carries exactly one runtime/config/source identity")


def write_parquet(rows: list[dict], path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_table(rows)
    columns = [*KEY_COLUMNS, *STRUCTURE_COLUMNS, *VALUE_COLUMNS, *WITNESS_COLUMNS, *IDENTITY_COLUMNS]
    fields = []
    for column in columns:
        if column in INTEGER_COLUMNS:
            fields.append((column, pa.int64()))
        elif column == "latency":
            fields.append((column, pa.float64()))
        elif column == "used_cuda_graph":
            fields.append((column, pa.bool_()))
        else:
            fields.append((column, pa.string()))
    table = pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows], schema=pa.schema(fields))
    pq.write_table(table, path)


# --------------------------------------------------------------------------------------------
# admission (raw rank streams -> table rows)
# --------------------------------------------------------------------------------------------
RAW_EXTRA = ("sample", "invocation", "tp_rank", "case_id", "case_plan_sha256", "collection_purpose")


def expected_keys(plan: dict, manifest: dict) -> set[tuple]:
    """Every (invocation, physical key) the plan must produce per sample and rank."""
    keys = set()
    for case in plan["cases"]:
        components = (
            ATTENTION_COMPONENTS
            if case["kind"] == "attention"
            else tuple(c for c in TOKEN_COMPONENTS if c in plan["components"])
        )
        for entry in representative_entries(manifest, case["phase"], components):
            regime, graph = regime_for(plan, entry["component"], case["phase"])
            row = make_row(
                entry,
                case,
                tp_size=plan["tp_size"],
                latency=1.0,
                kernel_source="x",
                regime=regime,
                used_cuda_graph=graph,
            )
            keys.add((case["index"], physical_key(row)))
    return keys


def aggregate_run(
    raw: Path, *, framework_commit: str, framework_version: str, expected_sm: dict, required_sources
) -> tuple[list[dict], dict]:
    """Admit one run directory (plan.json, manifest.json, rank-*.json, rank-*.jsonl).

    Returns the per-key rows (median over samples of the per-sample rank maximum) and the common
    provenance. Fails closed on incomplete coverage, mixed identities or receipts that differ from
    the frozen plan.
    """
    plan = json.loads((raw / "plan.json").read_text())
    manifest = json.loads((raw / "manifest.json").read_text())
    validate_plan(
        plan,
        manifest,
        framework_commit=framework_commit,
        framework_version=framework_version,
        expected_sm=expected_sm,
        required_sources=required_sources,
    )
    tp = plan["tp_size"]
    plan_sha = sha256_file(raw / "plan.json")
    sources = None
    receipts: list[dict] = []
    for rank in range(tp):
        receipt = json.loads((raw / f"rank-{rank}.json").read_text())
        receipts.append(receipt)
        expected = dict(
            schema=RECEIPT_SCHEMA,
            state="complete_pending_admission",
            tp_rank=rank,
            plan_sha256=plan_sha,
            manifest_sha256=sha256_file(raw / "manifest.json"),
            runtime_digest=plan["runtime_digest"],
            image_sha256=plan["image_sha256"],
            purpose=plan["purpose"],
            framework_version=framework_version,
            collector_revision=plan["collector_revision"],
            checkpoint_weights_loaded=False,
        )
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError(
                f"rank {rank} receipt differs from the frozen plan: "
                + canonical_json({k: receipt.get(k) for k in expected})
            )
        witness = receipt["allocated_device_witness"]
        if witness["returncode"] != 0 or witness["sm"] != plan["expected_sm"]:
            raise ValueError(f"rank {rank} ran on a different GPU than planned")
        if any(receipt["source_hashes"].get(p) != d for p, d in plan["source_pins"].items()):
            raise ValueError(f"rank {rank} ran different native sources than pinned")
        if sources is not None and sources != receipt["source_hashes"]:
            raise ValueError("ranks ran different native sources")
        sources = receipt["source_hashes"]
        if receipt.get("regime_violations"):
            raise ValueError(f"rank {rank} recorded measurement-regime violations: {receipt['regime_violations']}")
    provenance = dict(
        source_sha256=sha256_json(sources),
        config_sha256=manifest["config_sha256"],
        runtime_digest=plan["runtime_digest"],
    )
    expected = expected_keys(plan, manifest)
    # cases a preserved attempt failed on (framework-side failures, observed and recorded by every rank):
    # their keys are not expected; the admission reports them so the publisher can record the gap
    failed = receipts[0].get("failed_cases") or {}
    # the SET of failed cases must agree (the ranks skip the same work); the recorded messages may differ
    # (each rank names the marker file it read)
    if any(set(r.get("failed_cases") or {}) != set(failed) for r in receipts):
        raise ValueError("ranks disagree on the failed cases")
    if failed:
        failed_indices = {c["index"] for c in plan["cases"] if c["case_id"] in failed}
        if len(failed_indices) != len(failed):
            raise ValueError(f"failed cases are not all planned: {sorted(failed)}")
        expected = {k for k in expected if k[0] not in failed_indices}
    # component-scoped kernel limits (one component of a token case refused up front): the same
    # (case, component) set on every rank; only those keys leave the expectation
    failed_components = receipts[0].get("failed_components") or {}
    if any(
        {c: set(v) for c, v in (r.get("failed_components") or {}).items()}
        != {c: set(v) for c, v in failed_components.items()}
        for r in receipts
    ):
        raise ValueError("ranks disagree on the failed components")
    if failed_components:
        by_index = {c["index"]: c["case_id"] for c in plan["cases"]}
        if not set(failed_components) <= set(by_index.values()):
            raise ValueError(f"failed components name unplanned cases: {sorted(failed_components)[:5]}")
        component_at = PHYSICAL_KEY.index("component")
        expected = {k for k in expected if k[1][component_at] not in failed_components.get(by_index.get(k[0], ""), {})}
    samples = defaultdict(lambda: defaultdict(dict))  # key -> sample -> rank -> row
    for rank in range(tp):
        for line in (raw / f"rank-{rank}.jsonl").read_text().splitlines():
            row = json.loads(line)
            validate_row(row)
            if (
                any(row.get(k) != v for k, v in provenance.items())
                or row.get("case_plan_sha256") != plan_sha
                or row.get("collection_purpose") != plan["purpose"]
            ):
                raise ValueError("raw row carries a different provenance than the run")
            key = (row["invocation"], physical_key(row))
            if key not in expected or row["tp_rank"] != rank or row["sample_count"] != 1:
                raise ValueError(f"unplanned raw row: {row['case_id']} {row['component']}")
            if rank in samples[key][row["sample"]]:
                raise ValueError("duplicate raw sample")
            samples[key][row["sample"]][rank] = row
    missing = expected - set(samples)
    if missing:
        raise ValueError(f"incomplete coverage: {len(missing)} planned keys missing, e.g. {sorted(missing)[:3]}")
    by_key: dict[tuple, tuple[dict, list[float]]] = {}
    for (invocation, key), per_sample in samples.items():
        if len(per_sample) != plan["iterations"] or any(set(ranks) != set(range(tp)) for ranks in per_sample.values()):
            raise ValueError(f"incomplete samples/ranks for {key}")
        maxima = [max(r["latency"] for r in ranks.values()) for _, ranks in sorted(per_sample.items())]
        template = {k: v for k, v in next(iter(per_sample.values()))[0].items() if k not in RAW_EXTRA}
        if key in by_key:
            # the same physical key from two invocations is a plan bug, not something to average away
            raise ValueError(f"two invocations produced the same physical key {key}")
        by_key[key] = (template, maxima)
    rows = []
    for template, maxima in by_key.values():
        row = dict(template)
        row["latency"] = statistics.median(maxima)
        row["sample_count"] = len(maxima)
        rows.append(row)
    validate_table(rows)
    return rows, dict(
        provenance,
        plan=plan,
        manifest=manifest,
        plan_sha256=plan_sha,
        failed_cases=failed,
        failed_components=failed_components,
    )


def pool_runs(per_run: list[list[dict]]) -> list[dict]:
    """Concatenate admitted runs (shards / TPs); an identical key must carry an identical row."""
    merged: dict[tuple, dict] = {}
    for rows in per_run:
        for row in rows:
            key = physical_key(row)
            if key in merged:
                if {k: v for k, v in merged[key].items() if k not in VALUE_COLUMNS} != {
                    k: v for k, v in row.items() if k not in VALUE_COLUMNS
                }:
                    raise ValueError(f"runs disagree on the identity of {key}")
                raise ValueError(f"physical key measured by two runs; split the grid by shard instead: {key}")
            merged[key] = row
    rows = list(merged.values())
    validate_table(rows)
    return rows
