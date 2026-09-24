# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured GLM-5.3-Flash operation contract (CPU-only).

Measurements use the production graph's complete serialized geometry. Checkpoint
format remains a physical dimension even where two checkpoints have BF16 local
projections: their native dispatch equivalence has not been established.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from collector.glm53flash_jsonl import file_sha256, iter_records

PER_OPERATION_TP_MAX = "per_operation_tp_max_v1"
WHOLE_FORWARD_RANK = "whole_forward_slowest_rank_v1"

COMPONENTS = {
    "Glm53Attention": "attention",
    "Glm53Mhc": "mhc",
    "Glm53Router": "router",
    "Glm53Ffn": "ffn",
    "Glm53Primitive": "primitive",
}
BACKENDS = {
    "vllm": ("0.30.0", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"),
    "sglang": ("0.5.20", "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"),
}
CHECKPOINTS = {
    "fp8": ("zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"),
    "nvfp4": ("nvidia/GLM-5.3-Flash-NVFP4", "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"),
}
KEY_COLUMNS = ("component", "geometry", "batch_size", "prefix", "x")
INTEGER_COLUMNS = ("batch_size", "prefix", "x", "sample_count")
PROVENANCE_COLUMNS = (
    "backend",
    "backend_version",
    "backend_revision",
    "checkpoint_revision",
    "source_sha256",
    "config_sha256",
    "runtime_digest",
    "used_cuda_graph",
    "kernel_source",
    "state_mode",
)
EVIDENCE_COLUMNS = ("dataset_role", "request_set", "corpus_sha256", "evidence_sha256")
ROW_COLUMNS = (
    *KEY_COLUMNS,
    "latency",
    "sample_count",
    "measurement_scope",
    "kv_seed_regime",
    "dispatch_fingerprint",
    "aggregation_policy",
    "rank_selection_sha256",
    *PROVENANCE_COLUMNS,
    *EVIDENCE_COLUMNS,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def operation_geometry(body: dict) -> str:
    return canonical_json({key: value for key, value in body.items() if key not in ("name", "children")})


def build_model_manifest(backend: str, checkpoint_format: str, tp_size: int) -> dict:
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

    if backend not in BACKENDS or checkpoint_format not in CHECKPOINTS or tp_size not in (1, 2, 4):
        raise ValueError("unqualified GLM collection model configuration")
    if tp_size == 1 and checkpoint_format != "nvfp4":
        raise ValueError("TP1 is optional only for admitted NVFP4")
    model_path, revision = CHECKPOINTS[checkpoint_format]
    model = get_model(model_path, ModelConfig(tp_size=tp_size, moe_tp_size=tp_size, moe_ep_size=1), backend)
    return {
        **build_manifest(model),
        "model_path": model_path,
        "checkpoint_revision": revision,
        "backend": backend,
        "backend_version": BACKENDS[backend][0],
        "backend_revision": BACKENDS[backend][1],
        "config_sha256": sha256_json(_load_pre_downloaded_hf_config(model_path)),
        "tp_size": tp_size,
        "ep_size": 1,
    }


def build_manifest(model) -> dict:
    """Take identities from a configured production model, never reconstruct them."""
    phases = {}
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        entries = []

        def visit(spec):
            kind, body = next(iter(spec.items()))
            if kind in COMPONENTS:
                entries.append(
                    {"component": COMPONENTS[kind], "name": body["name"], "geometry": operation_geometry(body)}
                )
            elif kind == "Overlap":
                for child in (*body["group_a"], *body["group_b"]):
                    visit(child)
            else:
                raise ValueError(f"GLM production graph has an unobserved native boundary: {kind}")

        for op in ops:
            visit(json.loads(op._spec_json()))
        if any(
            sum(entry["component"] == component for entry in entries) != count
            for component, count in (("attention", 45), ("ffn", 45), ("primitive", 94))
        ):
            raise ValueError("GLM manifest must cover all text attention, FFN and primitive boundaries")
        if len({entry["name"] for entry in entries}) != len(entries):
            raise ValueError("native operation display names must uniquely identify graph occurrences")
        phases[phase] = entries
    return {"schema_version": 1, "phases": phases}


def _uint32(value, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not (int(positive) <= value <= 2**32 - 1):
        raise ValueError(f"{label} must be an exact {'positive ' if positive else ''}uint32")


def validate_native_workload(backend: str, phase: str, prefix: int, query: int) -> None:
    """Conservative stock-native admission after source and GB300 cache oracle."""
    if backend == "vllm" and phase in ("context", "prefill") and prefix % 4 and query >= 2:
        raise ValueError(
            "stock vLLM GLM cached prefill with unaligned IndexPool start is unqualified "
            "(prefix % 4 != 0, query >= 2); native cache oracle failed crossing pools"
        )


def validate_row(row: dict) -> None:
    """Reject unverifiable identities before they can enter a measured table."""
    if row.get("component") not in COMPONENTS.values():
        raise ValueError("unknown GLM-5.3-Flash component")
    shape = json.loads(row["geometry"])
    if (
        not isinstance(shape, dict)
        or "name" in shape
        or "children" in shape
        or canonical_json(shape) != row["geometry"]
    ):
        raise ValueError("geometry must be canonical JSON excluding the display name")
    checkpoint_format = shape.get("checkpoint_format")
    if checkpoint_format not in CHECKPOINTS:
        raise ValueError("geometry must preserve the exact checkpoint format")
    if row["checkpoint_revision"] != CHECKPOINTS[checkpoint_format][1]:
        raise ValueError("unqualified checkpoint revision")
    if row["backend"] not in BACKENDS or (row["backend_version"], row["backend_revision"]) != BACKENDS[row["backend"]]:
        raise ValueError("unqualified backend revision")
    if shape.get("backend") != row["backend"]:
        raise ValueError("operation geometry and observed backend disagree")
    for key in INTEGER_COLUMNS:
        _uint32(row[key], key, positive=key != "prefix")
    if isinstance(row["latency"], bool) or not math.isfinite(row["latency"]) or row["latency"] <= 0:
        raise ValueError("latency must be finite positive milliseconds")
    for key in ("source_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"invalid {key}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["runtime_digest"]):
        raise ValueError("immutable platform image digest required")
    scope = (
        (
            "communication"
            if shape.get("role") == "allreduce"
            else "compute_and_communication"
            if shape.get("role") == "logits"
            else "local_compute"
        )
        if row["component"] == "primitive"
        else "local_compute"
    )
    if row["measurement_scope"] != scope or not row["kernel_source"].strip():
        raise ValueError("observed local compute dispatch is required")
    if row.get("dispatch_fingerprint", "") and not re.fullmatch(r"[0-9a-f]{64}", row["dispatch_fingerprint"]):
        raise ValueError("invalid native kernel dispatch fingerprint")
    if not isinstance(row["used_cuda_graph"], bool):
        raise ValueError("used_cuda_graph must be boolean")
    if row["component"] == "attention":
        _uint32(shape.get("tp_size"), "tp_size", positive=True)
        if shape["tp_size"] not in (1, 2, 4) or (shape["tp_size"] == 1 and checkpoint_format != "nvfp4"):
            raise ValueError("unqualified TP/checkpoint combination")
        if not isinstance(shape.get("is_context"), bool):
            raise ValueError("attention is_context must be boolean")
        if shape["is_context"]:
            validate_native_workload(row["backend"], "context", row["prefix"], row["x"])
            expected_modes = ("cached_prefill", "chunked_prefill") if row["prefix"] else ("full_prefill",)
            if row["state_mode"] not in expected_modes:
                raise ValueError("prefill state mode disagrees with its measured prefix")
            if row["prefix"] + row["x"] > 131072:
                raise ValueError("prefill exceeds the qualified 128K context")
        elif row["prefix"] or row["state_mode"] != "decode" or row["x"] + 1 > 131072:
            raise ValueError("decode requires absolute past-KV x, prefix=0, and decode state mode")
        if row["kv_seed_regime"] != ("real_kv" if row["prefix"] or not shape["is_context"] else "empty"):
            raise ValueError("cached prefill/decode requires native real-prefix state")
    elif (row["batch_size"], row["prefix"], row["kv_seed_regime"], row["state_mode"]) != (1, 0, "n/a", "token_only"):
        raise ValueError("token-only components require batch=1, prefix=0 and no state label")


def validate_calibration_row(row: dict) -> None:
    policy = row.get("aggregation_policy", PER_OPERATION_TP_MAX)
    if policy not in (PER_OPERATION_TP_MAX, WHOLE_FORWARD_RANK):
        raise ValueError("unknown native Ops TP aggregation policy")
    selection = row.get("rank_selection_sha256", "")
    if (policy == WHOLE_FORWARD_RANK and not re.fullmatch(r"[0-9a-f]{64}", selection)) or (
        policy == PER_OPERATION_TP_MAX and selection
    ):
        raise ValueError("native Ops TP aggregation lacks its rank-selection evidence")
    if row.get("dataset_role") != "calibration" or not row.get("request_set"):
        raise ValueError("measured rows require a frozen calibration request set")
    for key in ("corpus_sha256", "evidence_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")):
            raise ValueError(f"invalid calibration {key}")
    if row["sample_count"] < 10:
        raise ValueError("formal calibration rows require ten measured repetitions")


def canonical_native_source(row: dict) -> str:
    """Normalize source ownership for the explicitly pinned mHC forwarding calls.

    glm5_next.py@94602c9:727-746 contains two unconditional calls to the same
    _hc_pre method, which calls the imported hc_pre implementation at 710-725.
    Constructors at 664-679 give both parameter sets identical FP32 shapes;
    the fused output RMSNorm shape/epsilon is identical at 657-660. Raw entry
    labels and measured windows stay intact. No kernel fingerprint is invented.
    """
    source = row["kernel_source"]
    vllm_module = "vllm.models.glm5next.nvidia.model.Glm5NextDecoderLayer"
    entrypoint = source.split("/", 1)[0]
    vllm_roles = {
        "hc_pre": ("pre", "MHCPreOp"),
        "hc_post": ("post", "MHCPostOp"),
        "hc_fused_post_pre": ("fused_post_pre", "MHCFusedPostPreOp"),
    }
    for method, (role, native_class) in vllm_roles.items():
        if entrypoint != f"{vllm_module}.{method}":
            continue
        pins = json.loads(
            (Path(__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json").read_bytes()
        )
        geometry = json.loads(row["geometry"])
        if (
            row["backend"] != "vllm"
            or (row["backend_version"], row["backend_revision"]) != BACKENDS["vllm"]
            or row["component"] != "mhc"
            or geometry.get("backend") != "vllm"
            or geometry.get("role") != role
            or pins.get("vllm/models/glm5next/nvidia/model.py")
            != "d7353ea0c5708e40d65364b6a32ab63372aafbbebbf954722d635a125252a813"
            or row["source_sha256"] != sha256_json(pins)
        ):
            raise ValueError("vLLM mHC source ownership lacks its exact native source/geometry identity")
        # Pinned model.py:557-619 calls only the corresponding mHC CustomOp.
        # Historical observers enumerated the entire owning DecoderLayer,
        # including attention/FFN modules this method never calls. Preserve
        # those raw labels, but require the actual selected mHC dispatch.
        selected = f"m{method}_op:forward=vllm.model_executor.layers.mhc.{native_class}.forward_cuda"
        parts = source.split("/", 1)[1].split(";") if "/" in source else []
        allowed = (
            "input_layernorm:",
            "post_attention_layernorm:",
            "mhc_pre_op:",
            "mhc_post_op:",
            "mhc_fused_post_pre_op:",
            "mlp.",
            "self_attn.",
        )
        if parts.count(selected) != 1 or any(not part.startswith(allowed) for part in parts):
            raise ValueError("vLLM mHC selected native callee is missing, changed or ambiguous")
        return f"{entrypoint}/{selected}"
    module = "sglang.srt.models.glm5_next.Glm5NextDecoderLayer"
    if source not in (f"{module}.hc_attn_pre", f"{module}.hc_ffn_pre"):
        return source
    pins_path = Path(__file__).parent / "fpm_forward/runtime/glm53flash_sglang/runtime-source-sha256.json"
    pins = json.loads(pins_path.read_bytes())
    geometry = json.loads(row["geometry"])
    if (
        row["backend"] != "sglang"
        or (row["backend_version"], row["backend_revision"]) != BACKENDS["sglang"]
        or row["component"] != "mhc"
        or geometry.get("backend") != "sglang"
        or geometry.get("role") != "pre"
        or pins.get("srt/models/glm5_next.py") != "12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2"
        or row["source_sha256"] != sha256_json(pins)
    ):
        raise ValueError("SGLang mHC forwarding-source equivalence lacks its exact native source/geometry identity")
    return f"{module}._hc_pre/sglang.kernels.ops.layernorm.mhc.hc_pre"


def select_forward_ranks(paths: list[Path], tp_size: int) -> dict:
    """Choose one complete native rank timeline, never an op sum or fitted target."""
    records = {}
    inputs = []
    joins = ("request_set", "phase", "benchmark_id", "repetition", "invocation", "sampling_role", "request_ids")
    for rank in range(tp_size):
        path = paths[rank].parent / f"forward-rank-{rank}.jsonl"
        inputs.append({"path": path.name, "sha256": file_sha256(path)})
        for row in iter_records(path):
            if row.get("stage") != "measure":
                continue
            if (
                type(row.get("tp_rank")) is not int
                or row.get("tp_rank") != rank
                or row.get("gpu_completed") is not True
                or row.get("state_layout_admitted") is not True
            ):
                raise ValueError("rank selection requires actual completed native forward evidence")
            key = tuple(row[k] for k in ("phase", "benchmark_id", "repetition", "invocation"))
            join = {k: row[k] for k in joins}
            native_identity = {
                k: row.get(k)
                for k in (
                    "backend",
                    "backend_version",
                    "backend_revision",
                    "checkpoint_revision",
                    "source_sha256",
                    "config_sha256",
                    "runtime_digest",
                    "used_cuda_graph",
                    "batch_size",
                    "query_lengths",
                    "prefix_lengths",
                    "num_padded_tokens",
                )
            }
            if (
                row.get("whole_forward_boundary") == "embedding_to_logits_gpu_v1"
                or row.get("whole_forward_boundary") == "native_full_graph_metadata_to_logits_gpu_v1"
            ):
                value, boundary = row.get("whole_forward_gpu_ms"), row["whole_forward_boundary"]
            elif (
                row.get("backend") == "sglang"
                and row.get("used_cuda_graph") is False
                and row.get("timing_boundary") == "sglang_native_forward_device_timer"
            ):
                value, boundary = row.get("native_forward_ms"), row["timing_boundary"]
            else:
                raise ValueError("rank selection requires a recorded whole-forward interval; op sums cannot substitute")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("rank selection whole-forward interval must be finite positive milliseconds")
            group = records.setdefault(
                key,
                {"join": join, "native_identity": native_identity, "timing_boundary": boundary, "whole_forward_ms": {}},
            )
            if (
                group["join"] != join
                or group["native_identity"] != native_identity
                or group["timing_boundary"] != boundary
                or rank in group["whole_forward_ms"]
            ):
                raise ValueError("rank selection cannot join different or repeated native forwards")
            group["whole_forward_ms"][rank] = value
    selected = []
    for key, group in sorted(records.items()):
        if set(group["whole_forward_ms"]) != set(range(tp_size)):
            raise ValueError("rank selection requires every TP rank of the same native forward")
        group["selected_rank"] = max(range(tp_size), key=lambda rank: (group["whole_forward_ms"][rank], -rank))
        selected.append(group)
    if not selected:
        raise ValueError("rank selection has no native target forwards")
    return {
        "schema": "glm53flash_rank_selection_v1",
        "aggregation_policy": WHOLE_FORWARD_RANK,
        "tie_policy": "lowest_tp_rank",
        "raw_forward_files": inputs,
        "raw_module_files": [{"path": path.name, "sha256": file_sha256(path)} for path in sorted(paths)],
        "forwards": selected,
    }


def aggregate_rank_records(
    paths: list[Path],
    tp_size: int,
    manifest: dict,
    *,
    evidence_sha256: str,
    point_ids: dict[int, int] | None = None,
    aggregation_policy: str = PER_OPERATION_TP_MAX,
) -> list[dict]:
    """Reduce native intervals after complete graph/rank coverage.

    The historical default takes a conservative per-operation TP maximum.
    New collection explicitly selects all operations from the rank with the
    largest recorded whole-forward interval. Ties choose the lowest TP rank.
    Neither reduction reconstructs an exact cross-rank critical path.
    The raw records retain layer occurrence, workload and sample identities.
    Identical shapes in different layers may reduce together only after every
    occurrence in each observed phase has been observed. Failed/incomplete attempts
    must remain on disk and cannot be repaired by merging attempts.
    Shared physical keys use the lowest frozen original point ID; all points
    still require complete evidence and compatible dispatch. No latency decides
    ownership. Shards supply their immutable native-to-original point mapping.
    """
    if {path.name for path in paths} != {f"rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError("missing or unexpected TP rank files")
    if aggregation_policy not in (PER_OPERATION_TP_MAX, WHOLE_FORWARD_RANK):
        raise ValueError("unknown native Ops TP aggregation policy")
    selection = select_forward_ranks(sorted(paths), tp_size) if aggregation_policy == WHOLE_FORWARD_RANK else None
    selected = (
        {}
        if selection is None
        else {
            tuple(item["join"][k] for k in ("phase", "benchmark_id", "repetition", "invocation")): item
            for item in selection["forwards"]
        }
    )
    selection_sha = sha256_json(selection) if selection is not None else ""
    if point_ids is not None:
        for native, original in point_ids.items():
            _uint32(native, "native benchmark ID", positive=True)
            _uint32(original, "original point ID", positive=True)
        if len(set(point_ids.values())) != len(point_ids):
            raise ValueError("shard point mapping aliases original benchmark IDs")
    groups = {}
    coverage = defaultdict(int)
    expected = {
        phase: {(entry["name"], entry["component"], entry["geometry"]): index for index, entry in enumerate(entries)}
        for phase, entries in manifest["phases"].items()
    }
    for path in paths:
        expected_rank = int(path.stem.split("-")[1])
        for row in iter_records(path):
            validate_row(row)
            # Work on the parsed record only; original raw entrypoint evidence
            # remains unchanged on disk and bound by calibration-evidence.json.
            row["kernel_source"] = canonical_native_source(row)
            if row.get("dataset_role") != "calibration" or not row.get("request_set"):
                raise ValueError("raw native observations must identify their frozen calibration corpus")
            if not re.fullmatch(r"[0-9a-f]{64}", row.get("corpus_sha256", "")):
                raise ValueError("raw native observations must identify their frozen calibration corpus")
            row["evidence_sha256"] = evidence_sha256
            row["aggregation_policy"] = aggregation_policy
            row["rank_selection_sha256"] = selection_sha
            if row.get("stage") != "measure" or row.get("sampling_role") not in ("warmup", "measurement"):
                raise ValueError("formal native measurements require frozen target and sampling roles")
            for label in ("benchmark_id", "repetition"):
                _uint32(row[label], label)
            if point_ids is not None and row["benchmark_id"] not in point_ids:
                raise ValueError("observed benchmark is absent from frozen shard point mapping")
            if row["sample"] != row["repetition"]:
                raise ValueError("native sample and frozen repetition disagree")
            if row["tp_rank"] != expected_rank or isinstance(row["tp_rank"], bool):
                raise ValueError("rank record does not belong to its evidence file")
            for key in ("sample", "invocation"):
                _uint32(row[key], key)
            phase = row["phase"]
            identity = (row["name"], row["component"], row["geometry"])
            if identity not in expected.get(phase, {}):
                raise ValueError("observed native operation is absent from the production graph")
            occurrence = expected[phase][identity]
            bit = 1 << occurrence
            invocation_key = (expected_rank, phase, row["sample"], row["invocation"])
            if coverage[invocation_key] & bit:
                raise ValueError("native graph occurrence was observed more than once")
            coverage[invocation_key] |= bit
            physical = tuple(row[key] for key in KEY_COLUMNS)
            if physical not in groups:
                groups[physical] = {
                    "row": {key: row.get(key, "") for key in ROW_COLUMNS},
                    "provenance": tuple(row[key] for key in (*PROVENANCE_COLUMNS, *EVIDENCE_COLUMNS)),
                    "samples": {},
                    "repetitions": defaultdict(lambda: defaultdict(set)),
                    "dispatches": defaultdict(set),
                }
            group = groups[physical]
            if tuple(row[key] for key in (*PROVENANCE_COLUMNS, *EVIDENCE_COLUMNS)) != group["provenance"]:
                raise ValueError("incompatible native invocations collide on one physical key")
            group["repetitions"][row["benchmark_id"]][row["sampling_role"]].add(row["repetition"])
            if row["sampling_role"] == "measurement":
                group["dispatches"][row["benchmark_id"]].add(row.get("dispatch_fingerprint", ""))
            sample_key = (
                phase,
                row["sample"],
                row["invocation"],
                occurrence,
                row["sampling_role"],
                row["benchmark_id"],
            )
            sample = group["samples"].setdefault(sample_key, [0, 0.0])
            rank_bit = 1 << expected_rank
            if sample[0] & rank_bit:
                raise ValueError("duplicate rank within one native invocation")
            sample[0] |= rank_bit
            if selection is None:
                sample[1] = max(sample[1], row["latency"])
            else:
                chosen = selected.get((phase, row["benchmark_id"], row["repetition"], row["invocation"]))
                if chosen is None or any(row.get(k) != value for k, value in chosen["join"].items()):
                    raise ValueError("operation cannot borrow rank selection from a different native forward")
                if expected_rank == chosen["selected_rank"]:
                    sample[1] = row["latency"]
    if not groups:
        raise ValueError("no native measurements")
    for (_, phase, _, _), observed in coverage.items():
        if observed != (1 << len(expected[phase])) - 1:
            raise ValueError(f"incomplete native {phase} graph coverage")
    output = []
    for group in groups.values():
        samples, repetitions, dispatches = group["samples"], group["repetitions"], group["dispatches"]
        if any(sample[0] != (1 << tp_size) - 1 for sample in samples.values()):
            raise ValueError("incomplete TP rank set within a native invocation")
        if any(len(roles["warmup"]) < 5 or len(roles["measurement"]) < 10 for roles in repetitions.values()):
            raise ValueError("each frozen native point requires at least five warmups and ten measured repetitions")
        if len({tuple(sorted(signatures)) for signatures in dispatches.values()}) != 1:
            raise ValueError("shared physical key has incompatible measured dispatch across benchmark points")
        owner = min(repetitions, key=lambda native: native if point_ids is None else point_ids[native])
        measured = [sample for key, sample in samples.items() if key[-2:] == ("measurement", owner)]
        result = group["row"]
        result["owner_benchmark_id"] = owner
        result["original_point_id"] = owner if point_ids is None else point_ids[owner]
        signatures = dispatches[owner]
        result["dispatch_fingerprint"] = sha256_json(sorted(signatures)) if signatures and "" not in signatures else ""
        result["latency"] = statistics.median(sample[1] for sample in measured)
        result["sample_count"] = len(measured)
        validate_calibration_row(result)
        output.append(result)
    return output


def write_parquet(rows: list[dict], destination: Path) -> None:
    """Write one complete table; never silently replace duplicate physical keys."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("cannot publish an empty measured table")
    keys = set()
    identities = {}
    policies = {row.get("aggregation_policy", PER_OPERATION_TP_MAX) for row in rows}
    if len(policies) != 1:
        raise ValueError("table mixes native Ops TP aggregation policies")
    rows = [{"aggregation_policy": PER_OPERATION_TP_MAX, "rank_selection_sha256": "", **row} for row in rows]
    for row in rows:
        validate_row(row)
        validate_calibration_row(row)
        key = tuple(row[column] for column in KEY_COLUMNS)
        if key in keys:
            raise ValueError("duplicate GLM-5.3-Flash physical key")
        keys.add(key)
        checkpoint_format = json.loads(row["geometry"])["checkpoint_format"]
        identity = tuple(
            row[key] for key in PROVENANCE_COLUMNS if key not in ("used_cuda_graph", "kernel_source", "state_mode")
        )
        if identities.setdefault(checkpoint_format, identity) != identity:
            raise ValueError("table mixes runtime/config/source identities within a checkpoint format")
    schema = pa.schema(
        [
            (
                column,
                pa.int64()
                if column in INTEGER_COLUMNS
                else pa.float64()
                if column == "latency"
                else pa.bool_()
                if column == "used_cuda_graph"
                else pa.string(),
            )
            for column in ROW_COLUMNS
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
