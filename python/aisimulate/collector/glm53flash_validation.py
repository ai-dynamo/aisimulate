# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind measured Ops tables to retained native calibration and independent truth.

Content hashes identify retained evidence; they do not by themselves prove a
measurement. Admission also checks native request continuity, actual coordinates,
complete TP/graph coverage and reaggregates original observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from collector.glm53flash_contract import (
    BACKENDS,
    CHECKPOINTS,
    KEY_COLUMNS,
    PER_OPERATION_TP_MAX,
    ROW_COLUMNS,
    WHOLE_FORWARD_RANK,
    aggregate_rank_records,
    build_model_manifest,
    canonical_json,
    select_forward_ranks,
    sha256_json,
    validate_calibration_row,
    validate_native_workload,
)
from collector.glm53flash_jsonl import file_sha256, iter_records

BOUNDARY = "embedding_to_logits_gpu_v1"


def file_sha(path: Path) -> str:
    return file_sha256(path)


def _required_files(tp_size: int, backend: str) -> set[str]:
    files = {
        "requests.json",
        "manifest.json",
        "points.json",
        "provenance.json",
        "command.json",
        "runtime-preflight.json",
    }
    files.update(
        f"{stem}-{rank}.{suffix}"
        for rank in range(tp_size)
        for stem, suffix in (
            ("rank", "jsonl"),
            ("forward-rank", "jsonl"),
            ("state-layout-rank", "json"),
        )
    )
    if backend == "sglang":
        files.update({"sglang-provenance.json", "sglang-declared-config.json", "sglang-resolved-config.json"})
        files.update(f"retained-rank-{rank}.jsonl" for rank in range(tp_size))
    return files


def _sglang_execution(root: Path, fmt: str, tp: int, points: list[dict], *, graph: bool = False) -> dict:
    """Reuse the shared runtime receipt contract without the FPM timing reader."""
    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
    from aisimulate_core.sdk.utils import get_model_config_from_model_path
    from collector.fpm_forward.sglang_artifact import _validate_runtime_receipts, file_receipt

    provenance = json.loads((root / "sglang-provenance.json").read_bytes())
    raw_config = get_model_config_from_model_path(CHECKPOINTS[fmt][0])["raw_config"]
    cell = SimpleNamespace(
        execution_identity=execution_identity(raw_config, backend="sglang", input_modality="text"),
        topology=SimpleNamespace(tp=tp),
    )
    payload = {
        **provenance,
        "input_provenance": {"tokenizer_revision": CHECKPOINTS[fmt][1]},
        "producer": {"telemetry_policy": provenance.get("telemetry_policy")},
        "results": [{"point": point} for point in points],
    }
    receipts = {
        key: file_receipt(root / name)
        for key, name in (
            ("runtime_preflight", "runtime-preflight.json"),
            ("declared_config", "sglang-declared-config.json"),
            ("resolved_config", "sglang-resolved-config.json"),
        )
    }
    identity = _validate_runtime_receipts(cell, payload, root, receipts)
    resolved = json.loads((root / "sglang-resolved-config.json").read_bytes())
    expected = {"prefill": "disabled", "decode": "full" if graph else "disabled"}
    if any(resolved["cuda_graph_config"][phase]["backend"] != mode for phase, mode in expected.items()):
        raise ValueError(
            "Ops runtime receipts require disabled prefill and FULL decode"
            if graph
            else "Ops runtime receipts require both native SGLang graph phases disabled"
        )
    if set(identity["execution_identity"]) != set(EXECUTION_COLUMNS):
        raise ValueError("Ops native execution identity is incomplete")
    return identity


def expected_runtime_version(run: dict) -> str:
    """Frozen plan wins; standalone runs must explicitly opt into a repair."""
    from collector.glm53flash_runtime_identity import validate_backend_version

    backend = run["key"][0]
    planned = run.get("plan", {}).get("capability", {}).get("aic_database_version")
    explicit = run["spec"].get("backend_version")
    if planned is not None and explicit is not None and planned != explicit:
        raise ValueError("Ops explicit runtime differs from frozen plan capability")
    return validate_backend_version(backend, planned or explicit or BACKENDS[backend][0])


def _runtime_audit(root: Path, backend: str, expected_version: str | None = None) -> dict:
    audit = json.loads((root / "runtime-preflight.json").read_bytes())
    runtime = "glm53flash" if backend == "vllm" else "glm53flash_sglang"
    from collector.glm53flash_runtime_identity import (
        validate_backend_version,
        validate_vllm_runtime_closure,
        vllm_source_pins,
    )

    version = validate_backend_version(backend, audit.get("backend_version"))
    if expected_version is not None and version != expected_version:
        raise ValueError("Ops actual native runtime differs from frozen expected backend version")
    manifest = Path(__file__).parent / "fpm_forward/runtime" / runtime / "runtime-source-sha256.json"
    pins = vllm_source_pins(version, manifest) if backend == "vllm" else json.loads(manifest.read_bytes())
    if (audit.get("status"), audit.get("backend"), audit.get("backend_version"), audit.get("sources")) != (
        "passed",
        backend,
        version,
        pins,
    ):
        raise ValueError("Ops actual native source audit differs from pinned runtime")
    if backend == "vllm":
        validate_vllm_runtime_closure(version, manifest, audit.get("runtime_closure"))
    return pins


def _state_layout(layout: dict, backend: str) -> None:
    from collector.glm53flash_protocol import validate_gb300_identity

    hardware = layout.get("hardware")
    validate_gb300_identity(hardware)
    groups = layout.get("groups", {})
    expected_device = f"cuda:{hardware['cuda_device_index']}"
    if any(tensor.get("device") != expected_device for tensors in groups.values() for tensor in tensors):
        raise ValueError("Ops allocated state tensors differ from the observed worker GPU")
    expected = {"kda_conv": "torch.bfloat16", "kda_temporal": "torch.float32", "pooled_index_packed": "torch.uint8"}
    expected.update(
        {"index_tail": "torch.bfloat16"}
        if backend == "vllm"
        else {"index_tail_key": "torch.bfloat16", "index_tail_score": "torch.bfloat16"}
    )
    if layout.get("admitted") is not True or not groups.get("mla_latent"):
        raise ValueError("native hybrid state allocation was not admitted")
    if backend == "sglang" and layout.get("logical_kv_dtype") != "torch.float8_e4m3fn":
        raise ValueError("Ops actual MLA logical cache is not FP8")
    if backend == "vllm" and layout.get("native_cache_dtype") not in ("fp8", "fp8_e4m3"):
        raise ValueError("Ops actual MLA logical cache is not FP8")
    for name, dtype in expected.items():
        tensors = groups.get(name, [])
        if not tensors or any(t.get("dtype") != dtype for t in tensors):
            raise ValueError(f"Ops actual {name} dtype differs from pinned state")
    if backend == "vllm" and any(
        len(groups[name]) != count
        for name, count in {
            "kda_conv": 34,
            "kda_temporal": 34,
            "mla_latent": 11,
            "pooled_index_packed": 11,
            "index_tail": 11,
        }.items()
    ):
        raise ValueError("Ops allocated state does not cover all native layers")


def freeze_evidence(
    output: Path, tp_size: int, manifest: dict, *, aggregation_policy: str = PER_OPERATION_TP_MAX
) -> str:
    """Freeze only after native execution and completeness checks have succeeded."""
    requests = json.loads((output / "requests.json").read_text())
    if requests["dataset_role"] != "calibration":
        raise ValueError("only calibration observations may produce measured rows")
    required = _required_files(tp_size, manifest["backend"])
    if aggregation_policy == WHOLE_FORWARD_RANK:
        selection = select_forward_ranks([output / f"rank-{rank}.jsonl" for rank in range(tp_size)], tp_size)
        (output / "rank-selection.json").write_text(canonical_json(selection) + "\n")
        required.add("rank-selection.json")
    elif aggregation_policy != PER_OPERATION_TP_MAX:
        raise ValueError("unknown native Ops TP aggregation policy")
    if not all((output / name).is_file() for name in required):
        raise FileNotFoundError("native calibration evidence is incomplete")
    audit = json.loads((output / "runtime-preflight.json").read_text())
    if audit.get("status") != "passed":
        raise ValueError("native calibration source preflight did not pass")
    receipt = {
        "schema": "glm53flash_ops_calibration_evidence_v1",
        "aggregation_policy": aggregation_policy,
        "dataset_role": "calibration",
        "request_set": requests["request_set"],
        "corpus_sha256": requests["corpus_sha256"],
        "tp_size": tp_size,
        "backend": manifest["backend"],
        "checkpoint_revision": manifest["checkpoint_revision"],
        "files": [{"path": name, "sha256": file_sha(output / name)} for name in sorted(required)],
    }
    path = output / "calibration-evidence.json"
    path.write_text(canonical_json(receipt) + "\n")
    return file_sha(path)


def _read_evidence(root: Path) -> dict:
    path = root / "calibration-evidence.json"
    receipt = json.loads(path.read_bytes())
    if (
        receipt.get("schema") != "glm53flash_ops_calibration_evidence_v1"
        or receipt.get("dataset_role") != "calibration"
    ):
        raise ValueError("unknown native Ops calibration evidence schema/role")
    names = set()
    for item in receipt["files"]:
        source = root / item["path"]
        if source.name != item["path"] or source.resolve().parent != root.resolve() or item["path"] in names:
            raise ValueError("Ops evidence paths must be unique local files")
        names.add(item["path"])
        if file_sha(source) != item["sha256"]:
            raise ValueError("retained native Ops evidence changed after calibration")
    if not _required_files(receipt["tp_size"], receipt["backend"]) <= names:
        raise ValueError("Ops calibration evidence omits required native files")
    policy = receipt.get("aggregation_policy", PER_OPERATION_TP_MAX)
    if policy == WHOLE_FORWARD_RANK:
        if "rank-selection.json" not in names:
            raise ValueError("coherent native Ops calibration omits rank-selection proof")
        actual = select_forward_ranks(
            [root / f"rank-{rank}.jsonl" for rank in range(receipt["tp_size"])], receipt["tp_size"]
        )
        if json.loads((root / "rank-selection.json").read_bytes()) != json.loads(canonical_json(actual)):
            raise ValueError("native Ops rank-selection sidecar differs from actual whole-forward evidence")
    elif policy != PER_OPERATION_TP_MAX:
        raise ValueError("unknown native Ops TP aggregation policy")
    _runtime_audit(root, receipt["backend"])
    return receipt


def _validate_vllm_context(row: dict) -> None:
    from collector.glm53flash_protocol import MAX_MEASURED_CONTEXT, VLLM_CONTEXT_POLICY_VERSION, vllm_context_policy

    policy = vllm_context_policy(MAX_MEASURED_CONTEXT)
    if (
        row.get("context_policy") != policy
        or type(row.get("context_policy_version")) is not int
        or row["context_policy_version"] != VLLM_CONTEXT_POLICY_VERSION
        or type(row.get("native_max_model_len")) is not int
        or row["native_max_model_len"] != policy["runtime_context_length"]
    ):
        raise ValueError("Ops actual vLLM worker context differs from frozen measured context policy")


def load_native(run: dict, base: Path) -> dict:
    """Read independent GPU truth for Ops; never substitute FPM's host interval."""
    return _load_native(run, base)


def _load_native(run: dict, base: Path, *, calibration_evidence: bool = True) -> dict:
    """The exporter alone may validate original execution before freezing its sidecar."""
    raw_root = run["spec"].get("raw_root")
    if not raw_root:
        raise FileNotFoundError("native Ops raw collection is not supplied")
    root = (base / raw_root).resolve()
    backend, fmt, tp, phase = run["key"]
    mode = run["spec"].get("ops_execution_mode", "eager")
    if mode not in ("eager", "native_full_graph") or (
        mode == "native_full_graph" and (backend != "sglang" or phase != "decode")
    ):
        raise ValueError("unsupported native Ops execution mode/backend/phase")
    graph = mode == "native_full_graph"
    if run["role"] not in ("calibration", "holdout") and not (graph and run["role"] == "control"):
        raise ValueError("unsupported native Ops dataset role")
    unprofiled = run["role"] in ("holdout", "control")
    boundary = "native_full_graph_metadata_to_logits_gpu_v1" if graph else BOUNDARY
    version = expected_runtime_version(run)
    if version != BACKENDS[backend][0] and tp not in (2, 4):
        raise ValueError("repaired Ops runtime is qualified only at TP2/TP4")
    pins = _runtime_audit(root, backend, version)
    execution = _sglang_execution(root, fmt, tp, run["points"], graph=graph) if backend == "sglang" else None
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

    expected_config = sha256_json(_load_pre_downloaded_hf_config(CHECKPOINTS[fmt][0]))
    if version != BACKENDS[backend][0]:
        manifest = json.loads((root / "manifest.json").read_bytes())
        provenance = json.loads((root / "provenance.json").read_bytes())
        expected_identity = {
            "backend": backend,
            "backend_version": version,
            "backend_revision": BACKENDS[backend][1],
            "checkpoint_revision": CHECKPOINTS[fmt][1],
            "config_sha256": expected_config,
        }
        if (
            any(
                value.get(key) != expected
                for value in (manifest, provenance)
                for key, expected in expected_identity.items()
            )
            or manifest.get("tp_size") != tp
            or provenance.get("source_sha256") != sha256_json(pins)
        ):
            raise ValueError("repaired Ops manifest/provenance differs from frozen runtime identity")
    requests = json.loads((root / "requests.json").read_bytes())
    dataset_role = "calibration" if run["role"] == "control" else run["role"]
    if requests["dataset_role"] != dataset_role or requests["corpus_sha256"] != run["corpus"]:
        raise ValueError("Ops request role/corpus differs from frozen plan")
    if backend == "sglang":
        from collector.glm53flash_sglang_retained import validate_retained_states

        validate_retained_states(
            requests,
            {rank: root / f"forward-rank-{rank}.jsonl" for rank in range(tp)},
            {rank: root / f"retained-rank-{rank}.jsonl" for rank in range(tp)},
        )
    expected = {
        p["benchmark_id"]: (
            p["batch_size"],
            p["total_prefill_tokens"] // p["batch_size"] if phase == "prefill" else 1,
            p["total_kv_read_tokens"] // p["batch_size"],
        )
        for p in run["points"]
    }
    declared = {}
    for rid, point in requests["requests"].items():
        if point["target_phase"] != ("context" if phase == "prefill" else "generation"):
            raise ValueError("Ops request phase differs from frozen plan")
        key = point["benchmark_id"], point["repetition"]
        coordinates = point["target_batch_size"], point["target_query"], point["target_prefix"]
        if expected.get(key[0]) != coordinates:
            raise ValueError("Ops native request geometry differs from frozen plan")
        group = declared.setdefault(key, {"ids": set(), "role": point["sampling_role"], "coordinates": coordinates})
        if group["role"] != point["sampling_role"] or rid in group["ids"]:
            raise ValueError("Ops request identities or roles disagree")
        group["ids"].add(rid)
    for bid, coordinates in expected.items():
        matching = [(key, item) for key, item in declared.items() if key[0] == bid]
        if (
            len([1 for _, item in matching if item["role"] == "warmup"]) < 5
            or len([1 for _, item in matching if item["role"] == "measurement"]) < 10
        ):
            raise ValueError("Ops native point needs five warmups and ten measurements")
        if any(len(item["ids"]) != coordinates[0] for _, item in matching):
            raise ValueError("Ops frozen request batch is incomplete")
    timings = {}
    rank_inputs = {}
    hardware_by_rank, hardware_uuids = {}, set()
    for rank in range(tp):
        layout = json.loads((root / f"state-layout-rank-{rank}.json").read_bytes())
        # The two adapters preserve their established semantic hash conventions.
        layout_hash = hashlib.sha256(
            json.dumps(layout, sort_keys=True, **({"separators": (",", ":")} if backend == "vllm" else {})).encode()
        ).hexdigest()
        _state_layout(layout, backend)
        if backend == "vllm":
            from collector.glm53flash_runtime_identity import validate_vllm_runtime_closure

            source_manifest = Path(__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
            validate_vllm_runtime_closure(version, source_manifest, layout.get("runtime_closure"))
        if type(layout.get("tp_rank")) is not int or layout["tp_rank"] != rank:
            raise ValueError("Ops actual hardware/state inventory belongs to another TP rank")
        hardware = layout["hardware"]
        if "uuid" in hardware:
            if hardware["uuid"] in hardware_uuids:
                raise ValueError("Ops TP workers reused the same physical GPU UUID")
            hardware_uuids.add(hardware["uuid"])
        hardware_by_rank[str(rank)] = hardware
        previous, observed, seen_forward_ids, completed_requests = {}, set(), set(), set()
        for row in iter_records(root / f"forward-rank-{rank}.jsonl"):
            if (row.get("backend"), row.get("backend_version"), row.get("backend_revision")) != (
                backend,
                version,
                BACKENDS[backend][1],
            ):
                raise ValueError("Ops native seed/target runtime differs from frozen plan")
            if backend == "vllm":
                _validate_vllm_context(row)
                if row.get("native_runner") != "v2":
                    raise ValueError("Ops requires the pinned native V2 runner observation boundary")
            if execution is not None and any(row.get(key) != value for key, value in execution.items()):
                raise ValueError("Ops raw forward execution differs from the retained native runtime receipts")
            if row.get("forward_id") in seen_forward_ids:
                raise ValueError("Ops native forward identity was reused")
            seen_forward_ids.add(row["forward_id"])
            if (
                row.get("allocated_fake_tokens") != 0
                or row.get("state_protocol") != "glm53flash_same_request_real_hybrid_v1"
            ):
                raise ValueError("Ops cannot admit synthetic or unknown state initialization")
            if row.get("config_sha256") != expected_config or row.get("source_sha256") != sha256_json(pins):
                raise ValueError("Ops forward differs from native config/source audit")
            if row["tp_rank"] != rank or row.get("gpu_completed") is not True:
                raise ValueError("Ops forward lacks native rank/completion evidence")
            if row.get("state_layout_sha256") != layout_hash or row.get("state_layout_admitted") is not True:
                raise ValueError("Ops forward state inventory differs from its retained allocation")
            target_graph = graph and row.get("stage") == "measure"
            if row.get("used_cuda_graph") is not target_graph or row.get("runtime_mode") != (
                "FULL" if target_graph else "NONE"
            ):
                raise ValueError("Ops actual forward differs from declared eager/FULL execution")
            if (
                row.get("request_ids") != [request["request_id"] for request in row["requests"]]
                or len(row["requests"]) != row["batch_size"]
            ):
                raise ValueError("Ops actual request order/batch differs from native metadata")
            if (
                sum(row["query_lengths"]) != row["total_new_tokens"]
                or sum(row["prefix_lengths"]) != row["total_past_kv_tokens"]
            ):
                raise ValueError("Ops native token totals disagree")
            for request, query, prefix in zip(
                row["requests"], row["query_lengths"], row["prefix_lengths"], strict=True
            ):
                validate_native_workload(backend, row["phase"], prefix, query, version)
                rid = request["request_id"]
                if rid not in requests["requests"]:
                    raise ValueError("Ops observed an unfrozen request")
                if rid in completed_requests:
                    raise ValueError("Ops completed request was reused after its target")
                tokens = request.get("native_query_token_ids")
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != query
                    or any(type(t) is not int or t < 0 for t in tokens)
                ):
                    raise ValueError("Ops requires actual native input token IDs")
                prior = previous.get(rid)
                if prefix:
                    if not prior or request["previous_forward_id"] != prior[0] or prefix != len(prior[1]):
                        raise ValueError("Ops same-request real-prefix history is incomplete")
                    history = prior[1] + tokens
                else:
                    if prior or request.get("previous_forward_id") is not None:
                        raise ValueError("Ops request identity was restarted")
                    history = tokens
                prompt = request["prompt_token_ids"]
                if not prompt or history[: len(prompt)] != prompt[: len(history)] or (prior and prior[2] != prompt):
                    raise ValueError("Ops native request prompt/history changed")
                if not request["same_request_real_prefix"] or (
                    request["computed_tokens_before"],
                    request["computed_tokens_after"],
                ) != (prefix, prefix + query):
                    raise ValueError("Ops forward did not advance native state")
                if request["input_tokens_sha256"] != sha256_json(history):
                    raise ValueError("Ops native input history digest mismatch")
                sample = request.get("sampled_token_id")
                if prefix >= len(prompt) and (not prior or tokens != [prior[3]]):
                    raise ValueError("Ops decode differs from preceding sampled token")
                if type(sample) is not int or sample < 0:
                    raise ValueError("Ops lacks completed native sampled token")
                previous[rid] = row["forward_id"], history, prompt, sample
                token_key = (rid, prefix, query)
                signature = (sha256_json(history), sha256_json(prompt), sample)
                if rank_inputs.setdefault(token_key, signature) != signature:
                    raise ValueError("Ops TP ranks disagree on actual input/sample history")
                if row["stage"] == "measure":
                    previous.pop(rid)
                    completed_requests.add(rid)
            if row["stage"] != "measure":
                continue
            if any(row.get(field) != requests[field] for field in ("dataset_role", "request_set", "corpus_sha256")):
                raise ValueError("Ops forward request/corpus provenance differs from frozen manifest")
            key = row["benchmark_id"], row["repetition"]
            target = declared.get(key)
            ids = {request["request_id"] for request in row["requests"]}
            actual = row["batch_size"], row["query_lengths"][0], row["prefix_lengths"][0]
            if (
                key in observed
                or target is None
                or ids != target["ids"]
                or actual != target["coordinates"]
                or row["sampling_role"] != target["role"]
            ):
                raise ValueError("Ops native target is duplicate, incomplete or differs from frozen geometry")
            if len(set(row["query_lengths"])) != 1 or len(set(row["prefix_lengths"])) != 1:
                raise ValueError("Ops native target is heterogeneous")
            if (
                row.get("backend"),
                row.get("backend_version"),
                row.get("backend_revision"),
                row.get("checkpoint_revision"),
            ) != (backend, version, BACKENDS[backend][1], CHECKPOINTS[fmt][1]):
                raise ValueError("Ops native runtime/checkpoint differs from pinned identity")
            observed.add(key)
            if unprofiled:
                value = row.get("whole_forward_gpu_ms")
                if (
                    row.get("ops_instrumented") is not False
                    or row.get("whole_forward_boundary") != boundary
                    or isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("Ops independent holdout lacks uninstrumented whole-GPU truth")
                timings.setdefault(key, {})[rank] = value
            elif row.get("ops_instrumented") is not True:
                raise ValueError("Ops calibration lacks native module observations")
        if observed != declared.keys():
            raise ValueError("Ops native rank did not complete all frozen point repetitions")
    values = {}
    if unprofiled:
        for bid in expected:
            samples = [
                max(timings[key].values())
                for key, target in declared.items()
                if key[0] == bid and target["role"] == "measurement"
            ]
            values[bid] = statistics.median(samples)
    elif calibration_evidence and not graph:
        receipt = _read_evidence(root)
        if any(
            receipt.get(key) != value
            for key, value in {
                "request_set": requests["request_set"],
                "tp_size": tp,
                "corpus_sha256": run["corpus"],
                "backend": backend,
                "checkpoint_revision": CHECKPOINTS[fmt][1],
            }.items()
        ):
            raise ValueError("Ops calibration evidence belongs to another native run")
    graph_proof = None
    if graph:
        from collector.glm53flash_graph_export import read_graph_run, verify_evidence

        graph_proof = read_graph_run(root, run)
        if run["role"] == "calibration" and calibration_evidence:
            verify_evidence(root, graph_proof)
    execution_policy = {}
    if backend == "sglang":
        from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy

        execution_policy = _sglang_execution_policy(json.loads((root / "sglang-resolved-config.json").read_bytes()))
    return {
        **execution_policy,
        "values": values,
        "request_ids": set(requests["requests"]),
        "receipts": [
            {"path": str(path.relative_to(root)), "sha256": file_sha(path)}
            for path in sorted(root.iterdir())
            if path.is_file()
        ],
        "runtime_run_id": requests["request_set"],
        "runtime_grid_digest": hashlib.sha256(canonical_json(expected).encode()).hexdigest(),
        "input_provenance": {"text_sha256": requests["corpus_sha256"], "tokenizer_revision": CHECKPOINTS[fmt][1]},
        "backend_version": version,
        "timing_boundary": boundary,
        "evidence_root": str(root),
        "hardware_by_rank": hardware_by_rank,
        "graph_policy": graph_proof["policy"] if graph_proof else None,
    }


def _bind_raw_to_forwards(root: Path, tp: int) -> None:
    for rank in range(tp):
        forwards = {}
        for forward in iter_records(root / f"forward-rank-{rank}.jsonl"):
            if forward["stage"] == "measure":
                if forward["invocation"] in forwards:
                    raise ValueError("Ops native invocation identity reused")
                # Token histories have already been validated. Retain only the
                # per-forward scalar/identity join fields needed for op rows.
                forwards[forward["invocation"]] = {key: value for key, value in forward.items() if key != "requests"}
                forwards[forward["invocation"]]["history_ids"] = (
                    [request["previous_forward_id"] for request in forward["requests"]]
                    if forward["prefix_lengths"][0]
                    else []
                )
        for row in iter_records(root / f"rank-{rank}.jsonl"):
            forward = forwards.get(row["invocation"])
            if forward is None or row["tp_rank"] != rank:
                raise ValueError("Ops module observation has no admitted native forward")
            fields = (
                "stage",
                "phase",
                "benchmark_id",
                "repetition",
                "sampling_role",
                "dataset_role",
                "request_set",
                "corpus_sha256",
                "backend",
                "backend_version",
                "backend_revision",
                "checkpoint_revision",
                "source_sha256",
                "config_sha256",
                "runtime_digest",
                "used_cuda_graph",
                "request_ids",
            )
            if any(row.get(field) != forward.get(field) for field in fields):
                raise ValueError("Ops module observation belongs to a different native request/forward")
            batch, query, prefix = forward["batch_size"], forward["query_lengths"][0], forward["prefix_lengths"][0]
            if row.get("history_ids") != forward["history_ids"]:
                raise ValueError("Ops module history differs from admitted native state")
            shape = json.loads(row["geometry"])
            if row["component"] == "attention":
                coordinates = (
                    batch,
                    prefix if forward["phase"] == "context" else 0,
                    query if forward["phase"] == "context" else prefix,
                )
            else:
                coordinates = (1, 0, batch if shape.get("token_selection") == "last_per_request" else batch * query)
            if tuple(row[k] for k in ("batch_size", "prefix", "x")) != coordinates:
                raise ValueError("Ops module workload differs from admitted native forward")
            fingerprint, kernels = row.get("dispatch_fingerprint", ""), row.get("dispatch_kernels", [])
            if fingerprint and (not kernels or sha256_json(kernels) != fingerprint):
                raise ValueError("Ops dispatch fingerprint lacks actual attributed CUDA kernels")


def bind_calibration(paths: list[Path], frozen_run: dict, native_receipt: dict) -> dict:
    """Reaggregate retained native rows and compare every selected physical row."""
    if frozen_run["spec"].get("ops_execution_mode") == "native_full_graph":
        from collector.glm53flash_graph_export import bind_calibration as bind_graph

        return bind_graph(paths, frozen_run, native_receipt)
    import pyarrow.parquet as pq

    root = Path(native_receipt["evidence_root"])
    evidence = _read_evidence(root)
    digest = file_sha(root / "calibration-evidence.json")
    manifest = json.loads((root / "manifest.json").read_bytes())
    backend, fmt, tp, phase = frozen_run["key"]
    version = expected_runtime_version(frozen_run)
    if native_receipt.get("backend_version", version) != version:
        raise ValueError("Ops calibration runtime differs from frozen plan")
    production = build_model_manifest(backend, fmt, tp)
    if manifest.get("phases") != production["phases"]:
        raise ValueError("Ops calibration manifest differs from the complete production graph")
    _bind_raw_to_forwards(root, tp)
    rows = aggregate_rank_records(
        [root / f"rank-{rank}.jsonl" for rank in range(tp)],
        tp,
        manifest,
        evidence_sha256=digest,
        point_ids=frozen_run.get("original_point_ids"),
        aggregation_policy=evidence.get("aggregation_policy", PER_OPERATION_TP_MAX),
    )
    expected = {tuple(row[key] for key in KEY_COLUMNS): row for row in rows}
    selected = {}
    for path in paths:
        if path.name != "glm53flash_module_perf.parquet":
            continue
        for row in pq.read_table(path).to_pylist():
            row.setdefault("aggregation_policy", PER_OPERATION_TP_MAX)
            row.setdefault("rank_selection_sha256", "")
            geometry = json.loads(row["geometry"])
            if (
                geometry["backend"],
                geometry["checkpoint_format"],
                geometry.get("tp_size"),
                geometry.get("is_context"),
            ) != (backend, fmt, tp, phase == "prefill"):
                continue
            validate_calibration_row(row)
            key = tuple(row[column] for column in KEY_COLUMNS)
            if (
                key in selected
                or key not in expected
                or any(row[column] != expected[key][column] for column in ROW_COLUMNS)
            ):
                raise ValueError("consumer Ops rows differ from their native calibration evidence")
            selected[key] = row
    if selected.keys() != expected.keys():
        raise ValueError("consumer Ops table omits native calibration physical rows")
    return {
        "evidence_sha256": digest,
        "rows": len(selected),
        "native_runtime_run_id": native_receipt["runtime_run_id"],
        "source_plan_sha256": frozen_run["plan"]["sha256"],
    }


def _sharded_calibration_rows(children: list[tuple[dict, dict]], frozen_shard_manifest: dict):
    from collector.glm53flash_shards import merge_shard_rows

    if not children or len({tuple(run["key"]) for run, _ in children}) != 1:
        raise ValueError("Ops calibration shards must describe one complete deployment phase")
    backend, fmt, tp, _ = children[0][0]["key"]
    versions = {expected_runtime_version(run) for run, _ in children}
    if len(versions) != 1 or any(
        native.get("backend_version", next(iter(versions))) not in versions for _, native in children
    ):
        raise ValueError("Ops calibration shards cannot mix native runtime versions")
    if backend == "sglang":
        from collector.fpm_forward.glm53flash_validation import _same_sglang_policy

        for _, native in children:
            _same_sglang_policy(children[0][1], native, "Ops calibration shards")
    production = build_model_manifest(backend, fmt, tp)
    declared = {shard["child_cell_id"]: shard for shard in frozen_shard_manifest["shards"]}
    rows_by_shard, parents, evidence_receipts, request_ids, run_ids = {}, set(), [], set(), set()
    for run, native in children:
        shard = declared.get(run["cell"]["cell_id"])
        if not shard or run["plan"]["sha256"] != shard["child_plan_sha256"] or run["role"] != "calibration":
            raise ValueError("Ops calibration child differs from frozen shard identity")
        point_ids = {item["native_benchmark_id"]: item["original_point_id"] for item in shard["point_map"]}
        if run.get("original_point_ids") != point_ids:
            raise ValueError("Ops child point ownership differs from frozen shard map")
        if native["runtime_run_id"] in run_ids or request_ids.intersection(native["request_ids"]):
            raise ValueError("Ops shards must preserve independent native runs and request identities")
        run_ids.add(native["runtime_run_id"])
        request_ids.update(native["request_ids"])
        root = Path(native["evidence_root"])
        receipt = _read_evidence(root)
        if receipt["corpus_sha256"] != run["corpus"] or receipt["request_set"] != native["runtime_run_id"]:
            raise ValueError("Ops shard evidence belongs to another native run or corpus")
        manifest = json.loads((root / "manifest.json").read_bytes())
        if manifest.get("phases") != production["phases"]:
            raise ValueError("Ops shard omits native production graph boundaries")
        _bind_raw_to_forwards(root, tp)
        digest = file_sha(root / "calibration-evidence.json")
        rows = aggregate_rank_records(
            [root / f"rank-{rank}.jsonl" for rank in range(tp)],
            tp,
            manifest,
            evidence_sha256=digest,
            point_ids=point_ids,
            aggregation_policy=receipt.get("aggregation_policy", PER_OPERATION_TP_MAX),
        )
        if shard["shard_id"] in rows_by_shard:
            raise ValueError("Ops child shard was supplied twice")
        rows_by_shard[shard["shard_id"]] = rows
        parents.add(shard["parent_cell_id"])
        evidence_receipts.append(
            {"shard_id": shard["shard_id"], "evidence_sha256": digest, "runtime_run_id": native["runtime_run_id"]}
        )
    if len(parents) != 1:
        raise ValueError("Ops calibration child shards belong to different parent cells")
    rows, ownership = merge_shard_rows(production, frozen_shard_manifest, parents.pop(), rows_by_shard)
    return rows, ownership, evidence_receipts


def publish_sharded_calibration(
    children: list[tuple[dict, dict]], frozen_shard_manifest: dict, destination: Path
) -> dict:
    """Publish only after callers admitted every child with load_native()."""
    from collector.glm53flash_contract import write_parquet

    rows, ownership, receipts = _sharded_calibration_rows(children, frozen_shard_manifest)
    if destination.exists():
        raise FileExistsError("never replace an existing native calibration publication")
    write_parquet(rows, destination)
    publication = {
        "schema": "glm53flash_ops_shard_publication_v1",
        "ownership": ownership,
        "children": receipts,
        "table_sha256": file_sha(destination),
    }
    destination.with_suffix(".evidence.json").write_text(canonical_json(publication) + "\n")
    return publication


def bind_sharded_calibration(paths: list[Path], children: list[tuple[dict, dict]], frozen_shard_manifest: dict) -> dict:
    """Reproduce frozen ownership and validate a complete final table, without averaging shards."""
    import pyarrow.parquet as pq

    expected_rows, ownership, receipts = _sharded_calibration_rows(children, frozen_shard_manifest)
    expected = {tuple(row[key] for key in KEY_COLUMNS): row for row in expected_rows}
    selected = {}
    backend, fmt, tp, phase = children[0][0]["key"]
    for path in paths:
        if path.name != "glm53flash_module_perf.parquet":
            continue
        for row in pq.read_table(path).to_pylist():
            row.setdefault("aggregation_policy", PER_OPERATION_TP_MAX)
            row.setdefault("rank_selection_sha256", "")
            shape = json.loads(row["geometry"])
            if (shape["backend"], shape["checkpoint_format"], shape.get("tp_size"), shape.get("is_context")) != (
                backend,
                fmt,
                tp,
                phase == "prefill",
            ):
                continue
            key = tuple(row[column] for column in KEY_COLUMNS)
            if (
                key in selected
                or key not in expected
                or any(row[column] != expected[key][column] for column in ROW_COLUMNS)
            ):
                raise ValueError("consumer Ops shard rows differ from frozen ownership/native evidence")
            selected[key] = row
    if selected.keys() != expected.keys():
        raise ValueError("consumer Ops table omits frozen calibration shard coverage")
    return {"rows": len(selected), "physical_ownership_sha256": sha256_json(ownership), "children": receipts}
