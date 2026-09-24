# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU evidence-integrity tests; the observations here are synthetic fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from collector.fpm_forward.sglang_artifact import (
    TELEMETRY_POLICY,
    file_receipt,
    read_observations,
    validate_sglang_repetitions,
)
from collector.fpm_forward.sglang_driver import freeze_requests, result_payload
from collector.glm53flash_protocol import PROTOCOL, TIMING_BOUNDARIES
from collector.glm53flash_sglang_retained import PRODUCER_PROTOCOL

pytestmark = pytest.mark.unit


def provenance():
    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
    from aisimulate_core.sdk.utils import get_model_config_from_model_path

    config = get_model_config_from_model_path("zai-org/GLM-5.3-Flash")["raw_config"]
    return {
        "run_id": "run",
        "execution_identity": dict(
            zip(EXECUTION_COLUMNS, execution_identity(config, backend="sglang", input_modality="text"), strict=True)
        ),
        "telemetry_policy": TELEMETRY_POLICY,
        "producer_protocol": PRODUCER_PROTOCOL,
        "context_policy": {
            "measured_context_limit": 131072,
            "runtime_context_length": 131079,
            "native_admission_headroom": 7,
        },
    }


def fixture():
    point = {
        "benchmark_id": 1,
        "point_type": "decode",
        "batch_size": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2,
    }
    manifest = freeze_requests([point], request_set="independent", dataset_role="calibration", corpus_sha256="a" * 64)
    records = {0: [], 1: []}
    for rank in records:
        for rid, entry in manifest["requests"].items():
            rep = entry["repetition"]
            for step, (query, prefix, tokens, sampled) in enumerate(((2, 0, [4, 5], 7), (1, 2, [7], 9))):
                fid = f"rank-{rank}/forward-{rep * 2 + step}"
                history = [4, 5] if step == 0 else [4, 5, 7]
                record = {
                    **provenance(),
                    "tp_rank": rank,
                    "state_protocol": PROTOCOL,
                    "allocated_fake_tokens": 0,
                    "gpu_completed": True,
                    "ops_instrumented": False,
                    "timing_boundary": TIMING_BOUNDARIES["sglang"],
                    "forward_id": fid,
                    "request_ids": [rid],
                    "batch_size": 1,
                    "query_lengths": [query],
                    "prefix_lengths": [prefix],
                    "total_new_tokens": query,
                    "total_past_kv_tokens": prefix,
                    "phase": "context" if not step else "generation",
                    "stage": "seed" if not step else "measure",
                    "benchmark_id": 1,
                    "repetition": rep,
                    "sampling_role": entry["sampling_role"],
                    "request_set": manifest["request_set"],
                    "dataset_role": "calibration",
                    "corpus_sha256": "a" * 64,
                    "native_forward_ms": 999 if rep < 5 else rep + 1,
                    "runtime_mode": "NONE" if not step else "FULL",
                    "used_cuda_graph": bool(step),
                    "num_padded_tokens": query,
                    "requests": [
                        {
                            "request_id": rid,
                            "native_query_token_ids": tokens,
                            "prompt_token_ids": [4, 5],
                            "sampled_token_id": sampled,
                            "previous_forward_id": f"rank-{rank}/forward-{rep * 2}" if step else None,
                            "same_request_real_prefix": True,
                            "computed_tokens_before": prefix,
                            "computed_tokens_after": prefix + query,
                            "input_tokens_sha256": hashlib.sha256(
                                json.dumps(history, separators=(",", ":")).encode()
                            ).hexdigest(),
                        }
                    ],
                }
                records[rank].append(record)
    return point, manifest, records


def raw(records):
    return {rank: ("\n".join(json.dumps(row) for row in rows) + "\n").encode() for rank, rows in records.items()}


def artifact(tmp_path):
    from collector.fpm_forward import sglang_artifact

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS
    from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS

    point, manifest, records = fixture()
    pins = json.loads(
        (
            sglang_artifact.Path(sglang_artifact.__file__).parent
            / "runtime/glm53flash_sglang/runtime-source-sha256.json"
        ).read_text()
    )
    native_config = {
        "model_path": "/models/GLM-5.3-Flash",
        "revision": MODEL_REVISIONS["zai-org/GLM-5.3-Flash"],
        "tp_size": 2,
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "attn_cp_size": 1,
        "nnodes": 1,
        "context_length": 131079,
        "kv_cache_dtype": "fp8_e4m3",
        "disable_radix_cache": True,
        "chunked_prefill_size": 8192,
        "cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}},
    }
    for name, value in {
        "runtime-preflight.json": {
            "backend": "sglang",
            "backend_version": "0.5.20",
            "status": "passed",
            "sources": pins,
        },
        "sglang-declared-config.json": {**native_config, "cuda_graph_config": None},
        "sglang-resolved-config.json": native_config,
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    layout = {
        "admitted": True,
        "logical_kv_dtype": "torch.float8_e4m3fn",
        "groups": {
            name: [{"dtype": dtype, "shape": [16, 4], "stride": [4, 1]}]
            for name, dtype in {
                "mla_latent": "torch.float8_e4m3fn",
                "kda_conv": "torch.bfloat16",
                "kda_temporal": "torch.float32",
                "pooled_index_packed": "torch.uint8",
                "index_tail_key": "torch.bfloat16",
                "index_tail_score": "torch.bfloat16",
            }.items()
        },
    }
    for rank, rows in records.items():
        (tmp_path / f"state-layout-rank-{rank}.json").write_text(json.dumps(layout))
        for row in rows:
            row.update(
                state_layout_admitted=True,
                state_layout_sha256=hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
            )
        events, parked = [], {}

        def digest(values):
            return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

        for row in rows:
            rid = row["request_ids"][0]
            prefix, query = row["prefix_lengths"][0], row["query_lengths"][0]
            indices = list(range(10, 10 + prefix + query))
            completed = {
                "req_pool_idx": 1,
                "mamba_pool_idx": 3,
                "committed_tokens": len(indices),
                "allocated_tokens": len(indices),
                "cached_prefix_tokens": prefix,
                "retained_tokens": prefix,
                "committed_indices_sha256": digest(indices),
                "cached_prefix_indices_sha256": digest(indices[:prefix]),
                "retained_indices_sha256": digest(indices[:prefix]),
            }
            after = (
                None
                if row["stage"] == "measure"
                else {
                    **completed,
                    "cached_prefix_tokens": len(indices),
                    "retained_tokens": len(indices),
                    "cached_prefix_indices_sha256": digest(indices),
                    "retained_indices_sha256": digest(indices),
                }
            )
            events.append(
                {
                    "producer_protocol": PRODUCER_PROTOCOL,
                    "tp_rank": rank,
                    "forward_id": row["forward_id"],
                    "requests": [
                        {
                            "request_id": rid,
                            "before": parked.get(rid),
                            "completed": completed,
                            "parked": after,
                            "released": after is None,
                        }
                    ],
                }
            )
            parked[rid] = after
        (tmp_path / f"retained-rank-{rank}.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    traces = raw(records)
    observations = read_observations(manifest, traces, [point])
    manifest_path = tmp_path / "sglang-requests.json"
    manifest_path.write_text(json.dumps(manifest))
    paths = [tmp_path / f"forward-rank-{rank}.jsonl" for rank in traces]
    for rank, path in enumerate(paths):
        path.write_bytes(traces[rank])
    payload = result_payload(
        [point],
        observations,
        output=tmp_path / "benchmark.json",
        manifest_path=manifest_path,
        trace_paths=paths,
        provenance=provenance(),
        input_provenance={"text_sha256": "a" * 64, "tokenizer_revision": native_config["revision"]},
        elapsed=100,
    )
    cell = SimpleNamespace(
        state_protocol=PROTOCOL,
        topology=SimpleNamespace(tp=2),
        execution_identity=tuple(provenance()["execution_identity"][key] for key in EXECUTION_COLUMNS),
    )
    return cell, payload


def test_native_sglang_median_excludes_warmup_and_roundtrips(tmp_path):
    cell, payload = artifact(tmp_path)
    assert payload["results"][0]["fpms"][0]["wall_time"] == pytest.approx(0.0105)
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")
    payload["results"][0]["fpms"][0]["wall_time"] = 0.999
    with pytest.raises(ValueError, match="median"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize("index", [10, 11])
@pytest.mark.parametrize("field", ["run_id", "execution_identity", "telemetry_policy", "context_policy"])
def test_native_sglang_binds_seed_and_target_identity(field, index):
    point, manifest, records = fixture()
    records[1][index].pop(field)
    with pytest.raises(ValueError, match="provenance"):
        read_observations(manifest, raw(records), [point])


@pytest.mark.parametrize("field", ["run_id", "execution_identity", "context_policy"])
def test_native_sglang_rejects_rehashed_consistent_trace_from_other_execution(tmp_path, field):
    cell, payload = artifact(tmp_path)
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for rank, receipt in enumerate(evidence["traces"]):
        path = tmp_path / receipt["file"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if field == "run_id":
                row[field] = "other-run"
            elif field == "execution_identity":
                row[field]["model_config_sha256"] = "b" * 64
            else:
                row[field]["runtime_context_length"] -= 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        evidence["traces"][rank] = {"tp_rank": rank, **file_receipt(path)}
    with pytest.raises(ValueError, match="provenance differs"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize("kind", ["runtime_preflight", "declared_config", "resolved_config"])
def test_native_sglang_requires_runtime_receipts(tmp_path, kind):
    cell, payload = artifact(tmp_path)
    del payload["input_provenance"]["native_forward_manifest"][kind]
    with pytest.raises(ValueError, match="receipt is missing"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("runtime_preflight", "sources", {}),
        ("runtime_preflight", "backend_version", "0.5.19"),
        ("declared_config", "revision", "other-checkpoint"),
        ("resolved_config", "tp_size", 4),
        ("resolved_config", "kv_cache_dtype", "bfloat16"),
        ("resolved_config", "disable_radix_cache", False),
        ("resolved_config", "context_length", 131078),
        ("resolved_config", "model_path", "/models/other-checkpoint"),
        ("resolved_config", "cuda_graph_config", None),
        ("resolved_config", "allow_auto_truncate", True),
        ("resolved_config", "dcp_size", 2),
        ("resolved_config", "enable_attn_tp_input_scattered", True),
    ],
)
def test_native_sglang_rejects_rehashed_runtime_config_corruption(tmp_path, kind, field, value):
    cell, payload = artifact(tmp_path)
    evidence = payload["input_provenance"]["native_forward_manifest"]
    path = tmp_path / evidence[kind]["file"]
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    evidence[kind] = file_receipt(path)
    with pytest.raises(ValueError):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize(
    "corruption", ["seed", "fake", "tokens", "ops", "padding", "rank", "missing", "reused", "phase", "role"]
)
def test_native_sglang_rejects_invalid_observations(corruption):
    point, manifest, records = fixture()
    selected = records[1][11]
    if corruption == "seed":
        records[1].pop(10)
    elif corruption == "fake":
        selected["allocated_fake_tokens"] = 1
    elif corruption == "tokens":
        selected["requests"][0]["native_query_token_ids"] = [88]
    elif corruption == "ops":
        selected["ops_instrumented"] = True
    elif corruption == "padding":
        selected["num_padded_tokens"] = None
    elif corruption == "rank":
        selected["tp_rank"] = 0
    elif corruption == "missing":
        selected["stage"] = "seed"
    elif corruption == "reused":
        records[1].append(selected)
    elif corruption == "phase":
        selected["phase"] = "context"
    else:
        selected["sampling_role"] = "warmup"
    with pytest.raises(ValueError):
        read_observations(manifest, raw(records), [point])


def test_ops_provenance_is_bound_to_loaded_config_and_native_source(tmp_path):
    from collector.fpm_forward.sglang_driver import read_ops_provenance

    from aisimulate_core.sdk.glm53flash import BACKEND_REVISIONS

    def sha256_json(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    config = {"actual": "loaded"}
    audit = {"status": "passed", "sources": {"native.py": "a" * 64}}
    provenance = {
        "backend": "sglang",
        "backend_version": "0.5.20",
        "backend_revision": BACKEND_REVISIONS["sglang"],
        "checkpoint_revision": "pinned-checkpoint",
        "config_sha256": sha256_json(config),
        "source_sha256": sha256_json(audit["sources"]),
        "runtime_digest": "sha256:" + "b" * 64,
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance))
    assert (
        read_ops_provenance(path, raw_config=config, checkpoint_revision="pinned-checkpoint", runtime_audit=audit)
        == provenance
    )
    with pytest.raises(ValueError, match="differs"):
        read_ops_provenance(
            path, raw_config={"other": "config"}, checkpoint_revision="pinned-checkpoint", runtime_audit=audit
        )
    path.write_text(json.dumps({**provenance, "execution_identity": {}}))
    with pytest.raises(ValueError, match="cannot replace"):
        read_ops_provenance(path, raw_config=config, checkpoint_revision="pinned-checkpoint", runtime_audit=audit)


@pytest.mark.parametrize("all_ranks", [False, True])
def test_native_sglang_rejects_rehashed_wrong_sample_chain(all_ranks):
    point, manifest, records = fixture()
    for rank in records if all_ranks else (1,):
        request = records[rank][11]["requests"][0]
        request["native_query_token_ids"] = [88]
        request["input_tokens_sha256"] = hashlib.sha256(b"[4,5,88]").hexdigest()
    with pytest.raises(ValueError, match="preceding sampled token"):
        read_observations(manifest, raw(records), [point])


def test_native_sglang_rejects_internally_valid_but_different_tp_token_chains():
    point, manifest, records = fixture()
    # Rank one is internally continuous, but its preceding sample and decode
    # input differ from rank zero. Geometry and dispatch are unchanged.
    records[1][10]["requests"][0]["sampled_token_id"] = 88
    request = records[1][11]["requests"][0]
    request["native_query_token_ids"] = [88]
    request["input_tokens_sha256"] = hashlib.sha256(b"[4,5,88]").hexdigest()
    with pytest.raises(ValueError, match="TP ranks disagree"):
        read_observations(manifest, raw(records), [point])


def test_native_sglang_rejects_different_tp_final_samples():
    point, manifest, records = fixture()
    records[1][11]["requests"][0]["sampled_token_id"] = 88
    with pytest.raises(ValueError, match="TP ranks disagree"):
        read_observations(manifest, raw(records), [point])


def test_native_sglang_offload_disabled_sentinel_and_active_group():
    from types import SimpleNamespace

    from collector.fpm_forward.sglang_driver import validate_server_args

    args = SimpleNamespace(
        tp_size=2,
        context_length=131072,
        kv_cache_dtype="fp8_e4m3",
        disable_radix_cache=True,
        chunked_prefill_size=8192,
        cpu_offload_gb=0,
        offload_group_size=-1,
        offload_num_in_group=1,
    )
    validate_server_args(args)
    args.context_length = 131079
    validate_server_args(args)
    args.context_length = 131080
    with pytest.raises(ValueError, match="native headroom"):
        validate_server_args(args)
    args.context_length = 131079
    with pytest.raises(ValueError, match="measured context limit"):
        validate_server_args(args, measured_context_limit=131073)
    args.offload_group_size = 4
    with pytest.raises(ValueError, match="grouped offloading disabled"):
        validate_server_args(args)
    args.offload_group_size = -1
    args.cpu_offload_gb = 1
    with pytest.raises(ValueError, match="rejects cpu_offload_gb"):
        validate_server_args(args)


def test_native_sglang_context_headroom_obeys_strict_input_admission():
    from collector.glm53flash_protocol import sglang_runtime_context_length

    measured_limit = 131072
    configured = sglang_runtime_context_length(measured_limit)
    # Native tp_worker leaves six slots; managers.utils rejects equality.
    native_max_input = min(configured - 1, 200000 - 1) - 5
    assert measured_limit < native_max_input
    assert not measured_limit < min(configured - 2, 200000 - 1) - 5
    # A smaller real allocator can still reject: headroom is not capacity proof.
    assert not measured_limit < min(configured - 1, measured_limit - 1) - 5


def test_ops_graph_policy_checks_native_declaration_then_resolution():
    from collector.fpm_forward.sglang_driver import validate_eager_args

    resolved = {"cuda_graph_config": None}
    args = SimpleNamespace(
        cuda_graph_config=None,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        resolved_dict=lambda: resolved,
    )
    validate_eager_args(args, resolved=False)
    with pytest.raises(ValueError, match="resolved"):
        validate_eager_args(args, resolved=True)
    resolved["cuda_graph_config"] = {"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}}
    validate_eager_args(args, resolved=True)
    assert args.cuda_graph_config is None  # Pinned native resolution preserves raw fields.
    resolved["cuda_graph_config"]["prefill"]["backend"] = "tc_piecewise"
    with pytest.raises(ValueError, match="resolved"):
        validate_eager_args(args, resolved=True)
