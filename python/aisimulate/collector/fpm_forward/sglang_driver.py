# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive real native SGLang requests and retain exact scheduled observations.

Integration: sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
python/sglang/srt/{entrypoints/engine.py,server_args.py}, Apache-2.0.
Uses public Engine/ServerArgs APIs; no upstream source is copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import time
import uuid
from pathlib import Path

from collector.glm53flash_protocol import (
    MAX_MEASURED_CONTEXT,
    PROTOCOL,
    SGLANG_CONTEXT_HEADROOM,
    TIMING_BOUNDARIES,
    sglang_runtime_context_length,
)
from collector.glm53flash_sglang_retained import PRODUCER_PROTOCOL

from .sglang_artifact import MEASUREMENTS, TELEMETRY_POLICY, WARMUPS, canonical, file_receipt, read_observations


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def wait_retained_release(output: Path, request_ids: list[str], tp: int, timeout_seconds: int) -> None:
    """Responses can precede worker-side release receipts; wait before hashing."""
    deadline = time.monotonic() + timeout_seconds
    pending = set(range(tp))
    while pending:
        for rank in tuple(pending):
            failure = output / f"retained-failed-rank-{rank}.json"
            if failure.exists():
                raise RuntimeError(f"native retained cohort failed on rank {rank}: {failure.read_text()}")
            try:
                with (output / f"retained-rank-{rank}.jsonl").open("rb") as stream:
                    stream.seek(0, 2)
                    size = stream.tell()
                    stream.seek(max(0, size - 262144))
                    raw = stream.read()
                if not raw.endswith(b"\n"):
                    continue
                event = json.loads(raw.splitlines()[-1])
            except (FileNotFoundError, json.JSONDecodeError, IndexError):
                continue
            rows = event.get("requests", [])
            if (
                event.get("producer_protocol") == PRODUCER_PROTOCOL
                and event.get("tp_rank") == rank
                and [row.get("request_id") for row in rows] == request_ids
                and all(row.get("released") is True and row.get("parked") is None for row in rows)
            ):
                pending.remove(rank)
        if time.monotonic() >= deadline and pending:
            raise TimeoutError(f"native retained release receipts missing on ranks {sorted(pending)}")
        if pending:
            time.sleep(0.01)


def observed_scheduler_process(*args, **kwargs):
    """Spawn-safe entrypoint: install before creating each native TP worker."""
    from sglang.srt.managers.scheduler import run_scheduler_process

    from collector.glm53flash_sglang_retained import install as install_retained
    from collector.glm53flash_sglang_runtime import install

    install()
    install_retained()
    return run_scheduler_process(*args, **kwargs)


def create_observed_engine(server_args):
    # The public sglang.Engine is a LazyImport proxy: attribute assignment on
    # that object does not reach the native class used by its __call__.
    from sglang.srt.entrypoints.engine import Engine

    Engine.run_scheduler_process_func = staticmethod(observed_scheduler_process)
    return Engine(server_args=server_args)


def verify_runtime(output: Path) -> None:
    import importlib.metadata

    import sglang

    audit = {"backend": "sglang", "backend_version": importlib.metadata.version("sglang"), "sources": {}}
    try:
        if audit["backend_version"] != "0.5.20":
            raise ValueError("GLM SGLang FPM requires version 0.5.20")
        root = Path(sglang.__file__).resolve().parent
        pins = json.loads((Path(__file__).parent / "runtime/glm53flash_sglang/runtime-source-sha256.json").read_text())
        for name, expected in pins.items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            audit["sources"][name] = actual
            if actual != expected:
                raise ValueError(f"SGLang pinned native source differs: {name}")
        audit["status"] = "passed"
    except Exception as error:
        audit.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "runtime-preflight.json", audit)


def freeze_requests(points: list[dict], *, request_set: str, dataset_role: str, corpus_sha256: str) -> dict:
    mappings = {}
    for point in points:
        from .native_artifact import _expected_scheduled

        _expected_scheduled(point)
        batch = point["batch_size"]
        prefill = point["point_type"] == "prefill"
        query_total = point["total_prefill_tokens"] if prefill else batch
        prefix_total = point["total_kv_read_tokens"]
        if query_total % batch or prefix_total % batch or not 1 <= batch <= 32:
            raise ValueError("GLM SGLang first campaign requires homogeneous batches of 1 through 32")
        query, prefix = query_total // batch, prefix_total // batch
        if point.get("partition") is not None or point.get("rows") not in (None, [[query, prefix]] * batch):
            raise ValueError("SGLang native driver cannot replace heterogeneous requests with homogeneous ones")
        if query < 1 or prefix < 0 or query + prefix > 131072 or query_total > 8192:
            raise ValueError("GLM SGLang point exceeds frozen context/token limits")
        if not prefill and prefix < 1:
            raise ValueError("GLM SGLang decode requires a positive real prefix")
        for repetition in range(WARMUPS + MEASUREMENTS):
            for index in range(batch):
                rid = f"{request_set}-p{point['benchmark_id']}-r{repetition}-q{index}"
                mappings[rid] = {
                    "benchmark_id": point["benchmark_id"],
                    "repetition": repetition,
                    "sampling_role": "warmup" if repetition < WARMUPS else "measurement",
                    "target_phase": "context" if prefill else "generation",
                    "target_query": query,
                    "target_prefix": prefix,
                    "target_batch_size": batch,
                }
    return {
        "request_set": request_set,
        "dataset_role": dataset_role,
        "corpus_sha256": corpus_sha256,
        "requests": mappings,
    }


def validate_eager_args(server, *, resolved: bool) -> None:
    """Respect native ServerArgs' separate declaration/resolution lifecycle."""
    if resolved:
        # Native resolution freezes raw input fields and writes a declaration
        # stash; direct attributes still contain the original CLI values.
        config = server.resolved_dict().get("cuda_graph_config")
        if not isinstance(config, dict) or any(
            config.get(phase, {}).get("backend") != "disabled" for phase in ("decode", "prefill")
        ):
            raise ValueError("resolved SGLang Ops execution must disable both native graph phases")
        return
    config = server.cuda_graph_config
    if config is None:
        if server.cuda_graph_backend_decode != "disabled" or server.cuda_graph_backend_prefill != "disabled":
            raise ValueError("declared SGLang Ops execution must disable both native graph phases")
    elif isinstance(config, dict):
        if any(config.get(phase, {}).get("backend") != "disabled" for phase in ("decode", "prefill")):
            raise ValueError("declared SGLang Ops graph config must disable both phases")
    elif any(getattr(config, phase).backend != "disabled" for phase in ("decode", "prefill")):
        raise ValueError("declared SGLang Ops graph config must disable both phases")


def validate_server_args(args, *, measured_context_limit=MAX_MEASURED_CONTEXT) -> None:
    for name in ("pp_size", "dp_size", "ep_size", "attn_cp_size", "dcp_size", "moe_dp_size", "dwdp_size", "nnodes"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"GLM SGLang collection requires {name}=1")
    runtime_limit = sglang_runtime_context_length(measured_context_limit)
    if args.tp_size not in (2, 4) or args.context_length is None or not 1 <= args.context_length <= runtime_limit:
        raise ValueError("GLM SGLang requires TP2/TP4 and context_length within measured limit plus native headroom")
    if args.kv_cache_dtype != "fp8_e4m3" or not args.disable_radix_cache:
        raise ValueError("GLM SGLang requires FP8 KV and disabled cross-request radix reuse")
    for name in (
        "speculative_algorithm",
        "enable_eplb",
        "cpu_offload_gb",
        "enable_dp_attention",
        "enable_dp_lm_head",
        "enable_fp32_lm_head",
        "enable_attn_tp_input_scattered",
        "enable_pdmux",
        "enable_mixed_chunk",
    ):
        if getattr(args, name, None):
            raise ValueError(f"GLM SGLang collection rejects {name}")
    # Native 0.5.20 uses -1 as the disabled group-size sentinel. A truthiness
    # check would reject every default ServerArgs before loading the model.
    if getattr(args, "offload_group_size", -1) > 0:
        raise ValueError("GLM SGLang requires grouped offloading disabled (offload_group_size <= 0)")
    if not 1 <= args.chunked_prefill_size <= 8192:
        raise ValueError("GLM SGLang requires a positive native chunk budget <= 8192")


def result_payload(
    points: list[dict],
    observations: dict,
    *,
    output: Path,
    manifest_path: Path,
    trace_paths: list[Path],
    provenance: dict,
    input_provenance: dict,
    elapsed: float,
) -> dict:
    from .native_artifact import _expected_scheduled

    results, groups = [], []
    measured = 0.0
    for point in points:
        bid = point["benchmark_id"]
        seconds = [
            observations[0][bid, rep]["native_forward_ms"] / 1000 for rep in range(WARMUPS, WARMUPS + MEASUREMENTS)
        ]
        fpm = {
            "counter_id": bid,
            "dp_rank": 0,
            "wall_time": statistics.median(seconds),
            "scheduled_requests": _expected_scheduled(point),
        }
        results.append({"point": point, "fpms": [fpm]})
        groups.append(
            {
                "benchmark_id": bid,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
                "wall_time": fpm["wall_time"],
            }
        )
        measured += fpm["wall_time"]
    input_provenance = {
        **input_provenance,
        "native_forward_manifest": {
            "runtime_preflight": file_receipt(output.parent / "runtime-preflight.json"),
            "declared_config": file_receipt(output.parent / "sglang-declared-config.json"),
            "resolved_config": file_receipt(output.parent / "sglang-resolved-config.json"),
            "requests": file_receipt(manifest_path),
            "traces": [{"tp_rank": rank, **file_receipt(path)} for rank, path in enumerate(trace_paths)],
            "state_layouts": [
                {"tp_rank": rank, **file_receipt(output.parent / f"state-layout-rank-{rank}.json")}
                for rank in range(len(trace_paths))
            ],
            "retained_states": [
                {"tp_rank": rank, **file_receipt(output.parent / f"retained-rank-{rank}.jsonl")}
                for rank in range(len(trace_paths))
            ],
        },
    }
    return {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "timing_valid": True,
        "stop_reason": None,
        "error": None,
        "skipped_points": [],
        "missing_phases": [],
        "config": {"mode": points[0]["point_type"]},
        "coverage": {"expected_points": len(points), "completed_points": len(points), "skipped_points": 0},
        "results": results,
        "iteration_groups": groups,
        "dp": {"rank": 0, "size": 1},
        "run_id": provenance["run_id"],
        "grid_digest": hashlib.sha256(canonical(points).encode()).hexdigest(),
        "kvwarm": {"enabled": True, "warm_eligible": True, "skip_reason": None, "state_protocol": PROTOCOL},
        "execution_identity": provenance["execution_identity"],
        "context_policy": provenance["context_policy"],
        "producer_protocol": PRODUCER_PROTOCOL,
        "execution_mode": "native_graph_policy",
        "input_provenance": input_provenance,
        "timing_boundary": TIMING_BOUNDARIES["sglang"],
        "producer": {
            "backend": "sglang",
            "backend_version": "0.5.20",
            "warmup_repeats": WARMUPS,
            "measurement_repeats": MEASUREMENTS,
            "telemetry_policy": TELEMETRY_POLICY,
            "timing_rank": 0,
            "format": "aisimulate_normalized_native_sglang_v1",
        },
        "timing": {"benchmark_elapsed_seconds": elapsed, "measured_iteration_seconds": measured},
    }


def raw_checkpoint_config(model_path: str, revision: str, expected_model: str) -> tuple[dict, str]:
    """Keep exact checkpoint-file provenance separate from SDK inferred fields."""
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

    local = Path(model_path) / "config.json"
    if not local.is_file():
        from huggingface_hub import hf_hub_download

        local = Path(hf_hub_download(model_path, "config.json", revision=revision))
    raw = local.read_bytes()
    config = json.loads(raw)
    if config != _load_pre_downloaded_hf_config(expected_model):
        raise ValueError("native checkpoint config file differs from pinned original config")
    return config, hashlib.sha256(raw).hexdigest()


def read_ops_provenance(path: Path, *, raw_config: dict, checkpoint_revision: str, runtime_audit: dict) -> dict:
    """Bind operation rows to the loaded config and verified native source files."""
    import re

    from aisimulate_core.sdk.glm53flash import BACKEND_REVISIONS

    def sha256_json(value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    supplied = json.loads(path.read_text())
    expected = {
        "backend": "sglang",
        "backend_version": "0.5.20",
        "backend_revision": BACKEND_REVISIONS["sglang"],
        "checkpoint_revision": checkpoint_revision,
        "config_sha256": sha256_json(raw_config),
        "source_sha256": sha256_json(runtime_audit["sources"]),
    }
    if runtime_audit.get("status") != "passed" or any(supplied.get(key) != value for key, value in expected.items()):
        raise ValueError("Ops provenance differs from loaded config/checkpoint or verified native source")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", supplied.get("runtime_digest", "")):
        raise ValueError("Ops requires an immutable runtime platform digest")
    if any(key in supplied for key in ("run_id", "execution_identity", "telemetry_policy")):
        raise ValueError("Ops provenance cannot replace the native driver's execution identity")
    return {**expected, "runtime_digest": supplied["runtime_digest"]}


def main(argv=None) -> None:
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoTokenizer

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
    from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS, Glm53FlashConfig
    from aisimulate_core.sdk.utils import get_model_config_from_model_path

    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--benchmark-mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--benchmark-points-file", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, default=Path("/results/benchmark.json"))
    parser.add_argument("--benchmark-max-context-length", type=int, default=MAX_MEASURED_CONTEXT)
    parser.add_argument("--input-text", type=Path, default=Path("/tmp/fpm-bench/fpm_text.txt"))
    parser.add_argument("--tokenizer-revision", required=True, choices=tuple(MODEL_REVISIONS.values()))
    parser.add_argument("--dataset-role", choices=("calibration", "holdout"), default="calibration")
    parser.add_argument("--run-id", default=os.environ.get("DYN_FPM_RUN_ID", "glm53flash"))
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--observation-purpose", choices=("fpm", "ops", "ops_holdout"), default="fpm")
    args = parser.parse_args(argv)
    server = ServerArgs.from_cli_args(args)
    validate_server_args(server, measured_context_limit=args.benchmark_max_context_length)
    if args.observation_purpose in ("ops", "ops_holdout"):
        if bool(os.environ.get("AISIM_GLM53_OPS_MANIFEST")) != (args.observation_purpose == "ops"):
            raise ValueError("SGLang Ops requires an explicit manifest and native eager execution")
        validate_eager_args(server, resolved=False)
    elif os.environ.get("AISIM_GLM53_OPS_MANIFEST"):
        raise ValueError("SGLang FPM cannot run with Ops instrumentation")
    output = args.benchmark_output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("SGLang refuses to overwrite an existing benchmark artifact")
    verify_runtime(output.parent)
    expected_model = next(model for model, revision in MODEL_REVISIONS.items() if revision == args.tokenizer_revision)
    raw_config = get_model_config_from_model_path(server.model_path)["raw_config"]
    expected_config = get_model_config_from_model_path(expected_model)["raw_config"]
    if raw_config != expected_config or server.revision != args.tokenizer_revision:
        raise ValueError("SGLang loaded model config/revision differs from the pinned checkpoint contract")
    Glm53FlashConfig.from_text_config(raw_config["text_config"])
    identity = dict(
        zip(EXECUTION_COLUMNS, execution_identity(raw_config, backend="sglang", input_modality="text"), strict=True)
    )
    point_payload = json.loads(args.benchmark_points_file.read_text())
    points = [
        {
            **point,
            "point_type": args.benchmark_mode,
            "benchmark_id": index,
            "total_prefill_tokens": point.get("total_prefill_tokens", 0),
            "sample_reasons": ["kvwarm_real_kv"],
        }
        for index, point in enumerate(point_payload[args.benchmark_mode], 1)
    ]
    if not points:
        raise ValueError("SGLang native driver requires a nonempty phase manifest")
    for point in points:
        queries = point["total_prefill_tokens"] if args.benchmark_mode == "prefill" else point["batch_size"]
        if (queries + point["total_kv_read_tokens"]) > args.benchmark_max_context_length * point["batch_size"]:
            raise ValueError("SGLang point exceeds frozen measured context limit")
    text_raw = args.input_text.read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(server.model_path, revision=args.tokenizer_revision)
    tokens = tokenizer.encode(text_raw.decode(), add_special_tokens=False)
    if len(set(tokens)) < 2:
        raise ValueError("SGLang corpus must contain multiple tokenizer-generated tokens")
    request_set = f"{args.dataset_role}-{uuid.uuid4().hex}"
    input_provenance = {
        "source": "tokenizer_text",
        "text_sha256": hashlib.sha256(text_raw).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "tokenizer_revision": args.tokenizer_revision,
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
    }
    manifest = freeze_requests(
        points, request_set=request_set, dataset_role=args.dataset_role, corpus_sha256=input_provenance["text_sha256"]
    )
    manifest_path = output.parent / "sglang-requests.json"
    write_json(manifest_path, manifest)
    provenance = {
        "run_id": args.run_id,
        "execution_identity": identity,
        "telemetry_policy": TELEMETRY_POLICY,
        "producer_protocol": PRODUCER_PROTOCOL,
        "context_policy": {
            "measured_context_limit": args.benchmark_max_context_length,
            "runtime_context_length": server.context_length,
            "native_admission_headroom": SGLANG_CONTEXT_HEADROOM,
        },
    }
    if args.observation_purpose in ("ops", "ops_holdout"):
        checkpoint_config, checkpoint_file_sha256 = raw_checkpoint_config(
            server.model_path, args.tokenizer_revision, expected_model
        )
        provenance = {
            **read_ops_provenance(
                Path(os.environ["AISIM_GLM53_OPS_PROVENANCE"]),
                raw_config=checkpoint_config,
                checkpoint_revision=args.tokenizer_revision,
                runtime_audit=json.loads((output.parent / "runtime-preflight.json").read_text()),
            ),
            **provenance,
            "checkpoint_config_file_sha256": checkpoint_file_sha256,
        }
    provenance_path = output.parent / "sglang-provenance.json"
    write_json(provenance_path, provenance)
    write_json(output.parent / "sglang-declared-config.json", server.resolved_dict())
    os.environ.update(
        AISIM_GLM53_PURPOSE=args.observation_purpose,
        AISIM_GLM53_TRACE_DIR=str(output.parent),
        AISIM_GLM53_PROVENANCE=str(provenance_path),
        AISIM_GLM53_REQUEST_MANIFEST=str(manifest_path),
    )
    start = time.monotonic()
    engine = None
    try:
        engine = create_observed_engine(server)
        if args.observation_purpose in ("ops", "ops_holdout"):
            validate_eager_args(engine.server_args, resolved=True)
        write_json(output.parent / "sglang-resolved-config.json", engine.server_args.resolved_dict())
        for point in points:
            for repetition in range(WARMUPS + MEASUREMENTS):
                selected = [
                    (rid, entry)
                    for rid, entry in manifest["requests"].items()
                    if (entry["benchmark_id"], entry["repetition"]) == (point["benchmark_id"], repetition)
                ]
                inputs = []
                for request_index, (_, entry) in enumerate(selected):
                    length = entry["target_prefix"] + (entry["target_query"] if args.benchmark_mode == "prefill" else 0)
                    offset = (point["benchmark_id"] * 997 + repetition * 53 + request_index * 17) % len(tokens)
                    inputs.append([tokens[(offset + index) % len(tokens)] for index in range(length)])

                def timeout(_signum, _frame):
                    raise TimeoutError(
                        f"native SGLang request timed out at point {point['benchmark_id']} repetition {repetition}"
                    )

                old_handler = signal.signal(signal.SIGALRM, timeout)
                signal.alarm(args.request_timeout_seconds)
                try:
                    result = engine.generate(
                        input_ids=inputs,
                        rid=[rid for rid, _ in selected],
                        sampling_params={
                            "temperature": 0,
                            "max_new_tokens": 2 if args.benchmark_mode == "decode" else 1,
                            "ignore_eos": True,
                        },
                    )
                    wait_retained_release(
                        output.parent, [rid for rid, _ in selected], server.tp_size, args.request_timeout_seconds
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                with (output.parent / "sglang-completed-requests.jsonl").open("a") as stream:
                    stream.write(
                        canonical(
                            {
                                "benchmark_id": point["benchmark_id"],
                                "repetition": repetition,
                                "request_ids": [rid for rid, _ in selected],
                                "responses": result,
                            }
                        )
                        + "\n"
                    )
        trace_paths = [output.parent / f"forward-rank-{rank}.jsonl" for rank in range(server.tp_size)]
        if args.observation_purpose in ("ops", "ops_holdout"):
            write_json(
                output.parent / "ops-run-summary.json",
                {
                    "status": "requests_complete",
                    "observation_purpose": args.observation_purpose,
                    "accuracy_acceptance": "NOT_EVALUATED",
                    "requested_points": points,
                    "request_manifest": file_receipt(manifest_path),
                    "native_traces": [file_receipt(path) for path in trace_paths],
                    "warmup_repeats": WARMUPS,
                    "measurement_repeats": MEASUREMENTS,
                },
            )
            return
        observations = read_observations(manifest, dict(enumerate(trace_paths)), points, compact=True)
        payload = result_payload(
            points,
            observations,
            output=output,
            manifest_path=manifest_path,
            trace_paths=trace_paths,
            provenance=provenance,
            input_provenance=input_provenance,
            elapsed=time.monotonic() - start,
        )
        write_json(output, payload)
    except BaseException as error:
        write_json(
            output.parent / "sglang-failed.json",
            {
                "status": "failed",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - start,
                "requested_points": points,
                "accuracy_acceptance": "NOT_EVALUATED",
            },
        )
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
