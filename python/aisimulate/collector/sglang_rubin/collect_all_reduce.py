# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the pinned Rubin image's single-node TP4 BF16 all-reduce dispatch.

Run with ``torchrun --standalone --nproc-per-node=4 --module
collector.sglang_rubin.collect_all_reduce --help``. This is a standalone variant
of collector/network/collect_all_reduce.py, with its existing table contract:
message_size is the number of elements, latency is milliseconds, and backend
is sglang_graph or sglang_eager. The SDK excludes eager rows outside b60:
crates/core/src/perfmodel/perf_database/communication.rs:528-578.

Serving API audit (API calls only; no upstream implementation is copied):
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b
  srt/managers/scheduler.py:465: ParallelState population for TP4/EP1.
  srt/model_executor/model_runner.py:385,1029: device selection and bootstrap.
  srt/distributed/bootstrap.py:67,187,212: native flags and process groups.
  srt/model_executor/runner/base_runner.py:262: fusion workspace initialization.
  srt/models/glm4_moe.py:1447 and deepseek_v2.py:1175: GLM's TP reduction.
  srt/distributed/communication_op.py:18: the serving all-reduce entrypoint.
  srt/distributed/parallel_state.py:586,648,925: graph and eager dispatch.
  srt/model_executor/runner/{decode_cuda_graph_runner.py:1037,
      prefill_cuda_graph_runner.py:1326}: native graph_capture context.

The inputs are contiguous [tokens, 6144] BF16 hidden states, matching the
post-expert tensor at deepseek_v2.py:1162-1175. This measures plain all-reduce;
the fused residual/RMSNorm operation and expert all-to-all are separate ops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from collector.registry_types import PerfFile
from collector.sglang_rubin import _shared_helper
from collector.sglang_rubin.registry import CHECKPOINT_METADATA_SHA256, SGLANG_COMMIT, SGLANG_DISTRIBUTION_VERSION
from collector.sglang_rubin.runtime import (
    IMAGE_ARM64_DIGEST,
    IMAGE_REF,
    IMAGE_REPOSITORY,
    collect_inventory,
    validate_runtime,
)

HIDDEN_SIZE = 6144
WORLD_SIZE = 4
DEFAULT_TOKENS = (1, 8, 32, 128, 1024, 8192, 16384)
REPEAT = 5
MODULE = "collector.sglang_rubin.collect_all_reduce"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_COMM_ARGS = (
    "disable_custom_all_reduce",
    "enable_mscclpp",
    "enable_torch_symm_mem",
    "enable_symm_mem",
    "enable_nccl_nvls",
    "flashinfer_allreduce_fusion_backend",
)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Frozen local GLM-5.2 NVFP4 checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; resuming is not supported")
    parser.add_argument("--launcher-image", required=True, help="Pinned image reference, recorded without attestation")
    parser.add_argument("--tokens", nargs="+", type=int, choices=DEFAULT_TOKENS, default=list(DEFAULT_TOKENS))
    parser.add_argument("--modes", nargs="+", choices=("graph", "eager"), default=["graph", "eager"])
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--iterations", type=_positive_int, default=20)
    parser.add_argument("--timeout", type=_positive_int, default=300, help="Distributed timeout in seconds")
    args = parser.parse_args(argv)
    for name in ("tokens", "modes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    if args.launcher_image not in (IMAGE_REF, f"{IMAGE_REPOSITORY}@{IMAGE_ARM64_DIGEST}"):
        parser.error("--launcher-image must be the pinned index or ARM64 manifest reference")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    return args


def _rank_from_environment(env):
    try:
        rank, local_rank, world_size, local_size = (
            int(env[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
        port = int(env["MASTER_PORT"])
    except (KeyError, ValueError) as error:
        raise ValueError("Launch with torchrun --standalone --nproc-per-node=4") from error
    if world_size != WORLD_SIZE or local_size != WORLD_SIZE or rank != local_rank or not 0 <= rank < WORLD_SIZE:
        raise ValueError("Exactly four ranks on one node are required, with RANK == LOCAL_RANK in [0, 3]")
    if not env.get("MASTER_ADDR") or not 0 < port < 65536 or env.get("GROUP_RANK", "0") != "0":
        raise ValueError("Invalid single-node torchrun rendezvous")
    return rank


def _preflight_errors(inventory):
    errors = validate_runtime(inventory)
    observed = inventory.get("observed", {})
    version = observed.get("package_versions", {}).get("sglang", {}).get("version")
    if version != SGLANG_DISTRIBUTION_VERSION:
        errors.append(f"Expected SGLang {SGLANG_DISTRIBUTION_VERSION!r}, observed {version!r}")
    devices = observed.get("cuda", {}).get("devices", [])
    if len(devices) != WORLD_SIZE or [device.get("index") for device in devices] != list(range(WORLD_SIZE)):
        errors.append("Exactly four visible SM107 CUDA devices with indices 0, 1, 2, 3 are required")
    if (
        any(not device.get("uuid") for device in devices)
        or len({device.get("uuid") for device in devices}) != WORLD_SIZE
    ):
        errors.append("Four distinct observed GPU UUIDs are required")
    files = observed.get("checkpoint", {}).get("files", {})
    for name, digest in CHECKPOINT_METADATA_SHA256.items():
        if files.get(name, {}).get("sha256") != digest:
            errors.append(f"Checkpoint {name} differs from the frozen GLM-5.2 NVFP4 snapshot")
    return errors


def _gather(dist, control, value):
    reports = [None] * WORLD_SIZE
    dist.all_gather_object(reports, value, group=control)
    return reports


def _stage(dist, control, name, action):
    """Make local validation, timing and persistence failures visible to all ranks.

    Control uses a separate Gloo group with a finite timeout. Failures inside a
    collective are ultimately terminated by torchrun; no rank may publish a
    successful table unless all ranks have finished every stage.
    """
    result, failure = None, None
    try:
        result = action()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(_gather(dist, control, failure)) if error]
    if failures:
        raise RuntimeError(f"{name} failed: {'; '.join(failures)}")
    return result


def _memory(torch, rank):
    free, total = torch.cuda.mem_get_info(rank)
    return {
        "free_bytes": free,
        "total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(rank),
        "torch_reserved_bytes": torch.cuda.memory_reserved(rank),
    }


def _initialize(torch, args, rank):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.bootstrap import init_torch_distributed
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    # Real ServerArgs resolves the GLM model/hardware defaults, including
    # FlashInfer fusion (arg_groups/overrides.py:2021). No communicator is forced.
    server_args = ServerArgs(
        model_path=str(args.checkpoint_dir),
        tp_size=WORLD_SIZE,
        ep_size=1,
        device="cuda",
        trust_remote_code=True,
        kv_cache_dtype="fp8_e4m3",
        dist_timeout=args.timeout,
    )
    _set_envs_and_config(server_args)
    publish(server_args, role="scheduler")
    model_config = ModelConfig.from_server_args(server_args)
    if model_config.hidden_size != HIDDEN_SIZE or model_config.dtype != torch.bfloat16:
        raise RuntimeError("The frozen model must have hidden_size=6144 and bfloat16 hidden states")
    # scheduler.py:465; all non-TP parallel dimensions are one in this pilot.
    ps = ParallelState.trivial(
        tp_rank=rank, tp_size=WORLD_SIZE, attn_tp_rank=rank, attn_tp_size=WORLD_SIZE, gpu_id=rank
    )
    os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] = "env://"
    result = init_torch_distributed(
        server_args=server_args,
        model_config=model_config,
        device="cuda",
        ps=ps,
        dist_port=int(os.environ["MASTER_PORT"]),
        is_draft_worker=False,
        local_omp_cpuid=None,
    )
    group = result.tp_group
    if group.world_size != WORLD_SIZE or group.rank_in_group != rank or group.device != torch.device(f"cuda:{rank}"):
        raise RuntimeError("SGLang TP group does not match the torchrun rank/device mapping")
    return group, server_args


def _initialize_workspaces(server_args, torch):
    from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
    from sglang.srt.layers.flashinfer_comm_fusion import _get_workspace_manager, pre_initialize_workspaces

    # BaseRunner._pre_initialize_flashinfer_allreduce_workspace:262-278. Keep
    # the serving workspace size, even for larger sampled messages: resizing it
    # to the sweep maximum would change the framework's all-reduce dispatch.
    if server_args.flashinfer_allreduce_fusion_backend is not None:
        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE, hidden_dim=HIDDEN_SIZE, dtype=torch.bfloat16
        )
    return {
        role: {
            "initialized": manager.initialized,
            "backend": manager.backend,
            "max_token_num": manager.max_token_num,
            "hidden_dim": manager.hidden_dim,
        }
        for role, manager in (("attn_tp", _get_workspace_manager(True)), ("moe", _get_workspace_manager(False)))
    }


def _validate_rank_reports(reports):
    if len(reports) != WORLD_SIZE or [report["rank"] for report in reports] != list(range(WORLD_SIZE)):
        raise RuntimeError("Missing or incorrectly ordered rank reports")
    if len({report["hostname"] for report in reports}) != 1:
        raise RuntimeError("All four ranks must run on the same host")
    if len({report["device"]["uuid"] for report in reports}) != WORLD_SIZE:
        raise RuntimeError("Ranks must own four distinct GPUs")
    for report in reports:
        if report["device"]["index"] != report["rank"] or report["device"]["capability"] != [10, 7]:
            raise RuntimeError("Incorrect rank-to-SM107 device mapping")
        if report["communication"] != reports[0]["communication"]:
            raise RuntimeError("Framework communicator configuration differs between ranks")


@contextmanager
def _trace_dispatch(group, dist, flashinfer_backend):
    """Observe invoked communicator methods; never reimplement their selection.

    Wrappers run only for correctness checks or graph capture, outside timing.
    A torch/NCCL fallback cannot acquire a custom-all-reduce label.
    """
    calls = []

    def observe(method, label):
        @wraps(method)
        def invoke(*args, **kwargs):
            result = method(*args, **kwargs)
            calls.append(label)
            return result

        return invoke

    with ExitStack() as stack:
        for name, methods in (
            ("ca_comm", ("custom_all_reduce",)),
            ("qr_comm", ("quick_all_reduce",)),
            ("pymscclpp_comm", ("all_reduce",)),
            ("torch_symm_mem_comm", ("all_reduce",)),
            ("pynccl_comm", ("all_reduce", "outplace_all_reduce")),
        ):
            comm = getattr(group, name)
            if comm is not None:
                for method_name in methods:
                    label = f"{type(comm).__module__}.{type(comm).__qualname__}.{method_name}"
                    stack.enter_context(patch.object(comm, method_name, observe(getattr(comm, method_name), label)))
        label = f"sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce[{flashinfer_backend}]"
        stack.enter_context(patch.object(group, "_flashinfer_allreduce", observe(group._flashinfer_allreduce, label)))
        stack.enter_context(
            patch.object(dist, "all_reduce", observe(dist.all_reduce, "torch.distributed.all_reduce[nccl]"))
        )
        yield calls


def _validate_outputs(torch, inputs, outputs):
    if len(outputs) != REPEAT:
        raise RuntimeError("All-reduce did not produce every output")
    expected = torch.full_like(inputs[0], sum(range(1, WORLD_SIZE + 1)))
    for output in outputs:
        if output.shape != expected.shape or output.dtype != expected.dtype or output.device != expected.device:
            raise RuntimeError("All-reduce output shape, dtype or device differs from the input")
        if not torch.equal(output, expected):
            raise RuntimeError("All-reduce sum validation failed; expected rank values 1+2+3+4=10")


def _benchmark(torch, dist, group, control, workspaces, args, rank, tokens, mode):
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
    from sglang.srt.distributed.parallel_state import graph_capture

    inputs = _stage(
        dist,
        control,
        f"{tokens}/{mode} allocation",
        lambda: [
            torch.full((tokens, HIDDEN_SIZE), rank + 1, dtype=torch.bfloat16, device=group.device)
            for _ in range(REPEAT)
        ],
    )
    backend = workspaces.get(group._fi_workspace_hint, {}).get("backend")

    def prepare():
        if mode == "graph":
            with graph_capture() as context:
                # Warm up on the same side stream before capturing. Some native
                # custom communicators return dummy outputs here; validate only
                # after capture has registered its buffers and the graph replays.
                for inp in inputs:
                    tensor_model_parallel_all_reduce(inp)
                for inp in inputs:
                    inp.fill_(rank + 1)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with _trace_dispatch(group, dist, backend) as calls, torch.cuda.graph(graph, stream=context.stream):
                    outputs = [tensor_model_parallel_all_reduce(inp) for inp in inputs]
            graph.replay()
            replay = graph.replay
        else:
            with _trace_dispatch(group, dist, backend) as calls:
                outputs = [tensor_model_parallel_all_reduce(inp) for inp in inputs]

            def replay():
                for inp in inputs:
                    tensor_model_parallel_all_reduce(inp)

        torch.cuda.synchronize()
        _validate_outputs(torch, inputs, outputs)
        if not calls:
            raise RuntimeError("No actual all-reduce implementation was observed")
        # In-place fallbacks would otherwise repeatedly multiply their inputs
        # until they overflow. Zero is stable under repeated reduction; resets,
        # correctness checks and allocations are outside the timed region.
        for inp in inputs:
            inp.zero_()
        torch.cuda.synchronize()
        return replay, "|".join(sorted(set(calls))) + f"_{mode}", outputs

    # Keep graph output allocations alive throughout replay, including native
    # out-of-place all-reduce buffers held in the graph's memory pool.
    replay, kernel_source, _outputs = _stage(dist, control, f"{tokens}/{mode} sum validation", prepare)
    sources = _gather(dist, control, kernel_source)
    if len(set(sources)) != 1:
        raise RuntimeError(f"All-reduce dispatch differs between ranks: {sources}")

    def warmup():
        for _ in range(args.warmup):
            replay()
        torch.cuda.synchronize()

    _stage(dist, control, f"{tokens}/{mode} warmup", warmup)
    dist.monitored_barrier(group=control, timeout=timedelta(seconds=args.timeout), wait_all_ranks=True)

    def time_replays():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            replay()
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / (args.iterations * REPEAT)
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError(f"Invalid measured latency: {latency}")
        return latency

    latency = _stage(dist, control, f"{tokens}/{mode} timing", time_replays)
    latencies = _gather(dist, control, latency)
    return {
        "allreduce_dtype": "bfloat16",
        "num_gpus": WORLD_SIZE,
        "message_size": tokens * HIDDEN_SIZE,
        "latency": max(latencies),
        "backend": f"sglang_{mode}",
        "kernel_source": kernel_source,
        "rank_latencies_ms": latencies,
    }


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _publish(args, inventory, reports, rows):
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta

    helper = _shared_helper()
    perf_path = args.output_dir / PerfFile.CUSTOM_ALLREDUCE.value
    module_hash = collector_hash(MODULE, _PACKAGE_ROOT, load_closures(_PACKAGE_ROOT / "collector/hash_closures.yaml"))
    runtime_meta = {
        "framework": "sglang",
        "version": SGLANG_DISTRIBUTION_VERSION,
        "image": args.launcher_image,
        "source_commit": SGLANG_COMMIT,
        "abi": {
            "torch": inventory["observed"]["cuda"]["torch_version"],
            "cuda": inventory["observed"]["cuda"]["torch_cuda_version"],
        },
        "transport": {"scope": "single_node", "tensor_parallel_size": WORLD_SIZE, "expert_parallel_size": 1},
        "backend_capability": {"operation": "plain_all_reduce", "fused_residual_rmsnorm_measured": False},
    }
    _write_json(
        args.output_dir / "all_reduce_observed.json",
        {
            "inventory": inventory,
            "ranks": reports,
            "measurements": rows,
            "timing": {
                "iterations": args.iterations,
                "warmup": args.warmup,
                "repeat": REPEAT,
                "aggregation": "max_rank",
            },
            "source_commit": SGLANG_COMMIT,
            "collector_hash": module_hash,
            "image_identity_verified": False,
            "operation_boundary": {
                "measured": "tensor_model_parallel_all_reduce on contiguous BF16 hidden states",
                "excluded": ["residual addition", "RMSNorm", "expert all-to-all"],
                "prediction_limit": (
                    "The SDK models add/norm separately; their sum with plain all-reduce "
                    "does not measure fusion savings."
                ),
            },
        },
    )
    for row in rows:
        helper.log_perf(
            item_list=[
                {key: row[key] for key in ("allreduce_dtype", "num_gpus", "message_size", "latency", "backend")}
            ],
            framework="SGLang",
            version=SGLANG_DISTRIBUTION_VERSION,
            device_name=reports[0]["device"]["name"],
            op_name="all_reduce",
            kernel_source=row["kernel_source"],
            perf_filename=str(perf_path),
        )
    finalized = helper.finalize_perf_files([perf_path], merge_existing=False)
    expected = perf_path.with_suffix(".parquet")
    if finalized != [expected]:
        raise RuntimeError(f"Expected finalized table {expected}, observed {finalized}")
    # Fresh-only output has no previous sidecar that could falsely attest a
    # partial publication. A crash before this write leaves an unqualified table.
    write_collection_meta(
        args.output_dir,
        runtime_meta,
        {
            expected.stem: {
                "collector_ref": MODULE,
                "collector_hash": module_hash,
                "case_plan_hash": case_plan_hash(
                    [f"tp4/bfloat16/{row['message_size']}/{row['backend']}" for row in rows]
                ),
                "collected_at": datetime.now(UTC).date().isoformat(),
                "rows": len(rows),
                "status": "complete",
            }
        },
    )


def run(args, rank):
    import torch
    import torch.distributed as dist

    # model_runner.py:385 selects the rank's device before distributed setup.
    torch.cuda.set_device(rank)
    inventory = collect_inventory(checkpoint_dir=args.checkpoint_dir, launcher_image=args.launcher_image)
    errors = _preflight_errors(inventory)
    if errors:
        raise RuntimeError("Runtime preflight failed: " + "; ".join(errors))
    if os.environ.get("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "0").lower() not in ("0", "false"):
        raise RuntimeError("This collector requires all four GPUs visible to each rank")
    before = _memory(torch, rank)
    group, server_args = _initialize(torch, args, rank)
    control = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo", timeout=timedelta(seconds=args.timeout))
    created_output = False
    try:

        def prepare_output():
            nonlocal created_output
            if rank == 0:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                created_output = True

        _stage(dist, control, "output setup", prepare_output)
        workspaces = _stage(
            dist, control, "serving workspace setup", lambda: _initialize_workspaces(server_args, torch)
        )
        communication = {
            "server_args": {name: getattr(server_args, name) for name in _COMM_ARGS},
            "workspaces": workspaces,
            "communicators": {
                name: None
                if (comm := getattr(group, name)) is None
                else {"class": f"{type(comm).__module__}.{type(comm).__qualname__}", "disabled": comm.disabled}
                for name in ("ca_comm", "pynccl_comm", "pymscclpp_comm", "torch_symm_mem_comm")
            },
        }
        report = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "device": inventory["observed"]["cuda"]["devices"][rank],
            "communication": communication,
            "memory_before_distributed": before,
            "memory_after_workspaces": _memory(torch, rank),
        }
        reports = _gather(dist, control, report)
        _validate_rank_reports(reports)
        rows = []
        for tokens in args.tokens:
            for mode in args.modes:
                row = _benchmark(torch, dist, group, control, workspaces, args, rank, tokens, mode)
                rows.append(row)
                if rank == 0:
                    print(
                        f"TP4 {row['backend']} elements={row['message_size']} latency={row['latency']:.6f} ms",
                        flush=True,
                    )
        _stage(dist, control, "publication", lambda: _publish(args, inventory, reports, rows) if rank == 0 else None)
    except Exception as error:
        if created_output:
            _write_json(
                args.output_dir / "errors_all_reduce.json",
                [
                    {
                        "classification": "unexpected",
                        "exception_type": type(error).__name__,
                        "error_message": str(error),
                        "case_parameters": {"tokens": args.tokens, "modes": args.modes, "num_gpus": WORLD_SIZE},
                    }
                ],
            )
        raise
    finally:
        # Do not enter another collective on a failed rank. torchrun terminates
        # its peers when this process exits nonzero. Successful ranks have already
        # synchronized CUDA and agreed on publication through the control group.
        dist.destroy_process_group(control)


def main(argv=None):
    args = _parse_args(argv)
    rank = _rank_from_environment(os.environ)
    run(args, rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
