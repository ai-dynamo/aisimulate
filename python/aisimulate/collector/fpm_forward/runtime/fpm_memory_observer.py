# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe initialized vLLM memory without changing execution or profiling.

Original collector adapters against vLLM 0.27.0, commit
4bdc8a788d2e2ce9165d552b3d4d8b72604626bf: vllm/v1/worker/gpu_worker.py
(determine_available_memory, initialize_from_config, compile_or_warm_up_model),
vllm/v1/worker/gpu_model_runner.py (initialize_kv_cache_tensors), and
vllm/v1/core/{kv_cache_manager,block_pool}.py. Cache specs are read, not rebuilt.
https://github.com/vllm-project/vllm/tree/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1

The same reads were audited against vLLM 0.32.0, commit
5df63a7203cdb9328d824a27b682e77c7b9928be, and 0.33.0, commit
4ba787ec9959e566d3b587a796dff07673324da5. Lines are given as 0.32.0/0.33.0:
- vllm/v1/worker/gpu_worker.py: determine_available_memory 634/641,
  initialize_from_config 853/866, compile_or_warm_up_model 908/933, and the
  runner choice in _make_model_runner 563-579/570-586.
- vllm/v1/worker/gpu/model_runner.py (Model Runner V2): initialize_kv_cache
  605/623, kv_caches 791-793/812-814.
- vllm/v1/worker/gpu_model_runner.py (Model Runner V1):
  initialize_kv_cache_tensors 6979/6914, initialize_kv_cache 7055/6990.
- vllm/v1/worker/utils.py (identical at both tags): allocate_kv_cache 564-627,
  bind_kv_cache_to_layers 868-885.
- vllm/v1/worker/gpu/attn_utils.py: init_kv_cache 396-443/397-444.
- vllm/v1/kv_cache_interface.py: KVCacheTensor 1409-1429/1475-1495,
  KVCacheGroupSpec 1439-1456/1505-1522, create_kv_cache_views 377-441/418-482.
- Layout: vllm/v1/core/kv_cache_utils.py:1588-1687 at 0.32.0 and
  vllm/v1/core/kv_cache_planning.py:642-741 at 0.33.0.
- vllm/v1/core/kv_cache_manager.py: block_pool 197/199, watermark_blocks 208/210.
- vllm/v1/core/block_pool.py: null_block 187-188/191-192, get_num_free_blocks 868/912.
https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1
https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1

A final patch release of 0.32 or 0.33, such as 0.33.1, is accepted only when every file in
SOURCE_MANIFESTS for its audited release (0.32.0 or 0.33.0) is installed once and byte-identical
to that release. Its records carry audited_backend_version. Any other version is refused.

Unsupported observation APIs write unresolved evidence. Exceptions from the
runtime itself are never caught by these observation helpers or wrappers.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import re
import socket
import threading
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_NAME = "aisimulate_fpm_runtime_memory"
SCHEMA_VERSION = 1
MEMORY_SUPPORTED_VERSIONS = ("0.27.0", "0.32.0", "0.33.0")
EXECUTION_SUPPORTED_VERSIONS = ("0.27.0", "0.28.0", "0.32.0", "0.33.0")
# Audited release to {path relative to the installation root: SHA-256}. Made by vllm_source_manifest.py.
SOURCE_MANIFESTS: dict[str, dict[str, str]] = {
    "0.32.0": {
        "vllm/compilation/decorators.py": "605fb8b1df7503dc17743e336d1a398a79c698fd05513d4bb2e462b3a1aa9fd7",
        "vllm/config/cache.py": "d37db21e7a6a1b477a580b54d90011d8708c4e7f6e3903221432d9a39830559b",
        "vllm/config/compilation.py": "d8a8086166e3f344b75d98d8350a77592f7b6c145319b3a9f0b9c8cb36bc39cb",
        "vllm/config/kernel.py": "dedb58b73dd702834d16f9183845ab40098c1f43a13ec70e9063d745e54a12fe",
        "vllm/config/model.py": "04ca37ea1d50128a7ddede30e7baa4e0a56bccec85091577a2c81c5de1a8fab8",
        "vllm/config/offload.py": "1dbcc23fedee34e09fe275b5889c43d77bbbafed7b36d097a620fb3e905c7aab",
        "vllm/config/parallel.py": "e8dcfb67c1ace957dcc7e12569ee62eb0f983aec3901fdef27c13014cea4b760",
        "vllm/config/scheduler.py": "2fb86b1572a4925be59f304d13b4f6ec03d496b67a7c4abbb976ecda2232839a",
        "vllm/config/vllm.py": "573692a80364d5c373321754c6213b4792ae292ba6aea7d6573ae013c2c71ae9",
        "vllm/distributed/__init__.py": "97f28c30b32662af8cab163939e4ee423037f11630752e13b55cbc25a7d0ab71",
        "vllm/distributed/parallel_state.py": "0e47cff9a89717183bdce6403f0454961848247f51306a0f1dfbe3d6e1b93336",
        "vllm/engine/arg_utils.py": "c10281277fd9a90dc8b22670db4f1af53708c5caa2aefee7aa861fa579329219",
        "vllm/model_executor/layers/attention/attention.py": "bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2",  # noqa: E501
        "vllm/model_executor/layers/attention_layer_base.py": "402470d2b7312988402971cee9c56b5816cf2d6c087c62d282feefca8384accc",  # noqa: E501
        "vllm/model_executor/layers/quantization/__init__.py": "d7656bbd3c34acc5df276cb99c72e96783c343c2751f7650107f035afbe8c295",  # noqa: E501
        "vllm/model_executor/layers/quantization/fp8.py": "940aed86660227851450da2ea564f81bc2db5e7f3de1ae2c99b4f19828c6ae1d",  # noqa: E501
        "vllm/model_executor/layers/quantization/modelopt.py": "4c47880b069b517e10d0b6d7167d98bcf8ff6f52e15cd8e1c973490fb2fefb10",  # noqa: E501
        "vllm/model_executor/offloader/__init__.py": "63775e5fd5fec29dee22f3b47f9fa861bde06a67dc8687ebe813046925fcbd15",
        "vllm/model_executor/offloader/base.py": "982e295299e2a282703ef6bec91897883f187f1ca31c21b6b46bda9566e09303",
        "vllm/models/inkling/nvidia/sconv_swa_attn.py": "7876005c24fb4753f549d332bc95c48419e15b1bda4e0f1a6aa3abf5703f0142",  # noqa: E501
        "vllm/platforms/cuda.py": "18ae1dfc34848afcade72821892db50fd3a2e3a57b48829521cc263e3f877caa",
        "vllm/v1/core/block_pool.py": "6d1e0d89b43e3b2ae396906b9c2f0551dbe192b60d05f8c1ca2d93a91bbbeea3",
        "vllm/v1/core/kv_cache_coordinator.py": "efd7193391197c555ebc0c8f2b593a6c4cb5b7da2978cd57414c0b71170f5971",
        "vllm/v1/core/kv_cache_manager.py": "d08315437c6b07040d1b7e12d7bd7d6e966c976a286f05a9788098cfdd9754c7",
        "vllm/v1/core/kv_cache_utils.py": "af4164bca819f1cdc416077522a7019936fd38a523a482283eb91b08dae0e57a",
        "vllm/v1/core/sched/scheduler.py": "d9a39870286c424cef2fc0730e24092ee3df67e848608e2278944846bab804ae",
        "vllm/v1/core/single_type_kv_cache_manager.py": "fe38816a388bb5b615231f621adc208e3893d20349865d041cc4f646d4d89b0a",  # noqa: E501
        "vllm/v1/engine/core.py": "458fd729771681b124d3a4d5b1c575f689bf789a5e0c8cc9f5c524eb6ec84f87",
        "vllm/v1/kv_cache_interface.py": "39ea5be3054f97238f81ba070ea661f1b5975a6268ced5f413d4a057ea449d4a",
        "vllm/v1/worker/gpu/attn_utils.py": "4ff080ab9c815037b53daceae4bd4654b5e9077b02e534bfb86b40f2a0edbe82",
        "vllm/v1/worker/gpu/cudagraph_utils.py": "ef2aae2eb3ec5ea97d4732f65c886b7a4688247721aacf23e536485e0ffc05ab",
        "vllm/v1/worker/gpu/model_runner.py": "5240fe8d62d403e466a4aa07a28fb7af5c0f1e9b9c3bb08c4003185bde461f87",
        "vllm/v1/worker/gpu/model_states/encoder_only.py": "6b5755a31a0457fd7dc26a62e827df64d87c5b02476dabc0794c9aedd4996a3c",  # noqa: E501
        "vllm/v1/worker/gpu_model_runner.py": "4cf95efe4ea8caa3110dcd231ff95ceb97e8136f5056e6bdd388300a484c4a81",
        "vllm/v1/worker/gpu_worker.py": "e970aa29a738aa15cd963d9f9202ca6d3d504e98fd344959fd998e940472a5a0",
        "vllm/v1/worker/utils.py": "85b23497017e5db07ea8f3fa56807c44dd7dd269f1977031bd86acdadfb0b5ec",
    },
    "0.33.0": {
        "vllm/compilation/decorators.py": "cec7c84bb2d4c36349a25048ee546fb6f8104215ad8c72f263ff28537e19247e",
        "vllm/config/cache.py": "55f26e023dc568f510a5d256dab638d3ca38114c2f634ba68b558d8ac25f8385",
        "vllm/config/compilation.py": "83c6970be0b7306793e768ea8d871bfffa0553f84af4085fd6ad9915d8e55cda",
        "vllm/config/kernel.py": "106d4529f3d7be8de5ef9d63deb9b2c77d955cc429ec9c8c28b1b668cec20cc3",
        "vllm/config/model.py": "e35090bc42befb81217dbb42c1508531b5402a872f5b8a12f36ba772c604d654",
        "vllm/config/offload.py": "1dbcc23fedee34e09fe275b5889c43d77bbbafed7b36d097a620fb3e905c7aab",
        "vllm/config/parallel.py": "61f4e257fbe61145ddac89740952d0bcceed80f9a0fc48eb653e549d3738007a",
        "vllm/config/scheduler.py": "61485c7a1ef1c7c65339767f4b223f55de76d2ca152eb19df915c88f526be9de",
        "vllm/config/vllm.py": "892088ef8dc5c7fa9d50cdf083dd776fd45e8d148d53b4a3279b2e79abeab0ca",
        "vllm/distributed/__init__.py": "97f28c30b32662af8cab163939e4ee423037f11630752e13b55cbc25a7d0ab71",
        "vllm/distributed/parallel_state.py": "c40796553e6797ce3e10d116ad58a30ce52a95156f4c393a4d11758c15fd1f1f",
        "vllm/engine/arg_utils.py": "f97af20917395ae717976f6022a78dd996a63c564a31db98567dd979cf0e2aa8",
        "vllm/model_executor/layers/attention/attention.py": "3914b60f5407d193cfd1cbe75c6689a6da71f89e37d177a04bb0d95b3244ea44",  # noqa: E501
        "vllm/model_executor/layers/attention_layer_base.py": "402470d2b7312988402971cee9c56b5816cf2d6c087c62d282feefca8384accc",  # noqa: E501
        "vllm/model_executor/layers/quantization/__init__.py": "d7656bbd3c34acc5df276cb99c72e96783c343c2751f7650107f035afbe8c295",  # noqa: E501
        "vllm/model_executor/layers/quantization/fp8.py": "8bf381da9ac6a9db81c488d0347019cb05235a247b3c8cac94695d4a366ee691",  # noqa: E501
        "vllm/model_executor/layers/quantization/modelopt.py": "e76a666bdb653f86effc1094e9427e89a9acba99f0673f1c9e2f0a8784366b4f",  # noqa: E501
        "vllm/model_executor/offloader/__init__.py": "63775e5fd5fec29dee22f3b47f9fa861bde06a67dc8687ebe813046925fcbd15",
        "vllm/model_executor/offloader/base.py": "982e295299e2a282703ef6bec91897883f187f1ca31c21b6b46bda9566e09303",
        "vllm/models/inkling/nvidia/sconv_swa_attn.py": "7876005c24fb4753f549d332bc95c48419e15b1bda4e0f1a6aa3abf5703f0142",  # noqa: E501
        "vllm/platforms/cuda.py": "0cfa80cce99aae194689d822edef808e5c78454eec696fb1c2ded7d17dfe71bb",
        "vllm/platforms/interface.py": "11f06cfa184fa7380177eaf26bca58c421306f0eea273c9f35519d35ff4a6f75",
        "vllm/v1/core/block_pool.py": "64d304d2b431a6ce9a0a0425610122354e1ac11b6aff1ca7d467d4218183774c",
        "vllm/v1/core/kv_cache_coordinator.py": "6bf7583c5fd3fb85a99437eb3d7f3eddd319314ff93737e847d071fda59383e3",
        "vllm/v1/core/kv_cache_manager.py": "39ae1c1564a034fd59736a92d962bfab5a0af03529e01349b765422dbd5a355d",
        "vllm/v1/core/kv_cache_planning.py": "b3c200004a57b1da274127ad267af2c3e6349b94c7543e4ef5a39cfddc74f722",
        "vllm/v1/core/kv_cache_utils.py": "de64494f8316d3e706802d1461cbf17ddbd421bb4c18946c085145aefd9846ea",
        "vllm/v1/core/sched/scheduler.py": "d76cc22257550972137df9c893505d0bd0f5d482374691e6eca4372772de501a",
        "vllm/v1/core/single_type_kv_cache_manager.py": "4b8c34ff759855b94a5a3a4a3a1b4784077f27077539e01509c1b3f72e75455f",  # noqa: E501
        "vllm/v1/engine/core.py": "d02d32df5d15f0a8b8ffc731ed4bc6f7fcb68b8a9273ec5e4c5d1ab4bd7d7e01",
        "vllm/v1/kv_cache_interface.py": "1e0b5027cc13ed845488ff3f671292e6c57fcae99c5b7130ad92c9d176a38684",
        "vllm/v1/worker/gpu/attn_utils.py": "705a1eb16d26dc0ec658af9e10e00831f16d1925c14361228853807e456f6553",
        "vllm/v1/worker/gpu/cudagraph_utils.py": "1ff1655db4ca85ad2efcd8de9b3de37f053eb68d1d22202d45301c11866b0d40",
        "vllm/v1/worker/gpu/model_runner.py": "9145e2601f9e93db5e6c7e648c6741cff55f2a49f046800f79311e4977550121",
        "vllm/v1/worker/gpu/model_states/encoder_only.py": "f592382117198e70b23974c362e688221134083718d5e271d9a4429f0c9636bf",  # noqa: E501
        "vllm/v1/worker/gpu_model_runner.py": "3ba41733f7062b9b70801025ee263979a0b11b8c0cea418d5d18caa26de6524d",
        "vllm/v1/worker/gpu_worker.py": "e46a2659842b5b0238ca5d13ecb4a2dcff91f028493a7117d666edb0785a742c",
        "vllm/v1/worker/utils.py": "85b23497017e5db07ea8f3fa56807c44dd7dd269f1977031bd86acdadfb0b5ec",
    },
}

_SOURCE_PROBLEMS: dict[str, str | None] = {}


def audited_release(version: str | None, audited: tuple[str, ...]) -> str | None:
    """Return the audited release whose source a version stands for, or None.

    An audited version stands for itself. A final release (digits and dots, at least three parts)
    stands for the newest audited release of its release line (first two parts) that has a source
    manifest. Pre-release, development, post-release and local builds are refused.
    """
    if not isinstance(version, str):
        return None
    if version in audited:
        return version
    if not re.fullmatch(r"\d+(?:\.\d+){2,}", version, re.ASCII):
        return None
    line = version.split(".")[:2]
    candidates = [release for release in audited if release in SOURCE_MANIFESTS and release.split(".")[:2] == line]
    return max(candidates, key=lambda release: tuple(int(part) for part in release.split(".")), default=None)


def _source_problem(release: str) -> str | None:
    """Compare every audited vLLM file of a release with the installed one, once per process."""
    if release not in _SOURCE_PROBLEMS:
        roots = [Path(root) for root in importlib.import_module("vllm").__path__]
        problem = None
        for path in sorted(SOURCE_MANIFESTS[release]):
            relative = path.split("/", 1)[1]
            found = [root / relative for root in roots if (root / relative).is_file()]
            if len(found) != 1:
                problem = f"{path} is not installed exactly once"
            elif hashlib.sha256(found[0].read_bytes()).hexdigest() != SOURCE_MANIFESTS[release][path]:
                problem = f"{path} differs"
            if problem:
                break
        _SOURCE_PROBLEMS[release] = problem
    return _SOURCE_PROBLEMS[release]


def _check_release(payload: dict[str, Any], audited: tuple[str, ...], refusal: str) -> None:
    """Refuse an unaudited vLLM; accept a patch release only when its audited files are unchanged."""
    version = payload["backend_version"]
    release = audited_release(version, audited)
    if release is None:
        raise ValueError(refusal)
    if release != version:
        payload["audited_backend_version"] = release
        problem = _source_problem(release)
        if problem:
            raise ValueError(f"installed vLLM {version} does not match audited vLLM {release}: {problem}")


RESULTS_DIR = Path("/results")
_SPEC_MODULE = "vllm.v1.kv_cache_interface."
_CONV_CLASS = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
_CONFIG_FIELDS = {
    "model_config": ("model", "revision", "dtype", "quantization", "max_model_len", "enforce_eager"),
    "cache_config": (
        "cache_dtype",
        "gpu_memory_utilization",
        "kv_cache_memory_bytes",
        "enable_prefix_caching",
        "num_gpu_blocks_override",
    ),
    "scheduler_config": ("max_num_batched_tokens", "max_num_seqs", "async_scheduling"),
    "parallel_config": (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_external_lb",
        "enable_expert_parallel",
        "enable_eplb",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ),
    "compilation_config": ("mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size"),
    "kernel_config": ("moe_backend",),
}


def cpu_affinity_snapshot(*, task_directory: Path = Path("/proc/self/task")) -> dict[str, Any]:
    """Observe Linux thread affinity once, without binding or running a model.

    Linux affinity is per thread. Querying PID explicitly observes the main
    thread even when a runtime hook executes on another Python/native thread.
    The task scan is not atomic; changing or unreadable threads stay partial.
    CPU IDs are logical CPUs, not a physical-core or NUMA placement claim.
    """
    pid = os.getpid()
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": pid,
        "observer_thread_id": threading.get_native_id(),
        "main_thread_allowed_cpus": [],
        "threads": [],
        "thread_ids_before": [],
        "thread_ids_after": [],
        "thread_errors": [],
        "errors": [],
        "status": "unavailable",
    }
    try:
        payload["main_thread_allowed_cpus"] = sorted(os.sched_getaffinity(pid))
        if not payload["main_thread_allowed_cpus"]:
            raise ValueError("main thread has an empty CPU affinity mask")
    except (AttributeError, OSError, ValueError) as error:
        payload["errors"].append(f"main thread affinity: {type(error).__name__}: {error}")
        return payload
    try:
        before = payload["thread_ids_before"] = sorted(
            int(path.name) for path in task_directory.iterdir() if path.name.isdecimal()
        )
        for tid in before:
            try:
                cpus = sorted(os.sched_getaffinity(tid))
                if not cpus:
                    raise ValueError("thread has an empty CPU affinity mask")
                payload["threads"].append({"tid": tid, "allowed_cpus": cpus})
            except (OSError, ValueError) as error:
                payload["thread_errors"].append({"tid": tid, "error": f"{type(error).__name__}: {error}"})
        after = payload["thread_ids_after"] = sorted(
            int(path.name) for path in task_directory.iterdir() if path.name.isdecimal()
        )
        if before != after:
            payload["errors"].append("thread list changed during observation")
        if payload["observer_thread_id"] not in before:
            payload["errors"].append("observation thread is missing from the task snapshot")
        main = [thread for thread in payload["threads"] if thread["tid"] == pid]
        if not main or main[0]["allowed_cpus"] != payload["main_thread_allowed_cpus"]:
            payload["errors"].append("main thread was missing or its affinity changed during observation")
    except OSError as error:
        payload["errors"].append(f"thread enumeration: {type(error).__name__}: {error}")
    payload["status"] = "partial" if payload["errors"] or payload["thread_errors"] else "observed"
    return payload


def observe_cpu(
    kind: str,
    *,
    dp_rank: int | None = None,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    requested_cpus_per_task: int | None = None,
    cpu_bind: str | None = None,
    local_gpu_count: int | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Save independent CPU evidence; unavailable observation never changes memory.

    The launcher uses the same container and srun step as the engine, before
    engine startup. Its caller must inspect this saved result before proceeding.
    This helper records affinity; it does not set it or claim it remains fixed.
    """
    measurements = {
        "launcher": "before_engine_start",
        "worker": "after_warmup",
        "scheduler": "scheduler_initialized",
    }
    if kind not in measurements:
        raise ValueError("CPU observation requires launcher, worker or scheduler role")
    ranks = {"dp_rank": dp_rank, "tp_rank": tp_rank, "pp_rank": pp_rank}
    required = set(ranks) if kind == "worker" else {"dp_rank"} if kind == "scheduler" else set()
    if any(
        (type(value) is not int or value < 0) if key in required else value is not None for key, value in ranks.items()
    ):
        raise ValueError("CPU observation has invalid role/rank coordinates")
    if directory is None:
        directory, binding = _execution_destination()
    else:
        source = directory / "collector-provenance.json"
        raw = source.read_bytes()
        provenance = json.loads(raw)
        if not isinstance(provenance, dict):
            raise ValueError("CPU observation collector provenance must be an object")
        binding = {
            "collector_provenance": provenance,
            "execution_provenance": provenance,
            "provenance_source": {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            },
        }
    payload = {
        "schema_name": "aisimulate_fpm_cpu_affinity",
        "schema_version": 1,
        "kind": kind,
        "measurement": measurements[kind],
        **ranks,
        **binding,
        **cpu_affinity_snapshot(),
    }
    if kind == "launcher":
        payload.update(
            requested_cpus_per_task=requested_cpus_per_task, cpu_bind=cpu_bind, local_gpu_count=local_gpu_count
        )
    suffix = (
        f"-dp{dp_rank}-tp{tp_rank}-pp{pp_rank}" if kind == "worker" else f"-dp{dp_rank}" if kind == "scheduler" else ""
    )
    path = directory / f"fpm-cpu-{kind}{suffix}.json"
    try:
        _write_execution_file(path, payload)
    except FileExistsError:
        raise RuntimeError(f"refusing duplicate runtime CPU observation: {path}") from None
    return payload


def _class_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def compilation_config(config: Any) -> dict[str, Any]:
    """Snapshot the graph settings before the worker resolves backend support."""
    compilation = config.compilation_config
    return {key: _json_value(getattr(compilation, key)) for key in _CONFIG_FIELDS["compilation_config"]}


def _write_execution_file(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish atomically without replacing an earlier observation, including
        # when two worker processes accidentally report the same rank.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execution_destination() -> tuple[Path, dict[str, Any]]:
    output_value = os.environ.get("FPM_EXECUTION_OUTPUT_DIR")
    provenance_value = os.environ.get("FPM_EXECUTION_PROVENANCE_FILE")
    custom = output_value is not None or provenance_value is not None
    if custom:
        if not output_value or not provenance_value:
            raise ValueError("FPM_EXECUTION_OUTPUT_DIR and FPM_EXECUTION_PROVENANCE_FILE must be supplied together")
        for name, value in (("output directory", output_value), ("provenance file", provenance_value)):
            if not Path(value).is_absolute() or any(char in value for char in ("\n", "\r", "\x00")):
                raise ValueError(f"execution observation {name} must be an absolute path without control characters")
        directory, source = Path(output_value), Path(provenance_value)
    else:
        directory, source = RESULTS_DIR, RESULTS_DIR / "collector-provenance.json"
    raw = source.read_bytes()
    provenance = json.loads(raw)
    if not isinstance(provenance, dict):
        raise ValueError("execution provenance must be a JSON object")
    binding = {
        "execution_provenance": provenance,
        "provenance_source": {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)},
    }
    if not custom:
        binding["collector_provenance"] = provenance
        return directory, binding
    if (
        provenance.get("purpose") != "matched_serving_accuracy"
        or not isinstance(provenance.get("run_id"), str)
        or not provenance["run_id"].strip()
        or "collector_attempt_id" in provenance
        or provenance.get("schema_name") == "aic_fpm_collector_provenance"
    ):
        raise ValueError("custom execution provenance requires a serving run_id and purpose=matched_serving_accuracy")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".execution-run.json"
    if (directory / "collector-provenance.json").exists() or (
        not marker.exists() and any(directory.glob("fpm-execution-worker-*.json"))
    ):
        raise ValueError("serving execution output requires a fresh directory, separate from collector attempts")
    try:
        _write_execution_file(marker, binding)
    except FileExistsError:
        if json.loads(marker.read_bytes()) != binding:
            raise ValueError("execution observation directory already belongs to a different serving run") from None
    return directory, binding


def observe_execution(worker: Any, *, dp_rank: int, tp_rank: int, pp_rank: int) -> None:
    """Record selected backend objects after initialization, outside timed forwards.

    This original adapter reads AttentionGroup.backend/layer_names populated by
    GPUModelRunner.initialize_attn_backend, including runtime backend wrappers:
    https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/gpu_model_runner.py#L7020
    https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/worker/utils.py#L217
    The same read-only interfaces were verified for vLLM 0.28.0 at:
    https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/gpu_model_runner.py#L7155
    https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/utils.py#L241
    They were also verified for vLLM 0.32.0, commit 5df63a7203cdb9328d824a27b682e77c7b9928be,
    and 0.33.0, commit 4ba787ec9959e566d3b587a796dff07673324da5. AttentionGroup keeps these
    fields (vllm/v1/worker/utils.py:261-266, identical at both tags). Model Runner V1 fills
    attn_groups in initialize_attn_backend (vllm/v1/worker/gpu_model_runner.py 0.32.0:6688,
    0.33.0:6623). Model Runner V2 assigns them from init_attn_backend
    (vllm/v1/worker/gpu/model_runner.py 0.32.0:665-674, 0.33.0:689-698;
    vllm/v1/worker/gpu/attn_utils.py 0.32.0:196, 0.33.0:197), which appends an empty list
    for a cache group without layer views (attn_utils.py 0.32.0:220-222, 0.33.0:221-223).
    An empty list adds no group.
    It does not infer a backend from requested flags or trace per-point dispatch.

    Ordinary serving can use FpmExecutionWorker with absolute
    FPM_EXECUTION_OUTPUT_DIR and FPM_EXECUTION_PROVENANCE_FILE paths. Both are
    required together; the provenance identifies a fresh matched-serving run.
    Collector launches retain their default /results output and provenance.
    """
    directory, binding = _execution_destination()
    payload = {
        "schema_name": "aisimulate_fpm_runtime_execution",
        "schema_version": 1,
        **binding,
        "backend_version": importlib.metadata.version("vllm"),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "per_point_dispatch": "unreported",
    }
    try:
        _check_release(
            payload,
            EXECUTION_SUPPORTED_VERSIONS,
            f"execution observation requires audited vLLM {EXECUTION_SUPPORTED_VERSIONS}",
        )
        groups = []
        for cache_groups in worker.model_runner.attn_groups:
            for group in cache_groups:
                names = list(group.layer_names)
                if not names or any(not isinstance(name, str) or not name for name in names):
                    raise ValueError("runtime attention group has no explicit layer names")
                backend = group.backend
                if not isinstance(backend, type):
                    raise ValueError("runtime attention group backend is not a class")
                groups.append(
                    {
                        "kv_cache_group_id": group.kv_cache_group_id,
                        "backend_class": f"{backend.__module__}.{backend.__qualname__}",
                        "layer_names": names,
                    }
                )
        if not groups:
            raise ValueError("runtime reported no initialized attention groups")
        payload.update(
            attention_groups=groups,
            graph_config=compilation_config(worker.vllm_config),
            resolved_config=execution_config(worker.vllm_config),
            status="observed",
        )
    except Exception as error:
        payload.update(status="unresolved", error=f"{type(error).__name__}: {error}")
        logging.getLogger(__name__).warning("FPM execution evidence is incomplete: %s", payload["error"])
    path = directory / f"fpm-execution-worker-dp{dp_rank}-tp{tp_rank}-pp{pp_rank}.json"
    if path.exists():
        raise RuntimeError(f"refusing duplicate runtime execution observation: {path}")
    try:
        _write_execution_file(path, payload)
    except FileExistsError:
        raise RuntimeError(f"refusing duplicate runtime execution observation: {path}") from None


def _cached_config_commit(model_config: Any) -> str | None:
    """Return the Hub snapshot commit in the local path of the loaded config.json, else None.

    File-system lookups only; nothing here may reach the network.
    """
    try:
        source = getattr(model_config, "hf_config_path", None) or model_config.model
        if os.path.isdir(source):
            resolved = os.path.join(source, "config.json")
            if not os.path.isfile(resolved):
                return None
        else:
            from huggingface_hub import try_to_load_from_cache

            # Only a pinned commit is evidence: the current cache ref is not proof of the loaded commit
            # (REVIEW.md lines 131-133). vLLM's ResolvedRevision.resolved holds it (huggingface_hub/_revision.py;
            # vLLM v0.32.0 config/model.py lines 632-641, 672).
            effective = getattr(model_config, "_hf_config_revision", None) or model_config.revision
            pinned = getattr(effective, "resolved", None)
            if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned)):
                pinned = effective
            if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned)):
                return None
            revision = str(pinned)
            resolved = try_to_load_from_cache(source, "config.json", revision=revision)
        if not isinstance(resolved, str):
            return None
        # Same rule as transformers' extract_commit_hash: first snapshots/<segment>/, lowercase commit.
        match = re.search(r"snapshots/([^/]+)/", resolved.replace("\\", "/"))
        if match and re.fullmatch(r"[0-9a-f]{40}", match.group(1)):
            return match.group(1)
    except Exception as error:
        logging.getLogger(__name__).debug("loaded config commit lookup failed: %s", type(error).__name__)
    return None


def execution_config(config: Any) -> dict[str, Any]:
    """Read execution identity without requiring the memory-finalization APIs."""
    result = {
        section: {
            key: _json_value(getattr(getattr(config, section), key))
            for key in fields
            if hasattr(getattr(config, section), key)
        }
        for section, fields in _CONFIG_FIELDS.items()
        if getattr(config, section, None) is not None
    }
    hf_config = getattr(config.model_config, "hf_config", None)
    if hf_config is not None and hasattr(hf_config, "_commit_hash"):
        if hf_config._commit_hash is not None:
            result["model_config"]["loaded_config_commit_hash"] = hf_config._commit_hash
    elif hf_config is not None:
        # transformers 5.19.0 (Dynamo 1.7.0-rc.1 image, configuration_utils.py line 366) discards
        # `_commit_hash`; 5.17.0 and earlier keep it. Read the commit from the local cache path instead.
        commit = _cached_config_commit(config.model_config)
        if commit is not None:
            result["model_config"]["loaded_config_commit_hash"] = commit
    quant = config.quant_config
    result["quantization_config"] = (
        {
            "type": _class_name(quant),
            **{
                key: _json_value(getattr(quant, key))
                for key in ("quant_method", "activation_scheme", "weight_block_size")
                if hasattr(quant, key)
            },
        }
        if quant is not None
        else None
    )
    return result


def resolved_config(config: Any) -> dict[str, Any]:
    result = execution_config(config)
    # vLLM 0.27.0 config/offload.py owns weight offload; cache_config does not.
    offload = config.offload_config
    result["offload_config"] = {
        "offload_backend": offload.offload_backend,
        "uva": {"cpu_offload_gb": offload.uva.cpu_offload_gb},
        "prefetch": {
            key: getattr(offload.prefetch, key)
            for key in ("offload_group_size", "offload_num_in_group", "offload_prefetch_step")
        },
    }
    return result


def _offloader_type() -> str:
    # vLLM 0.27.0 model_executor/offloader/base.py:get_offloader is read-only.
    from vllm.model_executor.offloader import get_offloader

    return _class_name(get_offloader())


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def cache_groups(cache_config: Any) -> list[dict[str, Any]]:
    groups = []
    all_names: set[str] = set()
    for group in cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        spec_type = _class_name(spec)
        if spec_type not in {_SPEC_MODULE + "FullAttentionSpec", _SPEC_MODULE + "SlidingWindowSpec"}:
            raise ValueError(f"unsupported runtime cache spec: {spec_type}")
        if getattr(group, "is_eagle_group", False):
            raise ValueError("speculative cache groups are unsupported")
        # vLLM 0.32.0 KVCacheGroupSpec adds host_resident and role (v1/kv_cache_interface.py:1451, :1456;
        # 0.33.0 :1517, :1522). A missing field means the 0.27.0 group: device-resident, role DEFAULT.
        if getattr(group, "host_resident", False):
            raise ValueError("host-resident cache groups are unsupported")
        role = _json_value(getattr(group, "role", "DEFAULT"))
        if role != "DEFAULT":
            raise ValueError(f"unsupported cache group role: {role}")
        names = list(group.layer_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("runtime cache groups require explicit layer names")
        if len(names) != len(set(names)) or all_names.intersection(names):
            raise ValueError("runtime cache group layer aliases are unsupported")
        all_names.update(names)
        window = getattr(spec, "sliding_window", None)
        if spec_type.endswith(".SlidingWindowSpec"):
            _positive(window, "sliding_window")
        elif window is not None or getattr(spec, "attention_chunk_size", None) is not None:
            raise ValueError("full-attention cache with alternate retention is unsupported")
        groups.append(
            {
                "layer_names": names,
                "spec_type": spec_type,
                "block_size_tokens": _positive(spec.block_size, "cache block size"),
                "spec_page_size_bytes": _positive(spec.page_size_bytes, "cache spec page size"),
                "sliding_window": window,
                "dtype": str(spec.dtype),
            }
        )
    if not groups:
        raise ValueError("runtime emitted no cache groups")
    return groups


def _storage(tensor: Any) -> tuple[tuple[str, int], int]:
    storage = tensor.untyped_storage()
    device = str(storage.device)
    if not device.startswith("cuda:"):
        raise ValueError(f"cache storage is not on a CUDA device: {device}")
    return (device, _positive(storage.data_ptr(), "storage pointer")), _positive(storage.nbytes(), "storage bytes")


def _layer_storages(kv_cache: Any) -> set[tuple[str, int]]:
    """Storage keys of one cache tensor, or of every tensor of a tuple or list.

    A vLLM 0.31.0 Mamba layer binds a tuple of state views of one page.
    """
    tensors = kv_cache if isinstance(kv_cache, (tuple, list)) else (kv_cache,)
    if not tensors:
        raise ValueError("cache layer binds no tensors")
    return {_storage(tensor)[0] for tensor in tensors}


def _tensor_storages(tensors: Any) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            key, size = _storage(value)
            if key in result and result[key] != size:
                raise ValueError("aliased cache storage reports inconsistent sizes")
            result[key] = size

    visit(tensors)
    if not result:
        raise ValueError("runtime emitted no physical cache storage")
    if len({device for device, _pointer in result}) != 1:
        raise ValueError("rank cache spans multiple devices")
    return result


def _shared_by_allocations(
    config: Any, layers: dict[str, Any], storages: dict[tuple[str, int], int]
) -> list[dict[str, Any]]:
    """Check a vLLM 0.27.0 or 0.28.0 plan, where ``shared_by`` names the aliases of each tensor."""
    layer_storages = {name: _storage(layer.kv_cache) for name, layer in layers.items()}
    if dict(layer_storages.values()) != storages:
        raise ValueError("cache group layers do not account for every model-runner storage")

    # Every observed alias must be declared by the runtime allocation plan.
    # Packed tensors explicitly share one backing allocation through block_stride.
    expected: dict[tuple[str, int], int] = {}
    covered: set[str] = set()
    packed_key = None
    allocations = []
    for tensor in config.kv_cache_tensors:
        names = list(tensor.shared_by)
        if not names or covered.intersection(names) or any(name not in layers for name in names):
            raise ValueError("runtime tensor allocation has missing or duplicate cache layers")
        keys = {layer_storages[name][0] for name in names}
        if len(keys) != 1:
            raise ValueError("runtime-declared shared tensor uses different physical storages")
        key = next(iter(keys))
        size = _positive(tensor.size, "configured cache tensor bytes")
        stride = getattr(tensor, "block_stride", 0)
        if type(stride) is not int or stride < 0:
            raise ValueError("invalid packed cache block stride")
        if stride:
            if packed_key is None:
                if key in expected:
                    raise ValueError("packed cache aliases an unpacked allocation")
                packed_key = key
            elif key != packed_key:
                raise ValueError("packed cache tensors use multiple backing allocations")
        elif key in expected:
            raise ValueError("undeclared cache tensor alias")
        if key in expected and expected[key] != size:
            raise ValueError("packed allocation sizes disagree")
        expected[key] = size
        covered.update(names)
        allocations.append(
            {"size": size, "shared_by": names, "block_stride": stride, "offset": getattr(tensor, "offset", 0)}
        )
    if covered != set(layers) or expected != storages:
        raise ValueError("runtime allocation plan disagrees with physical cache storage")
    return allocations


def _layer_strided_allocations(
    config: Any, groups: list[dict[str, Any]], layers: dict[str, Any], storages: dict[tuple[str, int], int]
) -> list[dict[str, Any]]:
    """Check a vLLM 0.32.0 or 0.33.0 plan, where every tensor places its ``layers`` in one buffer.

    The page of layer l for block b starts at offset + l * layer_stride + b * block_stride
    (v1/kv_cache_interface.py 0.32.0:1413-1415, 0.33.0:1479-1481), so the last page of the last
    layer bounds the region in layer-outer and block-outer layouts. A layer's page is the spec
    page of its group. Tensors whose regions overlap alias the same bytes: cache groups overlay
    each other (0.32.0:1419-1421, 0.33.0:1485-1487).
    """
    num_blocks = _positive(config.num_blocks, "cache block count")
    layer_storages = {name: _layer_storages(layer.kv_cache) for name, layer in layers.items()}
    if set().union(*layer_storages.values()) != set(storages):
        raise ValueError("cache group layers do not account for every model-runner storage")
    pages = {name: group["spec_page_size_bytes"] for group in groups for name in group["layer_names"]}
    expected: dict[tuple[str, int], int] = {}
    covered: set[str] = set()
    allocations = []
    for tensor in config.kv_cache_tensors:
        if getattr(tensor, "host_resident", False):
            raise ValueError("host-resident cache tensors are unsupported")
        names = list(tensor.layers)
        if not names or covered.intersection(names) or any(name not in layers for name in names):
            raise ValueError("runtime tensor allocation has missing or duplicate cache layers")
        keys = set().union(*(layer_storages[name] for name in names))
        if len(keys) != 1:
            raise ValueError("runtime-declared shared tensor uses different physical storages")
        key = next(iter(keys))
        if expected and key not in expected:
            raise ValueError("cache tensors with layer strides use multiple backing allocations")
        size = _positive(tensor.size, "configured cache tensor bytes")
        if key in expected and expected[key] != size:
            raise ValueError("packed allocation sizes disagree")
        layer_stride, block_stride, offset = (
            getattr(tensor, field, None) for field in ("layer_stride", "block_stride", "offset")
        )
        if any(type(value) is not int or value < 0 for value in (layer_stride, block_stride, offset)):
            raise ValueError("invalid cache tensor layer stride, block stride or offset")
        page = max(pages[name] for name in names)
        if page > block_stride:
            raise ValueError("cache tensor page exceeds its block stride")
        if offset + (len(names) - 1) * layer_stride + (num_blocks - 1) * block_stride + page > size:
            raise ValueError("cache tensor layers exceed their backing allocation")
        expected[key] = size
        covered.update(names)
        allocations.append(
            {
                "size": size,
                "layers": list(names),
                "layer_stride": layer_stride,
                "block_stride": block_stride,
                "offset": offset,
            }
        )
    if covered != set(layers) or expected != storages:
        raise ValueError("runtime allocation plan disagrees with physical cache storage")
    return allocations


def worker_memory(worker: Any) -> dict[str, Any]:
    if not getattr(worker, "_fpm_cache_initialized", False):
        raise ValueError("worker cache initialization was not observed")
    runner = worker.model_runner
    if getattr(runner, "shared_kv_cache_layers", None):
        raise ValueError("cross-layer KV sharing is not yet supported by memory finalization")
    config = runner.kv_cache_config
    groups = cache_groups(config)
    storages = _tensor_storages(runner.kv_caches)
    context = worker.vllm_config.compilation_config.static_forward_context
    layers = {name: context[name] for group in groups for name in group["layer_names"]}
    # The record shape selects the reader. vLLM 0.27.0 names the aliases of each tensor in shared_by
    # (v1/kv_cache_interface.py:952); 0.32.0 and 0.33.0 place layers by stride and have no shared_by
    # (0.32.0:1424-1429, 0.33.0:1490-1495). Every other shape stays unresolved.
    tensors = config.kv_cache_tensors
    if all(
        hasattr(item, "layers") and hasattr(item, "layer_stride") and not hasattr(item, "shared_by") for item in tensors
    ):
        allocations = _layer_strided_allocations(config, groups, layers, storages)
    elif all(hasattr(item, "shared_by") and not hasattr(item, "layers") for item in tensors):
        allocations = _shared_by_allocations(config, layers, storages)
    else:
        raise ValueError("unsupported runtime cache tensor record")
    for group in groups:
        classes = {name: _class_name(layers[name]) for name in group["layer_names"]}
        conv = {value == _CONV_CLASS for value in classes.values()}
        if len(conv) != 1:
            raise ValueError("mixed convolution and attention cache group is unsupported")
        group["kind"] = "convolution" if conv == {True} else "attention"
        group["layer_classes"] = classes
        if group["kind"] == "convolution" and group["sliding_window"] != 4:
            raise ValueError("unsupported Inkling convolution retention")
    diagnostics = {
        name: _json_value(getattr(worker, name))
        for name in (
            "available_kv_cache_memory_bytes",
            "peak_activation_memory",
            "non_torch_memory",
            "total_consumed",
            "requested_memory",
            "cudagraph_memory_estimate",
        )
        if hasattr(worker, name)
    }
    return {
        "num_blocks": _positive(config.num_blocks, "cache block count"),
        "available_cache_bytes": _positive(worker._fpm_available_cache_bytes, "profiled cache budget"),
        "allocated_cache_bytes": sum(storages.values()),
        "storages": [
            {"device": key[0], "pointer": key[1], "size_bytes": size} for key, size in sorted(storages.items())
        ],
        "tensor_allocations": allocations,
        "groups": groups,
        "diagnostics": diagnostics,
    }


def scheduler_memory(scheduler: Any, config: Any) -> dict[str, Any]:
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    managers = manager.coordinator.single_type_managers
    if not managers or any(item.block_pool is not pool for item in managers):
        raise ValueError("runtime cache groups must share one block pool")
    num_blocks = _positive(config.num_blocks, "scheduler cache block count")
    if pool.num_gpu_blocks != num_blocks:
        raise ValueError("scheduler and configured block pool sizes disagree")
    free = _positive(pool.get_num_free_blocks(), "initial free cache blocks")
    if free >= num_blocks or getattr(pool.null_block, "is_null", False) is not True:
        raise ValueError("scheduler null block reservation is not observable")
    if type(manager.watermark_blocks) is not int or manager.watermark_blocks != 0:
        raise ValueError("nonzero scheduler cache watermark is unsupported")
    return {
        "num_blocks": num_blocks,
        "initial_free_blocks": free,
        "reserved_blocks": num_blocks - free,
        "null_block_id": pool.null_block.block_id,
        "watermark_blocks": manager.watermark_blocks,
        "pool_count": 1,
        "groups": cache_groups(config),
    }


def observe(
    kind: str,
    owner: Any,
    *,
    dp_rank: int,
    tp_rank: int | None = None,
    pp_rank: int | None = None,
    cache_config: Any = None,
) -> None:
    """Write atomic evidence after successful initialization, never run a model."""
    provenance = json.loads((RESULTS_DIR / "collector-provenance.json").read_text())
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "collector_provenance": provenance,
        "backend_version": importlib.metadata.version("vllm"),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
    }
    try:
        _check_release(
            payload,
            MEMORY_SUPPORTED_VERSIONS,
            f"memory observation currently supports vLLM {', '.join(MEMORY_SUPPORTED_VERSIONS)}; "
            f"found {payload['backend_version']}",
        )
        if (
            getattr(owner.vllm_config, "speculative_config", None) is not None
            or getattr(owner.vllm_config, "kv_transfer_config", None) is not None
        ):
            raise ValueError("runtime memory observation supports non-speculative local cache only")
        payload["resolved_config"] = resolved_config(owner.vllm_config)
        offload = payload["resolved_config"]["offload_config"]
        if kind == "worker":
            payload["effective_offloader"] = _offloader_type()
            initial = getattr(owner, "_fpm_initial_compilation_config", None)
            if initial is not None:
                payload["initial_compilation_config"] = initial
        cache = owner.vllm_config.cache_config
        if (
            getattr(cache, "kv_cache_memory_bytes", None) is not None
            or getattr(cache, "num_gpu_blocks_override", None) is not None
            or offload["offload_backend"] != "auto"
            or offload["uva"]["cpu_offload_gb"] != 0
            or offload["prefetch"]["offload_group_size"] != 0
            or (
                kind == "worker"
                and payload["effective_offloader"] != "vllm.model_executor.offloader.base.NoopOffloader"
            )
        ):
            raise ValueError("manual KV allocation or offloaded cache is unsupported by runtime memory finalization")
        payload["cache"] = worker_memory(owner) if kind == "worker" else scheduler_memory(owner, cache_config)
        payload["status"] = "resolved"
    except Exception as error:
        payload.update(status="unresolved", error=f"{type(error).__name__}: {error}")
        logging.getLogger(__name__).warning("FPM runtime memory remains unresolved: %s", payload["error"])
    suffix = f"-tp{tp_rank}-pp{pp_rank}" if kind == "worker" else ""
    path = RESULTS_DIR / f"fpm-memory-{kind}-dp{dp_rank}{suffix}.json"
    if path.exists():
        raise RuntimeError(f"refusing duplicate runtime memory observation: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
