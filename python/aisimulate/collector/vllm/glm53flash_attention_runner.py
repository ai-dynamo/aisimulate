# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time one real GLM-5.3-Flash sparse-MLA layer standalone on stock vLLM 0.31.0.

The checkpoint is built by vLLM's own ``LLM`` engine (model builder, KV/IndexPool
and retained-tail cache allocation, scheduler and metadata builders) with the
pinned serving configuration. The in-process engine core (``VLLM_ENABLE_V1_
MULTIPROCESSING=0``, v1/engine/core_client.py InprocClient) lets this driver
set, before each step, the scheduler's per-step token budget
(``Scheduler.max_num_scheduled_tokens``, v1/core/sched/scheduler.py:132-136,
582) and per-request chunk (``SchedulerConfig.long_prefill_token_threshold``,
scheduler.py:611-626, 679-680; ignored for a lone request, which the budget
then bounds) so B homogeneous requests advance in lockstep. Those two knobs
only decide how real requests are batched; every KV/IndexPool slot and
metadata field is still produced by vLLM. The worker probe
(``glm53flash_attention_worker``) times the module.

IndexPool alignment: stock 0.31.0 leaves the boundary pool of a prefill chunk
that starts off the 4-token pool grid unwritten or fills it with another
request's tokens (models/glm5next/nvidia/sparse_indexer.py:47-90
``_kpool_compress_insert``, "Assumes pool-aligned chunk starts"). Every planned
chunk starts on a multiple of 4 (contract ``seed_chunk``; aligned targets); the
driver checks every scheduled prefill chunk and raises ``KpoolAlignmentError``
before an unaligned one could run, and records the chunk starts per target.

Graph mode: the serving default ``FULL_AND_PIECEWISE`` with the deployment's
capture sizes: uniform decode batches replay FULL graphs and prefill steps up
to 8192 tokens replay breakable PIECEWISE graphs. See
``collector/README.glm53flash_attention.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from collector.glm53flash_attention_contract import (
    KPOOL_ALIGN,
    RUNTIME_VERSIONS,
    build_plan,
    geometry,
    representative_layer_is_uniform,
    selected_max_model_len,
    selected_plan,
    unaligned_targets,
)
from collector.glm53flash_attention_runtime import (
    config_sha256,
    package_source_sha256,
    request_tokens,
    target_id,
)
from collector.glm53flash_attention_tokens import manifest_tokens
from collector.glm53flash_attention_tokens import spec as input_token_spec

PLUGIN_ENTRY = "glm53flash_w4 = collector.vllm.glm53flash_attention_worker:register\n"


def write_plugin(directory: Path) -> Path:
    """Expose the probe through vLLM's general-plugin entry-point group."""
    dist = directory / "glm53flash_w4_probe-1.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: glm53flash-w4-probe\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text("[vllm.general_plugins]\n" + PLUGIN_ENTRY)
    return directory


class KpoolAlignmentError(RuntimeError):
    """A vLLM prefill chunk would start off the IndexPool grid (stock 0.31.0 defect)."""


class Driver:
    def __init__(self, llm, plan, tokens, output):
        self.llm = llm
        self.engine = llm.llm_engine
        self.scheduler = self.engine.engine_core.engine_core.scheduler
        self.plan = plan
        self.tokens = tokens
        self.output = output
        self.counter = 0
        self.chunk_starts: set[int] = set()

    def rpc(self, name, *args):
        from collector.vllm import glm53flash_attention_worker as worker

        return self.llm.collective_rpc(getattr(worker, name), args=args)

    def _requests(self, prefix):
        return sorted(
            (r for r in self.scheduler.requests.values() if r.request_id.startswith(prefix)),
            key=lambda r: r.request_id,
        )

    def _budget(self, batch, chunk):
        self.scheduler.max_num_scheduled_tokens = batch * chunk
        self.scheduler.scheduler_config.long_prefill_token_threshold = chunk

    def _step(self, prefix, batch, expected_computed):
        """One engine step whose prefill chunks must all start on the pool grid."""
        before = {r.request_id: (r.num_computed_tokens, r.num_prompt_tokens) for r in self._requests(prefix)}
        unaligned = sorted({c for c, prompt in before.values() if c < prompt and c % KPOOL_ALIGN})
        if unaligned:
            raise KpoolAlignmentError(f"prefill chunk would start at {unaligned} (not a multiple of {KPOOL_ALIGN})")
        self.engine.step()
        after = {r.request_id: r.num_computed_tokens for r in self._requests(prefix)}
        for request, (computed, prompt) in before.items():
            if computed < prompt and after.get(request, prompt) > computed:
                self.chunk_starts.add(computed)
        state = list(after.values())
        if expected_computed is not None and state and state != [expected_computed] * batch:
            raise RuntimeError(f"requests computed {state}, planned {expected_computed}")

    def _add(self, prefix, batch, length, max_tokens):
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False)
        for request in range(batch):
            self.engine.add_request(
                f"{prefix}-{request:02d}",
                {"prompt_token_ids": request_tokens(self.tokens, request, length)},
                params,
            )
        self.chunk_starts = set()

    def _seed(self, prefix, batch, chunk, start, end):
        computed = start
        while computed < end:
            step = min(chunk, end - computed)
            self._budget(batch, step)
            self._step(prefix, batch, computed + step)
            computed += step
        return computed

    def run_prefill(self, request_set, progress):
        batch, query = request_set["batch_size"], request_set["query"]
        self.counter += 1
        prefix = f"w4-{self.counter:04d}-{request_set['set_id']}"
        last = request_set["targets"][-1] + query
        self._add(prefix, batch, last, 1)
        computed = 0
        for value in request_set["targets"]:
            started = time.monotonic()
            computed = self._seed(prefix, batch, request_set["seed_chunk"], computed, value)
            self._budget(batch, query)
            replays = {s["pw_replays"] for s in self.rpc("rpc_status")}
            if len(replays) != 1:
                raise RuntimeError(f"workers disagree on PIECEWISE replay counts {replays}")
            final = value + query == last
            self._step(prefix, batch, None if final else value + query)
            target = {
                "phase": "context",
                "batch_size": batch,
                "prefix": value,
                "x": query,
                "target_id": target_id("context", batch, value, query),
                "chunk_starts": sorted(self.chunk_starts),
            }
            result = self.rpc("rpc_measure_prefill", target, replays.pop())
            computed = value + query
            progress(
                {
                    "set_id": request_set["set_id"],
                    "target_id": target["target_id"],
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "passed",
                    "chunk_starts": target["chunk_starts"],
                    "rpc": result,
                }
            )
        if self.engine.has_unfinished_requests():
            raise RuntimeError("prefill request set did not finish at its last target")

    def run_decode(self, request_set, progress):
        batch = request_set["batch_size"]
        for length in request_set["targets"]:
            started = time.monotonic()
            self.counter += 1
            prefix = f"w4-{self.counter:04d}-{request_set['set_id']}-l{length}"
            self._add(prefix, batch, length - 1, 2)
            self._seed(prefix, batch, request_set["seed_chunk"], 0, length - 1)
            target = {
                "phase": "generation",
                "batch_size": batch,
                "prefix": 0,
                "x": length,
                "target_id": target_id("generation", batch, 0, length),
                "chunk_starts": sorted(self.chunk_starts),
            }
            self._budget(batch, request_set["seed_chunk"])
            before = [r.num_computed_tokens for r in self._requests(prefix)]
            if before != [length - 1] * batch:
                raise RuntimeError(f"decode seed computed {before}, planned {length - 1}")
            replays = {s["full_replays"] for s in self.rpc("rpc_status")}
            if len(replays) != 1:
                raise RuntimeError(f"workers disagree on FULL replay counts {replays}")
            self._step(prefix, batch, None)
            result = self.rpc("rpc_measure_decode", target, replays.pop())
            if self.engine.has_unfinished_requests():
                raise RuntimeError("decode request set did not finish after its measured step")
            progress(
                {
                    "set_id": request_set["set_id"],
                    "target_id": target["target_id"],
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "passed",
                    "chunk_starts": target["chunk_starts"],
                    "rpc": result,
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--only-sets", nargs="*", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="resolve everything, then exit before CUDA work")
    options = parser.parse_args()
    manifest_path = Path(options.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    plan = manifest["plan"]
    if plan != build_plan(manifest["sweep"]):
        raise ValueError("manifest plan differs from its frozen sweep")
    if manifest.get("input_tokens") != input_token_spec(plan):
        raise ValueError("manifest input_tokens differ from this collector's generator")
    if manifest.get("only_sets") is not None:
        # A split attempt measures exactly the manifest's set selection.
        if options.only_sets and sorted(options.only_sets) != manifest["only_sets"]:
            raise ValueError("--only-sets differs from the manifest selection")
        options.only_sets = manifest["only_sets"]
    unaligned = unaligned_targets(selected_plan(manifest))
    if unaligned:
        raise KpoolAlignmentError(f"kpool_align4: planned prefill chunks off the pool grid {unaligned[:6]}")
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("rank-*.jsonl")) or (output / "COMPLETE").exists():
        raise RuntimeError("output has prior raw records; use a fresh attempt directory")
    config = json.loads((Path(options.model_path) / "config.json").read_text())
    expected = geometry(config, "vllm", manifest["geometry"]["checkpoint_format"], manifest["geometry"]["tp_size"])
    if expected != manifest["geometry"]:
        raise ValueError("checkpoint/TP geometry differs from the manifest")
    representative_layer_is_uniform(config, manifest["layer_id"], expected["checkpoint_format"])

    plugin = write_plugin(output / "plugin")
    os.environ["PYTHONPATH"] = f"{plugin}:{os.environ.get('PYTHONPATH', '')}"
    sys.path.insert(0, str(plugin))
    os.environ["GLM53_W4_MANIFEST"] = str(manifest_path)
    os.environ["GLM53_W4_FRAMEWORK_VERSION"] = manifest["framework_version"]
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from importlib.metadata import version

    import vllm

    installed = version("vllm")
    if (
        installed != RUNTIME_VERSIONS["vllm"]
        or vllm.__version__ != installed
        or manifest["framework_version"] != installed
    ):
        raise RuntimeError(f"vllm {installed}/{vllm.__version__} is not the pinned {RUNTIME_VERSIONS['vllm']}")
    from vllm import LLM

    # Serving default FULL_AND_PIECEWISE (breakable CUDA graph under
    # CompilationMode.NONE) with the deployment's 62 capture sizes.
    compilation_config = {"cudagraph_capture_sizes": manifest["serving_graph"]["vllm_cudagraph_capture_sizes"]}
    max_model_len = selected_max_model_len(manifest)
    if manifest["max_model_len"] != max_model_len:
        raise ValueError("manifest max_model_len differs from its selected context class")
    engine_args = {
        "model": options.model_path,
        "tensor_parallel_size": expected["tp_size"],
        "kv_cache_dtype": "fp8_e4m3",
        # Serving context limit of the selected context class (capacity only).
        "max_model_len": max_model_len,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "language_model_only": True,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": options.gpu_memory_utilization,
        "compilation_config": compilation_config,
        "disable_log_stats": True,
        "seed": 0,
    }
    (output / "execution-contract.json").write_text(
        json.dumps({"engine_args": engine_args, "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True)
    )
    if options.dry_run:
        from importlib.metadata import entry_points

        from vllm.engine.arg_utils import EngineArgs

        EngineArgs(**engine_args)
        plugins = [e.value for e in entry_points(group="vllm.general_plugins")]
        if "collector.vllm.glm53flash_attention_worker:register" not in plugins:
            raise RuntimeError(f"probe plugin is not discoverable: {plugins}")
        print(
            json.dumps(
                {
                    "dry_run": "ok",
                    "framework": installed,
                    "geometry": expected,
                    "sets": sum(not options.only_sets or s["set_id"] in options.only_sets for s in plan["sets"]),
                    "engine_args": engine_args,
                    "plugins": plugins,
                },
                sort_keys=True,
            )
        )
        return
    llm = LLM(**engine_args)
    source_sha, sources = package_source_sha256(Path(vllm.__file__).resolve().parent)
    provenance = {
        "framework_version": installed,
        "source_sha256": source_sha,
        "config_sha256": config_sha256(Path(options.model_path)),
        "checkpoint_revision": manifest["checkpoint_revision"],
        "runtime_digest": manifest["runtime_digest"],
        "layer_id": manifest["layer_id"],
    }
    tokenizer = llm.get_tokenizer()
    tokens, inputs = manifest_tokens(manifest, tokenizer, Path(options.model_path))
    (output / "source_hashes.json").write_text(json.dumps(sources, sort_keys=True))
    (output / "input_provenance.json").write_text(json.dumps({**inputs, **provenance}, sort_keys=True))
    driver = Driver(llm, plan, tokens, output)
    setup = driver.rpc(
        "rpc_setup",
        str(output),
        dict(manifest["geometry"]),
        provenance,
        {"warmup": plan["warmup"], "iterations": plan["iterations"]},
    )
    (output / "worker-setup.json").write_text(json.dumps(setup, sort_keys=True))
    if sorted(r["rank"] for r in setup) != list(range(expected["tp_size"])):
        raise RuntimeError(f"unexpected worker ranks {setup}")

    def progress(payload):
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")

    for request_set in plan["sets"]:
        if options.only_sets and request_set["set_id"] not in options.only_sets:
            continue
        try:
            if request_set["phase"] == "context":
                driver.run_prefill(request_set, progress)
            else:
                driver.run_decode(request_set, progress)
        except BaseException as error:
            progress(
                {
                    "set_id": request_set["set_id"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            raise
    (output / "COMPLETE").write_text("glm53flash attention collection completed\n")


if __name__ == "__main__":
    main()
