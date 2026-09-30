# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time one real GLM-5.3-Flash sparse-MLA layer standalone on vLLM 0.30.0+glm53tail.

The checkpoint is built by vLLM's own ``LLM`` engine (model builder, KV/IndexPool
and retained-tail cache allocation, scheduler and metadata builders) with the
pinned serving configuration. The in-process engine core (``VLLM_ENABLE_V1_
MULTIPROCESSING=0``, v1/engine/core_client.py InprocClient) lets this driver
set, before each step, the scheduler's per-step token budget
(``Scheduler.max_num_scheduled_tokens``) and per-request chunk
(``SchedulerConfig.long_prefill_token_threshold``, v1/core/sched/scheduler.py)
so B homogeneous requests advance in lockstep. Those two knobs only decide how
real requests are batched; every KV/IndexPool slot and metadata field is still
produced by vLLM. The worker probe (``glm53flash_attention_worker``) times the
module. See ``collector/README.glm53flash_attention.md``.

Graph mode: serving resolved ``cudagraph_mode=FULL_AND_PIECEWISE`` with capture
sizes up to 64 tokens. Pure decode batches replay FULL graphs there; this
collector uses ``FULL_DECODE_ONLY``, which captures the same FULL uniform-decode
graphs but runs every prefill/mixed step eagerly (serving also runs >64-token
steps eagerly; <=64-token prefill steps use breakable piecewise graphs in
serving and are therefore eager-overestimated here, see README).
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
    RUNTIME_VERSIONS,
    build_plan,
    geometry,
    representative_layer_is_uniform,
)
from collector.glm53flash_attention_runtime import (
    config_sha256,
    corpus_tokens,
    package_source_sha256,
    request_tokens,
    target_id,
)

PLUGIN_ENTRY = "glm53flash_w4 = collector.vllm.glm53flash_attention_worker:register\n"


def write_plugin(directory: Path) -> Path:
    """Expose the probe through vLLM's general-plugin entry-point group."""
    dist = directory / "glm53flash_w4_probe-1.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: glm53flash-w4-probe\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text("[vllm.general_plugins]\n" + PLUGIN_ENTRY)
    return directory


class Driver:
    def __init__(self, llm, plan, tokens, output):
        self.llm = llm
        self.engine = llm.llm_engine
        self.scheduler = self.engine.engine_core.engine_core.scheduler
        self.plan = plan
        self.tokens = tokens
        self.output = output
        self.counter = 0

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
        self.engine.step()
        state = [r.num_computed_tokens for r in self._requests(prefix)]
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
            target = {
                "phase": "context",
                "batch_size": batch,
                "prefix": value,
                "x": query,
                "target_id": target_id("context", batch, value, query),
            }
            self._budget(batch, query)
            self.rpc("rpc_arm", target)
            try:
                final = value + query == last
                self.engine.step()
                if not final:
                    state = [r.num_computed_tokens for r in self._requests(prefix)]
                    if state != [value + query] * batch:
                        raise RuntimeError(f"target step computed {state}, planned {value + query}")
            finally:
                status = self.rpc("rpc_status")
                self.rpc("rpc_arm", None)
            if any(s["done"] != target["target_id"] for s in status):
                raise RuntimeError(f"probe did not measure {target['target_id']}: {status}")
            computed = value + query
            progress(
                {
                    "set_id": request_set["set_id"],
                    "target_id": target["target_id"],
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "passed",
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
            }
            self._budget(batch, request_set["seed_chunk"])
            before = [r.num_computed_tokens for r in self._requests(prefix)]
            if before != [length - 1] * batch:
                raise RuntimeError(f"decode seed computed {before}, planned {length - 1}")
            replays = {s["full_replays"] for s in self.rpc("rpc_status")}
            if len(replays) != 1:
                raise RuntimeError(f"workers disagree on FULL replay counts {replays}")
            self.engine.step()
            result = self.rpc("rpc_measure_decode", target, replays.pop())
            if self.engine.has_unfinished_requests():
                raise RuntimeError("decode request set did not finish after its measured step")
            progress(
                {
                    "set_id": request_set["set_id"],
                    "target_id": target["target_id"],
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "passed",
                    "rpc": result,
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--only-sets", nargs="*", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    options = parser.parse_args()
    manifest_path = Path(options.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    plan = manifest["plan"]
    if plan != build_plan(manifest["sweep"]):
        raise ValueError("manifest plan differs from its frozen sweep")
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

    engine_args = {
        "model": options.model_path,
        "tensor_parallel_size": expected["tp_size"],
        "kv_cache_dtype": "fp8_e4m3",
        "max_model_len": 131079,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "language_model_only": True,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": options.gpu_memory_utilization,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "disable_log_stats": True,
        "seed": 0,
    }
    (output / "execution-contract.json").write_text(
        json.dumps({"engine_args": engine_args, "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True)
    )
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
    longest = max(s["targets"][-1] + (s.get("query") or 0) for s in plan["sets"])
    tokens, corpus = corpus_tokens(tokenizer, Path(options.corpus), longest + 32 * 4099)
    (output / "source_hashes.json").write_text(json.dumps(sources, sort_keys=True))
    (output / "input_provenance.json").write_text(json.dumps({**corpus, **provenance}, sort_keys=True))
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
