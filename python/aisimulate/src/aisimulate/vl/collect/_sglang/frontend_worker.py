# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record SGLang frontend service times for one image workload; prints one JSON document.

Runs under the serving host's interpreter that has sgl-project/sglang v0.5.19
installed (Python 3.10 or newer, standard library plus sglang). Both frontends
are measured as black boxes at request level under a closed loop of one request
up to the width of the pool being measured, each stage on the real SGLang objects:

* Python frontend: the tokenizer manager's multimodal processor is instantiated
  in this process with its own IO and processor pools; `process` is the wall
  time of `process_mm_data_async`, `send` the loop's synchronous
  `wrap_shm_features` + `msgpack_encode`, and `receive` the scheduler's
  per-request preparation of the produced payload (msgpack decode, shared-memory
  materialization, `MultimodalInputs.from_processor_output`, placeholder padding).
* Rust frontend: the embedded Rust server is launched alone (no GPU, no
  scheduler) under the requested tensor-parallel width, which decides whether
  features ride inline or through shared memory; `process` is the time from the
  HTTP send to the moment the scheduler-side drain returns the request, and
  `receive` the scheduler's preparation of that drained request. Each drained
  request is then answered with an error, which the client only uses to move on.

Excluded on both paths: HTTP parsing and chat templating in the Python server,
and every scheduler cost beyond receive preparation. Progress goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import platform
import socket
import sys
import threading
import time
from array import array
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samples import MIN_STEADY_SAMPLES, Span, steady_samples
from workload import generate_images, image_data_url

SGLANG_VERSION = "0.5.19"
REQUEST_DEADLINE_S = 120.0


def _release(version: str) -> str:
    """`major.minor.patch` of a version string, local suffixes dropped (mirrors `vl.table.sglang_release`)."""
    return ".".join(version.split("+", 1)[0].split(".")[:3])


def _sglang_version() -> str:
    from sglang.version import __version__

    if _release(str(__version__)) != SGLANG_VERSION:
        raise SystemExit(f"installed sglang {__version__} is not {SGLANG_VERSION}; the stage boundaries are pinned")
    return str(__version__)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _serving_components(model, mm_process_config, port=None, image_processor_backend="pil"):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.tokenizer_manager import get_processor_wrapper
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer

    extra = {}
    if mm_process_config:
        extra["mm_process_config"] = mm_process_config
    if port is not None:
        extra.update(host="127.0.0.1", port=port)
    server_args = ServerArgs(
        model_path=model,
        dtype="bfloat16",
        enable_multimodal=True,
        skip_server_warmup=True,
        image_processor_backend=image_processor_backend,
        **extra,
    )
    set_global_server_args_for_tokenizer(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return server_args, model_config, get_processor_wrapper()


def _prompt(processor, count, text_tokens):
    tokenizer = processor.tokenizer
    ids = tokenizer.encode("Describe the images carefully. " * (text_tokens + 1))[:text_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"}] * count}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def closed_loop(call, level, budget_per_worker=100):
    """Keep `level` jobs in flight until the level has enough steady samples.

    A burst of short jobs never runs at its target concurrency: submission and
    wake-up skew exceed a few-millisecond service, so the burst drains before its
    last job starts. Resubmitting on completion keeps the load constant; ramp-up
    and drain samples fall out of the steady criterion.
    """
    spans = []
    state = {"issued": 0, "done": 0, "stop": False}
    budget = budget_per_worker * level

    async def worker():
        while not state["stop"] and state["issued"] < budget:
            index = state["issued"]
            state["issued"] += 1
            spans.append(await call(index))
            state["done"] += 1
            if state["done"] % level == 0 and len(steady_samples(spans, level)) >= MIN_STEADY_SAMPLES:
                state["stop"] = True

    await asyncio.gather(*(worker() for _ in range(level)))
    _progress(f"level {level}: {len(steady_samples(spans, level))} steady of {len(spans)} samples")
    return spans


async def measure_levels(call, levels):
    recorded = {}
    for level in range(1, levels + 1):
        spans = await closed_loop(call, level)
        recorded[str(level)] = [[span.started_ns, span.ended_ns] for span in spans]
    return recorded


def _feature_devices(items):
    devices = set()
    for item in items:
        features = item.feature if isinstance(item.feature, list) else [item.feature]
        devices.update(str(getattr(feature, "device", "cpu")) for feature in features if feature is not None)
    return devices


def _tokenized_request(rid, input_ids, mm_inputs):
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids) if input_ids is not None else None,
        input_embeds=None,
        mm_inputs=mm_inputs,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def _mean_ms(prepare, timed, cleanup=None, samples=MIN_STEADY_SAMPLES + 2):
    """Mean wall time of `timed` over freshly prepared inputs; the first two runs warm up.

    Preparation and cleanup stay outside the timed region, and one input is alive
    at a time so the samples never pile up feature buffers or shared memory.
    """
    durations = []
    for _ in range(samples):
        state = prepare()
        started = time.perf_counter_ns()
        output = timed(state)
        durations.append((time.perf_counter_ns() - started) / 1e6)
        if cleanup is not None:
            cleanup(output)
    return sum(durations[2:]) / len(durations[2:])


def _release_shm(request):
    from sglang.srt.managers.mm_utils import ShmPointerMMData

    for item in request.mm_inputs.mm_items:
        features = item.feature if isinstance(item.feature, list) else [item.feature]
        for feature in features:
            if isinstance(feature, ShmPointerMMData):
                feature.close_and_unlink()


def _send_ms(output):
    """Tokenizer-manager loop work after the processor: shared-memory wrapping and msgpack encoding."""
    from sglang.srt.managers.io_struct import msgpack_encode
    from sglang.srt.managers.mm_utils import wrap_shm_features

    def prepare():
        return _tokenized_request("collect-send", list(output.input_ids), copy.deepcopy(output))

    def send(request):
        wrap_shm_features(request)
        msgpack_encode(request)
        return request

    return _mean_ms(prepare, send, cleanup=_release_shm)


def _receive_steps():
    import torch
    from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, unwrap_shm_features
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    # The scheduler process runs torch single-threaded (`ModelRunner.load_model`
    # sets it for GPU devices and never restores it); with the default pool the
    # shared-memory clone thrashes on the two cores the Rust server leaves it.
    torch.set_num_threads(1)

    def prepare(request):
        unwrap_shm_features(request)
        mm_inputs = MultimodalInputs.from_processor_output(request.mm_inputs)
        MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(list(request.input_ids), mm_inputs)
        return request

    return prepare


def _python_receive_ms(output):
    """Scheduler-thread preparation of one received request, decoded from the real payload."""
    from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
    from sglang.srt.managers.mm_utils import wrap_shm_features

    steps = _receive_steps()

    def prepare():
        request = _tokenized_request("collect-receive", list(output.input_ids), copy.deepcopy(output))
        wrap_shm_features(request)
        return msgpack_encode(request)

    def receive(payload):
        request = msgpack_decode(payload)
        request.unwrap_pickle_fields()
        return steps(request)

    # Materializing the features unlinks their segments; nothing is left to clean up.
    return _mean_ms(prepare, receive)


def _environment(version):
    """What stays fixed across collections on this host; the table's identity."""
    return {"cpu": _cpu_model(), "sglang_version": version, "python": sys.version.split()[0]}


async def measure_python(args, images, mm_process_config):
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode

    server_args, model_config, processor = _serving_components(args.model, mm_process_config)
    import_processors("sglang.srt.multimodal.processors")
    mm = get_mm_processor(
        model_config.hf_config, server_args, processor, determine_tensor_transport_mode(), model_config=model_config
    )
    urls = [image_data_url(encoded, args.encoding) for encoded in images]
    prompt = _prompt(processor, len(images), args.text_tokens)
    workers = int(mm.io_executor._max_workers)
    levels = workers
    devices = set()
    try:

        async def one(index):
            started = time.perf_counter_ns()
            output = await mm.process_mm_data_async(
                image_data=urls,
                input_text=prompt,
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid=f"collect-{index}"),
            )
            span = Span(started, time.perf_counter_ns())
            devices.update(_feature_devices(output.mm_items))
            one.output = output
            return span

        # Warm the pools and the processor before timing.
        await asyncio.gather(*(one(-1 - index) for index in range(2 * levels)))
        if devices != {"cpu"}:
            raise SystemExit(
                f"the Python frontend produced features on {sorted(devices)}; only the CPU path is modeled"
            )
        recorded = await measure_levels(one, levels)
        output = one.output
        send_ms = _send_ms(output)
        receive_ms = _python_receive_ms(output)
    finally:
        mm.shutdown()
    return {
        "frontend": "python",
        "workers": workers,
        "levels": recorded,
        "send_ms": send_ms,
        "receive_ms": receive_ms,
        "provenance": {
            "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
            "mm_processor_class": f"{type(mm).__module__}.{type(mm).__name__}",
            "image_processor_backend": mm.image_processor_backend,
            "processor_workers": mm.mm_processor_worker_num,
            "text_tokens": args.text_tokens,
            "input_tokens": len(output.input_ids),
            "includes": {
                "process": ["BaseMultimodalProcessor.process_mm_data_async (image load, HF processor, layout)"],
                "send": ["wrap_shm_features", "msgpack_encode(TokenizedGenerateReqInput)"],
                "receive": [
                    "msgpack_decode + unwrap_pickle_fields",
                    "unwrap_shm_features",
                    "MultimodalInputs.from_processor_output (hash_feature per item)",
                    "MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens",
                ],
            },
        },
    }


async def measure_rust(args, images, mm_process_config):
    import aiohttp
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.rust_server.multimodal import RustMmProcessor
    from sglang.srt.rust_server.server import RustServer

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_args, model_config, processor = _serving_components(args.model, mm_process_config, port=port)
    workers = int(server_args.mm_processor_worker_num or getattr(RustMmProcessor, "AUTO_MM_WORKERS", 8))
    levels = workers
    descriptor = SimpleNamespace(
        server_args=server_args,
        model_config=model_config,
        processor=processor,
        ps=SimpleNamespace(attn_dp_rank=0, dp_size=1),
        max_total_num_tokens=model_config.context_len,
    )
    prompt = _prompt(processor, len(images), args.text_tokens)
    payload_images = [image_data_url(encoded, args.encoding) for encoded in images]
    done = {}
    kept = []
    stop = threading.Event()
    failure = []

    def drain(server):
        # The scheduler-side drain marks the worker's completion. A few drained
        # requests are kept to time the scheduler's receive preparation later; the
        # rest release their shared-memory features at once, as the scheduler's
        # materialization would (the unlink is Python's once a request is drained).
        # Control messages (client aborts after our error reply) carry no work.
        try:
            while not stop.is_set():
                items = server.drain(256)
                now = time.perf_counter_ns()
                for item in items:
                    if not isinstance(item, TokenizedGenerateReqInput):
                        continue
                    done[str(item.rid).split("#", 1)[0]] = now
                    if len(kept) < MIN_STEADY_SAMPLES + 2:
                        kept.append(item)
                    else:
                        _release_shm(item)
                    server.server.push_error(item.rid, "aisimulate collect: frontend-only recording")
                if not items:
                    time.sleep(0.0002)
        except Exception as exc:
            failure.append(exc)

    # The feature transport follows the tensor-parallel width the serving host runs;
    # `override` publishes it without a distributed init.
    with get_parallel().override(tp_size=args.tp):
        server = RustServer.launch(descriptor)
    if server.mm_spec is None:
        raise SystemExit("the Rust server did not enable its multimodal pipeline for this model")
    feature_shm = bool(server.mm_spec.feature_shm)
    if feature_shm != (args.tp > 1):
        raise SystemExit(
            f"sglang chose feature_shm={feature_shm} for tp={args.tp}; the transport model expects otherwise"
        )
    drainer = threading.Thread(target=drain, args=(server,), daemon=True)
    drainer.start()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_DEADLINE_S)) as session:

            async def one(index):
                rid = f"collect-{index}"
                started = time.perf_counter_ns()
                deadline = time.monotonic() + REQUEST_DEADLINE_S
                async with session.post(
                    f"http://127.0.0.1:{port}/generate",
                    json={
                        "rid": rid,
                        "text": prompt,
                        "image_data": payload_images,
                        "sampling_params": {"max_new_tokens": 1},
                    },
                ) as response:
                    body = await response.read()
                if rid not in done and response.status != 200:
                    raise RuntimeError(f"the Rust server rejected request {rid}: HTTP {response.status} {body[:200]!r}")
                while rid not in done:
                    if failure:
                        raise RuntimeError("the scheduler-side drain failed") from failure[0]
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"request {rid} never reached the scheduler-side drain")
                    await asyncio.sleep(0.0002)
                return Span(started, done.pop(rid))

            await asyncio.gather(*(one(-1 - index) for index in range(2 * levels)))
            recorded = await measure_levels(one, levels)
    finally:
        stop.set()
        drainer.join(timeout=5)
        server.server.shutdown()
    try:
        if len(kept) < MIN_STEADY_SAMPLES + 2:
            raise SystemExit("too few drained requests were kept to time the receive preparation")
        pending = iter(kept)
        receive_ms = _mean_ms(lambda: next(pending), _receive_steps(), samples=len(kept))
    finally:
        # Materialization unlinked the timed requests; release whatever was not reached.
        for item in kept:
            _release_shm(item)
    extension = sys.modules.get("sglang.srt.rust_extensions._server")
    return {
        "frontend": "rust",
        "workers": workers,
        "levels": recorded,
        "receive_ms": receive_ms,
        "provenance": {
            "tp": args.tp,
            "feature_shm": feature_shm,
            "extension": getattr(extension, "__file__", None),
            "text_tokens": args.text_tokens,
            "includes": {
                "process": [
                    "HTTP receive and tokenization in the Rust server",
                    "multimodal worker: payload, fetch, hash, decode, patchify, layout, pack"
                    + (", shared-memory publish" if feature_shm else ""),
                    "channel to the scheduler drain",
                ],
                "receive": [
                    "unwrap_shm_features",
                    "MultimodalInputs.from_processor_output (hash_feature per item)",
                    "MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens",
                ],
            },
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="frontend_worker", description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frontend", choices=("python", "rust"), required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--encoding", choices=("png", "jpeg"), default="png")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel width of the serving host (Rust frontend)")
    args = parser.parse_args(argv)
    version = _sglang_version()
    # Recorded before the Rust server narrows this thread's affinity.
    environment = _environment(version)
    collected = {"host": platform.node(), "threads": _threads()}
    bounds = (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels))
    image = {name: int(value) for name, value in bounds if value}
    mm_process_config = {"image": image} if image else None
    images = generate_images(args.height, args.width, args.count, args.encoding)
    measure = measure_python if args.frontend == "python" else measure_rust
    result = asyncio.run(measure(args, images, mm_process_config))
    result["environment"] = environment
    result["provenance"].update(collected, sampled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
