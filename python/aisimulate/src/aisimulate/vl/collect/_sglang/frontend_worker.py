# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record SGLang frontend service times for one image workload; prints one JSON document.

Runs under the serving host's interpreter that has sgl-project/sglang v0.5.19
installed (Python 3.10 or newer, standard library plus sglang). Both frontends
are measured as black boxes at request level under a closed loop of one to
`--levels` concurrent requests:

* Python frontend: the real multimodal processor of the tokenizer manager is
  instantiated in this process with its own IO and processor pools, and each
  request is the wall time of `process_mm_data_async`. The synchronous send
  continuation (`wrap_shm_features` + `msgpack_encode`) and the scheduler's
  per-request receive preparation (msgpack decode, shared-memory
  materialization, `MultimodalInputs.from_processor_output`, placeholder
  padding) are timed on the produced payload with the same real functions.
* Rust frontend: the embedded Rust server is launched alone (no GPU, no
  scheduler); each request is the time from its HTTP send to the moment the
  scheduler-side drain returns it, after which it is answered with an error.

Excluded on both paths: HTTP parsing and chat templating in the Python server,
and every scheduler cost beyond receive preparation.
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


def _check_sglang_version() -> str:
    from sglang.version import __version__

    if not str(__version__).startswith(SGLANG_VERSION):
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
    return spans


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


def _mean_of_steps(prepare, steps):
    """Mean wall time of running `steps` in order on a freshly prepared input; warm-up excluded."""
    samples = []
    for _ in range(MIN_STEADY_SAMPLES + 2):
        state = prepare()
        started = time.perf_counter_ns()
        for step in steps:
            state = step(state)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return sum(samples[2:]) / len(samples[2:])


def _send_continuation_ms(output):
    """Tokenizer-manager loop work after the processor: shared-memory wrapping and msgpack encoding."""
    from sglang.srt.managers.io_struct import msgpack_encode
    from sglang.srt.managers.mm_utils import ShmPointerMMData, wrap_shm_features

    def prepare():
        return _tokenized_request("collect-send", list(output.input_ids), copy.deepcopy(output))

    def send(request):
        wrap_shm_features(request)
        payload = msgpack_encode(request)
        for item in request.mm_inputs.mm_items:
            features = item.feature if isinstance(item.feature, list) else [item.feature]
            for feature in features:
                if isinstance(feature, ShmPointerMMData):
                    feature.close_and_unlink()
        return payload

    return _mean_of_steps(prepare, [send])


def _receive_ms(output):
    """Scheduler-thread preparation of one received request, on the real payload with the real functions."""
    from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
    from sglang.srt.managers.mm_utils import (
        MultiModalityDataPaddingPatternMultimodalTokens,
        unwrap_shm_features,
        wrap_shm_features,
    )
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    input_ids = list(output.input_ids)

    def prepare():
        request = _tokenized_request("collect-receive", input_ids, copy.deepcopy(output))
        wrap_shm_features(request)
        return msgpack_encode(request)

    def decode(payload):
        request = msgpack_decode(payload)
        request.unwrap_pickle_fields()
        return request

    def materialize(request):
        unwrap_shm_features(request)
        return request

    def hash_and_pad(request):
        mm_inputs = MultimodalInputs.from_processor_output(request.mm_inputs)
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(list(request.input_ids), mm_inputs)

    return _mean_of_steps(prepare, [decode, materialize, hash_and_pad])


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
        await asyncio.gather(*(one(-1 - index) for index in range(2 * args.levels)))
        levels = {}
        for level in range(1, args.levels + 1):
            levels[level] = await closed_loop(one, level)
        output = one.output
        if devices != {"cpu"}:
            raise SystemExit(
                f"the Python frontend produced features on {sorted(devices)}; only the CPU path is modeled"
            )
        tm_loop_ms = _send_continuation_ms(output)
        receive_ms = _receive_ms(output)
    finally:
        mm.shutdown()
    return {
        "frontend": "python",
        "workers": args.levels,
        "levels": {str(level): [[span.started_ns, span.ended_ns] for span in spans] for level, spans in levels.items()},
        "tm_loop_ms": tm_loop_ms,
        "receive_ms": receive_ms,
        "provenance": {
            "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
            "mm_processor_class": f"{type(mm).__module__}.{type(mm).__name__}",
            "image_processor_backend": mm.image_processor_backend,
            "io_workers": mm.io_executor._max_workers,
            "processor_workers": mm.mm_processor_worker_num,
            "input_tokens": len(output.input_ids),
            "includes": {
                "pool": ["BaseMultimodalProcessor.process_mm_data_async (image load, HF processor, layout)"],
                "tm_loop": ["wrap_shm_features", "msgpack_encode(TokenizedGenerateReqInput)"],
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
    from sglang.srt.rust_server.multimodal import RustMmProcessor
    from sglang.srt.rust_server.server import RustServer

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_args, model_config, processor = _serving_components(args.model, mm_process_config, port=port)
    mm_workers = args.levels or (server_args.mm_processor_worker_num or getattr(RustMmProcessor, "AUTO_MM_WORKERS", 8))
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
    stop = threading.Event()

    def drain(server):
        # The scheduler-side drain marks the worker's completion; the request is
        # then answered with an error, which the client only uses to move on.
        while not stop.is_set():
            items = server.drain(256)
            now = time.perf_counter_ns()
            for item in items:
                done[str(item.rid).split("#", 1)[0]] = now
                server.server.push_error(item.rid, "aisimulate collect: frontend-only recording")
            if not items:
                time.sleep(0.0002)

    server = RustServer.launch(descriptor)
    drainer = threading.Thread(target=drain, args=(server,), daemon=True)
    drainer.start()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:

            async def one(index):
                rid = f"collect-{index}"
                started = time.perf_counter_ns()
                async with session.post(
                    f"http://127.0.0.1:{port}/generate",
                    json={
                        "rid": rid,
                        "text": prompt,
                        "image_data": payload_images,
                        "sampling_params": {"max_new_tokens": 1},
                    },
                ) as response:
                    await response.read()
                while rid not in done:
                    await asyncio.sleep(0.0002)
                return Span(started, done.pop(rid))

            await asyncio.gather(*(one(-1 - index) for index in range(2 * mm_workers)))
            levels = {}
            for level in range(1, mm_workers + 1):
                levels[level] = await closed_loop(one, level)
    finally:
        stop.set()
        drainer.join(timeout=5)
        server.server.shutdown()
    extension = sys.modules.get("sglang.srt.rust_extensions._server")
    return {
        "frontend": "rust",
        "workers": mm_workers,
        "levels": {str(level): [[span.started_ns, span.ended_ns] for span in spans] for level, spans in levels.items()},
        "provenance": {
            "mm_workers": mm_workers,
            "extension": getattr(extension, "__file__", None),
            "includes": {
                "pool": [
                    "HTTP receive and tokenization in the Rust server",
                    "multimodal worker: payload, fetch, hash, decode, patchify, layout, pack",
                    "channel to the scheduler drain",
                ]
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
    parser.add_argument("--levels", type=int, default=0, help="highest concurrency to sample; 0 = frontend default")
    args = parser.parse_args(argv)
    version = _check_sglang_version()
    if args.frontend == "python" and args.levels <= 0:
        args.levels = 16
    bounds = (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels))
    image = {name: int(value) for name, value in bounds if value}
    mm_process_config = {"image": image} if image else None
    images = generate_images(args.height, args.width, args.count, args.encoding)
    measure = measure_python if args.frontend == "python" else measure_rust
    result = asyncio.run(measure(args, images, mm_process_config))
    result["provenance"].update(
        sampled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(),
        sglang_version=version,
        cpu=_cpu_model(),
        threads=_threads(),
        python=sys.version.split()[0],
    )
    result["cpu"] = _cpu_model()
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
