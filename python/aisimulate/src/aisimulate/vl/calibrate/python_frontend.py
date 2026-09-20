# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the Python frontend of sgl-project/sglang v0.5.19 (`0bcd822`).

The stages wrap real worker functions inside their own executors:
image decode (`BaseMultimodalProcessor._load_single_item` on the IO pool),
the Hugging Face processor call on the processor pool, and the synchronous
remainder of `process_mm_data_async` on the tokenizer-manager loop. Only the
PIL image backend with PNG input is a CPU path; a run that decodes on a GPU is
rejected because the pools would then model the wrong resource.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from ...config.engine import CostFnConfig, FrontendPredictionConfig, FrontendStageConfig
from .samples import MIN_STEADY_SAMPLES, Span, stage_costs, steady_samples
from .workload import image_data_url


class Recorder:
    """Collect service intervals of one wrapped function; queue wait stays outside."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.enabled = False
        self._lock = threading.Lock()

    def call(self, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        started = time.perf_counter_ns()
        result = function(*args, **kwargs)
        ended = time.perf_counter_ns()
        with self._lock:
            self.spans.append(Span(started, ended))
        return result


async def _measure_curve(call, recorder: Recorder, capacity: int) -> dict[int, list[Span]]:
    """Bounded bursts per concurrency until enough steady samples exist."""
    curves: dict[int, list[Span]] = {}
    await asyncio.gather(*(call(index) for index in range(2 * capacity)))
    for concurrency in range(1, capacity + 1):
        recorder.spans = []
        recorder.enabled = True
        for _ in range(100):
            await asyncio.gather(*(call(index) for index in range(concurrency)))
            if len(steady_samples(recorder.spans, concurrency)) >= MIN_STEADY_SAMPLES:
                break
        recorder.enabled = False
        curves[concurrency] = list(recorder.spans)
    return curves


def _serving_components(model: str):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.tokenizer_manager import get_processor_wrapper
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer

    server_args = ServerArgs(
        model_path=model,
        dtype="bfloat16",
        enable_multimodal=True,
        skip_server_warmup=True,
        image_processor_backend="pil",
    )
    set_global_server_args_for_tokenizer(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return server_args, model_config, get_processor_wrapper()


def _prompt(processor, count: int, text_tokens: int) -> str:
    tokenizer = processor.tokenizer
    ids = tokenizer.encode("Describe the images carefully. " * (text_tokens + 1))[:text_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"}] * count}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def _measure(model: str, images: Sequence[bytes], encoding: str, text_tokens: int) -> dict[str, Any]:
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.managers.schedule_batch import Modality
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode

    server_args, model_config, processor = _serving_components(model)
    import_processors("sglang.srt.multimodal.processors")
    mm = get_mm_processor(
        model_config.hf_config, server_args, processor, determine_tensor_transport_mode(), model_config=model_config
    )
    urls = [image_data_url(encoded, encoding) for encoded in images]
    prompt = _prompt(processor, len(images), text_tokens)
    decode_recorder, processor_recorder = Recorder(), Recorder()
    processor_type = type(mm)
    original_load = processor_type.__dict__.get("_load_single_item")
    bound_load = mm._load_single_item
    original_call = type(processor).__call__
    devices: set[str] = set()

    def load(cls, *args, **kwargs):
        result = decode_recorder.call(bound_load, *args, **kwargs)
        devices.add(str(getattr(result, "device", "cpu")))
        return result

    def hf_call(instance, *args, **kwargs):
        result = processor_recorder.call(original_call, instance, *args, **kwargs)
        devices.add(str(getattr(result.get("pixel_values"), "device", "cpu")))
        return result

    processor_type._load_single_item = classmethod(load)
    type(processor).__call__ = hf_call
    loop = asyncio.get_running_loop()
    try:
        io_workers = mm.io_executor._max_workers
        processor_workers = mm.mm_processor_worker_num

        async def decode_one(index):
            await loop.run_in_executor(mm.io_executor, bound_load, urls[index % len(urls)], Modality.IMAGE)

        decoded = [[bound_load(url, Modality.IMAGE) for url in urls] for _ in range(2 * processor_workers)]

        async def process_one(index):
            loaded = decoded[index % len(decoded)]
            if mm.mm_processor_executor is None:
                return mm.process_mm_data(prompt, images=loaded)
            return await mm.mm_processor_executor.run(mm.process_mm_data, prompt, images=loaded)

        decode_curves = await _measure_curve(decode_one, decode_recorder, io_workers)
        processor_curves = await _measure_curve(process_one, processor_recorder, processor_workers)
        # The tokenizer-manager loop owns whatever the full request path spends
        # outside the two pools; measure it alone so no pool wait is attributed to it.
        decode_recorder.enabled = processor_recorder.enabled = True
        loop_spans = []
        for index in range(MIN_STEADY_SAMPLES):
            decode_recorder.spans, processor_recorder.spans = [], []
            started = time.perf_counter_ns()
            await mm.process_mm_data_async(
                image_data=urls,
                input_text=prompt,
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid=f"calibrate-{index}"),
            )
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            inside = sum(span.service_ms for span in decode_recorder.spans + processor_recorder.spans)
            loop_spans.append(max(wall_ms - inside, 0.0))
        decode_recorder.enabled = processor_recorder.enabled = False
    finally:
        type(processor).__call__ = original_call
        if original_load is None:
            del processor_type._load_single_item
        else:
            processor_type._load_single_item = original_load
        mm.shutdown()
    if devices != {"cpu"}:
        raise RuntimeError(f"Python frontend decoded on {sorted(devices)}; only the CPU path is modeled")
    decode_cost, decode_scale = stage_costs(decode_curves)
    processor_cost, processor_scale = stage_costs(processor_curves)
    frontend = FrontendPredictionConfig(
        io_workers=io_workers,
        processor_workers=processor_workers,
        mm_workers=1,
        stages=[
            FrontendStageConfig(resource="io_decode", unit="image", cost=decode_cost, concurrency_scale=decode_scale),
            FrontendStageConfig(
                resource="processor", unit="request", cost=processor_cost, concurrency_scale=processor_scale
            ),
            FrontendStageConfig(
                resource="tm_loop",
                unit="request",
                cost=CostFnConfig(const_ms=sum(loop_spans) / len(loop_spans)),
            ),
        ],
    )
    provenance = {
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "image_processor_backend": mm.image_processor_backend,
        "decode_samples": sum(len(spans) for spans in decode_curves.values()),
        "processor_samples": sum(len(spans) for spans in processor_curves.values()),
        "loop_ms": loop_spans,
    }
    return {"frontend": frontend, "provenance": provenance}


def measure_python_frontend(model: str, images: Sequence[bytes], encoding: str, *, text_tokens: int = 128):
    """Frontend stages of the PIL/PNG Python frontend for one image workload."""
    return asyncio.run(_measure(model, images, encoding, text_tokens))
