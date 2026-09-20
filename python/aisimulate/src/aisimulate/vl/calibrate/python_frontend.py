# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the Python frontend of sgl-project/sglang v0.5.19 (`0bcd822`).

Each stage is the unit its executor really runs, wrapped at the class attribute
the pinned code dispatches through so every call is recorded:

* ``io_decode`` (unit ``image``): `BaseMultimodalProcessor._load_single_item`,
  submitted per image to ``io_executor``.
* ``processor`` (unit ``request``): `BaseMultimodalProcessor.process_and_combine_mm_data`,
  the whole function `process_and_combine_mm_data_async` submits to the
  thread-based `MultimodalProcessorExecutor` (Hugging Face processor call, item
  collection, offsets, per-image split, pad values, transport wrapping). With
  one processor worker the executor is ``None`` and the same function runs
  synchronously on the tokenizer-manager loop, so the stage is then attributed
  to ``tm_loop``.
* ``tm_loop`` (unit ``request``): the request's `process_mm_data_async` wall time
  minus the interval union of its pool spans (regex split, future dispatch,
  padded ids, grids, M-RoPE positions, output assembly), plus the send-side
  continuation the tokenizer manager runs before the scheduler handoff
  (`wrap_shm_features`, i.e. the POSIX shared-memory copy of every feature, and
  `msgpack_encode` of the multimodal output).

Only the PIL image backend with CPU features is a CPU path; a run whose
features land on a GPU is rejected because the pools would then model the
wrong resource. Excluded from every stage: HTTP parsing, chat templating, text
tokenization of the surrounding prompt, ZMQ transport.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from ...config.engine import CostFnConfig, FrontendPredictionConfig, FrontendStageConfig
from .samples import MIN_STEADY_SAMPLES, Span, stage_costs, steady_samples, union_ms
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


def install_recorders(
    processor_type: type, decode_recorder: Recorder, processor_recorder: Recorder
) -> Callable[[], None]:
    """Route the class-level worker entry points through the recorders.

    `_load_single_item` is a classmethod the pinned code submits as
    ``self.__class__._load_single_item``; `process_and_combine_mm_data` is
    submitted as a bound method of the instance. Both resolve through the
    class, so replacing the attributes there records the executor calls and
    the calls made by `process_mm_data_async` alike. Returns the restore hook.
    """
    saved = {name: processor_type.__dict__.get(name) for name in ("_load_single_item", "process_and_combine_mm_data")}
    load_function = processor_type._load_single_item.__func__
    combine_function = processor_type.process_and_combine_mm_data

    def load(cls, *args, **kwargs):
        return decode_recorder.call(load_function, cls, *args, **kwargs)

    def combine(instance, *args, **kwargs):
        return processor_recorder.call(combine_function, instance, *args, **kwargs)

    processor_type._load_single_item = classmethod(load)
    processor_type.process_and_combine_mm_data = combine

    def restore() -> None:
        for name, original in saved.items():
            if original is None:
                delattr(processor_type, name)
            else:
                setattr(processor_type, name, original)

    return restore


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


def image_process_config(min_pixels: int | None, max_pixels: int | None) -> dict[str, Any] | None:
    """SGLang's `--mm-process-config` for a workload that overrides the processor's pixel budget."""
    image = {name: int(value) for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value}
    return {"image": image} if image else None


def _serving_components(model: str, mm_process_config: dict[str, Any] | None = None):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.tokenizer_manager import get_processor_wrapper
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer

    server_args = ServerArgs(
        model_path=model,
        dtype="bfloat16",
        enable_multimodal=True,
        skip_server_warmup=True,
        image_processor_backend="pil",
        **({"mm_process_config": mm_process_config} if mm_process_config else {}),
    )
    set_global_server_args_for_tokenizer(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return server_args, model_config, get_processor_wrapper()


def _multimodal_processor(model: str, mm_process_config: dict[str, Any] | None = None):
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode

    server_args, model_config, processor = _serving_components(model, mm_process_config)
    import_processors("sglang.srt.multimodal.processors")
    mm = get_mm_processor(
        model_config.hf_config, server_args, processor, determine_tensor_transport_mode(), model_config=model_config
    )
    return server_args, model_config, processor, mm


def _prompt(processor, count: int, text_tokens: int) -> str:
    tokenizer = processor.tokenizer
    ids = tokenizer.encode("Describe the images carefully. " * (text_tokens + 1))[:text_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"}] * count}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _feature_devices(items) -> set[str]:
    devices = set()
    for item in items:
        features = item.feature if isinstance(item.feature, list) else [item.feature]
        devices.update(str(getattr(feature, "device", "cpu")) for feature in features if feature is not None)
    return devices


def _send_continuation_ms(output) -> float:
    """Tokenizer-manager work between the processor output and the scheduler handoff.

    `_create_tokenized_object` wraps every CPU feature into POSIX shared memory
    and the request is msgpack-encoded for the scheduler; both run on the loop.
    """
    from sglang.srt.managers.io_struct import msgpack_encode
    from sglang.srt.managers.mm_utils import ShmPointerMMData, wrap_shm_features

    samples = []
    for _ in range(MIN_STEADY_SAMPLES + 2):
        payload = copy.deepcopy(output)
        started = time.perf_counter_ns()
        wrap_shm_features(SimpleNamespace(mm_inputs=payload))
        msgpack_encode(payload)
        samples.append((time.perf_counter_ns() - started) / 1e6)
        for item in payload.mm_items:
            features = item.feature if isinstance(item.feature, list) else [item.feature]
            for feature in features:
                if isinstance(feature, ShmPointerMMData):
                    feature.close_and_unlink()
    return sum(samples[2:]) / len(samples[2:])


async def _measure(
    model: str,
    images: Sequence[bytes],
    encoding: str,
    text_tokens: int,
    mm_process_config: dict[str, Any] | None,
) -> dict[str, Any]:
    from sglang.srt.managers.schedule_batch import Modality

    _, _, processor, mm = _multimodal_processor(model, mm_process_config)
    urls = [image_data_url(encoded, encoding) for encoded in images]
    prompt = _prompt(processor, len(images), text_tokens)
    decode_recorder, processor_recorder = Recorder(), Recorder()
    restore = install_recorders(type(mm), decode_recorder, processor_recorder)
    loop = asyncio.get_running_loop()
    devices: set[str] = set()
    try:
        io_workers = mm.io_executor._max_workers
        processor_workers = mm.mm_processor_worker_num
        synchronous_processor = mm.mm_processor_executor is None

        async def decode_one(index):
            # Resolved through the class, i.e. through the recorder, as the IO pool submission does.
            decoded = await loop.run_in_executor(
                mm.io_executor, mm._load_single_item, urls[index % len(urls)], Modality.IMAGE
            )
            # nvJPEG decodes land on a CUDA device before the features are moved to the CPU.
            devices.add(str(getattr(decoded, "device", "cpu")))

        # Loaded inputs for the processor stage, one per job that can be in flight.
        base_outputs = [
            await mm.load_mm_data(prompt=prompt, image_data=urls, multimodal_tokens=mm.mm_tokens)
            for _ in range(2 * processor_workers)
        ]
        for base_output in base_outputs:
            devices.update(str(getattr(image, "device", "cpu")) for image in (base_output.images or []))

        async def process_one(index):
            items, _, _ = await mm.process_and_combine_mm_data_async(
                base_outputs[index % len(base_outputs)], mm.mm_tokens
            )
            devices.update(_feature_devices(items))

        decode_curves = await _measure_curve(decode_one, decode_recorder, io_workers)
        processor_curves = await _measure_curve(process_one, processor_recorder, processor_workers)

        # The loop owns whatever the request path spends while no pool job runs;
        # parallel pool spans are subtracted once, as their interval union.
        decode_recorder.enabled = processor_recorder.enabled = True
        loop_ms = []
        output = None
        for index in range(MIN_STEADY_SAMPLES):
            decode_recorder.spans, processor_recorder.spans = [], []
            started = time.perf_counter_ns()
            output = await mm.process_mm_data_async(
                image_data=urls,
                input_text=prompt,
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid=f"calibrate-{index}"),
            )
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            loop_ms.append(max(wall_ms - union_ms(decode_recorder.spans + processor_recorder.spans), 0.0))
        decode_recorder.enabled = processor_recorder.enabled = False
        devices.update(_feature_devices(output.mm_items))
        continuation_ms = _send_continuation_ms(output)
    finally:
        restore()
        mm.shutdown()
    if devices != {"cpu"}:
        raise RuntimeError(
            f"Python frontend decoded or produced features on {sorted(devices)}; only the CPU path is modeled"
        )
    decode_cost, decode_scale = stage_costs(decode_curves, capacity=io_workers)
    processor_cost, processor_scale = stage_costs(processor_curves, capacity=processor_workers)
    frontend = FrontendPredictionConfig(
        io_workers=io_workers,
        processor_workers=processor_workers,
        mm_workers=1,
        stages=[
            FrontendStageConfig(resource="io_decode", unit="image", cost=decode_cost, concurrency_scale=decode_scale),
            FrontendStageConfig(
                resource="tm_loop" if synchronous_processor else "processor",
                unit="request",
                cost=processor_cost,
                concurrency_scale=[] if synchronous_processor else processor_scale,
            ),
            FrontendStageConfig(
                resource="tm_loop",
                unit="request",
                cost=CostFnConfig(const_ms=sum(loop_ms) / len(loop_ms) + continuation_ms),
            ),
        ],
    )
    provenance = {
        "processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "mm_processor_class": f"{type(mm).__module__}.{type(mm).__name__}",
        "image_processor_backend": mm.image_processor_backend,
        "processor_executor": "tm_loop" if synchronous_processor else "MultimodalProcessorExecutor",
        "decode_samples": sum(len(spans) for spans in decode_curves.values()),
        "processor_samples": sum(len(spans) for spans in processor_curves.values()),
        "loop_ms": loop_ms,
        "send_continuation_ms": continuation_ms,
        "includes": {
            "io_decode": ["BaseMultimodalProcessor._load_single_item"],
            "processor": ["BaseMultimodalProcessor.process_and_combine_mm_data"],
            "tm_loop": [
                "process_mm_data_async minus pool spans",
                "wrap_shm_features",
                "msgpack_encode(MultimodalProcessorOutput)",
            ],
        },
    }
    return {"frontend": frontend, "provenance": provenance}


def measure_python_frontend(
    model: str,
    images: Sequence[bytes],
    encoding: str,
    *,
    text_tokens: int = 128,
    mm_process_config: dict[str, Any] | None = None,
):
    """Frontend stages of the PIL Python frontend for one image workload."""
    return asyncio.run(_measure(model, images, encoding, text_tokens, mm_process_config))
