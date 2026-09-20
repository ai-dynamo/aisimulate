# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-thread costs of sgl-project/sglang v0.5.19 (`0bcd822`).

`receive` is the per-request preparation the scheduler thread performs between
`recv_requests` and the request joining the waiting queue, which differs by
frontend path:

* Python frontend: `msgpack_decode` of the `TokenizedGenerateReqInput` and
  `unwrap_pickle_fields`; `unwrap_shm_features`, which materializes every
  feature from the tokenizer manager's shared-memory segment;
  `MultimodalInputs.from_processor_output`, whose `set_pad_value` hashes each
  feature with `hash_feature`; and the placeholder padding
  (`MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens`, the
  `pad_input_ids` pattern of the Qwen-VL family).
* Rust frontend: `msgpack_decode` of the columnar header (no multimodal payload),
  `RustMmProcessor.build_output`, which wraps the worker's buffers on the
  scheduler loop, `MultimodalInputs.from_processor_output` with the
  worker-precomputed hash (no `hash_feature`), and the same placeholder padding.
  Single-rank serving keeps features inline; the shared-memory materialization
  of tensor-parallel deployments is not sampled.

Excluded on both paths: the socket or ring receive itself, `Req` construction,
tensor-parallel broadcast. Batch-level costs (batch selection, input
preparation, kernel launches, result processing) depend on the GPU forward and
are read from a measurement taken during a serving run; without one they stay
missing so predictions cannot silently treat them as free.
"""

from __future__ import annotations

import copy
import json
import time
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...config.engine import CostFnConfig, HostPredictionConfig
from .samples import MIN_STEADY_SAMPLES

BATCH_COSTS = (
    "select",
    "prepare_extend",
    "launch_extend",
    "prepare_vision",
    "launch_vision",
    "launch_decode",
    "result",
)


@dataclass(frozen=True)
class ReceiveMeasurement:
    """Scheduler receive cost of one path with the operations it covers."""

    cost: CostFnConfig
    includes: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


def _time_steps(prepare: Callable[[], Any], steps: Sequence[Callable[[Any], Any]]) -> float:
    """Mean wall time of running `steps` in order on a freshly prepared input; warm-up excluded."""
    samples = []
    for _ in range(MIN_STEADY_SAMPLES + 2):
        state = prepare()
        started = time.perf_counter_ns()
        for step in steps:
            state = step(state)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return sum(samples[2:]) / len(samples[2:])


def _processor_output(
    model: str,
    images: Sequence[bytes],
    encoding: str,
    text_tokens: int,
    mm_process_config: dict[str, Any] | None = None,
):
    """The tokenizer manager's output for the workload, produced by the real Python pipeline."""
    import asyncio

    from .python_frontend import _multimodal_processor, _prompt
    from .workload import image_data_url

    server_args, model_config, processor, mm = _multimodal_processor(model, mm_process_config)
    try:
        output = asyncio.run(
            mm.process_mm_data_async(
                image_data=[image_data_url(encoded, encoding) for encoded in images],
                input_text=_prompt(processor, len(images), text_tokens),
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid="calibrate-receive"),
            )
        )
    finally:
        mm.shutdown()
    return server_args, model_config, processor, output


def _pad(input_ids: Sequence[int], mm_inputs) -> array:
    from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens

    return array("q", MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(list(input_ids), mm_inputs))


def _tokenized_request(rid: str, input_ids, mm_inputs):
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


def _measure_python_receive(output) -> ReceiveMeasurement:
    from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
    from sglang.srt.managers.mm_utils import unwrap_shm_features, wrap_shm_features
    from sglang.srt.managers.schedule_batch import MultimodalInputs

    input_ids = list(output.input_ids)

    def prepare():
        # Tokenizer-manager side: shared-memory wrapping and encoding are the sender's work.
        request = _tokenized_request("calibrate-receive", input_ids, copy.deepcopy(output))
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
        return _pad(request.input_ids, mm_inputs)

    cost = _time_steps(prepare, [decode, materialize, hash_and_pad])
    return ReceiveMeasurement(
        cost=CostFnConfig(const_ms=cost),
        includes=[
            "msgpack_decode(TokenizedGenerateReqInput) + unwrap_pickle_fields",
            "unwrap_shm_features (materialize features from shared memory)",
            "MultimodalInputs.from_processor_output (hash_feature per item)",
            "MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens",
        ],
        provenance={"feature_transport": "shm", "items": len(output.mm_items)},
    )


def _rust_entry(output):
    """The worker buffers `RustMmProcessor.build_output` drains, shaped like the pinned `MmEncodeResult`."""
    import torch

    features, grids, hashes, offsets = [], [], [], []
    for item in output.mm_items:
        features.append(item.feature.reshape(item.feature.shape[0], -1).to(torch.float32))
        grids.append(tuple(int(v) for v in item.image_grid_thw.reshape(-1).tolist()))
        hashes.append(int(item.hash) if item.hash is not None else int(torch.randint(1, 1 << 60, ()).item()))
        offsets.append(tuple(int(v) for v in item.offsets[0]))
    return SimpleNamespace(
        shm_names=None,
        features=torch.cat(features).numpy(),
        grids=grids,
        hashes=hashes,
        offsets=offsets,
        mrope=output.mrope_positions.reshape(3, -1).to(torch.int64).numpy(),
        mrope_delta=int(output.mrope_position_delta.reshape(-1)[0]),
    )


def _measure_rust_receive(server_args, model_config, processor, output) -> ReceiveMeasurement:
    from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
    from sglang.srt.managers.schedule_batch import MultimodalInputs
    from sglang.srt.rust_server.multimodal import RustMmProcessor

    spec = RustMmProcessor(server_args=server_args, model_config=model_config, processor=processor).resolve_spec()
    if spec is None:
        raise ValueError(f"{model_config.model_path} has no Rust multimodal pipeline in the pinned SGLang release")
    if spec.feature_shm:
        raise ValueError("the Rust receive path is sampled for single-rank inline features only")
    input_ids = list(output.input_ids)
    entry = _rust_entry(output)
    header = msgpack_encode(_tokenized_request("calibrate-receive", None, None))

    def prepare():
        return SimpleNamespace(header=header, entry=copy.deepcopy(entry))

    def decode_header(state):
        request = msgpack_decode(state.header)
        request.unwrap_pickle_fields()
        return state

    def build(state):
        return RustMmProcessor.build_output(spec, state.entry)

    def pad(processor_output):
        return _pad(input_ids, MultimodalInputs.from_processor_output(processor_output))

    cost = _time_steps(prepare, [decode_header, build, pad])
    return ReceiveMeasurement(
        cost=CostFnConfig(const_ms=cost),
        includes=[
            "msgpack_decode(TokenizedGenerateReqInput header) + unwrap_pickle_fields",
            "RustMmProcessor.build_output (inline features, worker hash)",
            "MultimodalInputs.from_processor_output (precomputed hash, no hash_feature)",
            "MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens",
        ],
        provenance={"feature_transport": "inline", "family": spec.family, "items": len(output.mm_items)},
    )


def measure_receive(
    model: str,
    images: Sequence[bytes],
    encoding: str,
    *,
    frontend: str,
    text_tokens: int = 128,
    mm_process_config: dict[str, Any] | None = None,
) -> ReceiveMeasurement:
    """Per-request scheduler preparation of the given frontend path on one CPU thread."""
    import torch

    torch.set_num_threads(1)
    server_args, model_config, processor, output = _processor_output(
        model, images, encoding, text_tokens, mm_process_config
    )
    if frontend == "python":
        return _measure_python_receive(output)
    if frontend == "rust":
        return _measure_rust_receive(server_args, model_config, processor, output)
    raise ValueError(f"unknown frontend {frontend!r}")


def batch_costs(path: str | Path | None) -> tuple[dict[str, CostFnConfig], list[str]]:
    """Operator-measured batch-level costs; the keys not provided are reported missing."""
    measured: dict[str, CostFnConfig] = {}
    if path is not None:
        raw = json.loads(Path(path).read_text())
        for name in BATCH_COSTS:
            if name in raw:
                measured[name] = CostFnConfig.model_validate(raw[name])
    return measured, [name for name in BATCH_COSTS if name not in measured]


def host_from_costs(receive: CostFnConfig, batch: dict[str, CostFnConfig]) -> HostPredictionConfig:
    return HostPredictionConfig(receive=receive, **batch)
