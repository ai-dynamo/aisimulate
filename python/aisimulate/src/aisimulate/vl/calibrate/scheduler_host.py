# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-thread costs of sgl-project/sglang v0.5.19 (`0bcd822`).

`receive` runs the real per-request preparation the scheduler performs when it
takes a request off its inbox: the feature hash (`mm_utils.hash_feature`) over
the processor output and the placeholder padding. Batch-level costs (batch
selection, kernel launches, result processing) depend on the GPU forward and
are read from a measurement taken during a serving run; without one they stay
missing so predictions cannot silently treat them as free.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from ...config.engine import CostFnConfig, HostPredictionConfig
from .samples import MIN_STEADY_SAMPLES

BATCH_COSTS = ("select", "launch_extend", "launch_vision", "launch_decode", "result")


def measure_receive(model: str, images: Sequence[bytes], encoding: str, *, text_tokens: int = 128) -> CostFnConfig:
    """Per-request scheduler preparation on one CPU thread."""
    import torch
    from sglang.srt.managers.mm_utils import hash_feature
    from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
    from sglang.srt.managers.schedule_batch import Modality
    from sglang.srt.multimodal.transport import determine_tensor_transport_mode

    from .python_frontend import _prompt, _serving_components
    from .workload import image_data_url

    torch.set_num_threads(1)
    server_args, model_config, processor = _serving_components(model)
    import_processors("sglang.srt.multimodal.processors")
    mm = get_mm_processor(
        model_config.hf_config, server_args, processor, determine_tensor_transport_mode(), model_config=model_config
    )
    try:
        loaded = [mm._load_single_item(image_data_url(encoded, encoding), Modality.IMAGE) for encoded in images]
        output = mm.process_mm_data(_prompt(processor, len(images), text_tokens), images=loaded)
    finally:
        mm.shutdown()
    features = [item.feature for item in output.mm_items if getattr(item, "feature", None) is not None]
    samples = []
    for _ in range(MIN_STEADY_SAMPLES + 2):
        started = time.perf_counter_ns()
        for feature in features:
            hash_feature(feature)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return CostFnConfig(const_ms=sum(samples[2:]) / len(samples[2:]))


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
