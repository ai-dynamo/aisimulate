# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real-request repetition ordering without importing GPU frameworks."""

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import dsv41_producer_lifecycle as precedent

path = Path(os.environ["AIC_FPM_GLM53FLASH_PRODUCER"])
spec = importlib.util.spec_from_file_location("glm53flash_tested", path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)
impl.DeepseekV41RealKVScheduler = impl.Glm53FlashRealKVScheduler
precedent.impl = impl


@dataclass
class GraphStats:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int = 0
    runtime_mode: str = "FULL"


def record(self, output, model_output):
    if self._real_callback_stage in {"admission", "measure"}:
        point = self._bench_current_point
        decode = point.point_type == "decode"
        kv = point.total_kv_read_tokens
        if decode and self._real_callback_stage == "admission":
            kv -= point.batch_size
        self._bench_current_fpms.append(
            {
                "wall_time": (self._real_repeat + 1) / 1000,
                "counter_id": point.benchmark_id,
                "dp_rank": 0,
                "scheduled_requests": {
                    "num_prefill_requests": 0 if decode else point.batch_size,
                    "sum_prefill_tokens": point.total_prefill_tokens,
                    "sum_prefill_kv_tokens": 0 if decode else kv,
                    "num_decode_requests": point.batch_size if decode else 0,
                    "sum_decode_kv_tokens": kv if decode else 0,
                },
            }
        )


precedent.Base._update_from_output = record


class Driver(precedent.Driver):
    def finish(self):
        output, snapshots = self.inflight.pop(0)
        for rid, (start, count) in snapshots.items():
            request = self.obj.requests[rid]
            memory = self.memory.setdefault(rid, [])
            assert len(memory) == start, "forward read uninitialized hybrid history"
            tokens = request._all_token_ids[start : start + count]
            assert len(tokens) == count
            memory.extend(tokens)
            if start + count >= len(request.prompt_token_ids):
                request._all_token_ids.append(13 + request.num_output_tokens)
                request.num_output_tokens += 1
                request.num_output_placeholders -= 1
        count = output.total_num_scheduled_tokens
        self.obj._update_from_output(output, SimpleNamespace(cudagraph_stats=GraphStats(count, count)))


for phase, context, suffix in [("prefill", 0, 64), ("prefill", 4099, 3), ("decode", 1024, 0)]:
    point = precedent.point(phase, batch=2, context=context, new=suffix)
    scheduler = precedent.scheduler(point)
    scheduler._real_repeat = 0
    scheduler._real_repetitions = {}
    scheduler._real_dispatches = []
    driver = Driver(scheduler)
    for _ in range(10000):
        output = scheduler._real_step(phase)
        driver.submit(output)
        if driver.inflight:
            driver.finish()
        if scheduler.saved:
            break
    else:
        raise AssertionError("producer did not finish")
    assert len(scheduler.saved) == 1
    assert len(scheduler._real_repetitions[1]) == 15
    histories = [json.loads(row) for row in scheduler._real_token_streams]
    assert [h["sampling_role"] for h in histories] == ["warmup"] * 5 + ["measurement"] * 10
    assert len({r["request_id"] for h in histories for r in h["requests"]}) == 30
    assert math.isclose(scheduler.saved[0][1][-1]["wall_time"], 0.0105, rel_tol=1e-12)
    for repetition in scheduler._real_repetitions[1]:
        assert repetition["completed_seed_tokens"] == point.total_kv_read_tokens - (2 if phase == "decode" else 0)
        assert len(repetition["dispatches"]) == (2 if phase == "decode" else 1)
    # A missing dispatch receipt fails before a latency can be accepted.
    scheduler._real_tags[123] = "measure"
    bad = SimpleNamespace()
    scheduler._real_tags[id(bad)] = "measure"
    try:
        scheduler._update_from_output(bad, SimpleNamespace(cudagraph_stats=None))
    except RuntimeError as error:
        assert "missing actual graph dispatch" in str(error)
    else:
        raise AssertionError("missing dispatch silently accepted")
print("GLM real-state repetition, tail, median and dispatch gates passed")
