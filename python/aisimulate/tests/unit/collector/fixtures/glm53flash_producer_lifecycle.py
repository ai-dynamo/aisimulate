# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real-request repetition ordering without importing GPU frameworks."""

import hashlib
import importlib.util
import json
import math
import os
import tempfile
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
    scheduler._real_request_set = "fixture-run"
    scheduler._real_request_manifest = {}
    scheduler._real_input = {"text_sha256": "a" * 64}
    manifest_dir = tempfile.TemporaryDirectory()
    manifest_path = Path(manifest_dir.name) / "requests.json"
    os.environ["AISIM_GLM53_REQUEST_MANIFEST"] = str(manifest_path)
    scheduler._real_token_stream_count = 0
    scheduler._real_token_stream_digest = hashlib.sha256()
    scheduler._bench_config.output_path = str(Path(manifest_dir.name) / "benchmark.json")
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
    stream_path = Path(scheduler._bench_config.output_path).with_suffix(".token-streams.jsonl")
    histories = [json.loads(row) for row in stream_path.read_bytes().splitlines()]
    assert scheduler._real_token_stream_count == 15
    assert hashlib.sha256(stream_path.read_bytes()).hexdigest() == scheduler._real_token_stream_digest.hexdigest()
    assert scheduler._real_token_streams == [], "must not accumulate completed prompt arrays"
    # A fresh attempt cannot silently overwrite a previous attempt's evidence.
    scheduler._real_token_stream_count = 0
    try:
        scheduler._append_real_token_history(b"{}")
    except FileExistsError:
        pass
    else:
        raise AssertionError("old raw evidence was overwritten")
    scheduler._real_token_stream_count = 15
    scheduler._real_identity = {}
    scheduler._real_purpose = "fpm"
    for _ in range(2):
        scheduler._bench_write_results()
        published = json.loads(Path(scheduler._bench_config.output_path).read_text())
        receipt = published["input_provenance"]["token_stream_manifest"]
        assert receipt["records"] == 15
        assert receipt["sha256"] == hashlib.sha256(stream_path.read_bytes()).hexdigest()
    with stream_path.open("ab") as destination:
        destination.write(b"{}\n")
    try:
        scheduler._bench_write_results()
    except RuntimeError as error:
        assert "changed before publication" in str(error)
    else:
        raise AssertionError("modified raw history was published")
    assert stream_path.read_bytes().endswith(b"{}\n"), "failed raw evidence must survive"
    assert [h["sampling_role"] for h in histories] == ["warmup"] * 5 + ["measurement"] * 10
    assert len({r["request_id"] for h in histories for r in h["requests"]}) == 30
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["requests"]) == 30
    assert manifest["corpus_sha256"] == "a" * 64
    for entry in manifest["requests"].values():
        assert entry["target_prefix"] == context
        assert entry["target_phase"] == ("generation" if phase == "decode" else "context")
        assert entry["sampling_role"] == ("warmup" if entry["repetition"] < 5 else "measurement")
    del os.environ["AISIM_GLM53_REQUEST_MANIFEST"]
    manifest_dir.cleanup()
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
candidate = precedent.point("prefill", batch=1, context=4096, new=3)
scheduler = precedent.scheduler(candidate)
scheduler._bench_hash_block_size = 4352
scheduler._bench_capacity_limit = {
    "max_num_running_reqs": 4,
    "max_num_scheduled_tokens": 8192,
    "max_model_len": 131072,
}.__getitem__
scheduler._bench_grid_usable_blocks = lambda batch: 4067


class NativeManager:
    def __init__(self, recurrent):
        self.recurrent = recurrent

    def get_num_blocks_to_allocate(self, **kwargs):
        assert kwargs["new_computed_blocks"] == []
        assert not kwargs["apply_admission_cap"]
        return 1 if self.recurrent else math.ceil(kwargs["num_tokens"] / 4352)


scheduler.kv_cache_manager.coordinator = SimpleNamespace(
    single_type_managers=[NativeManager(False), NativeManager(True)]
)
assert scheduler._bench_prefill_kv_read_lengths(8195, 2, rows=[[3, 4096], [3, 4099]]) == [4096, 4099]
assert scheduler._bench_prefill_kv_read_lengths(4096, 1) == [4096]
assert scheduler._bench_prefill_point_feasible(3, 1, 4096)
assert scheduler._bench_prefill_point_feasible(3, 1, 4099)
assert scheduler._bench_prefill_point_feasible(8192, 1, 122880)
assert not scheduler._bench_prefill_point_feasible(8192, 1, 122881)
assert not scheduler._bench_prefill_point_feasible(8193, 1, 0)
scheduler._bench_grid_usable_blocks = lambda batch: 1
assert not scheduler._bench_prefill_point_feasible(3, 1, 4096)

for prefix, admitted in [(131071, True), (131072, False)]:
    scheduler = precedent.scheduler(precedent.point("decode", batch=1, context=prefix))
    try:
        scheduler._real_validate_grid()
    except ValueError:
        assert not admitted
    else:
        assert admitted
print("GLM real-state repetition, unaligned-tail capacity, inclusive-context, median and dispatch gates passed")
