# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle of the default-serving (prefix-cache) GLM vLLM FPM producer.

A small fake of the native scheduler admits every injected request in one step,
serves prefix-cache hits on the 4-token grid and records Dynamo-shaped FPMs.
This checks the producer's state machine, validation and evidence; it is not
hardware or timing evidence.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PRODUCER = (
    Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime/glm53flash/glm53flash_prefix_scheduler.py"
)


@dataclass
class Point:
    point_type: str
    benchmark_id: int
    batch_size: int
    total_prefill_tokens: int
    total_kv_read_tokens: int
    rows: list | None = None
    partition: dict | None = None
    sample_reasons: list = field(default_factory=lambda: ["explicit"])


@dataclass
class Metrics:
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0


@dataclass
class Stats:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


class Request:
    def __init__(self, request_id, prompt_token_ids, sampling_params, pooling_params, block_hasher, cache_salt):
        self.request_id = request_id
        self.prompt = list(prompt_token_ids)
        self.max_tokens = sampling_params.max_tokens
        self.salt = cache_salt
        self.computed = 0
        self.outputs = 0


class FakeBase:
    """Subset of Dynamo InstrumentedScheduler + vLLM Scheduler used by the producer."""

    def _bench_init(self, config):  # pragma: no cover - the test builds state directly
        raise AssertionError

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def _bench_stop_at_timeout_boundary(self, point_type):
        return False

    def _bench_pop_next(self, point_type):
        if self._bench_grid and self._bench_grid[0].point_type == point_type:
            return self._bench_grid.popleft()
        return None

    def _bench_cleanup_requests(self):
        for rid in list(self._bench_active_req_ids):
            self.requests.pop(rid, None)
        self._bench_active_req_ids.clear()

    @staticmethod
    def _bench_decode_context_lengths(total, batch):
        quotient, remainder = divmod(total - batch, batch)
        return [1 + quotient + int(i < remainder) for i in range(batch)]

    @staticmethod
    def _bench_prefill_new_token_lengths(total, batch, partition=None, rows=None):
        return [int(new) for new, _ in rows]

    @staticmethod
    def _bench_fpm_validation_failure(point, fpm):
        s = fpm["scheduled_requests"]
        if point.point_type == "prefill":
            if (s["num_prefill_requests"], s["sum_prefill_tokens"], s["sum_prefill_kv_tokens"]) != (
                point.batch_size,
                point.total_prefill_tokens,
                point.total_kv_read_tokens,
            ):
                return "prefill mismatch"
        elif (s["num_decode_requests"], s["sum_decode_kv_tokens"]) != (point.batch_size, point.total_kv_read_tokens):
            return "decode mismatch"
        return None

    def _bench_save_current_point(self):
        fpms = self._bench_current_fpms[-1:]
        self.saved.append((self._bench_current_point, fpms))

    def _update_from_output(self, output, model_output):
        metrics = output.metrics
        self.clock += 1
        if self._bench_should_record_scheduled(metrics):
            self._bench_current_fpms.append(
                {
                    "counter_id": self._bench_current_point.benchmark_id,
                    "dp_rank": 0,
                    "wall_time": self.wall(output),
                    "scheduled_requests": metrics.__dict__.copy(),
                }
            )


def load_producer(monkeypatch):
    native = types.ModuleType("dynamo.vllm.instrumented_scheduler")
    native.InstrumentedScheduler = FakeBase
    native._BenchPhase = types.SimpleNamespace(DECODE_SWEEP="decode", DONE="done")
    modules = {
        "dynamo": types.ModuleType("dynamo"),
        "dynamo.vllm": types.ModuleType("dynamo.vllm"),
        "dynamo.vllm.instrumented_scheduler": native,
        "vllm": types.ModuleType("vllm"),
        "vllm.sampling_params": types.SimpleNamespace(SamplingParams=lambda **kw: types.SimpleNamespace(**kw)),
        "vllm.tokenizers": types.SimpleNamespace(get_tokenizer=None),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.request": types.SimpleNamespace(
            Request=Request, RequestStatus=types.SimpleNamespace(PREEMPTED="preempted", WAITING_FOR_REMOTE_KVS="remote")
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("glm53flash_prefix_tested", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Engine:
    """Admits every waiting request in one step; 4-token prefix-cache grid."""

    def __init__(self, scheduler, *, unit=4, first_miss=()):
        self.s = scheduler
        self.unit = unit
        self.cache = collections.defaultdict(set)  # salt -> cached prefix tuples
        self.first_miss = set(first_miss)

    def hit(self, request):
        limit = (len(request.prompt) - 1) // self.unit * self.unit
        for length in range(limit, 0, -self.unit):
            if tuple(request.prompt[:length]) in self.cache[request.salt]:
                if request.salt in self.first_miss:
                    self.first_miss.discard(request.salt)
                    return 0
                return length
        return 0

    def step(self):
        s = self.s
        metrics = Metrics()
        active = [r for r in s.requests.values() if r.outputs < r.max_tokens]
        if not active:
            return False
        new = [r for r in active if r in s.waiting]
        new_reqs, cached, scheduled = [], types.SimpleNamespace(req_ids=[], num_computed_tokens=[]), {}
        if new:
            for r in new:
                r.computed = self.hit(r)
                new_reqs.append(types.SimpleNamespace(req_id=r.request_id, num_computed_tokens=r.computed))
                scheduled[r.request_id] = len(r.prompt) - r.computed
                metrics.num_prefill_requests += 1
                metrics.sum_prefill_tokens += len(r.prompt) - r.computed
                metrics.sum_prefill_kv_tokens += r.computed
            s.waiting.clear()
            for r in new:
                r.computed = len(r.prompt)
                r.outputs = 1
                for length in range(self.unit, len(r.prompt) + 1, self.unit):
                    self.cache[r.salt].add(tuple(r.prompt[:length]))
            stepped = new
        else:
            for r in active:
                cached.req_ids.append(r.request_id)
                cached.num_computed_tokens.append(r.computed)
                scheduled[r.request_id] = 1
                metrics.num_decode_requests += 1
                metrics.sum_decode_kv_tokens += r.computed
                r.computed += 1
                r.outputs += 1
            stepped = active
        tokens = metrics.sum_prefill_tokens + metrics.num_decode_requests
        out = types.SimpleNamespace(
            total_num_scheduled_tokens=tokens,
            metrics=metrics,
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=scheduled,
        )
        model = types.SimpleNamespace(cudagraph_stats=Stats(tokens, tokens, 0, "FULL"))
        s._update_from_output(out, model)
        for r in stepped:
            if r.outputs >= r.max_tokens:
                s.requests.pop(r.request_id, None)
        return True


def make_scheduler(module, tmp_path, points):
    s = module.Glm53FlashPrefixSeedScheduler.__new__(module.Glm53FlashPrefixSeedScheduler)
    s.requests, s.waiting, s.saved, s.clock = {}, [], [], 0
    s._bench_active_req_ids, s._bench_seq = set(), 0
    s._bench_current_point, s._bench_current_fpms = None, []
    s._bench_grid = collections.deque(points)
    s._bench_phase = "prefill"
    s._bench_block_hasher = None
    s._bench_config = types.SimpleNamespace(output_path=str(tmp_path / "benchmark.json"), mode=points[0].point_type)
    s._glm_point = s._glm_stage = s._glm_seed = s._glm_rows = None
    s._glm_repetitions, s._glm_rejected, s._glm_point_records = [], [], {}
    s._glm_evidence_count = 0
    s._glm_step_outputs, s._glm_prompt_lengths = [], {}
    s._glm_evidence_digest = module.hashlib.sha256()
    s._glm_request_set = "test"
    s._glm_tokens = list(range(100, 197))
    s._glm_serving = {"hash_block_size": 4}
    s._glm_stage_deadline = 0.0
    s.wall = lambda output: 0.001 * (1 + output.metrics.num_prefill_requests) + 1e-6 * s.clock
    return s


def run(scheduler, engine, point_type, limit=10000):
    for _ in range(limit):
        scheduler._glm_step(point_type)
        if scheduler._glm_point is None and not scheduler._bench_grid:
            return
        engine.step()
    raise AssertionError("producer did not finish")


def test_prefill_prefix_seed_lifecycle(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    points = [
        Point("prefill", 1, 2, 64, 0, rows=[[32, 0], [32, 0]]),
        Point("prefill", 2, 2, 64, 240, rows=[[32, 120], [32, 120]]),
    ]
    s = make_scheduler(module, tmp_path, points)
    # The first hit of slot 0's seeded prefix misses once (as native vLLM does for
    # block-multiple prefixes); that repetition must be rejected and replaced.
    engine = Engine(s, first_miss={"test-p2-s0"})
    run(s, engine, "prefill")
    assert [p.benchmark_id for p, _ in s.saved] == [1, 2]
    for point, fpms in s.saved:
        assert len(fpms) == 1 and fpms[0]["scheduled_requests"]["sum_prefill_kv_tokens"] == point.total_kv_read_tokens
    records = [json.loads(line) for line in (tmp_path / "benchmark.repetitions.jsonl").read_text().splitlines()]
    assert [len(r["repetitions"]) for r in records] == [15, 15]
    assert len(records[1]["rejected_repetitions"]) == 1
    assert "prefill_real_seed" in points[1].sample_reasons
    seeded = records[1]["seed"]
    assert seeded["lengths"] == [120, 120] and seeded["specs"][0] == [[17 * 2, 120]]
    # Prompts are reconstructible from specs and every repetition is distinct.
    pool = s._glm_tokens
    prompts = {tuple(module.build_prompt(pool, rep["prompt_specs"][0])) for rep in records[1]["repetitions"]}
    assert len(prompts) == 15 and all(len(p) == 152 for p in prompts)
    measured = [rep["fpms"][0]["wall_time"] for rep in records[1]["repetitions"] if rep["role"] == "measurement"]
    assert s.saved[1][1][0]["wall_time"] == pytest.approx(sorted(measured)[4:6][0] / 2 + sorted(measured)[4:6][1] / 2)


@pytest.mark.parametrize(("context", "seed"), [(201, 196), (200, 196), (198, 196), (197, 192)])
def test_decode_steady_step_lifecycle(monkeypatch, tmp_path, context, seed):
    module = load_producer(monkeypatch)
    points = [Point("decode", 1, 3, 0, 3 * context)]
    s = make_scheduler(module, tmp_path, points)
    run(s, Engine(s), "decode")
    ((point, fpms),) = s.saved
    assert fpms[0]["scheduled_requests"]["sum_decode_kv_tokens"] == 3 * context
    record = json.loads((tmp_path / "benchmark.repetitions.jsonl").read_text())
    # True context: prompt context - 1; the cached seed (measured chunk start) stays on the 4-token grid.
    assert record["rows"]["prompt"] == [context - 1] * 3 and record["rows"]["measured_decode_step"] == 2
    assert record["seed"]["lengths"] == [seed] * 3
    assert all(len(rep["fpms"]) == 2 for rep in record["repetitions"])
    assert all(
        rep["fpms"][0]["scheduled_requests"]["sum_decode_kv_tokens"] == 3 * (context - 1)
        for rep in record["repetitions"]
    )
    assert all(rep["prefill_chunks"] == [[[seed, context - 1 - seed]] * 3] for rep in record["repetitions"])
    assert record["seed"]["prefill_chunks"] == [[[0, seed]] * 3]
    assert not record["rejected_repetitions"]


def test_unaligned_prefill_chunk_start_fails_the_run(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    points = [Point("prefill", 1, 1, 32, 120, rows=[[32, 120]])]
    s = make_scheduler(module, tmp_path, points)
    engine = Engine(s)
    engine.hit = lambda request: 118 if len(request.prompt) == 152 else 0
    with pytest.raises(RuntimeError, match="not divisible by 4"):
        run(s, engine, "prefill")
    assert not s.saved


def test_mismatched_geometry_is_never_published(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    points = [Point("prefill", 1, 1, 32, 120, rows=[[32, 120]])]
    s = make_scheduler(module, tmp_path, points)
    engine = Engine(s, unit=8)  # hits land on an 8-token grid: 120 -> 120 ok, force misses below
    engine.hit = lambda request: 116  # never the planned prefix
    with pytest.raises(RuntimeError, match="too many rejected repetitions"):
        run(s, engine, "prefill")
    assert not s.saved


def test_explicit_align4_decode_contexts_override_the_even_split(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    contexts = tmp_path / "decode-contexts.json"
    contexts.write_text(json.dumps([[2, 246, [125, 121]]]))  # TEST ONLY uneven split
    monkeypatch.setenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", str(contexts))
    monkeypatch.setattr(module, "_DECODE_CONTEXTS", None)
    s = make_scheduler(module, tmp_path, [Point("decode", 1, 2, 0, 246)])
    run(s, Engine(s), "decode")
    ((point, fpms),) = s.saved
    assert fpms[0]["scheduled_requests"]["sum_decode_kv_tokens"] == 246
    record = json.loads((tmp_path / "benchmark.repetitions.jsonl").read_text())
    assert record["rows"] == {"context": [125, 121], "prompt": [124, 120], "measured_decode_step": 2}
    assert record["seed"]["lengths"] == [120, 116]
    assert all(rep["prefill_chunks"] == [[[120, 4], [116, 4]]] for rep in record["repetitions"])
    with pytest.raises(ValueError, match="no frozen per-request contexts"):
        module.Glm53FlashPrefixSeedScheduler._bench_decode_context_lengths(250, 2)
