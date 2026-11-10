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
import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from collector import glm53flash_attention_tokens as seeded
from collector.fpm_forward.glm53flash_points import glm_align_stop

from aisimulate_core.sdk import utils as sdk_utils

pytestmark = pytest.mark.unit

PRODUCER = (
    Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime/glm53flash/glm53flash_prefix_scheduler.py"
)
MANIFEST = PRODUCER.with_name("runtime-source-sha256.json")
FP8_MODEL, NVFP4_MODEL = "zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4"
FP8_PIN, NVFP4_PIN = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
# A revision that is not a pin, whose files equal the pinned FP8 files.
LOADED_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"


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

    def _bench_build_grid(self):
        # Dynamo 395f0240 materializes the manifest and then plans native warm-up stages (instrumented_scheduler:2718).
        if self._bench_grid_built:
            return
        self._bench_grid_built = True
        self._bench_grid = collections.deque(self._bench_explicit_points)
        self._kvwarm_prepare(self._bench_config.mode)

    def _kvwarm_prepare(self, mode):
        self.warmup_planned.append(mode)

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


class BudgetEngine(Engine):
    """Schedules like vLLM v0.31.0 with a token budget per step, on hash blocks of ``unit`` tokens.

    Running requests go first, then waiting requests in order (scheduler.py:630, :1073). A prefill chunk that ends
    before the prompt end is cut to a block boundary when the block fits the budget. Mamba align mode of vLLM 0.33.0
    also stops a chunk at the last cacheable block boundary before the prompt end and, after a start inside a block, at
    the next boundary. A prompt's last chunk that can save a prefill checkpoint of ``alignment`` inside it takes neither
    stop (vLLM 0.33.0 scheduler.py:579-606, :644-661, kv_cache_interface.py:1169-1210). A chunk that the cut empties
    waits: a running request is skipped, and a waiting one ends the admissions of the step (scheduler.py:473-480,
    :724-742, :1134-1135). vLLM discards the token sampled after a partial prefill (gpu_model_runner.py:2089-2091,
    :3719-3722), so a request has no output token until its prompt is complete. Until then Dynamo counts it as a prefill
    request (output.py:189-191, instrumented_scheduler.py:2078-2085 at ai-dynamo/dynamo@395f0240).
    """

    def __init__(self, scheduler, *, budget, unit, alignment=None):
        super().__init__(scheduler, unit=unit)
        self.budget = budget
        self.alignment = alignment

    def chunk(self, request, tokens):
        start, prompt, unit = request.computed, len(request.prompt), self.unit
        end = start + tokens
        # Without hash lookahead tokens and EAGLE block drop, the checkpoint position is the last cache position.
        last = (prompt - 1) // unit * unit
        internal = (
            self.alignment is not None
            and end >= prompt
            and start % unit == 0
            and -(-end // unit) - 2 > (start - 1) // unit
            and start + unit <= last
            and start < last < end
            and (last - start) % self.alignment == 0
        )
        if end < prompt and unit <= self.budget:
            end = end // unit * unit
        stops = (0 if internal else last, (start // unit + 1) * unit if start % unit and not internal else 0)
        end = min((stop for stop in stops if start < stop < end), default=end)
        return max(end - start, 0)

    def step(self):
        s = self.s
        active = [r for r in s.requests.values() if r.outputs < r.max_tokens]
        if not active:
            return False
        budget, stepped, new_reqs = self.budget, [], []
        cached = types.SimpleNamespace(req_ids=[], num_computed_tokens=[])
        for r in [r for r in active if r not in s.waiting]:
            tokens = min(1, budget) if r.outputs else self.chunk(r, min(len(r.prompt) - r.computed, budget))
            if tokens:
                cached.req_ids.append(r.request_id)
                cached.num_computed_tokens.append(r.computed)
                stepped.append((r, tokens))
                budget -= tokens
        for r in [r for r in s.waiting if r in active]:
            r.computed = self.hit(r)
            tokens = self.chunk(r, min(len(r.prompt) - r.computed, budget))
            if not tokens:
                break
            s.waiting.remove(r)
            new_reqs.append(types.SimpleNamespace(req_id=r.request_id, num_computed_tokens=r.computed))
            stepped.append((r, tokens))
            budget -= tokens
        metrics, scheduled = Metrics(), {}
        for r, tokens in stepped:
            scheduled[r.request_id] = tokens
            if r.outputs:
                metrics.num_decode_requests += 1
                metrics.sum_decode_kv_tokens += r.computed
            else:
                metrics.num_prefill_requests += 1
                metrics.sum_prefill_tokens += tokens
                metrics.sum_prefill_kv_tokens += r.computed
        for r, tokens in stepped:
            r.computed += tokens
            if r.outputs:
                r.outputs += 1
            elif r.computed == len(r.prompt):
                r.outputs = 1
                for length in range(self.unit, len(r.prompt) + 1, self.unit):
                    self.cache[r.salt].add(tuple(r.prompt[:length]))
        total = sum(scheduled.values())
        out = types.SimpleNamespace(
            total_num_scheduled_tokens=total,
            metrics=metrics,
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=scheduled,
        )
        s._update_from_output(out, types.SimpleNamespace(cudagraph_stats=Stats(total, total, 0, "FULL")))
        for r, _ in stepped:
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
    s._bench_grid_built, s.warmup_planned = False, []
    s._glm_point = s._glm_stage = s._glm_seed = s._glm_rows = None
    s._glm_repetitions, s._glm_rejected, s._glm_point_records = [], [], {}
    s._glm_evidence_count = 0
    s._glm_step_outputs, s._glm_prompt_lengths = [], {}
    s._glm_evidence_digest = module.hashlib.sha256()
    s._glm_request_set = "test"
    s._glm_tokens = list(range(100, 197))
    s._glm_serving = {"hash_block_size": 4}
    # vLLM's scheduler configuration with the token budget of the onboarding request, not an unlimited one.
    s.scheduler_config = types.SimpleNamespace(max_num_batched_tokens=8192)
    # With prefix caching, vLLM runs GLM-5.3-Flash in Mamba align mode. The block is the hash block of this fake.
    s.need_mamba_block_aligned_split = True
    s.cache_config = types.SimpleNamespace(block_size=4)
    # The launch state that vLLM 0.33.0's scheduler reads in its align-mode chunk rule: no EAGLE block drop, no hash
    # lookahead tokens, no partial hash hits and no prefill checkpoint block (scheduler.py:218, :331, :342, :350,
    # :437-456).
    s.block_size = s.hash_block_size = 4
    s.use_eagle_block_drop = False
    s.num_hash_lookahead_tokens = 0
    s.mamba_partial_cache_hit = False
    s.mamba_has_prefill_checkpoint_blocks = False
    s.mamba_prefill_checkpoint_alignment = None
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


def _parallel(tp, **overrides):
    fields = dict(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        use_ubatching=False,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
    )
    return types.SimpleNamespace(**(fields | overrides))


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_pure_tp_admission_is_capacity_neutral(monkeypatch, tp):
    # Both checkpoints share one TP set; fitting a system is decided at runtime.
    module = load_producer(monkeypatch)
    assert module.pure_tensor_parallel(_parallel(tp), None)


@pytest.mark.parametrize(
    ("tp", "overrides", "speculative"),
    [
        (3, {}, None),
        (16, {}, None),
        (4, {"data_parallel_size": 2}, None),
        (4, {"pipeline_parallel_size": 2}, None),
        (4, {"enable_expert_parallel": True}, None),
        (4, {"use_ubatching": True}, None),
        (4, {"decode_context_parallel_size": 2}, None),
        (4, {}, object()),
    ],
)
def test_non_pure_tp_layouts_are_rejected(monkeypatch, tp, overrides, speculative):
    module = load_producer(monkeypatch)
    assert not module.pure_tensor_parallel(_parallel(tp, **overrides), speculative)


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


def test_publication_enrichment_needs_only_the_fpm_measurement(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.setattr(sys.modules["vllm"], "__version__", "0.31.0", raising=False)
    points = [Point("prefill", 1, 2, 64, 0, rows=[[32, 0], [32, 0]])]
    s = make_scheduler(module, tmp_path, points)
    run(s, Engine(s), "prefill")
    s._glm_input = {"source": "seeded_tokens", "seed": 53}
    s._glm_context_policy = {"measured_context_limit": 131072}
    s._glm_identity = {"model_config_sha256": "test"}
    output = {
        "results": [{"point": {"point_type": "prefill", "total_kv_read_tokens": 0, "benchmark_id": 1}}],
        "iteration_groups": [],
    }
    s._glm_enrich(output, tmp_path / "benchmark.json")
    assert output["input_provenance"]["seed"] == 53
    assert output["input_provenance"]["repetition_evidence_manifest"]["records"] == s._glm_evidence_count
    assert output["producer"]["vllm_package_version"] == "0.31.0"
    # The derived code keeps its revision. The scheduler file that ran is named separately.
    assert output["producer"]["instrumentation_revision"] == "54960177085413259859c88bd34ed0734d4c2ea9"
    assert output["producer"]["runtime_scheduler_revision"] == "395f02405c1dd1835ab73a39f7a5984ae5fd4552"
    assert output["producer"]["runtime_distribution"] == "ai-dynamo 1.6.0.dev20261010"
    assert output["results"][0]["kv_seed_regime"] == "not_applicable"


def test_dynamo_runtime_pins_match_the_ai_dynamo_1_6_0_dev20261010_build():
    pins = json.loads(MANIFEST.read_text())
    # The ai-dynamo 1.6.0.dev20261010 build carries these four files of ai-dynamo/dynamo@395f0240.
    expected = {
        "dynamo/vllm/instrumented_scheduler.py": "06fcb26ffcfeb4d58a01704d8d7842d9d838d4a34170a1086f09e5617442da8e",
        "dynamo/vllm/benchmark_points.py": "bdc6f2f0bb222498e8542859ed2b4f950d1949868f02f50aaee8199257172f91",
        "dynamo/vllm/gc_policy.py": "4221e5a20718525559ad4c9e9bee34928ee22dcf2cfdf8036936be2a360153a4",
        "dynamo/common/forward_pass_metrics.py": "523dfc4e7ccbc5a37dbcf03b331a41b628227ad8a5d9c64fda8abc96dd31b188",
    }
    assert {name: pins[name] for name in expected} == expected


def test_native_warmup_planning_never_runs_under_the_producer(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    # One block of new tokens, which vLLM's Mamba align mode runs in one step without a prefill checkpoint.
    points = [Point("prefill", 1, 1, 4, 0, rows=[[4, 0]])]
    s = make_scheduler(module, tmp_path, points)
    s._bench_explicit_points = points
    s._bench_build_grid()
    # The native grid build ran. The producer seeds every point through its own stages, so the native warm-up
    # planning that the grid build calls must not run.
    assert s._bench_grid_built
    assert s.warmup_planned == []


def test_explicit_decode_points_stay_within_the_measured_context(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.delenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", raising=False)

    def build(total_kv_read_tokens):
        points = [Point("decode", 1, 1, 0, total_kv_read_tokens)]
        s = make_scheduler(module, tmp_path, points)
        # A run that serves a 1,048,576-token context and measures up to 131,072 tokens. The native decode check
        # bounds a context by the runtime context only.
        s._glm_context_policy = module.glm_context_policy(1048576, 131072)
        s._bench_explicit_points = points
        s._bench_build_grid()
        return s

    assert [point.total_kv_read_tokens for point in build(131072)._bench_grid] == [131072]
    # The artifact states the measured context as kvwarm.max_context, so a deeper point is refused.
    with pytest.raises(ValueError) as refused:
        build(262144)
    assert str(refused.value) == "GLM-5.3-Flash decode context 262144 exceeds the measured context limit 131072"


def test_explicit_decode_points_stay_within_the_producer_batch_limit(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.delenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", raising=False)

    def build(batch_size):
        points = [Point("decode", 1, batch_size, 0, batch_size * 4096)]
        s = make_scheduler(module, tmp_path, points)
        s._glm_context_policy = module.glm_context_policy(1048576, 131072)
        s._bench_explicit_points = points
        s._bench_build_grid()
        return s

    assert [point.batch_size for point in build(32)._bench_grid] == [32]
    # The artifact states the producer's batch limit as kvwarm.max_batch, and the native decode check allows a batch up
    # to the running-request limit, so the producer refuses a larger batch.
    with pytest.raises(ValueError) as refused:
        build(33)
    assert str(refused.value) == "GLM-5.3-Flash decode batch 33 exceeds the producer limit 32"


def test_explicit_decode_points_prefill_their_suffixes_in_one_step(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.delenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", raising=False)

    def build(batch_size, budget=None):
        points = [Point("decode", 1, batch_size, 0, batch_size * 4096)]
        s = make_scheduler(module, tmp_path, points)
        # The native hash block of GLM-5.3-Flash at TP4. Each 4,095-token prompt has a seed of 2,176 tokens, so each
        # request prefills 1,919 tokens before its first decode step.
        s._glm_serving = {"hash_block_size": 2176}
        if budget is not None:
            s.scheduler_config.max_num_batched_tokens = budget
        s._glm_context_policy = module.glm_context_policy(1048576, 131072)
        s._bench_explicit_points = points
        s._bench_build_grid()
        return s

    accepted = build(4)
    assert accepted.scheduler_config.max_num_batched_tokens == 8192
    assert [point.batch_size for point in accepted._bench_grid] == [4]
    # vLLM would prefill the suffixes of eight requests over two steps, so no repetition would start with the decode
    # step of all eight requests, and the producer refuses the point before it seeds anything.
    with pytest.raises(ValueError) as refused:
        build(8)
    assert str(refused.value) == (
        "GLM-5.3-Flash decode point of 8 requests at context 4096 needs 15352 suffix tokens in one step, "
        "above the token budget 8192"
    )
    # The budget is the one that vLLM's scheduler configuration holds.
    assert [point.batch_size for point in build(8, budget=16384)._bench_grid] == [8]
    with pytest.raises(ValueError) as refused:
        build(2, budget=3836)
    assert str(refused.value) == (
        "GLM-5.3-Flash decode point of 2 requests at context 4096 needs 3838 suffix tokens in one step, "
        "above the token budget 3836"
    )
    # One request may prefill its suffix over several steps, because no other request can decode before it.
    assert [point.batch_size for point in build(1, budget=1024)._bench_grid] == [1]


def chunked_decode(module, tmp_path, batch_size, budget):
    """A decode point at context 250 on 64-token hash blocks, scheduled with a token budget.

    Each request has a 249-token prompt and a 192-token seed, so it prefills a suffix of 57 tokens before it decodes.
    """
    points = [Point("decode", 1, batch_size, 0, batch_size * 250)]
    s = make_scheduler(module, tmp_path, points)
    s._glm_serving = {"hash_block_size": 64}
    s.scheduler_config.max_num_batched_tokens = budget
    return s, BudgetEngine(s, budget=budget, unit=64)


def test_one_request_records_only_its_decode_steps_after_a_chunked_suffix(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.delenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", raising=False)
    # The 16-token budget is below the 64-token block, so vLLM prefills the 57-token suffix in four chunks.
    s, engine = chunked_decode(module, tmp_path, 1, 16)
    run(s, engine, "decode")

    ((point, fpms),) = s.saved
    record = json.loads((tmp_path / "benchmark.repetitions.jsonl").read_text())
    assert not record["rejected_repetitions"] and len(record["repetitions"]) == 15
    for repetition in record["repetitions"]:
        assert repetition["prefill_chunks"] == [[[192, 16]], [[208, 16]], [[224, 16]], [[240, 9]]]
        # The producer records no step with a prefill request, so the first recorded step is the first decode step,
        # at the full prompt, and the second one is the measured step.
        recorded = [
            (step["num_prefill_requests"], step["num_decode_requests"], step["sum_decode_kv_tokens"])
            for step in (fpm["scheduled_requests"] for fpm in repetition["fpms"])
        ]
        assert recorded == [(0, 1, 249), (0, 1, 250)]
    assert fpms[0]["scheduled_requests"]["sum_decode_kv_tokens"] == 250


def test_two_requests_whose_suffixes_exceed_the_budget_are_never_in_lockstep(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    monkeypatch.delenv("DYN_FPM_GLM53FLASH_DECODE_CONTEXTS", raising=False)
    # Two suffixes of 57 tokens need 114 tokens. A budget of 116 prefills both in one step, so they decode together.
    s, engine = chunked_decode(module, tmp_path, 2, 116)
    run(s, engine, "decode")
    record = json.loads((tmp_path / "benchmark.repetitions.jsonl").read_text())
    assert not record["rejected_repetitions"]
    for repetition in record["repetitions"]:
        assert repetition["prefill_chunks"] == [[[192, 57], [192, 57]]]
        assert repetition["fpms"][0]["scheduled_requests"]["sum_decode_kv_tokens"] == 498

    # A budget of 112 admits the second suffix one step later. The first request then decodes while the second one
    # prefills, so the first recorded step holds the two requests at contexts 250 and 249, and every repetition is
    # rejected: the cached prefix cannot grow beyond the seed.
    (tmp_path / "split").mkdir()
    s, engine = chunked_decode(module, tmp_path / "split", 2, 112)
    with pytest.raises(RuntimeError, match="too many rejected repetitions"):
        run(s, engine, "decode")
    assert not s.saved
    first_steps = [record["fpms"][0]["scheduled_requests"] for record in s._glm_rejected]
    assert [(step["num_decode_requests"], step["sum_decode_kv_tokens"]) for step in first_steps] == [(2, 499)] * 11


def align_prefill(module, tmp_path, new, alignment=None):
    """A prefill point of one request with ``new`` tokens after one block of prefix, on the 2,176-token blocks of
    GLM-5.3-Flash at TP 4, scheduled like vLLM in Mamba align mode with an 8,192-token budget.

    With an ``alignment``, every KDA group of the launch holds a prefill checkpoint block of that alignment."""
    points = [Point("prefill", 1, 1, new, 2176, rows=[[new, 2176]])]
    s = make_scheduler(module, tmp_path, points)
    s._glm_serving = {"hash_block_size": 2176}
    s.cache_config.block_size = s.block_size = s.hash_block_size = 2176
    s.mamba_has_prefill_checkpoint_blocks = alignment is not None
    s.mamba_prefill_checkpoint_alignment = alignment
    s._bench_explicit_points = points
    return s, BudgetEngine(s, budget=8192, unit=2176, alignment=alignment)


def test_explicit_prefill_points_that_vllm_splits_are_refused_at_grid_build(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    # Without a prefill checkpoint block, vLLM 0.33.0 stops the chunk of two blocks after one block at 4,352, the last
    # cacheable block boundary before the end of its prompt. Every repetition then records two prefill steps, and the
    # producer rejects it.
    s, engine = align_prefill(module, tmp_path, 4352)
    with pytest.raises(RuntimeError, match="too many rejected repetitions"):
        run(s, engine, "prefill")
    assert not s.saved
    assert [(record["rejection_reason"], record["prefill_chunks"]) for record in s._glm_rejected] == [
        ("recorded 2 native FPMs, expected 1", [[[2176, 2176]], [[4352, 2176]]])
    ] * 11
    # The grid build refuses the point at engine start, before it seeds anything.
    (tmp_path / "refused").mkdir()
    s, _ = align_prefill(module, tmp_path / "refused", 4352)
    with pytest.raises(ValueError) as refused:
        s._bench_build_grid()
    assert str(refused.value) == (
        "benchmark_id=1: vLLM's Mamba align mode stops the prefill of 4352 new tokens after a prefix of 2176 at 4352, "
        "a boundary of its 2176-token blocks, so the point runs in more than one step and the producer measures one"
    )
    # With a checkpoint block of a 16-token alignment, vLLM saves the checkpoint at 4,352 inside the chunk and takes
    # neither stop, so two blocks and 4,096 tokens after one block of prefix run in one step.
    for new in (4352, 4096):
        (tmp_path / f"checkpoint-{new}").mkdir()
        s, engine = align_prefill(module, tmp_path / f"checkpoint-{new}", new, alignment=16)
        s._bench_build_grid()
        run(s, engine, "prefill")
        assert [point.benchmark_id for point, _ in s.saved] == [1]
        record = json.loads((tmp_path / f"checkpoint-{new}" / "benchmark.repetitions.jsonl").read_text())
        assert not record["rejected_repetitions"]
        assert all(repetition["prefill_chunks"] == [[[2176, new]]] for repetition in record["repetitions"])


def _scheduler(module, tmp_path, **launch):
    """A producer of one prefill point whose vLLM scheduler holds the launch attributes ``launch``."""
    points = [Point("prefill", 1, 1, 4, 0, rows=[[4, 0]])]
    s = make_scheduler(module, tmp_path, points)
    s._bench_explicit_points = points
    for name, value in launch.items():
        setattr(s, name, value)
    return s


def test_producer_and_planner_align_rules_agree(monkeypatch):
    scheduler = load_producer(monkeypatch).Glm53FlashPrefixSeedScheduler
    for start in range(0, 257, 16):
        for end in range(start + 4, 449, 12):
            for alignment in (None, 16, 48):
                assert scheduler._glm_align_stop(
                    start, end, 64, hash_block=64, checkpoint_alignment=alignment
                ) == glm_align_stop(start, end, 64, hash_block=64, checkpoint_alignment=alignment)


def test_producer_refuses_a_launch_that_the_align_rule_does_not_model(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    for name, value in (
        ("use_eagle_block_drop", True),
        ("num_hash_lookahead_tokens", 1),
        ("mamba_partial_cache_hit", True),
        # The scheduler block of vLLM differs from the block of the cache configuration.
        ("block_size", 8),
    ):
        scheduler = _scheduler(module, tmp_path, **{name: value})
        with pytest.raises(ValueError, match="align-mode rule") as refused:
            scheduler._bench_build_grid()
        assert str(refused.value) == (
            f"GLM-5.3-Flash collection does not model this launch in its align-mode rule: {name}={value}"
        )
    # A launch that the rule models builds its grid.
    scheduler = _scheduler(module, tmp_path)
    scheduler._bench_build_grid()
    assert [point.benchmark_id for point in scheduler._bench_grid] == [1]


def hub_stand_in(monkeypatch, cached):
    """Install a ``huggingface_hub`` that answers every cache lookup with ``cached``. Returns the lookups."""
    lookups = []

    def try_to_load_from_cache(repo_id, filename, *, revision=None):
        lookups.append((repo_id, filename, revision))
        return cached

    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = try_to_load_from_cache
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return lookups


def test_checkpoint_dir_resolves_folders_and_hub_ids(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    # A folder launch is its own checkpoint folder and needs no cache lookup.
    lookups = hub_stand_in(monkeypatch, None)
    assert module.glm_checkpoint_dir(str(tmp_path), LOADED_REVISION) == tmp_path
    assert lookups == []
    # A Hub ID launch resolves to the snapshot folder of the loaded revision in the local Hugging Face cache.
    snapshot = tmp_path / "snapshots" / "04c4e9e9"
    lookups = hub_stand_in(monkeypatch, str(snapshot / "config.json"))
    assert module.glm_checkpoint_dir(FP8_MODEL, LOADED_REVISION) == snapshot
    assert lookups == [(FP8_MODEL, "config.json", LOADED_REVISION)]


@pytest.mark.parametrize("cached", [None, object()], ids=["unknown-file", "known-absent-sentinel"])
def test_checkpoint_dir_refuses_hub_ids_without_cached_files(monkeypatch, cached):
    module = load_producer(monkeypatch)
    hub_stand_in(monkeypatch, cached)
    with pytest.raises(ValueError, match="absent from the local Hugging Face cache"):
        module.glm_checkpoint_dir(FP8_MODEL, LOADED_REVISION)


@pytest.mark.parametrize("revision", ["main", "04c4e9e9", ""], ids=["branch", "abbreviated", "unset"])
def test_checkpoint_dir_requires_the_40_hex_loaded_revision(monkeypatch, tmp_path, revision):
    module = load_producer(monkeypatch)
    hub_stand_in(monkeypatch, str(tmp_path / "config.json"))
    for model in (str(tmp_path), FP8_MODEL):
        with pytest.raises(ValueError, match="DYN_FPM_TOKENIZER_REVISION must be the 40-hex revision"):
            module.glm_checkpoint_dir(model, revision)


OTHER_CONFIG = {"architectures": ["Glm5NextForConditionalGeneration"], "hidden_size": 1}


def patch_pinned_configs(monkeypatch):
    """Give the two pinned models different configurations. Returns them as ``(fp8, nvfp4)``."""
    fp8 = {"architectures": ["Glm5NextForConditionalGeneration"]}
    nvfp4 = {"architectures": ["Glm5NextForConditionalGeneration"], "quantization_config": {"quant_algo": "NVFP4"}}
    configs = {FP8_MODEL: {"raw_config": fp8}, NVFP4_MODEL: {"raw_config": nvfp4}}
    monkeypatch.setattr(sdk_utils, "get_model_config_from_model_path", configs.__getitem__)
    return fp8, nvfp4


def test_pinned_model_resolution_accepts_equal_config_revisions(monkeypatch):
    module = load_producer(monkeypatch)
    fp8, nvfp4 = patch_pinned_configs(monkeypatch)
    # A revision that is no pin is accepted when its configuration equals a pinned model's configuration.
    assert module.glm_pinned_model(dict(fp8), LOADED_REVISION) == (FP8_MODEL, FP8_PIN)
    assert module.glm_pinned_model(dict(nvfp4), LOADED_REVISION) == (NVFP4_MODEL, NVFP4_PIN)
    with pytest.raises(ValueError, match="loaded GLM-5.3-Flash checkpoint matches no pinned model configuration"):
        module.glm_pinned_model(dict(OTHER_CONFIG), LOADED_REVISION)


def test_a_pinned_revision_needs_the_configuration_of_its_model(monkeypatch):
    module = load_producer(monkeypatch)
    fp8, nvfp4 = patch_pinned_configs(monkeypatch)
    assert module.glm_pinned_model(dict(fp8), FP8_PIN) == (FP8_MODEL, FP8_PIN)
    assert module.glm_pinned_model(dict(nvfp4), NVFP4_PIN) == (NVFP4_MODEL, NVFP4_PIN)
    # A revision label that contradicts the loaded files must not select another model. This holds for the
    # configuration of the other pinned model and for a configuration that no pin has.
    for pin, model, files in (
        (FP8_PIN, FP8_MODEL, nvfp4),
        (NVFP4_PIN, NVFP4_MODEL, fp8),
        (FP8_PIN, FP8_MODEL, OTHER_CONFIG),
    ):
        refusal = f"GLM-5.3-Flash checkpoint at pinned revision {pin} does not match the configuration of {model}"
        with pytest.raises(ValueError) as refused:
            module.glm_pinned_model(dict(files), pin)
        assert str(refused.value) == refusal


def test_runtime_context_rule(monkeypatch):
    module = load_producer(monkeypatch)
    # The runtime context is the configured one. The measured limit stays what the point list may reach.
    assert module.glm_context_policy(1048576, 131072) == {
        "measured_context_limit": 131072,
        "runtime_context_length": 1048576,
        "native_admission_headroom": 7,
    }
    assert module.glm_context_policy(131079, 131072)["measured_context_limit"] == 131072
    with pytest.raises(ValueError, match="at least the measured context plus seven internal positions"):
        module.glm_context_policy(131078, 131072)


def init_scheduler(module, monkeypatch, *, model, checkpoint):
    """A producer scheduler whose ``_bench_init`` runs end to end on fakes of the native base and the vLLM config.

    ``checkpoint`` is the folder that holds the loaded checkpoint's ``tokenizer.json``. Its configuration is the
    pinned FP8 checkpoint's. vLLM loaded the revision that ``DYN_FPM_TOKENIZER_REVISION`` names, as in the launch that
    the collector renders. Returns the scheduler, the vLLM config and the configuration and tokenizer loads that the
    producer made.
    """
    monkeypatch.setattr(FakeBase, "_bench_init", lambda self, config: setattr(self, "_bench_active", True))
    for name in (
        "DYN_FPM_GLM53FLASH_MEASURED_CONTEXT",
        "DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT",
        "DYN_FPM_GLM53FLASH_DECODE_CONTEXTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DYN_FPM_TOKENIZER_REVISION", LOADED_REVISION)
    tokenizer_json = checkpoint / "tokenizer.json"
    tokenizer_json.write_bytes(b'{"stand-in": "tokenizer.json"}')
    monkeypatch.setattr(seeded, "TOKENIZER_JSON_SHA256", hashlib.sha256(tokenizer_json.read_bytes()).hexdigest())
    real_config = sdk_utils.get_model_config_from_model_path
    loads = {"config": [], "tokenizer": []}

    def load_config(path):
        loads["config"].append(path)
        return real_config(FP8_MODEL if path == str(checkpoint) else path)

    monkeypatch.setattr(sdk_utils, "get_model_config_from_model_path", load_config)

    def get_tokenizer(name, **kwargs):
        loads["tokenizer"].append((name, kwargs["revision"]))
        return types.SimpleNamespace(vocab_size=seeded.VOCAB_HIGH)

    monkeypatch.setattr(module, "get_tokenizer", get_tokenizer)
    scheduler = module.Glm53FlashPrefixSeedScheduler.__new__(module.Glm53FlashPrefixSeedScheduler)
    scheduler._bench_hash_block_size = scheduler.block_size = 2176
    scheduler.connector = scheduler.ec_connector = None
    # The checkpoint attributes that vLLM 0.33.0's scheduler sets for a launch without a prefill checkpoint block
    # (scheduler.py:437-449).
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler._bench_explicit_points = object()
    scheduler._bench_config = types.SimpleNamespace(warmup_iterations=0, collect_imbalanced=True)
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            model=model,
            revision=LOADED_REVISION,
            tokenizer=model,
            tokenizer_mode="auto",
            trust_remote_code=False,
            max_model_len=1048576,
            enforce_eager=False,
            architecture="Glm5NextForConditionalGeneration",
            hf_config=types.SimpleNamespace(model_type="glm5_next"),
        ),
        observability_config=types.SimpleNamespace(cudagraph_metrics=True),
        cache_config=types.SimpleNamespace(
            enable_prefix_caching=True, mamba_cache_mode="align", prefix_match_unit=None, cache_dtype="fp8_e4m3"
        ),
        scheduler_config=types.SimpleNamespace(async_scheduling=False, max_num_batched_tokens=8192),
        parallel_config=_parallel(4),
        speculative_config=None,
    )
    return scheduler, config, loads


@pytest.mark.parametrize("launch", ["folder", "hub-id"])
def test_bench_init_reads_the_resolved_checkpoint(monkeypatch, tmp_path, launch):
    module = load_producer(monkeypatch)
    checkpoint = tmp_path / "snapshots" / "04c4e9e9"
    checkpoint.mkdir(parents=True)
    model = str(checkpoint)
    if launch == "hub-id":
        model = FP8_MODEL
        hub_stand_in(monkeypatch, str(checkpoint / "config.json"))
    scheduler, config, loads = init_scheduler(module, monkeypatch, model=model, checkpoint=checkpoint)
    scheduler._bench_init(config)
    recorded = scheduler._glm_input
    assert recorded["checkpoint_dir"] == str(checkpoint)
    assert recorded["tokenizer_json_sha256"] == seeded.TOKENIZER_JSON_SHA256
    # The artifact records the loaded revision and the pinned checkpoint that its configuration matched.
    assert recorded["tokenizer_revision"] == LOADED_REVISION
    assert (recorded["pinned_model"], recorded["pinned_revision"]) == (
        "zai-org/GLM-5.3-Flash",
        "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    )
    # The loaded configuration and the tokenizer file come from the resolved folder. The tokenizer itself still
    # loads from the launch value at the loaded revision.
    assert str(checkpoint) in loads["config"]
    assert loads["tokenizer"] == [(model, LOADED_REVISION)]
    assert scheduler._glm_context_policy == {
        "measured_context_limit": 131072,
        "runtime_context_length": 1048576,
        "native_admission_headroom": 7,
    }


def test_producer_records_the_checkpoint_evidence_of_its_launch(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    scheduler, config, _ = init_scheduler(module, monkeypatch, model=str(tmp_path), checkpoint=tmp_path)
    # Every Mamba group holds a checkpoint block (scheduler.py:445-449), and the first alignment is 16.
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    scheduler.mamba_prefill_checkpoint_alignment = 16
    scheduler._bench_init(config)
    assert scheduler._glm_serving["mamba_prefill_checkpoint"] == {"blocks": True, "alignment": 16}
    # A launch without checkpoint blocks records no alignment, whatever the first Mamba group declares.
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler._bench_init(config)
    assert scheduler._glm_serving["mamba_prefill_checkpoint"] == {"blocks": False, "alignment": None}


def test_bench_init_refuses_an_unset_tokenizer_revision(monkeypatch, tmp_path):
    module = load_producer(monkeypatch)
    scheduler, config, _ = init_scheduler(module, monkeypatch, model=str(tmp_path), checkpoint=tmp_path)
    # A folder launch without a revision, so vLLM records none.
    config.model_config.revision = None
    monkeypatch.delenv("DYN_FPM_TOKENIZER_REVISION")
    with pytest.raises(ValueError, match="DYN_FPM_TOKENIZER_REVISION must be the 40-hex revision"):
        scheduler._bench_init(config)


@pytest.mark.parametrize("loaded", ["main", FP8_PIN], ids=["hub-default-branch", "other-commit"])
def test_bench_init_refuses_a_launch_revision_that_differs_from_the_tokenizer_revision(monkeypatch, tmp_path, loaded):
    module = load_producer(monkeypatch)
    lookups = hub_stand_in(monkeypatch, str(tmp_path / "config.json"))
    scheduler, config, loads = init_scheduler(module, monkeypatch, model=FP8_MODEL, checkpoint=tmp_path)
    # vLLM loads the weights of the revision in its model configuration. A launch by Hub ID without --revision loads
    # the default branch: vLLM 0.31.0 stores the requested revision as a string subclass whose value is "main".
    config.model_config.revision = loaded
    with pytest.raises(ValueError) as refused:
        scheduler._bench_init(config)
    assert str(refused.value) == (
        f"GLM-5.3-Flash launch revision {loaded!r} differs from DYN_FPM_TOKENIZER_REVISION {LOADED_REVISION!r}"
    )
    # The producer refuses before it reads the configuration or the tokenizer of a snapshot.
    assert (lookups, loads) == ([], {"config": [], "tokenizer": []})
