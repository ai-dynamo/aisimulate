# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU native-interface doubles verify lifecycle; no GPU admission is claimed."""

import json
from types import SimpleNamespace

import pytest
from collector.glm53flash_sglang_retained import PRODUCER_PROTOCOL, RetainedRequestLoop, validate_retained_states

pytestmark = pytest.mark.unit


class Tensor:
    def __init__(self, value):
        self.value = value

    def __len__(self):
        return len(self.value)

    def item(self):
        return self.value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value.copy()


class Rows:
    def __init__(self):
        self.values = {}

    def __getitem__(self, key):
        row, column = key
        return Tensor(self.values[row][column])


class Req:
    def __init__(self, rid, length, decode):
        self.rid = rid
        self.origin_input_ids = list(range(length))
        self.output_ids = []
        self.full_untruncated_fill_ids = []
        self.prefix_indices = Tensor([])
        self.inflight_middle_chunks = 0
        self.sampling_params = SimpleNamespace(max_new_tokens=2 if decode else 1, ignore_eos=True)
        self.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)

    def finished(self):
        return len(self.output_ids) == self.sampling_params.max_new_tokens

    def init_next_round_input(self, cache=None):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids
        if cache is not None:
            self.prefix_indices = Tensor([])

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end)


class NativeBatch:
    @classmethod
    def init_new(cls, **kwargs):
        value = cls()
        value.__dict__.update(kwargs)
        return value

    def prepare_for_extend(self):
        self.phase = "context"
        pool = self.req_to_token_pool
        for req in self.reqs:
            if not req.kv.holds_kv:
                slot = len(pool.req_to_token.values) + 1
                req.kv.req_pool_idx = slot
                req.kv.mamba_pool_idx = Tensor(slot + 100)
                req.kv.holds_kv = req.kv.holds_mamba = True
                pool.req_index_to_mamba_index_mapping[slot] = Tensor(slot + 100)
            size = req.extend_range.end
            req.kv.kv_committed_len = req.kv.kv_allocated_len = size
            pool.req_to_token.values[req.kv.req_pool_idx] = list(
                range(req.kv.req_pool_idx * 200000, req.kv.req_pool_idx * 200000 + size)
            )


class Scheduler:
    def __init__(self, entries):
        self.entries = entries
        self.ps = SimpleNamespace(tp_rank=0)
        self.enable_pdmux = self.enable_overlap = False
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.model_config = object()
        self.model_worker = SimpleNamespace(_aisim_glm53_release_requests=self.retire_requests)
        self.token_to_kv_pool_allocator = object()
        self.req_to_token_pool = SimpleNamespace(req_to_token=Rows(), req_index_to_mamba_index_mapping={})
        self.tree_cache = SimpleNamespace(is_chunk_cache=lambda: True, supports_mamba=lambda: False)
        self.computed, self.traces, self.calls = {}, [], []
        self.sample_launched = False
        self.retired = []
        self.is_initializing = False
        self.cur_batch_for_debug = None
        self.forward_ct = 0

    def retire_requests(self, ids):
        assert all(req.finished() and not req.kv.holds_kv and not req.kv.holds_mamba for req in self.last_reqs)
        self.retired.extend(ids)

    def run_batch(self, batch):
        assert self.cur_batch_for_debug is batch
        self.forward_ct += 1
        self.last_reqs = batch.reqs
        prefixes, queries = [], []
        for req in batch.reqs:
            history = self.computed.setdefault(req.rid, [])
            prefix = len(history)
            tokens = (
                req.full_untruncated_fill_ids[prefix : req.extend_range.end]
                if batch.phase == "context"
                else req.output_ids[-1:]
            )
            assert tokens
            history.extend(tokens)
            assert len(history) == req.kv.kv_committed_len
            prefixes.append(prefix)
            queries.append(len(tokens))
        first = self.entries[batch.reqs[0].rid]
        target = (
            batch.phase == first["target_phase"]
            and len(batch.reqs) == first["target_batch_size"]
            and prefixes == [first["target_prefix"]] * len(batch.reqs)
            and queries == [first["target_query"]] * len(batch.reqs)
        )
        witness = {
            "producer_protocol": PRODUCER_PROTOCOL,
            "gpu_completed": True,
            "forward_id": str(len(self.traces)),
            "request_ids": [req.rid for req in batch.reqs],
            "prefix_lengths": prefixes,
            "query_lengths": queries,
            "phase": batch.phase,
            "stage": "measure" if target else "seed",
        }
        self.model_worker._aisim_glm53_last_forward = witness
        self.traces.append(witness)
        self.calls.append((batch.phase, tuple(prefixes), tuple(queries)))
        return SimpleNamespace()

    def process_batch_result(self, batch, result):
        assert self.sample_launched
        self.sample_launched = False
        for req in batch.reqs:
            if req.inflight_middle_chunks:
                req.inflight_middle_chunks -= 1
            else:
                req.output_ids.append(123)
                if req.finished():
                    req.kv.holds_kv = req.kv.holds_mamba = False
                else:
                    self.stash_chunked_request(req)

    def launch_batch_sample_if_needed(self, result, batch):
        self.sample_launched = True

    def stash_chunked_request(self, req):
        assert len(self.computed[req.rid]) == req.kv.kv_committed_len
        req.prefix_indices = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, : req.kv.kv_committed_len]

    def update_running_batch(self, batch):
        for req in batch.reqs:
            req.set_extend_range(req.kv.kv_committed_len, req.kv.kv_committed_len + 1)
        batch.prepare_for_extend()
        batch.phase = "generation"
        return batch


def campaign(tmp_path, batch, prefix, query, decode=False):
    entries = {
        f"request-{i}": {
            "benchmark_id": 1,
            "repetition": 0,
            "target_phase": "generation" if decode else "context",
            "target_query": query,
            "target_prefix": prefix,
            "target_batch_size": batch,
        }
        for i in range(batch)
    }
    scheduler = Scheduler(entries)
    manifest = {"requests": entries}
    reqs = [Req(rid, prefix if decode else prefix + query, decode) for rid in entries]
    loop = RetainedRequestLoop(scheduler, manifest, tmp_path, batch_type=NativeBatch)
    return loop, scheduler, manifest, reqs


@pytest.mark.parametrize(
    "batch,prefix,query",
    [(1, 0, 3), (2, 7, 3), (4, 1023, 1), (8, 8195, 7), (16, 4353, 9), (32, 9, 256), (1, 131071, 1)],
)
@pytest.mark.parametrize("decode", [False, True])
def test_real_seed_park_native_target_and_release(tmp_path, batch, prefix, query, decode):
    if decode:
        prefix, query = max(prefix, 1), 1
    loop, scheduler, manifest, reqs = campaign(tmp_path, batch, prefix, query, decode)
    loop.run_cohort(reqs)
    assert scheduler.calls[-1] == ("generation" if decode else "context", (prefix,) * batch, (query,) * batch)
    assert all(not req.kv.holds_kv and not req.kv.holds_mamba and req.finished() for req in reqs)
    assert scheduler.retired == [req.rid for req in reqs]
    assert scheduler.model_worker._aisim_glm53_last_forward is None
    assert all(sum(call[2]) <= 8192 for call in scheduler.calls)
    trace = ("\n".join(json.dumps(row) for row in scheduler.traces) + "\n").encode()
    validate_retained_states(manifest, {0: trace}, {0: (tmp_path / "retained-rank-0.jsonl").read_bytes()})


@pytest.mark.parametrize("corruption", ["slot", "indices", "unreleased", "missing_forward", "gpu_incomplete"])
def test_rejects_broken_retained_gpu_lifecycle(tmp_path, corruption):
    loop, scheduler, manifest, reqs = campaign(tmp_path, 2, 7, 3)
    loop.run_cohort(reqs)
    events = [json.loads(line) for line in (tmp_path / "retained-rank-0.jsonl").read_text().splitlines()]
    row = events[-1]["requests"][0]
    if corruption == "slot":
        row["completed"]["mamba_pool_idx"] += 99
    elif corruption == "indices":
        row["completed"]["retained_indices_sha256"] = "a" * 64
    elif corruption == "unreleased":
        row["released"] = False
    elif corruption == "missing_forward":
        events.pop(0)
    else:
        scheduler.traces[0]["gpu_completed"] = False
    serialize = lambda rows: ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    with pytest.raises(ValueError):
        validate_retained_states(manifest, {0: serialize(scheduler.traces)}, {0: serialize(events)})


def test_response_publication_waits_for_each_native_rank_release(tmp_path):
    from collector.fpm_forward.sglang_driver import wait_retained_release

    rows = [{"request_id": "request", "released": True, "parked": None}]
    event = {"producer_protocol": PRODUCER_PROTOCOL, "tp_rank": 0, "requests": rows}
    (tmp_path / "retained-rank-0.jsonl").write_text(json.dumps(event) + "\n")
    with pytest.raises(TimeoutError, match="ranks \\[1\\]"):
        wait_retained_release(tmp_path, ["request"], 2, 0)
    event["tp_rank"] = 1
    (tmp_path / "retained-rank-1.jsonl").write_text(json.dumps(event) + "\n")
    wait_retained_release(tmp_path, ["request"], 2, 0)
    with pytest.raises(TimeoutError):
        wait_retained_release(tmp_path, ["another-cohort"], 2, 0)
    (tmp_path / "retained-failed-rank-1.json").write_text('{"error":"native allocation failed"}')
    with pytest.raises(RuntimeError, match="allocation failed"):
        wait_retained_release(tmp_path, ["request"], 2, 0)


@pytest.mark.parametrize("overlap", [False, True])
def test_loop_initializes_native_idle_state_before_ingesting_requests(tmp_path, overlap):
    scheduler = Scheduler({})
    scheduler.enable_overlap = overlap
    scheduler.gracefully_exit = False
    scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler._sched_idled = False
    seen = []

    def ingest():
        if overlap:
            assert not scheduler.result_queue
        seen.append("ingest")

    def idle():
        assert scheduler._sched_idled
        if overlap:
            assert not scheduler.result_queue
        seen.append("idle")
        scheduler.gracefully_exit = True

    scheduler.ingest_requests = ingest
    scheduler.on_idle = idle
    RetainedRequestLoop(scheduler, {"requests": {}}, tmp_path, batch_type=NativeBatch).run()
    assert seen == ["ingest", "idle"]


def test_loop_refuses_to_discard_pending_native_overlap_result(tmp_path):
    scheduler = Scheduler({})
    scheduler.enable_overlap = True
    pending = object()
    scheduler.result_queue = [pending]
    with pytest.raises(RuntimeError, match="pending native overlap"):
        RetainedRequestLoop(scheduler, {"requests": {}}, tmp_path, batch_type=NativeBatch)
    assert scheduler.result_queue == [pending]


def test_paused_loop_uses_native_scheduler_state_accounting(tmp_path):
    scheduler = Scheduler({})
    scheduler.enable_overlap = True
    scheduler.gracefully_exit = False
    scheduler._engine_paused = True
    scheduler.ingest_requests = lambda: None
    seen = []

    def paused():
        assert not scheduler.result_queue
        seen.append("paused")
        scheduler.gracefully_exit = True

    scheduler._record_scheduler_state_for_paused_engine = paused
    RetainedRequestLoop(scheduler, {"requests": {}}, tmp_path, batch_type=NativeBatch).run()
    assert seen == ["paused"]


def native_watchdog_active(scheduler):
    # SGLang 94602c9c, scheduler_components/invariant_checker.py:505-527.
    # This exact native predicate gates the unchanged forward_ct/300s watchdog.
    return scheduler.is_initializing or scheduler.cur_batch_for_debug is not None


@pytest.mark.parametrize("decode", [False, True])
def test_watchdog_tracks_native_work_until_cohort_release_and_retirement(tmp_path, monkeypatch, decode):
    loop, scheduler, _manifest, reqs = campaign(tmp_path, 2, 7, 1 if decode else 3, decode)
    observed = []
    for name in ("run_batch", "launch_batch_sample_if_needed", "process_batch_result"):
        original = getattr(scheduler, name)

        def active_call(*args, _name=name, _original=original):
            assert native_watchdog_active(scheduler)
            observed.append(_name)
            return _original(*args)

        monkeypatch.setattr(scheduler, name, active_call)
    original_retire = scheduler.model_worker._aisim_glm53_release_requests

    def retire(ids):
        assert native_watchdog_active(scheduler)
        assert all(req.finished() and not req.kv.holds_kv and not req.kv.holds_mamba for req in reqs)
        observed.append("retire")
        original_retire(ids)

    monkeypatch.setattr(scheduler.model_worker, "_aisim_glm53_release_requests", retire)
    loop.run_cohort(reqs)
    assert observed == ["run_batch", "launch_batch_sample_if_needed", "process_batch_result"] * len(scheduler.calls) + [
        "retire"
    ]
    assert scheduler.forward_ct == len(scheduler.calls)
    assert scheduler.cur_batch_for_debug is None
    assert not native_watchdog_active(scheduler)


@pytest.mark.parametrize("failure", ["unfinished", "unreleased", "retirement"])
def test_watchdog_remains_active_when_cohort_completion_fails(tmp_path, monkeypatch, failure):
    loop, scheduler, _manifest, reqs = campaign(tmp_path, 1, 0, 3)
    if failure == "retirement":

        def fail_retire(_ids):
            assert native_watchdog_active(scheduler)
            raise RuntimeError("TEST_ONLY retirement failed")

        monkeypatch.setattr(scheduler.model_worker, "_aisim_glm53_release_requests", fail_retire)
        message = "retirement failed"
    else:
        original = loop._execute

        def incomplete(*args, **kwargs):
            value = original(*args, **kwargs)
            if failure == "unfinished":
                reqs[0].output_ids.clear()
            else:
                reqs[0].kv.holds_kv = reqs[0].kv.holds_mamba = True
            return value

        monkeypatch.setattr(loop, "_execute", incomplete)
        message = "did not release"
    with pytest.raises(RuntimeError, match=message):
        loop.run_cohort(reqs)
    assert scheduler.forward_ct == 1
    assert native_watchdog_active(scheduler)
    assert scheduler.model_worker._aisim_glm53_last_forward is not None


def test_empty_idle_clears_only_batch_marker_before_native_housekeeping(tmp_path):
    scheduler = Scheduler({})
    scheduler.gracefully_exit = scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler.cur_batch_for_debug = object()
    scheduler.forward_ct = 42
    scheduler.ingest_requests = lambda: None

    def idle():
        assert not native_watchdog_active(scheduler)
        assert scheduler.forward_ct == 42
        assert scheduler._sched_idled
        scheduler.gracefully_exit = True

    scheduler.on_idle = idle
    RetainedRequestLoop(scheduler, {"requests": {}}, tmp_path, batch_type=NativeBatch).run()


def test_pending_cohort_does_not_take_idle_watchdog_reset(tmp_path):
    loop, scheduler, _manifest, reqs = campaign(tmp_path, 2, 7, 3)
    scheduler.gracefully_exit = scheduler._engine_paused = False
    marker = scheduler.cur_batch_for_debug = object()
    polls = 0

    def ingest():
        nonlocal polls
        polls += 1
        if polls == 1:
            scheduler.waiting_queue = [reqs[0]]
        else:
            raise RuntimeError("TEST_ONLY stop incomplete cohort")

    scheduler.ingest_requests = ingest
    scheduler.on_idle = lambda: pytest.fail("pending cohort must not enter idle housekeeping")
    with pytest.raises(RuntimeError, match="stop incomplete cohort"):
        loop.run()
    assert scheduler.cur_batch_for_debug is marker
    assert native_watchdog_active(scheduler)
    assert scheduler.forward_ct == 0
    assert not loop.completed
