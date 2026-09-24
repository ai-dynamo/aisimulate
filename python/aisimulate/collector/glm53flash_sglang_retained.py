# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialized native request benchmark with retained, actually computed state.

Integration: sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e,
Apache-2.0. Calls native Scheduler, ScheduleBatch, ChunkCache and hybrid pools;
no upstream implementation is copied. See README.glm53flash_sglang.md and the
root THIRD_PARTY_NOTICES.md for population sites and lifecycle attribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

PRODUCER_PROTOCOL = "sglang_retained_request_benchmark_v1"


def _digest(values) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def validate_retained_states(manifest: dict, traces: dict[int, bytes], receipts: dict[int, bytes]) -> None:
    """Verify lifecycle receipts against every actual completed native forward."""
    if set(traces) != set(receipts):
        raise ValueError("retained-state rank coverage differs from native forwards")

    def snapshot(value):
        integer_fields = (
            "req_pool_idx",
            "mamba_pool_idx",
            "committed_tokens",
            "allocated_tokens",
            "cached_prefix_tokens",
            "retained_tokens",
        )
        hash_fields = (
            "committed_indices_sha256",
            "cached_prefix_indices_sha256",
            "retained_indices_sha256",
        )
        if (
            not isinstance(value, dict)
            or any(type(value.get(key)) is not int or value[key] < 0 for key in integer_fields)
            or any(
                not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key])
                for key in hash_fields
            )
            or not max(value["cached_prefix_tokens"], value["retained_tokens"])
            <= value["committed_tokens"]
            <= value["allocated_tokens"]
        ):
            raise ValueError("retained-state physical snapshot is invalid")
        return value

    for rank, raw in traces.items():
        forwards = [json.loads(line) for line in raw.splitlines()]
        events = [json.loads(line) for line in receipts[rank].splitlines()]
        if len(forwards) != len(events):
            raise ValueError("retained-state receipt omitted or added native forwards")
        parked, released = {}, set()
        for forward, event in zip(forwards, events, strict=True):
            if (
                forward.get("producer_protocol") != PRODUCER_PROTOCOL
                or forward.get("gpu_completed") is not True
                or event.get("producer_protocol") != PRODUCER_PROTOCOL
                or event.get("tp_rank") != rank
                or event.get("forward_id") != forward.get("forward_id")
            ):
                raise ValueError("retained-state receipt is not bound to its completed native forward")
            rows = event.get("requests", [])
            ids = forward["request_ids"]
            if [row.get("request_id") for row in rows] != ids:
                raise ValueError("retained-state request order differs from native execution")
            occupied = {
                key: {value[key] for rid, value in parked.items() if rid not in ids}
                for key in ("req_pool_idx", "mamba_pool_idx")
            }
            for row, prefix, query in zip(rows, forward["prefix_lengths"], forward["query_lengths"], strict=True):
                rid = row["request_id"]
                if rid not in manifest["requests"] or rid in released:
                    raise ValueError("retained-state request was unplanned or reused after release")
                before = row.get("before")
                if before != parked.get(rid) or (before is None) != (prefix == 0):
                    raise ValueError("retained-state prefix did not survive the preceding native result")
                current = snapshot(row.get("completed"))
                if current["committed_tokens"] != prefix + query or current["retained_tokens"] != prefix:
                    raise ValueError("retained-state committed length differs from native scheduled tokens")
                for key in occupied:
                    if current[key] in occupied[key]:
                        raise ValueError("distinct retained requests alias a native hybrid state slot")
                    occupied[key].add(current[key])
                if before is not None and (
                    before["committed_tokens"] != prefix
                    or any(current[key] != before[key] for key in occupied)
                    or current["retained_indices_sha256"] != before["committed_indices_sha256"]
                ):
                    raise ValueError("retained-state slot or GPU prefix indices changed before the target")
                if type(row.get("released")) is not bool or row["released"] != (forward.get("stage") == "measure"):
                    raise ValueError("native request release is inconsistent with the frozen target")
                if row["released"]:
                    if row.get("parked") is not None:
                        raise ValueError("released native request still claims retained state")
                    parked.pop(rid, None)
                    released.add(rid)
                else:
                    after = snapshot(row.get("parked"))
                    if (
                        any(
                            after[key] != current[key]
                            for key in (
                                "req_pool_idx",
                                "mamba_pool_idx",
                                "committed_tokens",
                                "allocated_tokens",
                                "committed_indices_sha256",
                            )
                        )
                        or after["cached_prefix_tokens"] != after["committed_tokens"]
                        or after["retained_tokens"] != after["committed_tokens"]
                        or after["cached_prefix_indices_sha256"] != after["committed_indices_sha256"]
                        or after["retained_indices_sha256"] != after["committed_indices_sha256"]
                    ):
                        raise ValueError("native chunk cache failed to retain its completed GPU prefix")
                    parked[rid] = after
        if parked or released != set(manifest["requests"]):
            raise ValueError("native retained-state request lifecycle is incomplete")


def state_snapshot(scheduler, req, *, retained_tokens: int | None = None) -> dict:
    """Read actual GPU index rows and the native Mamba mapping outside timing."""
    kv = req.kv
    if not kv.holds_kv or not kv.holds_mamba:
        raise RuntimeError("retained request lost its native KV or Mamba allocation")
    row, mamba = int(kv.req_pool_idx), int(kv.mamba_pool_idx.item())
    pool = scheduler.req_to_token_pool
    if int(pool.req_index_to_mamba_index_mapping[row].item()) != mamba:
        raise RuntimeError("native request-to-Mamba mapping changed independently of its request")
    committed = int(kv.kv_committed_len)
    indices = pool.req_to_token[row, :committed].detach().cpu().tolist()
    prefix = req.prefix_indices.detach().cpu().tolist()
    if len(indices) != committed or indices[: len(prefix)] != prefix:
        raise RuntimeError("native retained prefix does not equal the allocated GPU KV index row")
    if committed < len(prefix) or int(kv.kv_allocated_len) < committed:
        raise RuntimeError("native retained request has inconsistent committed allocation")
    retained = len(prefix) if retained_tokens is None else retained_tokens
    if not 0 <= retained <= committed:
        raise RuntimeError("retained GPU prefix exceeds its completed allocation")
    return {
        "req_pool_idx": row,
        "mamba_pool_idx": mamba,
        "committed_tokens": committed,
        "allocated_tokens": int(kv.kv_allocated_len),
        "cached_prefix_tokens": len(prefix),
        "committed_indices_sha256": _digest(indices),
        "cached_prefix_indices_sha256": _digest(prefix),
        "retained_tokens": retained,
        "retained_indices_sha256": _digest(indices[:retained]),
    }


class RetainedRequestLoop:
    """Own only benchmark scheduling; native code owns allocation and execution."""

    def __init__(self, scheduler, manifest: dict, output: Path, *, batch_type=None):
        self.time_batch = None
        if batch_type is None:
            from sglang.srt.managers.schedule_batch import ScheduleBatch
            from sglang.srt.observability.req_time_stats import set_schedule_time_batch, set_time_batch

            batch_type = ScheduleBatch
            self.time_batch = (set_time_batch, set_schedule_time_batch)
        self.scheduler = scheduler
        self.batch_type = batch_type
        self.manifest = manifest
        self.output = output
        self.rank = int(scheduler.ps.tp_rank)
        self.pending = {}
        self.completed = set()
        self.cohorts = {}
        for rid, entry in manifest["requests"].items():
            self.cohorts.setdefault((entry["benchmark_id"], entry["repetition"]), []).append(rid)
        if not scheduler.tree_cache.is_chunk_cache() or scheduler.tree_cache.supports_mamba():
            raise RuntimeError("retained benchmark requires the native non-sharing ChunkCache")
        if scheduler.enable_pdmux or not scheduler.spec_algorithm.is_none():
            raise RuntimeError("retained benchmark requires ordinary native target forwards")

    def _before(self, reqs) -> dict:
        return {req.rid: state_snapshot(self.scheduler, req) if req.kv.holds_kv else None for req in reqs}

    def _batch(self, reqs, *, middle=False):
        scheduler = self.scheduler
        batch = self.batch_type.init_new(
            reqs=reqs,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            tree_cache=scheduler.tree_cache,
            model_config=scheduler.model_config,
            enable_overlap=scheduler.enable_overlap,
            spec_algorithm=scheduler.spec_algorithm,
            chunked_req=reqs[0] if middle else None,
        )
        batch.contains_last_prefill_chunk = not middle
        batch.prepare_for_extend()
        return batch

    def _execute(self, batch, before: dict, *, middle=False):
        scheduler = self.scheduler
        if self.time_batch is not None:
            self.time_batch[0](batch.reqs, "set_forward_entry_time")
            self.time_batch[1](batch)
        if getattr(scheduler, "enable_fpm", False):
            batch.fpm_start_time = time.monotonic()
        scheduler.cur_batch_for_debug = batch
        scheduler.chunked_req = batch.reqs[0] if middle else None
        result = scheduler.run_batch(batch)
        scheduler.launch_batch_sample_if_needed(result, batch)
        if scheduler.enable_overlap:
            # Native overlap loop fences shared input-buffer reads before
            # processing results (scheduler.py:1991-1997). We drain each step.
            scheduler._apply_war_barrier()
        witness = getattr(scheduler.model_worker, "_aisim_glm53_last_forward", None)
        ids = [req.rid for req in batch.reqs]
        if (
            not isinstance(witness, dict)
            or witness.get("request_ids") != ids
            or witness.get("gpu_completed") is not True
            or witness.get("producer_protocol") != PRODUCER_PROTOCOL
        ):
            raise RuntimeError("retained benchmark lacks its actual completed native forward")
        completed = {}
        for req, prefix, query in zip(batch.reqs, witness["prefix_lengths"], witness["query_lengths"], strict=True):
            prior = before[req.rid]
            prior_tokens = 0 if prior is None else prior["committed_tokens"]
            current = state_snapshot(scheduler, req, retained_tokens=prior_tokens)
            if prefix != prior_tokens or current["committed_tokens"] != prefix + query:
                raise RuntimeError("native completed forward does not extend its observed retained state")
            if prior is not None and (
                any(current[key] != prior[key] for key in ("req_pool_idx", "mamba_pool_idx"))
                or current["retained_indices_sha256"] != prior["committed_indices_sha256"]
            ):
                raise RuntimeError("native target changed retained hybrid slots or prefix GPU indices")
            completed[req.rid] = current
        scheduler.process_batch_result(batch, result)
        if middle:
            # Native normal scheduling stashes completed middle chunks before
            # the next prepare (scheduler.py:3456-3457,3588-3598). ChunkCache
            # obtains prefix_indices from the actual GPU row (chunk_cache.py:86).
            scheduler.stash_chunked_request(batch.reqs[0])
        requests = []
        for req in batch.reqs:
            if bool(req.kv.holds_kv) != bool(req.kv.holds_mamba):
                raise RuntimeError("native result released only part of the hybrid request state")
            parked = state_snapshot(scheduler, req) if req.kv.holds_kv else None
            if parked is not None and (
                parked["cached_prefix_tokens"] != parked["committed_tokens"] or req.inflight_middle_chunks != 0
            ):
                raise RuntimeError("native result did not commit a complete parked prefix")
            requests.append(
                {
                    "request_id": req.rid,
                    "before": before[req.rid],
                    "completed": completed[req.rid],
                    "parked": parked,
                    "released": parked is None,
                }
            )
        receipt = {
            "producer_protocol": PRODUCER_PROTOCOL,
            "tp_rank": self.rank,
            "forward_id": witness["forward_id"],
            "requests": requests,
        }
        with (self.output / f"retained-rank-{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + "\n")
        scheduler.chunked_req = None
        scheduler.last_batch = None
        return batch

    def _seed(self, req, prefix: int):
        while len(req.prefix_indices) < prefix:
            before = self._before([req])
            start = len(req.prefix_indices)
            end = min(prefix, start + 8192)
            # These are native scheduling choices, not cache contents/counters:
            # Req.set_extend_range and the middle-chunk result flag have the
            # native population sites scheduler.py:3991 and schedule_policy.py:1164.
            req.init_next_round_input()
            req.set_extend_range(start, end)
            req.inflight_middle_chunks += 1
            self._execute(self._batch([req], middle=True), before, middle=True)

    def run_cohort(self, reqs):
        entries = [self.manifest["requests"][req.rid] for req in reqs]
        fields = ("benchmark_id", "repetition", "target_phase", "target_query", "target_prefix", "target_batch_size")
        if len({tuple(entry[key] for key in fields) for entry in entries}) != 1:
            raise RuntimeError("native requests do not belong to one frozen cohort")
        entry = entries[0]
        batch_size, prefix, query = len(reqs), entry["target_prefix"], entry["target_query"]
        decode = entry["target_phase"] == "generation"
        if (
            batch_size != entry["target_batch_size"]
            or not 1 <= batch_size <= 32
            or not prefix >= 0
            or not query >= 1
            or batch_size * query > 8192
            or prefix + query > 131072
            or (decode and (prefix < 1 or query != 1))
        ):
            raise RuntimeError("frozen retained cohort is outside the admitted geometry")
        seed_prefix = prefix - 1 if decode else prefix
        for req in reqs:
            expected_prompt = prefix if decode else prefix + query
            if (
                len(req.origin_input_ids) != expected_prompt
                or req.sampling_params.max_new_tokens != (2 if decode else 1)
                or not req.sampling_params.ignore_eos
                or req.finished()
                or req.kv.holds_kv
                or req.kv.holds_mamba
            ):
                raise RuntimeError("native request admission altered the frozen prompt or sampling budget")
            req.init_next_round_input(self.scheduler.tree_cache)
            if len(req.prefix_indices):
                raise RuntimeError("a fresh benchmark request unexpectedly reused external prefix state")
            self._seed(req, seed_prefix)
        before = self._before(reqs)
        for req in reqs:
            req.init_next_round_input()  # No tree-cache match: preserve this Req's real prefix.
            req.set_extend_range(len(req.prefix_indices), len(req.full_untruncated_fill_ids))
        batch = self._execute(self._batch(reqs), before)
        if decode:
            before = self._before(reqs)
            batch = self.scheduler.update_running_batch(batch)
            if batch is None or [req.rid for req in batch.reqs] != [req.rid for req in reqs]:
                raise RuntimeError("native capacity admission retracted the frozen decode cohort")
            self._execute(batch, before)
        if any(not req.finished() or req.kv.holds_kv or req.kv.holds_mamba for req in reqs):
            raise RuntimeError("native completion did not release the benchmark cohort's hybrid state")
        retire = getattr(self.scheduler.model_worker, "_aisim_glm53_release_requests", None)
        if not callable(retire):
            raise RuntimeError("native observer cannot retire the released cohort's token history")
        retire([req.rid for req in reqs])
        self.scheduler.model_worker._aisim_glm53_last_forward = None

    def run(self):
        scheduler = self.scheduler
        while not scheduler.gracefully_exit:
            scheduler.ingest_requests()
            if scheduler._engine_paused:
                continue
            for req in scheduler.waiting_queue:
                if req.rid not in self.manifest["requests"] or req.rid in self.completed:
                    raise RuntimeError("retained benchmark received an unplanned or reused request")
                self.pending[req.rid] = req
            scheduler.waiting_queue = []
            ready = next((ids for ids in self.cohorts.values() if all(rid in self.pending for rid in ids)), None)
            if ready is None:
                if not self.pending:
                    scheduler.on_idle()
                continue
            reqs = [self.pending.pop(rid) for rid in ready]
            self.run_cohort(reqs)
            self.completed.update(ready)


def install() -> None:
    """Called in every native scheduler process before Engine constructs it."""
    from sglang.srt.managers.scheduler import DynamicGradMode, Scheduler

    if getattr(Scheduler, "_aisim_retained_benchmark", False):
        return
    manifest = json.loads(Path(os.environ["AISIM_GLM53_REQUEST_MANIFEST"]).read_text())
    output = Path(os.environ["AISIM_GLM53_TRACE_DIR"])

    @DynamicGradMode()
    def event_loop(scheduler):
        try:
            RetainedRequestLoop(scheduler, manifest, output).run()
        except BaseException as error:
            (output / f"retained-failed-rank-{scheduler.ps.tp_rank}.json").write_text(
                json.dumps(
                    {
                        "producer_protocol": PRODUCER_PROTOCOL,
                        "request_set": manifest["request_set"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "accuracy_acceptance": "NOT_EVALUATED",
                    },
                    indent=2,
                )
                + "\n"
            )
            raise

    Scheduler.event_loop_normal = event_loop
    Scheduler.event_loop_overlap = event_loop
    Scheduler._aisim_retained_benchmark = True
