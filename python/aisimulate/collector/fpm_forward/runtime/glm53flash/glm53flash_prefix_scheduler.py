# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scheduler integration adapted from ai-dynamo/dynamo at
# 54960177085413259859c88bd34ed0734d4c2ea9, components/src/dynamo/vllm/instrumented_scheduler.py
# (InstrumentedScheduler benchmark state machine, injected requests scheduled by the
# parent vLLM scheduler). The prefix-cache real-seed staging (seed shot, untimed
# warm shot, validated measured shot) is ported from ai-dynamo/dynamo
# 1.5.0.dev20260917 (as shipped in lmsysorg/sglang@sha256:b0d8718a...),
# components/src/dynamo/vllm/instrumented_scheduler.py
# (_bench_realseed_stage_point / _bench_realseed_pending_step). Apache-2.0.
# Modified: GLM-5.3-Flash default-serving hybrid state (prefix caching on, Mamba
# "align" mode), real text tokens, repeated measurements and compact deferred evidence.
"""GLM-5.3-Flash native FPM producer on the default serving configuration.

Timing is the unchanged Dynamo ``InstrumentedScheduler`` FPM ``wall_time``:
prefill = schedule() end to update_from_output() arrival of the measured step;
steady decode = consecutive output arrivals (second decode step). This module
never builds a ``SchedulerOutput``: every request -- prefix seed, warmup and
measured -- is added to the native waiting queue and scheduled by vLLM's own
scheduler, so prefix-cache lookup, Mamba align-mode chunking, KDA/conv, MLA
and IndexPool state handling are the serving code paths.

State construction (prefix caching on):

* prefill ``(new_i, prefix_i)``: one seed request per slot computes exactly
  ``prefix_i`` real tokens (cached by the native prefix cache); each repetition
  then submits ``prefix_i`` + a repetition-unique suffix, which must hit exactly
  ``prefix_i`` (validated from the native measured FPM, never assumed).
* decode ``context_i``: a seed request computes the hit-aligned part of the
  ``context_i - 1`` token prompt; each repetition submits that prompt with
  ``max_tokens=3``; the second pure decode step (all B requests, KV read
  ``sum(context_i)``) is the measured steady sample.

No host work is added between steps of a repetition: the only per-step work is
the native Dynamo FPM bookkeeping plus a reference to the native
``cudagraph_stats`` object. Validation, medians and evidence serialization run
after a repetition's requests have finished, and the compact evidence record of
a point (prompt specifications, not token arrays) is written after all of its
repetitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import dynamo.vllm.instrumented_scheduler as native
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.request import Request

DYNAMO_SHA = "54960177085413259859c88bd34ed0734d4c2ea9"
REAL_SEED_SOURCE = "ai-dynamo/dynamo 1.5.0.dev20260917 instrumented_scheduler._bench_realseed_*"
VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
MODEL_SHAS = {
    "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
}
STATE_PROTOCOL = "glm53flash_prefix_cache_real_seed_v1"
TIMING_BOUNDARY = "dynamo_vllm_instrumented_scheduler_fpm_wall_time"
WARMUP_REPEATS = 5
MEASUREMENT_REPEATS = 10
MAX_REJECTED_REPETITIONS = 10
MAX_BATCH = 32
MAX_CONTEXT = 131072
MAX_NEW = 8192
STAGE_TIMEOUT_SECONDS = 1800.0
PREFILL_REAL_SEED_REASON = "prefill_real_seed"
DECODE_REAL_KV_REASON = "kvwarm_real_kv"


def token_slice(pool: list[int], length: int, offset: int) -> list[int]:
    """Rotate and repeat the tokenizer's real-text stream deterministically."""
    if not pool or length < 0:
        raise ValueError("non-empty token pool and nonnegative length required")
    size = len(pool)
    start = offset % size
    repeats, tail = divmod(start + length, size)
    stream = pool * repeats + pool[:tail]
    return stream[start : start + length]


def prompt_spec(segments: list[tuple[int, int]]) -> list[list[int]]:
    """Compact, exactly reconstructible prompt: ``[[offset, length], ...]``."""
    return [[int(offset), int(length)] for offset, length in segments]


def build_prompt(pool: list[int], spec: list[list[int]]) -> list[int]:
    tokens: list[int] = []
    for offset, length in spec:
        tokens.extend(token_slice(pool, length, offset))
    return tokens


def scheduled_dict(metrics) -> dict:
    return {
        "num_prefill_requests": int(metrics.num_prefill_requests),
        "sum_prefill_tokens": int(metrics.sum_prefill_tokens),
        "sum_prefill_kv_tokens": int(metrics.sum_prefill_kv_tokens),
        "num_decode_requests": int(metrics.num_decode_requests),
        "sum_decode_kv_tokens": int(metrics.sum_decode_kv_tokens),
    }


class Glm53FlashPrefixSeedScheduler(native.InstrumentedScheduler):
    # ------------------------------------------------------------------
    # Initialization and identity
    # ------------------------------------------------------------------
    def _bench_init(self, config):
        self._glm_point = None
        self._glm_stage = None
        self._glm_stage_deadline = 0.0
        self._glm_rows = None
        self._glm_repetitions = []
        self._glm_rejected = []
        self._glm_rep_index = 0
        self._glm_rep_record = None
        self._glm_rep_dispatches = []
        self._glm_rep_scheduled = []
        self._glm_seed = None
        self._glm_point_records = {}
        self._glm_evidence_count = 0
        self._glm_evidence_digest = hashlib.sha256()
        self._glm_request_set = f"{os.environ.get('FPM_RUN_ID', 'glm53flash')}-{uuid.uuid4().hex}"
        self._glm_tokens = None
        self._glm_input = None
        self._glm_identity = None
        measured = int(os.environ.get("DYN_FPM_GLM53FLASH_MEASURED_CONTEXT", MAX_CONTEXT))
        if not 1 <= measured <= MAX_CONTEXT:
            raise ValueError("GLM measured context limit must be between 1 and 131072")
        self._glm_context_policy = {
            "measured_context_limit": measured,
            "runtime_context_length": measured + 7,
            "native_admission_headroom": 7,
        }
        if config.model_config.max_model_len != self._glm_context_policy["runtime_context_length"]:
            raise ValueError("GLM vLLM runtime context must equal measured context plus seven internal positions")
        super()._bench_init(config)
        if not self._bench_active:
            raise ValueError("GLM-5.3-Flash prefix-seed producer requires native benchmark mode")
        if config.model_config.enforce_eager:
            raise ValueError("GLM-5.3-Flash FPM requires the native CUDA graph policy")
        if not config.observability_config.cudagraph_metrics:
            raise ValueError("GLM-5.3-Flash FPM requires actual CUDA graph dispatch metrics")
        cache = config.cache_config
        if not cache.enable_prefix_caching:
            raise ValueError("default-serving GLM FPM requires native prefix caching (vLLM default)")
        self._glm_serving = {
            "enable_prefix_caching": bool(cache.enable_prefix_caching),
            "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
            "prefix_match_unit": getattr(cache, "prefix_match_unit", None),
            "hash_block_size": int(self._bench_hash_block_size),
            "block_size": int(self.block_size),
            "need_mamba_block_aligned_split": bool(getattr(self, "need_mamba_block_aligned_split", False)),
            "async_scheduling": bool(getattr(config.scheduler_config, "async_scheduling", False)),
        }
        expected_unit = os.environ.get("DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT")
        if expected_unit is not None and self._glm_serving["hash_block_size"] != int(expected_unit):
            raise ValueError("native prefix-match unit differs from the frozen campaign identity")
        if self._glm_serving["async_scheduling"]:
            raise ValueError("GLM FPM campaign freezes synchronous scheduling (--no-async-scheduling)")
        parallel = config.parallel_config
        self._glm_tp_size = parallel.tensor_parallel_size
        if (
            parallel.tensor_parallel_size not in (2, 4)
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.use_ubatching
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_expert_parallel
            or config.speculative_config is not None
        ):
            raise ValueError("GLM-5.3-Flash requires pure TP2/TP4, DP/PP/CP 1, no EP/ubatching/speculation")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("GLM-5.3-Flash FPM forbids KV/encoder connectors")
        if self._bench_explicit_points is None:
            raise ValueError("GLM-5.3-Flash requires a frozen explicit sampling manifest")
        if self._bench_config.warmup_iterations != 0:
            raise ValueError("native global warmup must be zero; the producer runs five warmups per point")
        if not self._bench_config.collect_imbalanced:
            raise ValueError("frozen executed geometries require DYN_FPM_BENCH_COLLECT_IMBALANCED=1")
        from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
        from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS, Glm53FlashConfig
        from aisimulate_core.sdk.utils import get_model_config_from_model_path

        raw_config = get_model_config_from_model_path(config.model_config.model)["raw_config"]
        Glm53FlashConfig.from_text_config(raw_config["text_config"])
        if (
            config.model_config.architecture != "Glm5NextForConditionalGeneration"
            or config.model_config.hf_config.model_type != "glm5_next"
        ):
            raise ValueError("loaded model must be GLM-5.3-Flash")
        self._glm_identity = dict(
            zip(
                EXECUTION_COLUMNS,
                execution_identity(raw_config, backend="vllm", input_modality="text"),
                strict=True,
            )
        )
        revision = os.environ.get("DYN_FPM_TOKENIZER_REVISION")
        if revision not in MODEL_SHAS:
            raise ValueError("tokenizer revision must equal the pinned GLM-5.3-Flash checkpoint revision")
        expected_model = next(model for model, pin in MODEL_REVISIONS.items() if pin == revision)
        if raw_config != get_model_config_from_model_path(expected_model)["raw_config"]:
            raise ValueError("loaded model metadata differs from the pinned GLM checkpoint")
        text_bytes = Path(os.environ["DYN_FPM_INPUT_TEXT"]).read_bytes()
        tokenizer = get_tokenizer(
            config.model_config.tokenizer,
            tokenizer_mode=config.model_config.tokenizer_mode,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=revision,
        )
        self._glm_tokens = list(tokenizer.encode(text_bytes.decode("utf-8"), add_special_tokens=False))
        if len(set(self._glm_tokens)) < 2 or not all(type(x) is int and x >= 0 for x in self._glm_tokens):
            raise ValueError("tokenizer text stream must contain distinct valid token ids")
        self._glm_input = {
            "source": "tokenizer_text",
            "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(self._glm_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "token_ids": list(self._glm_tokens),
            "tokenizer_revision": revision,
            "token_count": len(self._glm_tokens),
            "unique_token_count": len(set(self._glm_tokens)),
            "sampling": "prompt = concatenated [offset, length] rotations of token_ids (see prompt specs)",
        }

    def _bench_eager_warmup_points(self):
        return []  # five warmup repetitions run per exact geometry

    def _kvwarm_warm_eligible(self):
        return True

    def _bench_cache_fake_prefixes(self, *args, **kwargs):
        raise RuntimeError("synthetic prefix KV is forbidden for GLM-5.3-Flash")

    def _bench_inject_fake_decode(self, *args, **kwargs):
        raise RuntimeError("synthetic decode KV is forbidden for GLM-5.3-Flash")

    def _bench_new_request_counts_as_decode(self, req_id):
        # Every request here is an ordinary new request admitted by the native
        # scheduler; its first step is a (cache-hit) prefill.
        return False

    def _bench_blocks_per_req(self, num_tokens, *, has_cache_hit=False, apply_admission_cap=False):
        # Native capacity semantics of the pinned managers (hybrid, align mode).
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id="__glm53flash_capacity_probe__",
                num_tokens=num_tokens,
                new_computed_blocks=[],
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=num_tokens,
                apply_admission_cap=apply_admission_cap,
            )
            for manager in self.kv_cache_manager.coordinator.single_type_managers
        )

    def _bench_prefill_kv_read_lengths(self, total, batch, partition=None, rows=None):
        if rows is not None:
            lengths = [int(prefix) for _, prefix in rows]
        elif partition is not None:
            raise ValueError("GLM default-serving FPM uses explicit executed rows, not partitions")
        else:
            quotient, remainder = divmod(total, batch)
            lengths = [quotient + int(index < remainder) for index in range(batch)]
        if len(lengths) != batch or sum(lengths) != total or min(lengths) < 0:
            raise ValueError("prefix rows differ from requested token totals")
        return lengths

    def _bench_prefill_point_feasible(
        self, total_prefill_tokens, batch_size, total_kv_read_tokens, partition=None, rows=None
    ):
        if not (
            1 <= batch_size <= min(MAX_BATCH, self._bench_capacity_limit("max_num_running_reqs"))
            and batch_size
            <= total_prefill_tokens
            <= min(MAX_NEW, self._bench_capacity_limit("max_num_scheduled_tokens"))
        ):
            return False
        try:
            prefixes = self._bench_prefill_kv_read_lengths(total_kv_read_tokens, batch_size, partition, rows)
            queries = self._bench_prefill_new_token_lengths(total_prefill_tokens, batch_size, partition, rows)
        except ValueError:
            return False
        limit = min(self._glm_context_policy["measured_context_limit"], self._bench_capacity_limit("max_model_len"))
        if len(queries) != batch_size or sum(queries) != total_prefill_tokens or min(queries) < 1:
            return False
        lengths = [prefix + query for prefix, query in zip(prefixes, queries, strict=True)]
        if max(lengths) > limit:
            return False
        required = sum(self._bench_blocks_per_req(length) for length in lengths)
        return required <= self._bench_grid_usable_blocks(batch_size)

    def _bench_build_grid(self):
        built = self._bench_grid_built
        super()._bench_build_grid()
        if built:
            return
        unit = self._glm_serving["hash_block_size"]
        for point in self._bench_grid:
            if point.point_type == "prefill":
                prefixes, queries = self._glm_prefill_rows(point)
                if any(prefix % unit for prefix in prefixes):
                    raise ValueError(
                        f"benchmark_id={point.benchmark_id}: prefix not on the native prefix-match grid ({unit}); "
                        "the frozen manifest must carry the executed geometry"
                    )
            else:
                contexts = self._bench_decode_context_lengths(point.total_kv_read_tokens, point.batch_size)
                if min(contexts) < 3:
                    raise ValueError("GLM real decode requires per-request context >= 3; no clamping")

    def _glm_prefill_rows(self, point):
        prefixes = self._bench_prefill_kv_read_lengths(
            point.total_kv_read_tokens, point.batch_size, point.partition, point.rows
        )
        queries = self._bench_prefill_new_token_lengths(
            point.total_prefill_tokens, point.batch_size, point.partition, point.rows
        )
        return prefixes, queries

    # ------------------------------------------------------------------
    # Recording: only the measured repetition, only the native FPM
    # ------------------------------------------------------------------
    def _bench_should_record_scheduled(self, scheduled):
        if self._glm_stage != "measure" or self._glm_point is None:
            return False
        if self._glm_point.point_type == "prefill":
            return scheduled.num_prefill_requests > 0
        return scheduled.num_decode_requests > 0 and scheduled.num_prefill_requests == 0

    def _update_from_output(self, scheduler_output, model_runner_output):
        result = super()._update_from_output(scheduler_output, model_runner_output)
        if self._glm_stage == "measure" and scheduler_output.total_num_scheduled_tokens > 0:
            # References only: conversion happens after the repetition.
            self._glm_rep_dispatches.append(getattr(model_runner_output, "cudagraph_stats", None))
            self._glm_rep_scheduled.append(scheduler_output.total_num_scheduled_tokens)
        return result

    # ------------------------------------------------------------------
    # State machine (every request is natively scheduled)
    # ------------------------------------------------------------------
    def schedule(self, throttle_prefills=False):
        # Native Dynamo returns an empty output during DECODE_SWEEP while
        # benchmark requests are active (its decode points use synthetic
        # outputs). Here decode requests are ordinary native requests, so
        # every phase advances the state machine and then schedules natively.
        if not (self._bench_active and self._bench_phase == native._BenchPhase.DECODE_SWEEP):
            return super().schedule(throttle_prefills)
        try:
            if self._bench_step() is not None:
                raise RuntimeError("GLM prefix-seed producer never builds a SchedulerOutput")
        except Exception as error:
            self._bench_abort(error)
            raise
        return self._schedule_and_record_time(throttle_prefills)

    def _bench_step_prefill(self):
        return self._glm_step("prefill")

    def _bench_step_decode(self):
        return self._glm_step("decode")

    def _glm_alive(self):
        return any(rid in self.requests for rid in self._bench_active_req_ids)

    def _glm_inject(self, prompts, *, max_tokens, salts, role):
        ids = []
        for prompt, salt in zip(prompts, salts, strict=True):
            req_id = f"{self._glm_request_set}-{role}-{self._bench_seq}"
            self._bench_seq += 1
            request = Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=salt,
            )
            self.add_request(request)
            self._bench_active_req_ids.add(req_id)
            ids.append(req_id)
        self._glm_stage_deadline = time.monotonic() + STAGE_TIMEOUT_SECONDS
        return ids

    def _glm_step(self, point_type):
        if self._glm_point is None:
            if self._bench_stop_at_timeout_boundary(point_type):
                return None
            point = self._bench_pop_next(point_type)
            if point is None:
                self._bench_phase = (
                    native._BenchPhase.DECODE_SWEEP
                    if point_type == "prefill" and self._bench_config.mode == "agg"
                    else native._BenchPhase.DONE
                )
                return None
            self._glm_begin_point(point)
            return None
        if self._glm_alive():
            if time.monotonic() >= self._glm_stage_deadline:
                raise RuntimeError(f"GLM {self._glm_stage} stage timed out; no fallback")
            return None
        self._bench_cleanup_requests()
        if self._glm_stage == "seed":
            self._glm_seed["completed_monotonic"] = time.monotonic()
            self._glm_start_repetition()
        elif self._glm_stage == "measure":
            if self._glm_finish_repetition():
                self._glm_finish_point()
            else:
                self._glm_start_repetition()
        return None

    def _glm_slot_offset(self, point, slot):
        return 131 * slot + 17 * point.benchmark_id

    def _glm_begin_point(self, point):
        self._glm_point = point
        self._bench_current_point = None
        self._bench_current_fpms = []
        self._glm_repetitions = []
        self._glm_rejected = []
        self._glm_rep_index = 0
        salt_base = f"{self._glm_request_set}-p{point.benchmark_id}"
        self._glm_salts = [f"{salt_base}-s{slot}" for slot in range(point.batch_size)]
        if point.point_type == "prefill":
            prefixes, queries = self._glm_prefill_rows(point)
            self._glm_rows = {"prefix": prefixes, "new": queries}
            seed_lengths = prefixes
            reason = PREFILL_REAL_SEED_REASON if point.total_kv_read_tokens > 0 else None
        else:
            contexts = self._bench_decode_context_lengths(point.total_kv_read_tokens, point.batch_size)
            unit = self._glm_serving["hash_block_size"]
            prompt_lengths = [context - 1 for context in contexts]
            self._glm_rows = {"context": contexts, "prompt": prompt_lengths}
            # Largest hit-aligned prefix strictly shorter than the prompt.
            seed_lengths = [(length - 1) // unit * unit for length in prompt_lengths]
            reason = DECODE_REAL_KV_REASON
        if reason is not None and reason not in point.sample_reasons:
            point.sample_reasons.append(reason)
        self._glm_seed = {"lengths": list(seed_lengths), "specs": [], "request_ids": []}
        slots = [slot for slot, length in enumerate(seed_lengths) if length > 0]
        if not slots:
            self._glm_start_repetition()
            return
        specs = [prompt_spec([(self._glm_slot_offset(point, slot), seed_lengths[slot])]) for slot in slots]
        self._glm_seed["specs"] = specs
        self._glm_stage = "seed"
        self._glm_seed["request_ids"] = self._glm_inject(
            [build_prompt(self._glm_tokens, spec) for spec in specs],
            max_tokens=1,
            salts=[self._glm_salts[slot] for slot in slots],
            role="seed",
        )

    def _glm_repetition_prompts(self, point, attempt):
        specs, salts = [], []
        if point.point_type == "decode":
            for slot, length in enumerate(self._glm_rows["prompt"]):
                specs.append(prompt_spec([(self._glm_slot_offset(point, slot), length)]))
                salts.append(self._glm_salts[slot])
            return specs, salts
        for slot, (prefix, new) in enumerate(zip(self._glm_rows["prefix"], self._glm_rows["new"], strict=True)):
            base = self._glm_slot_offset(point, slot)
            if prefix:
                # Same seeded prefix; a repetition-unique suffix (shifted
                # rotation) so earlier repetitions cannot extend the hit.
                suffix_offset = base + prefix + 1 + 7 * (attempt + 1)
                specs.append(prompt_spec([(base, prefix), (suffix_offset, new)]))
                salts.append(self._glm_salts[slot])
            else:
                specs.append(prompt_spec([(base + 53 * attempt, new)]))
                salts.append(f"{self._glm_salts[slot]}-r{attempt}")
        return specs, salts

    def _glm_start_repetition(self):
        point = self._glm_point
        attempt = len(self._glm_repetitions) + len(self._glm_rejected)
        specs, salts = self._glm_repetition_prompts(point, attempt)
        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_expected_fpms = 2 if point.point_type == "decode" else 1
        self._glm_rep_dispatches = []
        self._glm_rep_scheduled = []
        self._glm_rep_record = {"attempt": attempt, "prompt_specs": specs, "salts": salts}
        self._glm_stage = "measure"
        prompts = [build_prompt(self._glm_tokens, spec) for spec in specs]
        self._glm_rep_record["request_ids"] = self._glm_inject(
            prompts, max_tokens=3 if point.point_type == "decode" else 1, salts=salts, role=f"r{attempt}"
        )

    def _glm_rejection(self, point, fpms):
        expected_count = 2 if point.point_type == "decode" else 1
        if len(fpms) != expected_count:
            return f"recorded {len(fpms)} native FPMs, expected {expected_count}"
        reason = self._bench_fpm_validation_failure(point, fpms[-1])
        if reason is not None:
            return reason
        scheduled = fpms[-1]["scheduled_requests"]
        if point.point_type == "prefill" and scheduled.get("num_decode_requests", 0):
            return "measured prefill step is mixed with decode"
        if point.point_type == "decode":
            first = fpms[0]["scheduled_requests"]
            if (
                first.get("num_decode_requests") != point.batch_size
                or first.get("sum_decode_kv_tokens") != point.total_kv_read_tokens - point.batch_size
            ):
                return "first decode step is not the all-B lockstep step"
        return None

    def _glm_finish_repetition(self) -> bool:
        point = self._glm_point
        fpms = [dict(fpm) for fpm in self._bench_current_fpms]
        dispatches = []
        for stats in self._glm_rep_dispatches:
            dispatches.append(None if stats is None else asdict(stats))
        record = {
            **self._glm_rep_record,
            "fpms": fpms,
            "scheduled_token_counts": list(self._glm_rep_scheduled),
            "dispatches": dispatches,
        }
        reason = self._glm_rejection(point, fpms)
        if reason is None and any(item is None for item in dispatches[-len(fpms) :]):
            reason = "missing actual graph dispatch receipt"
        self._bench_current_fpms = []
        if reason is not None:
            record["rejection_reason"] = reason
            self._glm_rejected.append(record)
            if len(self._glm_rejected) > MAX_REJECTED_REPETITIONS:
                raise RuntimeError(
                    f"benchmark_id={point.benchmark_id}: too many rejected repetitions; last reason: {reason}"
                )
            return False
        index = len(self._glm_repetitions)
        record["repetition"] = index
        record["role"] = "warmup" if index < WARMUP_REPEATS else "measurement"
        self._glm_repetitions.append(record)
        return len(self._glm_repetitions) == WARMUP_REPEATS + MEASUREMENT_REPEATS

    def _glm_finish_point(self):
        point = self._glm_point
        measured = [record for record in self._glm_repetitions if record["role"] == "measurement"]
        fpms = [dict(fpm) for fpm in measured[-1]["fpms"]]
        for index, fpm in enumerate(fpms):
            fpm["wall_time"] = statistics.median(record["fpms"][index]["wall_time"] for record in measured)
        self._bench_current_point = point
        self._bench_current_fpms = fpms
        self._glm_stage = None
        self._bench_save_current_point()
        evidence = {
            "benchmark_id": point.benchmark_id,
            "point_type": point.point_type,
            "rows": self._glm_rows,
            "seed": self._glm_seed,
            "repetitions": self._glm_repetitions,
            "rejected_repetitions": self._glm_rejected,
        }
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        path = Path(self._bench_config.output_path).with_suffix(".repetitions.jsonl")
        with path.open("xb" if self._glm_evidence_count == 0 else "ab") as stream:
            stream.write(raw + b"\n")
        self._glm_evidence_digest.update(raw + b"\n")
        self._glm_evidence_count += 1
        self._glm_point_records[point.benchmark_id] = hashlib.sha256(raw).hexdigest()
        self._glm_point = None
        self._glm_seed = None
        self._glm_rows = None

    def _bench_cleanup_requests(self):
        super()._bench_cleanup_requests()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _bench_write_results(self):
        destination = Path(self._bench_config.output_path)
        staging = destination.with_name(destination.name + ".native")
        # The runtime wrapper treats the final path as completion. Write the
        # native document elsewhere, enrich it, then publish atomically.
        self._bench_config.output_path = str(staging)
        try:
            super()._bench_write_results()
        finally:
            self._bench_config.output_path = str(destination)
        output = json.loads(staging.read_text())
        try:
            self._glm_enrich(output, destination)
        except Exception as error:
            # Publish the native document anyway; readers reject it.
            output["error"] = f"{output.get('error') or ''}; GLM enrichment failed: {error!r}".lstrip("; ")
            output["valid"] = output["usable"] = False
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(output, indent=2))
        os.replace(temporary, destination)
        staging.unlink()

    def _glm_enrich(self, output, destination):
        evidence_path = destination.with_suffix(".repetitions.jsonl")
        if self._glm_evidence_count:
            with evidence_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != self._glm_evidence_digest.hexdigest():
                raise RuntimeError("GLM repetition evidence changed before publication")
        else:
            digest = None
        hardware = []
        for rank in range(self._glm_tp_size):
            path = destination.with_name(f"native-device-rank-{rank}.json")
            raw = path.read_bytes()
            receipt = json.loads(raw)
            if receipt.get("status") != "passed" or receipt.get("tp_rank") != rank:
                raise RuntimeError("native GLM worker hardware qualification did not pass")
            hardware.append({"tp_rank": rank, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        output["input_provenance"] = {
            **(self._glm_input or {}),
            "context_policy": self._glm_context_policy,
            "native_hardware_manifest": hardware,
            "repetition_evidence_manifest": {
                "schema_version": 1,
                "file": evidence_path.name,
                "sha256": digest,
                "records": self._glm_evidence_count,
            },
        }
        output["context_policy"] = self._glm_context_policy
        output["execution_identity"] = self._glm_identity
        output["execution_mode"] = "native_graph_policy"
        output["observation_purpose"] = "fpm"
        output["ops_instrumented"] = False
        output["timing_boundary"] = TIMING_BOUNDARY
        output["serving_config"] = self._glm_serving
        output["kvwarm"] = {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "native_prefix_cache_real_seed",
            "state_protocol": STATE_PROTOCOL,
            "max_batch": MAX_BATCH,
            "max_context": self._glm_context_policy["measured_context_limit"],
        }
        from collector.glm53flash_runtime_identity import vllm_source_manifest_sha256

        output["producer"] = {
            "instrumentation_revision": DYNAMO_SHA,
            "real_seed_port_source": REAL_SEED_SOURCE,
            "vllm_package_version": __import__("vllm").__version__,
            "reviewed_scheduler_api_revision": VLLM_SHA,
            "runtime_source_manifest_sha256": vllm_source_manifest_sha256(
                __import__("vllm").__version__, Path(__file__).with_name("runtime-source-sha256.json")
            ),
            "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context_policy_version": 1,
            "hardware_contract_version": 1,
            "warmup_repeats": WARMUP_REPEATS,
            "measurement_repeats": MEASUREMENT_REPEATS,
        }
        for item in output["results"]:
            point = item["point"]
            item["kv_seed_regime"] = (
                "real_kv"
                if point["point_type"] == "decode"
                else "real_prefix"
                if point["total_kv_read_tokens"] > 0
                else "not_applicable"
            )
            item["repetition_evidence_sha256"] = self._glm_point_records.get(point["benchmark_id"])
        for group in output["iteration_groups"]:
            group["kv_seed_regime"] = "real_kv"


# Spawned processes may import this module before the native base finishes;
# publish only after class creation (see sitecustomize.py).
if os.environ.get("DYN_FPM_GLM53FLASH_PREFIX_SEED") == "1":
    native.InstrumentedScheduler = Glm53FlashPrefixSeedScheduler
