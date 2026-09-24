# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scheduler integration adapted from ai-dynamo/dynamo at
# 54960177085413259859c88bd34ed0734d4c2ea9, components/src/dynamo/vllm/instrumented_scheduler.py.
# Modified: GLM-5.3-Flash real hybrid state, repeated native observations, and dispatch evidence.
"""GLM-5.3-Flash native FPM producer with same-request real hybrid state.

Derived from the attributed DeepSeek canary integration in this repository.
Each repetition executes its complete history before the measured suffix or
second decode step. Native scheduler intervals remain the timing boundary.
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
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

DYNAMO_SHA = "54960177085413259859c88bd34ed0734d4c2ea9"
VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
MODEL_SHAS = {
    "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
}
WARMUP_REPEATS = 5
MEASUREMENT_REPEATS = 10
MAX_BATCH = 32
MAX_CONTEXT = 131072
MAX_NEW = 8192
# This DP1 producer executes a real seed chain and first-use kernel compilation
# before measurement. The native short result timeout bounds a synthetic point,
# not this complete real-KV lifecycle. Keep a finite whole-point deadline without
# changing native measurement timing, DP synchronization or campaign timeout.
REAL_POINT_TIMEOUT_SECONDS = 900.0


def token_slice(pool: list[int], length: int, offset: int) -> list[int]:
    """Rotate and repeat the tokenizer's real-text stream deterministically."""
    if not pool or length < 0:
        raise ValueError("non-empty token pool and nonnegative length required")
    return [pool[(offset + index) % len(pool)] for index in range(length)]


class Glm53FlashRealKVScheduler(native.InstrumentedScheduler):
    def _bench_init(self, config):
        self._real_request_set = f"{os.environ.get('FPM_RUN_ID', 'glm53flash')}-{uuid.uuid4().hex}"
        self._real_request_manifest = {}
        self._real_purpose = os.environ.get("AISIM_GLM53_PURPOSE", "fpm")
        if self._real_purpose not in {"fpm", "ops", "ops_holdout"}:
            raise ValueError("unsupported GLM observation purpose")
        self._real_tags = {}
        self._real_callback_stage = None
        self._real_stage = None
        self._real_requests = []
        self._real_stages = []
        self._real_stage_index = 0
        self._real_submitted = set()
        self._real_outstanding = 0
        self._real_deadline = 0.0
        self._real_token_stream_count = 0
        self._real_token_stream_digest = hashlib.sha256()
        self._real_witnesses = {}
        self._real_warmup_results = []
        self._real_expected_warmup_ids = []
        self._real_seed_tokens = 0
        self._real_input = None
        self._real_identity = None
        self._real_configure_context(config)
        super()._bench_init(config)
        if not self._bench_active:
            raise ValueError("GLM-5.3-Flash canary overlay requires native benchmark mode")
        if self._real_purpose == "fpm" and config.model_config.enforce_eager:
            raise ValueError("GLM-5.3-Flash formal collection requires native graph policy")
        if self._real_purpose != "fpm" and not config.model_config.enforce_eager:
            raise ValueError("GLM operation observation currently requires explicit native eager execution")
        if (self._real_purpose == "ops") != bool(os.environ.get("AISIM_GLM53_OPS_MANIFEST")):
            raise ValueError("GLM Ops instrumentation must match its explicit collection purpose")
        if not config.observability_config.cudagraph_metrics:
            raise ValueError("GLM-5.3-Flash FPM requires actual CUDA graph dispatch metrics")
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size not in (2, 4)
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
        ):
            raise ValueError("GLM-5.3-Flash requires pure TP2/TP4 / PP1 / DP1")
        if (
            parallel.use_ubatching
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("GLM-5.3-Flash canary requires no ubatching/DBO and no context parallelism")
        if parallel.enable_expert_parallel or config.speculative_config is not None:
            raise ValueError("GLM-5.3-Flash canary does not support EP or speculative decoding")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("GLM-5.3-Flash canary forbids KV/encoder connectors")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("GLM same-request state requires cross-request prefix caching disabled")
        if (
            parallel.enable_eplb
            or config.offload_config.uva.cpu_offload_gb
            or config.offload_config.prefetch.offload_group_size
            or config.cache_config.kv_offloading_size
        ):
            raise ValueError("GLM-5.3-Flash disables EPLB and CPU offload")
        if self._bench_explicit_points is None:
            raise ValueError("GLM-5.3-Flash requires a frozen explicit sampling manifest")
        if self._bench_config.warmup_iterations != 0:
            raise ValueError("native global warmup must be zero; producer runs five real warmups per point")
        self._real_repeat = 0
        self._real_repetitions = {}
        self._real_dispatches = []
        from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS, Glm53FlashConfig
        from aisimulate_core.sdk.utils import get_model_config_from_model_path

        raw_config = get_model_config_from_model_path(config.model_config.model)["raw_config"]

        Glm53FlashConfig.from_text_config(raw_config["text_config"])
        if (
            config.model_config.architecture != "Glm5NextForConditionalGeneration"
            or config.model_config.hf_config.model_type != "glm5_next"
        ):
            raise ValueError("loaded model must be GLM-5.3-Flash")
        from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity

        self._real_identity = dict(
            zip(
                EXECUTION_COLUMNS,
                execution_identity(
                    raw_config,
                    backend="vllm",
                    input_modality="text",  # This producer creates only text token-ID requests.
                ),
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
        self._real_tokens = list(tokenizer.encode(text_bytes.decode("utf-8"), add_special_tokens=False))
        if len(set(self._real_tokens)) < 2 or not all(type(x) is int and x >= 0 for x in self._real_tokens):
            raise ValueError("tokenizer text stream must contain distinct valid token ids")
        self._real_input = {
            "source": "tokenizer_text",
            "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(self._real_tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "tokenizer_revision": revision,
            "token_count": len(self._real_tokens),
            "unique_token_count": len(set(self._real_tokens)),
            "sampling": "rotate stream by 131*request_index+17*benchmark_id; repeat to requested length",
        }

    def _real_configure_context(self, config):
        # Keep the spawned native overlay independent of the host collector package.
        measured = int(os.environ.get("DYN_FPM_GLM53FLASH_MEASURED_CONTEXT", MAX_CONTEXT))
        if not 1 <= measured <= MAX_CONTEXT:
            raise ValueError("GLM measured context limit must be between 1 and 131072")
        self._real_context_policy = {
            "measured_context_limit": measured,
            "runtime_context_length": measured + 7,
            "native_admission_headroom": 7,
        }
        if config.model_config.max_model_len != self._real_context_policy["runtime_context_length"]:
            raise ValueError("GLM vLLM runtime context must equal measured context plus seven internal positions")

    def _bench_eager_warmup_points(self):
        # Every exact geometry receives five real warmups below, in either
        # native eager dispatch or CUDA graph replay.
        return []

    def _kvwarm_warm_eligible(self):
        return True

    def _bench_blocks_per_req(self, num_tokens, *, has_cache_hit=False, apply_admission_cap=False):
        # Use the pinned managers' native capacity semantics, including one-block rings.
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

    def _bench_build_grid(self):
        built = self._bench_grid_built
        super()._bench_build_grid()
        if not built:
            self._real_validate_grid()

    def _bench_prefill_kv_read_lengths(self, total, batch, partition=None, rows=None):
        # These are completed tokens on the SAME live request, not synthetic
        # prefix-cache hits. The hybrid allocation page/hash size (4352 in the
        # first TP2 run) cannot restrict the logical IndexPool tail coordinate.
        if batch < 1 or total < 0:
            raise ValueError("invalid real prefix geometry")
        if rows is not None:
            lengths = [int(prefix) for _, prefix in rows]
        elif total and partition is not None and partition.get("axis") in {"kv", "both"}:
            lengths = native._imbalanced_partition(
                total,
                batch,
                unit=1,
                minimum_units=0,
                high_count=int(partition["high_count"]),
                fraction=float(partition["fraction"]),
            )
        else:
            quotient, remainder = divmod(total, batch)
            lengths = [quotient + int(index < remainder) for index in range(batch)]
        if len(lengths) != batch or sum(lengths) != total or min(lengths) < 0:
            raise ValueError("real prefix rows differ from requested token totals")
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
        context_limit = min(
            self._real_context_policy["measured_context_limit"], self._bench_capacity_limit("max_model_len")
        )
        if len(queries) != batch_size or sum(queries) != total_prefill_tokens or min(queries) < 1:
            return False
        lengths = [prefix + query for prefix, query in zip(prefixes, queries, strict=True)]
        if max(lengths) > context_limit:
            return False
        # Use native allocation-manager capacity. A later allocation failure is
        # still a recorded failure; there is no fake-KV or smaller-point retry.
        required = sum(self._bench_blocks_per_req(length, apply_admission_cap=False) for length in lengths)
        return required <= self._bench_grid_usable_blocks(batch_size)

    def _real_validate_grid(self):
        from collector.glm53flash_runtime_identity import vllm_unaligned_prefill_admitted

        repaired_start = vllm_unaligned_prefill_admitted(__import__("vllm").__version__)
        self._real_expected_warmup_ids = []
        context_limit = self._real_context_policy["measured_context_limit"]
        for point in self._bench_grid:
            prefix, suffix = self._real_lengths(point)
            if (
                not 1 <= point.batch_size <= MAX_BATCH
                or max(prefix, default=0) + 2 * int(point.point_type == "decode") > context_limit
            ):
                raise ValueError("GLM-5.3-Flash canary point exceeds batch/context bound")
            if point.point_type == "prefill":
                # Stock vLLM's prefill pooling assumes pool-aligned starts.
                # GB300 split/one-shot probes fail at P4097/Q3 and Q4; keep
                # the broader unaligned-start contract unqualified, including
                # geometries that were not individually numerically probed.
                if not repaired_start and any(p % 4 and q >= 2 for p, q in zip(prefix, suffix, strict=True)):
                    raise ValueError(
                        "stock vLLM IndexPool cached-prefill start is unqualified: "
                        f"benchmark_id={point.benchmark_id}, prefixes={prefix}, queries={suffix}; "
                        "requires a separately qualified runtime repair, no coordinate substitution"
                    )
                if sum(suffix) > MAX_NEW:
                    raise ValueError("GLM-5.3-Flash canary prefill exceeds total new-token bound")
                if any(p + q > context_limit for p, q in zip(prefix, suffix, strict=True)):
                    raise ValueError("GLM-5.3-Flash canary prefill exceeds prefix plus new-token context bound")
            if point.point_type == "decode" and min(prefix) < 1:
                raise ValueError("GLM-5.3-Flash real decode requires context >=2; no coordinate clamping")

    def _bench_materialize_prefill_candidate(self, candidate, path, *, generated=False):
        partition = candidate.partition.model_dump() if candidate.partition is not None else None
        if not self._bench_prefill_point_feasible(
            candidate.total_prefill_tokens,
            candidate.batch_size,
            candidate.total_kv_read_tokens,
            partition,
            candidate.rows,
        ):
            self._bench_raise_explicit_infeasible(path, candidate)
        capture, padding, reasons = self._bench_cudagraph_metadata(
            candidate.total_prefill_tokens,
            self._bench_prefill_capture_sizes,
            self._bench_capacity_limit("max_num_scheduled_tokens"),
        )
        return native.BenchmarkPoint(
            point_type="prefill",
            total_prefill_tokens=candidate.total_prefill_tokens,
            total_kv_read_tokens=candidate.total_kv_read_tokens,
            batch_size=candidate.batch_size,
            expected_cudagraph_mode=self._bench_prefill_cudagraph_mode if capture is not None else "NONE",
            expected_capture_size=capture,
            padding_tokens=padding,
            partition=partition,
            rows=candidate.rows,
            sample_reasons=[native._bench_origin_reason(generated), *reasons],
        )

    def _real_lengths(self, point):
        if point.point_type == "decode":
            contexts = self._bench_decode_context_lengths(point.total_kv_read_tokens, point.batch_size)
            return [x - 1 for x in contexts], [1] * point.batch_size
        return (
            self._bench_prefill_kv_read_lengths(
                point.total_kv_read_tokens, point.batch_size, point.partition, point.rows
            ),
            self._bench_prefill_new_token_lengths(
                point.total_prefill_tokens, point.batch_size, point.partition, point.rows
            ),
        )

    def _bench_cache_fake_prefixes(self, *args, **kwargs):
        raise RuntimeError("synthetic prefix KV is forbidden for GLM-5.3-Flash")

    def _bench_inject_fake_decode(self, *args, **kwargs):
        raise RuntimeError("synthetic decode KV is forbidden for GLM-5.3-Flash")

    def _bench_new_request_counts_as_decode(self, req_id):
        return False  # Every newly registered request starts at actual position zero.

    def _bench_should_record_scheduled(self, scheduled):
        return self._real_callback_stage in {"admission", "measure"} and super()._bench_should_record_scheduled(
            scheduled
        )

    def _real_begin(self, point):
        prefix, suffix = self._real_lengths(point)
        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_expected_fpms = 2 if point.point_type == "decode" else 1
        self._bench_admission_kv_tokens = sum(prefix)
        self._real_seed_tokens = 0
        self._real_dispatches = []
        self._real_expected_seed = sum(prefix)
        self._real_deadline = time.monotonic() + REAL_POINT_TIMEOUT_SECONDS
        self._real_requests = []
        self._real_submitted = set()
        self._real_stages = []
        remaining = list(prefix)
        chunk = min(MAX_NEW, self.max_num_scheduled_tokens // point.batch_size)
        if chunk < 1:
            raise RuntimeError("native token budget cannot admit one token per request")
        while any(remaining):
            counts = [min(chunk, value) for value in remaining]
            self._real_stages.append(("seed", counts))
            remaining = [value - count for value, count in zip(remaining, counts, strict=True)]
        self._real_stages.extend(
            [("admission", suffix), ("measure", suffix)] if point.point_type == "decode" else [("measure", suffix)]
        )
        self._real_stage_index = 0
        for index in range(point.batch_size):
            prompt_len = prefix[index] if point.point_type == "decode" else prefix[index] + suffix[index]
            req_id = f"{self._real_request_set}_{point.benchmark_id}_{index}_{self._bench_seq}"
            request = Request(
                request_id=req_id,
                prompt_token_ids=token_slice(self._real_tokens, prompt_len, 131 * index + 17 * point.benchmark_id),
                sampling_params=SamplingParams(
                    max_tokens=3 if point.point_type == "decode" else 1, ignore_eos=True, temperature=0
                ),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=req_id,
            )
            request.status = RequestStatus.RUNNING
            self.requests[req_id] = request
            self.running.append(request)
            self._bench_active_req_ids.add(req_id)
            self._real_requests.append(request)
            self._real_request_manifest[req_id] = {
                "benchmark_id": point.benchmark_id,
                "repetition": self._real_repeat,
                "sampling_role": "warmup" if self._real_repeat < WARMUP_REPEATS else "measurement",
                "target_phase": "generation" if point.point_type == "decode" else "context",
                "target_query": suffix[index],
                "target_prefix": prefix[index] + int(point.point_type == "decode"),
                "target_batch_size": point.batch_size,
            }
        manifest_path = os.environ.get("AISIM_GLM53_REQUEST_MANIFEST")
        if manifest_path:
            destination = Path(manifest_path)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "request_set": self._real_request_set,
                        "dataset_role": os.environ.get("DYN_FPM_DATASET_ROLE", "calibration"),
                        "corpus_sha256": self._real_input["text_sha256"],
                        "requests": self._real_request_manifest,
                    },
                    sort_keys=True,
                )
            )
            os.replace(temporary, destination)
        self._bench_seq += 1

    def _real_output(self, stage, counts):
        if sum(counts) > self.max_num_scheduled_tokens:
            raise RuntimeError("requested real-forward point exceeds native scheduling token budget")
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = self.finished_req_ids
        output.num_common_prefix_blocks = [0] * self.kv_cache_manager.num_kv_cache_groups
        for request, count in zip(self._real_requests, counts, strict=True):
            if count == 0:
                continue
            if request.request_id not in self.requests:
                raise RuntimeError("real warm request was lost before measurement")
            blocks = self.kv_cache_manager.allocate_slots(request, count, delay_cache_blocks=True)
            if blocks is None:
                raise RuntimeError("real KV allocation failed; synthetic fallback is forbidden")
            rid = request.request_id
            output.num_scheduled_tokens[rid] = count
            if rid not in self._real_submitted:
                output.scheduled_new_reqs.append(
                    NewRequestData.from_request(
                        request, blocks.get_block_ids(), prefill_token_ids=request._all_token_ids
                    )
                )
                self._real_submitted.add(rid)
            else:
                cached = output.scheduled_cached_reqs
                cached.req_ids.append(rid)
                cached.all_token_ids[rid] = request._all_token_ids.copy()
                cached.new_block_ids.append(blocks.get_block_ids(allow_none=True))
                cached.num_computed_tokens.append(request.num_computed_tokens)
                cached.num_output_tokens.append(request.num_output_tokens + request.num_output_placeholders)
        output.total_num_scheduled_tokens = sum(counts)
        output.new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids() or None
        copies, retained = self.kv_cache_manager.take_kv_cache_block_copies()
        if copies or retained:
            raise RuntimeError("unexpected copy-on-write in private same-request real KV path")
        self._real_tags[id(output)] = stage
        self._real_outstanding += 1
        if stage == "admission" or (stage == "measure" and self._bench_current_point.point_type == "prefill"):
            self._bench_sync_pending = True
        return output

    def _real_step(self, point_type):
        if self._real_stage is not None:
            if time.monotonic() >= self._real_deadline:
                raise RuntimeError("real KV warm/measurement timed out; no fallback")
            if self._real_stage_index < len(self._real_stages):
                stage, counts = self._real_stages[self._real_stage_index]
                # Complete every seed forward before proceeding. Admission and
                # steady decode may pipeline, matching native second-step timing.
                if self._real_outstanding and stage != "measure":
                    return None
                if self._real_outstanding and self._real_stage == "seed":
                    return None
                if stage != "seed" and self._real_seed_tokens != self._real_expected_seed:
                    raise RuntimeError("measured forward requested before all real seed tokens completed")
                self._real_stage_index += 1
                self._real_stage = stage
                return self._real_output(stage, counts)
            if self._real_outstanding:
                return None
            if len(self._bench_current_fpms) != self._bench_expected_fpms:
                raise RuntimeError("real KV point did not yield the exact native FPM count")
            point = self._bench_current_point
            role = "warmup" if self._real_repeat < WARMUP_REPEATS else "measurement"
            stream = {
                "benchmark_id": point.benchmark_id,
                "sampling_role": role,
                "repetition": self._real_repeat,
                "requests": [
                    {
                        "request_index": index,
                        "request_id": request.request_id,
                        "prompt_token_ids": list(request.prompt_token_ids),
                        "output_token_ids": list(request._all_token_ids[len(request.prompt_token_ids) :]),
                        "computed_tokens": request.num_computed_tokens,
                    }
                    for index, request in enumerate(self._real_requests)
                ],
            }
            raw = json.dumps(stream, sort_keys=True, separators=(",", ":")).encode()
            self._append_real_token_history(raw)
            receipt = {
                "repetition": self._real_repeat,
                "role": role,
                "completed_seed_tokens": self._real_seed_tokens,
                "same_request": True,
                "allocated_fake_tokens": 0,
                "token_stream_sha256": hashlib.sha256(raw).hexdigest(),
                "dispatches": self._real_dispatches,
                "fpms": list(self._bench_current_fpms),
            }
            repetitions = self._real_repetitions.setdefault(point.benchmark_id, [])
            repetitions.append(receipt)
            if self._real_repeat + 1 < WARMUP_REPEATS + MEASUREMENT_REPEATS:
                self._bench_cleanup_requests()
                self._real_repeat += 1
                self._real_begin(point)
                self._real_stage = "ready"
                return None
            measured = [r for r in repetitions if r["role"] == "measurement"]
            # Preserve native FPM shape validation and reduction. The ten
            # original observations are retained independently of this median.
            self._bench_current_fpms = [dict(fpm) for fpm in measured[-1]["fpms"]]
            for index, fpm in enumerate(self._bench_current_fpms):
                fpm["wall_time"] = statistics.median(r["fpms"][index]["wall_time"] for r in measured)
            point.sample_reasons.append("kvwarm_real_kv")
            self._bench_save_current_point()
            self._bench_cleanup_requests()
            self._real_repeat = 0
            self._real_stage = None
            self._real_requests = []
            return None
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
        self._real_begin(point)
        self._real_stage = "ready"
        return self._real_step(point_type)

    def _bench_step_prefill(self):
        return self._real_step("prefill")

    def _bench_step_decode(self):
        return self._real_step("decode")

    def schedule(self, throttle_prefills=False):
        if not self._bench_active:
            return super().schedule(throttle_prefills)
        try:
            self.current_step += 1
            self.kv_cache_manager.new_step_starts()
            output = self._bench_step()
            if output is None:
                output = SchedulerOutput.make_empty()
                output.finished_req_ids = self.finished_req_ids
            if output.total_num_scheduled_tokens:
                self.sched_step_seq += 1
            self._update_after_schedule(output)
            self._bench_synchronize_output(output)
            if output.total_num_scheduled_tokens:
                self._schedule_times.append(time.monotonic())
            return output
        except Exception as error:
            self._bench_abort(error)
            raise

    def _update_from_output(self, scheduler_output, model_runner_output):
        stage = self._real_tags.pop(id(scheduler_output), None)
        self._real_callback_stage = stage
        try:
            if stage in {"admission", "measure"}:
                stats = model_runner_output.cudagraph_stats
                if stats is None:
                    raise RuntimeError("missing actual graph dispatch receipt; no expected-mode substitution")
                self._real_dispatches.append({"stage": stage, **asdict(stats)})
            result = super()._update_from_output(scheduler_output, model_runner_output)
            if stage == "seed":
                self._real_seed_tokens += scheduler_output.total_num_scheduled_tokens
            if stage is not None:
                self._real_outstanding -= 1
            return result
        finally:
            self._real_callback_stage = None

    def _append_real_token_history(self, raw: bytes) -> None:
        # Called only after every native interval of this repetition completes.
        # Preserve completed histories even if a later repetition fails, without
        # holding every long-context prompt in scheduler host memory.
        stream_path = Path(self._bench_config.output_path).with_suffix(".token-streams.jsonl")
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "xb" if self._real_token_stream_count == 0 else "ab"
        with stream_path.open(mode) as destination:
            destination.write(raw)
            destination.write(b"\n")
        self._real_token_stream_digest.update(raw)
        self._real_token_stream_digest.update(b"\n")
        self._real_token_stream_count += 1

    def _bench_write_results(self):
        super()._bench_write_results()
        destination = Path(self._bench_config.output_path)
        output = json.loads(destination.read_text())
        stream_path = destination.with_suffix(".token-streams.jsonl")
        if not stream_path.exists() and self._real_token_stream_count == 0:
            stream_path.touch(exist_ok=False)
        with stream_path.open("rb") as source:
            stream_digest = hashlib.file_digest(source, "sha256").hexdigest()
        if stream_digest != self._real_token_stream_digest.hexdigest():
            raise RuntimeError("GLM token history changed before publication")
        output["input_provenance"] = dict(self._real_input or {})
        output["input_provenance"]["context_policy"] = self._real_context_policy
        output["input_provenance"]["token_stream_manifest"] = {
            "schema_version": 3,
            "file": stream_path.name,
            "sha256": stream_digest,
            "records": self._real_token_stream_count,
        }
        if output.get("limits", {}).get("max_model_len") != self._real_context_policy["runtime_context_length"]:
            raise RuntimeError("native result context limit differs from the admitted GLM context policy")
        output["context_policy"] = self._real_context_policy
        output["execution_identity"] = self._real_identity
        output["execution_mode"] = "native_graph_policy" if self._real_purpose == "fpm" else "eager_ops"
        output["observation_purpose"] = self._real_purpose
        output["ops_instrumented"] = self._real_purpose == "ops"
        output["timing_boundary"] = "vllm_native_scheduler_output_interval"
        output["kvwarm"] = {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "same_request_real_forward",
            "state_protocol": "glm53flash_same_request_real_hybrid_v1",
            "max_batch": MAX_BATCH,
            "max_context": self._real_context_policy["measured_context_limit"],
        }
        from collector.glm53flash_runtime_identity import vllm_source_manifest_sha256

        output["producer"] = {
            "instrumentation_revision": DYNAMO_SHA,
            "vllm_package_version": __import__("vllm").__version__,
            "reviewed_scheduler_api_revision": VLLM_SHA,
            "runtime_source_manifest_sha256": vllm_source_manifest_sha256(
                __import__("vllm").__version__, Path(__file__).with_name("runtime-source-sha256.json")
            ),
            "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context_policy_version": 1,
            "warmup_repeats": WARMUP_REPEATS,
            "measurement_repeats": MEASUREMENT_REPEATS,
        }
        for item in output["results"]:
            item["kv_seed_regime"] = "real_kv"
            item["real_hybrid_repetitions"] = self._real_repetitions[item["point"]["benchmark_id"]]
        for item in output["iteration_groups"]:
            item["kv_seed_regime"] = "real_kv"
        temporary = destination.with_suffix(destination.suffix + ".real.tmp")
        temporary.write_text(json.dumps(output, indent=2))
        os.replace(temporary, destination)


# Spawned workers may import this class by its defining module before the
# configured native scheduler path. Publish only after class creation so the
# lazy source-checking hook can complete either import order without recursion.
if os.environ.get("DYN_FPM_GLM53FLASH_REAL_KV") == "1":
    native.InstrumentedScheduler = Glm53FlashRealKVScheduler
