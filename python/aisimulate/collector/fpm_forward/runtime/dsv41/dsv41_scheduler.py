# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scheduler integration adapted from ai-dynamo/dynamo at
# 54960177085413259859c88bd34ed0734d4c2ea9, components/src/dynamo/vllm/instrumented_scheduler.py.
# Modified: bounded same-request real-KV collection for vLLM V4.1 preview.
"""Canary-only native FPM extension; see README.md for immutable API pins.

Every state-bearing request is filled by model forwards from position zero.
The same request and block tables survive through its measured suffix/decode.
This module never assigns num_computed_tokens or imports synthetic KV.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import dynamo.vllm.instrumented_scheduler as native
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

DYNAMO_SHA = "54960177085413259859c88bd34ed0734d4c2ea9"
VLLM_SHA = "79a7108d9aea27ddab99ce1779290d300b17fc23"
MODEL_SHA = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"
MAX_BATCH = 2
MAX_CONTEXT = 2048
MAX_NEW = 512


def token_slice(pool: list[int], length: int, offset: int) -> list[int]:
    """Rotate and repeat the tokenizer's real-text stream deterministically."""
    if not pool or length < 0:
        raise ValueError("non-empty token pool and nonnegative length required")
    return [pool[(offset + index) % len(pool)] for index in range(length)]


class DeepseekV41RealKVScheduler(native.InstrumentedScheduler):
    def _bench_init(self, config):
        self._real_tags = {}
        self._real_callback_stage = None
        self._real_stage = None
        self._real_requests = []
        self._real_stages = []
        self._real_stage_index = 0
        self._real_submitted = set()
        self._real_outstanding = 0
        self._real_deadline = 0.0
        self._real_token_streams = []
        self._real_witnesses = {}
        self._real_warmup_results = []
        self._real_expected_warmup_ids = []
        self._real_seed_tokens = 0
        self._real_input = None
        self._real_identity = None
        super()._bench_init(config)
        if not self._bench_active:
            raise ValueError("V4.1 canary overlay requires native benchmark mode")
        if config.model_config.enforce_eager is not True:
            raise ValueError("V4.1 real-KV collection requires enforce_eager=true; graph timing is not qualified")
        parallel = config.parallel_config
        if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size) != (4, 1, 1):
            raise ValueError("V4.1 canary requires pure TP4 / PP1 / DP1")
        if (
            parallel.use_ubatching
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("V4.1 canary requires no ubatching/DBO and no context parallelism")
        if parallel.enable_expert_parallel or config.speculative_config is not None:
            raise ValueError("V4.1 canary does not support EP or speculative decoding")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError("V4.1 canary forbids KV/encoder connectors")
        engram = config.engram_config
        if engram is None or engram.cpu_offload:
            raise ValueError("V4.1 canary requires explicit Engram cpu_offload=false")
        if self._bench_explicit_points is None and (
            self.max_model_len > 2050
            or self.max_num_running_reqs > MAX_BATCH
            or self.max_num_scheduled_tokens > MAX_NEW
        ):
            raise ValueError("automatic V4.1 canary grid requires model_len<=2050, max_seqs<=2, batched_tokens<=512")
        if self._bench_config.warmup_iterations != 0:
            raise ValueError("set native warmup_iterations=0; eager shape warmups remain enabled")
        model_path = Path(config.model_config.model)
        raw_config = json.loads((model_path / "config.json").read_text())
        if (
            config.model_config.architecture != "DeepseekV41ForCausalLM"
            or config.model_config.hf_config.model_type != "deepseek_v41"
        ):
            raise ValueError("loaded model must be DeepSeek V4.1")
        from aiconfigurator_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity

        self._real_identity = dict(zip(EXECUTION_COLUMNS, execution_identity(raw_config), strict=True))
        revision = os.environ.get("DYN_FPM_TOKENIZER_REVISION")
        if revision != MODEL_SHA:
            raise ValueError("tokenizer revision must equal the pinned V4.1 checkpoint revision")
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

    def _kvwarm_warm_eligible(self):
        return True

    def _bench_blocks_per_req(self, num_tokens, *, has_cache_hit=False, apply_admission_cap=False):
        # Use the pinned managers' native capacity semantics, including one-block rings.
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id="__dsv41_capacity_probe__",
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

    def _real_validate_grid(self):
        self._real_expected_warmup_ids = sorted(
            point.benchmark_id for point in self._bench_grid if native.EAGER_WARMUP_REASON in point.sample_reasons
        )
        for point in self._bench_grid:
            prefix, suffix = self._real_lengths(point)
            if (
                not 1 <= point.batch_size <= MAX_BATCH
                or max(prefix, default=0) + int(point.point_type == "decode") > MAX_CONTEXT
            ):
                raise ValueError("V4.1 canary point exceeds batch/context bound")
            if point.point_type == "prefill":
                if sum(suffix) > MAX_NEW:
                    raise ValueError("V4.1 canary prefill exceeds total new-token bound")
                if any(p + q > MAX_CONTEXT for p, q in zip(prefix, suffix, strict=True)):
                    raise ValueError("V4.1 canary prefill exceeds prefix plus new-token context bound")
            if point.point_type == "decode" and min(prefix) < 1:
                raise ValueError("V4.1 real decode requires context >=2; no coordinate clamping")

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
        raise RuntimeError("synthetic prefix KV is forbidden for V4.1")

    def _bench_inject_fake_decode(self, *args, **kwargs):
        raise RuntimeError("synthetic decode KV is forbidden for V4.1")

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
        self._real_expected_seed = sum(prefix)
        self._real_deadline = time.monotonic() + self._bench_point_result_timeout_seconds
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
            req_id = f"__dsv41_real_{point.benchmark_id}_{index}_{self._bench_seq}"
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
            stream = {
                "benchmark_id": self._bench_current_point.benchmark_id,
                "sampling_role": "warmup"
                if native.EAGER_WARMUP_REASON in self._bench_current_point.sample_reasons
                else "measurement",
                "requests": [
                    {
                        "request_index": index,
                        "prompt_token_ids": list(request.prompt_token_ids),
                        "output_token_ids": list(request._all_token_ids[len(request.prompt_token_ids) :]),
                        "computed_tokens": request.num_computed_tokens,
                    }
                    for index, request in enumerate(self._real_requests)
                ],
            }
            stream_bytes = json.dumps(stream, sort_keys=True, separators=(",", ":")).encode()
            self._real_token_streams.append(stream_bytes)
            self._bench_current_point.sample_reasons.append("kvwarm_real_kv")
            self._real_witnesses[self._bench_current_point.benchmark_id] = {
                "completed_seed_tokens": self._real_seed_tokens,
                "same_request": True,
                "allocated_fake_tokens": 0,
                "token_stream_sha256": hashlib.sha256(stream_bytes).hexdigest(),
            }
            if stream["sampling_role"] == "warmup":
                # Native 5496017 prepends eager replicas with IDs after the
                # measured range and discards them at save time (2376-2500,
                # 4225-4243). Preserve their real histories under an explicit
                # role, while only measured rows enter calibration curves.
                point = self._bench_current_point
                self._real_warmup_results.append(
                    {
                        "point": {
                            name: getattr(point, name)
                            for name in (
                                "benchmark_id",
                                "point_type",
                                "batch_size",
                                "total_prefill_tokens",
                                "total_kv_read_tokens",
                                "sample_reasons",
                            )
                        },
                        "real_kv_witness": dict(self._real_witnesses[point.benchmark_id]),
                        "fpms": list(self._bench_current_fpms),
                    }
                )
            self._bench_save_current_point()
            self._bench_cleanup_requests()
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
            result = super()._update_from_output(scheduler_output, model_runner_output)
            if stage == "seed":
                self._real_seed_tokens += scheduler_output.total_num_scheduled_tokens
            if stage is not None:
                self._real_outstanding -= 1
            return result
        finally:
            self._real_callback_stage = None

    def _bench_write_results(self):
        super()._bench_write_results()
        destination = Path(self._bench_config.output_path)
        output = json.loads(destination.read_text())
        stream_bytes = b"\n".join(self._real_token_streams) + (b"\n" if self._real_token_streams else b"")
        stream_path = destination.with_suffix(".token-streams.jsonl")
        stream_tmp = stream_path.with_suffix(".jsonl.tmp")
        stream_tmp.write_bytes(stream_bytes)
        os.replace(stream_tmp, stream_path)
        output["input_provenance"] = dict(self._real_input or {})
        output["input_provenance"]["token_stream_manifest"] = {
            "schema_version": 2,
            "warmup_benchmark_ids": self._real_expected_warmup_ids,
            "file": stream_path.name,
            "sha256": hashlib.sha256(stream_bytes).hexdigest(),
            "records": len(self._real_token_streams),
        }
        output["warmup_results"] = self._real_warmup_results
        output["execution_identity"] = self._real_identity
        output["execution_mode"] = "eager"
        output["kvwarm"] = {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "same_request_real_forward",
            "max_batch": MAX_BATCH,
            "max_context": MAX_CONTEXT,
        }
        output["producer"] = {
            "instrumentation_revision": DYNAMO_SHA,
            "dynamo_revision": None,
            "vllm_revision": None,
            "vllm_package_version": __import__("vllm").__version__,
            "reviewed_scheduler_api_revision": VLLM_SHA,
            "runtime_source_manifest_sha256": hashlib.sha256(
                Path(__file__).with_name("runtime-source-sha256.json").read_bytes()
            ).hexdigest(),
            "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        for item in output["results"]:
            item["kv_seed_regime"] = "real_kv"
            item["real_kv_witness"] = self._real_witnesses[item["point"]["benchmark_id"]]
        for item in output["iteration_groups"]:
            item["kv_seed_regime"] = "real_kv"
        temporary = destination.with_suffix(destination.suffix + ".real.tmp")
        temporary.write_text(json.dumps(output, indent=2))
        os.replace(temporary, destination)


# Spawned workers may import this class by its defining module before the
# configured native scheduler path. Publish only after class creation so the
# lazy source-checking hook can complete either import order without recursion.
if os.environ.get("DYN_FPM_DSV41_REAL_KV") == "1":
    native.InstrumentedScheduler = DeepseekV41RealKVScheduler
