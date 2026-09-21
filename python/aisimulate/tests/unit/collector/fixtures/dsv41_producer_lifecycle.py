# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU state-machine contract tests; live pinned-image preflight remains required."""

import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS


def module(name, **values):
    obj = types.ModuleType(name)
    obj.__dict__.update(values)
    sys.modules[name] = obj
    return obj


class Output:
    @classmethod
    def make_empty(cls):
        return SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], num_computed_tokens=[], num_output_tokens=[], all_token_ids={}
            ),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
        )


class NewData:
    @classmethod
    def from_request(cls, request, block_ids, **kwargs):
        return SimpleNamespace(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids[:],
            num_computed_tokens=request.num_computed_tokens,
            block_ids=block_ids,
        )


class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._all_token_ids = self.prompt_token_ids[:]
        self.num_computed_tokens = 0
        self.num_output_tokens = 0
        self.num_output_placeholders = 0


class Base:
    def _bench_decode_context_lengths(self, total, batch):
        assert total % batch == 0
        return [total // batch] * batch

    def _bench_prefill_kv_read_lengths(self, total, batch, *args):
        return [total // batch] * batch

    _bench_prefill_new_token_lengths = _bench_prefill_kv_read_lengths

    def _bench_stop_at_timeout_boundary(self, point_type):
        return False

    def _bench_pop_next(self, point_type):
        return self._bench_grid.popleft() if self._bench_grid else None

    def _bench_save_current_point(self):
        self.saved.append((self._bench_current_point, self._bench_current_fpms[:]))
        self._bench_current_point = None
        self._bench_current_fpms = []

    def _bench_cleanup_requests(self):
        self.requests.clear()
        self.running.clear()
        self._bench_active_req_ids.clear()

    def _update_from_output(self, output, model_output):
        if self._real_callback_stage in {"admission", "measure"}:
            self._bench_current_fpms.append(self._real_callback_stage)

    def _bench_should_record_scheduled(self, scheduled):
        return True

    def _bench_write_results(self):
        payload = {
            "schema_version": 2,
            "artifact_type": "rank",
            "status": "complete",
            "valid": True,
            "results": [{"point": {"benchmark_id": 1}, "fpms": []}],
            "iteration_groups": [{"benchmark_id": 1}],
        }
        Path(self._bench_config.output_path).write_text(json.dumps(payload))

    def _bench_build_explicit_grid(self, points, *, generated=False):
        self._bench_grid = deque(points)


for name in ["dynamo", "dynamo.vllm", "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"]:
    module(name)
module(
    "dynamo.vllm.instrumented_scheduler",
    InstrumentedScheduler=Base,
    EAGER_WARMUP_REASON="eager_warmup",
    _BenchPhase=SimpleNamespace(DECODE_SWEEP="decode", DONE="done"),
)
module("vllm.sampling_params", SamplingParams=lambda **kw: SimpleNamespace(**kw))
module("vllm.tokenizers", get_tokenizer=None)
module("vllm.v1.core.sched.output", CachedRequestData=None, NewRequestData=NewData, SchedulerOutput=Output)
module("vllm.v1.request", Request=Request, RequestStatus=SimpleNamespace(RUNNING="running"))
sys.modules["vllm"].__version__ = "test-runtime"
spec = importlib.util.spec_from_file_location("dsv41_tested", Path(os.environ["AIC_FPM_DSV41_PRODUCER"]))
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


def point(kind, batch=2, context=512, new=64):
    return SimpleNamespace(
        point_type=kind,
        batch_size=batch,
        total_kv_read_tokens=batch * context,
        total_prefill_tokens=batch * new if kind == "prefill" else 0,
        partition=None,
        rows=None,
        sample_reasons=[],
        benchmark_id=1,
    )


def scheduler(pt):
    obj = object.__new__(impl.DeepseekV41RealKVScheduler)
    obj._real_tags = {}
    obj._real_callback_stage = None
    obj._real_stage = None
    obj._real_outstanding = 0
    obj._real_tokens = [3, 8, 27, 42, 11, 79, 14]
    obj._real_token_streams = []
    obj._real_witnesses = {}
    obj._real_warmup_results = []
    obj._real_expected_warmup_ids = []
    obj._bench_point_result_timeout_seconds = 30
    obj._bench_seq = 0
    obj._bench_block_hasher = None
    obj._bench_active_req_ids = set()
    obj._bench_grid = deque([pt])
    obj._bench_config = SimpleNamespace(mode=pt.point_type)
    obj.max_num_scheduled_tokens = 1024
    obj.requests = {}
    obj.running = []
    obj.finished_req_ids = set()
    obj.saved = []
    obj._bench_sync_pending = False
    obj.kv_cache_manager = SimpleNamespace(
        num_kv_cache_groups=3, take_new_block_ids=lambda: [], take_kv_cache_block_copies=lambda: ([], [])
    )
    obj.kv_cache_manager.allocate_slots = Mock(
        side_effect=lambda request, count, **kwargs: SimpleNamespace(
            get_block_ids=lambda **kw: ([hash(request.request_id) % 101], [7], [11])
        )
    )
    return obj


class Driver:
    """A FIFO worker stand-in that only fills memory after a forward completes."""

    def __init__(self, obj):
        self.obj = obj
        self.memory = {}
        self.inflight = []
        self.identities = {}

    def submit(self, output):
        if output is None:
            return
        snapshots = {}
        for rid, count in output.num_scheduled_tokens.items():
            request = self.obj.requests[rid]
            self.identities.setdefault(rid, id(request))
            assert self.identities[rid] == id(request)
            snapshots[rid] = (request.num_computed_tokens, count)
            request.num_computed_tokens += count
            if request.num_computed_tokens >= len(request.prompt_token_ids):
                request.num_output_placeholders += 1
        self.inflight.append((output, snapshots))

    def finish(self):
        output, snapshots = self.inflight.pop(0)
        for rid, (start, count) in snapshots.items():
            request = self.obj.requests[rid]
            memory = self.memory.setdefault(rid, [])
            assert len(memory) == start, "a forward cannot read uninitialized KV"
            tokens = request._all_token_ids[start : start + count]
            assert len(tokens) == count, "forward tokens must exist in real history"
            memory.extend(tokens)
            if start + count >= len(request.prompt_token_ids):
                request._all_token_ids.append(13 + request.num_output_tokens)
                request.num_output_tokens += 1
                request.num_output_placeholders -= 1
        self.obj._update_from_output(output, SimpleNamespace())


class RealKVTests(unittest.TestCase):
    def test_initialization_uses_validated_top_level_engram(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}")
            config = SimpleNamespace(
                model_config=SimpleNamespace(
                    enforce_eager=True,
                    model=directory,
                    architecture="DeepseekV41ForCausalLM",
                    hf_config=SimpleNamespace(model_type="deepseek_v41"),
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    use_ubatching=False,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                    enable_expert_parallel=False,
                ),
                engram_config=SimpleNamespace(cpu_offload=False),
                speculative_config=None,
            )
            obj = object.__new__(impl.DeepseekV41RealKVScheduler)
            obj._bench_active = True
            obj.connector = obj.ec_connector = None
            obj._bench_explicit_points = [point("prefill")]
            obj._bench_config = SimpleNamespace(warmup_iterations=0)
            identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
            with (
                patch.object(Base, "_bench_init", create=True),
                patch("aisimulate_core.sdk.fpm_identity.execution_identity", return_value=identity) as identify,
                patch.dict(os.environ, {"DYN_FPM_TOKENIZER_REVISION": "invalid"}),
                self.assertRaisesRegex(ValueError, "tokenizer revision"),
            ):
                obj._bench_init(config)
            identify.assert_called_once_with({}, engram_cpu_offload=False, input_modality="text")
            self.assertEqual(obj._real_identity, dict(zip(EXECUTION_COLUMNS, identity, strict=True)))

    def test_decode_warms_actual_requests_then_pipelines_two_decode_steps(self):
        obj = scheduler(point("decode"))
        worker = Driver(obj)
        seed = obj._real_step("decode")
        worker.submit(seed)
        self.assertEqual(seed.total_num_scheduled_tokens, 1022)
        self.assertIsNone(obj._real_step("decode"))
        self.assertEqual(obj._real_seed_tokens, 0)
        worker.finish()
        admission = obj._real_step("decode")
        worker.submit(admission)
        self.assertEqual(admission.scheduled_cached_reqs.num_computed_tokens, [511, 511])
        steady = obj._real_step("decode")
        worker.submit(steady)
        self.assertEqual(steady.scheduled_cached_reqs.num_computed_tokens, [512, 512])
        self.assertIsNone(obj._real_step("decode"))
        worker.finish()
        worker.finish()
        obj._real_step("decode")
        self.assertEqual(obj.saved[0][1], ["admission", "measure"])
        self.assertEqual(obj._real_witnesses[1]["completed_seed_tokens"], 1022)
        self.assertTrue(all(len(values) == 513 for values in worker.memory.values()))
        self.assertTrue(all(len(set(values)) > 2 for values in worker.memory.values()))
        manifest = json.loads(obj._real_token_streams[0])
        self.assertEqual(len(manifest["requests"][0]["prompt_token_ids"]), 511)
        self.assertEqual(len(manifest["requests"][0]["output_token_ids"]), 3)
        self.assertEqual(
            obj._real_witnesses[1]["token_stream_sha256"], impl.hashlib.sha256(obj._real_token_streams[0]).hexdigest()
        )

    def test_cached_prefill_preserves_full_prompt_history_and_same_state(self):
        obj = scheduler(point("prefill", context=1024, new=64))
        worker = Driver(obj)
        for _ in range(2):
            output = obj._real_step("prefill")
            worker.submit(output)
            self.assertIsNone(obj._real_step("prefill"))
            worker.finish()
        measured = obj._real_step("prefill")
        self.assertEqual(measured.scheduled_cached_reqs.num_computed_tokens, [1024, 1024])
        self.assertEqual(measured.scheduled_cached_reqs.num_output_tokens, [0, 0])
        worker.submit(measured)
        worker.finish()
        obj._real_step("prefill")
        self.assertEqual(obj.saved[0][1], ["measure"])
        self.assertEqual(obj._real_witnesses[1]["completed_seed_tokens"], 2048)
        self.assertTrue(all(len(values) == 1088 for values in worker.memory.values()))

    def test_zero_prefix_prefill_is_a_real_forward(self):
        obj = scheduler(point("prefill", context=0))
        worker = Driver(obj)
        output = obj._real_step("prefill")
        self.assertEqual(len(output.scheduled_new_reqs), 2)
        worker.submit(output)
        worker.finish()
        obj._real_step("prefill")
        self.assertEqual(obj.saved[0][1], ["measure"])

    def test_native_eager_warmup_keeps_a_separate_history_role(self):
        pt = point("prefill", context=0)
        pt.benchmark_id = 5
        pt.sample_reasons = ["eager_warmup"]
        obj = scheduler(pt)
        obj._real_validate_grid()
        worker = Driver(obj)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        obj._real_step("prefill")
        stream = json.loads(obj._real_token_streams[0])
        self.assertEqual(stream["sampling_role"], "warmup")
        self.assertEqual(stream["benchmark_id"], 5)
        self.assertEqual(obj._real_warmup_results[0]["point"]["benchmark_id"], 5)
        self.assertEqual(obj._real_warmup_results[0]["real_kv_witness"], obj._real_witnesses[5])
        self.assertEqual(obj._real_warmup_results[0]["fpms"], ["measure"])
        self.assertEqual(obj._real_expected_warmup_ids, [5])

    def test_allocation_failure_has_no_fallback(self):
        obj = scheduler(point("decode"))
        obj.kv_cache_manager.allocate_slots.return_value = None
        obj.kv_cache_manager.allocate_slots.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
            obj._real_step("decode")
        self.assertEqual(obj.saved, [])

    def test_lost_state_fails(self):
        obj = scheduler(point("prefill"))
        worker = Driver(obj)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        obj.requests.clear()
        with self.assertRaisesRegex(RuntimeError, "was lost"):
            obj._real_step("prefill")

    def test_timeout_never_publishes_real_kv(self):
        obj = scheduler(point("decode"))
        obj._real_step("decode")
        obj._real_deadline = 0
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            obj._real_step("decode")
        self.assertEqual(obj.saved, [])

    def test_real_seed_jit_has_bounded_deadline_independent_of_fake_point_timeout(self):
        obj = scheduler(point("prefill", context=512, new=64))
        obj._bench_point_result_timeout_seconds = 8.0
        worker = Driver(obj)
        with patch.object(impl.time, "monotonic", return_value=100.0):
            worker.submit(obj._real_step("prefill"))
        self.assertEqual(obj._real_deadline, 220.0)
        with patch.object(impl.time, "monotonic", return_value=109.0):
            self.assertIsNone(obj._real_step("prefill"))
        worker.finish()
        with patch.object(impl.time, "monotonic", return_value=190.0):
            worker.submit(obj._real_step("prefill"))
        worker.finish()
        with patch.object(impl.time, "monotonic", return_value=191.0):
            obj._real_step("prefill")
        self.assertEqual(obj.saved[0][1], ["measure"])
        self.assertEqual(obj._bench_point_result_timeout_seconds, 8.0)

        obj = scheduler(point("prefill", context=512, new=64))
        with patch.object(impl.time, "monotonic", return_value=100.0):
            obj._real_step("prefill")
        with (
            patch.object(impl.time, "monotonic", return_value=220.0),
            self.assertRaisesRegex(RuntimeError, "timed out"),
        ):
            obj._real_step("prefill")
        self.assertEqual(obj.saved, [])
        self.assertEqual(obj._real_witnesses, {})

    def test_synthetic_entry_points_are_blocked(self):
        obj = scheduler(point("decode"))
        for method in [obj._bench_cache_fake_prefixes, obj._bench_inject_fake_decode]:
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                method()

    def test_canary_limits_fail_instead_of_silently_skipping(self):
        for pt in [
            point("decode", batch=3),
            point("decode", context=2049),
            point("decode", context=1),
            point("prefill", new=513),
            point("prefill", batch=2, new=257),
            point("prefill", context=2000, new=128),
        ]:
            with self.subTest(point=pt):
                obj = scheduler(pt)
                with self.assertRaises(ValueError):
                    obj._real_validate_grid()
        for pt in [
            point("decode", context=2048),
            point("prefill", batch=1, context=1536, new=512),
            point("prefill", batch=2, context=1536, new=256),
        ]:
            with self.subTest(boundary=pt):
                scheduler(pt)._real_validate_grid()

    def test_native_ring_capacity_semantics_are_used(self):
        obj = scheduler(point("decode"))
        ring = SimpleNamespace(get_num_blocks_to_allocate=Mock(return_value=1))
        dense = SimpleNamespace(get_num_blocks_to_allocate=Mock(return_value=32))
        obj.kv_cache_manager.coordinator = SimpleNamespace(single_type_managers=[ring, dense])
        self.assertEqual(obj._bench_blocks_per_req(2048), 33)
        self.assertEqual(ring.get_num_blocks_to_allocate.call_args.kwargs["num_tokens"], 2048)

    def test_marker_is_earned_after_real_seed_and_measure_complete(self):
        pt = point("prefill")
        obj = scheduler(pt)
        obj._real_validate_grid()
        self.assertNotIn("kvwarm_real_kv", pt.sample_reasons)
        worker = Driver(obj)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        self.assertNotIn("kvwarm_real_kv", pt.sample_reasons)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        obj._real_step("prefill")
        self.assertIn("kvwarm_real_kv", pt.sample_reasons)

    def test_uneven_native_lengths_preserve_each_request_prefix(self):
        pt = point("prefill", context=512, new=64)
        obj = scheduler(pt)
        obj._real_lengths = lambda _: ([512, 256], [129, 128])
        worker = Driver(obj)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        measured = obj._real_step("prefill")
        self.assertEqual(measured.scheduled_cached_reqs.num_computed_tokens, [512, 256])
        self.assertEqual(list(measured.num_scheduled_tokens.values()), [129, 128])
        worker.submit(measured)
        worker.finish()
        obj._real_step("prefill")
        self.assertEqual(sorted(map(len, worker.memory.values())), [384, 641])

    def test_shorter_prefix_parks_without_rebuilding_its_state(self):
        obj = scheduler(point("prefill"))
        obj._real_lengths = lambda _: ([1024, 256], [129, 128])
        worker = Driver(obj)
        worker.submit(obj._real_step("prefill"))
        worker.finish()
        parked = obj._real_step("prefill")
        self.assertEqual(len(parked.num_scheduled_tokens), 1)
        worker.submit(parked)
        worker.finish()
        measured = obj._real_step("prefill")
        self.assertEqual(measured.scheduled_cached_reqs.num_computed_tokens, [1024, 256])
        self.assertEqual(len(measured.scheduled_cached_reqs.all_token_ids), 2)
        worker.submit(measured)
        worker.finish()
        obj._real_step("prefill")
        self.assertEqual(sorted(map(len, worker.memory.values())), [384, 1153])

    def test_producer_serialized_provenance_matches_actual_collector_reader(self):
        # Execute the actual reader function from the concurrently developed
        # Collector source, without importing its unrelated planning dependencies.
        consumer_path = Path(os.environ["AIC_FPM_NATIVE_ARTIFACT"])
        tree = ast.parse(consumer_path.read_text())
        names = {"_validate_execution_provenance", "_validate_kvwarm_contract"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions), 2)
        columns = EXECUTION_COLUMNS
        scope = {
            "Path": Path,
            "Any": Any,
            "FPMCell": SimpleNamespace,
            "re": re,
            "EXECUTION_COLUMNS": columns,
            "KVWARM_STRATEGIES": {"pure_tp"},
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(consumer_path), "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            obj = scheduler(point("decode"))
            path = Path(directory) / "benchmark_results.json"
            obj._bench_config.output_path = str(path)
            obj._real_identity = dict(zip(columns, ("a" * 64, "full", "hbm_tp_sharded", "text"), strict=True))
            obj._real_input = {
                "source": "tokenizer_text",
                "text_sha256": "b" * 64,
                "token_ids_sha256": "c" * 64,
                "tokenizer_revision": impl.MODEL_SHA,
                "token_count": 10,
                "unique_token_count": 7,
            }
            obj._real_witnesses = {1: {"completed_seed_tokens": 1022, "same_request": True, "allocated_fake_tokens": 0}}
            obj._bench_write_results()
            payload = json.loads(path.read_text())
            self.assertIsNone(payload["producer"]["vllm_revision"])
            self.assertIsNone(payload["producer"]["dynamo_revision"])
            self.assertEqual(payload["producer"]["instrumentation_revision"], impl.DYNAMO_SHA)
            self.assertEqual(payload["producer"]["vllm_package_version"], "test-runtime")
            self.assertEqual(payload["producer"]["collection_timeouts"]["same_request_seed_and_measure_seconds"], 120.0)
            cell = SimpleNamespace(
                execution_identity=tuple(obj._real_identity.values()),
                input_text_sha256="b" * 64,
                workload_kind="decode",
                parallel_strategy="pure_tp",
            )
            evidence = scope["_validate_execution_provenance"](cell, payload, path)
            for field, value in obj._real_input.items():
                self.assertEqual(evidence[field], value)
            self.assertEqual(evidence["token_stream_manifest"]["records"], 0)
            self.assertTrue((path.parent / evidence["token_stream_manifest"]["file"]).is_file())
            scope["_validate_kvwarm_contract"](cell, payload["kvwarm"], path)
            self.assertEqual(payload["results"][0]["kv_seed_regime"], "real_kv")
            self.assertEqual(payload["results"][0]["real_kv_witness"]["completed_seed_tokens"], 1022)
            payload["execution"] = payload.pop("execution_identity")
            with self.assertRaisesRegex(ValueError, "execution identity"):
                scope["_validate_execution_provenance"](cell, payload, path)

    def test_token_slices_are_repeatable_and_offset_varies_history(self):
        self.assertEqual(impl.token_slice([1, 3, 7], 5, 1), [3, 7, 1, 3, 7])
        self.assertNotEqual(impl.token_slice([1, 3, 7], 8, 0), impl.token_slice([1, 3, 7], 8, 1))


if __name__ == "__main__":
    unittest.main()
