#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original CPU-only regression tests for the separately attributed native patch.

Run with unittest. Set SGLANG_FPM_BASE_SOURCE to the exact upstream full file to
add full-file patch/hash verification; the bounded fixture remains offline.
No engine packages, CUDA, tokenizer, model, or GPU execution are involved.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "provenance.json").read_text())
SOURCE_PATH = Path(META["upstream_path"])
PATCH_PATH = ROOT / "sglang-native-five-fields.patch"


def patched_text(original):
    with tempfile.TemporaryDirectory(prefix="sglang-fpm-patch-test-") as name:
        work = Path(name)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_text(original)
        for check in (True, False):
            command = ["git", "apply"] + (["--check"] if check else [])
            subprocess.run(command + [str(PATCH_PATH)], cwd=work, check=True, capture_output=True)
        return target.read_text()


def method_from_source(source):
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerMetricsReporter"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_scheduled_request_metrics"
    )
    namespace = {}
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(method)
    exec(compile(ast.fix_missing_locations(module), "<native-metrics-method>", "exec"), namespace)
    return namespace[method.name]


class Accumulator:
    """Minimal independent scalar stand-in for the native accumulator API."""

    def __init__(self):
        self.values = []

    def add(self, value):
        self.values.append(value)

    @property
    def count(self):
        return len(self.values)

    @property
    def total(self):
        return sum(self.values)

    def variance(self):
        if not self.values:
            return 0.0
        mean = self.total / self.count
        return sum((value - mean) ** 2 for value in self.values) / self.count


class Mode:
    def __init__(self, value):
        self.value = value

    def is_mixed(self):
        return self.value == "mixed"

    def is_extend(self):
        return self.value in ("extend", "mixed")

    def is_decode(self):
        return self.value == "decode"


def request(prompt=31, mutated_prefix=31, seqlen=32):
    return SimpleNamespace(origin_input_ids=range(prompt), prefix_indices=range(mutated_prefix), seqlen=seqlen)


def batch(mode, reqs, extend=None, prefix=None, decoding=(), seq_lens=(), stats=None):
    return SimpleNamespace(
        forward_mode=Mode(mode),
        reqs=reqs,
        extend_lens=extend,
        prefix_lens=prefix,
        decoding_reqs=decoding,
        seq_lens_cpu=seq_lens,
        prefill_stats=stats,
    )


def stats(tokens, count=1):
    return SimpleNamespace(log_input_tokens=tokens, num_new_seqs=count)


def fields(result):
    return tuple(
        getattr(result, key)
        for key in (
            "num_prefill_requests",
            "sum_prefill_tokens",
            "sum_prefill_kv_tokens",
            "num_decode_requests",
            "sum_decode_kv_tokens",
        )
    )


class NativeFiveFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "fixtures/metrics_reporter_scheduled_method.py.txt").read_text()
        cls.original = staticmethod(method_from_source(source))
        cls.modified = staticmethod(method_from_source(patched_text(source)))

    def call(self, sample, original=False):
        module_name = "sglang.srt.observability.forward_pass_metrics"
        module = ModuleType(module_name)
        module.ScheduledRequestMetrics = SimpleNamespace
        module.WelfordAccumulator = Accumulator
        with patch.dict("sys.modules", {module_name: module}):
            method = self.original if original else self.modified
            return method(None, sample)

    def test_archived_cold_prefill_reproduces_stock_mismatch(self):
        sample = batch("extend", [request()], [31], [0], stats=stats(256))
        self.assertEqual(fields(self.call(sample, original=True)), (1, 256, 31, 0, 0))
        self.assertEqual(fields(self.call(sample)), (1, 31, 0, 0, 0))

    def test_short_prompt_ignores_allocation_page_rounding(self):
        sample = batch("extend", [request(4, 0)], [4], [0], stats=stats(256))
        self.assertEqual(fields(self.call(sample)), (1, 4, 0, 0, 0))

    def test_cache_hit_uses_prefix_before_result_cache_update(self):
        sample = batch("extend", [request(519, 519)], [7], [512], stats=stats(256))
        self.assertEqual(fields(self.call(sample)), (1, 7, 512, 0, 0))

    def test_chunked_prefill_counts_chunk_with_its_prior_context(self):
        sample = batch("extend", [request(2048, 1056)], [32], [1024], stats=stats(256))
        self.assertEqual(fields(self.call(sample)), (1, 32, 1024, 0, 0))

    def test_multiple_prefills_do_not_round_individual_requests(self):
        sample = batch(
            "extend",
            [request(257, 257), request(769, 769)],
            [1, 257],
            [256, 512],
            stats=stats(768, 2),
        )
        self.assertEqual(fields(self.call(sample)), (2, 258, 768, 0, 0))

    def test_missing_logging_stats_does_not_zero_real_prefill(self):
        sample = batch("extend", [request(17, 17)], [17], [0])
        self.assertEqual(fields(self.call(sample)), (1, 17, 0, 0, 0))

    def test_mixed_indices_preserve_decode_and_skip_decode_positions(self):
        d0, p0, d1, p1 = request(seqlen=13), request(), request(seqlen=43), request()
        sample = batch(
            "mixed",
            [d0, p0, d1, p1],
            [1, 31, 1, 19],
            [12, 512, 42, 1024],
            decoding=[d1, d0],
            stats=stats(512, 2),
        )
        self.assertEqual(fields(self.call(sample)), (2, 50, 1536, 2, 56))
        self.assertEqual(fields(self.call(sample))[3:], fields(self.call(sample, True))[3:])

    def test_pure_decode_uses_original_native_inclusive_sequence_lengths(self):
        sample = batch("decode", [request(), request()], seq_lens=[32, 48])
        self.assertEqual(fields(self.call(sample)), (0, 0, 0, 2, 80))
        self.assertEqual(vars(self.call(sample)), vars(self.call(sample, True)))

    def test_all_decode_mixed_batch_needs_no_prefill_metadata(self):
        req = request(seqlen=32)
        sample = batch("mixed", [req], decoding=[req])
        self.assertEqual(fields(self.call(sample)), (0, 0, 0, 1, 32))

    def test_idle_remains_zero(self):
        self.assertEqual(fields(self.call(batch("idle", []))), (0, 0, 0, 0, 0))

    def test_padded_metadata_tail_does_not_add_synthetic_requests(self):
        sample = batch("extend", [request()], [31, 225], [0, 0], stats=stats(256))
        self.assertEqual(fields(self.call(sample)), (1, 31, 0, 0, 0))

    def test_recorded_derivative_hashes(self):
        for name, key in (
            ("sglang-native-five-fields.patch", "patch_sha256"),
            ("fixtures/metrics_reporter_scheduled_method.py.txt", "fixture_sha256"),
            (META["license_path"], "license_sha256"),
        ):
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), META[key])

    @unittest.skipUnless(os.environ.get("SGLANG_FPM_BASE_SOURCE"), "Full upstream file not supplied")
    def test_full_file_hash_and_unchanged_timer_decode_code(self):
        source = Path(os.environ["SGLANG_FPM_BASE_SOURCE"]).read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), META["upstream_file_sha256"])
        modified = patched_text(source)
        self.assertEqual(hashlib.sha256(modified.encode()).hexdigest(), META["patched_file_sha256"])
        before, after = ast.parse(source), ast.parse(modified)
        for tree in (before, after):
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "SchedulerMetricsReporter":
                    node.body = [
                        child
                        for child in node.body
                        if not (isinstance(child, ast.FunctionDef) and child.name == "_build_scheduled_request_metrics")
                    ]
        self.assertEqual(ast.dump(before), ast.dump(after))


if __name__ == "__main__":
    unittest.main()
