# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY native-base double: exercise the original purpose guard/envelope.

No scheduler execution, cache state or GPU qualification is represented here.
"""

import ast
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).parents[3] / "collector/fpm_forward/runtime/glm53flash/glm53flash_scheduler.py"
PURPOSES = ("fpm", "ops", "ops_holdout", "ops_graph", "ops_graph_holdout")


@pytest.fixture
def scheduler_class():
    class NativeBase:
        def _bench_init(self, config):
            self._bench_active = True

        def _bench_write_results(self):
            Path(self._bench_config.output_path).write_text(
                json.dumps({"limits": {"max_model_len": 131079}, "results": [], "iteration_groups": []})
            )

    # Compile the actual class intact, isolating only unavailable native imports.
    tree = ast.parse(SOURCE.read_text())
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = dict(
        native=SimpleNamespace(InstrumentedScheduler=NativeBase),
        hashlib=hashlib,
        json=json,
        os=os,
        uuid=uuid,
        Path=Path,
        __file__=str(SOURCE),
        DYNAMO_SHA="54960177085413259859c88bd34ed0734d4c2ea9",
        VLLM_SHA="ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        MAX_BATCH=32,
        WARMUP_REPEATS=5,
        MEASUREMENT_REPEATS=10,
    )
    # Explicit future import keeps unused native annotation names unevaluated.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), native_class],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    cls = namespace["Glm53FlashRealKVScheduler"]
    cls._real_configure_context = lambda self, config: None
    return cls


class AcceptedPurpose(Exception):
    pass


@pytest.mark.parametrize("purpose", PURPOSES)
@pytest.mark.parametrize("manifest", [False, True])
def test_manifest_required_exactly_for_eager_and_graph_calibration(scheduler_class, monkeypatch, purpose, manifest):
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", purpose)
    monkeypatch.setenv("AISIM_GLM53_OPS_MANIFEST", "TEST_ONLY-manifest.json" if manifest else "")

    class Config:
        model_config = SimpleNamespace(enforce_eager=purpose in ("ops", "ops_holdout"))

        @property
        def observability_config(self):
            # The original method reaches this immediately after its manifest guard.
            raise AcceptedPurpose

    expected = AcceptedPurpose if manifest == (purpose in ("ops", "ops_graph")) else ValueError
    with pytest.raises(expected):
        scheduler_class()._bench_init(Config())


@pytest.mark.parametrize("purpose", PURPOSES)
def test_result_envelope_records_actual_instrumentation(scheduler_class, monkeypatch, tmp_path, purpose):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.30.0"))
    instance = scheduler_class()
    instance._bench_config = SimpleNamespace(output_path=str(tmp_path / "benchmark.json"))
    instance._real_purpose = purpose
    instance._real_token_stream_count = 0
    instance._real_token_stream_digest = hashlib.sha256()
    instance._real_input = {}
    instance._real_context_policy = {"runtime_context_length": 131079, "measured_context_limit": 131072}
    instance._real_identity = {"scope": "TEST_ONLY"}
    instance._bench_write_results()
    result = json.loads((tmp_path / "benchmark.json").read_text())
    assert result["ops_instrumented"] is (purpose in ("ops", "ops_graph"))
    assert result["observation_purpose"] == purpose
    assert result["timing_boundary"] == "vllm_native_scheduler_output_interval"
    assert result["producer"]["warmup_repeats"] == 5
    assert result["producer"]["measurement_repeats"] == 10
