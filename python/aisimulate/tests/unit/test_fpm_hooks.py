# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FPM per-request hooks against stand-ins for the SGLang and Dynamo vLLM producers."""

import importlib
import sys
import types

import pytest

msgspec = pytest.importorskip("msgspec")

from aisimulate_core import fpm_hooks
from aisimulate_core.fpm_hooks import _struct, dynamo_vllm, sglang


def _stock_fpm_module(name: str) -> types.ModuleType:
    """A module shaped like the producers' forward_pass_metrics: frozen aggregate-only struct + codec."""
    mod = types.ModuleType(name)

    class ScheduledRequestMetrics(msgspec.Struct, frozen=True, gc=False):
        num_prefill_requests: int = 0
        sum_prefill_tokens: int = 0
        var_prefill_length: float = 0.0
        sum_prefill_kv_tokens: int = 0
        num_decode_requests: int = 0
        sum_decode_kv_tokens: int = 0
        var_decode_kv_tokens: float = 0.0

    class ForwardPassMetrics(msgspec.Struct, frozen=True, gc=False):
        version: int = 1
        wall_time: float = 0.0
        scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()

    mod.ScheduledRequestMetrics = ScheduledRequestMetrics
    mod.ForwardPassMetrics = ForwardPassMetrics
    mod.encode = msgspec.msgpack.Encoder().encode
    return mod


@pytest.fixture
def fake_sglang(monkeypatch):
    fpm = _stock_fpm_module("sglang.srt.observability.forward_pass_metrics")
    reporter = types.ModuleType("sglang.srt.managers.scheduler_components.metrics_reporter")

    class SchedulerMetricsReporter:  # real name in SGLang; the hook finds it by method, not by name
        def _build_scheduled_request_metrics(self, batch):
            cls = sys.modules["sglang.srt.observability.forward_pass_metrics"].ScheduledRequestMetrics
            if batch.forward_mode.is_decode():
                return cls(num_decode_requests=len(batch.reqs), sum_decode_kv_tokens=sum(r.seqlen for r in batch.reqs))
            ext = getattr(batch, "extend_lens", None) or [r.extend_input_len for r in batch.reqs]
            pre = getattr(batch, "prefix_lens", None) or [len(r.prefix_indices) for r in batch.reqs]
            return cls(
                num_prefill_requests=len(batch.reqs),
                sum_prefill_tokens=sum(ext),
                sum_prefill_kv_tokens=sum(pre),
            )

    reporter.SchedulerMetricsReporter = SchedulerMetricsReporter
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.observability",
        "sglang.srt.managers",
        "sglang.srt.managers.scheduler_components",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fpm.__name__, fpm)
    monkeypatch.setitem(sys.modules, reporter.__name__, reporter)
    return fpm, reporter


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"

    def is_mixed(self):
        return self.kind == "mixed"


class _Req:
    def __init__(self, seqlen=0, extend_input_len=0, prefix=0):
        self.seqlen = seqlen
        self.extend_input_len = extend_input_len
        self.prefix_indices = list(range(prefix))


def test_extend_struct_roundtrips_through_msgpack():
    fpm = _stock_fpm_module("x.fpm")
    ext = _struct.extend_struct(fpm.ScheduledRequestMetrics)
    assert _struct.has_request_fields(ext) and not _struct.has_request_fields(fpm.ScheduledRequestMetrics)
    base = fpm.ScheduledRequestMetrics(num_decode_requests=2, sum_decode_kv_tokens=300)
    out = _struct.with_pairs(ext, base, [(1, 200), (1, 100)])
    assert out.num_decode_requests == 2 and out.extend_lengths == [1, 1] and out.past_kv_lengths == [200, 100]
    payload = fpm.encode(fpm.ForwardPassMetrics(wall_time=0.01, scheduled_requests=out))
    raw = msgspec.msgpack.decode(payload)
    assert raw["scheduled_requests"]["past_kv_lengths"] == [200, 100]
    # aggregate-only consumers still decode the same bytes
    legacy = msgspec.msgpack.Decoder(fpm.ForwardPassMetrics).decode(payload)
    assert legacy.scheduled_requests.num_decode_requests == 2
    # typed decode into the extended struct validates the new lists like any other field
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.Decoder(ext).decode(msgspec.msgpack.encode({"extend_lengths": [1], "past_kv_lengths": ["x"]}))


def test_sglang_hook_adds_lists_from_schedule_batch(fake_sglang):
    fpm, reporter = fake_sglang
    assert sglang.patch_sglang_metrics_reporter(reporter) is True
    assert sglang.patch_sglang_metrics_reporter(reporter) is False  # idempotent
    mixin = reporter.SchedulerMetricsReporter()
    batch = types.SimpleNamespace(
        forward_mode=_Mode("extend"),
        reqs=[_Req(), _Req()],
        extend_lens=[16384, 512],
        prefix_lens=[32768, 0],
        decoding_reqs=None,
    )
    m = mixin._build_scheduled_request_metrics(batch)
    assert m.extend_lengths == [16384, 512] and m.past_kv_lengths == [32768, 0]
    assert m.sum_prefill_tokens == 16896
    decode_batch = types.SimpleNamespace(forward_mode=_Mode("decode"), reqs=[_Req(seqlen=700), _Req(seqlen=90)])
    d = mixin._build_scheduled_request_metrics(decode_batch)
    assert d.extend_lengths == [1, 1] and d.past_kv_lengths == [700, 90]
    # fallback when the schedule-time lists are absent
    fb = types.SimpleNamespace(
        forward_mode=_Mode("extend"), reqs=[_Req(extend_input_len=300, prefix=40)], decoding_reqs=None
    )
    f = mixin._build_scheduled_request_metrics(fb)
    assert f.extend_lengths == [300] and f.past_kv_lengths == [40]
    # module attribute now points at the extended struct, as the producer's call-time import expects
    assert _struct.has_request_fields(fpm.ScheduledRequestMetrics)


def test_sglang_hook_is_noop_when_native(fake_sglang):
    fpm, reporter = fake_sglang
    fpm.ScheduledRequestMetrics = _struct.extend_struct(fpm.ScheduledRequestMetrics)
    assert sglang.patch_sglang_metrics_reporter(reporter) is False


def test_dynamo_vllm_hook_adds_sorted_lists(monkeypatch):
    fpm = _stock_fpm_module("dynamo.common.forward_pass_metrics")
    sched_mod = types.ModuleType("dynamo.vllm.instrumented_scheduler")
    sched_mod.ScheduledRequestMetrics = fpm.ScheduledRequestMetrics

    class InstrumentedScheduler:
        def _bench_new_request_counts_as_decode(self, req_id):
            return req_id == "bench"

        def _extract_scheduled(self, output):
            return sched_mod.ScheduledRequestMetrics(
                num_prefill_requests=2, sum_prefill_tokens=600, num_decode_requests=2
            )

    sched_mod.InstrumentedScheduler = InstrumentedScheduler
    for name in ("dynamo", "dynamo.common", "dynamo.vllm"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fpm.__name__, fpm)
    monkeypatch.setitem(sys.modules, sched_mod.__name__, sched_mod)
    assert dynamo_vllm.patch_dynamo_vllm_instrumented_scheduler(sched_mod) is True
    output = types.SimpleNamespace(
        scheduled_new_reqs=[
            types.SimpleNamespace(req_id="a", num_computed_tokens=0),
            types.SimpleNamespace(req_id="bench", num_computed_tokens=5000),
        ],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=["c", "d"], num_computed_tokens=[9000, 100]),
        num_scheduled_tokens={"a": 500, "c": 100, "d": 1},
    )
    m = InstrumentedScheduler()._extract_scheduled(output)
    # sorted by past desc, then extend desc; bench decode request gets extend 1
    assert m.past_kv_lengths == [9000, 5000, 100, 0]
    assert m.extend_lengths == [100, 1, 1, 500]
    assert m.num_prefill_requests == 2
    assert _struct.has_request_fields(sched_mod.ScheduledRequestMetrics)


def test_post_import_patcher_runs_patch_after_module_exec(tmp_path, monkeypatch):
    pkg = tmp_path / "fpm_hooks_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    fpm_hooks.install({"fpm_hooks_probe.target": lambda mod: seen.append(mod.VALUE)})
    mod = importlib.import_module("fpm_hooks_probe.target")
    assert mod.VALUE == 1 and seen == [1]
    sys.modules.pop("fpm_hooks_probe.target", None)
    sys.modules.pop("fpm_hooks_probe", None)


def test_hook_path_points_at_sitecustomize():
    import os

    assert os.path.exists(os.path.join(fpm_hooks.hook_path(), "sitecustomize.py"))


def test_sglang_hook_finds_owner_class_by_method_name(fake_sglang):
    fpm, reporter = fake_sglang
    # rename the owner class: the hook must still locate the builder
    reporter.SomeOtherReporter = reporter.SchedulerMetricsReporter
    del reporter.SchedulerMetricsReporter
    assert sglang.patch_sglang_metrics_reporter(reporter) is True
    m = reporter.SomeOtherReporter()._build_scheduled_request_metrics(
        types.SimpleNamespace(forward_mode=_Mode("decode"), reqs=[_Req(seqlen=5)])
    )
    assert m.past_kv_lengths == [5]
    # a module without any builder is skipped, not an error
    empty = types.ModuleType("sglang.srt.managers.scheduler_components.metrics_reporter_empty")
    assert sglang.patch_sglang_metrics_reporter(empty) is False
