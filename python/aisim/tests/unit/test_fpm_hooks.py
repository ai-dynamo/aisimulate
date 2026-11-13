# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FPM per-request hooks against stand-ins for the SGLang and Dynamo vLLM producers."""

import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.unit

msgspec = pytest.importorskip("msgspec")

from aisim_core import fpm_hooks
from aisim_core.fpm_hooks import _dynamo_vllm as dynamo_vllm
from aisim_core.fpm_hooks import _sglang as sglang
from aisim_core.fpm_hooks import _struct


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
        """Mirrors SGLang 0.5.20: DeviceTimer reporter summed into wall_time, emit drains it, then builds."""

        def __init__(self, spec=(0, 0)):
            self._spec = spec
            self.published = []
            self._init_fpm()

        def _init_fpm(self):
            self.gpu_time_acc = 0.0

            def accumulate(t, **_metadata):
                self.gpu_time_acc += t

            self.forward_pass_device_timer = _Timer()
            self.forward_pass_device_timer.add_reporter(accumulate)

        def _active_spec_config_snapshot(self):
            return {"num_steps": self._spec[0], "num_draft_tokens": self._spec[1]}

        def _emit_forward_pass_metrics(self, batch, result=None):
            self.forward_pass_device_timer._report()
            wall_time, self.gpu_time_acc = self.gpu_time_acc, 0.0
            if wall_time == 0.0:
                return
            self.published.append((wall_time, self._build_scheduled_request_metrics(batch)))

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


class _Timer:
    """DeviceTimer stand-in: intervals reported with their category metadata on ``_report``."""

    def __init__(self):
        self._reporters = []
        self._finished = []

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def finish(self, t, category):
        self._finished.append((t, category))

    def _report(self):
        for t, category in self._finished:
            for reporter in self._reporters:
                reporter(t=t, category=category)
        self._finished.clear()


class _Lens(list):
    """Tensor stand-in for ``GenerationBatchResult.accept_lens``."""

    def __init__(self, values, is_cpu=True):
        super().__init__(values)
        self.is_cpu = is_cpu

    def tolist(self):
        return list(self)


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
    # decode: schedule-time seq_lens_cpu (the aggregate's source); req.seqlen is +1 after the step
    decode_batch = types.SimpleNamespace(
        forward_mode=_Mode("decode"), reqs=[_Req(seqlen=701), _Req(seqlen=91)], seq_lens_cpu=[700, 90]
    )
    d = mixin._build_scheduled_request_metrics(decode_batch)
    assert d.extend_lengths == [1, 1] and d.past_kv_lengths == [700, 90]
    no_lens = types.SimpleNamespace(forward_mode=_Mode("decode"), reqs=[_Req(seqlen=701)])
    assert mixin._build_scheduled_request_metrics(no_lens).past_kv_lengths == [700]
    # fallback when the schedule-time lists are absent
    fb = types.SimpleNamespace(
        forward_mode=_Mode("extend"), reqs=[_Req(extend_input_len=300, prefix=40)], decoding_reqs=None
    )
    f = mixin._build_scheduled_request_metrics(fb)
    assert f.extend_lengths == [300] and f.past_kv_lengths == [40]
    # module attribute now points at the extended struct, as the producer's call-time import expects
    assert _struct.has_request_fields(fpm.ScheduledRequestMetrics)


def test_extend_struct_step_fields_default_and_roundtrip():
    fpm = _stock_fpm_module("x.fpm")
    ext = _struct.extend_struct(fpm.ScheduledRequestMetrics)
    plain = _struct.with_pairs(ext, fpm.ScheduledRequestMetrics(num_decode_requests=1), [(1, 10)])
    assert plain.spec_num_draft_tokens == 0 and plain.spec_accept_lengths == []
    assert plain.device_time_s_by_category == {} and plain.host_emitted_at_s == 0.0
    step = {
        "spec_num_steps": 3,
        "spec_num_draft_tokens": 4,
        "spec_accept_lengths": [4, 1],
        "device_time_s_by_category": {"target_verify": 0.02, "eagle_draft": 0.006},
        "host_scheduled_at_s": 10.0,
        "host_emitted_at_s": 10.03,
        "not_a_step_field": 1,
    }
    out = _struct.with_pairs(ext, fpm.ScheduledRequestMetrics(num_decode_requests=2), [(1, 200), (1, 100)], step)
    raw = msgspec.msgpack.decode(fpm.encode(fpm.ForwardPassMetrics(wall_time=0.026, scheduled_requests=out)))
    scheduled = raw["scheduled_requests"]
    assert scheduled["spec_accept_lengths"] == [4, 1] and scheduled["spec_num_draft_tokens"] == 4
    assert scheduled["device_time_s_by_category"] == {"target_verify": 0.02, "eagle_draft": 0.006}
    assert "not_a_step_field" not in scheduled
    # re-wrapping an extended record replaces, not duplicates, the hook fields
    again = _struct.with_pairs(ext, out, [(1, 5)])
    assert again.past_kv_lengths == [5] and again.spec_accept_lengths == []


def test_sglang_hook_records_speculative_step_diagnostics(fake_sglang):
    _, reporter = fake_sglang
    assert sglang.patch_sglang_metrics_reporter(reporter) is True
    mixin = reporter.SchedulerMetricsReporter(spec=(3, 4))
    timer = mixin.forward_pass_device_timer
    batch = types.SimpleNamespace(
        forward_mode=_Mode("decode"),
        reqs=[_Req(seqlen=701), _Req(seqlen=91)],
        seq_lens_cpu=[700, 90],
        fpm_start_time=100.0,
    )
    for t, category in (
        (0.020, "target_verify"),
        (0.002, "eagle_draft"),
        (0.002, "eagle_draft"),
        (0.003, "eagle_draft_extend"),
    ):
        timer.finish(t, category)
    mixin._emit_forward_pass_metrics(batch, types.SimpleNamespace(accept_lens=_Lens([4, 2])))
    wall_time, m = mixin.published[-1]
    # a speculative decode request's extend is the verify width 1 + k
    assert m.extend_lengths == [4, 4] and m.past_kv_lengths == [700, 90]
    assert (m.spec_num_steps, m.spec_num_draft_tokens, m.spec_accept_lengths) == (3, 4, [4, 2])
    assert m.device_time_s_by_category == pytest.approx(
        {"target_verify": 0.020, "eagle_draft": 0.004, "eagle_draft_extend": 0.003}
    )
    assert sum(m.device_time_s_by_category.values()) == pytest.approx(wall_time)
    assert m.host_scheduled_at_s == 100.0 and m.host_emitted_at_s > 0.0

    # next record: categories were drained; accept_lens still on the device is not read;
    # the per-request fallback adds the bonus token to the correct drafts
    timer.finish(0.010, "target_verify")
    mixin._emit_forward_pass_metrics(batch, types.SimpleNamespace(accept_lens=_Lens([4, 2], is_cpu=False)))
    _, m2 = mixin.published[-1]
    assert m2.device_time_s_by_category == pytest.approx({"target_verify": 0.010}) and m2.spec_accept_lengths == []
    timer.finish(0.010, "target_verify")
    fallback = types.SimpleNamespace(num_correct_drafts_per_req_cpu=[3, 0], num_non_draft_tokens_per_req=1)
    mixin._emit_forward_pass_metrics(batch, fallback)
    assert mixin.published[-1][1].spec_accept_lengths == [4, 1]

    # outside an emission the builder adds lists only and leaves device time for the next record
    timer.finish(0.005, "decode")
    timer._report()
    direct = mixin._build_scheduled_request_metrics(batch)
    assert direct.past_kv_lengths == [700, 90] and direct.device_time_s_by_category == {}
    mixin._emit_forward_pass_metrics(batch)
    assert mixin.published[-1][1].device_time_s_by_category == pytest.approx({"decode": 0.005})

    # prefill under speculation keeps the chunk as the extend
    prefill = types.SimpleNamespace(
        forward_mode=_Mode("extend"), reqs=[_Req()], extend_lens=[8192], prefix_lens=[4096], decoding_reqs=None
    )
    timer.finish(0.050, "extend")
    mixin._emit_forward_pass_metrics(prefill, types.SimpleNamespace(accept_lens=None))
    p = mixin.published[-1][1]
    assert p.extend_lengths == [8192] and p.past_kv_lengths == [4096] and p.spec_accept_lengths == []


def test_sglang_step_diagnostics_failure_keeps_lists(fake_sglang, caplog):
    _, reporter = fake_sglang
    sglang.patch_sglang_metrics_reporter(reporter)
    mixin = reporter.SchedulerMetricsReporter()

    def broken():
        raise RuntimeError("snapshot moved")

    mixin._active_spec_config_snapshot = broken
    batch = types.SimpleNamespace(forward_mode=_Mode("decode"), reqs=[_Req(seqlen=6)], seq_lens_cpu=[5])
    mixin.forward_pass_device_timer.finish(0.001, "decode")
    mixin._emit_forward_pass_metrics(batch, None)
    _, m = mixin.published[-1]
    assert m.past_kv_lengths == [5] and m.spec_num_draft_tokens == 0
    assert "step diagnostics failed" in caplog.text


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
        types.SimpleNamespace(forward_mode=_Mode("decode"), reqs=[_Req(seqlen=6)], seq_lens_cpu=[5])
    )
    assert m.past_kv_lengths == [5]
    # a module without any builder is skipped, not an error
    empty = types.ModuleType("sglang.srt.managers.scheduler_components.metrics_reporter_empty")
    assert sglang.patch_sglang_metrics_reporter(empty) is False


def test_record_frames_keeps_hook_fields_and_reports_gaps():
    import io
    import json

    from aisim_core.fpm_hooks import _record

    fpm = _stock_fpm_module("x.fpm")
    ext = _struct.extend_struct(fpm.ScheduledRequestMetrics)
    scheduled = _struct.with_pairs(
        ext,
        fpm.ScheduledRequestMetrics(num_decode_requests=1),
        [(4, 700)],
        {"spec_accept_lengths": [3], "device_time_s_by_category": {"target_verify": 0.02}},
    )
    payload = fpm.encode(fpm.ForwardPassMetrics(wall_time=0.02, scheduled_requests=scheduled))
    frames = [(n.to_bytes(8, "big"), payload) for n in (5, 6, 9)] + [((10).to_bytes(8, "big"), b"\xc1")]
    out = io.BytesIO()
    summary = _record.record_frames(frames, out, clock=lambda: (1.0, 2.0))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["sequence"] for line in lines] == [5, 6, 9]
    first = lines[0]
    assert first["received_unix_s"] == 1.0 and first["received_monotonic_s"] == 2.0
    # untyped decode: the hook fields survive, unlike a decode with the stock typed struct
    assert first["fpm"]["scheduled_requests"]["extend_lengths"] == [4]
    assert first["fpm"]["scheduled_requests"]["spec_accept_lengths"] == [3]
    assert summary["messages"] == 3 and summary["undecodable"] == 1
    assert summary["gaps"] == [[6, 9]] and (summary["first_sequence"], summary["last_sequence"]) == (5, 10)


def test_record_cli_records_a_live_zmq_stream(tmp_path):
    import json
    import threading
    import time

    zmq = pytest.importorskip("zmq")
    from aisim_core.fpm_hooks import __main__ as cli

    endpoint = f"ipc://{tmp_path / 'fpm'}.0"
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind(endpoint)
    payload = msgspec.msgpack.encode({"wall_time": 0.01, "scheduled_requests": {"extend_lengths": [1]}})

    def publish():
        time.sleep(0.5)  # let the subscriber connect
        for n in range(3):
            pub.send_multipart((b"", n.to_bytes(8, "big"), payload))

    sender = threading.Thread(target=publish)
    sender.start()
    out = tmp_path / "fpm.jsonl"
    assert cli.main(["record", "--endpoint", endpoint, "--output", str(out), "--duration", "1.5"]) == 0
    sender.join()
    pub.close(linger=0)
    context.term()
    assert [json.loads(line)["sequence"] for line in out.read_text().splitlines()] == [0, 1, 2]
    summary = json.loads((tmp_path / "fpm.jsonl.summary.json").read_text())
    assert summary["messages"] == 3 and summary["gaps"] == []
