# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest
from collector.sglang.dsv41_contract import build_manifest
from collector.sglang.dsv41_native_runner import run_workload
from collector.sglang.dsv41_workloads import baseline_tokens, coordinates, coverage_report, freeze_workloads

pytestmark = pytest.mark.unit


def _payload():
    return {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 2, "total_prefill_tokens": 6, "total_kv_read_tokens": 256, "rows": [[3, 128], [3, 128]]}
        ],
        "decode": [{"batch_size": 2, "total_kv_read_tokens": 4096}],
    }


def test_exact_homogeneous_points_freeze_without_rounding_or_extra_decode():
    frozen = freeze_workloads(_payload())
    assert frozen["cases"] == [
        {"case_id": "prefill-0000", "phase": "context", "batch_size": 2, "query": 3, "prefix": 128},
        {
            "case_id": "decode-0000",
            "phase": "generation",
            "batch_size": 2,
            "query": 1,
            "prefix": 2048,
            "canonical_past_kv": 2048,
            "native_inclusive_kv": 2049,
        },
    ]
    assert baseline_tokens(frozen["cases"]) == [2, 6]
    assert frozen["source_payload"] == _payload()


@pytest.mark.parametrize(
    "change",
    [
        {"total_prefill_tokens": 7},
        {"total_kv_read_tokens": 257},
        {"rows": [[2, 128], [4, 128]]},
        {"partition": {"axis": "new", "high_count": 1, "fraction": 0.5}},
    ],
)
def test_heterogeneous_or_unrepresentable_points_fail_instead_of_substituting(change):
    payload = _payload()
    payload["prefill"][0].update(change)
    with pytest.raises(ValueError, match="homogeneous"):
        freeze_workloads(payload)


def test_duplicate_configurations_are_not_silently_deduplicated():
    payload = _payload()
    payload["prefill"].append(copy.deepcopy(payload["prefill"][0]))
    with pytest.raises(ValueError, match="duplicate logical"):
        freeze_workloads(payload)


def _lifecycle(case, *, decode_steps=0):
    events, measured = [], []
    requests = [object() for _ in range(case["batch_size"])]
    batch = object()

    def prepare(size, initial, ids):
        events.append(("prepare", size, initial, len(ids[0])))
        return requests

    def seed_extend(reqs):
        assert reqs is requests
        events.append(("extend",))
        return [7] * len(reqs), [0.0], batch

    def suffix(args, ids, reqs, torch_runner):
        assert reqs is requests
        events.append(("suffix", args.cut_len, len(ids[0])))

    def execute(call, phase, batch_size, query, prefix, real_kv):
        measured.append((phase, batch_size, query, prefix, real_kv))
        return call()

    def decode(ids, given_batch):
        assert given_batch is batch and ids == [7] * len(requests)
        events.append(("decode",))
        return ids, [0.0]

    runner = SimpleNamespace(
        clear=lambda: events.append(("clear",)),
        extend=seed_extend,
        decode=decode,
        cleanup=lambda given_batch: events.append(("cleanup", given_batch is batch)),
        torch_runner=object(),
    )
    bench = SimpleNamespace(
        prepare_synthetic_inputs_for_latency_test=prepare, prepare_extend_inputs_for_correctness_test=suffix
    )
    recorder = SimpleNamespace(active=True, phase="generation")
    run_workload(runner, recorder, bench, list(range(4096)), case, execute, decode_steps=decode_steps)
    return events, measured


def test_cached_prefill_keeps_original_request_and_measures_only_suffix():
    case = freeze_workloads(_payload())["cases"][0]
    events, measured = _lifecycle(case)
    assert events == [
        ("clear",),
        ("prepare", 2, 128, 128),
        ("extend",),
        ("suffix", 128, 131),
        ("extend",),
        ("cleanup", True),
    ]
    assert measured == [("context", 2, 3, 128, True)]


def test_decode_seeds_exact_past_k_and_measures_one_inclusive_generation():
    case = freeze_workloads(_payload())["cases"][1]
    events, measured = _lifecycle(case)
    assert events == [("clear",), ("prepare", 2, 2048, 2048), ("extend",), ("decode",), ("cleanup", True)]
    assert measured == [("generation", 2, 1, 2048, True)]
    assert coordinates("attention", {}, "generation", 2, 1, 2048) == (2, 0, 2049)


@pytest.mark.parametrize("decode_steps", [0, 2])
def test_context_decode_steps_advance_prefix(decode_steps):
    case = freeze_workloads(_payload())["cases"][0]
    events, measured = _lifecycle(case, decode_steps=decode_steps)
    expected = [("context", 2, 3, 128, True)]
    if decode_steps:
        expected += [("generation", 2, 1, 131, True), ("generation", 2, 1, 132, True)]
    assert measured == expected
    assert events.count(("decode",)) == decode_steps
    assert events[-1] == ("cleanup", True)


def test_bounded_coordinate_preserves_long_prefix_and_short_actual_extension():
    geometry = {"bounded_prefill": True, "window_size": 128}
    assert coordinates("attention", geometry, "context", 2, 3, 1536) == (2, 1536, 3)
    assert coordinates("attention", geometry, "context", 2, 192, 128) == (2, 192, 128)
    assert coordinates("mhc", {}, "context", 2, 192, 128) == (1, 0, 384)


@pytest.mark.parametrize("decoder_replay", [False, True])
def test_coverage_report_exposes_bounded_prefix_holes(decoder_replay):
    def workload_payload(queries):
        return {
            "schema_version": 3,
            "prefill": [
                {
                    "batch_size": 2,
                    "total_prefill_tokens": 2 * query,
                    "total_kv_read_tokens": 512,
                    "rows": [[query, 256], [query, 256]],
                }
                for query in queries
            ],
            "decode": [],
        }

    calibration = freeze_workloads(workload_payload([128, 256]))
    heldout = freeze_workloads(workload_payload([192]))
    report = coverage_report(build_manifest(4, decoder_replay), calibration, heldout)
    assert report["calibration_configuration_count"] == 2
    assert report["heldout_configuration_count"] == 1
    if decoder_replay:
        # The 128-token tail starts at prefix 320 for the held-out workload;
        # neither calibration point measures that prefix (256 or 384).
        assert report["heldout_with_missing_curves"] == 1
        assert report["heldout_with_complete_interpolation_domain"] == 0
        missing = report["heldout"][0]["missing_curves"]
        assert missing
        assert all(key[0] == "attention" and key[-2:] == [320, 128] for key in missing)
    else:
        assert report["heldout_with_missing_curves"] == 0
        assert report["heldout_with_complete_interpolation_domain"] == 1


def test_native_benchmark_wall_boundary_synchronizes_before_and_after(monkeypatch):
    import collector.sglang.dsv41_native_runner as native

    events = []
    ticks = iter([1.0, 1.125])
    monkeypatch.setattr(native.time, "perf_counter", lambda: next(ticks))
    runner = SimpleNamespace(synchronize=lambda: events.append("sync"))
    result, elapsed = native.timed_native_forward(runner, lambda: (events.append("native-call"), "logits"))
    assert result == (None, "logits")
    assert elapsed == 125.0
    assert events == ["sync", "native-call", "sync"]


@pytest.mark.parametrize(
    ("forward_only", "missing_flag"),
    [
        (False, None),
        (True, None),
        (True, "--disable-custom-all-reduce"),
        (True, "--enforce-disable-flashinfer-allreduce-fusion"),
        (True, "--disable-shared-experts-fusion"),
    ],
)
@pytest.mark.parametrize("profile", ["full", "decoder_bounded"])
@pytest.mark.parametrize("native_bounded", [False, True])
def test_native_cli_binds_decoder_profile_before_execution(
    tmp_path, monkeypatch, forward_only, missing_flag, profile, native_bounded
):
    import sys

    import collector.sglang.dsv41_native_runner as native

    plan = freeze_workloads(_payload())
    plan_path = tmp_path / "input-plan.json"
    plan_path.write_text(json.dumps(plan))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(4, profile == "decoder_bounded")))
    output = tmp_path / "output"
    flags = [
        "--disable-custom-all-reduce",
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--disable-shared-experts-fusion",
    ]

    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            for flag in flags:
                parser.add_argument(flag, action="store_true")
            parser.add_argument("--enable-decoder-swa-bounded-replay", action="store_true")

        @staticmethod
        def from_cli_args(args):
            return args

    class BenchArgs:
        @staticmethod
        def add_cli_args(parser):
            pass

        @staticmethod
        def from_cli_args(args):
            return SimpleNamespace()

    executed = []

    def execute(server_args, bench_args):
        executed.append(server_args)
        assert bench_args.dsv41_options.forward_only is forward_only
        assert bench_args.dsv41_options.workload_plan == plan
        assert server_args.enable_decoder_swa_bounded_replay is (profile == "decoder_bounded")
        (output / "COMPLETE").write_text("fixture completes transport only")

    bench = SimpleNamespace(ServerArgs=ServerArgs, BenchArgs=BenchArgs, main=execute)
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.benchmark", SimpleNamespace(one_batch=bench))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--manifest",
            str(manifest_path),
            "--runtime-digest",
            "sha256:" + "a" * 64,
            "--prompt-file",
            "unused.txt",
            "--output",
            str(output),
            "--workload-plan",
            str(plan_path),
            *(["--forward-only"] if forward_only else []),
            *(["--enable-decoder-swa-bounded-replay"] if native_bounded else []),
            *(flag for flag in flags if flag != missing_flag),
        ],
    )
    if missing_flag:
        with pytest.raises(ValueError, match="forward-only requires unfused"):
            native.main()
        assert executed == []
        assert list(output.iterdir()) == []
        monkeypatch.setattr(sys, "argv", [*sys.argv, missing_flag])
    if native_bounded != (profile == "decoder_bounded"):
        with pytest.raises(ValueError, match="manifest execution_profile=.*requires enable_decoder_swa_bounded_replay"):
            native.main()
        assert executed == []
        assert not (output / "COMPLETE").exists()
        assert not (output / "workload-plan.json").exists()
        assert not (output / "execution-contract.json").exists()
        assert not list(output.glob("*.jsonl"))
        corrected = [arg for arg in sys.argv if arg != "--enable-decoder-swa-bounded-replay"]
        if profile == "decoder_bounded":
            corrected.append("--enable-decoder-swa-bounded-replay")
        monkeypatch.setattr(sys, "argv", corrected)
    native.main()
    assert len(executed) == 1
    assert json.loads((output / "workload-plan.json").read_text()) == plan
    receipt = json.loads((output / "execution-contract.json").read_text())
    assert receipt["component_recorder"] is not forward_only
    assert receipt["timing_boundary"] == (
        "sglang_one_batch_synchronized_wall_including_prepare_forward_sample"
        if forward_only
        else "cuda_events_local_components"
    )
    assert not list(output.glob("rank-*.jsonl"))
    with pytest.raises(RuntimeError, match="prior raw records"):
        native.main()
