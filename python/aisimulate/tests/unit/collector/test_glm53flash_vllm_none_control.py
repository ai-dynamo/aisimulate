# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY public NONE control/export tests; no native or model qualification.

The existing original-shaped fixture supplies actual reader inputs. Only the
common native history/hardware prerequisite is stubbed, as in the serving
export tests. No original campaign data or timing is used here.
"""

import copy
import json
from pathlib import Path

import pytest
from collector import glm53flash_validation as native
from collector import glm53flash_vllm_serving_export as serving
from collector.glm53flash_graph_nodes import trace_forward_identity
from collector.glm53flash_jsonl import file_sha256, iter_records

from .test_glm53flash_ops_evidence import put, put_lines
from .test_glm53flash_vllm_none_export import none_files
from .test_glm53flash_vllm_serving_export import complete_full_files

pytestmark = pytest.mark.unit


def truth(run, root, **kwargs):
    return {
        "evidence_root": str(root.resolve()),
        "runtime_run_id": f"TEST_ONLY-{run['role']}",
        "receipts": [{"path": path.name, "sha256": file_sha256(path)} for path in root.iterdir() if path.is_file()],
    }


def prepare(tmp_path, monkeypatch, observed, *, graph=False):
    create = complete_full_files if graph else none_files
    cal_run, cal = create(tmp_path / "cal", monkeypatch)
    control_run, control = create(tmp_path / "control", monkeypatch, "control")
    for root, values in ((cal, observed), (control, [100.0] * len(observed))):
        for rank in range(2):
            path = root / f"forward-rank-{rank}.jsonl"
            original = list(iter_records(path))
            op_path = root / f"serving-none-measured-ops-rank-{rank}.jsonl"
            original_ops = list(iter_records(op_path)) if op_path.exists() else []
            rows, ops = [], []
            for bid, elapsed in enumerate(values, 1):
                by_invocation = {}
                for raw in original:
                    row = copy.deepcopy(raw)
                    row.update(
                        benchmark_id=bid,
                        invocation=raw["invocation"] + (bid - 1) * 100,
                        whole_forward_gpu_ms=elapsed,
                    )
                    row["forward_id"] = f"rank-{rank}/forward-{row['invocation']}"
                    row["request_ids"] = [rid + f"/point-{bid}" for rid in raw["request_ids"]]
                    if "native_none_profile" in row:
                        profile = copy.deepcopy(row["native_none_profile"])
                        trace = json.loads((root / profile["trace_file"]).read_bytes())
                        trace["aisim_native_forward"] = trace_forward_identity(row)
                        target = root / f"none-profile-rank-{rank}-forward-{row['invocation']}.json"
                        put(target, trace)
                        profile.update(trace_file=target.name, trace_sha256=file_sha256(target))
                        row["native_none_profile"] = profile
                    by_invocation[raw["invocation"]] = row
                    rows.append(row)
                for raw in original_ops:
                    row = copy.deepcopy(raw)
                    forward = by_invocation[raw["invocation"]]
                    for field in ("benchmark_id", "invocation", "forward_id", "request_ids"):
                        row[field] = forward[field]
                    ops.append(row)
            put_lines(path, rows)
            if original_ops:
                put_lines(op_path, ops)
            if graph and root == cal:
                # This fixture is one FULL point; retain its replay evidence.
                graph_path = root / f"graph-forward-rank-{rank}.jsonl"
                graph_rows = list(iter_records(graph_path))
                for row in graph_rows:
                    row["whole_forward_gpu_ms"] = values[0]
                    # Keep graph trace forward identities unchanged.
                    row["request_ids"] = [rid + "/point-1" for rid in row["request_ids"]]
                    trace_path = root / row["replay_nodes"]["trace_file"]
                    trace = json.loads(trace_path.read_bytes())
                    trace["aisim_native_forward"] = trace_forward_identity(row)
                    put(trace_path, trace)
                    row["replay_nodes"]["trace_sha256"] = file_sha256(trace_path)
                put_lines(graph_path, graph_rows)
    for run in (cal_run, control_run):
        run["points"] = [{**run["points"][0], "benchmark_id": bid} for bid in range(1, len(observed) + 1)]
    monkeypatch.setattr(native, "load_native", truth)
    monkeypatch.setattr(native, "_load_native", truth)
    return cal_run, cal, control_run, control


def export(data, output):
    cal_run, cal, control_run, control = data
    return serving.export_serving(cal, cal_run, output, control_root=control, control_run=control_run)


@pytest.mark.parametrize("observed", [95.0, 105.0, 100.0, 94.999, 105.001])
def test_none_public_export_symmetric_exact_boundary_preserves_v1(tmp_path, monkeypatch, observed):
    data = prepare(tmp_path, monkeypatch, [observed])
    cal_run, cal, control_run, control = data
    output = tmp_path / serving.BASENAME
    proof = serving.read_serving_run(cal, cal_run)
    original_report = serving.profile_control(cal, proof, control, control_run)
    originals = {str(path): file_sha256(path) for root in (cal, control) for path in root.iterdir() if path.is_file()}
    if abs(observed - 100) <= 5:
        assert export(data, output)["rows"] == 278
        # The v1 report shape is unchanged: an existing valid NONE receipt
        # needs no additional status field or new sidecar to remain readable.
        assert serving.bind_calibration([output], cal_run, truth(cal_run, cal))["rows"] == 278
    else:
        with pytest.raises(ValueError, match="five-percent.*\\[1\\]"):
            export(data, output)
        assert not output.exists()
        assert not (cal / "serving-rank-selection.json").exists()
        assert not (cal / "serving-calibration-evidence.json").exists()
    assert (cal / "serving-profile-control.json").read_text() == serving.canonical_json(original_report)
    assert {path: file_sha256(Path(path)) for path in originals} == originals


@pytest.mark.parametrize("observed", [[95.0, 100.0, 105.0, 99.0], [94.0, 100.0, 105.0, 106.0]])
def test_all_four_original_points_retained_before_any_failure(tmp_path, monkeypatch, observed):
    data = prepare(tmp_path, monkeypatch, observed)
    output = tmp_path / serving.BASENAME
    if observed[0] == 95:
        export(data, output)
    else:
        with pytest.raises(ValueError, match="five-percent.*\\[1, 4\\]"):
            export(data, output)
        assert not output.exists()
    report = json.loads((data[1] / "serving-profile-control.json").read_bytes())
    assert [row["benchmark_id"] for row in report["results"]] == [1, 2, 3, 4]
    assert [row["profiled_median_ms"] for row in report["results"]] == observed
    assert all(len(row["samples"]) == 10 for row in report["results"])
    assert all(row["control_median_ms"] == 100 for row in report["results"])


@pytest.mark.parametrize("observed", [95.0, 106.0])
def test_preexisting_table_rederives_control_without_new_v1_fields(tmp_path, monkeypatch, observed):
    data = prepare(tmp_path, monkeypatch, [observed])
    output = tmp_path / serving.BASENAME
    gate = serving.require_none_timing_control
    # Reproduce the previous export behavior without changing any test raw.
    monkeypatch.setattr(serving, "require_none_timing_control", lambda proof, control: None)
    export(data, output)
    monkeypatch.setattr(serving, "require_none_timing_control", gate)
    before = file_sha256(output)
    if observed == 95:
        assert serving.bind_calibration([output], data[0], truth(data[0], data[1]))["rows"] == 278
    else:
        with pytest.raises(ValueError, match="five-percent"):
            serving.bind_calibration([output], data[0], truth(data[0], data[1]))
    assert file_sha256(output) == before


@pytest.mark.parametrize("status", [None, False])
def test_missing_or_failed_native_completion_never_reaches_export(tmp_path, monkeypatch, status):
    data = prepare(tmp_path, monkeypatch, [100.0])
    path = data[1] / "forward-rank-0.jsonl"
    rows = list(iter_records(path))
    if status is None:
        rows[-1].pop("gpu_completed")
    else:
        rows[-1]["gpu_completed"] = status
    put_lines(path, rows)
    original = file_sha256(path)
    output = tmp_path / serving.BASENAME
    with pytest.raises(ValueError, match="completion"):
        export(data, output)
    assert file_sha256(path) == original and not output.exists()


def test_graph_public_export_retains_descriptive_large_ratio(tmp_path, monkeypatch):
    data = prepare(tmp_path, monkeypatch, [200.0], graph=True)
    output = tmp_path / serving.BASENAME
    assert export(data, output)["rows"] == 278
    report = json.loads((data[1] / "serving-profile-control.json").read_bytes())
    assert report["timing_equivalence"] == "REPORTED_NOT_ASSUMED"
    assert report["results"][0]["profiled_to_control_ratio"] == 2
    assert serving.bind_calibration([output], data[0], truth(data[0], data[1]))["rows"] == 278


@pytest.mark.parametrize("defect", ["missing_point", "duplicate", "negative", "missing_median", "short_samples"])
def test_none_gate_rejects_incomplete_or_invalid_original_report(defect):
    proof = {"forwards": {(bid, 5): {0: {"phase": "context", "runtime_mode": "NONE"}} for bid in range(1, 5)}}
    control = {
        "results": [
            {"benchmark_id": bid, "profiled_median_ms": 100.0, "control_median_ms": 100.0, "samples": [{}] * 10}
            for bid in range(1, 5)
        ]
    }
    if defect == "missing_point":
        control["results"].pop()
    elif defect == "duplicate":
        control["results"][-1]["benchmark_id"] = 1
    elif defect == "negative":
        control["results"][-1]["control_median_ms"] = -100
    elif defect == "missing_median":
        control["results"][-1].pop("control_median_ms")
    else:
        control["results"][-1]["samples"].pop()
    original = copy.deepcopy(control)
    with pytest.raises(ValueError):
        serving.require_none_timing_control(proof, control)
    assert control == original


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
def test_graph_ratio_has_no_none_threshold(mode):
    proof = {"forwards": {(1, 5): {0: {"phase": "context", "runtime_mode": mode}}}}
    control = {"results": [{"benchmark_id": 1, "profiled_median_ms": 200.0, "control_median_ms": 100.0}]}
    original = copy.deepcopy(control)
    serving.require_none_timing_control(proof, control)
    assert control == original
