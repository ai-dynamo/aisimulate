# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent design, provenance, repetition, and source-precedence gates."""

import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location("prefix_refinement", Path(__file__).with_name("refinement.py"))
refinement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refinement)


def dump(path, value):
    path.write_text(json.dumps(value) + "\n")


def lines(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(refinement, "ROOT", tmp_path)
    refinement.freeze()
    return tmp_path


def test_frozen_design_disjoint_and_all_new_holds_inside_curves(frozen):
    receipt = refinement.load_frozen()
    assert receipt["new_calibration_configurations"] == 18
    assert receipt["new_measured_forward_holdout_invocations"] == 920
    reports = json.loads((frozen / "coverage-projection.json").read_bytes())
    assert [r["projected_calibration_module_points"] for r in reports] == [848, 948]
    assert [r["heldout_with_complete_interpolation_domain"] for r in reports] == [46, 46]
    plan = json.loads((frozen / "heldout-plan.json").read_bytes())
    decode = [r for r in plan["cases"] if r["phase"] == "generation"]
    assert len(decode) == 10
    assert all(r["native_inclusive_kv"] == r["canonical_past_kv"] + 1 for r in decode)
    with pytest.raises(ValueError, match="already frozen"):
        refinement.freeze()


def test_frozen_plan_edit_rejected(frozen):
    path = frozen / "heldout-plan.json"
    plan = json.loads(path.read_bytes())
    plan["cases"][0]["query"] += 1
    dump(path, plan)
    with pytest.raises(ValueError, match="document changed"):
        refinement.load_frozen()


def test_old_or_new_calibration_cannot_be_relabelled_new_holdout():
    calibration, heldout = map(refinement.freeze_workloads, refinement.points())
    heldout["cases"][0] = calibration["cases"][0]
    with pytest.raises(ValueError, match="overlaps"):
        refinement.validate_design(calibration, heldout)


@pytest.mark.parametrize("profile,expected_count", [("full", 848), ("decoder_bounded", 840)])
def test_pilot_rebuild_and_formal_same_key_precedence(profile, expected_count):
    formal = pq.read_table(refinement.BASE / "study" / profile / "systems" / refinement.MODULE).to_pylist()
    pilot = refinement.pilot_rows(profile)
    rows, origins = refinement.merge_rows(formal, [], pilot, profile)
    assert len(rows) == expected_count
    assert origins["formal"] == len(formal)
    actual = {refinement.key(r): r for r in rows}
    assert all(actual[refinement.key(r)] == r for r in formal)
    assert {r["component"] for r in pilot} == {"attention"}


def test_duplicate_or_changed_dispatch_or_source_is_rejected():
    formal = pq.read_table(refinement.BASE / "study/full/systems" / refinement.MODULE).to_pylist()
    with pytest.raises(ValueError, match="duplicate"):
        refinement.merge_rows(formal, [formal[0], formal[0]], [], "full")
    with pytest.raises(ValueError, match="dispatch"):
        refinement.merge_rows(formal, [formal[0] | {"kernel_source": "other"}], [], "full")
    with pytest.raises(ValueError, match="source/config"):
        refinement.merge_rows(formal, [formal[0] | {"runtime_digest": "different"}], [], "full")


@pytest.fixture
def synthetic_calibration(frozen):
    raw = frozen / "raw"
    raw.mkdir()
    original = refinement.BASE / "study/decoder_bounded/calibration/evidence"
    for name in ("execution-contract.json", "source_hashes.json", "input_provenance.json"):
        (raw / name).write_bytes((original / name).read_bytes())
    (raw / "COMPLETE").write_text("complete\n")
    plan = json.loads((frozen / "calibration-plan.json").read_bytes())
    dump(raw / "workload-plan.json", plan)
    formal = pq.read_table(refinement.BASE / "study/decoder_bounded/systems" / refinement.MODULE).to_pylist()
    templates = {(r["component"], r["geometry"]): r for r in formal}
    inputs = json.loads((raw / "input_provenance.json").read_bytes())
    manifest = refinement.build_manifest(4, True)
    for rank in range(4):
        progress, invocations, components = [], [], []
        for index, case in enumerate(plan["cases"]):
            for sample in range(4):
                progress.append(
                    case
                    | {
                        "case_index": index,
                        "sample": sample,
                        "tp_rank": rank,
                        "measured": sample > 0,
                        "status": "passed",
                    }
                )
                if sample == 0:
                    continue
                invocation = index * 4 + sample
                invocations.append(
                    case
                    | inputs
                    | {
                        "sample": sample,
                        "invocation": invocation,
                        "tp_rank": rank,
                        "real_kv": True,
                        "finite_logits": True,
                        "instrumented_forward_ms": 10.0,
                    }
                )
                for key in refinement.projected_keys(manifest, case):
                    row = dict(templates[key[:2]])
                    row.update(zip(refinement.KEY, key, strict=True))
                    row.update(sample=sample, invocation=invocation, tp_rank=rank, latency=1.0 + rank / 100)
                    components.append(row)
        lines(raw / f"workloads-rank-{rank}.jsonl", progress)
        lines(raw / f"invocations-rank-{rank}.jsonl", invocations)
        lines(raw / f"rank-{rank}.jsonl", components)
    return raw


def test_complete_exact_raw_calibration_is_admitted(synthetic_calibration):
    rows = refinement.admit_calibration(synthetic_calibration)
    assert rows and all(r["latency"] == 1.03 for r in rows)


@pytest.mark.parametrize("family", ["workloads", "invocations", "component"])
def test_missing_or_duplicate_raw_observations_rejected(synthetic_calibration, family):
    filename = "rank-0.jsonl" if family == "component" else f"{family}-rank-0.jsonl"
    path = synthetic_calibration / filename
    rows = refinement.read_lines(path)
    lines(path, rows[:-1] if family == "component" else rows + [rows[0]])
    with pytest.raises(ValueError, match="omitted|incomplete|duplicate"):
        refinement.admit_calibration(synthetic_calibration)


def test_unexpected_rank_or_runtime_drift_rejected(synthetic_calibration):
    extra = synthetic_calibration / "workloads-rank-4.jsonl"
    extra.write_text("")
    with pytest.raises(ValueError, match="rank files"):
        refinement.admit_calibration(synthetic_calibration)
    extra.unlink()
    path = synthetic_calibration / "execution-contract.json"
    contract = json.loads(path.read_bytes())
    contract["native_cli_args"].extend(["--extra-setting", "1"])
    dump(path, contract)
    with pytest.raises(ValueError, match="runtime arguments"):
        refinement.admit_calibration(synthetic_calibration)


def test_public_evidence_allowlist_and_model_path_scrubbed(synthetic_calibration, tmp_path):
    source = synthetic_calibration
    (source / "private-note.json").write_text('{"path":"private"}')
    path = source / "execution-contract.json"
    contract = json.loads(path.read_bytes())
    contract["native_cli_args"][1] = "/private/model"
    dump(path, contract)
    output = tmp_path / "published"
    refinement.publish_evidence(source, output)
    assert not (output / "private-note.json").exists()
    assert json.loads((output / path.name).read_bytes())["native_cli_args"][1] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert (output / "rank-0.jsonl.gz").is_file()


def synthetic_forward(frozen, profile):
    import gzip

    raw = frozen / f"raw-forward-{profile}"
    raw.mkdir()
    original = refinement.BASE / "study" / profile / "precision-v2/evidence"
    for name in ("execution-contract.json", "source_hashes.json", "input_provenance.json"):
        (raw / name).write_bytes((original / name).read_bytes())
    (raw / "COMPLETE").write_text("complete\n")
    plan = json.loads((frozen / "heldout-plan.json").read_bytes())
    dump(raw / "workload-plan.json", plan)
    template_rows = [
        json.loads(line)
        for line in gzip.decompress((original / "forward-rank-0.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    templates = {r["phase"]: r for r in template_rows}
    for rank in range(4):
        progress, observations = [], []
        for index, case in enumerate(plan["cases"]):
            for sample in range(11):
                progress.append(
                    case
                    | {
                        "case_index": index,
                        "sample": sample,
                        "tp_rank": rank,
                        "measured": sample > 0,
                        "status": "passed",
                    }
                )
                if sample == 0:
                    continue
                observations.append(
                    templates[case["phase"]]
                    | case
                    | {
                        "sample": sample,
                        "invocation": index * 11 + sample,
                        "tp_rank": rank,
                        "real_kv": case["phase"] == "generation" or case["prefix"] > 0,
                        "native_benchmark_forward_ms": 10.0 + rank,
                    }
                )
        lines(raw / f"workloads-rank-{rank}.jsonl", progress)
        lines(raw / f"forward-rank-{rank}.jsonl", observations)
    return raw


def test_build_only_after_three_complete_attempts_preserves_sources(synthetic_calibration, frozen):
    full = synthetic_forward(frozen, "full")
    bounded = synthetic_forward(frozen, "decoder_bounded")
    (bounded / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="46-by-10"):
        refinement.build(synthetic_calibration, full, bounded)
    assert not (frozen / "full").exists()
    (bounded / "COMPLETE").write_text("complete\n")
    refinement.build(synthetic_calibration, full, bounded)
    assert (frozen / "refinement-complete.json").exists()
    for profile, points in (("full", 848), ("decoder_bounded", 948)):
        receipt = json.loads((frozen / profile / "admission.json").read_bytes())
        assert receipt["module_points"] == points
        assert receipt["independent_holdouts"] == 46
        old = refinement.BASE / "study" / profile / "systems"
        for path in old.rglob("*.parquet"):
            if path.relative_to(old) != refinement.MODULE:
                assert path.read_bytes() == (frozen / profile / "systems" / path.relative_to(old)).read_bytes()
    refinement.check_preserved(refinement.load_frozen())
    with pytest.raises(ValueError, match="already exists"):
        refinement.build(synthetic_calibration, full, bounded)
