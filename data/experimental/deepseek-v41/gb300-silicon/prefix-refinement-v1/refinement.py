# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze independent prefix refinement and admit explicit source precedence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from collector.sglang.collect_dsv41_module import aggregate_rank_records
from collector.sglang.dsv41_contract import build_manifest, write_parquet
from collector.sglang.dsv41_forward_results import forward_admission_report
from collector.sglang.dsv41_workloads import coverage_report, freeze_workloads, projected_keys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
KEY = ("component", "geometry", "batch_size", "prefix", "x")
PROVENANCE = (
    "source_sha256",
    "config_sha256",
    "runtime_digest",
    "used_cuda_graph",
    "execution_profile",
    "measurement_scope",
)
MODULE = Path("data/gb300/dsv41/sglang/0.0.0.dev0/dsv41_module_perf.parquet")
IMAGE = "sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d"
SOURCE = "d50217d8f78e4bd173774c36713650bbf44b058c9575ac8babba208a5c5173a2"
CONFIG = "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(case):
    return tuple(case[k] for k in ("phase", "batch_size", "query", "prefix"))


def context(batch, query, prefix):
    return {
        "batch_size": batch,
        "total_prefill_tokens": batch * query,
        "total_kv_read_tokens": batch * prefix,
        "rows": [[query, prefix] for _ in range(batch)],
    }


def points():
    additions = [context(1, 128, p) for p in (64, 192, 576, 768, 1600, 1792)]
    additions += [context(2, 128, p) for p in (64, 192, 576, 1600)]
    additions += [context(b, 128, p + 2) for b in (1, 2) for p in (0, 128, 512, 1536)]
    heldout = [context(b, q, p) for b in (1, 2) for p in (0, 128, 512, 1536) for q in (31, 80, 126)]
    heldout += [context(1, 257, 0), context(1, 320, 0), context(1, 64, 256), context(1, 96, 256)]
    heldout += [context(b, 130, p) for b in (1, 2) for p in (0, 128, 512, 1536)]
    decode = [{"batch_size": b, "total_kv_read_tokens": b * k} for b in (1, 2) for k in (80, 160, 320, 640, 1920)]
    return (
        {"schema_version": 3, "prefill": additions, "decode": []},
        {"schema_version": 3, "prefill": heldout, "decode": decode},
    )


def old_plan(role):
    return json.loads((BASE / "study/full" / role / "evidence/workload-plan.json").read_bytes())


def pilot_plan():
    return freeze_workloads({"schema_version": 3, "prefill": [context(1, 3, 256), context(1, 128, 256)], "decode": []})


def validate_design(calibration, heldout):
    if len(calibration["cases"]) != 18 or len(heldout["cases"]) != 46:
        raise ValueError("refinement requires exactly 18 new calibration and 46 fresh holdout configurations")
    old = old_plan("calibration")["cases"] + old_plan("heldout")["cases"]
    if set(map(identity, heldout["cases"])) & set(map(identity, old + calibration["cases"] + pilot_plan()["cases"])):
        raise ValueError("new holdout overlaps previously measured or current calibration configurations")
    if set(map(identity, calibration["cases"])) & set(map(identity, old)):
        raise ValueError("new calibration duplicates previous frozen workloads")
    for case in calibration["cases"] + heldout["cases"]:
        if case["batch_size"] not in (1, 2) or case["batch_size"] * case["query"] > 512:
            raise ValueError("workload exceeds the qualified batch/extension domain")
        if case["prefix"] + case["query"] > 2048:
            raise ValueError("workload exceeds the qualified native context domain")


def frozen_inputs():
    result = {}
    for directory in ("full", "decoder_bounded", "study", "report"):
        for path in sorted((BASE / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                result[path.relative_to(BASE).as_posix()] = sha(path)
    return result


def check_preserved(receipt):
    for relative, expected in receipt["preserved_inputs_sha256"].items():
        if sha(BASE / relative) != expected:
            raise ValueError(f"original evidence changed: {relative}")


def freeze():
    cal_points, hold_points = points()
    calibration, heldout = freeze_workloads(cal_points), freeze_workloads(hold_points)
    validate_design(calibration, heldout)
    reports = []
    for bounded in (False, True):
        manifest = build_manifest(4, bounded)
        combined = {
            "cases": old_plan("calibration")["cases"]
            + pilot_plan()["cases"]
            + (calibration["cases"] if bounded else [])
        }
        projection = coverage_report(manifest, combined, heldout)
        if projection["heldout_with_complete_interpolation_domain"] != 46:
            raise ValueError("fresh holdouts need missing-prefix or x-axis extrapolation")
        reports.append(projection)
    documents = {
        "calibration-points.json": cal_points,
        "heldout-points.json": hold_points,
        "calibration-plan.json": calibration,
        "heldout-plan.json": heldout,
        "pilot-reuse-plan.json": pilot_plan(),
        "coverage-projection.json": reports,
        "full-manifest.json": build_manifest(4, False),
        "decoder_bounded-manifest.json": build_manifest(4, True),
    }
    receipt = {
        "schema": "dsv41.prefix.refinement.plan.v1",
        "status": "frozen_unmeasured",
        "new_calibration_profiles": ["decoder_bounded"],
        "new_calibration_configurations": 18,
        "new_holdout_configurations_per_profile": 46,
        "holdout_profiles": ["full", "decoder_bounded"],
        "calibration_warmup": 1,
        "calibration_repetitions": 3,
        "holdout_warmup": 1,
        "holdout_repetitions": 10,
        "new_measured_calibration_invocations": 54,
        "new_measured_forward_holdout_invocations": 920,
        "pilot_reuse_configurations_per_profile": 2,
        "pilot_reuse_components": ["attention"],
        "same_key_precedence": ["formal", "new_calibration", "pilot_selected"],
        "old_holdouts_are_not_new_accuracy_evidence": True,
        "fit_corrections": False,
        "source_sha256": SOURCE,
        "config_sha256": CONFIG,
        "runtime_digest": IMAGE,
        "preserved_inputs_sha256": frozen_inputs(),
        "preparation_source_sha256": sha(__file__),
        "documents_canonical_sha256": {
            name: hashlib.sha256(canonical(value).encode()).hexdigest() for name, value in documents.items()
        },
    }
    outputs = documents | {"plan-receipt.json": receipt}
    if any((ROOT / name).exists() for name in outputs):
        raise ValueError("plan already frozen; create a new version to change it")
    for name, value in outputs.items():
        path = ROOT / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_frozen():
    receipt = json.loads((ROOT / "plan-receipt.json").read_bytes())
    check_preserved(receipt)
    for name, expected in receipt["documents_canonical_sha256"].items():
        value = json.loads((ROOT / name).read_bytes())
        if hashlib.sha256(canonical(value).encode()).hexdigest() != expected:
            raise ValueError("frozen refinement document changed")
    return receipt


def key(row):
    return tuple(row[k] for k in KEY)


def checked_rows(rows, profile):
    keys = set()
    for row in rows:
        if key(row) in keys:
            raise ValueError("duplicate physical key within source table")
        keys.add(key(row))
        expected = (SOURCE, CONFIG, IMAGE, False, profile, "local_compute")
        if tuple(row[k] for k in PROVENANCE) != expected:
            raise ValueError("source/config/runtime/profile/scope differs from qualified module identity")
    return rows


def select_pilot(rows, profile):
    manifest = build_manifest(4, profile == "decoder_bounded")
    selected = set().union(*(projected_keys(manifest, case) for case in pilot_plan()["cases"]))
    selected = {k for k in selected if k[0] == "attention"}
    result = [r for r in checked_rows(rows, profile) if key(r) in selected]
    if {key(r) for r in result} != selected:
        raise ValueError("pilot lacks an explicitly selected attention point")
    return result


def merge_rows(formal, new, pilot, profile):
    merged, origins = {}, {}
    kernels = {(r["component"], r["geometry"]): r["kernel_source"] for r in formal}
    for name, values in (("formal", formal), ("new_calibration", new), ("pilot_selected", pilot)):
        for row in checked_rows(values, profile):
            if kernels.get((row["component"], row["geometry"])) != row["kernel_source"]:
                raise ValueError("component dispatch differs from the original calibration")
            k = key(row)
            if k not in merged:
                merged[k] = row
                origins[k] = name
    return [merged[k] for k in sorted(merged)], {
        name: sum(v == name for v in origins.values()) for name in ("formal", "new_calibration", "pilot_selected")
    }


def unpack(source, output):
    output.mkdir()
    for path in source.iterdir():
        if path.suffix == ".gz":
            (output / path.name.removesuffix(".gz")).write_bytes(gzip.decompress(path.read_bytes()))
        elif path.is_file():
            shutil.copyfile(path, output / path.name)


def pilot_rows(profile):
    """Rebuild pilot from all four raw ranks before selecting its local attention."""
    source = BASE / profile
    with tempfile.TemporaryDirectory(prefix="dsv41-pilot-admission-") as directory:
        raw = Path(directory) / "raw"
        unpack(source / "evidence", raw)
        if not (raw / "COMPLETE").is_file():
            raise ValueError("pilot has no completion receipt")
        sources = json.loads((raw / "source_hashes.json").read_bytes())
        if hashlib.sha256(canonical(sources).encode()).hexdigest() != SOURCE:
            raise ValueError("pilot native source snapshot differs")
        inputs = json.loads((raw / "input_provenance.json").read_bytes())
        original = json.loads((BASE / "study" / profile / "calibration/evidence/input_provenance.json").read_bytes())
        if inputs != original:
            raise ValueError("pilot text/config/runtime provenance differs")
        rows = aggregate_rank_records(sorted(raw.glob("rank-*.jsonl")), 4)
        table = pq.read_table(source / "systems" / MODULE).to_pylist()
        if {key(r): r for r in rows} != {key(r): r for r in table}:
            raise ValueError("pilot raw samples differ from its published table")
    return select_pilot(rows, profile)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def admit_calibration(attempt):
    plan = json.loads((ROOT / "calibration-plan.json").read_bytes())
    if not (attempt / "COMPLETE").is_file() or json.loads((attempt / "workload-plan.json").read_bytes()) != plan:
        raise ValueError("new calibration has not completed its exact frozen plan")
    if list(attempt.glob("baseline-rank-*.jsonl")) or list(attempt.glob("forward-rank-*.jsonl")):
        raise ValueError("refinement calibration may not contain new baselines or forward holdouts")
    contract = json.loads((attempt / "execution-contract.json").read_bytes())
    original_contract = json.loads(
        (BASE / "study/decoder_bounded/calibration/evidence/execution-contract.json").read_bytes()
    )
    normalized = json.loads(canonical(contract))
    args = normalized["native_cli_args"]
    args[args.index("--model-path") + 1] = "deepseek-ai/DeepSeek-V4.1-Flash"
    if normalized != original_contract:
        raise ValueError("new calibration native runtime arguments differ from the qualified original")
    if (contract["mode"], contract["component_recorder"], contract["warmup"], contract["iterations"]) != (
        "local_components",
        True,
        1,
        3,
    ):
        raise ValueError("refinement calibration requires one warmup and three component repetitions")
    argv = contract["native_cli_args"]
    for flag in (
        "--disable-custom-all-reduce",
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--disable-shared-experts-fusion",
        "--enable-decoder-swa-bounded-replay",
    ):
        if flag not in argv:
            raise ValueError("new calibration changed the qualified native serving contract")
    for flag in ("--cuda-graph-backend-decode", "--cuda-graph-backend-prefill"):
        if flag not in argv or argv[argv.index(flag) + 1] != "disabled":
            raise ValueError("new calibration must remain eager")
    sources = json.loads((attempt / "source_hashes.json").read_bytes())
    if hashlib.sha256(canonical(sources).encode()).hexdigest() != SOURCE:
        raise ValueError("native source snapshot changed")
    inputs = json.loads((attempt / "input_provenance.json").read_bytes())
    original = json.loads((BASE / "study/decoder_bounded/calibration/evidence/input_provenance.json").read_bytes())
    if inputs != original:
        raise ValueError("new calibration input text/tokenizer/config provenance differs from frozen study")
    manifest = build_manifest(4, True)
    expected = {(identity(case), sample) for case in plan["cases"] for sample in (1, 2, 3)}
    expected_progress = {(index, sample) for index in range(18) for sample in range(4)}
    cases = {identity(case): case for case in plan["cases"]}
    for family in ("workloads", "invocations"):
        if {p.name for p in attempt.glob(f"{family}-rank-*.jsonl")} != {
            f"{family}-rank-{rank}.jsonl" for rank in range(4)
        }:
            raise ValueError("missing or unexpected new calibration rank files")
    for rank in range(4):
        progress = read_lines(attempt / f"workloads-rank-{rank}.jsonl")
        if len(progress) != 72 or {(r["case_index"], r["sample"]) for r in progress} != expected_progress:
            raise ValueError("missing or duplicate new calibration warmup/progress")
        for row in progress:
            if (
                row["tp_rank"] != rank
                or row["status"] != "passed"
                or row["measured"] != (row["sample"] > 0)
                or any(row[k] != v for k, v in plan["cases"][row["case_index"]].items())
            ):
                raise ValueError("failed or mismatched new calibration workload")
        invocations = read_lines(attempt / f"invocations-rank-{rank}.jsonl")
        if len(invocations) != 54 or {(identity(r), r["sample"]) for r in invocations} != expected:
            raise ValueError("new calibration measured invocation set is incomplete")
        by_id = {r["invocation"]: r for r in invocations}
        if len(by_id) != len(invocations) or any(
            r["tp_rank"] != rank
            or r["finite_logits"] is not True
            or not math.isfinite(r["instrumented_forward_ms"])
            or r["instrumented_forward_ms"] <= 0
            or r["real_kv"] is not True
            or any(r[k] != inputs[k] for k in ("source_sha256", "config_sha256", "runtime_digest", "execution_profile"))
            for r in invocations
        ):
            raise ValueError("invalid new calibration invocation/rank identity")
        grouped = {}
        for row in read_lines(attempt / f"rank-{rank}.jsonl"):
            invocation = by_id.get(row["invocation"])
            if invocation is None or row["sample"] != invocation["sample"] or row["tp_rank"] != rank:
                raise ValueError("component record is not bound to its actual invocation")
            keys = grouped.setdefault(row["invocation"], set())
            if key(row) in keys:
                raise ValueError("duplicate component within native invocation")
            keys.add(key(row))
        if set(grouped) != set(by_id):
            raise ValueError("native invocation omitted all component records")
        for invocation, keys in grouped.items():
            if keys != projected_keys(manifest, cases[identity(by_id[invocation])]):
                raise ValueError("native invocation omitted an expected physical component")
    rows = checked_rows(aggregate_rank_records(sorted(attempt.glob("rank-*.jsonl")), 4), "decoder_bounded")
    return rows


def admit_forward(attempt, profile):
    result = forward_admission_report(attempt)
    plan = json.loads((ROOT / "heldout-plan.json").read_bytes())
    if (
        result["status"] != "accepted"
        or result["case_count"] != 46
        or result["iterations"] != 10
        or result["warmup"] != 1
        or result["execution_profile"] != profile
        or json.loads((attempt / "workload-plan.json").read_bytes()) != plan
    ):
        raise ValueError("fresh forward holdout does not satisfy the frozen 46-by-10 design")
    original = json.loads((BASE / "study" / profile / "precision-v2/evidence/input_provenance.json").read_bytes())
    if json.loads((attempt / "input_provenance.json").read_bytes()) != original:
        raise ValueError("fresh forward input/source/profile differs from the preserved study")
    contract = json.loads((attempt / "execution-contract.json").read_bytes())
    argv = contract["native_cli_args"]
    argv[argv.index("--model-path") + 1] = "deepseek-ai/DeepSeek-V4.1-Flash"
    original_contract = json.loads(
        (BASE / "study" / profile / "precision-v2/evidence/execution-contract.json").read_bytes()
    )
    if contract != original_contract:
        raise ValueError("fresh forward native runtime arguments differ from the qualified precision study")
    return result


def publish_evidence(source, destination):
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        allowed = path.name in {
            "COMPLETE",
            "workload-plan.json",
            "execution-contract.json",
            "source_hashes.json",
            "input_provenance.json",
        } or re.fullmatch(r"(?:rank|(?:workloads|invocations|forward)-rank)-[0-3]\.jsonl", path.name)
        if not path.is_file() or not allowed:
            continue
        data = path.read_bytes()
        if path.name == "execution-contract.json":
            contract = json.loads(data)
            argv = contract["native_cli_args"]
            argv[argv.index("--model-path") + 1] = "deepseek-ai/DeepSeek-V4.1-Flash"
            data = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
        if path.suffix == ".jsonl":
            (destination / (path.name + ".gz")).write_bytes(gzip.compress(data, mtime=0))
        else:
            (destination / path.name).write_bytes(data)


def build(calibration, full_forward, bounded_forward):
    import yaml

    receipt = load_frozen()
    new = admit_calibration(calibration)
    forwards = {
        "full": admit_forward(full_forward, "full"),
        "decoder_bounded": admit_forward(bounded_forward, "decoder_bounded"),
    }
    if any((ROOT / profile).exists() for profile in forwards):
        raise ValueError("refinement output already exists; preserve previous attempt")
    prepared = {}
    for profile in forwards:
        systems = BASE / "study" / profile / "systems"
        formal = pq.read_table(systems / MODULE).to_pylist()
        candidates = new if profile == "decoder_bounded" else []
        kernel_sources = {(r["component"], r["geometry"]): r["kernel_source"] for r in formal}
        if any(kernel_sources[(r["component"], r["geometry"])] != r["kernel_source"] for r in candidates):
            raise ValueError("new native component dispatch differs from the original calibration")
        pilot = pilot_rows(profile)
        rows, origins = merge_rows(formal, candidates, pilot, profile)
        expected_count = 848 if profile == "full" else 948
        if len(rows) != expected_count:
            raise ValueError("admitted refinement physical key count differs from frozen projection")
        prepared[profile] = (rows, origins)
    with tempfile.TemporaryDirectory(prefix=".admission-", dir=ROOT) as directory:
        staging = Path(directory)
        for profile, (rows, origins) in prepared.items():
            destination = staging / profile
            systems = destination / "systems"
            shutil.copytree(BASE / "study" / profile / "systems", systems)
            write_parquet(rows, systems / MODULE)
            meta_path = systems / MODULE.parent / "collection_meta.yaml"
            metadata = yaml.safe_load(meta_path.read_text())
            metadata["tables"]["dsv41_module_perf"] = {
                "status": "complete",
                "rows": len(rows),
                "data_sha256": sha(systems / MODULE),
                "refinement_plan_sha256": sha(ROOT / "plan-receipt.json"),
                "provenance_admission_sha256": sha(__file__),
                "source_precedence": origins,
                "original_formal_same_keys_preserved": True,
                "pilot_scope": "selected local attention only; collective excluded from timed interval",
            }
            meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
            publish_evidence(full_forward if profile == "full" else bounded_forward, destination / "heldout/evidence")
            (destination / "heldout/forward-results.json").write_text(
                json.dumps(forwards[profile], sort_keys=True, indent=2) + "\n"
            )
            if profile == "decoder_bounded":
                publish_evidence(calibration, destination / "calibration/evidence")
            (destination / "admission.json").write_text(
                json.dumps(
                    {
                        "schema": "dsv41.prefix.refinement.admission.v1",
                        "status": "accepted",
                        "profile": profile,
                        "module_points": len(rows),
                        "origin_counts": origins,
                        "baseline_tables_unchanged": True,
                        "independent_holdouts": 46,
                        "heldout_repetitions": 10,
                        "fit_corrections": False,
                        "plan_receipt_sha256": sha(ROOT / "plan-receipt.json"),
                        "admission_source_sha256": sha(__file__),
                        "table_sha256": sha(systems / MODULE),
                        "private_raw_input_files_sha256": {
                            name: {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}
                            for name, path in {
                                "new_calibration": calibration,
                                "heldout": full_forward if profile == "full" else bounded_forward,
                            }.items()
                        },
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
        check_preserved(receipt)
        for profile in prepared:
            (staging / profile).rename(ROOT / profile)
        complete = {
            "schema": "dsv41.prefix.refinement.complete.v1",
            "status": "accepted",
            "plan_receipt_sha256": sha(ROOT / "plan-receipt.json"),
            "admission_files_sha256": {profile: sha(ROOT / profile / "admission.json") for profile in prepared},
        }
        (ROOT / "refinement-complete.json").write_text(json.dumps(complete, sort_keys=True, indent=2) + "\n")

    check_preserved(receipt)


def verify():
    """Rebuild only admitted calibration sources, then query every physical key."""
    import aiconfigurator_core._aiconfigurator_core as native
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    load_frozen()
    complete = json.loads((ROOT / "refinement-complete.json").read_bytes())
    if complete["status"] != "accepted" or complete["plan_receipt_sha256"] != sha(ROOT / "plan-receipt.json"):
        raise ValueError("refinement lacks a complete admission receipt")
    for profile in ("full", "decoder_bounded"):
        output = ROOT / profile
        admission = json.loads((output / "admission.json").read_bytes())
        if complete["admission_files_sha256"][profile] != sha(output / "admission.json"):
            raise ValueError("refinement admission receipt changed")
        original_systems = BASE / "study" / profile / "systems"
        systems = output / "systems"
        if admission["table_sha256"] != sha(systems / MODULE):
            raise ValueError("refinement table changed after admission")
        for path in original_systems.rglob("*"):
            if path.is_file():
                relative = path.relative_to(original_systems)
                if relative not in (MODULE, MODULE.parent / "collection_meta.yaml") and sha(path) != sha(
                    systems / relative
                ):
                    raise ValueError("original baseline or system specification changed")
        with tempfile.TemporaryDirectory(prefix="dsv41-refinement-verify-") as directory:
            scratch = Path(directory)
            heldout = scratch / "heldout"
            unpack(output / "heldout/evidence", heldout)
            forward = admit_forward(heldout, profile)
            if forward != json.loads((output / "heldout/forward-results.json").read_bytes()):
                raise ValueError("published forward result differs from raw independent holdouts")
            additions = []
            if profile == "decoder_bounded":
                calibration = scratch / "calibration"
                unpack(output / "calibration/evidence", calibration)
                additions = admit_calibration(calibration)
            formal = pq.read_table(original_systems / MODULE).to_pylist()
            rows, origins = merge_rows(formal, additions, pilot_rows(profile), profile)
            if origins != admission["origin_counts"] or rows != sorted(
                pq.read_table(systems / MODULE).to_pylist(), key=key
            ):
                raise ValueError("published table differs from explicit calibration source precedence")
        database = PerfDatabase(
            "gb300",
            "sglang",
            "0.0.0.dev0",
            str(systems.resolve()),
            database_mode="SILICON",
            shared_layer=False,
            strict_provenance=True,
        )
        for row in rows:
            variant = (
                "Dsv41"
                + {"attention": "Attention", "mhc": "Mhc", "linear": "Linear", "engram": "Engram"}[row["component"]]
            )
            geometry = json.loads(row["geometry"])
            op = native.op_from_spec_json(json.dumps({variant: geometry | {"name": "refinement_point_check"}}))
            latency = _evaluate_single_op(
                database,
                op,
                is_context=geometry.get("is_context", True),
                batch_size=row["batch_size"],
                s=row["x"],
                prefix=row["prefix"],
                x=row["x"],
            )
            if latency.source != "silicon" or abs(float(latency) - row["latency"]) >= 1e-6:
                raise ValueError("strict native table lookup differs from its admitted observation")
        print(json.dumps({"profile": profile, "strict_module_points": len(rows), "new_independent_forward_cases": 46}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "check-plan", "build", "verify"])
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--full-forward", type=Path)
    parser.add_argument("--bounded-forward", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "verify":
        verify()
    elif args.mode == "build":
        if not all((args.calibration, args.full_forward, args.bounded_forward)):
            parser.error("build requires all three complete raw attempts")
        build(args.calibration, args.full_forward, args.bounded_forward)
    else:
        receipt = load_frozen()
        calibration = json.loads((ROOT / "calibration-plan.json").read_bytes())
        heldout = json.loads((ROOT / "heldout-plan.json").read_bytes())
        validate_design(calibration, heldout)
        for profile in ("full", "decoder_bounded"):
            print(json.dumps({"profile": profile, "selected_pilot_rows": len(pilot_rows(profile))}))
        print(
            json.dumps(
                {
                    "status": "frozen_plan_valid",
                    "new_calibration": receipt["new_calibration_configurations"],
                    "new_holdouts": 2 * receipt["new_holdout_configurations_per_profile"],
                }
            )
        )


if __name__ == "__main__":
    main()
