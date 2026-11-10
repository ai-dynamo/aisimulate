# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Jan Fostier
# SPDX-License-Identifier: Apache-2.0

"""CPU-only integrity checks and statistics for direct and historical evidence."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = re.compile(r"fp8_gemm_kernel_swapAB<(\d+)u, (\d+)u,")
PROJECTIONS = [(6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def positive(values):
    require(all(math.isfinite(x) and x > 0 for x in values), "Nonpositive or nonfinite timing")


def direct(root):
    protocol = read_json(root / "protocol.json")
    jobs = read_json(root / "jobs.json")
    require(len(jobs) == 2 * protocol["replicates"], "Incomplete independent starts")
    require(all(job["exit_code"] == 0 for job in jobs), "Failed collector worker")
    groups = defaultdict(list)
    environments = []
    keys = set()
    for arm in ["base", "head"]:
        for replicate in range(protocol["replicates"]):
            environment = read_json(root / f"{arm}-{replicate}-environment.json")
            environments.append(environment)
            rows = read_rows(root / f"{arm}-{replicate}-measurements.jsonl")
            require(len(rows) == len(protocol["shapes_m_n_k"]) * protocol["samples_per_shape"], "Missing rows")
            for row in rows:
                shape = tuple(row[k] for k in ["m", "n", "k"])
                key = (arm, replicate, shape, row["sample"])
                require(key not in keys, f"Duplicate measurement: {key}")
                keys.add(key)
                require(list(shape) == protocol["shapes_m_n_k"][row["shape_index"]], "Shape mismatch")
                require(row["arm"] == arm and row["replicate"] == replicate, "Worker mismatch")
                require(row["revision"] == protocol[f"{arm}_revision"], "Revision mismatch")
                require(row["used_cuda_graph"] is True and row["num_runs_executed"] == 6, "Timing mode mismatch")
                require(row["distinct_weight_storages"] == row["copies"], "Aliased weights")
                positive([row["published_latency_ms"], row["loop_latency_ms"]])
                require(
                    math.isclose(row["published_latency_ms"] * row["copies"], row["loop_latency_ms"], rel_tol=1e-12),
                    "Normalization mismatch",
                )
                wanted = (
                    1
                    if arm == "base"
                    else max(1, min(6, math.ceil(2 * environment["device"]["l2_bytes"] / (row["n"] * row["k"]))))
                )
                require(row["copies"] == wanted, "Allocation-count mismatch")
                groups[arm, replicate, shape].append(row)
    for field in ["device", "torch", "torch_cuda", "vllm", "runtime_source_sha256", "protocol_sha256", "runner_sha256"]:
        require(all(e[field] == environments[0][field] for e in environments), f"Runtime changed: {field}")
    require(
        hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() == environments[0]["protocol_sha256"],
        "Protocol changed after measurement",
    )
    with tarfile.open(root / "runner-source.tar.gz") as source:
        original = source.extractfile("run.py").read()
    require(hashlib.sha256(original).hexdigest() == environments[0]["runner_sha256"], "Runner snapshot mismatch")
    # The executable copy adds the repository-required SPDX copyright line;
    # the archived copy preserves the exact bytes used for collection.
    require(
        ast.dump(ast.parse(original)) == ast.dump(ast.parse((HERE / "run.py").read_bytes())),
        "Packaged runner behavior differs from executed snapshot",
    )
    for dependency in ["helper.py", "vllm/utils.py"]:
        require(
            len({e["collector_source_sha256"][dependency] for e in environments}) == 1,
            f"Collector dependency changed: {dependency}",
        )
    outliers = []
    comparisons = []
    for shape_list in protocol["shapes_m_n_k"]:
        shape = tuple(shape_list)
        comparison = {"m": shape[0], "n": shape[1], "k": shape[2]}
        kernels = set()
        for arm in ["base", "head"]:
            medians = []
            for replicate in range(protocol["replicates"]):
                rows = groups[arm, replicate, shape]
                require({r["sample"] for r in rows} == set(range(protocol["samples_per_shape"])), "Sample gap")
                values = [row["published_latency_ms"] for row in rows]
                median = statistics.median(values)
                medians.append(median)
                for row in rows:
                    if abs(row["published_latency_ms"] / median - 1) > protocol["outlier_relative_threshold"]:
                        outliers.append(
                            {
                                "arm": arm,
                                "replicate": replicate,
                                "shape": shape,
                                "sample": row["sample"],
                                "latency_ms": row["published_latency_ms"],
                                "within_start_median_ms": median,
                            }
                        )
                kernels.update(row["kernel_source"] for row in rows)
            comparison[f"{arm}_start_medians_ms"] = medians
            comparison[f"{arm}_median_ms"] = statistics.median(medians)
            comparison[f"{arm}_copies"] = groups[arm, 0, shape][0]["copies"]
        require(len(kernels) == 1, f"Kernel dispatch changed between arms: {shape}: {sorted(kernels)}")
        comparison["kernel_source"] = next(iter(kernels))
        comparison["change_pct"] = (comparison["head_median_ms"] / comparison["base_median_ms"] - 1) * 100
        comparisons.append(comparison)
    return {
        "rows": comparisons,
        "measurement_count": len(keys),
        "missing_or_invalid_measurements": 0,
        "kernel_dispatch_matches": True,
        "flagged_outliers": outliers,
        "excluded_measurements": 0,
        "outlier_rule": protocol["anomaly_checks"],
        "runner_sha256": environments[0]["runner_sha256"],
        "device": environments[0]["device"],
        "unit": "ms per published collector operation",
    }


def pairs(events):
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    groups = defaultdict(list)
    for index, event in enumerate(kernels):
        match = KERNEL.search(event["name"])
        if not match:
            continue
        require(index > 0, "GEMM without preceding quantization kernel")
        quant = kernels[index - 1]
        require("scale_1x128_kernel" in quant["name"] and quant["tid"] == event["tid"], "Unmatched quantization")
        overlap = max(0, min(quant["ts"] + quant["dur"], event["ts"] + event["dur"]) - max(quant["ts"], event["ts"]))
        positive([event["dur"], quant["dur"]])
        groups[tuple(map(int, match.groups()))].append(
            {
                "core_us": event["dur"],
                "pair_us": quant["dur"] + event["dur"] - overlap,
                "identity": [event["name"], event["args"]["grid"], event["args"]["block"]],
            }
        )
    return groups


def historical(root):
    protocol = read_json(root / "protocol.json")
    with tarfile.open(root / "collection-sources.tar.gz") as sources:
        for name, expected_hash in read_json(root / "freeze.json")["hashes"].items():
            require(
                hashlib.sha256(sources.extractfile(name).read()).hexdigest() == expected_hash,
                f"Frozen historical source changed: {name}",
            )
    model = {}
    operations = {}
    for replicate in range(3):
        records = read_rows(root / f"forward-{replicate}.jsonl")
        require(len(records) == 2, "Missing model cases")
        for record in records:
            require(
                record["finite_logits"] and record["capture_max_abs_logit_difference"] == 0,
                "Model graph output differs from runtime reference",
            )
            require(
                record["actual_requests"] == record["batch"] and record["input_tokens"] == record["batch"],
                "Model request geometry mismatch",
            )
            require(all(past == record["context"] for past in record["past_kv_lengths"]), "KV geometry mismatch")
        records = read_rows(root / f"ops-{replicate}.jsonl")
        require(len(records) == 67, "Missing sweep cases")
        for record in records:
            key = (replicate, record["m"], record["n"], record["k"], record["copies"])
            require(key not in operations, "Duplicate sweep case")
            require(record["graph_ops"] == 72 and record["profiled_replays"] == 5, "Unexpected sweep graph")
            positive(record["per_op_samples_ms"])
            operations[key] = record
    expected = {
        (rep, m, n, k, copies)
        for rep in range(3)
        for m in [1, 16]
        for n, k in PROJECTIONS
        for copies in protocol["copies"]
    }
    expected.update((rep, 16, 12288, 8192, copies) for rep in range(3) for copies in [1, 2, 4])
    require(set(operations) == expected, "Historical shape/copy coverage differs from the frozen protocol")
    extracted = {}
    with tarfile.open(root / "cuda-traces.tar.gz") as archive:
        members = {m.name: m for m in archive.getmembers()}
        expected_members = {r["trace_file"] for r in operations.values()}
        expected_members.update(
            f"graph-r{rep}-b{m}-c{128 if m == 1 else 512}.json.gz" for rep in range(3) for m in [1, 16]
        )
        require(set(members) == expected_members, "Missing or unexpected raw traces")
        for name, member in members.items():
            handle = archive.extractfile(member)
            require(handle is not None, f"Trace is not a regular file: {name}")
            payload = gzip.decompress(handle.read())
            extracted[name] = pairs(json.loads(payload)["traceEvents"])
            for record in operations.values():
                if record["trace_file"] == name:
                    require(hashlib.sha256(payload).hexdigest() == record["trace_json_sha256"], "Trace hash mismatch")
    for replicate in range(3):
        for m in [1, 16]:
            name = f"graph-r{replicate}-b{m}-c{128 if m == 1 else 512}.json.gz"
            groups = extracted[name]
            require(set(groups) == set(PROJECTIONS), "Unexpected model projections")
            require(all(len(values) == 180 for values in groups.values()), "Incomplete model kernel observations")
            model[replicate, m] = groups
    for (replicate, m, n, k, copies), record in operations.items():
        groups = extracted[record["trace_file"]]
        require(set(groups) == {(n, k)} and len(groups[n, k]) == 360, "Incomplete isolated kernel observations")
        if (n, k) in PROJECTIONS:
            identities = {json.dumps(p["identity"]) for p in groups[n, k]}
            reference = {json.dumps(p["identity"]) for p in model[replicate, m][n, k]}
            require(identities == reference, "Kernel name, grid or block mismatch")
    rows = []
    l2 = {record["l2_bytes"] for record in operations.values()}
    require(l2 == {60 << 20}, "Unexpected historical L2 capacity")
    for m in [1, 16]:
        for n, k in PROJECTIONS:
            copies = min(6, max(1, math.ceil(2 * (60 << 20) / (n * k))))
            row = {"m": m, "n": n, "k": k, "copies": copies}
            for metric in ["core_us", "pair_us"]:
                reference = [statistics.median(p[metric] for p in model[rep, m][n, k]) for rep in range(3)]
                row[f"model_{metric}"] = statistics.median(reference)
                row[f"model_start_medians_{metric}"] = reference
                for arm, count in [("base", 1), ("head", copies)]:
                    medians = [
                        statistics.median(
                            p[metric] for p in extracted[operations[rep, m, n, k, count]["trace_file"]][n, k]
                        )
                        for rep in range(3)
                    ]
                    row[f"{arm}_{metric}"] = statistics.median(medians)
                    row[f"{arm}_start_medians_{metric}"] = medians
                    row[f"{arm}_error_pct_{metric}"] = abs(row[f"{arm}_{metric}"] / row[f"model_{metric}"] - 1) * 100
            rows.append(row)
    summary = {}
    for metric in ["core_us", "pair_us"]:
        summary[metric] = {
            f"{arm}_mape_pct": statistics.mean(r[f"{arm}_error_pct_{metric}"] for r in rows) for arm in ["base", "head"]
        }
        summary[metric]["improved_shapes"] = sum(
            r[f"head_error_pct_{metric}"] < r[f"base_error_pct_{metric}"] for r in rows
        )
    negative = read_json(root / "full-forward-comparison.json")
    decode = [p for p in negative["points"] if p["phase"] == "decode"]
    require(len(decode) == 12, "Incomplete historical full-forward comparison")
    full_forward = {}
    for arm in ["baseline", "rotate_2l2"]:
        positive([p[f"{arm}_ms"] for p in decode] + [p["measured_ms"] for p in decode])
        value = statistics.mean(abs(p[f"{arm}_ms"] / p["measured_ms"] - 1) * 100 for p in decode)
        require(math.isclose(value, negative["summary"]["decode"][arm]["mape_pct"]), "Full-forward MAPE mismatch")
        full_forward[f"{arm}_mape_pct"] = value
    return {
        "rows": rows,
        "summary": summary,
        "trace_count": len(extracted),
        "sweep_case_count": len(operations),
        "kernel_name_grid_block_matches": True,
        "model_output_parity": "exact for all six model cases",
        "missing_or_invalid_measurements": 0,
        "excluded_measurements": 0,
        "earlier_full_forward": {
            **full_forward,
            "input_status": "Recorded comparison points from the earlier uncapped-policy investigation; not a new run.",
        },
        "aggregation": "Median within start, median across three starts; MAPE weights eight shapes equally.",
        "scope": "Earlier fixed-72-operation graph sweep; reanalysis of this PR's copy counts."
        " Not fresh production-row accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, default=HERE / "direct")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    result = {"schema_version": 1, "direct": direct(args.direct), "historical": historical(HERE / "historical")}
    args.output.mkdir(exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ["direct", "historical"]:
        rows = result[name]["rows"]
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "direct_measurements": result["direct"]["measurement_count"],
                "direct_flagged_outliers": len(result["direct"]["flagged_outliers"]),
                "direct_rows": result["direct"]["rows"],
                "historical": result["historical"]["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
