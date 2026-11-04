#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report bounded collection estimates; unavailable evidence remains explicitly unknown."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from tools.perf_database.collection_cost import case_key, load_tables

# These are workload axes, never geometry, precision, checkpoint or dispatch.
AXES = {
    "batch_size",
    "batch_sizes",
    "num_tokens",
    "seq_lens",
    "seq_len",
    "input_len",
    "prefix_len",
    "sequence_length",
    "kv_cache_length",
}


def clean_case(case):
    return {k: str(v) for k, v in case.items() if not k.startswith("_")}


def identity(sample):
    return sample["system"], sample["backend"], sample["operation"]


def geometry(case):
    return tuple(sorted((k, v) for k, v in clean_case(case).items() if k not in AXES))


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def accepted(samples):
    """Failures/timeouts are evidence of gaps, never cheap successful observations."""
    return [s for s in samples if s["status"] == "passed" and positive(s.get("collection_seconds"))]


def measured_cost(samples):
    """Inclusive point times plus explicitly separate setup once per group.

    Group/direct-call clocks already include setup and forwards. Never add the
    latency or nested phase clocks to an inclusive collection clock.
    """
    groups = defaultdict(list)
    for sample in accepted(samples):
        groups[(*identity(sample), sample["group_id"])].append(sample)
    total = 0.0
    for group in groups.values():
        inclusive = [s for s in group if s["measurement_scope"] in {"group", "direct_call"}]
        if inclusive:
            if len(inclusive) != 1:
                raise ValueError("ambiguous duplicate inclusive group clocks")
            total += inclusive[0]["collection_seconds"]
        else:
            setup = {s["setup_seconds"] for s in group if s.get("setup_seconds") is not None}
            if len(setup) > 1 or any(not positive(v) for v in setup):
                raise ValueError("inconsistent group setup clocks")
            total += sum(s["collection_seconds"] for s in group) + (next(iter(setup)) if setup else 0)
    return total


def predict(case, samples, component=None):
    """Exact observations or multilinear interpolation inside measured corners.

    Every non-workload field must match. No extrapolation or proxy-to-time fit.
    Returns an empirical repeated-observation range, not a confidence interval.
    """
    case = clean_case(case)
    matched = [s for s in accepted(samples) if geometry(s["case"]) == geometry(case)]
    if len({s["source_commit"] for s in matched}) > 1 or len({identity(s) for s in matched}) > 1:
        return None
    full_causal = bool(matched) and matched[0]["operation"] == "attention_context" and case.get("window_size") == "0"
    if (
        full_causal
        and component is None
        and all(
            positive(s.get("preparation_seconds"))
            and positive(s.get("execution_seconds"))
            and math.isclose(
                s["preparation_seconds"] + s["execution_seconds"], s["collection_seconds"], rel_tol=1e-5, abs_tol=1e-6
            )
            for s in matched
        )
    ):
        phases = [
            predict(case, [{**s, "collection_seconds": s[field]} for s in matched], component=field)
            for field in ("preparation_seconds", "execution_seconds")
        ]
        return (
            tuple(sum(values) for values in zip(*phases, strict=True)) if all(p is not None for p in phases) else None
        )
    exact = [s["collection_seconds"] for s in matched if clean_case(s["case"]) == case]
    if exact:
        return min(exact), max(exact)
    axes = sorted(set(case) & AXES)
    if not axes or any(not str(case[a]).isdigit() for a in axes):
        return None
    matched = [s for s in matched if set(clean_case(s["case"])) == set(case)]
    corners = []
    for axis in axes:
        values = sorted({int(s["case"][axis]) for s in matched if str(s["case"][axis]).isdigit()})
        target = int(case[axis])
        if not values or target < values[0] or target > values[-1]:
            return None
        index = bisect.bisect_left(values, target)
        corners.append([target] if values[index] == target else [values[index - 1], values[index]])
    low = high = 0.0
    for point in itertools.product(*corners):
        observations = [
            s["collection_seconds"]
            for s in matched
            if all(int(s["case"][a]) == v for a, v in zip(axes, point, strict=True))
        ]
        if not observations:
            return None
        weight = 1.0
        for axis, value, bounds in zip(axes, point, corners, strict=True):
            if len(bounds) == 2:
                power = 2 if full_causal and axis == "input_len" and component != "preparation_seconds" else 1
                lower, upper, target = bounds[0] ** power, bounds[1] ** power, int(case[axis]) ** power
                weight *= (
                    (upper - target) / (upper - lower) if value == bounds[0] else (target - lower) / (upper - lower)
                )
        low += weight * min(observations)
        high += weight * max(observations)
    return low, high


def validation(samples):
    points = [s for s in samples if s["measurement_scope"] == "point"]
    fit = [s for s in points if s.get("role") != "holdout"]
    holdouts = [s for s in points if s.get("role") == "holdout"]
    errors = []
    for sample in holdouts:
        estimate = predict(sample["case"], fit)
        if sample["status"] != "passed" or estimate is None or not positive(sample.get("collection_seconds")):
            return {"passed": False, "reason": "uncovered_or_failed_holdout", "max_relative_error": None}
        actual = sample["collection_seconds"]
        error = abs(statistics.mean(estimate) - actual) / max(actual, 1e-12)
        errors.append(error)
    return {
        "passed": bool(errors) and max(errors) <= 0.20,
        "reason": "heldout_20_percent" if errors else "no_holdout",
        "max_relative_error": max(errors) if errors else None,
    }


def phase_accounting_check(samples):
    """Validate reconstructed point+setup totals against independent group clocks."""
    errors = []
    for group in accepted(samples):
        if group["measurement_scope"] != "group" or group.get("role") != "holdout":
            continue
        points = [
            s
            for s in accepted(samples)
            if s["measurement_scope"] == "point"
            and s["group_id"] == group["group_id"]
            and identity(s) == identity(group)
        ]
        if not points or not any(s.get("setup_seconds") is not None for s in points):
            return {"passed": False, "max_relative_error": None, "reason": "missing_group_members_or_setup"}
        predicted = measured_cost(points)
        errors.append(abs(predicted - group["collection_seconds"]) / max(group["collection_seconds"], 1e-12))
    return {
        "passed": bool(errors) and max(errors) <= 0.15,
        "max_relative_error": max(errors) if errors else None,
        "reason": "group_15_percent" if errors else "no_group_holdout",
    }


def member_keys(sample):
    return frozenset(case_key(clean_case(case)) for case in sample.get("group_member_cases", []))


def controlled_protocol(sample):
    """Unproven cold/cache state cannot qualify a marginal setup estimate."""
    scope = sample.get("cache_scope_declared", "")
    if (
        sample.get("pilot_splits_same_cache_proven") is not True
        or not scope
        or not (sample.get("predecessor_cache_warmed") is True or "already-warmed" in scope)
    ):
        return None
    ranks = sample.get("participating_gpus")
    if not isinstance(ranks, int) or ranks <= 0:
        return None
    observed = sample.get("direct_benchmark_measurements", [])
    repetitions = sorted(
        {(m.get("num_runs_executed"), m.get("requested_warmups"), m.get("repeat_n")) for m in observed}, key=str
    )
    return (
        identity(sample),
        sample.get("_qualified_runtime_source", sample["source_commit"]),
        ranks,
        sample["cache_scope_declared"],
        json.dumps(sample.get("runtime_protocol"), sort_keys=True),
        json.dumps(repetitions),
    )


def independent_group_check(groups):
    """Two independent training repeats predict a third native group clock.

    Same-group point clocks plus their residual setup are only an accounting
    identity. They cannot establish forecast accuracy.
    """
    if len(groups) < 3 or len({s["group_id"] for s in groups}) != len(groups):
        return {"passed": False, "reason": "fewer_than_three_controlled_repeats", "max_relative_error": None}
    ordered = sorted(groups, key=lambda sample: (sample.get("paired_group_id", ""), sample["sample_id"]))
    train, holdout = ordered[:-1], ordered[-1]
    prediction = statistics.mean(s["collection_seconds"] for s in train)
    error = abs(prediction - holdout["collection_seconds"]) / max(holdout["collection_seconds"], 1e-12)
    return {
        "passed": error <= 0.15,
        "reason": "independent_repeat_15_percent",
        "max_relative_error": error,
        "training_sample_ids": [s["sample_id"] for s in train],
        "holdout_sample_id": holdout["sample_id"],
    }


def group_validation(samples):
    pools = defaultdict(list)
    for sample in accepted(samples):
        protocol = controlled_protocol(sample)
        if sample["measurement_scope"] == "group" and protocol is not None and member_keys(sample):
            pools[(protocol, member_keys(sample))].append(sample)
    checks = [independent_group_check(group) for group in pools.values()]
    errors = [check["max_relative_error"] for check in checks if check["max_relative_error"] is not None]
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "reason": "independent_repeat_15_percent" if checks else "no_controlled_group_repeats",
        "max_relative_error": max(errors) if errors else None,
        "checks": checks,
    }


def paired_groups(samples, scenario, added, removed, baseline_scenario="baseline"):
    """Fold controlled physical-manifest repeats once; never multiply coverage.

    Source eligibility is attached by report_lane before this function runs.
    A pair ID alone proves neither runtime equivalence nor cache/rank equality.
    """
    groups = defaultdict(list)
    for sample in accepted(samples):
        if sample["measurement_scope"] == "group" and sample.get("paired_group_id"):
            groups[sample["paired_group_id"]].append(sample)
    pools = defaultdict(list)
    excluded = []
    for pair_id, group in groups.items():
        before = [s for s in group if s.get("scenario") == baseline_scenario]
        after = [s for s in group if s.get("scenario") == scenario]
        if len(before) > 1 or len(after) > 1:
            excluded.append({"pair_id": pair_id, "reason": "ambiguous_pair_id_collision"})
            continue
        if not before or not after:
            continue
        before, after = before[0], after[0]
        protocol = controlled_protocol(before)
        if protocol is None or controlled_protocol(after) != protocol:
            excluded.append({"pair_id": pair_id, "reason": "uncontrolled_or_incompatible_source_protocol_ranks"})
            continue
        old, new = member_keys(before), member_keys(after)
        if not old or not new:
            continue
        old_geometry = {geometry(c) for c in before["group_member_cases"]}
        new_geometry = {geometry(c) for c in after["group_member_cases"]}
        if old_geometry != new_geometry:
            excluded.append({"pair_id": pair_id, "reason": "incompatible_physical_geometry"})
            continue
        additions, removals = new - old, old - new
        if not additions and not removals:
            continue
        if not additions <= added or not removals <= removed:
            continue
        pools[(protocol, old, new)].append((before, after))
    additions, removals, low, high = set(), set(), 0.0, 0.0
    details = []
    for (protocol, old, new), repeats in pools.items():
        new_keys, old_keys = new - old, old - new
        if additions & new_keys or removals & old_keys:
            excluded.append({"reason": "overlapping_incompatible_physical_pools"})
            continue
        additions.update(new_keys)
        removals.update(old_keys)
        deltas = [
            (after["collection_seconds"] - before["collection_seconds"]) * protocol[2] for before, after in repeats
        ]
        low += min(deltas)
        high += max(deltas)
        before_check = independent_group_check([before for before, _after in repeats])
        after_check = independent_group_check([after for _before, after in repeats])
        details.append(
            {
                "pair_ids": [before["paired_group_id"] for before, _after in repeats],
                "repeat_count": len(repeats),
                "gpu_seconds_range": [min(deltas), max(deltas)],
                "independent_validation": {
                    "passed": before_check["passed"] and after_check["passed"],
                    "baseline": before_check,
                    "candidate": after_check,
                },
            }
        )
    inventory_complete = additions == added and removals == removed
    qualified = all(pool["independent_validation"]["passed"] for pool in details)
    return {
        "covered_added_points": len(additions),
        "covered_removed_points": len(removals),
        "gpu_seconds_range": [low, high],
        "inventory_complete": inventory_complete,
        "complete": inventory_complete and qualified,
        "repeat_pools": details,
        "excluded_pairs": excluded,
    }


@lru_cache(maxsize=3)
def frozen_tables(directory):
    """Bounded per-lane cache; CLI inputs are frozen inventory directories."""
    return load_tables(Path(directory))


def inventory_difference(baseline, candidate):
    before, after = frozen_tables(str(baseline)), frozen_tables(str(candidate))
    if before.keys() != after.keys():
        raise ValueError("baseline and candidate must contain the same operation tables")
    return {
        name: {
            "baseline": len(old),
            "candidate": len(after[name]),
            "added": [after[name][key] for key in sorted(after[name].keys() - old.keys())],
            "removed": [old[key] for key in sorted(old.keys() - after[name].keys())],
        }
        for name, old in before.items()
    }


def non_history_scenario_proof(sample, operation, candidate_rows, alternate_rows):
    """Coverage flags do not change MoE/KDA execution; prove identical tasks.

    Every selected physical member and its complete generated task geometry must
    match both inventories. History setup is never transferred this way.
    """
    if operation not in {"moe", "kda"} or sample["measurement_scope"] != "group":
        return None
    members = member_keys(sample)
    if not members or not members <= candidate_rows.keys() or not members <= alternate_rows.keys():
        return None
    task_keys = set()
    for key in members:
        candidate, alternate = candidate_rows[key], alternate_rows[key]
        task = case_key(candidate.get("_task_case", candidate))
        if task != case_key(alternate.get("_task_case", alternate)):
            return None
        task_keys.add(task)
    return {
        "scope": "non_history_identical_native_tasks",
        "operation": operation,
        "sample_id": sample["sample_id"],
        "source_commit": sample["source_commit"],
        "physical_members": len(members),
        "physical_manifest_sha256": hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest(),
        "complete_task_manifest_sha256": hashlib.sha256("\n".join(sorted(task_keys)).encode()).hexdigest(),
    }


def report_lane(
    baseline,
    candidate,
    samples,
    system,
    backend,
    scenario="model",
    baseline_scenario="baseline",
    source_commit=None,
    equivalents=frozenset(),
    alternate_candidate=None,
):
    result = {}
    for table, delta in inventory_difference(baseline, candidate).items():
        operation = table.split("_", 2)[2]
        lane = []
        for sample in samples:
            if identity(sample) != (system, backend, operation):
                continue
            models = {case.get("model_path") for case in sample.get("group_member_cases", [sample["case"]])}
            eligible = (
                source_commit is None
                or sample["source_commit"] == source_commit
                or (sample["source_commit"], backend, operation) in equivalents
                or (
                    bool(models)
                    and all((sample["source_commit"], backend, operation, model) in equivalents for model in models)
                )
            )
            if eligible:
                lane.append({**sample, "_qualified_runtime_source": source_commit or sample["source_commit"]})
        buckets = defaultdict(list)
        for sample in lane:
            buckets[(geometry(sample["case"]), sample["source_commit"])].append(sample)
        covered = 0
        low = high = 0.0
        gaps = defaultdict(int)
        for case in delta["added"]:
            matches = [bucket for (shape, _commit), bucket in buckets.items() if shape == geometry(case)]
            estimates = []
            for bucket in matches:
                point_samples = [s for s in bucket if s["measurement_scope"] == "point"]
                estimate = predict(case, point_samples)
                exact = any(clean_case(s["case"]) == clean_case(case) for s in accepted(point_samples))
                if estimate is not None and (exact or validation(bucket)["passed"]):
                    estimates.append(estimate)
            estimate = (min(e[0] for e in estimates), max(e[1] for e in estimates)) if estimates else None
            if estimate is None:
                gaps["unmeasured_geometry_or_unbracketed_shape"] += 1
            else:
                covered += 1
                low += estimate[0]
                high += estimate[1]
        # Setup growth of retained workers is unknown without matched old/new
        # group evidence. Do not silently assume a sunk initialization cost.
        validations = [group_validation(bucket) for bucket in buckets.values()]
        pair_lane, scenario_proofs = [], []
        alternate = frozen_tables(str(alternate_candidate)).get(table, {}) if alternate_candidate is not None else {}
        current = frozen_tables(str(candidate))[table]
        for sample in lane:
            proof = None
            if scenario == "shared" and sample.get("scenario") == "model" and alternate:
                proof = non_history_scenario_proof(sample, operation, current, alternate)
            if proof is not None:
                pair_lane.append({**sample, "scenario": "shared"})
                scenario_proofs.append(proof)
            else:
                pair_lane.append(sample)
        paired = paired_groups(
            pair_lane,
            scenario,
            {case_key(clean_case(c)) for c in delta["added"]},
            {case_key(clean_case(c)) for c in delta["removed"]},
            baseline_scenario,
        )
        result[operation] = {
            "baseline_points": delta["baseline"],
            "candidate_points": delta["candidate"],
            "added_points": len(delta["added"]),
            "removed_points": len(delta["removed"]),
            "covered_added_points": covered,
            "uncovered_added_points": len(delta["added"]) - covered,
            "covered_point_seconds_range": [low, high],
            "uncovered_reasons": dict(gaps),
            "group_validation": validations,
            "phase_accounting_check": phase_accounting_check(lane),
            "strata": [
                {
                    "geometry": dict(shape),
                    "source_commit": commit,
                    "point_validation": validation(bucket),
                    "group_validation": group_validation(bucket),
                    "phase_accounting_check": phase_accounting_check(bucket),
                }
                for (shape, commit), bucket in buckets.items()
            ],
            "paired_group_measurement": paired,
            "scenario_equivalence_proofs": scenario_proofs,
            "extra_gpu_hours_range": [v / 3600 for v in paired["gpu_seconds_range"]] if paired["complete"] else None,
            "total_status": "measured_group_difference"
            if paired["complete"]
            else "unknown_setup_delta_or_uncovered_points",
            "setup_delta_seconds": None if delta["added"] or delta["removed"] else 0.0,
            "category": "non_history" if operation in {"moe", "kda", "nccl"} else "history",
        }
    return result


def verified_sources(proofs, repository, target):
    """Verify explicit per-operation equivalence against immutable Git file bytes.

    The proof declares its complete native-runtime dependency scope; matching a
    single entrypoint is insufficient. Planner/case changes do not prove runtime
    equivalence and are not silently accepted as such.
    """
    result = set()
    for proof in proofs:
        if proof["target_commit"] != target:
            raise ValueError("source equivalence must name target and complete runtime dependency scope")
        required = {
            f"python/aisimulate/collector/{proof['backend']}/{name}" for name in ("collect_attn.py", "registry.py")
        }
        required.update(
            f"python/aisimulate/collector/{name}" for name in ("helper.py", "case_generator.py", "registry_types.py")
        )
        if proof["backend"] == "vllm":
            required.add("python/aisimulate/collector/vllm/utils.py")
        supported = proof["operation"] in {"attention_context", "attention_generation"}
        if (
            proof["backend"] == "sglang"
            and proof["operation"] in {"dsa_generation_module", "dsa_generation_module_skip_indexer"}
            and proof.get("model_path")
        ):
            supported = True
            required.discard("python/aisimulate/collector/sglang/collect_attn.py")
            required.update(
                f"python/aisimulate/collector/sglang/{name}"
                for name in ("collect_mla_module.py", "runtime_compat.py", "runtime_limits.py", "dsa_prefill_graph.py")
            )
            model = proof["model_path"].replace("/", "--")
            required.update(
                f"python/aisimulate/src/aisimulate_core/model_configs/{model}_{name}.json"
                for name in ("config", "hf_quant_config")
            )
        if not supported or not required <= {f["path"] for f in proof["files"]}:
            raise ValueError("unsupported or incomplete per-operation equivalence proof")
        for file in proof["files"]:
            for revision in (proof["source_commit"], target):
                payload = subprocess.check_output(["git", "-C", str(repository), "show", f"{revision}:{file['path']}"])
                expected = file.get("sha256") or file.get(
                    "source_sha256" if revision == proof["source_commit"] else "target_sha256"
                )
                if (
                    file.get("source_sha256", expected) != file.get("target_sha256", expected)
                    or hashlib.sha256(payload).hexdigest() != expected
                ):
                    raise ValueError(f"source equivalence hash mismatch: {revision}:{file['path']}")
        key = (proof["source_commit"], proof["backend"], proof["operation"])
        result.add((*key, proof["model_path"]) if proof.get("model_path") else key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--source-equivalence", type=Path, help="JSON list of per-op verified runtime dependency hashes"
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-commit", required=True, help="exact candidate source revision eligible for point interpolation"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    proofs = json.loads(args.source_equivalence.read_text()) if args.source_equivalence else []
    if isinstance(proofs, dict):
        proofs = proofs["proofs"]
    equivalents = verified_sources(proofs, args.repository, args.source_commit)
    result = {
        "source_equivalence_proofs": proofs,
        "range_kind": "empirical_not_statistical_confidence_interval",
        "allocation_time_included": False,
        "fleet_gpu_hours_range": None,
        "comparison_accounting": (
            "shared/model/model_vs_shared are alternatives; reused observations are not additional collection spend"
        ),
        "coverage_scope": "declared inventory only; deferred or undeclared model routes remain unmeasured",
        "lanes": {},
    }
    for directory in sorted(args.baseline.glob("*/*")):
        if not directory.is_dir():
            continue
        system, backend = directory.parent.name, directory.name
        result["lanes"][f"{system}/{backend}"] = {
            name: report_lane(
                directory,
                root / system / backend,
                samples,
                system,
                backend,
                name,
                source_commit=args.source_commit,
                equivalents=equivalents,
                alternate_candidate=args.model / system / backend if name == "shared" else None,
            )
            for name, root in (("shared", args.shared), ("model", args.model))
        }
        result["lanes"][f"{system}/{backend}"]["model_vs_shared"] = report_lane(
            args.shared / system / backend,
            args.model / system / backend,
            samples,
            system,
            backend,
            "model",
            "shared",
            source_commit=args.source_commit,
            equivalents=equivalents,
        )
        frozen_tables.cache_clear()
    result["samples"] = samples
    result["phase_accounting"] = (
        "phase_seconds are supporting diagnostics, may be nested, and are never added to inclusive collection clocks"
    )
    result["sources"] = sorted({s["source_commit"] for s in samples})
    result["sample_status_counts"] = {
        status: sum(s["status"] == status for s in samples) for status in sorted({s["status"] for s in samples})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
