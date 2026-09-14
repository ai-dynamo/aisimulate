# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover exact historical cold ReplaySpecs through their original SHA-256.

No predicted timing is used as a replay input. This is retained-report input
closure, not a new admission of unavailable raw lifecycle evidence.
"""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cases(report):
    if "segments" in report:
        return [
            (segment_index, c)
            for segment_index, s in enumerate(report["segments"])
            for c in s["e2e_cases"]
        ]
    return [(0, c) for c in report["cohorts"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fpm-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    BASE = args.repo.resolve() / "data/experimental/deepseek-v41/gb300-silicon/report"
    FPM = args.fpm_repo.resolve() / "data/experimental/deepseek-v41/gb300-fpm"
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    summaries = []
    for scope in ("off", "on"):
        source_path = (
            BASE
            / f"sol-review-v2/{scope}"
            / (
                "silicon-results.json.gz"
                if scope == "on"
                else "e2e-silicon-results.json.gz"
            )
        )
        donor_path = (
            FPM
            / f"{scope}-union-v3-tracewait2/reports/core"
            / ("segments.json.gz" if scope == "on" else "e2e.json.gz")
        )
        plan_path = (
            BASE
            / "serving-v4"
            / (
                "on-segmented/logical-plan.json.gz"
                if scope == "on"
                else "off-prefix-refined/main-plan.json.gz"
            )
        )
        engine_path = (
            BASE / "sol-review-v2/field-on/e2e-silicon-results.json.gz"
            if scope == "on"
            else source_path
        )
        pins = {
            str(p): sha(p) for p in {source_path, donor_path, plan_path, engine_path}
        }
        original, donor = read(source_path), read(donor_path)
        plan = read(plan_path)
        plans = {c["cohort_id"]: c for c in plan["cohorts"]}
        donors = {c["cohort_id"]: c for _, c in cases(donor)}
        engine = read(engine_path)["replay_engine"]
        recovered = []
        for ordinal, (segment_index, cohort) in enumerate(cases(original)):
            row = {
                k: copy.deepcopy(cohort[k])
                for k in (
                    "cohort_id",
                    "purpose",
                    "trial_index",
                    "trial_seed",
                    "declared_coverage_role",
                    "corpus_role",
                    "observed",
                    "replay_spec_sha256",
                    "native_initial_cached_tokens",
                    "cache_semantics_match",
                )
                if k in cohort
            }
            row.update(
                input_ordinal=ordinal,
                segment_ordinal=segment_index,
                previous_prediction_status=cohort["status"],
            )
            if cohort["purpose"] == "prefix-reuse-B":
                row.update(
                    recovery_status="unavailable",
                    reason="Original seed-A submit offsets are not retained",
                )
            elif cohort["cohort_id"] not in plans:
                row.update(
                    recovery_status="unavailable",
                    reason="Original token-bearing plan cohort absent",
                )
            else:
                requests = cohort.get("requests")
                row["arrival_metadata_source"] = "original_silicon_report"
                if requests is None:
                    counterpart = donors.get(cohort["cohort_id"], {})
                    if counterpart.get("observed") != cohort["observed"]:
                        raise ValueError("cross-model report observed metrics differ")
                    requests = counterpart.get("requests")
                    row["arrival_metadata_source"] = (
                        "same-observation_fpm_report_request_metadata"
                    )
                if requests is None:
                    row.update(
                        recovery_status="unavailable",
                        reason="Original per-request submit offsets absent",
                    )
                else:
                    by_id = {r["request_id"]: r for r in requests}
                    planned = plans[cohort["cohort_id"]]["requests"]
                    if set(by_id) != {r["request_id"] for r in planned}:
                        raise ValueError("request identity mismatch")
                    materialized = []
                    for r in planned:
                        assert (
                            hashlib.sha256(
                                canonical(r["input_token_ids"]).encode()
                            ).hexdigest()
                            == r["input_token_ids_sha256"]
                        )
                        assert (
                            len(r["input_token_ids"])
                            == by_id[r["request_id"]]["input_length"]
                        )
                        assert (
                            r["output_tokens"]
                            == by_id[r["request_id"]]["output_length"]
                        )
                        materialized.append(
                            {
                                "id": r["request_id"],
                                "input_tokens": len(r["input_token_ids"]),
                                "input_token_ids": r["input_token_ids"],
                                "output_tokens": r["output_tokens"],
                                "arrival_time_ms": by_id[r["request_id"]][
                                    "arrival_time_ms"
                                ],
                            }
                        )
                    spec = {
                        "version": 1,
                        "topology": {
                            "kind": "aggregated",
                            "workers": {"initial_workers": 1},
                        },
                        "engine": copy.deepcopy(engine),
                        "record_per_request": True,
                        "requests": materialized,
                    }
                    actual = hashlib.sha256(canonical(spec).encode()).hexdigest()
                    if actual != cohort["replay_spec_sha256"]:
                        row.update(
                            recovery_status="unavailable",
                            reason="Reconstructed input does not match original ReplaySpec SHA",
                            candidate_sha256=actual,
                        )
                    else:
                        row.update(
                            recovery_status="exact_original_spec_sha256",
                            replay_spec=spec,
                            cohort_start_ms=0.0,
                            request_ids=[r["request_id"] for r in planned],
                        )
            recovered.append(row)
        assert pins == {p: sha(Path(p)) for p in pins}
        summary = {
            "scope": scope,
            "planned_cohorts": len(recovered),
            "exact_specs": sum(
                c["recovery_status"] == "exact_original_spec_sha256" for c in recovered
            ),
        }
        result = {
            "method": "Exact original SILICON ReplaySpec SHA closure from retained observation inputs",
            "fresh_raw_admission": False,
            "new_measurements": False,
            "predicted_timing_as_input": False,
            "program_sha256": sha(Path(__file__)),
            "source_files_sha256": pins,
            "plan_decompressed_sha256": hashlib.sha256(
                gzip.decompress(plan_path.read_bytes())
            ).hexdigest(),
            "summary": summary,
            "cohorts": recovered,
        }
        path = OUT / f"{scope}.json.gz"
        with path.open("xb") as stream:
            stream.write(
                gzip.compress(
                    (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
                    mtime=0,
                )
            )
        summaries.append(summary | {"sha256": sha(path)})
        print(json.dumps(summaries[-1]), flush=True)
    with (OUT / "completion.json").open("x") as stream:
        json.dump(
            {
                "program_sha256": sha(Path(__file__)),
                "complete": True,
                "scopes": summaries,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
