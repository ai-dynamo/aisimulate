# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare actual returned token IDs for the prespecified first 40 OFF/ON trials."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from render_report import checksum, read, require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def returned_ids(raw, planned):
    require(
        raw["request_id"] == planned["request_id"] and raw["input_token_ids"] == planned["input_token_ids"],
        "raw response request/input identity differs",
    )
    require(
        raw["input_token_ids_sha256"] == planned["input_token_ids_sha256"] == digest(planned["input_token_ids"]),
        "actual input token checksum differs",
    )
    require(
        raw["done"] is True and raw["summary"]["valid"] is True and not raw["errors"], "request did not close validly"
    )
    require(raw["requested_output_tokens"] == planned["output_tokens"], "output length policy changed")
    output = []
    unavailable = []
    for number, event in enumerate(raw["events"]):
        if not event.get("is_output"):
            continue
        ids = event.get("token_ids")
        if ids is None:
            unavailable.append(number)
            continue
        require(
            isinstance(ids, list) and ids and all(type(v) is int and v >= 0 for v in ids), "invalid returned token IDs"
        )
        require(
            event["data"]["nvext"]["completion_token_ids"] == ids, "parsed token IDs differ from original HTTP frame"
        )
        # Also bind the structured frame to the original SSE JSON bytes.
        require(json.loads(event["raw_data"]) == event["data"], "structured frame differs from original SSE bytes")
        output.extend(ids)
    if unavailable:
        return None, {"unavailable_output_frames": unavailable}
    require(
        len(output) == raw["summary"]["completion_tokens"] == planned["output_tokens"],
        "returned token IDs are incomplete",
    )
    return output, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("off-root", "on-root", "on-segment-admission", "paired-input-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {"input_proof": args.paired_input_proof, "on_admission": args.on_segment_admission}
    roots = {"off": args.off_root, "on": args.on_root}
    for profile, root in roots.items():
        paths[profile + "_plan"] = root / "strata/primary/main-plan.json"
        paths[profile + "_summary"] = root / "strata/primary/main-client/summary.json"
        paths[profile + "_closed_audit"] = root / "closed-segment-audit.json"
        paths[profile + "_execution"] = root / "execution.json"
    hashes = {key: checksum(path) for key, path in paths.items()}
    values = {key: read(path) for key, path in paths.items()}
    proof = values["input_proof"]
    require(
        proof["schema"] == "dsv41.verification.paired-input-proof.v1"
        and proof["exact_input_tokens_and_output_lengths_match"] is True,
        "prespecified paired input proof absent",
    )
    require(
        proof["paired_trial_indices"] == [0, 39] and proof["requests"] == 960 and proof["cohorts"] == 680,
        "paired subset changed",
    )
    require(
        values["on_admission"]["valid"] is True
        and values["on_admission"]["source_inputs_sha256"]["closed_audit"] == hashes["on_closed_audit"],
        "ON response comparison requires independently admitted closed physical data",
    )
    cases = {}
    for profile in roots:
        require(hashes[profile + "_plan"] == proof[profile + "_plan_sha256"], "frozen paired plan changed")
        require(
            values[profile + "_plan"]["run_id"]
            == proof[profile + "_run_id"]
            == values[profile + "_execution"]["run_id"],
            "physical paired run changed",
        )
        if profile == "on":
            require(
                values["on_admission"]["physical_run_id"] == proof["on_run_id"],
                "admitted physical run differs from paired ON data",
            )
        audit = values[profile + "_closed_audit"]
        require(audit["valid"] is True and audit["errors"] == [], "paired source was not closed and audited")
        planned = [c for c in values[profile + "_plan"]["cohorts"] if c["trial_index"] < 40]
        actual_ids = {c["cohort_id"] for c in values[profile + "_summary"]["cohorts"]}
        require(all(c["cohort_id"] in actual_ids for c in planned), "paired first40 are not fully observed")
        cases[profile] = {(c["trial_index"], c["purpose"]): c for c in planned}
        require(len(cases[profile]) == len(planned) == 680, "duplicate or omitted paired cohort")
    require(cases["off"].keys() == cases["on"].keys(), "paired scenario set differs")
    rows, request_files = [], {}
    for key in sorted(cases["off"]):
        off, on = [cases[profile][key] for profile in ("off", "on")]
        require(
            off["trial_seed"] == on["trial_seed"] and len(off["requests"]) == len(on["requests"]),
            "paired trial differs",
        )
        for ordinal, (left, right) in enumerate(zip(off["requests"], on["requests"], strict=True)):
            require(
                left["input_token_ids"] == right["input_token_ids"] and left["output_tokens"] == right["output_tokens"],
                "paired request inputs differ",
            )
            outputs, limitations = {}, {}
            for profile, request in (("off", left), ("on", right)):
                path = roots[profile] / "strata/primary/main-client/requests" / (request["request_id"] + ".json")
                raw = read(path)
                outputs[profile], limitations[profile] = returned_ids(raw, request)
                request_files[f"{profile}:{request['request_id']}"] = (path, checksum(path))
            comparable = all(value is not None for value in outputs.values())
            row = dict(
                trial_index=key[0],
                trial_seed=off["trial_seed"],
                purpose=key[1],
                request_ordinal=ordinal,
                off_request_id=left["request_id"],
                on_request_id=right["request_id"],
                requested_output_tokens=left["output_tokens"],
                input_token_ids_sha256=left["input_token_ids_sha256"],
                status="equal"
                if comparable and outputs["off"] == outputs["on"]
                else "different"
                if comparable
                else "unavailable",
                output_token_ids_sha256={p: digest(v) if v is not None else None for p, v in outputs.items()},
                limitations=limitations,
            )
            if comparable:
                row["equal_token_positions"] = sum(a == b for a, b in zip(outputs["off"], outputs["on"], strict=True))
                row["first_different_token_position"] = next(
                    (i for i, (a, b) in enumerate(zip(outputs["off"], outputs["on"], strict=True)) if a != b), None
                )
            rows.append(row)
    require(len(rows) == 960, "paired request count differs")
    require(hashes == {k: checksum(p) for k, p in paths.items()}, "source changed during paired comparison")
    require(all(checksum(p) == h for p, h in request_files.values()), "raw output changed during paired comparison")
    result = dict(
        schema="dsv41.paired.returned-token-ids.v1",
        scope="prespecified first40 OFF/ON trials; descriptive output equality only",
        sampling_exclusion=False,
        timing_observations_unchanged=True,
        input_files_sha256=hashes,
        source_sha256=checksum(Path(__file__)),
        physical_runs={p: proof[p + "_run_id"] for p in roots},
        paired_trial_count=40,
        paired_cohort_count=680,
        paired_request_count=len(rows),
        requests_by_status=dict(collections.Counter(r["status"] for r in rows)),
        raw_request_sha256={k: h for k, (_, h) in request_files.items()},
        requests=rows,
        limitations=[
            "Exact returned token equality does not establish task quality or numerical equivalence of hidden states.",
            "Different outputs do not authorize removal of timing observations.",
            "Execution profiles and physical runtimes differ; this is not a causal attribution of discrepancies.",
        ],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["requests_by_status"]))


if __name__ == "__main__":
    main()
