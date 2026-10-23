<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model Data Quality Gate: phase-one shadow rollout

The root [workflow](../.github/workflows/model-data-quality.yml) runs on every
pull request, including drafts and forks, on an ephemeral GitHub-hosted runner.
It uses no secrets or self-hosted infrastructure, read-only repository permission,
no persisted checkout credentials, and pinned actions. Its visible job is `Model Data Quality Gate (shadow)`. It does not modify branch
rules. **This is the first phase of AIC-1916, not the completed quality gate.**

Run from a clean checkout of the exact head, with the portable dependencies
listed in the workflow:

```sh
python python/aisimulate/tools/model_data_gate/run.py \
  --base <full-base-sha> --head <full-head-sha> --out /tmp/model-data-quality
```

The command writes `report.json`, `summary.md`, exact per-file hashes and schemas,
and row-diff CSVs. It returns 1 for any failed or incomplete required stage,
including a missing/crashed stage. `Not applicable` returns 0 only for an empty
or explicitly Markdown/reStructuredText documentation-only change set.
Machine-readable schemas under documentation paths remain applicable. Unknown paths, collector,
resolver, interpolation, schema, gate-tool and workflow changes are applicable.
Both old and new paths of moves, and deletions, are included. Comparison uses the
supplied base and head trees directly; it never silently substitutes a merge base.
Snapshots are read from Git objects, and symlinks in the exported data tree fail.
The checkout must match the head, including tracked tooling changes.

`--classify-only` is a cheap planning command used before dependency installation.
Its exit status says whether classification completed, **not** whether validation
passed. For applicable changes it writes a provisional failing report. Later
setup/test failures therefore cannot leave behind a false passing gate report.

## What this phase proves

| Stage | Implemented evidence | Explicit remaining coverage |
| --- | --- | --- |
| Artifact integrity | Readability/LFS rejection; schemas and exact row additions/removals/modifications; finite and non-null values; timing bounds; duplicate shape/kernel keys; GEMM required columns and integer dimensions; catalog family, framework/version, Collector metadata validation and row counts; existing power schema checks | Other operation-specific schemas/keys; device/topology identity; full legacy/reuse provenance; attested measured-versus-estimated source |
| Numerical sanity | Existing cross-backend/curve/GEMM SOL detectors run on each affected version, even when it is older than the latest backend version; head findings are reported with exact-base diagnostic counts | Reviewed baseline and thresholds; non-GEMM physical bounds; extrapolation; unsupported schemas and unpaired comparisons |
| Production reachability | A required, explicit INCOMPLETE result | Real native exact-key/interpolation-boundary probes; intended file/row/SILICON attribution; warm/cold paths; packaged-but-unreachable and fallback negatives |
| Behavior and parity | A required, explicit INCOMPLETE result identifying the existing prediction workflow | Exact-base/head affected predictions, native parity, supported-to-error and nonfinite output coverage, explained discontinuities, held-out accuracy where applicable |

The aggregate therefore **fails for every applicable change in this phase**.
An apparently valid data addition also remains incomplete until its source,
reachability and behavior are proved. Documentation changes can be not applicable.
This deliberately prevents partial diagnostics from being mistaken for acceptance
of all four stages. Existing Collector Data Check and Prediction Regression Gate
remain independent Full CI diagnostics; their success is not imported as proof of
missing per-coordinate coverage.

Artifact checks reuse `parquet_diff`, `collector.op_catalog`, the Collector's
`validate_collection_meta_for_update`, and the shared power storage checks.
Existing historical/reduced sidecars do not acquire stronger attestation merely
because their syntax is accepted. Legacy sidecars that are not valid for a fresh
collection fail the artifact check and require review during calibration.
GEMM has an explicit minimum schema. Other operation schemas remain incomplete.
Only `computescale` permits signed finite latency deltas; the two score-calibration
tables permit zero durations, while other timings must be positive. Null and
nonfinite values are never exempted.

The numerical report does **not** treat the PR base as an approved anomaly
baseline. No head finding is suppressed, even when present at base. Base/head
fingerprint counts are diagnostics for review only. The existing detector has
known limits (latest peer selection, no comparison for one backend, statistical
noise, grouped finding identities); these cannot satisfy the reviewed baseline
and ratchet acceptance criteria. New baseline policy must include reviewed
examples, calibrated thresholds and a regression fixture for each confirmed miss.

## Completing the rollout

1. Calibrate artifact and numerical findings against representative real changes,
   establish a committed reviewed baseline, and prevent both new coordinates and
   worsening existing findings from being absorbed by that baseline.
2. Integrate production native probes and affected prediction/parity evidence
   tied to both immutable revisions. Add end-to-end valid-addition, unreachable,
   fallback, parity and behavior regression fixtures before allowing PASS.
3. Extend schema, provenance and applicability coverage with explicit operation
   contracts; indirectly affected coordinates must not disappear from selection.
4. Remove workflow-level shadow tolerance after calibration, rename the job to
   `Model Data Quality Gate`, then activate only
   the aggregate `Model Data Quality Gate` as a required branch-rule context with
   AIC-1911. Verify effective rules and controlled failures separately.

During shadow rollout only the diagnostic execution step tolerates its nonzero
exit. Setup, contract tests, missing reports and artifact-upload failures remain
workflow failures. A green workflow wrapper means the shadow run completed, not
that `report.json` passed or that hardware accuracy was qualified.

The portable hosted contract suite runs the existing Parquet, anomaly and
prediction comparator/report tests without loading the application-wide native
fixtures. The report suite's `test_version_sort_is_version_aware` requires the
native SDK and is explicitly left in its existing Full CI application unit shard;
it is not mocked or claimed as covered by this portable job.
