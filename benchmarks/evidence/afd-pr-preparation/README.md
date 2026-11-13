<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD historical provenance and host performance

This directory retains historical source attribution, host-performance samples
and their reproduction helpers. Test-run logs, JUnit reports and PR preparation
records are kept outside the source tree; test results belong in the PR description.

## Historical provenance


[historical-summary.json](historical-summary.json) retains the original source and
binary identities, archive-file hashes and attribution of six retired E2E records.
The complete 64-file archive remains at revision
`baff0871ff5495d76ed3ddf8c0f45f461932bff4`, preserved by the original branch,
backup reference and Git bundle. Recover any retained file with
`git show baff0871ff5495d76ed3ddf8c0f45f461932bff4:PATH` after obtaining that history.
Recorded source/binary digests identify historical artifacts; they do not restore
missing measurement-time binaries or uncommitted source trees.

The historical campaigns covered source-ordered execution unification,
configuration ownership, dense-prefix pricing, pure-EP restrictions, local FFNs
and A-side TP, routing placement, heterogeneous fabrics and worker-role lowering.
Their measured timings and test counts apply to those snapshots. They establish
neither current-tree qualification nor physical serving accuracy. Earlier schema
numbers and supported configuration sets must not be interpreted as today's API.
The MTP communication note is available at the same archive revision under
`benchmarks/evidence/afd-mtp-comm/README.md`; it reports modeled effects and static
host-cost analysis, not measured host performance.

The six retired Qwen3-32B E2E records were already marked `historical` and were not
selected by the active parity suite. Their source attribution is retained in the
summary. The six active E2E cases and fourteen execution-plan cases remain live
regression fixtures, not historical evidence.

## Measurement identity (2026-10-25)

These measurements describe candidate `51e22a92af3cafd20c2b97a3b938c2d57185329b` against
baseline `f3a3e3156530574f5e7a10446f19e571b82db871`, including main's GLM runtime and
Task validation fixes. The candidate is the code/documentation commit immediately
before the evidence-only commit. Loaded extension paths and hashes remain in the
raw performance records. Both extensions were rebuilt from their own checkout;
source mtimes were not used as evidence of source equality.

Earlier measurements at candidate `754bdba1dfdd5571e3730eca1901778d76343ad1`
against `aeb05094a243dc0a054731ac11e63d1af49b6e89` remain archived with the
pre-rewrite history and local backup. They are historical and are not counted as
validation of this revision.

| Identity | Baseline | Candidate |
| --- | --- | --- |
| Release extension SHA256 | `3a07c6f0361e788cddf231afd4db7a209a1d675d4a4ed88ecb87cd1f85bbc923` | `77b6efb1c8f096b481613b3c4c0c17f8b39c2532048dbf97c661016801847af4` |
| Rust core Git tree | `66ddb72beaa3f4885d838dbf7ae0b3d9b35f038a` | `c67cb041285c1c0900c976ff8b948b6dd2fb2d72` |

Both arms used the same inputs:

- `Cargo.lock` SHA256: `b8ba470a555be37b373d0f0098260ae3339a7667ade6a7ab50e228328dbadfed`.
- `python/aisimulate/uv.lock` SHA256: `a321d06e9d3e6da5ee209d523b3b0526dfa0b18c0a5c673421c94cbfc23ed338`.
- Packaged systems/data Git tree: `302be64c650f4f03c58ed286b08c04f9f401038b`.

Both extensions used Python 3.13.16, the release profile and the
`extension-module` feature. Toolchain: `rustc 1.99.0 (b940084d7 2026-09-28)`;
`cargo 1.99.0 (5f94df478 2026-08-27)`.

## Local host performance

The [forward gate summary](forward-gate/summary.md) covers all 64
cases, five alternating paired rounds, ten warmups and 100 iterations. Predicted
values match within 1e-9. The 128 cold/warm points classify as
29 IMPROVEMENT, 41 OK, 58 UNSTABLE. There are no run errors; the unchanged repository comparator
reports `blocking=false`. Raw samples and full comparison are retained as
deterministic gzip JSON files.

[replay-summary.json](replay-summary.json) contains medians, ranges and paired
ratios; [replay-performance.json.gz](replay-performance.json.gz) retains all samples.
Each row has five fresh processes per arm and workload, fixed thread counts,
1024 input tokens and 128 output tokens per request. Ordinary simulated summary
metrics agree exactly after excluding only the three wall-clock fields. All
native and fresh CLI work counts are checked. AFD rows measure candidate scaling
only; the retired analytical main runner has different semantics.

| Workload | Requests | CLI baseline / candidate (ms) | Warm runner baseline / candidate (ms) |
| --- | ---: | ---: | ---: |
| ordinary_fixed | 16 | 654.90 / 657.04 | 0.84 / 0.86 |
| ordinary_fixed | 1024 | 716.72 / 715.92 | 41.08 / 40.91 |
| ordinary_op_level | 16 | 1998.81 / 1993.03 | 14.34 / 14.30 |
| ordinary_op_level | 1024 | 2042.44 / 2047.46 | 57.68 / 57.18 |
| afd_four_stage | 16 | — / 2179.33 | — / 20.66 |
| afd_four_stage | 1024 | — / 2301.40 | — / 68.14 |
| afd_two_stage | 16 | — / 2231.60 | — / 20.80 |
| afd_two_stage | 1024 | — / 2345.11 | — / 67.64 |

CLI time includes supervision, construction, replay and output. A warm runner
sample is a complete runner call after one first call and three warmups; native
providers are reconstructed each call. Compile time, empty native preparation,
first runner time and whole-process peak RSS are reported separately in JSON.
The first runner call follows an empty native preparation; it is not process-cold.
RSS describes the complete process, not retained-cache allocation.

Ordinary paired median ratios are CLI 0.994–1.003, warm runner 0.994–1.021. These local observations
do not establish a speedup. Measurements ran sequentially on macOS without CPU
affinity and without concurrent tests or builds. Unstable points remain visible
in the gate output; no Linux affinity-controlled, long-corpus or real
serving-accuracy qualification is claimed by this record.

## Reproduction

Use separate baseline and candidate checkouts with their matching release native
extensions installed. The helpers verify the actual loaded extension path and
hash. Build each checkout with the same lockfiles, systems data, Python and
profile listed above. The macOS build command was:

```sh
cargo rustc --offline --release -p aisimulate-core --features extension-module --lib -- -C link-arg=-undefined -C link-arg=dynamic_lookup
```

Use `PYO3_PYTHON` to select the matching Python executable when needed. Avoid
reusing crate outputs between different checkouts without forcing recompilation;
verify the build log, source identity and actual loaded extension. Run the
helpers from the candidate checkout, with no concurrent builds or tests:

```sh
python benchmarks/evidence/afd-pr-preparation/measure_forward.py --baseline /path/to/baseline --output /tmp/afd-forward
python benchmarks/evidence/afd-pr-preparation/measure_replay.py --baseline /path/to/baseline --output /tmp/afd-replay.json
```

The forward helper writes compressed raw/comparison JSON automatically. The
replay helper writes plain JSON; the retained copy is gzip-compressed with
mtime=0 without changing its JSON content. Local paths inside raw records
identify the loaded binaries; use your own checkout paths to reproduce.
`historical-summary.json` remains historical when new measurements are collected.
