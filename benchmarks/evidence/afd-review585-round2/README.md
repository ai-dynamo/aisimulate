<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR 585 second review corrections (2026-11-01)

## Immutable inputs

- Main fixed at implementation start: `e8cbe2dc150f0df309993474842c2e417753873d`.
- Integrated, pre-fix baseline: `444c71d55a66b5bc50ea4ddab30e69a024c30525`.
- Measured runtime/generator source: `d139f3646774dced5be67d3c20f3c51cde11ba72`.
- Signed contribution commit: `45f446825ddc34f1add75999a3fb009410fe79a6`; its tree is identical to measured source (`ad8d1f0324ec2878c95fb1bb8d107c562661e94d`). Measurement identity is not relabeled after the DCO metadata correction.
- The two compressed manifests pin 943 runtime/config/dependency input hashes, source trees, data trees, compiler/features and loaded release binary hashes.
- Both binaries use matching Python, locks, release/extension-module features and macOS dynamic-lookup linker flags. Cargo build diagnostics were 189/185 seconds: one build per arm, not a build-speed gate.
- `source.bundle` contains measured source and historical baseline `6b57265e`. Its prerequisites are upstream `e8cbe2dc` and ancestor `dcadbc508`. The integrated baseline is the measured source's parent. Import into a separate checkout of pinned main. Native binaries are not checked in.

## Corrections and independent checks

AFD copies public source metadata; only AFD gets Gemma4's shared-branch count. Public helpers are byte-identical to pinned main. Ordinary Gemma4 context/generation/encoder OpSpecs remain unchanged before/after AFD preparation. Strict precision resolution is shared by AFD timing and memory only, preserving requested modes and effective provenance.

Invalid microbatch geometry is filtered before candidate limits; direct/pinned inputs reject. Trace/session/agentic explicit frozen timing rejects before materialization. Synthetic defaults and supported overrides remain. Symmetric transfer queries are reused while retaining both direction records, scaling, energy and source. Checked timing conversions precede queries, cache and evidence mutation.

Main integration preserves ngram/async checks, ordinary PP preflight, detokenization and finite trace scanning. Relevant retired analytical AFD CLI assertions are migrated to role-based validation.

## Validation

`validation.json` records scopes and local logs:

- Rust library: 2,670 passed, one ignored, with embed-python.
- Public Rust API: 21 passed.
- Related Python: 1,415 passed; generator request regressions: 120 passed; import/public contracts: 140 passed.
- Parity: 389 general cases plus 36 AFD cases verified across runs. Initial AFD failures from a missing isolated-checkout venv entry point were rerun with the correct entry point.
- Ruff check/format, Cargo format, whitespace and packaged legal-file checks pass. These are local results, not hosted Fast/Full CI or an independent review.

Only `qwen_tp1_mtp` and `qwen_tp2_mtp` goldens change. Main `18d71dcd` (#637) switches the shared speculative sampler from StdRng to SmallRng, changing the seed stream. Independent integrated-baseline and candidate runs agree exactly. Counts and TTFT remain unchanged; duration and TPOT deltas are in `mtp-deltas.json`. The verified generator preserves previous generation identities. All six E2E and twelve other layout records remain byte-identical.

## Performance scopes

The unchanged forward protocol/matrix runs five alternating pairs, ten warmups and 100 warm iterations. All 64 cases complete, predictions agree at 1e-9, and the comparator reports `blocking=false`: 66 improvement, 35 OK and 27 unstable cold/warm points. This advisory classification is not a blanket performance approval.

`measure.py` reuses the existing nine-call replay worker. Each of 13 workloads has five alternating paired fresh processes, five retained warm calls, and a separate full supervised CLI command. Loaded binary hashes and complete request/input/output totals are checked. All modeled summary metrics match between arms. No builds or correctness campaigns run concurrently with timing.

Warm complete-runner paired median ratios are 0.994-1.008. The ordinary fixed CLI control is bimodal (roughly 620/680 ms), with opposite-signed differences and paired range 0.901-1.115. Its 1.086 median is **inconclusive**, not a stable regression or a pass. Every invocation is retained. This unpinned macOS campaign does not qualify all long traces or physical serving accuracy.

`compile_ms` covers only prediction-to-ReplaySpec construction, not module/process startup. Empty-native, first runner, warm runner, native wall, full CLI and peak RSS samples are retained separately. Moving work outside a narrow timer is not a speedup. Native wall excludes some provider preparation and final aggregation; full runner/CLI timers cover those costs. No request-sized retained state is added: metadata copies are bounded at preparation, successful conversions allocate nothing, and reuse removes the second transfer query.

Supplemental Gemma4 ordinary/AFD and Hopper weight-only/AFD nvfp4 rows use SOL explicitly, not measured-kernel accuracy. Ordinary native CLI uses effective `nvfp4_wo`; the separate legacy CLI campaign directly requests `nvfp4` on H100, exercising the restored public helper. Its five paired complete estimates have identical numerical reports, ratio 0.994 and range 0.987-0.995. This is an analytical estimate, not request replay.

Legacy stdout includes timestamps and Python source locations. The initial raw-text probe rejected those dynamic prefixes. The reported five-pair campaign normalizes only those prefixes, retaining all messages and numeric fields. Raw/normalized digests and the measured harness checksum are retained. `legacy-harness.py.gz` preserves that exact script; the readable `.py` has only import/line formatting changes, with identical function ASTs; the failed text-comparison probe is not counted as a measured pair.

## Paired timing detail

| Workload | Requests | Warm runner ratio | CLI ratio | CLI pair range |
|---|---:|---:|---:|---|
| ordinary_fixed | 16 | 0.996 | 1.086 | 0.901-1.115 |
| ordinary_op_level | 16 | 1.004 | 0.996 | 0.984-1.009 |
| ordinary_op_level | 1024 | 1.000 | 0.999 | 0.983-1.016 |
| afd_four_stage_low | 16 | 1.005 | 1.001 | 0.979-1.011 |
| afd_four_stage_busy | 1024 | 1.004 | 0.994 | 0.971-1.017 |
| afd_two_stage | 1024 | 1.001 | 1.006 | 0.955-1.015 |
| afd_ragged | 64 | 1.008 | 1.001 | 0.991-1.016 |
| afd_depth16 | 64 | 1.003 | 1.004 | 0.994-1.017 |
| afd_restore | 32 | 1.003 | 1.001 | 0.981-1.017 |
| ordinary_gemma4 | 16 | 0.994 | 1.004 | 0.982-1.041 |
| afd_gemma4 | 16 | 0.997 | 1.004 | 0.998-1.016 |
| ordinary_nvfp4_wo_hopper | 16 | 0.998 | 1.005 | 0.978-1.021 |
| afd_nvfp4_hopper | 16 | 0.996 | 1.004 | 0.993-1.006 |

All absolute preparation/first/warm/native/CLI medians and pair ranges are in `summary.json`; raw paired samples are retained in the compressed reports.

## Reproduction

Build the pinned source snapshots in separate checkouts with matching dependencies and flags; verify loaded binaries and create manifests for those builds. Set PYTHONPATH to the selected checkout. Embedded Rust tests additionally need the venv site-packages directory.

```sh
python benchmarks/evidence/afd-pr-preparation/measure_forward.py --baseline /path/to/baseline --candidate-revision d139f3646774dced5be67d3c20f3c51cde11ba72 --output /tmp/forward
python benchmarks/evidence/afd-review585-round2/measure.py --baseline /path/to/baseline --candidate-revision d139f3646774dced5be67d3c20f3c51cde11ba72 --output /tmp/replay.json
python benchmarks/evidence/afd-review585-round2/measure_legacy.py --baseline /path/to/baseline --output /tmp/legacy.json
```

Historical campaigns and identities remain unchanged. No remote CI, PR comment or push was performed.
