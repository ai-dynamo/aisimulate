# Simulation regression detection experiment

This temporary branch runs the current simulation performance check against three
independent reverse patches. It is not intended for merge. No PR is required.

Baseline: `8a709362b5f0692d263eceab4cba0a13790fe138`.

| Comparison | Candidate | Reversed fix |
|---|---|---|
| Control | `8a709362b5f0692d263eceab4cba0a13790fe138` | None; independent base and head builds |
| Without #295 | `4fe1aff3e1003f4b8994f5b9721800f42c8cf1f2` | `2e01b14a543889f96639a4b0bb4c1d17523b545b` |
| Without #321 | `71e54cb06b24e2e774c4583fe017cdb49298b834` | `0ac7512b3f8ef6ad4d9b4fb8ead04d20b1ccf44e` |
| Without #386 | `00732f0f4bb5637f52d15c4f8fec612602db10e8` | `3d5ecfd860df537e25180c663a34e55257e9bf15` |

## Reverse-patch adjustments

- **#295:** Restore owned timing evidence, cache-entry copies, checked accumulation,
  and the Python conversion fallback. Retain the later FPM evidence history and
  reset, shared canonical model ownership, and native FPM KV-ceiling lookup.
  Remove tests of optimized private types removed by the reverse patch; retain
  the current diagnostic and evidence behavior tests, adapting the diagnostic
  fixture to the current `Arc` types. The existing diagnostic-provider route
  remains active when selected; the experiment does not force the Python fallback.
- **#321:** Restore eager cache snapshots, owned radix edge arrays, and copies of
  lease page vectors. Retain the complete current SGLang scheduler test file,
  including later bounded-checkpoint and rollback coverage. The snapshot-sharing
  implementation tests return to their pre-fix form with the reverse patch.
- **#386:** Restore eager request-ID membership sets and duplicate collection
  before sorting. Retain the complete current replay component test module,
  including worker-wide membership and idle-rank coverage.

All candidates start from the same baseline. Model data, Python source,
dependencies, simulation gate code, fixture, thresholds, and workload sizes match
the baseline exactly. The pipeline checks those invariants before measurement.

## Pipeline

The experiment branch adapts the existing `simulation-performance.yml` only:
manual dispatch selects five wheel builds and four comparisons with fixed SHAs.
It retains the existing runner label, container, Python/Rust/build settings,
locked installation, artifact verification, and revision-local workers.
All builds finish before comparisons start. Each comparison alternates base/head
on one CPU for five paired rounds across all 12 current cases. Separate comparison
jobs use the normal CI runner allocation; CPU affinity does not establish exclusive
physical-host ownership.

Ordinary comparison mode is used, including for the control. Qualification mode
and its separate two-second duration target are not enabled. A regression is
greater than 10% and 100 ms in at least four of five rounds. There is no workload
resizing, threshold change, or selective repetition to obtain detection.

Matrix failure does not cancel other entries. Comparisons attempt to run even
when another build fails; a missing required artifact remains an explicit failure.
The benchmark's nonzero exit code is preserved. Summaries, timing records, logs,
wheel provenance, and each candidate patch are retained for 30 days.

## Local validation

Rust tests used installed Rust 1.96.1 and locked dependencies. CI release builds
use Rust 1.99.0, exactly as the pinned workflow does.

- Without #295: 49 embedded-Python tests and 17 timing-evidence tests passed.
- Without #321: 686 tests matching `engine::` passed, including SGLang rollback.
- Without #386: 27 replay component tests and 167 vLLM scheduler tests passed.
- Existing simulation performance gate tests: 44 passed.
- Rust formatting, diff checks, and actionlint passed.

## Interpretation

The unchanged-revision control must pass before drawing a firm detection
conclusion. Each variant is reported as detected, not detected, or invalid/incomplete.
All 12 cases and all five rounds must be present for a complete comparison.
Invalid or missing measurements are not a pass or evidence that coverage missed
the regression. A valid pass means this fixed suite did not detect the restored
cost; it does not prove the original regression was absent.

This measures current CI coverage on current code with restored slow paths, not
the original historical revision pairs. There is no full historical workload or
output-equivalence campaign in this experiment.
