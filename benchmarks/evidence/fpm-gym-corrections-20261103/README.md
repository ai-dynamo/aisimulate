<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FPM Gym seed, metadata and spline qualification

These CPU replays compare the corrected Gym with the published results on the
same measured populations. The original-A mixed fallback remains enabled for A
and A+C. No new GPU measurements or online Gym deployment are included.

## Scope and provenance

The fixes make Gym training consume the canonical native loader's effective
`fake_fallback` targets, recover incomplete explicitly reconstructed MiniMax
sidecars after verifying the source contracts, retain valid mixed bootstrap
labels when the prior returns a typed missing-data error, and remove Parquet
serialization-order bias from offline spline initialization.

Inputs come from
[`nvidia/aisimulate-fpm-dataset`](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/03e006de4b7c96276cc94f62b102b1b1275d24ac),
revision `03e006de4b7c96276cc94f62b102b1b1275d24ac`.
Configuration membership and scoring-population hashes match the published
snapshot. MAPE averages successful predictions; coverage is shown separately.
Marked fake-KV targets are estimates from the existing native extrapolation
policy. Genuine measured targets and source data files are unchanged. Unknown
producer revisions remain unknown.

[`accepted-results.json`](accepted-results.json) indexes all 16 accepted
method/configuration evaluations, with result hashes and scoring-population
hashes. Its referenced JSON files include model diagnostics and provenance;
local filesystem locations are omitted. [`final-source-manifest.json`](final-source-manifest.json)
binds the 28 changed implementation, documentation and test files to their
SHA-256 hashes and base commit `9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`.
[`publication-build-receipt.json`](publication-build-receipt.json) identifies the
release wheel and native library by hash; binaries are not committed.

## Accepted results

All A+C rows use the strict five-second adaptation protocol. Scenario B controls
retain their stated training fraction. New publications use protocol 14; the
viewer continues to distinguish and accept historical protocol 13.

| Case / method | Scenario | Before MAPE | After MAPE | Before coverage | After coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `minimax-tep4-warmup` | A | preparation failure | 2.364907% | 0/286 | 286/286 |
| `minimax-tep4-gbdt` | A | preparation failure | 2.851255% | 0/286 | 286/286 |
| `minimax-tep4-spline` | A | 52.931746% | 12.368270% | 286/286 | 286/286 |
| `minimax-tp4-gbdt` | A | 7.530548% | 5.642960% | 157006/157007 | 157006/157007 |
| `minimax-tp4-spline` | A | 29.935397% | 9.095725% | 153270/157007 | 153270/157007 |
| `glm-b50-spline` | B (50% train) | 8.553690% | 8.553690% | 2532/45664 | 2532/45664 |
| `minimax-b60-spline` | B (60% train) | 16.526334% | 16.526334% | 25739/79604 | 25739/79604 |
| `kimi-a-gbdt` | A | 4.462161% | 4.462161% | 282217/284398 | 282217/284398 |
| `kimi-ac-gbdt` | A+C | 1.835707% | 1.835707% | 282217/284398 | 282217/284398 |
| `kimi-ac-selector` | A+C | 1.857038% | 1.857038% | 282217/284398 | 282217/284398 |
| `kimi-regression-a-spline` | A | 19.250602% | 16.672593% | 282217/284398 | 282217/284398 |
| `glm-regression-a-spline` | A | 7.766798% | 4.899699% | 86430/91330 | 86430/91330 |
| `minimax-tep4-linear` | A | 27.785984% | 25.914232% | 286/286 | 286/286 |
| `minimax-tp4-linear` | A | 30.881671% | 24.616183% | 153270/157007 | 153270/157007 |
| `kimi-regression-a-linear` | A | 11.444988% | 11.444988% | 282217/284398 | 282217/284398 |
| `glm-regression-a-linear` | A | 26.684445% | 16.745136% | 86430/91330 | 86430/91330 |

`minimax-tep4` denotes H200 TEP4 snapshot `aisim-commit-unknown-4baad028`;
`minimax-tp4` is the distinct H200 pure-TP4 configuration. Exact configuration
identities are in the index. Kimi's spline improvement isolates ordering because
that profile has no marked targets. GLM's spline improvement combines target
normalization and ordering. Linear changes come only from target normalization.

## Validation

Local qualification used Python 3.12 on macOS ARM64 with the release native
library identified in the build receipt. Commands below use repository-relative
paths; run them from the repository root with the development environment and
that revision's native library installed.

```bash
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false \
  python/aisimulate/.venv/bin/python -m pytest \
  -c python/aisimulate/pytest.ini -p no:timeout tests/fpm_accuracy

python/aisimulate/.venv/bin/python -m pytest \
  -c python/aisimulate/pytest.ini -p no:timeout \
  python/aisimulate/tests/unit/sdk/test_fpm_profile.py \
  python/aisimulate/tests/unit/sdk/test_fpm_spline.py

cargo test -p aisimulate-core --lib perfmodel::fpm::regression
python/aisimulate/.venv/bin/python scripts/pages/check_fpm_scenarios_browser.py
cargo fmt --all -- --check
git diff --check
```

- Complete Gym suite: **757 passed** on the final source.
- Native profile/spline SDK tests: **286 passed** on the final native build.
- Rust regression tests: **63 passed**, including the finite zero-slope optimum.
- Protocol 14 and historical protocol 13 synthetic browser fixtures: passed.
- Ruff lint/format, Rust formatting and whitespace checks: passed.
- External Rust public-API compile fixture: passed. The fixture was copied to a
  temporary standalone workspace because this checkout was nested in another
  Cargo workspace.
- An earlier combined run passed **1,136 tests**, including the Gym, mixed
  composition/quality/selector, original-A manifest, snapshot and single-oracle
  suites. The final spline-only ordering scope and compact-key changes were
  followed by the complete 757-test Gym rerun above; native source was unchanged.

The temporary Git environment above applies only to repository fixtures created
by tests. It does not alter developer commit-signing settings. `-p no:timeout`
avoids the macOS SIGALRM crash-reporter behavior. Browser checks require the
project's browser-test dependencies and Chromium. These are local results;
GitHub Fast CI, Full CI and required reviews are separate publication gates.

Boundary tests exercise raw-mode preservation, unchanged genuine measurements,
normalization report consistency, corrupt or conflicting metadata, validation
beyond one scan batch, invalid bootstrap labels, typed missing-data failures,
duplicate ordering keys, source-row permutations and historical viewer schemas.
Exact normalization expectations use the existing hand-computed same-cell,
same-batch real-KV extrapolation fixtures. Zero-slope assertions check a finite
constrained optimum and preserve the existing store-readiness contract.

## Cost and remaining limits

Normalization performs extra preparation only for profiles with relevant marked
rows. Recovery verifies file hashes and scans contract columns in bounded
batches. These changes add no prediction-loop scan. Spline ordering adds one
offline O(N log N) sort and O(N) shuffle per prepared seed set; it leaves native
capacity, online arrival order and GBDT/linear seed order unchanged.

Kimi A+C retains 17 more valid mixed labels, increasing pending rows from 2 to 19
in the affected worker. The minimum remains 20, so this does not create a new
mixed store or change aggregate prediction coverage. The 17 prior errors remain
visible. Missing KDA timing data is still unresolved.

The native spline readiness guard is preserved. GLM B50 and MiniMax B200 B60
controls retain their original coverage and MAPE. A ready spline component does
not by itself make a store ready when its linear companion has no usable fit.
The fix addresses offline seed order, not all limits of a finite retained-sample
regression model. Evaluation populations and GBDT hyperparameters are unchanged.
