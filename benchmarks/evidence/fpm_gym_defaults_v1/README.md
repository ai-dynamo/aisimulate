<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Learned FPM defaults: Gym accuracy comparison

This evidence compares PR #534's defaults with the defaults introduced by
`08f0a2d2e2f9685369701d3c485368ad53e0b111`. It replays existing measured
observations through both native libraries. It does not collect new serving
measurements, refit the offline seeds, or tune settings against these results.

## Results

All 42 runs completed with zero prediction or tuning errors. Online and
selector MAPE improved in each selected cell when compared within the same
model class. Hybrid controls show the small variation expected from sampling.

Values below are MAPE percentages; brackets give the min–max across repeats.
Delta is after minus before, in percentage points. The detailed
[comparison](comparison.json) also includes every workload and repeat.

| Cell | Model | Runs per revision | Before mean [range] | After mean [range] | Delta (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| Kimi A+C | online | 1 | 2.325395 | 1.835707 | -0.489688 |
| Kimi A+C | selector | 3 | 2.111510 [2.111097–2.112338] | 1.858805 [1.858052–1.859368] | -0.252705 |
| Kimi A+C | hybrid | 3 | 2.098299 [2.096205–2.100266] | 2.098080 [2.095611–2.099845] | -0.000219 |
| Kimi B+C (50%) | online | 1 | 1.435335 | 1.361896 | -0.073439 |
| Kimi B+C (50%) | selector | 3 | 1.367574 [1.367574–1.367574] | 1.227428 [1.227405–1.227464] | -0.140145 |
| Kimi B+C (50%) | hybrid | 3 | 1.356395 [1.356237–1.356710] | 1.356395 [1.356237–1.356710] | 0.000000 |
| GLM B+C (50%) | online | 1 | 6.698535 | 5.894693 | -0.803842 |
| GLM B+C (50%) | selector | 3 | 4.353160 [4.353160–4.353160] | 4.134191 [4.134191–4.134191] | -0.218968 |
| GLM B+C (50%) | hybrid | 3 | 4.309745 [4.306982–4.311127] | 4.311127 [4.311127–4.311127] | +0.001382 |

Coverage is identical across revisions, model classes and repeats:

| Cell | Successful / eligible | Coverage |
| --- | ---: | ---: |
| Kimi A+C | 282,217 / 284,398 | 99.233% |
| Kimi B+C (50%) | 142,199 / 142,199 | 100.000% |
| GLM B+C (50%) | 45,665 / 45,665 | 100.000% |

Kimi A+C retains 2,181 unavailable mixed-workload predictions because the
offline seed lacks that store. Online receives only 19 eligible warmup
admissions for it, below the 20-row bootstrap threshold. No missing prediction
is scored as zero error or silently removed from the coverage denominator.

**Changing the implicit model choice has a different outcome from changing
online fitting controls.** The old implicit frozen/hybrid behavior versus the
new implicit online behavior is:

| Cell | Old frozen/hybrid default | New online default | Delta (pp) |
| --- | ---: | ---: | ---: |
| Kimi A+C | 2.098299% | 1.835707% | -0.262592 |
| Kimi B+C (50%) | 1.356395% | 1.361896% | +0.005502 |
| GLM B+C (50%) | 4.309745% | 5.894693% | +1.584947 |

Thus the requested mode change improves Kimi A+C, slightly worsens Kimi B+C,
and worsens GLM B+C. The controlled online/selector gains above do not imply
that choosing online is always more accurate than choosing frozen hybrid.
The old behavior is represented by the frozen artifact control; source-load
equivalence is checked separately below.

## Scope and metric

The comparison uses the local Gym's `scripts/fpm_accuracy/scenarios.py`
(`score_scenario`) and `scripts/fpm_accuracy/evaluate.py` (`Metric`) directly.
For each successful prediction, with prediction and measured latency in
milliseconds:

```text
APE_i (%) = 100 * abs(predicted_i - measured_i) / measured_i
MAPE (%) = sum(APE_i) / successful_prediction_count
coverage (%) = 100 * successful_prediction_count / eligible_observation_count
```

There is no clipping. Missing predictions and errors reduce coverage rather
than contributing a zero error. The `all` row pools successful predictions
across workload kinds within a cell; it does not average workload MAPEs. The
Gym has no cross-cell aggregate objective, and this evidence reports none.

The selected cells are Kimi K3 A+C and the default 50% B+C split for Kimi K3
and GLM-5.2-NVFP4. B+C trains from the chronological prefix and scores the
remaining suffix. A+C uses the Gym's separate self-benchmark seed and scores
the full runtime stream. Prediction precedes feedback; the Gym admits the first
observation at or after each new five-second tick. Intermediate
labels are not buffered for later admission. `update_every` independently
counts admitted observations per native store. The runner does not flush a
partial fitting batch at the end of the stream.

Cold-start C is excluded: the local Gym uses `append_learned_checkpoint`,
which PR #534 does not expose. Selector and hybrid use a complete worker
model; online retains the Gym's per-worker, per-native-store partition.
All use the public `best_available` / estimation / tuning APIs. This preserves
the native selector's expert state and hybrid calibration sharing.

The three cells contain 472,262 eligible scoring observations per mode and
revision (284,398 Kimi A+C, 142,199 Kimi B+C, 45,665 GLM B+C). The two Kimi
cells overlap; their counts are not independent observations. These selected
deployments do not establish accuracy for other models or configurations.
They come from the Gym corpus used to choose the recipes, rather than an
independent holdout corpus.

## Comparison arms

- **Online:** before explicitly selects `adaptation: online` and disables
  correction, then omits every online fitting control. After supplies only
  the checkpoint and uses the new online/correction defaults. This isolates
  the changed online recipe.
- **Selector:** both revisions explicitly select `causal_selector`; all
  online, selector and correction controls are omitted.
- **Frozen hybrid control:** before uses the old implicit frozen default;
  after explicitly selects `frozen`. Both omit correction controls. This
  checks the unchanged standalone hybrid recipe with the same offline forest.

A literal checkpoint-only configuration previously selected frozen hybrid
and now selects online. The **before hybrid** arm loads the identical frozen
forest through its inference artifact; it represents the old frozen/hybrid
behavior, but does not execute a checkpoint-only request. The online-to-online
pair explicitly selects online before the change.
This frozen control uses the same 150-tree seed as the streaming arms; it is
not the Gym's separately trained A-only/B-only forest recipe.

An additional baseline smoke comparison loaded the same seed through
`artifact_path` and `checkpoint_path`: all 256 predictions matched exactly
while admitting 32 feedback observations. Forests, features and targets
remained identical; the checkpoint path additionally attached verified
identity metadata. See [the validation receipt](frozen_source_equivalence.json).
The source loaders converge on the same frozen predictor and correction
implementation. The smoke stays below the 64-row correction capacity to
avoid eviction tie randomness.

All arms use `estimation_mode: fpm_learned`, `fallback_policy: deny`, identical
ordered scoring populations, and identical offline forests: 150 trees, at
most 15 leaves, learning rate 0.2, and the 17-feature `fpm5` ABI. The runner
verifies the seed manifests, forests, fitted rows, feature values, targets,
row provenance and deployment identity before loading them.

Online runs once per revision and cell. Selector and hybrid run in three
independent processes per revision and cell; their results report the mean
and min–max of the three cell MAPEs. This repeat summary does not introduce
cross-cell averaging. Fixed calibration breaks equal-size eviction ties
through randomized hash-map iteration, so calibration-based predictions can
vary slightly even with identical forests and observations. The unchanged
hybrid arm is a control for that variation.

## Provenance and reproduction

Baseline: PR #534 head `0c53fab44dabc171e56bfbddf8361b154b57ee06`.
The preserved native library was built at
`747458b0828e52bcbcc9b3546b215149c70e426f`; `git diff --quiet` confirms
`crates/core`, `Cargo.toml`, and `Cargo.lock` are identical at those revisions.
Its SHA-256 is
`a7a414e773426a448325290c8c324aacb24faeb6e8034994d0f83cdcb9772316`.

Candidate: `08f0a2d2e2f9685369701d3c485368ad53e0b111`, based on main
`d2dc433e`. Its native library SHA-256 is
`7d87627741dca4eea8d05fd90a23294f983096662f1208a55956f3f20b549c2c`.
The learned estimator, online, selector and combined implementation sources
on main were unchanged from PR #534 before applying this fix.

The Gym is the local development worktree based on
`8ead44198c4e8b7674c92a16a71a604ff230b40f`, with uncommitted scenario code.
Per-file source hashes are recorded in every result; the base commit alone
does not identify these local changes. Prepared inputs come from
`fpm_gym_selector_sweep_v2/prepared`; immutable checkpoints come from the
local Gym's `fpm-gym-selector-w8-hybrid64/seed-checkpoints` evidence.
Raw cases, checkpoint assets and the local Gym source are not packaged in
this evidence directory. Reproduction requires those hash-matching inputs.

`compare_defaults.py worker --help` describes the explicit source, SDK,
native-library and input paths. Each arm/cell runs in a fresh Python process
loading the requested native library. `summarize` pairs result files and
checks matching training/scoring populations, seed receipts and admissions.
Results preserve workload metrics, raw APE sums, coverage, error counts,
normalized runtime configuration, source hashes and admission receipts.
The run used Python 3.12.13, NumPy 2.5.4, PyArrow 25.0.1,
scikit-learn 1.9.1 and huggingface_hub 1.32.0, with numerical-library thread
counts set to one. The candidate SDK transport was shared by both native
revisions; fitting and default resolution execute in the selected Rust library.

For example, with the hash-matching local inputs available:

```sh
python compare_defaults.py worker \
  --gym-source "$GYM_SOURCE" --sdk-source "$CANDIDATE_SDK_SOURCE" \
  --prepared-root "$PREPARED_EVIDENCE_ROOT" --seed-root "$SEED_ROOT" \
  --native "$BASELINE_NATIVE" \
  --revision 0c53fab44dabc171e56bfbddf8361b154b57ee06 \
  --variant before --mode online --cell glm_bc_50 --replica 0 \
  --output results/before-online-glm_bc_50-r0.json
python compare_defaults.py summarize --output comparison.json results/*.json
```

Repeat the worker command for both revisions and all three cells, with one
online run and three selector/hybrid runs for each. The complete comparison
contains 42 worker results.

## Correctness validation

On candidate `08f0a2d2`, 195 native SDK/API tests passed, including omitted
defaults versus explicit Gym recipes, frozen artifact loading, explicit
overrides, checkpoint continuation and configuration error paths:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python/aisimulate/src \
  python -m pytest -c python/aisimulate/pytest.ini -p no:timeout \
  python/aisimulate/tests/unit/sdk/test_fpm_selector_integration.py \
  python/aisimulate/tests/unit/sdk/test_fpm_online_export.py \
  python/aisimulate/tests/unit/sdk/test_fpm_online_integration.py \
  python/aisimulate/tests/unit/sdk/test_fpm_online_structural_reference.py \
  python/aisimulate/tests/unit/sdk/test_fpm_learned.py \
  python/aisimulate/tests/cross_package/test_core_public_api.py -q
```

Before rebasing onto main, the implementation worktree also passed 2,228
Rust core tests (one ignored), 16 public Rust API tests, 904 CLI/Sweeper/runner
tests and 389 native parity tests. These broader suites were not rerun after
the rebase; the native library was rebuilt and the 195 SDK/API tests above
were rerun. Independent code review of `08f0a2d2` found no blocking issues.
