# Scenario Comparison integration evidence

These are partial-data and synthetic API checks, not a complete qualified HF
campaign or a claim about serving accuracy across the dataset. Browser fixtures
are synthetic and are not included in the production publication artifacts.

## Sources and runtime

- HF dataset: `nvidia/aisimulate-fpm-dataset`, immutable revision
  `5487a4599a7fbc012c07bcd3699754bdf4a8bef7` from the verified local Hub cache.
- Configuration: `data/moonshotai--Kimi-K3/gb300/vllm/0.1.dev19262/tep8`.
  The real-data smoke uses the first 180 seconds: 10,270 observations spanning
  179,987 ms. It preserves source order, ranks, timestamps, and targets.
- Native extension built from PR639 revision
  `f0442fee83bdbeb791fbeb1bf8710f45754be5bd`; SHA-256
  `7d7fda4edc843d989b178b610c4363e13b189e5201d22614a3ba5d135a0b111a`.
  The integration branch starts at `e93ad9eed342adc599d6d825c3806987f9050da0`,
  which additionally fixes checkpoint I/O. This is not an exact-head wheel
  qualification. CI builds and verifies the evaluated branch's exact wheel.
- Python 3.12.13, scikit-learn 1.9.1; native, OpenMP, and BLAS thread limits are
  one. Reported elapsed costs are local smoke measurements, not CI budgets.

## Checks

`native-matrix-smoke.json` contains all seven methods, all five cases, and the
nine B/B+C split settings. Unsupported combinations remain null. The complete
matrix passed the publication row contract against independently counted source
populations. Diagnostics are omitted from this evidence file after validation;
the full serialized row size is recorded separately.

Strict C admits 35 observations in this window, including its bootstrap
observations. There are no immediate warmup calls. Each admitted target becomes
available after its prediction is scored. Learned C models need 20 observations
for a trainable store; cold-start misses remain in coverage. At the default 50%
split, B+C admits 16 observations after its training prefix.

The original-A mixed fallback is exercised by GBDT A and GBDT/hybrid/selector
A+C. Kimi has 60 A-backed prediction failures and one A+C update failure where
the measured prior has no usable neighbor and the native SOL path has no
`context_kda_conv1d` implementation. These are retained as errors. Ordinary typed
interpolation domain misses remain unavailable. Neither class is removed from
the measured denominator; MAPE uses successful predictions only.

The same runtime limitations also affect standalone interpolation: warmup-on
returns 651 predictions (6.34% coverage, 1.80% MAPE) with 9,426 errors;
warmup-off returns 9,773 predictions (95.16% coverage, 1.45% MAPE) with 304
errors. The remaining observations are ordinary unavailable queries. These
figures describe this selected library/window, not all Kimi measurements.

`mixed-continuation.json` is a separate synthetic direct-profile API check.
GBDT and selector append one mixed forest after 20 admitted labels, with zero
paired comparisons immediately after append. Nine later labels produce nine
paired comparisons. The GBDT quality gate resolves native defaults
`alpha=0.05, min_comparisons=5`. Hybrid creates no new forest. This check establishes
causal state transitions and frozen-forest preservation, not measured accuracy.

The Python evaluator/publication/Pages tests passed 742 tests plus 138 subtests
(22 skipped for optional runtime dependencies). Twelve workflow tests and both
dashboard browser checks passed. Native adapter and seed checks were also run
with the extension above. Later narrowly scoped checks are recorded in the PR.

## Reproduce

Use this PR's checkout with its evaluator requirements and an installed AISim
wheel supporting PR639. `--native-library` is optional; it records the explicitly
provided extension hash when testing an external build.

```bash
PYTHONPATH=.:python/aisimulate/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
python benchmarks/evidence/fpm_scenario_publication_20261029/smoke.py \
  --hf-snapshot /path/to/hub/snapshots/5487a4599a7fbc012c07bcd3699754bdf4a8bef7 \
  --configuration data/moonshotai--Kimi-K3/gb300/vllm/0.1.dev19262/tep8 \
  --output /tmp/native-matrix-smoke.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
python benchmarks/evidence/fpm_scenario_publication_20261029/mixed_continuation.py \
  --output /tmp/mixed-continuation.json
```

The public page is deployed by the existing main-branch Pages workflow after
PR639 and this change merge. A full campaign must finish before scenario values
become available. The page rejects legacy paged-C artifacts and mismatched
snapshots; this partial evidence cannot qualify for publication.
