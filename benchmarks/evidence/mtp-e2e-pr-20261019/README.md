# Public E2E MTP integration validation

- Measurement source: [InferenceX September 28 snapshot](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-28).
- Selection: 2,281 unchanged STP points plus 1,039 MTP candidates, 3,320 total.
  `selection.json` reconciles exclusions against the complete measurement table.
- Native CPU replay smoke: benchmark 415317, 80/80 requests completed, explicit
  op-level MTP depth 2. Acceptance comes from rounded SGLang interval means and
  is labeled estimated. `validation.json` records input, log, native, and
  performance-data identities plus paired latencies. This single point validates
  integration; it is not a refreshed full-cohort MAPE estimate.
- Validation: source/campaign/export/Pages regression tests; Node UI/workflow
  tests; Playwright Chromium checks including MTP filtering and reload persistence;
  Ruff checks. See the PR for final test counts.

Reproduction uses the normal campaign, with `.github/e2e-accuracy-dataset.json`
(policy `gym-resolved-config-v3`), an installed wheel from the evaluated revision,
`--configuration-mode estimated`, and `--fetch-runtime-evidence`. Select and resolve
inputs with `source.cohort.select_points(..., include_mtp=True)` and
`run_e2e_accuracy.resolve_points(..., allow_estimated_defaults=True)`. Run an
individual saved point through:

```sh
python scripts/e2e_accuracy/run_e2e_accuracy.py point < smoke-resolved.json
```

Local raw inputs, downloaded logs, source caches, wheels, and smoke outputs remain
ignored beside this report. No new GPU collection or model calibration was used.
The public page receives fresh MTP results after the merged campaign completes;
this change does not publish the older standalone evaluation as current data.
