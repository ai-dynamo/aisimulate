# GLM-5.3-Flash canonical dataset integration

This tool implements the next local step after AISimulate's
`collector.fpm_forward.glm53flash_publication` stage succeeds. It contains
original NVIDIA Apache-2.0 code. No real GLM dataset has been imported or uploaded;
the tests use synthetic inputs under pytest temporary `TEST_ONLY_*` directories.
Passing these tests is an integrity check, not GPU or prediction acceptance.

The inspected dataset base is
`nvidia/aisimulate-fpm-dataset@c60c25294145fe11da2d24eba8f703191d18c684`.
Its [scripts/manage_dataset.py API](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/blob/c60c25294145fe11da2d24eba8f703191d18c684/scripts/manage_dataset.py)
and manifest/catalog formats were read to implement interoperability. That file's
SHA256 is `9686df97a10496e06e346ee2749ba6e823c24dff18f69cf6f9be72b824d11b52`.
Upstream dataset code and source fragments are not included in this tool or
copied into AISimulate. The importer copies an entire user-provided local
snapshot to a **new** destination. It applies AST-selected routing edits only
to that external destination's validator and adds the original `glm53flash.py`
policy module there. Source fragments are obtained only from the external
snapshot at runtime; unrelated bytes and copyright/license comments remain
unchanged. No upstream license is assumed or invented. Both the pinned file
hash and structural hook counts must match before applying edits. A changed
validator requires a new source inspection and pin.

## What it checks

- Exactly eight accepted configurations and sixteen prefill/decode cells, fixed
  model revisions, all required context groups, complete predictions, recomputed
  per-phase MAPE no greater than 10%, and the installed consumer payload identity.
- Every partition equals its recorded source Arrow rows, including schema
  metadata. The union of partition indices must include every source row once.
  Source Parquet and metadata bytes remain archived; canonical calibration
  Parquet is a byte copy of the already validated partition.
- The original acceptance input manifest is retained and checked twice: its
  exact file bytes match the stage receipt, and its canonical JSON digest
  matches the acceptance report. These are deliberately different identities.
- Exact accepted native runtime versions match each partition for calibration
  and holdout. The canonical path normalizes `+` and case using the dataset's
  existing API; the actual backend version remains unchanged in all identity
  fields and calibration rows.
- KV cache precision must be the pinned `fp8` policy in every row; `auto` or
  BF16 cache rows cannot enter this campaign. Actual accepted precision is
  preserved in source and partition identity.
- All calibration/holdout phase roles have external evidence locations with
  SHA256 and byte counts. URIs must have no embedded credentials or query
  tokens. External raw archives are referenced; this tool does not fetch them
  or attest that storage access remains available.
- The full pre-existing dataset and new canonical manifests/catalogs pass the
  dataset's existing validator with explicit GLM routing. Old campaign policy
  and counts remain enforced.

This policy validates a publication stage and its exact evidence identity. It
does not rerun native GPU collection or independently regenerate predictions;
that work remains the stage exporter's fresh acceptance gate. It also does not
establish trust in arbitrary user-written reports. Review the actual stage and
archive receipts before committing them.

## Run after the formal stage exists

```sh
python tools/glm53flash_hf/import_glm53flash.py \
  --base /absolute/path/to/fresh-immutable-dataset-snapshot \
  --stage /absolute/path/to/accepted-glm-stage \
  --destination /absolute/path/to/new-canonical-dataset \
  --external-receipts /absolute/path/to/verified-raw-archive-receipts.json \
  --source-revision ACTUAL_40_CHARACTER_AISIMULATE_COMMIT \
  --evidence-date ACTUAL_YYYY_MM_DD
```

External receipts are a JSON list with fields `backend`, `weight_quantization`
(`fp8` or `nvfp4`), `tp`, `phase`, `role` (`calibration` or `holdout`), `uri`,
`sha256`, and `bytes`. At least one archive receipt must cover every one of the
32 phase/role combinations. A shared archive can appear in multiple role
records when it actually contains each referenced role. Large raw and failed
artifacts should remain in the archive; do not claim a compact metric report is
the raw archive. Download/extract and verify the archived originals separately.

The new dataset contains the entire original publication stage and raw archive
receipts under `campaigns/glm53flash-pr324/<stage-sha256>/`, and eight normal
canonical `data/.../manifest.json` leaves. Original stage metadata is preserved;
canonical sidecars add only import policy and an explicit provenance receipt.
Empty measurement manifests are intentional: calibration is not recast as
independent measurement truth. Native accuracy reports stay in the separately
identified stage validation folder and retain their timing boundaries.

`import-result.json` says `CANONICAL_LOCAL_NOT_PUBLISHED`. Before the authorized
HF write, re-read the Hub's current parent commit, preserve concurrent changes,
review the exact local upload diff, and upload the policy/module/data/catalogs
together. Then obtain the real immutable HF commit, update all eight consumer
profiles in PR #324, and verify installed-wheel/offline predictions. This tool
does not perform those remaining actions, create an HF PR, or invent a pin.

Existing GLM canonical leaves cause a hard error. A later revision needs an
explicit current-to-history migration; silently overwriting accepted history
is not implemented.

## Generate consumer profiles after the real Hub commit

`profile.py` requires the accepted stage, the canonical dataset copy, its explicit
`import-result.json`, and the actual 40-character commit returned by HF after
publication. It confirms that commit through the Hub API, then reads and hashes
the pinned consumer files and evidence receipts. A syntactically valid SHA alone
does not establish a published pin. There is no offline or test-admission override
for this production command, and it performs no HF writes.

```sh
python tools/glm53flash_hf/profile.py \
  --stage /absolute/path/to/accepted-glm-stage \
  --dataset /absolute/path/to/published-canonical-dataset-copy \
  --import-result /absolute/path/to/published-canonical-dataset-copy/campaigns/glm53flash-pr324/STAGE_SHA/import-result.json \
  --revision ACTUAL_HF_40_CHARACTER_COMMIT \
  --destination /absolute/path/to/new-profile-output/ACTUAL_HF_40_CHARACTER_COMMIT
```

The output `hf_dataset.json` uses the existing SDK format and contains exactly
eight serving profiles named `gb300-<backend>-<precision>-tp<tp>-full`. Every
profile retains the exact checkpoint revision, execution identity and source row
lineage. Its Parquet and sidecar paths come from the canonical manifests. Its
`gb300.yaml` comes from the accepted consumer data receipts, including the
original source path and byte hash; no repository default is substituted.
Prefill and decode must have used identical YAML bytes. Targets follow that
YAML's actual relative `data_dir` and the exact runtime version. For example,
canonical HF paths can contain `0.30.0-glm53kpool...` while the materialized
consumer directory contains `0.30.0+glm53kpool...`.

`generation.json` records verified remote file hashes and says
`PIN_VERIFIED_OFFLINE_VALIDATION_PENDING`. Check the generated pin into the FPM
PR, build/install its wheel, materialize each profile with
`aisimulate_core.sdk.fpm_dataset.materialize_fpm_profile`, reproduce the accepted
predictions, then repeat with `local_files_only=True`. Generating and verifying
the pin does not replace those installed-consumer and offline accuracy checks.
Explicit synthetic/test markers in stage, report, original input or rows are
rejected before a production profile can be written.

## Tests

```sh
GLM_TEST_BASE=/absolute/path/to/immutable-baseline \
PYTHONPATH=/absolute/path/to/aisimulate/python/aisimulate \
python -m pytest -q tests/unit/tools/test_glm53flash_hf_publication.py
```

The baseline integration test preserves all baseline source bytes, constructs
synthetic temporary leaves, runs the complete canonical validator, verifies
candidate-version path normalization, and rejects changed catalog identity.
Unit tests reject missing/duplicate phase cells, invented metrics, incomplete
predictions, missing consumer identity, archive evidence gaps, unsafe URIs,
changed source bytes, escaping symlinks, and mismatched accepted runtimes.
The complete baseline integration test is skipped unless `GLM_TEST_BASE` is
explicitly set. Normal CI runs the other local unit tests without GPU access,
a dataset download, or an HF connection.
**Never upload test output.**

The profile contract tests are
`tests/unit/tools/test_glm53flash_hf_profiles.py`. They use an explicitly mocked
Hub, test-only bypasses installed through pytest monkeypatch, and isolated test
outputs. They check all eight SDK materializations and subsequent offline loads;
they do not claim native prediction or real GLM publication acceptance.
