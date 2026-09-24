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
policy modules there. Source fragments are obtained only from the external
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

The importer accepts only `glm53flash_bound_raw_evidence_v1` records produced
by the archive binding workflow below. There must be **exactly 32** records:
one per backend/precision/TP/phase/role. Labels-only receipts from the low-level
archive helper are deliberately insufficient. Binding verifies:

- Exact stage, original input-manifest and accepted report content SHA256.
- Frozen plan file content SHA256 and selected cell/phase/role. Sharded roles
  require the exact accepted child-cell union and every child plan.
- The exact accepted native file set and every SHA256 under each recorded
  `raw_root`, mapped to an explicit prefix in the archive inventory. Missing,
  extra or changed files under that accepted root fail; other files (including
  failed attempts) remain in the complete campaign inventory.
- `consumer_sources`: every original `consumer_data` path and SHA matches the
  accepted prediction receipts and `stage.sources[].original_consumer_paths`;
  the archived stage source bytes are rehashed. Consumer files remain in the
  stage; they are not falsely represented as native raw files.
- The complete inventory, archive input manifest, archive verification receipt,
  byte counts, verified method and source recheck, with immutable small-file
  receipts. The importer preserves all these files, and profile generation
  verifies them at the actual immutable Hub revision.

## Archive the closed campaigns on the storage host

`raw_archive.py`, `raw_campaign.py` and `external_control.py` are original Apache-2.0 implementations
using only Python's standard library. They can run with Python 3.12 on the
remote Lustre host without importing AISimulate, torch, Arrow or a GPU runtime.
These are repository maintenance tools, not installed SDK entry points. For
remote use, deploy all three modules together in a new
versioned bundle directory, retain the Apache-2.0 license, and record the exact
AISimulate source commit and SHA256 of every file before transfer. Recheck all
hashes on the destination and invoke `python3.12 /bundle/raw_campaign.py`; do not
add a generic `tools` package to the SDK wheel. The final dataset importer still
runs the full accepted-stage/Arrow validator.
The earlier host-local `glm53flash-raw-archive-v1` helper is the original source
of `raw_archive.py`; no external implementation was copied.

First stop writers and preserve failed attempts. On the host where the original
acceptance manifest's paths are valid, create a new plan:

```sh
python3.12 tools/glm53flash_hf/raw_campaign.py plan \
  --stage /lustre/evidence/accepted-stage \
  --manifest-base /lustre/campaign/original-manifest-directory \
  --output /lustre/evidence/archive-plan.json
```

`manifest-base` is the directory against which the original acceptance input
manifest resolved relative `plan.path`, `shard_manifest.path`, and `raw_root`.
Absolute original paths remain absolute. No alternate mount or path prefix is
guessed. Each of the 32 jobs contains its accepted raw roots and an entry index.
Fill only `source_root` and `uri`: the former must be an explicit absolute path
to the **entire closed campaign directory, including failed attempts**, enclosing
all its accepted raw roots; the latter must be a stable credential-free `ssh://`,
`s3://`, or `https://` archive location. Source roots and URIs are deliberately
not inferred. For example, a URI can be
`ssh://ocijhb/lustre/evidence/ROLE/campaign.tar.gz`. URI availability is not
checked, and this tool performs no upload or remote write.

The production sharder places both phases beneath one parent. For example,
with `artifact_root=/lustre/campaign/vllm-fp8-tp2-calibration/artifacts` and
`checkpoint_dir=/lustre/campaign/vllm-fp8-tp2-calibration/checkpoints`, its layout is:

```text
vllm-fp8-tp2-calibration/             # explicit source_root
  checkpoints/                      # original resume/failure state
  artifacts/<parent-plan-sha16>/
    collection-plan.json
    shard-manifest.json
    plans/<child-cell-id>.json
    shards/<child-plan-sha16>/cells/<child-cell-id>/
      raw/                          # one accepted child root
      attempts/<attempt-id>/        # preserved superseded/failed attempts
```

Assign that same `source_root` and same URI to the parent's prefill and decode
jobs. `archive` derives the exact shared label set from the 32-job plan. Archive
that physical parent once, then map both logical labels to its bundle. Separate
calibration/holdout parent directories normally require 16 physical archives
for all eight deployments. No directory or URI is inferred; one URI may not name
different source roots. An archive's attested label set must exactly equal all
bound records referencing its URI, SHA and byte size.

Each distinct physical archive can run independently, making retries explicit
new outputs:

```sh
python3.12 tools/glm53flash_hf/raw_campaign.py archive \
  --stage /lustre/evidence/accepted-stage \
  --plan /lustre/evidence/archive-plan.json \
  --label sglang-fp8-2-decode-calibration \
  --output /lustre/evidence/archive-bundles/sglang-fp8-2-calibration
```

The output parent must already exist and the output itself must be new and
outside its source. Run once for each distinct source-root/URI pair, using one
of its labels (the five fields `backend`, `weight_quantization`, `tp`, `phase`,
`role` joined with `-`). Then write an explicit `archive-bundles.json` object
mapping all 32 labels to their successful absolute bundle directories; shared
labels must point to the same verified bundle. A failed run's output is preserved;
choose a new output directory on retry and point the map to that verified run.

```sh
python3.12 tools/glm53flash_hf/raw_campaign.py bind \
  --stage /lustre/evidence/accepted-stage \
  --plan /lustre/evidence/archive-plan.json \
  --bundles /lustre/evidence/archive-bundles.json \
  --output /lustre/evidence/new-bound-raw-evidence
```

`archive` records every source file (including hidden/empty files and failure
logs), hashes it while reading, streams gzip/tar with bounded buffers, fully
re-reads each tar member and checks the original SHA, size and exact membership,
then rechecks source stat and hashes. It rejects symlinks, special files,
traversal, overwrite, source mutation and malformed archives. `bind` repeats
the complete tar verification once per physical bundle within that invocation.
Before reusing that result and before writing success, it rechecks the same
physical file identities and rehashes every small receipt and the full inventory.
A replacement, symlink or changed byte/stat invalidates reuse. No verification
cache survives the invocation; each logical record independently binds its exact
native and consumer source set. There is no on-disk extraction or copy of
payloads to the root host.
Memory does not scale with raw payload bytes; inventory/native file lists do
scale with file count. These are observations of a quiescent tree, not an atomic
filesystem snapshot. The tool cannot discover failed attempts outside the
explicitly supplied campaign root; the operator must choose its complete scope.

Only transfer the **new-bound-raw-evidence directory** to the dataset import
host, preserving its relative paths. Pass its `external-raw-evidence.json` as
`--external-receipts`. It contains inventory JSONL, source input/verification
receipts, and exact frozen plan/shard-manifest bytes. Large `campaign.tar.gz`
files stay at their external locations. The portable importer rechecks every
small-file hash and the full native/consumer binding; it does not fetch the tar
or claim an external URI was reached. Low-level archive receipts keep
`native_or_accuracy_acceptance=NOT_EVALUATED`: byte preservation cannot grant
native or prediction acceptance. Formal binding requires a separately accepted
stage; synthetic/test/diagnostic markers are rejected with no production bypass.

### Retain external cache-hook execution controls

For a campaign using the task-private `sitecustomize.py` cache hook, the accepted
input manifest must include an `external_control: {path, sha256}` receipt on each
calibration/holdout role spec. This is a **post-execution attachment** preserving
the existing launch chain: original `started.json` hashes the frozen admission
and launcher manifest; those files enumerate the original source and CPU proof
bytes. Attachment creation does not establish that new files existed before a
measurement. The original native data and frozen plans remain unchanged.
The current adapter is explicitly `sglang_pid_private_sitecustomize_v1`, pinned
to the observed SGLang 0.5.20 launch. It requires the admitted hook at the first
native PYTHONPATH entry and its read-only `/opt/glm53flash-cache` mount. A vLLM
bootstrap that then invokes a packaged observer needs a separately validated
adapter bound to its actual freeze; this adapter cannot qualify that chain.

After all children have closed, create a request JSON with these fields:

```json
{
  "original_task_root": "/lustre/task",
  "anchors": {
    "launcher_manifest": "launch/manifest.sha256",
    "admission": "launch/admission.json",
    "cache_hook": "preparation/cache-hook/sitecustomize.py",
    "source_identity": "preparation/source.json",
    "cache_cpu_result": "cache-cpu/609540/result.json"
  },
  "runs": [
    {
      "cell_id": "fpm-shard-<actual-child-id>",
      "raw_root": "/lustre/task/runs/<child>/609709/artifacts/<plan-prefix>/cells/<child>/raw/node0000",
      "started": "runs/<child>/609709/started.json",
      "collector_provenance": "runs/<child>/609709/artifacts/<plan-prefix>/cells/<child>/raw/node0000/collector-provenance.json"
    }
  ]
}
```

The abbreviated array above must enumerate **every** child from that original
admission, with actual paths. The helper rejects missing, duplicate or wrong
children. `--source-root` is the explicitly mapped local or storage-host task
root, while `original_task_root` preserves the original absolute path spelling
from the plans and launch. The physical source root and its members must be free
of symlinks; where a site has a `/lustre` alias, use its known canonical physical
root as `--source-root`. No original absolute path becomes an archive member.

```sh
python3.12 /bundle/external_control.py \
  --source-root /scratch/canonical-task-root \
  --request /scratch/new-evidence/external-control-request.json \
  --output /scratch/new-evidence/external-control
```

Put the resulting `external-control.json` path and SHA in the original acceptance
input manifest **before** acceptance and publication staging. Reference the same
attachment for roles sharing one frozen launch. Only the finite launcher-manifest
and admission-binding closure is copied, plus original per-child started/native
provenance receipts. Source, hook, and CPU proof must already be hashed in the
admission. Identical bytes are stored once, while distinct original paths remain
visible. Historical failed-attempt or source-diff evidence in that closure is
preserved as history; it is not promoted to a passing observation.

Choose an archive source root containing the original per-child `started.json`
as well as the accepted native roots. `bind` joins these original bytes to the
archive inventory, accepted attempt/plan, frozen hook's read-only mount and native
PYTHONPATH. It copies the small control attachment into the portable evidence
bundle, deduplicated across roles; the dataset importer preserves and revalidates
every byte. Neither arbitrary parent directories nor large raw payloads are
copied into this attachment. A missing attachment is rejected when frozen plans
mount the hook or native receipts contain `cache-setup-*`; campaigns without an
external hook retain their existing behavior.

For the first SG campaign, parent runtime preflight hashes 24 SGLang source files;
rank evidence separately records state layout, GPU identity and cache startup.
The control attachment preserves this distinction and creates no per-rank source
hashes or accuracy acceptance claims.

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

Raw archive and binding tests are `test_glm53flash_raw_archive.py` and
`test_glm53flash_raw_campaign.py`; `test_glm53flash_shared_archive.py` uses
actual production planner/sharder schemas and the strict plan reader. Tiny
TEST_ONLY fixtures exercise 32 roles bound to 16 shared physical archives,
sharded accepted roots, complete failure preservation, corruption, traversal,
source mutation, rehashed-inventory mismatch and consumer-origin mismatch.
Test-only monkeypatches permit synthetic receipts solely inside pytest; no CLI
flag permits them in a formal binding or import. Local tests do not establish
ARM/Lustre qualification or formal campaign acceptance.
