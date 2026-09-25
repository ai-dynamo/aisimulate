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

The standalone dataset includes every module listed in
`glm53flash.POLICY_MODULES`, including the versioned external-control adapters. Profile
generation checks every module against the reviewed source and includes their
hashes in immutable Hub readback verification. An isolated-process test imports
the copied modules without the repository on Python's import path.

## Current formal launcher attachments

Legacy `glm53flash_external_control_v1` inputs retain their original contract.
An explicit `glm53flash_external_control_v2` attachment supports
`vllm_nested_formal_v2` and `sglang_split_host_formal_v2`. Select the adapter in
the existing `external_control.py --request` JSON. The attachment is assembled
after execution; it never modifies a launcher, CPU result, plan, native receipt
or `started.json`, and does not claim to have existed before collection.

The vLLM adapter reads the original four nested deployment inventories, their
dictionary of file bindings, the actual public CPU role/phase receipts and the
original per-deployment qualification results. The SGLang adapter separately
binds the host renderer and native producer commits/wheels through the original
CPU controller, factory manifests and prepared inventory. It rejects a draft,
an unknown CPU result, a missing deployment qualification, or a mixed allocator
policy. Wheel blobs remain externally pinned artifacts; this attachment binds
their original identities and installed verification evidence, without claiming
to reinstall or independently rehash the blobs on portable replay.

Both adapters require all 72 original children for their backend, preserving
phase-local IDs: 250 prefill + 147 decode calibration points and 144 prefill +
77 decode independent holdout points per deployment. Original child plan bytes,
corpus/execution options and startup environment remain hash-bound. Every
accepted child must match its original job, attempt, raw root, and native
provenance. The vLLM adapter reuses the original worker/cache/NVML join; SGLang
uses its existing strict native hardware/state/allocator evidence and does not
acquire a new external worker witness requirement. Archive binding also retains
the original execution controls, including original NVML evidence where used.

`external_control_current.frozen_contract(document, resolver)` is a source-only
readiness check before GPU attempts exist. It returns an in-memory index and
does not create an accepted attachment. Full `prepare`/`validate` still requires
all original execution records. The public publication stage independently
reruns native validation, shard union, installed-consumer prediction and every
holdout acceptance gate. Neither this source check nor a completed collection
can substitute for that stage.

An analysis run must record the actual installed consumer wheel and the
publication-tool revision separately from the original producer identities.
Do not label newer analysis code as an earlier installed consumer. No v2
adapter changes calibration membership, interpolation, MAPE thresholds, failed
attempt preservation, Hub publication authorization or offline validation.

## Separate original SGLang admissions

`glm53flash_external_control_v3` with adapter
`sglang_split_host_formal_mixed_v3` accepts an explicit `deployment_controls`
map for the four deployments. Each value binds that deployment's original
factory, CPU result, launcher and admission references by path and SHA256.
For example, an original three-deployment admission and a later independent
fourth admission can coexist without rewriting either admission or any
`started.json`. The validator joins each started record to its own deployment's
exact manifest, qualification, plan, wheel, allocator policy and point set.
The original schema2 contract continues to require its one complete admission.

`external_control_sglang_mixed.inspect_partial(document, resolver)` reports
source preparation separately from admission. A source inventory of 72 children
with only 54 admitted children remains preparation-only. Full attachments,
archive binding and publication still require all four qualified deployments
and 72 explicit complete original children. A later successful whole-child
attempt may be chosen with its own strict evidence while preserving the failed
attempt; neither point splicing nor automatic newest-job selection is allowed.

## Mandatory original-attempt history

Current schema2/schema3/schema4 controls require import policy
`glm53flash-accepted-arrow-partitions-history-v2`. The archive/import and
portable offline checks are separate named contracts:

- `fpm_closed_attempt_history_v2` verifies actual local tar files. Its
  `archive_closed` and `bind_closed` APIs check complete original job membership,
  terminal bytes and the explicit 72-child selection for each backend before
  and after archive operations. Tar members, their complete inventory, all 32
  role labels and selected native attempt IDs must agree. Late attempts, altered
  terminals or a job directory without its started record reject. A pre-start
  result-only failure needs an explicit reviewed representation; it is never
  omitted from history.
- `fpm_portable_attempt_history_metadata_v1` is prepared only after the actual
  full-archive import gate. It copies original history, receipt, inventory and
  input-manifest bytes into content-addressed files. Offline validation
  mandatorily rederives their hashes, membership, labels and selected native
  attempts. Its result is
  `PORTABLE_METADATA_HISTORY_PASS_NO_FRESH_TAR_VERIFICATION`: it does not reopen
  tar, query a live producer filesystem, or establish native/accuracy acceptance.

Missing local archives reject at import. Missing copied proofs or metadata
reject offline; missing tar never selects a weaker import path. The importer
records the identical portable proof reference in its receipt and each
manifest's provenance. Removing both the proof and the new policy cannot
relabel current controls as legacy. Genuine legacy controls retain the previous
policy. Optional full-archive revalidation is an explicit call to
`closed_history.require_publication_history` with local bundles.

The canonical copy contains all sixteen modules in `POLICY_MODULES`. The history
helper pins its fifteen sibling source files; the canonical/profile closure pins
the helper itself and every copied evidence file. These maintenance tools are
source-distributed separately from installed native producers and analysis
consumers. `MAINTENANCE_IDENTITY` retains the exact reviewed
`fpm_sglang_public_factory_v4` source-profile label and base revision
`d53b406d88d1b77e42d9d03c27cd6a7a095319fe`; these identify the public-factory
adapter follow-up, not unchanged d53 bytes or an installed consumer. Record
the actual repository/tool revision separately.
Changes to a pinned sibling require review and a coherent new hash map.

## Public SGLang factory controls

Explicit glm53flash_external_control_v4 / sglang_public_factory_formal_v4
supports the new 72-child public ARM factory and sixteen CPU transports.
Its admission, started and final schemas are separate from historical 636/642
controls. See [the exact v4 contract](README.sglang-factory-v4.md) for source,
qualification, storage, execution and portable history requirements. No
actual admission is created by this implementation or its TEST_ONLY fixtures.

## Cleanup-only reconciliation

A completed original child can fail solely because post-run resource teardown
could not be verified. Ordinary resume is unsuitable for preserving that
history: it may update the checkpoint or schedule native work. The explicit
[cleanup executor](cleanup_executor.py) instead retains the original failed
terminal/checkpoint and emits a separate proof after verified teardown and a
complete strict read of the same native attempt.

The supported original host source identities are pinned in
[cleanup_reconciliation.py](cleanup_reconciliation.py). Eligibility requires
all execution/collection phase marks, no primary error, only
resource_cleanup_failed errors, exact original plan/job/attempt/owner identity,
and an independently reviewed terminal-log disposition with no native,
watchdog, or unresolved failure. Point counts alone do not qualify. A teardown
log error is neither automatically ignored nor treated as a native failure;
its original source and timestamped evidence must be reviewed.

In a separately qualified analysis environment, invoke the cleanup-only API
with an explicit request and a new diagnostic output:

    python -I tools/glm53flash_hf/cleanup_executor.py --request /absolute/request.json --output /absolute/new-output

The request has task_root, attempt_directory, host_source and references.
host_source contains the original commit, runner_sha256 and slurm_sha256.
references has seven original task-root-relative {path, sha256, bytes} entries:
started, final, checkpoint, plan, owner, allocation and failure_review.
The original allocation.json command receipt must bind the same job ID and one
explicit cluster through its successful `scontrol show job JOB --json` output. The review contract
is fpm_original_terminal_failure_review_v1 and binds the original final and
checkpoint hashes, job, attempt, all log hashes/sizes, outcome
SOLE_POST_COLLECTION_CLEANUP_FAILURE, and empty native_failures and
unresolved_findings. The executor does not generate or infer that review.

The live mode of proof contract `fpm_cleanup_reconciliation_v4` requires a fresh successful
`scontrol --local show config` response with exactly the original ClusterName
before any step query or cancellation. Subsequent step queries explicitly use
`--local --clusters=ORIGINAL`, and cancellation uses `--clusters=ORIGINAL`.
The default request routing_mode is explicit-cluster. An explicit verified-local
mode supports sites where the --clusters/database route is unavailable; there
is no automatic fallback. This mode uses squeue --local and exact-step scancel
without --clusters, checking authoritative original ClusterName before/after
every operation. Controller addresses/ports, federation and client settings,
the actual client configuration file, CLI binaries and routing-affecting
environment must remain identical across all boundaries. Non-default federation
parameters, missing configuration, changed owner or any failed post-check reject
before raw reads. Raw commands and checks are retained; environment values are
represented only by a digest.
Inherited SLURM_CLUSTERS and SQUEUE_/SCANCEL_/SCONTROL_ options must be unset so
filters or federation settings cannot produce a misleading empty response.
An unknown, different, or ambiguous cluster, malformed step output, or failed
command rejects before raw reads. Original allocation bytes, the fresh config
response and every actual command remain in the proof and archive closure;
offline validation rechecks this evidence without claiming a live cluster check.
No actual v1/v2/v3 reconciliation was produced; those proof objects are rejected. Ordinary
legacy selection without reconciliation is unchanged. These CLI semantics follow
[SchedMD scontrol](https://slurm.schedmd.com/scontrol.html),
[squeue](https://slurm.schedmd.com/squeue.html) and
[scancel](https://slurm.schedmd.com/scancel.html); no upstream implementation is copied.

The exact pinned Slurm cleanup algorithm only queries and cancels observed
steps matching the original owner job and deterministic name. It does not set
a replacement allocation ID or cancel other jobs. Failed/timed-out queries
never establish absence. Command streams and all errors go to the new output,
not the original cell. After verified absence, public aggregate_cell reparses
the complete original native attempt. Full file inventories, external small
inputs and canonical storage identity are checked before/after the read.
Internal member symlinks or retargeted storage reject. An exception preserves
new diagnostics without a successful proof; no native retry is performed.

A selection using that proof must explicitly use
fpm_complete_child_selection_reconciled_v2 and add a digest-bound reconciliation
reference to the selected whole child. The original failed status stays in
the complete attempt ledger. Legacy v1 selections still require original
success and cannot carry this extension. The original native producer and
the actual analysis consumer identities remain separate from this maintenance
source revision.

Archive creation embeds the reconciliation proof, verifies its whole original
inventory against actual tar members, and preserves all other failed attempts.
Portable offline validation rechecks the same copied inventory/proof closure;
it does not perform fresh teardown, native parsing or tar verification.
Missing proof, changed logs, new or omitted attempt files, mixed original
identity, and a failed reconciliation reject. All72 whole-child choices per
backend, all32 role labels, full original point denominators and ordinary
prediction/accuracy gates still apply. A reconciliation is not acceptance.

The executor is an operational API: source availability does not authorize
scheduler cleanup. No actual reconciliation or dataset acceptance is claimed
by its TEST_ONLY fixtures.

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
  --history /absolute/path/to/explicit-archive-history.json \
  --bundles /absolute/path/to/archive-bundles.json \
  --source-revision ACTUAL_40_CHARACTER_AISIMULATE_COMMIT \
  --evidence-date ACTUAL_YYYY_MM_DD
```

For current schema2/schema3/schema4 controls, `--history` contains
`contract="fpm_closed_attempt_history_v2"` and a `proof` object with `path`,
`sha256` and integer `bytes` fields. The proof path is relative to the bound
external-evidence directory. `--bundles`
is the exact 32-label map to local bundle directories. Only genuine legacy
controls may omit these arguments. The importer streams the actual tar gate
before creating its canonical destination, then copies the verified metadata
under `campaigns/.../history/`. Installed offline cache loading verifies that
copied evidence and does not download raw tar on every prediction.

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

`raw_archive.py`, `raw_campaign.py`, `external_control.py`,
`external_control_current.py` and `external_control_vllm.py` are original Apache-2.0 implementations
using only Python's standard library. They can run with Python 3.12 on the
remote Lustre host without importing AISimulate, torch, Arrow or a GPU runtime.
These are repository maintenance tools, not installed SDK entry points. For
remote use, deploy all sixteen `POLICY_MODULES` together in a new
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

When original metadata uses a storage alias such as `/lustre` but the physical
root is under `/scratch`, preserve the original paths and bind the roots
explicitly. Run the following on the storage host, using real absolute roots and
new output files:

```sh
python3.12 tools/glm53flash_hf/raw_archive.py storage-root \
  --original /lustre/campaign/task --canonical /scratch/campaign/task \
  --output /scratch/evidence/storage-root.json
python3.12 tools/glm53flash_hf/raw_campaign.py plan \
  --stage /scratch/evidence/accepted-stage \
  --manifest-base /lustre/campaign/task/analysis \
  --storage-root-binding /scratch/evidence/storage-root.json \
  --output /scratch/evidence/archive-plan.json
```

The proof records actual root resolution, `samefile`, device/inode and observed
alias ancestors. Planning, archive creation and binding recheck it before and
after their reads; a retargeted alias fails. Only a contained suffix may be
translated to the verified canonical root. Canonical paths and all source
members retain the original no-symlink checks, including `O_NOFOLLOW` reads.
The original plan, started receipt, accepted `raw_root` and external-control
path equality remain unchanged. Use canonical paths for stage, archive and
bound-output directories. The explicit mapping covers this one task root;
unrelated files require their own safe canonical paths.

The root proof and its digest enter the archive input manifest, which is embedded
in the archive and bound by the receipt. Portable verification checks that closed
proof, inventory and original/canonical suffix correspondence. It does not claim
the old host's inode still exists. Without a binding the previous path rules
remain unchanged; this option does not authorize arbitrary member symlinks.
Only current external-control attachments accept this explicit storage mapping.

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

For current schema2/schema3/schema4 controls, use the history-aware APIs with the
same accepted stage and archive plan. `history_inputs` maps each backend to
hash-and-size-bound original request and selection-ledger references. Each
ledger retains every original attempt and chooses complete children explicitly.
These APIs are available from the source checkout; no SDK wheel or native loop
is changed:

```python
import json
from pathlib import Path
from tools.glm53flash_hf import closed_history, raw_archive, raw_campaign

stage = Path("/lustre/evidence/accepted-stage")
plan = json.loads(Path("/lustre/evidence/archive-plan.json").read_text())
history_inputs = json.loads(Path("/lustre/evidence/history-inputs.json").read_text())
closed_history.archive_closed(
    stage, plan, "sglang-fp8-2-decode-calibration",
    Path("/lustre/evidence/archive-bundles/sglang-fp8-2-calibration"),
    history_inputs, campaign=raw_campaign, archive=raw_archive,
)
# Repeat for each distinct source-root/URI pair, then bind all 32 labels.
bundles = json.loads(Path("/lustre/evidence/archive-bundles.json").read_text())
bound = Path("/lustre/evidence/new-bound-raw-evidence")
proof = closed_history.bind_closed(
    stage, plan, bundles, bound, history_inputs,
    campaign=raw_campaign, archive=raw_archive,
)
proof["path"] = closed_history.BOUND_PROOF
with Path("/lustre/evidence/explicit-archive-history.json").open("x") as output:
    json.dump({"contract": closed_history.CONTRACT, "proof": proof}, output)
```

Each distinct physical archive can run independently, with retries recorded in
new output directories. The following low-level commands retain their legacy
contract. A bare low-level archive/bind output cannot substitute for the required
current history-aware proof:

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

For a campaign using a task-private Python startup cache hook, the accepted
input manifest must include an `external_control: {path, sha256}` receipt on each
calibration/holdout role spec. This is a **post-execution attachment** preserving
the existing launch chain: original `started.json` hashes the frozen admission
and launcher manifest; those files enumerate the original source and CPU proof
bytes. Attachment creation does not establish that new files existed before a
measurement. The original native data and frozen plans remain unchanged.
The default adapter is `sglang_pid_private_sitecustomize_v1`, pinned
to the observed SGLang 0.5.20 launch. It requires the admitted hook at the first
native PYTHONPATH entry and its read-only `/opt/glm53flash-cache` mount.
The separate vLLM adapter below requires the original packaged observer and
Python's `usercustomize` route. It does not accept the historical bootstrap that
manually invoked another `sitecustomize`.

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
Some original preparation `source.json` files omit `wheel_sha256`; when present
it must match. In both cases, the frozen launcher must retain the actual passed
producer CPU receipt and installed-wheel source record, and their source/wheel
identities must agree with admission and every original child started receipt.

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

### vLLM packaged observer plus usercustomize

Set the request's `adapter` to `vllm_packaged_observer_usercustomize_v1`.
Retain the same launcher/admission/source anchors and add the following exact
eight-anchor set (paths below illustrate the schema, not published evidence):

```json
{
  "launcher_manifest": "launch/manifest.sha256",
  "admission": "launch/admission.json",
  "source_identity": "preparation/source.json",
  "cache_cpu_result": "cpu/public-environment-receipt.json",
  "cache_hook": "cpu-bundle/cache-hook/cache_hook.py",
  "usercustomize": "cpu-bundle/cache-hook/usercustomize.py",
  "observer_entry": "cpu/diagnostic-cell/slurm-runtime/sitecustomize.py",
  "runtime_manifest": "cpu/diagnostic-cell/slurm-runtime/runtime-source-sha256.json"
}
```

Before executing any formal child, its original admission must contain `backend:
"vllm"`, the exact `framework_version`, and a `vllm_startup` object:

```json
{
  "cpu_result": "cpu/result.json",
  "cpu_source_identity": "cpu-bundle/source.json",
  "cpu_bundle_manifest": "cpu-bundle/manifest.sha256",
  "cpu_identity_paths": [
    "cpu/diagnostic-cell/raw/node0000/preparation-identity.json",
    "cpu/diagnostic-cell/raw/node0000/execution-identity.json"
  ],
  "mounts": {
    "/opt/glm53flash-cache": "/lustre/task/cpu-bundle/cache-hook",
    "/opt/glm53flash-current": "/lustre/task/current-producer-python",
    "/opt/glm53flash-candidate": "/lustre/task/candidate-python",
    "/opt/glm53flash-dynamo": "/lustre/task/dynamo-python"
  }
}
```

Admission bindings must enumerate all CPU bundle manifest members, original
result/public/prepare/execute JSON, both CPU processes' original cache/order JSON,
and the staged observer, runtime manifest, scheduler and worker hook. Include
every child's staged `run.sh`, `collector-runtime-env.sh`, observer, manifest,
scheduler and worker hook. CPU source/wheel/version must equal the admitted
producer; a historical CPU receipt for a different producer cannot qualify it.
The child scheduler is compared with the original CPU staging bytes. The worker
hook is also compared with its original CPU observed source SHA. This preserves
what was actually recorded without inventing a historical scheduler source hash.

The frozen native environment must use the exact PYTHONPATH order
`/tmp/fpm-bench:/opt/glm53flash-cache:/opt/glm53flash-current:/opt/glm53flash-candidate:/opt/glm53flash-dynamo`,
normal Python startup, `DYN_FPM_GLM53FLASH_REAL_KV=1` and
`AISIM_GLM53_PURPOSE=fpm`. Runtime mounts must be read-only and cannot shadow
those paths. Every child keeps its own original started/admission/launcher chain.

After execution, each request `runs[]` item also supplies `nvml_receipt` and
`nvml_stdout` paths, plus `workers` in exact TP-rank order. Every worker object
has exactly `device`, `cache`, and `order` paths: its original
`native-device-rank-N.json`, `cache-setup-PID.json` and
`cache-startup-order-PID.json` from the selected native raw root. The independent
original NVML command receipt must retain the job, observation timestamp, exit
code, output SHA and exact `srun --jobid=JOB --overlap --ntasks=1 --nodes=1
--cpus-per-task=1 nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory
--format=csv,noheader` argv. Preserve its original CSV; a derived join assertion
alone is insufficient.

The adapter recomputes worker rank → native GPU UUID → original NVML
`VLLM::Worker_TPN` PID → original startup receipts. It requires distinct workers,
complete TP2/TP4 coverage, matching native provenance/version, the same observed
47-file runtime closure (including 19 binaries) as both original CPU processes,
and PID/allocation-private cache paths created before framework imports. Final
binding also requires each worker receipt's exact bytes in the accepted native
file inventory and physical archive. This does not replace the strict native
reader or its runtime admission checks. CPU helpers' cache receipts cannot stand
in for native GPU workers.

This adapter's complete launch tests are **TEST_ONLY**. The original CPU 610254
and GPU 610709 files support component replay only; no formal vLLM admission or
accepted archive has been constructed from those qualification jobs. Final
formal archive acceptance remains pending the actual frozen launch, its own
worker witnesses, and accepted calibration/holdout data. Failed attempts and
diagnostics remain preserved under their original scope.

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

## Planner, native producer, and analysis identities

New partitions retain the source table's `aic_revision` unchanged and add
`planner_revision` with that exact value under
`revision_identity_schema=glm53flash_revision_identity_v1`. The value can be an
installed distribution RECORD identity, rather than a Git commit. The historical
`producer_revision` remains as an explicitly tagged
`legacy_planner_revision_alias`; it must not be interpreted as a worker revision.
Original source tables, metadata, plans, and execution receipts remain unchanged.

For these partitions, canonical import requires all four original prefill/decode
and calibration/holdout v2 control attachments for each configuration. It checks
every original child plan's `aic_revision`, revalidates the complete attachments,
and derives `planner_source_commit`/`planner_wheel_sha256` and
`native_producer_revision`/`native_producer_wheel_sha256` from their source and
installed-wheel evidence. The four identities must agree. This represents a
642b host renderer with a 636 native worker without relabeling either one.
Missing identities, mixed workers, or mismatched plan revisions reject import.

The import receipt and configuration provenance retain the resulting
`revision_identity`, including original control hashes. Its `analysis_revision`
contains the exact acceptance report's `installed_consumer` object and the
independent `publication_tool_revision` supplied by `--source-revision`.
The consumer payload hash covers the installed public SDK/native consumer bytes
checked by acceptance; it is not a whole-wheel hash or an inferred Git commit.
The tool revision does not attest which consumer was installed. A newly built,
verified installed consumer remains a separate prerequisite for formal analysis.

For new explicit revision metadata, catalog/configuration `aisim_commit` is the
verified native producer commit and carries
`aisim_commit_semantics=native_producer_revision`. The legacy
`provenance.producer_revisions` list retains its planner-alias meaning with an
explicit tag. Snapshot validation re-derives these fields from the archived
originals, including the installed analysis identity. Historical stages and
snapshots without the new marker retain their old contract; current v2 controls
cannot silently downgrade to that legacy naming path. These metadata changes do
not alter calibration rows, native timings, predictions, or acceptance limits.

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

The current-history suites are `test_glm53flash_closed_history.py`,
`test_glm53flash_portable_history.py`, `test_glm53flash_history_import.py` and
`test_glm53flash_current_history_import.py`. They exercise real tiny TEST_ONLY
tar files, failed/late attempt preservation, rehashed-metadata tampering,
mandatory importer routing, and offline validation after deleting tar and live
source trees. The new-policy importer fixture explicitly mocks outer native
acceptance and dataset transport; the existing legacy integration test separately
uses the real dataset-manager API. Mixed original admission tests are in
`test_glm53flash_external_control_sglang_mixed.py`. These tests do not admit any
actual dataset or establish new GPU accuracy.

### Explicit historical accounting mode

When live-controller cleanup cannot run, a separately authorized request can
select `termination_mode: historical-accounting`. It cannot also select a live
`routing_mode`. There is no automatic fallback. The original seven references,
terminal-log eligibility, complete native reader and immutable archive inventory
remain mandatory. The distinct outcome `HISTORICAL_OWNED_RUN_TERMINATED` says only
that the exact historical Slurm-managed run has positive terminal accounting
records. It does not claim a live empty queue, current free GPUs, cancellation,
native success, or permission to schedule work. ExitCode zero is not an
acceptance criterion; the unchanged original native/log gates remain decisive.

The additional `accounting` request object has `client` and `known_history`.
`client.executable` and `client.config` each contain an absolute `path` and its
pre-reviewed `sha256`. `known_history` is a normal task-relative
`{path, sha256, bytes}` reference to a `fpm_accounting_known_step_history_v1`
document. Its nonempty `captures` list contains `reference`, `command_pointer`,
`time_pointer`, `timezone`, and `cluster`. References bind independent original
capture JSON bytes. Pointers are lists of object keys/list indices locating the
actual recorded command and epoch-nanosecond observation time; for example,
`["rows", 0]`/`["observed_ns"]` or `["query"]`/`["before", "epoch_ns"]`.
The command's original argv, return code and literal stdout/stderr (or original
base64 plus SHA fields) are reparsed. No reconstructed row list is accepted.
Both supported historical column formats are explicit in
[accounting_termination.py](accounting_termination.py). At least one independent
17-column capture must bind DBIndex and parent UID/Restarts0. Earlier 12-column
captures may add known steps and preserve revised state/end history. Every
capture remains embedded unchanged; historical terminal updates are never
silently overwritten or treated as new native evidence.

Fresh before/after captures use the pinned executable/config, exact original job
and explicit cluster, `--duplicates`, and all 17 fields with UTC timestamps.
There is no allocation-only or state filter. Installed version/helpformat are
checked before each data query. Original cluster/UID/Submit/Start/DBIndex and
Restarts0, batch/extern and all independently known steps must match; unknown,
ambiguous, missing, nonterminal or truncated records fail. All returned steps
must belong to that same database run and be terminal with valid End, including
steps not bearing the collector's name. Parent End is not an upper bound on
step End. The collector never interprets empty output as absence.

The executor reads the actual original owner before and after each accounting
capture and checks its canonical path, bytes and filesystem identity. Client
files, configuration and effective environment must stay unchanged. Inherited
SACCT_/SLURM_ filters/routes are removed; the pinned SLURM_CONF, UTC and standard
time format are explicit. Original metadata/storage are checked before raw
inventory/native replay and afterward. Complete accounting captures must bracket
the full inventory → original native read → inventory interval and agree on
all record fields/membership. A change, timeout, unsupported field or native
failure preserves diagnostics and fails; it is never retried into acceptance.

The same v4 verifier is mandatory in canonical selection, actual archive checks
and portable offline history. Both captures, original independent history,
original owner/metadata, exact inventory, native-reader identity and source
closure remain inside the proof. Offline verification performs no new scheduler
query. These are maintenance tools only; native producer/runtime bytes and
ordinary legacy selections are unchanged. Original watchdog failures remain
ineligible regardless of accounting state. The source follows the public
[SchedMD sacct contract](https://slurm.schedmd.com/sacct.html); no Slurm source is
copied or modified, and TEST_ONLY tests do not qualify actual originals.


## Explicit collection and original pod roots

The public FPM native validator takes the collection directory
.../cells/<cell_id>/raw; original reader, worker and cleanup-reconciliation
receipts identify its pod directory .../raw/node0000. A manifest must not pass
the latter as the validator root. That loses the pod component required to locate
the original collector-provenance receipt and fails before prediction.

For new full-publication manifests, each native child must opt in explicitly:

    {
      "cell_id": "<original child cell id>",
      "raw_root": "/<original attempt>/artifacts/<plan prefix>/cells/<original child cell id>/raw",
      "native_root_scope": "glm53flash_collection_node0000_v1",
      "attempt_id": "<original collector attempt id>"
    }

Keep the original external-control document, its runs[].raw_root pod path,
collector provenance and worker references unchanged. The current external-control
adapter binds the new collection root to exactly that original pod and checks
accepted evidence keys such as node0000/collector-provenance.json against the
original SHA. The new scope requires original current external controls. All native
children in a stage use the same scope; unknown, missing within a marked set, or
mixed scopes fail. Unmarked legacy manifests retain their original equality rule;
there is no path-based fallback or automatic upgrade.

Archive plans carry both accepted_raw_roots (collection roots) and explicit
accepted_native_roots with cell, attempt, scope and original_pod_root.
Closed history derives the expected plan prefix, child and pod again from the
unchanged original started/checkpoint/selection bytes, including any separately
verified cleanup reconciliation. Archive and portable metadata validators reject
extra or missing pods, including empty foreign directories, as well as unsafe
paths, symlink members, omitted provenance and mismatched evidence bytes. Existing
explicit storage-alias proof and closed-source checks remain required.

The source profile fpm_collection_pod_boundary_v1 is based on immutable
b490efc66aa78d6ee84f246c6fd611ea68829849; its coherent policy bundle now
includes native_roots.py. It does not relabel a native producer or installed
analysis consumer. Import still verifies actual archive bytes separately; offline
validation rederives the copied inventory/history/source closure without claiming
a fresh tar read. No new native collection, prediction, threshold or acceptance
is implied by a successful directory-boundary check.
