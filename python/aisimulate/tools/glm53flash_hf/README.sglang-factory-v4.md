# Public SGLang factory controls (v4)

Schema glm53flash_external_control_v4 selects only adapter
sglang_public_factory_formal_v4. It describes a separately generated public
SGLang campaign: eight parents, 72 children and 2,472 original point mappings.
Each deployment retains 397 calibration and 221 independent holdout points,
the original ordered 100M token partitions, input files and native arguments
apart from the publicly derived run ID. Historical rows are not reused.
The native run ID is deterministic; the collector creates a fresh attempt.
Neither an AMD source render nor the sixteen CPU transports qualifies a GPU.

The implementation in external_control_sglang_factory.py is the shared
prospective launcher/publication contract. This document and its TEST_ONLY
fixtures do not create an actual admission. The old v1–v3 adapters keep their
own original fields and statuses. There is no status alias or conversion to
the historical 636-producer / 642-host launcher.

## Admission

The attachment has the usual schema, adapter, original_task_root, anchors,
runs and copied files fields. Its anchors are exactly launcher_manifest and
admission, both task-relative paths. The original launcher manifest must contain
the admission. The admission keys are exactly ADMISSION_FIELDS:

| Key | Value |
| --- | --- |
| schema | sglang_public_factory_formal_admission_v1 |
| status | FROZEN_PUBLIC_FACTORY_FORMAL |
| host, producer | Separate {source_commit, wheel_sha256, target} objects |
| cpu_controller_manifest, cpu_source | Original CPU controller manifest and its source.json references |
| factory_manifest, factory_source | Original factory manifest and its source.json references |
| cache_hook | Reference to the actual mounted original hook, crossbound to the factory's exact copy |
| storage_binding | Original public raw_archive.create_storage_binding object, or null when no alias is used |
| actual_cpu | Exactly {job_id, directory, files}; positive integer job, job directory, directory-relative SHA map |
| qualification_points | Reference to the original six-prefill / three-decode point document |
| qualifications | Exactly four deployment keys, each containing exactly prefill and decode references described below |

Every reference is exactly {path, sha256}. Paths may use an original absolute
spelling only when the explicit storage binding maps it to original_task_root,
which is the canonical root. Relative paths are already relative to that root.
JSON values and original file bytes are never rewritten. The prepare API
verifies the live public storage proof before and after copying: it must run
where the original alias and canonical root are available. A local mirror
cannot silently substitute for that live check. The resulting portable
validate API uses copied metadata only and requires no live path.

The CPU file map binds actual final result.json, bound-inputs.json,
entry-{identity,render,transport}.json, corresponding process-*/process-result.json,
both identity/*-installed-wheel.json, identity/actual-image.json,
transport-release.json, prepared/inventory.json and transport/receipt.json.
The prepared inventory expands to all original generated plans, crosswalks,
native files and receipts. Each of the sixteen selected transports needs its
individual original transport/<deployment>-<role>-<phase>/receipt.json and one
unambiguous raw/<pod>/collector-provenance.json plus all members of that
receipt's raw_files. These are CPU import/transport records, not GPU timings.

The actual final state must be PUBLIC_ARM_FACTORY_AND_16_CPU_TRANSPORTS_PASS;
the original transport aggregate must be 16_PUBLIC_CPU_TRANSPORT_FIXTURES_PASS.
Full installed maps must equal the frozen ARM RECORD map, including runtime
and all collector bytes. Each preparation/execution pair retains its original
import origins and distinct PID/private cache evidence. A cleanup failure or
incomplete terminal aggregate does not pass. Frozen source declarations may
still contain null qualification fields: they remain source declarations, not
rewritten final results.

The adapter independently rederives the original-to-new phase-local union,
parent/child identities, point geometry, ordered partitions, input bytes, full
crosswalk and sixteen deterministic fixture selections. Only FP8 TP2 uses
allocator 16384; the other deployments use native default. All retain .82.
The installed planner aic_revision, native producer Git/wheel identity and
eventual analysis consumer identity retain their separate meanings.

## Qualification references

Each phase object has exactly:
reader, aggregation, rows, plan, started, checkpoint, installed_reader,
reader_manifest, idle.

The first eight are original references. They bind the new reader's actual
started_sha256, checkpoint_sha256, source_manifest_sha256, raw_directory and
attempt fields; historical field names are not aliases. The reader uses the
separately installed host target. All seven strict reader source hashes and
the complete installed wheel map must match. The native start, original plan,
checkpoint and all six or three rows must identify the same new producer,
phase, attempt, runtime, geometry, 5+10 sampling and actual allocator policy.
Both phases must agree on that policy.

For prefill, idle is exactly {start, observed, return} references to the three
original qualification-only idle records. They prove the unchanged 660-second
post-release interval, unchanged worker PID/start ticks and successful
post-idle health call within the original 900-second request budget. They do
not claim a direct marker read or clean worker exit. For decode, idle is null.
All four deployments need both strict phases. Collection completion alone does
not satisfy this contract.

## Started, final and native controls

Each new original started.json has exactly STARTED_FIELDS:

- schema = sglang_public_factory_formal_started_v1, state = RUNNING,
  decimal-string job, deployment, mode = formal.
- source_commit, wheel_sha256, host_source_commit, host_wheel_sha256.
- admission_sha256, launcher_manifest_sha256, actual_cpu_job.
- selected: the entire unchanged selected prepared/receipt.json formal_children
  row, including both new_identity and historical original_identity;
  plan_sha256 is the new child plan.
- qualification: the exact admission's deployment qualification object.
- requested_allocator_policy: the original native allocator request document.

The selected source row's null new_attempt_id and native_request_set remain
null; the fresh actual attempt is in the original collector provenance. The
actual native request-set identity remains native evidence. A source
declaration is never relabeled as a runtime UUID.

The new final result.json uses schema = sglang_public_factory_formal_result_v1.
It retains every fixed started field verbatim except schema and state. Its
terminal state is one of COLLECTION_PASSED, COLLECTION_FAILED_PRESERVED,
FAILED_PRESERVED. Additional original result/error/timestamp fields are
preserved. The versioned history reader checks the complete fixed identity,
then uses selected.new_identity; it never treats original_identity as the new
child. Failed attempts remain in closed history and cannot be selected as
passing native data merely because they have a final record.

Each attachment run contains cell_id, started, raw_root, collector_provenance,
host_wheel_verification, producer_wheel_verification. The two full installed
proofs are the original actual-host-wheel.json and actual-producer-wheel.json
in that job directory. All 72 selected executions must be present with
separate collector attempts. Public native validation, independent holdout,
accuracy thresholds and complete failed-attempt archive/history gates still
apply. This adapter does not change run_collection, native loops,
measurements or evaluation math.

## Distributed policy identity

Deploy all sixteen glm53flash.POLICY_MODULES together. The closed-history
profile is now fpm_sglang_public_factory_history_v5, based on the recorded
0346 revision with the explicit history changes below. Its fifteen sibling
SHA pins include the factory adapter. The original v4/d53 closure remains a
historical maintenance identity. Neither profile denotes a new native producer
installation. Keep older source bundles and proofs under their original identities.

## Factory attempt history and startup failures

The public factory's attempt directory has four components relative to the
campaign root: `deployment/index-child/job/started.json`. A history request for
this layout must explicitly set
`factory_history_scope: sglang_public_factory_depth4_original_history_v1` and
use the v4 factory external-control schema/adapter. Live capture, actual archive
inventory verification and portable metadata verification use that same named
scope. Unmarked legacy SGLang histories keep the older three-component layout.
Unknown or mixed scopes fail; scope is never inferred from available files.

The scoped request also requires `pre_native_failure_floor`, binding original
`receipt`, `inventory`, and `accounting` path/SHA references, the task-relative
`originals_root`, and the original `jobs` list. `native_started` must be false and
`original_started_records` null. The original closure receipt must describe
`ORIGINAL_PRE_NATIVE_SHARED_STORAGE_GATE_FAILURES_PRESERVED`, empty deployment
output lists, exact inventory/accounting digests and complete member/byte counts.
Scheduler accounting must retain every failed parent, with original stdout and
stderr for each. Do not synthesize started/final metadata for these jobs or put
them in the native selection ledger.

`closed_history.snapshot` copies and hashes all original supplement bytes after
safe storage/path checks. The closure is bounded to 512 original members and
8 MiB, with at most 16 MiB total embedded evidence including metadata. Logs are
binary-safe and never decoded as UTF-8. Review the chosen originals for secrets
before including them in a public artifact; the tool preserves exact bytes and
does not silently redact evidence. Unknown, missing, changed or extra proof
members, symlinks and unsafe paths are rejected. Capture before and after each
archive/bind operation must remain byte-identical.

These originals are embedded in the mandatory hash-bound history sidecar with
storage marker `ORIGINAL_BYTES_IN_HISTORY_SIDECAR_NOT_NATIVE_TAR`. The native tar
continues to cover the unchanged original native campaign tree. The supplement
is explicitly separate from its tar member/byte claims. Existing bound and
portable history transport carries and independently rechecks every embedded
original, even after producer storage is unavailable. Metadata-only offline
verification still does not claim a fresh tar read, native validation, or model
accuracy. The full 32-label/eight-configuration/sixteen-phase gates remain intact.

The original 89-member startup closure produces a 6,192,940-byte serialized
supplement. The existing bound-proof format repeats a shared SG bundle proof
across sixteen labels, adding approximately 99 MB before the remaining proof
data. This is bounded by the fixed 32-label matrix and supplement limits; this
revision does not redesign that transport. A future execution envelope must use
an explicit allowance derived from those limits plus the remaining proof size,
rather than a generic 32 MiB metadata limit. Ordinary raw-artifact guards remain
unchanged. Before publication, the full offline and qualified-wheel consumer
checks must exercise the actual-sized artifact; small TEST_ONLY fixtures prove
the source contract, not this later artifact gate.

This history support is a new maintenance source profile
`fpm_sglang_public_factory_history_v5` based on 0346. Existing native producer,
0346 consumer results, frozen launchers and historical proof/source closures
are not renamed or updated in place. A new consumer build and explicit source
qualification are required before using this revision in a new acceptance run.
