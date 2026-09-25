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
profile is fpm_sglang_public_factory_v4, based on the recorded d53 source
revision with newly reviewed adapter changes. Its fifteen sibling SHA pins
include the new module. This is a new maintenance closure, not a new native
producer installation. Keep older source bundles and proofs under their
original identities.
