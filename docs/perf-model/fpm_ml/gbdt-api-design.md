<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Seeded GBDT workflow and opt-in adoption

This design extends the existing seeded learned-model API with authenticated
source bundles, consistent readiness and dependency reporting, complete portable
snapshots, and a runnable B→B+C example. Applications own telemetry collection and
initial training. The AISimulate runtime updates an accepted seed; it does not
provide a source-free C-only collection or initial-fit service.

**Status:** revised implementation scope. The earlier native first-fit proposal,
its twenty-row initialization threshold, reserved native tree capacity, and new
Gym protocol are removed from these four PRs. Historical experiment reports remain
historical evidence; they do not qualify this revision. The accepted global default
is still `estimation_mode: auto`, with OP → interpolation → regression selection.
Learned use is explicit. No Dynamo or Gym migration is required to preserve an
existing caller's behavior.

**Baseline:** merged [PR #658](https://github.com/ai-dynamo/aisimulate/pull/658),
merge revision `9f4819e2` (earlier source audits used `a8b30ca6`). Implementation
must preserve its public signatures, ordinary
configuration round trips, seeded predictions and online updates, supported
checkpoint readers, and existing Gym protocols. Recheck these claims against the
actual prerequisite revision before delivery.

## Part I. Implementation design: four PRs

### 1. Outcome, boundaries, and terminology

The runtime path is:

`best_available(accepted source)` → predict → admit measured C → explicitly flush
when intended → inspect readiness → save → restore fixed or continued state.

The production constructor remains
`RustForwardPassPerfModel.best_available(config)` in Python and its canonical Rust
counterpart. Follow the [performance-model API rule](../../../python/aisimulate/.claude/rules/perfmodel-api.md)
and [numerical parity rule](../../../python/aisimulate/.claude/rules/rust-core/parity.md).

| Term | Meaning |
| --- | --- |
| A | Accepted whole-forward self-benchmark observations; distinct from operation tables. |
| B | Accepted serving observations reserved for offline training. |
| C | Chronological measured feedback admitted after prediction or scoring. |
| Store | A workload class within a fixed worker model: pure prefill, pure decode, locally mixed or cross-rank aggregated. |
| Readiness | An available forest or explicitly qualified fallback for the required store; not proof of coverage for every query or measured serving accuracy. |
| Source bundle | A versioned identity, seed, training recipe, provenance and dependency manifest. |
| Model snapshot | Complete numerical state and required timing-source closure. It does not contain application stream offsets. |

A and B describe evidence, not separate estimator modes. An accepted B seed with
no admitted C remains B; admitting C to that instance makes it B+C. Online
adaptation does not occur merely because a prediction method is called.

| Workflow | AISimulate responsibility | Application responsibility |
| --- | --- | --- |
| A or B seed | Validate source/identity, predict, expose readiness and dependencies. | Collect and qualify observations; train and publish an appropriate source. |
| A+C or B+C | Update existing seeded stores and retain numerical state. | Telemetry, rank grouping, sampling, deduplication, order, admission cadence, errors and promotion. |
| No accepted seed | Reject unresolved learned construction; accept a compatible source once provided. | Buffer real observations, perform initial fitting and validate the resulting seed. |
| Snapshot continuation | Restore model state and timing dependencies exactly. | Bind the matching input cursor and coordinate durable delivery/recovery. |

The public offline trainer can produce the initial seed, including from an
application's first accepted observations. That helper does not make the runtime
model a collection service. Existing caller-managed C-only experiments and Gym
bootstrap policies remain outside this runtime scope.

### 2. Compatibility requirements

- Keep `auto` and `fallback_policy: deny` defaults and candidate order unchanged.
- Retain public method signatures and complete canonical configurations. New fields
  must be optional and must not rewrite ordinary existing configurations.
- Preserve raw `artifact_path`/`checkpoint_path` and `source_sha256` workflows.
  They do not acquire mandatory bundle/fallback pins or new source-selection rules.
- Preserve old diagnostic meanings, not-ready reasons and online checkpoint
  version-5 semantics. New readiness/dependency detail is additive.
- Preserve frozen, online and causal-selector predictions and feedback behavior,
  original-A mixed-policy compatibility, fixed bins and imported tree capacities.
- Preserve existing state snapshot readers and verify actual saved fields strictly;
  absent newly additive metadata must not cause an old valid snapshot to fail.
- Preserve Gym's existing recipe identifiers, source scopes, feedback cadence,
  output layouts and historical readers. No new native initial-fit protocol.
- Preserve Dynamo's existing default regression path and learned config pass-through.
  Downstream shortcomings must be reported separately from changes caused by this stack.

Bad source hashes, incompatible identity, invalid configuration and update errors
are errors. They must not silently become a fallback estimator, a fresh learner,
or an unavailable search candidate.

### 3. Public contract after the four PRs

#### 3.1 Preserve the method surface

| Method | Contract |
| --- | --- |
| `best_available(config)` | Select and validate the model once. Explicit learned online mode needs a checkpoint or an eligible discovered bundle. |
| Prediction methods | Query the selected model without admitting observations or fitting an absent store. |
| `tune_with_fpms(iterations)` | Consume complete measured iterations and update existing seeded stores under the established numerical contract. |
| `flush_updates()` | Explicitly fit a final pending update batch; no source-free initial fit. |
| `readiness_report()` | Preserve old store readiness and expose required execution coverage additively. |
| `dependency_info()` | Describe verified timing dependencies of the constructed model. |
| `save_state_snapshot(path)` | Atomically publish a new complete snapshot without fitting pending rows. |
| `best_available(state_snapshot=path)` | Restore through the canonical constructor, with fixed `on_tune: reject` by default and explicit `continue` for learning. |

#### 3.2 Source configuration additions

New controls belong to `estimator_config.fpm_learned`; Rust owns validation and
selection. The existing [configuration reference](configuration.md) is normative.

| Field | Behavior |
| --- | --- |
| `bundle_path` | Explicit authenticated bundle manifest; exclusive with raw source paths. |
| `source_policy` | `auto` permits exact learned catalogue discovery for online selection; `explicit_only` requires an explicit source. Does not change global estimator selection. |
| `fallback` | `auto` permits a supplied qualified bundle prior; `disabled` selects learned predictions only. A/B labels alone never create a prior. |
| `resolved_source` | Generated complete source pin for new bundle/catalogue paths. Preserve when transporting that resolved configuration. |
| `resolved_fallback` | Generated binding for new fallback/source composition, including explicit absence. |

There is no `bootstrap` configuration block. A bundle records its actual offline
`training_recipe`; there is no native first-fit recipe or fixed row threshold.
Legacy raw-source requests remain compatible with their existing path/digest
contract. Explicit new pins, once supplied, remain strict.

#### 3.3 Ownership and shared boundaries

Rust owns identity normalization, source resolution, fallback routing, prediction,
seeded updates, readiness and dependency validation. Python and all consumers pass
through complete configurations; they must not parse manifests to reimplement
source selection or infer required workload stores independently.

The additive execution report identifies required stores from the canonical
worker role, backend scheduler and attention-DP width. Native execution enforces
actual width when identity width is unknown. A source can be inspected before it
is executable; construction success and the legacy coarse readiness flag do not
establish complete coverage. Consumers may display detailed coverage without
changing existing coarse diagnostic values.

Dependency inspection exposes manifest declarations; canonical construction
verifies payload bytes and semantic coherence. Validate dependencies at explicit
preparation/assessment boundaries, not inside prediction loops. Keep architecture,
hardware and KV-capacity evidence separate from timing-source ownership.

#### 3.4 Feature PRs own usable workflows

| PR | Complete responsibility |
| --- | --- |
| PR1 | Source production/install, catalogue, strict new source pins and qualified prior, seeded caller/onboarding integration, reference docs and tests. |
| PR2 | Additive execution readiness/dependencies, explicit learned candidate filtering and compatibility checks, without changing seeded fitting or caller-managed initial training. |
| PR3 | Complete snapshots, source/prior closure, relocation, compatible restore/continuation, caller/onboarding consumption and tests. |
| PR4 | Trainer-free packaged verification, runnable B→externally supplied C→B+C example, supported trainer packaging and consolidated guidance. |

Feature PRs include necessary integration and tests rather than leaving a required
runtime change for publication. Existing interpolation onboarding and its six
stages remain intact.

### 4. PR1: stable fallback and source resolution

**Suggested title:** `feat: add pinned learned bundles and seeded workflows`.

#### 4.1 Bundle and catalogue contract

A versioned bundle records:

| Record | Required information |
| --- | --- |
| Format | Kind, schema, producer and offline recipe revision; externally pinned manifest digest. |
| Identity | Complete canonical source identity, including role and known deployment facts. |
| Seed | Training/runtime checkpoint or inference artifact, relative path/digest, supported adaptations and covered stores. |
| Training recipe | Actual artifact estimator name, checkpoint trainer version, ordered features, target transform and hyperparameters. Describes this seed only. |
| Provenance | A/B anchor, derived accepted-row counts, immutable input references/hashes, separate evaluation inputs, and explicit C lineage when applicable. |
| Optional prior | Qualified original-A source and interpolation dependencies, eligibility, calibration and quality controls. |
| Dependencies | Relative file paths, sizes, SHA256 hashes and purpose; reject escaping paths and undeclared required files. |

The training recipe uses the artifact's actual estimator metadata and the training
checkpoint's scikit-learn version. `recipe_version` identifies the producer's
named offline recipe revision independently. Online sources need fitting state,
not inference trees alone. Frozen artifacts remain supported, and selector
construction retains its training-seed requirement.

The catalogue is `<systems_root>/fpm_learned/index.json`, outside operation-table
discovery. Ordered roots and exact identity matching use existing `systems_paths`.
Explicit sources win. A catalogue identity has one designated default; corruption
or ambiguous declarations fail. A missing source is an error for learned
construction. No network download, benchmark-directory scan, or automatic switch
from another selected estimator is introduced.

New resolved exports keep strict source/fallback pins. Catalogue updates affect
new unresolved requests only. Pristine A/B sources must not silently contain C;
validate or explicitly label admission lineage to prevent evaluation leakage.
Unknown facts cannot establish an exact indexed deployment match.

#### 4.2 Qualified fallback

Internal fallback remains part of the selected learned estimator. It does not
change `selected_estimation_mode` or imply global `fallback_policy: allow`.
Only explicitly qualified original-A evidence can provide the mixed interpolation
prior. B and B+C do not manufacture an A prior. Preserve existing routing,
calibration, quality-gate and selector semantics for
`temporary_mixed_composition` configurations. New bundle intent may prepare that
same qualified policy without making callers author its internal details.

Every predicted route must remain available after source relocation where the
snapshot contract owns it. A missing forest does not justify generic OP fallback.
An eligible prior can make a required route ready but does not prove support for
all query coordinates. Never convert validation/fitting failures into fallback.

#### 4.3 Producer, installation and acceptance

Extend `aisimulate_core.sdk.fpm_learned` with bundle train/export/inspect/install
helpers and its existing module CLI. Training remains an explicit offline action;
export accepts an already accepted checkpoint. Return complete configurations
resolved through the canonical native constructor, including prepared prior pins.

Use new immutable destinations, authenticated dependency copies and atomic
catalogue updates. Installing a second recipe preserves the default unless
explicitly requested otherwise. Include source bytes and notices in distributions
as required; a synthetic fixture must identify itself as functional evidence.

Tests cover source precedence, exact identity, tampering, missing/duplicate
catalogue entries, adaptation eligibility, legacy raw-source compatibility,
producer/install round trips, real seeded predictions and installed-package use.
Optional onboarding uses existing commands and a selected prediction config;
collection-quality interpolation and generated default configs remain unchanged.

### 5. PR2: shared readiness and seeded caller compatibility

**Suggested title:** `feat: expose learned execution readiness to consumers`.

This PR adds shared consumer information and integration. It does not change the
seeded learner's objective, initial training, binning, tree capacity, online
checkpoint schema, admission counters or failure-atomicity contract.

Keep existing readiness fields and reasons. Add required-store execution coverage
and dependency information where consumers need them. Native execution must check
its actual role/topology; fixed replay must not infer readiness from one covered
store. Preserve old diagnostics and snapshots when the additive fields are absent.

For explicitly learned search, skip only automatically enumerated candidates that
are genuinely unavailable for the required deployment. Preserve explicit user
identity, source and controls. Missing deployment coverage is distinct from a bad
source, invalid config or update failure. Existing Auto/native/regression search
behavior remains unchanged. Do not add redundant model construction or gating to
configuration normalization merely to obtain readiness early.

Onboarding validation reports incomplete coverage for partly ready sources and
never fits absent stores. Invalid or absent required sources are errors. Serving
validation honors the selected complete config instead of rebuilding interpolation
implicitly. These behaviors belong in the existing stage-six commands, not a new
stage or a live service.

Acceptance includes unchanged baseline SDK tests, source-config round trips,
coarse diagnostic compatibility, exact actual-width coverage, seeded searches,
replay/onboarding error reporting, and pinned downstream consumer checks. Gym
keeps its current initial-fit and evaluation protocol; no new protocol readers or
published metric-layout migration are part of this PR.

### 6. PR3: complete portable learned snapshots

**Suggested title:** `feat: preserve complete learned snapshots and continuation`.

A complete snapshot owns forests, pending updates, retained training/recent rows,
existing fitted bins and tree capacities, calibration, selector evidence, routing
state, counters, deterministic update state, source evidence and required native
prior dependencies. It preserves all state of supported seeded online, frozen
calibrated and causal-selector models. It does not contain application buffers or
stream offsets, and saving does not fit pending rows.

Use a versioned manifest with file sizes and streaming hashes; copy or serialize
required timing closure into a staging directory, sync it, and atomically publish
a new destination. Keep source provenance distinct from mutable runtime state.
Construction verifies hashes and semantic agreement before use. The snapshot's
owned paths relocate; explicit external resources remain external and declared.

The outer snapshot format/layout may advance with explicit readers for existing
supported snapshots. Online runtime checkpoint version 5 remains unchanged.
Absent newly additive readiness fields in old snapshots may be projected out
for comparison; saved readiness values, counts and model state remain strict.
Unsupported, corrupt or incompatible snapshots fail without fresh initialization.

Restoration defaults to `on_tune: reject`; `continue` restores exact numerical
continuation through `best_available`. Tests cover pending rows, fixed/continued
state, frozen calibration, selector and fallback state, deletion of original
sources, relocation in a separate process, strict tamper checks, old snapshots,
and identical later-feedback predictions/counters. Empty or untrained regression
snapshots retain their existing behavior; learned snapshots remain seeded.

Python consumers use the native summary's resolved config and dependency
inventory. Inspection is manifest-only; model construction verifies payloads.
Preserve independent model architecture and resource validation, including a pinned
checkpoint `config.json` where `fpm_profile` requires it. Owning timing sources does
not establish GPU or KV-memory feasibility.

### 7. PR4: seeded examples and adoption guidance

**Suggested title:** `feat: add reproducible opt-in learned examples`.

Publish a small self-authored synthetic decode B bundle, with exact provenance,
training recipe and separate evaluation/feedback inputs. A decode-only identity
makes required-store readiness complete for its intended functional workload.
Do not present the synthetic identity as a supported model/hardware deployment or
publish it as an operational catalogue default.

The installed `gbdt_example` module must exercise public train/export/install,
explicit and discovered selection, then tune the actual B instance with externally
provided chronological C. It predicts before admission, explicitly flushes, saves
and relocates a ready B+C snapshot, rejects tuning on a fixed restore, and compares
continued feedback with an uninterrupted twin. A separate application cursor binds
the feedback digest, consumed offset and snapshot digest. Document that production
applications need their own durable commit/reconciliation protocol.

`verify-bundle` must work without importing the optional trainer, validate all
input/source hashes and lineage, and compare native predictions and complete
execution readiness. `run` uses the supported scikit-learn 1.9.1 optional dependency;
the base runtime remains trainer-free. Installed-wheel tests run outside a source
checkout and verify module/native origins.

Guidance belongs on existing workflow/configuration/onboarding pages. Explain B,
C, admission ownership, exact identity, fixed evaluation, missing coverage and
independent resource/accuracy checks. Do not imply the offline onboarding CLI
collects live traffic or automatically learns. Keep the six-stage process and
accepted interpolation choices unchanged.

### 8. Worked journey and failure branches

1. The application collects and accepts B for one fixed deployment identity.
2. An explicit offline trainer exports a compatible seed/bundle. Verify all required
   workload stores and independent holdout/serving evidence.
3. Construct with explicit `fpm_learned`, preserving the complete returned config.
4. The application assembles complete C iterations, predicts/scores first, then
   admits each selected measured target under its sampling and deduplication policy.
5. Inspect errors and native readiness. Reconcile partial admission before retrying;
   do not regenerate labels from predictions or blindly replay an entire batch.
6. Explicitly flush if desired, save a new snapshot, and durably bind the matching
   application cursor. Promote only after coverage, memory and serving validation.
7. Restore fixed for replay/evaluation or explicitly continued for later measured C.

| Failure | Required outcome |
| --- | --- |
| No accepted learned source | Report source unavailability; application obtains/trains one. |
| Bad hash, identity or malformed source | Hard error; no replacement discovery or fresh model. |
| Missing required store/query support | Incomplete prediction/replay; qualify another source or eligible prior. |
| Update exception after earlier accepted rows | Preserve diagnostic evidence and reconcile input admission; no blind batch retry. |
| Cursor does not match snapshot or stream | Application stops/reconciles before further feedback. |
| Invalid snapshot state or dependency | Hard restore error. |
| Architecture/resource infeasibility | Reject deployment independently of timing readiness. |

### 9. Verification and implementation handoff

Keep historical evidence immutable. Record new receipts under a separate evidence
directory with exact source revision, loaded native hash, dependency versions,
input hashes and commands. Compare against the prerequisite using unchanged tests
and actual native consumer paths; distinguish test mocks from real estimator calls.

Verify existing seeded numerical predictions, updates, configuration round trips,
checkpoint readers and Gym C/B+C results remain unchanged. Run required native
parity suites without golden changes. Verify source/bundle and snapshot failures
fail closed, old snapshots continue identically, and packaged examples use public
installed APIs. Measure preparation, source validation, snapshot I/O and host cost
where implementation changes affect them; do not interpret synthetic functional
results as hardware accuracy or performance qualification.

Pinned Dynamo compatibility checks must preserve the original adapter, engine
query and config files. Existing ADP capacity synthesis or coarse readiness gaps
are separate downstream findings when they reproduce on the control. No blanket
claim of full Dynamo deployment qualification follows from module-level checks.

## Part II. Estimator selection and default behavior

### 10. Existing defaults remain unchanged

`auto` searches OP → interpolation → regression. Denied fallback restricts an
explicit mode; it does not remove Auto's candidates. Invalid configuration must
not be reinterpreted as a selection miss. A selected regression can begin
untrained and receive its existing online fitting; offline execution still needs
usable coverage. Learned mode remains outside Auto.

An explicit learned source supports frozen evaluation, seeded online adaptation
or the existing causal selector under their established requirements. Raw source
paths and digests remain supported. A new bundle packages accepted evidence and
complete identity; it does not silently opt an existing caller into GBDT.

### 11. Agreed opt-in target

Prefer a verified matching A/B bundle for learned use. Keep the same model for
A→A+C or B→B+C. Applications without an accepted seed own initial collection and
training before constructing the learned runtime. Missing learned stores do not
initialize themselves from later feedback.

For independent replay, retain `on_tune: reject` and validate actual required
stores, query coverage, model architecture, hardware and memory. For continuation,
restore numerical state with `continue` and reconcile the application's separate
input cursor. Existing Dynamo configuration pass-through and Gym orchestration
remain usable; new source/snapshot use requires qualification of the selected
assets and workloads, not a mandatory protocol migration.

### 12. Compatibility checklist for opt-in adoption

- Omitted and explicit Auto behavior are unchanged.
- Existing raw-source controls, public calls, diagnostics and online checkpoint
  layout remain compatible; optional bundle pins are strict only where selected.
- Source and dependency validation is native-owned and performed outside hot queries.
- Readiness detail is additive; actual execution checks role/topology coverage.
- C input and initial training remain application-owned; no implicit collector,
  label generation, missing-store first fit or stream-offset management.
- Complete snapshots preserve seeded continuation and required timing dependencies.
- Onboarding preserves all six stages, accepted configurations, interpolation
  collection-quality checks and separate memory/coverage/accuracy evidence.
- Examples demonstrate a fully ready B→C→B+C route with trainer-free verification.
- Historical experiment data is not relabeled as qualification of this revision.
