# Revised stack descriptions

The review follow-up is included in PR1: stream bulk original-A manifests without
the small metadata limit, and preserve omitted worker timing/default aggregated
mode through learned onboarding. PR3 adds the oversized-manifest relocation
regression. Fresh checks on rebuilt native `d7c69b78` pass 435 Rust FPM tests,
26 public-API tests and 343 affected Python tests; see
[review-fix evidence](../gbdt_review_fixes_20261103/README.md). Earlier campaign and
wheel counts below retain their pre-review build identities. The synthetic
example retains its actual producing revision and unchanged artifact hashes.

The prerequisite is merged PR #658 at
`9f4819e2b77afb4798bc8133c42ccc1d1b045e5c`. These descriptions use the final
seeded scope. They do not claim a default change, native initial training,
a mandatory downstream migration, or improved prediction quality.

| PR | Branch | Commit |
| --- | --- | --- |
| 1 | `codex/gbdt-source-bundles` | `98a2998fc38263fed96bd71dd779717c95c3da40` |
| 2 | `codex/gbdt-native-bootstrap` | `bc35afdfb4159000dd616132cac085972c527916` |
| 3 | `codex/gbdt-complete-snapshots` | `3630fed8e30ec24edee78f569b9d7276e60ddb4a` |
| 4 | `codex/gbdt-adoption-examples` | Identified by branch and scope; its own commit is intentionally not embedded here. |

The PR2 branch name is historical. Runtime, legacy, consumer, real-data and
host-cost qualification accompanies PR3. PR4 adds example/installed-wheel evidence,
adoption guidance and this reference ledger. Reported tests qualify the assembled
stack at their recorded native hashes; they are not independent test totals for
each intermediate commit. Historical native-first-fit results remain in local
archive refs and separate storage, with original digests and paths preserved.

## PR1 — source bundles and seeded callers

**Title:** `feat: add pinned learned bundles and seeded workflows`

Learned sources currently require callers to manage their seed, deployment
identity and dependencies separately. Add verified source bundles and public
train/export/inspect/install helpers that return the canonical resolved config,
with exact-identity catalogue discovery and an explicitly qualified original-A
prior. Existing raw artifact/checkpoint configurations preserve their path/hash
contract, and Auto keeps its existing estimator order.

Pass complete seeded configurations through existing callers and optional
onboarding replay/serving validation. The six-stage onboarding workflow keeps its
interpolation choices, collection-quality assay and independent memory checks.
Applications continue to collect observations and perform initial fitting.

Assembled-stack validation includes 922 current SDK tests, 192 unchanged
prerequisite SDK tests, 1,079 caller tests and 1,916 onboarding tests. The final
optimized native reruns the 922/192 SDK tests and 22 learned-onboarding/dependency
checks. Counts overlap; exact scopes/builds and prior failure/fix records are in
`benchmarks/evidence/gbdt_seeded_revision_20261103/README.md`.

## PR2 — shared execution readiness

**Title:** `feat: expose learned execution readiness to consumers`

A model's coarse readiness does not prove that all workload stores required by a
deployment are available. Expose additive native execution coverage and dependency
details, use them in explicitly learned search and onboarding, and preserve
source/configuration errors as errors. Existing diagnostic meanings, public
signatures, seeded fitting and online checkpoint version-5 semantics remain intact.

The unchanged Gym suite passes 691 tests and 13 real-native scenarios; successful
receipts match baseline. The unchanged Dynamo adapter passes 31 unit tests and
11 shared native cases, plus three new bundle/snapshot pass-through cases. Three
pre-existing Dynamo failures reproduce on both builds and remain documented.

The full36 C/BC and HF smoke campaign preserves all 6,407,633 scored records'
prediction bits/availability and admission payloads, including all thirteen M3
cases. This demonstrates preserved numerical behavior and coverage limits, not
new accuracy. Those receipts identify native 7fe75475; later verifier/receipt
changes have source-delta proofs and final consumer/SDK reruns on 6a12ba8d.
Native engine/compile parity passes 389 tests without changing prerequisite goldens.

## PR3 — complete snapshots and continuation

**Title:** `feat: preserve complete learned snapshots and continuation`

A forest checkpoint alone loses calibrated fallback, selector and continuation
state, and source paths can prevent relocation. Save complete seeded model state
with its timing-source/prior closure checked against recorded hashes. Restore
fixed or continued state through the canonical constructor and expose owned
versus external dependencies. Preserve supported legacy snapshots and the existing
version-5 online checkpoint contract.

Reuse fresh construction-local verification receipts to avoid rereading already
checked checkpoint payloads. Source/identity and extra-dependency checks remain
strict; snapshots do not contain application feedback cursors or replace
architecture and resource validation.

Pre-review native 6a12ba8d passes 429 Rust FPM tests, 26 public API tests, 922 SDK tests,
22 learned-support tests, 57 snapshot tests and 192 unchanged prerequisite SDK
tests. Six actual old snapshot variants relocate and continue identically; old
version-5 online checkpoint continuation also passes. The source proof binds all
268 native build inputs to the tested implementation.

Five fresh paired host-cost trials per case show raw construction close to
baseline after duplicate scans were removed. Complete snapshots add 40–145 KB
in comparable cases, 0.41–0.87 ms median restore cost and 2.87–14.03 ms median save
cost. Warm SDK paired medians range 0.985–1.040 and retained training state matches
baseline. Synthetic size limits, I/O variation and unsupported baseline fallback
snapshot capability are explicit in the cost report; no universal performance
threshold is claimed.

## PR4 — optional adoption and packaged example

**Title:** `feat: add reproducible opt-in learned examples`

Provide a self-authored synthetic decode B bundle and an installed public-API
example that admits externally supplied C into that actual seed, then saves,
relocates and continues B+C state. Compare continuation with an uninterrupted
twin and bind a separate application cursor to the input and snapshot digests.
Trainer-free verification checks packaged bytes, provenance, complete decode
readiness and identical explicit/installed/discovered predictions.

Document application ownership of telemetry, rank synchronization, sampling,
deduplication, initial fitting, durable offsets and recovery. Keep Auto and the
six onboarding stages unchanged. The optional learned extra pins supported
scikit-learn 1.9.1; the base runtime does not need the trainer. This functional
example does not qualify serving accuracy or resource feasibility.

Recorded artifact regeneration uses producing PR3
`3e6e82c1b67e4448da01d60ae9d06893d1f8489f`, with the PR4 generator pinned by a
separate digest. All ten B→C→B+C lifecycle checks and seven packaged verification
checks pass. Verification passes with every sklearn import rejected, and altered
evaluation lineage is rejected under optimized Python. Generated bundle/seed/input
bytes match preparation exactly; only final build provenance changes. The final
wheel passes the installed end-to-end test outside the checkout, covering the
seeded source/snapshot workflow, public B+C example, trainer-free verification and
tamper rejection. Its native extension is byte-identical to qualified build
6a12ba8d. Documentation links, production Python lint and formatting pass.
