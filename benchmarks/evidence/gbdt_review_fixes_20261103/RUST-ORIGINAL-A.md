# Original-A bulk manifest regression

The original-A observation manifest is bulk input, but source dependency collection and portable remapping read it with the 16 MiB metadata reader. This rejected a valid legacy mixed-fallback configuration before construction completed.

## Reproduction and checks

All commands ran in `work/gbdt-implementation` on base `497480757978d631433b26c5daa39b85e462a339`, with the environment and exact command recorded in the adjacent JSON receipts. This report records focused correctness checks, not performance qualification.

| Receipt | Result |
| --- | --- |
| `rust-original-a-before.json` / `.log` | Before the production fix, the new real-observation manifest over 16 MiB failed public legacy construction with `manifest exceeds 16 MiB`; cargo exit 101. |
| `rust-original-a-after.json` / `.log` | Initial compilation caught a missing `Option` unwrap in the new test; no tests ran. Retained as an intermediate attempt. |
| `rust-original-a-after-final.json` / `.log` | `cargo test -p aisimulate-core --lib original_a -- --test-threads=2`: 11 passed, 0 failed. Ten matching FPM tests plus one unrelated scheduler test matched the filter. |

The negative test was initially named `large_original_a_legacy_construction_and_snapshot_relocation`; after reproducing the failure, it was split to keep PR1 and PR3 independently applicable. The final PR1 test uses the same oversized real observation fixture and public constructor, verifies the recorded manifest length and digest, and verifies portable source/seed references. The PR3 test saves and moves the complete snapshot, deletes all original fixture inputs, restores, compares prediction, and verifies dependencies.

The focused parser tests cover complete byte hashing and length (including whitespace), valid extra reference metadata, duplicate keys in root/provenance/reference objects and discarded nested observations/metadata, malformed input followed by an unread tail, missing references, trailing non-JSON content, changed bytes, and the unchanged 16 MiB limit for small metadata. Existing matching mixed-fallback tests also cover corrupt/inconsistent/missing sources and frozen/online seed qualification. The only compiler warning is the existing unused `TimingEvidenceAccumulator::slot_layouts` method outside the changed FPM code.

## Implementation and cost boundary

`source.rs::prior_dependencies` and `portable_prior_remaps` now use `source/original_a.rs`. A 64 KiB buffered reader computes SHA-256 and byte length from the same opened stream that is parsed. It retains only source/seed dependency references, while consuming each accepted observation immediately. Parse errors drain the remaining bytes before digest verification, so malformed prefixes cannot bypass verification of a changed tail. Reference digests, dependency remap checks, and subsequent semantic original-A qualification remain in force.

The projection's retained memory scales with dependency-reference bytes, keys in currently open objects, the parser's largest scalar buffer, and nesting state; it does not retain an accepted-observation `Value` population. Work is linear in input bytes plus object-key duplicate checks. This changes construction, snapshot packing, and explicit dependency verification only; prediction and update paths are untouched. The existing mixed-policy semantic loader still decodes the observations it needs for qualification; this fix does not claim to remove that existing population allocation. Small bundle/catalogue/checkpoint/artifact metadata continues to use the unchanged 16 MiB reader.

## Restack ownership

- PR1: `source.rs` changes; new `source/original_a.rs` including its four unit tests; `large_original_a_fixture()`; `large_original_a_legacy_construction_and_portable_references()`.
- PR3: `large_original_a_complete_snapshot_relocation()` only, reusing the PR1 fixture helper.

The PR1 pieces were applied unstaged to `work/gbdt-review-restack`, preserving the existing staged PR1 and other workers' Python changes. No commits or branch mutations were made by this task. `rust-original-a-source-hashes.json` records the final official Rust source/test bytes before the parent's broader checks and rebuild.
