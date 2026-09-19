# B200 / vLLM 0.25.0 operator data — partial publication

This is a publication of existing production measurements, **not a new full
recollection**, and not a declaration that every planned case succeeded.
The data belong to the vLLM 0.25.0 image with source commit
`dd10e03f95f94edbea1975c67ace3a35ec9a8a40`.

## Format and provenance

- Same family/backend/version directories, filenames, columns, units and logical
  types as the previous packaged Parquet tables. UTF8 offset widths are aligned
  losslessly to the previous Arrow schemas. No timing is averaged or synthesized.
- Physically disjoint production shards are concatenated; duplicate identity keys
  fail publication validation. No smoke rows are inserted into performance tables.
- `collection_meta.yaml` uses the existing schema-2 multi-collection format to
  retain each original collector reference, content hash, case-plan hash, date,
  row count. Upstream runtime revision and observed per-shard failure counts are
  retained in the adjacent collection report (the public v2 sidecar schema does
  not admit those fields). Old measurements are **not** relabeled as
  collected by the smoke-test commit.
- Per the repository's provenance writer, a table status of `complete` means its
  collection finished; classified individual case failures are recorded separately.
  It does not mean every planned shape succeeded. The overall release is partial.
- `vllm-0.25.0-collection-report.json` lists table hashes, source jobs, coverage,
  withheld outputs and reuse. `vllm-0.25.0-failures.json.gz` preserves the original
  checkpoint failure records. The frozen source archive is SHA-256 identified in
  the report; original raw records and clock traces remain preserved in that archive.

## Historical environment evidence

The optional multi-node campaign runner and Slurm completion hook are no longer
part of this PR. They are not needed for normal prediction or direct
`collect.py` use. Their matching source/configuration/tests were archived
locally for future reuse; no new GitHub archive branch or package is published.

The report's `reproduction.artifact_location` identifies the immutable Git tree
`54b073e140219b6e923a57a09acd05e96797a901`. Its `runtime_manifest` and each
`historical_source_overrides[].patch_path` are relative to `python/aisimulate`
**in that historical tree**, not paths in the current checkout. Their SHA-256
values and the original measurement refs are unchanged. For example, retrieve
an evidence file with `git show COMMIT:python/aisimulate/PATH_FROM_REPORT` from
a clone containing that commit, and verify it against the report's digest.
The local archive contains the same bytes independently of Git history.

The generic hash-checked runtime-manifest interface remains available for direct
collection with an explicitly supplied declaration. The fleet default has not
been switched to 0.25.0; the removed campaign's runtime choice is not applied
automatically. This scope change neither remeasures nor requalifies the data.

## Frequency policy

The user approved Slurm `--gpu-freq=1965` plus observed clock sampling, matching
the targeted collection procedure. **Hard clock locking was not verified.**
Do not describe these results as newly verified hard-locked measurements.

## What is not included yet

- `compute_scale`: its 1,628 old successful measurements are preserved, but the
  latest fresh smoke found an existing producer/finalization defect: the callback
  also emits `scale_matrix_perf.txt`, which the collector does not register as an
  owned output. Five smoke callbacks succeeded, but finalization failed. Both
  outputs are withheld from this batch rather than suppressing the extra file or
  weakening finalization checks.
- DSA context, DSA generation and MSA generation: interrupted full sweeps are being
  continued from their checkpoints. They are not replaced by smoke data and are
  not marked complete here.
- CAR/NCCL: raw communication measurements exist but their independent publication
  and validation are pending; NCCL retains its actual library version, not the
  enclosing vLLM version.
- KDA: explicit user-approved reuse of existing `0.1.dev19262` preview data is
  declared in `kda/vllm/0.25.0/reuse.yaml`. Its 1,203 rows and original measured
  build are unchanged. They are not new vLLM 0.25.0 measurements.

Do not promote this partial release to a fully recollected default or describe a
prediction that still uses older fallback tables as exclusively new 0.25.0 data.
All publication and follow-up changes remain in PR #219.
