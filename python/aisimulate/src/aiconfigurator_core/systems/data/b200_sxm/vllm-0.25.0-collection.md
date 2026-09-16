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
