# FPM consumer parity on the migrated tree (AISIM-2338)

Tree: the AISIM-2337 migration of `nvidia/aisimulate-fpm-dataset@dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`
(1,129 files, 1.50 GB, local, read only; nothing was read from or written to Hugging Face). Both sides ran on
this one tree with `python -m scripts.fpm_accuracy.parity_unified_layout --tree <tree>`:

* **old**: a symlink view in the legacy layout (`data/fpm/<rest>` shown as `data/<rest>`, legacy
  `catalog/index.json`), the whole tree, `HfDataset` in `legacy` layout;
* **new**: `HfDataset.from_hub` against a local transport: catalog read, configurations selected, only the
  selected envelopes' files fetched through the library with role `fpm-provenance` excluded.

Both sides use the same repository label and revision placeholder, so case IDs (which embed them) are comparable.

## Results

| Check | Result |
| --- | --- |
| Selected libraries: configuration IDs, paths, snapshot IDs, order, manifest SHA-256, worker config records | 61 of 61 identical |
| Truth membership and hashes, helper files, FPM artifacts and sidecar hashes, observation IDs and values, case IDs, membership digests, warnings, provenance links (old URL with `data/` to `data/fpm/`) | identical for 61 configurations: 65 truth files, 175 helper files, 74 FPM artifacts, 2,155,267 observations |
| Files read by the new pipeline | 354 MB of 1,498 MB; no `fpm-provenance` file (1,086 MB) was fetched or needed |
| Legacy catalog counts (61 configurations, 10 history manifests) | match |
| History | old read fails closed (the legacy catalog lists 2 DSV4.1 history manifests that no current manifest points to); new read lists the 8 pointer-reachable snapshots, all from catalog-listed manifests |
| Profile and recipe pins (`data/fpm/` path exists, SHA-256 equal, also present in the old view) | 266 files pass; 15 files of `systems/dsv41_fpm_hf.json` stay on the retired repository (see below) |
| Evaluation rows (scores, coverage counts), 6 configurations, old versus new, full JSON equality | identical |

`parity-full-resolution.json` is the full run (769 s of case resolution); `parity-scoring-sample.json` is the
scoring run with the new side limited by `--configuration` (18 MB fetched for 6 configurations).

## Limits and findings

* Scoring ran on 6 of 61 configurations (the smallest with measurements, vLLM and SGLang), with the shared
  development environment: regression, warmup and nowarmup predicted real points (for example 1,356 of 1,356 and
  1,336 of 1,356); `learned` was unsupported (scikit-learn is not installed there) and one MiniMax
  warmup/nowarmup pair reported `predictor_error` on both sides. Equality of rows includes those statuses. The
  other 55 configurations were compared by their case (membership digest, hashes, observations), from which
  scoring is a deterministic function; they were not scored.
* `systems/dsv41_fpm_hf.json` (3 quarantined or development history profiles) pins `b35883ee`. Its metadata
  sidecars were rewritten in later dataset commits and the history directory was renamed, so those sidecar
  bytes no longer exist in the migrated tree. The file keeps the retired repository and commit.
* The migrated catalog still names `nvidia/aisimulate-fpm-dataset` as `dataset_id`; `HfDataset` accepts it for any
  requested repository. Regenerate the catalog with the target repository when publishing.
* The 2 orphan history manifests above are a property of the legacy catalog, not of this change.
