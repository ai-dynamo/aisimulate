# FPM dataset move to `data/fpm/` (AISIM-2337)

Source: `nvidia/aisimulate-fpm-dataset` at `main` =
`dbfcdd9eb62ea733efbc3f0cbd78277f2960061a` (1,129 files, 1.50 GB). The migrated tree was
built locally by `aisim_core.data.fpm_migration`; nothing was written to Hugging Face.

## Result (exit gate)

| Check | Result |
| --- | --- |
| Source bytes equal the HF listing (LFS SHA-256 / git blob ID, 1,129 files) | pass |
| `data/**` files moved with unchanged size and SHA-256 | 1,056 of 1,056 |
| Other files unchanged (README, schemas, scripts, audit, legacy `catalog/*`) | 72 of 73; `.gitattributes` rewritten (LFS rules `data/` to `data/fpm/`) |
| Library `check_dataset` (every envelope artifact re-hashed, row counts, identity hashes) | pass |
| Every moved file declared by exactly one envelope with the same SHA-256 | pass (71 envelopes) |
| Catalog lists every current legacy configuration | 61 of 61, 61 unique IDs; coverage shards count 61 |
| Historical snapshots | 10 of 10 have envelopes; not in the default index |

`hash-audit.parquet` has one row per source file: source path, target path, SHA-256 before and
after, bytes. `audit-summary.json` has the rest (counts per model and hardware).

## What changed

* Path rule only: `data/<rest>` to `data/fpm/<rest>` (history: `data/fpm/history/...`). Directory
  names (`<org>--<model>`, `b300-sxm`, `pure-tp4`) are unchanged, so no byte, name or parallelism
  label is rewritten.
* New beside each leaf: `aisim-manifest.json` (`aisim-artifact-manifest/v1`, category `fpm`,
  purpose `execution-profile`) listing every file with role, SHA-256, bytes and Parquet rows.
* New: `catalog/dataset.json`, `catalog/fpm/index.json` (61 entries), `catalog/fpm/<hardware>.parquet`
  and `catalog/systems.parquet` from the library's `build_catalog` (90 KB in total).
* Library fix: Parquet artifacts that declare a non-`aisim-` schema may lack an embedded schema ID
  (the FPM libraries are unstamped and must keep their bytes).

## Findings that need a decision

1. **Embedded old paths.** 202 moved text files (all 71 leaf `manifest.json`, FPM metadata, import
   records) contain `data/<model>/...` strings. They stay unchanged to keep hashes, so
   `configuration_path` is now a logical key, and the physical path is `data/fpm/` plus the
   remainder. Consumers (AISIM-2338) apply that rule; there is no adapter here.
2. **History is not indexed.** Selecting a model and hardware must not pull superseded snapshots,
   and the library has no filter on snapshot status. Rebuild with `--index-history` to include them.
3. **Legacy catalog and tooling are stale after the move.** `catalog/index.json`, `fpm.json`,
   `measurements.json` (and Parquet twins), `scripts/manage_dataset.py` and the README still
   describe the old layout and are left unchanged. Remove or rewrite them with the consumer update.
4. **Directory names vs system IDs.** Paths keep `b300-sxm`; the DEP's `<hardware>` is the system
   ID (`b300_sxm`), which is what `identity.hardware` and the catalog use. Renaming the directory
   is a pure path change if wanted.
5. **Default fetch size is not bounded by role yet.** An envelope lists all files of a
   configuration. Median is 0.29 MB, but the largest is 517 MB and total is 1.44 GB, mostly
   `fpm/provenance` (1.08 GB). Role-based tiers in `data fetch` would fix this.
6. `deployment` is null (a recorded evidence gap): the legacy manifests do not say aggregated or
   disaggregated.

## Reproduce

```bash
cd python/aisim
M="python -m aisim_core.data.fpm_migration"   # with PYTHONPATH=src
$M download --revision dbfcdd9e... --dest SRC --listing-out LISTING.json   # read-only; needs huggingface_hub
$M migrate --source SRC --output OUT --listing LISTING.json --revision <full sha>
$M audit --source SRC --output OUT --listing LISTING.json --revision <full sha> --report-dir REPORT
$M plan-upload --source SRC --output OUT --listing LISTING.json --plan-out REPORT
```

## Upload plan (needs approval; not executed)

One commit of 2,201 operations, computed in `upload-operations.parquet` (summary in
`upload-summary.json`):

| Operation | Count | Bytes |
| --- | --- | --- |
| copy (LFS, server side, no upload) | 280 | 1,458,122,223 |
| add (non-LFS moved files, 71 envelopes, catalog, `.gitattributes`) | 865 | 40,454,598 uploaded |
| delete (old `data/<rest>` paths; old commits keep them) | 1,056 | 0 |

Old pins stay valid because every old path exists at `dbfcdd9e` and earlier commits.

Recommended: land on a new branch of the existing repo first, review, then merge by a reviewed
HF pull request (the maintainers' step, not the agent's):

```bash
# dry run (default): builds the 2,201 operations and sends nothing
bash -ic "uv run --no-project --with 'huggingface_hub==0.35.3' --with hf_xet --with pyarrow --with 'numpy>=2' \
  python benchmarks/evidence/fpm-unified-layout-20261107/apply_upload_plan.py \
  --tree <OUT> --branch unified-layout-candidate"
# execute (after approval): create_branch from dbfcdd9e + one create_commit with parent_commit=dbfcdd9e
... same command plus --execute
```

`apply_upload_plan.py` refuses `--branch main` and fails if the branch parent moved. If a new
repository is preferred (rename or org move is still open in the DEP), push `<OUT>` to it with
`upload_folder`; the tree is the complete post-move repository.
Rollback: delete the branch, or revert consumers' pins to the previous commit.
