# InferenceX cache to aisim-e2e/v1: full-cohort parity (AISIM-2332)

Inputs: private cache `ai-dynamo/aisim-infx-cache` at the pinned revision
`8ec77d043d236ae92aa0cee6a9896f8195ea7718` (7,743 observations, 187,347 metric rows, `e2e-source-index/v1`);
frozen legacy cohort: InferenceX `db-dump/2026-09-28` (85,309 benchmark rows) selected with
`gym-resolved-config-v3`. Nothing was written to Hugging Face; the converted dataset is local only.
Machine-readable results: [`data-parity.json`](data-parity.json), [`cohort-parity.json`](cohort-parity.json).
Reproduce with `aisim_core.data.migrate_infx` and `scripts/e2e_accuracy/migrate_infx_parity.py`.

## Result in one table

| Dimension | Pinned HF path vs raw-backed aisim-e2e | Typed tables only (after follow-up) |
| --- | --- | --- |
| Observation identities, record hashes, source config IDs | identical (7,743) | identical |
| Membership / snapshot | identical; legacy snapshot ID kept | identical |
| Metric values, units | 187,347 cached rows: max relative difference 0.0 | same |
| Provenance (run URL, image, date, raw record) | identical, raw bytes hash-linked | same |
| Selected cohort, exclusions, freshness and image decisions | identical (2,855 selected; 3,306 filter, 1,009 stale, 572 image, 1 row) | identical |
| Resolved deployment evidence (coverage) | identical for all 2,855 points (1,970 resolved) | identical (1,970 resolved; was 1,635 before the typed columns) |
| Reviewed MTP evidence bindings | 872 / 872 | 872 / 872 (was 160) |
| Score inputs (mean TTFT/TPOT/E2EL, per-GPU throughput) | identical (0 of 14,275 differ) | identical |
| Predictions and scores | not re-executed (no wheel); inputs digest-identical, see limits | coverage equal; still not re-executed |

The local cache copy equals what the current HF reader downloads from that revision (`live_current_path`).

## Decisions (accepted) and what changed in the follow-up

1. **Identity hash v3.** `dp_attention` (per-role boolean, never a degree), `multinode` and `offload_mode` are
   first-class `Identity` fields; v1/v2 IDs are unchanged (later fields are excluded from earlier hash versions).
   `identity_settings` keeps `spec_method` and worker counts.
2. **Parallelism reported under metrics.** `pp`, `pcp_size`, `dcp_size` (and per-role variants) are identity
   dimensions `pp`, `cp`, `dcp`; one source configuration can span several `configuration_id`s (2 cases).
3. **Telemetry.** 21 source configurations differ only in the GPU-telemetry list `workers`; 4 `configuration_id`s
   merge them. `source_config_id` per observation keeps the upstream grouping.
4. **Typed columns.** `aisim-e2e-observations/v1` gains `recipe_fingerprint`, `source_config_json` (verbatim
   upstream configuration record, including telemetry and the zero-degree row) and `unmapped_metrics_json`
   (source metrics without a metric row, verbatim types). A typed-only reader now reproduces the legacy tables
   exactly (0 field differences in configs, runs and benchmark rows; selected-row differences 0; cohort input
   digest equal). The raw-backed view remains as the reference.
5. **Raw bundles.** One gzip bundle per configuration (1,546 raw files instead of 12,401; 6,204 files in the
   dataset). Each observation keeps `raw_offset`/`raw_length` and the SHA-256 of its own record, so every row is
   audited individually (tampered range or hash is rejected by the reader).
6. **Result: 1,546 configurations** (cache: 1,565 source configurations).

## Differences, all explained (data level, `unexplained` is empty)

| Code | Count | Reason |
| --- | --- | --- |
| `identity.configuration_ids_new` | 1,546 | `cfg-<sha256>` replaces `hf:<sha256>` |
| `identity.configuration_merge_telemetry` | 4 | see 3 |
| `identity.configuration_split_parallelism` | 2 | see 2 |
| `metric.timing_boundary_null` | 187,347 | cache stamped "unknown"; schema stores null |
| `metric.explicit_missing_rows` | 10,942 | required statistic not reported is stored `missing`, never zero |
| `metric.interactivity_added` | 41,154 | `*_intvty` was dropped by the cache normalizer; now mapped, values verbatim |
| `metric.source_key_parallelism` | 9,081 | degrees moved to identity (see 2) |
| `metric.source_key_unmapped` | 78,402 | 93 other source keys (power, qps, token counts, hit rates, ...) have no metric row; typed `unmapped_metrics_json` and the raw bundle keep them, the cache had no row either |

Converter fixes found by the parity run: `p99.9_*` statistics were unmapped; one gb200 row reports decode tp/ep = 0
(treated as unknown, with an evidence gap).

## Remaining gaps

- Predictions and scores were not re-executed (no wheel build here). Inputs are digest-identical for all 2,855
  selected rows and resolution matches for all points (1,970 resolved, 885 unresolved, same as the legacy path), so
  a campaign run on the converted data is still the final proof.
- `source_config_json` and `unmapped_metrics_json` are verbatim JSON text, not columns per field; fields worth
  querying (power, qps, cache hit rates) have no metric rows yet.
- The converted dataset is local only; nothing was published.

## Legacy cohort (db-dump 2026-09-28, v3: 3,320 selected)

The HF path and the migrated dataset give identical results against it (same numbers in both blocks of
`cohort-parity.json`), so none of these differences comes from the migration:

- 7,135 observations shared; 78,174 database-only (older history), 608 HF-only.
- Selected in both: 2,829 (matches the earlier review); database-only 491 (433 not in the HF current view, 58 stale
  there); HF-only 26 (9 absent from the dump, 17 where the dump superseded the image).
- 223 shared rows differ in non-transport fields: 217 have `offload_mode` `on` in the API but absent in the dump,
  6 lack producer run metadata on the HF side. These were already outside the reviewed 2,829 overlap.

## Rollback

The cache is untouched. Rollback is the previous adapter plus the existing pin
(`.github/e2e-accuracy-dataset.json`); old HF commits keep the old paths.

Converted datasets (local, not published): pinned revision 1,546 configurations, 7,743 observations, 239,443 metric
rows; current `main` (37cdb031, 8,007 observations) converts with no unexplained difference either.
