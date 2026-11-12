# Packaged operation data and the wheel (AISIM-2341)

Measured on the `simonec/2341-nonpython-consumers` tree (based on `simonec/unified-data`)
with `python3 scripts/ci/measure_wheel_data.py`; raw output is `wheel-data-estimate.json`.
Re-run it against a built wheel with `--wheel PATH` (authoritative) before Phase 6.

Method: every file the maturin `include` list ships from `src/aisim_core/` is
deflated (zlib level 6, as in a wheel). The native extension and the Python packages
are not rebuilt here; the native size is the 13.6 MB `_runtime.abi3.so` of an
unrelated dev build and is only an order-of-magnitude figure.

| Part of `aisim_core/` payload | Files | Bytes | Deflated bytes |
|---|---:|---:|---:|
| `systems/data/**` (production op data) | 1,589 | 122,361,067 | 109,047,469 |
| `systems/` other (specs, version tables, profiles) | 119 | 12,276,976 | 732,602 |
| Other included core files (model configs, schemas, md) | 141 | 8,218,627 | 295,843 |
| Total | 1,849 | 142,856,670 | 110,075,914 |

- Op data is 99.1 % of the deflated payload (109.0 of 110.1 MB). Parquet is already
  zstd-compressed, so deflate saves only 11 %: wheel size tracks raw size.
- Without `systems/data/` the core payload is 1.03 MB deflated; the wheel would be
  dominated by the native extension and Python sources.
- 110.1 MB is already 105 MiB, above PyPI's default 100 MiB file limit, before the native
  extension and Python sources are added. Confirm with `--wheel` on a real build.
- Per system (deflated): b200_sxm 19.6 MB, b300_sxm 15.9 MB, gb300 15.6 MB, gb200 15.3 MB,
  h200_sxm 14.0 MB, h100_sxm 14.1 MB, rtx_pro_6000_server 7.0 MB, l40s 4.8 MB,
  a100_sxm 1.3 MB, b60 1.3 MB. a30, a100_pcie, h100_pcie and l4 hold only placeholders.

## Real Hugging Face check (2026-11-07, read-only)

- `aisim data fetch --dry-run` against `nvidia/aisimulate-fpm-dataset@main` authenticated and
  resolved `main` to `dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`, then failed with 404 on
  `catalog/dataset.json`: the public repo has no unified catalog and no `op-latency` category yet.
  No op-latency download could be exercised for real.
- `--offline --dry-run` with a full-SHA pin and an empty cache made no request and exited 0, listing the
  missing files.
- The materialize step (`materialize_systems_root`) is verified only with a stubbed `ensure_data` in unit tests.
