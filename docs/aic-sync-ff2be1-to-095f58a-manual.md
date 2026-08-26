# AIConfigurator manual synchronization report

- From: `ff2be1fd434fd516474e42b77f94cd5a5f841b9b`
- To: `095f58a51c4ca8e61b66ec108d86f223f8d559ce`

## `.github`

Reason: Only AISimulate repository-root workflows are active.

AISimulate disposition:

- The imported workflow/action snapshots remain inactive under
  `python/aisimulate/.github/`; applicable platform-wheel behavior is adapted
  to the unified `aisimulate` wheel.
- Active repository-root CI remains AISimulate-owned and validates the
  two-artifact release contract.
- The daily manifest rename is reflected by the canonical
  `perf_data_reuse_manifest.yaml`, its generator, and its path-trigger tests.

Changed upstream entries:

- `M	.github/actions/build-platform-wheel/action.yml`
- `M	.github/workflows/build-platform-wheels.yml`
- `M	.github/workflows/build-test.yml`
- `R076	.github/workflows/op-kernel-source-manifest-daily-run.yml	.github/workflows/perf-data-reuse-manifest-daily-run.yml`
