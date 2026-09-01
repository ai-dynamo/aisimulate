# AIConfigurator manual synchronization report

- From: `095f58a51c4ca8e61b66ec108d86f223f8d559ce`
- To: `ce2824e8abd9bef71c3b162f651e63704a0eb4c1`
- AISimulate base: `0f0f4b33d61283d7c05a95289ac9da58c57e92c2`

## `pyproject.toml`

Reason: AISimulate has one wheel manifest and a combined dependency set.

Upstream changed `plotext` to `<6`. The combined AISimulate manifest already
carried that bound, so no further manifest edit was required.

## `uv.lock`

Reason: regenerate from the combined AISimulate manifest.

The upstream lockfile change was the matching `plotext` constraint. The
combined AISimulate lock already resolved the compatible dependency set and
remained authoritative.

## `aic-core/rust/aiconfigurator-core/Cargo.toml`

Reason: adapt dependency and feature changes into `crates/core/Cargo.toml`.

The upstream core added `log = "0.4"`. The dependency was added to the
combined AISimulate crate and the repository `Cargo.lock` was regenerated.

## Migration-only adaptations

- Extended rename and copy headers are rewritten into the configured mirror
  paths by `scripts/render_aic_sync_patch.py`; a regression test covers the
  renderer behavior.
- AISimulate-specific packaging, active workflows, and generated root
  `CODEOWNERS` remain owned by this repository rather than copied from AIC.
- The performance-data reuse manifest and report were regenerated from the
  final AISimulate data tree. That correctly keeps the pruned vLLM
  `chunk_gated_delta_rule` lane absent; the imported AIC test is adjusted to
  assert the final data state instead of reintroducing the stale donor lane.
- The vLLM 0.20.1 generator golden allowlist includes the migrated
  `--gpu-memory-utilization` flag, whose rendering is separately covered by
  sweeper request tests.
