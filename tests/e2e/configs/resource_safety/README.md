# Reported Mac sweep regression

`reported-mac-sweep.yaml` preserves every behavioral field from the September
10 report: 10,240 input tokens, 1,024 output tokens, all seven concurrency
choices through 64,512, 100 requests per load unit, 72 GPUs, and the original
scheduler, memory-fraction, and optimizer domains. The normalized SHA-256 is
`af1a9d53df52c5eab2c07f9be20b69457e51ddd4c43933c42bcca7e489698577`.
No private crash log is included. The kernel-panic root cause was not established.

The historical eager prompt allocation at the largest candidate is
6,451,200 × 10,240 × 4 = 264,241,152,000 bytes (246.09375 GiB), excluding
reporting and engine memory. This is a calculated allocation, not measured crash
RSS. Do not reproduce it by running an unguarded historical binary.

Run the deterministic preflight tests with the resource-safety Python package:

```sh
python -m pytest -p no:timeout tests/test_reported_resource_regression.py
```

Those tests use an allocation sentinel to prove refusal precedes search and
runner creation. They are not native integration evidence. To generate that
separate evidence with the installed candidate AISimulate and Dynamo wheels:

```sh
python scripts/verify_reported_resource_safety.py --output-dir /tmp/resource-proof-new \
  --dynamo-wheel /path/to/ai_dynamo_runtime.whl --dynamo-source-dir /path/to/dynamo
```

The directory must not exist. Repeat on macOS and Linux. The script requires
native imports, a loaded binary that matches the supplied candidate wheel,
the lazy-allocation capability, and a completed small Dynamo replay before running the full
reported sweep and a deterministic largest-candidate variant through the public
`aisimulate recommend --stack dynamo` CLI. It applies a 2 GiB execution limit,
requires exit 3 plus a qualified resource plan, checks monitored peak RSS and
process cleanup, and records package versions, source revision/diff, native
binary hash, input identities, and runtime evidence. Missing packages, missing
preflight, timeouts, crashes, or watchdog-triggered memory overshoot fail; they
cannot produce a `safe_refusal` verdict.

Safe refusal is the acceptance target for this limited host. Full-scale
completion, exact original panic reproduction, and reporter confirmation are
separate claims. Preserve the manifest with build provenance for both wheels;
a package version alone does not identify a locally rebuilt native library.
This regression depends on the host-budget and supervision PRs, the Dynamo lazy
binding PR, and the bounded-reporting PR. Do not mark the integrated fix verified
until both platform manifests have been produced from the intended builds.

## Recorded macOS result

The [macOS manifest](evidence/macos-arm64.json) records an actual native run on
macOS 26.6.2 arm64 with Python 3.13.7. The eight-request native smoke replay
completed. Both CLI cases returned `resource_limited` (exit 3) with a qualified
6,451,200-request plan, before candidate execution. Both confirmed descendant
cleanup under the 2 GiB execution budget.

| Case | Observed child-process-tree RSS peak (bytes) | Wall time |
| --- | ---: | ---: |
| Full reported sweep | 163,037,184 | 0.512 s |
| Largest candidate | 148,455,424 | 0.487 s |

RSS values are polled observations of the supervised child tree, not a
measurement of total host memory or the historical crash. The coordinator's
separate memory allowance is recorded in the manifest. The only added full-sweep
configuration is the execution resource budget; simulation settings are intact.

Run identities:

- AISimulate CLI and evidence script: `912c7745d8b4fe3036af52a4d1a4613e15d6d7b8`.
- Dynamo Python checkout: `9e48b26837a088caa4d826c57d66f8079e83b43d`.
- Dynamo native wheel built from `d70c5b10ae0ab2f9ccd4708f92c098021cf82418`;
  the subsequent commit only updates Python tests. Embedded AISimulate core:
  `0480320fdd20c2e063a28a90faa111180fce8e47` from #173.
- Wheel SHA-256: `8d68d7271968a780705a76620843fbde46a85df516bc6230e9d223d35a6f7ef8`.
  The runner also verifies the installed Dynamo binary against the wheel member.

The [Dynamo build log](evidence/macos-dynamo-build.log) retains compiler output
with the local workspace prefix replaced by `$WORKSPACE`. Build from the stated
Dynamo native revision with Rust 1.96.0 and Python 3.13:

```sh
uv tool run --from maturin maturin build \
  --manifest-path lib/bindings/python/Cargo.toml --no-default-features \
  -j 2 -i /path/to/python3.13 --out /path/to/wheels
```

This was a development-profile, locally built Dynamo wheel with editable Dynamo
Python sources. The AISimulate Python package came from the recorded source
checkout; its separately imported, pre-existing local `_runtime` module is
identified by binary hash in the manifest and was not rebuilt in this run.
The smoke replay executes the new Dynamo native wheel and its pinned core.
These records provide local build evidence, not a release-wheel attestation.
Linux native qualification, dependency integration, and the reporter's rerun
remain pending. The evidence proves safe refusal on this Mac, not completion of
the large workload or the cause of the original kernel panic.
