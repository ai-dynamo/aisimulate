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
