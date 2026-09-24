<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GLM-5.3-Flash independent holdout acceptance

`glm53flash_validation.py` reconstructs native forward latency from collector
receipts and calls the installed public `RustForwardPassPerfModel` for every
frozen holdout point. It accepts no supplied scores or prediction arrays.
Synthetic unit fixtures test this protocol; they are not GPU acceptance data.

Run with a noneditable AISimulate wheel installed, using the collector checkout
on `PYTHONPATH` without its `src` directory:

```sh
PYTHONPATH=/path/to/aisimulate/python/aisimulate python -m collector.fpm_forward.glm53flash_validation \
  --manifest /results/holdout-campaign.json --output /results/acceptance.json
```

The command verifies the loaded public SDK and native extension against the
installed wheel's `RECORD`, including every hashed `aisimulate_core` payload.
It records that payload digest. An editable checkout cannot certify installed
consumer acceptance. Exit code zero requires all 16 phase cells to pass;
`FAILED` and `NOT_EVALUATED` return two.

## Manifest from native producer outputs

All relative paths resolve against the campaign manifest. Each entry pairs one
calibration plan/cell with a separately collected holdout plan/cell. Populate
`plan` from the actual frozen collection-plan JSON, `cell_id` from its cells,
and `attempt_id`/`raw_root` from that collection attempt. Both plans must freeze
explicit benchmark points and distinct input corpora with their real SHA256s;
the holdout plan must declare `options.dataset_role="holdout"`. Do not manufacture
plans after observing timings. Generate a file receipt with
`hashlib.sha256(Path(path).read_bytes()).hexdigest()`.

```json
{
  "schema": "glm53flash_independent_holdout_v1",
  "mode": "fpm",
  "entries": [{
    "calibration": {
      "plan": {"path": "calibration/plan.json", "sha256": "<actual-file-sha256>"},
      "cell_id": "<actual-prefill-cell-id>",
      "attempt_id": "<actual-attempt-id>",
      "raw_root": "calibration/cell/raw"
    },
    "holdout": {
      "plan": {"path": "holdout/plan.json", "sha256": "<actual-file-sha256>"},
      "cell_id": "<actual-prefill-cell-id>",
      "attempt_id": "<different-actual-attempt-id>",
      "raw_root": "holdout/cell/raw"
    },
    "consumer_config": {
      "backend_version": "0.30.0",
      "systems_paths": ["calibration-systems"]
    },
    "consumer_data": [
      {"path": "calibration-systems/<actual-file>", "sha256": "<actual-file-sha256>"}
    ]
  }],
  "http_metrics": []
}
```

Provide all eight backend/precision/TP configurations, each with separate
prefill and decode entries: vLLM 0.30.0 or SGLang 0.5.20, original FP8 or NVIDIA
NVFP4 checkpoint, TP2 or TP4. Missing entries remain visible as unevaluated cells.
The consumer model/checkpoint, GB300, pure TP, FP8 KV, strict measured database,
no fallback, no MTP/EPLB and estimation mode are fixed by the validator.
Additional public consumer selection options can be supplied in
`consumer_config`; conflicting fixed settings and online estimator tuning fail.

`consumer_data` must cover every parquet/YAML/JSON/text data or configuration
file under the explicit roots, not merely the main table. For FPM,
`fpm_forward_perf.parquet` rows for the selected cell must exactly match the
native calibration plan, attempt, runtime/grid, corpus/tokenizer, execution
identity, requested geometry and observed latency. Extra or missing points,
duplicate rows, renamed attempts and substituted scores fail validation.

Native readers verify the producer's full hybrid-state protocol before returning
measurements. vLLM token-stream receipts and SGLang request manifests supply
actual request IDs. Corpus hashes, request IDs and workload geometries are
required to be disjoint globally between calibration and holdout, including
across configurations. Geometry is phase/batch/total query/total cached tokens.
Raw file digests and native runtime identities are retained in the report.

## Gates and reports

FPM requires per-configuration, per-phase MAPE at most 10%. Explicit `mode="ops"`
uses 20% and `op_level` predictions. Ops additionally requires the Ops producer's
`collector.glm53flash_validation.bind_calibration(paths, frozen_run,
native_receipt)` adapter to verify its calibration evidence. Without that
adapter/evidence, the result is `NOT_EVALUATED`; generic operator rows cannot
certify GLM Ops acceptance.

Every frozen holdout point stays in the report and coverage denominator. Each
phase needs full compared-point coverage and requested points in each context
band: 1K–32K, above 32K through 64K, and above 64K through 128K. Context length is
(query + cached tokens) / batch; below-1K points are reported separately. These
labels describe bands, not a requirement to pick exact boundary lengths.

Metrics include MAPE, WAPE (sum absolute error / sum observed latency), nearest
rank P95 absolute percentage error, maximum absolute percentage error, requested
and compared counts, and coverage. They are computed separately per phase and
per context band. Missing native artifacts or an unavailable installed consumer
remain `NOT_EVALUATED`; invalid evidence, failed prediction or a missed gate
is `FAILED`. No missing or failed point is deleted to improve a score.

Optional `http_metrics` contains independently hashed JSON receipts. They are
preserved in a separate, unevaluated HTTP end-to-end section and never enter
native-forward metrics or the FPM/Ops gate. An empty manifest yields 16 explicit
`NOT_EVALUATED` cells and cannot report a pass.
