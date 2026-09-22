---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Role-specific Context Limits
subtitle: Recommendation precedence and selection contract
---

## Contract

`engine.context_length` is the shared fallback. In disaggregated mode,
`engine.workers.prefill.context_length` and `engine.workers.decode.context_length` override it for
their respective workers. Every supplied value must be a positive integer. Aggregated workers do
not accept a worker-level override because there is only one engine limit.

The same effective value is used in two places:

- KV-feasibility filtering during parallel-shape enumeration.
- The worker payload's `max_model_len`.

When an override is absent, behavior remains the legacy shared-limit behavior. When the shared
limit is absent, the resolved model metadata supplies the fallback. A role-specific value may be
used even when model metadata has no shared maximum; this is required for models whose serving
configuration declares independent worker limits.

## Provenance

The model fallback is `max_context` from the resolved AISimulate model metadata. It is not a
measured performance datum and does not change the performance model. Role-specific limits are
operator-declared serving constraints, validated as positive integers at the public configuration
boundary. The recommendation layer copies those values into the search space, and the search
layer passes them to the existing KV-capacity estimator without changing its equations.

## Selection Evidence

The machine-readable comparison below is the acceptance oracle for the selection contract. `before`
is the shared-limit behavior; `after` is the role-aware behavior introduced by this change. The
listed candidate IDs are abstract parallel-shape identities, so the fixture is deterministic and
does not depend on checked-in performance data.

```json
{
  "schema_version": 1,
  "cases": [
    {
      "name": "aggregated",
      "before": {"limit": 1000000, "selected": ["agg:tp1"]},
      "after": {"limit": 1000000, "selected": ["agg:tp1"]}
    },
    {
      "name": "disaggregated",
      "before": {"limit": 1000000, "selected": [], "candidate_count": 0},
      "after": {
        "limits": {"prefill": 64000, "decode": 128000},
        "selected": ["pd:prefill-tp1/decode-tp1"],
        "candidate_count": 1
      }
    },
    {
      "name": "heterogeneous_disaggregated",
      "before": {"limit": 1000000, "selected": [], "candidate_count": 0},
      "after": {
        "limits": {"prefill": 64000, "decode": 128000},
        "selected": ["pd:h200-prefill-tp1/gb200-decode-tp1"],
        "candidate_count": 1
      }
    },
    {
      "name": "afd_companion",
      "before": {"limit": 1000000, "selected": [], "candidate_count": 0},
      "after": {
        "limits": {"companion": 128000},
        "selected": ["afd+pd:decode-companion-tp1"],
        "candidate_count": 1
      }
    }
  ],
  "anomaly_checks": [
    "role limits are positive",
    "unset role limits fall back to shared limit",
    "aggregate worker overrides are rejected",
    "role-aware selection does not silently use the shared limit"
  ]
}
```

The executable regression coverage is `test_role_context_limits_work_without_shared_model_context`
in `python/aisimulate/tests/unit/test_modern_recommend_context.py`. Run the focused evidence check
with:

```bash
python/aisimulate/.venv/bin/pytest -c python/aisimulate/pytest.ini \
  python/aisimulate/tests/unit/test_modern_recommend_context.py -m unit -p no:timeout
```
