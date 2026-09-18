<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B200 TRT-LLM power import

The 18 tables below derive from [AIConfigurator commit
915f590680d8a79fe9c39f6f3a9ff13bc267fcce](https://github.com/ai-dynamo/aiconfigurator/tree/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/aic-core/src/aiconfigurator_core/systems/data/b200_sxm)
([upstream PR #1584](https://github.com/ai-dynamo/aiconfigurator/pull/1584)).
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
The upstream [Apache-2.0 license](https://github.com/ai-dynamo/aiconfigurator/blob/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/LICENSE)
applies. Root and packaged `THIRD_PARTY_NOTICES.md` also record this import.

Upstream paths start with `src/aisimulate_core/systems/data/b200_sxm/`.
Local paths are relative to this directory. Each table is at
`<family>/trtllm/1.3.0rc20/<table>_perf.parquet` under both roots.

| Family | Table | Upstream rows | Packaged rows | Positive power pairs |
| --- | --- | ---: | ---: | ---: |
| attention | context_attention | 69,419 | 70,847 | 69,404 |
| attention | generation_attention | 50,520 | 51,854 | 50,492 |
| comm | custom_allreduce | 69 | 69 | 69 |
| encoder_attention | encoder_attention | 7,679 | 7,679 | 7,653 |
| gemm | gemm | 130,156 | 130,156 | 129,256 |
| linear_attention | gdn | 1,882 | 1,882 | 1,725 |
| linear_attention | mamba2 | 685 | 685 | 684 |
| mhc | mhc_module | 139 | 139 | 139 |
| mla | context_mla | 1,760 | 1,760 | 1,759 |
| mla | generation_mla | 2,896 | 2,896 | 2,894 |
| mla | mla_context_module | 5,856 | 5,856 | 5,838 |
| mla | mla_generation_module | 8,831 | 8,831 | 8,831 |
| mla_bmm | mla_bmm | 848 | 848 | 825 |
| moe | moe | 218,295 | 218,295 | 179,033 |
| quantize | computescale | 1,628 | 1,628 | 1,623 |
| quantize | scale_matrix | 1,628 | 1,628 | 1,625 |
| sparse_attention | dsa_context_module | 14,640 | 14,640 | 14,625 |
| sparse_attention | dsa_generation_module | 11,040 | 11,040 | 10,947 |

Sixteen tables are byte-identical copies. The two attention tables are modified
derivatives: all upstream identities, schemas, timings and power measurements
are preserved, alongside 1,428 context and 1,334 generation identities already
present in AISimulate. Those 2,762 local rows retain their timings and use the
paired `0.0/0.0` unavailable-power sentinel. Another 43 upstream attention rows
already use that sentinel (15 context and 28 generation).

Across the 530,733 packaged rows, 487,422 contain positive power pairs and 43,311
use paired-zero sentinels. Latency is in milliseconds; `power` and `power_limit`
are watts per GPU. These are import-integrity facts, not hardware-accuracy
qualification. Collector V3 sidecars, reuse rules and evidence policy govern
collection and reuse.

`power_upstream/context_attention.parquet.source` and
`power_upstream/generation_attention.parquet.source` are unmodified copies of
the corresponding upstream attention tables. The `.source` suffix keeps them
out of runtime `*.parquet` discovery. Focused tests in
`tests/unit/tools/test_power_data.py` (relative to the Python package root) pin
the imported files and check both merges against these source copies. Existing
Parquet review tooling and packaged-data tests check the shared power-field
storage contract.

The retained local identity/latency pairs are independently pinned from AISimulate
commit `ffcb6576b3a60077ea1200788f1b784213f979cf`, immediately before import commit
`717f973bea4ebc07673192475b3a0f743d82c168`. The test records their digests and the
deterministic serialization used to reproduce them from that baseline.
