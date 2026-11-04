# Review-series disposition

The series remains open for owner review. No PR is merged by this campaign.
Cost qualification and exact-head CI are reported separately from code completion.

| Request | Disposition and evidence |
|---|---|
| Stack #664 on #676 | Completed against pinned `69232f77ce94de85e8ca7c40da8c2e484a6bdff4`; after #676 merged as `eb04c6888cf33ef4f48a995511a23429127f3baa`, #664 was rebased and retargeted to `main`. The original #664 head is preserved on `simonec/pr664-before-model-history-20261103`. |
| Model-owned optional history | `--include-long-history` defaults off. Twenty-four normal plan fingerprints match pinned #676. Correlated numerical declarations reuse model geometry and precision; the enabled mode adds declared points before representative selection. Model configuration, effective serving recipe and target workload are documented in [the audit](README.md) and [model evidence](model-declaration-evidence.json). |
| Stable IDs and resume | Preserve #676's device-independent case identities and persisted schema. No memory-filtered manifest is restored inside task IDs. Coverage mode propagates through plans, campaign wrappers and worker rows; tests cover same-mode and normal-to-expanded selection. |
| Collection-cost totals | [The audit](README.md) freezes B/S/M inventories on one baseline and separates history from retained non-history changes. Actual system-specific samples, paired initialization costs, failures and uncovered strata determine which GPU-hour estimates can be published. Missing estimates remain unknown; case-count reductions are not claimed as time savings. |
| SGLang B52 serving shape and consumer | [Native buffer evidence](../pr664-padded-serving-b300/README.md) compares 52 real requests plus four sentinel slots with the installed serving runner. Real B52 remains the persisted consumer coordinate. The native Rust consumer regression passes; #665 replaces the affected 44 rows. This is not a full-model end-to-end qualification. |
| GLM shard selection | Explicit checkpoint selection precedes deduplication. #665's recipe selects `nvidia/GLM-5.2-NVFP4`, enables optional history and uses matching filters. Tests verify nonempty disjoint full/skip-indexer populations. The recipe requires #664's collector changes. |
| Model context limits and memory | Pinned #676 supplies model-bounded plan points and runtime classified capacity refusals. Optional declarations validate context/query/prefix and decode `past + 1`. Runtime KV-pool and kernel limits remain device decisions. B300 recollection does not establish capacity on other systems. |
| MoE refusal and continuation | [H100 evidence](../pr664-h100-moe-capacity/README.md) includes a proven resident-weight refusal before allocation, an actual FP8 workspace failure, and a safe group that persists its result before normal worker recycling. No speculative workspace estimate or silent grid filtering is introduced. |
| SM90 FP8 head-512, n4/KV1 | [Native comparison](../pr664-sm90-fp8-native/README.md) reproduces the FA4/CuTE stride failure through native `Attention.forward` and the collector in both prefill and decode. No workaround or performance rows are added. Existing passing n16/KV2 evidence remains; n4/KV1 is an explicit native-framework qualification gap. |
| Timing boundary and attribution | Collector documentation identifies native QKV projection, fresh inputs, graph/eager policy and upstream sources. Graph-eligible SGLang native Tensor Core prefill raises `MeasurementBoundaryError`; eager timing applies only to graph-ineligible cases. The paired timing experiment is preserved with #665. The remaining unisolated indexer difference is a stated limitation. |
| #665 overlaps and provenance | [Normalization and validation](https://github.com/ai-dynamo/aisimulate/tree/bfdb5f727ad601c9dde43933f6e18f2d97aad1d4/benchmarks/evidence/collector/pr665-b300-dsa-boundary) preserve the original append and superseded rows, reconcile 308 overlapping physical coordinates, and retain 6,312 unique coordinates. Only 44 affected B52 rows are recollected using signed collector `c4b44d81f526b3c5299f7c15ccafe02d90b39229`. Global loader precedence is unchanged. |
| #666 provenance and page layout | [Durable original evidence](https://github.com/ai-dynamo/aisimulate/tree/a73e3b57800a9df71e16c3bb7ac2aa9ac6f91c19/benchmarks/evidence/collector/qwen35-page-layout-20261102) preserves the historical 95-row table and sidecar byte-for-byte. Thirty paired page-layout checks support the isolated mapping with median ratios 0.9938–1.0082; they do not qualify all rows or replace full TP8 end-to-end evidence. |
| Donor-priority redesign | Filed separately as [#714](https://github.com/ai-dynamo/aisimulate/issues/714), with Qwen and non-Qwen examples. No global loader or donor-priority redesign is included in these data PRs. |
| SGLang Blackwell KDA verify | The pinned 0.5.21 image lacks the Kimi-K3 verify module for sampled B200/B300 widths 2–8. These strict zero-row failures remain a native verify qualification gap; do not generalize them to KDA context/generation or substitute another kernel. The inherited normal coverage is preserved. |
| TRT large-context pilot omission | The preexisting H20 offset warning is an unverified causal inference. B4/Q32/d256/64K remains planned but unmeasured on H100/B300; the bounded pilot uses 16K/32K. This omission is not a planner filter or a zero-cost success. |
| Unsupported model routes | New optional Kimi MLA points are deferred because the generic DeepSeek adapter does not implement native Kimi NoPE/cache/output-gate semantics. Existing #676 defaults and KDA additions remain. Other unqualified AgentX routes are listed in the audit; omitted coverage is not equivalent-workload savings. |

Validation before the final audit additions: **4,699 collector tests passed,
31 skipped**; the native Rust B52 consumer test and #665 canonical data validator
passed. After integration with current main, 763 focused coverage/planner/audit tests passed.
The final audit helpers passed all41 focused tests; lint and copyright checks
passed for all1,763 tracked source files. New pushes require
fresh exact-head Full CI and GitHub-verified signatures. #665 and #666 require
the final #664 collector before merge. Infrastructure failures before checkout
must be retried without claiming that the affected tests ran.
