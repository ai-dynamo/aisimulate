# GLM circular-tail runtime candidate

Unqualified candidate. Formal FPM/Ops/HTTP admission remains closed. This combines the previously reviewed unaligned retained-pool repair with an explicit `KpoolTailSpec.uses_slot_mapping=False`. It does not modify attention metadata, block-table kernels, cache allocation, scheduling, forward/control loops or timing.

The original generic kernel then reads valid block 0 and emits padding for this group; the unchanged native KpoolTailMetadataBuilder computes each actual token's circular position from the first block. Native positions must be present in every admitted path; the positions=None branch is not qualified. The metadata builder, KDA and MLA caches remain unchanged.

Sources: vLLM https://github.com/vllm-project/vllm at immutable ced6857afa0ea7b2e3f0846a62e1394e90f15607, original paths vllm/model_executor/layers/sparse_attn_indexer_kpool.py and vllm/v1/kv_cache_interface.py. Copyright contributors to the vLLM project; modified by NVIDIA CORPORATION & AFFILIATES,2026. Apache-2.0; full LICENSE adjacent, original license/copyright retained and modifications marked. The upstream root has no NOTICE. Build driver is original project code, derived from its preceding checked-in driver.

The lossless combined-repair.patch.b64 is the build input; review.diff strips context trailing whitespace for review only. Identity pins both source files, complete patch bytes and actual new distribution version. build.py uses the verified original source archive and image-derived original binary wheel with upstream precompiled build, preserving all19 original native binaries and legal material. It never changes an existing installation. Build does not grant correctness or performance admission.

Required next gates: direct actual spec/group and positions witness; original/repaired slot input control; real ordinary128K and retained cohort; unaligned original suite, both precisions/TP2/TP4; capacity and5+10 state/graph qualification; new formal campaigns. Old runtime data remains quarantined and cannot populate the new identity.
