<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime observation adapters

These are original AISimulate adapters written against inspected runtime interfaces. They contain no copied vLLM or Dynamo implementation. The immutable upstream revisions and installation-relative file SHA-256 values are recorded in each manifest. The version-specific source note is part of the hashed bundle. vLLM and Dynamo are Apache-2.0 projects; these references document the interfaces used, without redistributing their source.

`vllm-0.27.0.json` is the bundled selection. `vllm-0.28.0.example.json` is an explicit campaign-local example for its inspected runtime build. Freeze the latter with `load_instrumentation` and `freeze_instrumentation` into a fresh campaign directory, as shown in the [user guide](../../../../../../docs/fpm-self-service.md#resolve-cache-geometry-with-a-runtime-probe). Other runtime pins need their own source audit, manifest and adapter changes; changing the version string alone is insufficient. An example bundle is implementation evidence, not qualification of every model, backend, GPU or topology on that version.

## Runtime interface

The existing collector stages the frozen bundle on `PYTHONPATH` and supplies these environment variables:

| Variable | Contents |
| --- | --- |
| `AISIMULATE_RUNTIME_CONTEXT` | Absolute path to the runner-owned `aisimulate-runtime-probe-launch/v1` JSON document: `attempt_id`, `configuration`, `phase`, `bundle_sha256`, exact `launch`, and `expected_ranks`. Sampling is a separate field. |
| `AISIMULATE_RUNTIME_INSTRUMENTATION` | Absolute path to the frozen JSON `manifest.json`. |
| `AISIMULATE_RUNTIME_OBSERVATION_DIR` | The current process's results directory; the collector uses `/results`. |

Worker and scheduler classes inherit the native vLLM worker and Dynamo `InstrumentedScheduler`. The worker preserves return values and exceptions from memory profiling, cache initialization and warm-up. The scheduler calls the native constructor first. Neither intercepts timed forwards, selects kernels, changes scheduler settings, or changes sampling. Python modules are imported only by those runtime processes; bundle preview/freezing and observation import do not import them.

Both adapters export `ObservedInstrumentedScheduler`. The runner passes the manifest's `scheduler_class` unchanged to `--scheduler-cls`; both pinned Dynamo argument consumers require its string to contain `InstrumentedScheduler` before importing the class. Preserve this name constraint as well as native inheritance. Each version's source notes link the exact consumer.

## Observation meanings

| Field | Source, units and lifecycle |
| --- | --- |
| `runtime.version` | Installed `importlib.metadata.version("vllm")`, checked against both class version and manifest. |
| `runtime.source_revision`, `runtime.source_files` | Pinned interpretation revision plus SHA-256 of actual installed files, including the Dynamo scheduler and inspected backends. All must match. A supplied generated vLLM commit ID is checked and retained separately in `source_identity`; file hashes do not attest the entire source tree. |
| `launch`, `identity`, attempt/configuration/phase/bundle | Runner context copied verbatim. These declarations are independently checked against actual resolved configuration and hardware on import. |
| `resolved_config` | Actual initialized vLLM model/cache/scheduler/parallel/compilation/kernel/quantization/offload fields, including speculative and transfer configuration. No launch value substitutes for an unobserved field. |
| `model_config_sha256`, `loaded_model_config` | SHA-256 of the actual `hf_config_path or model` directory's `config.json`, or the local HF cache's `config.json` at `hf_config._commit_hash`. No network request is made. Missing files, non-HF formats and version-selected alternative configuration files remain unresolved. The launch passes the reviewed `--revision` to native model construction, including for local directories. Import requires the observed revision and independently checks the loaded config bytes; this binds configuration and the declared source revision, not weight bytes. |
| `model_config_source_files` | When the launch declares adjacent model-config sidecars, SHA-256 values are read from the actual config directory. Missing files remain unresolved; expected hashes are never echoed. The primary config bytes remain unchanged. |
| `hardware` | Worker device properties: CUDA major/minor combined as SM, actual device name and physical total bytes. The requested GPU label is retained separately for independent family/SM checking. |
| `cache.available_cache_bytes` | The byte allowance returned by native `determine_available_memory`, preserved after `super` succeeds. It is not post-warm-up free HBM. |
| `raw_cache_config` | Unnormalized spec fields and allocation declarations retained independently, including when a spec or layout cannot be represented. |
| `cache.num_blocks`, `groups` | Initialized `KVCacheConfig`: exact layer membership, raw spec class, logical block tokens, spec page bytes, dtype and retention. Spec page bytes come from the runtime property, including padding. |
| `cache.storages` | Unique `(CUDA device, untyped storage pointer)` with actual backing-storage bytes and a stable rank-local storage ID. Shared backing storage is counted once. |
| `cache.tensor_allocations` | Runtime `size/shared_by/block_stride/offset`, with the observed storage ID. Size is the entire backing storage; stride and offset are bytes. Zero block stride denotes an unpacked allocation. Runtime declarations must explain all aliases. |
| `cache.layer_tensors` | Every layer's actual tensor shape, byte strides, byte storage offset and element width. Block axis comes from its initialized backend's `get_kv_cache_block_dim`, never from matching a shape value. Kernel block size comes from the exact initialized runner: V1 uses `_kernel_block_sizes`, V2 uses `kernel_block_sizes`. The observer records the actual runner class and sizes, requires that implementation's source hash, and rejects unknown classes or missing initialized state. |
| `cache.pool_id` | Stable logical ID per DP worker/scheduler pair. Actual scheduler manager identity proves every group shares the same pool; process pointers are not compared across processes. |
| `cache.initial_free_blocks`, reservations | Initial block-pool count and free queue after native scheduler construction; set difference gives reserved IDs. These audited constructors reserve only the null block. Additional reservations remain unresolved until their permanence has a source mapping. |
| `lifecycle` | Workers record only after native warm-up returns successfully and cache initialization was observed. `capture_completed` means the runtime's selected capture policy completed, including its intentional eager/no-capture path. Actual graph policy and sizes remain in `resolved_config`. Schedulers record after native initialization. |

The supported representation is one shared HBM block pool with full/sliding retention, zero watermark, no speculative decoding, transfer or offload, and no compressed storage blocks. The existing audited `InklingConvState`/window-four mapping is preserved by the historical helper; any model-specific backend also needs its own declared source hash before this new bundle can complete. Unknown classes and retention are not renamed into supported specs. Actual prefix-caching configuration is retained; `semantics.prefix_reuse=false` does not authorize cross-request replay reuse.

Both logical K/V-plane-first and packed views retain their physical strides. The CPU importer independently checks affine byte bounds, pool/kernel block ratio, packed slot intervals, padding, declared aliases, reservations and compatible rank/phase settings. An adapter-supplied status cannot authorize import.

Independent collection errors are kept in `unresolved_fields` and `error` with available raw data. Duplicate records fail instead of overwriting earlier evidence. Native initialization/warm-up exceptions propagate unchanged. Formal collection also emits the unchanged `fpm-execution-worker-*.json` evidence for existing execution-quality consumers.

## Checks for a new campaign-local adapter

Check the frozen manifest's scheduler string through the pinned native benchmark argument consumer before direct class construction, and verify the exported class inherits the native scheduler. Exercise the runtime extension classes against small runtime-interface fakes before using GPUs. Cover actual backend block axes (including repeated dimension lengths), packed offsets, padded pages, declared aliases, source/version mismatches, actual config-byte hashes, absent fields, retention changes and native exceptions. Then use the normal probe command on the selected hardware and import both phases for every selected configuration. Preserve failed attempts and corroborating initialization logs. A source audit and fake-runtime test cannot replace that live campaign evidence.
