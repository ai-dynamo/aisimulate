<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

AISimulate contains source material derived from the projects identified below.
These notices apply only to the identified third-party material. Dependencies
installed separately by package managers are governed by the license material
distributed with those packages.

Unless otherwise stated, AISimulate file paths in this document are relative
to `python/aisimulate/` in the repository source tree.

## AIConfigurator

The repository-root `.coderabbit.yaml` is adapted and modified from
AIConfigurator's `.coderabbit.yaml` at commit
`13b5cf2697876692b0a52098266c81162add11fc`. The original imported policy is
preserved at `python/aisimulate/.coderabbit.yaml` as migration provenance.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/blob/13b5cf2697876692b0a52098266c81162add11fc/.coderabbit.yaml

Pinned upstream collection and preserved-file SHA-256:
`5fb7a61a53f71f476169fa8e2419d3073c8d7e96206d8986b7d4fbb0f11fbcdd`

AISimulate-modified root overlay SHA-256:
`a70a21425533b419feaafcec67b176b3048a0893c4833bb802f46dc6e9a8f57e`

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This material is licensed under the Apache License 2.0. The upstream license
at the identified revision is available at:
https://github.com/ai-dynamo/aiconfigurator/blob/13b5cf2697876692b0a52098266c81162add11fc/LICENSE

The 18 B200 TensorRT-LLM 1.3.0rc20 performance tables under
`src/aisimulate_core/systems/data/b200_sxm/*/trtllm/1.3.0rc20/` include
power measurements derived from AIConfigurator commit
`915f590680d8a79fe9c39f6f3a9ff13bc267fcce` (PR #1584). Fourteen tables remain
unmodified, byte-identical copies after the September 17-18 refresh. The
context-MLA table was replaced with locally collected measurements, and the
MoE table has locally refreshed MXFP4 rows with unavailable-power sentinels.
The context-attention and
generation-attention tables are modified derivatives: AISimulate preserves
newer local timing rows and adds the typed `0.0` / `0.0` unavailable sentinel
to those local-only identities. Source paths, row counts, measured coverage,
and merge details are recorded in
`src/aisimulate_core/systems/data/b200_sxm/README.md`. The two unmodified
upstream attention copies under
`src/aisimulate_core/systems/data/b200_sxm/power_upstream/` support focused
import regression tests, which pin source and packaged SHA-256 digests.

The corresponding energy expectations in the repository-root file
`crates/core/parity_tests/perfmodel/goldens/per_op.json` are modified generated
derivatives of those measurements. AISimulate's native pinning workflow at
commit `36dcc8f3afe9e6e2e9de976737b6337fad8c4d74` produced the two case updates;
the adjacent parity README records the reviewed energy-only delta.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/tree/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/aic-core/src/aiconfigurator_core/systems/data/b200_sxm

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This material is licensed under the Apache License 2.0. The upstream license
at the identified revision is available at:
https://github.com/ai-dynamo/aiconfigurator/blob/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/LICENSE

## Dynamo V4.1 FPM collection adapter

`collector/fpm_forward/runtime/dsv41/dsv41_scheduler.py` is modified code
adapted from `components/src/dynamo/vllm/instrumented_scheduler.py` in
https://github.com/ai-dynamo/dynamo/tree/54960177085413259859c88bd34ed0734d4c2ea9.
It adds bounded same-request real-KV collection while preserving the native
benchmark and FPM contracts. Copyright (c) 2025-2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved. Licensed under Apache-2.0; the upstream
license is preserved in the adapter's adjacent `LICENSE`. The adjacent README
records the inspected vLLM API revision and immutable runtime image/source
hashes. vLLM implementation files are not vendored. The text fixture and
lifecycle tests are original work for this change, with no external corpus.

## NVIDIA AIConfigurator speculative decoding

The speculation SDK, compatibility exports, CLI/task integration, attention and whole-forward FPM operation changes, native bindings, and their tests are adapted and modified from AIConfigurator PR #1563, pinned at commit `6290c161a354da5250c391bd43372b2e9c6f4a51`. Original paths are under `aic-core/src/aiconfigurator_core/sdk/`, `src/aiconfigurator/`, `aic-core/rust/aiconfigurator-core/`, `aic-core/rust/tests/public-api/`, and `tests/`.

Derived AISimulate paths are under `python/aisimulate/src/aisimulate_core/sdk/`, `python/aisimulate/src/aisimulate/sdk/speculation/`, `python/aisimulate/src/aisimulate_core/sdk/speculation/`, `python/aisimulate/src/aisimulate/legacy_cli/`, `python/aisimulate/src/aisimulate/sdk/{speculative,task_v2}.py`, and `python/aisimulate/tests/`; repository-root Rust paths are under `crates/core/src/perfmodel/`, `crates/core/parity_tests/perfmodel/`, and `crates/tests/public-api/`. The repository's `docs/aic-pr1563-migration.md` lists the exact original and mapped paths. Changes preserve AISimulate's current native contracts and strengthen configuration validation and regression coverage.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/tree/6290c161a354da5250c391bd43372b2e9c6f4a51

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Licensed under Apache-2.0; the original license is at:
https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/LICENSE

This records provenance for NVIDIA-authored predecessor code, without implying ownership by an unaffiliated third party.

## NVIDIA AIConfigurator CI provenance

Selected CI definitions and a recommendation test were adapted from NVIDIA's
AIConfigurator repository at commit
`77fd0773407b3683d8a671fe24a30a7110651b64` and modified for AISimulate's
unified package layout and Fast CI / Full CI execution model:

- repository-root `.github/workflows/ci.yml` (selected jobs)
- repository-root `.github/actions/build-platform-wheel/action.yml`
- repository-root `.github/actions/setup-python-rust/action.yml`
- repository-root `.github/workflows/collector-check.yml`
- repository-root `.github/workflows/prediction-regression-gate.yml`
- repository-root `.github/workflows/validate-platform-wheels.yml`
- `tests/e2e/cli/test_cli_recommend.py`

Upstream source:
https://github.com/ai-dynamo/AIConfigurator/tree/77fd0773407b3683d8a671fe24a30a7110651b64

Copyright (c) NVIDIA CORPORATION & AFFILIATES.

AIConfigurator is licensed under the Apache License, Version 2.0. The full
Apache-2.0 license text is reproduced in `LICENSE`. This section records
cross-repository provenance for NVIDIA-authored predecessor code; it is not a
claim that AIConfigurator is owned by an unaffiliated third party.

## SGLang feed-forward/decode composition and quantization exclusions

The dense-prefix composition and exclusion resolver in `src/aisimulate_core/sdk/models/deepseek_v32.py` and their regression fixtures in `tests/unit/sdk/models/test_deepseek_v32_dense.py` and `tests/unit/sdk/models/test_large_ep_model_graphs.py` adapt and modify the tensor-parallel communication and packed-linear selection behavior from SGLang revision `02c5a855aceb968c310e6fbc6632270e26edc84b`. Original source paths are `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, and `python/sglang/srt/layers/quantization/utils.py`. The adaptation models operator composition and projection precision without importing the serving runtime or its GPU dependencies.

The scoped GLM-5.2 NVFP4 Rubin decode composition helper in the same model file, its fixtures in `tests/unit/sdk/models/test_sglang_rubin_decode_composition.py`, and the three-operation oracle in repository-root `crates/core/src/perfmodel/engine/runtime.rs` also adapt and modify the embedding reduction, post-join routed/shared expert addition and terminal residual RMSNorm inventory from that revision. Original source paths are `python/sglang/srt/layers/vocab_parallel_embedding.py:566–579`, `python/sglang/srt/models/deepseek_v2.py:1009–1030,2906–2910`, and `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:374–406`. This composition applies to the existing TP4 collection/serving contract with `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0`, although the image default is true, and inactive embedding replication/shared-expert-TP1. It uses existing analytical BF16 memory operations and all-reduce measurements; native operation implementations are not copied into execution and no native latency for the new terms is claimed.

Upstream source:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b/python/sglang/srt

Copyright contributors to the vLLM project. Copyright 2023-2024 SGLang Team. These source files are licensed under the Apache License, Version 2.0. The upstream license is at:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/blob/02c5a855aceb968c310e6fbc6632270e26edc84b/LICENSE

## vLLM

The inference-mode scope and MSA query-position metadata integration in
`collector/vllm/collect_mla_module.py` and
`collector/vllm/collect_msa_module.py` are adapted (modified) from serving
behavior at vLLM commit `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`:

- `vllm/v1/worker/gpu_model_runner.py` (query positions, common attention metadata,
  and the inference-mode model execution boundary).
- `vllm/models/minimax_m3/nvidia/indexer_msa.py` (MSA positions metadata contract).
- `vllm/models/minimax_m3/nvidia/model.py` (versioned shared top-k buffer layout).

Upstream source:
https://github.com/vllm-project/vllm/tree/dd10e03f95f94edbea1975c67ace3a35ec9a8a40

Copyright contributors to the vLLM project. Licensed under Apache-2.0;
modifications adapt the serving contracts to synthetic collector batches.


The independently written Rust deferred-queue and post-lookup-touch behavior in
`crates/core/src/engine/scheduler/vllm/{core,host_offload}.rs`
and related G3 fixtures in `crates/core/src/replay/agg_tests.rs` reference vLLM at
immutable revision `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`:
[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py),
`vllm/v1/core/sched/request_queue.py` and `vllm/v1/kv_offload/tiering/manager.py`.
The upstream project is copyright contributors to the vLLM project and licensed
under Apache-2.0. These repository-root-relative files adapt behavioral contracts;
no upstream method bodies are copied. The Rust implementation and fixtures are
modified for simulation and do not reproduce native filesystem timing.

The following files are derived from vLLM's attention test utilities at tag
`v0.11.0` (commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`):

- `collector/vllm/utils.py`
- `collector/vllm/utils_xpu.py`

Upstream source:
https://github.com/vllm-project/vllm/blob/v0.11.0/tests/v1/attention/utils.py

The Gemma 4 visual-mask behavior and replicated multimodal adapter in
`src/aiconfigurator_core/sdk/models/gemma4.py` and
`src/aiconfigurator_core/sdk/models/blocks/vit.py` are adapted and modified
for performance modeling from vLLM at commit
`d2906091bfc579cebefe3d8e8fb9077397ce9882`:

https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4.py
https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4_mm.py

The Gemma 4 source preserves these upstream notices:

Copyright contributors to the vLLM project.
Copyright 2025 The vLLM team.
Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

This material is licensed under the Apache License 2.0.

The Kimi K2.5 and Kimi K3 vision-tower topology, pooled PatchMerger, and
PatchMergerV2 adapters modeled in
`src/aiconfigurator_core/sdk/models/blocks/vit.py` and parsed in
`src/aiconfigurator_core/sdk/utils.py`, with Kimi K3 rotary-grid validation in
`src/aiconfigurator_core/sdk/backends/base_backend.py` and regression derivatives in
`tests/unit/sdk/models/test_kimi_k25_vision.py` and
`tests/unit/sdk/models/test_kimi_k3_vision.py`, are modified adaptations of the
MoonViT3D implementation that Kimi K3 reuses from Kimi K2.5 in vLLM at commit
`d2906091bfc579cebefe3d8e8fb9077397ce9882`:

- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25.py
- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py
- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/layers/quantization/modelopt.py

The upstream license at that revision is available at:
https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/LICENSE

The Llama 4 encoder operation topology in
`src/aiconfigurator_core/sdk/models/blocks/vit.py` is adapted (modified) from vLLM's Llama 4
implementation at tag `v0.8.5` (commit
`ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`):

https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/mllama4.py

Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team. All rights reserved.

The upstream license at that revision is available at:
https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/LICENSE

Copyright contributors to the vLLM project.

The vLLM `LICENSE` file at commit
`b8b302cde434df8c9289a2b465406b47ebab1c2d` is reproduced verbatim below:

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## DeepEP

The following files are derived from DeepEP test utilities at commit
`73b6ea4a439ba03a695563f9fd242c8e4b02b37c` and contain NVIDIA modifications:

- `collector/wideep/sglang/deepep/test_internode.py`
- `collector/wideep/sglang/deepep/test_intranode.py`
- `collector/wideep/sglang/deepep/test_low_latency.py`
- `collector/wideep/sglang/deepep/utils.py`

The patch `collector/wideep/vllm/patches/deepep_73b_nvl4.patch` is also a
modification of DeepEP source at that commit.

Upstream source:
https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c

```text
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SGLang DeepSeek-V4.1 serving contracts

The V4.1 execution and performance-model logic in
`src/aisimulate_core/sdk/deepseek_v41.py`, `sdk/models/deepseek_v41.py`
(with the same `src/aisimulate_core/` prefix), and repository-root
`crates/core/src/perfmodel/operators/dsv41.rs` is informed by and modified from
SGLang's serving architecture at immutable commit
`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`:

- `python/sglang/srt/models/deepseek_v4.py` and `deepseek_v2.py`
- `python/sglang/srt/layers/engram.py`
- `python/sglang/srt/layers/attention/dsv4/compressor.py`
- `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`
- `python/sglang/benchmark/one_batch.py`

The original integration adapter
`collector/sglang/dsv41_native_runner.py` calls that pinned benchmark's model
builder and request lifecycle. Its component boundaries are modified from
the serving contracts above; it does not copy framework metadata builders.
The matching loaded-dimension guards in `collector/sglang/dsv41_contract.py`
and their CPU fixtures in `tests/unit/collector/test_dsv41_contract.py` are
modified analytical adaptations of the indexer layout in `dsv41_sparse.py`.

The measured operator databases and adjacent documentation under
`src/aisimulate_core/systems/profiles/dsv41/` contain AISimulate timings
and geometry adapted from the same serving contracts (modified). Their
README and adjacent provenance identify the immutable collection runtime
and source identities. The restored GB300 tables preserve their historical
timings and documented source-audited indexer metadata correction. The
B300 TP4 and TP2 tables are new native measurements with loaded-module dimension
validation; they do not inherit that historical correction.

The FPM table execution identities, geometry, collection sidecars and accompanying
README under `src/aisimulate_core/systems/profiles/dsv41_fpm/`, including
B300 TP2/TP4 and GB300 TP2 full/bounded measurements, are modified AISimulate
adaptations of the SGLang serving-contract paths and immutable commit
listed above (Copyright 2023-2024 SGLang Team and SGLang contributors,
Apache-2.0), and of `config.json`, `inference/model.py` and
`DeepSeek_V41_Tech_Report.pdf` from
`deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d`
(Copyright (c) 2023 DeepSeek, MIT; source and license below). The latencies are
new AISimulate measurements; these files contain no upstream model execution
code. The SGLang source revision identifies the serving-contract reference,
not the entire measured runtime image; the immutable image and captured source
identities remain collection provenance.

Source: https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc
Copyright 2023-2024 SGLang Team and SGLang contributors. Licensed under Apache-2.0; its terms are
reproduced in the repository `LICENSE`. These are analytical adaptations,
not a copy of the model execution implementation. The modified analytical
scoring/storage adaptations and their independently written regression cases
also appear in `python/aisimulate/tests/unit/sdk/models/test_deepseek_v41.py`,
Rust operator/spec unit tests, `docs/deepseek-v41.md`, and
`docs/deepseek-v41-storage.md`. They distinguish candidate masking from scoring
and physical FlashMLA cache payload from logical FP4 values.

## DeepSeek model configuration files

The following model configuration files are copied from, or formatting-only
adaptations of, the named DeepSeek model repositories:

| AISimulate file | Upstream revision |
| --- | --- |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-R1_config.json` | `deepseek-ai/DeepSeek-R1@56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V3_config.json` | `deepseek-ai/DeepSeek-V3@e815299b0bcbac849fa540c768ef21845365c9eb` |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json` | `deepseek-ai/DeepSeek-V3.2@c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3` |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4-Flash_config.json` | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4.1-Flash_config.json` | `deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| `src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json` | `deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1` |

The V4.1 descriptor and performance formulas in `src/aisimulate_core/sdk/deepseek_v41.py`,
`src/aisimulate_core/sdk/models/deepseek_v41.py`, and repository-root
`crates/core/src/perfmodel/operators/dsv41.rs` are AISimulate performance-model
adaptations of the architecture described by `inference/model.py` and
`DeepSeek_V41_Tech_Report.pdf` at the same V4.1 revision (modified; no model execution code).
Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d

Upstream repositories:
https://huggingface.co/deepseek-ai

```text
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Meta Muse Glimmer model configuration

`src/aisimulate_core/model_configs/meta-models--Muse-Glimmer-30B_config.json`
is an unmodified copy of `config.json` from the Meta Muse Glimmer model
repository at immutable revision
`f84ecc3a0ea984a4c04542a84269e3d065350a6e`:

https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/f84ecc3a0ea984a4c04542a84269e3d065350a6e/config.json

Copyright owner: Meta Platforms, Inc. and affiliates.

License: Apache License 2.0. The upstream license is available at:
https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/f84ecc3a0ea984a4c04542a84269e3d065350a6e/LICENSE

The Apache License 2.0 terms are reproduced in this distribution's `LICENSE`
file.

## Qwen3.8-Max model configuration files

The following model configuration files are copied byte-for-byte from the
named Qwen model repositories at the immutable revisions shown:

| Packaged file | Upstream revision |
| --- | --- |
| `aisimulate_core/model_configs/Qwen--Qwen3.8-2.4T-A95B_config.json` | `Qwen/Qwen3.8-2.4T-A95B@207bd685a7e3696cfaff12ded7c6a7ea0f88c996` |
| `aisimulate_core/model_configs/Qwen--Qwen3.8-2.4T-A95B-FP8_config.json` | `Qwen/Qwen3.8-2.4T-A95B-FP8@d2dc35658bcf77e66643428cb52e774cc3b5bd29` |

Upstream repositories:
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B and
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8

Copyright owner: Qwen. The copied files are unmodified. Both upstream
revisions apply the same custom Qwen3.8-Max License, reproduced in full
below:

```text
Qwen3.8-Max License

Copyright (c) 2026 Qwen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software, including the model weights, parameters, configuration files, inference code and associated documentation files (collectively, the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from (collectively, "Use" or "Using") copies of the Software; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, and the aggregate revenue of the licensee and its affiliates exceeds US$50,000,000 (or the equivalent amount in any other currencies) during any consecutive twelve (12) months, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties.
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.

THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL QWEN, ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS, AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY.

For any questions regarding this license, please contact model-business@notice.qwencloud.com.
```

## Meta Llama 4

The following bundled model configs are modified copies of Meta Llama 4
checkpoint configuration files:

- `src/aisimulate_core/model_configs/meta-llama--Llama-4-Scout-17B-16E-Instruct_config.json`
  from revision `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3`:
  https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/92f3b1597a195b523d8d9e5700e57e4fbb8f20d3/config.json
- `src/aisimulate_core/model_configs/meta-llama--Llama-4-Maverick-17B-128E-Instruct_config.json`
  from revision `73d14711bcc77c16df3470856949c3764056b617`:
  https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/73d14711bcc77c16df3470856949c3764056b617/config.json

Required attribution: Llama 4 is licensed under the Llama 4 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.

The Llama 4 Community License below is reproduced from the official Meta Llama
repository at commit `0e0b8c519242d5833d8c11bffc1232b77ad7f301`:

https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/LICENSE

```text
LLAMA 4 COMMUNITY LICENSE AGREEMENT

Llama 4 Version Effective Date: April 5, 2025

“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein.

“Documentation” means the specifications, manuals and documentation accompanying Llama 4 distributed by Meta at https://www.llama.com/docs/overview.

“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

“Llama 4” means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by Meta at https://www.llama.com/llama-downloads.

“Llama Materials” means, collectively, Meta’s proprietary Llama 4 and Documentation (and any portion thereof) made available under this Agreement.

“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.

1. License Rights and Redistribution.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials.

b. Redistribution and Use.

i. If you distribute or make available the Llama Materials (or any derivative works thereof), or a product or service (including another AI model) that contains any of them, you shall (A) provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at the beginning of any such AI model name.

ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part of an integrated end user product, then Section 2 of this Agreement will not apply to you.

iii. You must retain in all copies of the Llama Materials that you distribute the following attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

iv. Your use of the Llama Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama Materials (available at https://www.llama.com/llama4/use-policy), which is hereby incorporated by reference into this Agreement.

2. Additional Commercial Terms. If, on the Llama 4 version release date, the monthly active users of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta, which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. No trademark licenses are granted under this Agreement, and in connection with the Llama Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines (currently accessible at https://about.meta.com/brand/resources/meta/company-brand/). All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Llama Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 4 outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Llama Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Llama Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.
```

## Hugging Face Transformers

The `rotate_half` function in `collector/trtllm/collect_mla.py` is copied from
`transformers.models.llama.modeling_llama.rotate_half` in the Hugging Face
Transformers project at tag `v4.57.1`:

https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py#L109-L113

The Gemma 4 vision-tower graph, bidirectional visual-block mask behavior, and
aspect-ratio-preserving processor math in these files are adapted and modified
for performance modeling from Hugging Face
Transformers at commit `cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`:

- `src/aiconfigurator_core/sdk/models/blocks/vit.py`
- `src/aiconfigurator_core/sdk/models/gemma4.py`
- `src/aiconfigurator_core/sdk/backends/base_backend.py`

Upstream sources:

- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/gemma4/modeling_gemma4.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/gemma4/image_processing_gemma4.py

Copyright 2026 the HuggingFace Team. All rights reserved.
This Gemma 4 material is licensed under the Apache License 2.0.

The Kimi K2.5 and Kimi K3 spatial-temporal vision-tower, processor, and temporal-pooling
behavior in `src/aiconfigurator_core/sdk/models/blocks/vit.py`,
`src/aiconfigurator_core/sdk/utils.py`, and `src/aiconfigurator_core/sdk/backends/base_backend.py`,
with regression derivatives in `tests/unit/sdk/models/test_kimi_k25_vision.py`
and `tests/unit/sdk/models/test_kimi_k3_vision.py`,
are modified adaptations of Hugging Face Transformers at
commit `cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`:

- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/modeling_kimi_k25.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py

Copyright 2026 the HuggingFace Inc. team. All rights reserved.
Copyright 2026 the HuggingFace Team. All rights reserved.
This material is licensed under the Apache License 2.0.

The upstream license at that revision is available at:
https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/LICENSE

The fixed-tile canvas and global-tile logic in
`src/aiconfigurator_core/sdk/backends/base_backend.py`, and the processor metadata
normalization in `src/aiconfigurator_core/sdk/utils.py`, are modified adaptations
of the Llama 4 image processor (Copyright 2025 HuggingFace Inc. team. All rights
reserved.) at tag `v4.51.0` (commit
`0720e206c6ba28887e4d60ef60a6a089f6c1cc76`):

https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/models/llama4/image_processing_llama4_fast.py

The upstream license at that revision is available at:
https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/LICENSE

The Transformers `LICENSE` file at tag `v4.57.1` is reproduced verbatim below:

```text
Copyright 2018- The Hugging Face team. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## AISim FPM Gym

- Source: https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym
- Revision: `e8221729db2802e822f6919fd68bc2941743385b`.
- Original paths: `src/aisim_fpm/{hf,types,models,evals/fpt}`,
  `dashboard/index.html`, `dashboard/assets/gym.css`, and `tests/test_hf_dataset.py`.
- Derived files: `scripts/fpm_accuracy/`, `pages/fpm-accuracy/`, and
  `tests/fpm_accuracy/test_hf_dataset.py`.
- Copyright: NVIDIA CORPORATION & AFFILIATES.
- License: Apache-2.0; NVIDIA maintainer confirmed permission to migrate and
  publish this code under Apache-2.0.
- Modified: development-only two-predictor evaluation, public overview export,
  GitHub Pages presentation, local import paths, and canonical estimator API
  adaptation with older-wheel compatibility. No Plotly assets included.
