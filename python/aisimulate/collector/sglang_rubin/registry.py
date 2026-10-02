# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-specific implementations of the existing GLM-5.2 collector contracts."""

from collector.registry_types import OpEntry, PerfFile

MODEL_PATH = "nvidia/GLM-5.2-NVFP4"
SM_VERSION = 107
SGLANG_COMMIT = "02c5a855aceb968c310e6fbc6632270e26edc84b"
SGLANG_REPOSITORY = "https://gitlab-master.nvidia.com/dl/sglang/sglang"
# Observed on Hecate in JET job 443897220 (2026-09-17), separately from the
# image's SGLANG_VERSION build label. The checkpoint is hf-aec724e_orig.
SGLANG_DISTRIBUTION_VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
CHECKPOINT_METADATA_SHA256 = {
    "config.json": "d3783a603e5aa9cb58eff7a5d8ac9c42d83156b229efe185c72e9e2dc8444923",
    "hf_quant_config.json": "212394aabaa6a4e7823668e7a03d6295422aa2ca2cbdbfbfc519eab602ad20e6",
}

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.sglang_rubin.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="moe",
        module="collector.sglang_rubin.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.sglang_rubin.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.sglang_rubin.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsa_context_module_skip_indexer",
        module="collector.sglang_rubin.collect_mla_module",
        get_func="get_dsa_context_module_skip_indexer_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
        worker_perf_filename=PerfFile.DSA_CONTEXT_MODULE_SKIP_INDEXER,
    ),
    OpEntry(
        op="dsa_generation_module_skip_indexer",
        module="collector.sglang_rubin.collect_mla_module",
        get_func="get_dsa_generation_module_skip_indexer_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
        worker_perf_filename=PerfFile.DSA_GENERATION_MODULE_SKIP_INDEXER,
    ),
    OpEntry(
        op="glm5_mqa_logits_module",
        module="collector.sglang_rubin.glm5_dsa_sparse_modules",
        get_func="get_glm5_mqa_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_MQA_LOGITS_MODULE,
    ),
    OpEntry(
        op="glm5_topk_module",
        module="collector.sglang_rubin.glm5_dsa_sparse_modules",
        get_func="get_glm5_topk_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_TOPK_MODULE,
    ),
    OpEntry(
        op="glm5_dsa_attn_module",
        module="collector.sglang_rubin.glm5_dsa_sparse_modules",
        get_func="get_glm5_dsa_attn_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_DSA_ATTN_MODULE,
    ),
]
