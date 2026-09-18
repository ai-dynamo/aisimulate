# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the routing IDs produced by FlashInfer's fused TRT-LLM MoE kernel.

SGLang's non-routed FlashInfer backend normally keeps top-k selection inside
the fused MoE kernel, so the framework recorder cannot see the selected expert
IDs.  FlashInfer exposes ``routing_replay_out`` specifically for retrieving the
IDs selected by that same kernel.  This fail-closed bridge supplies that output
only while the recorder is active and forwards it to SGLang's recorder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

SGLANG_VERSION = "0.5.14"
FLASHINFER_VERSION = "0.6.13"
FLASHINFER_DISTRIBUTIONS = {
    "flashinfer-python": FLASHINFER_VERSION,
    "flashinfer-cubin": FLASHINFER_VERSION,
    "flashinfer-jit-cache": f"{FLASHINFER_VERSION}+cu130",
}
FLASHINFER_ROUTING_REPLAY_FIX = "b54d28bea0639510d79c5ac58a60a4087585ff00"
EXPECTED_SOURCE_SHA256 = "067753d34e2b258939508c98e65b5ac5883217245b78563b0a7e759310b6e3b5"
EXPECTED_RUNNER_SOURCE_SHA256 = "09a54bdf8636ed9f9af3dd946bf61d4b40f06c246ba8baaa077ff3c459ea92ca"
EXPECTED_MXFP4_SOURCE_SHA256 = "b514d889f7ef55a8dca5f941702ec2871c8980000b6da355e7f52eb414ca9e5f"
EXPECTED_COMPRESSED_MXINT4_SOURCE_SHA256 = "61311a5c3407352c0399fd596ffe59ed8762796a79b72351505352431aa052c3"

_IMPORT_ORIGINAL = "from sglang.srt.utils.custom_op import register_custom_op\n"
_IMPORT_PATCHED = """from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.utils.custom_op import register_custom_op
"""

_RETURN_ORIGINAL = "    return trtllm_fp8_block_scale_moe(**kwargs)\n"
_RETURN_PATCHED = """    recorder = get_global_expert_distribution_recorder()
    routing_replay_out = None
    if recorder.recording:
        routing_replay_out = torch.full(
            (hidden_states.shape[0], top_k),
            -1,
            dtype=torch.int16,
            device=hidden_states.device,
        )
        kwargs["routing_replay_out"] = routing_replay_out

    output = trtllm_fp8_block_scale_moe(**kwargs)
    if routing_replay_out is not None:
        # The producer and recorder operations use the same CUDA stream, so
        # scatter-add observes the IDs emitted by the fused routing kernel.
        recorder.on_select_experts(topk_ids=routing_replay_out)
    return output
"""

_RUNNER_ORIGINAL = """    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
"""
_RUNNER_PATCHED = """    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    # DeepSeek-V4 hash-routed layers already produce and record exact Standard
    # top-k IDs. Keep those layers on the routed kernel while learned-routing
    # layers remain on the fused internal-routing path observed via replay.
    use_routed_topk = use_routed_topk or TopKOutputChecker.format_is_standard(
        topk_output
    )
    if TopKOutputChecker.format_is_bypassed(topk_output):
"""

_RUNNER_IMPORT_ORIGINAL = "from sglang.srt.environ import envs\n"
_RUNNER_IMPORT_PATCHED = """from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
"""

_BF16_CALL_ORIGINAL = """            # Call the fused kernel
            final_hidden_states = trtllm_bf16_moe(
                routing_logits=topk_output.router_logits,
                routing_bias=topk_config.correction_bias,
                hidden_states=hidden_states,
                gemm1_weights=quant_info.gemm1_weights,
                gemm2_weights=quant_info.gemm2_weights,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=runner_config.intermediate_size_per_partition,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=runner_config.num_local_experts,
                routing_method_type=runner_config.routing_method_type,
                routed_scaling_factor=runner_config.routed_scaling_factor,
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
                activation_type=activation_type,
            )
"""
_BF16_CALL_PATCHED = """            # Call the same fused kernel and request only its selected IDs.
            recorder = get_global_expert_distribution_recorder()
            routing_replay_out = None
            if recorder.recording:
                routing_replay_out = torch.full(
                    (hidden_states.shape[0], topk_config.top_k),
                    -1,
                    dtype=torch.int16,
                    device=hidden_states.device,
                )
            final_hidden_states = trtllm_bf16_moe(
                routing_logits=topk_output.router_logits,
                routing_bias=topk_config.correction_bias,
                hidden_states=hidden_states,
                gemm1_weights=quant_info.gemm1_weights,
                gemm2_weights=quant_info.gemm2_weights,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=runner_config.intermediate_size_per_partition,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=runner_config.num_local_experts,
                routing_method_type=runner_config.routing_method_type,
                routed_scaling_factor=runner_config.routed_scaling_factor,
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
                activation_type=activation_type,
                routing_replay_out=routing_replay_out,
            )
            if routing_replay_out is not None:
                recorder.on_select_experts(topk_ids=routing_replay_out)
"""

_MXFP4_IMPORT_ORIGINAL = "from sglang.srt.environ import envs\n"
_MXFP4_IMPORT_PATCHED = """from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
"""

_MXFP4_SETUP_ORIGINAL = """            top_k = topk_output.topk_config.top_k
            router_logits = topk_output.router_logits

            with use_symmetric_memory(
"""
_MXFP4_SETUP_PATCHED = """            top_k = topk_output.topk_config.top_k
            router_logits = topk_output.router_logits
            recorder = get_global_expert_distribution_recorder()
            routing_replay_out = None
            if recorder.recording:
                routing_replay_out = torch.full(
                    (x_quant.shape[0], top_k),
                    -1,
                    dtype=torch.int16,
                    device=x_quant.device,
                )

            with use_symmetric_memory(
"""

_MXFP4_CALL_TAIL_ORIGINAL = """                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                output=symm_output,
            )[0]
            return StandardCombineInput(hidden_states=trtllm_gen_output)
"""
_MXFP4_CALL_TAIL_PATCHED = """                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                output=symm_output,
                routing_replay_out=routing_replay_out,
            )[0]
            if routing_replay_out is not None:
                recorder.on_select_experts(topk_ids=routing_replay_out)
            return StandardCombineInput(hidden_states=trtllm_gen_output)
"""

_COMPRESSED_MXINT4_IMPORT_ORIGINAL = "from sglang.srt.distributed import get_tp_group\n"
_COMPRESSED_MXINT4_IMPORT_PATCHED = """from sglang.srt.distributed import get_tp_group
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
"""

_COMPRESSED_MXINT4_SETUP_ORIGINAL = """        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = (
"""
_COMPRESSED_MXINT4_SETUP_PATCHED = """        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        recorder = get_global_expert_distribution_recorder()
        routing_replay_out = None
        if recorder.recording:
            routing_replay_out = torch.full(
                (x.shape[0], topk_config.top_k),
                -1,
                dtype=torch.int16,
                device=x.device,
            )
        correction_bias = (
"""

_COMPRESSED_MXINT4_CALL_TAIL_ORIGINAL = """            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            output=symm_output,
        )

        return StandardCombineInput(hidden_states=symm_output)
"""
_COMPRESSED_MXINT4_CALL_TAIL_PATCHED = """            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            output=symm_output,
            routing_replay_out=routing_replay_out,
        )
        if routing_replay_out is not None:
            recorder.on_select_experts(topk_ids=routing_replay_out)

        return StandardCombineInput(hidden_states=symm_output)
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_bridge(report_path: Path) -> dict:
    installed_version = importlib.metadata.version("sglang")
    if installed_version != SGLANG_VERSION:
        raise RuntimeError(f"SGLang {installed_version!r} != pinned {SGLANG_VERSION!r}")
    installed_flashinfer_distributions = {name: importlib.metadata.version(name) for name in FLASHINFER_DISTRIBUTIONS}
    mismatched_flashinfer_distributions = {
        name: {"expected": expected, "actual": installed_flashinfer_distributions[name]}
        for name, expected in FLASHINFER_DISTRIBUTIONS.items()
        if installed_flashinfer_distributions[name] != expected
    }
    if mismatched_flashinfer_distributions:
        raise RuntimeError(
            f"FlashInfer distributions do not match the pinned runtime: "
            f"{mismatched_flashinfer_distributions}; "
            f"custom routing replay requires upstream fix {FLASHINFER_ROUTING_REPLAY_FIX}"
        )

    wrapper_spec = importlib.util.find_spec("sglang.srt.layers.moe.flashinfer_trtllm_moe")
    runner_spec = importlib.util.find_spec("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")
    mxfp4_spec = importlib.util.find_spec("sglang.srt.layers.quantization.mxfp4")
    compressed_mxint4_spec = importlib.util.find_spec(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe"
    )
    if wrapper_spec is None or wrapper_spec.origin is None:
        raise RuntimeError("could not locate SGLang FlashInfer TRT-LLM MoE wrapper")
    if runner_spec is None or runner_spec.origin is None:
        raise RuntimeError("could not locate SGLang FlashInfer TRT-LLM MoE runner")
    if mxfp4_spec is None or mxfp4_spec.origin is None:
        raise RuntimeError("could not locate SGLang MXFP4 quantization source")
    if compressed_mxint4_spec is None or compressed_mxint4_spec.origin is None:
        raise RuntimeError("could not locate SGLang compressed-tensors MXINT4 MoE source")
    wrapper_path = Path(wrapper_spec.origin)
    runner_path = Path(runner_spec.origin)
    mxfp4_path = Path(mxfp4_spec.origin)
    compressed_mxint4_path = Path(compressed_mxint4_spec.origin)
    wrapper_source = wrapper_path.read_bytes()
    runner_source = runner_path.read_bytes()
    mxfp4_source = mxfp4_path.read_bytes()
    compressed_mxint4_source = compressed_mxint4_path.read_bytes()
    wrapper_original_sha256 = _sha256(wrapper_source)
    runner_original_sha256 = _sha256(runner_source)
    mxfp4_original_sha256 = _sha256(mxfp4_source)
    compressed_mxint4_original_sha256 = _sha256(compressed_mxint4_source)
    if wrapper_original_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {wrapper_path}: {wrapper_original_sha256}; expected {EXPECTED_SOURCE_SHA256}"
        )
    if runner_original_sha256 != EXPECTED_RUNNER_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {runner_path}: {runner_original_sha256}; "
            f"expected {EXPECTED_RUNNER_SOURCE_SHA256}"
        )
    if mxfp4_original_sha256 != EXPECTED_MXFP4_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {mxfp4_path}: {mxfp4_original_sha256}; "
            f"expected {EXPECTED_MXFP4_SOURCE_SHA256}"
        )
    if compressed_mxint4_original_sha256 != EXPECTED_COMPRESSED_MXINT4_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {compressed_mxint4_path}: {compressed_mxint4_original_sha256}; "
            f"expected {EXPECTED_COMPRESSED_MXINT4_SOURCE_SHA256}"
        )

    wrapper_decoded = wrapper_source.decode("utf-8")
    runner_decoded = runner_source.decode("utf-8")
    mxfp4_decoded = mxfp4_source.decode("utf-8")
    compressed_mxint4_decoded = compressed_mxint4_source.decode("utf-8")
    if wrapper_decoded.count(_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected custom-op import was not uniquely present")
    if wrapper_decoded.count(_RETURN_ORIGINAL) != 1:
        raise RuntimeError("expected fused FP8 MoE return was not uniquely present")
    if runner_decoded.count(_RUNNER_ORIGINAL) != 1:
        raise RuntimeError("expected fused FP8 MoE runner dispatch block was not uniquely present")
    if runner_decoded.count(_RUNNER_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected fused MoE runner env import was not uniquely present")
    if runner_decoded.count(_BF16_CALL_ORIGINAL) != 1:
        raise RuntimeError("expected fused BF16 MoE call was not uniquely present")
    if mxfp4_decoded.count(_MXFP4_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected MXFP4 env import was not uniquely present")
    if mxfp4_decoded.count(_MXFP4_SETUP_ORIGINAL) != 1:
        raise RuntimeError("expected MXFP4 fused routing setup was not uniquely present")
    if mxfp4_decoded.count(_MXFP4_CALL_TAIL_ORIGINAL) != 1:
        raise RuntimeError("expected MXFP4 fused routing call tail was not uniquely present")
    if compressed_mxint4_decoded.count(_COMPRESSED_MXINT4_IMPORT_ORIGINAL) != 1:
        raise RuntimeError("expected compressed MXINT4 distributed import was not uniquely present")
    if compressed_mxint4_decoded.count(_COMPRESSED_MXINT4_SETUP_ORIGINAL) != 1:
        raise RuntimeError("expected compressed MXINT4 fused routing setup was not uniquely present")
    if compressed_mxint4_decoded.count(_COMPRESSED_MXINT4_CALL_TAIL_ORIGINAL) != 1:
        raise RuntimeError("expected compressed MXINT4 fused routing call tail was not uniquely present")
    wrapper_patched = wrapper_decoded.replace(_IMPORT_ORIGINAL, _IMPORT_PATCHED, 1)
    wrapper_patched = wrapper_patched.replace(_RETURN_ORIGINAL, _RETURN_PATCHED, 1).encode("utf-8")
    runner_patched = runner_decoded.replace(_RUNNER_IMPORT_ORIGINAL, _RUNNER_IMPORT_PATCHED, 1)
    runner_patched = runner_patched.replace(_RUNNER_ORIGINAL, _RUNNER_PATCHED, 1)
    runner_patched = runner_patched.replace(_BF16_CALL_ORIGINAL, _BF16_CALL_PATCHED, 1).encode("utf-8")
    mxfp4_patched = mxfp4_decoded.replace(_MXFP4_IMPORT_ORIGINAL, _MXFP4_IMPORT_PATCHED, 1)
    mxfp4_patched = mxfp4_patched.replace(_MXFP4_SETUP_ORIGINAL, _MXFP4_SETUP_PATCHED, 1)
    mxfp4_patched = mxfp4_patched.replace(_MXFP4_CALL_TAIL_ORIGINAL, _MXFP4_CALL_TAIL_PATCHED, 1).encode("utf-8")
    compressed_mxint4_patched = compressed_mxint4_decoded.replace(
        _COMPRESSED_MXINT4_IMPORT_ORIGINAL, _COMPRESSED_MXINT4_IMPORT_PATCHED, 1
    )
    compressed_mxint4_patched = compressed_mxint4_patched.replace(
        _COMPRESSED_MXINT4_SETUP_ORIGINAL, _COMPRESSED_MXINT4_SETUP_PATCHED, 1
    )
    compressed_mxint4_patched = compressed_mxint4_patched.replace(
        _COMPRESSED_MXINT4_CALL_TAIL_ORIGINAL, _COMPRESSED_MXINT4_CALL_TAIL_PATCHED, 1
    ).encode("utf-8")

    for source_path, patched in (
        (wrapper_path, wrapper_patched),
        (runner_path, runner_patched),
        (mxfp4_path, mxfp4_patched),
        (compressed_mxint4_path, compressed_mxint4_patched),
    ):
        temporary = source_path.with_suffix(".py.collector-replay-tmp")
        temporary.write_bytes(patched)
        temporary.replace(source_path)

    report = {
        "status": "APPLIED",
        "framework": "sglang",
        "framework_version": installed_version,
        "flashinfer_distributions": installed_flashinfer_distributions,
        "flashinfer_routing_replay_fix": FLASHINFER_ROUTING_REPLAY_FIX,
        "observation": "flashinfer_bf16_fp8_mxfp4_and_compressed_mxint4_fused_moe_routing_replay_out",
        "source_files": {
            "flashinfer_trtllm_moe.py": {
                "path": str(wrapper_path),
                "original_sha256": wrapper_original_sha256,
                "patched_sha256": _sha256(wrapper_patched),
            },
            "moe_runner/flashinfer_trtllm.py": {
                "path": str(runner_path),
                "original_sha256": runner_original_sha256,
                "patched_sha256": _sha256(runner_patched),
            },
            "quantization/mxfp4.py": {
                "path": str(mxfp4_path),
                "original_sha256": mxfp4_original_sha256,
                "patched_sha256": _sha256(mxfp4_patched),
            },
            "quantization/compressed_tensors_w4a4_mxint4_moe.py": {
                "path": str(compressed_mxint4_path),
                "original_sha256": compressed_mxint4_original_sha256,
                "patched_sha256": _sha256(compressed_mxint4_patched),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_bridge(args.report), sort_keys=True))


if __name__ == "__main__":
    main()
