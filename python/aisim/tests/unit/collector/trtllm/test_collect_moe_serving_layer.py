# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free: the TRT-LLM MoE collector builds the routed-expert layer each architecture's serving builds.

``collector/trtllm/collect_moe.py`` imports torch and tensorrt_llm at module
scope, so the construction helpers are AST-extracted and run against recording
fakes of the TensorRT-LLM classes (same technique as test_moe_serving_gate.py).
"""

import ast
import enum
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECT_MOE = Path(__file__).resolve().parents[4] / "collector" / "trtllm" / "collect_moe.py"

FUNCTIONS = {
    "_zero_routing_bias",
    "_serving_moe_gate",
    "_serving_moe_layer",
    "_moe_kernel_source",
    "_post_load_weights",
}


class ActivationType(enum.Enum):
    Swiglu = 0
    Relu2 = 1
    Geglu = 2


class _Recorded:
    """Records the constructor call, like the TensorRT-LLM class it stands in for."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _recording(name):
    return type(name, (_Recorded,), {})


class _Bias:
    zeroed = False

    def zero_(self):
        self.zeroed = True


def _gate(kind):
    class Gate(_Recorded):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.kind = kind
            self.e_score_correction_bias = _Bias()
            self.routing_method = ("gate-routing", kind)

    return Gate


class _TritonFusedMoE:
    pass


@pytest.fixture
def helpers(monkeypatch):
    tree = ast.parse(COLLECT_MOE.read_text())

    def wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return node.name in FUNCTIONS
        if isinstance(node, ast.Assign):
            return any(isinstance(t, ast.Name) and t.id.endswith("_ARCHITECTURES") for t in node.targets) or any(
                isinstance(t, ast.Name) and t.id == "_MOE_KERNEL_SOURCES" for t in node.targets
            )
        return False

    namespace = {
        "torch": SimpleNamespace(
            no_grad=nullcontext,
            zeros=lambda n, dtype: ("zeros", n, dtype),
            ones=lambda n, dtype: ("ones", n, dtype),
            float32="fp32",
            bfloat16="bf16",
        ),
        "SimpleNamespace": SimpleNamespace,
        "ActivationType": ActivationType,
        "DEFAULT_MOE_ACTIVATION": "default-swiglu",
        "DeepseekV3Gate": _gate("v3"),
        "DeepSeekV3MoeRoutingMethod": _recording("DeepSeekV3MoeRoutingMethod"),
        "Llama4RenormalizeMoeRoutingMethod": _recording("Llama4RenormalizeMoeRoutingMethod"),
        "MiniMaxM2MoeRoutingMethod": _recording("MiniMaxM2MoeRoutingMethod"),
        "RenormalizeMoeRoutingMethod": _recording("RenormalizeMoeRoutingMethod"),
        "SimpleActivation": _recording("SimpleActivation"),
        "SiTuActivation": _recording("SiTuActivation"),
        "SwigluActivation": _recording("SwigluActivation"),
        "SwigluBiasActivation": _recording("SwigluBiasActivation"),
        "TritonFusedMoE": _TritonFusedMoE,
        "_TRTLLM_VERSION": "1.4.0rc2",
        "_packaged_model_config": lambda model: {
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            # GLM-5.3-Flash text config fields Glm5NextGate and Glm5NextMoE read.
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "swiglu_limit": 10.0,
        },
    }
    exec(
        compile(ast.Module(body=[n for n in tree.body if wanted(n)], type_ignores=[]), str(COLLECT_MOE), "exec"),
        namespace,
    )
    for name in ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for module_name, attr, value in (
        ("modeling_deepseekv4", "DeepseekV4Gate", _gate("v4")),
        ("modeling_minimaxm3", "MiniMaxM3Gate", _gate("m3")),
        ("modeling_gemma4", "Gemma4MoeRoutingMethod", _recording("Gemma4MoeRoutingMethod")),
        ("modeling_glm5_next", "Glm5NextGate", _gate("glm5next")),
    ):
        module = ModuleType(f"tensorrt_llm._torch.models.{module_name}")
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return namespace


def _declared(architecture, **fields):
    values = {
        "architecture": architecture,
        "sglang_moe_num_expert_group": 8,
        "sglang_moe_topk_group": 4,
        "sglang_moe_routed_scaling_factor": 2.5,
        "sglang_moe_swiglu_limit": None,
    }
    values.update(fields)
    return SimpleNamespace(**values)


def _layer(helpers, architecture, moe_backend="CUTLASS", **fields):
    return helpers["_serving_moe_layer"](
        _declared(architecture, **fields), "org/model", 7168, 256, 8, "bf16", moe_backend
    )


@pytest.mark.parametrize(
    "architecture", ["DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM", "KimiK25ForConditionalGeneration"]
)
def test_deepseek_v3_family_routes_through_its_gate_with_default_swiglu(helpers, architecture):
    layer = _layer(helpers, architecture, moe_backend="TRTLLM")
    assert layer.routing_method == ("gate-routing", "v3")
    assert layer.gate.e_score_correction_bias.zeroed
    assert layer.gate.kwargs["moe_backend"] == "TRTLLM"
    assert layer.activation == "default-swiglu"
    assert layer.router_logits_dtype == "fp32"
    assert (layer.bias, layer.apply_router_weight_on_input, layer.routing_scales_from_backend) == (False, False, False)


def test_nemotron_h_runs_relu2_experts_whatever_the_row_declares(helpers):
    # The FP8 rows of cases/models declare silu; modeling_nemotron_h.py has one activation.
    layer = _layer(helpers, "NemotronHForCausalLM", sglang_moe_num_expert_group=None, sglang_moe_topk_group=None)
    assert layer.routing_method == ("gate-routing", "v3")
    assert layer.activation.kwargs == {"kind": ActivationType.Relu2}
    assert layer.gate.kwargs["n_group"] == 1 and layer.gate.kwargs["topk_group"] == 1


def test_deepseek_v4_clamps_swiglu_at_the_checkpoint_limit(helpers):
    layer = _layer(helpers, "DeepseekV4ForCausalLM", moe_backend="TRTLLM", sglang_moe_swiglu_limit=10.0)
    assert layer.routing_method == ("gate-routing", "v4")
    assert layer.gate.kwargs["is_hashed"] is False
    assert layer.activation.kwargs == {"clamp": 10.0}
    assert layer.router_logits_dtype == "fp32"


@pytest.mark.parametrize("architecture", ["MiniMaxM3ForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
def test_minimax_m3_runs_the_swiglu_bias_package(helpers, architecture):
    layer = _layer(helpers, architecture)
    assert layer.routing_method == ("gate-routing", "m3")
    assert layer.activation.kwargs == {"gate_sigmoid_scale": 1.702, "linear_offset": 1.0, "clamp": 7.0}
    assert layer.bias is False


def test_kimi_k3_routes_deepseek_style_with_situ_experts(helpers):
    layer = _layer(
        helpers, "KimiK3ForConditionalGeneration", moe_backend="TRTLLM", sglang_moe_routed_scaling_factor=1.0
    )
    routing = layer.routing_method
    assert type(routing).__name__ == "DeepSeekV3MoeRoutingMethod"
    assert routing.kwargs["is_fused"] is True and routing.kwargs["top_k"] == 8
    assert routing.kwargs["callable_e_score_correction_bias"]() == ("zeros", 256, "fp32")
    assert type(layer.activation).__name__ == "SiTuActivation"
    assert layer.activation.kwargs == {"gate_softcap": 4.0, "linear_softcap": 25.0}
    assert layer.router_logits_dtype == "fp32"


@pytest.mark.parametrize("moe_backend", ["CUTLASS", "TRTLLM"])
def test_glm53_flash_routes_through_its_gate_with_clamped_swiglu(helpers, moe_backend):
    layer = _layer(helpers, "Glm5NextForConditionalGeneration", moe_backend=moe_backend)
    assert layer.routing_method == ("gate-routing", "glm5next")
    assert layer.gate.e_score_correction_bias.zeroed
    (gate_config,) = layer.gate.args
    # The case's shape with the checkpoint's routing constants, as Glm5NextMoE builds it.
    assert (gate_config.hidden_size, gate_config.n_routed_experts, gate_config.num_experts_per_tok) == (7168, 256, 8)
    assert (gate_config.n_group, gate_config.topk_group, gate_config.routed_scaling_factor) == (1, 1, 2.5)
    assert gate_config.scoring_func == "sigmoid" and gate_config.norm_topk_prob is True
    assert layer.gate.kwargs == {"moe_backend": moe_backend}
    assert type(layer.activation).__name__ == "SwigluActivation"
    assert layer.activation.kwargs == {"clamp": 10.0}
    assert layer.router_logits_dtype == "fp32"
    assert layer.bias is False and layer.apply_router_weight_on_input is False


def test_minimax_m2_routes_with_an_fp32_bias(helpers):
    layer = _layer(helpers, "MiniMaxM2ForCausalLM")
    assert type(layer.routing_method).__name__ == "MiniMaxM2MoeRoutingMethod"
    assert layer.routing_method.kwargs["callable_e_score_correction_bias"]() == ("zeros", 256, "fp32")
    assert layer.router_logits_dtype == "fp32"


@pytest.mark.parametrize(
    ("architecture", "from_backend"),
    [
        ("MixtralForCausalLM", False),
        ("Qwen3MoeForCausalLM", True),
        ("Qwen3VLMoeForConditionalGeneration", True),
        # TRT-LLM's Qwen3Next gate is Renormalize, whatever SGLang declares.
        ("Qwen3_5MoeForCausalLM", False),
        ("Qwen3_5MoeForConditionalGeneration", False),
    ],
)
def test_softmax_top_k_families_renormalize_on_bf16_logits(helpers, architecture, from_backend):
    layer = _layer(helpers, architecture)
    assert type(layer.routing_method).__name__ == "RenormalizeMoeRoutingMethod"
    assert layer.routing_method.kwargs == {"top_k": 8}
    assert layer.router_logits_dtype == "bf16"
    assert layer.activation == "default-swiglu"
    assert layer.routing_scales_from_backend is from_backend


def test_llama4_applies_router_weight_on_input(helpers):
    layer = _layer(helpers, "Llama4ForConditionalGeneration")
    assert type(layer.routing_method).__name__ == "Llama4RenormalizeMoeRoutingMethod"
    assert layer.apply_router_weight_on_input is True


def test_gemma4_runs_geglu_with_its_scaled_routing(helpers):
    layer = _layer(helpers, "Gemma4ForConditionalGeneration")
    assert type(layer.routing_method).__name__ == "Gemma4MoeRoutingMethod"
    assert layer.routing_method.kwargs["callable_per_expert_scale"]() == ("ones", 256, "bf16")
    assert layer.activation.kwargs == {"kind": ActivationType.Geglu}


@pytest.mark.parametrize(
    ("moe_backend", "routing_dtype"), [("TRTLLM", "bf16"), ("TRITON", "fp32"), ("CUTLASS", "fp32")]
)
def test_gpt_oss_has_expert_bias_and_swiglu_bias(helpers, moe_backend, routing_dtype):
    layer = _layer(helpers, "GptOssForCausalLM", moe_backend=moe_backend)
    assert layer.routing_method.kwargs == {"top_k": 8, "output_dtype": routing_dtype}
    assert layer.activation.kwargs == {"gate_sigmoid_scale": 1.702, "linear_offset": 1.0, "clamp": 7.0}
    assert layer.bias is True


def test_an_architecture_serving_cannot_load_fails_before_the_build(helpers):
    with pytest.raises(ValueError, match="builds no MoE layer for architecture 'Step3p7FlashForCausalLM'"):
        _layer(helpers, "Step3p7FlashForCausalLM")


def _built(provider, technique, quant="nvfp4"):
    identity = SimpleNamespace(
        provider=provider, technique=technique, canonical=lambda: f"{provider}.{technique}.x.{quant}"
    )
    return SimpleNamespace(backend=SimpleNamespace(descriptor=SimpleNamespace(identity=identity)))


@pytest.mark.parametrize(
    ("provider", "technique", "label"),
    [
        ("trtllm", "cutlass", "moe_torch_flow_cutlass"),
        ("trtllm", "triton", "moe_torch_flow_triton_fp8_block"),
        ("trtllm", "trtllm_gen", "moe_torch_flow_trtllm_gen"),
        ("deepgemm", "cuda", "deepgemm"),
    ],
)
def test_kernel_source_names_the_resolved_implementation(helpers, provider, technique, label):
    assert helpers["_moe_kernel_source"](_built(provider, technique), False) == label


def test_kernel_source_of_the_triton_layer_and_the_min_latency_row(helpers):
    assert helpers["_moe_kernel_source"](_TritonFusedMoE(), False) == "moe_torch_flow_triton"
    assert helpers["_moe_kernel_source"](_built("trtllm", "trtllm_gen"), True) == "moe_torch_flow_min_latency"
    with pytest.raises(RuntimeError, match="min-latency row resolved to trtllm.cutlass"):
        helpers["_moe_kernel_source"](_built("trtllm", "cutlass"), True)


def test_an_unlabeled_implementation_fails_the_case(helpers):
    with pytest.raises(RuntimeError, match="no kernel_source label for MoE implementation flashinfer.trtllm_gen"):
        helpers["_moe_kernel_source"](_built("flashinfer", "trtllm_gen", "bf16"), False)


def test_post_load_weights_runs_on_every_module_serving_would_visit(helpers):
    visited = []

    class Module:
        def __init__(self, name, removed=False, hook=True):
            self.name = name
            self._weights_removed = removed
            if hook:
                self.post_load_weights = lambda: visited.append(name)

    wrapper = Module("ConfigurableMoE", removed=True)
    unit = Module("impl")
    routing = Module("routing", hook=False)
    wrapper.modules = lambda: [wrapper, unit, routing]
    helpers["_post_load_weights"](wrapper)
    assert visited == ["impl"]
