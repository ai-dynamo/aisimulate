# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"

MOE_RUNTIME_DEFAULTS = {
    "sglang_moe_backends": {},
    "sglang_moe_activation": "silu",
    "sglang_moe_is_gated": True,
    "sglang_moe_has_bias": False,
    "sglang_moe_gemm1_alpha": None,
    "sglang_moe_gemm1_clamp_limit": None,
    "sglang_moe_swiglu_limit": None,
    "sglang_moe_scoring_func": "softmax",
    "sglang_moe_routing_method_type": None,
    "sglang_moe_routed_scaling_factor": None,
    "sglang_moe_renormalize": True,
    "sglang_moe_has_correction_bias": False,
    "sglang_moe_num_expert_group": None,
    "sglang_moe_topk_group": None,
    "sglang_moe_apply_router_weight_on_input": False,
}


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    from collector.helper import WORKER_RESTART
    from collector.sglang.moe_config_observation import observe_triton_config_selection

    # The lane guard has dedicated tests; it returns the installed version.
    loaded = {
        "WORKER_RESTART": WORKER_RESTART,
        "_raise_if_unverified_moe_lane": lambda _moe_type: "0.5.23",
        # Record-only Triton config observation (own tests in
        # test_moe_config_observation.py); no SGLang module to observe here.
        "_triton_config_namespaces": lambda: [],
        "observe_triton_config_selection": observe_triton_config_selection,
        "append_triton_config_record": lambda *_args, **_kwargs: None,
        # The architecture only widens SGLang's FlashInfer tuning range
        # (test_collect_moe_serving_paths.py covers it).
        "moe_model_layer": lambda *_args: SimpleNamespace(architecture=None),
        **(namespace or {}),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


def _gptoss_case(*, tp: int, ep: int):
    return SimpleNamespace(
        **(
            MOE_RUNTIME_DEFAULTS
            | {
                "sglang_moe_has_bias": True,
                "sglang_moe_gemm1_alpha": 1.702,
                "sglang_moe_gemm1_clamp_limit": 7.0,
            }
        ),
        num_tokens_list=[128],
        hidden_size=2880,
        inter_size=2880,
        topk=4,
        num_experts=128,
        tp=tp,
        ep=ep,
        model_name="openai/gpt-oss-120b",
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GptOssForCausalLM",
    )


def _populate_gptoss_cases(
    cases,
    *,
    sm_version=100,
    allowed_mode="w4a8_mxfp4_mxfp8",
    module_config=None,
    resolved_backend=None,
):
    """SGLang's queued MoE rows for these model layers through the shared op plan."""
    from collector import op_plans
    from collector.case_generator import MoeCommonTestCase

    allowed_modes = {allowed_mode} if isinstance(allowed_mode, str) else set(allowed_mode)
    layers = [MoeCommonTestCase(**vars(case)) for case in cases]
    by_row = {
        (layer.model_name, layer.hidden_size, layer.inter_size, layer.topk, layer.num_experts): layer
        for layer in layers
    }

    def runner(_case, mode, sm):
        if resolved_backend:
            return resolved_backend
        if sm == 90 and mode == "int4_wo":
            return "marlin"
        if mode == "nvfp4" and sm in {100, 103}:
            return "flashinfer_trtllm"
        return "flashinfer_mxfp4" if mode in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"} else "triton"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(op_plans, "get_common_moe_test_cases", lambda: layers)
        patch.setattr(op_plans, "moe_planned_modes", lambda _model, _declared: sorted(allowed_modes))
        patch.setattr(op_plans, "get_moe_quantization_module_config", lambda *_args, **_kwargs: module_config or {})
        patch.setattr(op_plans, "get_sglang_moe_backend", runner)
        patch.setattr(op_plans, "moe_model_layer", lambda *row: by_row[row])
        return op_plans.plan_cases("moe", "sglang", sm_version)


def _glm5_case(model_name: str):
    return SimpleNamespace(
        **(
            MOE_RUNTIME_DEFAULTS
            | {
                "sglang_moe_scoring_func": "sigmoid",
                "sglang_moe_routing_method_type": "DeepSeekV3",
                "sglang_moe_routed_scaling_factor": 2.5,
                "sglang_moe_has_correction_bias": True,
                "sglang_moe_num_expert_group": 1,
                "sglang_moe_topk_group": 1,
            }
        ),
        num_tokens_list=[128],
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        tp=1,
        ep=1,
        model_name=model_name,
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GlmMoeDsaForCausalLM",
    )


def test_glm_nvfp4_artifacts_deduplicate_on_the_moe_consumer_key():
    populated = _populate_gptoss_cases(
        [
            _glm5_case("nvidia/GLM-5-NVFP4"),
            _glm5_case("nvidia/GLM-5.2-NVFP4"),
        ],
        allowed_mode="nvfp4",
    )

    assert populated
    assert len(populated) == 1
    assert {case[8] for case in populated} == {"nvidia/GLM-5-NVFP4"}


def test_kimi_int4_population_restores_all_hopper_ep_slices():
    from collector.case_generator import get_common_moe_test_cases

    common_cases = [case for case in get_common_moe_test_cases() if case.model_name == "moonshotai/Kimi-K2.5"]
    populated = _populate_gptoss_cases(
        common_cases,
        sm_version=90,
        allowed_mode="int4_wo",
        module_config={"group_size": 32},
    )

    # 114 (quant, shape, tp, ep, distribution) series x 29 token points,
    # including the added 32K and 64K anchors.
    assert len(populated) == 114
    assert sum(len(case[1]) for case in populated) == 3306
    assert sum(len(case[1]) for case in populated if case[7] == 64) == 261
    assert sum(len(case[1]) for case in populated if case[7] == 128) == 174
    # int4_wo opened to SM100/103: serving
    # auto-selects flashinfer_trtllm for Kimi INT4 there (server_args.py:3736),
    # replacing the old zero-population contract.
    sm100 = _populate_gptoss_cases(
        common_cases,
        sm_version=100,
        allowed_mode="int4_wo",
        module_config={"group_size": 32},
    )
    assert len(sm100) == 114 and sum(len(case[1]) for case in sm100) == 3306
    from types import SimpleNamespace

    from collector.case_generator import get_sglang_moe_backend

    assert get_sglang_moe_backend(SimpleNamespace(sglang_moe_backends={}), "int4_wo", 100) == "flashinfer_trtllm"


def test_fp8_block_population_keeps_runtime_alignment_failures_attempted():
    invalid = _glm5_case("example/fp8")
    invalid.hidden_size = 128
    invalid.inter_size = 384
    invalid.tp = 2
    invalid.num_experts = 8
    invalid.topk = 2
    valid = SimpleNamespace(**vars(invalid))
    valid.inter_size = 512

    assert len(_populate_gptoss_cases([invalid], sm_version=90, allowed_mode="fp8_block")) == 1
    assert len(_populate_gptoss_cases([valid], sm_version=90, allowed_mode="fp8_block")) == 1


def test_gemma_gelu_population_keeps_runtime_vector_failures_attempted():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "google/gemma-4-26B-A4B"
    invalid.hidden_size = 2816
    invalid.inter_size = 704
    invalid.topk = 8
    invalid.sglang_moe_activation = "gelu"
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8

    assert len(_populate_gptoss_cases([invalid], sm_version=90, allowed_mode="bfloat16")) == 1
    assert len(_populate_gptoss_cases([valid], sm_version=90, allowed_mode="bfloat16")) == 1

    blackwell_valid = SimpleNamespace(**vars(invalid))
    blackwell_valid.tp = 4
    assert len(_populate_gptoss_cases([valid], sm_version=100, allowed_mode="bfloat16")) == 1
    assert len(_populate_gptoss_cases([blackwell_valid], sm_version=100, allowed_mode="bfloat16")) == 1


def test_bfloat16_flashinfer_cutlass_population_keeps_runtime_failures_attempted():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    invalid.hidden_size = 2688
    invalid.inter_size = 1856
    invalid.topk = 6
    invalid.sglang_moe_activation = "relu2"
    invalid.sglang_moe_is_gated = False
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8
    ultra = SimpleNamespace(**vars(invalid))
    ultra.model_name = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    ultra.hidden_size = 8192
    ultra.inter_size = 5120

    assert (
        len(
            _populate_gptoss_cases(
                [invalid], sm_version=90, allowed_mode="bfloat16", resolved_backend="flashinfer_cutlass"
            )
        )
        == 1
    )
    assert (
        len(
            _populate_gptoss_cases(
                [valid, ultra], sm_version=90, allowed_mode="bfloat16", resolved_backend="flashinfer_cutlass"
            )
        )
        == 2
    )
    assert (
        len(
            _populate_gptoss_cases(
                [invalid], sm_version=100, allowed_mode="bfloat16", resolved_backend="flashinfer_trtllm"
            )
        )
        == 1
    )


def test_dsv4_fp4_population_keeps_runtime_limits_attempted():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "deepseek-ai/DeepSeek-V4-Pro"
    invalid.architecture = "DeepseekV4ForCausalLM"
    invalid.hidden_size = 7168
    invalid.inter_size = 3072
    invalid.topk = 6
    invalid.num_experts = 384
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8

    assert len(_populate_gptoss_cases([invalid], sm_version=90, allowed_mode="w4a16_mxfp4")) == 1
    populated = _populate_gptoss_cases([valid], sm_version=90, allowed_mode="w4a16_mxfp4")
    assert len(populated) == 1
    assert populated[0][-1] is True

    blackwell = _populate_gptoss_cases([valid], sm_version=100, allowed_mode="w4a8_mxfp4_mxfp8")
    assert len(blackwell) == 1
    assert blackwell[0][0] == "w4a8_mxfp4_mxfp8"

    runtime_limited = SimpleNamespace(**vars(valid))
    runtime_limited.ep = 2
    runtime_limited.tp = 16
    runtime_limited.num_tokens_list = [16384]
    attempted = _populate_gptoss_cases(
        [runtime_limited],
        sm_version=100,
        allowed_mode="w4a8_mxfp4_mxfp8",
    )
    assert len(attempted) == 1
    assert attempted[0][1:8] == [[16384], 7168, 3072, 6, 384, 16, 2]


def test_sm120_shared_memory_predictions_do_not_remove_moe_cases():
    case = _gptoss_case(tp=32, ep=1)
    case.model_name = "mistralai/Mixtral-8x7B-v0.1"
    case.hidden_size = 4096
    case.inter_size = 14336
    case.topk = 2
    case.num_experts = 8
    case.num_tokens_list = [16]

    populated = _populate_gptoss_cases(
        [case],
        sm_version=120,
        allowed_mode="fp8_block",
        resolved_backend="triton",
    )

    assert len(populated) == 1


def test_sm90_population_excludes_nvfp4_instead_of_using_marlin():
    assert not _populate_gptoss_cases(
        [_glm5_case("nvidia/GLM-5.2-NVFP4")],
        sm_version=90,
        allowed_mode="nvfp4",
    )


def test_moe_population_deduplicates_equal_persisted_keys_and_rejects_semantic_conflicts():
    first = _gptoss_case(tp=4, ep=8)
    duplicate = SimpleNamespace(**vars(first))
    duplicate.model_name = "example/equivalent-gptoss"

    assert len(_populate_gptoss_cases([first, duplicate])) == 1

    duplicate.sglang_moe_activation = "gelu"
    with pytest.raises(ValueError, match="time different layers under one consumer key"):
        _populate_gptoss_cases([first, duplicate])


def test_case_generator_preserves_representative_sglang_moe_runtime_contracts():
    from collector.case_generator import get_common_moe_test_cases, get_sglang_moe_backend

    cases = get_common_moe_test_cases()

    def model_case(model_name):
        return next(case for case in cases if case.model_name == model_name)

    deepseek = model_case("deepseek-ai/DeepSeek-V3")
    assert (deepseek.sglang_moe_scoring_func, deepseek.sglang_moe_routed_scaling_factor) == ("sigmoid", 2.5)
    assert get_sglang_moe_backend(deepseek, "fp8_block", 100) == "flashinfer_trtllm"

    gemma = model_case("google/gemma-4-26B-A4B")
    assert gemma.sglang_moe_activation == "gelu"

    gpt_oss = model_case("openai/gpt-oss-120b")
    assert (gpt_oss.sglang_moe_has_bias, gpt_oss.sglang_moe_gemm1_alpha, gpt_oss.sglang_moe_gemm1_clamp_limit) == (
        True,
        1.702,
        7.0,
    )
    assert get_sglang_moe_backend(gpt_oss, "w4a16_mxfp4", 90) == "triton"

    kimi = model_case("moonshotai/Kimi-K2.5")
    assert get_sglang_moe_backend(kimi, "int4_wo", 90) == "marlin"
    assert (
        kimi.sglang_moe_scoring_func,
        kimi.sglang_moe_routing_method_type,
        kimi.sglang_moe_routed_scaling_factor,
        kimi.sglang_moe_has_correction_bias,
        kimi.sglang_moe_num_expert_group,
        kimi.sglang_moe_topk_group,
    ) == ("sigmoid", "DeepSeekV3", 2.827, True, 1, 1)

    dsv4 = model_case("deepseek-ai/DeepSeek-V4-Pro")
    assert (dsv4.sglang_moe_scoring_func, dsv4.sglang_moe_has_correction_bias) == ("sqrtsoftplus", True)
    # w4a16_mxfp4 is no longer in the Pro artifact's allowed_modes (owner
    # decision); the retained native-FP4 plan is the w4a8 path.
    assert get_sglang_moe_backend(dsv4, "w4a8_mxfp4_mxfp8", 100) == "flashinfer_mxfp4"

    llama4 = model_case("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    assert (
        llama4.sglang_moe_scoring_func,
        llama4.sglang_moe_routing_method_type,
        llama4.sglang_moe_renormalize,
        llama4.sglang_moe_apply_router_weight_on_input,
    ) == ("sigmoid", "Llama4", False, True)

    nemotron = model_case("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert (nemotron.sglang_moe_activation, nemotron.sglang_moe_is_gated) == ("relu2", False)
    assert get_sglang_moe_backend(nemotron, "nvfp4", 100) == "flashinfer_trtllm"

    source = SOURCE_PATH.read_text()
    assert "use_grouped_topk=num_expert_group is not None and topk_group is not None" in source
    assert "num_expert_group=num_expert_group" in source


def test_sm120_population_runs_gptoss_as_w4a8_and_skips_w4a16():
    # SGLang serves GPT-OSS MXFP4 on SM120 as MXFP8 x MXFP4 on flashinfer_mxfp4
    # (gpt_oss.py:69-74, mxfp4.py:426-427) and no W4A16 lane (collector/skips.py),
    # so only the w4a8_mxfp4_mxfp8 slice is planned there.
    populated = _populate_gptoss_cases(
        [_gptoss_case(tp=1, ep=1)],
        sm_version=120,
        allowed_mode={"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"},
    )

    assert {case[0] for case in populated} == {"w4a8_mxfp4_mxfp8"}
    assert all(case[12] == "flashinfer_mxfp4" for case in populated)


def _k3_case(*, tp: int = 1, ep: int = 1):
    k3 = _gptoss_case(tp=tp, ep=ep)
    k3.model_name = "moonshotai/Kimi-K3"
    k3.architecture = "KimiK3ForConditionalGeneration"
    return k3


def test_sm120_population_keeps_kimi_k3_w4a16_and_skips_its_mxfp8_activations():
    # Kimi-K3 keeps auto on SM120 (kimi_k3.py:172-190): Mxfp4MoEMethod's Triton
    # runner on BF16-upcast experts, so only the W4A16 slice is served.
    populated = _populate_gptoss_cases(
        [_k3_case()], sm_version=120, allowed_mode={"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}, resolved_backend="triton"
    )
    assert {case[0] for case in populated} == {"w4a16_mxfp4"}
    assert all(case[12] == "triton" for case in populated)


def test_kimi_k3_row_carries_serving_situ_and_triton_pins():
    from collector.case_generator import get_common_moe_test_cases, get_sglang_moe_backend

    k3 = next(case for case in get_common_moe_test_cases() if case.model_name == "moonshotai/Kimi-K3")
    # models/kimi_k3.py:503-521: activation=config.hidden_act, the SiTU betas as gemm1.
    assert (k3.sglang_moe_activation, k3.sglang_moe_gemm1_alpha, k3.sglang_moe_gemm1_clamp_limit) == (
        "situ",
        4.0,
        25.0,
    )
    # auto off SM100/103/107 resolves Mxfp4MoEMethod to Triton (mxfp4.py:1443-1450).
    assert [get_sglang_moe_backend(k3, "w4a16_mxfp4", sm) for sm in (89, 90, 120)] == ["triton"] * 3
    assert get_sglang_moe_backend(k3, "w4a8_mxfp4_mxfp8", 100) == "flashinfer_mxfp4"


@pytest.mark.parametrize("sm_version", [100, 103])
def test_sm100_population_runs_kimi_k3_as_w4a8_only(sm_version):
    # SM100/103 serve K3 on flashinfer_mxfp4 at the default precision, MXFP8
    # activations (fields/exec_.py:892-897, moe_runner/flashinfer_trtllm.py:1145-1166).
    populated = _populate_gptoss_cases(
        [_k3_case()], sm_version=sm_version, allowed_mode={"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
    )
    assert {case[0] for case in populated} == {"w4a8_mxfp4_mxfp8"}


@pytest.mark.parametrize("sm_version", [89, 90])
def test_gptoss_w4a16_runs_triton_kernel_at_ep1_where_serving_keeps_auto(sm_version):
    # SGLang's GPT-OSS override serves triton_kernel at ep_size 1 and leaves
    # auto (Triton, the per-SM pin) at EP>1 (model_overrides/gpt_oss.py:102-120).
    populated = _populate_gptoss_cases(
        [_gptoss_case(tp=1, ep=1), _gptoss_case(tp=4, ep=1), _gptoss_case(tp=1, ep=8)],
        sm_version=sm_version,
        allowed_mode="w4a16_mxfp4",
        resolved_backend="triton",
    )
    assert {(case[6], case[7]): case[12] for case in populated} == {
        (1, 1): "triton_kernel",
        (4, 1): "triton_kernel",
        (1, 8): "triton",
    }

    # Kimi-K3 has no such override: auto stays Triton at every EP.
    k3 = _populate_gptoss_cases(
        [_k3_case()], sm_version=sm_version, allowed_mode="w4a16_mxfp4", resolved_backend="triton"
    )
    assert [case[12] for case in k3] == ["triton"]


@pytest.mark.parametrize("sm_version", [100, 103])
def test_gptoss_on_blackwell_runs_w4a8_on_flashinfer_mxfp4_at_every_ep(sm_version):
    # gpt_oss.py:64-75 selects flashinfer_mxfp4 on SM100/103 before the EP check, at
    # MXFP8 activations: the W4A8 cases run it and collector/skips.py leaves W4A16 out.
    populated = _populate_gptoss_cases(
        [_gptoss_case(tp=1, ep=1), _gptoss_case(tp=1, ep=8)], sm_version=sm_version, allowed_mode="w4a8_mxfp4_mxfp8"
    )
    assert [case[12] for case in populated] == ["flashinfer_mxfp4", "flashinfer_mxfp4"]
    assert not _populate_gptoss_cases([_gptoss_case(tp=1, ep=1)], sm_version=sm_version, allowed_mode="w4a16_mxfp4")


def test_moe_population_rejects_an_architecture_sglang_builds_differently_under_one_key():
    # GPT-OSS (EP1 runner) and Gemma-4 (routing function) build a different
    # layer than another architecture's row with the same fields.
    with pytest.raises(ValueError, match="time different layers under one consumer key"):
        _populate_gptoss_cases([_gptoss_case(tp=1, ep=1), _k3_case()], sm_version=90, allowed_mode="w4a16_mxfp4")


@pytest.mark.parametrize(("tp", "ep"), [(4, 8), (32, 1), (32, 8)])
def test_gptoss_mxfp4_population_retains_tp_and_ep_buckets(tp, ep):
    populated = _populate_gptoss_cases([_gptoss_case(tp=tp, ep=ep)])

    assert len(populated) == 1
    assert populated[0][0] == "w4a8_mxfp4_mxfp8"
    assert populated[0][6:8] == [tp, ep]
    assert populated[0][12] == "flashinfer_mxfp4"
    assert populated[0][15:18] == [True, 1.702, 7.0]
    assert populated[0][-1] is False


@pytest.mark.parametrize(
    ("mode", "sm_version", "expected"),
    [
        ("nvfp4", 100, "flashinfer_trtllm"),
        ("nvfp4", 103, "flashinfer_trtllm"),
        ("nvfp4", 120, "flashinfer_cutlass"),
        ("int4_wo", 89, "marlin"),
        ("int4_wo", 90, "marlin"),
        ("int4_wo", 120, "marlin"),
        ("w4a16_mxfp4", 90, "triton"),
        ("w4a16_mxfp4", 120, "marlin"),
        ("w4a8_mxfp4_mxfp8", 100, "flashinfer_mxfp4"),
        ("w4a8_mxfp4_mxfp8", 120, "flashinfer_mxfp4"),
    ],
)
def test_yaml_backend_map_matches_sglang_0514(mode, sm_version, expected):
    from collector.case_generator import get_sglang_moe_backend

    assert get_sglang_moe_backend(SimpleNamespace(sglang_moe_backends={}), mode, sm_version) == expected


@pytest.mark.parametrize("mode", ["nvfp4", "w4a8_mxfp4_mxfp8"])
def test_fp4_modes_reject_marlin_backend(mode):
    from collector.case_generator import get_sglang_moe_backend

    test_case = SimpleNamespace(sglang_moe_backends={mode: {90: "marlin"}})
    with pytest.raises(ValueError, match="Marlin is only valid for the weight-only modes"):
        get_sglang_moe_backend(test_case, mode, 90)


def test_w4a16_mxfp4_accepts_marlin_backend():
    # SGLang 0.5.14 serving itself selects Marlin for MXFP4 w4a16 on SM120
    # (server_args.py:3876-3887); weight-only identity is preserved.
    from collector.case_generator import get_sglang_moe_backend

    test_case = SimpleNamespace(sglang_moe_backends={})
    assert get_sglang_moe_backend(test_case, "w4a16_mxfp4", 120) == "marlin"


def test_sm90_nvfp4_has_no_backend():
    from collector.case_generator import get_sglang_moe_backend

    with pytest.raises(ValueError, match="No SGLang MoE backend"):
        get_sglang_moe_backend(SimpleNamespace(sglang_moe_backends={}), "nvfp4", 90)


@pytest.mark.parametrize("persisted", [True, False])
@pytest.mark.parametrize(
    ("moe_type", "moe_backend", "model_name", "sm_version", "kernel_source"),
    [
        ("int4_wo", "marlin", "moonshotai/Kimi-K2.5", 90, "sglang_marlin_moe"),
        ("int4_wo", "flashinfer_trtllm", "moonshotai/Kimi-K2.5", 100, "sglang_flashinfer_trtllm_moe"),
        ("w4a16_mxfp4", "triton", "openai/gpt-oss-120b", 90, "sglang_fused_moe_triton"),
        (
            "w4a8_mxfp4_mxfp8",
            "flashinfer_mxfp4",
            "openai/gpt-oss-120b",
            100,
            "sglang_flashinfer_trtllm_moe",
        ),
    ],
)
def test_quantized_moe_uses_framework_path_and_fails_closed(
    persisted,
    moe_type,
    moe_backend,
    model_name,
    sm_version,
    kernel_source,
):
    framework_calls = []
    logged = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 26,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "H200",
        ),
    )

    def framework_benchmark(**kwargs):
        framework_calls.append(kwargs)
        for num_tokens in kwargs["num_tokens_list"]:
            kwargs["on_result"](num_tokens, 1.25, {"power": 100.0}, kernel_source)
        return kernel_source

    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "_benchmark_moe_layer": framework_benchmark,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {"group_size": 32},
            "get_sm_version": lambda: sm_version,
            "_fmoe_kernels_mod": SimpleNamespace(_B_DESC_CACHE=SimpleNamespace(clear=lambda: None)),
            "gc": SimpleNamespace(collect=lambda: None),
            "log_perf": lambda **kwargs: logged.append(kwargs) or persisted,
            "_dist_version": lambda _name: "0.5.14",
            "EXIT_CODE_RESTART": 10,
        },
    )["run_moe_torch"]

    args = (
        moe_type,
        128,
        7168,
        2048,
        8,
        384,
        1,
        1,
        model_name,
    )
    if persisted:
        from collector.helper import WorkerRestartSignal

        assert isinstance(
            run(
                *args,
                distributed="balanced",
                moe_backend=moe_backend,
                perf_filename="moe.csv",
            ),
            WorkerRestartSignal,
        )
    else:
        with pytest.raises(RuntimeError, match="Failed to persist SGLang MoE performance row"):
            run(
                *args,
                distributed="balanced",
                moe_backend=moe_backend,
                perf_filename="moe.csv",
            )

    assert framework_calls[0]["moe_backend"] == moe_backend
    assert logged[0]["kernel_source"] == kernel_source


def test_bfloat16_triton_case_builds_the_serving_layer_and_fails_closed():
    """Mixtral/gemma-4 BF16 and FP8-block Triton cases build FusedMoE + TopK like the
    other cases: serving runs no other path, and the retired raw fused_moe call piled
    every non-local EP slot onto expert 0 (topk_ids.clamp(min=0))."""
    layer_calls = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        device=lambda value: value,
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "H200",
        ),
    )

    def layer_benchmark(**kwargs):
        layer_calls.append(kwargs)
        for num_tokens in kwargs["num_tokens_list"]:
            kwargs["on_result"](num_tokens, 1.25, {"power": 100.0}, "sglang_fused_moe_triton")
        return "sglang_fused_moe_triton"

    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "_benchmark_moe_layer": layer_benchmark,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "get_sm_version": lambda: 90,
            "log_perf": lambda **_kwargs: False,
        },
    )["run_moe_torch"]

    with pytest.raises(RuntimeError, match="Failed to persist SGLang MoE performance row"):
        run(
            "bfloat16",
            128,
            4096,
            14336,
            2,
            8,
            1,
            8,
            "mistralai/Mixtral-8x7B-v0.1",
            distributed="power_law",
            power_law_alpha=1.01,
            perf_filename="moe.csv",
        )

    assert [call["moe_backend"] for call in layer_calls] == ["triton"]
    assert layer_calls[0]["moe_ep_size"] == 8


@pytest.mark.parametrize(
    ("moe_type", "hidden_size", "inter_size", "tp", "is_gated", "error"),
    [
        ("fp8_block", 128, 384, 2, True, "fp8_block.*local_inter_size"),
        ("int4_wo", 128, 80, 1, True, "INT4-WO.*group_size"),
    ],
)
def test_runtime_rejects_misaligned_quantized_cases(moe_type, hidden_size, inter_size, tp, is_gated, error):
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "get_sm_version": lambda: 90,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {"group_size": 32},
        },
    )["run_moe_torch"]

    with pytest.raises(ValueError, match=error):
        run(
            moe_type,
            128,
            hidden_size,
            inter_size,
            2,
            8,
            tp,
            1,
            "example/model",
            moe_backend="triton" if moe_type == "fp8_block" else "marlin",
            is_gated=is_gated,
            perf_filename="moe.csv",
        )


@pytest.mark.parametrize("moe_type", ["nvfp4", "w4a8_mxfp4_mxfp8"])
def test_runtime_rejects_fp4_modes_on_marlin(moe_type):
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={"torch": fake_torch, "get_sm_version": lambda: 90},
    )["run_moe_torch"]

    with pytest.raises(ValueError, match="Marlin is only valid for the weight-only modes"):
        run(
            moe_type,
            128,
            128,
            128,
            2,
            8,
            1,
            1,
            "example/model",
            moe_backend="marlin",
            perf_filename="moe.csv",
        )


def test_runtime_rejects_misaligned_sm90_dsv4_w4a16_case():
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={"torch": fake_torch, "get_sm_version": lambda: 90},
    )["run_moe_torch"]

    with pytest.raises(ValueError, match=r"SM90 DeepSeek-V4 W4A16.*local_inter_size"):
        run(
            "w4a16_mxfp4",
            128,
            7168,
            3072,
            6,
            384,
            16,
            1,
            "deepseek-ai/DeepSeek-V4-Pro",
            moe_backend="flashinfer_mxfp4",
            is_fp4_experts=True,
            perf_filename="moe.csv",
        )


def test_sm100_dsv4_w4a8_unaligned_local_inter_runs_as_serving_pads_it():
    """Mxfp4FlashinferTrtllmMoEMethod.process_weights_after_loading pads the local
    intermediate width to 128 (sglang v0.5.23
    srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:55-100), so the collector no
    longer refuses these cases before building the layer."""
    layer_calls = []
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            memory_allocated=lambda _device: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "B200",
        ),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "get_sm_version": lambda: 100,
            "_benchmark_moe_layer": lambda **kwargs: layer_calls.append(kwargs),
        },
    )["run_moe_torch"]

    run(
        "w4a8_mxfp4_mxfp8",
        128,
        7168,
        3072,
        6,
        384,
        32,
        1,
        "deepseek-ai/DeepSeek-V4-Pro",
        moe_backend="flashinfer_mxfp4",
        is_fp4_experts=True,
        perf_filename="moe.csv",
    )

    assert [(call["moe_tp_size"], call["is_fp4_experts"]) for call in layer_calls] == [(32, True)]


def test_framework_moe_router_logits_are_explicitly_float32():
    source = ast.get_source_segment(
        SOURCE_PATH.read_text(),
        next(
            node
            for node in ast.parse(SOURCE_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_moe_layer"
        ),
    )

    assert "balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)" in source
    assert "device=device, dtype=torch.float32" in source.split("power_law_logits_v3", maxsplit=1)[1]


def test_framework_int4_builds_grouped_compressed_tensors_config():
    source = ast.get_source_segment(
        SOURCE_PATH.read_text(),
        next(
            node
            for node in ast.parse(SOURCE_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_quant_config"
        ),
    )

    assert "CompressedTensorsConfig.from_config" in source
    assert '"strategy": "group"' in source
    assert '"group_size": int4_group_size' in source


def test_framework_moe_source_comes_from_the_constructed_quant_method():
    source_text = SOURCE_PATH.read_text()
    tree = ast.parse(source_text)
    functions = {
        node.name: ast.get_source_segment(source_text, node) for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    kernel_source = functions["_kernel_source"]
    run_moe = functions["run_moe_torch"]

    assert "moe_layer.runner_backend" in kernel_source
    for method_name in (
        "Mxfp4MoEMethod",
        "Mxfp4FlashinferCutlassMoEMethod",
        "Mxfp4FlashinferTrtllmMoEMethod",
        "CompressedTensorsWNA16MoE",
        "CompressedTensorsMxInt4MoE",
        "ModelOptNvFp4FusedMoEMethod",
        "Fp8MoEMethod",
        "UnquantizedFusedMoEMethod",
    ):
        assert method_name in source_text
    assert 'model_name == "deepseek-ai/DeepSeek-V4-Pro"' not in run_moe


def test_moe_population_emits_one_case_per_token_sweep():
    """SGLang nvfp4 shards took 4.4-5.8 h against 1-1.4 h for vLLM on the same
    shapes because every token point was its own case and rebuilt the framework layer.
    Cases now carry the whole sweep (vLLM/TRT-LLM layout) and dedup on the series."""
    case = _gptoss_case(tp=1, ep=1)
    case.num_tokens_list = [1, 8, 64, 512]
    duplicate = SimpleNamespace(**vars(case))
    duplicate.model_name = "example/equivalent-gptoss"

    populated = _populate_gptoss_cases([case, duplicate])

    assert len(populated) == 1
    assert populated[0][1] == [1, 8, 64, 512]
    assert populated[0][8] == "openai/gpt-oss-120b"


def test_run_moe_torch_times_every_token_point_on_one_framework_layer():
    logged = []
    framework_calls = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        device=lambda value: value,
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "B200",
        ),
    )

    def framework_benchmark(**kwargs):
        framework_calls.append(kwargs["num_tokens_list"])
        for num_tokens in kwargs["num_tokens_list"]:
            kwargs["on_result"](num_tokens, float(num_tokens), {"power": 1.0}, "sglang_flashinfer_trtllm_moe")
        return "sglang_flashinfer_trtllm_moe"

    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "_benchmark_moe_layer": framework_benchmark,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "get_sm_version": lambda: 100,
            "_fmoe_kernels_mod": SimpleNamespace(_B_DESC_CACHE=SimpleNamespace(clear=lambda: None)),
            "gc": SimpleNamespace(collect=lambda: None),
            "log_perf": lambda **kwargs: logged.append(kwargs) or True,
            "EXIT_CODE_RESTART": 10,
        },
    )["run_moe_torch"]

    assert (
        run(
            "nvfp4",
            [1, 64, 4096],
            7168,
            2048,
            8,
            256,
            1,
            8,
            "nvidia/GLM-5-NVFP4",
            distributed="power_law",
            power_law_alpha=1.2,
            moe_backend="flashinfer_trtllm",
            perf_filename="moe.csv",
        )
        is None
    )
    # One layer build for the whole sweep, one persisted row per token point.
    assert framework_calls == [[1, 64, 4096]]
    assert [row["item_list"][0]["num_tokens"] for row in logged] == [1, 64, 4096]
    assert [row["item_list"][0]["latency"] for row in logged] == [1.0, 64.0, 4096.0]
    assert {row["item_list"][0]["distribution"] for row in logged} == {"power_law_1.2"}
    assert {row["kernel_source"] for row in logged} == {"sglang_flashinfer_trtllm_moe"}

    # A bare int (fills, direct callers) still measures exactly that point.
    logged.clear()
    run(
        "nvfp4",
        768,
        7168,
        2048,
        8,
        256,
        1,
        8,
        "nvidia/GLM-5-NVFP4",
        moe_backend="flashinfer_trtllm",
        perf_filename="moe.csv",
    )
    assert [row["item_list"][0]["num_tokens"] for row in logged] == [768]


def test_triton_case_persists_each_token_point_with_its_config_record():
    logged = []
    sidecar = []
    layer_calls = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        device=lambda value: value,
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "H200",
        ),
    )

    def layer_benchmark(**kwargs):
        layer_calls.append(kwargs["num_tokens_list"])
        for num_tokens in kwargs["num_tokens_list"]:
            kwargs["on_result"](num_tokens, 0.5 * num_tokens, {"power": 1.0}, "sglang_fused_moe_triton")
        return "sglang_fused_moe_triton"

    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "_benchmark_moe_layer": layer_benchmark,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "get_sm_version": lambda: 90,
            "log_perf": lambda **kwargs: logged.append(kwargs) or True,
            "append_triton_config_record": lambda _path, row, source, _observed: sidecar.append(
                (row["num_tokens"], source)
            ),
        },
    )["run_moe_torch"]

    run(
        "bfloat16",
        [16, 32],
        4096,
        14336,
        2,
        8,
        1,
        1,
        "mistralai/Mixtral-8x7B-v0.1",
        distributed="balanced",
        perf_filename="moe.csv",
    )

    assert layer_calls == [[16, 32]]
    assert [row["item_list"][0]["num_tokens"] for row in logged] == [16, 32]
    assert [row["kernel_source"] for row in logged] == ["sglang_fused_moe_triton"] * 2
    assert sidecar == [(16, "sglang_fused_moe_triton"), (32, "sglang_fused_moe_triton")]
