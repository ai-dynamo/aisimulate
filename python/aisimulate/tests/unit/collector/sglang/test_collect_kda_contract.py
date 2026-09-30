# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_kda.py"


def test_kda_context_raises_on_conv_int32_offset_overflow():
    # Silicon-proven Triton kernel limit (see the guard's own comment in the
    # collector): the guard must span the full 3-block mixed_qkv buffer, not
    # the per-block proj_size.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "total_tokens * conv_channels >= 2**31" in source


def test_kda_case_phases_cover_context_generation_verify():
    # The registry getter must emit all three phases for every declared shape;
    # verify rows carry the draft-token width in the seq_len slot. Asserted on
    # the shared spec generator (importable without torch), which both the
    # sglang and vllm backend getters adapt.
    from collector.case_generator import get_common_kda_test_cases

    phases = {case.phase for case in get_common_kda_test_cases()}
    assert phases == {"context", "generation", "verify"}


def test_kda_case_generator_adds_small_batch_long_context_sweep():
    # Every shape gets the full context grid plus a separate small-batch
    # context case reaching 131072 tokens per request; the long lengths never
    # cross the full batch sweep.
    from collector.case_generator import get_common_kda_test_cases

    cases = get_common_kda_test_cases()
    shapes = {(case.d_model, case.num_v_heads) for case in cases}
    for shape in shapes:
        context = [case for case in cases if (case.d_model, case.num_v_heads) == shape and case.phase == "context"]
        assert len(context) == 2
        full, long = sorted(context, key=lambda case: max(case.seq_len_list))
        assert max(full.seq_len_list) == 32768
        assert long.seq_len_list == [65536, 131072]
        assert long.batch_size_list == [1, 2]
        assert not set(full.seq_len_list) & set(long.seq_len_list)


def _exec_functions(names, namespace):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_glm5_next_phase_router_keeps_one_token_extends_on_prefill():
    # SGLang runs every non-verify extend batch through forward_extend, so
    # seq_len=1 context cells stay on the prefill kernels; verify is not
    # collected for the nextn=0 GLM baseline.
    calls = []
    namespace = {
        "run_glm5_next_kda_context": lambda **kwargs: calls.append(("context", kwargs["seq_len_list"])),
        "run_glm5_next_kda_generation": lambda **kwargs: calls.append(("generation", kwargs["batch_size_list"])),
    }
    _exec_functions({"run_glm5_next_kda_torch"}, namespace)
    router = namespace["run_glm5_next_kda_torch"]
    args = (4096, 4, 32, 128, 32, 128, [1, 2])
    router("context", *args, [1, 2], model_name="m")
    router("generation", *args, None, model_name="m")
    assert calls == [("context", [1, 2]), ("generation", [1, 2])]
    with pytest.raises(NotImplementedError, match="verify"):
        router("verify", *args, [2], model_name="m")


def test_glm5_next_model_paths_route_only_glm_rows():
    namespace = _exec_functions({"GLM5_NEXT_KDA_MODEL_PATHS", "_is_glm5_next_kda"}, {})
    assert namespace["_is_glm5_next_kda"]("zai-org/GLM-5.3-Flash")
    assert namespace["_is_glm5_next_kda"]("nvidia/GLM-5.3-Flash-NVFP4")
    assert not namespace["_is_glm5_next_kda"]("moonshotai/Kimi-K3")


def test_glm5_next_decode_uses_bounded_gate_recurrence_not_packed_decode():
    # kda_backend.py:739-742 skips the packed T=1 kernel when lower_bound is
    # set, so GLM decode runs fused_sigmoid_gating_delta_rule_update(is_kda).
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    decode = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_glm5_next_kda_generation"
    )
    names = {
        call.func.id for call in ast.walk(decode) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "fused_sigmoid_gating_delta_rule_update" in names
    assert "causal_conv1d_update" in names
    assert "fused_recurrent_kda_packed_decode" not in names


@pytest.mark.parametrize(
    ("version", "accepted"),
    [("0.5.16", True), ("0.5.17", False), ("0.5.19", False), ("0.5.20", True), ("0.5.21", False)],
)
def test_kda_compat_admits_kimi_branch_and_glm_runtime(version, accepted):
    from collector.version_resolver import _check_compat

    declaration = next(
        node.value.value
        for node in ast.parse(SOURCE_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__compat__" for t in node.targets)
    )
    assert _check_compat(declaration, version) is accepted


def test_glm5_next_chunk_kda_int32_offset_guard_precedes_the_scan():
    # chunk_kda's Triton kernels use int32 (bos * H + i_h) * K offsets
    # (fla/kda.py:265-268 @v0.5.20); an overflowing launch is an illegal
    # address that poisons the worker for every later cell.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    context = source[source.index("def run_glm5_next_kda_context") : source.index("def run_glm5_next_kda_generation")]
    assert "if nt * proj >= 2**31:" in context
    assert context.index("if nt * proj >= 2**31:") < context.index("def run_chunk():")


_GATE_NAMES = {
    "GLM5_NEXT_KDA_MODEL_PATHS",
    "_is_glm5_next_kda",
    "KIMI_K3_KDA_ARCHITECTURE",
    "GLM5_NEXT_KDA_ARCHITECTURE",
    "_KDA_ARCHITECTURE_COMPAT",
    "KdaRuntimeNotAuditedError",
    "_kda_architecture",
    "_require_audited_runtime",
}


def _gate_namespace():
    from collector.version_resolver import _check_compat

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _GATE_NAMES)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _GATE_NAMES for t in node.targets))
    ]
    namespace = {"_check_compat": _check_compat}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_kda_audited_release_sets_do_not_overlap():
    from collector.version_resolver import _check_compat

    compat = _gate_namespace()["_KDA_ARCHITECTURE_COMPAT"]
    audited = {arch: spec.split("==", 1)[1] for arch, spec in compat.items()}
    assert set(audited) == {"KimiK3ForConditionalGeneration", "Glm5NextForConditionalGeneration"}
    for arch, version in audited.items():
        for other, spec in compat.items():
            assert _check_compat(spec, version) is (other == arch), (arch, other)


@pytest.mark.parametrize(
    ("model_name", "version", "audited"),
    [
        ("moonshotai/Kimi-K3", "0.5.16", True),
        ("moonshotai/Kimi-K3", "0.5.20", False),
        ("zai-org/GLM-5.3-Flash", "0.5.20", True),
        ("zai-org/GLM-5.3-Flash", "0.5.16", False),
    ],
)
def test_kda_runtime_gate_rejects_unaudited_releases(model_name, version, audited):
    namespace = _gate_namespace()
    if audited:
        namespace["_require_audited_runtime"](model_name, version)
    else:
        with pytest.raises(namespace["KdaRuntimeNotAuditedError"], match="not an audited KDA runtime"):
            namespace["_require_audited_runtime"](model_name, version)
