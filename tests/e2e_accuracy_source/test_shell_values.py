# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for reviewed InferenceX shell constructs; never run shell."""

import pytest
from e2e_accuracy_source.recipes.inferencex_recipe import InferenceXRecipeError
from e2e_accuracy_source.recipes.shell_values import expand, resolve_lines


def test_conditional_scheduler_interval():
    expression = "$( [[ $CONC -gt 4 ]] && echo 30 || echo 10 )"
    assert expand(expression, {"CONC": 8}) == "30"
    assert expand(expression, {"CONC": 4}) == "10"
    with pytest.raises(InferenceXRecipeError):
        expand(expression, {})


def test_unknown_condition_cannot_select_serving_knobs():
    recipe = """
ARGS="--tensor-parallel-size 2"
if [[ $UNKNOWN == yes ]]; then
ARGS="--tensor-parallel-size 8"
fi
vllm serve model $ARGS
"""
    with pytest.raises(InferenceXRecipeError, match="ARGS"):
        resolve_lines(recipe, {})


def test_array_case_and_generated_config():
    recipe = """
case "$TP" in
  2) TOKENS=8192 ;;
  4) TOKENS=16384 ;;
esac
cat > config.yaml <<EOF
max_num_tokens: $TOKENS
kv_cache_config:
  tokens_per_block: 32
EOF
SERVE_CMD=(
  trtllm-serve model
  --config config.yaml
)
"${SERVE_CMD[@]}" > "$SERVER_LOG" 2>&1
"""
    commands, _, files = resolve_lines(recipe, {"TP": 4})
    assert commands == ["trtllm-serve model --config config.yaml"]
    assert "max_num_tokens: 16384" in files["config.yaml"]
    assert "tokens_per_block: 32" in files["config.yaml"]


def test_arbitrary_command_substitution_is_not_executed(tmp_path):
    marker = tmp_path / "must-not-exist"
    with pytest.raises(InferenceXRecipeError):
        expand(f"$(touch {marker})", {})
    assert not marker.exists()


def test_arithmetic_comparison_is_not_output_redirection():
    commands, _, _ = resolve_lines(
        'sglang serve model --max-running-requests "$(( CONC * 3 / 2 > 8 ? CONC * 3 / 2 : 8 ))" >> $SERVER_LOG 2>&1 &',
        {"CONC": 32},
    )
    assert commands == ['sglang serve model --max-running-requests "48"']


def test_lowercase_capture_array_keeps_all_sizes():
    _, values, _ = resolve_lines(
        """
capture_tokens=(1 2 4 8)
capture_tokens+=( $(seq 16 16 $MAX_NUM_TOKENS))
CAPTURE_TOKENS_LIST=$(printf "%s, " "${capture_tokens[@]}")
""",
        {"MAX_NUM_TOKENS": 48},
    )
    assert values["CAPTURE_TOKENS_LIST"] == "1, 2, 4, 8, 16, 32, 48, "


def test_empty_variable_uses_shell_default():
    assert expand("${X:-32}", {"X": ""}) == "32"


def test_nested_serving_options_survive_normalization():
    from e2e_accuracy_source.recipes.inferencex_recipe import _normalize_yaml_server_args

    raw = {
        "kv_cache_config": {"tokens_per_block": 32, "enable_block_reuse": False},
        "cuda_graph_config": {"batch_sizes": [1, 2, 4, 8]},
        "max_seq_len": 9216,
    }
    normalized = _normalize_yaml_server_args(raw)
    assert normalized["kv_cache_config"] == raw["kv_cache_config"]
    assert normalized["cuda_graph_config"] == raw["cuda_graph_config"]
    assert normalized["max_model_len"] == 9216
