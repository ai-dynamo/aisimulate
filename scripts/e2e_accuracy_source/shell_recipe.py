# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a restricted serving-command grammar without executing shell code."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from e2e_accuracy_source.inferencex_recipe import (
    InferenceXRecipeError,
    RecipeSource,
    _load_yaml_mapping,
    _normalize_yaml_server_args,
    _read_first,
)
from e2e_accuracy_source.legacy_recipe import legacy_shell_source, legacy_workload, source_record
from e2e_accuracy_source.schema import SiliconRow
from e2e_accuracy_source.shell_values import resolve_lines


def command_args(command: str, variables: dict[str, Any]) -> dict[str, Any]:
    def substitute(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        if name not in variables:
            raise InferenceXRecipeError(f"unresolved shell variable {name}")
        return str(variables[name])

    command = re.sub(r"\$\{([A-Z_]+)\}|\$([A-Z_]+)", substitute, command)
    if "$" in command or "`" in command or ";" in command:
        raise InferenceXRecipeError("unsupported shell expression in serving arguments")
    tokens = shlex.split(command)
    args: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in (">", ">>", "2>&1", "&") or token.startswith(">"):
            break
        if token.startswith("--"):
            key, sep, value = token[2:].partition("=")
            if not sep:
                value = True
                if i + 1 < len(tokens) and not tokens[i + 1].startswith(("--", ">", "&", "2>")):
                    i += 1
                    value = tokens[i]
            if isinstance(value, str):
                if re.fullmatch(r"-?\d+", value):
                    value = int(value)
                elif re.fullmatch(r"\d+\.\d+", value):
                    value = float(value)
                elif value in ("true", "false"):
                    value = value == "true"
            if isinstance(value, str) and value.startswith(("{", "[")):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            name = key.replace("-", "_")
            if name in args and args[name] != value:
                raise InferenceXRecipeError(f"conflicting serving flag {key}")
            args[name] = value
        i += 1
    return args


def read_shell_recipe(
    row: SiliconRow, source: RecipeSource, model: str, *, legacy_context: dict | None = None
) -> tuple[str, str, dict, dict]:
    return _read_static_shell_recipe(row, source, model, legacy_context=legacy_context or {})


def read_workload(row: SiliconRow, source: RecipeSource, text: str) -> dict:
    workflow_path, workflow_text = _read_first(source, row.head_sha, (".github/workflows/benchmark-tmpl.yml",))
    workflow = _load_yaml_mapping(workflow_text, "benchmark workflow")
    ratio = workflow.get("env", {}).get("RANDOM_RANGE_RATIO")
    if not isinstance(ratio, (int, float)) or not re.search(r'''--random-range-ratio\s+"\$RANDOM_RANGE_RATIO"''', text):
        raise InferenceXRecipeError("benchmark distribution is not established")
    count = re.search(r'--num-prompts\s+["\']?\$\(\(\s*\$?CONC\s*\*\s*(\d+)\s*\)\)', text)
    if count is None:
        raise InferenceXRecipeError("benchmark request count is not established")
    benchmark = {
        "type": "benchmark_serving.py",
        "random_range_ratio": ratio,
        "num_prompts_mult": int(count.group(1)),
        "source": f"{row.head_sha}:{workflow_path}",
    }
    return benchmark


def _read_static_shell_recipe(
    row: SiliconRow, source: RecipeSource, model: str, *, legacy_context: dict
) -> tuple[str, str, dict, dict]:
    backend = row.framework.removeprefix("dynamo-")
    stems = [
        f"{row.silicon_model}_{row.precision}_{row.hardware}_{backend}",
        f"{row.silicon_model}_{row.precision}_{row.hardware}",
    ]
    paths = tuple(f"benchmarks/single_node/{folder}{stem}.sh" for folder in ("fixed_seq_len/", "") for stem in stems)
    legacy = False
    try:
        path, text = _read_first(source, row.head_sha, paths)
    except InferenceXRecipeError as error:
        if "does not exist at" not in str(error):
            raise
        path, text, benchmark_path, benchmark_text = legacy_shell_source(row, source)
        legacy = True
    variables = dict(
        MODEL=model,
        MODEL_PATH=model,
        TP=row.decode_tp,
        EP_SIZE=row.decode_ep,
        DP_ATTENTION=str(row.decode_dp_attention).lower(),
        CONC=row.conc,
        ISL=row.isl,
        OSL=row.osl,
        PORT=8000,
        PORT_OFFSET=0,
        EVAL_ONLY="false",
        RUN_EVAL="false",
    )
    if legacy_context.get("_legacy_max_model_len") is not None:
        variables["MAX_MODEL_LEN"] = legacy_context["_legacy_max_model_len"]
    else:
        _, generator = _read_first(
            source,
            row.head_sha,
            (
                "utils/matrix_logic/generate_sweep_configs.py",
                "utils/matrix-logic/generate_sweep_configs.py",
                "utils/generate_sweep_configs.py",
            ),
        )
        padding = set(
            re.findall(
                r"(?:Fields.MAX_MODEL_LEN.value|FIELD_MAX_MODEL_LEN|['\"]max-model-len['\"])\s*:\s*isl\s*\+\s*osl\s*\+\s*(\d+)",
                generator,
            )
        )
        if len(padding) == 1:
            variables["MAX_MODEL_LEN"] = row.isl + row.osl + int(next(iter(padding)))
    commands, values, files = resolve_lines(text, variables)
    if len(commands) != 1:
        raise InferenceXRecipeError(
            f"expected one statically resolved serving command in {path}; found {len(commands)}"
        )
    command = commands[0]
    marker = {
        "vllm": r"vllm serve|python3? -m vllm.entrypoints.openai.api_server",
        "sglang": r"python3? -m sglang.launch_server|sglang serve",
        "trt": r"trtllm-serve",
    }[backend]
    match = re.search(marker, command)
    if not match:
        raise InferenceXRecipeError("serving executable does not match recorded framework")
    args = command_args(command[match.end() :], {})
    if backend in {"trt", "vllm"}:
        filename = args.pop("config", None) or args.pop("extra_llm_api_options", None)
        if filename:
            if filename not in files:
                raise InferenceXRecipeError(f"unresolved serving config file: {filename}")
            config_args = _load_yaml_mapping(files[filename], "generated serving config")
            normalized_config = _normalize_yaml_server_args(config_args)
            normalized_cli = _normalize_yaml_server_args(args)
            conflicts = {
                key
                for key in normalized_config.keys() & normalized_cli.keys()
                if normalized_config[key] != normalized_cli[key]
            }
            if conflicts:
                raise InferenceXRecipeError(
                    "serving CLI/config precedence needs version evidence: " + ", ".join(sorted(conflicts))
                )
            args = config_args | args
    if values.get("SGLANG_RADIX_FORCE_MISS") == "1" or "SGLANG_RADIX_FORCE_MISS=1" in command[: match.start()]:
        args["enable_prefix_caching"] = False
    args["recipe_environment"] = {
        k: v
        for k, v in values.items()
        if k.startswith(("VLLM_", "SGLANG_", "SGL_", "TRTLLM_")) or k == "OVERRIDE_QUANT_ALGO"
    }
    for token in shlex.split(command[: match.start()]):
        name, separator, value = token.partition("=")
        if separator and name == "OVERRIDE_QUANT_ALGO":
            args["recipe_environment"][name] = value
    if "OVERRIDE_QUANT_ALGO" in text and "OVERRIDE_QUANT_ALGO" not in args["recipe_environment"]:
        raise InferenceXRecipeError("unresolved OVERRIDE_QUANT_ALGO in serving recipe")
    if legacy:
        benchmark = legacy_workload(row, source, benchmark_text, legacy_context)
        benchmark["source_files"] = [source_record(benchmark_path, benchmark_text)]
        if "run_benchmark_serving" in benchmark_text:
            helper_path = "benchmarks/benchmark_lib.sh"
            benchmark["source_files"].append(source_record(helper_path, source.read_text(row.head_sha, helper_path)))
    else:
        benchmark = read_workload(row, source, text)
    return path, text, args, benchmark
