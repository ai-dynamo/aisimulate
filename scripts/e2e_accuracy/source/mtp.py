# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MTP inputs from benchmark-linked InferenceX logs; never infer acceptance from latency."""

import ast
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import requests

from .recipes.inferencex_recipe import _normalize_yaml_server_args

API = "https://inferencex.semianalysis.com/api/v1/"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_api(route, params, cache: Path):
    url = API + route + "?" + urlencode(params)
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text()), url
    with requests.get(url, timeout=45, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > 64 * 1024 * 1024:
                raise ValueError("server log exceeds 64 MiB")
    value = json.loads(content)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as stream:
        json.dump(value, stream)
        temporary = Path(stream.name)
    temporary.replace(path)
    return value, url


def log_files(point, cache):
    """The public API joins benchmark_result_id to its stored log bundle."""
    bid = point["benchmark"]["id"]
    names, _ = read_api("server-log-files", {"id": bid}, cache)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("invalid server log file list")
    if not point["config"]["disagg"] and len(names) != 1:
        raise ValueError("ambiguous aggregated server log bundle")
    roles = {"prefill": [], "decode": []} if point["config"]["disagg"] else {"aggregated": names}
    if point["config"]["disagg"]:
        for role in roles:
            roles[role] = [n for n in names if re.search(rf"_{role}_w\d+\.out$", n)]
    result = {}
    for role, files in roles.items():
        if not files:
            raise ValueError(f"no {role} runtime logs")
        result[role] = []
        for name in files:
            value, url = read_api("server-log", {"id": bid, "file": name}, cache)
            if value.get("id") != bid or not isinstance(value.get("serverLog"), str):
                raise ValueError("server log benchmark identity mismatch")
            text = value["serverLog"]
            result[role].append((text, {"url": url, "file": name, "sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return result


def validate_mtp(value):
    depth, accepted = value["depth"], value["accepted_draft_tokens"]
    if type(depth) is not int or not 1 <= depth <= 5:
        raise ValueError("unsupported MTP depth: replay supports 1..5")
    if type(accepted) not in (int, float) or not math.isfinite(accepted) or not 0 <= accepted <= depth:
        raise ValueError("accepted draft tokens must be finite and within [0, depth]")
    if value.get("kind") not in {"observed_counters", "observed_log_samples", "configured_synthetic"}:
        raise ValueError("acceptance provenance is required")
    return value


def parse_runtime(text, backend):
    """Read literal configuration only. Logs are data, never executable Python."""
    text = ANSI.sub("", text)
    args = {}
    for line in text.splitlines():
        if "server_args=ServerArgs(" in line:
            expr = ast.parse(line.split("server_args=", 1)[1], mode="eval").body
            if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "ServerArgs":
                raise ValueError("invalid ServerArgs record")
            args.update({kw.arg: kw.value.value for kw in expr.keywords if isinstance(kw.value, ast.Constant)})
        elif "non-default args:" in line:
            raw = line.split("non-default args:", 1)[1].strip()
            expression = ast.parse(re.sub(r"<[A-Za-z_][^>\n]*>", "None", raw), mode="eval").body
            if not isinstance(expression, ast.Dict):
                raise ValueError("invalid vLLM arguments")
            for key, value in zip(expression.keys, expression.values, strict=True):
                try:
                    args[ast.literal_eval(key)] = ast.literal_eval(value)
                except ValueError:
                    continue  # Runtime object representations are retained in the source log.
        elif backend == "trtllm" and "speculative_config=MTPDecodingConfig(" in line:
            raw = re.sub(r"<[A-Za-z_][^>\n]*>", "None", line)
            raw = re.sub(r"(?<!,) (?=[a-z_][a-z_0-9]*=)", ", ", raw)
            expression = ast.parse("Runtime(" + raw + ")", mode="eval").body
            for kw in expression.keywords:
                if isinstance(kw.value, ast.Constant):
                    args[kw.arg] = kw.value.value
                elif kw.arg in {"kv_cache_config", "speculative_config"} and isinstance(kw.value, ast.Call):
                    args[kw.arg] = {
                        v.arg: v.value.value for v in kw.value.keywords if isinstance(v.value, ast.Constant)
                    }
        if "Initializing a V1 LLM engine (v" in line:
            version = re.search(r"Initializing a V1 LLM engine \(v([^)]*)\)", line)
            if version:
                args["runtime_framework_version"] = version[1]
        summary = re.search(
            r"max_total_num_tokens=\d+, chunked_prefill_size=(\d+), max_prefill_tokens=(\d+), "
            r"max_running_requests=(\d+), context_len=(\d+)",
            line,
        )
        if summary:
            args.update(
                zip(
                    ("chunked_prefill_size", "max_prefill_tokens", "max_running_requests", "context_length"),
                    map(int, summary.groups()),
                    strict=True,
                )
            )
    if not args:
        raise ValueError("runtime server configuration is absent")
    if isinstance(args.get("speculative_config"), str):
        args["speculative_config"] = json.loads(args["speculative_config"])
    if backend == "sglang" and "chunked_prefill_size" in args:
        args["enable_chunked_prefill"] = args["chunked_prefill_size"] > 0
        args["enable_prefix_caching"] = not args.get("disable_radix_cache", False)
    return _normalize_yaml_server_args({k: v for k, v in args.items() if v is not None})


def mtp_depth(args, backend):
    spec = args.get("speculative_config") or {}
    if backend == "sglang":
        if args.get("speculative_draft_model_path") not in (None, args.get("model_path")):
            raise ValueError("external draft checkpoint requires a separate draft-model evaluation")
        if str(args.get("speculative_algorithm", "")).upper() not in {"EAGLE", "NEXTN", "MTP"}:
            raise ValueError("unsupported speculation scheme (not native MTP)")
        if args.get("speculative_eagle_topk") != 1:
            raise ValueError("tree speculation is not the linear MTP model")
        depth = args.get("speculative_num_steps")
        if type(depth) is not int or args.get("speculative_num_draft_tokens") != depth + 1:
            raise ValueError("SGLang draft width differs from linear MTP depth + 1")
    else:
        if spec.get("model") not in (None, args.get("model_path", args.get("model"))):
            raise ValueError("external draft checkpoint requires a separate draft-model evaluation")
        method = str(spec.get("method", spec.get("decoding_type", ""))).lower()
        if method not in {"mtp", "deepseek_mtp", "qwen3_next_mtp", "qwen3_5_mtp"}:
            raise ValueError(f"unsupported speculation scheme: {method or 'missing'}")
        depth = spec.get("num_speculative_tokens", spec.get("max_draft_len"))
    if type(depth) is not int or not 1 <= depth <= 5:
        raise ValueError("unsupported or missing MTP depth")
    return depth


def acceptance(args, text, backend):
    text = ANSI.sub("", text)
    depth = mtp_depth(args, backend)
    spec = args.get("speculative_config") or {}
    if spec.get("rejection_sample_method") == "synthetic":
        length = spec.get("synthetic_acceptance_length")
        rates = spec.get("synthetic_acceptance_rates")
        if length is not None:
            value = float(length) - 1
        elif rates is not None:
            if len(rates) != depth or any(not 0 <= r <= 1 for r in rates) or rates != sorted(rates, reverse=True):
                raise ValueError("invalid synthetic unconditional acceptance rates")
            value = sum(rates)
        else:
            raise ValueError("missing configured synthetic acceptance")
        return validate_mtp(dict(depth=depth, accepted_draft_tokens=value, kind="configured_synthetic"))
    # vLLM counters are per logging interval, not cumulative. Sum counts, never percentages.
    counts = re.findall(r"Accepted:\s*(\d+) tokens, Drafted:\s*(\d+) tokens", text)
    if counts and backend == "vllm":
        accepted = sum(int(a) for a, _ in counts)
        drafted = sum(int(d) for _, d in counts)
        if drafted:
            return validate_mtp(
                dict(
                    depth=depth,
                    accepted_draft_tokens=depth * accepted / drafted,
                    kind="observed_counters",
                    accepted=accepted,
                    drafted=drafted,
                )
            )
    lengths = [float(v) for v in re.findall(r"accept len:\s*([\d.]+)", text)]
    if lengths and backend == "sglang":
        # Rounded interval means lack their verification counts. Label this reconstruction explicitly.
        return validate_mtp(
            dict(
                depth=depth,
                accepted_draft_tokens=sum(lengths) / len(lengths) - 1,
                kind="observed_log_samples",
                samples=len(lengths),
                min_length=min(lengths),
                max_length=max(lengths),
                limitation="Unweighted mean of rounded logged interval acceptance lengths; warmup may be included.",
            )
        )
    raise ValueError("missing acceptance: real rejection sampling has no fixed workload acceptance default")


def runtime_inputs(point, parsed, logs):
    model, roles, version, backend, benchmark, evidence = parsed
    roles = {name: dict(args) for name, args in roles.items()}
    mtp = {}
    evidence = dict(evidence)
    evidence["mtp_runtime_logs"] = {}
    for role, files in logs.items():
        controls = [parse_runtime(text, backend) for text, _ in files]
        # Per-worker rank addresses and ports differ; modeled controls must agree.
        relevant = (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "block_size",
            "kv_cache_dtype",
            "speculative_config",
            "speculative_num_steps",
            "speculative_algorithm",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "speculative_draft_model_path",
            "model_path",
            "model",
            "dtype",
            "quantization",
            "mem_fraction_static",
            "gpu_memory_utilization",
            "kv_cache_config",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "enable_dp_attention",
            "expert_parallel_size",
            "attention_data_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "enable_expert_parallel",
            "max_model_len",
            "prefill_schedule_interval",
            "prefill_decode_interval",
            "revision",
        )
        if any(any(c.get(k) != controls[0].get(k) for k in relevant) for c in controls[1:]):
            raise ValueError(f"{role} workers disagree on modeled controls")
        roles[role] = roles[role] | controls[0]
        evidence.setdefault("mtp_runtime_controls", {})[role] = sorted(controls[0])
        runtime_model = roles[role].get("model_path", roles[role].get("model"))
        if (
            isinstance(runtime_model, str)
            and "/" in runtime_model
            and not runtime_model.startswith("/")
            and runtime_model != model
        ):
            raise ValueError(f"runtime model {runtime_model} differs from source model {model}")
        evidence["mtp_runtime_logs"][role] = [proof for _, proof in files]
        if role == "prefill":
            if not roles[role].get("speculative_algorithm") and not roles[role].get("speculative_config"):
                roles[role] = {k: v for k, v in roles[role].items() if not k.startswith("speculative_")}
        else:
            mtp[role] = acceptance(roles[role], "\n".join(text for text, _ in files), backend)
    prefill = roles.get("prefill", {})
    if prefill.get("speculative_algorithm") or prefill.get("speculative_config"):
        if mtp_depth(prefill, backend) != mtp["decode"]["depth"]:
            raise ValueError("prefill and decode MTP depths differ")
        # P/D serves the same requests. Decode owns acceptance; prefill still needs its MTP timing identity.
        mtp["prefill"] = {**mtp["decode"], "acceptance_source_role": "decode"}
    return (model, roles, version, backend, benchmark, evidence), mtp
