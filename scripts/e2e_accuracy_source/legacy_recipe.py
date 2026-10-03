# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read pre-master workflow matrices and pre-single_node recipe locations.

Only immutable repository files are used. No shell command is executed and
external moving branches are never treated as historical source evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex

from e2e_accuracy_source.inferencex_recipe import (
    INFERENCEX_REPOSITORY_URL,
    InferenceXRecipeError,
    RecipeSource,
    _load_yaml_mapping,
    _read_first,
)
from e2e_accuracy_source.schema import SiliconRow
from e2e_accuracy_source.sources import load_manifest, manifest_text


def source_record(path: str, text: str) -> dict:
    return {"path": path, "content_sha256": hashlib.sha256(text.encode()).hexdigest()}


def workflow_inputs(workflow: dict) -> dict:
    # PyYAML 1.1 treats the unquoted GitHub Actions `on` key as boolean True.
    return workflow.get("on", workflow.get(True, {})).get("workflow_call", {}).get("inputs", {})


def load_source_config(row: SiliconRow, source: RecipeSource) -> tuple[str, str, dict]:
    """Normalize legacy workflow jobs while retaining their original content hash."""
    try:
        path, text = _read_first(
            source, row.head_sha, ("configs/nvidia-master.yaml", ".github/configs/nvidia-master.yaml")
        )
        return path, text, _load_yaml_mapping(text, "source configuration")
    except InferenceXRecipeError as error:
        if "does not exist at" not in str(error):
            raise
    prefix = {"gptoss120b": "gptoss"}.get(row.silicon_model, row.silicon_model)
    if row.disagg or prefix not in {"dsr1", "gptoss"}:
        raise InferenceXRecipeError("no reviewed legacy workflow family")
    path = f".github/workflows/{prefix}-tmpl.yml"
    text = source.read_text(row.head_sha, path)
    workflow = _load_yaml_mapping(text, "legacy model workflow")
    sweep_path = ".github/workflows/full-sweep-tmpl.yml"
    sweep_text = source.read_text(row.head_sha, sweep_path)
    sweep = _load_yaml_mapping(sweep_text, "legacy sequence workflow")
    sequences = [
        job["with"]
        for job in sweep.get("jobs", {}).values()
        if job.get("uses") == "./" + path
        and job.get("with", {}).get("isl") == row.isl
        and job.get("with", {}).get("osl") == row.osl
    ]
    if len(sequences) != 1:
        raise InferenceXRecipeError("legacy workflow sequence is missing or ambiguous")
    sequence = sequences[0]
    benchmark_path = ".github/workflows/benchmark-tmpl.yml"
    benchmark_text = source.read_text(row.head_sha, benchmark_path)
    benchmark = _load_yaml_mapping(benchmark_text, "legacy benchmark workflow")
    inputs = workflow_inputs(benchmark)
    config = {}
    for key, job in workflow.get("jobs", {}).items():
        if job.get("uses") != "./" + benchmark_path:
            continue
        values = job.get("with", {})
        if values.get("framework") != row.framework or values.get("precision") != row.precision:
            continue
        if str(values.get("runner", "")).split("-")[0] != row.hardware:
            continue
        # Every inherited scalar has to be the reviewed direct workflow input.
        for name in ("isl", "osl", "max-model-len", "random-range-ratio"):
            if values.get(name) != "${{ inputs." + name + " }}":
                raise InferenceXRecipeError(f"unreviewed legacy workflow override: {name}")
        try:
            tps = json.loads(values["tp-list"])
            concs = json.loads(values.get("conc-list", inputs.get("conc-list", {}).get("default", "null")))
        except (ValueError, TypeError, KeyError) as error:
            raise InferenceXRecipeError("unresolved legacy TP/concurrency matrix") from error
        if not isinstance(tps, list) or not isinstance(concs, list):
            raise InferenceXRecipeError("unresolved legacy TP/concurrency matrix")
        config[key] = {
            **values,
            "model-prefix": prefix,
            "seq-len-configs": [
                {"isl": row.isl, "osl": row.osl, "search-space": [{"tp": tp, "conc-list": concs} for tp in tps]}
            ],
            "_legacy_max_model_len": sequence.get("max-model-len"),
            "_legacy_random_range_ratio": sequence.get("random-range-ratio"),
            "_legacy_sources": [
                source_record(path, text),
                source_record(sweep_path, sweep_text),
                source_record(benchmark_path, benchmark_text),
            ],
        }
    return path, text, config


def legacy_shell_source(row: SiliconRow, source: RecipeSource) -> tuple[str, str, str, str]:
    """Return server and benchmark source for the old Docker/Slurm split."""
    prefix = {"gptoss120b": "gptoss"}.get(row.silicon_model, row.silicon_model)
    if prefix not in {"dsr1", "gptoss"} or row.disagg:
        raise InferenceXRecipeError("no reviewed legacy shell family")
    suffix = "_trt" if row.framework == "trt" else ""
    base = f"benchmarks/{prefix}_{row.precision}_{row.hardware}{suffix}"
    paths = (base + "_slurm.sh",) if row.framework == "trt" or row.hardware == "h200" else (base + "_docker.sh",)
    path, text = _read_first(source, row.head_sha, paths)
    if "benchmark_serving.py" in text:
        return path, text, path, text
    if 'source "$(dirname "$0")/benchmark_lib.sh"' in text and "run_benchmark_serving" in text:
        library_path = "benchmarks/benchmark_lib.sh"
        library = source.read_text(row.head_sha, library_path)
        # Verify the helper forwards workload flags unchanged to its client.
        for flag, name in (
            ("random-range-ratio", "random_range_ratio"),
            ("num-prompts", "num_prompts"),
            ("input-len", "input_len"),
            ("output-len", "output_len"),
        ):
            if not re.search(re.escape("--" + flag + ")") + r"\s+" + name + r"=\"\$2\"", library):
                raise InferenceXRecipeError("unreviewed legacy benchmark helper argument")
            client_flag = "random-" + flag if flag in {"input-len", "output-len"} else flag
            if f'--{client_flag} "${name}"' not in library:
                raise InferenceXRecipeError("unreviewed legacy benchmark helper forwarding")
        return path, text, path, text
    runner_path = {"b200": "runners/launch_b200-tg.sh", "h100": "runners/launch_h100-cr.sh"}.get(row.hardware)
    if runner_path is None:
        raise InferenceXRecipeError("legacy benchmark runner is unresolved")
    runner = source.read_text(row.head_sha, runner_path)
    # The Docker runner must actually dispatch this model/precision family.
    if "_${PRECISION}_" + row.hardware not in runner or "_docker.sh" not in runner:
        raise InferenceXRecipeError("legacy runner does not establish server dispatch")
    return path, text, runner_path, runner


def legacy_workload(row: SiliconRow, source: RecipeSource, text: str, context: dict) -> dict:
    """Resolve legacy workload from explicit workflow values or declared inputs."""
    workflow_path = ".github/workflows/benchmark-tmpl.yml"
    workflow_text = source.read_text(row.head_sha, workflow_path)
    workflow = _load_yaml_mapping(workflow_text, "legacy benchmark workflow")
    ratio = context.get("_legacy_random_range_ratio")
    if ratio is None:
        # Only use the input default when the shared caller omits that argument.
        caller_path = ".github/workflows/full-sweep-tmpl.yml"
        caller_path, caller_text = _read_first(
            source, row.head_sha, (caller_path, ".github/workflows/full-sweep-test.yml")
        )
        caller = _load_yaml_mapping(caller_text, "legacy benchmark caller")
        calls = [job for job in caller.get("jobs", {}).values() if job.get("uses") == "./" + workflow_path]
        if not calls or any("random-range-ratio" in job.get("with", {}) for job in calls):
            raise InferenceXRecipeError("legacy benchmark distribution caller is unresolved")
        ratio = workflow_inputs(workflow).get("random-range-ratio", {}).get("default")
    try:
        ratio = float(ratio)
    except (ValueError, TypeError) as error:
        raise InferenceXRecipeError("legacy benchmark distribution is unresolved") from error
    if workflow.get("env", {}).get("RANDOM_RANGE_RATIO") != "${{ inputs.random-range-ratio }}":
        raise InferenceXRecipeError("legacy benchmark ratio is not passed to the runner")
    ratio_flag = r'--random-range-ratio(?:=|\s+)["\']?\$RANDOM_RANGE_RATIO\b'
    counts = set(re.findall(r'--num-prompts(?:=|\s+)["\']?\$\(\(\s*\$?CONC\s*\*\s*(\d+)\s*\)\)', text))
    if not re.search(ratio_flag, text) or len(counts) != 1:
        raise InferenceXRecipeError("legacy benchmark distribution or request count is unresolved")
    # Historical scripts clone this driver without a revision. Keep the gap
    # explicit instead of claiming the client implementation was pinned.
    return {
        "type": "benchmark_serving.py",
        "random_range_ratio": ratio,
        "num_prompts_mult": int(next(iter(counts))),
        "source": f"{row.head_sha}:{workflow_path}",
        "client_revision_verified": False,
    }


def launcher_recipe_copies(text: str, destination: str) -> list[dict]:
    """Locate literal cp -rT mappings; callers must verify the executed branch.

    These are candidates, not permission to use an inactive launch branch.
    Exact run logs can establish which candidate was actually copied.
    """
    if not destination.startswith("recipes/") or ".." in destination.split("/"):
        raise InferenceXRecipeError("invalid external recipe destination")
    copies = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line.startswith("cp -rT "):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) < 4:
            continue
        src, dst = tokens[2:4]
        src = src.removeprefix("${GITHUB_WORKSPACE}/").removeprefix("$GITHUB_WORKSPACE/")
        if not src.startswith("benchmarks/multi_node/srt-slurm-recipes/") or not dst.startswith("recipes/"):
            continue
        src, dst = src.rstrip("/"), dst.rstrip("/")
        if any(char in src + dst for char in "$`*") or ".." in (src + "/" + dst).split("/"):
            continue
        if destination.startswith(dst + "/"):
            candidate = {
                "path": src + destination[len(dst) :],
                "copy_source": src,
                "copy_destination": dst,
                "requires_execution_evidence": True,
            }
            if candidate not in copies:
                copies.append(candidate)
    return copies


def verified_launcher_recipe_copy(row: SiliconRow, destination: str, source: RecipeSource) -> dict | None:
    """Resolve a reviewed copy only with exact-attempt runner and source evidence."""
    manifest = load_manifest("launcher_recipe_sources.json")
    if not (
        row.disagg
        and row.is_multinode
        and row.benchmark_type == "single_turn"
        and row.framework == "dynamo-vllm"
        and row.silicon_model == "minimaxm2.5"
        and row.hardware == "b300"
        and row.precision == "fp4"
        and row.spec_method == "none"
        and destination.startswith(manifest["destination_directory"] + "/")
    ):
        return None
    matches = [
        entry
        for entry in manifest["points"]
        if all(getattr(row, key) == value for key, value in entry["point"].items())
        and str(entry["job"]["run_id"]) == str(row.github_run_id)
        and entry["job"]["run_attempt"] == row.run_attempt
        and entry["job"]["head_sha"] == row.head_sha
    ]
    if len(matches) != 1:
        return None
    job = matches[0]["job"]
    if job["conclusion"] != "success" or job["runner_name"].split("_")[0] != "b300-nv":
        raise InferenceXRecipeError("launcher copy lacks successful exact-attempt runner evidence")
    topology = " ".join(
        f"{label}(tp{getattr(row, role + '_tp')}/ep{getattr(row, role + '_ep')}"
        f"/dp{str(getattr(row, role + '_dp_attention')).lower()}/nw{getattr(row, role + '_num_workers')})"
        for role, label in (("prefill", "P"), ("decode", "D"))
    )
    suffix = (
        f"minimaxm2.5_{row.isl // 1024}k{row.osl // 1024}k fp4 b300 dynamo-vllm | "
        f"{topology} | disagg-true spec-none conc-"
    )
    tail = job["name"].rsplit(" / ", 1)[-1]
    if (
        row.isl % 1024
        or row.osl % 1024
        or not tail.startswith(suffix)
        or str(row.conc) not in tail.removeprefix(suffix).split("x")
    ):
        raise InferenceXRecipeError("launcher copy job configuration differs from measurement")
    texts = {}
    for kind in ("launcher", "workflow"):
        text = source.read_text(row.head_sha, manifest[kind + "_path"])
        if hashlib.sha256(text.encode()).hexdigest() != manifest[kind + "_sha256"]:
            raise InferenceXRecipeError("reviewed launcher copy source checksum mismatch")
        texts[kind] = text
    candidates = launcher_recipe_copies(texts["launcher"], destination)
    if len(candidates) != 1 or candidates[0]["copy_source"] != manifest["source_directory"]:
        raise InferenceXRecipeError("reviewed launcher copy is missing or ambiguous")
    return candidates[0] | {
        "requires_execution_evidence": False,
        "evidence_type": "reviewed launcher branch and exact-attempt job metadata",
        "job": job,
        "job_url": f"{INFERENCEX_REPOSITORY_URL}/actions/runs/{job['run_id']}/job/{job['id']}",
        "jobs_endpoint": manifest["jobs_endpoint"],
        "sources": [source_record(manifest[kind + "_path"], texts[kind]) for kind in ("launcher", "workflow")],
    }


def verified_launcher_workload_identity(
    row: SiliconRow, destination: str, *, server_source_sha: str | None = None
) -> dict | None:
    """Use reviewed executed checkouts only for their exact measured points.

    Hardware alone does not identify a launcher. These records retain the
    actual job command, source checkout, upload hashes, and four matching
    latency statistics, including corrected versus recorded attempt numbers.
    """
    manifest_name = "launcher_workload_sources.json"
    records = load_manifest(manifest_name)["records"]
    matches = [
        record
        for record in records
        if record["recipe_destination"] == destination
        and all(getattr(row, key, None) == value for key, value in record["point"].items())
        and str(row.github_run_id) == record["job"]["github_run_id"]
        and row.run_attempt == record["job"]["recorded_run_attempt"]
        and all(row.metrics.get(key) == value for key, value in record["metrics"].items())
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise InferenceXRecipeError("ambiguous reviewed workload checkout evidence")
    record = matches[0]
    if (server_source_sha or row.head_sha) != record["job"]["source_git_sha"]:
        raise InferenceXRecipeError("server recipe source differs from verified workload execution source")
    return {
        "repository": "https://github.com/NVIDIA/srt-slurm",
        "git_sha": record["runner_git_sha"],
        "evidence_type": "exact job checkout with matching benchmark measurements",
        "recipe_destination": destination,
        "job": record["job"],
        "artifact": record["artifact"],
        "reviewed_manifest": source_record(manifest_name, manifest_text(manifest_name)),
    }
