# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read immutable InferenceX sources and validate recipe search-point selection.

Shell interpretation and deployment resolution live in their dedicated modules.
Source commands are never executed.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

from scripts.e2e_accuracy.source.schema import SiliconRow

INFERENCEX_REPOSITORY = "SemiAnalysisAI/InferenceX"
INFERENCEX_REPOSITORY_URL = f"https://github.com/{INFERENCEX_REPOSITORY}"
_FULL_GIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


class InferenceXRecipeError(ValueError):
    """A source recipe cannot be resolved exactly or safely interpreted."""


class RecipeSource(Protocol):
    def read_text(self, git_sha: str, path: str) -> str: ...


class GitHubRecipeSource:
    """Read immutable public InferenceX files from raw.githubusercontent.com."""

    repository_url = INFERENCEX_REPOSITORY_URL

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: Path | None = None,
        archived_runtime: bool = False,
    ) -> None:
        self._session = session or requests.Session()
        self._cache_dir = cache_dir
        self.archived_runtime = archived_runtime
        self._cache: dict[tuple[str, str], str] = {}
        self._missing: set[tuple[str, str]] = set()
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def read_text(self, git_sha: str, path: str) -> str:
        with self._locks_guard:
            lock = self._locks.setdefault((git_sha, path), threading.Lock())
        with lock:
            return self._read_text(git_sha, path)

    def _read_text(self, git_sha: str, path: str) -> str:
        if not _FULL_GIT_SHA.fullmatch(git_sha) or path.startswith("/") or ".." in path.split("/"):
            raise InferenceXRecipeError("recipe source requires an immutable SHA and relative path")
        disk_path = self._cache_dir / git_sha / path if self._cache_dir else None
        if disk_path is not None and disk_path.is_file():
            return disk_path.read_text()
        key = (git_sha, path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key in self._missing:
            raise InferenceXRecipeError(f"InferenceX recipe does not exist at {git_sha}:{path}")
        url = f"https://raw.githubusercontent.com/{INFERENCEX_REPOSITORY}/{git_sha}/{path}"
        try:
            response = self._session.get(url, timeout=60)
            if response.status_code == 404:
                self._missing.add(key)
                raise InferenceXRecipeError(f"InferenceX recipe does not exist at {git_sha}:{path}")
            response.raise_for_status()
        except requests.RequestException as error:
            raise InferenceXRecipeError(f"could not fetch InferenceX recipe at {git_sha}:{path}: {error}") from error
        self._cache[key] = response.text
        if disk_path is not None:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            # Same-directory replace prevents a reader from seeing partial content.
            descriptor, temporary = tempfile.mkstemp(dir=disk_path.parent, prefix=".recipe-")
            try:
                with os.fdopen(descriptor, "w") as output:
                    output.write(response.text)
                os.replace(temporary, disk_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return response.text


def _read_first(source: RecipeSource, git_sha: str, paths: tuple[str, ...]) -> tuple[str, str]:
    errors: list[str] = []
    for path in paths:
        try:
            return path, source.read_text(git_sha, path)
        except InferenceXRecipeError as error:
            errors.append(str(error))
    raise InferenceXRecipeError("could not find reviewed InferenceX recipe: " + "; ".join(errors))


def _select_disaggregated_search_point(row: SiliconRow, family: dict[str, Any]) -> dict[str, Any]:
    scenarios = family.get("scenarios")
    sequence_configs = scenarios.get("fixed-seq-len") if isinstance(scenarios, dict) else None
    if sequence_configs is None:
        sequence_configs = family.get("seq-len-configs")
    if not isinstance(sequence_configs, list):
        raise InferenceXRecipeError("source config has no fixed-sequence recipe search space")

    matches: list[dict[str, Any]] = []
    for sequence_config in sequence_configs:
        if not isinstance(sequence_config, dict):
            continue
        if _as_int(sequence_config.get("isl")) != row.isl or _as_int(sequence_config.get("osl")) != row.osl:
            continue
        search_space = sequence_config.get("search-space")
        if not isinstance(search_space, list):
            continue
        for point in search_space:
            if not isinstance(point, dict) or row.conc not in _int_list(point.get("conc-list")):
                continue
            if point.get("spec-decoding", "none") != row.spec_method:
                continue
            if _role_topology_matches(row, point.get("prefill"), "prefill") and _role_topology_matches(
                row, point.get("decode"), "decode"
            ):
                matches.append(point)
    if len(matches) != 1:
        raise InferenceXRecipeError(
            "expected one exact source-config recipe point for "
            f"isl={row.isl} osl={row.osl} conc={row.conc}; found {len(matches)}"
        )
    return matches[0]


def _role_topology_matches(row: SiliconRow, role: Any, prefix: str) -> bool:
    if not isinstance(role, dict):
        return False
    expected = {
        "num-worker": getattr(row, f"{prefix}_num_workers"),
        "tp": getattr(row, f"{prefix}_tp"),
        "ep": getattr(row, f"{prefix}_ep"),
        "dp-attn": getattr(row, f"{prefix}_dp_attention"),
    }
    return all(role.get(key) == value for key, value in expected.items())


def _single_config_file(search_point: dict[str, Any]) -> str:
    settings: list[Any] = []
    for role_name in ("prefill", "decode"):
        role = search_point.get(role_name)
        if isinstance(role, dict) and isinstance(role.get("additional-settings"), list):
            settings.extend(role["additional-settings"])
    matches = [
        setting.split("=", 1)[1]
        for setting in settings
        if isinstance(setting, str) and setting.startswith("CONFIG_FILE=")
    ]
    if len(matches) != 1:
        raise InferenceXRecipeError(f"expected one CONFIG_FILE in source recipe point; found {len(matches)}")
    return matches[0]


def _normalize_yaml_server_args(raw: dict[str, Any]) -> dict[str, Any]:
    args = {
        str(key).replace("-", "_"): value
        for key, value in raw.items()
        if isinstance(value, (str, int, float, bool, dict, list)) or value is None
    }
    aliases = {
        "tp": "tensor_parallel_size",
        "tp_size": "tensor_parallel_size",
        "pp_size": "pipeline_parallel_size",
        "dp": "data_parallel_size",
        "dp_size": "data_parallel_size",
        "ep_size": "moe_expert_parallel_size",
        "page_size": "block_size",
        "context_length": "max_model_len",
        "max_seq_len": "max_model_len",
        "chunked_prefill_size": "max_num_batched_tokens",
        "cuda_graph_max_bs": "max_cudagraph_capture_size",
        "expert_parallel_size": "moe_expert_parallel_size",
        "max_running_requests": "max_num_seqs",
    }
    for source_name, canonical_name in aliases.items():
        if source_name in args:
            args[canonical_name] = args[source_name]
    if args.get("no_enable_prefix_caching") is True or args.get("disable_radix_cache") is True:
        args["enable_prefix_caching"] = False
    if args.get("enforce_eager") is True or args.get("disable_cuda_graph") is True:
        args["cuda_graph_enabled"] = False
    elif "max_cudagraph_capture_size" in args:
        args["cuda_graph_enabled"] = True
    if "data_parallel_size" in args:
        args["attention_data_parallel_size"] = args["data_parallel_size"]
    if "enable_expert_parallel" not in args and isinstance(args.get("moe_expert_parallel_size"), int):
        args["enable_expert_parallel"] = args["moe_expert_parallel_size"] > 1
    return args


@lru_cache(maxsize=256)
def _load_yaml_mapping(content: str, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(content, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    except yaml.YAMLError as error:
        raise InferenceXRecipeError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceXRecipeError(f"{label} is not a mapping")
    return value


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [parsed for item in value if (parsed := _as_int(item)) is not None]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
