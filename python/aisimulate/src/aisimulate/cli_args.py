# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight argument parsing before supervised runtime imports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .detail import parse_detail_sections


class _CliConfigError(ValueError):
    pass


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.MappingNode] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # Check authored keys before merge expansion adds inherited keys. A
        # shared alias may already be flattened, so validate each node once.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            seen: set[Any] = set()
            merge_key = object()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    key = merge_key
                elif key_node.tag == "tag:yaml.org,2002:value":
                    key = self.construct_scalar(key_node)
                else:
                    key = self.construct_object(key_node)
                try:
                    duplicate = key in seen
                    seen.add(key)
                except TypeError:
                    # Let SafeLoader produce its normal unhashable-key error.
                    continue
                if duplicate:
                    label = "<<" if key is merge_key else key
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"duplicate mapping key {label!r}",
                        key_node.start_mark,
                    )
        super().flatten_mapping(node)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisimulate",
        description="Predict or recommend an LLM serving configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "recommend"):
        child = subparsers.add_parser(command)
        child.add_argument("-c", "--config", required=True)
        child.add_argument(
            "--stack",
            help="execution stack; defaults to engine, or dynamo-policy when router is configured",
        )
        child.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="PATH=YAML_VALUE",
        )
        child.add_argument("--output-dir", default="./aisimulate-output")
        child.add_argument("--overwrite", action="store_true")
        child.add_argument("--format", choices=("table", "json"), default="table")
    subparsers.choices["predict"].add_argument("--capture-per-request", action="store_true")
    subparsers.choices["predict"].epilog = (
        "AgentX replay: use weka, agentic_mooncake, or agentic Dynamo traces with "
        "trace_timestamps and positive agentic_lanes. The offline engine stack supports "
        "aggregated and P/D vLLM/SGLang, HBM-only, speculative decoding disabled. "
        "agentic_snapshot selects seeded starts; agentic_warmup primes saved prefixes; "
        "agentic_profile enables duration controls on the offline engine or optional dynamo-policy stack; "
        "legacy --stack dynamo does not support profiles. Results are functional_only; "
        "hardware accuracy and complete AgentX recipe parity are not qualified."
    )
    subparsers.choices["predict"].add_argument(
        "--detail",
        type=parse_detail_sections,
        default=(),
        metavar="SECTIONS",
        help="comma-separated summary,memory,time,energy,source, or all; unavailable evidence is explicit",
    )
    subparsers.choices["predict"].add_argument(
        "--diagnostics", choices=("power",), help="compatibility alias for power diagnostics; prefer --detail energy"
    )
    subparsers.choices["predict"].add_argument(
        "--detail-top-n",
        "--diagnostics-top-n",
        dest="diagnostics_top_n",
        type=_positive_int,
        default=12,
        metavar="N",
        help="maximum operations per phase in detail tables; JSON retains all operations (default: 12)",
    )
    subparsers.choices["predict"].add_argument(
        "--online",
        action="store_true",
        help="pace prediction against the real wall clock instead of virtual time",
    )
    return parser


def select_stack(explicit: str | None, config: dict[str, Any]) -> str:
    """Select an optional integration only when no stack was explicitly requested."""

    if explicit is not None:
        return explicit
    return "dynamo-policy" if "router" in config else "engine"


def _load_mapping(path: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except OSError as exc:
        raise _CliConfigError(f"could not read configuration {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise _CliConfigError(f"malformed YAML in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise _CliConfigError(f"configuration {source} must contain one YAML mapping")
    return value


def _apply_overrides(data: dict[str, Any], overrides: list[str], *, command: str) -> None:
    for assignment in overrides:
        if "=" not in assignment:
            raise _CliConfigError(f"invalid --set {assignment!r}; expected PATH=YAML_VALUE")
        raw_path, raw_value = assignment.split("=", 1)
        parts = raw_path.split(".")
        if not raw_path or any(not part or part.isdigit() for part in parts):
            raise _CliConfigError(f"invalid --set path {raw_path!r}; sequence indexes are unsupported")
        if command == "predict" and parts[0] in {"optimization", "optimizer"}:
            raise _CliConfigError(f"--set path {raw_path!r} is not in the schema")
        current: Any = data
        for part in parts[:-1]:
            if not isinstance(current, dict):
                raise _CliConfigError(f"--set path {raw_path!r} crosses a non-mapping value")
            if part not in current:
                current[part] = {}
            current = current[part]
        leaf = parts[-1]
        if not isinstance(current, dict):
            raise _CliConfigError(f"--set path {raw_path!r} crosses a non-mapping value")
        try:
            current[leaf] = yaml.load(raw_value, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as exc:
            raise _CliConfigError(f"invalid YAML value for --set {raw_path!r}: {exc}") from exc
