#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned AgentX client without changing its workload/controller semantics."""

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import agentx_cli
import aiperf.cli_runner
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.resolution.resolvers import build_default_resolver_chain
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.semianalysis_cc_traces_weka import SemiAnalysisCCTracesWekaLoader
from datasets import load_dataset

DATASET = "semianalysisai/cc-traces-weka-062126"
REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
CONTEXT = 262144


def peak(value):
    maximum = 0
    if isinstance(value, dict):
        if isinstance(value.get("in"), int) and isinstance(value.get("out"), int):
            maximum = value["in"] + value["out"]
        for child in value.values():
            if isinstance(child, dict):
                maximum = max(maximum, peak(child))
            elif isinstance(child, list) and child and isinstance(child[0], dict):
                maximum = max(maximum, *(peak(item) for item in child))
    return maximum


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, revision=REVISION, split="train", trust_remote_code=False)
    rows = [{"id": row["id"], "peak_context": peak(row)} for row in dataset]
    eligible = [row["id"] for row in rows if row["peak_context"] <= CONTEXT]
    if len(rows) != 393 or len(eligible) != 220:
        raise ValueError("Pinned Weka corpus eligibility differs from reviewed campaign")
    receipt = {
        "dataset": DATASET,
        "revision": REVISION,
        "context_limit": CONTEXT,
        "plays": rows,
        "eligible": eligible,
        "harness_version": importlib.metadata.version("agentx-harness"),
    }
    if receipt["harness_version"] != "1.0.12":
        raise ValueError("Unexpected AgentX version")
    (output / "corpus-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"plays": len(rows), "eligible": len(eligible), "revision": REVISION}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args, command = parser.parse_known_args()
    if args.prepare:
        prepare(args.prepare)
        return
    if not args.record or not command:
        parser.error("--record and AgentX arguments are required")
    if command[0] == "--":
        command = command[1:]
    if importlib.metadata.version("agentx-harness") != "1.0.12":
        raise ValueError("Unexpected AgentX version")
    args.record.mkdir(parents=True, exist_ok=False)
    BaseHFDatasetLoader.hf_revision = REVISION
    if os.environ.get("AGENTX_COLLECTION_RECORD") != str(args.record):
        raise ValueError("The pinned client loader hook is required")
    (args.record / "command.json").write_text(json.dumps(command, indent=2))
    if args.preflight:

        def preflight(plan):
            if not plan.is_single_run:
                raise ValueError("Collection requires exactly one run")
            run = aiperf.cli_runner._make_benchmark_run(plan.configs[0], random_seed=42)
            build_default_resolver_chain().resolve_all(run)
            Tokenizer.from_pretrained(run.cfg.tokenizer.name, trust_remote_code=True)

            async def validate_loader():
                loader = SemiAnalysisCCTracesWekaLoader(run=run, hf_dataset_name=DATASET)
                dataset = load_dataset(DATASET, revision=REVISION, split="train")
                return loader._validate_rows(dataset)

            selected = asyncio.run(validate_loader())
            if len(selected) != 220:
                raise ValueError("Official loader eligibility changed")
            print(json.dumps({"preflight": "passed", "eligible": len(selected)}))

        aiperf.cli_runner.run_benchmark = preflight
    sys.argv = ["agentx", *command]
    agentx_cli.main()


if __name__ == "__main__":
    main()
