#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collect four single-lane diagnostic chains before the full cache-gated sweep."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from collect import Collector, save_json


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.cache_receipt.read_text())
    save_json(args.output / "cache-geometry.json", receipt)
    manifest = {"purpose": "diagnostic cache qualification; excluded from campaign", "cases": [], "status": "running"}
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600), read_bufsize=4 * 1024 * 1024
        ) as session:
            collector = Collector(session, args.base_url, args.model, args.output)
            for length in (4096, 32768):
                prompts, tool = await collector.prompts(f"qualify-p{length}", length, 1)
                for cache in ("cold", "warm"):
                    case = await collector.case(f"qualify-p{length}-{cache}", prompts, tool, cache)
                    manifest["cases"].append(case)
                    save_json(args.output / "collector.json", manifest)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save_json(args.output / "collector.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8800")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
