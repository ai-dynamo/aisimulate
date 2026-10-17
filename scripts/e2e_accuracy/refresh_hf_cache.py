# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal scheduled-job entry point; public contribution APIs remain libraries."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.e2e_accuracy.hf_cache import InferenceXClient, prepare_snapshot
from scripts.e2e_accuracy.hf_publish import publish_snapshot


def main():
    output = Path(os.environ.get("E2E_CACHE_OUTPUT", "results/hf-e2e-cache"))
    http_cache = Path(os.environ.get("E2E_HTTP_CACHE", "results/inferencex-http"))
    options = InferenceXClient(http_cache).get("views/options")
    models = sorted({entry["name"] for entry in options.json()["models"]})
    if not models:
        raise ValueError("Model discovery returned no models")

    def fetch(model):
        return model, InferenceXClient(http_cache).get("benchmarks", model=model)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = dict(pool.map(fetch, models))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "scripts/e2e_accuracy/hf_cache.py"], text=True
    ).strip():
        raise ValueError("Commit the normalizer before freezing source evidence")
    publication = {
        "redistribution_status": "unresolved",
        "terms_reference": None,
        "terms_request": "https://github.com/SemiAnalysisAI/InferenceX-app/pull/1362#issuecomment-6204169098",
        "attribution": "InferenceX by SemiAnalysis",
        "notice": "Unofficial frozen copy; not an upstream release.",
    }
    index = prepare_snapshot(output, options, responses, normalizer_revision=revision, publication=publication)
    result = {"status": "prepared", "snapshot_id": index["snapshot_id"], "models": len(models)}
    if os.environ.get("E2E_PUBLISH") == "true":
        identity = subprocess.check_output(["git", "var", "GIT_AUTHOR_IDENT"], text=True).rsplit(" ", 2)[0]
        result = publish_snapshot(output, os.environ["HF_DATASET_ID"], signoff=identity)
    print(json.dumps(result, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write("### InferenceX HF cache\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
