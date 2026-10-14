# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze public DSpark reference provenance; never label it matched validation."""

import hashlib
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

POINTS = [(442741, 10585002129), (442271, 10268876053)]
OUTPUT = Path(__file__).parent / "references.json"


def main():
    results = []
    for benchmark_id, artifact_id in POINTS:
        endpoint = f"https://inferencex.semianalysis.com/api/v1/benchmark-siblings?id={benchmark_id}"
        with urllib.request.urlopen(endpoint, timeout=60) as response:
            metadata = response.read()
        archive = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/SemiAnalysisAI/InferenceX/actions/artifacts/{artifact_id}/zip",
            ]
        )
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            server = files.read("results/server.log")
        text = server.decode(errors="replace")
        patterns = {
            "synthetic_acceptance_length": r"synthetic_acceptance_length[\"']?[:= ]+([0-9.]+)",
            "num_speculative_tokens": r"num_speculative_tokens[\"']?[:= ]+(\d+)",
            "speculative_num_draft_tokens": r"speculative_num_draft_tokens[\"']?[:= ]+(\d+)",
            "dspark_block_size": r"dspark_block_size[\"']?[:= ]+(\d+)",
            "adaptive_verification": r"enable_adaptive_verification[\"']?[:= ]+(True|False|true|false)",
            "ragged_verify_mode": r"SGLANG_RAGGED_VERIFY_MODE[=: ]+(\w+)",
        }
        results.append(
            {
                "benchmark_id": benchmark_id,
                "artifact_id": artifact_id,
                "metadata_endpoint": endpoint,
                "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "server_log_sha256": hashlib.sha256(server).hexdigest(),
                "observed_fields": {name: sorted(set(re.findall(pattern, text))) for name, pattern in patterns.items()},
                "metadata": json.loads(metadata),
                "status": "reference_only_pending_runtime_and_workload_reconciliation",
            }
        )
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
