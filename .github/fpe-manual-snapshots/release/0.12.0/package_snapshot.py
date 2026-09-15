# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the completed manual run and create the reviewed release bootstrap artifact."""

import csv
import datetime
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "1f728534910187ecb2f021ef5e4bd4949bd555d0"
TOOLING_SHA = "58256c63de898b57c207d8829e5bb0cb3a5c7c3c"
BRANCH = "release/0.12.0"
WHEEL_SHA = "a06fb0c31d65956bb70effb6c3a4e9777f327db20c2ee918de12f2099869f4b9"
PREFIX = "python/aisimulate/src/aiconfigurator_core/systems/fpe_support_matrix/"
output = ROOT / "snapshot"
output.mkdir()
shards = json.loads((ROOT / "shards.json").read_text())
required = ROOT / "tooling/.github/fpe-required-probes.json"
tools = ROOT / "tooling/python/aisimulate/tools/support_matrix"
subprocess.run(
    [
        sys.executable,
        str(tools / "qualify_fpe_support_matrix.py"),
        str(ROOT / "collection"),
        "--expected-shards",
        json.dumps(shards),
        "--expected-sha",
        SOURCE_SHA,
        "--expected-wheel-sha256",
        WHEEL_SHA,
        "--required-probes",
        str(required),
        "--output",
        str(output / "fpe-qualification.json"),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(tools / "build_fpe_support_matrix.py"),
        str(ROOT / "collection"),
        "--output-dir",
        str(output / "data"),
    ],
    check=True,
)
qualification = json.loads((output / "fpe-qualification.json").read_text())
index = json.loads((output / "data/index.json").read_text())
rows = []
for name in index["files"]:
    path = output / "data" / name
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        shard_rows = list(reader)
    # Reproduce against the verified release wheel using the same manual harness.
    command_prefix = "python python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py "
    for row in shard_rows:
        assert row["Command"].startswith(command_prefix), row["Command"]
        row["Command"] = "python run_shard.py " + row["Command"][len(command_prefix) :]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(shard_rows)
    rows.extend(shard_rows)
ledger = []
for path in sorted((ROOT / "collection").rglob("fpe_support_matrix.json")):
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    ledger.append(
        {
            "path": str(path.relative_to(ROOT / "collection")),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "metrics": json.loads(path.with_name("run_metrics.json").read_text()),
        }
    )
(output / "raw-reports.json").write_text(json.dumps(ledger, indent=2) + "\n")
execution = []
for path in sorted((ROOT / "collection").rglob("progress.json")):
    progress = json.loads(path.read_text())
    assert "completed_at" in progress and all(s["status"] == "complete" for s in progress["shards"]), path
    execution.append(progress)
(output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
with zipfile.ZipFile(output / "snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for name in ["fpe-qualification.json", "raw-reports.json", "execution.json"]:
        bundle.writestr(name, (output / name).read_bytes())
    for name in ["index.json", *index["files"]]:
        bundle.writestr(PREFIX + name, (output / "data" / name).read_bytes())
summary = {
    "capability_rows": len(rows),
    "models": len({r["HuggingFaceID"] for r in rows}),
    "systems": len({r["System"] for r in rows}),
    "web_status_counts": dict(Counter(r["Status"] for r in rows)),
    "systems_summary": {
        system: dict(Counter(r["Status"] for r in rows if r["System"] == system))
        for system in sorted({r["System"] for r in rows})
    },
}
manifest = {
    "schema_version": 1,
    "branch": BRANCH,
    "source_sha": SOURCE_SHA,
    "source_version": "0.12.0",
    "tooling_sha": TOOLING_SHA,
    "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "archive_sha256": hashlib.sha256((output / "snapshot.zip").read_bytes()).hexdigest(),
    "qualification": qualification,
    "summary": summary,
    "execution": {
        "provider": "Brev",
        "cpu_workers": 8,
        "vcpus_per_worker": 8,
        "max_threads_per_shard": 8,
        "native_build_jobs": 6,
        "source_inventory": "release checkout",
        "harness": "pinned newer FPE harness",
    },
    "raw_report_ledger_sha256": hashlib.sha256((output / "raw-reports.json").read_bytes()).hexdigest(),
}
with (ROOT / "source.tar").open("rb") as f:
    manifest["source_archive_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
(output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2))
