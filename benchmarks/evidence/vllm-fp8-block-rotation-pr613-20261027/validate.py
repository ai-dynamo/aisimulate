# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Jan Fostier
# SPDX-License-Identifier: Apache-2.0

"""Exercise the evidence reader against corrupt records without changing raw data."""

import json
import shutil
import tempfile
from pathlib import Path

import analyze

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    for scenario in ["valid", "missing", "duplicate", "negative", "wrong_kernel", "outlier"]:
        with tempfile.TemporaryDirectory(prefix="pr613-reader-check-") as temporary:
            direct = Path(temporary) / "direct"
            shutil.copytree(HERE / "direct", direct)
            path = direct / "base-0-measurements.jsonl"
            rows = analyze.read_rows(path)
            if scenario == "missing":
                rows.pop()
            elif scenario == "duplicate":
                rows[1] = dict(rows[0])
            elif scenario == "negative":
                rows[0]["published_latency_ms"] = -1
            elif scenario == "wrong_kernel":
                rows[0]["kernel_source"] = "intentionally_incorrect_kernel"
            elif scenario == "outlier":
                rows[0]["published_latency_ms"] *= 2
                rows[0]["loop_latency_ms"] *= 2
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            try:
                result = analyze.direct(direct)
            except ValueError as error:
                if scenario in ["valid", "outlier"]:
                    raise
                checks.append({"scenario": scenario, "result": "rejected", "reason": str(error)})
            else:
                if scenario not in ["valid", "outlier"]:
                    raise RuntimeError(f"Corrupt records were accepted: {scenario}")
                if result["measurement_count"] != 504 or result["excluded_measurements"] != 0:
                    raise RuntimeError("Reader discarded measurements")
                if scenario == "outlier" and len(result["flagged_outliers"]) != 1:
                    raise RuntimeError("Injected outlier was not flagged exactly once")
                checks.append(
                    {
                        "scenario": scenario,
                        "result": "accepted; all samples retained",
                        "outliers": len(result["flagged_outliers"]),
                    }
                )
    report = {"status": "passed", "checks": checks, "raw_measurements_modified": False}
    (HERE / "analysis/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
