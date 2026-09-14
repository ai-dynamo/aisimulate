# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actually rerun the published strict SILICON geometry through the current native API.

Uses observed geometry/configuration only. Published prediction values are read
after each native call solely to verify reproduction; they are never model inputs.
"""

import argparse
import copy
import math
from pathlib import Path
import subprocess

from export import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--fpm-repo",
        type=Path,
        required=True,
        help="Repository containing the unchanged GB200 experimental overlay",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    source_commit = "f21ed55168074f8d32153a04d3cdfc524484722b"
    if subprocess.check_output(
        ["git", "diff", source_commit, "--", "crates/core", "python/aisimulate/src"],
        cwd=repo,
    ):
        raise ValueError("Prediction source differs from the recorded code")
    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    if not Path(native.__file__).resolve().is_relative_to(repo):
        raise ValueError(
            "The actual loaded native library is outside the selected worktree"
        )
    here = Path(__file__).resolve().parent
    published = read(here / "rows.json.gz")
    original_bindings = read(here / "original-input-bindings.json")
    gb200_binding = read(here / "gb200-probe-input-bindings.json")
    models, identities = {}, {}
    for profile, scope in (("full", "off"), ("decoder_bounded", "on")):
        binding = original_bindings[scope]
        config = copy.deepcopy(binding["prediction_config"])
        config["systems_path"] = str(repo / config["systems_path"])
        root = Path(config["systems_path"])
        actual = {
            str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        if actual != binding["systems_files_sha256"]:
            raise ValueError("Frozen operation calibration tables differ")
        models[profile] = RustForwardPassPerfModel.from_native(config)
        identities[profile] = {
            "config": binding["prediction_config"],
            "table_sha256": actual,
        }
    config = copy.deepcopy(gb200_binding["prediction_config"])
    config["systems_path"] = str(args.fpm_repo.resolve() / config["systems_path"])
    root = Path(config["systems_path"])
    actual = {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
    }
    if actual != gb200_binding["systems_files_sha256"]:
        raise ValueError("Original GB200 overlay differs")
    models["gb200"] = RustForwardPassPerfModel.from_native(config)
    identities["gb200"] = {
        "config": gb200_binding["prediction_config"],
        "table_sha256": actual,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "predictor_source_commit": source_commit,
        "actual_worktree_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "native_extension_sha256": sha(Path(native.__file__)),
        "program_sha256": sha(Path(__file__)),
        "published_rows_sha256": sha(here / "rows.json.gz"),
        "configuration_and_tables": identities,
        "correction_fitting": False,
    }
    write(args.output / "started.json", identity)
    rows, mismatches = [], []
    for original in published["rows"]:
        geometry = copy.deepcopy(
            original.get(
                "prediction_input",
                {
                    "version": 1,
                    "scheduled_requests": original.get("scheduled_requests"),
                },
            )
        )
        profile = (
            "gb200"
            if original["scope"] == "gb200-strict-op-coverage-probe"
            else original["profile"]
        )
        row = {
            k: original[k]
            for k in ("scope", "profile", "input_ordinal", "observed_ms")
            if k in original
        }
        try:
            predicted = models[profile].estimate_forward_pass_time_ms(geometry)
            if not math.isfinite(predicted) or predicted <= 0:
                raise ValueError(
                    "Native prediction is not finite positive milliseconds"
                )
        except Exception as error:
            row.update(
                status="prediction_unavailable",
                failure_type=type(error).__name__,
                failure=str(error),
            )
        else:
            row.update(status="predicted", predicted_ms=predicted)
        same = row["status"] == original["status"]
        if row["status"] == "predicted":
            same = same and row["predicted_ms"] == original.get("predicted_ms")
        elif profile == "gb200":
            same = (
                same
                and {"type": row["failure_type"], "message": row["failure"]}
                == original["failure"]
            )
        else:
            same = (
                same
                and row["failure_type"] == original.get("failure_type")
                and row["failure"] == original.get("failure")
            )
        if not same:
            mismatches.append(
                {k: row[k] for k in ("scope", "profile", "input_ordinal")}
            )
        rows.append(row)
    write(
        args.output / "actual-predictions.json.gz", {"identity": identity, "rows": rows}
    )
    write(
        args.output / "completion.json",
        {
            "actual_prediction_calls": len(rows),
            "exact_published_result_matches": len(rows) - len(mismatches),
            "mismatches": mismatches,
            "native_unchanged": sha(Path(native.__file__))
            == identity["native_extension_sha256"],
            "output_sha256": sha(args.output / "actual-predictions.json.gz"),
        },
    )
    if mismatches:
        raise ValueError(
            "Actual prediction differences retained in the new output directory"
        )


if __name__ == "__main__":
    main()
