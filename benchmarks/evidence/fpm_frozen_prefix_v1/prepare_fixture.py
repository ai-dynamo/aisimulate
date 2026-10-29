# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit a longer synthetic seed after verifying its original feature/target rows."""

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path


def main(source: Path, output: Path, trees: int):
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import fpm_learned

    manifest = json.loads((source / "seed/manifest.json").read_text())
    artifact = json.loads((source / "seed" / manifest["artifact_file"]).read_text())
    assert list(manifest["stores"]) == ["pure_decode"]
    store = manifest["stores"]["pure_decode"]
    generator = source.parent / "benchmark.py"
    iterations = list(runpy.run_path(str(generator))["rows"](store["row_count"], 0))
    assert len(iterations) == store["row_count"]
    columns = [f"x{index}" for index in range(len(artifact["features"]))]
    original = pq.read_table(
        source / "seed" / store["rows_file"], columns=columns + ["wall_time_ms"]
    ).to_pylist()
    for metrics, row in zip(iterations, original, strict=True):
        features = fpm_learned.compute_features(metrics)
        assert [features[name] for name in artifact["features"]] == [
            row[name] for name in columns
        ]
        assert fpm_learned.iteration_wall_ms(metrics) == row["wall_time_ms"]
    output.mkdir()
    fitted = fpm_learned.train(
        iterations,
        "decode",
        features=artifact["features"],
        target=artifact["target"],
        max_iter=trees,
        learning_rate=store["learning_rate"],
        max_leaf_nodes=store["max_leaf_nodes"],
        min_samples_leaf=store["min_samples_leaf"],
        random_state=0,
        early_stopping=False,
        online_checkpoint_dir=output / "seed",
        identity=manifest["identity"],
    )
    assert len(fitted["stores"]["pure_decode"]["trees"]) == trees
    shutil.copyfile(source / "queries.json", output / "queries.json")
    print(
        json.dumps(
            {
                "history_rows": len(iterations),
                "trees": trees,
                "verified_feature_target_rows": len(iterations),
                "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(
                    (source / "seed/manifest.json").read_bytes()
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    (output / "seed/manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    main(args.source, args.output, args.trees)
