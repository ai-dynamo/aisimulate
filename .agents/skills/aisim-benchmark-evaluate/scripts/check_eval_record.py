# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check, and optionally verify and attest, a stored ``aisim-eval/v1`` record.

Without ``--predictor`` it loads the record (the file name must equal its content
hash) and reports ``gpu_time`` findings at archival and qualification level.

With ``--predictor FILE.py`` (a module exposing ``predict(inputs) -> float | None``
and ``fingerprint() -> str``, both built from the canonical ``best_available``
predictor) it recomputes the prediction from the recorded truth table with
``verify_record``. ``--attest-root DIR`` then writes the superseding ``verified``
record with ``attest`` + ``write_record``. Exit code 1 means a failed check.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from aisim_core.data.eval import attest, check_record_gpu_time, load_record, verify_record, write_record
from aisim_core.data.gpu_time import check_gpu_time


def _load_predictor(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_predictor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict, module.fingerprint()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=Path)
    p.add_argument(
        "--truth-root", type=Path, help="directory holding the truth artifacts (default: record's directory)"
    )
    p.add_argument("--predictor", type=Path)
    p.add_argument("--attest-root", type=Path, help="dataset root to write the verified record into")
    args = p.parse_args(argv)

    record = load_record(args.record)
    out = {
        "record_id": record.record_id,
        "tier": record.tier,
        "status": record.status,
        "n_scored": record.results.n_scored,
        "n_total": record.results.n_total,
        "mape_pct": record.results.mape_pct,
        "gpu_time_findings": {
            level: [f.__dict__ for f in check_gpu_time(record.gpu_time, level)]
            for level in ("archival", "qualification")
        },
        "tier_findings": [f.__dict__ for f in check_record_gpu_time(record)],
    }
    code = 0
    if args.predictor:
        predict, fingerprint = _load_predictor(args.predictor)
        report = verify_record(record, args.truth_root or args.record.parent, predict, fingerprint)
        out["verification"] = {"ok": report.ok, "problems": report.problems}
        if not report.ok:
            code = 1
        elif args.attest_root:
            verified = attest(record, report)
            out["verified_record_path"] = str(write_record(args.attest_root, verified))
            out["verified_record_id"] = verified.record_id
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
