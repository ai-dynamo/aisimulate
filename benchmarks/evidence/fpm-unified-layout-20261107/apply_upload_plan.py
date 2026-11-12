# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply upload-operations.parquet as ONE Hugging Face commit. Dry run unless --execute.

Not run by the AISIM-2337 agent. A maintainer reviews the plan, then runs it.
Needs huggingface_hub; the token comes from HF_TOKEN and is never printed.

  uv run --no-project --with 'huggingface_hub==0.35.3' --with hf_xet --with pyarrow --with 'numpy>=2' \
      python apply_upload_plan.py --tree PATH/fpm-unified --branch unified-layout-candidate [--execute]
"""

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

REPO = "nvidia/aisimulate-fpm-dataset"
BASE = "dbfcdd9eb62ea733efbc3f0cbd78277f2960061a"  # the revision the plan was computed from

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True, help="the migrated tree built by fpm_migration migrate")
parser.add_argument("--branch", required=True, help="a new branch; the public main branch is refused")
parser.add_argument("--ops", type=Path, default=Path(__file__).with_name("upload-operations.parquet"))
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
if args.branch == "main":
    raise SystemExit("refusing to write to main; land it through a reviewed pull request")

rows = pq.read_table(args.ops).to_pylist()
operations = []
for row in rows:
    if row["op"] == "copy":  # LFS only: server-side, no bytes uploaded
        operations.append(CommitOperationCopy(src_path_in_repo=row["src"], path_in_repo=row["dst"], src_revision=BASE))
    elif row["op"] == "add":
        operations.append(CommitOperationAdd(path_in_repo=row["dst"], path_or_fileobj=str(args.tree / row["src"])))
    else:
        operations.append(CommitOperationDelete(path_in_repo=row["dst"]))
print({kind: sum(r["op"] == kind for r in rows) for kind in ("copy", "add", "delete")}, "-> branch", args.branch)
if not args.execute:
    raise SystemExit("dry run: nothing sent")

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_branch(REPO, branch=args.branch, revision=BASE, repo_type="dataset")
info = api.create_commit(
    REPO,
    operations=operations,
    commit_message="Move FPM files to data/fpm/ byte-identical and add envelope manifests and catalog (AISIM-2337)",
    repo_type="dataset",
    revision=args.branch,
    parent_commit=BASE,
)
print(info.oid)
