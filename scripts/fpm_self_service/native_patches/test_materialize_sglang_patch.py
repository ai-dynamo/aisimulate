# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original CPU-only byte/provenance tests; no upstream module is embedded."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import materialize_sglang_patch as materializer


class MaterializerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.clean)

    def clean(self):
        # Only this test's own private read-only outputs are made removable.
        for path in self.root.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        self.temp.cleanup()

    def test_exact_hunks_reject_offsets_and_changed_context(self):
        original = b"first\nold\nlast\n"
        patch = b"--- a/example.py\n+++ b/example.py\n@@ -1,3 +1,3 @@\n first\n-old\n+new\n last\n"
        self.assertEqual(
            materializer.apply_exact_patch(original, patch, "example.py"),
            b"first\nnew\nlast\n",
        )
        with self.assertRaisesRegex(ValueError, "context"):
            materializer.apply_exact_patch(b"extra\n" + original, patch, "example.py")
        with self.assertRaisesRegex(ValueError, "position"):
            materializer.apply_exact_patch(original, patch.replace(b"+1,3", b"+2,3"), "example.py")

    def test_wrong_upstream_is_rejected_before_creating_output(self):
        base = self.root / "base.py"
        base.write_text("wrong upstream bytes\n")
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "exact reviewed base"):
            materializer.materialize(output, base)
        self.assertFalse(output.exists())

    def test_patch_or_license_changes_are_rejected(self):
        source = self.root / "patch-source"
        source.mkdir()
        for name in (
            "provenance.json",
            materializer.PATCH_NAME,
            "licenses/sglang-e087e662-LICENSE",
        ):
            target = source / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(materializer.ROOT / name, target)
        (source / materializer.PATCH_NAME).write_bytes(b"changed patch\n")
        with self.assertRaisesRegex(ValueError, "Patch or license hash"):
            materializer.contract(source)

    @unittest.skipUnless(
        os.environ.get("SGLANG_FPM_BASE_SOURCE"),
        "exact upstream bytes supplied separately",
    )
    def test_full_native_materialization_and_tamper_rejection(self):
        output = self.root / "materialized"
        receipt = materializer.materialize(output, Path(os.environ["SGLANG_FPM_BASE_SOURCE"]))
        meta = materializer.contract()
        self.assertEqual(
            materializer.sha256(output / "metrics_reporter.py"),
            meta["patched_file_sha256"],
        )
        self.assertFalse((output / "metrics_reporter.py").stat().st_mode & 0o222)
        self.assertFalse((output / materializer.RECEIPT_NAME).stat().st_mode & 0o222)
        self.assertEqual(
            materializer.verify_materialization(output)["receipt_sha256"],
            materializer.sha256(output / materializer.RECEIPT_NAME),
        )
        self.assertEqual(receipt["license_sha256"], meta["license_sha256"])
        self.assertFalse(receipt["serving_qualified"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            materializer.materialize(output, Path(os.environ["SGLANG_FPM_BASE_SOURCE"]))
        target = output / "metrics_reporter.py"
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"# altered\n")
        target.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            materializer.verify_materialization(output)
        self.assertEqual(json.loads((output / "provenance.json").read_text()), meta)


if __name__ == "__main__":
    unittest.main()
