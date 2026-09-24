# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic TEST_ONLY temporary fixtures; no real campaign or remote writes."""

import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from tools.glm53flash_hf import raw_archive as archive

pytestmark = pytest.mark.unit

LABELS = [
    {
        "backend": "vllm",
        "weight_quantization": "fp8",
        "tp": 2,
        "phase": "prefill",
        "role": "calibration",
    }
]
URI = "https://example.invalid/TEST_ONLY/campaign.tar.gz"


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glm53-raw-TEST_ONLY-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "TEST_ONLY_SOURCE"
        self.source.mkdir()
        (self.source / "failed-attempts").mkdir()
        (self.source / "failed-attempts/traceback.txt").write_bytes(b"TEST_ONLY original failure\n")
        (self.source / "empty-directory").mkdir()
        (self.source / "empty.log").touch()
        (self.source / "native-世界.jsonl").write_bytes(b"TEST_ONLY retained sample\n" * 100)
        (self.source / "blob.bin").write_bytes(bytes(range(256)) * 12_000)
        self.output = self.base / "TEST_ONLY_BUNDLE"

    def snapshot(self):
        return {
            str(path.relative_to(self.source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.source.rglob("*")
            if path.is_file()
        }

    def create(self):
        return archive.create_archive(self.source, self.output, URI, copy.deepcopy(LABELS))

    def rewrite_tar(self, transform):
        path = self.output / archive.ARCHIVE
        with tarfile.open(path, "r:gz") as source:
            entries = [(member, source.extractfile(member).read() if member.isfile() else None) for member in source]
        entries = transform(entries)
        with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as target:
            for member, raw in entries:
                target.addfile(member, io.BytesIO(raw) if raw is not None else None)

    def stream_verify(self):
        return archive.verify_archive(
            self.output / archive.ARCHIVE,
            self.output / archive.INVENTORY,
            self.output / archive.INPUT_MANIFEST,
        )

    def test_complete_payload_failed_attempts_empty_directories_and_metadata_preserved(
        self,
    ):
        before = self.snapshot()
        receipt = self.create()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(receipt["verification"]["status"], "PASS")
        self.assertEqual(receipt["verification"]["files"], len(before))
        self.assertEqual(receipt["verification"]["directories"], 3)
        self.assertEqual(receipt["native_or_accuracy_acceptance"], "NOT_EVALUATED")
        self.assertEqual(archive.verify_bundle(self.output), receipt["verification"])
        records = list(archive.inventory_records(self.output / archive.INVENTORY))
        self.assertEqual({r["path"]: r["sha256"] for r in records if r["kind"] == "file"}, before)
        external = json.loads((self.output / "external-raw-evidence.json").read_text())
        self.assertEqual(external[0]["sha256"], receipt["archive"]["sha256"])
        self.assertEqual(external[0]["external_uri_verification"], "NOT_CHECKED")

    def test_payload_reads_are_bounded_and_do_not_extract_files(self):
        observed = []
        original = archive.HashingReader.read

        def record_size(reader, size=-1):
            observed.append(size)
            return original(reader, size)

        with mock.patch.object(archive.HashingReader, "read", record_size):
            self.create()
        self.assertTrue(observed)
        self.assertLessEqual(max(observed), archive.CHUNK)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {
                archive.ARCHIVE,
                archive.INVENTORY,
                archive.INPUT_MANIFEST,
                "receipt.json",
                "external-raw-evidence.json",
            },
        )

    def test_symlinks_and_existing_or_nested_outputs_are_rejected(self):
        for kind in ("input_link", "root_link", "output_link", "existing", "nested"):
            with self.subTest(kind=kind):
                source, output = self.source, self.base / kind
                if kind == "input_link":
                    link = self.source / "danger"
                    link.symlink_to(self.source / "blob.bin")
                elif kind == "root_link":
                    source = self.base / "ROOT_LINK"
                    source.symlink_to(self.source, target_is_directory=True)
                elif kind == "output_link":
                    output.symlink_to(self.base / "missing")
                elif kind == "existing":
                    output.mkdir()
                    (output / "preserve.txt").write_text("TEST_ONLY existing")
                else:
                    output = self.source / "output"
                with self.assertRaises((ValueError, OSError)):
                    archive.create_archive(source, output, URI, LABELS)
                if kind == "input_link":
                    self.assertTrue((output / "failure.json").exists())
                    link.unlink()
                if kind == "existing":
                    self.assertEqual((output / "preserve.txt").read_text(), "TEST_ONLY existing")

    def test_source_changes_between_inventory_and_archive_fail_and_keep_original_failure(
        self,
    ):
        original = archive.write_archive

        def change(root_fd, output):
            target = self.source / "blob.bin"
            info = target.stat()
            raw = target.read_bytes()
            target.write_bytes(b"x" + raw[1:])
            os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
            return original(root_fd, output)

        with (
            mock.patch.object(archive, "write_archive", change),
            self.assertRaisesRegex(archive.ArchiveError, "changed"),
        ):
            self.create()
        self.assertTrue((self.output / archive.ARCHIVE).exists())
        self.assertTrue((self.output / "failure.json").exists())
        self.assertFalse((self.output / "external-raw-evidence.json").exists())
        self.assertEqual(
            (self.source / "failed-attempts/traceback.txt").read_bytes(),
            b"TEST_ONLY original failure\n",
        )

    def test_change_during_file_hash_is_detected(self):
        original = archive.hash_stream
        changed = False

        def mutate(stream):
            nonlocal changed
            result = original(stream)
            if not changed:
                changed = True
                with (self.source / "blob.bin").open("ab") as target:
                    target.write(b"TEST_ONLY mutation during hash")
            return result

        with (
            mock.patch.object(archive, "hash_stream", mutate),
            self.assertRaisesRegex(archive.ArchiveError, "stat changed"),
        ):
            self.create()
        self.assertTrue((self.output / "failure.json").exists())

    def test_regular_file_replaced_by_fifo_does_not_block_or_get_archived(self):
        original = archive.hash_source_file

        def change(root_fd, record):
            if record["path"] == "blob.bin":
                (self.source / "blob.bin").unlink()
                os.mkfifo(self.source / "blob.bin")
            return original(root_fd, record)

        with (
            mock.patch.object(archive, "hash_source_file", change),
            self.assertRaisesRegex(archive.ArchiveError, "changed type"),
        ):
            self.create()
        self.assertFalse((self.output / "receipt.json").exists())

    def test_file_set_changes_after_archive_are_detected(self):
        original = archive.verify_archive

        def change(*args):
            result = original(*args)
            (self.source / "late-file.txt").write_text("TEST_ONLY late evidence")
            return result

        with mock.patch.object(archive, "verify_archive", change), self.assertRaises(archive.ArchiveError):
            self.create()
        self.assertTrue((self.source / "late-file.txt").exists())
        self.assertFalse((self.output / "receipt.json").exists())

    def test_corrupt_file_bytes_in_valid_gzip_tar_are_rejected(self):
        self.create()

        def change(entries):
            for index, (member, raw) in enumerate(entries):
                if member.name.endswith("traceback.txt"):
                    entries[index] = member, b"x" + raw[1:]
            return entries

        self.rewrite_tar(change)
        with self.assertRaisesRegex(archive.ArchiveError, "SHA256 mismatch"):
            self.stream_verify()
        with self.assertRaisesRegex(archive.ArchiveError, "receipt SHA256"):
            archive.verify_bundle(self.output)

    def test_traversal_duplicate_missing_and_special_members_are_rejected(self):
        for mutation in ("traversal", "duplicate", "missing", "symlink"):
            with self.subTest(mutation=mutation):
                self.output = self.base / ("TEST_ONLY_" + mutation)
                self.create()

                def change(entries):
                    if mutation == "traversal":
                        entries[1][0].name = "payload/../../escape"
                    elif mutation == "duplicate":
                        entries.insert(1, entries[1])
                    elif mutation == "missing":
                        entries.pop(1)
                    else:
                        entries[1][0].type = tarfile.SYMTYPE
                        entries[1][0].linkname = "/unsafe"
                        entries[1] = entries[1][0], None
                    return entries

                self.rewrite_tar(change)
                with self.assertRaises((archive.ArchiveError, tarfile.TarError)):
                    self.stream_verify()
        self.assertFalse((self.base / "escape").exists())

    def test_truncated_gzip_and_nonzero_trailing_tar_data_are_rejected(self):
        self.create()
        path = self.output / archive.ARCHIVE
        original = path.read_bytes()
        path.write_bytes(original[:-8])
        with self.assertRaises((EOFError, OSError, archive.ArchiveError, tarfile.TarError)):
            self.stream_verify()
        path.write_bytes(gzip.compress(gzip.decompress(original) + b"TEST_ONLY forbidden trailing data"))
        with self.assertRaisesRegex(archive.ArchiveError, "trailing"):
            self.stream_verify()

    def test_credentials_invalid_labels_and_duplicate_inventory_are_rejected(self):
        for uri in (
            "https://user:secret@example.invalid/raw",
            URI + "?token=TEST_ONLY",
            "file:///tmp/raw",
        ):
            with self.assertRaises(archive.ArchiveError):
                archive.validate_identity(uri, LABELS)
        with self.assertRaises(archive.ArchiveError):
            archive.validate_identity(URI, LABELS * 2)
        self.create()
        path = self.output / archive.INVENTORY
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join([lines[0], lines[0], *lines[1:]]))
        with self.assertRaisesRegex(archive.ArchiveError, "duplicate root"):
            list(archive.inventory_records(path))

    def test_bundle_byte_counts_and_external_receipts_cannot_be_rewritten(self):
        self.create()
        path = self.output / "receipt.json"
        raw = path.read_bytes()
        receipt = json.loads(raw)
        receipt["archive"]["bytes"] += 1
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(archive.ArchiveError, "byte count"):
            archive.verify_bundle(self.output)
        path.write_bytes(raw)
        external = self.output / "external-raw-evidence.json"
        records = json.loads(external.read_text())
        records[0]["sha256"] = "0" * 64
        external.write_text(json.dumps(records))
        with self.assertRaisesRegex(archive.ArchiveError, "external evidence"):
            archive.verify_bundle(self.output)

    def test_cli_creates_and_stream_verifies_synthetic_bundle(self):
        labels = self.base / "TEST_ONLY_LABELS.json"
        labels.write_text(json.dumps(LABELS))
        script = str(Path(archive.__file__).resolve())
        create = subprocess.run(
            [
                sys.executable,
                script,
                "create",
                "--source",
                str(self.source),
                "--output",
                str(self.output),
                "--uri",
                URI,
                "--labels",
                str(labels),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.base,
        )
        self.assertEqual(json.loads(create.stdout)["status"], "PASS")
        verify = subprocess.run(
            [sys.executable, script, "verify", "--bundle", str(self.output)],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.base,
        )
        self.assertEqual(json.loads(verify.stdout)["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
