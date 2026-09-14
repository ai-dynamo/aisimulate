# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for local documentation destination checks."""

import tempfile
import unittest
from pathlib import Path

from scripts.check_documentation_links import check_file, local_links


class DocumentationLinksTest(unittest.TestCase):
    def test_extracts_inline_images_references_and_balanced_destinations(self):
        text = (
            "[guide](guide.md#setup) ![plot](plot.png)\n"
            '[named](<my guide.md> "Title") [nested](name_(one).md)\n'
            '[ref]: ../guide.md "Title"\n'
            "[encoded](my%20guide.md) [escaped](name_\\(one\\).md)\n"
        )
        self.assertEqual(
            local_links(text),
            [
                (1, "guide.md"),
                (1, "plot.png"),
                (2, "my guide.md"),
                (2, "name_(one).md"),
                (3, "../guide.md"),
                (4, "my guide.md"),
                (4, "name_(one).md"),
            ],
        )

    def test_ignores_code_comments_and_nonlocal_destinations(self):
        text = (
            "<!-- [hidden](absent.md) -->\n"
            "```md\n[example](absent.md)\n```\n"
            "~~~md\n[example](absent.md)\n~~~\n"
            "`[inline](absent.md)` [web](https://example.com/a)\n"
            "[mail](mailto:docs@example.com) [anchor](#heading) [cdn](//example.com/a)\n"
            "[real](real.md)\n"
        )
        self.assertEqual(local_links(text), [(10, "real.md")])

    def test_checks_multiline_link_labels(self):
        self.assertEqual(local_links("[Sweep Configuration\nProviders](provider.md)"), [(2, "provider.md")])

    def test_checks_code_formatted_link_labels(self):
        self.assertEqual(local_links("[`file.py`](file.py)"), [(1, "file.py")])

    def test_reports_deleted_target_and_repository_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            docs = root / "docs"
            docs.mkdir()
            target = root / "source.py"
            target.write_text("")
            page = docs / "guide.md"
            page.write_text("[source](../source.py) [folder](../docs/)\n")
            self.assertEqual(check_file(page, root), [])
            target.unlink()
            self.assertEqual(check_file(page, root), ["docs/guide.md:1: missing local destination: ../source.py"])
            page.write_text("[escape](../../)")
            self.assertEqual(check_file(page, root), ["docs/guide.md:1: missing local destination: ../../"])


if __name__ == "__main__":
    unittest.main()
