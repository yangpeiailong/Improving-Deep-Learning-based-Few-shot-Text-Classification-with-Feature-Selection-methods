from __future__ import annotations

import unittest

from fs_lrtc.data.preparation import SourceRecord, select_primary_records, strip_flattened_20ng_headers, text_sha256


def record(row: int, label: str, text: str) -> SourceRecord:
    digest = text_sha256(text)
    return SourceRecord(row, label, text, text, digest, digest, "none")


class PreparationTests(unittest.TestCase):
    def test_deduplication_and_conflict_policy(self):
        kept, excluded = select_primary_records(
            [record(1, "a", "same"), record(2, "a", "same"), record(3, "x", "conflict"), record(4, "y", "conflict"), record(5, "b", "unique")]
        )
        self.assertEqual([item.original_row for item in kept], [1, 5])
        self.assertEqual([item["reason"] for item in excluded].count("same_label_exact_duplicate"), 1)
        self.assertEqual([item["reason"] for item in excluded].count("conflicting_exact_text_group"), 2)

    def test_flattened_header_removal(self):
        text = "Path: a!b From: x@example.org Subject: Example Lines: 2  This is the body."
        body, changed = strip_flattened_20ng_headers(text)
        self.assertTrue(changed)
        self.assertEqual(body, "This is the body.")

    def test_repeated_flattened_header_block(self):
        text = (
            "Xref: x Newsgroups: a Path: p From: f Lines: 2  "
            "Newsgroups: a Subject: s Organization: o  Actual body."
        )
        body, changed = strip_flattened_20ng_headers(text)
        self.assertTrue(changed)
        self.assertEqual(body, "Actual body.")


if __name__ == "__main__":
    unittest.main()
