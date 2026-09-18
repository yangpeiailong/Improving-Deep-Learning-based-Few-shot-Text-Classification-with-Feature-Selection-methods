from __future__ import annotations

import unittest

from fs_lrtc.environment import compare_versions, package_versions


class EnvironmentTests(unittest.TestCase):
    def test_compare_versions_reports_only_mismatches(self):
        observed = {"a": "1", "b": "2", "c": None}
        expected = {"a": "1", "b": "3", "c": "4"}
        self.assertEqual(
            compare_versions(observed, expected),
            ["b: expected 3, observed 2", "c: expected 4, observed None"],
        )

    def test_installed_project_is_detected(self):
        self.assertIsNotNone(package_versions(["fs-lrtc"])["fs-lrtc"])


if __name__ == "__main__":
    unittest.main()
