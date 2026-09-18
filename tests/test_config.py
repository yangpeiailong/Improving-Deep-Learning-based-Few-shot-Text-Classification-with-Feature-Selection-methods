import tempfile
import unittest
from pathlib import Path

from fs_lrtc.config import load_paths


class ConfigTests(unittest.TestCase):
    def test_load_paths_expands_project_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "paths.yaml"
            config.write_text(
                """
paper_root: "D:/paper"
code_root: "D:/code"
paths:
  raw_data_root: "${paper_root}/03_data/raw"
  audit_root: "${paper_root}/03_data/audit"
  cache_root: "${code_root}/artifacts/cache"
""".strip(),
                encoding="utf-8",
            )
            paths = load_paths(config)
            self.assertEqual(paths["raw_data_root"], Path("D:/paper/03_data/raw"))
            self.assertEqual(paths["cache_root"], Path("D:/code/artifacts/cache"))


if __name__ == "__main__":
    unittest.main()
