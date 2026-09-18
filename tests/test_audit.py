import tempfile
import unittest
from pathlib import Path

import yaml

from fs_lrtc.data.audit import run_audit
from fs_lrtc.utils.hashing import file_sha256


class AuditTests(unittest.TestCase):
    def test_audit_detects_duplicates_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            paper = tmp_path / "paper"
            code = tmp_path / "code"
            raw = paper / "03_data" / "raw" / "demo"
            audit = paper / "03_data" / "audit"
            raw.mkdir(parents=True)
            code.mkdir()
            texts = raw / "texts.txt"
            labels = raw / "labels.txt"
            texts.write_text("same\nsame\nunique\n", encoding="utf-8")
            labels.write_text("a\nb\na\n", encoding="utf-8")
            metadata = {
                "dataset_id": "demo",
                "files": {
                    "text_sha256": file_sha256(texts),
                    "label_sha256": file_sha256(labels),
                },
                "observed": {"samples": 3, "classes": 2},
            }
            (raw / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")

            paths_file = tmp_path / "paths.yaml"
            paths_file.write_text(
                yaml.safe_dump(
                    {
                        "paper_root": str(paper),
                        "code_root": str(code),
                        "paths": {
                            "raw_data_root": str(paper / "03_data" / "raw"),
                            "audit_root": str(audit),
                        },
                    }
                ),
                encoding="utf-8",
            )
            datasets_file = tmp_path / "datasets.yaml"
            datasets_file.write_text(
                yaml.safe_dump(
                    {
                        "text_filename": "texts.txt",
                        "label_filename": "labels.txt",
                        "metadata_filename": "metadata.yaml",
                        "main_datasets": ["demo"],
                        "reserve_datasets": [],
                        "registry": {"demo": {"directory": "demo", "metadata": "demo/metadata.yaml"}},
                    }
                ),
                encoding="utf-8",
            )

            report = run_audit(paths_file, datasets_file)
            self.assertEqual(report["fatal_issue_count"], 0)
            self.assertEqual(report["warning_count"], 3)
            self.assertTrue((audit / "data_summary.csv").is_file())
            self.assertEqual(
                (audit / "label_conflicts.csv").read_text(encoding="utf-8-sig").count("\n"), 2
            )
            self.assertTrue((audit / "audit_report.json").is_file())

    def test_audit_supports_explicit_replacement_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            paper = tmp_path / "paper"
            code = tmp_path / "code"
            raw = paper / "03_data" / "raw" / "demo"
            audit = paper / "03_data" / "audit"
            raw.mkdir(parents=True)
            code.mkdir()
            texts = raw / "texts.txt"
            labels = raw / "labels.txt"
            texts.write_bytes(b"valid\ninvalid:\xef\xbf\n")
            labels.write_text("a\nb\n", encoding="utf-8")
            metadata = {
                "dataset_id": "demo",
                "decoding": {"encoding": "utf-8", "errors": "replace"},
                "files": {
                    "text_sha256": file_sha256(texts),
                    "label_sha256": file_sha256(labels),
                },
                "observed": {"samples": 2, "classes": 2},
            }
            (raw / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
            paths_file = tmp_path / "paths.yaml"
            paths_file.write_text(
                yaml.safe_dump(
                    {
                        "paper_root": str(paper),
                        "code_root": str(code),
                        "paths": {
                            "raw_data_root": str(paper / "03_data" / "raw"),
                            "audit_root": str(audit),
                        },
                    }
                ),
                encoding="utf-8",
            )
            datasets_file = tmp_path / "datasets.yaml"
            datasets_file.write_text(
                yaml.safe_dump(
                    {
                        "text_filename": "texts.txt",
                        "label_filename": "labels.txt",
                        "metadata_filename": "metadata.yaml",
                        "main_datasets": ["demo"],
                        "reserve_datasets": [],
                        "registry": {"demo": {"directory": "demo", "metadata": "demo/metadata.yaml"}},
                    }
                ),
                encoding="utf-8",
            )
            report = run_audit(paths_file, datasets_file)
            self.assertEqual(report["fatal_issue_count"], 0)
            self.assertTrue(
                any(issue["code"] == "decode_replacement_characters" for issue in report["issues"])
            )



if __name__ == "__main__":
    unittest.main()
