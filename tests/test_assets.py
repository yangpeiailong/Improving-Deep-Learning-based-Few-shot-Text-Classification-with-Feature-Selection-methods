import tempfile
import unittest
from pathlib import Path

from fs_lrtc.assets import load_asset_config, validate_bert_directory
from fs_lrtc.config import ConfigError


class AssetTests(unittest.TestCase):
    def test_loads_and_expands_local_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "assets.yaml"
            config.write_text(
                """schema_version: 1
fasttext:
  source_path: D:/fasttext/cc.en.300.bin
  reduced_path: D:/fasttext/cc.en.100.bin
  expected_source_dimension: 300
  target_dimension: 100
  overwrite_reduced: false
bert:
  path: D:/bert-base-uncased
  identity: google-bert/bert-base-uncased
  local_files_only: true
manifest_path: ${audit_root}/assets.json
""",
                encoding="utf-8",
            )
            loaded = load_asset_config(config, {"audit_root": Path("D:/paper/audit")})
            self.assertEqual(loaded.fasttext.target_dimension, 100)
            self.assertEqual(loaded.manifest_path, Path("D:/paper/audit/assets.json"))
            self.assertTrue(loaded.bert.local_files_only)

    def test_rejects_same_fasttext_source_and_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "assets.yaml"
            config.write_text(
                """fasttext:
  source_path: D:/same.bin
  reduced_path: D:/same.bin
  expected_source_dimension: 300
  target_dimension: 100
bert:
  path: D:/bert
  identity: bert-base-uncased
  local_files_only: true
manifest_path: D:/manifest.json
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_asset_config(config, {})

    def test_validates_minimal_bert_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "vocab.txt", "pytorch_model.bin"):
                (root / name).write_bytes(b"test")
            groups = validate_bert_directory(root)
            self.assertEqual([item.name for item in groups["required"]], ["config.json", "vocab.txt"])
            self.assertEqual(groups["weights"][0].name, "pytorch_model.bin")

    def test_rejects_bert_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "vocab.txt").write_text("[UNK]\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                validate_bert_directory(root)


if __name__ == "__main__":
    unittest.main()
