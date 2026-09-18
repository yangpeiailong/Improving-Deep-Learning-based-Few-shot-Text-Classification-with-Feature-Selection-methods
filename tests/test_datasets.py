import tempfile
import unittest
from pathlib import Path

from fs_lrtc.data.datasets import load_dataset_specs, read_utf8_lines


class DatasetTests(unittest.TestCase):
    def test_registry_resolves_expected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            raw = tmp_path / "raw"
            dataset = raw / "demo"
            dataset.mkdir(parents=True)
            (dataset / "texts.txt").write_text("one\ntwo\n", encoding="utf-8")
            (dataset / "labels.txt").write_text("a\nb\n", encoding="utf-8")
            config = tmp_path / "datasets.yaml"
            config.write_text(
                """
text_filename: texts.txt
label_filename: labels.txt
metadata_filename: metadata.yaml
main_datasets: [demo]
reserve_datasets: []
registry:
  demo: {directory: demo, metadata: demo/metadata.yaml}
""".strip(),
                encoding="utf-8",
            )
            [spec] = load_dataset_specs(config, raw, scope="main")
            self.assertEqual(spec.dataset_id, "demo")
            self.assertEqual(spec.role, "main")
            self.assertEqual(read_utf8_lines(spec.texts_path), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
