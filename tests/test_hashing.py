import hashlib
import tempfile
import unittest
from pathlib import Path

from fs_lrtc.utils.hashing import file_sha256, text_sha256


class HashingTests(unittest.TestCase):
    def test_hash_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "value.txt"
            target.write_bytes(b"abc")
            expected = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(file_sha256(target), expected)
            self.assertEqual(text_sha256("abc"), expected)


if __name__ == "__main__":
    unittest.main()
