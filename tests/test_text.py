from __future__ import annotations

import unittest

from fs_lrtc.text import TextTokenizer


class TextTokenizerTests(unittest.TestCase):
    def test_normalization_and_tokenization(self):
        tokenizer = TextTokenizer()
        self.assertEqual(
            tokenizer.tokenize("<b>DON’T</b> &amp; café_42!"),
            ["don’t", "café", "42"],
        )


if __name__ == "__main__":
    unittest.main()
