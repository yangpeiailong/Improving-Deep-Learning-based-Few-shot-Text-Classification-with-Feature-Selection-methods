"""Deterministic lightweight tokenization shared by classic and BERT inputs."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


_HTML_TAG = re.compile(r"<[^>]*>")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class TextTokenizer:
    unicode_normalization: str = "NFKC"
    html_unescape: bool = True
    remove_html_tags: bool = True
    lowercase: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TextTokenizer":
        token_pattern = config.get("token_pattern", "unicode_words_with_internal_apostrophe")
        if token_pattern != "unicode_words_with_internal_apostrophe":
            raise ValueError(f"Unsupported token pattern: {token_pattern}")
        return cls(
            unicode_normalization=str(config.get("unicode_normalization", "NFKC")),
            html_unescape=bool(config.get("html_unescape", True)),
            remove_html_tags=bool(config.get("remove_html_tags", True)),
            lowercase=bool(config.get("lowercase", True)),
        )

    def normalize(self, text: str) -> str:
        value = unicodedata.normalize(self.unicode_normalization, text)
        if self.html_unescape:
            value = html.unescape(value)
        if self.remove_html_tags:
            value = _HTML_TAG.sub(" ", value)
        if self.lowercase:
            value = value.lower()
        return " ".join(value.split())

    def tokenize(self, text: str) -> list[str]:
        return _WORD.findall(self.normalize(text))

    def tokenize_many(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]
