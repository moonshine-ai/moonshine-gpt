"""CMUdict TSV → hybrid encoder strings for lexicon-informed G2P.

Unambiguous entries (exactly one distinct IPA for a spelling, case-insensitive)
are wrapped as ``<k>{ipa}</k>`` so the Transformer sees the dictionary phones
directly. Ambiguous spellings, OOV words, punctuation, and digit runs use
``<u>{original_text}</u>`` so the model still predicts those spans from context.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict

# ASCII “words” for dictionary lookup; everything else is passed through as unknown spans.
_TOKEN_RE = re.compile(r"[A-Za-z']+|\s+|[^A-Za-z'\s]+")


class CmuDictLexicon:
    def __init__(self, word_to_ipa: dict[str, str]):
        self._word_to_ipa = word_to_ipa

    @classmethod
    def from_tsv(cls, path: str) -> CmuDictLexicon:
        path = os.path.abspath(path)
        groups: dict[str, set[str]] = defaultdict(set)
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or not row[0].strip():
                    continue
                if row[0].lstrip().startswith("#"):
                    continue
                if row[0].strip() == "text" and len(row) >= 2 and row[1].strip() == "ipa":
                    continue
                if len(row) < 2:
                    continue
                word, ipa = row[0].strip(), row[1].strip()
                if not word or not ipa:
                    continue
                groups[word.lower()].add(ipa)
        word_to_ipa: dict[str, str] = {}
        for key, ipas in groups.items():
            if len(ipas) == 1:
                word_to_ipa[key] = next(iter(ipas))
        return cls(word_to_ipa)

    def lookup_unambiguous(self, word: str) -> str | None:
        return self._word_to_ipa.get(word.lower())

    def build_hybrid_source(self, text: str) -> str:
        parts: list[str] = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            if tok.isspace():
                parts.append(tok)
            elif re.fullmatch(r"[A-Za-z']+", tok):
                ipa = self.lookup_unambiguous(tok)
                if ipa is not None:
                    parts.append(f"<k>{ipa}</k>")
                else:
                    parts.append(f"<u>{tok}</u>")
            else:
                parts.append(f"<u>{tok}</u>")
        return "".join(parts)
