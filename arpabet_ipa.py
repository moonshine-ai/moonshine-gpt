"""ARPAbet (ASCII, optional stress digits) → Unicode IPA for US English–oriented G2P exports."""

from __future__ import annotations

import re

_STRESS_SUFFIX = re.compile(r"[012]$")

SKIP_PHONEMES = frozenset(
    {
        "SIL",
        "SP",
        "SPN",
        "PAU",
        "HES",
        "BRTH",
        "NOISE",
        "NSN",
        "+",
    }
)

# AH/ER: when stress is stripped, AH0/AH1 collapse; ə is a common compromise for mixed
# contexts. ER → ɚ for r-colored vowel.
ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
    "AX": "ə",
    "IX": "ɨ",
    "AXR": "ɚ",
    "UX": "ʉ",
    "EL": "əl",
    "EM": "əm",
    "EN": "ən",
    "ENG": "ŋ",
}


def arpabet_tokens_to_ipa(tokens: list[str], *, skip_unknown: bool = False) -> str | None:
    """Map a sequence of ARPAbet tokens to a single IPA string (concatenated).

    Strips trailing stress ``0``/``1``/``2`` from each token. Returns ``None`` if
    *skip_unknown* is true and any token is missing from the table.
    """
    parts: list[str] = []
    for raw in tokens:
        if raw is None:
            continue
        p = _STRESS_SUFFIX.sub("", raw.strip().upper())
        if not p or p in SKIP_PHONEMES:
            continue
        ipa = ARPABET_TO_IPA.get(p)
        if ipa is None:
            if skip_unknown:
                return None
            raise ValueError(f"Unknown phoneme token: {raw!r} (normalized {p!r})")
        parts.append(ipa)
    return "".join(parts)
