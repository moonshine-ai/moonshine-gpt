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

# Known / unknown spans in hybrid encoder strings (IPA inside <k> must not contain ``</k>``).
_HYBRID_KU_RE = re.compile(r"<k>.*?</k>|<u>.*?</u>")
UNKNOWN_U_RE = re.compile(r"<u>.*?</u>")


def hybrid_ku_chunks(window: str) -> list[tuple[int, str]]:
    """Return ``(start_index, chunk)`` for each ``<k>…</k>`` or ``<u>…</u>`` in *window*."""
    return [(m.start(), m.group(0)) for m in _HYBRID_KU_RE.finditer(window)]


def extract_ipa_for_target_unknown(window: str, target_u_start: int, hyp_ipa: str) -> str:
    """Pick the IPA token aligned with the ``<u>…</u>`` that begins at *target_u_start*.

    Training targets use space-separated word IPA; we align on the sequence of ``<k>`` / ``<u>``
    chunks in *window*. Falls back to the full hypothesis when alignment is unclear.
    """
    chunks_pos = hybrid_ku_chunks(window)
    ku = [ch for _, ch in chunks_pos]
    hyp_words = hyp_ipa.split()
    if not ku:
        return hyp_ipa.strip()
    try:
        slot_idx = next(i for i, (pos, ch) in enumerate(chunks_pos) if ch.startswith("<u>") and pos == target_u_start)
    except StopIteration:
        return hyp_ipa.strip()
    if len(hyp_words) == len(ku):
        return hyp_words[slot_idx].strip()
    if len(ku) == 1:
        return hyp_ipa.strip()
    return hyp_ipa.strip()


def hybrid_context_window_bounds(h: str, u_start: int, u_end: int, n_chars: int) -> tuple[int, int]:
    """Char budget *n_chars* before/after an unknown span, expanded so no ``<k>``/``<u>`` tag is cut.

    Starts from ``[u_start - n_chars, u_end + n_chars)`` (clamped), then repeatedly grows the
    interval to fully contain any chunk that overlaps it (fixpoint).
    """
    lo = max(0, u_start - n_chars)
    hi = min(len(h), u_end + n_chars)
    chunks = [(m.start(), m.end()) for m in _HYBRID_KU_RE.finditer(h)]
    for _ in range(len(chunks) + 2):
        changed = False
        for s, e in chunks:
            if e <= lo or s >= hi:
                continue
            if s < lo or e > hi:
                lo, hi = min(lo, s), max(hi, e)
                lo = max(0, lo)
                hi = min(len(h), hi)
                changed = True
        if not changed:
            break
    return lo, hi


def clip_hybrid_window(window: str, u_start: int, u_end: int, max_len: int) -> tuple[str, int]:
    """Return a substring of *window* with length at most *max_len* that fully contains ``[u_start:u_end]``.

    Also returns the new start index of that span inside the clipped string.
    """
    if max_len <= 0 or len(window) <= max_len:
        return window, u_start
    span_len = u_end - u_start
    if span_len >= max_len:
        return window[u_start:u_end], 0
    margin = max_len - span_len
    left = margin // 2
    start = max(0, u_start - left)
    end = start + max_len
    if end > len(window):
        end = len(window)
        start = max(0, end - max_len)
    return window[start:end], u_start - start


def hybrid_all_known_to_ipa(h: str) -> str:
    """Strip ``<k>…</k>`` wrappers; *h* must not contain ``<u>`` spans."""
    if "<u>" in h:
        raise ValueError("hybrid string still contains <u> spans")
    return re.sub(r"<k>(.*?)</k>", r"\1", h, flags=re.DOTALL)


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


def iter_lexicon_window_training_pairs(
    text: str,
    ipa: str,
    lexicon: CmuDictLexicon,
    unknown_context_chars: int,
) -> list[tuple[str, str]]:
    """Split one (text, ipa) row into windowed (src, tgt) examples aligned with inference.

    For each ``<u>…</u>`` left-to-right, earlier unknowns are teacher-forced to ``<k>gold_ipa</k>``.
    Each example uses ``hybrid_context_window_bounds`` for the encoder window; *tgt* is the
    space-joined gold IPA for every ``<k>/<u>`` chunk fully inside that window.

    If the number of hybrid chunks does not match ``len(ipa.split())``, returns a single
    ``(full_hybrid, ipa)`` pair so training still sees the sentence.
    """
    ipa_words = ipa.split()
    h0 = lexicon.build_hybrid_source(text)

    def all_chunks(s: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group(0)) for m in _HYBRID_KU_RE.finditer(s)]

    ch0 = all_chunks(h0)
    if not ch0:
        return [(h0, ipa)]
    if len(ch0) != len(ipa_words):
        return [(h0, ipa)]
    if not UNKNOWN_U_RE.search(h0):
        return [(h0, ipa)]

    out: list[tuple[str, str]] = []
    h_work = h0
    while True:
        m = UNKNOWN_U_RE.search(h_work)
        if not m:
            break
        i, j = m.start(), m.end()
        lo, hi = hybrid_context_window_bounds(h_work, i, j, unknown_context_chars)
        window = h_work[lo:hi]
        chw = all_chunks(h_work)
        global_idxs = [gi for gi, (s, e, _) in enumerate(chw) if s >= lo and e <= hi]
        tgt = " ".join(ipa_words[gi] for gi in global_idxs) if global_idxs else ipa
        out.append((window, tgt))
        gi_u = next(gi for gi, (s, e, c) in enumerate(chw) if c.startswith("<u>") and s == i)
        gold = ipa_words[gi_u]
        h_work = h_work[:i] + f"<k>{gold}</k>" + h_work[j:]
    return out
