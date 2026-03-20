#!/usr/bin/env python3
"""
Export the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
to training TSVs: orthographic word (graphemes) and IPA from ARPAbet phones.

Default source is upstream ``cmudict.dict`` on the ``master`` branch (downloaded
via HTTPS). Stress marks (0/1/2) are stripped before the same ARPAbet→IPA mapping
as ``build_librig2p_nostress_tsv.py`` (see ``arpabet_ipa.py``).

Inline `` # `` comments on dictionary lines are removed before parsing.

By default writes three columns: ``text``, ``ipa``, and ``ambiguous`` (``true`` if
that spelling has more than one distinct IPA in CMUdict after conversion, else
``false``). ``train_g2p.py`` / ``g2p_tsv_io`` still load only the first two columns.

Use ``--no-ambiguous-column`` for a strict two-column file like other dataset scripts.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import urllib.request

from arpabet_ipa import arpabet_tokens_to_ipa
from g2p_tsv_io import write_pairs_tsv, write_tsv_table

# Current Sphinx repo ships ``cmudict.dict`` (older tarballs used ``cmudict-0.7b``).
DEFAULT_CMUDICT_URL = (
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
)

# Alternate pronunciations use a trailing (2), (3), … after the spelling.
_HEADWORD = re.compile(r"^([^(]+)(?:\(\d+\))?$")


def _parse_cmudict_line(line: str) -> tuple[str, list[str]] | None:
    line = line.strip()
    if not line or line.startswith(";;;"):
        return None
    if " #" in line:
        line = line.split(" # ", 1)[0].rstrip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    raw_word = parts[0]
    phones = parts[1:]
    m = _HEADWORD.match(raw_word)
    if not m:
        return None
    word = m.group(1).strip()
    if not word:
        return None
    return word, phones


def _url_open(url: str):
    """HTTPS with ``certifi`` CA bundle when available (helps macOS Python installs)."""
    try:
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=ctx)
    except ImportError:
        return urllib.request.urlopen(url)


def _fetch_cmudict(url: str, dest_path: str) -> None:
    d = os.path.dirname(dest_path)
    if d:
        os.makedirs(d, exist_ok=True)
    print(f"Downloading {url} ...")
    with _url_open(url) as resp, open(dest_path, "wb") as out:
        out.write(resp.read())


def _iter_cmudict_entries(path: str):
    with open(path, encoding="latin-1", errors="replace") as f:
        for line in f:
            parsed = _parse_cmudict_line(line)
            if parsed is None:
                continue
            yield parsed


def main() -> None:
    p = argparse.ArgumentParser(description="CMUdict → IPA TSV for train_g2p.py")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .tsv path")
    p.add_argument(
        "--cmudict-path",
        type=str,
        default=None,
        help="Local cmudict file (e.g. cmudict.dict). If omitted, uses --url or default URL",
    )
    p.add_argument(
        "--url",
        type=str,
        default=DEFAULT_CMUDICT_URL,
        help="URL to download when --cmudict-path is not set",
    )
    p.add_argument(
        "--download-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="If set with no --cmudict-path, save the download to this path and parse it",
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase the grapheme column (CMUdict mixes case; this folds for consistency)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows written after a full scan (0 = all). Ambiguity uses the entire parsed lexicon",
    )
    p.add_argument(
        "--skip-unknown-phonemes",
        action="store_true",
        help="Drop entries with unknown phone symbols instead of failing",
    )
    p.add_argument(
        "--no-ambiguous-column",
        action="store_true",
        help="Omit the ambiguous column; write only text and ipa",
    )
    args = p.parse_args()

    temp_path: str | None = None
    if args.cmudict_path:
        path = os.path.abspath(args.cmudict_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    elif args.download_cache:
        path = os.path.abspath(args.download_cache)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.isfile(path):
            _fetch_cmudict(args.url, path)
    else:
        fd, temp_path = tempfile.mkstemp(suffix="-cmudict.dict", text=True)
        os.close(fd)
        path = temp_path
        _fetch_cmudict(args.url, path)

    print(f"Reading {path} ...")
    pairs: list[tuple[str, str]] = []
    skipped_unknown = 0
    try:
        for word, phones in _iter_cmudict_entries(path):
            text = word.lower() if args.lowercase else word
            ipa = arpabet_tokens_to_ipa(phones, skip_unknown=args.skip_unknown_phonemes)
            if ipa is None:
                skipped_unknown += 1
                continue
            if not ipa:
                continue
            pairs.append((text, ipa))
    finally:
        if temp_path and os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    ipa_by_grapheme: dict[str, set[str]] = {}
    for text, ipa in pairs:
        ipa_by_grapheme.setdefault(text, set()).add(ipa)
    ambiguous_grapheme = {g: len(s) > 1 for g, s in ipa_by_grapheme.items()}

    if args.max_rows:
        pairs = pairs[: args.max_rows]

    comments = [
        "# source=cmudict (cmusphinx/cmudict)",
        f"# url={args.url}" if not args.cmudict_path else "# local file",
        "# ARPAbet phones → Unicode IPA (stress digits stripped; see arpabet_ipa.py)",
    ]
    if not args.no_ambiguous_column:
        comments.append(
            "# ambiguous=true: this spelling has multiple distinct IPAs in CMUdict "
            "(e.g. read/bass); false: only one pronunciation in the converted lexicon"
        )
        rows_3: list[tuple[str, str, str]] = [
            (t, ipa, "true" if ambiguous_grapheme[t] else "false") for t, ipa in pairs
        ]
        write_tsv_table(args.output, ["text", "ipa", "ambiguous"], rows_3, comments)
    else:
        write_pairs_tsv(args.output, pairs, comments)
    print(f"Wrote {len(pairs)} pairs to {args.output}")
    if skipped_unknown:
        print(f"Skipped {skipped_unknown} entries with unknown phoneme tokens.", file=sys.stderr)


if __name__ == "__main__":
    main()
