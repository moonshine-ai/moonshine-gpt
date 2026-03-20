#!/usr/bin/env python3
"""
Build a TSV of (sentence, IPA) pairs from WikiText using espeak-ng.

Output format matches what ``train_g2p.py`` expects under ``--data-dir``:
comment lines (optional), header ``text<TAB>ipa``, then data rows.
"""

from __future__ import annotations

import argparse

from g2p_espeak_data import build_ipa_with_espeak, fetch_sentences_from_wikitext
from g2p_tsv_io import write_pairs_tsv


def main():
    p = argparse.ArgumentParser(description="WikiText sentences → IPA TSV (espeak-ng)")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .tsv path")
    p.add_argument("--max-sentences", type=int, default=20_000, help="Max sentences to sample")
    p.add_argument(
        "--dataset",
        type=str,
        default="wikitext-103-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="WikiText config on Hugging Face",
    )
    p.add_argument("--min-len", type=int, default=5, help="Min sentence length (chars)")
    p.add_argument("--max-len", type=int, default=256, help="Max sentence length (chars)")
    p.add_argument("--voice", type=str, default="en-us", help="espeak-ng voice")
    args = p.parse_args()

    print(f"Loading up to {args.max_sentences} lines from WikiText ({args.dataset})...")
    sentences = list(
        fetch_sentences_from_wikitext(
            args.max_sentences, config=args.dataset, min_len=args.min_len, max_len=args.max_len
        )
    )
    print(f"Got {len(sentences)} sentences. Running espeak-ng G2P...")
    pairs = build_ipa_with_espeak(sentences, voice=args.voice)
    comments = [
        "# source=wikitext-corpus-espeak",
        f"# dataset={args.dataset} max_sentences={args.max_sentences}",
    ]
    write_pairs_tsv(args.output, pairs, comments)
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
