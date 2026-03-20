#!/usr/bin/env python3
"""
Build a TSV of (word, IPA) pairs from a Hugging Face word-list dataset using espeak-ng.

Output format matches what ``train_g2p.py`` expects under ``--data-dir`` (combine with
corpus TSVs in the same folder, or train on this file alone).
"""

from __future__ import annotations

import argparse

from g2p_espeak_data import build_ipa_with_espeak, fetch_words_from_dict
from g2p_tsv_io import write_pairs_tsv


def main():
    p = argparse.ArgumentParser(description="HF word list → IPA TSV (espeak-ng)")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .tsv path")
    p.add_argument(
        "--dict-dataset",
        type=str,
        default="Maximax67/English-Valid-Words",
        metavar="HF_NAME",
        help="Hugging Face dataset id",
    )
    p.add_argument(
        "--dict-config",
        type=str,
        default="sorted_by_frequency",
        metavar="CONFIG",
        help="Dataset config name if required (use '' to load default config only)",
    )
    p.add_argument("--max-words", type=int, default=10_000, help="Max words to sample")
    p.add_argument("--min-len", type=int, default=1, help="Min word length (letters)")
    p.add_argument("--max-len", type=int, default=64, help="Max word length (letters)")
    p.add_argument("--voice", type=str, default="en-us", help="espeak-ng voice")
    args = p.parse_args()

    dict_config = args.dict_config if args.dict_config else None
    print(f"Loading up to {args.max_words} words from {args.dict_dataset!r}...")
    words = fetch_words_from_dict(
        args.dict_dataset,
        args.max_words,
        min_len=args.min_len,
        max_len=args.max_len,
        dict_config=dict_config,
    )
    print(f"Got {len(words)} words. Running espeak-ng G2P...")
    pairs = build_ipa_with_espeak(words, voice=args.voice)
    comments = ["# source=dict-espeak", f"# dataset={args.dict_dataset} max_words={args.max_words}"]
    if dict_config:
        comments.append(f"# config={dict_config}")
    write_pairs_tsv(args.output, pairs, comments)
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
