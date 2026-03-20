#!/usr/bin/env python3
"""
Export [flexthink/librig2p-nostress](https://huggingface.co/datasets/flexthink/librig2p-nostress)
to training TSVs: ``char`` (graphemes) and IPA built by mapping each ``phn`` token
(ARPAbet-style ASCII, stress digits omitted) to Unicode IPA and concatenating.

Uses ``huggingface_hub`` to download ``dataset/<split>.json`` (the Hub dataset
script is not loadable on recent ``datasets`` versions without remote code).

Output matches ``g2p_tsv_io.write_pairs_tsv`` / ``train_g2p.py --data-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from arpabet_ipa import arpabet_tokens_to_ipa
from g2p_tsv_io import write_pairs_tsv


def _resolve_json_path(split: str, json_path: str | None) -> str:
    if json_path:
        return os.path.abspath(json_path)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Install huggingface_hub (e.g. pip install huggingface_hub) or pass --json-path"
        ) from e
    filename = f"dataset/{split}.json"
    return hf_hub_download(
        repo_id="flexthink/librig2p-nostress",
        filename=filename,
        repo_type="dataset",
    )


def _iter_librig2p_records(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    for _item_id, item in data.items():
        yield item


def main() -> None:
    splits = (
        "lexicon_train",
        "lexicon_valid",
        "lexicon_test",
        "sentence_train",
        "sentence_valid",
        "sentence_test",
    )
    p = argparse.ArgumentParser(description="librig2p-nostress → IPA TSV for train_g2p.py")
    p.add_argument("--output", "-o", type=str, required=True, help="Output .tsv path")
    p.add_argument(
        "--split",
        type=str,
        default="lexicon_train",
        choices=splits,
        help="Which split JSON to export (downloaded from the Hub unless --json-path)",
    )
    p.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Local path to <split>.json (skips download; must match --split)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all)")
    p.add_argument(
        "--skip-unknown-phonemes",
        action="store_true",
        help="Drop rows that contain an unknown phn token instead of failing",
    )
    args = p.parse_args()

    path = _resolve_json_path(args.split, args.json_path)
    print(f"Reading {path} ...")

    pairs: list[tuple[str, str]] = []
    skipped_unknown = 0
    n = 0
    for item in _iter_librig2p_records(path):
        if args.max_rows and n >= args.max_rows:
            break
        text = (item.get("char") or "").strip()
        phn = item.get("phn")
        if not text or not isinstance(phn, list):
            continue
        ipa = arpabet_tokens_to_ipa([str(x) for x in phn], skip_unknown=args.skip_unknown_phonemes)
        if ipa is None:
            skipped_unknown += 1
            continue
        if not ipa:
            continue
        pairs.append((text, ipa))
        n += 1

    comments = [
        "# source=librig2p-nostress",
        f"# split={args.split}",
        "# phn=ARPAbet-nostress → Unicode IPA (concatenated per token; arpabet_ipa.py)",
    ]
    write_pairs_tsv(args.output, pairs, comments)
    print(f"Wrote {len(pairs)} pairs to {args.output}")
    if skipped_unknown:
        print(f"Skipped {skipped_unknown} rows with unknown phoneme tokens.", file=sys.stderr)


if __name__ == "__main__":
    main()
