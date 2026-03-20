#!/usr/bin/env python3
"""Run the trained G2P model to convert English text to IPA."""

import argparse
import json
import os
import sys

import torch

# Import from train script (same dir)
from g2p_lexicon import CmuDictLexicon
from train_g2p import CharVocab, G2PTransformer, predict


def main():
    parser = argparse.ArgumentParser(description="Convert text to IPA using trained G2P model")
    parser.add_argument("text", nargs="?", default=None, help="Input text (or read from stdin)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory with g2p_transformer.pt and vocabs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu); default auto")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    for name in ("g2p_transformer.pt", "src_vocab.json", "tgt_vocab.json", "config.json"):
        if not os.path.isfile(os.path.join(ckpt_dir, name)):
            print(f"Missing {ckpt_dir}/{name}. Train first with: python train_g2p.py --out-dir {ckpt_dir}", file=sys.stderr)
            sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src_vocab = CharVocab.load(os.path.join(ckpt_dir, "src_vocab.json"))
    tgt_vocab = CharVocab.load(os.path.join(ckpt_dir, "tgt_vocab.json"))
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)

    model = G2PTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        max_src_len=config["max_src_len"],
        max_tgt_len=config["max_tgt_len"],
        **{k: config[k] for k in ("d_model", "nhead", "dim_feedforward", "dropout")},
        num_encoder_layers=config["num_layers"],
        num_decoder_layers=config["num_layers"],
    )
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "g2p_transformer.pt"), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    lexicon: CmuDictLexicon | None = None
    if config.get("use_lexicon"):
        tsv = config.get("dict_tsv") or config.get("cmudict_tsv")
        if not tsv or not os.path.isfile(tsv):
            print(
                f"Checkpoint expects use_lexicon but TSV missing: {tsv!r}. Re-train or fix config.json.",
                file=sys.stderr,
            )
            sys.exit(1)
        lexicon = CmuDictLexicon.from_tsv(tsv)

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("No input text.", file=sys.stderr)
        sys.exit(1)

    ipa = predict(model, src_vocab, tgt_vocab, text, device, lexicon=lexicon)
    print(ipa)


if __name__ == "__main__":
    main()
