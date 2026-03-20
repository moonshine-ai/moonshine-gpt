#!/usr/bin/env python3
"""
Train a small Transformer for grapheme-to-phoneme (G2P): English text → IPA.

Training pairs come from ``*.tsv`` files under ``--data-dir`` (see ``build_corpus_tsv.py``
and ``build_dictionary_tsv.py``).

By default the encoder sees a *hybrid* source built from the lexicon TSV (``--dict-tsv``,
default ``<data-dir>/dict.tsv``): dictionary
entries with a single IPA are wrapped as ``<k>…</k>``; ambiguous / OOV tokens and
punctuation use ``<u>…</u>``. Use ``--no-lexicon`` for raw grapheme input only.
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from g2p_lexicon import CmuDictLexicon
from g2p_tsv_io import load_pairs_from_tsv_dir


# ------------------------------------------------------------------------------
# Vocabulary (character-level)
# ------------------------------------------------------------------------------

class CharVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self):
        self.char2idx = {c: i for i, c in enumerate(self.SPECIAL)}
        self.idx2char = {i: c for i, c in enumerate(self.SPECIAL)}

    def add(self, s: str):
        for c in s:
            if c not in self.char2idx:
                i = len(self.char2idx)
                self.char2idx[c] = i
                self.idx2char[i] = c

    def encode(self, s: str, add_sos_eos: bool = False) -> list[int]:
        out = []
        if add_sos_eos:
            out.append(self.SOS_IDX)
        for c in s:
            out.append(self.char2idx.get(c, self.UNK_IDX))
        if add_sos_eos:
            out.append(self.EOS_IDX)
        return out

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if strip_special and i in (self.PAD_IDX, self.SOS_IDX, self.EOS_IDX):
                continue
            chars.append(self.idx2char.get(i, self.SPECIAL[self.UNK_IDX]))
        return "".join(chars)

    def __len__(self):
        return len(self.char2idx)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": {int(k): v for k, v in self.idx2char.items()}}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v = cls()
        v.char2idx = data["char2idx"]
        v.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return v


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class G2PDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: CharVocab,
        tgt_vocab: CharVocab,
        max_src: int,
        max_tgt: int,
        lexicon: CmuDictLexicon | None = None,
    ):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.lexicon = lexicon

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, ipa = self.pairs[i]
        if self.lexicon is not None:
            text = self.lexicon.build_hybrid_source(text)
        src = self.src_vocab.encode(text, add_sos_eos=False)
        tgt = self.tgt_vocab.encode(ipa, add_sos_eos=True)
        src = src[: self.max_src]
        tgt = tgt[: self.max_tgt]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_g2p(batch):
    src_list, tgt_list = zip(*batch)
    src_lens = [s.size(0) for s in src_list]
    tgt_lens = [t.size(0) for t in tgt_list]
    src_pad = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=CharVocab.PAD_IDX, batch_first=True)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_list, padding_value=CharVocab.PAD_IDX, batch_first=True)
    return src_pad, tgt_pad, torch.tensor(src_lens, dtype=torch.long), torch.tensor(tgt_lens, dtype=torch.long)


# ------------------------------------------------------------------------------
# Model: small Transformer encoder-decoder
# ------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class G2PTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=CharVocab.PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=CharVocab.PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(max_src_len, max_tgt_len), dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        src_emb = self.pos_enc(self.src_embed(src))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_key_padding = (tgt == CharVocab.PAD_IDX)
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding,
        )
        return self.fc_out(out)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        src_emb = self.pos_enc(self.src_embed(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, memory: torch.Tensor, tgt: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None, tgt_mask: torch.Tensor | None = None):
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(out)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, pad_idx):
    model.train()
    total_loss = 0.0
    n = 0
    for src, tgt, src_lens, tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_key_padding_mask = (src == pad_idx)
        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in, src_key_padding_mask=src_key_padding_mask)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.tgt_vocab_size),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * tgt_out.numel()
        n += tgt_out.numel()
    return total_loss / n if n else 0.0


@torch.no_grad()
def predict(
    model: G2PTransformer,
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    text: str,
    device: torch.device,
    *,
    lexicon: CmuDictLexicon | None = None,
    max_decode_len: int = 200,
) -> str:
    """Run greedy decoding to convert text to IPA.

    If *lexicon* is set, *text* is converted with ``build_hybrid_source`` first
    (``<k>…</k>`` for unambiguous dictionary hits, ``<u>…</u>`` otherwise).
    """
    model.eval()
    if lexicon is not None:
        text = lexicon.build_hybrid_source(text)
    src = torch.tensor([src_vocab.encode(text)], dtype=torch.long, device=device)
    src_key_padding_mask = (src == CharVocab.PAD_IDX)
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
    ys = [tgt_vocab.SOS_IDX]
    for _ in range(max_decode_len - 1):
        tgt = torch.tensor([ys], dtype=torch.long, device=device)
        logits = model.decode(memory, tgt, memory_key_padding_mask=src_key_padding_mask)
        next_tok = logits[0, -1].argmax().item()
        if next_tok == tgt_vocab.EOS_IDX:
            break
        ys.append(next_tok)
    return tgt_vocab.decode(ys, strip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Train G2P Transformer (text → IPA)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing one or more *.tsv files (text<TAB>ipa); see build_corpus_tsv.py / build_dictionary_tsv.py",
    )
    parser.add_argument("--max-src-len", type=int, default=256, help="Max source (grapheme) length")
    parser.add_argument("--max-tgt-len", type=int, default=200, help="Max target (IPA) length")
    parser.add_argument("--d-model", type=int, default=256, help="Transformer dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Encoder/decoder layers")
    parser.add_argument("--dim-ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Save model and vocabs here (defaults to --resume path when resuming)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="DIR",
        help="Resume from checkpoint in DIR; pass the same --data-dir as the original run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-data",
        action="store_true",
        help="Shuffle concatenated TSV pairs before training (off by default; DataLoader still shuffles batches)",
    )
    parser.add_argument(
        "--dict-tsv",
        type=str,
        default=None,
        metavar="PATH",
        help="Pronouncing-dictionary TSV (text, ipa, …); unambiguous spellings → <k>…</k> in the encoder. "
        "Default: <data-dir>/dict.tsv",
    )
    parser.add_argument(
        "--no-lexicon",
        action="store_true",
        help="Disable dictionary hybrid source; encoder sees raw text (matches older checkpoints).",
    )
    args = parser.parse_args()

    if args.resume:
        args.out_dir = args.resume  # save back into the same checkpoint dir when resuming
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.abspath(args.data_dir)
    dict_tsv_path = os.path.abspath(args.dict_tsv) if args.dict_tsv else os.path.join(data_dir, "dict.tsv")
    pairs = load_pairs_from_tsv_dir(data_dir)
    if args.shuffle_data:
        random.shuffle(pairs)
    print(f"Loaded {len(pairs)} (text, IPA) pairs from {data_dir}")

    resume_dir = args.resume
    lexicon: CmuDictLexicon | None = None
    lexicon_tsv_path: str | None = None

    if resume_dir:
        config_path = os.path.join(resume_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Resume directory must contain config.json: {resume_dir}")
        with open(config_path) as f:
            ckpt_config = json.load(f)
        saved_data = ckpt_config.get("data_dir")
        if saved_data and os.path.abspath(saved_data) != data_dir:
            print(
                f"Warning: checkpoint data_dir {saved_data!r} differs from --data-dir {data_dir!r}; "
                "vocabs must still match the data you trained on."
            )
        if ckpt_config.get("use_lexicon"):
            tsv = ckpt_config.get("dict_tsv") or ckpt_config.get("cmudict_tsv")
            if not tsv or not os.path.isfile(tsv):
                raise FileNotFoundError(
                    f"Checkpoint was trained with use_lexicon but dict TSV missing: {tsv!r}"
                )
            lexicon_tsv_path = os.path.abspath(tsv)
            lexicon = CmuDictLexicon.from_tsv(lexicon_tsv_path)
            print(f"Lexicon hybrid source enabled ({lexicon_tsv_path})")
        if len(pairs) < 100:
            raise RuntimeError("Too few training pairs in --data-dir (need at least 100).")
        src_vocab = CharVocab.load(os.path.join(resume_dir, "src_vocab.json"))
        tgt_vocab = CharVocab.load(os.path.join(resume_dir, "tgt_vocab.json"))
        model = G2PTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            max_src_len=ckpt_config["max_src_len"],
            max_tgt_len=ckpt_config["max_tgt_len"],
            d_model=ckpt_config["d_model"],
            nhead=ckpt_config["nhead"],
            num_encoder_layers=ckpt_config["num_layers"],
            num_decoder_layers=ckpt_config["num_layers"],
            dim_feedforward=ckpt_config["dim_feedforward"],
            dropout=ckpt_config["dropout"],
        ).to(device)
        state_path = os.path.join(resume_dir, "g2p_transformer.pt")
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
        print(f"Resuming from {resume_dir}. Loaded model from {state_path}")
        args.max_src_len = ckpt_config["max_src_len"]
        args.max_tgt_len = ckpt_config["max_tgt_len"]
        args.d_model = ckpt_config["d_model"]
        args.nhead = ckpt_config["nhead"]
        args.num_layers = ckpt_config["num_layers"]
        args.dim_ff = ckpt_config["dim_feedforward"]
        args.dropout = ckpt_config["dropout"]
    else:
        if len(pairs) < 100:
            raise RuntimeError("Too few training pairs in --data-dir (need at least 100).")
        if not args.no_lexicon:
            tsv = dict_tsv_path
            if not os.path.isfile(tsv):
                raise FileNotFoundError(
                    f"Lexicon TSV not found: {tsv!r}. Build it (e.g. build_cmudict_tsv.py -o …/dict.tsv) or pass --no-lexicon."
                )
            lexicon_tsv_path = tsv
            lexicon = CmuDictLexicon.from_tsv(tsv)
            print(f"Lexicon hybrid source enabled ({lexicon_tsv_path})")
        src_vocab = CharVocab()
        tgt_vocab = CharVocab()
        for text, ipa in pairs:
            src_line = lexicon.build_hybrid_source(text) if lexicon is not None else text
            src_vocab.add(src_line)
            tgt_vocab.add(ipa)
        print(f"Source vocab size: {len(src_vocab)}, target vocab size: {len(tgt_vocab)}")
        model = G2PTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            dropout=args.dropout,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        ).to(device)

    # Dataset and loader (shared by resume and fresh)
    dataset = G2PDataset(pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len, lexicon=lexicon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_g2p, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5) Train
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, CharVocab.PAD_IDX)
        print(f"Epoch {epoch}/{args.epochs}  loss = {loss:.4f}")

    # 6) Save
    torch.save(model.state_dict(), os.path.join(args.out_dir, "g2p_transformer.pt"))
    src_vocab.save(os.path.join(args.out_dir, "src_vocab.json"))
    tgt_vocab.save(os.path.join(args.out_dir, "tgt_vocab.json"))
    config = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_ff,
        "dropout": args.dropout,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "data_dir": data_dir,
        "use_lexicon": lexicon is not None,
        "dict_tsv": lexicon_tsv_path,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f)
    print(f"Saved model and vocabs to {args.out_dir}")


if __name__ == "__main__":
    main()
