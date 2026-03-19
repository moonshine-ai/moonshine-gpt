#!/usr/bin/env python3
"""
Train a small Transformer for grapheme-to-phoneme (G2P): English text → IPA.
Uses WikiText-2 as the text corpus and py-espeak-ng for IPA ground truth.
"""

import argparse
import csv
import json
import math
import os
import random
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional imports for data
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    from espeakng import ESpeakNG
except ImportError:
    ESpeakNG = None


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
# Data: WikiText + espeak-ng IPA
# ------------------------------------------------------------------------------

def fetch_sentences_from_wikitext(max_sentences: int, config: str = "wikitext-103-raw-v1", min_len: int = 5, max_len: int = 300):
    """Yield English text lines from WikiText (raw). Use config 'wikitext-103-raw-v1' for ~1.8M lines, 'wikitext-2-raw-v1' for ~37k."""
    if load_dataset is None:
        raise ImportError("Install 'datasets': pip install datasets")
    ds = load_dataset("wikitext", config, split="train")
    n = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if not text or len(text) < min_len or len(text) > max_len:
            continue
        if text.startswith("=") and text.endswith("="):
            continue
        yield text
        n += 1
        if n >= max_sentences:
            break


def build_ipa_with_espeak(sentences: list[str], voice: str = "en-us") -> list[tuple[str, str]]:
    """Return list of (text, ipa) using espeak-ng. Skips failures."""
    if ESpeakNG is None:
        raise ImportError("Install 'py-espeak-ng' and ensure espeak-ng binary is in PATH")
    esng = ESpeakNG()
    esng.voice = voice
    out = []
    for text in tqdm(sentences, desc="G2P (espeak-ng)"):
        try:
            ipa = esng.g2p(text, ipa=2)
            if ipa and isinstance(ipa, str) and ipa.strip():
                ipa = ipa.strip()
                out.append((text, ipa))
        except Exception:
            continue
    return out


# ------------------------------------------------------------------------------
# Data: dictionary word list (single words)
# ------------------------------------------------------------------------------

# Common column names for word in HF datasets (try in order)
WORD_COLUMN_CANDIDATES = ("Word", "word", "token", "text", "words")


def fetch_words_from_dict(
    dict_dataset: str, max_words: int, min_len: int = 1, max_len: int = 64, dict_config: str | None = None
) -> list[str]:
    """Yield single words from a Hugging Face dataset (e.g. Maximax67/English-Valid-Words)."""
    if load_dataset is None:
        raise ImportError("Install 'datasets': pip install datasets")
    if dict_config:
        ds = load_dataset(dict_dataset, dict_config, split="train")
    else:
        ds = load_dataset(dict_dataset, split="train")
    if hasattr(ds, "column_names") and ds.column_names:
        cols = ds.column_names
        word_col = next((c for c in WORD_COLUMN_CANDIDATES if c in cols), cols[0])
    else:
        word_col = "Word"
    n = 0
    out = []
    for ex in ds:
        w = (ex.get(word_col) or ex.get(word_col.lower()) or "").strip()
        if not w or len(w) < min_len or len(w) > max_len:
            continue
        if not w.isalpha():  # skip if any non-letter
            continue
        out.append(w)
        n += 1
        if n >= max_words:
            break
    return out


# ------------------------------------------------------------------------------
# Cache: TSV with metadata (dataset name + n_sentences) for reuse
# ------------------------------------------------------------------------------

def _cache_basename(dataset: str, max_sentences: int) -> str:
    """Safe filename component from dataset config and max_sentences."""
    safe = re.sub(r"[^\w\-]", "_", dataset)
    return f"g2p_cache_{safe}_{max_sentences}.tsv"


def load_cached_pairs(cache_path: str, dataset: str, max_sentences: int) -> list[tuple[str, str]] | None:
    """Load (text, ipa) pairs from cache if it exists and matches dataset + max_sentences."""
    if not os.path.isfile(cache_path):
        return None
    pairs = []
    with open(cache_path, "r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        if not first.startswith("# dataset=") or " max_sentences=" not in first:
            return None
        # Parse "# dataset=wikitext-103-raw-v1 max_sentences=20000"
        parts = first[2:].strip().split()
        meta = {}
        for p in parts:
            k, v = p.split("=", 1)
            meta[k] = v
        if meta.get("dataset") != dataset or meta.get("max_sentences") != str(max_sentences):
            return None
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        if header != ["text", "ipa"]:
            return None
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
    return pairs if pairs else None


def save_cached_pairs(cache_path: str, dataset: str, max_sentences: int, pairs: list[tuple[str, str]]) -> None:
    """Write (text, ipa) pairs to TSV with metadata header."""
    d = os.path.dirname(cache_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# dataset={dataset} max_sentences={max_sentences}\n")
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "ipa"])
        writer.writerows(pairs)


def _cache_basename_dict(dict_dataset: str, max_words: int, dict_config: str | None = None) -> str:
    """Safe filename for dictionary G2P cache."""
    safe = re.sub(r"[^\w\-]", "_", dict_dataset.replace("/", "_"))
    if dict_config:
        safe_cfg = re.sub(r"[^\w\-]", "_", dict_config)
        return f"g2p_cache_dict_{safe}_{safe_cfg}_{max_words}.tsv"
    return f"g2p_cache_dict_{safe}_{max_words}.tsv"


def load_cached_pairs_dict(
    cache_path: str, dict_dataset: str, max_words: int, dict_config: str | None = None
) -> list[tuple[str, str]] | None:
    """Load (word, ipa) pairs from dict cache if it exists and matches."""
    if not os.path.isfile(cache_path):
        return None
    pairs = []
    with open(cache_path, "r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        if not first.startswith("# source=dict") or " dataset=" not in first or " max_words=" not in first:
            return None
        parts = first[2:].strip().split()
        meta = {}
        for p in parts:
            k, v = p.split("=", 1)
            meta[k] = v
        if meta.get("dataset") != dict_dataset or meta.get("max_words") != str(max_words):
            return None
        if dict_config is not None and meta.get("config") != dict_config:
            return None
        if dict_config is None and meta.get("config") is not None:
            return None
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        if header != ["text", "ipa"]:
            return None
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
    return pairs if pairs else None


def save_cached_pairs_dict(
    cache_path: str, dict_dataset: str, max_words: int, pairs: list[tuple[str, str]], dict_config: str | None = None
) -> None:
    """Write dict (word, ipa) pairs to TSV with metadata header."""
    d = os.path.dirname(cache_path)
    if d:
        os.makedirs(d, exist_ok=True)
    meta = f"# source=dict dataset={dict_dataset} max_words={max_words}"
    if dict_config:
        meta += f" config={dict_config}"
    meta += "\n"
    with open(cache_path, "w", encoding="utf-8", newline="") as f:
        f.write(meta)
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "ipa"])
        writer.writerows(pairs)


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class G2PDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], src_vocab: CharVocab, tgt_vocab: CharVocab, max_src: int, max_tgt: int):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, ipa = self.pairs[i]
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
def predict(model: G2PTransformer, src_vocab: CharVocab, tgt_vocab: CharVocab, text: str, device: torch.device, max_decode_len: int = 200) -> str:
    """Run greedy decoding to convert text to IPA."""
    model.eval()
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
    parser.add_argument("--max-sentences", type=int, default=20_000, help="Max training sentences to use")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1", choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"], help="WikiText config: 103 has ~1.8M train lines, 2 has ~37k")
    parser.add_argument("--dict-dataset", type=str, default="Maximax67/English-Valid-Words", metavar="HF_NAME", help="Add single-word pairs from a HF dataset (e.g. Maximax67/English-Valid-Words)")
    parser.add_argument("--dict-config", type=str, default="sorted_by_frequency", metavar="CONFIG", help="Config name for --dict-dataset if required (e.g. sorted_by_frequency for Maximax67/English-Valid-Words)")
    parser.add_argument("--max-dict-words", type=int, default=10_000, help="Max single words to add when --dict-dataset is set")
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
    parser.add_argument("--resume", type=str, default=None, metavar="DIR", help="Resume training from checkpoint in DIR (loads model, vocabs, config; uses cached data from original run)")
    parser.add_argument("--cache-dir", type=str, default=".", help="Directory for espeak-ng TSV cache (default: current dir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voice", type=str, default="en-us", help="espeak-ng voice for IPA")
    args = parser.parse_args()

    if args.resume:
        args.out_dir = args.resume  # save back into the same checkpoint dir when resuming
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_dir = args.resume
    if resume_dir:
        # Load checkpoint: config, vocabs, model state; then load cached pairs
        config_path = os.path.join(resume_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Resume directory must contain config.json: {resume_dir}")
        with open(config_path) as f:
            ckpt_config = json.load(f)
        dataset_name = ckpt_config.get("dataset") or args.dataset
        ckpt_max_sentences = ckpt_config.get("max_sentences")
        if ckpt_max_sentences is not None:
            ckpt_max_sentences = int(ckpt_max_sentences)
        else:
            ckpt_max_sentences = args.max_sentences
        cache_path = os.path.join(args.cache_dir, _cache_basename(dataset_name, ckpt_max_sentences))
        pairs = load_cached_pairs(cache_path, dataset_name, ckpt_max_sentences)
        if pairs is None:
            raise FileNotFoundError(
                f"Resume requires cached G2P data. Not found: {cache_path} "
                f"(dataset={dataset_name}, max_sentences={ckpt_max_sentences}). Run without --resume first to create the cache."
            )
        print(f"Resuming from {resume_dir}. Using cached data: {len(pairs)} pairs.")
        dict_dataset_ckpt = ckpt_config.get("dict_dataset")
        dict_max_words_ckpt = ckpt_config.get("max_dict_words")
        dict_config_ckpt = ckpt_config.get("dict_config")
        if dict_dataset_ckpt and dict_max_words_ckpt is not None:
            dict_max_words_ckpt = int(dict_max_words_ckpt)
            dict_cache_path = os.path.join(
                args.cache_dir, _cache_basename_dict(dict_dataset_ckpt, dict_max_words_ckpt, dict_config_ckpt)
            )
            dict_pairs = load_cached_pairs_dict(
                dict_cache_path, dict_dataset_ckpt, dict_max_words_ckpt, dict_config_ckpt
            )
            if dict_pairs is not None:
                pairs = pairs + dict_pairs
                random.shuffle(pairs)
                print(f"Added {len(dict_pairs)} dict-word pairs (total {len(pairs)}).")
            else:
                print(f"Warning: checkpoint used dict_dataset={dict_dataset_ckpt} but cache not found at {dict_cache_path}; resuming without dict words.")
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
        print(f"Loaded model from {state_path}")
        # Use checkpoint config for everything so saved config matches the model
        args.dataset = dataset_name
        args.max_sentences = ckpt_max_sentences
        args.max_src_len = ckpt_config["max_src_len"]
        args.max_tgt_len = ckpt_config["max_tgt_len"]
        args.d_model = ckpt_config["d_model"]
        args.nhead = ckpt_config["nhead"]
        args.num_layers = ckpt_config["num_layers"]
        args.dim_ff = ckpt_config["dim_feedforward"]
        args.dropout = ckpt_config["dropout"]
        args.dict_dataset = ckpt_config.get("dict_dataset")
        args.max_dict_words = int(ckpt_config["max_dict_words"]) if ckpt_config.get("max_dict_words") is not None else getattr(args, "max_dict_words", 10_000)
        args.dict_config = ckpt_config.get("dict_config")
    else:
        # 1) Load sentences and get IPA (from cache or espeak-ng)
        cache_path = os.path.join(args.cache_dir, _cache_basename(args.dataset, args.max_sentences))
        pairs = load_cached_pairs(cache_path, args.dataset, args.max_sentences)
        if pairs is not None:
            print(f"Using cached G2P data from {cache_path} ({len(pairs)} pairs).")
        else:
            print(f"Loading sentences from WikiText ({args.dataset})...")
            sentences = list(fetch_sentences_from_wikitext(args.max_sentences, config=args.dataset, min_len=5, max_len=args.max_src_len))
            print(f"Got {len(sentences)} sentences. Generating IPA with espeak-ng...")
            pairs = build_ipa_with_espeak(sentences, voice=args.voice)
            print(f"Collected {len(pairs)} (text, IPA) pairs. Caching to {cache_path}")
            save_cached_pairs(cache_path, args.dataset, args.max_sentences, pairs)

        if len(pairs) < 100:
            raise RuntimeError("Too few pairs; install espeak-ng and ensure py-espeak-ng works.")

        # Optional: add single-word pairs from a dictionary dataset
        if args.dict_dataset:
            dict_cache_path = os.path.join(
                args.cache_dir, _cache_basename_dict(args.dict_dataset, args.max_dict_words, getattr(args, "dict_config", None))
            )
            dict_pairs = load_cached_pairs_dict(
                dict_cache_path, args.dict_dataset, args.max_dict_words, getattr(args, "dict_config", None)
            )
            if dict_pairs is not None:
                print(f"Using cached dict words from {dict_cache_path} ({len(dict_pairs)} pairs).")
            else:
                print(f"Loading words from dictionary dataset {args.dict_dataset}...")
                words = fetch_words_from_dict(
                    args.dict_dataset,
                    args.max_dict_words,
                    min_len=1,
                    max_len=min(64, args.max_src_len),
                    dict_config=getattr(args, "dict_config", None),
                )
                print(f"Got {len(words)} words. Generating IPA with espeak-ng...")
                dict_pairs = build_ipa_with_espeak(words, voice=args.voice)
                print(f"Collected {len(dict_pairs)} (word, IPA) pairs. Caching to {dict_cache_path}")
                save_cached_pairs_dict(
                    dict_cache_path, args.dict_dataset, args.max_dict_words, dict_pairs, getattr(args, "dict_config", None)
                )
            pairs = pairs + dict_pairs
            random.shuffle(pairs)

        # 2) Build vocabs
        src_vocab = CharVocab()
        tgt_vocab = CharVocab()
        for text, ipa in pairs:
            src_vocab.add(text)
            tgt_vocab.add(ipa)
        print(f"Source vocab size: {len(src_vocab)}, target vocab size: {len(tgt_vocab)}")

        # 3) Model (dataset/loader built below for both branches)
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
    dataset = G2PDataset(pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)
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
        "dataset": args.dataset,
        "max_sentences": args.max_sentences,
    }
    if getattr(args, "dict_dataset", None):
        config["dict_dataset"] = args.dict_dataset
        config["max_dict_words"] = getattr(args, "max_dict_words", 10_000)
        if getattr(args, "dict_config", None):
            config["dict_config"] = args.dict_config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f)
    print(f"Saved model and vocabs to {args.out_dir}")


if __name__ == "__main__":
    main()
