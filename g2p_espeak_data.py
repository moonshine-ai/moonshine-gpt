"""Fetch text from Hugging Face and run espeak-ng G2P (used by dataset build scripts)."""

from __future__ import annotations

import signal

from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    from espeakng import ESpeakNG
except ImportError:
    ESpeakNG = None

# Common column names for word in HF datasets (try in order)
WORD_COLUMN_CANDIDATES = ("Word", "word", "token", "text", "words")


def fetch_sentences_from_wikitext(
    max_sentences: int, config: str = "wikitext-103-raw-v1", min_len: int = 5, max_len: int = 300
):
    """Yield English text lines from WikiText (raw)."""
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


def fetch_words_from_dict(
    dict_dataset: str, max_words: int, min_len: int = 1, max_len: int = 64, dict_config: str | None = None
) -> list[str]:
    """Load single words from a Hugging Face dataset (e.g. Maximax67/English-Valid-Words)."""
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
    out: list[str] = []
    for ex in ds:
        w = (ex.get(word_col) or ex.get(word_col.lower()) or "").strip()
        if not w or len(w) < min_len or len(w) > max_len:
            continue
        if not w.isalpha():
            continue
        out.append(w)
        n += 1
        if n >= max_words:
            break
    return out


def build_ipa_with_espeak(sentences: list[str], voice: str = "en-us") -> list[tuple[str, str]]:
    """Return list of (text, ipa) using espeak-ng. Skips failures."""
    if ESpeakNG is None:
        raise ImportError("Install 'py-espeak-ng' and ensure espeak-ng binary is in PATH")
    esng = ESpeakNG()
    esng.voice = voice
    out: list[tuple[str, str]] = []

    class _G2PTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _G2PTimeout()

    for text in tqdm(sentences, desc="G2P (espeak-ng)"):
        try:
            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, 1.0)
            try:
                ipa = esng.g2p(text, ipa=2)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, prev_handler)
            if ipa and isinstance(ipa, str) and ipa.strip():
                ipa = ipa.strip()
                out.append((text, ipa))
        except _G2PTimeout:
            continue
        except Exception:
            continue
    return out
