#!/usr/bin/env python3
"""
Evaluate the trained G2P model on out-of-distribution text from Hugging Face.

Uses espeak-ng IPA output as ground truth and computes:
  - Phoneme Error Rate (PER): edit distance at the character/phoneme level
  - Word-level PER: edit distance on space-split IPA tokens
  - Exact match accuracy
"""

import argparse
import json
import os
import signal
import statistics
import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from g2p_lexicon import CmuDictLexicon
from train_g2p import CharVocab, G2PTransformer, predict, predict_lexicon_windowed

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    from espeakng import ESpeakNG
except ImportError:
    ESpeakNG = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Compute Levenshtein edit distance.

    Returns (distance, substitutions, insertions, deletions).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (dist, sub, ins, del)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                ins = dp[i][j - 1]
                dele = dp[i - 1][j]
                best = min(sub, ins, dele, key=lambda x: x[0])
                if best is sub:
                    dp[i][j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                elif best is ins:
                    dp[i][j] = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])
                else:
                    dp[i][j] = (dele[0] + 1, dele[1], dele[2], dele[3] + 1)
    return dp[n][m]


def phoneme_error_rate(ref_ipa: str, hyp_ipa: str) -> tuple[float, int, int]:
    """Character-level PER between two IPA strings.

    Returns (per, edit_dist, ref_len).
    """
    ref_chars = list(ref_ipa)
    hyp_chars = list(hyp_ipa)
    dist, _, _, _ = edit_distance(ref_chars, hyp_chars)
    ref_len = len(ref_chars) or 1
    return dist / ref_len, dist, len(ref_chars)


def word_error_rate(ref_ipa: str, hyp_ipa: str) -> tuple[float, int, int]:
    """Word-level error rate: split IPA on spaces and compute edit distance.

    Returns (wer, edit_dist, ref_len).
    """
    ref_words = ref_ipa.split()
    hyp_words = hyp_ipa.split()
    dist, _, _, _ = edit_distance(ref_words, hyp_words)
    ref_len = len(ref_words) or 1
    return dist / ref_len, dist, len(ref_words)


# ---------------------------------------------------------------------------
# OOD dataset fetchers
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "ag_news": {
        "path": "fancyzhx/ag_news",
        "split": "test",
        "text_field": "text",
        "description": "AG News (news articles – 4 topics)",
    },
    "emotion": {
        "path": "dair-ai/emotion",
        "split": "test",
        "text_field": "text",
        "description": "Emotion tweets (6 emotions)",
    },
    "imdb": {
        "path": "stanfordnlp/imdb",
        "split": "test",
        "text_field": "text",
        "description": "IMDB movie reviews",
    },
    "sst2": {
        "path": "stanfordnlp/sst2",
        "split": "validation",
        "text_field": "sentence",
        "description": "Stanford Sentiment Treebank v2",
    },
}


def fetch_ood_sentences(
    dataset_name: str,
    max_samples: int,
    min_len: int = 10,
    max_len: int = 250,
) -> list[str]:
    """Pull sentences from an OOD Hugging Face dataset."""
    if load_dataset is None:
        raise ImportError("Install 'datasets': pip install datasets")

    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    print(f"Loading OOD dataset: {cfg['description']} ({cfg['path']}, split={cfg['split']})")
    ds = load_dataset(cfg["path"], split=cfg["split"])
    text_field = cfg["text_field"]

    sentences = []
    for ex in ds:
        text = (ex.get(text_field) or "").strip()
        # Trim very long texts to max_len (take first max_len chars up to last space)
        if len(text) > max_len:
            text = text[:max_len].rsplit(" ", 1)[0]
        if not text or len(text) < min_len:
            continue
        sentences.append(text)
        if len(sentences) >= max_samples:
            break
    return sentences


# ---------------------------------------------------------------------------
# espeak ground-truth generation
# ---------------------------------------------------------------------------

def get_espeak_ipa(sentences: list[str], voice: str = "en-us") -> list[tuple[str, str]]:
    """Generate IPA ground truth for sentences using espeak-ng.

    Returns list of (text, ipa) pairs; skips failures.
    """
    if ESpeakNG is None:
        raise ImportError("Install 'py-espeak-ng' and ensure espeak-ng is in PATH")
    esng = ESpeakNG()
    esng.voice = voice

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    pairs = []
    for text in tqdm(sentences, desc="espeak-ng ground truth"):
        try:
            prev = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, 2.0)
            try:
                ipa = esng.g2p(text, ipa=2)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, prev)
            if ipa and isinstance(ipa, str) and ipa.strip():
                pairs.append((text, ipa.strip()))
        except _Timeout:
            continue
        except Exception:
            continue
    return pairs


# ---------------------------------------------------------------------------
# Bucketed analysis helpers
# ---------------------------------------------------------------------------

def length_bucket(n: int) -> str:
    if n <= 30:
        return "short (≤30 chars)"
    elif n <= 80:
        return "medium (31-80 chars)"
    elif n <= 150:
        return "long (81-150 chars)"
    else:
        return "very long (>150 chars)"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate G2P model on out-of-distribution text (espeak as ground truth)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory with g2p_transformer.pt, vocabs, config.json",
    )
    parser.add_argument(
        "--dataset", type=str, default="ag_news",
        choices=list(DATASET_CONFIGS.keys()),
        help="OOD dataset to evaluate on (default: ag_news)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max-len", type=int, default=250,
        help="Maximum source text length (chars)",
    )
    parser.add_argument(
        "--voice", type=str, default="en-us",
        help="espeak-ng voice for ground truth",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--show-errors", type=int, default=10,
        help="Number of worst-PER examples to display (0 to disable)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write JSON results",
    )
    parser.add_argument(
        "--lexicon-full-sentence",
        action="store_true",
        help="With lexicon: single forward pass on the full hybrid (disable per-unknown windows).",
    )
    parser.add_argument(
        "--unknown-context-chars",
        type=int,
        default=None,
        metavar="N",
        help="Hybrid chars before/after each <u>…</u> for windowed lexicon inference (default: config or 48).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    ckpt_dir = args.checkpoint_dir
    for name in ("g2p_transformer.pt", "src_vocab.json", "tgt_vocab.json", "config.json"):
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            print(f"Missing {path}. Train first.", file=sys.stderr)
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
    model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "g2p_transformer.pt"), map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt_dir}  (device={device})")

    lexicon: CmuDictLexicon | None = None
    if config.get("use_lexicon"):
        tsv = config.get("dict_tsv") or config.get("cmudict_tsv")
        if not tsv or not os.path.isfile(tsv):
            print(
                f"Checkpoint expects use_lexicon but TSV missing: {tsv!r}. Fix config.json or re-train.",
                file=sys.stderr,
            )
            sys.exit(1)
        lexicon = CmuDictLexicon.from_tsv(tsv)

    ctx_n = args.unknown_context_chars
    if ctx_n is None:
        ctx_n = int(config.get("unknown_context_chars", 48))

    # ------------------------------------------------------------------
    # 2. Fetch OOD data + espeak ground truth
    # ------------------------------------------------------------------
    sentences = fetch_ood_sentences(
        args.dataset, args.max_samples, min_len=10, max_len=args.max_len,
    )
    print(f"Fetched {len(sentences)} sentences from '{args.dataset}'")

    pairs = get_espeak_ipa(sentences, voice=args.voice)
    print(f"Generated espeak IPA for {len(pairs)}/{len(sentences)} sentences")

    if not pairs:
        print("No valid pairs – cannot evaluate.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Run model predictions and compute metrics
    # ------------------------------------------------------------------
    results = []
    total_per_dist = 0
    total_per_ref = 0
    total_wer_dist = 0
    total_wer_ref = 0
    exact_matches = 0
    latencies: list[float] = []  # seconds per sample

    bucket_stats = defaultdict(lambda: {"per_dist": 0, "per_ref": 0, "exact": 0, "n": 0, "latencies": []})

    for text, ref_ipa in tqdm(pairs, desc="Model inference + metrics"):
        t0 = time.perf_counter()
        if lexicon is not None and not args.lexicon_full_sentence:
            hyp_ipa = predict_lexicon_windowed(
                model,
                src_vocab,
                tgt_vocab,
                text,
                device,
                lexicon=lexicon,
                unknown_context_chars=ctx_n,
                max_src_len=config.get("max_src_len", 0),
            )
        else:
            hyp_ipa = predict(model, src_vocab, tgt_vocab, text, device, lexicon=lexicon)
        t1 = time.perf_counter()
        latency = t1 - t0
        latencies.append(latency)

        per, per_d, per_r = phoneme_error_rate(ref_ipa, hyp_ipa)
        wer, wer_d, wer_r = word_error_rate(ref_ipa, hyp_ipa)
        exact = int(ref_ipa == hyp_ipa)

        total_per_dist += per_d
        total_per_ref += per_r
        total_wer_dist += wer_d
        total_wer_ref += wer_r
        exact_matches += exact

        bkt = length_bucket(len(text))
        bucket_stats[bkt]["per_dist"] += per_d
        bucket_stats[bkt]["per_ref"] += per_r
        bucket_stats[bkt]["exact"] += exact
        bucket_stats[bkt]["n"] += 1
        bucket_stats[bkt]["latencies"].append(latency)

        results.append({
            "text": text,
            "ref_ipa": ref_ipa,
            "hyp_ipa": hyp_ipa,
            "per": per,
            "wer": wer,
            "exact": exact,
            "latency_s": latency,
        })

    n = len(results)
    avg_per = total_per_dist / total_per_ref if total_per_ref else 0.0
    avg_wer = total_wer_dist / total_wer_ref if total_wer_ref else 0.0
    exact_pct = exact_matches / n * 100 if n else 0.0

    # Latency statistics
    lat_mean = statistics.mean(latencies) if latencies else 0.0
    lat_median = statistics.median(latencies) if latencies else 0.0
    lat_p90 = sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0.0
    lat_p99 = sorted(latencies)[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0.0
    lat_min = min(latencies) if latencies else 0.0
    lat_max = max(latencies) if latencies else 0.0
    total_time = sum(latencies)
    throughput = n / total_time if total_time > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Print report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  G2P Evaluation — dataset: {args.dataset}  ({n} samples)")
    print("=" * 70)
    print(f"  Phoneme Error Rate (PER):   {avg_per:.4f}  ({avg_per * 100:.2f}%)")
    print(f"  Word Error Rate (WER):      {avg_wer:.4f}  ({avg_wer * 100:.2f}%)")
    print(f"  Exact Match Accuracy:       {exact_matches}/{n}  ({exact_pct:.2f}%)")
    print("-" * 70)
    print("  Inference Latency (per sample):")
    print(f"    Mean:    {lat_mean * 1000:>8.1f} ms")
    print(f"    Median:  {lat_median * 1000:>8.1f} ms")
    print(f"    P90:     {lat_p90 * 1000:>8.1f} ms")
    print(f"    P99:     {lat_p99 * 1000:>8.1f} ms")
    print(f"    Min:     {lat_min * 1000:>8.1f} ms")
    print(f"    Max:     {lat_max * 1000:>8.1f} ms")
    print(f"  Total inference time: {total_time:.2f}s  |  Throughput: {throughput:.1f} samples/s")
    print("=" * 70)

    # Per-bucket breakdown
    print("\n  Breakdown by input length:")
    print(f"  {'Bucket':<25s} {'N':>5s}  {'PER':>8s}  {'Exact':>8s}  {'Avg ms':>8s}  {'P90 ms':>8s}")
    print("  " + "-" * 72)
    for bkt in ["short (≤30 chars)", "medium (31-80 chars)", "long (81-150 chars)", "very long (>150 chars)"]:
        s = bucket_stats.get(bkt)
        if s is None or s["n"] == 0:
            continue
        bkt_per = s["per_dist"] / s["per_ref"] if s["per_ref"] else 0.0
        bkt_exact = s["exact"] / s["n"] * 100
        bkt_lats = sorted(s["latencies"])
        bkt_avg_ms = statistics.mean(bkt_lats) * 1000
        bkt_p90_ms = bkt_lats[int(len(bkt_lats) * 0.9)] * 1000
        print(f"  {bkt:<25s} {s['n']:>5d}  {bkt_per:>7.2%}  {bkt_exact:>7.2f}%  {bkt_avg_ms:>8.1f}  {bkt_p90_ms:>8.1f}")

    # Worst examples
    if args.show_errors and args.show_errors > 0:
        print(f"\n  Top-{args.show_errors} worst PER examples:")
        print("  " + "-" * 66)
        worst = sorted(results, key=lambda r: r["per"], reverse=True)[: args.show_errors]
        for i, r in enumerate(worst, 1):
            print(f"  [{i}] PER={r['per']:.3f}")
            print(f"      text: {r['text'][:100]}")
            print(f"      ref:  {r['ref_ipa'][:100]}")
            print(f"      hyp:  {r['hyp_ipa'][:100]}")

    # ------------------------------------------------------------------
    # 5. Optionally write JSON results
    # ------------------------------------------------------------------
    if args.output:
        summary = {
            "dataset": args.dataset,
            "n_samples": n,
            "per": avg_per,
            "wer": avg_wer,
            "exact_match": exact_pct,
            "latency": {
                "mean_ms": lat_mean * 1000,
                "median_ms": lat_median * 1000,
                "p90_ms": lat_p90 * 1000,
                "p99_ms": lat_p99 * 1000,
                "min_ms": lat_min * 1000,
                "max_ms": lat_max * 1000,
                "total_s": total_time,
                "throughput_samples_per_s": throughput,
            },
            "buckets": {
                bkt: {
                    "n": s["n"],
                    "per": s["per_dist"] / s["per_ref"] if s["per_ref"] else 0.0,
                    "exact_pct": s["exact"] / s["n"] * 100 if s["n"] else 0.0,
                    "latency_mean_ms": statistics.mean(s["latencies"]) * 1000 if s["latencies"] else 0.0,
                    "latency_p90_ms": sorted(s["latencies"])[int(len(s["latencies"]) * 0.9)] * 1000 if s["latencies"] else 0.0,
                }
                for bkt, s in bucket_stats.items()
            },
            "per_sample": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  Full results written to {args.output}")


if __name__ == "__main__":
    main()
