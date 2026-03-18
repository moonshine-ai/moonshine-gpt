# G2P Transformer (English text → IPA)

Train a small Transformer that maps English text to IPA phonemes. Training uses **WikiText** (by default WikiText-103, ~1.8M lines) as the text corpus and **espeak-ng** (via the `py-espeak-ng` Python package) to generate ground-truth IPA for each sentence.

## Requirements

- **Python 3.10+**
- **espeak-ng** must be installed on your system and available in `PATH` (the Python package is a wrapper around the binary).

  - macOS: `brew install espeak-ng`
  - Ubuntu/Debian: `sudo apt install espeak-ng`

## Install

```bash
pip install -r requirements.txt
```

## Training

Default run (20k sentences, 10 epochs, small transformer):

```bash
python train_g2p.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-sentences` | 20000 | Max training sentences to use |
| `--dataset` | wikitext-103-raw-v1 | WikiText config: `wikitext-103-raw-v1` (~1.8M lines) or `wikitext-2-raw-v1` (~37k) |
| `--max-src-len` | 256 | Max grapheme sequence length |
| `--max-tgt-len` | 200 | Max IPA sequence length |
| `--d-model` | 256 | Transformer dimension |
| `--nhead` | 8 | Number of attention heads |
| `--num-layers` | 3 | Encoder and decoder layers |
| `--dim-ff` | 1024 | Feedforward hidden size |
| `--dropout` | 0.1 | Dropout |
| `--batch-size` | 64 | Batch size |
| `--epochs` | 10 | Training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--out-dir` | checkpoints | Where to save model and vocabs |
| `--cache-dir` | . | Directory for espeak-ng TSV cache (reused when dataset + max_sentences match) |
| `--voice` | en-us | espeak-ng voice used for IPA |

Example (smaller/faster run):

```bash
python train_g2p.py --max-sentences 5000 --epochs 5 --batch-size 32 --out-dir out
```

Outputs in `--out-dir`:

- `g2p_transformer.pt` – model state dict
- `src_vocab.json` – source (grapheme) character vocabulary
- `tgt_vocab.json` – target (IPA) character vocabulary

## Model

- **Encoder–decoder** Transformer (PyTorch `nn.Transformer`).
- **Character-level** input and output: no subword tokenizer; both text and IPA are encoded as sequences of Unicode characters.
- Training is **teacher-forced** with cross-entropy on the decoder output.

## Data

- **Source**: Hugging Face `wikitext` dataset. Default config is `wikitext-103-raw-v1` (WikiText-103, ~1.8M train lines); use `--dataset wikitext-2-raw-v1` for the smaller WikiText-2 (~37k lines). Only the train split is used; lines are filtered by length (5–256 chars) and section headers are skipped.
- **Target**: Each selected line is passed to `espeakng.ESpeakNG().g2p(text, ipa=2)` to obtain IPA. Pairs where `g2p` fails or returns empty are dropped.
- **Cache**: The (text, IPA) pairs are written to a TSV file under `--cache-dir` (default: current directory), with a first-line header encoding the data source and `max_sentences` (e.g. `# dataset=wikitext-103-raw-v1 max_sentences=20000`). Re-runs with the same `--dataset` and `--max-sentences` reuse this cache and skip espeak-ng. Cache files are named like `g2p_cache_wikitext-103-raw-v1_20000.tsv`.

## Inference

After training, run:

```bash
python infer.py "Hello world"
# or
echo "Hello world" | python infer.py
```

Use `--checkpoint-dir` if you saved to a different directory (e.g. `python infer.py --checkpoint-dir out "Hello world"`).
