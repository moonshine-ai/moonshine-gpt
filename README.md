# G2P Transformer (English text → IPA)

Train a small Transformer that maps English text to IPA phonemes. **Training data** is a folder of tab-separated files (`text<TAB>ipa`). You can build TSVs with **WikiText → espeak-ng** (`build_corpus_tsv.py`), **HF word list → espeak-ng** (`build_dictionary_tsv.py`), **[librig2p-nostress](https://huggingface.co/datasets/flexthink/librig2p-nostress)** (`build_librig2p_nostress_tsv.py`), or **[CMUdict](https://github.com/cmusphinx/cmudict)** (`build_cmudict_tsv.py`); the last three use ARPAbet → Unicode IPA in `arpabet_ipa.py` (no espeak-ng).

## Requirements

- **Python 3.10+**
- **espeak-ng** must be installed on your system and available in `PATH` when running the dataset build scripts (the Python package is a wrapper around the binary).

  - macOS: `brew install espeak-ng`
  - Ubuntu/Debian: `sudo apt install espeak-ng`

## Install

```bash
pip install -r requirements.txt
```

## Building training TSVs

Corpus (WikiText lines + IPA):

```bash
python build_corpus_tsv.py -o data/corpus.tsv --max-sentences 20000
```

Dictionary words + IPA (example HF dataset):

```bash
python build_dictionary_tsv.py -o data/dictionary.tsv --max-words 10000
```

[LibriG2P (no stress)](https://huggingface.co/datasets/flexthink/librig2p-nostress) lexicon or sentence splits (downloads `dataset/<split>.json` via `huggingface_hub`):

```bash
python build_librig2p_nostress_tsv.py -o data/librig2p_lex.tsv --split lexicon_train
# optional: --json-path /path/to/lexicon_train.json  --max-rows 50000  --skip-unknown-phonemes
```

[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) (`cmudict.dict` from GitHub by default):

```bash
python build_cmudict_tsv.py -o data/cmudict.tsv
# optional: --cmudict-path ./cmudict.dict  --lowercase  --download-cache ~/.cache/cmudict.dict
# default TSV has a third column ``ambiguous`` (true/false); training still uses only text+ipa
# use --no-ambiguous-column for a two-column file
```

Phoneme tokens (`phn`) are treated as stress-free ARPAbet (trailing `0`/`1`/`2` stripped if present), mapped to IPA, and concatenated into one string per example so the target side stays character-level like the other TSV builders. IPA symbols and conventions can differ from espeak-ng; mixing this TSV with espeak-built TSVs in one `--data-dir` is allowed but may widen the target alphabet. CMUdict and LibriG2P exports share the same mapping in `arpabet_ipa.py`.

Put one or more `*.tsv` files in a directory (e.g. `data/`) and point training at that folder. Files are read in sorted path order and concatenated.

## Training

```bash
python train_g2p.py --data-dir data
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | *(required)* | Directory containing one or more `*.tsv` files (`text<TAB>ipa` after optional `#` comment lines) |
| `--shuffle-data` | off | Shuffle all pairs after loading (loader still shuffles batches) |
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
| `--resume` | — | Resume from checkpoint in DIR; use the same `--data-dir` as the original run |

Example (smaller/faster run):

```bash
python train_g2p.py --data-dir data --epochs 5 --batch-size 32 --out-dir out
```

Continue training from an existing checkpoint:

```bash
python train_g2p.py --data-dir data --resume checkpoints --epochs 5
```

Outputs in `--out-dir`:

- `g2p_transformer.pt` – model state dict
- `src_vocab.json` – source (grapheme) character vocabulary
- `tgt_vocab.json` – target (IPA) character vocabulary
- `config.json` – hyperparameters and `data_dir` (absolute path) for reference

## Model

- **Encoder–decoder** Transformer (PyTorch `nn.Transformer`).
- **Character-level** input and output: no subword tokenizer; both text and IPA are encoded as sequences of Unicode characters.
- Training is **teacher-forced** with cross-entropy on the decoder output.

## Data format

Each TSV may start with `#` comment lines, then a header row whose first two columns are `text` and `ipa` (additional columns are ignored by training):

```text
text	ipa
```

Then one training pair per row. The loader in `g2p_tsv_io.py` merges every `*.tsv` in `--data-dir`. CMUdict exports may add `ambiguous` (see above).

## Inference

After training, run:

```bash
python infer.py "Hello world"
# or
echo "Hello world" | python infer.py
```

Use `--checkpoint-dir` if you saved to a different directory (e.g. `python infer.py --checkpoint-dir out "Hello world"`).
