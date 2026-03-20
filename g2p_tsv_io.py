"""Load and write tab-separated (text, ipa) pairs for G2P training."""

from __future__ import annotations

import csv
import glob
import os

HEADER = ["text", "ipa"]


def load_pairs_from_tsv_dir(data_dir: str) -> list[tuple[str, str]]:
    """Load and concatenate all ``*.tsv`` files in *data_dir* (sorted by path).

    Each file must contain optional ``#`` comment lines, then a header row
    ``text<TAB>ipa``, then one pair per row.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths = sorted(glob.glob(os.path.join(data_dir, "*.tsv")))
    if not paths:
        raise FileNotFoundError(f"No *.tsv files in {data_dir}")
    pairs: list[tuple[str, str]] = []
    for path in paths:
        pairs.extend(_load_pairs_from_tsv_file(path))
    return pairs


def _load_pairs_from_tsv_file(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header_seen = False
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if not header_seen:
                normalized = [c.strip() for c in row]
                if normalized != HEADER:
                    raise ValueError(f"{path}: expected header row {HEADER!r}, got {row!r}")
                header_seen = True
                continue
            if len(row) >= 2:
                pairs.append((row[0], row[1]))
    if not header_seen:
        raise ValueError(f"{path}: missing header row {HEADER!r}")
    return pairs


def write_pairs_tsv(path: str, pairs: list[tuple[str, str]], comments: list[str] | None = None) -> None:
    """Write TSV with optional comment lines (each line should start with ``#``)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in comments or []:
            line = line.rstrip("\n")
            if not line.startswith("#"):
                line = "# " + line.lstrip()
            f.write(line + "\n")
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        writer.writerows(pairs)
