"""Labelling sentimen berbasis lexicon (InSet-style).

Metode sama dengan notebook `Codingan labelling.ipynb`:
    skor = sum(weight[token]) untuk token yang ada di lexicon
    skor > 0  -> positive
    skor < 0  -> negative
    skor == 0 -> neutral   (termasuk kasus tidak ada token yang cocok)

Tokenisasi di notebook hanya `str(text).lower().split()` (whitespace), BUKAN
NLTK — direplikasi persis agar label identik. Input `data_stemmed` diasumsikan
sudah berupa teks ber-spasi hasil preprocessing.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> dict[str, int]:
    """Muat full_lexicon.csv -> dict {word: weight}. Dicache.

    Catatan: kolom `number_of_words` (n-gram) diabaikan — scoring unigram-only,
    sama seperti notebook (entri multi-kata tak pernah cocok karena split spasi).
    """
    csv_path = path or str(config.LEXICON_PATH)
    df = pd.read_csv(csv_path)
    return dict(zip(df["word"].astype(str), df["weight"].astype(int)))


def score_text(text: str, lexicon: dict[str, int]) -> int:
    """Jumlahkan bobot lexicon dari token (split whitespace, lowercase)."""
    tokens = str(text).lower().split()
    return sum(lexicon.get(tok, 0) for tok in tokens)


def label_sentiment(text: str, lexicon: dict[str, int]) -> str:
    score = score_text(text, lexicon)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def label_dataframe(
    df: pd.DataFrame,
    text_col: str = "data_stemmed",
    lexicon_path: str | None = None,
) -> pd.DataFrame:
    """Tambahkan kolom `sentiment` ke DataFrame."""
    lexicon = load_lexicon(lexicon_path)
    out = df.copy()
    out["sentiment"] = out[text_col].fillna("").map(lambda t: label_sentiment(t, lexicon))
    return out
