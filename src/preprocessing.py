"""Preprocessing teks tweet Bahasa Indonesia.

Urutan tahap (sama persis dengan notebook `Codingan_Preprocessing_Data.ipynb`):
    1. case folding          -> lowercase
    2. cleaning              -> buang url, mention, hashtag, emoji, tanda baca (kecuali %)
    3. expand repeated words -> "data4" -> "data data data data"
    4. tokenizing            -> RegexpTokenizer(r'\\w+')
    5. slang normalization   -> ganti token slang -> formal (kamus_slang.csv)
    6. stopword removal      -> NLTK Indonesian + tweak custom
    7. stemming              -> Sastrawi

Perbedaan vs notebook (disengaja, lihat komentar di tiap titik):
    - Jalan in-memory (DataFrame), bukan 7x baca-tulis CSV.
    - Stemming dilakukan pada teks yang sudah di-`join`, bukan pada string-repr
      list token (notebook asli men-stem `"['pajak','naik']"` dan mengandalkan
      Sastrawi membuang kurung/kutip — rapuh).
"""
from __future__ import annotations

import re
import string
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

# Tanda baca tetap menyisakan '%' agar token seperti "12%" / "66%" tidak rusak
# (topik dataset = "PPN 12%"). Sama seperti notebook asli.
PUNCT_NO_PERCENT = string.punctuation.replace("%", "")

# Pola emoji — verbatim dari notebook.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "☀-⭕"
    "‍"
    "️"
    "]+",
    flags=re.UNICODE,
)

_TOKEN_RE = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# Tahap 1-3: cleaning + case folding
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lowercase + buang url/mention/hashtag/emoji/tanda baca + expand repeat."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)        # url
    text = re.sub(r"@\w+", "", text)           # mention
    text = re.sub(r"#\w+", "", text)           # hashtag
    text = EMOJI_PATTERN.sub("", text)          # emoji
    text = text.translate(str.maketrans("", "", PUNCT_NO_PERCENT))  # tanda baca
    # expand "kata5" -> "kata kata kata kata kata" (verbatim dari notebook)
    text = re.sub(
        r"([A-Za-z]+)(\d+)\b",
        lambda m: " ".join([m.group(1)] * int(m.group(2))),
        text,
    )
    return text.strip()


# --------------------------------------------------------------------------- #
# Tahap 4: tokenizing
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """RegexpTokenizer(r'\\w+') — pecah jadi token alfanumerik."""
    return _TOKEN_RE.findall(str(text))


# --------------------------------------------------------------------------- #
# Tahap 5: slang normalization
# --------------------------------------------------------------------------- #
def _read_slang_csv(path: str) -> dict[str, str]:
    """Baca satu CSV slang -> dict {slang: formal}."""
    df = pd.read_csv(path)
    return dict(zip(df["slang"].astype(str), df["formal"].astype(str)))


@lru_cache(maxsize=1)
def load_slang_dict(default_path: str | None = None, extra_path: str | None = None) -> dict[str, str]:
    """Kamus slang gabungan: default bawaan + (opsional) file tambahan hasil upload.

    Default (DEFAULT_SLANG_PATH) selalu dimuat. Kalau file tambahan (SLANG_PATH) ada,
    entri-nya di-merge DI ATAS default — tambahan MENANG saat key bentrok (boleh
    override + nambah entri). Tanpa tambahan → default saja.

    Dicache (maxsize=1); storage.save_support_file meng-clear cache saat tambahan
    di-upload supaya pembacaan berikut pakai gabungan terbaru.
    """
    dpath = default_path or str(config.DEFAULT_SLANG_PATH)
    epath = extra_path or str(config.SLANG_PATH)
    merged = _read_slang_csv(dpath)
    if Path(epath).exists():
        merged.update(_read_slang_csv(epath))  # tambahan override default
    return merged


def normalize_slang(tokens: list[str], slang_dict: dict[str, str]) -> list[str]:
    return [slang_dict.get(tok, tok) for tok in tokens]


# --------------------------------------------------------------------------- #
# Tahap 6: stopword removal
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    """NLTK Indonesian stopwords + tweak custom (config.STOPWORD_REMOVE/ADD)."""
    import nltk
    from nltk.corpus import stopwords

    try:
        words = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords")
        words = set(stopwords.words("indonesian"))

    words.difference_update(config.STOPWORD_REMOVE)
    words.update(config.STOPWORD_ADD)
    return frozenset(words)


def remove_stopwords(tokens: list[str], stop_words: frozenset[str]) -> list[str]:
    return [tok for tok in tokens if tok not in stop_words]


# --------------------------------------------------------------------------- #
# Tahap 7: stemming
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    return StemmerFactory().create_stemmer()


def stem_tokens(tokens: list[str]) -> str:
    """Stem teks gabungan token. Mengembalikan string ber-spasi.

    Beda dari notebook: di sini token di-`join` jadi kalimat dulu baru di-stem,
    bukan men-stem string-repr list. Sastrawi paling andal pada teks normal.
    """
    stemmer = _get_stemmer()
    return stemmer.stem(" ".join(tokens))


# --------------------------------------------------------------------------- #
# Pipeline lengkap
# --------------------------------------------------------------------------- #
def preprocess_text(text: str, slang_dict: dict[str, str], stop_words: frozenset[str]) -> str:
    """Jalankan semua tahap untuk satu teks -> string ter-stem (siap TF-IDF)."""
    tokens = tokenize(clean_text(text))
    tokens = normalize_slang(tokens, slang_dict)
    tokens = remove_stopwords(tokens, stop_words)
    return stem_tokens(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "full_text",
    slang_extra_path: str | None = None,
) -> pd.DataFrame:
    """Tambahkan kolom hasil tiap tahap ke DataFrame.

    `slang_extra_path`: override path file slang tambahan (default: config.SLANG_PATH).
    Mengembalikan df dengan kolom baru:
        data_cleaned, data_token, data_slang, data_stopwords, data_stemmed
    """
    slang_dict = load_slang_dict(extra_path=slang_extra_path)
    stop_words = load_stopwords()

    out = df.copy()
    out["data_cleaned"] = out[text_col].fillna("").map(clean_text)
    out["data_token"] = out["data_cleaned"].map(tokenize)
    out["data_slang"] = out["data_token"].map(lambda t: normalize_slang(t, slang_dict))
    out["data_stopwords"] = out["data_slang"].map(lambda t: remove_stopwords(t, stop_words))
    out["data_stemmed"] = out["data_stopwords"].map(stem_tokens)
    return out
