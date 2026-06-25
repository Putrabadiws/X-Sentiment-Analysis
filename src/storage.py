"""Simpan file pendukung yang di-upload via UI ke path kanonik di data/.

Kenapa modul terpisah (bukan inline di app.py): logika validasi kolom + clear
cache loader bisa diuji tanpa menjalankan Streamlit; app.py jadi lapisan UI tipis.

Invariant yang dijaga: setelah file di-upload, ia tersimpan ke path yang dibaca
pipeline (config.SLANG_PATH / config.LEXICON_PATH) DAN cache loader terkait
di-clear, supaya pembacaan berikutnya pakai file baru — bukan hasil cache lama.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from . import config, labelling, preprocessing

# Kolom minimum yang harus ada per jenis file (number_of_words opsional untuk lexicon).
REQUIRED_COLUMNS = {
    "slang": {"slang", "formal"},
    "lexicon": {"word", "weight"},
}

# Loader ber-cache yang harus di-clear setelah file ditimpa, per jenis file.
_LOADERS = {
    "slang": preprocessing.load_slang_dict,
    "lexicon": labelling.load_lexicon,
}


def dest_path(kind: str) -> Path:
    """Path kanonik tujuan simpan untuk `kind` ∈ {"slang", "lexicon"}.

    Sengaja resolve dari config tiap panggil (bukan snapshot di module-level)
    supaya path bisa di-monkeypatch saat tes tanpa menulis ke data/ asli.
    """
    if kind == "slang":
        return config.SLANG_PATH
    if kind == "lexicon":
        return config.LEXICON_PATH
    raise ValueError(f"kind tak dikenal: {kind!r}")


def validate_columns(df: pd.DataFrame, kind: str) -> None:
    """Raise ValueError kalau kolom wajib untuk `kind` tidak lengkap."""
    if kind not in REQUIRED_COLUMNS:
        raise ValueError(f"kind tak dikenal: {kind!r}")
    missing = REQUIRED_COLUMNS[kind] - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib hilang untuk '{kind}': {sorted(missing)}. "
            f"Butuh {sorted(REQUIRED_COLUMNS[kind])}, dapat {list(df.columns)}."
        )


def save_support_file(raw: bytes, kind: str) -> Path:
    """Validasi lalu simpan bytes CSV ke path kanonik; clear cache loader.

    Validasi dilakukan SEBELUM menimpa file lama — kalau CSV tak valid, file
    yang sudah ada tidak ikut rusak. Mengembalikan path tujuan.
    """
    dest = dest_path(kind)  # juga memvalidasi `kind`
    df = pd.read_csv(io.BytesIO(raw))
    validate_columns(df, kind)

    config.ensure_dirs()
    dest.write_bytes(raw)
    _LOADERS[kind].cache_clear()  # buang hasil baca lama; baca ulang saat dipakai
    return dest
