"""Konfigurasi terpusat: path file & parameter pipeline.

Semua path Colab (`/content/drive/...`) dari notebook asli diganti jadi path
relatif ke folder project, biar aplikasi bisa jalan di mesin lokal/manapun.
"""
from __future__ import annotations

import os
from pathlib import Path

# Root project = parent dari folder src/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- File pendukung ---
# Kamus slang: default bawaan SELALU ada (ikut repo), upload = TAMBAHAN opsional
# yang di-merge di atas default (tambahan menang). Lihat preprocessing.load_slang_dict.
DEFAULT_SLANG_PATH = BASE_DIR / "src" / "defaults" / "kamus_slang_default.csv"  # bawaan
SLANG_PATH = DATA_DIR / "kamus_slang.csv"  # tambahan opsional (hasil upload, kolom slang/formal)
# Lexicon masih WAJIB disediakan user (kolom `word`, `weight`, `number_of_words` opsional).
LEXICON_PATH = DATA_DIR / "full_lexicon.csv"

# --- Artefak model tersimpan ---
MODEL_PATH = MODELS_DIR / "svm_sentiment.joblib"  # berisi vectorizer + classifier

# --- Parameter labelling ---
# Skor > 0 -> positive, < 0 -> negative, == 0 -> neutral (sama seperti notebook).
LABELS = ("positive", "neutral", "negative")

# --- Parameter modeling ---
# Best params dari GridSearch di notebook asli (di notebook tidak pernah dipakai,
# di sini kita pakai sebagai default — alasan ada di modeling.py).
SVM_BEST_PARAMS = {"kernel": "sigmoid", "C": 10, "coef0": 1, "gamma": 0.0005}
RANDOM_STATE = 42
TEST_SIZE = 0.2  # held-out test set; sisanya untuk train (+CV)

# --- Stopword tweak dari notebook preprocessing ---
STOPWORD_REMOVE = {"hari", "apa", "ada", "lama"}  # dikeluarkan dari daftar stopword
STOPWORD_ADD = {"aja", "halo", "eh"}              # ditambahkan jadi stopword

# --- Crawling (opsional, via twikit) ---
# Cookie sesi X diambil dari env var, JANGAN hardcode di sini.
# Butuh keduanya: auth_token (token sesi) + ct0 (CSRF token). Lihat crawling.py.
X_AUTH_TOKEN = os.environ.get("X_AUTH_TOKEN", "")
X_CT0 = os.environ.get("X_CT0", "")


def ensure_dirs() -> None:
    """Pastikan folder data/ & models/ ada."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
