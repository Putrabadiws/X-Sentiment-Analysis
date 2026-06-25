"""Tes preprocessing — fokus fungsi murni (tanpa file/Sastrawi/NLTK)."""
import pandas as pd

from src import config, preprocessing


# --- clean_text -------------------------------------------------------------
def test_clean_text_lowercases_and_strips_url_mention_hashtag():
    out = preprocessing.clean_text("Halo @budi #PPN cek http://t.co/abc DISINI")
    assert "@budi" not in out
    assert "#ppn" not in out
    assert "http" not in out
    assert out == out.lower()


def test_clean_text_keeps_percent_sign():
    # '%' sengaja dipertahankan (topik "PPN 12%").
    assert "%" in preprocessing.clean_text("PPN naik 12% tahun ini")


def test_clean_text_removes_other_punctuation():
    assert "," not in preprocessing.clean_text("pajak, naik!!!")


def test_clean_text_handles_non_string():
    # Input NaN/angka tidak boleh meledak.
    assert preprocessing.clean_text(123) == "123"


def test_clean_text_expands_repeated_word_digit():
    # Bug bawaan notebook yang sengaja dipertahankan: "ya3" -> "ya ya ya".
    assert preprocessing.clean_text("ya3") == "ya ya ya"


# --- tokenize ---------------------------------------------------------------
def test_tokenize_splits_on_word_boundaries():
    assert preprocessing.tokenize("pajak naik lagi") == ["pajak", "naik", "lagi"]


def test_tokenize_empty_string_returns_empty_list():
    assert preprocessing.tokenize("") == []


# --- normalize_slang --------------------------------------------------------
def test_normalize_slang_replaces_known_and_keeps_unknown():
    slang = {"gak": "tidak", "bgt": "banget"}
    assert preprocessing.normalize_slang(["gak", "suka", "bgt"], slang) == [
        "tidak",
        "suka",
        "banget",
    ]


def test_normalize_slang_empty_tokens():
    assert preprocessing.normalize_slang([], {"a": "b"}) == []


# --- remove_stopwords -------------------------------------------------------
def test_remove_stopwords_filters_members():
    stop = frozenset({"yang", "di"})
    assert preprocessing.remove_stopwords(["pajak", "yang", "di", "naik"], stop) == [
        "pajak",
        "naik",
    ]


# --- load_slang_dict (default + tambahan) -----------------------------------
def _write_slang(path, mapping):
    pd.DataFrame({"slang": list(mapping), "formal": list(mapping.values())}).to_csv(path, index=False)


def test_load_slang_default_only_when_no_extra(tmp_path, monkeypatch):
    default = tmp_path / "def.csv"
    _write_slang(default, {"udah": "sudah", "gua": "aku"})
    monkeypatch.setattr(config, "DEFAULT_SLANG_PATH", default)
    monkeypatch.setattr(config, "SLANG_PATH", tmp_path / "tak_ada.csv")  # tambahan tak ada
    preprocessing.load_slang_dict.cache_clear()
    assert preprocessing.load_slang_dict() == {"udah": "sudah", "gua": "aku"}
    preprocessing.load_slang_dict.cache_clear()


def test_load_slang_merges_extra_over_default(tmp_path, monkeypatch):
    default = tmp_path / "def.csv"
    _write_slang(default, {"gak": "engga", "udah": "sudah"})  # 'gak' akan dioverride
    extra = tmp_path / "extra.csv"
    _write_slang(extra, {"gak": "tidak", "bgt": "banget"})    # override + entri baru
    monkeypatch.setattr(config, "DEFAULT_SLANG_PATH", default)
    monkeypatch.setattr(config, "SLANG_PATH", extra)
    preprocessing.load_slang_dict.cache_clear()
    d = preprocessing.load_slang_dict()
    assert d["udah"] == "sudah"   # default tetap
    assert d["bgt"] == "banget"   # entri baru dari tambahan
    assert d["gak"] == "tidak"    # tambahan MENANG atas default ("engga")
    preprocessing.load_slang_dict.cache_clear()
