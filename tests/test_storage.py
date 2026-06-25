"""Tes penyimpanan file pendukung hasil upload via UI."""
import pandas as pd
import pytest

from src import config, labelling, preprocessing, storage


def _slang_csv(slang: str = "gak", formal: str = "tidak") -> bytes:
    return pd.DataFrame({"slang": [slang], "formal": [formal]}).to_csv(index=False).encode()


def _lexicon_csv() -> bytes:
    return (
        pd.DataFrame({"word": ["bagus"], "weight": [5], "number_of_words": [1]})
        .to_csv(index=False)
        .encode()
    )


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Arahkan path file pendukung ke tmp_path supaya tak menyentuh data/ asli.

    DEFAULT_SLANG_PATH juga diarahkan ke default kecil terkontrol (bukan default
    bawaan repo 1.5k entri) supaya assertion merge tak terganggu isi default asli.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "SLANG_PATH", tmp_path / "kamus_slang.csv")  # tambahan
    monkeypatch.setattr(config, "LEXICON_PATH", tmp_path / "full_lexicon.csv")
    default_slang = tmp_path / "default_slang.csv"
    # default sengaja tak memuat 'gak'/'bgt' supaya nilai tambahan-lah yang teruji.
    pd.DataFrame({"slang": ["xx"], "formal": ["yy"]}).to_csv(default_slang, index=False)
    monkeypatch.setattr(config, "DEFAULT_SLANG_PATH", default_slang)
    preprocessing.load_slang_dict.cache_clear()
    labelling.load_lexicon.cache_clear()
    yield
    preprocessing.load_slang_dict.cache_clear()
    labelling.load_lexicon.cache_clear()


# --- happy path -------------------------------------------------------------
def test_save_slang_writes_file_and_is_loadable(isolated_data):
    dest = storage.save_support_file(_slang_csv(), "slang")
    assert dest.exists()
    # load_slang_dict() = default (xx→yy) + tambahan tersimpan (gak→tidak).
    merged = preprocessing.load_slang_dict()
    assert merged["gak"] == "tidak"
    assert merged["xx"] == "yy"  # default bawaan ikut


def test_save_lexicon_writes_file_and_is_loadable(isolated_data):
    dest = storage.save_support_file(_lexicon_csv(), "lexicon")
    assert dest.exists()
    assert labelling.load_lexicon(str(dest))["bagus"] == 5


# --- cache invalidation (inti fitur) ----------------------------------------
def test_resave_invalidates_cache(isolated_data):
    storage.save_support_file(_slang_csv("gak", "tidak"), "slang")
    # load default-arg (None) -> baca config.SLANG_PATH, ter-cache
    assert preprocessing.load_slang_dict()["gak"] == "tidak"
    storage.save_support_file(_slang_csv("bgt", "banget"), "slang")
    # tanpa cache_clear di save_support_file, baris ini masih lihat data lama
    assert preprocessing.load_slang_dict()["bgt"] == "banget"


# --- error / edge cases -----------------------------------------------------
def test_unknown_kind_raises(isolated_data):
    with pytest.raises(ValueError):
        storage.save_support_file(_slang_csv(), "bukan_jenis")


def test_missing_columns_raises(isolated_data):
    bad = pd.DataFrame({"foo": [1]}).to_csv(index=False).encode()
    with pytest.raises(ValueError):
        storage.save_support_file(bad, "slang")


def test_invalid_csv_does_not_overwrite_existing(isolated_data):
    # File valid disimpan dulu, lalu upload CSV rusak harus gagal TANPA menimpa.
    storage.save_support_file(_slang_csv("gak", "tidak"), "slang")
    bad = pd.DataFrame({"foo": [1]}).to_csv(index=False).encode()
    with pytest.raises(ValueError):
        storage.save_support_file(bad, "slang")
    assert preprocessing.load_slang_dict()["gak"] == "tidak"
