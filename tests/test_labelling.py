"""Tes labelling lexicon-based."""
import pandas as pd

from src import labelling

LEX = {"bagus": 5, "buruk": -4, "naik": -1, "untung": 3}


# --- score_text -------------------------------------------------------------
def test_score_text_sums_weights():
    assert labelling.score_text("bagus untung", LEX) == 8


def test_score_text_ignores_unknown_words():
    assert labelling.score_text("kata asing random", LEX) == 0


def test_score_text_is_case_insensitive():
    assert labelling.score_text("BAGUS", LEX) == 5


# --- label_sentiment --------------------------------------------------------
def test_label_positive():
    assert labelling.label_sentiment("bagus untung", LEX) == "positive"


def test_label_negative():
    assert labelling.label_sentiment("buruk naik", LEX) == "negative"


def test_label_neutral_on_zero_score():
    # bagus(5) + buruk(-4) + naik(-1) = 0 -> neutral
    assert labelling.label_sentiment("bagus buruk naik", LEX) == "neutral"


def test_label_neutral_on_no_match():
    assert labelling.label_sentiment("tidak ada di kamus", LEX) == "neutral"


# --- label_dataframe (file) -------------------------------------------------
def test_label_dataframe_adds_sentiment_column(tmp_path):
    lex_path = tmp_path / "lex.csv"
    pd.DataFrame(
        {"word": ["bagus", "buruk"], "weight": [5, -4], "number_of_words": [1, 1]}
    ).to_csv(lex_path, index=False)
    labelling.load_lexicon.cache_clear()

    df = pd.DataFrame({"data_stemmed": ["bagus", "buruk", "kosong"]})
    out = labelling.label_dataframe(df, lexicon_path=str(lex_path))
    assert list(out["sentiment"]) == ["positive", "negative", "neutral"]
    labelling.load_lexicon.cache_clear()
