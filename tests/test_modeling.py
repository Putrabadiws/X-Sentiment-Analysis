"""Tes modeling — train/save/load/predict pada dataset sintetis kecil."""
import pandas as pd
import pytest

from src import modeling


def _toy_df(n_per_class: int = 12) -> pd.DataFrame:
    """Dataset terpisah jelas per kelas agar SVM mudah belajar."""
    rows = []
    for _ in range(n_per_class):
        rows.append(("bagus mantap untung senang", "positive"))
        rows.append(("buruk rugi sedih kecewa", "negative"))
        rows.append(("biasa saja standar netral", "neutral"))
    return pd.DataFrame(rows, columns=["data_stemmed", "sentiment"])


def test_build_pipeline_has_expected_steps():
    pipe = modeling.build_pipeline()
    assert [name for name, _ in pipe.steps] == ["tfidf", "scaler", "smote", "svm"]


def test_train_model_returns_metrics():
    result = modeling.train_model(_toy_df())
    assert 0.0 <= result.accuracy <= 1.0
    assert set(result.labels) == {"positive", "negative", "neutral"}
    assert result.n_train > 0 and result.n_test > 0
    assert result.confusion.shape == (3, 3)


def test_train_model_learns_separable_data():
    # Data sangat terpisah -> akurasi harus tinggi.
    result = modeling.train_model(_toy_df(n_per_class=20))
    assert result.accuracy >= 0.8


def test_save_load_predict_roundtrip(tmp_path):
    result = modeling.train_model(_toy_df())
    path = tmp_path / "model.joblib"
    modeling.save_model(result.pipeline, path=str(path))
    assert path.exists()

    loaded = modeling.load_model(str(path))
    preds = modeling.predict(loaded, ["bagus mantap untung", "buruk rugi sedih"])
    assert len(preds) == 2
    assert all(p in {"positive", "negative", "neutral"} for p in preds)


def test_train_model_drops_nan_rows():
    df = _toy_df()
    df.loc[len(df)] = [None, None]
    result = modeling.train_model(df)  # tidak boleh error karena baris NaN
    assert result.n_train > 0


def test_predict_empty_list_returns_empty():
    result = modeling.train_model(_toy_df())
    assert modeling.predict(result.pipeline, []) == []


def test_train_model_handles_small_imbalanced_classes_without_nan_cv():
    # Kelas minoritas kecil -> SMOTE k_neighbors default (6) akan gagal & CV=nan.
    # Pastikan k_neighbors adaptif membuat CV tetap menghasilkan angka valid.
    import math

    rows = (
        [("bagus mantap untung", "positive")] * 30
        + [("buruk rugi sedih", "negative")] * 8
        + [("biasa saja standar", "neutral")] * 6
    )
    df = pd.DataFrame(rows, columns=["data_stemmed", "sentiment"])
    result = modeling.train_model(df)
    assert not math.isnan(result.cv_accuracy)
    assert 0.0 <= result.cv_accuracy <= 1.0
