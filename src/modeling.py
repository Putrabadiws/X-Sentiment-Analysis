"""TF-IDF + SVM (SMOTE) — training, evaluasi, simpan/muat, prediksi.

Pipeline: TfidfVectorizer -> StandardScaler(with_mean=False) -> SMOTE -> SVC(sigmoid)

Perbaikan vs notebook `Pembobotan Kata(TF-IDF,SVM,DLL)` (disengaja):
    1. SMOTE DITARUH DI DALAM imblearn.Pipeline, bukan di-`fit_resample` sebelum
       split K-Fold. Di notebook, SMOTE dijalankan dulu baru di-CV -> sampel
       sintetis bocor ke fold validasi -> akurasi 92% over-optimistic. Dengan
       SMOTE di dalam pipeline, resample hanya terjadi pada fold train tiap CV.
    2. Param terbaik hasil GridSearch (C=10, coef0=1, gamma=0.0005) DIPAKAI di
       model final. Di notebook hasil GridSearch dicetak tapi tak pernah dipakai.
    3. Ada held-out test set sungguhan (di notebook x_test/x_val dibuat tapi tak
       pernah dievaluasi).
    4. Vectorizer + model DISIMPAN (joblib) -> UI bisa prediksi tanpa training ulang.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


@dataclass
class TrainResult:
    """Hasil training: model + metrik evaluasi pada held-out test set."""

    pipeline: ImbPipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy: float
    labels: list[str]
    confusion: np.ndarray
    report: str
    n_train: int = 0
    n_test: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)


def build_pipeline(svm_params: dict | None = None, smote_k: int = 5) -> ImbPipeline:
    """Bangun pipeline TF-IDF -> Scaler -> SMOTE -> SVC.

    smote_k = jumlah tetangga SMOTE. Wajib < jumlah sampel kelas terkecil di
    setiap fold CV, kalau tidak SMOTE error (default-nya 6).
    """
    params = dict(svm_params or config.SVM_BEST_PARAMS)
    return ImbPipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            # with_mean=False wajib untuk matriks sparse TF-IDF.
            ("scaler", StandardScaler(with_mean=False)),
            ("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=smote_k)),
            ("svm", SVC(random_state=config.RANDOM_STATE, **params)),
        ]
    )


def train_model(
    df: pd.DataFrame,
    text_col: str = "data_stemmed",
    label_col: str = "sentiment",
    svm_params: dict | None = None,
    cv_folds: int = 5,
) -> TrainResult:
    """Latih model & evaluasi pada held-out test set.

    SMOTE di dalam pipeline -> tidak ada kebocoran data ke test/fold validasi.
    """
    data = df[[text_col, label_col]].dropna()
    X = data[text_col].astype(str)
    y = data[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Jumlah fold aman: tak boleh melebihi jumlah sampel kelas terkecil.
    min_class = int(y_train.value_counts().min())
    safe_folds = max(2, min(cv_folds, min_class))

    # k_neighbors SMOTE harus < jumlah sampel kelas terkecil DI DALAM fold train.
    # Tiap fold train menyisakan ~(safe_folds-1)/safe_folds dari tiap kelas.
    min_class_in_fold = min_class - -(-min_class // safe_folds)  # = min_class - ceil(min/folds)
    smote_k = max(1, min(5, min_class_in_fold - 1))

    pipeline = build_pipeline(svm_params, smote_k=smote_k)

    # CV pada data train saja (SMOTE re-resample tiap fold di dalam pipeline).
    skf = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")

    # Fit final pada seluruh train, evaluasi pada held-out test.
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = sorted(y.unique())
    return TrainResult(
        pipeline=pipeline,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        cv_accuracy=float(cv_scores.mean()),
        labels=labels,
        confusion=confusion_matrix(y_test, y_pred, labels=labels),
        report=classification_report(y_test, y_pred, labels=labels, zero_division=0),
        n_train=len(X_train),
        n_test=len(X_test),
        class_distribution=y.value_counts().to_dict(),
    )


def save_model(pipeline: ImbPipeline, path: str | None = None) -> str:
    """Simpan pipeline (sudah termasuk vectorizer) ke joblib."""
    config.ensure_dirs()
    out_path = str(path or config.MODEL_PATH)
    joblib.dump(pipeline, out_path)
    return out_path


def load_model(path: str | None = None) -> ImbPipeline:
    return joblib.load(str(path or config.MODEL_PATH))


def predict(pipeline: ImbPipeline, texts: list[str]) -> list[str]:
    """Prediksi label untuk daftar teks yang SUDAH dipreprocessing (ter-stem).

    Guard list kosong: TfidfVectorizer.transform([]) error, jadi pintas di sini.
    """
    if not texts:
        return []
    return list(pipeline.predict(texts))
