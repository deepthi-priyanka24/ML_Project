from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from backend.dataset import load_dataset


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@lru_cache(maxsize=1)
def train_model() -> tuple[Pipeline, dict[str, object]]:
    df = get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    word_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=1,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=12000,
        min_df=1,
    )

    model = Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("word", word_vectorizer, 0),
                        ("char", char_vectorizer, 0),
                    ],
                    remainder="drop",
                    sparse_threshold=0.3,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced"),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    model.fit(x_train.to_frame(), y_train)

    predictions = model.predict(x_test.to_frame())
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions, labels=["fake", "real"]).tolist(),
        "samples": int(len(df)),
        "labels": df["label"].value_counts().to_dict(),
    }
    return model, metrics


def predict_text(text: str) -> dict[str, object]:
    model, _ = train_model()
    probabilities = model.predict_proba(pd.DataFrame({"text": [text]}))[0]
    classes = list(model.classes_)
    fake_index = classes.index("fake")
    real_index = classes.index("real")
    best_index = int(probabilities.argmax())
    label = classes[best_index]
    confidence = float(probabilities[best_index])
    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": float(probabilities[fake_index]),
        "real_probability": float(probabilities[real_index]),
    }
