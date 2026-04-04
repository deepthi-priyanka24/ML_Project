from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "news_archive"


def _load_source(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    title = df.get("title", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    text = df.get("text", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    combined = (title + " " + text).str.strip()

    return pd.DataFrame(
        {
            "label": label,
            "text": combined,
            "title": title,
            "subject": df.get("subject", pd.Series(dtype=str)).fillna("").astype(str).str.strip(),
            "date": df.get("date", pd.Series(dtype=str)).fillna("").astype(str).str.strip(),
        }
    )


def load_dataset() -> pd.DataFrame:
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"

    fake_df = _load_source(fake_path, "fake")
    real_df = _load_source(real_path, "real")

    dataset = pd.concat([fake_df, real_df], ignore_index=True)
    dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset = dataset[dataset["text"].str.len() > 0].reset_index(drop=True)
    return dataset
