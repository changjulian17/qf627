"""General helper utilities for the project."""
from __future__ import annotations
import os
import pandas as pd
import pickle


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def save_df_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    df.to_csv(path, index=True)


def save_pickle(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


__all__ = ["ensure_dir", "save_df_csv", "save_pickle", "load_pickle"]
