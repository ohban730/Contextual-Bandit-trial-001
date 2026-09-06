import numpy as np
import pandas as pd

from .genre import DEFAULT_WINDOW_DAYS


def _mean_vector(vectors: pd.Series) -> np.ndarray:
    return np.stack(vectors.to_numpy()).mean(axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_recent_interest_vector(
    history_df: pd.DataFrame, embedding_df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS, now=None
):
    """直近window_days日に見た動画タイトルの平均ベクトル（＝いまの興味の方向）を返す。"""
    now = now or history_df["watched_at"].max()
    recent = history_df[history_df["watched_at"] >= now - pd.Timedelta(days=window_days)]
    merged = recent.merge(embedding_df, on="video_id", how="inner")
    if merged.empty:
        return None
    return _mean_vector(merged["embedding"])


def add_semantic_scores(
    channel_df: pd.DataFrame, history_df: pd.DataFrame, embedding_df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS
) -> pd.DataFrame:
    """各チャンネルの動画タイトルの平均ベクトルが「最近の興味」とどれだけ意味的に近いかをスコア化する。"""
    df = channel_df.copy()
    recent_vector = build_recent_interest_vector(history_df, embedding_df, window_days)
    if recent_vector is None:
        df["semantic_score"] = 0.0
        return df

    merged = history_df.merge(embedding_df, on="video_id", how="inner")
    channel_vectors = merged.groupby("channel_name")["embedding"].apply(_mean_vector)

    similarity = df["channel_name"].map(
        lambda name: _cosine_similarity(channel_vectors[name], recent_vector) if name in channel_vectors else 0.0
    )
    df["semantic_score"] = (similarity + 1) / 2  # コサイン類似度[-1,1] を [0,1] にスケール
    return df
