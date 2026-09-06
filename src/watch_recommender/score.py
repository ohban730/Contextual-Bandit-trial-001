from datetime import datetime, timezone

import pandas as pd

from .config import ScoringWeights


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def score_channels(channel_df: pd.DataFrame, weights: ScoringWeights = ScoringWeights(), now=None) -> pd.DataFrame:
    """「久しぶり度」と「好きだった度」からチャンネルにスコアを付ける（ジャンル関連度はPhase Bで追加）。"""
    now = now or datetime.now(timezone.utc)
    df = channel_df.copy()
    df["days_since_last_watch"] = (now - df["last_watched"]).dt.total_seconds() / 86400

    df["recency_score"] = _normalize(df["days_since_last_watch"])
    df["frequency_score"] = _normalize(df["watch_count"])

    weighted_sum = weights.recency * df["recency_score"] + weights.frequency * df["frequency_score"]
    active_weight = weights.recency + weights.frequency

    if "genre_score" in df.columns:  # Phase B: YouTube Data APIでのメタデータ補完済みの場合のみ有効
        weighted_sum = weighted_sum + weights.genre * df["genre_score"]
        active_weight += weights.genre

    if "semantic_score" in df.columns:  # Phase B: タイトル埋め込みの類似度が計算済みの場合のみ有効
        weighted_sum = weighted_sum + weights.semantic * df["semantic_score"]
        active_weight += weights.semantic

    df["score"] = weighted_sum / active_weight

    return df.sort_values("score", ascending=False).reset_index(drop=True)
