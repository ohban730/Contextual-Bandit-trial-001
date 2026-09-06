import pandas as pd

DEFAULT_WINDOW_DAYS = 14

# YouTubeの標準動画カテゴリID（表示用、スコア計算には使わない）
CATEGORY_NAMES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
}


def _dominant_category(categories: pd.Series):
    counts = categories.dropna().value_counts()
    return counts.index[0] if not counts.empty else None


def build_recent_genre_profile(
    history_df: pd.DataFrame, metadata_df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS, now=None
) -> pd.Series:
    """直近window_days日に見たカテゴリの正規化済み頻度分布（0〜1）を返す。"""
    now = now or history_df["watched_at"].max()
    recent = history_df[history_df["watched_at"] >= now - pd.Timedelta(days=window_days)]
    merged = recent.merge(metadata_df[["video_id", "category_id"]], on="video_id", how="inner")
    counts = merged["category_id"].dropna().value_counts()
    if counts.empty:
        return pd.Series(dtype=float)
    return counts / counts.max()


def add_genre_scores(
    channel_df: pd.DataFrame, history_df: pd.DataFrame, metadata_df: pd.DataFrame, window_days: int = DEFAULT_WINDOW_DAYS
) -> pd.DataFrame:
    """各チャンネルの主要ジャンルが「最近よく見ているジャンル」とどれだけ一致するかをスコア化する。"""
    recent_profile = build_recent_genre_profile(history_df, metadata_df, window_days)

    merged = history_df.merge(metadata_df[["video_id", "category_id"]], on="video_id", how="left")
    channel_dominant = merged.groupby("channel_name")["category_id"].apply(_dominant_category)

    df = channel_df.copy()
    df["dominant_category"] = df["channel_name"].map(channel_dominant)
    df["dominant_category_name"] = df["dominant_category"].map(CATEGORY_NAMES).fillna(df["dominant_category"])
    df["genre_score"] = df["dominant_category"].map(recent_profile).fillna(0.0)
    return df
