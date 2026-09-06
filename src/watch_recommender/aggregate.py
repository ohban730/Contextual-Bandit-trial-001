import pandas as pd


def aggregate_by_video(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["video_id"])
    return (
        valid.groupby("video_id")
        .agg(
            video_title=("video_title", "first"),
            video_url=("video_url", "first"),
            channel_name=("channel_name", "first"),
            channel_url=("channel_url", "first"),
            watch_count=("watched_at", "count"),
            first_watched=("watched_at", "min"),
            last_watched=("watched_at", "max"),
        )
        .reset_index()
    )


def aggregate_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["channel_name"]).sort_values("watched_at")
    return (
        valid.groupby("channel_name")
        .agg(
            channel_url=("channel_url", "first"),
            unique_videos=("video_id", "nunique"),
            watch_count=("watched_at", "count"),
            first_watched=("watched_at", "min"),
            last_watched=("watched_at", "max"),
            last_video_title=("video_title", "last"),
        )
        .reset_index()
    )
