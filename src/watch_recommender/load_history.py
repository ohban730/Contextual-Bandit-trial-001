import json
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import WATCH_HISTORY_PATH


# Takeoutのタイトルは表示言語によって "Watched {title}"（英語）や
# "{title} を視聴しました"（日本語）のように前後に定型文が付く。
_TITLE_PREFIXES = ["Watched "]
_TITLE_SUFFIXES = ["を視聴しました"]


def _extract_video_id(title_url: Optional[str]) -> Optional[str]:
    if not title_url:
        return None
    query = parse_qs(urlparse(title_url).query)
    values = query.get("v")
    return values[0] if values else None


def _clean_title(title: str) -> str:
    for prefix in _TITLE_PREFIXES:
        if title.startswith(prefix):
            return title[len(prefix):]
    for suffix in _TITLE_SUFFIXES:
        if title.endswith(suffix):
            return title[: -len(suffix)].rstrip()
    return title


def load_watch_history(path=None) -> pd.DataFrame:
    """Google Takeoutのwatch-history.jsonを読み込み、視聴イベント単位のDataFrameに変換する。"""
    path = path or WATCH_HISTORY_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        if entry.get("header") != "YouTube":
            continue

        video_id = _extract_video_id(entry.get("titleUrl"))
        if video_id is None:
            continue  # 広告再生・検索履歴など動画視聴以外のアクティビティは除外

        subtitles = entry.get("subtitles") or []
        channel_name = subtitles[0].get("name") if subtitles else None
        channel_url = subtitles[0].get("url") if subtitles else None

        time_str = entry.get("time")
        watched_at = datetime.fromisoformat(time_str.replace("Z", "+00:00")) if time_str else None

        rows.append(
            {
                "video_title": _clean_title(entry.get("title", "")),
                "video_url": entry.get("titleUrl"),
                "video_id": video_id,
                "channel_name": channel_name,
                "channel_url": channel_url,
                "watched_at": watched_at,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["watched_at"] = pd.to_datetime(df["watched_at"], utc=True)
    return df
