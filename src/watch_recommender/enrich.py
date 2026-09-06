import json
import re
from typing import Iterable, Optional

import pandas as pd
import requests

from .config import DATA_PROCESSED, YOUTUBE_API_KEY

VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
BATCH_SIZE = 50
METADATA_CACHE_PATH = DATA_PROCESSED / "video_metadata.json"

_DURATION_RE = re.compile(r"PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?")


def _parse_duration(duration: Optional[str]) -> int:
    match = _DURATION_RE.fullmatch(duration or "")
    if not match:
        return 0
    parts = match.groupdict()
    hours, minutes, seconds = (int(parts[k] or 0) for k in ("hours", "minutes", "seconds"))
    return hours * 3600 + minutes * 60 + seconds


def _load_cache() -> dict:
    if METADATA_CACHE_PATH.exists():
        with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_video_metadata(video_ids: Iterable[str], api_key: str = None, use_cache: bool = True) -> pd.DataFrame:
    """videos.list APIでカテゴリ・タグ・再生時間・統計情報を取得する（video_idごとにキャッシュ）。"""
    api_key = api_key or YOUTUBE_API_KEY
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEYが設定されていません。.envを確認してください。")

    video_ids = list(dict.fromkeys(video_ids))
    cache = _load_cache() if use_cache else {}
    missing = [vid for vid in video_ids if vid not in cache]

    for batch in _chunk(missing, BATCH_SIZE):
        response = requests.get(
            VIDEOS_ENDPOINT,
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
                "key": api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("items", [])

        found_ids = set()
        for item in items:
            vid = item["id"]
            found_ids.add(vid)
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            cache[vid] = {
                "category_id": snippet.get("categoryId"),
                "tags": snippet.get("tags", []),
                "duration_seconds": _parse_duration(item.get("contentDetails", {}).get("duration")),
                "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
            }
        for vid in batch:
            if vid not in found_ids:
                cache[vid] = None  # 削除済み・非公開などで取得できなかった動画

        if use_cache:
            _save_cache(cache)

    rows = [{"video_id": vid, **cache[vid]} for vid in video_ids if cache.get(vid) is not None]
    return pd.DataFrame(rows)
