import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

WATCH_HISTORY_PATH = DATA_RAW / "watch-history.json"


@dataclass
class ScoringWeights:
    recency: float = 0.4    # w1: 久しぶり度
    frequency: float = 0.2  # w2: 好きだった度
    genre: float = 0.2      # w3: ジャンル関連度（Phase B: カテゴリID一致）
    semantic: float = 0.2   # w4: 意味的関連度（Phase B: タイトル埋め込みの類似度）


YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
