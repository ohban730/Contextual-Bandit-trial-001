from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_PATH = DATA_PROCESSED / "video_embeddings.npz"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def _load_cache() -> dict:
    if EMBEDDING_CACHE_PATH.exists():
        data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=False)
        return dict(zip(data["video_ids"], data["vectors"]))
    return {}


def _save_cache(cache: dict) -> None:
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    video_ids = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values()))
    np.savez_compressed(EMBEDDING_CACHE_PATH, video_ids=video_ids, vectors=vectors)


def embed_video_titles(video_id_to_title: dict, use_cache: bool = True, batch_size: int = 256) -> pd.DataFrame:
    """動画タイトルを埋め込みベクトル化する（video_idごとにキャッシュ）。"""
    cache = _load_cache() if use_cache else {}
    missing = {vid: title for vid, title in video_id_to_title.items() if vid not in cache}

    if missing:
        model = _get_model()
        ids = list(missing.keys())
        titles = list(missing.values())
        vectors = model.encode(titles, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        for vid, vec in zip(ids, vectors):
            cache[vid] = vec.astype(np.float32)
        if use_cache:
            _save_cache(cache)

    rows = [{"video_id": vid, "embedding": cache[vid]} for vid in video_id_to_title if vid in cache]
    return pd.DataFrame(rows)
