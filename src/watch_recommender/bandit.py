import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DATA_PROCESSED

# 文脈(状態)として使う特徴量。score.pyがチャンネルごとに計算済みの0〜1スコアをそのまま使う。
FEATURE_NAMES = ["recency_score", "frequency_score", "genre_score", "semantic_score"]

BANDIT_STATE_PATH = DATA_PROCESSED / "bandit_state.json"
PENDING_SUGGESTION_PATH = DATA_PROCESSED / "pending_suggestion.json"
SUGGESTION_LOG_PATH = DATA_PROCESSED / "suggestion_log.json"

# good評価はこの日数だけ除外し、以後は再び候補になり得る（badは永久に除外）
GOOD_COOLDOWN_DAYS = 30


class LinearThompsonSamplingBandit:
    """線形回帰+Thompson Samplingによる文脈付きバンディット。

    フィードバック1件ごとに重み(theta)の確信度をその場で更新する。データが無いうちは
    事後分布が広く、候補選びはほぼランダム（探索）になり、フィードバックが貯まるほど
    学習した好みに沿った選択（活用）に寄っていく。
    """

    def __init__(
        self,
        feature_names=FEATURE_NAMES,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        state_path: Path = BANDIT_STATE_PATH,
    ):
        self.feature_names = feature_names
        self.alpha = alpha
        self.lambda_ = lambda_
        self.state_path = state_path
        self.d = len(feature_names) + 1  # +1 はバイアス項
        self.A, self.b = self._load_state()

    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return np.array(data["A"]), np.array(data["b"])
        return self.lambda_ * np.eye(self.d), np.zeros(self.d)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"A": self.A.tolist(), "b": self.b.tolist(), "feature_names": self.feature_names}, f)

    @staticmethod
    def _with_bias(context_matrix: np.ndarray) -> np.ndarray:
        bias = np.ones((context_matrix.shape[0], 1))
        return np.hstack([context_matrix, bias])

    @property
    def theta_mean(self) -> np.ndarray:
        return np.linalg.inv(self.A) @ self.b

    def sample_scores(self, context_matrix: np.ndarray) -> np.ndarray:
        """各候補について事後分布からthetaを1回サンプリングし、予測スコアを返す。"""
        X = self._with_bias(context_matrix)
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        theta_sample = np.random.multivariate_normal(theta_hat, (self.alpha**2) * A_inv)
        return X @ theta_sample

    def update(self, context: np.ndarray, reward: float) -> None:
        x = np.append(context, 1.0)
        self.A += np.outer(x, x)
        self.b += reward * x
        self._save_state()

    def save_pending(self, channel_name: str, context: np.ndarray, extra: Optional[dict] = None) -> None:
        payload = {"channel_name": channel_name, "context": context.tolist(), **(extra or {})}
        PENDING_SUGGESTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_SUGGESTION_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @staticmethod
    def load_pending() -> Optional[dict]:
        if not PENDING_SUGGESTION_PATH.exists():
            return None
        with open(PENDING_SUGGESTION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def clear_pending() -> None:
        if PENDING_SUGGESTION_PATH.exists():
            PENDING_SUGGESTION_PATH.unlink()

    @staticmethod
    def log_feedback(channel_name: str, label: str, judged_at: Optional[datetime] = None) -> None:
        """評価済みのチャンネルを記録する（同じチャンネルを繰り返し提案しないようにするため）。"""
        judged_at = judged_at or datetime.now(timezone.utc)
        log = LinearThompsonSamplingBandit._load_log()
        log.append({"channel_name": channel_name, "label": label, "judged_at": judged_at.isoformat()})
        SUGGESTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SUGGESTION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)

    @staticmethod
    def _load_log() -> list:
        if SUGGESTION_LOG_PATH.exists():
            with open(SUGGESTION_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @staticmethod
    def already_judged_channels(now: Optional[datetime] = None) -> set:
        """今、除外すべきチャンネル名の集合。

        bad評価は永久に除外する。good評価はGOOD_COOLDOWN_DAYS日だけ除外し、それ以降は
        再び候補になり得る（同じチャンネルに複数回評価がある場合は最新の評価を優先する）。
        """
        now = now or datetime.now(timezone.utc)
        latest_by_channel: dict = {}
        for entry in LinearThompsonSamplingBandit._load_log():
            latest_by_channel[entry["channel_name"]] = entry  # 後に追記された方で上書き

        excluded = set()
        for channel_name, entry in latest_by_channel.items():
            if entry["label"] == "bad":
                excluded.add(channel_name)
                continue
            judged_at_str = entry.get("judged_at")
            if judged_at_str is None:
                excluded.add(channel_name)  # 記録形式が古い場合は安全側に倒して除外を継続
                continue
            judged_at = datetime.fromisoformat(judged_at_str)
            if now - judged_at < timedelta(days=GOOD_COOLDOWN_DAYS):
                excluded.add(channel_name)
        return excluded
