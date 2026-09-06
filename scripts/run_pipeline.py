#!/usr/bin/env python
"""視聴履歴から「最近見てないけど昔よく見てた」チャンネルを1件、文脈付きバンディットで提案するCLI。"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from watch_recommender.aggregate import aggregate_by_channel  # noqa: E402
from watch_recommender.bandit import FEATURE_NAMES, LinearThompsonSamplingBandit  # noqa: E402
from watch_recommender.config import YOUTUBE_API_KEY  # noqa: E402
from watch_recommender.embeddings import embed_video_titles  # noqa: E402
from watch_recommender.enrich import fetch_video_metadata  # noqa: E402
from watch_recommender.genre import add_genre_scores  # noqa: E402
from watch_recommender.load_history import load_watch_history  # noqa: E402
from watch_recommender.score import score_channels  # noqa: E402
from watch_recommender.semantic import add_semantic_scores  # noqa: E402


def record_feedback(bandit: LinearThompsonSamplingBandit, label: str) -> None:
    pending = bandit.load_pending()
    if pending is None:
        print(
            "評価対象の提案がありません。先に `python scripts/run_pipeline.py` を実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    reward = 1.0 if label == "good" else 0.0
    bandit.update(np.array(pending["context"]), reward)
    bandit.log_feedback(pending["channel_name"], label)
    bandit.clear_pending()

    print(f"「{pending['channel_name']}」への評価（{label}）を記録しました。今後この候補は提案されません。")
    theta = bandit.theta_mean
    weights_str = ", ".join(f"{name}={value:.3f}" for name, value in zip(FEATURE_NAMES + ["bias"], theta))
    print(f"学習中の重み(theta): {weights_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-watch-count", type=int, default=2, help="候補とする最小累計視聴回数")
    parser.add_argument(
        "--no-enrich", action="store_true", help="YouTube Data APIでのメタデータ補完・ジャンル関連度の計算をスキップ"
    )
    parser.add_argument("--no-semantic", action="store_true", help="タイトル埋め込みによる意味検索スコアの計算をスキップ")
    parser.add_argument(
        "--category",
        help="このジャンルの候補だけに絞る（例: Music）。YouTube Data APIでのジャンル補完が必要",
    )
    parser.add_argument(
        "--title-regex",
        help="動画タイトルがこの正規表現にマッチする視聴だけを対象にする（例: 'MV|MAD'）。チャンネルではなく動画の種類で絞りたいときに使う",
    )
    parser.add_argument("--feedback", choices=["good", "bad"], help="直前に提案した候補への評価を記録してバンディットを更新")
    args = parser.parse_args()

    bandit = LinearThompsonSamplingBandit()

    if args.feedback:
        record_feedback(bandit, args.feedback)
        return

    history_df = load_watch_history()
    if history_df.empty:
        print("視聴履歴が読み込めませんでした。data/raw/watch-history.json を確認してください。", file=sys.stderr)
        sys.exit(1)

    if args.title_regex:
        try:
            pattern = re.compile(args.title_regex, re.IGNORECASE)
        except re.error as e:
            print(f"--title-regexの正規表現が不正です: {e}", file=sys.stderr)
            sys.exit(1)
        history_df = history_df[history_df["video_title"].str.contains(pattern, na=False)]
        if history_df.empty:
            print(f"「{args.title_regex}」に一致する視聴履歴が見つかりませんでした。", file=sys.stderr)
            sys.exit(1)

    channel_df = aggregate_by_channel(history_df)

    if args.no_enrich:
        pass
    elif not YOUTUBE_API_KEY:
        print("YOUTUBE_API_KEY未設定のため、ジャンル関連度はスキップします（.envを確認してください）。\n", file=sys.stderr)
    else:
        print("YouTube Data APIでメタデータを取得中（初回は数分かかる場合があります）...\n", file=sys.stderr)
        metadata_df = fetch_video_metadata(history_df["video_id"].dropna().unique())
        channel_df = add_genre_scores(channel_df, history_df, metadata_df)
    if "genre_score" not in channel_df.columns:
        channel_df["genre_score"] = 0.0

    if not args.no_semantic:
        print("動画タイトルを埋め込みベクトル化中（初回は数分かかる場合があります）...\n", file=sys.stderr)
        video_id_to_title = (
            history_df.dropna(subset=["video_id"])
            .drop_duplicates("video_id")
            .set_index("video_id")["video_title"]
            .to_dict()
        )
        embedding_df = embed_video_titles(video_id_to_title)
        channel_df = add_semantic_scores(channel_df, history_df, embedding_df)
    if "semantic_score" not in channel_df.columns:
        channel_df["semantic_score"] = 0.0

    scored_df = score_channels(channel_df)  # recency_score/frequency_scoreの算出、および従来ロジックの参考スコア

    candidates = scored_df[scored_df["watch_count"] >= args.min_watch_count]
    if candidates.empty:
        candidates = scored_df

    if args.category:
        if "dominant_category_name" not in candidates.columns:
            print(
                "`--category`を使うにはジャンル補完が必要です（`--no-enrich`を外し、.envにYOUTUBE_API_KEYを設定してください）。",
                file=sys.stderr,
            )
            sys.exit(1)
        filtered = candidates[candidates["dominant_category_name"].str.casefold() == args.category.casefold()]
        if filtered.empty:
            available = sorted(candidates["dominant_category_name"].dropna().unique())
            print(
                f"「{args.category}」に一致する候補が見つかりませんでした。候補にあるジャンル: {', '.join(available)}",
                file=sys.stderr,
            )
            sys.exit(1)
        candidates = filtered

    already_judged = bandit.already_judged_channels()
    if already_judged:
        remaining = candidates[~candidates["channel_name"].isin(already_judged)]
        if remaining.empty:
            print("条件に合う候補はすべて評価済みです。条件を変えて試してください。", file=sys.stderr)
            sys.exit(1)
        candidates = remaining

    context_matrix = candidates[FEATURE_NAMES].to_numpy(dtype=float)
    sampled_scores = bandit.sample_scores(context_matrix)
    best_idx = int(np.argmax(sampled_scores))
    chosen = candidates.iloc[best_idx]
    chosen_context = context_matrix[best_idx]

    bandit.save_pending(chosen["channel_name"], chosen_context, extra={"last_video_title": chosen["last_video_title"]})

    print(f"=== 視聴履歴サマリー: {len(history_df)}件 / {len(channel_df)}チャンネル ===\n")
    print("--- バンディットが選んだ今回のおすすめ ---")
    print(f"チャンネル: {chosen['channel_name']}")
    print(f"最後に見た動画: {chosen['last_video_title']}")
    print(
        "文脈: "
        + ", ".join(f"{name}={value:.3f}" for name, value in zip(FEATURE_NAMES, chosen_context))
    )
    print(f"サンプリング後の予測スコア: {sampled_scores[best_idx]:.3f}\n")
    print("これが良ければ:  python scripts/run_pipeline.py --feedback good")
    print("イマイチなら:    python scripts/run_pipeline.py --feedback bad\n")

    print("--- 参考: 従来ロジック（固定の重み）での上位5件 ---")
    print(
        candidates.head(5)[["channel_name", "last_video_title", "watch_count", "score"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
