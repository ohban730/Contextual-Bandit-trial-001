# Watch Recommender

YouTube視聴履歴から「最近見てないけど昔よく見た」動画を1件、文脈付きバンディットが提案するツール。「良い/イマイチ」を返すたびにその場で好みを学習していく。

## セットアップ

### 1. Python環境

既存のconda環境 `llm-sandbox`（`pandas`/`requests`/`python-dotenv`/`sentence-transformers`/`torch`が導入済み）を使う。

```bash
conda activate llm-sandbox
pip install -r requirements.txt  # 念のため（大半は導入済みのはず）
```

### 2. Google Takeoutから視聴履歴を取得

1. https://takeout.google.com/ にアクセス
2. 「すべて選択を解除」→ **YouTube と YouTube Music** のみチェック
3. 「複数の形式」→ **履歴** を **JSON** 形式に変更（デフォルトはHTMLなので注意）
4. エクスポートを作成しダウンロード
5. 展開後、`Takeout/YouTube and YouTube Music/history/watch-history.json` を、このプロジェクトの `data/raw/watch-history.json` にコピー

### 3. YouTube Data APIキー（任意、ジャンル関連度を使う場合）

1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクト作成 → 「APIとサービス」→「ライブラリ」から **YouTube Data API v3** を有効化 → 「認証情報」→「APIキーを作成」
2. `.env.example` を `.env` にコピーし、取得したキーを `YOUTUBE_API_KEY` に設定（1日あたり無料枠1万クォータ、動画情報取得は1リクエストあたり1クォータ程度なので個人利用では十分足りる）
3. キーが無い場合はジャンル関連度の計算だけが自動でスキップされ、他の機能はそのまま動く

### 4. 実行

```bash
python scripts/run_pipeline.py
```

出力例:
```
=== 視聴履歴サマリー: 27358件 / 7985チャンネル ===

--- バンディットが選んだ今回のおすすめ ---
チャンネル: サンプルチャンネル
最後に見た動画: サンプル動画タイトル
文脈: recency_score=0.847, frequency_score=0.375, genre_score=1.000, semantic_score=0.932
サンプリング後の予測スコア: 0.812

これが良ければ:  python scripts/run_pipeline.py --feedback good
イマイチなら:    python scripts/run_pipeline.py --feedback bad

--- 参考: 従来ロジック（固定の重み）での上位5件 ---
...
```

提案を見たら `--feedback good` か `--feedback bad` で評価を返す。これだけでバンディットの重みがその場で更新される（次に `run_pipeline.py` を実行する前に評価を返す必要がある。評価を返さず次を実行すると直前の提案は上書きされて評価できなくなる）。

オプション:
- `--min-watch-count`: 候補にする最小累計視聴回数（デフォルト2、いいね的な一見動画を除外）
- `--no-enrich`: YouTube Data APIでのメタデータ補完・ジャンル関連度をスキップ
- `--no-semantic`: タイトル埋め込みによる意味検索スコアをスキップ
- `--category <ジャンル名>`: 候補をそのジャンルだけに絞る（例: `--category Music`。`dominant_category_name`列に出る値と一致させる。ジャンル補完が必要）
- `--title-regex <正規表現>`: 動画タイトルがこの正規表現にマッチする視聴だけを対象にする。チャンネル単位ではなく「動画の種類」で絞りたいとき用（例: `--title-regex "\bMV\b|\bMAD\b"` でMV・MAD動画だけに絞る。単純に`"MV|MAD"`にすると英単語の"made"などに部分一致して誤検出が増えるので`\b`で単語境界を付けるのがコツ。マッチした視聴だけを対象に集計し直すので、`--min-watch-count`は1程度まで下げないと候補が出ないことが多い）
- `--feedback good|bad`: 直前の提案への評価を記録（他のオプションと同時指定不可、これのみ単独で使う）

初回実行時はYouTube Data APIの呼び出しとタイトルの埋め込みベクトル化が走るため数分かかることがある。結果は `data/processed/video_metadata.json`（メタデータ）と `data/processed/video_embeddings.npz`（埋め込みベクトル）にキャッシュされ、2回目以降は未取得分のみ処理するので数秒で終わる。`--feedback` の実行はこれらの再計算をスキップするので常に一瞬で終わる。

`good`か`bad`を返したチャンネルは `data/processed/suggestion_log.json` に評価日時とともに記録され、除外される。`good`と答えても実際にそのチャンネルを視聴したわけではなく（＝視聴履歴上の`recency_score`は現実には下がらない）、1回のフィードバックだけでもbanditの重みはその候補の文脈に強く引っ張られるため、この除外が無いと同じような候補ばかりが繰り返し提案され続けてしまう。

- **bad評価は永久に除外**（[bandit.py](src/watch_recommender/bandit.py) の `already_judged_channels`）。興味が無いという明確な意思表示なので、時間が経っても復活しない
- **good評価は`GOOD_COOLDOWN_DAYS`(デフォルト30日)だけ除外**し、それ以降は再び候補になり得る。このツールの趣旨は「たまに思い出させてくれる」ことなので、一度goodと言ったチャンネルを永久に除外する理由は薄いため
- 同じチャンネルに複数回評価がある場合は最新の評価が優先される
- 候補が尽きたとき、`suggestion_log.json`を丸ごと空にする必要は基本的にない。クールダウンにより時間経過で自然にgood評価の除外が切れていくため

## 提案ロジック — 文脈付きバンディット

各チャンネルは4つの0〜1スコアで表現される「文脈」を持つ。

```
recency_score   … 経過日数の正規化スコア（久しぶり度）
frequency_score … 過去の視聴頻度（好きだった度）
genre_score     … カテゴリIDの一致度（YouTube Data APIキーがある場合のみ、無ければ0固定）
semantic_score  … タイトル埋め込みの意味的な近さ
```

[bandit.py](src/watch_recommender/bandit.py) が線形回帰+Thompson Samplingでこの4つの重み(theta)を管理し、フィードバックのたびに更新する。データが無いうちはthetaの確信度が低く候補選びはほぼ探索的（ランダムに近い）になり、フィードバックが貯まるほど学習した好みに沿った選択に寄っていく。固定の重み(`w1〜w4`, [config.py](src/watch_recommender/config.py) の `ScoringWeights`)による従来ロジックのスコアは「参考」として毎回一緒に表示される。

### ジャンル関連度 — [genre.py](src/watch_recommender/genre.py)

直近14日間に見た動画のカテゴリID分布（「最近見ているジャンル」）を作り、各チャンネルの主要カテゴリがそこにどれだけ含まれるかをスコア化。カテゴリIDは`videos.list`のsnippetから取得。

### 意味検索 — [embeddings.py](src/watch_recommender/embeddings.py) / [semantic.py](src/watch_recommender/semantic.py)

多言語対応の埋め込みモデル `paraphrase-multilingual-MiniLM-L12-v2`（Hugging Face / sentence-transformers、GPU使用）で動画タイトルをベクトル化。直近14日間の視聴タイトルの平均ベクトル（＝いまの興味の方向）と、各チャンネルの視聴タイトル平均ベクトルとのコサイン類似度をスコアにする。カテゴリIDが同じでも実際のタイトルが似ていない場合や、逆にカテゴリは違っても内容が近い場合を拾えるのが狙い。

### 既知の制限: 「今ハマってないジャンル」は出にくい

`genre_score`も`semantic_score`も、直近14日間によく見ているジャンル・話題を基準に計算している。そのため、直近の視聴が特定ジャンル（例: アニメ反応系）に偏っていると、それ以外のジャンル（例: 音楽）は「久しぶりだから見たい」という動機があっても両スコアが低く出てしまい、構造的に上位に出にくい。数回のフィードバックだけでこの偏りをbanditが学習しきるのは難しいため、`--category`オプションで候補を明示的にそのジャンルへ絞り込むのが実用的な回避策になる。

## 今後の拡張（まだ未着手）

- フィードバックが数百件貯まったら、埋め込みモデル自体を対照学習でファインチューニングし、「あなた固有の近さの感覚」をより深く反映させる

## License

[MIT](LICENSE)
