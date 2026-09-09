# このプログラムぜんぶの「かんたん説明書」

このフォルダには、プロジェクトの中にあるプログラム(ファイル)を1つずつ、
**小学生にもわかるように**説明したノートが入っています。

## このプログラム全体は、なにをするもの？

あなたがYouTubeで見た動画の記録(「視聴履歴」)を読みこんで、

> 「むかしはよく見てたけど、さいきん見てないチャンネル」

を1つだけ選んで教えてくれる、**「思い出しレコメンドロボット」**です。

ロボットはただ選ぶだけじゃなくて、あなたが「良い(good)」「イマイチ(bad)」と
評価するたびに、すこしずつ**あなたの好みを覚えていきます**。これを
「文脈付きバンディット(contextual bandit)」という仕組みで実現しています。

## ロボットが動く順番(パイプライン)

`scripts/run_pipeline.py` が全体の司令塔です。上から順番に、こんなふうに
それぞれのファイルを呼び出しています。

```
①視聴履歴を読みこむ           → load_history.py
②チャンネルごとにまとめる      → aggregate.py
③(必要なら)ジャンルを調べる    → enrich.py, genre.py
④(必要なら)動画タイトルの意味を調べる → embeddings.py, semantic.py
⑤それぞれのチャンネルに点数をつける → score.py
⑥ロボットが1つ選ぶ・好みを覚える     → bandit.py
⑦ぜんぶをつなげて実行する           → run_pipeline.py
```

設定(APIキーや保存場所など)は `config.py` にまとまっています。

## 各ファイルの説明書 一覧

| ファイル | ひとことでいうと | 説明書 |
|---|---|---|
| `config.py` | ロボットの「設定書き込みノート」 | [config.md](config.md) |
| `load_history.py` | YouTubeの記録を読める形に整える係 | [load_history.md](load_history.md) |
| `aggregate.py` | 動画ごと・チャンネルごとに集計する係 | [aggregate.md](aggregate.md) |
| `enrich.py` | YouTubeに「この動画の種類は?」と聞きに行く係 | [enrich.md](enrich.md) |
| `genre.py` | 最近好きなジャンルと一致するか調べる係 | [genre.md](genre.md) |
| `embeddings.py` | 動画タイトルを「数字の意味コード」に変える係 | [embeddings.md](embeddings.md) |
| `semantic.py` | タイトルの意味が似ているか調べる係 | [semantic.md](semantic.md) |
| `score.py` | チャンネルに点数をつける係 | [score.md](score.md) |
| `bandit.py` | 1つ選んで、好みを学習していくロボットの頭脳 | [bandit.md](bandit.md) / 具体例つき: [bandit_walkthrough.md](bandit_walkthrough.md) |
| `run_pipeline.py` | 全部をつなげて動かす司令塔 | [run_pipeline.md](run_pipeline.md) |

読む順番は、上の表の上から順番がおすすめです。最後に `bandit.py` を読むと、
「なぜAIっぽく好みを学習できるのか」がいちばんよくわかります。
