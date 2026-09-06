# run_pipeline.py の説明書

## ひとことでいうと
これまで説明してきたすべての係(ファイル)を順番に呼び出して、
実際に「おすすめを1件出す」「評価を記録する」を行う、
**司令塔(しれいとう)となるプログラム**です。あなたが実際に
ターミナルで実行するのは、このファイルです。

## なにをしているところ？

料理でいうなら、これまでのファイルが「野菜を切る係」
「お肉を焼く係」「味付けする係」だとすると、`run_pipeline.py`は
それらを正しい順番で呼び出して「1皿の料理を完成させるレシピ」
にあたります。

## 2つの使い方(モード)

このプログラムには、大きく分けて2つの動かし方があります。

### モード1: おすすめを1件もらう
```
python scripts/run_pipeline.py
```
これを実行すると、次の順番で処理が進みます。

1. `load_watch_history()` で視聴履歴を読みこむ
2. (オプションで)`--title-regex` により、特定のタイトルの動画だけに絞る
3. `aggregate_by_channel()` でチャンネルごとに集計する
4. (`--no-enrich`を付けていなければ)`fetch_video_metadata` と
   `add_genre_scores` でジャンルの一致度を計算する
5. (`--no-semantic`を付けていなければ)`embed_video_titles` と
   `add_semantic_scores` で意味の近さを計算する
6. `score_channels()` で総合点を計算する
7. 視聴回数が少なすぎる候補(`--min-watch-count`未満)を除外する
8. (`--category`が指定されていれば)そのジャンルだけに絞る
9. `bandit.already_judged_channels()` で、すでに評価済み・お休み中の
   チャンネルを除外する
10. 残った候補について`bandit.sample_scores()`を呼び、Thompson
    Samplingで1件選ぶ
11. 選んだ結果を`save_pending`で一時保存し、画面に表示する

### モード2: さっきの提案を評価する
```
python scripts/run_pipeline.py --feedback good
python scripts/run_pipeline.py --feedback bad
```
こちらを実行すると、`record_feedback()`という関数が呼ばれます。

1. `load_pending()`で「さっき何を提案していたか」を思い出す
2. `good`なら`reward=1.0`、`bad`なら`reward=0.0`として
   `bandit.update()`を呼び、学習させる
3. `log_feedback()`で「もう提案しない(または30日間お休み)」対象として
   記録する
4. `clear_pending()`で「提案待ち」の状態をリセットする

## 出てくる主な道具

### `argparse`
ターミナルで打ちこむオプション(`--min-watch-count 3`のようなもの)を
読み取るための、Python標準の道具です。`--help`を付けて実行すると、
それぞれのオプションの説明が表示されます。

### `sys.path.insert(0, ...)`
このファイルは`scripts`フォルダにありますが、中身は`src`フォルダの
プログラム(`watch_recommender`パッケージ)を使いたいので、
「ここも探しにいってね」とPythonに教えているおまじないです。

### `re.compile(args.title_regex, re.IGNORECASE)`
`--title-regex`オプションで、たとえば`"MV|MAD"`のような
「動画タイトルの中に含まれていてほしい文字パターン」を指定できます。
`re.IGNORECASE`は「大文字・小文字を区別しない」という設定です。

## 具体的な流れの例

初めてこのプログラムを実行したとしましょう(まだ評価履歴がありません)。

```
python scripts/run_pipeline.py --no-enrich --no-semantic
```

このとき`bandit`はまだ何も学習していないので、`sample_scores`は
ほぼランダムに近い選び方でチャンネルを1つ選びます。表示された
提案が気に入ったら

```
python scripts/run_pipeline.py --feedback good
```

を実行すると、そのチャンネルの特徴(`recency_score`などの組み合わせ)が
「良い」として記憶され、次にまた実行したときには、似た特徴を持つ
チャンネルが選ばれやすくなっていきます。

## もっと知りたい人へ
- `sys.stdout.reconfigure(encoding="utf-8")`: 日本語などの文字が
  ターミナルで文字化けしないように、出力の文字コードを指定しています。
- `# noqa: E402`: 「本来はファイルの先頭にimportを書くべき」という
  Pythonの作法(スタイルチェッカーの警告)を、あえて無視する印です。
  ここでは`sys.path`を先に設定してからimportする必要があるための
  やむを得ない例外です。
