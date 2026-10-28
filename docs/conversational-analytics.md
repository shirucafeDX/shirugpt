# Conversational Analytics APIを中心とする最小設計

## 責任の分担

旧Geminiエージェントのプロンプトを短縮して転用する方式ではなく、独立したバックエンドを使用する。
アプリはAPIへの接続・会話ID・画面と保存を扱い、SQL作成・実行・修正・Python分析・グラフ作成はサービスへ委任する。
旧ツール定義、ツール選択、結果CSVの再送、作業メモの生成・検証、LLM反復ループは新経路へ接続しない。
既存UIとの接続に限り、AnswerResult、ToolContext、QueryRunという表示・保存用の型を共有する。

## 登録コンテキスト

`src/conversational_analytics.py`の短いINSTRUCTIONは目的、事実と仮説の区別、対象の確認、回答条件を指定する。
FIELD_CONTEXTは業務定義を列のdescriptionとして登録する。connected_idは期間内の重複なし人数、DU_idは重複なし来店数、storeは正式名と略称の対応、ordered_atは日本時間のDATETIME。
これらはデータの仕様であり、モデルの実行手順書ではない。

ANALYTICS_TABLESの実在をBigQueryの一覧で照合し、スキーマを確認する。保存した一覧・スキーマがあれば再利用する。
登録コンテキストとプロジェクト・リージョンのハッシュからデータエージェントIDを決め、同じ設定では再利用する。
定義が変われば新しいデータエージェント・会話を作る。APIに登録したコンテキストとハッシュも保存し、実行詳細で表示する。

## 1ターンの処理

1. 新規会話なら登録済みデータエージェントに結び付けたAPI会話を作る。
2. API会話参照と新しい質問1件、基準日時を送る。過去のアプリ会話・SQL・CSVは再送しない。
3. ストリームの最終回答、スキーマ、SQL、データ、BigQueryジョブ、グラフ、分析・エラー等の公開イベントを収集する。
4. 最終回答が届いても受信を終えない。その後のSQL・結果イベントも最後まで保存する。
5. ジョブIDからBigQueryの実行時刻・処理量・課金対象量を取得する。group_idが未指定の複数クエリも個々に処理し、同じジョブは二重計上しない。
6. 表とSQLは既存のデータ表示へ、ネイティブのグラフ仕様はそのまま描画へ渡す。旧方式のグラフ推測を重ねない。
7. 元の受信イベントと実行時間を保存会話とBigQueryのanalytics_call記録へ保存する。未公開の使用量はNULLのままとする。

アプリからのchat呼び出しが1回でも、サービス内部のLLM呼び出しが1回とは限らない。
外部APIの待ち時間とBigQueryジョブ時間は測れるが、内部モデルの計算時間・入力全文・履歴圧縮の全文は取得できない。

実試験ではTHOUGHTに3件のSQLと実行完了の説明があるのに、構造化されたgenerated_sql・ジョブ・結果は1件だけ返るケースがあった。公開イベントだけで全ジョブの課金量・原結果を収集できるとは限らない。
`scripts/audit_analytics_turn.py`で当該ターンの時刻と受信SQLを自分のBigQueryジョブへ照合できる。整形と識別子の大小文字を構文木で正規化し、一致する実行済み結果を追加取得できる。新しい分析SQLは実行しない。この照合も非公開の内部処理全体を保証するものではない。

## 状態と履歴

analytics_stateにはデータエージェント名、API会話名、登録コンテキスト、結果・ジョブの参照、課金済みジョブIDを保存する。
新しい質問では会話IDを参照し、アプリが作る要約やcontext_noteは送らない。結果の保存量が増えても、それを新しい質問の本文に付け加えない。

Cloud StorageのZIP保存はAPI会話参照と表データを保持する。サービス側の保存期間が切れた場合に会話を完全復元できることは保証しない。
方式を切り替えた履歴は表示用として読み込み、対象や期間の再指定を案内する。旧モデルの署名・未完ツール要求を新APIに送らない。

## API・ツールの採用判断

- Conversational Analytics API: 中心となる唯一の通常生成経路。一般相談も含め、初期試験では別モデルへのルーティングを追加しない。
- Python分析: データエージェントの組み込み機能を有効化。アプリ端末で生成コードを実行しない。利用と実行の確認は受信イベントで行う。
- CountTokens: 比較CLIの`--count-tokens`のみ。登録コンテキストJSON単独をGeminiで計測する参考値。通常の各ターンには追加しない。分析APIの内部入力や料金を測るものではない。
- ContextCache: 最小構成では導入しない。分析APIへcached_contentを渡す設計ではなく、登録済みエージェントと会話参照を使う。別のGemini APIのキャッシュ作成だけを追加しても、この経路の使用量削減は保証できない。
- 通常Gemini・外部オーケストレーター: 初期構成では追加しない。最終回答の再生成による余分な往復を避ける。

## 権限・費用・停止の変更

必要な権限はデータエージェントの作成・利用、会話作成・利用と、対象BigQueryデータの参照・ジョブ実行。
共有アプリを公開する場合、実行主体のIAMを分析対象へ限定する。ANALYTICS_TABLES、BQ_ALLOWED_DATASETSとシステム指示だけでサービス内部SQLのアクセス範囲を強制できるとは扱わない。
旧sql_guard・dry run承認UIは新API内部のSQL実行前に介入しない。MAX_BYTES_BILLEDをサービスのbig_query_max_billed_bytesへ登録する。

アプリの月次利用表示に分析APIのAI費用は含まれない。そのため従来と同じ精度で総額の月次上限を強制できるとは主張しない。BigQuery課金対象量は取得できた実行ジョブのみ集計する。
各回答の概算費用と会話内合計は別途表示する。使用量ログ設定と現在の未取得項目は[費用計測](analytics-costs.md)を参照。
分析APIの使用量・請求はCloud Billing側でも確認する。

停止はgRPCストリームをcancelする。内部BigQueryやPythonのすべてが即時取消しになる保証はない。
切断・タイムアウト後の自動再送は実装しない。完了した回答がなければ成功として扱わず、受信済みの実行情報は残す。

## 再現試験

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_analytics.py --output /tmp/analytics-new-run --count-tokens
PYTHONPATH=. .venv/bin/python scripts/benchmark_analytics.py --output /tmp/analytics-followup --resume /tmp/analytics-new-run/b4.zip --question '保存済み結果を使って比較してください。'
PYTHONPATH=. .venv/bin/python scripts/audit_analytics_turn.py /tmp/analytics-new-run/b3.json --output /tmp/analytics-audit.json --results
```

初期8問は単純計算、計算の追質問、一般相談、店舗集計、月別グラフ、期間・学年比較、保存復元後の計算、曖昧な依頼の確認。
試験用ユーザーを使い、実データとAPIを呼び出す。出力ディレクトリには質問・回答・SQL・イベント・結果・ターンID・会話ZIPと開始時ソースハッシュを保存する。
総合正答率にまとめず、集計、計算、指示遵守、SQL禁止、グラフ、履歴再利用、API失敗を分ける。
