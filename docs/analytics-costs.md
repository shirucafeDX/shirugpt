# 回答ごとの費用計測

Conversational Analytics方式の回答下部に、AI適用後・BigQuery・使用量確認クエリ・合計をUSDで表示する。回答終了後60秒間は5秒間隔で更新し、その後は「費用を再取得」で確認する。会話内の取得済み合計と未確定件数も表示する。生成中の課金メーターやCloud Billing請求確定額ではない。

## 計算

- AIの通常単価換算: 内部入力 × USD 3/100万 + 出力（思考を含む）× USD 20/100万。2026-10-28確認の[公式料金](https://cloud.google.com/products/data-agents/pricing?hl=en)。2026-12-31までのData Cloud agents無料試用条件ではAI適用後を0とする。通常単価換算とトークン数の未取得を0にはしない。
- BigQuery: 実行済みジョブのtotal_bytes_billed × アプリ既存のオンデマンド単価（USD 6.25/TiB）。キャッシュヒットで課金対象0なら0。予約・容量課金はバイト数で換算できないため未確定。
- 同じジョブIDは一度だけ計上。過去ジョブ参照の再送で追加費用を計上しない。
- 公開SQLに対応するジョブIDが欠ける場合、自分の実行ジョブをターン時間帯±3秒で取得し、SQL構文木を照合する。同じSQLの候補が複数なら未確定。非公開SQLや別実行主体のジョブまで網羅する保証はない。
- 使用量確認クエリ自体の課金対象バイトも集計。1回の上限は64MiB。保存・ログ取り込み・配信・契約割引・BigQuery無料枠などは計算対象外。

## ネイティブログ設定

[会話ログ](https://docs.cloud.google.com/bigquery/docs/conversation-logging-overview)は回答後に遅れて届くベストエフォートの記録。モデル呼び出しごとのLLM_RESPONSEをevent_idで重複排除し、完了イベントが揃った場合のみ通常単価換算を確定する。内部の使用量を質問単独のCountTokensで代用しない。

保存先は環境変数ANALYTICS_TRACE_DATASET（既定shirugpt_analytics_logs）とANALYTICS_TRACE_TABLE（既定agent_events）。分析対象とは別にする。初期設定は以下。

```sh
PYTHONPATH=. .venv/bin/python scripts/setup_analytics_costs.py --project-number YOUR_PROJECT_NUMBER --apply
```

専用データセットを作成し、同プロジェクトのConversational AnalyticsサービスエージェントだけにデータセットのWRITER権限を追加する。新規データセットのテーブル保存期限は30日。会話ログには質問・回答を含むため、参照権限は管理者向けとする。

アプリはSDKのログ有効化設定を試行し、サーバーがenabledを返したか検査する。未確認なら画面に明記する。Google Cloud Consoleのデータエージェント編集で会話ログを有効化・保存先を指定・公開することもできる。

**2026-10-28の環境検証では未完了**: データセット作成とサービスエージェントの権限付与は完了したが、SDKおよびRESTの有効化更新をサーバーが保持せず、実行後もログテーブルが生成されなかった。ブラウザーのログインアカウントでは対象プロジェクトが一覧に出ず、画面からの有効化も未検証。したがって現在のAI実トークン数・通常単価換算は取得待ち。BigQueryジョブに基づく計算は動作する。

## 保存と月次集計

費用・取得状態・対象ジョブはanalytics_state.costsに保持し、保存会話ZIPに含める。更新時には既存アプリ利用ログへanalytics_costイベントとしてJSONスナップショットを保存する。後から集計する場合はturn_idごとにevent_timeが最新の行を使用し、更新行を合算しない。既存の加算用トークン列には転記しない。

従来の月次集計・上限判定へ分析APIのAI費用はまだ統合していない。画面にもこの制約を表示する。確認できたBigQueryジョブと使用量確認クエリのバイト数は既存集計に加算する。

## 検証

実APIで「神大店の2026年7月の来店者数と来店数」を試験。15.44秒で582人・1,586回来店を回答。ジョブ1件の課金対象量0（キャッシュヒット）を取得し、無料試用適用後の概算合計USD 0.000000、AI使用量は取得待ちと区別した。結果は/tmp/shirugpt-live-cost-final-20261028/cost-result.json。
