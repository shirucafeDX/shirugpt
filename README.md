# ShiruGPT

日本語の質問から Gemini が SQL を作成し、BigQuery のデータを集計して回答する
Streamlit アプリです。現在は Streamlit Community Cloud に限定公開しているプロトタイプです。
一般知識を交えた課題発見、施策の検討、壁打ちにも対応します。

## 主な機能

- 期間・店舗・指標などが曖昧な場合、AI が確認質問を生成。選択肢・自由記述で回答できます。
- 棒・折れ線・面・散布図・円・ヒートマップに対応。AI が種類・軸・必要な内訳を選びます。
  元データ・SQL・実行量の確認と CSV 出力にも対応します。
- ログインした本人のチャット履歴を保存し、続きから分析できます。履歴は5件ずつ追加表示し、最近5件と開いた会話をキャッシュします。
- 利用者ごとの使用量記録・月次利用上限と、管理者向けの利用状況・設定確認に対応します。
- Gemini 3.8 Flashを通常はMediumで使用します。`/effort` でLow／Medium／Highを選択でき、
  入力欄の枠はLow＝緑、Medium＝青、High＝紫です。`/thinking` はHighと詳細分析用の指示を適用し、再入力で解除します。
- 入力欄で `/` を打つとコマンド候補を表示します。クリック、または上下キーとEnterで実行できます（Tabは入力補完）。
- `/usage` では当月の入力・出力（思考を含む）トークン、クエリ量、Gemini・BigQueryの費用と単価を確認できます。

## 使用サービスと構成

| サービス・技術 | 役割 |
| --- | --- |
| Streamlit / Streamlit Community Cloud | チャット画面とアプリの実行・公開 |
| Google OIDC | Google アカウントによるログイン |
| Gemini / Vertex AI | 質問の解釈、確認質問・SQL の作成、結果の説明 |
| BigQuery | 分析データの集計、利用ログ・利用上限設定の保存 |
| Cloud Storage | 会話履歴・SQL・取得結果の保存と復元（任意設定） |

```mermaid
flowchart TD
    User[ユーザー] <--> App[Streamlitアプリ<br/>Streamlit Community Cloud]
    App <-->|ログイン| Auth[Google OIDC]
    App <-->|質問・SQL作成・結果の解釈| AI[Gemini / Vertex AI]
    App <-->|SQL実行・利用ログと上限設定| BQ[BigQuery]
    App <-->|履歴の保存・復元| GCS[Cloud Storage]
```

Gemini はツール呼び出しを要求し、アプリが SQL を検査して BigQuery で実行します。
アプリは取得結果を統計情報とサンプル行にまとめて Gemini に返し、回答を画面に表示します。
利用ログと会話履歴は、Gemini が参照できる分析データから分離しています。

## クイックスタート

Python と Google Cloud CLI を用意し、対象プロジェクトの API・権限を設定してください。
詳しい手順と Google ログインの設定は [セットアップ](docs/setup.md) を参照してください。

```bash
git clone https://github.com/junya34/shirugpt.git
cd shirugpt
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

`.env` の `GCP_PROJECT_ID`、`BQ_DEFAULT_DATASET`、`BQ_ALLOWED_DATASETS` を設定して起動します。

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <プロジェクトID>
.venv/bin/streamlit run app.py --server.address 127.0.0.1
```

ローカルで Google ログインを設定しない場合、チャット履歴は永続化されません。

## 主な制約

- 読み取り専用 SQL と、許可したデータセットへのアクセスに限定します。
  高コストのクエリには実行前の確認と課金上限を設けています。
- 既定では Gemini に先頭500行のサンプルを渡し、画面用には最大5,000行を取得します。
  SQL に `LIMIT` がなければ1,000行を自動付与します。これらは設定で変更できます。
- SQL の集計粒度と回答の正しさは保証されません。重要な判断には「データ・実行詳細」で
  SQL と結果を確認してください。

## 開発・詳細資料

GCP に接続せずに回帰テストを実行できます。外部 API の実接続確認は別途必要です。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

- [セットアップ・環境変数](docs/setup.md)
- [処理の流れ・集計と表示の方針](docs/architecture.md)
- [安全機構・データの扱い](docs/security.md)
- [利用ログ・月次利用上限・管理者ページ](docs/admin.md)
