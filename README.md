# aibyss-lab-lounge (L2)

A.I.byss Suite の「会話ランタイム」。発話テキストを受け取り、Event Bus (Redis Streams) に
`utterance.final` → `llm.final` → `tts.done` の 3 イベントを publish する。

> **Walking Skeleton 実装状況**:
> Step 4 完了 — 開発用テキストエミッタ実装済み。
> 本物の STT / LLM / TTS は非スコープ（将来 Step で差し替え）。

---

## セットアップ

### 前提

- Python 3.11 以上
- [uv](https://github.com/astral-sh/uv) がインストール済み
- `aibyss-workspace` が兄弟ディレクトリに clone 済み（スキーマ自動探索に使用）
- `aibyss-workspace` で Redis が起動済み（本番実行時）

```
repos/
  aibyss-workspace/          <- specs/event-envelope-0.1.schema.json, docker-compose.yml
  aibyss-lab-lounge/         <- このリポジトリ
  aibyss-coral-chronicle/    <- C2（購読・タイムライン API）
```

### インストール

```bash
cd aibyss-lab-lounge

# 依存ライブラリをインストール（.venv を自動作成）
uv sync

# 開発用ツール（pytest 等）を含めてインストール
uv sync --extra dev
```

### 環境変数

```bash
cp .env.example .env
# 必要に応じて .env を編集する（既定値のまま動作する）
```

主要変数:

| 変数 | 既定値 | 説明 |
|------|--------|------|
| `REDIS_URL` | `redis://localhost:6379` | Redis 接続 URL |
| `REDIS_STREAM_KEY` | `aibyss:events` | publish 先の Redis Stream キー |
| `AIBYSS_SCHEMA_PATH` | *(自動探索)* | スキーマファイルの絶対パス（省略で兄弟 workspace を探索） |

---

## 実行方法（開発用テキストエミッタ）

```bash
uv run python -m lab_lounge.emitter "今日の天気を教えて"
```

出力例:

```
stream_id  : 3fa85f64-5717-4562-b3fc-2c963f66afa6
session_id : 7c9e6679-7425-40de-944b-e07fc1f90ae7
trace_id   : 550e8400-e29b-41d4-a716-446655440000
  published: type=utterance.final event_id=...
  published: type=llm.final       event_id=...
  published: type=tts.done        event_id=...
```

`--stream-id` で stream_id を固定することもできる:

```bash
uv run python -m lab_lounge.emitter "テスト発話" --stream-id my-stream-002
```

---

## C2 と組み合わせた確認手順（Step 4 受け入れ条件）

### 1. Redis を起動する

```bash
cd ../aibyss-workspace
docker compose up -d
docker compose exec redis redis-cli ping
# -> PONG
```

### 2. C2 を起動する

```bash
cd ../aibyss-coral-chronicle
# .env に REDIS_URL=redis://localhost:6379 が設定されていることを確認
uv run uvicorn coral_chronicle.main:app --host 0.0.0.0 --port 8100
```

C2 の起動ログに以下が出れば Consumer 準備完了:

```
INFO  coral_chronicle.bus  Consumer Group 'cg-c2' を確認/作成しました
```

### 3. テキストエミッタを実行する

別ターミナルで:

```bash
cd ../aibyss-lab-lounge
uv run python -m lab_lounge.emitter "今日の天気を教えて"
```

表示された `stream_id` をメモしておく。

### 4. C2 のタイムラインで 3 件を確認する

**PowerShell（推奨 — 日本語が文字化けしない）**:

```powershell
$sid = "<上記の stream_id>"
Invoke-RestMethod "http://localhost:8100/timeline?stream_id=$sid" | ConvertTo-Json -Depth 5
```

**curl を使う場合（注意）**: Windows では `curl.exe ... | python -m json.tool` は  
`python -m json.tool` が stdin を cp932 で読む場合があるため **日本語が文字化けして見える** ことがある。  
データ自体（SQLite / Redis 内）は正常な UTF-8 で保持されているため、  
`Invoke-RestMethod` や `curl.exe ... | python -c "import sys,json; ..."`  
で UTF-8 指定して確認すること。

```powershell
# UTF-8 を明示して標準入力を読む確認コマンド
$sid = "<上記の stream_id>"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
curl.exe -s "http://localhost:8100/timeline?stream_id=$sid" | `
  python -c "import sys,json; sys.stdin.reconfigure(encoding='utf-8'); print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

期待レスポンス:

```json
{
  "events": [
    {"type": "utterance.final", "stream_idx": 0, "...": "..."},
    {"type": "llm.final",       "stream_idx": 1, "...": "..."},
    {"type": "tts.done",        "stream_idx": 2, "...": "..."}
  ],
  "total": 3
}
```

### 5. links の因果関係を確認する

```bash
curl -s "http://localhost:8100/timeline?stream_id=<stream_id>" \
  | python -c "import json,sys; evs=json.load(sys.stdin)['events']; [print(e['type'], e.get('links')) for e in evs]"
```

期待出力:

```
utterance.final None
llm.final       ['<utterance.final の event_id>']
tts.done        ['<llm.final の event_id>']
```

---

## テスト

```bash
uv run pytest -v
```

テスト一覧:

| ファイル | 内容 |
|---------|------|
| `tests/test_events.py` | スキーマ検証・ビルダー・Guardrail G-1 確認 |
| `tests/test_pipeline.py` | 3 イベント順序・links 因果関係・seq 連番 |
| `tests/test_bus.py` | XADD フィールド名・JSON 値・クライアント close |

---

## ディレクトリ構成

```
aibyss-lab-lounge/
  .env.example
  .gitignore
  pyproject.toml
  src/
    lab_lounge/
      __init__.py
      events.py      # Event Envelope ビルダー + スキーマ検証
      bus.py         # Redis Streams publisher (XADD)
      pipeline.py    # 3 イベントを順に作って publish
      emitter.py     # CLI エントリポイント
  tests/
    conftest.py
    test_events.py
    test_pipeline.py
    test_bus.py
```

---

## Step 5 に入る前の前提条件

- [ ] `uv run pytest -v` が全テスト PASS する
- [ ] `docker compose up -d` で Redis が起動している
- [ ] C2 起動後にエミッタを実行し、`GET /timeline` で `stream_idx` 0/1/2 の 3 件が取得できる
- [ ] `llm.final.links` に `utterance.final` の `event_id` が入っている
- [ ] `tts.done.links` に `llm.final` の `event_id` が入っている

---

## Step 6 スモークテスト — L2 の役割

Step 6 では L2 エミッタを 1 回実行するだけで全イベントが流れる。

```powershell
cd aibyss-lab-lounge
uv run python -m lab_lounge.emitter "今日の天気を教えて"
```

出力例：

```
stream_id  : 3fa85f64-5717-4562-b3fc-2c963f66afa6   ← これをコピーして C2 確認に使う
session_id : 7c9e6679-7425-40de-944b-e07fc1f90ae7
trace_id   : 4bf92f3577b34da6a3ce929d0e0e4736
  published: type=utterance.final event_id=aaaaaaaa-...   seq=0
  published: type=llm.final       event_id=bbbbbbbb-...   seq=1
  published: type=tts.done        event_id=cccccccc-...   seq=2
```

`stream_id` をコピーして C2 タイムライン確認に使う:

```powershell
$sid = "3fa85f64-5717-4562-b3fc-2c963f66afa6"   # ← 上記出力から貼り付け
Invoke-RestMethod "http://localhost:8100/timeline?stream_id=$sid" | ConvertTo-Json -Depth 5
```

**L2 の成功ログ**: `published: type=...` が 3 行出ること。  
**V2 の成功**: ブラウザ `http://localhost:3200` の字幕エリアに `llm.final.payload.text` が表示されること（tts.done を V2 が受けて hud.caption を生成するため）。

---

## 設計概要（詳細は設計ドキュメント参照）

| ドキュメント | 場所 |
|------------|------|
| システム全体設計 | `../aibyss-workspace/docs/design/system.md` |
| L2 詳細設計 | `../aibyss-workspace/docs/design/lab-lounge.md` |
| Walking Skeleton 計画 | `../aibyss-workspace/docs/plan/walking-skeleton.md` |
| Event Envelope スキーマ | `../aibyss-workspace/specs/event-envelope-0.1.schema.json` |

**設計契約上の重要事項**:

- `stream_idx` を Event Envelope に含めない（G-1）。
- `POST /events` に直接送信しない（G-2）。Event Bus（Redis Streams）経由のみ。
- `seq` を全体ソートキーに使わない（G-3）。全体順序は C2 の `stream_idx` による。

---

## 1. Lab-Lounge（L2）機能概要

### 1.1 目的

AITuberとの会話を **止めずに・速く・安定して**成立させる会話実行基盤。

### 1.2 機能（想定）

- **会話パイプライン**
    - STT（音声→テキスト）
    - （必要時）RAG：C2検索 / Web検索
    - LLM応答生成（外部ホスト）
    - TTS（テキスト→音声）
- **3人AITuberの協調**
    - 通常：1人が応答（ルーターで担当決定）
    - 必要時：複数参加（素材収集）→ **最終出力の整形は1回**（API回数を抑える）
- **フォールバック設計（ライブ耐性）**
    - C2検索が遅い/失敗 → 検索無しで先に結論返す
    - Web検索は「必要時のみ」＆タイムアウト短め
- **短期記憶（実行状態）**
    - 直近Nターン、会話の一時状態、直前のツール結果など（LangGraph state）
- **ローカル活用（速度/安定）** *(将来目標)*
    - VAD/録音制御、STT、TTS、Embedding生成などは可能な限りローカルで実行（LLMは外部）
    - *v0.1 Walking Skeleton では開発用テキストエミッタで代替（本物の STT/TTS/Embedding は非スコープ）*

### 1.3 何が嬉しいのか（価値）

- **ライブ体験が崩れない**（沈黙が短い／失敗しても会話が続く）
- **会話の品質が上がる**（必要な時だけC2やWebを参照）
- **運用コストが読める**（呼び出し回数とタイムアウトで制御できる）
- **後続ツール（V2）につながるログが自動で溜まる**（C2にイベントとして残せる）

### 1.4 境界線（非目標）

- L2は「長期記憶の整理」や「統計・レポート作り」を主目的にしない
    
    → それはC2/V2側へ寄せる（L2はライブ優先）。