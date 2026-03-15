# aibyss-lab-lounge (L2)

A.I.byss Suite の「会話ランタイム」。発話テキストを受け取り、Event Bus (Redis Streams) に
`utterance.final` → `llm.final` → `tts.done` の 3 イベントを publish する。

> **Walking Skeleton 実装状況**:
> Step 4–6 完了 — 開発用テキスト・音声ファイルエミッタ実装済み。
> Phase 1 real LLM 対応済み (llm.py / graph.py)。
> Phase 2 real TTS 対応済み (tts.py)。
> Phase 3 real STT 対応済み (stt.py)。
> Phase 4 マイク入力・スピーカー出力対応済み (audio_io.py / run_once.py)。
> Phase 5 LangSmith 観測対応済み (observability.py)。

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
| `L2_USE_REAL_LLM` | `false` | `true` にすると real LLM を呼ぶ（ダミー応答を無効化） |
| `L2_LLM_PROVIDER` | `openai` | LLM プロバイダ（現在 `openai` のみ対応） |
| `L2_LLM_MODEL` | `gpt-4o-mini` | 使用するモデル名 |
| `OPENAI_API_KEY` | *(必須 / real mode のみ)* | OpenAI API キー（`.env` に記載。リポジトリにコミット禁止） |
| `L2_USE_REAL_TTS` | `false` | `true` にすると real TTS を呼ぶ |
| `L2_TTS_PROVIDER` | `voicevox` | TTS プロバイダ (`edge_tts` / `voicevox`) |
| `L2_USE_REAL_STT` | `false` | `true` にすると `--audio-file` で real STT を呼ぶ |
| `L2_STT_PROVIDER` | `openai` | STT プロバイダ（現在 `openai` Whisper API のみ） |
| `L2_STT_LANG` | `ja` | 認識言語 (ISO 639-1) |
| `LANGSMITH_TRACING` | `false` | `true` にすると LangChain / LangGraph 実行を LangSmith に送信する |
| `LANGSMITH_API_KEY` | *(必須 / tracing on のみ)* | LangSmith API キー（`.env` に記載。コミット禁止） |
| `LANGSMITH_PROJECT` | `aibyss-lab-lounge` | LangSmith プロジェクト名 |

---

## 実行方法（開発用エミッタ）

### テキスト入力（ダミーモード・デフォルト）

Redis に 3 イベントを publish する。LLM / TTS / STT は呼ばない。

```powershell
uv run python -m lab_lounge.emitter "今日の天気を教えて"
```

### 音声ファイル入力（real STT モード）

`--audio-file` で音声ファイルを指定する。`L2_USE_REAL_STT=true` のとき real STT API を呼ぶ。

```powershell
# 1. stt extra をインストール（初回のみ）
uv sync --extra stt

# 2. 環境変数をセット（OPENAI_API_KEY は LLM と共用）
$env:L2_USE_REAL_STT  = "true"
$env:L2_STT_PROVIDER  = "openai"
$env:L2_STT_LANG      = "ja"

# 3. 音声ファイルを指定して実行
uv run python -m lab_lounge.emitter --audio-file samples/q1.wav

# 環境変数をリセット
Remove-Item Env:\L2_USE_REAL_STT, Env:\L2_STT_PROVIDER, Env:\L2_STT_LANG
```

> **ダミー STT モード** (`L2_USE_REAL_STT=false`): `--audio-file` を指定しても 警告ログを出してダミーテキストでパイプラインが続く。

### real LLM モード

`L2_USE_REAL_LLM=true` を設定すると `llm.final` が OpenAI を実際に呼ぶ。

```bash
# 1. llm extra をインストール（初回のみ）
uv sync --extra llm

# 2. .env に API キーを設定
echo "L2_USE_REAL_LLM=true" >> .env
echo "OPENAI_API_KEY=sk-..." >> .env  # 実際のキーに差し替える

# 3. 実行
uv run python -m lab_lounge.emitter "今日の天気を教えて"
```

**PowerShell からお試しの場合**:

```powershell
$env:L2_USE_REAL_LLM = "true"; $env:L2_LLM_MODEL = "gpt-4o-mini"
uv run python -m lab_lounge.emitter "今日の天気を教えて"
Remove-Item Env:\L2_USE_REAL_LLM, Env:\L2_LLM_MODEL
```

### real TTS モード

```powershell
# VOICEVOX Engine を起動した後（別ターミナル）:
# docker run -p 50021:50021 voicevox/voicevox_engine:latest

$env:L2_USE_REAL_TTS      = "true"
$env:L2_TTS_PROVIDER      = "voicevox"
$env:L2_TTS_VOICE         = "89"   # Voidoll
$env:L2_TTS_SPEAKER       = "Voidoll"
$env:L2_TTS_OUTPUT_DIR    = "./data/audio"
$env:L2_TTS_VOICEVOX_URL  = "http://localhost:50021"
uv run python -m lab_lounge.emitter "こんにちは"
Remove-Item Env:\L2_USE_REAL_TTS, Env:\L2_TTS_PROVIDER, Env:\L2_TTS_VOICE, Env:\L2_TTS_SPEAKER, Env:\L2_TTS_OUTPUT_DIR, Env:\L2_TTS_VOICEVOX_URL
```

### real STT + real LLM + real TTS 最短手順

**前提**: Redis + VOICEVOX Engine 起動済み、`OPENAI_API_KEY` 設定済み。

```powershell
uv sync --extra stt --extra llm --extra tts

$env:L2_USE_REAL_STT      = "true"
$env:L2_USE_REAL_LLM      = "true"
$env:L2_USE_REAL_TTS      = "true"
$env:L2_TTS_PROVIDER      = "voicevox"
$env:L2_TTS_VOICE         = "89"
$env:L2_TTS_SPEAKER       = "Voidoll"
$env:L2_TTS_OUTPUT_DIR    = "./data/audio"
uv run python -m lab_lounge.emitter --audio-file samples/q1.wav
Remove-Item Env:\L2_USE_REAL_STT, Env:\L2_USE_REAL_LLM, Env:\L2_USE_REAL_TTS, Env:\L2_TTS_PROVIDER, Env:\L2_TTS_VOICE, Env:\L2_TTS_SPEAKER, Env:\L2_TTS_OUTPUT_DIR
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

### LangSmith 観測（Phase 5）

LangChain / LangGraph の内部 run / trace / latency を LangSmith で観測できる。  
**C2 は system of record、LangSmith は system of observation** として役割分離を保つ。

#### 有効化手順

```powershell
# 1. obs extra をインストール（初回のみ）
uv sync --extra obs

# 2. .env に追加
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=lsv2_pt_...
#    LANGSMITH_PROJECT=aibyss-lab-lounge
```

#### audio-file E2E を LangSmith で観測する最短手順

**前提**: Redis + VOICEVOX 起動済み、`.env` に `OPENAI_API_KEY` + `LANGSMITH_API_KEY` 設定済み。

```powershell
uv sync --extra stt --extra llm --extra tts --extra obs

$env:LANGSMITH_TRACING  = "true"
$env:LANGSMITH_PROJECT  = "aibyss-lab-lounge"
$env:L2_USE_REAL_STT    = "true"
$env:L2_USE_REAL_LLM    = "true"
$env:L2_USE_REAL_TTS    = "true"
$env:L2_TTS_PROVIDER    = "voicevox"
$env:L2_TTS_VOICE       = "89"
$env:L2_TTS_SPEAKER     = "Voidoll"
$env:L2_TTS_OUTPUT_DIR  = "./data/audio"
uv run python -m lab_lounge.emitter --audio-file samples/q1.wav
Remove-Item Env:\LANGSMITH_TRACING, Env:\LANGSMITH_PROJECT, `
  Env:\L2_USE_REAL_STT, Env:\L2_USE_REAL_LLM, Env:\L2_USE_REAL_TTS, `
  Env:\L2_TTS_PROVIDER, Env:\L2_TTS_VOICE, Env:\L2_TTS_SPEAKER, Env:\L2_TTS_OUTPUT_DIR
```

実行後、[https://smith.langchain.com](https://smith.langchain.com) → プロジェクト `aibyss-lab-lounge` を開くと  
1 往復の run / latency / token 数が `aibyss.trace_id` / `aibyss.stream_id` で検索できる。

> **tracing off のとき**: `LANGSMITH_TRACING=false`（既定）のままでも全機能が動く。  
> `observability.py` は metadata を組み立てるが外部に送信しない。

### マイク入力・スピーカー出力（Phase 4）

`run_once.py` は 1 回録音 → STT → LLM → TTS → デバイス再生までを 1 往復で実行する最小ランタイム。  
常時ストリーミングでなく、**1 往復終わると先に進む固定秒数録音方式**で実装している。

#### インストール

```powershell
# mic extra (録音 + 再生) を含む全 extra を同時インストール
uv sync --extra mic --extra stt --extra llm --extra tts
```

#### 実機確認手順 (LangSmith 無効)

**前提**: Redis + VOICEVOX 起動済み、`.env` に `OPENAI_API_KEY` 設定済み。

```powershell
# デフォルト 5 秒録音
uv run python -m lab_lounge.run_once

# 録音秒数を指定
uv run python -m lab_lounge.run_once --record-seconds 10

# TTS 再生をスキップ（ファイルパスのみ表示）
uv run python -m lab_lounge.run_once --no-play
```

#### 実機確認手順 (LangSmith 有効)

**前提**: Redis + VOICEVOX 起動済み、`.env` に `OPENAI_API_KEY` + `LANGSMITH_API_KEY` 設定済み。

```powershell
uv sync --extra mic --extra stt --extra llm --extra tts --extra obs

$env:LANGSMITH_TRACING = "true"
uv run python -m lab_lounge.run_once --record-seconds 5
Remove-Item Env:\LANGSMITH_TRACING
```

実行後、[https://smith.langchain.com](https://smith.langchain.com) → `aibyss-lab-lounge` で LLM run を確認できる。

#### デバイス一覧確認

```powershell
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

#### フォールバック動作

| 事象 | 内容 |
|---|---|
| 無音検出 | `処理を中断しました。` と表示して終了 |
| 録音デバイスエラー | 同上 |
| STT 失敗 | `処理を中断しました。` と表示。LLM / TTS には進まない |
| 再生失敗 (MP3 等) | 警告ログのみ。PipelineResult は返す |

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
| `tests/test_events.py` | スキーマ検証・ビルダー・それぞれの STT/LLM/TTS フィールド・ Guardrail G-1 確認 |
| `tests/test_pipeline.py` | 3 イベント順序・links 因果関係・seq 連番・ utterance_meta パススルー |
| `tests/test_bus.py` | XADD フィールド名・JSON 値・クライアント close |
| `tests/test_stt.py` | `transcribe_audio_file` ・ `_file_duration_ms` ・ ImportError 確認 |
| `tests/test_tts.py` | `synthesize` ・ provider ディスパッチ・ ImportError 確認 |
| `tests/test_llm.py` | `call_llm` ・ provider ディスパッチ・ LLMResult フィールド |
| `tests/test_graph.py` | `run_graph` ・ LangGraph ノード・ ImportError 確認 |
| `tests/test_observability.py` | `is_langsmith_enabled` ・ `build_run_metadata` ・ tracing on/off 分岐 |
| `tests/test_run_once.py` | 録音成功フロー・無音・録音失敗・ STT 失敗・ skip_playback・URI 変換 |

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
      emitter.py     # CLI エントリポイント (text / --audio-file)
      stt.py         # STT アダプタ (OpenAI Whisper API)
      llm.py         # LLM アダプタ (OpenAI)
      graph.py       # LangGraph 1-node グラフ
      tts.py         # TTS アダプタ (edge-tts / VOICEVOX)
      observability.py  # LangSmith 観測ヘルパー (tracing on/off 判定・run metadata 組み立て)
      audio_io.py       # 録音・再生アダプタ (sounddevice/soundfile ラッパー)
      run_once.py       # 1 往復会話 CLI (--mic / --record-seconds)
  tests/
    conftest.py
    test_events.py
    test_pipeline.py
    test_bus.py
    test_stt.py
    test_llm.py
    test_graph.py
    test_tts.py
    test_observability.py
    test_run_once.py
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