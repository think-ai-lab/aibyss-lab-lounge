# Irodori-TTS サイドカー

L2 (lab-lounge) から **irodori-tts**（Flow Matching ニューラル TTS／VoiceDesign）を
使うための HTTP サイドカーサーバです（確定版は VoiceDesign 一本）。

## なぜサイドカーなのか

irodori は `torch` + CUDA + git 依存（`dacvae` / `silentcipher`）の重いスタックで、独自の uv venv を
持ちます。これを軽量な L2（Python 3.11／stdlib 中心）に取り込むと依存解決が脆くなり、GPU メモリも
常時占有してしまいます。そこで **VOICEVOX が Docker＋HTTP で分離されているのと同じ発想**で、irodori も
別プロセス＋HTTP で疎結合にします。

- モデルは起動時に 1 回だけロードしてウォーム保持（実測 **~0.9s/合成、RTF ~0.1** @ RTX 5090）。
- サーバはキャラクター／感情ロジックを持たない **generic な VoiceDesign エンドポイント**。声の同一性
  （self-ref アンカー `ref_wav` ＋ `caption` ＋ `seed`）も感情（本文末の正規絵文字 ＋ caption サフィックス）も
  呼出側（L2 の `tts._call_irodori`）が組み立てて送ります。

## 前提

- `run_irodori_sidecar.ps1` は PowerShell 7（`pwsh`）で動かす（Windows PowerShell 5.1 は BOM のない UTF-8 を ANSI として読むので、日本語の Windows では構文エラーで止まる）
- `irodori-tts` を clone して `uv sync --extra cu128`（CUDA バックエンド）する。既定の場所は作者の環境の `T:\irodori-tts\Irodori-TTS` なので、別の場所なら `L2_IRODORI_REPO` で指す
- weights を `HF_HOME` に取得する。スクリプトは `L2_IRODORI_HF_HOME`（既定 `T:\irodori-tts\hf-cache`）を `HF_HOME` へ入れるので、別の場所なら `L2_IRODORI_HF_HOME` で指す
  （`Irodori-TTS-500M-v3` / `Irodori-TTS-600M-v3-VoiceDesign` / DACVAE codec / tokenizer / silentcipher）
  `run_irodori_sidecar.ps1` は取得済みの weights だけを使う（`HF_HUB_OFFLINE=1`。上流の更新を無言で取り込むと調整済みの声が変わるため）。初回や更新するときは `-AllowDownload` を付ける
- CUDA GPU が利用可能

## 起動

通常は workspace のコントロール卓から:

```powershell
# aibyss-workspace
.\scripts\aibyss.ps1 start dev -Irodori     # Redis / VOICEVOX / irodori / C2 / V2
```

単体起動:

```powershell
# aibyss-lab-lounge
.\scripts\run_irodori_sidecar.ps1                      # vd を 18080 で起動
```

モデルロードに数十秒かかります。「起動完了」ログが出れば準備完了です。

## API

### `GET /health`

```json
{ "status": "ok", "models": ["vd"] }
```

### `POST /synthesize` → `audio/wav`（PCM16、`X-Sample-Rate: 48000`）

```json
{
  "mode": "vd",
  "text": "ごきげんよう、アビスメイトの皆さま。🤭",
  "caption": "理知的で聡明な、気品のある若い女性の声。…（＋感情サフィックス）",
  "ref_wav": "<reference_voices の絶対パス>\\mimi_ref.wav",
  "duration_scale": 1.0,
  "num_steps": 24,
  "t_schedule_mode": "sway",
  "cfg_scale_speaker": 5.0,
  "seed": 3
}
```

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `mode` | ✓ | `"vd"`（VoiceDesign） |
| `text` | ✓ | 合成テキスト。本文末に正規絵文字が付くことがある |
| `caption` | ✓ | 声質＋話し方を指示する自然文（＋感情サフィックス） |
| `ref_wav` | 確定版必須 | self-ref アンカーの絶対パス（無ければ no-ref。確定版は必ず送る） |
| `duration_scale` | | `max(0.85, 100/speed)`。`>1` で長く、`<1` で短く |
| `num_steps` | | サンプリングステップ（既定 24＝実走の床） |
| `t_schedule_mode` | | `sway`（既定）/ `linear` |
| `cfg_scale_speaker` | | 話者条件付け強度（既定 5.0） |
| `seed` | | 省略でランダム（確定版はキャラ毎の固定値を L2 が送る） |

```powershell
# 動作確認の例
curl -X POST http://127.0.0.1:18080/synthesize `
  -H "Content-Type: application/json" `
  -d '{"mode":"vd","text":"テストです。","caption":"気品のある若い女性の声。落ち着いて優雅に。","duration_scale":1.0}' `
  --output test.wav
```

## トラブルシュート

| 症状 | 原因 / 対処 |
|------|------------|
| 起動時 `CUDA ... is False` | GPU/ドライバ未認識。`nvidia-smi` を確認。irodori を `--extra cu128` で sync したか |
| `irodori_tts` が import できない | `L2_IRODORI_REPO` が irodori リポを指しているか確認 |
| 起動時に weights が見つからない（`LocalEntryNotFoundError` など） | まず起動時に表示される `HF_HOME` が取得済みのキャッシュを指しているか確かめる（違えば `L2_IRODORI_HF_HOME` で指す）。未取得なら `run_irodori_sidecar.ps1 -AllowDownload` で取得する |
| L2 から繋がらない | `L2_TTS_IRODORI_URL`（既定 `http://127.0.0.1:18080`）とポートが一致しているか |

設定の全体像は L2 の `.env.example` の irodori セクションを参照してください。
