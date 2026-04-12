# ──────────────────────────────────────────────────────────────
# Lab Lounge — dev プロファイル run_loop 起動スクリプト
# ──────────────────────────────────────────────────────────────
# 本番履歴 (c2.db) を汚染せずに run_loop を試すための開発モード。
#
# 【env 読み込み順 (後勝ち)】
#   1. .env       (本番と共通の設定を継承: OPENAI_API_KEY, L2_USE_REAL_LLM 等)
#   2. .env.dev   (dev プロファイル固有の上書き: ストリーム名、C2 URL)
#   → プロセス環境変数 > .env.dev > .env の優先度
#
# 前提:
#   - Redis が起動中 (docker compose up -d redis)
#   - 本番 C2 が稼働中 (port 8100, c2.db)
#     → aibyss-coral-chronicle/scripts/run_prod.ps1
#   - 開発 C2 が稼働中 (port 8101, c2-dev.db, aibyss:events-dev)
#     → aibyss-coral-chronicle/scripts/run_dev.ps1
#   - .env.dev が存在する (cp .env.dev.example .env.dev)
#
# 使い方: PowerShell から実行
#   .\scripts\run_dev.ps1
#   .\scripts\run_dev.ps1 -MaxTurns 5
#   .\scripts\run_dev.ps1 -WakeBackend continuous -MaxTurns 3
#
# 【分離の保証】
#   このスクリプトで起動した L2 は:
#     - aibyss:events-dev のみに XADD (本番ストリームには一切書き込まない)
#     - 主 C2 = http://localhost:8101 (dev、書き込み先)
#     - 副 C2 = http://localhost:8100 (prod、read-only 参照)
#   よって発話データは本番 c2.db に混入せず、かつ過去の本番履歴も参照できる。
# ──────────────────────────────────────────────────────────────

param(
    [string]$WakeBackend = "continuous",
    [int]$MaxTurns = 3
)

$ErrorActionPreference = "Stop"

# ── dotenv パーサー (python-dotenv 準拠) ──
# 対応書式:
#   - KEY=value
#   - KEY = value       (前後空白)
#   - KEY="value"       (ダブルクォート)
#   - KEY='value'       (シングルクォート)
#   - export KEY=value  (bash 互換)
#   - KEY=value # comment   (クォート無し: 空白+# 以降をコメントとして除去)
#   - KEY="value # not"     (クォート内: # は文字列の一部として保持)
#   - KEY=value#literal     (# 直前が非空白: 文字列として残す)
function Import-DotEnvFile {
    param(
        [string]$Path,
        [switch]$Override
    )
    if (-not (Test-Path $Path)) { return }
    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        if (-not $line.Contains("=")) { continue }
        $eqIdx = $line.IndexOf("=")
        $key = $line.Substring(0, $eqIdx).Trim()
        $value = $line.Substring($eqIdx + 1).Trim()

        # クォート付きかチェック
        $isQuoted = $false
        if ($value.Length -ge 2) {
            $first = $value[0]
            if ($first -eq '"' -or $first -eq "'") {
                # 閉じクォートを探す
                $closeIdx = $value.IndexOf($first, 1)
                if ($closeIdx -gt 0) {
                    $value = $value.Substring(1, $closeIdx - 1)
                    $isQuoted = $true
                    # 閉じクォートの外側は全て無視 (コメント扱い)
                }
            }
        }

        if (-not $isQuoted) {
            # クォート無し: ASCII 空白 or タブ + # をコメント境界とする
            # `value # comment` → `value`
            # `value#literal`   → `value#literal` (# 直前が非空白なので保持)
            $spaceHash = $value.IndexOf(" #")
            $tabHash = $value.IndexOf("`t#")
            $commentIdx = -1
            if ($spaceHash -ge 0 -and $tabHash -ge 0) {
                $commentIdx = [Math]::Min($spaceHash, $tabHash)
            } elseif ($spaceHash -ge 0) {
                $commentIdx = $spaceHash
            } elseif ($tabHash -ge 0) {
                $commentIdx = $tabHash
            }
            if ($commentIdx -ge 0) {
                $value = $value.Substring(0, $commentIdx).TrimEnd()
            }
        }

        if ($Override -or -not [Environment]::GetEnvironmentVariable($key, "Process")) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$EnvFile = Join-Path $RepoRoot ".env.dev"
if (-not (Test-Path $EnvFile)) {
    Write-Error ".env.dev が見つかりません。先に 'cp .env.dev.example .env.dev' を実行してください。"
    exit 1
}

# ── 1. .env を base として読み込み (強制上書き) ──
# 【重要】-Override を指定する理由:
# セッション横断で PowerShell の環境変数が残り続けるため、過去の実行で
# 設定された値 (例: パーサーのバグで残った壊れた値) が新しい実行でも
# 再利用されてしまう事故を防ぐ。ファイルが常に source of truth となる。
# シェルレベルで一時上書きしたい場合は run_dev.ps1 実行後に設定する。
$BaseEnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $BaseEnvFile) {
    Write-Host "[lab-lounge/dev] .env を base として読み込み (API キー等を継承)..." -ForegroundColor Gray
    Import-DotEnvFile -Path $BaseEnvFile -Override
} else {
    Write-Host "[lab-lounge/dev] .env が無いので .env.dev のみで起動" -ForegroundColor Gray
}

# ── 2. .env.dev で上書き (プロファイル固有の設定、後勝ち) ──
Write-Host "[lab-lounge/dev] .env.dev で上書き中..." -ForegroundColor Cyan
Import-DotEnvFile -Path $EnvFile -Override

Write-Host ""
Write-Host "[lab-lounge/dev] 分離確認 (最終値):" -ForegroundColor Cyan
Write-Host "  REDIS_STREAM_KEY      = $env:REDIS_STREAM_KEY"
Write-Host "  L2_USE_C2_RETRIEVER   = $env:L2_USE_C2_RETRIEVER"
Write-Host "  L2_C2_URL (primary)   = $env:L2_C2_URL"
Write-Host "  L2_C2_URL_READONLY    = $env:L2_C2_URL_READONLY"
Write-Host "  L2_VAD_BACKEND        = $env:L2_VAD_BACKEND"
Write-Host "  L2_VAD_AGGRESSIVENESS = $env:L2_VAD_AGGRESSIVENESS"

# OPENAI_API_KEY が読み込まれているか簡易確認 (値そのものは表示しない)
if ($env:OPENAI_API_KEY) {
    $keyPreview = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(7, $env:OPENAI_API_KEY.Length))
    Write-Host "  OPENAI_API_KEY        = $keyPreview... (length=$($env:OPENAI_API_KEY.Length))"
} else {
    Write-Host "  OPENAI_API_KEY        = (未設定)" -ForegroundColor Yellow
}
Write-Host ""

# ── 分離違反の安全チェック ──
$ViolationFound = $false

if ($env:REDIS_STREAM_KEY -eq "aibyss:events") {
    Write-Warning "REDIS_STREAM_KEY が本番ストリーム 'aibyss:events' を指しています。dev 発話が本番 C2 に届いてしまいます。"
    $ViolationFound = $true
}
if ($env:L2_C2_URL -eq "http://localhost:8100") {
    Write-Warning "L2_C2_URL が本番ポート 8100 を指しています。dev 発話の読み取りで本番 DB を参照する設定になっています。"
    $ViolationFound = $true
}
if (-not $env:L2_USE_C2_RETRIEVER -or $env:L2_USE_C2_RETRIEVER.ToLower() -ne "true") {
    Write-Warning "L2_USE_C2_RETRIEVER が true ではありません。dev モードで C2Retriever を有効化するには .env.dev で true に設定してください。"
    # これは warning のみ (C2Retriever を使わない dev 実行も許容する)
}

if ($ViolationFound) {
    Write-Error ".env.dev で上記の項目を必ず上書きしてください (.env.dev.example を参照)。"
    exit 1
}

Write-Host "[lab-lounge/dev] run_loop を起動します (wake-backend=$WakeBackend max-turns=$MaxTurns)..." -ForegroundColor Green
uv run python -m lab_lounge.run_loop --wake-backend $WakeBackend --max-turns $MaxTurns
