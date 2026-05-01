# ──────────────────────────────────────────────────────────────
# Lab Lounge — prod プロファイル run_loop 起動スクリプト
# ──────────────────────────────────────────────────────────────
# 本番運用 / 実配信用の run_loop を起動する。
# .env のみ読み込み (.env.dev は触らない)。
#
# 【env 読み込み】
#   .env (本番設定: REDIS_STREAM_KEY=aibyss:events / L2_C2_URL=:8100 等)
#   → プロセス環境変数 > .env の優先度
#
# 前提:
#   - Redis が起動中 (aibyss-workspace で docker compose up -d redis)
#   - VOICEVOX が起動中 (port 50021)
#   - 本番 C2 が稼働中 (port 8100, c2.db)
#       → aibyss-coral-chronicle/scripts/run_prod.ps1
#   - 本番 V2 が稼働中 (推奨、配信時に必須)
#       → aibyss-nautilus-v2/scripts/run_prod.ps1
#   - .env が存在する (cp .env.example .env)
#
# 使い方: PowerShell から実行
#   .\scripts\run_prod.ps1
#   .\scripts\run_prod.ps1 -WakeBackend continuous
#   .\scripts\run_prod.ps1 -MaxTurns 5         # 5 ターンで終了 (デフォルト 0 = 無制限)
#
# 【dev/prod の違い】
#   prod は .env のみ読み込み aibyss:events ストリームに発話を publish する。
#   c2.db (本番履歴) に記録される点が dev (c2-dev.db) との最大の差。
# ──────────────────────────────────────────────────────────────

param(
    [string]$WakeBackend = "continuous",
    [int]$MaxTurns = 0  # 0 = 無制限 (run_loop の --max-turns を渡さない)
)

$ErrorActionPreference = "Stop"

# ── dotenv パーサー (run_dev.ps1 と同一ロジック) ──
# 対応書式: KEY=value / KEY="value" / KEY='value' / export KEY=value /
#           KEY=value # comment / KEY="value # keep"
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

        $isQuoted = $false
        if ($value.Length -ge 2) {
            $first = $value[0]
            if ($first -eq '"' -or $first -eq "'") {
                $closeIdx = $value.IndexOf($first, 1)
                if ($closeIdx -gt 0) {
                    $value = $value.Substring(1, $closeIdx - 1)
                    $isQuoted = $true
                }
            }
        }

        if (-not $isQuoted) {
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

$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Error ".env が見つかりません。先に 'cp .env.example .env' を実行してください。"
    exit 1
}

# ── .env を読み込み (強制上書き) ──
# 【重要】-Override を指定する理由は run_dev.ps1 と同じ。
# セッション横断の env 残留を防ぎ、ファイルを source of truth とする。
Write-Host "[lab-lounge/prod] .env を読み込み中..." -ForegroundColor Cyan
Import-DotEnvFile -Path $EnvFile -Override

Write-Host ""
Write-Host "[lab-lounge/prod] 起動設定 (最終値):" -ForegroundColor Cyan
Write-Host "  REDIS_STREAM_KEY      = $env:REDIS_STREAM_KEY"
Write-Host "  L2_C2_URL             = $env:L2_C2_URL"
Write-Host "  L2_USE_REAL_LLM       = $env:L2_USE_REAL_LLM"
Write-Host "  L2_USE_REAL_TTS       = $env:L2_USE_REAL_TTS"
Write-Host "  L2_USE_REAL_STT       = $env:L2_USE_REAL_STT"
Write-Host "  LANGSMITH_TRACING     = $env:LANGSMITH_TRACING"
Write-Host "  LANGSMITH_PROJECT     = $env:LANGSMITH_PROJECT"
Write-Host "  L2_VAD_BACKEND        = $env:L2_VAD_BACKEND"
Write-Host "  L2_VAD_AGGRESSIVENESS = $env:L2_VAD_AGGRESSIVENESS"

# OPENAI_API_KEY が読み込まれているか簡易確認 (値そのものは表示しない)
if ($env:OPENAI_API_KEY) {
    $keyPreview = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(7, $env:OPENAI_API_KEY.Length))
    Write-Host "  OPENAI_API_KEY        = $keyPreview... (length=$($env:OPENAI_API_KEY.Length))"
} else {
    Write-Host "  OPENAI_API_KEY        = (未設定)" -ForegroundColor Yellow
}
if ($env:LANGSMITH_API_KEY) {
    $keyPreview = $env:LANGSMITH_API_KEY.Substring(0, [Math]::Min(10, $env:LANGSMITH_API_KEY.Length))
    Write-Host "  LANGSMITH_API_KEY     = $keyPreview... (length=$($env:LANGSMITH_API_KEY.Length))"
}
Write-Host ""

# ── 安全チェック (prod モードに不適切な設定を検知) ──
$ViolationFound = $false

if ($env:REDIS_STREAM_KEY -eq "aibyss:events-dev") {
    Write-Warning "REDIS_STREAM_KEY が dev ストリーム 'aibyss:events-dev' を指しています。prod モードでは 'aibyss:events' が期待値です。"
    $ViolationFound = $true
}
if ($env:L2_C2_URL -eq "http://localhost:8101") {
    Write-Warning "L2_C2_URL が dev C2 ポート 8101 を指しています。prod モードでは 8100 が期待値です。"
    $ViolationFound = $true
}

if ($ViolationFound) {
    Write-Error ".env を本番設定に修正してください (.env.example を参照)。"
    exit 1
}

# ── run_loop 起動 ──
Write-Host "[lab-lounge/prod] run_loop を起動します (wake-backend=$WakeBackend)..." -ForegroundColor Green
if ($MaxTurns -gt 0) {
    Write-Host "  max-turns=$MaxTurns (有限)" -ForegroundColor Gray
    uv run python -m lab_lounge.run_loop --wake-backend $WakeBackend --max-turns $MaxTurns
} else {
    Write-Host "  max-turns=無制限 (Ctrl+C で停止)" -ForegroundColor Gray
    uv run python -m lab_lounge.run_loop --wake-backend $WakeBackend
}
