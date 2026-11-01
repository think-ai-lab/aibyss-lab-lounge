#Requires -Version 7.5
# ──────────────────────────────────────────────────────────────
# Lab Lounge — prod プロファイル run_loop 起動スクリプト
# ──────────────────────────────────────────────────────────────
# 本番運用 / 実配信用の run_loop を起動する。
# .env のみ読み込み (.env.dev は触らない)。
#
# 【env 読み込み】
#   .env (本番設定: REDIS_STREAM_KEY=aibyss:events / L2_C2_URL=:8100 等)
#   → .env > プロセス環境変数 の優先度 (ファイルが常に勝つ)
#   REDIS_STREAM_KEY は未設定なら既定の aibyss:events へ向くので通す。aibyss:events-dev と空の値は拒否する。
#   親プロセスから継承した LLM の接続先 (ANTHROPIC_BASE_URL など) は、読み込みの前に外す。
#
# 前提:
#   - PowerShell 7.5 以上 (pwsh)。共通処理が ?? などの PowerShell 7 の構文を使い、REDIS_STREAM_KEY= の検査には空の値の
#     環境変数が要る (7.4 以前は空の値を入れると環境変数ごと消える) ので、古い版は先頭の #Requires で原因を伝えて止める
#   - Redis が起動中 (aibyss-workspace で docker compose up -d redis)
#   - VOICEVOX が起動中 (port 15021)
#   - 本番 C2 が稼働中 (port 8100, c2.db)。検索に使わない設定なら、止まっていても黄色で知らせて起動する
#       → aibyss-coral-chronicle/scripts/run_prod.ps1
#   - 本番 V2 が稼働中 (port 3200。別の URL なら L2_V2_URL で指す)
#       → aibyss-nautilus-v2/scripts/run_prod.ps1
#   - .env が存在する (cp .env.example .env)
#
# 【起動前の確認】
#   run_loop の前に preflight --static を流し、C2 と V2 の /health を確かめる。失敗があれば赤く出して exit 1 で止まる。
#   /health の stream も publish 先と比べる。V2 のポートは dev と prod で同じなので、dev の V2 が残っていても 200 を返すため。
#   C2 は設定によらず読む (検索に使わなくても全発話を記録するので)。検索に使わない設定なら止めず、黄色で知らせる
#   (発話は Redis に残り、C2 を戻せば追いつく)。
#   OBS が別の PC (L2_OBS_WS_URL が loopback でない) なら、LAN からの接続が 0 のときに黄色で知らせる。
#   C2 も V2 も置かない構成では -SkipHealthCheck で /health の確認だけを省ける。
#
# 使い方: PowerShell から実行
#   .\scripts\run_prod.ps1
#   .\scripts\run_prod.ps1 -WakeBackend continuous
#   .\scripts\run_prod.ps1 -MaxTurns 5         # 5 ターンで終了 (デフォルト 0 = 無制限)
#   .\scripts\run_prod.ps1 -SkipHealthCheck    # C2 も V2 も置かない構成
#
# 【dev/prod の違い】
#   prod は .env のみ読み込み aibyss:events ストリームに発話を publish する。
#   c2.db (本番履歴) に記録される点が dev (c2-dev.db) との最大の差。
# ──────────────────────────────────────────────────────────────

param(
    [string]$WakeBackend = "continuous",
    [int]$MaxTurns = 0,  # 0 = 無制限 (run_loop の --max-turns を渡さない)
    # C2 と V2 の /health の確認を省く。preflight --static は省かない。
    [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "run_common.ps1")

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Error ".env が見つかりません。先に 'cp .env.example .env' を実行してください。"
    exit 1
}

# ── 継承した LLM の接続先を外す (.env を読む前に。理由は Remove-InheritedLlmVar を参照) ──
Remove-InheritedLlmVar -Tag "lab-lounge/prod"

# ── .env を読み込み (強制上書き) ──
# 【重要】-Override を指定する理由は run_dev.ps1 と同じ。
# セッション横断の env 残留を防ぎ、ファイルを source of truth とする。
Write-Host "[lab-lounge/prod] .env を読み込み中..." -ForegroundColor Cyan
Import-DotEnvFile -Path $EnvFile -Override

Write-Host ""
Write-Host "[lab-lounge/prod] 起動設定 (最終値):" -ForegroundColor Cyan
Write-Host "  REDIS_STREAM_KEY      = $env:REDIS_STREAM_KEY"
# WHY: 起動の出力は貼られることがあるので、/health の行と同じく host を伏せる (理由は Format-ServiceEndpoint)。
Write-Host "  L2_C2_URL             = $(if ($env:L2_C2_URL) { Format-ServiceEndpoint $env:L2_C2_URL })"
Write-Host "  L2_USE_REAL_LLM       = $env:L2_USE_REAL_LLM"
Write-Host "  L2_USE_REAL_TTS       = $env:L2_USE_REAL_TTS"
Write-Host "  L2_USE_REAL_STT       = $env:L2_USE_REAL_STT"
Write-Host "  LANGSMITH_TRACING     = $env:LANGSMITH_TRACING"
Write-Host "  LANGSMITH_PROJECT     = $env:LANGSMITH_PROJECT"
Write-Host "  L2_VAD_BACKEND        = $env:L2_VAD_BACKEND"
Write-Host "  L2_VAD_AGGRESSIVENESS = $env:L2_VAD_AGGRESSIVENESS"
Write-KeyStatus -Name "OPENAI_API_KEY" -Required
Write-KeyStatus -Name "LANGSMITH_API_KEY"
Write-Host ""

# ── 安全チェック (prod モードに不適切な設定を検知) ──
$ViolationFound = $false

if ($env:REDIS_STREAM_KEY -eq "aibyss:events-dev") {
    Write-Warning "REDIS_STREAM_KEY が dev ストリーム 'aibyss:events-dev' を指しています。prod モードでは 'aibyss:events' が期待値です。"
    $ViolationFound = $true
}
# WHY: 未設定は bus の既定の aibyss:events へ向くので、prod では正しい。空の値は既定へ戻らず空の名前へ publish する
#      (bus._stream_key) ので、C2 にも HUD にも届かない。
if ($null -ne $env:REDIS_STREAM_KEY -and [string]::IsNullOrWhiteSpace($env:REDIS_STREAM_KEY)) {
    Write-Warning "REDIS_STREAM_KEY が空です。C2 にも HUD にも発話が届きません。行を消すか 'aibyss:events' を指定してください。"
    $ViolationFound = $true
}
# WHY: URL の完全一致だと、ホスト名を 127.0.0.1 へ変えただけで無言で効かなくなる。
#      dev と prod を分けているのはポートなので、ポートで判定する。
if ($env:L2_C2_URL -match ':8101/?$') {
    Write-Warning "L2_C2_URL が dev C2 ポート 8101 を指しています。prod モードでは 8100 が期待値です。"
    $ViolationFound = $true
}

if ($ViolationFound) {
    Write-Error ".env を本番設定に修正してください (.env.example を参照)。"
    exit 1
}

# ── 起動前の確認 ──
# WHY: 1 つずつ止めると、直して流し直すたびに次の失敗を知ることになる。全部見てから止める。
#      --frozen は配信の直前に解決器を走らせないため。--all-extras は点検の版の検査が全 extra を見るのと揃えるため。
Write-Host "[lab-lounge/prod] 起動前の点検 (preflight --static)..." -ForegroundColor Cyan
$StartupFailed = $false
uv run --frozen --all-extras python -m lab_lounge.preflight --static --profile prod
if ($LASTEXITCODE -ne 0) {
    Write-Host "[lab-lounge/prod] preflight が失敗しました (exit $LASTEXITCODE)。上の FAIL の行を直してください。" -ForegroundColor Red
    $StartupFailed = $true
}
if ($SkipHealthCheck) {
    Write-Host "[lab-lounge/prod] -SkipHealthCheck: C2 と V2 の /health の確認を省きます。" -ForegroundColor Yellow
} elseif (-not (Test-ServicesHealthy -Tag "lab-lounge/prod")) {
    $StartupFailed = $true
}
if ($StartupFailed) {
    Write-Host "[lab-lounge/prod] 起動を中止します。" -ForegroundColor Red
    exit 1
}

# ── run_loop 起動 ──
Write-Host "[lab-lounge/prod] run_loop を起動します (wake-backend=$WakeBackend)..." -ForegroundColor Green
if ($MaxTurns -gt 0) {
    Write-Host "  max-turns=$MaxTurns (有限)" -ForegroundColor Gray
    uv run --frozen --all-extras python -m lab_lounge.run_loop --wake-backend $WakeBackend --max-turns $MaxTurns
} else {
    Write-Host "  max-turns=無制限 (Ctrl+C で停止)" -ForegroundColor Gray
    uv run --frozen --all-extras python -m lab_lounge.run_loop --wake-backend $WakeBackend
}
exit $LASTEXITCODE
