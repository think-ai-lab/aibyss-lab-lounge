# ──────────────────────────────────────────────────────────────
# Lab Lounge — dev プロファイル run_loop 起動スクリプト
# ──────────────────────────────────────────────────────────────
# 本番履歴 (c2.db) を汚染せずに run_loop を試すための開発モード。
#
# 【env 読み込み順 (後勝ち)】
#   1. .env       (本番と共通の設定を継承: OPENAI_API_KEY, L2_USE_REAL_LLM 等)
#   2. .env.dev   (dev プロファイル固有の上書き: ストリーム名、C2 URL)
#   → .env.dev > .env > プロセス環境変数 の優先度 (ファイルが常に勝つ)
#   親プロセスから継承した LLM の接続先 (ANTHROPIC_BASE_URL など) は、読み込みの前に外す。
#
# 前提:
#   - Redis が起動中 (docker compose up -d redis)
#   - 開発 C2 が稼働中 (port 8101, c2-dev.db, aibyss:events-dev)
#     → aibyss-coral-chronicle/scripts/run_dev.ps1
#   - V2 が稼働中 (port 3200。別の URL なら L2_V2_URL で指す)
#     → aibyss-nautilus-v2/scripts/run_dev.ps1
#   - .env.dev が存在する (cp .env.dev.example .env.dev)
#
# 【起動前の確認】
#   run_loop の前に preflight --static を流し、C2 と V2 の /health を確かめる。
#   C2 は記憶の検索に使う設定のときだけ確かめる。失敗があれば赤く出して exit 1 で止まる。
#   C2 も V2 も置かない構成では -SkipHealthCheck で /health の確認だけを省ける。
#
# 【本番 C2 (副 C2) について】
#   本番 C2 (port 8100, c2.db) は過去の本番履歴を読むための副 C2 で、止まっていても起動できる。
#   止まっている間は、検索のたびに副 C2 の分が予算 (L2_C2_RETRIEVER_BUDGET_MS、既定 600 ms) で
#   打ち切りになり、本番の履歴なしで続く (WARN が出る)。
#   本番 C2 を動かさない期間は、.env.dev の L2_C2_URL_READONLY を空にすれば待たずに済む。
#
# 使い方: PowerShell から実行
#   .\scripts\run_dev.ps1
#   .\scripts\run_dev.ps1 -MaxTurns 5
#   .\scripts\run_dev.ps1 -MaxTurns 0                          # 無制限 (Ctrl+C で停止、prod と同じ挙動)
#   .\scripts\run_dev.ps1 -WakeBackend continuous -MaxTurns 3
#   .\scripts\run_dev.ps1 -SkipHealthCheck                     # C2 も V2 も置かない構成
#
# 【分離の保証】
#   このスクリプトで起動した L2 は:
#     - aibyss:events-dev のみに XADD (本番ストリームには一切書き込まない)
#     - 主 C2 = http://127.0.0.1:8101 (dev、書き込み先)
#     - 副 C2 = http://127.0.0.1:8100 (prod、read-only 参照)
#   よって発話データは本番 c2.db に混入せず、かつ過去の本番履歴も参照できる。
# ──────────────────────────────────────────────────────────────

param(
    [string]$WakeBackend = "continuous",
    # MaxTurns: dev の default は 3 ターン (暴走防止)。
    # 0 以下を指定すると --max-turns を渡さず run_loop 側で無制限になる (prod と仕様統一)。
    [int]$MaxTurns = 3,
    # C2 と V2 の /health の確認を省く。preflight --static は省かない。
    [switch]$SkipHealthCheck
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "run_common.ps1")

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$EnvFile = Join-Path $RepoRoot ".env.dev"
if (-not (Test-Path $EnvFile)) {
    Write-Error ".env.dev が見つかりません。先に 'cp .env.dev.example .env.dev' を実行してください。"
    exit 1
}

# ── 継承した LLM の接続先を外す (.env を読む前に。理由は Remove-InheritedLlmVar を参照) ──
Remove-InheritedLlmVar -Tag "lab-lounge/dev"

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
Write-KeyStatus -Name "OPENAI_API_KEY" -Required
Write-Host ""

# ── 分離違反の安全チェック ──
$ViolationFound = $false

# WHY: bus は未設定を本番の aibyss:events と読む (bus._stream_key) ので、行を書き忘れただけで dev の発話が
#      本番の正史へ入る。正史は追記のみで戻しにくいので、未設定も本番と同じく止める。
#      空の値は既定へ戻らず空の名前へ publish するので、C2 にも HUD にも届かない。これも止める。
if ([string]::IsNullOrWhiteSpace($env:REDIS_STREAM_KEY)) {
    Write-Warning "REDIS_STREAM_KEY が未設定か空です。未設定は本番 'aibyss:events' へ向き、空の値は C2 にも HUD にも届きません。"
    $ViolationFound = $true
} elseif ($env:REDIS_STREAM_KEY -eq "aibyss:events") {
    Write-Warning "REDIS_STREAM_KEY が本番ストリーム 'aibyss:events' を指しています。dev 発話が本番 C2 に届いてしまいます。"
    $ViolationFound = $true
}
# WHY: URL の完全一致だと、ホスト名を 127.0.0.1 へ変えただけで無言で効かなくなる。
#      dev と prod を分けているのはポートなので、ポートで判定する。
if ($env:L2_C2_URL -match ':8100/?$') {
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

# ── 起動前の確認 ──
# WHY: 1 つずつ止めると、直して流し直すたびに次の失敗を知ることになる。全部見てから止める。
#      --frozen は配信の直前に解決器を走らせないため。--all-extras は点検の版の検査が全 extra を見るのと揃えるため。
Write-Host "[lab-lounge/dev] 起動前の点検 (preflight --static)..." -ForegroundColor Cyan
$StartupFailed = $false
uv run --frozen --all-extras python -m lab_lounge.preflight --static --profile dev
if ($LASTEXITCODE -ne 0) {
    Write-Host "[lab-lounge/dev] preflight が失敗しました (exit $LASTEXITCODE)。上の FAIL の行を直してください。" -ForegroundColor Red
    $StartupFailed = $true
}
if ($SkipHealthCheck) {
    Write-Host "[lab-lounge/dev] -SkipHealthCheck: C2 と V2 の /health の確認を省きます。" -ForegroundColor Yellow
} elseif (-not (Test-ServicesHealthy -Tag "lab-lounge/dev")) {
    $StartupFailed = $true
}
if ($StartupFailed) {
    Write-Host "[lab-lounge/dev] 起動を中止します。" -ForegroundColor Red
    exit 1
}

Write-Host "[lab-lounge/dev] run_loop を起動します (wake-backend=$WakeBackend)..." -ForegroundColor Green
# WHY: prod と仕様を揃え、$MaxTurns が 0 以下なら --max-turns を渡さず
#      run_loop 側で無制限ループにする (Python 側 default = None = 無制限)。
#      run_dev.ps1 -MaxTurns 0 で「即終了」になる罠を解消し、明示的に
#      無制限を指定したいケース (長時間の手動実験) に対応する。
if ($MaxTurns -gt 0) {
    Write-Host "  max-turns=$MaxTurns (有限)" -ForegroundColor Gray
    uv run --frozen --all-extras python -m lab_lounge.run_loop --wake-backend $WakeBackend --max-turns $MaxTurns
} else {
    Write-Host "  max-turns=無制限 (Ctrl+C で停止)" -ForegroundColor Gray
    uv run --frozen --all-extras python -m lab_lounge.run_loop --wake-backend $WakeBackend
}
exit $LASTEXITCODE
