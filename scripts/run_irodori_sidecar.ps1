# ──────────────────────────────────────────────────────────────
# Lab Lounge — Irodori-TTS サイドカー起動スクリプト
# ──────────────────────────────────────────────────────────────
# irodori-tts の HTTP サイドカー (sidecar/irodori_server.py) を irodori 専用の
# uv venv で起動する。L2 本体 (lab-lounge の venv) とは別環境で torch + CUDA を
# 使うため、--directory で irodori リポを指定して uv run する。
#
# 【前提】
#   - irodori-tts が T:\irodori-tts\Irodori-TTS に clone + uv sync 済み
#     (CUDA バックエンド extra で同期されていること)
#   - weights が HF_HOME (既定 T:\irodori-tts\hf-cache) に DL 済み
#   - GPU (CUDA) が利用可能
#
# 【使い方】
#   .\scripts\run_irodori_sidecar.ps1                       # vd を 18080 で起動
#   .\scripts\run_irodori_sidecar.ps1 -Port 50081 -Models vd
#
# 【環境変数による上書き (任意)】
#   L2_IRODORI_REPO     : irodori リポのパス (既定 T:\irodori-tts\Irodori-TTS)
#   L2_IRODORI_HF_HOME  : weights キャッシュ (既定 T:\irodori-tts\hf-cache)
#
# 通常は aibyss-workspace の start-services.ps1 (-Irodori) から別ウィンドウで起動される。
# ──────────────────────────────────────────────────────────────

param(
    [int]$Port = 18080,
    [string]$Models = "vd"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot                    # aibyss-lab-lounge
$SidecarScript = Join-Path $RepoRoot "sidecar\irodori_server.py"

# irodori リポ / HF キャッシュは環境変数で上書き可。未設定なら既定パス。
$IrodoriRepo = if ($env:L2_IRODORI_REPO) { $env:L2_IRODORI_REPO } else { "T:\irodori-tts\Irodori-TTS" }
$HfHome      = if ($env:L2_IRODORI_HF_HOME) { $env:L2_IRODORI_HF_HOME } else { "T:\irodori-tts\hf-cache" }

# ── 事前チェック (第三者が立ち上げやすいよう、失敗時に原因を明示) ──
if (-not (Test-Path $SidecarScript)) {
    Write-Error "サイドカースクリプトが見つかりません: $SidecarScript"
    exit 1
}
if (-not (Test-Path $IrodoriRepo)) {
    Write-Error "irodori リポが見つかりません: $IrodoriRepo (L2_IRODORI_REPO で上書き可)"
    exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv が PATH にありません。https://docs.astral.sh/uv/ を参照してください。"
    exit 1
}

# irodori_server.py が irodori_tts を import するためのパス + weights キャッシュ。
$env:HF_HOME = $HfHome
$env:L2_IRODORI_REPO = $IrodoriRepo

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Irodori-TTS サイドカー起動" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  irodori repo : $IrodoriRepo"
Write-Host "  HF_HOME      : $HfHome"
Write-Host "  port         : $Port"
Write-Host "  models       : $Models"
Write-Host ""
Write-Host "  モデルロードに数十秒かかります。'起動完了' ログが出るまでお待ちください。" -ForegroundColor Yellow
Write-Host ""

# irodori の venv で起動。--no-sync で CUDA バックエンド extra を維持したまま実行する
# (uv run のデフォルト同期は backend extra を落としてしまうため。bench と同じ流儀)。
uv run --directory $IrodoriRepo --no-sync python $SidecarScript --port $Port --models $Models
