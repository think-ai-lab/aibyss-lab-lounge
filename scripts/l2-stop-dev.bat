# ──────────────────────────────────────────────
# L2 (lab_lounge.run_loop) 停止スクリプト — Stream Deck 用
# python.exe のうち、コマンドラインに run_loop を含む
# プロセスだけを狙い撃ちで停止する (C2 の python は巻き込まない)。
# 注意: dev/prod を区別しない。run_loop 全個体が対象。
# ──────────────────────────────────────────────
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_loop' }

if (-not $procs) {
    Write-Host "[l2-stop] run_loop プロセスは見つかりませんでした (既に停止)" -ForegroundColor Yellow
    exit 0
}

foreach ($p in $procs) {
    Write-Host "[l2-stop] PID $($p.ProcessId) を停止: $($p.CommandLine)"
    Stop-Process -Id $p.ProcessId -Force
}

Write-Host "[l2-stop] 完了" -ForegroundColor Green