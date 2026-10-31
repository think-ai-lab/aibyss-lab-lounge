@echo off
REM ──────────────────────────────────────────────
REM L2 (lab-lounge run_loop) dev 起動 — Stream Deck 用
REM `start` で新規コンソールを明示生成するため、
REM 呼び出し元(Stream Deck)が窓を隠していても必ず可視で開く。
REM `cmd /k` によりエラー終了時もメッセージが窓に残る。
REM ──────────────────────────────────────────────
start "L2 dev (lab-lounge)" cmd /k pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dev.ps1" -WakeBackend bg-continuous -MaxTurns 0