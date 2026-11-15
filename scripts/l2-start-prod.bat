@echo off
REM ----------------------------------------------
REM L2 (lab-lounge run_loop) prod start - for Stream Deck
REM `start` opens a new console, so the window stays visible
REM even if the caller (Stream Deck) hides its own window.
REM `cmd /k` keeps the messages on screen after an error exit.
REM ----------------------------------------------
start "L2 prod (lab-lounge)" cmd /k pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_prod.ps1" -WakeBackend bg-continuous -MaxTurns 0
