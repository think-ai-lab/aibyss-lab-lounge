# ──────────────────────────────────────────────────────────────
# Lab Lounge — run_dev.ps1 / run_prod.ps1 の共通処理
# ──────────────────────────────────────────────────────────────
# 両方のスクリプトから dot-source して使う。単体で流しても関数を定義するだけで何も起きない。
# ──────────────────────────────────────────────────────────────

function Import-DotEnvFile {
    <#
    .SYNOPSIS
        .env 形式のファイルをプロセスの環境変数へ読み込む (python-dotenv 準拠)。
    .DESCRIPTION
        対応書式:
          - KEY=value
          - KEY = value       (前後空白)
          - KEY="value"       (ダブルクォート)
          - KEY='value'       (シングルクォート)
          - export KEY=value  (bash 互換)
          - KEY=value # comment   (クォート無し: 空白+# 以降をコメントとして除去)
          - KEY="value # not"     (クォート内: # は文字列の一部として保持)
          - KEY=value#literal     (# 直前が非空白: 文字列として残す)
    .PARAMETER Path
        読み込むファイル。なければ何もしない
    .PARAMETER Override
        既存の環境変数を上書きする
    #>
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
