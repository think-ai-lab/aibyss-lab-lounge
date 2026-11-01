# ──────────────────────────────────────────────────────────────
# Lab Lounge — run_dev.ps1 / run_prod.ps1 の共通処理
# ──────────────────────────────────────────────────────────────
# 両方のスクリプトから dot-source して使う。単体で流しても関数を定義するだけで何も起きない。
# ──────────────────────────────────────────────────────────────

# 親プロセスから継承すると LLM の向き先や認証を変えてしまう環境変数。
# lab_lounge.preflight の INHERITED_LLM_VARS と同じ一覧にする (tests/test_run_scripts.py で突き合わせる)。
$script:InheritedLlmVars = @(
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "GOOGLE_GEMINI_BASE_URL"
)

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

function Remove-InheritedLlmVar {
    <#
    .SYNOPSIS
        親プロセスから継承した LLM の接続先と認証を環境から外し、外した名前を表示する (値は出さない)。
    .DESCRIPTION
        WHY: Claude Code などのツールから起動すると、ツール自身の接続先を引き継ぐ。preflight は自分の中でだけ
             外すので、同じシェルから起動する run_loop には残り、点検が緑でも本番は別の相手へ繋がる。
        .env を読む前に呼ぶこと。.env がこれらを指定しているときは残り、preflight が終了コード 2 で止める。
    .PARAMETER Tag
        表示の先頭に付ける名前
    #>
    param([Parameter(Mandatory)][string]$Tag)
    $removed = @($script:InheritedLlmVars | Where-Object { Test-Path "Env:$_" })
    foreach ($name in $removed) {
        Remove-Item "Env:$name"
    }
    if ($removed) {
        Write-Host "[$Tag] 継承した変数を外しました: $($removed -join ', ')" -ForegroundColor Yellow
    }
}

function Write-KeyStatus {
    <#
    .SYNOPSIS
        秘匿値の有無だけを set / unset で表示する。
    .DESCRIPTION
        WHY: 起動の出力は配信中のコンソールや記録に貼られることがある。先頭の数文字でも、公開すればキーの手がかりになる。
    .PARAMETER Name
        環境変数の名前
    .PARAMETER Required
        unset のときに黄色で目立たせる
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [switch]$Required
    )
    $isSet = [bool][Environment]::GetEnvironmentVariable($Name, "Process")
    $line = "  {0,-21} = {1}" -f $Name, $(if ($isSet) { "set" } else { "unset" })
    if ($Required -and -not $isSet) {
        Write-Host $line -ForegroundColor Yellow
    } else {
        Write-Host $line
    }
}
