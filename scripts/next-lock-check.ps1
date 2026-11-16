#Requires -Version 7.5
# ──────────────────────────────────────────────────────────────
# Lab Lounge — 次の lock で契約テストを流す（校正）
# ──────────────────────────────────────────────────────────────
# 依存を上げた lock で、tests/contract（偽の LLM サーバを使う契約テスト）が何を赤くするかを、
# 実リポの .venv と uv.lock に触らずに確かめる。
#
# 【流れ】
#   1. git archive HEAD で、commit 済みの中身だけを短いパス（%TEMP%\l2-nlc-<時刻>）へ写す
#      WHY 短いパス: Windows の古い既定のパスの上限（260 文字）に、venv の中の深いパスが当たらないように
#      WHY zip: GNU tar は C: をホスト名と読むので、展開は Expand-Archive で行う
#   2. 写しで uv lock --upgrade（-UpgradePackage で一部だけ、-NoUpgrade で今の lock のまま）
#   3. 写しで uv sync --frozen --all-extras
#   4. pytest tests/contract を、通常と --runxfail の 2 回流す
#        通常:       挙動のテストは緑、方針のテストは xfail が期待（今の lock）
#        --runxfail: xfail の印を外して流す。方針のテストが、今も本当に赤いかを見る
#   5. テストごとの結果と、主なパッケージの版（実リポの lock → 写しの lock）を表にして出す
#   6. 写しを消す（-KeepCopy で残す）
#
# 【外への作用】
#   - uv lock --upgrade は PyPI から依存の情報と配布物を取る。uv のキャッシュにあるものは取り直さない
#   - -NoUpgrade -Offline なら、ネットワークを使わない（キャッシュに無ければ失敗する）
#   - 契約テストそのものは、conftest の socket の網の下で 127.0.0.1 の偽サーバだけに繋ぐ
#
# 前提:
#   - PowerShell 7.5 以上、git、uv
#   - 兄弟の aibyss-workspace（イベントのスキーマを読む）。無ければ -SchemaPath で指す
#
# 使い方:
#   .\scripts\next-lock-check.ps1 -NoUpgrade -Offline          # 今の lock（基準の表）
#   .\scripts\next-lock-check.ps1                              # 全部を上げた lock
#   .\scripts\next-lock-check.ps1 -UpgradePackage anthropic    # anthropic だけ上げた lock
#   .\scripts\next-lock-check.ps1 -ReportPath .\nlc.md         # 表を残す場所（既定は %TEMP%）
#
# 終了コード:
#   0 = 2 回の pytest が最後まで流れ、表を出せた（赤いテストがあっても 0。赤を読むのが目的なので）
#   1 = 写し、lock、sync、pytest の起動のどれかに失敗した（表は出ない）
# ──────────────────────────────────────────────────────────────

param(
    # 上げるパッケージの名前。指定すると、それだけを上げる（uv lock --upgrade-package）。
    [string[]]$UpgradePackage = @(),
    # lock を上げずに、今の lock のまま流す（基準の表を作る）。
    [switch]$NoUpgrade,
    # uv をネットワークなしで動かす。-NoUpgrade と組み合わせる。
    [switch]$Offline,
    # 写しを消さずに残す（中を調べたいとき）。
    [switch]$KeepCopy,
    # 表（Markdown）を書く場所。既定は %TEMP%\l2-nlc-<時刻>.md。
    [string]$ReportPath = "",
    # イベントのスキーマ。既定は兄弟の aibyss-workspace の中。
    [string]$SchemaPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
# WHY: uv や pytest は、赤いテストで 0 以外を返す。それを PowerShell の例外に変えると、表を作る前に落ちる。
# 終了コードは Invoke-Native が自分で見る。
$PSNativeCommandUseErrorActionPreference = $false

# 表に版を出すパッケージ。LLM の SDK、LangChain の層、HTTP クライアント。
$WatchedPackages = @(
    "anthropic", "openai", "google-genai",
    "langchain", "langchain-core", "langchain-anthropic", "langchain-openai", "langchain-google-genai",
    "langgraph", "httpx", "httpx2", "httpcore", "httpcore2", "pydantic"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

# 相対パスは、呼んだ場所からの相対として今のうちに絶対パスへ直す。
# WHY: 後で写しの中へ Push-Location するので、相対のままだと表やスキーマの探索が写しの中を指し、
#      表が写しと一緒に消える。
function Resolve-CallerPath {
    param([string]$Path)
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}
if ($ReportPath) { $ReportPath = Resolve-CallerPath $ReportPath }
if ($SchemaPath) { $SchemaPath = Resolve-CallerPath $SchemaPath }

if ($NoUpgrade -and $UpgradePackage.Count -gt 0) {
    Write-Host "[FAIL] -NoUpgrade と -UpgradePackage は同時に使えません" -ForegroundColor Red
    exit 1
}
if (-not $SchemaPath) {
    $SchemaPath = Join-Path (Split-Path -Parent $RepoRoot) "aibyss-workspace\specs\event-envelope-0.1.schema.json"
}
if (-not (Test-Path -LiteralPath $SchemaPath -PathType Leaf)) {
    Write-Host "[FAIL] イベントのスキーマが見つかりません: $SchemaPath（-SchemaPath で指してください）" -ForegroundColor Red
    exit 1
}

# WHY ミリ秒まで: 同じ秒に 2 回流しても、写しと表のパスがぶつからないように。
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) "l2-nlc-$Stamp"
$ZipPath = "$WorkDir.zip"
if (-not $ReportPath) {
    $ReportPath = "$WorkDir.md"
}

# 写しの venv のパスが長いと、古い Windows の上限（260 文字）に、venv の中の深いパスが当たる。
$VenvPath = Join-Path $WorkDir ".venv"
if ($VenvPath.Length -gt 100) {
    Write-Host "[WARN] 写しの venv のパスが 100 文字を超えます ($($VenvPath.Length) 文字)。TEMP を短い場所にしてください: $VenvPath" -ForegroundColor Yellow
}

# commit 済みの HEAD だけを流す。作業ツリーに未 commit の変更があっても、写しには入らない。
$dirty = & git -C $RepoRoot status --porcelain
if ($dirty) {
    Write-Host "[WARN] 作業ツリーに未 commit の変更があります。流すのは HEAD の中身だけで、この変更は含まれません。" -ForegroundColor Yellow
}

# 流している間だけ書き換える環境変数。呼んだシェルを汚さないよう、終わりに元へ戻す。
$ManagedEnv = @("VIRTUAL_ENV", "PYTEST_ADDOPTS", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "AIBYSS_SCHEMA_PATH")
$SavedEnv = @{}
foreach ($name in $ManagedEnv) {
    $SavedEnv[$name] = [System.Environment]::GetEnvironmentVariable($name)
}

function Invoke-Native {
    <#
    .SYNOPSIS
        外のコマンドを写しの中で流し、終了コードを返す。出力はそのまま画面へ流す。
    #>
    param([string]$FilePath, [string[]]$ArgumentList)
    Write-Host "  > $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkGray
    & $FilePath @ArgumentList | Out-Host
    return $LASTEXITCODE
}

function Read-LockVersions {
    <#
    .SYNOPSIS
        uv.lock の行から、[[package]] の name と version を読み、名前 → 版の辞書を返す。
    #>
    param([string[]]$Lines)
    $versions = @{}
    $name = $null
    foreach ($line in $Lines) {
        if ($line -eq "[[package]]") { $name = $null; continue }
        if ($line -match '^name = "(.+)"$') { $name = $Matches[1]; continue }
        if ($name -and $line -match '^version = "(.+)"$') { $versions[$name] = $Matches[1]; $name = $null }
    }
    return $versions
}

function Read-JUnit {
    <#
    .SYNOPSIS
        pytest の junit XML から、テストごとの結果（pass / FAIL / xfail / XPASS / skip）を読む。
    .NOTES
        xfail は <skipped type="pytest.xfail">、strict の XPASS は <failure> に「XPASS(strict)」として出る。
    #>
    param([string]$Path)
    $results = [ordered]@{}
    [xml]$doc = Get-Content -LiteralPath $Path -Raw
    foreach ($case in $doc.SelectNodes("//testcase")) {
        # 収集の失敗（import が壊れたなど）は classname が空で、name にモジュール名が入る。
        $file = if ($case.classname) { ($case.classname -split '\.')[-1] } else { "(collection)" }
        $id = "$file::$($case.name)"
        $failure = $case.SelectSingleNode("failure|error")
        $skipped = $case.SelectSingleNode("skipped")
        if ($failure) {
            $text = "$($failure.message) $($failure.InnerText)"
            $results[$id] = if ($text -match "XPASS\(strict\)") { "XPASS" } else { "FAIL" }
        } elseif ($skipped) {
            $results[$id] = if ($skipped.type -eq "pytest.xfail") { "xfail" } else { "skip" }
        } else {
            $results[$id] = "pass"
        }
    }
    return $results
}

$exitCode = 0
try {
    Write-Host "[1/5] HEAD を写す: $WorkDir" -ForegroundColor Cyan
    $code = Invoke-Native "git" @("-C", $RepoRoot, "archive", "--format=zip", "-o", $ZipPath, "HEAD")
    if ($code -ne 0) { throw "git archive が失敗しました (exit $code)" }
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $WorkDir
    Remove-Item -LiteralPath $ZipPath

    # 親のシェルから漏れると、写しではなく別の環境や別の設定で流れるもの。
    foreach ($name in @("VIRTUAL_ENV", "PYTEST_ADDOPTS", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH")) {
        [System.Environment]::SetEnvironmentVariable($name, $null)
    }
    $env:AIBYSS_SCHEMA_PATH = $SchemaPath
    $uvNet = @()
    if ($Offline) { $uvNet = @("--offline") }

    Push-Location -LiteralPath $WorkDir
    try {
        if ($NoUpgrade) {
            Write-Host "[2/5] lock は今のまま（-NoUpgrade）" -ForegroundColor Cyan
            $lockLabel = "今の lock"
        } else {
            $upgradeArgs = @("--upgrade")
            $lockLabel = "全部を上げた lock"
            if ($UpgradePackage.Count -gt 0) {
                $upgradeArgs = @($UpgradePackage | ForEach-Object { "--upgrade-package"; $_ })
                $lockLabel = "一部を上げた lock（$($UpgradePackage -join ', ')）"
            }
            Write-Host "[2/5] uv lock $($upgradeArgs -join ' ')" -ForegroundColor Cyan
            $code = Invoke-Native "uv" (@("lock") + $upgradeArgs + $uvNet)
            if ($code -ne 0) { throw "uv lock が失敗しました (exit $code)" }
        }

        Write-Host "[3/5] uv sync --frozen --all-extras" -ForegroundColor Cyan
        $code = Invoke-Native "uv" (@("sync", "--frozen", "--all-extras") + $uvNet)
        if ($code -ne 0) { throw "uv sync が失敗しました (exit $code)" }

        $runs = [ordered]@{ "通常" = @(); "--runxfail" = @("--runxfail") }
        $tables = [ordered]@{}
        $index = 0
        foreach ($label in $runs.Keys) {
            $index++
            $junit = Join-Path $WorkDir "contract-$index.xml"
            Write-Host "[4/5] pytest tests/contract ($label)" -ForegroundColor Cyan
            # --no-sync: 上の sync の結果をそのまま使う。-p no:cacheprovider: 写しに .pytest_cache を作らない。
            # --continue-on-collection-errors: import が壊れても、そのファイルだけを FAIL として表に出し、残りを流す
            # （SDK の更新で import が壊れることこそ、この表で見たい赤の 1 つ）。
            $pytestArgs = @("run", "--frozen", "--no-sync", "--all-extras", "pytest", "tests/contract", "-q",
                "-p", "no:cacheprovider", "--continue-on-collection-errors", "--junitxml=$junit") + $runs[$label]
            $code = Invoke-Native "uv" $pytestArgs
            # pytest の 1 は「赤いテストがある」で、表を作る。2 以上は、流れなかった（中断、収集の失敗、使い方の誤り）。
            if ($code -gt 1 -or -not (Test-Path -LiteralPath $junit)) { throw "pytest が最後まで流れませんでした (exit $code)" }
            $tables[$label] = Read-JUnit $junit
        }

        Write-Host "[5/5] 表を作る: $ReportPath" -ForegroundColor Cyan
        # WHY git show HEAD:uv.lock: 流したのは HEAD の中身。作業ツリーの uv.lock が未 commit で違っていても、
        # 「実リポの lock」の列は流した写しの元と同じ HEAD から読む。
        $headLock = & git -C $RepoRoot show "HEAD:uv.lock"
        if ($LASTEXITCODE -ne 0) { throw "git show HEAD:uv.lock が失敗しました (exit $LASTEXITCODE)" }
        $before = Read-LockVersions $headLock
        $after = Read-LockVersions (Get-Content -LiteralPath (Join-Path $WorkDir "uv.lock"))
        $head = (& git -C $RepoRoot rev-parse --short HEAD).Trim()

        $lines = [System.Collections.Generic.List[string]]::new()
        $lines.Add("# 次の lock の契約テスト（$lockLabel、HEAD $head、$Stamp）")
        $lines.Add("")
        $lines.Add("## 主な版")
        $lines.Add("")
        $lines.Add("| パッケージ | 実リポの lock | この lock |")
        $lines.Add("|---|---|---|")
        foreach ($name in $WatchedPackages) {
            $old = if ($before.ContainsKey($name)) { $before[$name] } else { "−" }
            $new = if ($after.ContainsKey($name)) { $after[$name] } else { "−" }
            if ($old -eq "−" -and $new -eq "−") { continue }
            $mark = if ($old -ne $new) { " **変化**" } else { "" }
            $lines.Add("| $name | $old | $new$mark |")
        }
        $lines.Add("")
        $lines.Add("## テストごとの結果")
        $lines.Add("")
        $lines.Add("| テスト | 通常 | --runxfail |")
        $lines.Add("|---|---|---|")
        $normal = $tables["通常"]
        $raw = $tables["--runxfail"]
        foreach ($id in $normal.Keys) {
            $lines.Add("| ``$id`` | $($normal[$id]) | $($raw[$id]) |")
        }
        $lines.Add("")
        $counts = $normal.Values | Group-Object | Sort-Object Name | ForEach-Object { "$($_.Name) $($_.Count)" }
        $lines.Add("通常の集計: $($counts -join '、')")

        $text = ($lines -join "`n") + "`n"
        Set-Content -LiteralPath $ReportPath -Value $text -Encoding utf8NoBOM -NoNewline
        Write-Host ""
        Write-Host $text
    } finally {
        Pop-Location
    }
} catch {
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    foreach ($name in $ManagedEnv) {
        [System.Environment]::SetEnvironmentVariable($name, $SavedEnv[$name])
    }
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    if ($KeepCopy) {
        Write-Host "写しを残しました: $WorkDir" -ForegroundColor Yellow
    } elseif (Test-Path -LiteralPath $WorkDir) {
        Remove-Item -LiteralPath $WorkDir -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $WorkDir) {
            Write-Host "写しを消しきれませんでした（開いているファイルがあるかもしれません）: $WorkDir" -ForegroundColor Yellow
        }
    }
}
exit $exitCode
