# ──────────────────────────────────────────────────────────────
# Lab Lounge — run_dev.ps1 / run_prod.ps1 の共通処理
# ──────────────────────────────────────────────────────────────
# 両方のスクリプトから dot-source して使う。単体で流しても関数を定義するだけで何も起きない。
# ──────────────────────────────────────────────────────────────

# WHY: 呼び出し側は $LASTEXITCODE を見て赤い案内を出す。この設定が有効な環境だと、ネイティブコマンドの
#      非 0 が例外になり、案内を出す前にスクリプトが止まる。
$PSNativeCommandUseErrorActionPreference = $false

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

function Get-DotEnvName {
    <#
    .SYNOPSIS
        .env 形式のファイルにある変数の名前だけを返す。値は取り出さない。
    .DESCRIPTION
        行の読み方は Import-DotEnvFile と同じ (tests/test_run_scripts.py で python-dotenv と突き合わせる)。
    .PARAMETER Path
        読むファイル。なければ何も返さない
    .OUTPUTS
        [string] 名前。ファイルに書かれた順
    #>
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        $eqIdx = $line.IndexOf("=")
        if ($eqIdx -lt 0) { continue }
        $name = $line.Substring(0, $eqIdx).Trim()
        if ($name) { $name }
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

function Remove-DevOnlyEnvVar {
    <#
    .SYNOPSIS
        .env.dev にあって .env にない名前を環境から外し、外した名前を灰色で表示する (値は取り出さず、出さない)。
    .DESCRIPTION
        WHY: Import-DotEnvFile の -Override は、ファイルにある名前しか上書きしない。同じシェルで run_dev.ps1 の後に
             run_prod.ps1 を流すと、.env.dev だけにある L2_C2_URL_READONLY や L2_ENABLE_RAG が本番の run_loop へ渡り、
             .env.example が本番では厳禁とする副 C2 を読むことになる。
        run_prod.ps1 が .env を読む前に呼ぶこと。シェルで同じ名前を手で立てていても外れるので、外した名前を表示する。
    .PARAMETER Tag
        表示の先頭に付ける名前
    .PARAMETER EnvFile
        本番の設定のファイル (.env)
    .PARAMETER DevEnvFile
        dev の設定のファイル (.env.dev)。なければ何も外さない
    #>
    param(
        [Parameter(Mandatory)][string]$Tag,
        [Parameter(Mandatory)][string]$EnvFile,
        [Parameter(Mandatory)][string]$DevEnvFile
    )
    $inEnvFile = @(Get-DotEnvName -Path $EnvFile)
    # WHY: 名前はファイルから来るので、* がワイルドカードとして他の変数を巻き込まないように -LiteralPath で扱う。
    #      -notin は大文字と小文字を区別しない。Windows の環境変数も区別しないので揃う。
    $removed = foreach ($name in Get-DotEnvName -Path $DevEnvFile) {
        if ($name -notin $inEnvFile -and (Test-Path -LiteralPath "Env:$name")) {
            Remove-Item -LiteralPath "Env:$name"
            $name
        }
    }
    if ($removed) {
        Write-Host "[$Tag] .env.dev だけにある変数を外しました: $($removed -join ', ')" -ForegroundColor Gray
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

function Test-EnvFlag {
    <#
    .SYNOPSIS
        L2 の機能フラグを run_loop と同じ規則で読む (1 / true / yes。前後の空白は認めない)。
    .PARAMETER Name
        環境変数の名前
    .OUTPUTS
        [bool]
    #>
    param([Parameter(Mandatory)][string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    return ($value ?? "").ToLower() -in @("1", "true", "yes")
}

function Test-C2RetrieverEnabled {
    <#
    .SYNOPSIS
        run_loop が C2 の検索を呼ぶ設定かどうかを返す (preflight の _c2_retriever_enabled と同じ規則)。
    .DESCRIPTION
        L2_USE_C2_RETRIEVER は "true" だけを真と読む。C2 を読むのは retrieve_memory だけなので、
        L2_ENABLE_TOOLS か L2_ENABLE_RAG が偽なら、値によらず呼ばれない。
    .OUTPUTS
        [bool]
    #>
    $useC2 = ($env:L2_USE_C2_RETRIEVER ?? "").ToLower() -eq "true"
    return $useC2 -and (Test-EnvFlag "L2_ENABLE_TOOLS") -and (Test-EnvFlag "L2_ENABLE_RAG")
}

function Format-ServiceEndpoint {
    <#
    .SYNOPSIS
        表示用に URL を host:port へ縮める。loopback 以外の host は [HOST] に伏せる。
    .DESCRIPTION
        WHY: 起動の出力は貼られることがあるので、preflight と同じく手元のアドレスを出さない。
    .PARAMETER Url
        サービスのベース URL
    .OUTPUTS
        [string]
    #>
    param([Parameter(Mandatory)][string]$Url)
    $uri = $null
    if (-not [Uri]::TryCreate($Url, [UriKind]::Absolute, [ref]$uri)) {
        return "(URL の形が不正)"
    }
    $name = if ($uri.IsLoopback) { $uri.Host } else { "[HOST]" }
    return "${name}:$($uri.Port)"
}

function Get-ServiceHealth {
    <#
    .SYNOPSIS
        サービスの GET /health を 1 度だけ確かめる。
    .PARAMETER BaseUrl
        サービスのベース URL
    .OUTPUTS
        Ok (200 で status が ok か disabled), Detail (失敗の理由), Body (JSON の中身。読めなければ $null) を持つオブジェクト
    #>
    param([Parameter(Mandatory)][string]$BaseUrl)
    # 進捗バーの表示で起動の出力が崩れないようにする。この関数の中だけで効く。
    $ProgressPreference = "SilentlyContinue"
    try {
        # WHY: -NoProxy は、環境の HTTP_PROXY へ手元のサービスの確認を流さないため。
        #      -SkipHttpErrorCheck は、503 を例外にせず本文の理由を読むため。
        $response = Invoke-WebRequest -Uri "$($BaseUrl.TrimEnd('/'))/health" -TimeoutSec 3 -NoProxy -SkipHttpErrorCheck
    } catch {
        return [pscustomobject]@{ Ok = $false; Detail = "接続できません"; Body = $null }
    }
    $body = $null
    try {
        $body = $response.Content | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{ Ok = $false; Detail = "HTTP $($response.StatusCode)。JSON でない応答で、別のサービスかもしれません"; Body = $null }
    }
    if ($response.StatusCode -eq 200) {
        # WHY: C2 と V2 の 200 は status が ok か disabled だけ。空の本文や形の違う JSON まで緑にすると、
        #      ポートにいる別のサービスを見逃して、無音のまま配信に入る。
        if ($body.status -cin @("ok", "disabled")) {
            return [pscustomobject]@{ Ok = $true; Detail = ""; Body = $body }
        }
        return [pscustomobject]@{ Ok = $false; Detail = "HTTP 200。status が ok でも disabled でもなく、別のサービスかもしれません"; Body = $body }
    }
    # 503 の本文にある理由を添える。C2 は reason、V2 は redis と consumer を返す。
    $parts = @("HTTP $($response.StatusCode)")
    if ($null -ne $body.reason) { $parts += "reason=$($body.reason)" }
    if ($null -ne $body.redis) { $parts += "redis.sub=$($body.redis.sub) redis.pub=$($body.redis.pub)" }
    if ($null -ne $body.consumer) { $parts += "consumer=$($body.consumer)" }
    return [pscustomobject]@{ Ok = $false; Detail = ($parts -join " "); Body = $body }
}

function Write-ServiceHealthLine {
    <#
    .SYNOPSIS
        200 を返したサービスの状態を 1 行で表示する。disabled は動くが役目を果たさないので黄色にする。
    .PARAMETER Tag
        表示の先頭に付ける名前
    .PARAMETER Label
        サービスと接続先の表示
    .PARAMETER Status
        /health の status の値
    .PARAMETER Extra
        行末に足す情報
    #>
    param(
        [Parameter(Mandatory)][string]$Tag,
        [Parameter(Mandatory)][string]$Label,
        [string]$Status,
        [string]$Extra = ""
    )
    $line = "[$Tag] $Label`: $Status"
    if ($Extra) { $line += " $Extra" }
    if ($Status -eq "disabled") {
        Write-Host "$line (REDIS_URL が空なので、発話を受け取っていません)" -ForegroundColor Yellow
    } else {
        Write-Host $line -ForegroundColor Green
    }
}

function Get-ServiceProblem {
    <#
    .SYNOPSIS
        /health の結果から、L2 の発話を受け取れない理由を返す。
    .DESCRIPTION
        WHY: V2 のポートは dev と prod で同じなので、別のモードの V2 が残っていても 200 を返す。200 だけを見ると、
             本番の字幕と bubble が出ないのに緑のまま配信に入る。そこで、読んでいるストリームも publish 先と比べる。
             status が ok なのに stream がないのは、stream を返す前の古いプロセスで、どこを読んでいるかを確かめられない。
             disabled はバスを読んでいないので比べない (Write-ServiceHealthLine が黄色で知らせる)。
             未設定の REDIS_STREAM_KEY は、bus._stream_key と同じく本番の aibyss:events と読む。
    .PARAMETER Health
        Get-ServiceHealth の戻り値
    .OUTPUTS
        [string] 理由。問題がなければ空文字列
    #>
    param([Parameter(Mandatory)]$Health)
    if (-not $Health.Ok) {
        return $Health.Detail
    }
    if ($Health.Body.status -ceq "disabled") {
        return ""
    }
    $stream = $Health.Body.stream
    if ($null -eq $stream) {
        return "/health が stream を返しません。古いプロセスなので、止めて起動し直してください"
    }
    $expected = $env:REDIS_STREAM_KEY ?? "aibyss:events"
    # WHY: Redis のキーは大文字と小文字を区別するので、-cne で比べる。
    if ($stream -cne $expected) {
        return "'$stream' を読んでいます (L2 の publish 先は '$expected')。別のモードで動いているので、止めて起動し直してください"
    }
    return ""
}

function Test-ObsOnAnotherPc {
    <#
    .SYNOPSIS
        L2_OBS_WS_URL が別の PC を指しているかどうかを返す。
    .DESCRIPTION
        obs.py の _parse_ws_url と同じく、ws:// のない host:port の形も受ける。
        未設定と読めない形は $false にする (読めない形は preflight の obs の行が知らせる)。
    .OUTPUTS
        [bool]
    #>
    $url = $env:L2_OBS_WS_URL
    if (-not $url) {
        return $false
    }
    if (-not $url.Contains("://")) {
        $url = "ws://$url"
    }
    $uri = $null
    if (-not [Uri]::TryCreate($url, [UriKind]::Absolute, [ref]$uri)) {
        return $false
    }
    return -not $uri.IsLoopback
}

function Write-HudClientNotice {
    <#
    .SYNOPSIS
        HUD に繋がっている画面が足りないときに黄色で知らせる。OBS を後から起動することもあるので、止めない。
    .DESCRIPTION
        WHY: OBS が別の PC だと、同じ機械で /status を開いているだけで total は 0 にならず、OBS の PC が繋がっていなくても
             気づけない。L2_OBS_WS_URL が別の PC を指すときは、LAN からの接続の数で判定する。
             OBS の PC のアドレスは、起動の出力が貼られることがあるので出さない。
    .PARAMETER Tag
        表示の先頭に付ける名前
    .PARAMETER Clients
        V2 の /health の ws_clients
    #>
    param(
        [Parameter(Mandatory)][string]$Tag,
        [Parameter(Mandatory)]$Clients
    )
    if (Test-ObsOnAnotherPc) {
        if ([int]$Clients.lan -eq 0) {
            Write-Host "[$Tag] 別の PC の OBS が HUD に繋がっていません (lan=0)。OBS の bubble のソースと、README の「OBS を別の PC で動かす」を確かめてください" -ForegroundColor Yellow
        }
    } elseif ([int]$Clients.total -eq 0) {
        Write-Host "[$Tag] HUD に繋がっている画面がありません。OBS を使うなら bubble のソースが読み込まれているかを確かめてください (別の PC なら README の「OBS を別の PC で動かす」)" -ForegroundColor Yellow
    }
}

function Test-ServicesHealthy {
    <#
    .SYNOPSIS
        C2 と V2 の /health を確かめ、問題があれば赤く一覧を出す。検索に使わない設定の C2 は止めずに黄色で知らせる。
    .DESCRIPTION
        WHY: ポートが開いているだけでは分からない。C2 も V2 も Redis の consumer が止まっても listen を続けるので、
             歴史書が記録せず OBS の HUD も動かないまま配信に入り、無音で初めて気づくことになる。
    .PARAMETER Tag
        表示の先頭に付ける名前
    .OUTPUTS
        [bool] 問題がなければ $true
    #>
    param([Parameter(Mandatory)][string]$Tag)
    $problems = [System.Collections.Generic.List[string]]::new()

    $c2Url = if ($env:L2_C2_URL) { $env:L2_C2_URL } else { "http://127.0.0.1:8100" }
    $label = "C2 $(Format-ServiceEndpoint $c2Url)"
    $c2 = Get-ServiceHealth -BaseUrl $c2Url
    $problem = Get-ServiceProblem -Health $c2
    if (-not $problem) {
        Write-ServiceHealthLine -Tag $Tag -Label $label -Status $c2.Body.status
    } elseif (Test-C2RetrieverEnabled) {
        $problems.Add("$label`: $problem")
    } else {
        # WHY: C2 は検索のフラグと無関係に consumer で全発話を記録するので、検索に使わない設定でも確かめる。
        #      ただし止めない。L2 と V2 だけで動かす構成があり、発話は Redis に残るので C2 を戻せば追いつく。
        #      stream を返さない古いプロセスは記録しているかもしれないので、言い切らない。
        #      append_failing だけは別で、追いつかない。C2 は書けなかった発話も ack し、以後は新しい発話しか読まないので、
        #      DB を直しても読み直さない。それでも赤で止めないのは、原因を直しても次の発話が届くまで 503 が残り、
        #      止めると L2 を起動できなくなるから。
        $effect = if ($c2.Ok -and $null -eq $c2.Body.stream) { "歴史書が記録しているかを確かめられません" } else { "歴史書が記録していません" }
        $recovery = if ($c2.Body.reason -ceq "append_failing") {
            "C2 は書けなかった発話を ack するので、DB を直しても自動では追いつきません。発話は Redis のストリームには残ります"
        } else {
            "発話は Redis に残り、C2 を戻せば追いつきます"
        }
        Write-Host "[$Tag] $label`: $problem。検索に使わない設定なので止めませんが、$effect。$recovery" -ForegroundColor Yellow
    }

    $v2Url = if ($env:L2_V2_URL) { $env:L2_V2_URL } else { "http://127.0.0.1:3200" }
    $label = "V2 $(Format-ServiceEndpoint $v2Url)"
    $v2 = Get-ServiceHealth -BaseUrl $v2Url
    $problem = Get-ServiceProblem -Health $v2
    if ($problem) {
        $problems.Add("$label`: $problem")
    } else {
        $clients = $v2.Body.ws_clients
        $extra = if ($null -ne $clients) { "ws_clients total=$($clients.total) loopback=$($clients.loopback) lan=$($clients.lan)" } else { "" }
        Write-ServiceHealthLine -Tag $Tag -Label $label -Status $v2.Body.status -Extra $extra
        if ($null -ne $clients) {
            Write-HudClientNotice -Tag $Tag -Clients $clients
        }
    }

    if ($problems.Count -eq 0) {
        return $true
    }
    Write-Host "[$Tag] 前提サービスの確認に失敗しました:" -ForegroundColor Red
    foreach ($problem in $problems) {
        Write-Host "  - $problem" -ForegroundColor Red
    }
    Write-Host "  直してから流し直してください。C2 や V2 を置かない構成なら -SkipHealthCheck を付けて起動できます。" -ForegroundColor Red
    return $false
}
