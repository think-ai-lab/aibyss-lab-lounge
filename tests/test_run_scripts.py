"""
test_run_scripts.py — run_dev.ps1 / run_prod.ps1 の起動前の確認を確かめる

起動のスクリプトは配信の入口なので、ここが嘘をつくと無音の配信に入ってしまう。
実サービスには繋がない。スクリプトの写しを一時ディレクトリへ置き、uv は呼び出しを記録するだけの偽物に、
C2 と V2 はこのプロセスの中の偽サーバに替えて、PowerShell で最後まで流す。
"""

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import dotenv_values

from lab_lounge import bus, preflight

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
_RUN_SCRIPTS = ["run_dev.ps1", "run_prod.ps1"]

# WHY: Windows の subprocess は env の PATH で実行ファイルを探さないので、親のうちに絶対パスにしておく。
_PWSH = shutil.which("pwsh")
needs_pwsh = pytest.mark.skipif(_PWSH is None, reason="pwsh がない環境")
# Windows に標準で入っている古い版 (5.1)。誤ってこちらで開いたときの動きを確かめる。
_WINDOWS_POWERSHELL = shutil.which("powershell")

# 値が出力に現れないことを確かめる印。先頭だけ出ても見つかるように、頭に短い印を置く。
_KEY_VALUE = "zqxKEYVALUE-not-a-real-key"
_INHERITED_VALUE = "http://192.0.2.1:9/zqxINHERITED"

# 偽の uv。引数と、外れているはずの変数の残りを記録して、preflight と run_loop で別の終了コードを返す。
# 継承した LLM の変数のほかに見張る名前は、場面ごとに UV_STUB_WATCH で渡す（Scene.watch）。
_INHERITED_NAMES = " ".join(preflight.INHERITED_LLM_VARS)
_UV_CMD = f"""@echo off
>>"%UV_STUB_LOG%" echo ARGS %*
for %%N in ({_INHERITED_NAMES} %UV_STUB_WATCH%) do if defined %%N >>"%UV_STUB_LOG%" echo LEFT %%N
echo %* | findstr /c:"lab_lounge.preflight" >nul && exit /b %UV_STUB_PREFLIGHT_EXIT%
exit /b %UV_STUB_RUN_EXIT%
"""
_UV_SH = f"""#!/bin/sh
echo "ARGS $*" >> "$UV_STUB_LOG"
for name in {_INHERITED_NAMES} $UV_STUB_WATCH; do
    eval "value=\\${{$name+x}}"
    [ -n "$value" ] && echo "LEFT $name" >> "$UV_STUB_LOG"
done
case "$*" in
    *lab_lounge.preflight*) exit "$UV_STUB_PREFLIGHT_EXIT" ;;
esac
exit "$UV_STUB_RUN_EXIT"
"""


# WHY: Write-Host の色はパイプへ流すと失われる。止めるか知らせるかの違いは色で伝えているので、同じ名前の関数を先に
#      定義して行の先頭へ写す（PowerShell は同じ名前のコマンドレットより関数を先に探す）。
_SHOW_COLORS = (
    "function global:Write-Host { param([Parameter(Position = 0)]$Object, $ForegroundColor) "
    '[Console]::Out.WriteLine("<$ForegroundColor>$Object") }; '
)


# V2 に繋がっている画面が足りないときの案内。テストで見分けられるように、片方がもう片方を含まない文にしてある。
_TOTAL_NOTICE = "HUD に繋がっている画面がありません"
_LAN_NOTICE = "別の PC の OBS が HUD に繋がっていません"

# 検索に使わない C2 の黄色の知らせの結び。止まっていた間の発話は読み直せるが、書けなかった発話は読み直されない。
_C2_CATCHES_UP = "発話は Redis に残り、C2 を戻せば追いつきます"
_C2_DOES_NOT_CATCH_UP = "C2 は書けなかった発話を ack するので、DB を直しても自動では追いつきません。発話は Redis のストリームには残ります"


def _stream_of(script: str) -> str:
    """スクリプトの publish 先を返す。.env.example と .env.dev.example の値と同じ。"""
    return "aibyss:events-dev" if script == "run_dev.ps1" else "aibyss:events"


def _v2_ok(stream: str, *, total: int = 8, loopback: int = 0, lan: int = 8) -> dict:
    """正常な V2 の /health の本文を返す。"""
    return {"status": "ok", "stream": stream, "ws_clients": {"total": total, "loopback": loopback, "lan": lan}}


class FakeService:
    """GET /health だけに答える偽サーバ。"""

    def __init__(self) -> None:
        self.status = 200
        # None はテストが応答を決めていない印。Scene が流すスクリプトに合う正常な応答で埋める。
        self.body: bytes | None = None
        self.delay = 0.0
        self.paths: list[str] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server の規約
                # WHY: self.path は先頭の // を / へ縮めるので、URL の末尾の / の扱いを確かめられない。送られた行から読む。
                service.paths.append(self.raw_requestline.split()[1].decode("ascii"))
                time.sleep(service.delay)
                try:
                    self.send_response(service.status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(service.body)
                except OSError:
                    pass  # 待たせる場面では、相手が先に切っている


            def log_message(self, *_args) -> None:
                """テストの出力へアクセスログを混ぜない。"""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def answer(self, status: int, body: dict | str) -> None:
        """次からの応答を変える。"""
        self.status = status
        self.body = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")

    def close(self) -> None:
        """サーバを止める。"""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class Scene:
    """スクリプトの写し、偽の uv、偽の C2 と V2 を置いた一時ディレクトリ。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.c2 = FakeService()
        self.v2 = FakeService()
        self.preflight_exit = 0
        self.run_exit = 0
        self.env: dict[str, str] = {}
        # 偽の uv が LEFT の行で残りを知らせる名前。継承した LLM の変数は常に見張る。
        self.watch: list[str] = []
        self.dotenv: dict[str, str] = {
            "OPENAI_API_KEY": _KEY_VALUE,
            "LANGSMITH_API_KEY": _KEY_VALUE,
            "L2_ENABLE_TOOLS": "true",
            "L2_ENABLE_RAG": "true",
            "L2_USE_C2_RETRIEVER": "true",
            "REDIS_STREAM_KEY": "aibyss:events",
            "L2_C2_URL": self.c2.url,
            "L2_V2_URL": self.v2.url,
        }
        self.dotenv_dev: dict[str, str] = {"REDIS_STREAM_KEY": "aibyss:events-dev"}
        # 置かないファイル（".env" か ".env.dev"）。
        self.absent_files: set[str] = set()
        self._log = root / "uv.log"
        scripts = root / "scripts"
        scripts.mkdir()
        for name in [*_RUN_SCRIPTS, "run_common.ps1"]:
            shutil.copy(_SCRIPTS / name, scripts / name)
        self._bin = root / "bin"
        self._bin.mkdir()
        if os.name == "nt":
            (self._bin / "uv.cmd").write_text(_UV_CMD, encoding="ascii", newline="\r\n")
        else:
            stub = self._bin / "uv"
            stub.write_text(_UV_SH, encoding="ascii")
            stub.chmod(0o755)

    def close(self) -> None:
        """偽サーバを止める。"""
        self.c2.close()
        self.v2.close()

    def set_stream_lines(self, dotenv_value: str | None, dotenv_dev_value: str | None) -> None:
        """.env と .env.dev の REDIS_STREAM_KEY の行を決める。None はその行を書かない。"""
        for values, value in ((self.dotenv, dotenv_value), (self.dotenv_dev, dotenv_dev_value)):
            values.pop("REDIS_STREAM_KEY", None)
            if value is not None:
                values["REDIS_STREAM_KEY"] = value

    def _prepare(self, script: str) -> None:
        """今の値で .env と .env.dev を書き、テストが決めていない応答を正常なものにする。"""
        # 消すと既定の URL の実サービスへ繋がるので、どの場面にも必ず置く。
        assert {"L2_C2_URL", "L2_V2_URL"} <= set(self.dotenv)
        for name, values in ((".env", self.dotenv), (".env.dev", self.dotenv_dev)):
            if name not in self.absent_files:
                (self.root / name).write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
        stream = _stream_of(script)
        if self.c2.body is None:
            self.c2.answer(200, {"status": "ok", "stream": stream, "last_read_age_sec": 0.1})
        if self.v2.body is None:
            self.v2.answer(200, _v2_ok(stream))

    def _child_env(self) -> dict[str, str]:
        """子の環境を返す。

        WHY: PATH を偽の uv とシステムのコマンドだけにする。本物の uv が先に見つかると、依存の同期が走ってしまう。
             conftest がテスト用に入れた REDIS_STREAM_KEY は渡さない。渡すと、env ファイルに行がない場面を作れない
             （シェルに値が残っている場面は self.env で明示する）。
        """
        system = str(Path(os.environ["SystemRoot"]) / "System32") if os.name == "nt" else "/usr/bin:/bin"
        inherited = {name: value for name, value in os.environ.items() if name != "REDIS_STREAM_KEY"}
        return {
            **inherited,
            "PATH": f"{self._bin}{os.pathsep}{system}",
            "UV_STUB_LOG": str(self._log),
            "UV_STUB_PREFLIGHT_EXIT": str(self.preflight_exit),
            "UV_STUB_RUN_EXIT": str(self.run_exit),
            "UV_STUB_WATCH": " ".join(self.watch),
            **self.env,
        }

    def run(self, script: str, *args: str, colors: bool = False) -> subprocess.CompletedProcess:
        """スクリプトを流す。日本語の出力を読むため、出力の文字コードを UTF-8 へ揃えて -Command で呼ぶ。

        colors を真にすると、Write-Host の行の先頭に色を <Yellow> の形で付ける。
        """
        self._prepare(script)
        command = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            f"{_SHOW_COLORS if colors else ''}"
            f"& '{self.root / 'scripts' / script}' {' '.join(args)}; exit $LASTEXITCODE"
        )
        return subprocess.run(
            [_PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
            env=self._child_env(), cwd=self.root,
        )

    def run_as_file(self, script: str, *args: str) -> subprocess.CompletedProcess:
        """l2-start-*.bat と同じ -File で流す。プロセスの終了コードを確かめるため。"""
        self._prepare(script)
        return subprocess.run(
            [_PWSH, "-NoProfile", "-NonInteractive", "-File", str(self.root / "scripts" / script), *args],
            capture_output=True, timeout=120, check=False, env=self._child_env(), cwd=self.root,
        )

    @property
    def uv_lines(self) -> list[str]:
        """偽の uv が記録した行。"""
        return self._log.read_text(encoding="ascii").splitlines() if self._log.exists() else []

    @property
    def uv_calls(self) -> list[str]:
        """偽の uv に渡った引数。"""
        return [line.removeprefix("ARGS ").strip() for line in self.uv_lines if line.startswith("ARGS ")]

    def called(self, module: str) -> bool:
        """そのモジュールを起動しようとしたかどうかを返す。"""
        return any(module in call for call in self.uv_calls)


def _run_common_probe(tmp_path: Path, lines: list[str]) -> str:
    """run_common.ps1 を読み込んで lines を流し、標準出力を返す。関数の表を 1 つのプロセスでまとめて確かめるため。"""
    probe = tmp_path / "probe.ps1"
    head = ["[Console]::OutputEncoding = [System.Text.Encoding]::UTF8", f". '{_SCRIPTS / 'run_common.ps1'}'"]
    probe.write_text("\n".join([*head, *lines]), encoding="utf-8-sig")
    done = subprocess.run(
        [_PWSH, "-NoProfile", "-NonInteractive", "-File", str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


# run_loop の python-dotenv と読み方を比べる .env の行。{} に変数の名前が入る。
_DOTENV_ROWS = [
    "{}=value",
    "{} = value",
    "{}=   value   ",
    '{}="value"',
    "{}='value'",
    "export {}=value",
    "{}=value # comment",
    "{}=value\t# comment after a tab",
    '{}="value # not a comment"',
    '{}="value" # comment after the quote',
    "{}=value#literal",
    "{}=two words",
    "{}=",
    "# {}=commented out",
    "   # {}=indented comment",
    "{}=http://127.0.0.1:8100/",
    "{}=a=b",
    "{}=値",
]


def _write_dotenv_rows(tmp_path: Path) -> tuple[list[str], Path]:
    """_DOTENV_ROWS を 1 つのファイルへ書き、行ごとの変数の名前とファイルのパスを返す。"""
    names = [f"L2_PROBE_{index:02}" for index in range(len(_DOTENV_ROWS))]
    env_file = tmp_path / ".env.probe"
    env_file.write_text("\n".join(row.format(name) for row, name in zip(_DOTENV_ROWS, names, strict=True)) + "\n", encoding="utf-8")
    return names, env_file


@pytest.fixture
def scene(tmp_path):
    """既定で正常に動く場面を返す。テストは偽サーバの応答や .env の値を変えてから run を呼ぶ。"""
    made = Scene(tmp_path)
    yield made
    made.close()


def _profile_of(script: str) -> str:
    """スクリプトが preflight へ渡すはずのプロファイルを返す。"""
    return "dev" if script == "run_dev.ps1" else "prod"


@needs_pwsh
class TestStartup:
    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_healthy_run_checks_first_and_starts_with_the_locked_environment(self, scene, script):
        """正常な起動の全体。親から継承した接続先も、キーの値も、この 1 回で確かめる。"""
        scene.env = dict.fromkeys(preflight.INHERITED_LLM_VARS, _INHERITED_VALUE)

        done = scene.run(script)

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        # 点検してから起動する。配信の直前に解決器を走らせず、extra も欠かさない。
        assert scene.uv_calls[0] == f"run --frozen --all-extras python -m lab_lounge.preflight --static --profile {_profile_of(script)}"
        assert scene.uv_calls[1].startswith("run --frozen --all-extras python -m lab_lounge.run_loop --wake-backend continuous")
        assert len(scene.uv_calls) == 2
        # 外した変数の名前は知らせ、値は出さず、run_loop へも点検へも渡さない。
        for name in preflight.INHERITED_LLM_VARS:
            assert name in output
        assert not [line for line in scene.uv_lines if line.startswith("LEFT ")]
        # キーは set / unset だけ。先頭の数文字も出さない。
        assert re.search(r"OPENAI_API_KEY\s+= set", output)
        if script == "run_prod.ps1":
            assert re.search(r"LANGSMITH_API_KEY\s+= set", output)
        assert "zqx" not in output
        # V2 の接続数が読めること。OBS の PC が繋がっていないことを、起動時に見つけるため。
        assert "total=8 loopback=0 lan=8" in output
        assert scene.c2.paths == ["/health"]
        assert scene.v2.paths == ["/health"]

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_failed_checks_are_all_reported_and_stop_the_start(self, scene, script):
        """preflight の非 0 は $ErrorActionPreference では止まらない。$LASTEXITCODE を見ていること。"""
        scene.preflight_exit = 1
        scene.v2.answer(503, {
            "status": "degraded", "stream": _stream_of(script), "redis": {"sub": False, "pub": True}, "consumer": "stopped",
            "ws_clients": {"total": 0, "loopback": 0, "lan": 0},
        })

        done = scene.run(script)

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert not scene.called("lab_lounge.run_loop")
        assert "preflight が失敗しました" in output
        # 1 つ目で止めず、V2 の理由まで 1 度に読めること。
        assert "HTTP 503" in output
        assert "consumer=stopped" in output
        assert "redis.sub=False redis.pub=True" in output
        assert "-SkipHealthCheck" in output

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_skip_health_check_is_for_a_setup_without_c2_and_v2(self, scene, script):
        """C2 も V2 も置かない利用者のための口。preflight は省かない。"""
        scene.c2.answer(503, {"status": "degraded"})
        scene.v2.answer(503, {"status": "degraded"})

        done = scene.run(script, "-SkipHealthCheck")

        assert done.returncode == 0, done.stdout + done.stderr
        assert scene.called("lab_lounge.preflight")
        assert scene.called("lab_lounge.run_loop")
        assert scene.c2.paths == []
        assert scene.v2.paths == []

    def test_unhealthy_c2_reason_and_foreign_service_on_the_v2_port(self, scene):
        """listen していても 503 なら止まること。JSON でない 200 は別のサービスがポートにいる印。"""
        scene.c2.answer(503, {"status": "degraded", "stream": "aibyss:events-dev", "reason": "consumer_stale", "last_read_age_sec": 42.0})
        scene.v2.answer(200, "<html>another service</html>")

        done = scene.run("run_dev.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert not scene.called("lab_lounge.run_loop")
        assert "reason=consumer_stale" in output
        assert "JSON でない応答" in output

    @pytest.mark.parametrize("body", ["", "[]", "{}", '{"detail":"x"}', '{"status":"degraded"}', '{"status":"UP"}', '{"status":"OK"}'])
    def test_200_outside_the_contract_stops_the_start(self, scene, body):
        """C2 と V2 の 200 は status が ok か disabled だけ。JSON でも形が違えば、ポートにいるのは別のサービス。"""
        scene.v2.answer(200, body)

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert "status が ok でも disabled でもなく、別のサービスかもしれません" in output
        assert not scene.called("lab_lounge.run_loop")

    @pytest.mark.parametrize("script,other", [
        ("run_prod.ps1", "aibyss:events-dev"),
        ("run_dev.ps1", "aibyss:events"),
        ("run_prod.ps1", "AIBYSS:EVENTS"),  # Redis のキーは大文字と小文字を区別する
    ])
    def test_v2_reading_another_stream_stops_the_start(self, scene, script, other):
        """V2 のポートは dev と prod で同じ。残った別のモードの V2 は 200 を返すが、発話は届かず HUD は無音になる。"""
        scene.v2.answer(200, _v2_ok(other))

        done = scene.run(script)

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert f"'{other}' を読んでいます" in output
        assert f"'{_stream_of(script)}'" in output
        assert not scene.called("lab_lounge.run_loop")
        assert "lan=8" not in output  # 止まる V2 を緑の行で出さない

    @pytest.mark.parametrize("script,other", [("run_prod.ps1", "aibyss:events-dev"), ("run_dev.ps1", "aibyss:events")])
    def test_c2_in_use_reading_another_stream_stops_the_start(self, scene, script, other):
        """検索に使う C2 が別のストリームを読むと、今の発話を覚えていない相手から思い出すことになる。"""
        scene.c2.answer(200, {"status": "ok", "stream": other, "last_read_age_sec": 0.1})

        done = scene.run(script)

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert re.search(rf"C2 127\.0\.0\.1:\d+: '{other}' を読んでいます", output)
        assert not scene.called("lab_lounge.run_loop")

    @pytest.mark.parametrize("service,body", [
        ("c2", {"status": "ok", "last_read_age_sec": 0.1}),
        ("v2", {"status": "ok", "ws_clients": {"total": 8, "loopback": 0, "lan": 8}}),
    ])
    def test_ok_without_the_stream_is_an_old_process_and_stops_the_start(self, scene, service, body):
        """stream を足す前のプロセスは、どこを読んでいるかを確かめられない。緑にせず、起動し直す案内を出す。"""
        getattr(scene, service).answer(200, body)

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert "古いプロセス" in output
        assert not scene.called("lab_lounge.run_loop")

    def test_disabled_is_not_compared_because_it_reads_no_stream(self, scene):
        """disabled はバスを読んでいないので、比べる相手がない。今までどおり知らせるだけで続ける。"""
        scene.c2.answer(200, {"status": "disabled", "stream": "aibyss:events-dev"})
        scene.v2.answer(200, {"status": "disabled", "stream": "aibyss:events-dev", "ws_clients": {"total": 8, "loopback": 0, "lan": 8}})

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert "REDIS_URL が空なので" in output
        assert scene.called("lab_lounge.run_loop")

    def test_custom_stream_name_is_compared_as_it_is(self, scene):
        """既定の 2 つと違う名前でも、L2 と同じなら嘘の赤にしない。"""
        scene.dotenv_dev["REDIS_STREAM_KEY"] = "lab:custom"
        scene.c2.answer(200, {"status": "ok", "stream": "lab:custom"})
        scene.v2.answer(200, _v2_ok("lab:custom"))

        done = scene.run("run_dev.ps1")

        assert done.returncode == 0, done.stdout + done.stderr

    def test_closed_port_stops_the_start(self, scene):
        """サービスが起動していない。"""
        with socket.create_server(("127.0.0.1", 0)) as unused:
            scene.dotenv["L2_V2_URL"] = f"http://127.0.0.1:{unused.getsockname()[1]}"
        done = scene.run("run_dev.ps1")
        assert done.returncode == 1, done.stdout + done.stderr
        assert "接続できません" in done.stdout + done.stderr
        assert not scene.called("lab_lounge.run_loop")

    @pytest.mark.parametrize("status,body,reason,effect,catches_up", [
        (503, {"status": "degraded", "stream": "aibyss:events", "reason": "consumer_stopped", "last_read_age_sec": None},
         "HTTP 503 reason=consumer_stopped", "歴史書が記録していません", True),
        (200, {"status": "ok", "stream": "aibyss:events-dev", "last_read_age_sec": 0.1},
         "'aibyss:events-dev' を読んでいます", "歴史書が記録していません", True),
        (200, "{}", "別のサービスかもしれません", "歴史書が記録していません", True),
        # 古いプロセスは記録しているかもしれないので、言い切らない。
        (200, {"status": "ok", "last_read_age_sec": 0.1}, "古いプロセス", "歴史書が記録しているかを確かめられません", True),
        # C2 は書けなかった発話を ack するので、この理由だけは「追いつきます」が嘘になる。
        (503, {"status": "degraded", "stream": "aibyss:events", "reason": "append_failing", "last_read_age_sec": 0.842},
         "HTTP 503 reason=append_failing", "歴史書が記録していません", False),
    ])
    def test_c2_that_search_does_not_use_is_reported_in_yellow_and_does_not_stop(self, scene, status, body, reason, effect, catches_up):
        """C2 は検索のフラグと無関係に全発話を記録する。止めないのは、発話が Redis に残り、C2 を戻せば追いつくから。

        append_failing は追いつかないが、それでも止めない。原因を直しても次の発話が届くまで 503 が残るので、
        赤で止めると L2 を起動できなくなる。
        """
        scene.dotenv["L2_ENABLE_RAG"] = "false"
        scene.c2.answer(status, body)

        done = scene.run("run_prod.ps1", colors=True)

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert scene.called("lab_lounge.run_loop")
        recovery = _C2_CATCHES_UP if catches_up else _C2_DOES_NOT_CATCH_UP
        assert re.search(
            rf"<Yellow>\[lab-lounge/prod\] C2 127\.0\.0\.1:\d+: .*{re.escape(reason)}.*"
            rf"検索に使わない設定なので止めませんが、{effect}。{recovery}",
            output,
        ), output
        assert ("追いつきます" in output) == catches_up, output
        assert "前提サービスの確認に失敗しました" not in output

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_setup_with_only_l2_and_v2_starts_without_a_false_red(self, scene, script):
        """C2 を置かない構成。接続できなくても、検索に使わないなら止めない。V2 の確認まで省く -SkipHealthCheck は要らない。"""
        scene.dotenv["L2_ENABLE_TOOLS"] = "false"
        with socket.create_server(("127.0.0.1", 0)) as unused:
            url = f"http://127.0.0.1:{unused.getsockname()[1]}"
        scene.dotenv["L2_C2_URL"] = url
        scene.dotenv_dev["L2_C2_URL"] = url

        done = scene.run(script, colors=True)

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert re.search(r"<Yellow>\[lab-lounge/\w+\] C2 127\.0\.0\.1:\d+: 接続できません。検索に使わない設定なので止めませんが", output), output
        assert scene.v2.paths == ["/health"]

    def test_c2_that_search_uses_stops_in_red_when_it_does_not_record(self, scene):
        """検索に使う設定では今までどおり止める。今の発話を思い出せないまま配信に入らないように。"""
        scene.c2.answer(503, {"status": "degraded", "stream": "aibyss:events", "reason": "consumer_stopped", "last_read_age_sec": None})

        done = scene.run("run_prod.ps1", colors=True)

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert re.search(r"<Red>  - C2 127\.0\.0\.1:\d+: HTTP 503 reason=consumer_stopped", output), output
        assert "検索に使わない設定" not in output
        assert not scene.called("lab_lounge.run_loop")

    def test_c2_that_search_does_not_use_is_read_and_other_states_are_reported(self, scene):
        """正常な C2 は通常の行で出す。あわせて、起動はするが知らせるべき状態を確かめる。"""
        scene.dotenv["L2_ENABLE_RAG"] = "false"
        del scene.dotenv["OPENAI_API_KEY"]
        scene.v2.answer(200, {"status": "disabled", "stream": "aibyss:events", "ws_clients": {"total": 0, "loopback": 0, "lan": 0}})

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert scene.c2.paths == ["/health"]
        assert scene.v2.paths == ["/health"]
        assert re.search(r"C2 127\.0\.0\.1:\d+: ok", output)
        assert "検索に使わない設定" not in output
        assert re.search(r"OPENAI_API_KEY\s+= unset", output)
        assert "disabled" in output
        assert "HUD に繋がっている画面がありません" in output

    @pytest.mark.parametrize("obs_url,clients,notice", [
        # OBS が別の PC。同じ機械で /status を開いていると total は増えるので、lan で見る。
        ("ws://192.0.2.20:4455", {"total": 1, "loopback": 1, "lan": 0}, _LAN_NOTICE),
        ("ws://192.0.2.20:4455", {"total": 0, "loopback": 0, "lan": 0}, _LAN_NOTICE),
        ("192.0.2.20:4455", {"total": 1, "loopback": 1, "lan": 0}, _LAN_NOTICE),  # 本番は ws:// のない形も受ける
        ("ws://192.0.2.20:4455", {"total": 2, "loopback": 1, "lan": 1}, None),
        # OBS が同じ機械か未設定。今までどおり total で見る。
        ("ws://127.0.0.1:4455", {"total": 1, "loopback": 1, "lan": 0}, None),
        ("ws://127.0.0.1:4455", {"total": 0, "loopback": 0, "lan": 0}, _TOTAL_NOTICE),
        ("localhost:4455", {"total": 1, "loopback": 1, "lan": 0}, None),
        ("ws://[::1]:4455", {"total": 1, "loopback": 1, "lan": 0}, None),  # IPv6 の loopback も同じ機械
        (None, {"total": 1, "loopback": 1, "lan": 0}, None),
        (None, {"total": 0, "loopback": 0, "lan": 0}, _TOTAL_NOTICE),
    ])
    def test_notice_about_the_screens_on_the_hud(self, scene, obs_url, clients, notice):
        """OBS の PC が繋がっていないことを起動時に見つける。OBS を後から起動することもあるので止めない。"""
        if obs_url:
            scene.dotenv["L2_OBS_WS_URL"] = obs_url
        scene.v2.answer(200, _v2_ok("aibyss:events", **clients))

        done = scene.run("run_prod.ps1", colors=True)

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert scene.called("lab_lounge.run_loop")
        for text in (_TOTAL_NOTICE, _LAN_NOTICE):
            if text == notice:
                assert f"<Yellow>[lab-lounge/prod] {text}" in output, output
            else:
                assert text not in output, output
        assert "192.0.2" not in output  # 起動の出力は貼られることがあるので、OBS の PC のアドレスを出さない

    @pytest.mark.parametrize("script,expected", [
        ("run_dev.ps1", ["[HOST]:8101", "[HOST]:8100"]),
        ("run_prod.ps1", ["[HOST]:8100", "[HOST]:8102"]),
    ])
    def test_header_does_not_print_a_non_loopback_url(self, scene, script, expected):
        """起動の出力は貼られることがある。/health の行と同じく host を伏せ、URL の資格情報も出さない。
        ポートは残すので、dev と prod の分離は読める。"""
        scene.dotenv["L2_C2_URL"] = "http://user:zqxPASSWORD@192.0.2.1:8100"
        scene.dotenv["L2_C2_URL_READONLY"] = "http://user:zqxPASSWORD@192.0.2.3:8102"
        scene.dotenv_dev["L2_C2_URL"] = "http://user:zqxPASSWORD@192.0.2.1:8101"
        scene.dotenv_dev["L2_C2_URL_READONLY"] = "http://user:zqxPASSWORD@192.0.2.2:8100"

        done = scene.run(script, "-SkipHealthCheck")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert "192.0.2" not in output
        assert "zqx" not in output
        for endpoint in expected:
            assert endpoint in output

    def test_header_shows_a_loopback_url_as_host_and_port(self, scene):
        """loopback は調査の手がかりなので残す。空の副 C2 は URL の形の不正としない。"""
        scene.dotenv_dev["L2_C2_URL"] = "http://127.0.0.1:8101/"
        scene.dotenv_dev["L2_C2_URL_READONLY"] = ""

        done = scene.run("run_dev.ps1", "-SkipHealthCheck")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert re.search(r"L2_C2_URL \(primary\)\s+= 127\.0\.0\.1:8101\n", output), output
        assert re.search(r"L2_C2_URL_READONLY\s+= \n", output), output

    def test_v2_without_the_client_counts_still_starts(self, scene):
        """接続数は知らせるための値なので、読めなくても起動を止めない。"""
        scene.v2.answer(200, {"status": "ok", "stream": "aibyss:events"})

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert _TOTAL_NOTICE not in output
        assert _LAN_NOTICE not in output

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_dotenv_that_sets_an_endpoint_reaches_preflight_which_refuses_it(self, scene, script):
        """.env の指定は黙って消さない。読む前に外すので残り、preflight が終了コード 2 で止める。"""
        scene.dotenv["ANTHROPIC_BASE_URL"] = _INHERITED_VALUE
        scene.preflight_exit = preflight.EXIT_REFUSED

        done = scene.run(script)

        assert done.returncode == 1, done.stdout + done.stderr
        assert "LEFT ANTHROPIC_BASE_URL" in scene.uv_lines
        assert not scene.called("lab_lounge.run_loop")

    @pytest.mark.parametrize("script,dotenv_value,dotenv_dev_value,reason", [
        ("run_dev.ps1", "aibyss:events", "aibyss:events", "本番ストリーム"),
        ("run_dev.ps1", "aibyss:events", None, "本番ストリーム"),  # .env.dev に行がなく、.env の本番の値が残る
        ("run_dev.ps1", None, None, "未設定か空"),  # bus は未設定を本番と読む
        ("run_dev.ps1", "aibyss:events", "", "未設定か空"),  # 空の値は C2 にも HUD にも届かない
        ("run_dev.ps1", None, '"  "', "未設定か空"),
        ("run_prod.ps1", "aibyss:events-dev", None, "dev ストリーム"),
        ("run_prod.ps1", "", None, "空です"),
        ("run_prod.ps1", '"  "', None, "空です"),
    ])
    def test_stream_refusal_comes_before_the_checks(self, scene, script, dotenv_value, dotenv_dev_value, reason):
        """publish 先がプロファイルと違うか、どこにも届かない名前なら、何も流さずに止まる。None は env ファイルに行がないこと。"""
        scene.set_stream_lines(dotenv_value, dotenv_dev_value)
        done = scene.run(script)
        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert scene.uv_calls == []
        assert reason in output

    @pytest.mark.parametrize("script,url,reason", [
        ("run_dev.ps1", "http://127.0.0.1:8100", "本番ポート 8100"),
        ("run_dev.ps1", "http://127.0.0.1:8100/", "本番ポート 8100"),
        ("run_prod.ps1", "http://127.0.0.1:8101", "dev C2 ポート 8101"),
        ("run_prod.ps1", "http://127.0.0.1:8101/", "dev C2 ポート 8101"),
    ])
    def test_c2_port_of_the_other_mode_stops_before_the_checks(self, scene, script, url, reason):
        """dev が本番の C2 を主に読む、本番が dev の C2 を読む設定。何も流さずに止まる。

        -SkipHealthCheck は、止まらなくなる退行が入っても実サービスの /health へ繋がないため。
        """
        file_values = scene.dotenv_dev if script == "run_dev.ps1" else scene.dotenv
        file_values["L2_C2_URL"] = url

        done = scene.run(script, "-SkipHealthCheck")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert scene.uv_calls == []
        assert reason in output

    @pytest.mark.parametrize("script,missing", [("run_prod.ps1", ".env"), ("run_dev.ps1", ".env.dev")])
    def test_missing_env_file_stops_before_anything_runs(self, scene, script, missing):
        """読むはずのファイルがないまま進むと、シェルに残った値か既定の接続先で動いてしまう。"""
        scene.absent_files = {missing}

        done = scene.run(script, "-SkipHealthCheck")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert scene.uv_calls == []
        assert f"{missing} が見つかりません" in output

    def test_prod_without_a_stream_line_starts_because_the_default_is_prod(self, scene):
        """prod で行がないのは正しい設定。bus の既定が本番なので、ここで止めると .env を最小にした利用者が起動できない。"""
        scene.set_stream_lines(None, None)
        done = scene.run("run_prod.ps1")
        assert done.returncode == 0, done.stdout + done.stderr
        assert scene.called("lab_lounge.run_loop")

    def test_prod_without_a_stream_line_is_compared_with_the_default_of_the_bus(self, scene, monkeypatch):
        """未設定の実効値は aibyss:events。比べないと、行のない prod が dev の V2 を見逃す。"""
        monkeypatch.delenv("REDIS_STREAM_KEY")
        assert bus._stream_key() == "aibyss:events"  # 比べる相手の既定が変わったら、スクリプトも変える
        scene.set_stream_lines(None, None)
        scene.v2.answer(200, _v2_ok("aibyss:events-dev"))

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert "'aibyss:events-dev' を読んでいます" in output

    @pytest.mark.parametrize("script,name,leftover", [
        ("run_prod.ps1", "REDIS_STREAM_KEY", "aibyss:events-dev"),
        ("run_dev.ps1", "REDIS_STREAM_KEY", "aibyss:events"),  # .env.dev が勝つ
        ("run_dev.ps1", "L2_V2_URL", "http://127.0.0.1:6"),  # .env が勝つ。閉じたポート
    ])
    def test_files_win_over_what_the_last_run_left_in_the_shell(self, scene, script, name, leftover):
        """同じシェルで dev と prod を続けて流しても、前の実行の値で別の相手へ繋がらない。"""
        scene.env = {name: leftover}

        done = scene.run(script)

        assert done.returncode == 0, done.stdout + done.stderr
        assert scene.called("lab_lounge.run_loop")
        assert scene.v2.paths == ["/health"]

    def test_prod_removes_what_only_env_dev_set_in_the_same_shell(self, scene):
        """-Override は .env にある名前しか上書きしない。run_dev.ps1 の後に同じシェルで流すと、.env.dev だけの値が本番へ渡る。

        .env.example が本番では厳禁とする副 C2 を読むことになる。外した名前は知らせ、値は出さない。
        """
        leftovers = {"L2_C2_URL_READONLY": "http://127.0.0.1:8100/zqx", "L2_ENABLE_RAG": "true"}
        del scene.dotenv["L2_ENABLE_RAG"]
        scene.dotenv_dev.update(leftovers)
        scene.env = leftovers
        scene.watch = list(leftovers)

        done = scene.run("run_prod.ps1", colors=True)

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert scene.called("lab_lounge.run_loop")
        assert [line for line in scene.uv_lines if line.startswith("LEFT ")] == []
        assert "<Gray>[lab-lounge/prod] .env.dev だけにある変数を外しました: L2_C2_URL_READONLY, L2_ENABLE_RAG" in output
        assert re.search(r"L2_C2_URL_READONLY\s+= \n", output), output
        assert "zqx" not in output

    def test_prod_keeps_the_names_env_also_sets(self, scene):
        """.env にもある名前は、.env の値で動く。外すのは .env.dev だけにある名前。"""
        scene.dotenv["L2_C2_URL_READONLY"] = "http://127.0.0.1:8102"
        scene.dotenv_dev["L2_C2_URL_READONLY"] = "http://127.0.0.1:8100"
        scene.env = {"L2_C2_URL_READONLY": "http://127.0.0.1:8100"}
        scene.watch = ["L2_C2_URL_READONLY"]

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert "LEFT L2_C2_URL_READONLY" in scene.uv_lines
        assert ".env.dev だけにある変数" not in output
        assert re.search(r"L2_C2_URL_READONLY\s+= 127\.0\.0\.1:8102\n", output), output

    def test_trailing_slash_of_the_url_does_not_double_the_slash(self, scene):
        """V2 や C2 の経路に // は合わないので、末尾の / だけで嘘の赤になる。"""
        scene.dotenv["L2_C2_URL"] = f"{scene.c2.url}/"
        scene.dotenv["L2_V2_URL"] = f"{scene.v2.url}/"

        done = scene.run("run_prod.ps1")

        assert done.returncode == 0, done.stdout + done.stderr
        assert scene.c2.paths == ["/health"]
        assert scene.v2.paths == ["/health"]

    def test_proxy_left_in_the_shell_is_not_used_for_the_check(self, scene):
        """手元のサービスの確認がプロキシへ向かうと、止まったプロキシで嘘の赤になる。"""
        with socket.create_server(("127.0.0.1", 0)) as unused:
            scene.env = {"HTTP_PROXY": f"http://127.0.0.1:{unused.getsockname()[1]}"}

        done = scene.run("run_prod.ps1")

        assert done.returncode == 0, done.stdout + done.stderr
        assert scene.c2.paths == ["/health"]
        assert scene.v2.paths == ["/health"]

    def test_service_that_hangs_stops_the_start_instead_of_hanging_it(self, scene):
        """/health が固まったサービスで、起動の確認まで固まらない。6 秒は確認の上限の 3 秒より長く待たせる値。"""
        scene.v2.delay = 6.0

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert "接続できません" in output
        assert not scene.called("lab_lounge.run_loop")

    @pytest.mark.skipif(_WINDOWS_POWERSHELL is None, reason="Windows PowerShell がない環境")
    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_windows_powershell_stops_with_the_reason(self, scene, script):
        """古い版で開いたときに、読めない構文エラーではなく、版が違うと伝えて止まる。何も起動しない。"""
        scene._prepare(script)

        done = subprocess.run(
            [_WINDOWS_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(scene.root / "scripts" / script)],
            capture_output=True, timeout=120, check=False, env=scene._child_env(), cwd=scene.root,
        )

        # 文面は OS の言語で変わるので、エラーの識別子で確かめる。
        assert b"ScriptRequiresUnmatchedPSVersion" in done.stdout + done.stderr
        assert done.returncode == 1
        assert scene.uv_calls == []

    @pytest.mark.parametrize("script,args,max_turns", [
        ("run_dev.ps1", ["-MaxTurns", "0"], None),
        ("run_prod.ps1", ["-MaxTurns", "5"], "5"),
    ])
    def test_process_exit_code_follows_run_loop(self, scene, script, args, max_turns):
        """-File での起動（l2-start-*.bat）でも終了コードが呼び出し元へ届くこと。"""
        scene.run_exit = 7
        done = scene.run_as_file(script, *args)
        assert done.returncode == 7
        call = scene.uv_calls[-1]
        if max_turns:
            assert call.endswith(f"--max-turns {max_turns}")
        else:
            assert "--max-turns" not in call

    def test_c2_rule_is_the_same_as_preflight(self, tmp_path):
        """preflight と食い違うと、点検は通るのに起動で止まる（またはその逆）。実物の関数を同じ値で比べる。"""
        table = [
            {"L2_USE_C2_RETRIEVER": c2, "L2_ENABLE_TOOLS": tools, "L2_ENABLE_RAG": rag}
            for c2 in (None, "true", "TRUE", "1", "yes", " true", "false")
            for tools in ("true", "YES", "1", " 1", None)
            for rag in ("true", "0", None)
        ]

        def assignment(name: str, value: str | None) -> str:
            """環境変数を設定する PowerShell の文を返す。None は未設定。"""
            return f"$env:{name} = $null" if value is None else f"$env:{name} = '{value}'"

        lines = [";".join(assignment(*item) for item in row.items()) + "; [int](Test-C2RetrieverEnabled)" for row in table]
        stdout = _run_common_probe(tmp_path, lines)

        expected = []
        for row in table:
            with patch.dict(os.environ):
                for name, value in row.items():
                    os.environ.pop(name, None)
                    if value is not None:
                        os.environ[name] = value
                expected.append("1" if preflight._c2_retriever_enabled() else "0")
        assert stdout.split() == expected

    def test_endpoint_shows_only_a_loopback_host_and_the_port(self, tmp_path):
        """起動の出力は貼られることがある。loopback だけを調査の手がかりとして残し、資格情報とパスは出さない。"""
        table = {
            "http://127.0.0.1:8101": "127.0.0.1:8101",
            "http://127.0.0.1:8101/": "127.0.0.1:8101",
            "http://localhost:3200": "localhost:3200",
            "http://[::1]:3200": "[::1]:3200",
            "http://user:zqx@127.0.0.1:8101": "127.0.0.1:8101",
            "http://192.0.2.1:8100": "[HOST]:8100",
            "http://user:zqx@192.0.2.1:8100/path": "[HOST]:8100",
            "http://box.example": "[HOST]:80",
            "not a url": "(URL の形が不正)",
        }
        stdout = _run_common_probe(tmp_path, [f"Format-ServiceEndpoint '{url}'" for url in table])
        assert stdout.splitlines() == list(table.values())

    def test_dotenv_reader_reads_like_python_dotenv(self, tmp_path):
        """スクリプトと run_loop で値の読み方が違うと、起動の確認と本番で別の相手を見る。本物の dotenv_values と比べる。

        ダブルクォートの中のエスケープは揃わない（preflight の env/quoting が知らせる）ので、表に入れない。
        """
        names, env_file = _write_dotenv_rows(tmp_path)

        reads = [f"[Console]::Out.WriteLine('{name}=' + $env:{name})" for name in names]
        stdout = _run_common_probe(tmp_path, [f"Import-DotEnvFile -Path '{env_file}' -Override", *reads])

        # 空の値と行なしの区別は pwsh の版で変わる（古い版は空の値で変数を消す）ので、どちらも空として比べる。
        read_by_the_script = dict(line.split("=", 1) for line in stdout.splitlines())
        read_by_dotenv = dotenv_values(env_file)
        assert read_by_the_script == {name: read_by_dotenv.get(name) or "" for name in names}

    def test_name_reader_finds_what_the_dotenv_reader_sets(self, tmp_path):
        """外す名前の読み方が読み込みとずれると、.env にある名前を外したり、残りを見逃したりする。本物の dotenv_values と比べる。"""
        _names, env_file = _write_dotenv_rows(tmp_path)

        stdout = _run_common_probe(tmp_path, [f"Get-DotEnvName -Path '{env_file}'"])

        assert stdout.split() == list(dotenv_values(env_file))

    def test_name_reader_skips_lines_without_a_name_and_a_missing_file(self, tmp_path):
        """名前のない行まで返すと、環境から外すときに例外で止まる。"""
        env_file = tmp_path / ".env.probe"
        env_file.write_text("=value\nNO_EQUALS\n\n# COMMENT=1\nL2_PROBE_NAME=1\n", encoding="utf-8")

        stdout = _run_common_probe(tmp_path, [f"Get-DotEnvName -Path '{env_file}'", f"Get-DotEnvName -Path '{tmp_path / 'missing'}'"])

        assert stdout.split() == ["L2_PROBE_NAME"]

    def test_only_the_names_that_env_lacks_are_removed(self, tmp_path):
        """コメントの行は .env にない名前として外す。大文字と小文字の違いは同じ名前として残す（Windows の環境変数は区別しない）。

        名前はファイルから来るので、* でほかの変数を巻き込まない。.env.dev がなければ何も外さない。
        """
        env_file = tmp_path / ".env"
        dev_file = tmp_path / ".env.dev"
        env_file.write_text("L2_PROBE_BOTH=1\n# L2_PROBE_COMMENTED=1\nexport L2_PROBE_EXPORTED=1\nl2_probe_lower=1\n", encoding="utf-8")
        dev_file.write_text(
            "L2_PROBE_BOTH=2\nL2_PROBE_COMMENTED=2\nL2_PROBE_EXPORTED=2\nL2_PROBE_LOWER=2\n"
            "L2_PROBE_DEV=zqx\nL2_PROBE_DEV=zqx\nL2_PROBE_UNSET=2\nL2_PROBE_*=2\n",
            encoding="utf-8",
        )
        in_shell = ["L2_PROBE_BOTH", "L2_PROBE_COMMENTED", "L2_PROBE_EXPORTED", "L2_PROBE_LOWER", "L2_PROBE_DEV", "L2_PROBE_OTHER"]

        stdout = _run_common_probe(tmp_path, [
            *[f"$env:{name} = 'left'" for name in in_shell],
            f"Remove-DevOnlyEnvVar -Tag 'probe' -EnvFile '{env_file}' -DevEnvFile '{tmp_path / 'missing'}'",
            f"Remove-DevOnlyEnvVar -Tag 'probe' -EnvFile '{env_file}' -DevEnvFile '{dev_file}'",
            "[Console]::Out.WriteLine('REMAINS ' + ((Get-ChildItem Env:L2_PROBE_*).Name -join ' '))",
        ])

        lines = stdout.splitlines()
        assert lines[0] == "[probe] .env.dev だけにある変数を外しました: L2_PROBE_COMMENTED, L2_PROBE_DEV"
        assert set(lines[1].removeprefix("REMAINS ").split()) == {"L2_PROBE_BOTH", "L2_PROBE_EXPORTED", "L2_PROBE_LOWER", "L2_PROBE_OTHER"}
        assert len(lines) == 2


class TestScriptText:
    """流さなくても分かる約束。pwsh がない環境でも確かめる。"""

    @staticmethod
    def _read(name: str) -> str:
        """scripts/ のファイルの本文を返す。"""
        return (_SCRIPTS / name).read_text(encoding="utf-8")

    def test_inherited_list_is_the_same_as_preflight(self):
        """片方だけ増やすと、点検と本番で別の相手へ繋がる穴が残る。"""
        block = self._read("run_common.ps1").split("$script:InheritedLlmVars = @(")[1].split(")")[0]
        assert tuple(re.findall(r'"([A-Z_]+)"', block)) == preflight.INHERITED_LLM_VARS

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_powershell_7_5_is_required_on_the_first_line(self, script):
        """run_common.ps1 は ?? を使い、REDIS_STREAM_KEY= の検査には空の値の環境変数が要る。

        7.4 以前は空の値を入れると環境変数ごと消えるので、prod の空の値が未設定に見えて検査を通る。
        実際に古い版で流すテストは TestStartup にある。
        """
        assert self._read(script).splitlines()[0] == "#Requires -Version 7.5"

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_every_uv_run_uses_the_locked_environment(self, script):
        """素の uv run は配信の開始時に解決器を走らせ、lock の更新後に extra が古いまま残ることもある。"""
        calls = [line.strip() for line in self._read(script).splitlines()
                 if line.strip().startswith("uv ")]
        assert len(calls) == 3
        for call in calls:
            assert call.startswith("uv run --frozen --all-extras python -m lab_lounge.")

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_no_part_of_a_key_is_printed(self, script):
        assert "Substring" not in self._read(script)

    @pytest.mark.parametrize("script", _RUN_SCRIPTS)
    def test_preflight_gets_the_profile_of_the_script(self, script):
        assert f"--static --profile {_profile_of(script)}\n" in self._read(script)

    def test_default_urls_are_the_documented_ones(self):
        """偽サーバの場面では .env で上書きするので、既定はここで確かめる。実物の C2 と V2 には繋がない。"""
        common = self._read("run_common.ps1")
        assert '"http://127.0.0.1:3200"' in common
        assert '"http://127.0.0.1:8100"' in common
        # C2 の既定は本番の読み方と同じ。
        assert 'os.environ.get("L2_C2_URL", "http://127.0.0.1:8100")' in (_REPO_ROOT / "src" / "lab_lounge" / "graph.py").read_text(encoding="utf-8")

    def test_prod_start_batch_file_has_the_same_shape_as_dev(self):
        """Stream Deck から押すボタンの入口。cmd は UTF-8 の日本語を読み違えることがあるので ASCII に限る。"""
        raw = (_SCRIPTS / "l2-start-prod.bat").read_bytes()
        assert raw.decode("ascii")
        assert raw.count(b"\n") == raw.count(b"\r\n")
        prod_start = raw.decode("ascii").strip().splitlines()[-1]
        dev_start = (_SCRIPTS / "l2-start-dev.bat").read_bytes().decode("utf-8").strip().splitlines()[-1]
        assert prod_start == dev_start.replace("L2 dev", "L2 prod").replace("run_dev.ps1", "run_prod.ps1")
        assert '"%~dp0run_prod.ps1"' in prod_start

    def test_stop_batch_file_runs_as_a_batch_file(self):
        """Stream Deck から押すボタンの入口。中身が PowerShell の本文だと、cmd は 1 行目から読めずに何も止めない。

        実物は流さない（手元で動いている run_loop を止めてしまう）。cmd は UTF-8 の日本語を読み違えることがあるので ASCII に限る。
        """
        raw = (_SCRIPTS / "l2-stop-dev.bat").read_bytes()
        assert raw.decode("ascii")
        assert raw.count(b"\n") == raw.count(b"\r\n")
        lines = raw.decode("ascii").splitlines()
        assert lines[0] == "@echo off"
        assert lines[1:] == ['pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0l2-stop-dev.ps1"']


class TestReadme:
    """第三者は README だけを読んで動かす。実態から離れた箇所で詰まらないように。"""

    @staticmethod
    def _readme() -> str:
        """README の本文を返す。"""
        return (_REPO_ROOT / "README.md").read_text(encoding="utf-8")

    def _obs_section(self) -> str:
        """OBS を別の PC で動かす手順の節を返す。"""
        return self._readme().split("### OBS を別の PC で動かす")[1].split("\n---\n")[0]

    def test_startup_checks_are_described_where_the_script_is_explained(self):
        section = self._readme().split("#### dev プロファイルで起動する")[1].split("### 4. `preflight`")[0]
        for word in ("-SkipHealthCheck", "preflight --static --profile dev", "/health", "L2_V2_URL", "l2-start-prod.bat"):
            assert word in section
        assert "uv run --frozen --all-extras python -m lab_lounge.run_loop" in section
        assert "先頭のみ" not in section  # キーの先頭はもう出さない

    def test_what_the_checks_do_is_described_as_they_are(self):
        """起動前の確認で止まる理由と止まらない理由を、スクリプトと同じ内容で書いていること。"""
        section = self._readme().split("#### dev プロファイルで起動する")[1].split("### 4. `preflight`")[0]
        for word in (
            "未設定",  # dev のストリームの拒否
            "`stream`",  # 読んでいるストリームの照合
            "古いプロセス",  # stream を返さない V2 と C2
            "検索に使わない設定",  # C2 は常に読み、止めずに黄色で知らせる
            "`reason=append_failing` だけは別",  # 書けなかった発話は追いつかない
            "`lan` が 0",  # OBS が別の PC のときの判定
            "`[HOST]`",  # 見出しの URL を伏せる
            "`.env.dev` にあって `.env` にない名前",  # 同じシェルの run_dev.ps1 の残りを prod で外す
            "では止まらない。先に L2 の窓を Ctrl+C で止める",  # 切り替えのときに dev の run_loop を残さない
        ):
            assert word in section, word
        assert "記憶の検索に使わない C2 では止めない" not in section  # 読まなかったころの説明

    def test_stream_row_does_not_promise_a_stop_for_a_c2_out_of_search(self):
        """検索に使わない C2 は黄色で続行する。止まると読んだ利用者が、黄色の行を見逃さないように。"""
        row = next(line for line in self._readme().splitlines() if line.startswith("| `REDIS_STREAM_KEY`"))
        assert "検索に使う C2" in row
        assert "検索に使わない C2 は黄色" in row

    def test_troubleshooting_tells_the_way_out_of_append_failing(self):
        """Redis を調べても原因にたどり着かない 503。原因を直しても 503 が残るので、再起動まで書く。"""
        row = next(line for line in self._readme().splitlines() if "が起動前の確認で止まる**" in line)
        assert "`reason=append_failing`" in row
        assert "DB に書けていない" in row
        assert "C2 を再起動" in row

    def test_prerequisites_state_how_the_old_powershell_stops(self):
        """前提の節は、スクリプトの #Requires と同じことを言う。"""
        section = self._readme().split("### 前提")[1].split("### インストール")[0]
        assert "#Requires" in section

    def test_v2_url_is_in_the_templates(self):
        assert "L2_V2_URL" in (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    def test_guide_for_obs_on_another_pc_covers_both_systems(self):
        section = self._obs_section()
        for word in ("Test-NetConnection", "SourceAddress", "route -n get", "ipconfig getifaddr", "ws_clients", "Firewall"):
            assert word in section

    def test_guide_warns_against_the_interface_route(self):
        """自分の MAC の固定エントリになる操作は、手順としてではなく禁止として書く。"""
        section = self._obs_section()
        command = next(line for line in section.splitlines() if "route add -host" in line)
        assert "使わない" in command

    def test_addresses_in_the_guide_are_for_documentation(self):
        section = self._obs_section()
        addresses = set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", section))
        assert addresses
        assert {address for address in addresses if not address.startswith(("192.0.2.", "127.0.0.1"))} == set()

    def test_test_command_uses_the_locked_environment(self):
        """素の uv run は dev の extra を入れないので、新しい clone では pytest が見つからない。"""
        section = self._readme().split("## テスト")[1].split("## ディレクトリ構成")[0]
        assert "uv run --frozen --all-extras pytest" in section
        assert "uv run pytest" not in section
