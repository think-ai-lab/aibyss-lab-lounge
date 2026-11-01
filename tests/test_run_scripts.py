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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_lounge import preflight

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
_RUN_SCRIPTS = ["run_dev.ps1", "run_prod.ps1"]

# WHY: Windows の subprocess は env の PATH で実行ファイルを探さないので、親のうちに絶対パスにしておく。
_PWSH = shutil.which("pwsh")
needs_pwsh = pytest.mark.skipif(_PWSH is None, reason="pwsh がない環境")

# 値が出力に現れないことを確かめる印。先頭だけ出ても見つかるように、頭に短い印を置く。
_KEY_VALUE = "zqxKEYVALUE-not-a-real-key"
_INHERITED_VALUE = "http://198.51.100.1:9/zqxINHERITED"

# 偽の uv。引数と、外れているはずの変数の残りを記録して、preflight と run_loop で別の終了コードを返す。
_INHERITED_NAMES = " ".join(preflight.INHERITED_LLM_VARS)
_UV_CMD = f"""@echo off
>>"%UV_STUB_LOG%" echo ARGS %*
for %%N in ({_INHERITED_NAMES}) do if defined %%N >>"%UV_STUB_LOG%" echo LEFT %%N
echo %* | findstr /c:"lab_lounge.preflight" >nul && exit /b %UV_STUB_PREFLIGHT_EXIT%
exit /b %UV_STUB_RUN_EXIT%
"""
_UV_SH = f"""#!/bin/sh
echo "ARGS $*" >> "$UV_STUB_LOG"
for name in {_INHERITED_NAMES}; do
    eval "value=\\${{$name+x}}"
    [ -n "$value" ] && echo "LEFT $name" >> "$UV_STUB_LOG"
done
case "$*" in
    *lab_lounge.preflight*) exit "$UV_STUB_PREFLIGHT_EXIT" ;;
esac
exit "$UV_STUB_RUN_EXIT"
"""


class FakeService:
    """GET /health だけに答える偽サーバ。"""

    def __init__(self) -> None:
        self.status = 200
        self.body: bytes = b'{"status":"ok"}'
        self.paths: list[str] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server の規約
                service.paths.append(self.path)
                self.send_response(service.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(service.body)

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
        self.v2.answer(200, {"status": "ok", "ws_clients": {"total": 8, "loopback": 0, "lan": 8}})
        self.preflight_exit = 0
        self.run_exit = 0
        self.env: dict[str, str] = {}
        self.dotenv: dict[str, str] = {
            "OPENAI_API_KEY": _KEY_VALUE,
            "L2_ENABLE_TOOLS": "true",
            "L2_ENABLE_RAG": "true",
            "L2_USE_C2_RETRIEVER": "true",
            "REDIS_STREAM_KEY": "aibyss:events",
            "L2_C2_URL": self.c2.url,
            "L2_V2_URL": self.v2.url,
        }
        self.dotenv_dev: dict[str, str] = {"REDIS_STREAM_KEY": "aibyss:events-dev"}
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

    def _write_env_files(self) -> None:
        for name, values in ((".env", self.dotenv), (".env.dev", self.dotenv_dev)):
            (self.root / name).write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")

    def _child_env(self) -> dict[str, str]:
        """子の環境を返す。

        WHY: PATH を偽の uv とシステムのコマンドだけにする。本物の uv が先に見つかると、依存の同期が走ってしまう。
        """
        system = str(Path(os.environ["SystemRoot"]) / "System32") if os.name == "nt" else "/usr/bin:/bin"
        return {
            **os.environ,
            "PATH": f"{self._bin}{os.pathsep}{system}",
            "UV_STUB_LOG": str(self._log),
            "UV_STUB_PREFLIGHT_EXIT": str(self.preflight_exit),
            "UV_STUB_RUN_EXIT": str(self.run_exit),
            **self.env,
        }

    def run(self, script: str, *args: str) -> subprocess.CompletedProcess:
        """スクリプトを流す。日本語の出力を読むため、出力の文字コードを UTF-8 へ揃えて -Command で呼ぶ。"""
        self._write_env_files()
        command = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            f"& '{self.root / 'scripts' / script}' {' '.join(args)}; exit $LASTEXITCODE"
        )
        return subprocess.run(
            [_PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
            env=self._child_env(), cwd=self.root,
        )

    def run_as_file(self, script: str, *args: str) -> subprocess.CompletedProcess:
        """l2-start-*.bat と同じ -File で流す。プロセスの終了コードを確かめるため。"""
        self._write_env_files()
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


@pytest.fixture
def scene(tmp_path):
    made = Scene(tmp_path)
    yield made
    made.close()


def _profile_of(script: str) -> str:
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
            "status": "degraded", "redis": {"sub": False, "pub": True}, "consumer": "stopped",
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
        scene.c2.answer(503, {"status": "degraded", "reason": "consumer_stale", "last_read_age_sec": 42.0})
        scene.v2.answer(200, "<html>another service</html>")

        done = scene.run("run_dev.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 1, output
        assert not scene.called("lab_lounge.run_loop")
        assert "reason=consumer_stale" in output
        assert "JSON でない応答" in output

    def test_closed_port_stops_the_start(self, scene):
        """サービスが起動していない。"""
        with socket.create_server(("127.0.0.1", 0)) as unused:
            scene.dotenv["L2_V2_URL"] = f"http://127.0.0.1:{unused.getsockname()[1]}"
        done = scene.run("run_dev.ps1")
        assert done.returncode == 1, done.stdout + done.stderr
        assert "接続できません" in done.stdout + done.stderr
        assert not scene.called("lab_lounge.run_loop")

    def test_c2_that_the_settings_do_not_use_is_not_checked(self, scene):
        """検索に使わない C2 で起動を止めない。あわせて、起動はするが知らせるべき状態を確かめる。"""
        scene.dotenv["L2_ENABLE_RAG"] = "false"
        del scene.dotenv["OPENAI_API_KEY"]
        scene.v2.answer(200, {"status": "disabled", "ws_clients": {"total": 0, "loopback": 0, "lan": 0}})

        done = scene.run("run_prod.ps1")

        output = done.stdout + done.stderr
        assert done.returncode == 0, output
        assert scene.c2.paths == []
        assert scene.v2.paths == ["/health"]
        assert re.search(r"OPENAI_API_KEY\s+= unset", output)
        assert "disabled" in output
        assert "HUD に繋がっている画面がありません" in output

    def test_dotenv_that_sets_an_endpoint_reaches_preflight_which_refuses_it(self, scene):
        """.env の指定は黙って消さない。読む前に外すので残り、preflight が終了コード 2 で止める。"""
        scene.dotenv["ANTHROPIC_BASE_URL"] = _INHERITED_VALUE
        scene.preflight_exit = preflight.EXIT_REFUSED

        done = scene.run("run_prod.ps1")

        assert done.returncode == 1, done.stdout + done.stderr
        assert "LEFT ANTHROPIC_BASE_URL" in scene.uv_lines
        assert not scene.called("lab_lounge.run_loop")

    def test_stream_refusal_comes_before_the_checks(self, scene):
        """dev で本番ストリームを指していたら、何も流さずに止まる。"""
        scene.dotenv_dev["REDIS_STREAM_KEY"] = "aibyss:events"
        done = scene.run("run_dev.ps1")
        assert done.returncode == 1, done.stdout + done.stderr
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
            return f"$env:{name} = $null" if value is None else f"$env:{name} = '{value}'"

        lines = [f". '{_SCRIPTS / 'run_common.ps1'}'"]
        lines += [";".join(assignment(*item) for item in row.items()) + "; [int](Test-C2RetrieverEnabled)" for row in table]
        probe = tmp_path / "probe.ps1"
        probe.write_text("\n".join(lines), encoding="utf-8-sig")
        done = subprocess.run(
            [_PWSH, "-NoProfile", "-NonInteractive", "-File", str(probe)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
        )
        assert done.returncode == 0, done.stderr

        expected = []
        for row in table:
            with patch.dict(os.environ):
                for name, value in row.items():
                    os.environ.pop(name, None)
                    if value is not None:
                        os.environ[name] = value
                expected.append("1" if preflight._c2_retriever_enabled() else "0")
        assert done.stdout.split() == expected


class TestScriptText:
    """流さなくても分かる約束。pwsh がない環境でも確かめる。"""

    @staticmethod
    def _read(name: str) -> str:
        return (_SCRIPTS / name).read_text(encoding="utf-8")

    def test_inherited_list_is_the_same_as_preflight(self):
        """片方だけ増やすと、点検と本番で別の相手へ繋がる穴が残る。"""
        block = self._read("run_common.ps1").split("$script:InheritedLlmVars = @(")[1].split(")")[0]
        assert tuple(re.findall(r'"([A-Z_]+)"', block)) == preflight.INHERITED_LLM_VARS

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

    def test_default_v2_url_is_the_documented_one(self):
        """偽サーバの場面では .env で上書きするので、既定はここで確かめる。実物の V2 には繋がない。"""
        assert '"http://127.0.0.1:3200"' in self._read("run_common.ps1")

    def test_prod_start_batch_file_has_the_same_shape_as_dev(self):
        """Stream Deck から押すボタンの入口。cmd は UTF-8 の日本語で壊れるので ASCII に限る。"""
        raw = (_SCRIPTS / "l2-start-prod.bat").read_bytes()
        assert raw.decode("ascii")
        assert raw.count(b"\n") == raw.count(b"\r\n")
        prod_start = raw.decode("ascii").strip().splitlines()[-1]
        dev_start = (_SCRIPTS / "l2-start-dev.bat").read_bytes().decode("utf-8").strip().splitlines()[-1]
        assert prod_start == dev_start.replace("L2 dev", "L2 prod").replace("run_dev.ps1", "run_prod.ps1")
        assert '"%~dp0run_prod.ps1"' in prod_start


class TestReadme:
    """第三者は README だけを読んで動かす。実態から離れた箇所で詰まらないように。"""

    @staticmethod
    def _readme() -> str:
        return (_REPO_ROOT / "README.md").read_text(encoding="utf-8")

    def test_startup_checks_are_described_where_the_script_is_explained(self):
        section = self._readme().split("#### dev プロファイルで起動する")[1].split("### 4. `preflight`")[0]
        for word in ("-SkipHealthCheck", "preflight --static --profile dev", "/health", "L2_V2_URL", "l2-start-prod.bat"):
            assert word in section
        assert "uv run --frozen --all-extras python -m lab_lounge.run_loop" in section
        assert "先頭のみ" not in section  # キーの先頭はもう出さない

    def test_v2_url_is_in_the_templates(self):
        assert "L2_V2_URL" in (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
