"""
test_isolation.py — テストの隔離そのものを確かめる

conftest の隔離が黙って外れると、以後のテストが手元の .env や実サービスへ触れても緑のまま気づけない。
ここで「外れたら赤くなる」状態を保つ。
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import redis

from lab_lounge import bus

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 手元で実サービスが待ち受けている既定のポート。
_SERVICE_PORTS = [6, 3200, 4455, 6379, 8100, 8101, 15021, 18080, 50021]


class TestEnvironment:
    """lab_lounge の import より前に整えた環境が、テストの間も保たれていること。"""

    def test_environment_is_ready_when_conftest_is_imported(self):
        """テストファイルの import は fixture より先に走り、その時点で load_dotenv やログの作成が動く。

        fixture だけで整えていても、ここまでのテストは緑になる。別プロセスで conftest だけを読んで確かめる。
        """
        code = textwrap.dedent("""
            import json, os, sys
            sys.path.insert(0, sys.argv[1])
            import conftest
            print(json.dumps({name: os.environ.get(name) for name in (
                "PYTHON_DOTENV_DISABLED", "REDIS_URL", "REDIS_STREAM_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "L2_USE_REAL_LLM",
            )} | {"log_dir_is_temporary": conftest._SESSION_DIR.name in os.environ["L2_LOG_DIR"]}))
        """)
        env = {
            **os.environ, "OPENAI_API_KEY": "x", "ANTHROPIC_BASE_URL": "http://192.0.2.1", "L2_USE_REAL_LLM": "true",
            "REDIS_URL": "redis://192.0.2.1:6379", "REDIS_STREAM_KEY": "aibyss:events",
        }
        del env["PYTHON_DOTENV_DISABLED"]
        done = subprocess.run(
            [sys.executable, "-c", code, str(_REPO_ROOT / "tests")],
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False, env=env,
        )
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout) == {
            "PYTHON_DOTENV_DISABLED": "1",
            "REDIS_URL": "redis://127.0.0.1:6",
            "REDIS_STREAM_KEY": "aibyss:events-test",
            "OPENAI_API_KEY": None,
            "ANTHROPIC_BASE_URL": None,
            "L2_USE_REAL_LLM": None,
            "log_dir_is_temporary": True,
        }

    def test_dotenv_loading_is_disabled(self, tmp_path):
        """明示したパスでも読まないこと。run_loop などの load_dotenv() が手元の .env を環境へ入れないように。"""
        from dotenv import load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("L2_ISOLATION_SENTINEL=leaked\n", encoding="utf-8")
        assert load_dotenv(env_file) is False
        assert "L2_ISOLATION_SENTINEL" not in os.environ

    def test_publish_goes_to_a_closed_port(self):
        """消すと既定の実 Redis へ届くので、閉じたポートを指していること。"""
        assert bus._redis_url() == "redis://127.0.0.1:6"

    def test_publish_does_not_use_the_production_stream(self):
        assert bus._stream_key() != "aibyss:events"

    @pytest.mark.parametrize("prefix", [
        "OPENAI", "ANTHROPIC", "GOOGLE", "GEMINI", "TAVILY", "VOYAGE", "LANGCHAIN", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    ])
    def test_keys_and_endpoints_are_removed(self, prefix):
        assert not [name for name in os.environ if name.upper().startswith(prefix)]

    def test_langsmith_cannot_send(self):
        """キーがなく、送信先は閉じたポート。トレースを有効にするテストでも外へ出ない。"""
        names = {name for name in os.environ if name.upper().startswith("LANGSMITH")}
        assert names == {"LANGSMITH_TRACING", "LANGSMITH_ENDPOINT"}
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGSMITH_ENDPOINT"] == "http://127.0.0.1:6"

    def test_only_known_l2_variables_are_set(self):
        """手元の L2_ の設定が 1 つでも残ると、テストの結果が .env で変わる。"""
        assert {name for name in os.environ if name.upper().startswith("L2_")} == {
            "L2_LOG_DIR",
            "L2_STREAM_CONTEXT_FILE",
            "L2_IRODORI_VOICES_JSON",
            "L2_IRODORI_READINGS_JSON",
            "L2_VOICEPEAK_RETRY_WAIT_SEC",
            "L2_VOICEPEAK_MAX_RETRIES",
        }

    def test_changes_do_not_leak_into_the_next_test_first(self):
        """対になる次のテストと合わせて、os.environ への直接の書き込みが持ち越されないことを見る。"""
        os.environ["L2_ISOLATION_LEFTOVER"] = "x"
        os.environ["OPENAI_API_KEY"] = "x"

    def test_changes_do_not_leak_into_the_next_test_second(self):
        assert "L2_ISOLATION_LEFTOVER" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ


class TestLocalData:
    """git に入らない手元のデータとログの置き場所が、リポジトリの外を指していること。"""

    @staticmethod
    def _outside_repo(path: str | Path) -> bool:
        """path がリポジトリの中（リポジトリ自身を含む）でなければ True を返す。"""
        resolved = Path(path).resolve()
        return resolved != _REPO_ROOT and _REPO_ROOT not in resolved.parents

    def test_working_directory_is_outside_the_repo(self):
        """./data/audio や ./logs の既定と、作業ディレクトリの .env を読むライブラリのため。"""
        assert self._outside_repo(Path.cwd())

    @pytest.mark.parametrize("name", ["L2_LOG_DIR", "L2_STREAM_CONTEXT_FILE", "L2_IRODORI_VOICES_JSON", "L2_IRODORI_READINGS_JSON"])
    def test_paths_are_outside_the_repo(self, name):
        assert self._outside_repo(os.environ[name])

    @pytest.mark.parametrize("name", ["L2_STREAM_CONTEXT_FILE", "L2_IRODORI_VOICES_JSON", "L2_IRODORI_READINGS_JSON"])
    def test_local_files_look_absent_as_in_a_fresh_clone(self, name):
        assert not Path(os.environ[name]).exists()

    @pytest.mark.parametrize("name", ["_FILLER_CACHE_DIR", "_FILLER_AUDIO_DIR"])
    def test_filler_files_are_outside_the_repo(self, name):
        from lab_lounge import filler

        assert self._outside_repo(getattr(filler, name))

    def test_schema_path_is_resolved_from_the_sibling_workspace(self):
        """兄弟の aibyss-workspace があれば、そのスキーマを指すこと。ないときは外からの指定に任せる。"""
        sibling = _REPO_ROOT.parent / "aibyss-workspace" / "specs" / "event-envelope-0.1.schema.json"
        if not sibling.is_file():
            pytest.skip("aibyss-workspace を併置していない環境")
        assert Path(os.environ["AIBYSS_SCHEMA_PATH"]) == sibling


class TestNetworkGuard:
    """実際の接続の手前で止まること。繋がる先がないアドレス (192.0.2.x, 2001:db8::) で確かめる。"""

    @pytest.mark.parametrize("family,address", [
        (socket.AF_INET, ("192.0.2.1", 80)),
        (socket.AF_INET6, ("2001:db8::1", 80)),
        (socket.AF_INET6, ("::ffff:192.0.2.1", 80)),
    ])
    def test_connect_outside_is_blocked_and_recorded(self, external_network_attempts, family, address):
        with socket.socket(family) as sock, pytest.raises(RuntimeError, match="外へ接続"):
            sock.connect(address)
        assert external_network_attempts == [f"{address[0]}:{address[1]}"]
        external_network_attempts.clear()

    def test_connect_ex_outside_is_blocked(self, external_network_attempts):
        """番号を返す connect_ex でも素通りさせない。"""
        with socket.socket() as sock, pytest.raises(RuntimeError, match="外へ接続"):
            sock.connect_ex(("192.0.2.1", 80))
        assert external_network_attempts == ["192.0.2.1:80"]
        external_network_attempts.clear()

    def test_name_lookup_is_blocked(self, external_network_attempts):
        """名前解決も外への通信で、ネットワークの有無で失敗の形も変わる。"""
        with pytest.raises(RuntimeError, match="外へ接続"):
            socket.create_connection(("example.com", 443), timeout=1)
        assert external_network_attempts == ["example.com:443"]
        external_network_attempts.clear()

    async def test_event_loop_connection_is_blocked(self, external_network_attempts):
        """Windows の既定のイベントループは socket.connect を呼ばないので、別に差し替えている。"""
        with pytest.raises(RuntimeError, match="外へ接続"):
            await asyncio.open_connection("192.0.2.1", 80)
        assert external_network_attempts == ["192.0.2.1:80"]
        external_network_attempts.clear()

    @pytest.mark.parametrize("port", _SERVICE_PORTS)
    def test_local_service_ports_are_refused_at_once(self, external_network_attempts, port):
        """手元で起動中の実サービスへ繋がらないこと。Windows の実際の拒否（約 2 秒）も待たない。"""
        started = time.monotonic()
        with socket.socket() as sock, pytest.raises(ConnectionRefusedError):
            sock.connect(("127.0.0.1", port))
        assert time.monotonic() - started < 1
        assert external_network_attempts == []

    def test_connect_ex_to_a_local_service_port_returns_the_error_number(self):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", 6379)) != 0

    def test_localhost_counts_as_local(self, external_network_attempts):
        """実サービスの範囲なら拒否され、外への試みには数えない。"""
        with socket.socket() as sock, pytest.raises(ConnectionRefusedError):
            sock.connect(("localhost", 6379))
        assert external_network_attempts == []

    def test_a_server_opened_by_the_test_is_reachable(self):
        """テストが自分で開くサーバ（ポート 0 で OS に選ばせる）は遮断しない。偽サーバのテストのため。"""
        with socket.create_server(("127.0.0.1", 0)) as server:
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                pass

    def test_publish_fails_without_reaching_redis(self):
        """本物の redis クライアントでも閉じたポートで止まること。"""
        with pytest.raises(redis.ConnectionError):
            bus.publish({"type": "test", "payload": {}})

    def test_swallowed_attempt_still_fails_the_test(self, tmp_path):
        """本番のコードは例外を広く握るので、握りつぶされた試みでもテストを赤くすること。

        内側の pytest は同じ conftest を使う別プロセスで流す。
        """
        (tmp_path / "conftest.py").write_bytes((_REPO_ROOT / "tests" / "conftest.py").read_bytes())
        (tmp_path / "test_swallow.py").write_text(textwrap.dedent("""
            import socket

            def test_swallows_the_error():
                try:
                    socket.create_connection(("192.0.2.1", 80), timeout=1)
                except Exception:
                    pass
        """), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
             f"--rootdir={tmp_path}", f"--basetemp={tmp_path / 'inner'}", str(tmp_path / "test_swallow.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False, cwd=tmp_path,
        )
        assert done.returncode == 1, done.stdout + done.stderr
        assert "192.0.2.1:80" in done.stdout
