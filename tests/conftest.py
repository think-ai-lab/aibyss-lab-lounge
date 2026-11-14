"""
conftest.py — テストを実環境から切り離す共通設定

テストは次のものに触れない。先に環境を整え、それでも触れようとしたら止める。
  - 手元の .env と API キー（課金と、実発話の送信につながる）
  - 実サービス（Redis の正史、C2、V2、TTS、OBS）
  - git に入らない手元のデータ（新しい clone と結果が変わる）

環境の変更をファイルの先頭で行う理由:
  run_loop / run_once / emitter は import の瞬間に load_dotenv() を呼び、run_loop はログファイルも作る。
  テストファイルの import は fixture より先に走るので、fixture では間に合わない。
  このファイルがテストファイルより先に読まれることは pytest が保証する。

ほかに、次のものもここで用意する:
  - AIBYSS_SCHEMA_PATH を兄弟の aibyss-workspace から解決し、validator のキャッシュをテストごとに捨てる
  - OBS モジュールの状態をテストごとに捨てる
"""

import asyncio
import atexit
import errno
import ipaddress
import logging
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

# 環境から外す変数の接頭辞。キー、接続先、機能フラグをまとめて外す。
# WHY: 名前を 1 つずつ列挙すると、新しい変数（SDK が増やす接続先など）を足し忘れた瞬間に実環境へ漏れる。
#      必要なテストは monkeypatch.setenv で明示する。AIBYSS_SCHEMA_PATH は秘匿値でも接続先でもないので残す。
_SCRUBBED_PREFIXES = (
    "L2_",
    "OPENAI",
    "ANTHROPIC",
    "GOOGLE",
    "GEMINI",
    "TAVILY",
    "VOYAGE",
    "LANGSMITH",
    "LANGCHAIN",
    "REDIS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)

# 閉じたポート。REDIS_URL を消すと bus の既定（実 Redis）へ届くので、明示的に向け先を潰す。
_CLOSED_REDIS_URL = "redis://127.0.0.1:6"
# 本番の既定 aibyss:events と同じ名前にしない。万一 Redis へ届いても正史を汚さないように。
_TEST_STREAM_KEY = "aibyss:events-test"

# テストのプロセスだけが使う置き場所。ログ、作業ディレクトリ、手元のデータの身代わりをここへ集める。
_SESSION_DIR = Path(tempfile.mkdtemp(prefix="l2-tests-"))
_WORK_DIR = _SESSION_DIR / "cwd"
_WORK_DIR.mkdir()

# loopback でも、OS が自動で割り当てる範囲より下のポートには繋がせない。
# WHY: 手元で動いている実サービス（6379, 8100, 8101, 3200, 15021, 18080, 4455 など）はこの範囲にいる。
#      繋がると、サービスが起動しているかどうかでテストの結果が変わり、実物にも書き込んでしまう。
#      テストが自分で開くサーバはポート 0 で OS に選ばせるので、この範囲の上に来る（Linux は 32768 から、
#      Windows と macOS は 49152 から）。
_EPHEMERAL_PORT_MIN = 32768
# 例外: VOICEVOX の macOS / Linux での既定のポートは、OS が割り当てる範囲の中にある。
_HIGH_SERVICE_PORTS = {50021}

# テスト中に外へ出ようとした接続先。テストの終わりに確かめる。
_external_attempts: list[str] = []


def _scrubbed_names() -> list[str]:
    """今の環境にある、外すべき変数の名前を返す。"""
    return [name for name in os.environ if name.upper().startswith(_SCRUBBED_PREFIXES)]


def _isolated_values() -> dict[str, str]:
    """テストの間ずっと固定する環境変数を返す。"""
    return {
        # lab_lounge の load_dotenv() を止める。テストが手元の .env で変わらないように。
        "PYTHON_DOTENV_DISABLED": "1",
        "REDIS_URL": _CLOSED_REDIS_URL,
        "REDIS_STREAM_KEY": _TEST_STREAM_KEY,
        # 本流の publish (utterance.final など) の失敗を握らない。本番の既定 (true) では bus.publish_best_effort が
        # ERROR ログにして先へ進むので、publish のモックの漏れが緑のまま埋もれる。既定を確かめるテストは消して使う。
        "L2_BUS_FAIL_OPEN": "false",
        "LANGSMITH_TRACING": "false",
        # LANGSMITH_TRACING=true を立てるテストでも、実発話を含むトレースを外へ送らない。
        "LANGSMITH_ENDPOINT": "http://127.0.0.1:6",
        # run_loop の import で作られるログを、実リポの logs/ へ残さない。
        "L2_LOG_DIR": str(_SESSION_DIR / "logs"),
        # 既定のパスは手元の実物を指す。存在しないパスにして、新しい clone と同じ状態にする。
        "L2_STREAM_CONTEXT_FILE": str(_SESSION_DIR / "no-such-stream-context.md"),
        "L2_IRODORI_VOICES_JSON": str(_SESSION_DIR / "no-such-voices.json"),
        "L2_IRODORI_READINGS_JSON": str(_SESSION_DIR / "no-such-readings.json"),
        # 既定の待ち時間（2 秒 x リトライ）がテストの数だけ積み上がるのを防ぐ。
        "L2_VOICEPEAK_RETRY_WAIT_SEC": "0",
        "L2_VOICEPEAK_MAX_RETRIES": "0",
    }


def _remove_session_dir() -> None:
    """置き場所を消す。ログファイルを開いたままだと Windows では消せないので、先に閉じる。"""
    logging.shutdown()
    shutil.rmtree(_SESSION_DIR, ignore_errors=True)


atexit.register(_remove_session_dir)

for _name in _scrubbed_names():
    del os.environ[_name]
os.environ.update(_isolated_values())


# ─── 外への接続を止める ─────────────────────────────────────────────


class ExternalNetworkBlocked(RuntimeError):
    """テストが loopback の外へ接続しようとした。

    ConnectionError にしない理由: SDK は接続エラーを待って再試行するので、テストが遅くなるだけで止まらない。
    """


def _is_loopback(host: str) -> bool:
    """host が自分自身を指すなら True を返す。"""
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def _block_external(destination: str) -> None:
    """接続先を記録して止める。

    Raises:
        ExternalNetworkBlocked: 必ず送出する
    """
    _external_attempts.append(destination)
    raise ExternalNetworkBlocked(f"テストから外へ接続しようとしました: {destination}")


def _check_destination(address: object) -> None:
    """接続先を確かめ、繋いではいけない相手なら接続の前に止める。

    Args:
        address: socket.connect に渡す値

    Raises:
        ExternalNetworkBlocked: loopback の外を指している
        ConnectionRefusedError: loopback の、実サービスがいる範囲のポートを指している。
            閉じたポートと同じ扱いにする。Windows の実際の拒否は約 2 秒かかるので、待たずに返す
    """
    if not isinstance(address, tuple):
        return  # Unix ドメインソケットなど、ネットワークの外
    host, port = address[0], address[1]
    host = host.decode() if isinstance(host, (bytes, bytearray)) else host
    if not _is_loopback(host):
        _block_external(f"{host}:{port}")
    if int(port) < _EPHEMERAL_PORT_MIN or int(port) in _HIGH_SERVICE_PORTS:
        raise ConnectionRefusedError(errno.ECONNREFUSED, f"テストでは接続しません: {host}:{port}")


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _guarded_connect(self: socket.socket, address: object) -> None:
    """socket.socket.connect の差し替え。"""
    _check_destination(address)
    _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    """socket.socket.connect_ex の差し替え。拒否は例外でなく番号で返す（本物と同じ）。"""
    try:
        _check_destination(address)
    except ConnectionRefusedError as exc:
        return exc.errno
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> list:
    """socket.getaddrinfo の差し替え。外の名前は解決の前に止める。

    WHY: 名前解決そのものが外への通信で、ネットワークの有無で失敗の形も変わる。
         asyncio は名前のある接続先をここで解決するので、この先の接続の差し替えを通らない経路も止められる。
    """
    name = host.decode() if isinstance(host, (bytes, bytearray)) else host
    if isinstance(name, str) and name and name != "localhost":
        try:
            ipaddress.ip_address(name.split("%")[0])
        except ValueError:
            port = args[0] if args else kwargs.get("port")
            _block_external(f"{name}:{port}")
    return _real_getaddrinfo(host, *args, **kwargs)


# 収集中の import や、前のテストが残したスレッドも対象なので、fixture ではなく最初に 1 度だけ差し替える。
socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.getaddrinfo = _guarded_getaddrinfo

if sys.platform == "win32":
    # WHY: Windows の既定のイベントループは socket.connect を呼ばず ConnectEx へ直接渡すので、上の差し替えを素通りする。
    _real_proactor_connect = asyncio.windows_events.IocpProactor.connect

    def _guarded_proactor_connect(self: object, conn: object, address: object) -> object:
        """IocpProactor.connect の差し替え。"""
        _check_destination(address)
        return _real_proactor_connect(self, conn, address)

    asyncio.windows_events.IocpProactor.connect = _guarded_proactor_connect


@pytest.fixture(autouse=True)
def fail_on_external_network():
    """外へ接続しようとしたテストを失敗にする。

    WHY: 本番のコードは配信を止めないために例外を広く握るので、遮断の例外だけでは緑のまま見逃す。
         連続するテストの間に起きた試み（前のテストが残したスレッド）も、見逃さないように次のテストで報告する。
    """
    yield
    if _external_attempts:
        found = sorted(set(_external_attempts))
        _external_attempts.clear()
        pytest.fail(f"テストの間に外への接続がありました。該当の呼び出しをモックしてください: {found}")


@pytest.fixture
def external_network_attempts() -> list[str]:
    """遮断した接続先の一覧を返す（遮断そのものを確かめるテスト用）。読んだら空にすること。"""
    return _external_attempts


# ─── 環境と手元のデータ ─────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="session")
def run_in_work_dir():
    """作業ディレクトリをリポジトリの外へ移す。

    WHY: ./data/audio や ./logs の既定はここから解決されるので、実リポへ書き込まない。
         pydantic-settings も作業ディレクトリの .env を読むので、手元の .env を開かない。
    """
    original = os.getcwd()
    os.chdir(_WORK_DIR)
    yield
    os.chdir(original)


@pytest.fixture(autouse=True)
def isolate_environment():
    """テストごとに環境を決まった状態へ戻す。

    前のテストが os.environ を直接変えていても、順序で結果が変わらないようにする。
    real mode が必要なテストは、それぞれの fixture で明示的に setenv する。

    WHY: monkeypatch を通さないのは、テストの後で前の値へ戻す仕組みが、書き残しまで元へ戻してしまうため。
    """
    for name in _scrubbed_names():
        del os.environ[name]
    os.environ.update(_isolated_values())


@pytest.fixture(autouse=True)
def redirect_local_data(monkeypatch):
    """git に入らないフィラーのキャッシュと音声を見ないようにする。

    WHY: キャッシュは手元にだけあるので、あると再生の経路に入り、新しい clone と結果が変わる。
         音声の置き場所はリポジトリの中の絶対パスなので、作業ディレクトリを移しても実リポへ書く。
    """
    monkeypatch.setattr("lab_lounge.filler._FILLER_CACHE_DIR", _SESSION_DIR / "filler_cache")
    monkeypatch.setattr("lab_lounge.filler._FILLER_AUDIO_DIR", _SESSION_DIR / "audio")


@pytest.fixture(autouse=True)
def set_schema_path(monkeypatch):
    """スキーマのパスを確実に設定し、テスト間で validator のキャッシュをリセットする。

    兄弟の aibyss-workspace がなければ、外から渡された AIBYSS_SCHEMA_PATH をそのまま使う。
    """
    here = Path(__file__).resolve().parent
    candidate = here.parents[1] / "aibyss-workspace" / "specs" / "event-envelope-0.1.schema.json"
    if candidate.is_file():
        monkeypatch.setenv("AIBYSS_SCHEMA_PATH", str(candidate))

    # validator キャッシュをテスト前後にクリア
    from lab_lounge.events import _reset_validator_cache
    _reset_validator_cache()
    yield
    _reset_validator_cache()


@pytest.fixture(autouse=True)
def reset_bus_state():
    """bus の「直前に Redis へ届かなかった」記録を、テストの前後で消す。

    WHY: 記録はモジュールの状態なので、残ると次のテストの publish が L2_BUS_RETRY_AFTER_SEC の間、
         接続を試さずに失敗する。テストの順序で結果が変わらないようにする。
    """
    from lab_lounge.bus import _reset_for_tests
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture(autouse=True)
def reset_obs_module_state():
    """
    obs モジュールのグローバル状態をテスト前後でリセットする。

    set_pose / init_obs がテスト間で状態を持ち越さないようにする。
    """
    from lab_lounge.obs import _reset_for_tests
    _reset_for_tests()
    yield
    _reset_for_tests()
