"""
obs.py — OBS WebSocket クライアント（立ち絵切り替え用）

責務:
  - OBS Studio の WebSocket に接続し、立ち絵ソースを show/hide する
  - LLM が返す pose 値（neutral/happy/angry/sad/fun）に応じて、
    指定キャラクターの対応ソースを表示、他ポーズのソースを非表示にする
  - OBS 未接続時・obsws-python 未インストール時は no-op（warning ログのみ）
    で L2 の主機能はクラッシュしない

【OBS 側の構成】
  Scene: (任意、親シーン名は L2 から意識しない)
  ├── Group: mimi              ← グループ名 = キャラクター slug
  │   ├── mimi_neutral (Image Source)
  │   ├── mimi_happy
  │   ├── mimi_angry
  │   ├── mimi_sad
  │   └── mimi_fun
  ├── Group: chisame
  │   └── (同 5 ソース)
  └── Group: sakura
      └── (同 5 ソース)

  ※ OBS WebSocket API では Group は内部的に "Scene" として扱われるため、
     API 呼び出し時の scene_name にはキャラクター slug（グループ名）を渡す。

【ソース命名規則】
  `{character_slug}_{pose}`
  例: mimi_neutral, mimi_happy, mimi_angry, mimi_sad, mimi_fun

【環境変数】
  L2_OBS_WS_URL      — OBS WebSocket URL (ws://host:port 形式)
  L2_OBS_WS_PASSWORD — OBS WebSocket のパスワード

【前提パッケージ】
  uv sync --extra obsws  # obsws-python が必要
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

VALID_POSES: frozenset[str] = frozenset({"neutral", "happy", "angry", "sad", "fun"})

# モジュールレベル状態（プロセス内シングルトン）
_client: Any = None
_connected: bool = False
_init_attempted: bool = False
# キー: (group_name, source_name) → SceneItemId
_item_id_cache: dict[tuple[str, str], int] = {}


def _parse_ws_url(url: str) -> tuple[str, int]:
    """
    ws://host:port または host:port 形式を (host, port) に分解する。

    port 省略時はデフォルト 4455。
    """
    if "://" not in url:
        url = "ws://" + url
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4455
    return host, port


def init_obs() -> None:
    """
    OBS WebSocket に接続する。起動時に一度呼ぶ。

    L2_OBS_WS_URL 未設定 / obsws-python 未インストール / 接続失敗時は
    no-op（_client=None のまま）。再呼び出しは無視される。
    """
    global _client, _connected, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    url = os.environ.get("L2_OBS_WS_URL")
    if not url:
        logger.info("L2_OBS_WS_URL 未設定。OBS 連携は無効。")
        return

    try:
        import obsws_python as _obsws  # noqa: F401
    except ImportError:
        logger.warning(
            "obsws-python 未インストール。OBS 連携は無効。"
            " uv sync --extra obsws でインストール可能。"
        )
        return

    host, port = _parse_ws_url(url)
    password = os.environ.get("L2_OBS_WS_PASSWORD")

    try:
        _client = _obsws.ReqClient(host=host, port=port, password=password, timeout=3)
        _connected = True
        # 配信中のコンソール表示で host/port を漏らさないため env var 名のみログ
        logger.info("OBS 接続成功 (env: L2_OBS_WS_URL)")
    except Exception as exc:  # noqa: BLE001
        # 例外メッセージに host/port が含まれる可能性があるため type 名のみ表示
        logger.warning("OBS 接続失敗: %s", type(exc).__name__)
        _client = None
        _connected = False


def _get_item_id(group_name: str, source_name: str) -> int | None:
    """
    グループ内のソースの SceneItemId を取得する（キャッシュ付き）。

    OBS WebSocket API v5 では Group は内部的に Scene として扱われるため、
    `scene_name` パラメータにはグループ名を渡す。
    """
    cache_key = (group_name, source_name)
    if cache_key in _item_id_cache:
        return _item_id_cache[cache_key]

    if _client is None:
        return None

    try:
        resp = _client.get_scene_item_id(group_name, source_name)
        # obsws-python v5 の戻り値は dataclass: scene_item_id 属性を持つ
        item_id: int | None = getattr(resp, "scene_item_id", None)
        if item_id is None:
            return None
        _item_id_cache[cache_key] = item_id
        return item_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("SceneItemId 取得失敗: %s/%s (%s)", group_name, source_name, exc)
        return None


def set_pose(character_slug: str, pose: str) -> None:
    """
    指定キャラクターの立ち絵を pose に切り替える。

    同キャラクターの他 pose ソースは非表示にし、target のみ表示する。
    OBS Group 構成前提: グループ名 = キャラクター slug。
    OBS 未接続時は no-op（警告なし）。
    未知 pose 値は "neutral" にフォールバック。

    Args:
        character_slug: キャラクター slug (e.g., "mimi") — グループ名としても使用される
        pose:           立ち絵 slug ("neutral" / "happy" / "angry" / "sad" / "fun")
    """
    global _init_attempted

    # 遅延初期化: 未呼び出しならここで接続を試みる
    if not _init_attempted:
        init_obs()

    if _client is None or not _connected:
        return

    # pose 値をサニタイズ
    if pose not in VALID_POSES:
        if pose:
            logger.warning("不明な pose 値: %r → 'neutral' にフォールバック", pose)
        pose = "neutral"

    # グループ名 = キャラクター slug として API を呼ぶ
    # (OBS WebSocket API v5 では Group が Scene として扱われるため)
    group_name = character_slug
    target_name = f"{character_slug}_{pose}"

    # ターゲットソース存在確認
    target_id = _get_item_id(group_name, target_name)
    if target_id is None:
        # キャラクターのグループ or ソースが OBS に存在しない → silent no-op
        return

    # 同キャラクターの全 pose ソースを enabled=(p == pose) に設定
    for p in VALID_POSES:
        source_name = f"{character_slug}_{p}"
        item_id = _get_item_id(group_name, source_name)
        if item_id is None:
            continue

        try:
            _client.set_scene_item_enabled(group_name, item_id, (p == pose))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 切替失敗: %s", source_name, exc)

    logger.info("OBS pose 切替: %s → %s", character_slug, pose)


def disconnect_obs() -> None:
    """OBS WebSocket 接続を閉じる（シャットダウン時）。"""
    global _client, _connected, _init_attempted
    if _client is not None:
        try:
            _client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    _client = None
    _connected = False
    _init_attempted = False
    _item_id_cache.clear()


# ─── テスト用 internal ──────────────────────────────────────────

def _reset_for_tests() -> None:
    """テスト用: 内部状態を完全リセット。"""
    global _client, _connected, _init_attempted
    _client = None
    _connected = False
    _init_attempted = False
    _item_id_cache.clear()


def _set_client_for_tests(client: Any, connected: bool = True) -> None:
    """テスト用: モッククライアントを注入し init_attempted=True にする。"""
    global _client, _connected, _init_attempted
    _client = client
    _connected = connected
    _init_attempted = True
    _item_id_cache.clear()
