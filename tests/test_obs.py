"""
test_obs.py — obs.py (OBS WebSocket クライアント) のテスト

obsws-python をモックして実接続なしで検証する。
conftest.py の reset_obs_module_state fixture でテスト間の状態分離を担保。

【Group 構成前提】
OBS WebSocket API v5 では Group は内部的に Scene として扱われるため、
API 呼び出し時の scene_name にはキャラクター slug（グループ名）を渡す。
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import lab_lounge.obs as obs_mod


# ─── init_obs ────────────────────────────────────────────────────


class TestInitObs:
    def test_no_url_is_noop(self, monkeypatch):
        """L2_OBS_WS_URL 未設定なら no-op (接続せずに終了)。"""
        monkeypatch.delenv("L2_OBS_WS_URL", raising=False)
        obs_mod.init_obs()
        assert obs_mod._client is None
        assert obs_mod._connected is False

    def test_connection_success(self, monkeypatch):
        """接続成功時は _client がセットされる。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://localhost:4455")
        monkeypatch.setenv("L2_OBS_WS_PASSWORD", "secret")

        mock_client_instance = MagicMock()
        mock_obsws = MagicMock()
        mock_obsws.ReqClient.return_value = mock_client_instance

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            obs_mod.init_obs()

        assert obs_mod._client is mock_client_instance
        assert obs_mod._connected is True
        # ReqClient が正しい引数で呼ばれたことを確認
        mock_obsws.ReqClient.assert_called_once()
        call_kwargs = mock_obsws.ReqClient.call_args.kwargs
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 4455
        assert call_kwargs["password"] == "secret"

    def test_connection_success_log_does_not_leak_host_port_password(
        self, monkeypatch, caplog,
    ):
        """接続成功ログに host / port / password が含まれない (配信中の漏洩防止)。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://192.168.11.101:4455")
        monkeypatch.setenv("L2_OBS_WS_PASSWORD", "secretpass123")

        mock_obsws = MagicMock()
        mock_obsws.ReqClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            with caplog.at_level("INFO", logger="lab_lounge.obs"):
                obs_mod.init_obs()

        all_messages = " ".join(r.getMessage() for r in caplog.records)
        # 機密情報がログに出ない
        assert "192.168.11.101" not in all_messages
        assert "4455" not in all_messages
        assert "secretpass123" not in all_messages
        # 環境変数名のみが含まれる
        assert "L2_OBS_WS_URL" in all_messages

    def test_connection_failure_log_does_not_leak_details(
        self, monkeypatch, caplog,
    ):
        """接続失敗ログに host / port / password / 例外詳細が含まれない。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://192.168.11.101:4455")
        monkeypatch.setenv("L2_OBS_WS_PASSWORD", "secretpass123")

        mock_obsws = MagicMock()
        # 例外メッセージに host/port が含まれる典型ケース
        mock_obsws.ReqClient.side_effect = ConnectionRefusedError(
            "Connection refused: 192.168.11.101:4455 (auth=secretpass123)"
        )

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            with caplog.at_level("WARNING", logger="lab_lounge.obs"):
                obs_mod.init_obs()

        all_messages = " ".join(r.getMessage() for r in caplog.records)
        assert "192.168.11.101" not in all_messages
        assert "secretpass123" not in all_messages
        # 例外クラス名のみが残る
        assert "ConnectionRefusedError" in all_messages

    def test_connection_failure_graceful(self, monkeypatch):
        """接続失敗時は _client=None のまま、例外を投げない。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://localhost:4455")

        mock_obsws = MagicMock()
        mock_obsws.ReqClient.side_effect = RuntimeError("connection refused")

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            obs_mod.init_obs()  # should not raise

        assert obs_mod._client is None
        assert obs_mod._connected is False

    def test_obsws_python_not_installed(self, monkeypatch):
        """obsws-python 未インストール時は no-op。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://localhost:4455")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "obsws_python":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            obs_mod.init_obs()

        assert obs_mod._client is None

    def test_init_only_runs_once(self, monkeypatch):
        """init_obs を複数回呼んでも接続試行は 1 回のみ。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://localhost:4455")

        mock_obsws = MagicMock()
        mock_obsws.ReqClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            obs_mod.init_obs()
            obs_mod.init_obs()  # 2 回目
            obs_mod.init_obs()  # 3 回目

        # ReqClient は 1 回しか呼ばれない
        assert mock_obsws.ReqClient.call_count == 1

    def test_parse_ws_url_with_scheme(self, monkeypatch):
        """ws://host:port 形式を正しくパース。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "ws://192.168.1.100:4455")

        mock_obsws = MagicMock()
        mock_obsws.ReqClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            obs_mod.init_obs()

        call_kwargs = mock_obsws.ReqClient.call_args.kwargs
        assert call_kwargs["host"] == "192.168.1.100"
        assert call_kwargs["port"] == 4455

    def test_parse_ws_url_without_scheme(self, monkeypatch):
        """host:port 形式（スキーマなし）も受け付ける。"""
        monkeypatch.setenv("L2_OBS_WS_URL", "192.168.1.100:4455")

        mock_obsws = MagicMock()
        mock_obsws.ReqClient.return_value = MagicMock()

        with patch.dict(sys.modules, {"obsws_python": mock_obsws}):
            obs_mod.init_obs()

        call_kwargs = mock_obsws.ReqClient.call_args.kwargs
        assert call_kwargs["host"] == "192.168.1.100"
        assert call_kwargs["port"] == 4455


# ─── set_pose ────────────────────────────────────────────────────


def _make_mock_client(item_ids: dict[tuple[str, str], int]) -> MagicMock:
    """
    `{(group_name, source_name): scene_item_id}` マッピングを返すモッククライアント。

    新実装の obs.py は GetGroupSceneItemList で group 全体を一括取得するため、
    `get_group_scene_item_list(group_name)` を mock してその group の全 item を
    list で返す。group が登録されていない場合は obsws-python と同様に例外を投げる。
    """
    from collections import defaultdict

    client = MagicMock()

    # item_ids を group 単位で再構成
    group_items: dict[str, list[dict]] = defaultdict(list)
    for (group, source), item_id in item_ids.items():
        group_items[group].append({"sourceName": source, "sceneItemId": item_id})

    def get_group_scene_item_list(group_name):
        items = group_items.get(group_name)
        if not items:
            raise RuntimeError(f"Group not found: {group_name}")
        resp = MagicMock()
        resp.scene_items = items
        return resp

    client.get_group_scene_item_list.side_effect = get_group_scene_item_list
    client.set_scene_item_enabled = MagicMock()
    return client


def _mimi_group_item_ids() -> dict[tuple[str, str], int]:
    """mimi グループの 5 ソース item_id マッピング。"""
    return {
        ("mimi", "mimi_neutral"): 1,
        ("mimi", "mimi_happy"): 2,
        ("mimi", "mimi_angry"): 3,
        ("mimi", "mimi_sad"): 4,
        ("mimi", "mimi_fun"): 5,
    }


class TestSetPose:
    def test_no_client_is_noop(self):
        """_client=None なら set_pose は no-op。"""
        obs_mod._reset_for_tests()
        obs_mod._init_attempted = True  # 遅延初期化をスキップ
        obs_mod.set_pose("mimi", "happy")  # should not raise

    def test_uses_group_name_as_scene_name(self):
        """set_scene_item_enabled にグループ名 (=slug) が渡される。"""
        client = _make_mock_client(_mimi_group_item_ids())
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "happy")

        # 全ての set_scene_item_enabled 呼び出しで scene_name="mimi" になる
        calls = client.set_scene_item_enabled.call_args_list
        assert len(calls) == 5
        for call in calls:
            assert call.args[0] == "mimi", (
                f"scene_name should be 'mimi' (group), got {call.args[0]}"
            )

    def test_target_shown_others_hidden(self):
        """target pose を enabled=True、他を enabled=False にする。"""
        client = _make_mock_client(_mimi_group_item_ids())
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "happy")

        # set_scene_item_enabled が 5 回呼ばれる
        assert client.set_scene_item_enabled.call_count == 5

        # mimi_happy (id=2) だけ enabled=True
        calls = client.set_scene_item_enabled.call_args_list
        happy_call = next(c for c in calls if c.args[1] == 2)
        assert happy_call.args[2] is True

        for other_id in [1, 3, 4, 5]:
            other_call = next(c for c in calls if c.args[1] == other_id)
            assert other_call.args[2] is False

    def test_unknown_pose_fallback_to_neutral(self):
        """未知の pose 値は neutral にフォールバック。"""
        client = _make_mock_client({
            ("mimi", "mimi_neutral"): 1,
            ("mimi", "mimi_happy"): 2,
        })
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "ecstatic")

        # mimi_neutral (id=1) だけ True になる
        calls = client.set_scene_item_enabled.call_args_list
        neutral_call = next(c for c in calls if c.args[1] == 1)
        assert neutral_call.args[2] is True

    def test_empty_pose_fallback_to_neutral(self):
        """空文字 pose も neutral にフォールバック。"""
        client = _make_mock_client({("mimi", "mimi_neutral"): 1})
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "")

        calls = client.set_scene_item_enabled.call_args_list
        assert len(calls) == 1
        assert calls[0].args[1] == 1
        assert calls[0].args[2] is True

    def test_target_group_not_found_noop(self):
        """キャラクターのグループが存在しなければ no-op。"""
        client = _make_mock_client({})  # 何もない
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("unknown_char", "happy")

        client.set_scene_item_enabled.assert_not_called()

    def test_group_loaded_once(self):
        """group は初回 set_pose で 1 度だけ一括ロードされ、以降は API 呼び出されない。"""
        client = _make_mock_client(_mimi_group_item_ids())
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "happy")
        # 初回: get_group_scene_item_list が 1 回呼ばれる
        assert client.get_group_scene_item_list.call_count == 1

        obs_mod.set_pose("mimi", "sad")
        # 2 回目: API は呼ばれない (cache から取得)
        assert client.get_group_scene_item_list.call_count == 1

        obs_mod.set_pose("mimi", "neutral")
        # 3 回目も同じ
        assert client.get_group_scene_item_list.call_count == 1

    def test_cache_is_per_group(self):
        """キャッシュはグループ単位で分離される (mimi と chisame で別々の API call)。"""
        client = _make_mock_client({
            **_mimi_group_item_ids(),
            ("chisame", "chisame_neutral"): 10,
            ("chisame", "chisame_happy"): 11,
            ("chisame", "chisame_angry"): 12,
            ("chisame", "chisame_sad"): 13,
            ("chisame", "chisame_fun"): 14,
        })
        obs_mod._set_client_for_tests(client, connected=True)

        obs_mod.set_pose("mimi", "happy")
        # mimi の初回ロードのみ
        assert client.get_group_scene_item_list.call_count == 1

        obs_mod.set_pose("chisame", "happy")
        # chisame の初回ロード → 2 回目
        assert client.get_group_scene_item_list.call_count == 2

        obs_mod.set_pose("chisame", "sad")
        # 同 chisame は cache 済み → 増えない
        assert client.get_group_scene_item_list.call_count == 2

    def test_set_scene_item_enabled_failure_logged_not_raised(self):
        """set_scene_item_enabled が失敗しても例外を伝播させない。"""
        client = _make_mock_client({
            ("mimi", "mimi_neutral"): 1,
            ("mimi", "mimi_happy"): 2,
        })
        client.set_scene_item_enabled.side_effect = RuntimeError("API error")
        obs_mod._set_client_for_tests(client, connected=True)

        # should not raise
        obs_mod.set_pose("mimi", "happy")

    def test_all_valid_poses_accepted(self):
        """5 種の pose 値を全て受け付ける。"""
        client = _make_mock_client(_mimi_group_item_ids())
        obs_mod._set_client_for_tests(client, connected=True)

        for pose in ("neutral", "happy", "angry", "sad", "fun"):
            obs_mod.set_pose("mimi", pose)  # should not raise

    def test_lazy_init_on_first_call(self, monkeypatch):
        """set_pose が init_obs を呼んでいない状態で呼ばれたら遅延初期化する。"""
        obs_mod._reset_for_tests()
        monkeypatch.delenv("L2_OBS_WS_URL", raising=False)

        obs_mod.set_pose("mimi", "happy")  # no-op (URL 未設定)

        # init_attempted が True になっている
        assert obs_mod._init_attempted is True


# ─── VALID_POSES 定数 ────────────────────────────────────────────


class TestValidPoses:
    def test_contains_base_poses(self):
        """base 5 個 (neutral/happy/angry/sad/fun) は必ず含まれる。"""
        for pose in ("neutral", "happy", "angry", "sad", "fun"):
            assert pose in obs_mod.VALID_POSES

    def test_contains_mimi_specials(self):
        """mimi 固有の special pose が含まれる (system_mimi.txt と同期)。"""
        for pose in ("special_sulky", "special_pondering", "special_amused"):
            assert pose in obs_mod.VALID_POSES

    def test_contains_chisame_specials(self):
        """chisame 固有の special pose が含まれる (system_chisame.txt と同期)。"""
        for pose in ("special_overdrive", "special_doya", "special_bosoboso"):
            assert pose in obs_mod.VALID_POSES

    def test_contains_sakura_specials(self):
        """sakura 固有の special pose が含まれる (system_sakura.txt と同期)。"""
        for pose in ("special_whisper", "special_cool"):
            assert pose in obs_mod.VALID_POSES
