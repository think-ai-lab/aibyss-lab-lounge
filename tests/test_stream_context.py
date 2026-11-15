"""
test_stream_context.py — 配信文脈ロード関数のユニットテスト

stream_context.load_stream_context() の振る舞い:
  - 引数 path 優先
  - 環境変数 L2_STREAM_CONTEXT_FILE で上書き可
  - ファイル未存在 / 空文字列なら None を返す (後方互換)
"""

from pathlib import Path

from lab_lounge.stream_context import load_stream_context


class TestLoadStreamContext:
    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """ファイル未存在時は None (= 配信文脈なしとして従来通り動作)。"""
        missing = tmp_path / "does_not_exist.md"
        assert load_stream_context(path=missing) is None

    def test_returns_none_when_file_empty(self, tmp_path: Path):
        """空ファイル時も None (空セクションが LLM を混乱させないため)。"""
        empty = tmp_path / "empty.md"
        empty.write_text("", encoding="utf-8")
        assert load_stream_context(path=empty) is None

    def test_returns_none_when_file_whitespace_only(self, tmp_path: Path):
        """空白のみのファイルも None (strip 後に空文字列扱い)。"""
        whitespace = tmp_path / "ws.md"
        whitespace.write_text("   \n\n   \t\n", encoding="utf-8")
        assert load_stream_context(path=whitespace) is None

    def test_returns_text_when_file_present(self, tmp_path: Path):
        """ファイル本文が strip 後に返る。"""
        ctx = tmp_path / "today.md"
        ctx.write_text(
            "\n\n## 今日の予定\n\n- 大神プレイ\n\n",
            encoding="utf-8",
        )
        result = load_stream_context(path=ctx)
        assert result is not None
        assert result.startswith("## 今日の予定")
        assert "大神プレイ" in result
        # strip 確認 (前後空白除去)
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    def test_env_var_override(self, tmp_path: Path, monkeypatch):
        """path=None かつ環境変数指定時は環境変数のパスを参照する。"""
        ctx = tmp_path / "from_env.md"
        ctx.write_text("環境変数経由の配信文脈", encoding="utf-8")
        monkeypatch.setenv("L2_STREAM_CONTEXT_FILE", str(ctx))
        result = load_stream_context()
        assert result == "環境変数経由の配信文脈"

    def test_path_arg_takes_precedence_over_env_var(
        self, tmp_path: Path, monkeypatch
    ):
        """path 引数があれば環境変数より優先される (テストの明示性確保)。"""
        env_ctx = tmp_path / "env.md"
        env_ctx.write_text("env", encoding="utf-8")
        arg_ctx = tmp_path / "arg.md"
        arg_ctx.write_text("arg", encoding="utf-8")
        monkeypatch.setenv("L2_STREAM_CONTEXT_FILE", str(env_ctx))
        assert load_stream_context(path=arg_ctx) == "arg"

    def test_default_path_when_no_arg_no_env(
        self, tmp_path: Path, monkeypatch
    ):
        """引数も環境変数もない場合はデフォルトパスを読む。

        WHY: 実物の current.md は git に入らない手元の配信メモなので、テストからは読まない。
             既定のパスを一時ファイルへ差し替えて、読む順序だけを確かめる。
        """
        default = tmp_path / "current.md"
        default.write_text("既定の配信文脈", encoding="utf-8")
        monkeypatch.setattr("lab_lounge.stream_context._DEFAULT_PATH", default)
        monkeypatch.delenv("L2_STREAM_CONTEXT_FILE", raising=False)
        assert load_stream_context() == "既定の配信文脈"

    def test_returns_none_when_default_path_missing(
        self, tmp_path: Path, monkeypatch
    ):
        """デフォルトパスにファイルがなければ None (新しい clone の状態)。"""
        monkeypatch.setattr(
            "lab_lounge.stream_context._DEFAULT_PATH", tmp_path / "current.md"
        )
        monkeypatch.delenv("L2_STREAM_CONTEXT_FILE", raising=False)
        assert load_stream_context() is None

    def test_default_path_points_at_the_documented_place(self):
        """README が案内する場所から動いていないこと (中身は読まない)。"""
        from lab_lounge.stream_context import _DEFAULT_PATH

        assert _DEFAULT_PATH.parts[-3:] == ("data", "stream_context", "current.md")
