"""
test_skill_loader.py — skill_loader のユニットテスト

Sprint Axis D Block 4: Skills 定義 v0.1
"""

import pytest

from lab_lounge.skill_loader import (
    build_skills_prompt,
    load_character_skill,
    load_common_skills,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """各テスト後にキャッシュをリセット。"""
    yield
    reset_cache()


class TestLoadCommonSkills:
    """共通 Skill ファイルの読み込みを検証する。"""

    def test_loads_all_common_skills(self, tmp_path):
        """skills/common/ の全 .md ファイルが読み込まれる。"""
        common = tmp_path / "common"
        common.mkdir()
        (common / "a_first.md").write_text("# First Skill", encoding="utf-8")
        (common / "b_second.md").write_text("# Second Skill", encoding="utf-8")

        result = load_common_skills(tmp_path)
        assert "# First Skill" in result
        assert "# Second Skill" in result

    def test_alphabetical_order(self, tmp_path):
        """ファイルはアルファベット順に結合される。"""
        common = tmp_path / "common"
        common.mkdir()
        (common / "b.md").write_text("BBB", encoding="utf-8")
        (common / "a.md").write_text("AAA", encoding="utf-8")

        result = load_common_skills(tmp_path)
        assert result.index("AAA") < result.index("BBB")

    def test_empty_directory(self, tmp_path):
        """common/ が空ディレクトリなら空文字を返す。"""
        common = tmp_path / "common"
        common.mkdir()
        result = load_common_skills(tmp_path)
        assert result == ""

    def test_missing_directory(self, tmp_path):
        """common/ ディレクトリが存在しなければ空文字を返す。"""
        result = load_common_skills(tmp_path)
        assert result == ""

    def test_non_md_files_ignored(self, tmp_path):
        """*.md 以外のファイルは無視される。"""
        common = tmp_path / "common"
        common.mkdir()
        (common / "skill.md").write_text("Skill Content", encoding="utf-8")
        (common / "notes.txt").write_text("Not a skill", encoding="utf-8")

        result = load_common_skills(tmp_path)
        assert "Skill Content" in result
        assert "Not a skill" not in result

    def test_real_skills_directory(self):
        """本番の skills/common/ が読み込めることを確認。"""
        result = load_common_skills()
        # 本番ディレクトリには少なくとも tool_routing.md がある
        assert "ツール" in result or "retrieve_memory" in result


class TestLoadCharacterSkill:
    """キャラ別 Skill ファイルの読み込みを検証する。"""

    def test_loads_character_file(self, tmp_path):
        """skills/characters/{slug}.md が読み込まれる。"""
        chars = tmp_path / "characters"
        chars.mkdir()
        (chars / "mimi.md").write_text("# ミミ Skill", encoding="utf-8")

        result = load_character_skill("mimi", tmp_path)
        assert "# ミミ Skill" in result

    def test_missing_character_returns_empty(self, tmp_path):
        """存在しないキャラの場合は空文字を返す。"""
        chars = tmp_path / "characters"
        chars.mkdir()
        result = load_character_skill("nonexistent", tmp_path)
        assert result == ""

    def test_real_mimi_skill(self):
        """本番の skills/characters/mimi.md が読み込めることを確認。"""
        result = load_character_skill("mimi")
        assert "ミミ" in result

    def test_real_chisame_skill(self):
        """本番の skills/characters/chisame.md が読み込めることを確認。"""
        result = load_character_skill("chisame")
        assert "ちさめ" in result

    def test_real_sakura_skill(self):
        """本番の skills/characters/sakura.md が読み込めることを確認。"""
        result = load_character_skill("sakura")
        assert "さくら" in result

    def test_real_ruka_skill(self):
        """本番の skills/characters/ruka.md が読み込めることを確認。"""
        result = load_character_skill("ruka")
        assert "ルカ" in result

    def test_octamaid_returns_empty(self):
        """オクタメイドは Skill ファイル未作成のため空文字。"""
        result = load_character_skill("octamaid")
        assert result == ""


class TestBuildSkillsPrompt:
    """共通 + キャラ別 Skill の結合を検証する。"""

    def test_combines_common_and_character(self, tmp_path):
        """共通 Skill とキャラ別 Skill が結合される。"""
        common = tmp_path / "common"
        common.mkdir()
        (common / "tool.md").write_text("## Tool Routing", encoding="utf-8")

        chars = tmp_path / "characters"
        chars.mkdir()
        (chars / "mimi.md").write_text("## Mimi Bias", encoding="utf-8")

        result = build_skills_prompt("mimi", tmp_path)
        assert "## Tool Routing" in result
        assert "## Mimi Bias" in result
        # 共通 Skill が先、キャラ別が後
        assert result.index("Tool Routing") < result.index("Mimi Bias")

    def test_common_only_when_no_character_file(self, tmp_path):
        """キャラ別ファイルがなければ共通 Skill のみ。"""
        common = tmp_path / "common"
        common.mkdir()
        (common / "tool.md").write_text("## Common Only", encoding="utf-8")
        (tmp_path / "characters").mkdir()

        result = build_skills_prompt("unknown_char", tmp_path)
        assert "## Common Only" in result

    def test_real_mimi_prompt(self):
        """本番環境でミミの Skills プロンプトが構築できる。"""
        result = build_skills_prompt("mimi")
        # 共通 Skill (tool_routing) とキャラ別 (ミミ) の両方が含まれる
        assert "retrieve_memory" in result  # tool_routing から
        assert "ミミ" in result  # キャラ別から
