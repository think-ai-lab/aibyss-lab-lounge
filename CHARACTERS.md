# Characters

本リポジトリのコードは Apache-2.0 ですが、Think-AI Lab. のキャラクター
（坂東ルカ／ミミ・オクタヴィア／波心ちさめ／八重笠さくら／オクタメイド／アルカ、
その他関連キャラクターの名称・人格設定・詳細設定・プロンプト・画像・音声設定）は
ライセンスの対象外であり、すべての権利を Think-AI Lab. が留保します。

- 学習・研究目的での参照や、出典明記のうえでの短い引用は歓迎します。
- 上記キャラクターとしての運用・配信・商用利用・なりすましはお断りします。
- 同梱のサンプルキャラクター（`src/lab_lounge/system_prompts/samples/` の
  `sample_logic.md` / `sample_empath.md`）は Apache-2.0 です。
  自由に改変して、あなた自身のキャラクターを作ってください。

## 補足: 実装上の扱い

- 実キャラクターの人格プロンプト（`src/lab_lounge/system_prompts/system_*.txt`）と
  詳細設定（`src/lab_lounge/character_detail/`）は `.gitignore` で追跡対象から外しており、
  公開リポジトリには含まれません（実体は Think-AI Lab. のローカルにのみ存在します）。
- これらが無い環境（fresh clone 等）では、ローダーが `system_prompts/samples/` の
  サンプルへ自動フォールバックして起動します。`src/lab_lounge/characters.py` の
  `load_system_prompt` を参照してください。
