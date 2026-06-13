# Characters

本リポジトリのコードは Apache-2.0 ですが、Think-AI Lab. のキャラクター
（坂東ルカ／ミミ・オクタヴィア／波心ちさめ／八重笠さくら／オクタメイド／アルカ、
その他関連キャラクターの名称・人格設定・詳細設定・プロンプト・画像・音声設定）は
ライセンスの対象外であり、すべての権利を Think-AI Lab. が留保します。

- これらの実プロンプト・詳細設定は、`src/lab_lounge/system_prompts/` および
  `src/lab_lounge/character_detail/` に**参考情報として公開**しています。閲覧・学習・
  出典明記のうえでの短い引用は歓迎しますが、再利用のライセンスは付与しません。
- 上記キャラクターとしての運用・配信・商用利用・なりすまし・再配布はお断りします。
- `src/lab_lounge/system_prompts/samples/` のサンプルキャラクターは Apache-2.0 です。
  自由に改変して、あなた自身のキャラクターを作ってください。

## 補足: 実装上の扱い

- 実キャラクターの人格プロンプト（`src/lab_lounge/system_prompts/system_*.txt`）と
  詳細設定（`src/lab_lounge/character_detail/`）は、参考情報としてリポジトリに同梱しています。
  ただしライセンス対象外（全権利留保）であり、再利用は許諾しません。
- これらを削除した環境（fork してキャラ資産を除いた場合など）では、ローダーが
  `src/lab_lounge/system_prompts/samples/` のサンプルへ自動フォールバックして起動します。
  `src/lab_lounge/characters.py` の `load_system_prompt` を参照してください。
- `character_detail/` はコードからは未使用の予約レイヤです（設定資料として同梱）。
