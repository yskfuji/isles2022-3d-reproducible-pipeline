# 再現性チェックリスト

このページは、公開版 ISLES 2022 3D リポジトリを外部レビューする際の簡易チェックリストです。

## 1. パッケージの完全性

- README とリリースノートで参照している安定スナップショットのタグを確認する。
- リポジトリに機微な医療データや private な run artifact が含まれていないことを確認する。
- 必要に応じて、`core/pipeline` で `python scripts/smoke_test.py --use_dummy_data` または `python tools/make_manifest.py` を実行し、新しいマニフェストを生成する。

## 2. ドキュメントの整合性

- 入口となるページとして `README.md` または `README_ja.md` を読む。
- タスク説明として `isles2022/README_en.md` または `isles2022/README.md` を読む。
- リリースノート原稿として `docs/releases/v0.4.0-isles_ja.md` を確認する。
- 主要指標とタスク説明が、これらのファイル間で矛盾していないことを確認する。

## 3. コード経路の妥当性

- 前処理、学習、評価のエントリポイントが存在することを確認する。
  - `core/pipeline/src/preprocess/prepare_isles2022.py`
  - `core/pipeline/src/training/train_3d_unet.py`
  - `core/pipeline/src/evaluation/evaluate_isles.py`
- しきい値スイープと後処理ロジックが公開ドキュメントで説明されていることを確認する。

## 4. スモークテスト

- `python scripts/smoke_test.py --use_dummy_data` を実行する。
- 実データなしでコマンドが完了することを確認する。
- 出力された summary が、想定どおりの公開ファイルとエントリポイントを指していることを確認する。

## 5. 評価レシピとしての見やすさ

- README がモデル構造の説明だけでなく、実際の評価レシピまで示していることを確認する。
- Dice だけでなく、病変サイズ別や病変単位の見方も分かることを確認する。
- 公開物に何が含まれ、何が含まれないかが明示されていることを確認する。

## 6. モデル登録の検証

- リポジトリ直下では `python core/pipeline/tools/verify_registration.py --run-dir <REPRESENTATIVE_RUN_DIR> --model-name isles-3d-unet --checkpoint best.pt --promotion-rule "val_dice>=0.75" --registered-model-name isles-3d-unet-verify`、`core/pipeline` では `python tools/verify_registration.py ...` を実行する。
- `artifacts/verification/registered_models/.../registration.json` が生成されることを確認する。
- 標準出力の JSON 要約に、想定どおりの `promotion_status` と alias 結果が含まれることを確認する。

## 7. レビュワー目線の合格条件

- 主要タスク、重要指標、最初に動かすコマンドを 3 分以内に把握できる。
- preprocess → train → evaluate の流れを、非公開スクリプトに頼らず追える。
- 保護データがなくても、リポジトリの配線と再現性の考え方を確認できる。

## 8. 既知の制約

- このチェックリストが検証するのは、公開物としての再現性導線であり、医療モデルの完全再現そのものではない。
- 完全な指標再現には、別途準備した ISLES 2022 データが必要になる。