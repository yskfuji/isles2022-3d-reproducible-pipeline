# ISLES 監査マップ

この公開物は ISLES 監査向けに最小導線で整理しています。

## 1. 読む順番

1. `./isles2022/README.md`（または `README_en.md`）
2. `./isles2022/docs/isles2022_unet_minimum_recipe_ja.md`
3. `./isles2022/docs/isles2022_3dunet_complete_debug_and_fix_plan.md`
4. `./core/ToReBrain-pipeline/src/preprocess/prepare_isles2022.py`
5. `./core/ToReBrain-pipeline/src/training/train_3d_unet.py`
6. `./core/ToReBrain-pipeline/src/evaluation/evaluate_isles.py`

## 2. 主な監査ポイント

- 前処理仕様（spacing/intensity/modalities）の固定
- 学習設定と評価設定（threshold sweep）の明示
- サイズ別指標や後処理評価の再現可能性

## 3. 除外物

- `Datasets/`
- `runs/`
- `results/`
- `logs/`