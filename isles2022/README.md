# ISLES-2022 — 再現性つき実験README（ポートフォリオ向け）

このフォルダは、**ISLES 2022 病変セグメンテーション公開版の入口**です。採用・監査・外部レビューの最初の接点として、価値・結果・最短の試し方を先に見せる構成へ寄せています。

## まず分かること

- **何ができるか**: 前処理 / 学習 / 評価 / threshold sweep / postprocess sweep
- **誰向けか**: 採用担当、MRI セグメンテーション実装を見たい ML エンジニア、再現性重視の研究者
- **最短確認**: `python ../scripts/smoke_test.py --use_dummy_data`
- **成果の目安**:
  - local test mean Dice: **約 0.622**
  - fold0 val mean Dice: **0.7539**
  - lesion-wise precision: **0.516**
  - HD95: **12.38 mm**

## すぐ使うリンク

- English version: [README_en.md](README_en.md)
- 詳細コード / 実験: `../core/pipeline/`
- 引用情報: `../CITATION.cff`
- リリースノート原稿: `../docs/releases/v1.0-interview.md`
- ロードマップ: `../ROADMAP.md`

## Stable Portfolio Version（固定スナップショット）

採用選考でレビューされた「再現評価」は、次のタグに対応します：

✅ isles2022-v1.0-interview

リポジトリは継続的に開発中です。

対応する公開リポジトリ名: `isles2022-3d-reproducible-pipeline`

このフォルダは、ISLES-2022 病変セグメンテーションの実験を
**再現可能な形で第三者に説明・実行してもらうための入口**です。

---

## TL;DR

- 主要コードは `../core/pipeline/` にあります。
- 3D U-Net ベースの前処理・学習・評価パイプラインを一式で実行できます。
- 公開物には `Datasets/`・`runs/`・`results/` を同梱していません（データは各自で準備）。
- まずは「前処理 → 学習 → 評価」の最短 3 ステップを通すと全体像を把握できます。

---

## 1. コードマップ

- 前処理（ISLES-2022 生データ→学習用形式）
  - `../core/pipeline/src/preprocess/prepare_isles2022.py`
- 学習（3D U-Net）
  - `../core/pipeline/src/training/train_3d_unet.py`
- 評価（sliding window / threshold sweep / size-stratified 指標）
  - `../core/pipeline/src/evaluation/evaluate_isles.py`
- Dataset 定義
  - `../core/pipeline/src/datasets/isles_dataset.py`

---

## 2. 再現手順（最短）

以下は `github_public/core/pipeline/` をカレントとして実行します。

### 2.1 前処理

```bash
python -m src.preprocess.prepare_isles2022 \
  --raw-root "/path/to/ISLES-2022" \
  --derivatives-root "/path/to/ISLES-2022/derivatives" \
  --out-root data/processed/isles2022_dwi_adc \
  --split-csv data/splits/isles2022_train_val_test.csv \
  --modalities DWI,ADC \
  --target-spacing 1.5,1.5,1.5 \
  --intensity percentile_chwise
```

### 2.2 学習

```bash
python -m src.training.train_3d_unet \
  --config configs/train_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced.yaml
```

### 2.3 評価

```bash
python -m src.evaluation.evaluate_isles \
  --model-path runs/3d_unet/<YOUR_RUN>/best.pt \
  --csv-path data/splits/isles2022_train_val_test.csv \
  --root data/processed/isles2022_dwi_adc \
  --split val \
  --out-dir results/isles_eval_demo \
  --patch-size 48,48,24 \
  --overlap 0.5 \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
  --normalize nonzero_zscore \
  --tta flip
```

---

## 3. 現状の要点（ポートフォリオ記載用）

- 3D U-Net を主軸に、しきい値 sweep と後処理 sweep を分離して検証しています。
- 小病変の難しさ（small lesion で Dice が落ちやすい）を、サイズ別指標で追跡しています。
- 既存レポートでは、test で mean Dice が 0.62 前後の結果が確認されています（設定依存）。

---

## 4. 追加資料

- 最小レシピ（日本語）
  - `./docs/isles2022_unet_minimum_recipe_ja.md`
- デバッグ/改善計画（日本語）
  - `./docs/isles2022_3dunet_complete_debug_and_fix_plan.md`

本公開物は `github_public/` 配下のみで参照が完結するように構成しています。
