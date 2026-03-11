# ISLES-2022 — 再現性のある実験 README（ポートフォリオ向け）

**言語:** 日本語 | [英語版](README_en.md)

このフォルダは、**ISLES 2022 病変セグメンテーション公開版の概要ページ**です。採用・監査・外部レビューの最初の接点として、価値・結果・最短の試し方を先に示す構成にしています。

## まず分かること

- **何ができるか**: 前処理 / 学習 / 評価 / しきい値スイープ / 後処理スイープ
- **誰向けか**: 採用担当、MRI セグメンテーション実装を見たい ML エンジニア、再現性重視の研究者
- **最短確認**: `python ../scripts/smoke_test.py --use_dummy_data`
- **成果の目安**:
  - ローカルテストの平均 Dice: **約 0.622**
  - fold0 検証の平均 Dice: **0.7539**
  - 病変単位の適合率: **0.516**
  - HD95: **12.38 mm**

## すぐ使うリンク

- 英語版: [README_en.md](README_en.md)
- 詳細コード / 実験: `../core/pipeline/`
- 引用情報: `../CITATION.cff`
- リリースノート原稿: `../docs/releases/v0.3.0-isles.md`
- ロードマップ: `../ROADMAP.md`

## 固定スナップショット（ポートフォリオ用）

現行のポートフォリオ用スナップショットは、次のタグに対応します：

✅ `v0.3.0-isles`

リポジトリは継続的に開発中です。

対応する公開リポジトリ名: `isles2022-3d-reproducible-pipeline`

このフォルダは、ISLES-2022 病変セグメンテーションの実験を
**第三者が再現可能な形で理解し、実行できるようにするためのガイド**です。

---

## TL;DR

- 主要コードは `../core/pipeline/` にあります。
- 3D U-Net ベースの前処理・学習・評価パイプラインを一式で実行できます。
- 公開物には `Datasets/`・`runs/`・`results/` を同梱していません（データは各自で用意してください）。
- まずは「前処理 → 学習 → 評価」の最短 3 ステップを通すと全体像を把握できます。

## ベンチマーク要約

### 全体の目安

| 対象 split | 評価条件 | Mean Dice | 病変単位 precision | 病変単位 recall | HD95 |
|---|---|---:|---:|---:|---:|
| fold0 検証 | TTA=flip, threshold 0.14 | 0.7539 | 0.7522 | n/a | n/a |
| ローカルテスト | 5-fold mean-prob ensemble, threshold 0.20 | 0.622 | 0.5161 | 0.5927 | 12.38 mm |

### 公開テストレシピのサイズ別 Dice

サイズ区分は evaluator の既定値に合わせています: small `<250` vox、medium `250-999` vox、large `>=1000` vox。

| GT 病変サイズ | 症例数 | Mean Dice @ threshold 0.20 | Median Dice | 検出率 |
|---|---:|---:|---:|---:|
| Small | 5 | 0.4818 | 0.5870 | 0.80 |
| Medium | 7 | 0.5107 | 0.5310 | 1.00 |
| Large | 13 | 0.7368 | 0.8157 | 1.00 |

---

## 1. コードマップ

- 前処理（ISLES-2022 生データ → 学習用形式）
  - `../core/pipeline/src/preprocess/prepare_isles2022.py`
- 学習（3D U-Net）
  - `../core/pipeline/src/training/train_3d_unet.py`
- 評価（sliding window / しきい値スイープ / サイズ別指標）
  - `../core/pipeline/src/evaluation/evaluate_isles.py`
- データセット定義
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

## 3. 現時点の要点（ポートフォリオ向け）

- 3D U-Net を主軸に、しきい値スイープと後処理スイープを分けて検証しています。
- 小病変では Dice が下がりやすいため、その傾向をサイズ別指標で追跡しています。
- 既存レポートでは、テストで平均 Dice が 0.62 前後となる結果を確認しています（設定依存）。

---

## 4. 追加資料

- 最小レシピ（日本語）
  - `./docs/isles2022_unet_minimum_recipe_ja.md`
- デバッグ/改善計画（日本語）
  - `./docs/isles2022_3dunet_complete_debug_and_fix_plan.md`

本公開物は `github_public/` 配下のみで参照が完結するように構成しています。
