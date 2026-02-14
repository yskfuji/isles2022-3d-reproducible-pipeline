# ISLES-2022 — Reproducible Experiment README (Portfolio)

日本語版: [README.md](README.md)

This folder is the public entry point for ISLES-2022 lesion-segmentation work,
organized so a third party can understand and rerun the pipeline with their own data.

---

## TL;DR

- Main implementation lives in `../core/pipeline/`.
- The repository includes end-to-end scripts for preprocessing, training, and evaluation.
- `Datasets/`, `runs/`, and `results/` are intentionally excluded from this public export.
- The fastest way to understand the project is: preprocess → train → evaluate.

---

## 1. Code map

- Preprocessing (raw ISLES-2022 to processed layout)
  - `../core/pipeline/src/preprocess/prepare_isles2022.py`
- Training (3D U-Net)
  - `../core/pipeline/src/training/train_3d_unet.py`
- Evaluation (sliding window / threshold sweep / size-stratified metrics)
  - `../core/pipeline/src/evaluation/evaluate_isles.py`
- Dataset definitions
  - `../core/pipeline/src/datasets/isles_dataset.py`

---

## 2. Quick reproducible run

Run the commands below from `github_public/core/pipeline/`.

### 2.1 Preprocess

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

### 2.2 Train

```bash
python -m src.training.train_3d_unet \
  --config configs/train_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced.yaml
```

### 2.3 Evaluate

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

## 3. Current highlights (portfolio notes)

- The pipeline focuses on 3D U-Net with explicit threshold/postprocess sweeps.
- Small-lesion difficulty is tracked using size-stratified evaluation.
- Existing reports include runs around mean Dice ~0.62 on test (configuration-dependent).

---

## 4. Additional documents

- Minimum recipe (Japanese source)
  - `./docs/isles2022_unet_minimum_recipe_ja.md`
- Full debug/improvement plan (Japanese source)
  - `./docs/isles2022_3dunet_complete_debug_and_fix_plan.md`

This public package is self-contained under `github_public/`.
