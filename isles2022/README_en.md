# ISLES-2022 — Reproducible Experiment README (Portfolio)

**Language:** English | [Japanese](README.md)

This folder is the **public entry point** for the ISLES 2022 lesion-segmentation pipeline. It is designed so an external reviewer can understand the project value, representative results, and the fastest way to try it before reading the full experiment notes.

## What a reviewer can verify quickly

- **What it does**: preprocess / train / evaluate a 3D U-Net based ISLES pipeline
- **What it does**: preprocesses data, trains models, and evaluates a 3D U-Net-based ISLES pipeline
- **Who it is for**: hiring managers, ML engineers, and researchers who want reproducible MRI segmentation work
- **Fastest first run**: `python ../scripts/smoke_test.py --use_dummy_data`
- **Representative metrics**:
  - local test mean Dice: **~0.622**
  - fold0 validation mean Dice: **0.7539**
  - lesion-wise precision: **0.516**
  - HD95: **12.38 mm**

## Quick links

- Japanese version: [README.md](README.md)
- Citation: `../CITATION.cff`
- Release-note source: `../docs/releases/v1.0-interview.md`
- Roadmap: `../ROADMAP.md`

## Stable Portfolio Version

The reproducible evaluation reviewed during recruitment corresponds to:

✅ `isles2022-v1.0-interview`

Active development continues on the repository.

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

- The pipeline centers on a 3D U-Net with explicit threshold and post-processing sweeps.
- Small-lesion difficulty is tracked using size-stratified evaluation.
- Existing reports include test runs with mean Dice around 0.62, depending on configuration.

---

## 4. Additional documents

- Minimum recipe (Japanese source)
  - `./docs/isles2022_unet_minimum_recipe_ja.md`
- Full debug/improvement plan (Japanese source)
  - `./docs/isles2022_3dunet_complete_debug_and_fix_plan.md`

This public package is self-contained under `github_public/`.
