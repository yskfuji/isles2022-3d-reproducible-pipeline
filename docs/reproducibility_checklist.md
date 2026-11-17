# Reproducibility Checklist

Use this page as a fast external-review checklist for the public ISLES 2022 3D repository.

## 1. Package integrity

- Confirm the stable snapshot tag referenced in the README and release note.
- Confirm the repository excludes protected medical data and run artifacts.
- Optionally generate a fresh manifest with `python scripts/smoke_test.py --use_dummy_data` or `python tools/make_manifest.py` from `core/pipeline`.

## 2. Documentation consistency

- Read the landing page: `README.md` or `README_ja.md`.
- Read the task-facing guide: `isles2022/README_en.md` or `isles2022/README.md`.
- Read the release note source under `docs/releases/v0.4.0-isles.md`.
- Confirm the reported metrics and task description are consistent across those files.

## 3. Code-path sanity

- Verify that preprocessing, training, and evaluation entrypoints exist:
  - `core/pipeline/src/preprocess/prepare_isles2022.py`
  - `core/pipeline/src/training/train_3d_unet.py`
  - `core/pipeline/src/evaluation/evaluate_isles.py`
- Verify that threshold sweep and postprocess logic are documented in the public guides.

## 4. Smoke-test validation

- Run `python scripts/smoke_test.py --use_dummy_data`.
- Confirm the command completes without requiring medical data.
- Confirm the generated summary points at the expected public files and entrypoints.

## 5. Evaluation-readiness checks

- Confirm the README exposes the practical evaluation recipe, not just model architecture.
- Confirm size-aware or lesion-aware metrics are surfaced, not only Dice.
- Confirm the repository states what is included and excluded from the public bundle.

## 6. Registration verification

- Run `python core/pipeline/tools/verify_registration.py --run-dir <REPRESENTATIVE_RUN_DIR> --model-name isles-3d-unet --checkpoint best.pt --promotion-rule "val_dice>=0.75" --registered-model-name isles-3d-unet-verify` from the repository root or `python tools/verify_registration.py ...` from `core/pipeline`.
- Confirm the command creates `artifacts/verification/registered_models/.../registration.json`.
- Confirm the printed JSON summary reports the expected `promotion_status` and alias outcome.

## 7. Reviewer pass criteria

- A reviewer can identify the main task, the key metrics, and the first commands to run in under 3 minutes.
- A reviewer can trace preprocess -> train -> evaluate without needing hidden scripts.
- A reviewer can validate repository wiring without access to protected data.

## 8. Known limits

- This checklist validates public reproducibility scaffolding, not full medical-model reproduction.
- Full metric reproduction still requires separately prepared ISLES 2022 data.