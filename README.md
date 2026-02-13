# ISLES-2022 Reproducible Audit Package

このディレクトリは **ISLES-2022 lesion segmentation** の公開・監査用パッケージです。

## 入口

- 監査マップ: `./AUDIT_MAP.md`
- ISLES README（JP/EN）:
  - `./isles2022/README.md`
  - `./isles2022/README_en.md`

## 同梱方針

- 同梱: コード、設定、監査用ドキュメント、ISLES補助資料
- 非同梱: `Datasets/`, `runs/`, `results/`, `logs/`

データは監査者側で準備し、READMEの再現コマンドに従って検証します。
