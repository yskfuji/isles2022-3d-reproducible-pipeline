# isles2022-3d-reproducible-pipeline

このディレクトリは **ISLES-2022 lesion segmentation** の公開・監査用パッケージです。
リポジトリ名: `isles2022-3d-reproducible-pipeline`

## Stable Portfolio Version（固定スナップショット）

リポジトリは継続的に開発中であるため、ポートフォリオ用のレビューを次のタグに対応させています：

✅ isles2022-v1.0-interview


## (はじめに)コミットメッセージ規約

レビューしやすさを担保するため、今後のコミットは Conventional Commits 形式（`type: summary`）に揃えます：

- fix: leakage check in group split
- feat: add calibration evaluation
- refactor: manifest validation logic
- docs: evaluation protocol clarification

## イントロダクション

- 監査マップ: `./AUDIT_MAP.md`
- ISLES README（JP/EN）:
  - `./isles2022/README.md`
  - `./isles2022/README_en.md`

## コンポーネントダイレクト

- 同梱: コード、設定、監査用ドキュメント、ISLES補助資料
- 非同梱: `Datasets/`, `runs/`, `results/`, `logs/`

データは監査者側で準備し、READMEの再現コマンドに従って検証します。
