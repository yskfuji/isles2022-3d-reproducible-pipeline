# Public repo polish checklist — isles2022-3d-reproducible-pipeline

テンプレ由来: RSNA `docs/audit/PUBLIC_REPO_POLISH_CHECKLIST.md`  
監査日: 2026-11-11  
対象コミット: （push 後に記入）

## 1. 証拠と主張

| 項目 | 状態 | メモ |
|---|---|---|
| 数値の一次成果からの再計算 | 該当なし / 部分 | 指標は README 表と評価メモ依存。RSNA 型 `check_metric_claims.py` は未移植 |
| 制約と数値の同一節 | 完了 | README / README_ja に medical-device caveat と指標表を併記 |
| claims 検査スクリプト | 該当なし | 未導入。ギャップとして記録 |

## 2. 公開衛生

| 項目 | 状態 | メモ |
|---|---|---|
| 機微パス・絶対パス検査 | 部分 | 既存 `.gitignore` で Datasets/runs 除外。専用 hygiene スクリプトは未移植 |
| 相対リンク | 完了 | README リンクを点検 |
| 無視規則 | 完了 | 医療データ・重みを公開物から除外 |

## 3. 図

| 項目 | 状態 | メモ |
|---|---|---|
| PNG + `width="100%"` | 完了 | architecture / repo_map / results_snapshot |
| 日本語 `*_ja.png` | 完了 | README_ja から参照 |
| Mermaid 非依存 | 完了 | 静的 SVG→PNG |
| raw ハッシュ照合 | 完了予定 | push 後に記録 |

## 4. 日本語

| 項目 | 状態 | メモ |
|---|---|---|
| である調（README_ja） | 完了 | 入口を書き換え |
| textlint 設定 | 完了 | `.textlintrc.json` 追加（ruleset 依存は任意） |
| 口語時間売り禁止 | 完了 | 「約30秒」等を排除 |

## 5. 完了定義

| 項目 | 状態 | メモ |
|---|---|---|
| ローカル図再生成 | 完了 | docs/assets/*.png |
| raw 照合 | 未 | push 後 |
| ブラウザ確認 | 未 | ハードリロード後 |

## 監査時ギャップ（整備前）

- README_ja がです・ます調
- SVG 埋め込みのみ（GitHub 溢れリスク）
- 医療機器ではない旨が弱い／欠落
- textlint / 用語集なし
- claims 自動検査なし

## 適用コミット

（push 後追記）
