# 公開整備記録 — isles2022-3d-reproducible-pipeline

手順の定義: [rsna-ich-reproducible-pipeline 公開整備プレイブック](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/blob/main/docs/audit/PUBLIC_REPO_POLISH_PLAYBOOK.md)  
記録日: 2026-11-11  
参照コミット: `7d4d528`（整備本体は `eb87609`）

## 1. 証拠と主張

| 項目 | 判定 | メモ |
|---|---|---|
| 数値の一次成果からの再計算 | 該当なし / 部分 | 指標は README 表と評価メモに依存する。RSNA 型の自動照合スクリプトは未導入 |
| 制約と数値の同一節 | 合格 | README / README_ja に medical-device caveat と指標表を併記 |
| claims 検査スクリプト | 該当なし | 未導入 |

## 2. 公開衛生

| 項目 | 判定 | メモ |
|---|---|---|
| 機微パス・絶対パス検査 | 部分 | `.gitignore` で Datasets / runs 等を除外。専用 hygiene スクリプトは未導入 |
| 相対リンク | 合格 | README 系リンクを点検 |
| 無視規則 | 合格 | 医療データ・重みを公開物から除外 |

## 3. 図

| 項目 | 判定 | メモ |
|---|---|---|
| PNG + `width="100%"` | 合格 | architecture / repo_map / results_snapshot |
| 日本語 `*_ja.png` | 合格 | README_ja から参照 |
| Mermaid 非依存 | 合格 | 静的 SVG から PNG を生成 |
| raw ハッシュ照合 | 合格 | `architecture_ja.png` の SHA-256 が raw と一致 |

## 4. 日本語

| 項目 | 判定 | メモ |
|---|---|---|
| である調（README_ja） | 合格 | 入口をである調へ統一 |
| textlint 設定 | 合格 | `.textlintrc.json` を追加 |
| 口語時間売り禁止 | 合格 | 時間を売る見出しを置かない |

## 5. 完了定義

| 項目 | 判定 | メモ |
|---|---|---|
| ローカル図再生成 | 合格 | `docs/assets/*.png` |
| raw 照合 | 合格 | push 後に照合 |
| ブラウザ確認 | 未確認 | 閲覧者側でハードリロードを推奨 |

## 公開面での到達点

- 日英 README で PNG を応答的に埋め込む
- 日本語入口をである調とし、医療機器ではない旨を明示する
- 公開整備の判定を本記録に残す
