# isles2022-3d-reproducible-pipeline

**言語:** 日本語 | [英語版](README.md)

ISLES 2022 向けの、**再現可能な 3D 脳梗塞病変セグメンテーションパイプライン**です。監査しやすいドキュメント、しきい値・後処理のスイープ、サイズ別評価を含みます。

**クイックリンク**
- 英語版: [isles2022/README_en.md](isles2022/README_en.md)
- 日本語版: [isles2022/README.md](isles2022/README.md)
- 実験詳細: [isles2022/README.md](isles2022/README.md)
- 再現性チェックリスト: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- GitHub About 設定原稿: [英語版](docs/github_about.md) | [日本語版](docs/github_about_ja.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [英語版](docs/releases/v1.0-interview.md) | [日本語版](docs/releases/v1.0-interview_ja.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## このリポジトリでできること

- ISLES 2022 病変セグメンテーションの前処理 → 学習 → 評価ワークフロー
- 3D U-Net ベースラインと、しきい値・連結成分のスイープ
- 小病変を意識したサイズ別レポート
- 外部レビュー向けに整理したポートフォリオ向けの案内
- 実データなしで公開物が正しく動くかを確かめる簡易動作確認

## 想定している読者

- 医療AIセグメンテーション実装を確認したい採用担当
- 監査しやすい MRI セグメンテーション基盤を見たい ML エンジニア
- 再現性重視の ISLES 系プロジェクト構成を探している研究者

## 3分で分かる概要

![ISLES パイプライン構成図](docs/assets/architecture.svg)

![ISLES リポジトリ構成図](docs/assets/repo_map.svg)

![ISLES 指標サマリー](docs/assets/results_snapshot.svg)

### 代表指標

| 指標 | 値 | 意味 |
|---|---:|---|
| ローカルテストの平均 Dice | ≈ 0.622 @ threshold 0.20 | 公開レシピの実用的な性能目安 |
| Fold0 検証の平均 Dice | 0.7539 | 同分布の検証データにおける性能 |
| 病変単位の適合率 | 0.516 | 偽陽性制御とのトレードオフ |
| HD95 | 12.38 mm | 境界品質の指標 |

> 数値は同梱レシピと評価メモに基づきます。医療データ本体は公開物に含めていません。

## 最短の確認方法

### 1. 実データなしで動作確認

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. 配布物マニフェストを確認

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. 実データで前処理 / 学習 / 評価

- 日本語詳細: [isles2022/README.md](isles2022/README.md)
- 英語版の詳細ガイド: [isles2022/README_en.md](isles2022/README_en.md)

## 含まれるものと含まれないもの

含まれるもの:
- ソースコード
- 設定ファイル
- 監査 / 評価ドキュメント
- 静的図表とリリースノート原稿

含まれないもの:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## 固定スナップショット（ポートフォリオ用）

開発は継続中ですが、ポートフォリオ / 面接レビュー用の固定スナップショットは次のタグです。

✅ `isles2022-v1.0-interview`

## 引用

[CITATION.cff](CITATION.cff) を参照してください。

## コミットメッセージの規約

今後の変更は Conventional Commits（`type: summary`）で揃えます。

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
