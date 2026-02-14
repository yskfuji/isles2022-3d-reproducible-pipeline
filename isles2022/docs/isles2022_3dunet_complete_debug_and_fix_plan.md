# ISLES2022 / 3D U-Net Dice≈0.40停滞の完全解析＆改善実装指示書
（Copilot / Claude Code / VS Code Agent 用・最終版）

作成日: 2025-12-19

---

## 0. この文書の位置づけ（重要）
本書は **「考察」ではなく「実装を前提とした研究実験設計書」** である。  
目的は以下の3点を **定量的に確定** すること。

1. なぜ Dice≈0.40 で頭打ちになっているか（主因の断定）
2. **3D U-Net縛りで Dice 0.60 が理論的・実務的に可能か**
3. 可能なら「最短でそこに近づく実装ルート」、不可能なら「どこが限界か」

※ **アーキテクチャ変更（Attention, Transformer, 深層化など）は禁止**。  
※ 本書の指示どおりに PR を切れば、そのまま論文・技術レポート級の検証になる。

---

## 1. 確定している前提条件（変更禁止）
### モデル・推論
- Model: **3D U-Net（レトロ構造）**
- Input: DWI / ADC / FLAIR（3ch）
- patch_size: [64,64,48]
- overlap: 0.5
- TTA: full
- temperature: 1.0
- threshold sweep: 0.05–0.95 (step=0.05)
- min_size: 0（baseline）

### データ特性（ISLES2022）
- test: 25 cases（全例 GT>0）
- GT voxel distribution:
  - min=25 / median=248 / max=18001
  - **小病変が多数**
- voxel spacing:
  - train/test z≈2.0mm
  - val: z≈2.0mm & 4.8mm 混在（IQR=2.8）
- axis: LAS（全split統一）
- intensity saturation:
  - sat0≈0.52–0.55
  - sat1≈0.011（上下clipあり）

### 観測結果（test sweep）
- best Dice = **0.385 @ thr=0.80**
- P≈0.684 / R≈0.430
- thr≤0.15: recall≈1.0, precision≈0.01（FPの海）

---

## 2. 数理的に見た「何が起きているか」
### 2.1 Dice–Precision–Recall 曲線の意味
この形状は以下を強く示唆する。

- モデルは **病変候補を広く拾う能力（recall）は持っている**
- しかし
  - 病変と非病変の境界が粗い
  - FPを抑える学習圧が弱い
- 結果として
  - 低thr → FP爆発
  - 高thr → 小病変から落ちる → recall低下

👉 **「小病変を“識別できていない”のではなく、“十分に学習させていない”可能性が高い**

---

## 3. 主要仮説と検証戦略（断定するための設計）

### H1: 小病変サイズ分布 × Dice loss の構造的不利
- Dice は体積が小さいほど 1 voxel の誤差で大きく下がる
- 現状 median=248 vox → patch内では **背景が圧倒的**
- 検証:
  - 病変サイズ別 Dice（small/medium/large）を必ず算出
  - small だけが著しく低ければ **支持**

### H2: z-spacing 混在がモデル選択を歪めている
- val の 4.8mm slice は 3D文脈を破壊
- best epoch が「厚スライス向け」になっている可能性
- 検証:
  - val を z≈2.0 / z≈4.8 に分けて Dice を算出
  - 最良epochがズレるなら **部分支持**

### H3: 確率未校正（thr=0.8依存）
- thr=0.8 で最良という事実は calibration 不良の兆候
- ただし校正は Dice を劇的には上げない
- 検証:
  - temperature / isotonic で thr が安定するか
  - Dice +0.01–0.03 なら **部分支持**

### H4: 後処理なし設計が FP を支配
- FP component 数・サイズ分布を見ると即断できる
- 検証:
  - CC + min_size sweep で Dice が +0.05 以上改善 → **支持**

### H5: fg033 サンプリング不足
- patch内 GT voxel 数が小さすぎると勾配が背景に吸われる
- 検証:
  - fg0.8 + GT中心crop で small Dice が上がるか → **支持/否定**

---

## 4. 実装PR設計（厳密・再現可能）

### PR-0: 評価分解インフラ
- サイズ別 Dice / Precision / Recall
- FP component 統計（count, size p90）
- thr=0.8 / thr=0.3 を必ず含める

### PR-1: 小病変優先サンプリング
- foreground prob p_fg = 0.8
- lesion CC 列挙
- 重み w = 1 / (size^α), α∈[0.5,1.0]
- CC→voxel→patch center
- background は brain mask 内

### PR-2: FP抑制 loss
- Dice + BCE(pos_weight)
- Dice + Focal
- Tversky (α/β sweep)

### PR-3: 後処理（CC + min_size）
- min_size sweep: 0, 20, 25, 30, 50, 100, 200
- top_k: None, 1, 3, 5

### PR-4: z-spacing 対応
- val 4.8mm → 2.0mm resample
- or val split別評価

---

## 5. 到達可能 Dice の理論推定（3D U-Net縛り）
- 正しいサンプリング + 後処理あり:
  - **Dice ≈ 0.55–0.62**
- 0.60超え条件:
  1. small lesion recall ≥ 0.6
  2. FP CC p90 size ≤ GT median
  3. spacing 整合

---

## 6. 結論
- 主因: **小病変サンプリング不足 + 後処理なし**
- 3D U-Net縛りで 0.60 到達は **条件付きでYes**
- 最短ルート: PR-1 → PR-3 → PR-2 → PR-4

---

## 7. le_3mm（薄スライス）小病変FNの改善プラン（2025-12-23 / 追加）

### 7.1 いま分かっている「根本原因」
- testの `le_3mm` は全例 max_zoom≈2.0mm で、評価側の `--resample-max-zoom-mm 2.0` は入力を変えない → **前処理(resample)でle_3mmは上がらない**
- 取り逃し症例の一部は **GT内確率が全体的に低い**（TTA平均でさらにピークが潰れる）→ **閾値/後処理では救えないFNが残る**
- 現行学習（pr2prod）の設定では、学習パッチのFG露出が弱い:
  - `foreground_prob=0.33` かつ `force_fg_prob=0.0` → **正例症例でもBG中心パッチが多い**
  - FG中心を引けても、FGボクセル一様サンプルだと **大病変に偏りやすく小病変が薄い**

→ le_3mmを上げる最短は **「小病変が見える頻度」と「FNを重く見る損失」** を上げること。

### 7.2 低〜中コストの優先順位（効く順）
1) **サンプリングを正す（最優先）**
- 目的: small lesion の「モデルがGTに確率を出さない」状態を崩す
- 具体:
  - `data.target_pos_patch_frac=0.8`（正例症例の8割をFG中心）
  - `data.fg_component_sampling=inverse_size`（小さいCCを優先して中心サンプル）
  - `data.fg_component_sampling_alpha=1.0`（まずは1.0、次に2.0）

2) **深い層にも勾配を入れる（深層監督）**
- `train.deep_supervision=true`（aux重みは既定 0.5/0.25）

3) **FN寄りの損失へ（Tversky/Focal-Tversky）**
- Dice/BCEだけで小病変が立たない場合に投入
- 例: `tversky_focal (alpha=0.3, beta=0.7, gamma=1.33)`

4) **解像度（パッチ）を上げる（高コスト）**
- 目的: downsampleで小病変が潰れる問題を緩和
- 例: `patch_size: [64,64,32]` または `[64,64,48]`（メモリが許せば）
- ただし、まず(1)-(3)で「GT内確率が上がる」ことを確認してから。

### 7.3 すぐ回せる実験（YAMLは生成済み）
以下を追加済み:
- `pipeline/configs/generated/_thin_fix_20251223/`

推奨順:
1. サンプリング修正のみ（損失は現状維持）
2. + 深層監督
3. + Focal-Tversky

### 7.4 実行コマンド（学習→test評価）

学習（例: サンプリング修正のみ）:
```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
PYTHONPATH=$PWD /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
  python -m src.training.train_3d_unet \
  --config configs/generated/_thin_fix_20251223/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced_fg033_ps482424_ch48_pr2prod_dice_bce_pw4p0_bw0p5_thinfix_pos80_fgccinv_a1.yaml
```

test評価（推論は `tta=none` と `tta=flip` を両方。resample=2.0, slice-spacingはraw固定）:
```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
MODEL=runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced_fg033_ps482424_ch48_pr2prod_dice_bce_pw4p0_bw0p5_thinfix_pos80_fgccinv_a1/best.pt

for TTA in none flip; do
  PYTHONPATH=$PWD /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
    python -m src.evaluation.evaluate_isles \
    --model-path $MODEL \
    --csv-path data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \
    --root data/processed/my_dataset_dwi_adc_flair \
    --split test \
    --out-dir results/diag/test_eval_${TTA}_thinfix_pos80_fgccinv_a1 \
    --patch-size 48,48,24 \
    --overlap 0.5 \
    --thresholds 0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50 \
    --min-size 0 \
    --top-k 0 \
    --cc-score none \
    --cc-score-thr 0.5 \
    --normalize nonzero_zscore \
    --tta $TTA \
    --resample-max-zoom-mm 2.0 \
    --slice-spacing-source raw \
    --quiet
done
```

### 7.5 合否判定（次の手を決める指標）
- 最優先KPI: `le_3mm mean_dice` と `le_3mm detection_rate_case`
- 期待レンジ（目安）:
  - サンプリング修正だけで `det_rate` が 15/19 → 16-18/19 に上がるなら「学習で救えるFN」が多い
  - `le_3mm mean_dice` が 0.36 → 0.45 以上に動くなら、(2)(3)の積み増しで0.5台が現実的
- もし `det_rate` が動かない（=完全FNが残る）場合:
  - (A) 解像度（patch_size↑）
  - (B) データ側（ラベル/前処理のミスアライン）
  - (C) モデル側（downsample段数の見直し＝ただしアーキ変更禁止なら“浅いU-Net”のみ）
  の順で疑う。
