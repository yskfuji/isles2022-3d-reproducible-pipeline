# ISLES2022 × 3D U-Net：Dice>=0.5 を安定させる「最低限の作法」完全定義（ローカル評価 / M5 MacBook Pro想定）

このMarkdownは、**「何の変哲もない3D U-Net」でも ISLES2022 のローカル評価で Dice>0.5 を安定させる**ための、**最低限の作法（=再現可能なレシピ）**をブレない形でまとめたものです。  
前提：ローカル評価（train/val 分割）、M5 MacBook Pro 32GB、TTA/アンサンブルは使用可。

---

## 0. このレシピの目的と範囲

- 目的：**val Dice を 0.5 以上で安定**させる（まず “0.5の壁” を越える）
- モデル：**素の 3D U-Net**
- 追加OK：**TTA / アンサンブル**
- 入力：**DWI + ADC（2ch）**を基本（最初はここを固定）
- 重要：レシピの比較が壊れないように **評価定義を固定**する

---

## 1) 入力モダリティ（DWI+ADCに固定）

### 定義
- 入力チャネル：**DWI と ADC の2種類（2ch入力）**
- FLAIR：**最低限レシピでは使わない**（高解像・位置ズレ・前処理負担が増えやすい）

### 直感的理由
- DWI/ADC は梗塞のコントラストが強く、まずここで十分学習が回る
- FLAIRは入れると前処理が増え、「最低限で安定」を壊しやすい

---

## 2) 空間スケール統一（Resampling）

### 定義
- 全画像（DWI/ADC）とラベルを **同一ボクセルサイズ**へ統一する
- 目標：**1×1×1 mm 等方**（重ければ **2×2×2 mm 等方**でも可）
- 補間：
  - 画像：三次スプライン等（滑らか系）
  - ラベル：**nearest（最近傍）**

### 直感的理由
- 症例ごとに画素サイズ・スライス厚が違うと、畳み込みが見ている “実空間” がズレて学習が不安定になる

---

## 3) 強度正規化（Normalization）

### 定義（各ケース・各モダリティごと）
1. **非ゼロ領域（脳領域）**を対象に強度外れ値をクリップ  
   - 例：**0.5〜99.5パーセンタイル**でクリップ（過剰な外れ値を抑える）
2. その後、同じ非ゼロ領域で **Z-score 正規化**  
   - mean=0, std=1 にする

### 直感的理由
- 施設・装置差で強度分布がズレると、モデルが “強度差そのもの” を覚えにいって汎化が落ちる

---

## 4) パッチ学習（Patch-based Training）

### 定義
- 学習は **3Dパッチ**で行う（フルボリューム入力は原則しない）
- パッチサイズ：
  - 第一候補：**128×128×128**
  - 重い場合：**96×96×96**
- バッチサイズ：**1**（ローカル/MPS想定の現実解）

### 直感的理由
- 3Dフルボリュームは重すぎてバッチが取れず学習が崩れやすい
- パッチにより、計算量を制御しつつ局所の病変を学びやすくする

---

## 5) データ拡張（Augmentation：やりすぎないが必ず入れる）

### 定義（学習時オンザフライ）
以下を独立に確率適用（各 0.2〜0.5 程度の確率で適用すると扱いやすい）：

**空間系**
- ランダムフリップ（左右・前後など）
- 小回転（±10〜15°）
- 小スケール（0.9〜1.1）

**強度系**
- 乗算スケール（0.9〜1.1）
- 加算シフト（正規化後に ±0.1 程度）
- ガンマ変換（軽め）

**ノイズ / ぼかし**
- ガウスノイズ（軽め）
- ガウスぼかし（軽め）

### 禁止事項
- “病変っぽい偽構造” を作りやすい極端な変換（過度の歪み・強すぎる強度変換）

### 直感的理由
- データ数が有限で、拡張無しだと過学習しやすい
- ただし強すぎる変換はセグメンテーションの意味を壊す

---

## 6) モデル（素の3D U-Netをこれで固定）

### 定義（最小で十分な設定）
- 深さ：**4段 downsample（=5解像度）**
- 初期チャネル：**32**（重ければ16）
- チャネル：32→64→128→256→512
- Conv：3×3×3 を各段2回
- 正規化：**InstanceNorm（推奨） / GroupNorm**
  - batch=1 のため **BatchNormは避ける**
- 活性化：ReLU（or LeakyReLU）
- 出力：1ch（sigmoid）

### 直感的理由
- “変な工夫” より、まずは学習を安定させるレシピが重要
- batch=1 では BatchNorm が崩れやすい

---

## 7) 損失関数（0.5の壁を越える核心）

### 定義
- **Loss = DiceLoss + BCE（またはCE）**（重み 1:1 を固定）
- DiceLoss は smooth（epsilon）を入れて数値不安定を避ける

### 直感的理由
- 背景が圧倒的なので、CE単独だと背景優先になりやすい
- Diceは小病変でも “重みが消えにくい”
- CE/BCEは学習初期の勾配を安定させる

---

## 8) Foreground（病変）優先サンプリング（ほぼ必須）

### 定義
- 学習パッチ抽出の確率を固定：
  - **50%：病変を含むパッチ（foreground patch）**
  - **50%：ランダムパッチ（background混在）**
- foreground patch の作り方：
  - GTラベルの病変voxel座標からランダムに1点選ぶ
  - その点を中心（＋ランダムオフセット）にパッチを切り出す

### 直感的理由
- ランダムに切るだけだと病変入りパッチが出にくく、モデルが背景だけ学習して終わる
- 小病変タスクでは「病変を必ず見せる」ことが重要

---

## 9) 学習ハイパラ（M5/32GB向けの現実解）

### 定義（まずこれを固定）
- Optimizer：**AdamW**
- 初期LR：**2e-4**
- weight_decay：**1e-5**
- LRスケジュール：**Cosine Annealing（終端0）**
- epoch：**200〜300**
  - val が伸びないなら早期終了（例：20 epoch 改善なし）
- AMP（混合精度）：可能ならON

---

## 10) バリデーション（評価定義を固定しないと比較が壊れる）

### 定義
- 推論：**3Dスライディングウィンドウ**でフルボリューム再構成
- 出力：確率マップ → 二値化
  - 学習中の簡易valは **0.5 固定**（学習の進捗監視用）
  - **比較・報告は run ごとに `evaluate_isles` の `best_all` 閾値を採用**（threshold sweep で決める）
- Dice計算：
  - **case-wise Dice（症例ごとDice → 平均）**
- GTが空の症例（空病変）がある場合：
  - ルールを固定する（推奨：**平均から除外**）
  - ※ここがブレると平均Diceが簡単に動く

---

## 11) 推論強化（TTA と アンサンブルで安定性を上げる）

### TTA（軽量で効く）
- flip（左右など）を適用して推論
- 予測確率を元に戻して平均
- 最後に閾値0.5で二値化

### アンサンブル（ローカルでも現実的）
- seedを変えた同一設定モデルを **3本**作る
- 出力確率を平均して二値化

---

# すぐ見返せる「完全固定」まとめ（この通り実装すれば最低限レシピ成立）

- 入力：DWI+ADC（2ch）
- resample：1mm等方（無理なら2mm）
- normalize：非ゼロ領域で clip(0.5–99.5%) → zscore
- patch：128³（無理なら96³）
- batch：1
- aug：flip + 軽回転/軽スケール + 強度ゆらぎ + 軽ノイズ/軽ぼかし
- model：3D U-Net（4down、32start、InstanceNorm）
- loss：Dice + BCE（1:1）
- sampling：foreground 50% / random 50%
- optim：AdamW（lr2e-4, wd1e-5, cosine）
- epochs：200–300（早期終了あり）

---

# 付録：このリポジトリでの「現状運用」実測と次アクション（ts222 / 5fold / test）

## 現状の結論（2026-01-01時点）

- 5fold平均確率アンサンブル（`ts222=[2,2,2]`）の **test best mean_dice は約 0.622（best_threshold=0.20）**。
- 病変サイズ別では **large は平均≈0.737** だが、**small/medium が平均≈0.48〜0.51** で頭打ち。
- `min_size` / `top_k` の単純後処理は悪化しやすかった。
- `cc_score`（連結成分の確率支持でフィルタ）は **FP連結成分は減るが mean_dice の改善はごく小さい**（≈0.6228 程度）。

## ポートフォリオ用（推奨）：従来課題（FP過多）に効いた設定（fp2 / fold0-val）

従来課題（**FP過多で病変を立てすぎる**）の解決に直結して効いたのは **fp2**。

- 設定：`medseg_3d_unet_fp2_conservative_bce2_pw05_dicebce_autothr_basech48_e120_kfold5_f0_ts222`
- 評価条件：fold0 / split=val / n=46 / TTA=flip / threshold sweep（mean Dice最大）
- 結果（best threshold = **0.14**）
  - best mean Dice：**0.753930**
  - lesion-wise：precision_micro **0.752174** / recall_micro **0.628000** / f1_micro **0.684501**
  - mean_fp_cc：**1.913043**（FP連結成分の平均）

出典（このファイルが「一次情報」）：

- `pipeline/results/diag/fp_precision_eval_20260111/medseg_3d_unet_fp2_conservative_bce2_pw05_dicebce_autothr_basech48_e120_kfold5_f0_ts222/eval_val_tta_flip_cpu/summary.json`

ポートフォリオ記載の最小テンプレ（1行で意図が伝わる）：

> ISLES2022（local fold0-val n=46）：mean Dice=0.7539（TTA=flip, thr=0.14）/ lesion-wise precision=0.7522（FP抑制）

再現コマンド（同じ評価を作り直す場合）：

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
TORCH_DEVICE=cpu PYTHONPATH=$PWD /opt/anaconda3/envs/medseg_unet/bin/python -m src.evaluation.evaluate_isles \
  --model-path runs/3d_unet/medseg_3d_unet_fp2_conservative_bce2_pw05_dicebce_autothr_basech48_e120_kfold5_f0_ts222/best.pt \
  --csv-path data/splits/kfold5_my_dataset/fold0.csv \
  --root data/processed/my_dataset_ts222 \
  --split val \
  --out-dir results/diag/fp_precision_eval_20260111/medseg_3d_unet_fp2_conservative_bce2_pw05_dicebce_autothr_basech48_e120_kfold5_f0_ts222/eval_val_tta_flip_cpu_rerun \
  --patch-size 96,96,96 \
  --overlap 0.5 \
  --normalize nonzero_zscore \
  --temperature 1.0 \
  --tta flip \
  --thresholds 0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50 \
  --min-size 0 \
  --cc-score none \
  --cc-score-thr 0.5 \
  --top-k 0 \
  --extra-metrics \
  --quiet
```

## ポートフォリオ用：公式4指標 + 境界距離 + Precision/Recall を1発で出す

ポートフォリオ記載の最小テンプレ（test / 5fold ensemble）：

> ISLES2022（local test, 5fold mean-prob ensemble）：mean Dice≈0.622（thr=0.20）/ lesion-wise precision=0.516（FP抑制の難しさも併記）

`evaluate_isles` に `--extra-metrics` を追加し、**mean Dice が最大になる閾値（best_threshold_by_mean_dice）**で以下を `summary.json` に保存します。

- 公式4指標相当（ローカル定義）
  - Dice（per-threshold から best を採用）
  - 体積差（mL）：`extra_metrics_best.volume_diff_ml`（mean / mean_abs）
  - 病変数差：`extra_metrics_best.lesion_count_diff`（mean / mean_abs）
  - lesion-wise F1：`extra_metrics_best.lesionwise.f1_micro`
- 境界距離（mm）：`extra_metrics_best.boundary_distance_mm`（ASSD / HD / HD95 の平均）
- Precision/Recall（voxel-level）：`per_threshold[].voxel_precision` / `per_threshold[].voxel_recall`

実行例（既存アンサンブル確率 `probs` を使って再評価）

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
/opt/anaconda3/envs/medseg_unet/bin/python -m src.evaluation.evaluate_isles \
  --model-path None \
  --csv-path /Users/yusukefujinami/ToReBrain/pipeline/data/splits/my_dataset_train_val_test.csv \
  --root /Users/yusukefujinami/ToReBrain/pipeline/data/processed/my_dataset_ts222 \
  --split test \
  --probs-dir results/diag/kfold_ensemble_20251230_103044/ensemble/probs \
  --out-dir results/diag/kfold_ensemble_20251230_103044/eval_ensemble_test_extra \
  --patch-size 96,96,96 \
  --overlap 0.5 \
  --normalize nonzero_zscore \
  --temperature 1.0 \
  --thresholds 0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.8 \
  --extra-metrics \
  --quiet
```

出力先：

- `results/diag/kfold_ensemble_20251230_103044/eval_ensemble_test_extra/summary.json`
  - `best_threshold_by_mean_dice: 0.2`
  - `extra_metrics_best`:
    - 体積差（mL）：mean=-11.8979 / mean_abs=17.6106
    - 病変数差：mean=2.64 / mean_abs=9.92
    - lesion-wise：precision=0.5161 / recall=0.5927 / F1=0.5518
    - 境界距離（mm）：ASSD=4.784 / HD=51.430 / HD95=12.380

## まずやる解析（症例別の落ち方を固定化）

`evaluate_isles` が吐く `metrics.json` から、best閾値（summaryのbest）で症例別Dice/Precision/Recallとエラータイプ（FP優勢/FN優勢/見逃し）をCSV化する。

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
/opt/anaconda3/envs/medseg_unet/bin/python3.11 tools/analyze_eval_run.py \
  --eval-dir results/diag/kfold_ensemble_20251230_103044/eval_ensemble_test \
  --top-k 20
```

出力：
- `.../eval_ensemble_test/analysis_report.json`
- `.../eval_ensemble_test/worst_cases.csv`
- `.../eval_ensemble_test/cases_sorted.csv`

## 後処理の追加スイープ（cc_score）

`min_size/top_k` で改善しない場合、FP連結成分だけ落とす `cc_score` を軽くスイープする。

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
export PYTHONPATH=$PWD
PY=/opt/anaconda3/envs/medseg_unet/bin/python3.11
PROBS=$PWD/results/diag/kfold_ensemble_20251230_103044/ensemble/probs
OUTBASE=$PWD/results/diag/kfold_ensemble_20251230_103044/ccscore_sweep
mkdir -p "$OUTBASE"

for mode in max_prob p95_prob mean_prob; do
  for thr in 0.3 0.4 0.5 0.6 0.7; do
   out="$OUTBASE/${mode}_ge${thr}"
   "$PY" -m src.evaluation.evaluate_isles \
    --probs-dir "$PROBS" \
    --csv-path "$PWD/data/splits/my_dataset_train_val_test.csv" \
    --root "$PWD/data/processed/my_dataset_ts222" \
    --split test \
    --out-dir "$out" \
    --patch-size 96,96,96 \
    --overlap 0.5 \
    --thresholds 0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.80 \
    --min-size 0 \
    --top-k 0 \
    --cc-score "$mode" \
    --cc-score-thr "$thr" \
    --normalize nonzero_zscore \
    --tta none \
    --resample-max-zoom-mm 0.0 \
    --slice-spacing-source effective \
    --quiet
  done
done
```

## 次に効きやすい改善（優先度順）

1. **small/medium を狙って recall を上げる**（今はFN優勢ケースが混在）
  - foregroundサンプリング比率の強化（例：70% fg / 30% rand）
  - Lossを「FNを強く罰する」方向へ（例：Tversky / Focal-Tversky / BCE重み）
2. **全foldを同一epochで再学習（公平な比較）**
  - fold0だけ200で他が100だった期間があるため、最終判断は「全fold=200（同条件）」を推奨
3. **解像度（target spacing）を上げる**
  - `2mm`は small 病変が潰れやすい。`1.5mm`や`1mm`（計算重い）を検討
4. **推論側：TTAを投入（まずflip、次にfull）**
  - コストは増えるが、small/mediumの揺れを抑えやすい
- val：case-wise Dice、空病変扱いは固定
- infer：sliding window + TTA（flip） + optional 3-model ensemble

---

## 付録：よくある「0.5で停滞」原因（チェック用）

- foreground sampling が無い / 弱い（病変を見ていない）
- CE単独 or Dice単独で不安定（Dice+CEで安定化）
- 正規化が症例間で一貫していない（brain領域でzscore推奨）
- resampleが不統一（空間スケールがズレている）
- BatchNormを使っている（batch=1で崩れやすい）
- val の Dice定義がブレている（case-wise/空病変の扱い）

---

# pipeline（この作業リポジトリ）との比較メモ

## いままでの実験（要点）とのズレ

- 入力モダリティ
  - レシピ: DWI+ADC（2ch）固定
  - 既存実験: DWI+ADC+FLAIR（3ch）を多用（条件付きCascade Stage2も3ch+stage1prob）
- Resampling
  - レシピ: 等方（1mm/2mm）へ統一
  - 既存実験: `resample_max_zoom_mm=2.0`（主に薄スライスのZ方向だけ「上げる」寄り）
- Patch
  - レシピ: 96^3〜128^3（重ければ縮小）
  - 既存実験: 56×56×24 等、Zを小さくしてMPSで回る設定が中心
- Loss
  - レシピ: Dice + BCE（1:1）
  - 既存実験: `tversky_ohem_bce` など（FN寄り・OHEM併用）
- Valしきい値
  - レシピ: まず0.5固定で比較を崩さない
  - 既存実験: valでしきい値をauto選択（testではthreshold sweepで最適を報告）

## レシピ準拠の「比較用」新条件（config）

以下を追加（DWI+ADCのみ、foreground 50%/random 50%、Dice+BCE、val thr=0.5固定、patchは大きめ）：

- `pipeline/configs/generated/_recipe_20251227/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200.yaml`

注意：`patch_size=[96,96,96]` はMPSでOOMする可能性があります。OOMしたらまず `patch_size` を下げる（例: `[80,80,80]`）か、`base_ch` を16へ落とすのが最短です。

---

# pipelineでの「公式（比較用）」評価定義（固定）

比較のブレを防ぐため **`pipeline/src/evaluation/evaluate_isles.py` の `summary.json` を“公式”として扱う**前提で進めます。

## 評価の定義（重要ポイント）

- `mean_dice`：case-wise Dice の平均（症例ごとDice→平均）
- GTが空（negative case）も平均に含める（FPを出すとDiceが下がる）
- `best_all`：`per_threshold[]` から `mean_dice` 最大の閾値を採用（threshold sweep）
- `detection_rate_case`：GT陽性（gt_vox>0）のみで「1 voxelでもTPがあれば検出」
- `false_alarm_rate_case`：GT陰性のみで「1 voxelでも予測が出たら誤警報」
- 薄スライス集計：`by_slice_spacing` は NIfTI zoom の max を bucket 分け（デフォルト `3.0mm`）
- Resample制約：`--resample-max-zoom-mm 2.0` は最大spacing軸だけを2.0mmまで“上げる”（基本ダウンサンプルしない）

## 比較ルール

- 学習中の監視：`val_threshold=0.5` 固定
- 実験比較/報告：threshold sweep で得た `best_all` を採用

---

# 次の手順（nnU-Net相当：前処理＋5-fold）

## 1) データ統計（spacing/軸/GTサイズ）

```bash
cd /Users/yusukefujinami/ToReBrain/pipeline
/opt/anaconda3/envs/medseg_unet/bin/python tools/data_report.py \
  --config configs/generated/_recipe_20251227/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200.yaml \
  --out-root results/diag
```

## 2) nnU-Net風の前処理プラン（target spacing候補）

```bash
cd /Users/yusukefujinami/ToReBrain/pipeline
/opt/anaconda3/envs/medseg_unet/bin/python tools/plan_nnunet_like_preprocess.py \
  --config configs/generated/_recipe_20251227/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200.yaml \
  --out results/diag/nnunet_plan_my_dataset.json
```

## 3) 5-fold split（test固定）

```bash
cd /Users/yusukefujinami/ToReBrain/pipeline
/opt/anaconda3/envs/medseg_unet/bin/python tools/make_kfold_splits.py \
  --csv-in data/splits/my_dataset_train_val_test.csv \
  --root data/processed/my_dataset \
  --out-dir data/splits/kfold5_my_dataset \
  --k 5 \
  --seed 42 \
  --slice-spacing-thr-mm 3.0

---

## 4) 運用：まずepochs=100で5-foldを1周→伸びたfoldだけ200へ延長

学習時間短縮のため、**最初は全foldをepochs=100で回してCVの当たりを付け**、有望だったfold（または設定）だけを**epochs=200へ延長**します。

### (A) まずepochs=100で回す

今回のk-fold設定ディレクトリ（例）：

- `pipeline/configs/generated/_kfold5_ts222_20251229_104534/`

この中の fold1-4 は `train.epochs: 100` に変更済み（fold0は200のまま）。

学習キュー起動（fold1-4を順次）：

```bash
cd /Users/yusukefujinami/ToReBrain/pipeline
stamp=$(date +%Y%m%d_%H%M%S)
nohup /opt/anaconda3/envs/medseg_unet/bin/python tools/run_train_queue.py \
  --python /opt/anaconda3/envs/medseg_unet/bin/python \
  --repo /Users/yusukefujinami/ToReBrain/pipeline \
  --configs \
    configs/generated/_kfold5_ts222_20251229_104534/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200_kfold5_f1_ts222.yaml \
    configs/generated/_kfold5_ts222_20251229_104534/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200_kfold5_f2_ts222.yaml \
    configs/generated/_kfold5_ts222_20251229_104534/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200_kfold5_f3_ts222.yaml \
    configs/generated/_kfold5_ts222_20251229_104534/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200_kfold5_f4_ts222.yaml \
  > results/diag/train_queue_driver_${stamp}.log 2>&1 &
disown
```

### (B) 伸びたfoldだけepochs=200へ延長（重み継続）

延長したいfoldのYAMLだけ、`train.epochs` を 200 に戻し、`train.init_from` を「そのfoldの `last.pt`」へ向けます。

例（fold1を延長したい場合）：

- config: `..._f1_ts222.yaml`
- init元: `pipeline/runs/3d_unet/<exp_name>/last.pt`

補足：このリポジトリでは “真のresume（optimizer/epoch復元）” ではなく、基本は `train.init_from` による **重みのみロード**で延長します。
```


