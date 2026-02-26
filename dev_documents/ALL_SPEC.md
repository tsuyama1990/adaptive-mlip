# mlip_pipeline 全体仕様書 (ALL_SPEC.md)

## 1. プロジェクト概要
本プロジェクトは、第一原理計算（DFT）データの自動生成から、機械学習原子間ポテンシャル（MLIP）の学習、検証までを一貫して自動化するパイプラインシステム `pyacemaker` を構築する。
最終目標は、「最小限の計算コスト（DFT）で、最高精度の多項式ポテンシャル（Pacemaker/ACE）を、完全に自動で構築すること」である。

本バージョンより、巨大基盤モデル（MACE）をサロゲートモデルとして活用する**「MACE Knowledge Distillation ワークフロー」**をコアアーキテクチャとして採用する。大域的最適化（DIRECTサンプリング）と能動学習（Active Learning）、そしてマルチフィデリティ学習（Delta Learning）を統合した最先端のパイプラインを実現する。

## 2. コア・ワークフロー (The 7-Step MACE Distillation)
`config.yaml` にて `enable_mace_distillation: true` が設定された場合、システムは以下の7ステップでポテンシャルを構築する。

1. **DIRECTサンプリング (Entropy Maximization):**
   記述子空間において、エントロピー（情報量）が最大となるようにDIRECTサンプリングを実行し、未知の構造空間を網羅する初期構造プール（デフォルト: 100点）を生成する。
2. **MACE Uncertainty-based Active Learning:**
   1で生成した構造プールに対し、事前学習済みMACEモデルを用いて不確実性（Uncertainty/分散）を評価。不確実性が高い上位構造のみを抽出し、高コストなDFT計算（VASP等）を実行して真値ラベルを取得する。
3. **MACE Fine-tuning:**
   2で得られた少量のDFTデータを用いて、MACEモデルを対象の系に特化させるようにファインチューニングする。
4. **Surrogate Data Generation:**
   3で賢くなったMACEを用いて短い分子動力学（MD）シミュレーション等を回し、MLIP（ACE）の学習に十分な数（デフォルト: 1000点）の多様な構造を抽出する。
5. **Surrogate Labeling:**
   4で抽出した構造に対し、ファインチューニング済みMACEを用いて高速にエネルギー・力・ストレスの「擬似ラベル」を付与する。
6. **Pacemaker Base Training:**
   5の大量の擬似ラベルデータセットを用いて、Pacemaker（ACEポテンシャル）のベースモデルを学習させる。
7. **Delta Learning (Fine-tuning with DFT):**
   **[最重要ステップ]** 2で取得した高精度なDFTデータ（真値）を用いて、6で構築したPacemakerモデルに対して「デルタ学習（ファインチューニング）」を実行。MACE由来の系統誤差を補正し、真のDFT精度へと着地させる。

## 3. システムアーキテクチャとモジュール拡張方針

本パイプラインは以下の主要モジュールから構成される。**過去の開発サイクルで構築した既存資産を破壊することなく、設定によってシームレスに新フローへ切り替えられるよう実装すること。**

### 3.1. `structure_generator` モジュール
* **既存資産の活用:** 既存のランダム変位やスケーリング手法（`mutations.py`）は維持。
* **新規実装:** DIRECTアルゴリズム等を用いた「エントロピー最大化サンプリング」ロジックを追加。記述子（B基底など）空間での距離を最大化するような目的関数を実装する。

### 3.2. `oracle` モジュール
* **既存資産の活用:** 既存の `Calculator` クラス（DFT実行・パース）をそのまま真値評価用として利用。
* **新規実装:** `MaceSurrogateOracle` を追加。MACEモデルのロード、不確実性（Uncertainty）の計算、および高速な擬似ラベル付与機能（Step 2, 5）を担当する。能動学習（Active Learning）のフィルタリングロジックもここに集約する。

### 3.3. `trainer` モジュール (Pacemaker & MACE)
* **既存資産の活用:** 既存の `active_set.py` や `wrapper.py` で構築したPacemakerの学習プロセス、および過去に検討された**「デルタ学習（Delta Learning）」の基盤をフル活用**する。
* **新規実装:** MACEのファインチューニング（Step 3）を実行するインターフェースを追加。また、Step 7において、既存の学習済みPacemakerモデルを少量のDFTデータで再学習（ロス関数の重みをDFTデータに強く偏らせる等）する機能としてデルタ学習ロジックを統合する。

### 3.4. `orchestrator.py`
* 上記7ステップの順次実行を管理するメインコントローラー。
* 各ステップの入出力を厳密に型定義（Domain Models）で受け渡し、途中でジョブが中断しても再開できる冪等性（Idempotency）を確保すること。

## 4. 開発者（AIエージェント）への特別指示
* **既存コードの最大限の再利用:** ゼロから作り直すのではなく、現在の `mlip_pipeline` ディレクトリにあるコードベース（特に `trainer` や `oracle` の構造）をベースに拡張すること。
* **Delta Learningのシームレスな統合:** Step 7のマルチフィデリティ・デルタ学習は、本システムの精度の要である。過去の実装や構想にある「ベースポテンシャルとの差分を学習する」あるいは「重み付きデータセットで再最適化する」仕組みを確実に動作させること。
* **疎結合とインターフェース:** MACE非依存の環境（従来通りの純粋なDFTループ）でも動くよう、インターフェース（`core/interfaces.py`）を遵守して実装すること。

## 5. 設定ファイル (config.yaml) の拡張スキーマ
```yaml
distillation:
  enable_mace_distillation: true
  
  step1_direct_sampling:
    target_points: 100
    objective: "maximize_entropy"
  
  step2_active_learning:
    uncertainty_threshold: 0.8  # MACEのアンサンブル分散等の閾値
    dft_calculator: "VASP"
  
  step3_mace_finetune:
    base_model: "MACE-MP-0"
    epochs: 50

  step4_surrogate_sampling:
    target_points: 1000
    method: "mace_md"

  step7_pacemaker_finetune:
    enable: true
    weight_dft: 10.0  # デルタ学習時のDFTデータに対する重み