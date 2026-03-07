PYACEMAKER 次世代アーキテクチャ要求定義書 (PRD)
Version: 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)
Date: 2026-02-28
Status: DRAFT
1. プロジェクト背景と目的
1.1. 現行システム（Phase 01）の限界と直面した課題
Phase 01で実装されたPYACEMAKERの基本オーケストレーションは、Active Learningループの基礎を確立した。しかし、実用的なHPC規模（数万〜数百万原子の長時間分子動力学シミュレーション）への展開を想定した場合、以下の致命的な物理的・システム的制約が判明した。
MD連続性の欠落 (Time-Continuity Break):
不確実性（）が閾値を超えてHaltした際、再学習後にMDが初期構造から再スタートする設計となっており、相変態や拡散といった長時間の物理現象に到達できない。
熱揺らぎによる過敏な停止 (Thermal Noise False Positives):
単一の不確実性閾値でHaltを判定しているため、MD中の熱振動による瞬間的なスパイク（物理的には安全なノイズ）に過敏に反応し、不要な計算と無限ループを引き起こす。
局所切り出しによる物理的破綻 (Dangling Bond / Dipole Divergence):
巨大なシステムから不確実な領域（クラスター）を真空中へ単純に切り出してDFTに渡すと、切断面に大量のダングリングボンドが発生し、電荷の不均衡や双極子モーメントの発散を引き起こす。結果としてDFTのSCFループが収束しない、あるいは「ゴミ（非物理的な電子状態）」を学習してしまう。
鈍重なバッチ再学習と破滅的忘却 (Batch Retraining & Catastrophic Forgetting):
毎ステップ全データを用いてバッチ学習をやり直す設計は計算量（）の爆発を招き、かつエラー構造ばかりを追加することで、安定なバルク構造の記述精度が劣化する。
LAMMPS連携の脆弱性 (System Fragility):
C++レイヤーでのLAMMPSのクラッシュ（Lost atoms等）がPythonのメインプロセスを道連れにし、状態（State）の保存すら行われずにオーケストレーター全体が停止する。
1.2. 先行研究（FLARE）からのベストプラクティス抽出
ハーバード大等で開発されたFLAREアーキテクチャのコードベース解析から、以下の4つのパラダイムシフトを本システムに導入する。
Master-Slave逆転 (Inversion of Control): PythonがLAMMPSを操作するのではなく、LAMMPSのC++ループ（fix python/invokeや定期的なコールバック）の内部にPythonを従属させ、MDの時間を巻き戻さずに「一時停止→ポテンシャル更新→再開」をシームレスに行う。
二段階の不確実性閾値 (Two-Tier Thresholds): 「DFTを呼び出すための閾値（threshold_call_dft）」と「学習データに追加するための閾値（threshold_add_train）」を分離し、ノイズへの耐性を持たせる。
計算は全体、学習は局所 (Global Calculation, Local Learning): 物理的に安定した環境で電子状態を計算し、学習モデルの更新には不確実性が高かった「中心原子の力（Force）」のみを用いる。
逐次更新 (Incremental Update): バッチ再学習ではなく、過去の重みを初期値とした差分学習（Delta Learning/インクリメンタル更新）により、計算コストをO(1)に保つ。
1.3. 次世代アーキテクチャの基本思想
上記の課題とFLAREの教訓を統合し、「階層的蒸留（Hierarchical Distillation）」、「インテリジェント・クラスター抽出（Site-specific Cutout & Passivation）」、および**「非同期マスター・スレーブMD（Seamless Resume）」**を中核とする次世代ワークフローを定義する。
FLAREが諦めた「数百万原子のMD」を可能にするため、「切り出さない」のではなく「物理的に完璧に修復して切り出す」アプローチを採用する。さらに、基盤モデル（MACE）からの蒸留とファインチューニングの活用をシステム全体で最大化することにより、最大のボトルネックである高コストなDFT計算の試行回数を極限まで最小化しつつ、ターゲット領域においてDFTに迫る高精度な計算を実現することを目指す。
2. 4段階・階層的蒸留ワークフロー (Core Workflow)
本システムは、対象となる系（例：Fe-Pt-Mg-Oの4元系）に対して、以下の4つのフェーズを順次実行する。
Phase 1: ゼロショット蒸留と初期ポテンシャル構築（Zero-Shot Distillation & Baseline Construction）
目的: DFTを一切呼び出さず、MACE-MP-0等の基盤モデルが持つ「宇宙の常識（広範な汎化性能）」を抽出し、ACEに焼き付ける。
空間分解とコンビナトリアル探索:
入力された元素群から、全単体（種）および全二元系（${N}C{2}$種）のサブシステムを自動定義する。各サブシステムに対し、ランダム構造、歪み（Strain）、Rattle、高温スナップショットに加え、**多様な組成比の変動（Stoichiometry variation）**や、欠陥（空孔、格子間原子、アンチサイト等のディフェクト）の導入を網羅した構造プールを生成する。
DIRECTサンプリングによる情報量最大化 (Active Set Selection):
既存資産である ActiveSetSelector（DIRECTサンプリング / D-Optimality）を活用し、生成された膨大な構造プールから特徴空間において最も情報密度が高く多様性に富んだ構造（数百〜数千個）を抽出する。これにより、冗長なデータを排除し、後段の推論・学習コストを極限まで抑える。
確信度フィルタリング:
抽出された構造を MACEManager（基盤モデルOracle）に渡し、エネルギー・力・不確実性を推論する。この際、MACEの不確実性が閾値以下の「MACEが自信を持っている安全な構造」のみを正解データとして採用する。
ベースラインACE学習 (LJ Delta Learning):
2, 3を通じてchemical spaceを十分に広い範囲で、なおかつ確信度フィルタリングを通過した高品質なデータを用いて PacemakerTrainer を起動し、基礎的な多体ポテンシャル（base.yace）を学習させる。この際、超近接領域での原子のすり抜けを防ぐ（物理的破綻を避ける）ための既存機能として、Lennard-Jones (LJ) ポテンシャルからの Delta Learning をデフォルト構成として適用し、短距離反発のベースラインを担保する。Delta Learningは既存のクラスなどを活用し、パラメータを元素によって最適化する機能などを適宜活用すること
Phase 2: 限界テストと物理バリデーション（Validation & Stress Test）
目的: Phase 1で構築された基礎ポテンシャルが、本番環境で最低限の物理的安定性を担保できるか検証する。
母物質の物理特性検査:
Validator を起動し、各サブシステムの安定相（例: bcc-Fe, fcc-Pt, NaCl-MgO）に対する弾性定数（Bornの安定性基準）、フォノン分散（虚数振動の不在）、および状態方程式（EOS）を計算する。合格基準に達しない場合はPhase 1のサンプリング密度を自動で上げて再学習する。
ミニチュアMDによるストレステスト:
本番環境の縮小版（数千原子程度のスラブモデル等）を作成し、構築したポテンシャルでMDを走らせる。早期にHaltが発生するか、どの温度帯で不確実性が高まるか（Uncertainty Map）をプロファイリングする。
Phase 3: インテリジェント・クラスター抽出（Intelligent Cutout & Passivation）
目的: 大規模MD中に未知の局所構造（界面、欠陥、衝突）に遭遇しHaltした際、DFTで計算可能な物理的に妥当なクリーン・クラスターを自動生成する。
二段階閾値に基づく震源地特定 (Two-Tier Evaluation):
MDのシステム最大不確実性が threshold_call_dft を数ステップ連続で超えた場合にのみHaltを発動させる（熱ノイズの排除）。その後、既存の _get_max_gamma_atom_index などの機能を拡張・活用して個別の原子の不確実性（Site-uncertainty）を評価し、threshold_add_train を超えている原子群を「震源地」として特定する。
球状切り出しと重み付け (Local Learning):
既存資産である utils.extraction.extract_local_region をそのまま活用し、震源地から半径  内の原子に force_weight = 1.0 を、半径  内の原子に force_weight = 0.0 を付与する。さらに既存の utils.embedding.embed_cluster 等を呼び出し、切り出したクラスターを真空層付きの周期境界セル（PBC）に安全に再配置する。これにより学習対象をコアのみに絞り込む。
MACEによる境界事前緩和（Pre-relaxation）:
既存のMLIPラッパー機構（現行の m3gnet_wrapper.py の枠組み）を基盤モデル（MACE）用に拡張し適用する。**コア原子の座標は固定（Freeze）**したまま、MACEを用いてバッファ領域の原子の座標のみをエネルギー極小化（Relax）する。これにより切り出し時の不自然な結合歪みを解消する。
自動終端処理（Auto-Passivation）:
バッファ領域外縁の切断された結合（特に酸化物のOやMgなど）に対し、Fractional Hydrogen等のダミー原子を自動配置し、クラスター全体の電荷とダイポールモーメントを中性化する（utils.structure等に新規統合）。
クリーンDFT計算:
物理的・電気的に安定化されたクラスターを、既存資産である interfaces.qe_driver.QEDriver および core.oracle.DFTManager に渡す。既存の自己修復（Self-Healing）機能（Smearingの拡張やMixing Betaの自動調整）をフル活用し、SCFを確実に収束させてコア原子に対する真の力（Ground Truth Force）を取得する。
Phase 4: 階層的ファインチューニング（Hierarchical Delta Learning）
目的: 取得した貴重な少数のDFTデータを用いて、MACEとACEを連鎖的にアップデートし、MDを再開する。
MACEの覚醒（Finetune MACE）:
取得したDFTデータを用いて、MACE自体をファインチューニングする。これにより基盤モデルが「対象系の特異な界面物理」を完全に理解した状態（覚醒MACE）となる。
サロゲートデータの爆発的生成:
覚醒MACEをOracleとして用い、Haltした周辺のフェーズ空間（ランダム変位や微小MD）で数千個のサロゲートデータを一瞬で生成・推論する。
ACEのデルタ学習 (Incremental Update):
大量のサロゲートデータと、アンカーとなるDFT真値データを PacemakerTrainer に入力し、ACEポテンシャルを更新する。この際、計算量爆発を防ぐためゼロからの学習は行わず、直前のポテンシャルからの差分学習とし、さらにリプレイバッファを混入させる。
シームレス・レジューム (Master-Slave Resume):
更新されたポテンシャルを読み込み、Haltした直後のステップ（時間、座標、速度）からMDを安全に再開する。
3. モジュール別要求仕様
3.1. pyacemaker.utils.extraction (大幅拡張)
クラスター抽出と終端処理を担う、本アーキテクチャの要となるモジュール。
extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms
入力: 巨大なASE Atomsオブジェクト、threshold_add_trainを超えた対象原子のインデックスリスト。
処理:
neighbor_listを用いた  と  の球状抽出。
force_weight 配列の付与（Core=1.0, Buffer=0.0）。
_pre_relax_buffer(cluster, mace_calc): コアを ase.constraints.FixAtoms で固定し、MACEを用いてバッファをLBFGSで緩和する。
_passivate_surface(cluster): 電気陰性度と結合半径から未結合手を検出し、適宜Hまたは擬似原子を追加する（追加原子の force_weight は当然0.0）。
出力: 周期境界を持ち、真空層が挿入され、終端処理が施された計算可能な Atoms オブジェクト。
3.2. pyacemaker.core.oracle (多段化)
Oracleを抽象化し、基盤モデル（MACE）と第一原理計算（DFT）を透過的に扱えるようにする。
class MACEManager(BaseOracle)
MACE-MP-0等の推論を実行するラッパー。GPU対応。
エネルギー、力に加え、アンサンブル分散や潜在特徴空間の距離に基づく不確実性（Uncertainty）を出力する機能が必須。
class TieredOracle(BaseOracle)
クエリ戦略を管理する。構造を受け取った際、まずは MACEManager で推論し、不確実性が特定の閾値を超えた場合のみ QEDriver (DFT) にフォールバックするルーティングロジックを持つ。
3.3. pyacemaker.core.engine (LAMMPS連携とシームレス再開)
LAMMPSのクラッシュに耐え、Halt後の時間を連続させる堅牢なエンジン。FLAREのMaster-Slaveパラダイムを適用。
fix python/invoke の活用 (推奨アプローチ):
LAMMPSのC++実行ループから直接Pythonの検証スクリプトを毎Nステップ呼び出す。不確実性が閾値を超えた場合はMDをポーズ状態にし、バックグラウンドでOrchestrator（学習パイプライン）を走らせる。完了後、pair_coeffを動的にリロードしてMDを継続する。
Process Isolation と read_restart (フォールバック・アプローチ):
C++連携が技術的課題となる場合は別プロセス化する。LAMMPSがクラッシュしてもメインループは生き残り、定期保存された.restartファイルから速度分布とアンサンブル状態を完全に引き継いで再開する。
ソフトスタート（温度急上昇防止）:
ポテンシャルが切り替わった直後のエネルギー不連続による系の破綻を防ぐため、再開直後の  ステップは強いLangevin熱浴（fix langevin）をかけて系を熱化（Thermalize）させるロジックを自動挿入する。
3.4. pyacemaker.core.trainer (Pacemaker & MACE Finetune)
FinetuneManager:
DFTから取得したクリーンなデータセットを用いて、MACEのPyTorchモデルの最終層付近（Readout layer）を短時間学習させるラッパー。
PacemakerTrainer のインクリメンタル更新・デルタ学習強化:
バッチ学習の計算量爆発を防ぐため、前回のポテンシャル状態を引継ぎ、過去の学習データ（training_history.extxyz）からランダムにサンプリングした固定サイズのリプレイバッファを現在の学習セットに混合する。
LJポテンシャルからのDelta Learningを実行するための設定を input.yaml に自動生成する機能を担保する。
4. データモデル要件 (domain_models/config.py)
新たなワークフローを制御するためのPydanticモデルの拡張。
class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation設定"""
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="MACEが自信を持つ閾値")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    """FLAREにインスパイアされた二段階閾値"""
    threshold_call_dft: float = Field(0.05, description="MDをHaltしてDFTを呼び出す基準")
    threshold_add_train: float = Field(0.02, description="学習セットに追加する原子を選ぶ基準")
    smooth_steps: int = Field(3, description="熱ノイズ排除のため、閾値超過が連続するべきステップ数")

class CutoutConfig(BaseModel):
    """Phase 3: インテリジェント切り出し設定"""
    core_radius: float = Field(4.0, description="Force Weight 1.0の半径")
    buffer_radius: float = Field(3.0, description="追加の緩和バッファ層の厚さ")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    """Active Learning ループの戦略設定"""
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="破滅的忘却を防ぐための過去データ保持数")
    baseline_potential_type: str = Field("LJ", description="ベースラインとなる物理ポテンシャル (LJなど)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)


5. 非機能要件・HPC運用要件
5.1. ステート管理とトランザクション (Robust Checkpointing)
Task-level Checkpointing:
イテレーション単位の粗い保存ではなく、DFTの1計算、サロゲートの1生成ごとにJSONまたはSQLiteベースのローカルDBに状態をコミットする。HPCジョブがWall-time（実行時間制限）で強制キルされても、再投入時に秒単位でレジューム可能とする。
Artifact Cleanup:
数百万ステップのMDから生成されるダンプファイルやQEの巨大な波動関数ファイル（.wfc）は、学習・推論に成功した直後に自動的に圧縮（gzip）または削除するデーモンプロセスを並行稼働させる。
5.2. スケジューラ連携と並列化 (HPC Dispatch)
Oracle（DFT計算）は直列実行ではなく、concurrent.futures や Dask 等を用いて、利用可能なノード/GPUへ非同期にディスパッチする。
PacemakerTrainer のサブプロセス呼び出し時、HPC環境（Slurmの srun, PBSの mpiexec）のプレフィックスを環境変数から動的に組み立てるジョブテンプレート機能（JobDispatcher）を実装する。
6. 実装フェーズ（マイルストーン）提案
Sprint 1: Core Extraction & Two-Tier Evaluator
extraction.py の再設計。MACEによるPre-relaxationとH終端のアルゴリズム実装・テスト。
二段階閾値判定（Call DFT vs Add Train）のロジック構築。
Sprint 2: Master-Slave Inversion & Seamless Resume
fix python/invokeを用いたLAMMPSとの密結合、または .restart ファイルを用いた堅牢なMD継続機構の実装。LJベースのDelta Learning設定の統合。
Sprint 3: Hierarchical Distillation Loop
Orchestratorの書き換え。Phase 1〜4のフローを統合し、MACEManager と TieredOracle を接続する。差分学習（Incremental Update）の導入。
Sprint 4: Scale & Robustness
SQLiteベースの細かいチェックポイント機能と、マルチノード実行向けの非同期ディスパッチ機構の導入。
End of Document
