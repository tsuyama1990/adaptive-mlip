PYACEMAKER 高効率MLIP構築・運用システム 要件定義書 (Comprehensive Edition)1. プロジェクト概要本プロジェクトは、原子間ポテンシャル作成ツール「Pacemaker (ACE: Atomic Cluster Expansion)」を核とし、材料科学の専門知識が浅いユーザーであっても、最小限の工数で「State-of-the-Art（最先端）」な機械学習ポテンシャル (MLIP) を構築・運用できる自動化システムを開発することを目的とする。1.1 背景と課題：原子シミュレーションの民主化に向けて現代の計算材料科学において、第一原理計算（DFT）の精度と古典分子動力学（MD）のスケールを両立させるMLIPは不可欠なツールとなっている。しかし、高品質なMLIPの構築には、データサイエンスと計算物理の両面にわたる深い専門知識が必要であり、多くの実験研究者や企業の研究者にとって参入障壁が高いのが現状である。従来のMLIP構築フローは、専門家による手動の繰り返し作業（構造作成→DFT→学習→検証）に依存しており、以下の構造的な課題を抱えていた。構造サンプリングの偏りと「外挿」の危険性:標準的な平衡状態のMDだけでは、相転移、化学反応、破壊現象などで現れる「レアイベント」や「高エネルギー配置」を網羅できない。未知の領域（Extrapolation region）にシミュレーションが突入した際、ポテンシャルが物理的にあり得ない力（例: 原子核同士が重なっても斥力が働かないなど）を出力し、シミュレーションが破綻するリスクがある。「ゴミ」データの蓄積と計算資源の浪費:物理的に類似した構造（相関の強いスナップショット）を大量にDFT計算しても、ポテンシャルの精度向上には寄与しない。情報量の低い構造に高価な計算リソースを費やすことは、プロジェクトのコスト効率を著しく低下させる。運用開始後のメンテナンスコスト:シミュレーション中にポテンシャルが破綻した場合、原因となる構造を特定し、再学習を行い、再度シミュレーションを流すという手戻り作業が煩雑であり、研究のボトルネックとなる。1.2 目標 (Success Metrics)本システムは、以下の指標を達成することを目標とする。工数の劇的削減 (Zero-Config Workflow):初期設定ファイル (YAML) 1つで、初期構造生成から学習完了までのパイプラインを無人で完走させる。ユーザーはPythonスクリプトを書く必要がない。データ効率の最大化 (Data Efficiency):能動学習 (Active Learning) と物理に基づいた高度なサンプリング手法を組み合わせ、ランダムサンプリングと比較して1/10以下のDFT計算量で同等の精度（RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å）を目指す。物理的堅牢性 (Physics-Informed Robustness):Core-Repulsion（原子核近傍の斥力）における物理的な正しさを、LJポテンシャルからのデルタ学習によって強制する。データが存在しない領域でも、少なくとも原子が重なり合って崩壊しない「物理的な安全性」を担保する。スケーラビリティと拡張性:局所的なActive Learningから、数百万原子規模のMDや、秒単位の時間スケールを扱うkMCシミュレーションへとシームレスに展開できるアーキテクチャを構築する。2. システムアーキテクチャ詳細本システムは、モジュール間の疎結合を保ちつつ、全体を統括する Pythonベースの「Orchestrator」 を中心に設計される。各モジュールは独立したコンテナ（Docker/Singularity）として動作可能とし、ローカルワークステーションからクラウド/HPC環境へのデプロイを容易にする。2.1 モジュール構成Structure Generator (構造探索モジュール):役割: 未知の化学空間・構造空間を探索し、学習候補となる原子配置を提案する「探検家」。特徴: 物理的直感に基づいたバイアス（温度、圧力、欠陥、化学ポテンシャル）を印加し、闇雲なランダム探索ではなく「意味のある」多様性を追求する。Oracle (教師データ生成モジュール):役割: 提案された構造に対し、第一原理計算 (DFT) を実行して正解データ（エネルギー、力、応力）を付与する「賢者」。特徴: 計算の失敗（SCF収束エラー等）を自動検知し、パラメータ（Mixing beta, Smearing等）を動的に調整して再計算する自己修復機能を持つ。Trainer (学習モジュール):役割: Pacemakerエンジンを駆動し、ACEポテンシャルをフィッティングする「学習者」。特徴: 物理ベースライン（LJ/ZBL）との差分学習を管理し、過学習を防ぐための正則化（Regularization）を自動調整する。Dynamics Engine (推論・運用モジュール):役割: 生成されたポテンシャルを用いてMD/kMCを実行し、リアルタイムで信頼性（Uncertainty）を監視する「実行者」。特徴: LAMMPSおよびEON (kMC) との密な連携インターフェース（Python C-API等）を持ち、シミュレーションを中断・再開する制御権を持つ。2.2 データフロー (The Active Learning Cycle)システムは以下のサイクルを自律的に繰り返す。Exploration (探索): Dynamics Engine または Structure Generator が構造空間を探索する。Detection (検知): 不確実性指標 $\gamma$ (extrapolation grade) が閾値を超えた構造、または幾何学的に新規な構造を検出する。Selection & Calculation (選別と計算):冗長な構造（既存データと類似した構造）を除外。Periodic Embedding により計算コストを最小化した小さなセルを作成。Oracle へ DFT 計算リクエストを送信。Refinement (学習):新規データをデータセットに統合。Trainer がポテンシャルを更新（Fine-tuning）。Deployment (配備):更新された potential.yace を Dynamics Engine へホットデプロイ。シミュレーションを再開。3. 機能要件詳細3.1 自動トレーニング構造作成機能 (Structure Generator)学習データの「質」を決定づける最重要モジュールである。本システムでは、従来の固定的なルールベース（暗黙知）による切り替えを廃止し、対象系の特徴量から最適な探索戦略を動的に導出する 「適応的探索ポリシー (Adaptive Exploration Policy)」 を実装する。適応的探索ポリシーエンジン (Adaptive Exploration Policy Engine):概要:「合金だからMC」といった単純な分岐ではなく、組成、予測物性、不確実性分布を入力とし、MD/MC比率や温度スケジュールなどのハイパーパラメータを出力する決定モデル（Policy）を構築する。入力特徴量 (Input Features):初期探索（M3GNet/CHGNet等）の結果およびPymatgen等の静的解析から以下を取得する。Material DNA: 元素種、組成比、平均価電子数、結晶系、空間群。Predicted Properties: 汎用ポテンシャルで推定したバンドギャップ ($E_g$)、融点 ($T_m^{pred}$)、体積弾性率 ($B_0$)。Uncertainty State: 初期構造に対する $\gamma$ 値の分布形状（平均値、分散、最大値）。出力戦略パラメータ (Output Strategy Parameters):以下のパラメータを動的に決定し、LAMMPS/Pythonスクリプトへ渡す。$R_{MD/MC}$ (MD/MC Ratio): 純粋なMDステップに対する原子交換MCの頻度。$T_{schedule}$ (Temperature Schedule): 最高温度 ($T_{max}$)、昇温速度 ($dT/dt$)。$N_{defects}$ (Defect Density): 導入すべき空孔・格子間原子の濃度。$\epsilon_{range}$ (Strain Range): EOS/弾性計算のための歪み印加範囲。ポリシー決定ロジック (Decision Logic Example):特徴量条件判定される物理的レジーム適用される探索ポリシー (Action)$E_g \approx 0$ (金属) & 多成分系拡散・配置エントロピー支配High-MC Policy: $R_{MD/MC}$を高く設定 (例: 100 step毎)。$T_{max}$は$0.8 T_m$付近を重点的に。$E_g > 0$ (絶縁体) & 複雑な単位胞格子歪み・欠陥支配Defect-Driven Policy: $N_{defects}$を高く設定 (スーパーセル内に多種類の欠陥導入)。MCはOFFまたは低頻度。$B_0$ が高い (硬い材料)共有結合・方向性支配Strain-Heavy Policy: $\epsilon_{range}$を広め ($\pm 15\%$) に設定し、せん断変形を多めにサンプリング。初期 $\gamma$ 分散が大きい知識不足 (High Uncertainty)Cautious Exploration: $T_{schedule}$を緩やかにし、低温域でのデータ収集を優先して基礎を固める。実行可能なアクション空間 (Action Space):ポリシーによって選択・パラメータ調整される具体的なサンプリング手法群。1. Initial Exploration via Universal Potentials (Cold Start):役割: Policyの入力となる初期特徴量 ($T_m^{pred}$ 等) を取得するための事前調査。動作: ユーザー入力組成に対し、M3GNet等を用いて超高速でスクリーニングMDを行い、大まかな安定構造と物性を推定する。2. Variable T-P Ramping (温度・圧力スキャン):制御: Policyが出力した $T_{max}$ と圧力範囲に基づき、LAMMPSの fix npt 制御変数を設定する。詳細: 低温（フォノン）、中温（熱膨張）、高温・高圧（液体・高密度相）を、指定されたスケジュールで走査する。3. Hybrid MD/MC Sampling (化学空間探索):制御: Policyが出力した $R_{MD/MC}$ に基づき、LAMMPSの fix atom/swap の頻度と確率を調整する。詳細: 拡散が遅い系でも、物理時間を無視して化学組成の局所平衡（規則化・偏析）を探索する。4. Defect & Distortion Engineering (構造空間探索):制御: Policyが出力した $N_{defects}$ と $\epsilon_{range}$ に基づき、Pythonスクリプトで構造生成を行う。詳細 (One Defect Strategy): 空孔、格子間原子、アンチサイトを導入。詳細 (EOS & Strain): 等方的な体積変化 ($\pm \epsilon_{range}$) と、ランダムなせん断歪み (Rattling) を加えた構造を生成し、弾性特性の学習データを確保する。5. Normal Mode & Torsion (分子・局所モード探索):適用: Policyが「分子性」または「低次元構造」と判定した場合に発動。詳細: 結合角や二面角を強制的に操作し、MDでは乗り越えられないエネルギー障壁の向こう側を探索する。3.2 教師データ生成・ラベリング機能 (Oracle)DFT計算は計算コストが高く、かつエラーが発生しやすいため、ここでの自動化と効率化がシステム全体の性能を左右する。添付資料「Automating Quantum Espresso Static Calculations」に基づき、堅牢なプロトコルを実装する。DFT自動計算パイプラインと自己修復機能:入力生成: Quantum Espresso (QE) または VASP の入力ファイルを、構造の特性に応じて動的に最適化する。K-space Sampling: 固定のk点グリッド数（例: $4 \times 4 \times 4$）ではなく、kspacing (逆空間密度、例: $0.03 \sim 0.05 \AA^{-1}$) を指標としてグリッドを自動生成する。これにより、構造探索やEmbeddingによってセルサイズが変動しても、計算精度（エネルギーの収束性）を一貫させ、かつ過剰なk点による計算資源の浪費を防ぐ。Pseudopotentials: ユーザーが個別に指定するのではなく、SSSP (Standard Solid State Pseudopotentials) などの検証済み標準ライブラリ（Precision または Efficiency モード）を自動的に参照・適用する。これにより、ポテンシャルの質に起因する計算エラーやゴーストステートを防ぐ。Smearing: 金属/絶縁体の区別が曖昧な場合、安全側に倒して occupation='smearing' (Marzari-Vanderbilt) を採用し、SCF収束性を高める。Spin Polarization: 遷移金属（Fe, Co, Ni, Mn等）が含まれる場合、自動的にスピン分極計算をONにし、初期磁気モーメントを強磁性に設定して計算崩壊を防ぐ。Error Handling (Self-Correction): 計算が収束しなかった場合、自動的に以下の対策を順次試行するロジックを実装する。混合パラメータ (mixing_beta) を下げる（例: 0.7 -> 0.3）。対角化アルゴリズムを変更する（例: david -> cg）。電子温度（Smearing width）を上げる。Static Calculation (一点計算):設定: 構造緩和 (relax) ではなく、原子位置を固定したまま電子状態のみを解く。QE固有の要件: calculation='scf' を指定するが、デフォルトでは力が計算されないため、必ず tprnfor=.true. および tstress=.true. を設定ファイルに追加する。これにより、原子を動かすことなく、ポテンシャル学習に必要なHellmann-Feynman力と応力テンソルを正確に抽出する。（nscf は電荷密度が必要なため、教師データ生成の文脈では通常使用しない）Periodic Embedding (周期的埋め込み) によるデータ切り出し:背景: 大規模MD中に検出された「不確実な局所領域」だけをDFT計算したいが、単純にクラスターとして切り出すと、表面（真空）の効果がノイズとして混入する。また、球状に切り出すと周期境界条件を適用できない。手法:不確実性の高い原子を中心とし、ACEのカットオフ半径 $R_{cut}$ をカバーする領域を特定する。この領域の周囲に、さらに $R_{buffer}$ (約2層分) のバッファ領域を含めた上で、これらを包含する**直方体状のセル（Orthorhombic Box）**を切り出す。球状に切り出すと周期境界条件を満たせないため、必ず空間を隙間なく埋められる形状とする。この直方体セルを真空に浮かべるのではなく、周期境界条件を持つ小さなスーパーセル として再定義する（Periodic Embedding）。DFT計算後、中心領域（半径 $R_{cut}$）の原子の力のみを正解データとして採用し、境界付近（$R_{buffer}$）の原子の力は「表面効果の影響を受けている」として学習ウェイトをゼロにする（Force Masking）。利点: 界面、転位芯、アモルファス構造などの複雑な局所環境を、表面効果というアーティファクトなしに学習可能にする。3.3 学習機能 (Trainer)Pacemakerの機能をフル活用し、「少ないデータで賢く」学習する。過学習を防ぐための物理的な制約を重視する。Pacemakerのサイトを参照の上フル活用すること (https://pacemaker.readthedocs.io/en/feature-docs/)Delta Learning (LJ Baseline) の強制とハイブリッド運用:理論: 全ポテンシャルエネルギー $E_{total}$ を、$E_{total} = E_{baseline} + E_{ACE}$ と分解する。実装 (学習フェーズ):各元素ペアに対し、原子半径に基づいた標準的なLennard-Jones (LJ) パラメータ、あるいはZBLポテンシャル（近距離核反発）を自動設定する。Pacemaker設定ファイルにおいて、この参照ポテンシャルを定義し、ACE部分は「LJでは表現しきれない多体相互作用の残差」のみを学習するよう構成する。実装 (推論・MDフェーズ):合成ポテンシャルの利用: 学習済みモデルをLAMMPSで実行する際も、単独のACEポテンシャルではなく、必ず ベースライン(LJ/ZBL) と ACE を重ね合わせた合成ポテンシャル を使用する。LAMMPS設定: pair_style hybrid/overlay コマンドを用い、lj/cut (または zbl) と pace を重畳させる記述を in.lammps に自動生成する。これにより、学習時と全く等価なエネルギー局面を再現する。効果: 学習データが存在しない極短距離（原子同士が重なる領域）において、ACE多項式の暴走を防ぎ、必ず物理的な強い斥力が働くことを保証する。これにより、MD中の原子衝突による爆発（Segmentation Fault）をほぼ確実に回避できる。Active Set Optimization (D-Optimality):背景: データ数が増えると学習コストは線形～二乗で増加するが、似通ったデータの追加は精度向上に寄与しない。手法: 蓄積された数千～数万の構造データ全てを使って学習するのではなく、線形代数的な情報量（行列式）が最大となるような「代表構造 (Active Set)」を選別する。ツール: pace_activeset コマンドをバックグラウンドで実行し、MaxVolアルゴリズムを用いて基底関数の係数決定に最も寄与する構造のみをフィルタリングする。これにより、数万の候補構造から数百の「真に重要な構造」だけを抽出して学習に用いる。3.4 推論・On-the-Fly (OTF) 学習機能 (Dynamics Engine)能動学習の現場となる実行エンジンであり、本システムの心臓部である。ここでは、古典分子動力学 (MD) とアダプティブ・キネティック・モンテカルロ (aKMC) をシームレスに統合し、自律的な学習と推論のサイクルを実現する詳細仕様を定義する。ハイブリッドポテンシャルの適用 (Hybrid Potential Application):目的と背景: 純粋なACEポテンシャルは多項式展開であるため、学習データが存在しない原子核同士の極端な接近領域（Core region）において、非物理的な引力井戸を形成する場合がある。これは高エネルギー衝突時にMDシミュレーションの即時崩壊（Segmentation Fault）を招くリスクがある。要件: 3.3節で定義した物理ベースライン（LJまたはZBL）を、推論時にも厳格に適用し、安全装置として機能させる。LAMMPS実装:pair_style hybrid/overlay コマンドを使用し、物理ベースライン (lj/cut や zbl) と機械学習部分 (pace) を加算的に重ね合わせる記述を in.lammps に自動生成する。設定例:# ZBLをベースラインとし、ACEを上乗せする設定例
pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace potential.yace Ti O
pair_coeff * * zbl 22 8  # 原子番号を指定
この構成により、万が一ACE部分が外挿領域で異常な値を返しても、物理ベースラインの強い斥力が支配的となり、原子の重なりとシミュレーションの破綻を物理的に阻止する。不確実性監視 (Uncertainty Quantification) と fix halt:メカニズム:PacemakerのACEポテンシャルは、推論時に各原子の局所環境記述子（$\mathbf{c}_i$）を計算し、学習データセットの張る部分空間からの逸脱度合い（マハラノビス距離に類似）を示す指標 $\gamma$ (extrapolation grade) をリアルタイムに出力する機能を持つ。$\gamma \approx 0$ は学習データに近い既知の領域、$\gamma \gg 1$ は未知の危険領域を意味する。LAMMPS実装とトリガー:compute pace コマンド（USER-PACEパッケージ）を用いて、全原子の最大 $\gamma$ 値 (v_max_gamma) を監視する。fix halt コマンドを使用し、v_max_gamma > threshold (推奨初期値: 5.0, 学習進行に伴い緩和可能) となった瞬間にシミュレーションを強制停止し、終了コードを返す。パフォーマンス最適化:毎ステップの $\gamma$ 計算はコストが高いため、run every 10 などの設定により、10〜100ステップごとの間欠的な監視を行う。これにより、OTF監視による計算オーバーヘッドを10%未満に抑えつつ、破綻前に検知する応答性を確保する。自動再学習ループ (The Anatomy of a Halt Event):停止イベントが発生した際、Pythonオーケストレーターは以下のシーケンスを実行し、ポテンシャルを「治療」して再開する。単なる「穴埋め（一点学習）」ではなく、不確実性が高い領域の「局所的な曲率」を学習し、壁を平滑化するための近傍サンプリング戦略を採用する。Halt & Diagnose (停止と診断):LAMMPSが特定の終了コードで停止し、制御がPythonドライバに戻る。ログを解析し、停止の原因となったタイムステップと、最大 $\gamma$ 値を記録した原子ID群を特定し、その構造 $S_0$ (Halt Structure) を抽出する。Generate Local Candidates (近傍候補の生成):抽出した $S_0$ を中心に、以下のいずれかの手法で 20〜30 個程度の近傍構造候補 $\{S_i\}$ を安価に生成する（この段階ではDFTは行わない）。(A) Normal Mode Approximation: $S_0$ におけるHessianをACEあるいは汎用ポテンシャルで近似計算し、最大曲率方向へ $\pm \epsilon$ 変位させる（推奨）。(B) MD Micro-burst: $S_0$ から非常に短い時間（5〜20ステップ）、高温または小刻みな時間刻みでMDを走らせる。(C) Random Displacement: 高 $\gamma$ 原子に対し、微小なランダム変位 ($0.01 \sim 0.05 \AA$) を加える。Local D-Optimality Selection (局所的な選別):生成された候補構造群 $\{S_i\}$ と $S_0$ に対し、pace_activeset をローカルに実行する。情報行列の行列式 (D-Optimality) を最大化するような 5〜10 個の構造セット $\{S_{selected}\}$ を選出する。この際、$S_0$ は必ず含める（Anchor）。設定: 推奨デフォルト数は 5 点だが、系（金属: 3-5, 分子: 6-10）やHaltの頻度に応じて config.yaml で調整可能とする。目的: 単一点だけでなく、その周囲の「勾配の変化（曲率）」を学習させることで、再び同じ穴に落ちるのを防ぐ（再Halt防止）。Embed (埋め込み):選出された $\{S_{selected}\}$ の各構造に対し、3.2節の Periodic Embedding 処理を適用する。単なるクラスター（真空あり）ではなく、周期境界条件を満たす直方体状の小型スーパーセルを作成し、バルク性質を保ったままDFT計算可能な形式に変換する。Compute (正解データの生成):生成されたセル群をOracleへ一括投入し、DFT計算を実行して正確な力とエネルギーを取得する。Update (再学習):新規データを既存の学習セットに追加する。Pacemakerを実行する際、ゼロから学習するのではなく、前回のモデル重みを初期値として読み込み (--initial_potential)、学習率を下げて数エポックだけ回す Fine-tuning を行う。これにより、学習時間を数時間から数分へと短縮する。Resume (再開):更新された potential.yace をLAMMPSの作業ディレクトリに配置する。LAMMPSを read_restart で停止直前の状態から復帰させ、新しいポテンシャルをロードしてシミュレーションを続行する。Scale-up: MD/kMC 連携:背景: MDはナノ秒スケールの現象しか追えないが、実際の材料劣化や拡散現象は秒〜年のスケールで進行する。これを解決するために、Adaptive Kinetic Monte Carlo (aKMC) を導入し、時間スケールの壁を突破する。kMC (Kinetic Monte Carlo) 実装:EON (Eon Client/Server) 等のaKMCソフトウェアと連携する。サドル点探索（Nudged Elastic BandやDimer法）において、エネルギーと力の計算エンジンとしてACEポテンシャルを使用する。連携ロジック:遷移状態（サドル点）は、ポテンシャルエネルギー曲面上で原子配置が歪んだ高エネルギー状態であり、「未知の領域」になりやすい。サドル点探索中に構造の $\gamma$ 値が高くなった場合、即座に上述のOTFループ（Extract -> Embed -> ...）を起動する。これにより、「MDでは到達できないが、熱力学的に重要な遷移パス」を能動的に学習し、拡散係数や反応速度定数の予測精度を劇的に向上させる。役割分担とシナジー:MD: 短時間の熱振動、液体構造、高速な拡散、およびエントロピー的な配置の探索を担当。kMC: 固体内の空孔拡散、表面吸着、相変態の核生成など、MDでは到達不可能な長時間スケール現象の探索を担当。この両輪を共通のポテンシャル・共通の学習ループで回すことで、時間スケールと空間スケールの双方をカバーする「全能型」のポテンシャルを育成する。3.5 Pythonオーケストレーション詳細設計 (Implementation Blueprints)システム全体を統括するPythonフレームワーク（pyacemaker）の実装設計。ファイル管理の厳格化と、各モジュール（LAMMPS, ASE, Pacemaker）の具体的な連携コードを定義する。ディレクトリ構造 (Directory Strategy):混乱を避けるため、以下のような階層構造を自動生成する。project_root/
├── config.yaml               # ユーザー設定
├── data/
│   ├── initial_train.pckl.gzip  # 初期学習データ
│   └── accumulated.pckl.gzip    # 蓄積された全データ
├── potentials/
│   ├── generation_000.yace      # 初期ポテンシャル
│   └── generation_NNN.yace      # 最新ポテンシャル
└── active_learning/
    ├── iter_001/
    │   ├── md_run/               # LAMMPS実行ディレクトリ
    │   ├── kmc_run/              # KMC実行ディレクトリ (EON)
    │   ├── candidates/           # 切り出された候補構造
    │   ├── dft_calc/             # DFT計算ディレクトリ
    │   └── report.json           # 実行ログ
    ├── iter_002/
    └── ...
Core Orchestrator Logic (The Brain):状態遷移を管理するメインループの実装イメージ。import time
from pathlib import Path
import shutil

class Orchestrator:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.md_engine = MDInterface(self.config['lammps'])
        self.oracle = DFTManager(self.config['dft'])
        self.trainer = PacemakerWrapper(self.config['training'])
        self.validator = Validator(self.config['validation'])
        self.iteration = 0

    def run_cycle(self):
        # 最新ポテンシャルの特定
        current_pot = self.trainer.get_latest_potential()

        # 作業ディレクトリ作成
        work_dir = Path(f"active_learning/iter_{self.iteration:03d}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. LAMMPS MD実行 (Exploration)
        halt_info = self.md_engine.run_exploration(
            potential=current_pot,
            work_dir=work_dir / "md_run"
        )

        if not halt_info['halted']:
            print("MD completed without high uncertainty. Converged?")
            return "CONVERGED"

        # 2. 構造抽出 (Selection) - NEW: Generate local candidates & D-opt select
        high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
            dump_file=halt_info['dump_file'],
            threshold=self.config['otf_loop']['uncertainty_threshold']
        )

        # Local D-Optimality Selection Logic (Pseudo-code)
        selected_structures = []
        for s0 in high_gamma_atoms:
            candidates = self.structure_generator.generate_local_candidates(s0, n=20)
            selected = self.trainer.select_local_active_set(candidates, anchor=s0, n=5)
            selected_structures.extend(selected)

        # 3. DFT計算 (Oracle)
        new_data = self.oracle.compute_batch(selected_structures, work_dir / "dft_calc")

        if not new_data:
            print("No valid data obtained from DFT.")
            return "ERROR"

        # 4. 学習データの更新 (Data Management)
        dataset_path = self.trainer.update_dataset(new_data)

        # 5. 再学習 (Refinement)
        new_pot_path = self.trainer.train(
            dataset=dataset_path,
            initial_potential=current_pot,
            output_dir=work_dir / "training"
        )

        # 6. 検証 (Validation)
        validation_result = self.validator.validate(new_pot_path)
        if not validation_result['passed']:
            print(f"Validation failed: {validation_result['reason']}")

        # 次のサイクルへ
        self.iteration += 1
        shutil.copy(new_pot_path, f"potentials/generation_{self.iteration:03d}.yace")
Module Interface Details:A. MDInterface (LAMMPS Control):lammps Pythonモジュール（または subprocess）を使用するが、ここではより柔軟なPythonモジュール版を推奨。from lammps import lammps

class MDInterface:
    def run_exploration(self, potential, work_dir):
        lmp = lammps()
        # ... (LAMMPSの初期化コマンド発行) ...

        # Hybrid Potentialの設定
        lmp.command("pair_style hybrid/overlay pace zbl 1.0 2.0")
        lmp.command(f"pair_coeff * * pace {potential} ...")

        # Watchdogの設定 (gamma監視)
        lmp.command("compute pace_gamma all pace ... gamma_mode=1")
        lmp.command("variable max_gamma equal max(c_pace_gamma)")
        lmp.command("fix watchdog all halt 10 v_max_gamma > 5.0 error hard")

        try:
            lmp.command("run 100000")
            return {'halted': False}
        except Exception as e:
            # LAMMPSがerror hardで終了した場合、Python側でキャッチ可能
            print("Halt triggered by uncertainty watchdog!")
            return {'halted': True, 'dump_file': f"{work_dir}/dump.lammps"}
B. DFTManager (ASE Integration):複雑なDFT入力作成を ase.calculators.espellipso (QE) 等に委譲する。from ase.io import read, write
from ase.calculators.espresso import Espresso

class DFTManager:
    def compute_batch(self, structures, calc_dir):
        results = []
        for atoms in structures:
            # Periodic Embedding処理 (直方体セル化 + バッファ)
            embedded_atoms = self._apply_periodic_embedding(atoms)

            # Calculator設定 (自己修復ロジック付き)
            calc = Espresso(
                pseudopotentials=self.pseudos,
                tprnfor=True, tstress=True,  # 必須: 力と応力
                kpts=self.kpts,
                # ...
            )
            embedded_atoms.calc = calc

            try:
                # 計算実行
                embedded_atoms.get_potential_energy()
                results.append(embedded_atoms)
            except Exception:
                # 収束失敗時のリトライロジックをここに記述
                pass
        return results
C. EONWrapper (kMC Integration):EONはクライアント・サーバーモデルで動作するが、ここではローカルで eonclient を実行し、ポテンシャル評価部分のみをフックする構成とする。ディレクトリ構造:active_learning/iter_XXX/
└── kmc_run/
    ├── config.ini          # EON設定ファイル
    ├── reactant.con        # 初期構造
    ├── potential.yace      # 現在のポテンシャル
    └── potentials/         # EON用ポテンシャルドライバ
        └── pace_driver.py  # EONからPacemakerを呼ぶスクリプト
EON設定 (config.ini) 自動生成:未知の遷移状態を探索するため、NEBではなくDimer法 (process_search) を指定する。[Main]
job = process_search    # Dimer法による探索
temperature = 300.0

[Potential]
potential = script
script_path = ./potentials/pace_driver.py

[Process Search]
min_mode_method = dimer
EON実行とOTF検知ロジック:pace_driver.py 内部でエネルギー計算と同時に $\gamma$ 値をチェックし、閾値を超えたら特定の終了コード（例: 100）でEONを強制終了させる。# pace_driver.py (EONから呼び出されるスクリプト)
import sys
from pyacemaker.calculator import PaceCalculator

# EONの仕様に合わせて座標読み込み -> 計算 -> 出力を行う
# (標準入力から座標を読み込み、標準出力にエネルギーと力を返す)

calc = PaceCalculator("potential.yace")
atoms = read_coordinates_from_stdin() # 擬似コード

gamma = calc.get_extrapolation_grade(atoms)

if gamma > THRESHOLD:
    # 高い不確実性を検知 -> 構造を出力して異常終了
    write_bad_structure("bad_structure.cfg", atoms)
    sys.exit(100)  # 特別な終了コード

# エネルギーと力を標準出力へ
print(calc.get_potential_energy(atoms))
print_forces(calc.get_forces(atoms))
Orchestrator側の制御:class EONWrapper:
    def run_kmc(self, work_dir):
        # EONクライアントを実行
        proc = subprocess.run(["eonclient"], cwd=work_dir)

        if proc.returncode == 100:
            # OTFイベント発生
            return {'halted': True, 'bad_structure': work_dir / "bad_structure.cfg"}
        return {'halted': False}
D. PacemakerWrapper (Dataset & Training):PacemakerはPython APIが公開されていない機能も多いため、CLIをラップするのが確実。import subprocess

class PacemakerWrapper:
    def update_dataset(self, new_atoms_list):
        # ASE atoms -> Pacemaker pckl.gzip 形式への変換
        # pace_collect コマンドなどを利用するか、内部フォーマットに合わせて保存
        pass

    def train(self, dataset, initial_potential, output_dir):
        cmd = [
            "pace_train",
            "--dataset", str(dataset),
            "--initial_potential", str(initial_potential),
            "--max_num_epochs", "50",  # Fine-tuningなので少なく
            "--output_dir", str(output_dir)
        ]
        subprocess.run(cmd, check=True)
        return output_dir / "output_potential.yace"
3.6 品質保証・検証機能 (Validator)作成されたポテンシャルが、学習データ以外の未知のデータや、マクロな物理特性に対しても妥当な予測を行うかを検証する、システム品質の「最後の砦（Quality Assurance Gate）」である。単に数値的な誤差が小さいだけでなく、物理的に正しい振る舞いをするかを多角的に診断する。検証プロトコル (Validation Suite):学習サイクル (Refinement) が完了するたびに、以下のテストスイートを自動実行し、総合的なスコアを算出する。Test Set Evaluation (汎化性能の数値評価):分割戦略: 全データセットを Training:Validation:Test = 80:10:10 にランダム分割、あるいは時系列分割する。特に時系列分割は、類似した構造（相関の強いデータ）がテストセットに含まれるのを防ぎ、真の汎化性能を評価するために推奨される。指標: 以下の目標値をクリアしているか確認する。Energy RMSE < 1-2 meV/atomForce RMSE < 0.03-0.05 eV/ÅStress RMSE < 0.1 GPa可視化: Parity Plot（正解 vs 予測）だけでなく、誤差の累積分布関数 (CDF) をプロットし、95パーセンタイル誤差が許容範囲内かを確認する。Dynamic Stability (Phonon Dispersion):目的: ポテンシャルが「見かけ上の安定点」ではなく、真に動力学的に安定な結晶構造を再現しているか確認する。手法: 外部ツール Phonopy と連携し、有限変位法を用いてフォノン分散関係（バンド構造）を計算する。スーパーセルは、長距離相互作用を捉えるのに十分な大きさ（例: $4 \times 4 \times 4$ 以上）を自動設定する。判定: ブリルアンゾーン全域において、虚数の振動数（Imaginary Frequency, $\omega^2 < 0$）が存在しないことを必須条件とする（ただし、$\Gamma$点近傍の微小な虚数モードは並進対称性の破れによる数値誤差として許容する）。Mechanical Stability & Elasticity (弾性・機械的特性):手法: 単位胞に独立な歪み（$\pm 1\%, \pm 2\%$）を印加し、エネルギー-歪み曲線を多項式フィッティングして弾性定数テンソル ($C_{ij}$) を算出する。安定性チェック: 結晶系に応じた Bornの安定条件（例: 立方晶なら $C_{11} - C_{12} > 0, C_{44} > 0, C_{11} + 2C_{12} > 0$）を満たしているか確認する。精度評価: Voigt-Reuss-Hill平均を用いて体積弾性率 ($B$) とせん断弾性率 ($G$) を算出し、DFT参照値との誤差が許容範囲内（通常 $\pm 10-15\%$ 以内）であることを検証する。Physical Stress Tests (極限環境テスト):EOS Curve (Birch-Murnaghan Fit): 平衡体積から $\pm 20\%$ の範囲で圧縮・膨張させ、得られたエネルギー曲線をBirch-Murnaghan状態方程式にフィッティングする。滑らかさ（微分の連続性）と、$V_0, B_0, B'_0$ の物理的妥当性をチェックする。Melting Behavior: 融点付近（$T_m \pm 100K$）でのNPT-MDを行い、二相共存法またはZ-methodを用いて融点を推定し、実験値またはDFT予測値と大きく乖離していないかを確認する。Automated Gatekeeper Logic (判定とアクション):各テストの結果に基づき、システムは以下のいずれかの決定を下す。PASS (Green): 全ての基準（RMSE, Phonon stability, Born criteria）をクリア。ポテンシャルを production フォルダへデプロイし、バージョン番号を更新する。CONDITIONAL (Yellow): 数値精度は良いが、一部の物理量（例: 弾性率）の誤差がわずかに大きい。警告付きでデプロイしつつ、次回の学習で重み付け調整 (Physics-Informed Loss) を予約する。FAIL (Red): 致命的な欠陥（虚数フォノン、Born条件違反、許容を超えたRMSE）が検出された。デプロイを中止し、失敗の原因となった構造領域（例: 特定の歪み方向）を特定して、Structure Generatorに追加のサンプリング指示を出す。Reporting:検証結果、Parity Plot、フォノンバンド図、EOS曲線、弾性率比較表をまとめた validation_report.html を自動生成し、ユーザーがブラウザで視覚的に品質を確認できるようにする。
