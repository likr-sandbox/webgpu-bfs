# WebGPU Multi-Source BFS (`webgpu-bfs`) Project Knowledge Base & Agent Guide

> **Document Version**: 1.7.0  
> **Last Updated**: 2026-10-15  
> **Scope**: Architecture, Milestones M1–M26, Bottleneck Analysis, Hardware & WebGPU Constraints, Algorithmic Insights, and M26 Road Network CPU Reversal Breakthrough

---

## 1. プロジェクト概要と現時点での主要成果 (Achievements to Date)

### 1.1 開発目標と背景
本プロジェクト (`webgpu-bfs`) は、Rust と WebGPU (WGSL / `wgpu`) を基盤とし、重みなし複数始点最短経路問題（MS-BFS: Multi-Source Breadth-First Search）を GPU 上で極限まで高速化する並列グラフ探索ライブラリおよびベンチマーク環境の開発を目的としています。

グラフ探索における古典的課題である「CPU に対するレイテンシ劣後（特に高直径・狭フロンティアグラフ）」を克服するため、GraphBLAST 等の代数的グラフアルゴリズム知見、ビット並列フロンティア、2D ワープタイリング、GPU 自律反復（`dispatch_workgroups_indirect`）、次数考慮型協調ハブ展開、マルチバンク距離バッファ等の最先端アーキテクチャを体系的に導入しました。

### 1.2 達成された定量的成果 (Breakthroughs)

1. **実データ `luxembourg_osm` での定量的 CPU 逆転と実機特性 (Milestones M7–M10, M11, M19)**:
   - 高直径（$D \approx 1,200$）、疎結合（$|V|=114,599$, $|E|=239,332$）の道路網において、2D ワープタイル Pull 走査（`MultiWordBitParallel`）を確立。
   - **M8 理論ハードウェアターゲット**: 理論 VRAM 帯域（168 GB/s）および完全合体アクセスに基づくモデル値として **19,504 sources/sec**（105 ms @ $S=2048$、CPU Rayon 比 $24.56\times$、CPU 1-Thread 比 $42.94\times$）を算出。
   - **M19 以降の実機実測値 (WSL2 D3D12 / RTX 3050 - 水平展開最適化)**:
     - $S=64$: **約 211〜221 ms**（CPU 単一スレッド 137〜610 ms に対し最大 $2.67\times$ 高速化、$\le 233\text{ ms}$ 目標達成）。
     - $S=512$: **約 684 ms**（従来の 8 パス直列から 1 パス 512 始点同時実行へ短縮）。
     - $S=2048$: **約 2,686〜2,701 ms**（従来の 32 パス直列・6,389 ms から **$2.38\times$ 高速化**、パス数を 4 回へ $8\times$ 削減、CPU 単一スレッド FIFO: 3,684 ms を逆転）。
2. **無向べき乗則グラフ `web-Stanford` での大規模バッチ定量的 CPU 逆転 (Milestones M15–M18)**:
   - $|V|=281,903$, $|E|=3,985,272$、最大次数 38,625 のハブノードが存在する無向べき乗則グラフにおいて、次世代の「8 大アーキテクチャピラー」（次数考慮型協調ハブ展開、512/256-bit マルチワード Push、マルチバンク距離ストレージバッファ、LDS キュー 1,024 エントリ化、永続バッファプール、単一エンコーダ GPU 内 `u16` 量子化、ホスト並列アンパック、トポロジー適応型ルーター `Auto`）により、大規模バッチ $S \in \{512, 1024, 2048\}$ において CPU Rayon（マルチスレッド）および CPU 単一スレッドを完全逆転。
   - **$S=512$**: GPU 実行時間 **469.91 ms**（最速）/ 497.84 ms（平均）、スループット **3,802 MTEPS**（1,090 sources/s）。CPU Multi-Thread (Rayon: 919.33 ms) に対し **$1.85\times \sim 2.20\times$ 高速化**、CPU Single-Thread (FIFO: 3,642.36 ms) に対し **$7.32\times \sim 7.98\times$ 高速化**。
   - **$S=1024$**: GPU 実行時間 **1,005.43 ms**（3,526 MTEPS、1,018 sources/s）。CPU Rayon (1,799.15 ms) に対し **$1.79\times$ 高速化**、CPU Single-Thread (7,310.65 ms) に対し **$7.27\times$ 高速化**。
   - **$S=2048$**: GPU 実行時間 **1,994.07 ms**（3,572 MTEPS、1,027 sources/s）。CPU Rayon (3,733.54 ms) に対し **$1.87\times$ 高速化**、CPU Single-Thread (14,211.33 ms) に対し **$7.13\times$ 高速化**。
3. **無向べき乗則グラフでの協調ハブ展開・LDS 4 分割・Ultra-Hub 最適化による新境地 (Milestone M20)**:
   - 次数 38,625 の超巨大ハブと大規模バッチにおける SM 内レイテンシ・バンク競合を打破するため、「第 3 世代協調アーキテクチャ」を導入:
     * **Phase 0 サブワープ協調ロード**: 1 レーンあたり 4 ワード分散ロードにより、直列不可分交換遅延を $4\times$ 削減。
     * **LDS キュー容量の 2,048 エントリ化**: 共有メモリ容量を 9,460 バイト（WebGPU 16 KB 制限内）まで活用し、VRAM スピルオーバー直列化を 90% 以上排除。
     * **LDS 4 サブクアドラント・アトミックカウンタ**: `s_sub_queue_count[qid]`（`qid = local_id.x & 3u`）により 64 スレッドのアトミックバンク競合を 4 バンクへ分散。
     * **2 階層ウルトラハブ展開 (Stride 256 アンロール)**: 次数 $\ge 1,024$ に対し 4 倍アンロールを展開、次数 38,625 の外側ループ回数を 604 回から 151 回へ激減。
     * **距離バンク判定の外出しホイスティング**: ビット走査ループ内の 8 ウェイ switch を外部へ巻き上げ、密なインラインストア命令列を生成。
   - **定量的ベンチマーク実測値 (WSL2 D3D12 / RTX 3050 - M20 最適化適用)**:
     - **$S=512$**: GPU 実行時間 **426.38 ms**（平均）/ **385.25 ms**（次数順）、スループット **4,272〜5,091 Harmonic MTEPS**（1,201〜1,329 sources/s）。CPU Rayon (905.16〜3,188.49 ms) に対し **$2.12\times \sim 8.27\times$ 高速化**。
     - **$S=1024$**: GPU 実行時間 **915.70 ms**（平均）/ **765.29 ms**（次数順）、スループット **3,923〜5,065 Harmonic MTEPS**（1,118〜1,338 sources/s）。CPU Rayon (1,974.96〜2,343.63 ms) に対し **$2.16\times$ 高速化**。
     - **$S=2048$**: GPU 実行時間 **1,814.86 ms**（平均）/ **1,518.74 ms**（次数順）、スループット **3,993〜5,092 Harmonic MTEPS**（1,128〜1,348 sources/s）。CPU Rayon (3,906.34〜4,755.33 ms) に対し **$2.15\times \sim 3.13\times$ 高速化**。
     - **全領域非退行 (Non-Regression)**: `web-Stanford` $S=64$ で **111.53 ms**（ベースライン ~130 ms から短縮）、`luxembourg_osm` $S=64$ で **211.61 ms**（$\le 233\text{ ms}$ 目標達成）。
4. **自動ルーター検証・ワークスペース完全パリティ・ドキュメンテーション統合 (Milestone M21)**:
   - `select_auto_method` により、グラフの次数特性（道路網 $d_{\max} \le 32$、べき乗則 $d_{\max} \ge 128$、中間グラフ）とバッチサイズ（$S=1$、小中バッチ $1 < S < 64$、大規模バッチ $S \ge 64$）に応じた決定論的最適ルーティングを確立・網羅的テスト。
   - 全ワークスペース（ユニット、結合、Adversarial、Doc テスト）において 0 失敗・100.0% ビット完全性（Parity ✅）を維持。
5. **100% ビット単位パリティ検証 (Bitwise Correctness)**:
   - 全 14 の GPU 手法・メタ手法（`Auto` 含む）と CPU リファレンス（Rayon / FIFO）の間で、数億ホップ計算にわたる最短距離配列のビット単位完全一致（Parity ✅ 100.0%）を機械的テストスイートで保証。
6. **全グラフ無向化ポリシーの標準適用**:
   - 入力されたグラフが有向グラフであっても自動的に対称化（無向化）して走査を実施。標準バッチサイズは $S \in \{1, 16, 64, 256, 1024, 2048\}$ に統一。
7. **Sparse Push 端数バッチ 1 パス化・極小バッチリセット最適化・第 3 世代協調ハブ展開の全面水平展開 (Milestone M22)**:
   - 従来マルチパス分割直列化（2 パス実行）されていた端数バッチ（$S=128$, $S=384$）および極小バッチ（$S \le 32$）における構造的遅延を打破し、M20 の第 3 世代協調アーキテクチャを全 Push シェーダーへ水平展開:
     * **端数バッチ 1 パス統合ルーティング**: 分岐閾値を上限基準（$256 < S \le 512 \to \text{pool\_512}$, $64 < S \le 256 \to \text{pool\_256}$）へ改修。Phase 0 に `w < uniforms.num_words` ガードを導入し、非アクティブワードの不可分交換をスキップ。最終バンクの `u16` 量子化・DMA リードバックもアクティブ始点分（`0..active_banks`）に厳格に制限。
     * **実要素数バウンド距離リセット**: 距離配列の初期化を固定最大容量（$64 \times |V| = 72.17\text{ MB}$）から厳密な実バッチ要素数（$S \times |V|$）へ縮小。$S=1$ では 1.13 MB のみのリセットとなり、初期化オーバーヘッドを $64\times$ 削減。さらに $S \le 32$ では `num_words = 1` を適用。
     * **細粒度並列 CPU アンパック**: `std::thread::scope` による 2 バイト境界（ワード整合）分割を用いて、小バッチでも全 CPU 論理コアを活用した並列アンパックを展開。
     * **第 3 世代 M20 最適化の全面水平展開**: `indirect_sparse_push_256.wgsl` および `indirect_sparse_push.wgsl` に対し、Phase 0 サブワープ協調ロード、2,048 エントリ LDS キュー（WebGPU 16KB 制約内）、4 サブクアドラント・アトミックカウンタ（`s_sub_queue_count[4]`）、Stride 256 ウルトラハブ展開（次数 $\ge 1,024$）、ホイスティングされた距離書き込み関数を全面移植。
     * **トポロジー適応型 `Auto` ルーター刷新**: べき乗則グラフ（$d_{\max} \ge 128$）において、$S \ge 1$ の全バッチサイズを永続バッファプール対応の `IndirectSparseQueuePush` へルーティング。
   - **定量的ベンチマーク実測値 (WSL2 D3D12 / RTX 3050 - M22 最適化適用)**:
     - **単一始点 ($S=1$)**: 実行時間 **25.25〜36.99 ms**（従来の未プール ~180 ms から **$\sim 5\times$ 高速化**）。
     - **小規模バッチ ($S=16$)**: 実行時間 **58.19〜68.39 ms**（従来 ~210 ms から大幅短縮）。CPU Rayon (76.37〜117.92 ms) に対し **$1.26\times \sim 1.72\times$ 高速化**、CPU Single-Thread FIFO (235.19 ms) に対し **$3.44\times \sim 3.89\times$ 高速化**。
     - **端数バッチの厳格な単調性確保**:
       * $S=128$: **165.62〜242.97 ms**（$S=256$: 290.46〜388.85 ms に対し **$1.60\times \sim 2.31\times$ 高速化**、$T(128) < T(256)$ 完全成立）。
       * $S=384$: **381.12〜733.41 ms**（$S=512$: 455.10〜1,379.13 ms に対し **$1.28\times \sim 1.88\times$ 高速化**、$T(384) < T(512)$ 完全成立）。
     - **全領域完全パリティ**: $S \in \{1, 16, 64, 128, 256, 384, 512, 1024, 2048\}$ および境界 $S \in \{33, 65, 257, 513\}$ において 100.0% ビット完全性（Parity ✅）を達成。
8. **高直径道路網スケーリング・Pull カーネルバリア全廃・Dirty-Check 条件付きゼロクリア・重複バッファ排除 (Milestone M23)**:
   - 高直径道路網（`luxembourg_osm`, $D \approx 1,311$）および長経路探索における WSL2 `/dev/dxg` ホスト同期遅延と DRAM トラフィックを劇的に削減:
     * **Pull カーネルのツリーリダクションバリア全廃 & LDS 0 バイト化**: `multi_word_bp_pull_512.wgsl` において、8 段階のツリーリダクション `workgroupBarrier()` ループを完全撤廃。終了判定が真偽値（`count == 0`）のみを参照する特性を利用し、`if (thread_new_count > 0u) { atomicStore(&active_count, 1u); }` による直接アトミックストアへ置換。1 探索あたり 1 億 3,750 万回以上のワークグループバリアを完全消去し、共有メモリ消費量を 0 バイトへ削減。
     * **Dirty-Check 付き条件付きフロンティアゼロクリア**: Ping-Pong フロンティアバッファにおいて、無条件クリア `next_frontier[state_idx] = 0u` を `if (next_frontier[state_idx] != 0u) { next_frontier[state_idx] = 0u; }` へ置換。Ping-Pong の数学的不変条件を 100% 保持したまま、99.92% の非アクティブ頂点に対する無駄な DRAM 書き込みをスキップし、パスあたり約 9.6 GB のメモリ書き込みトラフィックを排除。
     * **非アクティブ始点ワードの早期スキップ & バンク判定ホイスティング**: `src_start >= params.total_sources` による早期 `continue` ガードを追加し、未使用始点スロット（$S=64$ で 87.5%、$S=256$ で 50%）の探索・マスク計算をバイパス。さらにビット走査ループ外へ 8 ウェイ switch をホイスティング。
     * **適応的マクロバッチ拡大 (3 階層: $160 \to 320 \to 512$ ホップ)**: `indirect_sparse_push.rs` において、`level > 0` のチャンクホップ幅を固定 64 からレベル進捗に応じて適応的に 320 / 512 ホップへ拡大。Luxembourg OSM における `/dev/dxg` ホスト同期ラウンドトリップ回数を 19 回から 4 回へ激減。
     * **BFS 単調性を利用した最終ステップ単一カウンタ評価**: 探索フロンティアの単調縮小性（$F_h = \emptyset \implies F_{h+1} = \emptyset$）に基づき、各マクロチャンク内の全 64 ホップのカウンタコピーを廃止し、最終ホップ `counter_bufs[last_k]` のみ（4 バイト）をリードバック。ドライバコピー回数を $64\times$ 削減。
     * **`Arc<wgpu::Buffer>` による対称グラフバッファ重複排除**: `PushPullGraphBuffers` において CSC オフセットおよび行インデックスを CSR の `Arc<Buffer>` へエイリアスし、無向グラフにおける VRAM トポロジーメモリ消費および PCIe アップロード量を半減。さらに `from_symmetric_graph` によりホスト側の `is_symmetric()` 転置チェックを完全バイパス。
   - **定量的ベンチマーク実測値 (WSL2 D3D12 / RTX 3050 - M23 最適化適用)**:
     - **道路網大規模バッチ `luxembourg_osm` ($S=2048$)**: 実行時間 **1,765.48〜2,054.96 ms**（平均 2,054 ms / 最速 1,765 ms）。従来の M19 ベースライン（2,686 ms）から **約 631〜920 ms 短縮（$1.31\times \sim 1.52\times$ 高速化）**、CPU 単一スレッド (3,684 ms) を大きく引き離す。
     - **道路網中規模バッチ `luxembourg_osm` ($S=64$)**: 実行時間 **211.62〜230.05 ms**（最速 205.95 ms）。$\le 233\text{ ms}$ の目標を安定してクリア。
     - **無向べき乗則グラフ `web-Stanford` ($S=512$)**: 実行時間 **448.81〜593.16 ms**、スループット **2,946〜4,024 Harmonic MTEPS**（1,141 sources/s）。CPU Rayon に対する優位性を完全維持。
     - **全ワークスペース 100% ビット完全性**: 単体・統合・Adversarial 全テストスイートで 0 失敗、全バッチサイズ・全トポロジーで Parity ✅ 100.0% を達成。
9. **非同期マルチバッファ・パイプライン重複 & u8/u16 適応的量子化 & 2D グリッドディスパッチ (Milestone M24)**:
   - 複数チャンク探索における GPU コンピュートと CPU DMA リードバック／アンパック処理のダブルバッファリング重複実行（チャンク $k+1$ の GPU 実行中にチャンク $k$ の CPU アンパックを並行処理）。
   - 適応的チャンク幅スケーリング（バッチ規模とグラフ特性に応じて $64 \to 128 \to 256$ ホップ）。
   - スモールワールド網（$D \le 254$）における適応的 `u8` 距離量子化（`pack_distances_u32_to_u8.wgsl`）により、PCIe DMA 転送量を 288 MB から 144 MB へ半減（$2\times$ 圧縮）。
   - `prep_sparse_push_hop.wgsl` の 2D 間接グリッド（`workgroups_x = 256`, `workgroups_y = (total_wgs + 255) / 256`）対応により、100 万頂点超のアクティブキューに対する WebGPU 65,535 ワークグループ制限を完全突破。
10. **GPU 自律型 Direction-Optimizing ハイブリッド走査 & 到達可能性ビットマスク API & 動的バンク幅スケーリング (Milestone M25)**:
   - `prep_hybrid_hop.wgsl`, `indirect_pull_512.wgsl`, `clear_frontier_512.wgsl` によるホスト介在ゼロの GPU 完全自律 Push/Pull ハイブリッド探索。入次数およびフロンティア密度閾値（$in\_count > |V| / 16$）に基づき、ホップ単位で動的に Push と 8 バンク Pull を切り替え。
   - `ReachabilityBitmask` および `ReachabilityEngine` トレイトによる 1 bit/source/vertex 直接到達可能性評価 API。8 バンク距離配列アロケーションおよび u16 量子化をバイパスし、`visited_buf` を直接 DMA リードバックすることで、メモリ帯域・VRAM 消費を大幅削減。
   - $|V| > 524,288$ の超大規模グラフにおける動的バンク幅スケーリング（$64 \to 32 \to 16$ 始点/バンク）により、WebGPU 128 MB ストレージバッファ上限を数学的・決定論的に順守。
   - CLI デフォルト手法の `Auto` 化、`--reachability` フラグの統合（verify / bench）、WASM `run_gpu_reachability` / `verify_reachability` エクスポート、Web UI への `Auto` および `IndirectSparseQueuePush` 反映。
   - `web-Stanford` および `luxembourg_osm` を含む全トポロジー・全バッチサイズ（$S \in \{1, 16, 64, 128, 256, 384, 512, 1024, 2048\}$）における 100.0% ビット完全性（Parity ✅）を達成。
11. **WorkgroupPersistentBfs による道路網（`luxembourg_osm`）での定量的 CPU 完全逆転 (Milestone M26)**:
   - 高直径・均一低次数グラフ（`luxembourg_osm`, $D \approx 1,311$, $|V|=114,599$, $d_{\max}=7$）における長年の課題であった WSL2 `/dev/dxg` ホスト同期遅延および直列分割オーバーヘッドを、「1 ワークグループ = 1 始点」「単一コンピュートパス内 1,311 ホップ全走査」により完全打破:
     * **単一コンピュートディスパッチでの全ホップ自律走査**: `workgroup_persistent_bfs.wgsl` において、各 128 スレッドのワークグループが 1 つの始点を担当し、フロンティアが空になるまでホスト・デバイス間同期 0 回、間接ディスパッチオーバーヘッド 0 回で全 1,311 ホップを自律反復走査。
     * **100% オンチップ SRAM 格納 (WebGPU 16,384 バイト制限厳格順守)**: 114,599 頂点の訪問フラグビットマスク `s_visited: array<atomic<u32>, 3584>`（14,336 バイト）+ 508 要素フロンティアキュー `s_queue: array<u32, 508>`（2,032 バイト）+ 制御変数（16 バイト）= 合計 16,384 バイトを共有メモリ（LDS）内に完全収容。スレッドレジスタバッファリングにより第 2 Ping-Pong キューを排し、VRAM ストレージバリアを完全撤廃。頂点数 $> 114,688$ のグラフ向けには VRAM 訪問バッファへのフォールバックを自動提供。
     * **反転ビット `atomic<u32>` 半ワード直接距離エンコーディング**: 512 始点分の距離を 117.35 MB の単一ストレージバッファ（WebGPU 128 MB 制限内）にハードウェアクリア（`encoder.clear_buffer`）し、`inv_d = 0xFFFFu - min(level, 0xFFFFu)` として発見時に `atomicOr(&packed_dist[s_word_base + (w >> 1u)], inv_d << ((w & 1u) * 16u))` で直接記録。従来の 8 バンクリセットパス・8 パス量子化・8 ウェイ switch を完全排除。
     * **ハードウェアキューオーバーフロー検知 & 透明なフォールバック**: `@binding(7) overflow_buf` により、万一キュー容量（508 要素）を超える波動フロントが発生した場合にシェーダーが `s_overflow = 1u` を記録し、ホストディスパッチャが `MultiWordBitParallel` へ透明に自動フォールバック。あらゆるグラフトポロジー（2D グリッド $300 \times 300$, Erdős-Rényi 等）で 100.0% のビット完全性を数学的に保証。
     * **空間的始点ソート (Spatial Source Sorting)**: 始点インデックスを頂点 ID 順に事前ソートし、GPU L2 キャッシュ（1.5 MiB）内における CSR 出エッジリストのヒット率を最大化。
     * **ダブルバッファリング 2 段階非同期パイプライン**: $S > 512$ の複数チャンク実行において、Chunk $i$ の GPU 探索コンピュートと Chunk $i-1$ のホスト CPU 並列アンパック（`if raw == 0 { u32::MAX } else { 0xFFFF - raw }`）を完全にオーバーラップ。
     * **SRAM から VRAM への到達可能性協調フラッシュ**: 到達可能性モードでは `packed_dist` の書き込みをバイパスし、探索終了時に `s_visited`（14.3 KB/始点）を VRAM へ直接協調フラッシュすることで PCIe 転送量を $16\times$ 削減（$S=2048$ で 29.34 MB）。
   - **定量的ベンチマーク実測値 (WSL2 D3D12 / RTX 3050 - M26 実測)**:
     - **$S=16$**: GPU 実行時間 **9.28〜9.91 ms**。CPU Rayon (12.39〜21.13 ms) に対し **$1.33\times \sim 2.13\times$ 高速化**。
     - **$S=64$**: GPU 実行時間 **20〜24 ms**（最速 22.13 ms）、到達可能性モード **15〜19 ms**（最速 15.94 ms）。従来の M23 ベースライン（211 ms）から **最大 $10\times$ 高速化**、CPU Rayon (28.43〜31.83 ms) を逆転。
     - **$S=512$**: GPU 実行時間 **186.06〜213.58 ms**。CPU Rayon (258.73〜342.87 ms) に対し **$1.39\times \sim 1.61\times$ 高速化**。
     - **$S=2048$ (距離探索)**: GPU 実行時間 **683.98〜736 ms**（最速 683.98 ms、スループット 2,843 sources/s）。CPU Rayon (1,039.58〜1,155.31 ms) に対し **$1.52\times \sim 1.58\times$ 高速化**、従来の M23 ベースライン（1,765 ms）から約 1,000 ms 短縮。
     - **$S=2048$ (到達可能性探索)**: GPU 実行時間 **178.51〜205.84 ms**（最速 178.51 ms、スループット **11,473 sources/s**）。CPU Rayon (1,851.50 ms) に対し **$9.0\times$ 高速化**、M25 ベースライン（4,971 ms）に対し **$32.8\times$ 劇的高速化**。
     - **全領域完全パリティ**: $S \in \{1, 16, 33, 64, 65, 512, 513, 2048\}$、境界・非標準シード（42, 9999, 123456）、および `web-Stanford` を含む全構成で 100.0% ビット完全性（Parity ✅）を達成。


---

## 2. 実装済みアルゴリズム一覧 (Implemented Methods)

| 手法 ID / コマンド名 | 分類 | 探索方向 | コア技術・特徴 | 最適適用シナリオ |
| :--- | :--- | :---: | :--- | :--- |
| `CpuReference` | CPU | Push | Rayon による始点並列 CPU マルチスレッド探索 | 汎用 CPU 比較基準 (Ground Truth) |
| `CpuSingleThread` | CPU | Push | 古典的 FIFO キューによる CPU 1 スレッド BFS | 単一スレッド基準、極小バッチ |
| `NaiveSequential` | GPU | Push | 単一始点 SpMV を始点数 $S$ 回直列実行 | アルゴリズム比較のベースライン |
| `BatchedDenseSpMM` | GPU | Push | フロンティアを行列化し、頂点×始点の SpMM 実行 | 中密度グラフ、小〜中バッチ |
| `BitParallel32` | GPU | Push | 32 始点を 1 ワード (`u32`) にパッキングしたビット演算 | 中規模バッチの基礎・密フロンティア |
| `BitParallel64` | GPU | Push | 64 始点を 2 ワード (`vec2<u32>`) にパッキングしたビット並列 | 中規模密 Push 探索 |
| `BitParallel64Pull` | GPU | Pull | CSC 逆辺を用いた 64 本 Pull 走査（アトミック競合ゼロ） | 道路網・均一低次数グラフ（$S \le 2048$） |
| `BooleanReachability`| GPU | Push | 到達可能性ブーリアンベクトル更新＋距離更新 | 論理到達判定と距離更新の分離 |
| `PushPullAdaptive` | GPU | Hybrid | フロンティア密度（閾値 5%）に基づく動的 Push/Pull 切り替え | スケールフリー網・高密度グラフ |
| `SparseQueue` | GPU | Push | Prefix Sum / キュー集約によるアクティブ頂点のみの走査 | **単一始点 ($S=1$) スパース走査** |
| `SparseBitParallel64`| GPU | Push | 64 本ビット並列＋動的キュー集約フィルタリング | 小〜中規模バッチ ($S \le 16$) で強力 |
| `MultiWordBitParallel`| GPU | Pull | 512 始点 8 バンク Pull 走査（バリア 0 回・LDS 0B）、Dirty-Check Ping-Pong、最終単一カウンタ評価、GPU 内 u16 量子化、単一エンコーダ DMA、永続バッファプール、S<=64 高速パス | **無向高直径網（道路網）の大規模バッチ ($S \le 2048$) で最速・S=2048で1,765〜2,054 ms (1.52x高速化)** |
| `IndirectMacroBatch` | GPU | Pull | `dispatch_workgroups_indirect` による GPU 内完結反復 | ホスト・デバイス間通信の極小化 |
| `IndirectSparseQueuePush`| GPU | Push | 次数考慮型協調ハブ展開（2階層stride 64/256）＋Phase 0分散ロード＋LDS 2048キュー（4クアドラント・アトミック）＋512/256/64-bitマルチワード＋端数バッチ1パス統合＋実要素数リセット＋GPU内u16量子化 | **無向べき乗則グラフ（Web・ソーシャル）の全バッチ ($S \le 2048$) で最速・S=1..16でもCPU Rayon逆転・CPU比最大8.27x (5,091 MTEPS)** |
| `MultiWordBitParallelPush`| GPU | Push | 動的ストライドループによる複数ワード ($S \le 2048$) Push 型ビット並列 | 中〜大規模バッチ Push 探索 |
| `WorkgroupPersistentBfs`| GPU | Push (Autonomous) | 1 ワークグループ=1 始点自律 BFS、100% オンチップ SRAM 格納（16KB 制限内: 14.3KB visited + 2KB queue）、反転ビット半ワード直接距離書込、キュー溢れハードウェア検知 & `MultiWordBitParallel` 自動フォールバック、空間ソート、2 段階非同期パイプライン | **無向高直径・道路網（`luxembourg_osm`）の全バッチ ($S \ge 16$) で CPU Rayon 完全逆転、S=64 で 20〜24 ms（10x高速化）、S=2048 到達可能性で 178 ms (11,473 sources/s, CPU比9.0x)** |
| `Auto` | GPU (Meta) | Adaptive | グラフの最大次数とバッチサイズに基づくトポロジー適応型自動ルーティング（低次数道路網 $d_{\max} \le 8, |E| \le 4|V|$: WorkgroupPersistentBfs 全バッチ S>=1、べき乗則ハブ網: IndirectSparseQueuePush 全バッチ S>=1、中間グラフ小中バッチ: BitParallel64、単一始点: SparseQueue） | **プロダクション推奨デフォルト・全トポロジー自動最適化・全領域非退行** |

---

## 3. プロジェクトで得られた重要な技術的知見 (Learnings & Insights)

### 3.1 グラフ形状と探索方向（Push vs Pull）の適合性
- **道路網・高直径グラフ (`luxembourg_osm`, $D \approx 1,200$)**:
  - フロンティアが常に狭く長く伸びるため、GPU スレッドの過半数が空振りする問題が生じる。
  - 大規模マルチソース（$S \ge 64$）では、**Pull 走査（`BitParallel64Pull` / `MultiWordBitParallel`）**が極めて有利になる。
  - 理由: 全頂点の次数が $\le 7$ と均一で、Pull 方式では各頂点が入エッジを集約して自身の距離・フロンティアを単一スレッドで決定的に書き込むため、**アトミック競合が完全にゼロ**であり、合体メモリアクセスが最大効率で働く。
- **無向べき乗則グラフ (`web-Stanford`, $D \approx 160$)**:
  - 最大次数 38,625 の超巨大ハブノードが存在する場合、Pull 走査はワープダイバージェンスにより著しく破綻（実行時間 4.5〜5.5 秒）する。
  - また、ナイーブなサブワープ個別走査（stride 4）では、ハブノード処理時に 1 サブワープあたり 9,656 回の直列ループが発生し、ワークグループ内の他スレッドが待たされ深刻なワープストールを引き起こしていた。
  - **解決策 (Milestones M16–M18)**:
    - **次数考慮型協調ハブ展開（Workgroup Collaborative Expansion）**: 次数 $\ge 128$ のハブノードに対し、ワークグループ全体（64 スレッド）が協調して stride 64 で出エッジを走査。ループ回数を 150 回へ激減（$64\times$ 圧縮）させ、ワープストールを完全解消。
    - **512 ビット / 256 ビット マルチワード化**: 探索幅を 512 始点まで拡大し、$S=2048$ における走査パス数を 32 回から 4 回へ削減。
    - これにより、大規模バッチ $S \in \{512, 1024, 2048\}$ において GPU 実行時間が CPU Rayon に対し **$1.79\times \sim 2.20\times$**、CPU シングルスレッドに対し最大 **$7.98\times$ 高速化** を達成。

### 3.2 WebGPU / WGSL のハードウェア・API 制約と対処パターン
WebGPU は高いポータビリティを持つ反面、ネイティブ CUDA 等に比べて厳格な制約が存在します。本プロジェクトで確立した回避パターンは以下の通りです。

1. **`max_compute_workgroups_per_dimension = 65535` の壁**:
   - **事象**: 頂点数 $|V| > 65,535 \times W_v$ のグラフにおいて、1D ディスパッチを行うと WebGPU バリデーションエラーでパニックする。
   - **解決策 (2D Grid Dispatch)**:
     $$\text{dispatch\_x} = 256, \quad \text{dispatch\_y} = \left\lceil \frac{\text{total\_wgs}}{256} \right\rceil$$
     シェーダー内で `let wg_id = group_id.x + group_id.y * params.dispatch_x;`、`let v = wg_id * W_v + local_id.y;` として頂点インデックスを復元。
2. **単一ストレージバッファ上限 (`max_storage_buffer_binding_size = 128 MB`) に対するマルチバンク分割 (Milestone M17)**:
   - **事象**: WebGPU の標準仕様では 1 ストレージバッファあたり 128 MB まで。頂点数 $|V|=281,903$ において $S=512$ の距離配列（$512 \times 281,903 \times 4\text{ B} = 577.34\text{ MB}$）を単一バッファでバインドしようとすると、デバイス検証エラーでパニックする。
   - **解決策 (マルチバンクストライピング)**:
     距離出力を 64 始点ごとの独立したストレージバッファ（各 $72.17\text{ MB} \le 128\text{ MB}$）8 バンクに分割してバインドグループへ割り当て。シェーダー内で `let bank = w >> 1u;` によりバンクを切り替えてスカラー書き込みを行うことで、128 MB 制約を厳格に順守しつつ、512 始点の並列ディスパッチを 1 パスで同時実行可能にした。
3. **ストレージバッファスロット上限 (`max_storage_buffers_per_shader_stage = 16`) の活用とフォールバック階層**:
   - D3D12/Vulkan の Elevated Limits で得られる 16 スロットを活用し、512 バッチシェーダーで 15 バッファ（制御 7 + バンク 8）を単一パイプラインにバインド。制限が 8 スロットの環境では 256 バッチ（4 バンク）または 64 バッチへ自動フォールバックする安全性を担保。
4. **WSL2 Direct3D 12 ハードウェア GPU の直接活性化 (Milestone M11)**:
   - **事象**: WSL2 Linux 上では Linux DRM ノードが存在せず、wgpu がデフォルトでソフトウェアエミュレーション（Mesa `llvmpipe`）を選択し、RTX 3050 が使われない。
   - **解決策**: `/dev/dxg` を検出し、`GALLIUM_DRIVER=d3d12` および `MESA_LOADER_DRIVER_OVERRIDE=d3d12` を自動構成。アダプタースコアリングで `D3D12 (NVIDIA GeForce RTX 3050)` に +7,050 点を与えて強制選択。
   - **ドライバ安定性**: プロセス終了時のドライバ破棄 SIGSEGV を `INSTANCE_RETAINER` で抑止。Naga GLSL 変換エラーを `atomicMin` によるスカラー代入で解消。

### 3.3 有向べき乗則グラフで CPU 逆転を達成した「6 大アーキテクチャピラー」 (Milestone M14)
1. **サブワープ協調エッジ展開 (`indirect_sparse_push.wgsl`)**:
   - 頂点あたり 4 スレッドが協調して出エッジをストライド 4 で走査し、**100% 合体 16 バイトロード** を達成。
2. **共有メモリ集約型アクティブキュー**:
   - Workgroup Local Data Share（LDS）内に 256 要素のキューバッファを設け、グローバルアトミック競合を **$128\times$ 削減**。
3. **GPU 自律適応型マクロバッチ（160 ホップ）**:
   - `web-Stanford` の全探索径を単一のコマンドバッファにバッチ記録し、ホスト・デバイス同期のディスパッチ遅延を極小化。
4. **GPU 内ロスレス `u16` 量子化 (`pack_distances_u32_to_u16.wgsl`)**:
   - 探索完了後に GPU 上で距離配列を `u16` へ圧縮（0.38 ms）。PCIe DMA 転送量を **72 MB から 36 MB へ半減** させ、ホストリードバック時間を 40 ms $\to$ 18 ms へ短縮（到達不能マーカー `u32::MAX` を 100% 忠実に再現）。
5. **永続的バッファプーリング (`PersistentSparsePushBuffers`)**:
   - 反復間の作業バッファとバインドグループをキャッシュし、実行ごとのバッファ割り当て遅延を **34 ms $\to$ 160 ns** へ削減。
6. **ホストページフォルト事前フォールティング**:
   - メモリスライスを事前にフォールトさせることで、実行ごとの 18,432 回の Linux ページフォルト遅延（約 26 ms）を完全排除。

### 3.4 無向べき乗則グラフ・大規模バッチ ($S \le 2048$) で CPU を圧倒する「第 2 世代高速化ピラー」 (Milestones M15–M18)
1. **次数考慮型協調ハブ展開 (Degree-Aware Collaborative Hub Expansion, `indirect_sparse_push_512.wgsl`)**:
   - 次数 $\ge 128$（最大 38,625）のハブノードに対し、サブワープ単独走査（stride 4, 9,656 反復）からワークグループ全体協調走査（stride 64, 150 反復）へ切り替え。ワープストールを完全解消。
2. **512 ビット / 256 ビット マルチワード・スパース Push (Multi-Word Bit-Parallel Push)**:
   - 探索フロンティア幅を 64 本から 256 本（8 ワード）/ 512 本（16 ワード）へ拡張。$S=2048$ における走査パス数を 32 回から 4 回へ激減（$8\times$ 削減）。
3. **マルチバンク距離ストレージバッファ (Multi-Bank Distance Buffers)**:
   - 128 MB バインディング制限を回避するため、距離出力を 64 始点ごとの 8 バンク（各 72.17 MB）にストライピング。
4. **LDS 共有メモリキューの $4\times$ 拡大 (1,024 エントリ)**:
   - Workgroup Local Data Share（LDS）内のキュー容量を 256 から 1,024 へ拡大。グローバルアトミック競合を最大 $1,024\times$ 削減。
5. **単一コマンドエンコーダでの GPU 内一括 `u16` 量子化 & DMA 統合**:
   - 8 バンクの `u16` パッキングコンピュートパスと単一ステージングバッファへの DMA コピーを 1 本の `CommandEncoder` に融合。ホストとの同期（`map_async`）を 1 チャンクあたり 1 回のみに集約。PCIe 転送量を 577 MB から 288 MB へ半減。
6. **ホスト並列アンパッキング (`std::thread::scope`)**:
   - 取得した `u16` ステージングデータを Rayon / スコープドスレッドにより呼び出し元バッファへ並列展開。ホスト側アンパック処理のレイテンシを極小化。
7. **トポロジー適応型自動ルーター (`Auto` / `select_auto_method`)**:
   - グラフの最大次数とバッチサイズから、道路網型（MultiWordBitParallel）、ハブ型べき乗則（IndirectSparseQueuePush）、極小バッチ（SparseQueue / BitParallel64）を決定的に自動選択。
8. **大規模バッチ永続バッファプール (`PersistentSparsePushBuffers256/512`)**:
   - 512/256 バッチ用の作業バッファ、バンクバッファ、バインドグループをキャッシュし、実行ごとのアロケーションコストを完全排除。

### 3.5 高直径グラフにおける WSL2 D3D12 実機レイテンシの構造と教訓
- **ハイパーバイザ提出遅延の累積**:
  - 直径 $D \approx 1,311$ のグラフ（`luxembourg_osm` 等）では、投機的マクロバッチング（64 ホップ）を行っても約 21 回の提出・同期が発生する。
  - WSL2 の `/dev/dxg` 経由では 1 回の往復同期に 2〜4 ms を要するため、ホップループのドライバ通信だけで 1 パスあたり約 160 ms を消費する。
- **バッファ制限による直列パスの増殖**:
  - $S=2048$ の出力距離配列（938.8 MB）が単一ストレージバッファ上限（128 MB）を超えるため、従来は 32 チャンクに分割直列実行され、トータル時間が約 6.4 秒に達していた。
  - **M19 によるブレークスルー**: マルチバンク距離バッファ（8 バンク分割、各 64 始点）と単一エンコーダ GPU 内 `u16` 量子化を Pull 走査に水平展開し、走査パス数を 32 回から 4 回へ削減（$8\times$ 圧縮）。実行時間を **2,686 ms（$2.38\times$ 高速化）** へ短縮し、CPU 単一スレッド（3,684 ms）を逆転。

### 3.6 道路網大規模バッチを革新したマルチバンク Pull 走査 (Milestone M19)
1. **8 バンク分散型 512 始点 Pull シェーダー (`multi_word_bp_pull_512.wgsl`)**:
   - ワークグループ構成 $(W_k=16, W_v=8, 1) = 128$ スレッド。16 ワード（512 始点）を 8 頂点で協調走査。
   - 距離バッファを 64 始点ごとの 8 バンク（`dist_bank0`〜`dist_bank7`）に分割し、WebGPU 128 MB バインディング制限をクリア。
   - スカラーバンク書き込み `bank = k >> 1u; local_s = (k & 1u) * 32u + bit_pos;` により、アトミック競合ゼロで直接決定的に書き込み。
2. **単一コマンドエンコーダでの GPU 内一括 `u16` 量子化 & DMA 統合**:
   - 8 バンクの `u16` パッキングコンピュートパスと単一ステージングバッファへの DMA コピーを 1 本の `CommandEncoder` に融合。ホストとの同期（`map_async`）を 1 チャンクあたり 1 回のみに集約。
3. **永続的バッファプーリング (`PersistentPullBuffers512`)**:
   - 512 バッチ用の作業バッファ、バンクバッファ、バインドグループをキャッシュし、実行ごとのアロケーションコストを完全排除。
4. **小〜中規模バッチ高速パス保証 ($S \le 64$)**:
   - $S \le 64$ の要求に対しては内部で最適化された `BitParallel64Pull` へ即座にルーティングし、216〜221 ms（$\le 233$ ms）の低レイテンシを完全維持。

### 3.7 ハブ展開と LDS アクセス最適化の第 3 世代アーキテクチャ (Milestone M20)
1. **Phase 0 サブワープ協調フロンティアロード (`indirect_sparse_push_512.wgsl`)**:
   - 従来はサブワープの先頭レーン（`sub_lane == 0`）のみが 16 ワードを直列不可分交換し、残り 3 レーンはアイドル状態であった。
   - 4 スレッドがそれぞれ 4 ワードを担当（$w = \text{sub\_lane} + i \times 4$）し並列に不可分交換を実施することで、Phase 0 のメモリロード遅延を $4\times$ 削減。
2. **LDS キュー容量の 2,048 エントリへの倍増拡張 (`WG_QUEUE_CAP = 2048u`)**:
   - 共有メモリキュー容量を従来の 1,024 エントリから 2,048 エントリへ拡張。
   - 全ワークグループ LDS 消費量は 9,460 バイト（`4096..=12288` バイトの安全マージン内、WebGPU デフォルト上限 16,384 バイト以下）に厳格に収まり、ハブノード展開時のグローバル VRAM スピルオーバー（`queue_sizes[out_slot]` への直列フォールバック）を 90% 以上削減。
3. **LDS 4 サブクアドラント・アトミックカウンタによる 64 ウェイバンク競合の分散解消**:
   - 単一のアトミックカウンタ `wg_queue_count` に 64 スレッドが同時アクセスしていたのを、スレッド ID に応じた 4 つの独立カウンタ `s_sub_queue_count: array<atomic<u32>, 4>`（`qid = local_id.x & 3u`）へ分割。
   - LDS メモリバンク競合を 4 バンクへ分散させ、発見フェーズのアトミックスループットを大幅向上。Phase 3 ではスレッド 0 が 4 クアドラント分のグローバルキューを単一の不可分加算で一括確保し、ワークグループ全体（64 スレッド）で合体 stride 64 フラッシュを実行。
4. **2 階層ウルトラハブ展開 (Two-Tier Ultra-Hub Unrolling, Stride 256)**:
   - `web-Stanford` の最大ハブ頂点（次数 38,625）に対し、次数 $\ge 1,024$（`ULTRA_HUB_THRESHOLD`）のウルトラハブを検出し、4 倍アンロール（stride 256）で展開。
   - 外側ループ回数を 604 回から 151 回へ激減（$4\times$ 圧縮）させ、ロングテール SM におけるループ条件判定・カウンタ更新オーバーヘッドを排除。
5. **距離バンク判定スイッチの外出しホイスティング (`write_bank_bits`)**:
   - 従来のビット走査ループ内で行われていた 8 ウェイ switch 判定（`write_distance_bank`）を、発見ビットワード単位の `write_bank_bits` としてビット走査ループの外部へホイスティング。
   - シェーダーコンパイラによるインライン展開が最大効率化され、各バンクバッファへのタイトなスカラー書き込みシーケンスが生成される。
- **実測性能効果**:
  - `web-Stanford` $S=512$ でスループット **4,272〜5,091 Harmonic MTEPS**（426.38 ms）、CPU Rayon 比 **$2.12\times \sim 8.27\times$ 高速化**。
  - $S=1024$ で **3,923〜5,065 Harmonic MTEPS**（915.70 ms）、CPU Rayon 比 **$2.16\times$ 高速化**。
  - $S=2048$ で **3,993〜5,092 Harmonic MTEPS**（1,814.86 ms）、CPU Rayon 比 **$2.15\times \sim 3.13\times$ 高速化**。
  - 全領域で 100.0% ビット完全性（Parity ✅）を維持。

### 3.8 トポロジー適応型自動ルーターの決定論的最適ルーティング (Milestone M21)
- `select_auto_method` による動的ルーティング規則:
  * **均一低次数道路網 ($d_{\max} \le 8$ かつ $|E| \le 4|V|$, 道路網 `luxembourg_osm` 等 - M26)**:
    - 全バッチサイズ $S \ge 1$: `WorkgroupPersistentBfs` (1 ワークグループ=1 始点自律 BFS、100% オンチップ SRAM 格納、キュー溢れ時は `MultiWordBitParallel` 自動フォールバック)
  * **低次数疎グラフ ($d_{\max} \le 32$, 非道路網)**:
    - $S \ge 64$: `MultiWordBitParallel` (8 バンク Pull 走査、512 始点同時、アトミック競合ゼロ)
    - $32 \le S < 64$: `BitParallel64` (64 本ビット並列)
    - $1 < S < 32$: `BitParallel32` (32 本ビット並列、レジスタ節約)
    - $S = 1$: `SparseQueue` (アクティブ頂点集約型単一始点走査)
  * **べき乗則ハブ網 ($d_{\max} \ge 128$, Web グラフ `web-Stanford` 等 - M22)**:
    - 全バッチサイズ $S \ge 1$: `IndirectSparseQueuePush` (次数考慮型協調ハブ展開、512 始点、LDS 2048、GPU 内 u16 量子化、永続バッファプール)
  * **中間密度網 ($32 < d_{\max} < 128$)**:
    - $S \ge 64$: `IndirectSparseQueuePush`
    - $32 \le S < 64$: `BitParallel64`
    - $1 < S < 32$: `BitParallel32`
    - $S = 1$: `SparseQueue`
- **検証保証**:
  - `crates/gpu_bfs/src/engine.rs` 内の単体テストスイートおよび統合テストにより、すべての次数境界（$d_{\max} \in \{0, 1, 4, 7, 8, 32, 33, 64, 127, 128, 256, 1024, 38625\}$）とバッチサイズ境界（$S \in \{0, 1, 2, 16, 31, 32, 63, 64, 256, 512, 1024, 2048\}$）で決定論的に正しいアルゴリズムが選択されることを完全保証。

### 3.9 詳細プロファイリングで判明した構造的ボトルネックと設計原則 (Post-M21 Analysis)
1. **Pull 走査における Indirect 化の逆効果とバリア全廃の有効性**:
   - 道路網（`luxembourg_osm`, $D=1,311$）の Pull 走査では全アクティブホップで全頂点グリッドを起動するため、`dispatch_workgroups_indirect`（`IndirectMacroBatch`）を適用してもワークグループ削減効果がなく、WSL2 GL-on-D3D12 バックエンドにおける 1,311 回の `prep_hop` 起動と `GL_COMMAND_BARRIER_BIT` により 211.6 ms $\to$ 321.7 ms（+52%）へ劣化する。
   - 一方、`multi_word_bp_pull_512.wgsl` の末尾で行われている 8 回の `workgroupBarrier()` ツリーリダクション（1 パスあたり約 1.5 億回）は、ホスト側が `active_count == 0` の真偽値しか参照しないため、`if (thread_new_count > 0u) { atomicStore(&active_count, 1u); }` へ置換することで共有メモリとバリアを完全にゼロ化できる。
2. **Ping-Pong フロンティアの Dirty-Check 付き条件付きゼロクリア**:
   - Pull シェーダーの `frontier_buf_a` / `frontier_buf_b` はホップ間で `clear_buffer` されない Ping-Pong バッファであるため、`next_frontier[state_idx] = 0u` を単純削除すると 2 ホップ前の古いフロンティアビットが残存してパリティ破壊を起こす。
   - `if (next_frontier[state_idx] != 0u) { next_frontier[state_idx] = 0u; }` と Dirty Check を行うことで、Ping-Pong 不変条件を 100% 維持しつつ、99.92% の非アクティブ頂点に対する 9.6 GB/パスの無駄な DRAM 書き込みを L1/L2 キャッシュヒット判定へ変換できる。
3. **マルチワード・カーネルの端数バッチ上限ルーティングと実サイズリセット**:
   - 512-bit / 256-bit カーネルの分岐条件は下限（`>= 512` / `>= 256`）ではなく、1 パスで収容可能な上限基準（`num_sources > 256` $\to$ 512-bit、`num_sources > 64` $\to$ 256-bit）とし、シェーダー内で `w < uniforms.num_words` により非アクティブワードをスキップさせることで、$S=128, 384$ 等の端数バッチにおける複数パス直列劣化を防止する。
   - 距離バッファのリセットおよび `u16` パック範囲は常に最大容量（`64 * nv`）ではなく実バッチ要素数（`batch_size * nv`）に限定する。

### 3.10 高直径道路網で CPU Rayon を完全逆転した「WorkgroupPersistentBfs」アーキテクチャ (Milestone M26)

高直径・均一低次数グラフ（`luxembourg_osm` 等）における長年の課題であったホスト・デバイス同期遅延を解消するため、M26 では「1 ワークグループ = 1 始点」「単一コンピュートパス内 1,311 ホップ全自律走査」を実現する `WorkgroupPersistentBfs` を開発しました。

1. **1 Workgroup = 1 始点の単一ディスパッチ全ホップ自律反復 (Single-Dispatch Autonomous Traversal)**:
   - 道路網（`luxembourg_osm`, $D \approx 1,311$）において、各ワークグループ（128 スレッド）が独立して 1 つの始点の探索を開始から終了まで `for (var level = 1u; level <= nv; level = level + 1u)` ループ内で完遂。
   - 1,311 回に及んでいたホスト・デバイス同期（`/dev/dxg` ラウンドトリップ）や間接ディスパッチ（`IndirectMacroBatch`）のコマンドバリアオーバーヘッドを完全に 0 回に削減。
2. **100% オンチップ SRAM 格納（WebGPU 16,384 バイト制限内）**:
   - `s_visited: array<atomic<u32>, 3584>`: 114,688 頂点分の訪問ビットマスク（14,336 バイト）。`luxembourg_osm`（$|V|=114,599$）が 100% 収まる。
   - `s_queue: array<u32, 508>`: 単一フロンティアキュー（2,032 バイト）。
   - 制御変数: `s_q_read`, `s_q_write`, `s_next_count`, `s_overflow` 等（16 バイト）。
   - 合計 $14,336 + 2,032 + 16 = 16,384$ バイト（WebGPU 仕様上限 16 KB に 1 バイトの狂いもなく合致）。
   - レジスタバッファリングにより第 2 Ping-Pong キューを排し、VRAM ストレージバリアを不要化。
   - 頂点数 $> 114,688$ のグラフに対しては VRAM 訪問バッファ（`vram_visited`）へのフォールバックを自動切替。
3. **反転ビット `atomic<u32>` 半ワード直接距離エンコーディング**:
   - 1 チャンク 512 始点分の距離を 117.35 MB の単一ストレージバッファ（`packed_dist: array<atomic<u32>>`）に格納（128 MB 制限を遵守）。
   - ハードウェア初期化: `encoder.clear_buffer` により 0 で高速一括クリア。
   - 発見時直接書き込み: 未訪問頂点への到達時、`inv_d = 0xFFFFu - min(level, 0xFFFFu)` を計算し、`atomicOr(&packed_dist[s_word_base + (w >> 1u)], inv_d << ((w & 1u) * 16u))` で 16-bit 半ワードへ直接書き込み。
   - これにより、従来の 8 バンク距離リセットパス、8 バンク `u16` 量子化コンピュートパス、8 ウェイ switch 判定がすべて不要に。
   - ホスト側でのアンパック: `if raw == 0 { u32::MAX } else { 0xFFFF - raw }` により到達不能マーカーも含め完全復元。
4. **ハードウェアキューオーバーフロー検知 & 透明な `MultiWordBitParallel` フォールバック**:
   - 狭フロンティアを想定する 508 要素 LDS キューに対し、広範な波動フロントを持つグラフ（Erdős-Rényi や 2D グリッド $300 \times 300$ など）では、シェーダーが `s_overflow = 1u` を検知し `@binding(7) overflow_buf[s_idx] = 1u` に記録。
   - ホストディスパッチャがオーバーフローを検知すると、該当始点またはチャンク全体を透明に `MultiWordBitParallel` へフォールバック実行。
   - 未訪問ノードの脱落や距離破損、ハングアップを原理的に排除し、あらゆるトポロジーで 100.0% パリティを数学的に保証。
5. **空間的始点ソート (Spatial Source Sorting)**:
   - 始点インデックスを頂点 ID 昇順にソートしてディスパッチ。
   - 隣接するワークグループがグラフの近接領域を同時に探索するため、CSR 出エッジリストの GPU L2 キャッシュ（1.5 MiB）内ヒット率が飛躍的に向上。
6. **2 段階ダブルバッファリング非同期パイプライン**:
   - $S > 512$ の大規模バッチにおいて、Chunk $i$ の GPU 探索コマンド実行と、Chunk $i-1$ のホスト CPU 並列アンパック処理を 2 面のステージングバッファにより真のパイプライン並行処理化。
   - GPU アイドル時間を完全に隠蔽。
7. **SRAM 到達可能性直接協調フラッシュ**:
   - 到達可能性モード（`run_workgroup_persistent_reachability`）では、`packed_dist` への距離書き込みを一切行わず、探索終了時に SRAM 上の `s_visited`（14.3 KB/始点）を 128 スレッド協調で VRAM の `visited_out_buf` へ一括フラッシュ。
   - $S=2048$ における PCIe DMA 転送量を 117.35 MB から 29.34 MB へ激減させ、実行時間を 178 ms（11,473 sources/s）へ短縮。
8. **ホスト側ビットマスク変換のループ反転並列化**:
   - `ReachabilityBitmask::from_canonical_distances` において、従来の頂点外側／ワード内側ループによる激しい L1/L2 キャッシュスラッシング（2,738 ms）を解消。
   - ワード外側／頂点内側ループへの反転と Rayon 並列化により、ホスト変換遅延をほぼゼロ化。

---

## 4. 扱えるグラフ規模のハードウェア限界モデル (Scaling Limits)

本環境（16GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU 6GB GDDR6 @ 168 GB/s / WSL2 D3D12）における実効限界：

| 探索設定 | ボトルネック要因 | 道路網型 ($|E| \approx 3|V|$) | ソーシャル・Web型 ($|E| \approx 16|V|$) |
| :--- | :--- | :--- | :--- |
| **単一始点 ($S=1$)** | VRAM 容量 (実効 5.2 GB) | **$|V| \approx 2.5 \text{億}, |E| \approx 7.5 \text{億}$** | **$|V| \approx 7,500 \text{万}, |E| \approx 1.2 \text{億}$** |
| **中規模マルチソース ($S=64$)** | 距離バッファ + CSR | **$|V| \approx 1,800 \text{万}, |E| \approx 5,400 \text{万}$** | **$|V| \approx 1,500 \text{万}, |E| \approx 2.4 \text{億}$** |
| **大規模マルチソース ($S=512$, 1パス同時)** | 単一バンク 128MB 制限 ($S_{\text{bank}}=64$) | **$|V| \le 524,288$ 頂点** | **$|V| \le 524,288$ 頂点** |
| **大規模マルチソース ($S=2048$, 4チャンク分割)** | VRAM 容量 (実効 5.2 GB) | **$|V| \approx 200 \text{万}, |E| \approx 600 \text{万}$** | **$|V| \approx 150 \text{万}, |E| \approx 2,400 \text{万}$** |
| **大規模マルチソース ($S=2048$, ホスト退避)** | システム RAM (10GB 空き) | **$|V| \approx 120 \text{万}, |E| \approx 360 \text{万}$** | **$|V| \approx 100 \text{万}, |E| \approx 1,600 \text{万}$** |
| **CPU 単一始点 ($S=1$)** | システム RAM (10GB 空き) | **$|V| \approx 5.0 \text{億}, |E| \approx 15 \text{億}$** | **$|V| \approx 1.4 \text{億}, |E| \approx 22 \text{億}$** |

> **マルチバンクバッファ限界の数理モデル**:
> 各バンクは 64 始点を格納するため、頂点あたり $64 \times 4\text{ B} = 256\text{ B}$ を消費する。WebGPU の単一バッファバインディング上限が 128 MB であるため、1 パスで処理可能な頂点数上限は:
> $$|V|_{\max} = \frac{128 \times 1024 \times 1024}{256} = 524,288\text{ 頂点}$$
> となる。これを超える頂点数のグラフに対しては、さらにバンク始点幅を 32 始点へ分割するか、マクロチャンク分割によって VRAM とシステム RAM 間をストリーミングする。

---

## 5. 今後の方針とロードマップ (Future Roadmap)

### Phase 3: 完了テーマ (Completed)
1. **距離配列の 16 ビット量子化 (`u16` Distance)** 【M14 実装完了 ✅】:
   - ホップ数が 65,535 以内のグラフにおいて、距離配列を GPU 上で `u16` に圧縮して転送。PCIe DMA 転送量を半減（36 MB）。
2. **トポロジー特性に基づく Push/Pull 自動選定機構 (`Auto`)** 【M17 実装完了 ✅】:
   - グラフ読み込み時に最大次数を事前評価し、道路網（均一低次数）なら MultiWord Pull 型、べき乗則グラフ（ハブ存在）なら Indirect Sparse Push 型を自動選択するオプティマイザを導入。
3. **次数考慮型協調ハブ展開と 512-bit マルチワード Push** 【M16–M18 実装完了 ✅】:
   - ハブノード（次数 $\ge 128$）のワークグループ協調展開と 512-bit マルチワード化により、無向べき乗則グラフの大規模バッチ（$S \in \{512, 1024, 2048\}$）で CPU Rayon を完全逆転。
4. **道路網（`luxembourg_osm`）へのマルチバンク・u16 量子化の水平展開** 【M19 実装完了 ✅】:
   - 8 バンク分散型 Pull 走査（512 始点/パス）、単一エンコーダ GPU 内 `u16` 量子化、永続バッファプールにより、$S=2048$ のパス数を 32 回から 4 回へ削減、実行時間を 6,389 ms から 2,686 ms へ大幅高速化（$2.38\times$）。$S=64$ でも 216 ms を維持。
5. **ハブ展開・LDS アクセス最適化と 5,000+ MTEPS 突破** 【M20 実装完了 ✅】:
   - Phase 0 サブワープ協調ロード、LDS 2,048 キュー、4 サブクアドラント・アトミックバンク競合解消、Stride 256 ウルトラハブ展開、バンク判定外出しにより、無向べき乗則グラフの大規模バッチで 4,272〜5,091 MTEPS（CPU Rayon 比 $2.12\times \sim 8.27\times$ 高速化）を達成。
6. **トポロジー自動ルーター検証・全ワークスペース完全パリティ** 【M21 実装完了 ✅】:
   - `select_auto_method` によるグラフ次数・始点数に応じた最適アルゴリズム選定（低次数網: MultiWordBitParallel、べき乗則網: IndirectSparseQueuePush、小中バッチ: BitParallel64、単一始点: SparseQueue）の単体・結合テスト網羅、ワークスペース全テストでの 100.0% ビット完全性検証。
7. **`ISQP` 端数バッチ 1 パス化・距離リセット極小化・M20 水平展開・小中バッチ `Auto` 刷新** 【M22 実装完了 ✅】:
   - 端数バッチ（$256 < S \le 512 \to \text{pool\_512}$, $64 < S \le 256 \to \text{pool\_256}$）の 1 パス統合と Phase 0 `w < uniforms.num_words` ガードによる非アクティブワードスキップにより、端数バッチの複数パス直列劣化を完全解消。
   - 距離配列初期化を実要素数 $S \times |V|$（$S=1$ で 1.13 MB）へ縮小、$S \le 32$ で `num_words = 1` を適用、`std::thread::scope` 細粒度並列アンパックを導入。
   - `indirect_sparse_push_256.wgsl` および `indirect_sparse_push.wgsl` へ Phase 0 サブワープ協調ロード、2,048 エントリ LDS キュー、4 クアドラント・アトミック、Stride 256 ウルトラハブ展開、距離書き込みホイスティングを水平展開。
   - べき乗則グラフ（$d_{\max} \ge 128$）の $S \ge 1$ を永続プール対応 `IndirectSparseQueuePush` へルーティングし、$S=1$（25〜37 ms、5x 高速化）、$S=16$（58〜68 ms）で CPU Rayon を完全逆転。
8. **高直径網向け適応的マクロバッチ拡大 & Pull カーネルのバリア全廃・Dirty-Check 化・重複バッファ排除** 【M23 実装完了 ✅】:
   - `multi_word_bp_pull_512.wgsl` の 8 段階ツリーリダクション `workgroupBarrier()` を廃止し、`if (thread_new_count > 0u) { atomicStore(&active_count, 1u); }` による直接アトミックストア置換（バリア 0 回、LDS 0 バイト化）。
   - Ping-Pong フロンティアの Dirty-Check 付き条件付きゼロクリア（`if (next_frontier[state_idx] != 0u) { next_frontier[state_idx] = 0u; }`）により、パスあたり約 9.6 GB の DRAM 書き込みトラフィックを排除。
   - 未使用始点ワードの早期スキップ（`src_start >= params.total_sources`）および距離バンク判定のループ外ホイスティング。
   - `indirect_sparse_push.rs` の適応的マクロバッチ拡大（3 階層: $160 \to 320 \to 512$ ホップ）により、高直径道路網の `/dev/dxg` 同期を 19 回から 4 回へ削減。
   - BFS 単調性に基づく最終ステップ単一カウンタ評価（`counter_bufs[last_k]` のみ読み出し）により、ドライバカウンタコピーを $64\times$ 削減。
   - `PushPullGraphBuffers` の `Arc<wgpu::Buffer>` による CSC/CSR バッファ重複排除および `from_symmetric_graph` によるホスト転置バイパス。
   - `luxembourg_osm` $S=2048$ で 1,765〜2,054 ms（$1.31\times \sim 1.52\times$ 高速化）、$S=64$ で 211〜230 ms を達成。

### Phase 4: 実装完了マイルストーン (Milestones M24–M26) & 将来展望

#### Milestone M24: 適応的 `u8`/`u16` 距離量子化 & 非同期 DMA チャンクパイプライン & 2D Indirect Grid 【M24 実装完了 ✅】
1. **GPU 内 `u8` 距離量子化 (`pack_distances_u32_to_u8.wgsl`)**:
   - 最終到達レベル `st[0] < 255`（`web-Stanford` 等のスモールワールド網）のとき、4 頂点距離を 1 ワード（`u32`）へパッキング（到達不能は `0xFF`）。PCIe DMA 転送量を 288.67 MB $\to$ 144.33 MB（`u32` 比 $4\times$ 圧縮）へ半減。
2. **ダブルバッファリング非同期 DMA パイプライン**:
   - $S=1024, 2048$ の複数チャンク実行時に、チャンク $i$ の CPU アンパック処理とチャンク $i+1$ の GPU 探索コマンド実行をオーバーラップ。
3. **`prep_sparse_push_hop.wgsl` の 2D Indirect Grid 対応**:
   - `total_wgs > 65535u` の際に `workgroups_x = 256u; workgroups_y = (total_wgs + 255u) / 256u;` へ自動展開し、`@builtin(num_workgroups)` で `wg_id` を復元することで 100 万頂点超のアクティブキューに完全対応。

#### Milestone M25: GPU 自律型 Direction-Optimizing ハイブリッド & 到達可能性ビットマスク API & 動的バンク幅スケーリング 【M25 実装完了 ✅】
1. **GPU 自律型 Direction-Optimizing (Sparse Push $\leftrightarrow$ Multi-Bank Pull) ハイブリッド**:
   - `prep_hybrid_hop.wgsl` が `in_count > |V| / 16` を検知した高密度ホップのみ 8 バンク Pull パスを起動し、疎なホップは Sparse Queue Push を実行する GPU 自律型ハイブリッド反復（ホスト介入 0 回）。
2. **到達可能性（Reachability）ビットマスク直接出力 API & $|V| > 524,288$ 動的バンク幅スケーリング**:
   - `ReachabilityBitmask` および `ReachabilityEngine` トレイトを導入。8 バンク距離バッファ割り当てと u16 量子化をバイパスし、`visited` ビットマスク（1 bit/source/vertex）をそのままゼロコピー DMA リードバック。
   - $|V| > 524,288$ グラフ向けの動的バンク幅スケーリング（$64 \to 32 \to 16$ 始点/バンク）により、128 MB WebGPU バッファ制限を決定論的にクリア。
3. **CLI デフォルトの `Auto` 化と WASM 非同期 `map_async` 対応**:
   - `crates/cli/src/args.rs` のデフォルト手法を `Auto` に更新し、`--reachability` フラグを verify / bench に追加。
   - `crates/wasm` で `run_gpu_reachability` / `verify_reachability` をエクスポートし、Web UI に `Auto` (デフォルト) と `IndirectSparseQueuePush` を追加。

#### Milestone M26: WorkgroupPersistentBfs アーキテクチャ & 道路網 CPU 完全逆転 【M26 実装完了 ✅】
1. **1 ワークグループ = 1 始点の単一ディスパッチ全ホップ自律反復**:
   - `workgroup_persistent_bfs.wgsl` において、各ワークグループ（128 スレッド）が 1 始点を担当し、終了までホスト・デバイス同期 0 回で全 1,311 ホップを自律走査。
2. **100% オンチップ SRAM 格納 (WebGPU 16 KB 制限厳格遵守)**:
   - 訪問フラグ `s_visited`（14,336B）+ フロンティアキュー `s_queue`（2,032B）+ 制御変数（16B）= 合計 16,384B を共有メモリ内に収容。
3. **反転ビット半ワード直接距離エンコーディング**:
   - 単一 117.35 MB バッファ（128 MB 制限内）への `inv_d = 0xFFFFu - min(level, 0xFFFFu)` 直接アトミック書き込みにより、8 バンクリセット・量子化パスを全廃。
4. **ハードウェアキュー溢れ検知 & 透明フォールバック**:
   - `@binding(7) overflow_buf` による波動フロント溢れ検知と `MultiWordBitParallel` 自動フォールバックにより、あらゆるグラフで 100% ビット完全性を保証。
5. **空間的始点ソート & 2 段階非同期パイプライン**:
   - 頂点 ID 順の空間的ソートで GPU L2 キャッシュヒット率を最大化し、Chunk $i$ の GPU 探索と Chunk $i-1$ の CPU アンパックをダブルバッファリングで完全オーバーラップ。
6. **SRAM 到達可能性直接協調フラッシュ**:
   - 探索終了時に SRAM から VRAM へ `s_visited`（14.3 KB/始点）を一括フラッシュし、PCIe 転送量を $16\times$ 削減。
7. **定量的 CPU 逆転の達成**:
   - `luxembourg_osm` において $S=64$ で 20〜24 ms（前 baseline 211 ms 比 $10\times$ 高速化、CPU Rayon 逆転）、$S=2048$ 距離探索で 683〜736 ms（CPU Rayon 1,155 ms 逆転）、$S=2048$ 到達可能性探索で 178〜185 ms（11,473 sources/s、CPU Rayon 比 $9.0\times$、前 baseline 比 $32.8\times$ 高速化）を達成。

#### 将来展望 (Phase 5: Out-of-Core & Multi-GPU)
1. **Out-of-Core / エッジ分割ストリーミングおよび分散マルチ GPU 展開**:
   - VRAM 上限を超える超巨大グラフのブロック分割ストリーミングと複数 GPU への展開。

---

## 6. コントリビューション & 開発の指針 (Agent Guidelines)

- **パリティ検証の絶対死守**:
  新アルゴリズムや最適化シェーダーを追加した際は、必ず `cargo test` および `webgpu-bfs verify` により、全頂点・全始点で CPU リファレンスとホップ数が 100% 一致することを検証すること。
- **WebGPU ポータビリティと WSL2 D3D12 の維持**:
  WGSL コードおよびバインディング設計において、標準 WebGPU 仕様を維持しつつ、WSL2 `/dev/dxg` Direct3D 12 Gallium パスでの動作安定性を維持すること。
- **2D グリッドディスパッチの徹底**:
  頂点単位および Indirect キュー単位のコンピュートシェーダーを設計する際は、65535 を超えるワークグループを安全にディスパッチできる 2D グリッドマッピングを採用すること。
- **マルチバンク距離バッファ設計と端数バッチ上限ルーティングの遵守**:
  WebGPU の標準 128 MB バインディング制限を意識し、64 始点単位のバンク分割設計を維持すること。また、マルチワード・カーネルの分岐は `num_sources > 256` ($\to$ 512) / `num_sources > 64` ($\to$ 256) のように上限容量基準でルーティングし、リセットや DMA コピーは実バッチサイズ分のみに限定すること。
- **Ping-Pong バッファにおける Dirty-Check 原則の遵守**:
  ホップ間で `clear_buffer` されない Ping-Pong フロンティアバッファにおいて、非アクティブ要素のゼロクリアを単純削除してはならない（2 ホップ前の古いビットが残存するため）。必ず `if (next_frontier[idx] != 0u) { next_frontier[idx] = 0u; }` の Dirty Check 付き条件付き書き込みを用いること。
- **ハブノードに対する協調展開の徹底**:
  べき乗則グラフの Push 走査シェーダーでは、次数 $\ge 128$ のハブノードを単一スレッド／サブワープで直列走査せず、ワークグループ協調（stride 64/256）で展開すること。
- **バッファプールと事前フォールティングの遵守**:
  反復ベンチマークや実運用パイプラインにおいて、GPU バッファの頻繁な再割り当てや未フォールトメモリへの DMA コピーを避け、ゼロコピーおよびキャッシングを活用すること。
