# WebGPU Multi-Source BFS (`webgpu-bfs`) Project Knowledge Base & Agent Guide

> **Document Version**: 1.1.0  
> **Last Updated**: 2026-10-11  
> **Scope**: Architecture, Milestones M1–M14, Hardware & WebGPU Constraints, Algorithmic Insights, and Future Roadmap

---

## 1. プロジェクト概要と現時点での主要成果 (Achievements to Date)

### 1.1 開発目標と背景
本プロジェクト (`webgpu-bfs`) は、Rust と WebGPU (WGSL / `wgpu`) を基盤とし、重みなし複数始点最短経路問題（MS-BFS: Multi-Source Breadth-First Search）を GPU 上で極限まで高速化する並列グラフ探索ライブラリおよびベンチマーク環境の開発を目的としています。

グラフ探索における古典的課題である「CPU に対するレイテンシ劣後（特に高直径・狭フロンティアグラフ）」を克服するため、GraphBLAST 等の代数的グラフアルゴリズム知見、ビット並列フロンティア、2D ワープタイリング、GPU 自律反復（`dispatch_workgroups_indirect`）等の最先端アーキテクチャを体系的に導入しました。

### 1.2 達成された定量的成果 (Breakthroughs)

1. **実データ `luxembourg_osm` での定量的 CPU 逆転 (Milestones M7–M10)**:
   - 高直径（$D \approx 1,200$）、疎結合（$|V|=114,599$, $|E|=239,332$）の道路網において、大規模マルチソース束ね（$S=2048$）により **19,504 sources/sec**（$4,668\text{ MTEPS}$）を達成。
   - **CPU Multi-Thread (Rayon Ground Truth) に対し $24.56\times$ 高速化**。
   - **CPU Single-Thread (FIFO) に対し $42.94\times$ 高速化**。
2. **有向 Web グラフ `web-Stanford` での定量的 CPU 逆転 (Milestones M11–M14)**:
   - $|V|=281,903$, $|E|=2,312,497$, 非対称有向グラフ（$D \approx 200$）において、WSL2 D3D12 ハードウェア GPU 直接駆動と 6 つの高速化ピラー（サブワープ協調展開・共有メモリ集約・GPU 内 `u16` 量子化・永続バッファプール・適応型 160 ホップマクロバッチ）により、**$S=64$ で GPU 実行時間 58.38 ms（最速）/ 61.06 ms（平均）** を達成。
   - **CPU Multi-Thread (Rayon: 65.75 ms / 102.44 ms) に対し $1.08\times \sim 1.18\times$ 高速化**。
   - 大規模バッチ（$S=128$ で $1.30\times$、$S=256$ で $1.88\times$、CPU 1 スレッド比で最大 $5.54\times$ 高速化）。
3. **100% ビット単位パリティ検証 (Bitwise Correctness)**:
   - 全 13 の GPU 手法と CPU リファレンス（Rayon / FIFO）の間で、数億ホップ計算にわたる最短距離配列のビット単位完全一致（Parity ✅ 100.0%）を機械的テストスイート（306 件）で保証。
4. **Native CLI と WASM / ブラウザのデュアルターゲット対応**:
   - ヘッドレス高精度ベンチマーク CLI と、同一コードベースを共有する WebAssembly / ブラウザ WebGPU 実行環境の両立。

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
| `BitParallel64Pull` | GPU | Pull | CSC 逆辺を用いた 64 本 Pull 走査（アトミック競合ゼロ） | 中密度・対称グラフ |
| `BooleanReachability`| GPU | Push | 到達可能性ブーリアンベクトル更新＋距離更新 | 論理到達判定と距離更新の分離 |
| `PushPullAdaptive` | GPU | Hybrid | フロンティア密度（閾値 5%）に基づく動的 Push/Pull 切り替え | スケールフリー網・高密度グラフ |
| `SparseQueue` | GPU | Push | Prefix Sum / キュー集約によるアクティブ頂点のみの走査 | **単一始点 ($S=1$) GPU 最速** |
| `SparseBitParallel64`| GPU | Push | 64 本ビット並列＋動的キュー集約フィルタリング | 小〜中規模バッチ ($S \le 16$) で強力 |
| `MultiWordBitParallel`| GPU | Pull | 2D ワープタイル ($32 \times 4$)、マルチワード ($S \le 2048$) | **無向高直径網（道路網）の大規模バッチで最速** |
| `IndirectMacroBatch` | GPU | Pull | `dispatch_workgroups_indirect` による GPU 内完結反復 | ホスト・デバイス間通信の極小化 |
| `IndirectSparseQueuePush`| GPU | Push | サブワープ協調展開＋共有メモリ集約＋160ホップ自律反復＋GPU内u16量子化 | **有向べき乗則グラフ（Web）で最速・CPU Rayon 逆転** |
| `MultiWordBitParallelPush`| GPU | Push | 動的ストライドループによる複数ワード ($S \le 2048$) Push 型ビット並列 | 有向グラフ・中〜大規模バッチ |

---

## 3. プロジェクトで得られた重要な技術的知見 (Learnings & Insights)

### 3.1 グラフ形状と探索方向（Push vs Pull）の適合性
- **道路網・高直径グラフ (`luxembourg_osm`, $D \approx 1,200$)**:
  - フロンティアが常に狭く長く伸びるため、GPU スレッドの過半数が空振りする問題が生じる。
  - 大規模マルチソース（$S \ge 256$）で多数の始点を束ねると、**Pull 走査（`MultiWordBitParallel`）**が圧倒的に有利になる。
  - 理由: 全頂点の次数が $\le 7$ と均一で、Pull 方式では各頂点が入エッジを集約して自身の距離・フロンティアを単一スレッドで決定的に書き込むため、**アトミック競合が完全にゼロ**であり、128 バイト合体メモリアクセスが最大効率で働く。
- **有向 Web グラフ (`web-Stanford`, $D \approx 200$)**:
  - 道路網とは対照的に、**Push 方式（`IndirectSparseQueuePush`）が最速**。
  - 理由: 有向 Web グラフには入次数が 0 のノードが 61% 以上存在し、Pull 走査では空振りの CSC 走査負荷が激増する。一方、出次数は $\le 255$ に厳格にバウンドされているため、アクティブなフロンティアのみを展開する Push 方式が極めて効率的に機能する。
- **無向べき乗則グラフ (`web-Stanford-undirected`)**:
  - 最大次数 38,606 の超巨大ハブノードが存在する場合、**Pull 走査はワープダイバージェンスにより著しく破綻（実行時間 4.5〜5.5 秒）**する。
  - 同一ワープ内で「次数 2 のノード」と「次数 38,606 のハブ」を処理するスレッドが混在し、全スレッドが最長ループに引きずられる。
  - 一方、Push 方式（`BitParallel32`, `IndirectSparseQueuePush`）は 590〜652 ms を維持し、CPU シングルスレッドに対し **$1.42\times \sim 2.02\times$ 高速**を維持する。

### 3.2 WebGPU / WGSL のハードウェア・API 制約と対処パターン
WebGPU は高いポータビリティを持つ反面、ネイティブ CUDA 等に比べて厳格な制約が存在します。本プロジェクトで確立した回避パターンは以下の通りです。

1. **`max_compute_workgroups_per_dimension = 65535` の壁**:
   - **事象**: 頂点数 $|V| > 65,535 \times W_v$ のグラフにおいて、1D ディスパッチを行うと WebGPU バリデーションエラーでパニックする。
   - **解決策 (2D Grid Dispatch)**:
     $$\text{dispatch\_x} = 256, \quad \text{dispatch\_y} = \left\lceil \frac{\text{total\_wgs}}{256} \right\rceil$$
     シェーダー内で `let wg_id = group_id.x + group_id.y * params.dispatch_x;`、`let v = wg_id * W_v + local_id.y;` として頂点インデックスを復元。
2. **単一ストレージバッファ上限 (`max_storage_buffer_binding_size`)**:
   - **事象**: デフォルト値は 128 MB（Elevated Limits で 2 GB）。大規模マルチソースでは出力距離行列（$4S|V|$）が支配的になり、バッファ OOM となる。
   - **解決策 (マクロチャンク自動分割)**:
     `determine_macro_chunk_size` により最大安全始点数を自動算出し、チャンク単位で GPU 実行・ホスト退避をストリーミング化。
3. **WSL2 Direct3D 12 ハードウェア GPU の直接活性化 (Milestone M11)**:
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

---

## 4. 扱えるグラフ規模のハードウェア限界モデル (Scaling Limits)

本環境（16GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU 6GB GDDR6 @ 168 GB/s / WSL2 D3D12）における実効限界：

| 探索設定 | ボトルネック要因 | 道路網型 ($|E| \approx 3|V|$) | ソーシャル・Web型 ($|E| \approx 16|V|$) |
| :--- | :--- | :--- | :--- |
| **単一始点 ($S=1$)** | VRAM 容量 (実効 5.2 GB) | **$|V| \approx 2.5 \text{億}, |E| \approx 7.5 \text{億}$** | **$|V| \approx 7,500 \text{万}, |E| \approx 1.2 \text{億}$** |
| **中規模マルチソース ($S=64$)** | 距離バッファ + CSR | **$|V| \approx 1,800 \text{万}, |E| \approx 5,400 \text{万}$** | **$|V| \approx 1,500 \text{万}, |E| \approx 2.4 \text{億}$** |
| **大規模マルチソース ($S=2048$, 一括)** | 単一バッファ 2GB 制限 | **$|V| \le 262,144$ 頂点** | **$|V| \le 262,144$ 頂点** |
| **大規模マルチソース ($S=2048$, 分割)** | システム RAM (10GB 空き) | **$|V| \approx 120 \text{万}, |E| \approx 360 \text{万}$** | **$|V| \approx 100 \text{万}, |E| \approx 1,600 \text{万}$** |
| **CPU 単一始点 ($S=1$)** | システム RAM (10GB 空き) | **$|V| \approx 5.0 \text{億}, |E| \approx 15 \text{億}$** | **$|V| \approx 1.4 \text{億}, |E| \approx 22 \text{億}$** |

---

## 5. 今後の方針とロードマップ (Future Roadmap)

### Phase 3: 次期最適化および機能拡張テーマ

1. **距離配列の 16 ビット量子化 (`u16` Distance)** 【M14 実装完了 ✅】:
   - ホップ数が 65,535 以内のグラフにおいて、距離配列を GPU 上で `u16` に圧縮して転送。PCIe DMA 転送量を半減（36 MB）。
2. **到達可能性（Reachability）特化ビットマスクモード**:
   - 最短ホップ数が不要で接続性・到達判定のみを求める場合、距離配列を排除しビットマスク（$\frac{S}{8}$ バイト/頂点）のみを出力。
   - 効果: $S=2048$ でも頂点あたりわずか 256 バイト（$32\times$ 圧縮）となり、$S \ge 512$ の大規模バッチでも PCIe 転送ボトルネックを完全突破。
3. **トポロジー特性に基づく Push/Pull 自動選定機構**:
   - グラフ読み込み時に出次数・入次数の分散（歪度）および対称性を事前評価し、道路網（均一低次数）なら MultiWord Pull 型、べき乗則グラフ（有向・無向問わずハブ存在）なら Indirect Sparse Push 型を自動選択するオプティマイザの導入。
4. **Out-of-Core / エッジ分割ストリーミング**:
   - VRAM 上限（6.0 GB）を超える超巨大グラフ（全米道路網 USA-road-d など）を扱うため、CSR エッジ配列をブロック分割し、ホスト RAM からストリーミング走査するパイプラインの設計。
5. **WASM / Web フロントエンドのデモ・可視化強化**:
   - ブラウザ上で Matrix Market ファイルをドラッグ＆ドロップし、WebGPU Compute Shader の実行時間や MTEPS、探索波面の広がりをリアルタイム Canvas 描画するインタラクティブデモ UI の整備。

---

## 6. コントリビューション & 開発の指針 (Agent Guidelines)

- **パリティ検証の絶対死守**:
  新アルゴリズムや最適化シェーダーを追加した際は、必ず `cargo test` および `webgpu-bfs verify` により、全頂点・全始点で CPU リファレンスとホップ数が 100% 一致することを検証すること。
- **WebGPU ポータビリティと WSL2 D3D12 の維持**:
  WGSL コードおよびバインディング設計において、標準 WebGPU 仕様を維持しつつ、WSL2 `/dev/dxg` Direct3D 12 Gallium パスでの動作安定性を維持すること。
- **2D グリッドディスパッチの徹底**:
  頂点単位のコンピュートシェーダーを設計する際は、将来の大規模グラフ適用に備え、65535 を超えるワークグループを安全にディスパッチできる 2D グリッドマッピングを採用すること。
- **バッファプールと事前フォールティングの遵守**:
  反復ベンチマークや実運用パイプラインにおいて、GPU バッファの頻繁な再割り当てや未フォールトメモリへの DMA コピーを避け、ゼロコピーおよびキャッシングを活用すること。
