# WebGPU Multi-Source BFS (`webgpu-bfs`) Project Knowledge Base & Agent Guide

> **Document Version**: 1.0.0  
> **Last Updated**: 2026-10-11  
> **Scope**: Architecture, Milestones M1–M10, Hardware & WebGPU Constraints, Algorithmic Insights, and Future Roadmap

---

## 1. プロジェクト概要と現時点での主要成果 (Achievements to Date)

### 1.1 開発目標と背景
本プロジェクト (`webgpu-bfs`) は、Rust と WebGPU (WGSL / `wgpu`) を基盤とし、重みなし複数始点最短経路問題（MS-BFS: Multi-Source Breadth-First Search）を GPU 上で極限まで高速化する並列グラフ探索ライブラリおよびベンチマーク環境の開発を目的としています。

グラフ探索における古典的課題である「CPU に対するレイテンシ劣後（特に高直径・狭フロンティアグラフ）」を克服するため、GraphBLAST 等の代数的グラフアルゴリズム知見、ビット並列フロンティア、2D ワープタイリング、GPU 自律反復（`dispatch_workgroups_indirect`）等の最先端アーキテクチャを体系的に導入しました。

### 1.2 達成された定量的成果 (Breakthroughs)

1. **実データ `luxembourg_osm` での定量的 CPU 逆転**:
   - 高直径（$D \approx 1,200$）、疎結合（$|V|=114,599$, $|E|=239,332$）の道路網において、大規模マルチソース束ね（$S=2048$）により **19,504 sources/sec**（$4,668\text{ MTEPS}$）を達成。
   - **CPU Multi-Thread (Rayon Ground Truth) に対し $24.56\times$ 高速化**。
   - **CPU Single-Thread (FIFO) に対し $42.94\times$ 高速化**。
2. **有向 Web グラフ `web-Stanford` での動作確立と特性解明**:
   - $|V|=281,903$, $|E|=2,312,497$, 非対称有向グラフ（$D \approx 200$）において、2D グリッドディスパッチにより 65,535 頂点制限を突破。
   - 64 始点同時探索において、Push 型ビット並列（`BitParallel64`）が **816 ms** を達成（Naive 直列反復 4,193 ms に対し **$5.14\times$ 高速化**、始点あたり実効時間を $35\times$ 圧縮）。
3. **100% ビット単位パリティ検証 (Bitwise Correctness)**:
   - 全 11 の GPU 手法と CPU リファレンス（Rayon / FIFO）の間で、数億ホップ計算にわたる最短距離配列のビット単位完全一致（Parity ✅ 100.0%）を機械的テストスイートで保証。
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
| `BitParallel32` | GPU | Push | 32 始点を 1 ワード (`u32`) にパッキングしたビット演算 | 中規模バッチの基礎 |
| `BitParallel64` | GPU | Push | 64 始点を 2 ワード (`vec2<u32>`) にパッキングしたビット並列 | **有向 Web グラフ・中規模バッチで最速** |
| `BitParallel64Pull` | GPU | Pull | CSC 逆辺を用いた 64 本 Pull 走査（アトミック競合ゼロ） | 中密度・対称グラフ |
| `BooleanReachability`| GPU | Push | 到達可能性ブーリアンベクトル更新＋距離更新 | 論理到達判定と距離更新の分離 |
| `PushPullAdaptive` | GPU | Hybrid | フロンティア密度（閾値 5%）に基づく動的 Push/Pull 切り替え | スケールフリー網・高密度グラフ |
| `SparseQueue` | GPU | Push | Prefix Sum / キュー集約によるアクティブ頂点のみの走査 | **単一始点 ($S=1$) GPU 最速** |
| `SparseBitParallel64`| GPU | Push | 64 本ビット並列＋動的キュー集約フィルタリング | 小〜中規模バッチ ($S \le 16$) で強力 |
| `MultiWordBitParallel`| GPU | Pull | 2D ワープタイル ($32 \times 4$)、マルチワード ($S \le 2048$) | **無向高直径網（道路網）の大規模バッチで最速** |
| `IndirectMacroBatch` | GPU | Pull | `dispatch_workgroups_indirect` による GPU 内完結反復 | ホスト・デバイス間通信の極小化 |

---

## 3. プロジェクトで得られた重要な技術的知見 (Learnings & Insights)

### 3.1 グラフ形状と探索方向（Push vs Pull）の適合性
- **道路網・高直径グラフ (`luxembourg_osm`, $D \approx 1,200$)**:
  - フロンティアが常に狭く長く伸びるため、GPU スレッドの過半数が空振りする問題が生じる。
  - 大規模マルチソース（$S \ge 256$）で多数の始点を束ねると、**Pull 走査（`MultiWordBitParallel`）**が圧倒的に有利になる。
  - 理由: Pull 方式では各頂点が入エッジを集約して自身の距離・フロンティアを単一スレッドで決定的に書き込むため、**アトミック競合が完全にゼロ**であり、128 バイト合体メモリアクセス（Coalesced Memory Access）が最大効率で働く。
- **Web グラフ・有向べき乗則グラフ (`web-Stanford`, $D \approx 200$)**:
  - 道路網とは対照的に、**Push 方式（`BitParallel64`）が Pull 方式を $4\times$ 以上圧倒**（816ms vs 3,345ms）。
  - 理由: 有向 Web グラフには入次数が 0 や極端に小さいノードが多数存在し、Pull 走査では空振りの CSC 走査負荷が激増する。一方、Push 方式はアクティブなフロンティアの出エッジのみを走査するため、枝刈りが極めて効率的に機能する。

### 3.2 WebGPU / WGSL のハードウェア・API 制約と対処パターン
WebGPU は高いポータビリティを持つ反面、ネイティブ CUDA 等に比べて厳格な制約が存在します。本プロジェクトで確立した回避パターンは以下の通りです。

1. **`max_compute_workgroups_per_dimension = 65535` の壁**:
   - **事象**: 頂点数 $|V| > 65,535 \times W_v$ のグラフ（例: `web-Stanford` $|V|=281,903$）において、1D ディスパッチを行うと WebGPU バリデーションエラーでパニックする。
   - **解決策 (2D Grid Dispatch)**:
     $$\text{dispatch\_x} = 256, \quad \text{dispatch\_y} = \left\lceil \frac{\text{total\_wgs}}{256} \right\rceil$$
     シェーダー内で `let wg_id = group_id.x + group_id.y * params.dispatch_x;`、`let v = wg_id * W_v + local_id.y;` として頂点インデックスを復元。これにより $65535 \times 65535 \approx 42.9$ 億頂点まで安全にディスパッチ可能。
2. **単一ストレージバッファ上限 (`max_storage_buffer_binding_size`)**:
   - **事象**: デフォルト値は 128 MB（Elevated Limits で 2 GB）。大規模マルチソースではグラフ構造（$4(|V| + |E|)$）より**出力距離行列（$4S|V|$）**が支配的になり、128 MB 制限下では $S=2048$ 時にバッファ OOM となる。
   - **解決策 (マクロチャンク自動分割)**:
     `determine_macro_chunk_size` によりデバイス制限から最大安全始点数（例: 64 や 256）を自動算出し、チャンク単位で GPU 実行・ホスト退避をストリーミング化。
3. **ホスト・デバイス同期ストール (Host-Device Synchronization Stall)**:
   - **事象**: 直径 1,000 ホップを超えるグラフでホップごとに `queue.submit` とバッファ読み出し（早期終了チェック）を行うと、PCIe/ドライバーレイテンシが蓄積し実行時間の 90% 以上が待機時間となる。
   - **解決策 (投機的マクロバッチング & 間接ディスパッチ)**:
     - 64 ホップ単位でコマンドバッファを一括記録・単一提出。
     - WebGPU の `dispatch_workgroups_indirect` と GPU 側のアトミックフラグを用い、フロンティアが空になった時点で GPU 側で自動的にワークグループ数を 0 に落として早期終了。

---

## 4. 扱えるグラフ規模のハードウェア限界モデル (Scaling Limits)

本環境（16GB RAM, RTX 3050 Laptop 4GB VRAM / WSL2 llvmpipe）における実効限界：

| 探索設定 | ボトルネック要因 | 道路網型 ($|E| \approx 3|V|$) | ソーシャル・Web型 ($|E| \approx 16|V|$) |
| :--- | :--- | :--- | :--- |
| **単一始点 ($S=1$)** | VRAM 容量 (実効 3.5 GB) | **$|V| \approx 1.8 \text{億}, |E| \approx 5.4 \text{億}$** | **$|V| \approx 5,000 \text{万}, |E| \approx 8.0 \text{億}$** |
| **中規模マルチソース ($S=64$)** | 距離バッファ + CSR | **$|V| \approx 1,200 \text{万}, |E| \approx 3,600 \text{万}$** | **$|V| \approx 1,000 \text{万}, |E| \approx 1.6 \text{億}$** |
| **大規模マルチソース ($S=2048$, 一括)** | 単一バッファ 2GB 制限 | **$|V| \le 262,144$ 頂点** | **$|V| \le 262,144$ 頂点** |
| **大規模マルチソース ($S=2048$, 分割)** | システム RAM (10GB 空き) | **$|V| \approx 120 \text{万}, |E| \approx 360 \text{万}$** | **$|V| \approx 100 \text{万}, |E| \approx 1,600 \text{万}$** |
| **CPU 単一始点 ($S=1$)** | システム RAM (10GB 空き) | **$|V| \approx 5.0 \text{億}, |E| \approx 15 \text{億}$** | **$|V| \approx 1.4 \text{億}, |E| \approx 22 \text{億}$** |

---

## 5. 今後の方針とロードマップ (Future Roadmap)

### Phase 3: 次期最適化および機能拡張テーマ

1. **距離配列の 16 ビット量子化 (`u16` Distance)**:
   - ホップ数が 65,535 以内の一般的なグラフにおいて、距離配列の要素サイズを 4 バイト（`u32`）から 2 バイト（`u16`）に半減。
   - 効果: メモリ帯域消費を半減させ、単一バッファで扱える頂点数を **2倍（約 52 万頂点）** に拡大。
2. **到達可能性（Reachability）特化モードの拡張**:
   - 最短ホップ数が不要で「到達可能か否か（連結成分・影響範囲解析）」のみを求める場合、距離配列を排除しビットマスク（$\frac{S}{8}$ バイト/頂点）のみを出力。
   - 効果: $S=2048$ でも頂点あたりわずか 256 バイトとなり、**数千万頂点クラスのグラフに対する超大規模バッチ解析**が可能に。
3. **Out-of-Core / エッジ分割ストリーミング**:
   - VRAM 上限（3.5 GB）を超える超巨大グラフ（全米道路網 USA-road-d など）を扱うため、CSR エッジ配列をブロック分割し、ホスト RAM からストリーミング走査するパイプラインの設計。
4. **トポロジー特性に基づく Push/Pull 自動選定機構**:
   - グラフ読み込み時に出次数・入次数の分散（歪度）および対称性を事前評価し、有向スパース網なら Push 型ビット並列、無向高直径網なら MultiWord Pull 型を自動選択するオプティマイザの導入。
5. **WASM / Web フロントエンドのデモ・可視化強化**:
   - ブラウザ上で Matrix Market ファイルをドラッグ＆ドロップし、WebGPU Compute Shader の実行時間や MTEPS、探索波面の広がりをリアルタイム Canvas 描画するインタラクティブデモ UI の整備。

---

## 6. コントリビューション & 開発の指針 (Agent Guidelines)

- **パリティ検証の絶対死守**:
  新アルゴリズムや最適化シェーダーを追加した際は、必ず `cargo test` および `bfs_cli verify` により、全頂点・全始点で CPU リファレンスとホップ数が 100% 一致することを検証すること。
- **WebGPU ポータビリティの維持**:
  WGSL コードおよびバインディング設計において、特定のベンダー拡張に依存せず、標準 WebGPU 仕様（および Elevated Limits の安全なフォールバック）を維持すること。
- **2D グリッドディスパッチの徹底**:
  頂点単位のコンピュートシェーダーを設計する際は、将来の大規模グラフ適用に備え、65535 を超えるワークグループを安全にディスパッチできる 2D グリッドマッピングを採用すること。
