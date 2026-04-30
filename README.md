# 3d-packing

「汎用3D物体の高密度・非噛み込み・スケーラブルなスペクトルパッキング」論文の主要アイデアをもとにした、Rust製の3D STLパッキングCLIです。このリポジトリは、論文実装より高速であることではなく、指定されたトレイ・衝突・到達可能性などの制約を破らないパック結果を出すことを目的にしています。

現在のパック結果は、GitHub Pagesのビューアから閲覧できます。

https://tanashou1.github.io/3d-packing/

現状ロジックの詳細は [`docs/implementation-details.md`](docs/implementation-details.md) にまとめています。GitHub Pagesのデプロイ運用は [`docs/deployment.md`](docs/deployment.md) にまとめています。

この実装は、論文の配置パイプラインをベースにした制約充足型のパッカーです。性能最適化よりも、実装が扱う制約モデルの中で実行可能な配置だけを採用することを優先しています。

1. 各STLメッシュをボクセル化する。
2. 3D FFT相関で、すべての平行移動候補に対する衝突数を計算する。
3. マンハッタン距離場から近接スコアを作り、これもFFT相関で計算する。
4. 論文の3次高さペナルティを加える。
5. バウンディングボックスが大きい物体から順に、貪欲またはbeam searchで配置する。
6. 必要に応じて、衝突しないオフセット空間をFlood-fillし、境界から平行移動で到達可能な配置だけを採用する。
7. 採用した配置を、負のx/y/z方向への二分探索でサブボクセル単位に詰める。
8. ray-castingのDirectional Blocking Graphで噛み込み候補を検出し、取り外し・再挿入型の後処理で改善を試みる。
9. 最終トレイを再度ray-castingで解析し、直線移動で取り出せる物体を報告する。

姿勢ごとのボクセル化とFFT候補評価はCPU上で並列化しています。GPU高速化は未実装です。取り外し・再挿入型の後処理最適化は実装済みですが、全体最適を保証するものではなく、ray分解スコアが改善する再挿入だけを採用します。現状で厳密に保証する制約の範囲と、今後さらに厳密化すべき制約は詳細ドキュメントに記載しています。

## ビルドとテスト

```bash
cargo test
cargo build --release
```

## GitHub Pagesの公開

可視化サイトは `docs/` を静的サイトとして公開します。現在は `gh-pages` ブランチへ手動同期せず、GitHub Actionsの `.github/workflows/pages.yml` からPagesへデプロイします。

リポジトリ設定の Pages では、Build and deployment の Source を **GitHub Actions** にします。`main` の `docs/` またはPages workflowに変更が入ると、Actionsが `docs/` をartifact化して `github-pages` 環境へ公開します。

ローカルでビューアを確認する例:

```bash
python3 -m http.server 8000 --directory docs
```

## 手続き生成サンプルSTLの作成

`samples/stl` には、小さな手続き生成サンプルが入っています。再生成するには次を実行します。

```bash
cargo run -- sample --output samples/stl
```

## Thingi10Kサンプルケースの作成

`scripts/fetch_thingi10k_cases.py` は、Thingi10K公式APIから選択済みモデルのメタデータを取得し、APIが返すSTLリンクからメッシュをダウンロードします。各STLは扱いやすい小さなスケールへ正規化され、ケースごとにライセンス・出典情報も保存されます。

```bash
python3 scripts/fetch_thingi10k_cases.py --output samples/thingi10k
```

生成されるケース:

| ケース | モデル数 | 目的 |
| --- | ---: | --- |
| `micro` | 6 | 低面数の小さなスモークテスト |
| `mechanical` | 8 | 機械部品、カップ、スピーカー形状などのソリッド |
| `mixed` | 8 | 箱、航空機部品、シャーシ、柔軟メッシュの混合 |
| `stacked_small` | 24 | 積み重ね配置を必要とする小型低面数ソリッド群 |
| `stacked_mixed` | 36 | 多層配置を強制しやすい、やや大きな混合セット |
| `stacked_mixed_10x` | 360 | `stacked_mixed`を10回複製した時間打ち切り・大規模ベンチマーク |

各ケースディレクトリには `attribution.json` が含まれます。トップレベルの
`samples/thingi10k/manifest.json` には、file ID、Thingiverse ID、作者、ライセンス、元の面数、出典URL、正規化後のバウンディングボックスがまとめられています。

10倍ベンチマークケースだけを再生成するには、既存の `stacked_mixed` を取得した後に次を実行します。通常はSTL本体を複製せず、相対シンボリックリンクで作成します。

```bash
python3 scripts/create_thingi10k_benchmark_case.py \
  --samples samples/thingi10k \
  --base-case stacked_mixed \
  --case stacked_mixed_10x \
  --repeat 10
```

## STLファイルのパック

サンプルディレクトリ内のSTLをまとめて1つのSTLへパックする例です。

```bash
cargo run -- pack samples/stl \
  --out target/sample-packed.stl \
  --width 45 --depth 45 --height 35 \
  --voxel 2 \
  --rotations 24
```

主なオプション:

| オプション | 既定値 | 意味 |
| --- | ---: | --- |
| `--width`, `--depth`, `--height` | `80`, `80`, `60` | 直方体トレイの寸法 |
| `--voxel` | `2` | ボクセルの辺長。小さいほど高精度だが遅くなる |
| `--rotations` | `24` | 試す姿勢数。24までは90度刻み、24超では追加の角度姿勢も試す |
| `--height-weight` | `10` | `p * q_z^3` 高さペナルティの係数 |
| `--beam-width` | `1` | 複数の部分配置を残すbeam search幅。`1`なら従来の貪欲配置 |
| `--refine-margin` | `0.05` | サブボクセルrefinement中の三角形AABBクリアランス |
| `--post-opt-passes` | `4` | 取り外し・再挿入後処理の最大パス数 |
| `--repack-passes` | `2` | 未配置物体を入れるための局所再パック最大パス数 |
| `--repack-window` | `8` | 局所再パックで一度に外す配置済み物体数 |
| `--repack-unpacked-limit` | `8` | 局所再パックで一度に試す未配置物体数 |
| `--time-limit-seconds` | なし | 指定秒数を超えたら、その時点で配置済みの部分結果を出力して打ち切る |
| `--no-repack` | off | 未配置物体向け局所再パックを無効化する |
| `--no-post-opt` | off | 取り外し・再挿入後処理を無効化する |
| `--no-refine` | off | 連続サブボクセルrefinementを無効化する |
| `--no-interlock` | off | Flood-fill到達可能性フィルタを無効化する |
| `--no-ray-disassembly` | off | ray-castingによる分解可能性解析を無効化する |

入力には個別のSTLファイル、またはSTLファイルを含むディレクトリを指定できます。出力は、配置に成功した物体を1つにまとめたASCII STLです。

## Thingi10Kのbboxタイト検証

Thingi10Kケースは、各物体のバウンディングボックスから算出したトレイへパックしています。検証スクリプトは、各物体bboxのx/yフットプリント合計よりも小さい底面を意図的に選ぶため、単純な平置きでは満たせない条件になります。機械可読な結果は `samples/thingi10k/validation.json` に保存されます。

| ケース | bboxタイトトレイ | ボクセル | パック数 | ボクセル密度 | トレイ底面 / bbox底面合計 |
| --- | --- | ---: | ---: | ---: | ---: |
| `micro` | `15 x 15 x 15` | `2.5` | 6/6 | 42.59% | 225.00 / 454.94 |
| `mechanical` | `28 x 28 x 21` | `3.5` | 8/8 | 43.49% | 784.00 / 1206.94 |
| `mixed` | `21 x 21 x 24.5` | `3.5` | 8/8 | 50.00% | 441.00 / 567.40 |
| `stacked_small` | `18 x 18 x 20` | `2.0` | 24/24 | 59.01% | 324.00 / 928.69 |
| `stacked_mixed` | `25 x 25 x 22.5` | `2.5` | 36/36 | 63.44% | 625.00 / 1790.90 |
| `stacked_mixed_10x` | `25 x 25 x 87.5` | `2.5` | 131/360 | 74.11% | 625.00 / 17908.99 |

bboxタイトケースを再生成・再検証するには、次を実行します。

```bash
cargo build --release
python3 scripts/fetch_thingi10k_cases.py --output samples/thingi10k
python3 scripts/validate_thingi10k_tight.py \
  --samples samples/thingi10k \
  --output target/thingi10k/tight_bbox
```

10倍ケースを1回だけ、60秒の計算打ち切り付きで検証する例です。`--single-attempt` は、最初に計算したbboxタイトトレイだけを試し、全物体が入らない場合も部分結果を記録します。

```bash
python3 scripts/validate_thingi10k_tight.py \
  --case stacked_mixed_10x \
  --samples samples/thingi10k \
  --output target/thingi10k/benchmark_10x \
  --binary target/release/spectral-packing \
  --beam-width 1 \
  --time-limit-seconds 60 \
  --single-attempt
```
