# 3d-packing

「汎用3D物体の高密度・非噛み込み・スケーラブルなスペクトルパッキング」論文の主要アイデアをもとにした、Rust製の3D STLパッキングCLIです。このリポジトリは、論文実装より高速であることではなく、指定されたトレイ・衝突・到達可能性などの制約を破らないパック結果を出すことを目的にしています。

現在のパック結果は、GitHub Pagesのビューアから閲覧できます。

https://tanashou1.github.io/3d-packing/

現状ロジックの詳細は [`docs/implementation-details.md`](docs/implementation-details.md) にまとめています。

この実装は、論文の配置パイプラインをベースにした制約充足型のパッカーです。性能最適化よりも、実装が扱う制約モデルの中で実行可能な配置だけを採用することを優先しています。

1. 各STLメッシュをボクセル化する。
2. 3D FFT相関で、すべての平行移動候補に対する衝突数を計算する。
3. マンハッタン距離場から近接スコアを作り、これもFFT相関で計算する。
4. 論文の3次高さペナルティを加える。
5. バウンディングボックスが大きい物体から順に貪欲に配置する。
6. 必要に応じて、衝突しないオフセット空間をFlood-fillし、境界から平行移動で到達可能な配置だけを採用する。
7. 採用した配置を、負のx/y/z方向への二分探索でサブボクセル単位に詰める。
8. 最終トレイをray-castingのDirectional Blocking Graphと強連結成分で解析し、直線移動で取り出せる物体を報告する。

GPU高速化と、論文の完全な「取り外し・再挿入」型の後処理最適化は未実装です。現状で厳密に保証する制約の範囲と、今後さらに厳密化すべき制約は詳細ドキュメントに記載しています。

## ビルドとテスト

```bash
cargo test
cargo build --release
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

各ケースディレクトリには `attribution.json` が含まれます。トップレベルの
`samples/thingi10k/manifest.json` には、file ID、Thingiverse ID、作者、ライセンス、元の面数、出典URL、正規化後のバウンディングボックスがまとめられています。

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
| `--rotations` | `24` | 試す90度刻みの右手系姿勢数 |
| `--height-weight` | `10` | `p * q_z^3` 高さペナルティの係数 |
| `--refine-margin` | `0.05` | サブボクセルrefinement中の三角形AABBクリアランス |
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

bboxタイトケースを再生成・再検証するには、次を実行します。

```bash
cargo build --release
python3 scripts/fetch_thingi10k_cases.py --output samples/thingi10k
python3 scripts/validate_thingi10k_tight.py \
  --samples samples/thingi10k \
  --output target/thingi10k/tight_bbox
```
