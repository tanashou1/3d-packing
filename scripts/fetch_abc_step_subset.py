#!/usr/bin/env python3
"""HuggingFace上の軽量ABC STEPサブセットを取得し、STLベンチケースを作る。

ABC Dataset本体のtri-mesh/OBJチャンクは数GB単位のため、Pagesで確認する
小さな実データケースとして、整理済みHuggingFaceデータセットのsimple STEPを
固定またはAPIから列挙したリストで取得する。STEPからSTLへの変換には
`gmsh` コマンドを使う。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from fetch_thingi10k_cases import (
    fetch_stl,
    normalize_triangles,
    safe_name,
    triangle_bbox,
    write_ascii_stl,
)


DATASET_BASE_URL = (
    "https://huggingface.co/datasets/turiya-ai/abc-cad-dataset-organized/resolve/main"
)
DATASET_API_URL = "https://huggingface.co/api/datasets/turiya-ai/abc-cad-dataset-organized"
DATASET_SOURCE = "turiya-ai/abc-cad-dataset-organized"
CASE_NAME = "abc_micro"
CASE_CONFIG = {
    "count": 6,
    "target_max_dim": 10.0,
    "voxel": 2.5,
    "footprint_fraction": 0.50,
}

STEP_FILES = [
    "step_files/simple/00000007_b33a147f86da49879455d286_step_000.step",
    "step_files/simple/00000009_9b3d6a97e8de4aa193b81000_step_001.step",
    "step_files/simple/00000022_ad34a3f60c4a4caa99646600_step_002.step",
    "step_files/simple/00000061_767e4372b5f94a88a7a17d90_step_002.step",
    "step_files/simple/00000063_767e4372b5f94a88a7a17d90_step_004.step",
    "step_files/simple/00000102_5ed74bccca6f4e89829bcb5e_step_002.step",
    "step_files/simple/00000119_ce24e0cbec8e45c89735d148_step_011.step",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/abc"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="STEP/STLの中間ファイル保存先。未指定なら一時ディレクトリを使う。",
    )
    parser.add_argument("--gmsh", default="gmsh")
    parser.add_argument("--case-name", default=CASE_NAME)
    parser.add_argument("--count", type=int, default=CASE_CONFIG["count"])
    parser.add_argument("--target-max-dim", type=float, default=CASE_CONFIG["target_max_dim"])
    parser.add_argument("--voxel", type=float, default=CASE_CONFIG["voxel"])
    parser.add_argument(
        "--footprint-fraction", type=float, default=CASE_CONFIG["footprint_fraction"]
    )
    parser.add_argument("--max-faces", type=int, default=2500)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="HuggingFace simple STEPを先頭から何件まで試すか",
    )
    parser.add_argument(
        "--fixed-micro-list",
        action="store_true",
        help="従来のabc_micro固定STEPリストだけを使う",
    )
    args = parser.parse_args()

    ensure_gmsh(args.gmsh)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="abc-step-subset-") as tmp:
            build_case(args, Path(tmp))
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        build_case(args, args.work_dir)


def ensure_gmsh(gmsh: str) -> None:
    if shutil.which(gmsh) is None:
        raise FileNotFoundError(
            "`gmsh` が見つかりません。例: sudo apt-get install -y gmsh"
        )


def build_case(args: argparse.Namespace, work_dir: Path) -> None:
    if args.count <= 0:
        raise ValueError("--count は正の整数にしてください")

    output = args.output
    case_dir = output / args.case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    step_dir = work_dir / "step"
    raw_stl_dir = work_dir / "stl"
    step_dir.mkdir(parents=True, exist_ok=True)
    raw_stl_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    source_paths = STEP_FILES if args.fixed_micro_list else discover_step_files(args.max_candidates)
    for source_path in source_paths:
        if len(metadata) >= args.count:
            break
        step_path = step_dir / Path(source_path).name
        raw_stl_path = raw_stl_dir / f"{step_path.stem}.stl"
        download(source_path, step_path)
        try:
            convert_step_to_stl(args.gmsh, step_path, raw_stl_path)
            triangles = fetch_stl(raw_stl_path.resolve().as_uri())
        except Exception as error:
            print(f"{step_path.name}: 変換をスキップしました: {error}")
            raw_stl_path.unlink(missing_ok=True)
            continue
        if len(triangles) > args.max_faces:
            print(f"{step_path.name}: 面数{len(triangles)}のためスキップしました")
            raw_stl_path.unlink(missing_ok=True)
            continue

        normalized = normalize_triangles(triangles, args.target_max_dim)
        bbox = triangle_bbox(normalized)
        out_name = f"{len(metadata):03d}_{safe_name(step_path.stem)}.stl"
        out_path = case_dir / out_name
        write_ascii_stl(out_path, f"{args.case_name}_{len(metadata):03d}", normalized)

        metadata.append(
            {
                "source": f"{DATASET_BASE_URL}/{source_path}",
                "source_dataset": DATASET_SOURCE,
                "source_format": "step",
                "converted_with": "gmsh",
                "num_faces_converted": len(triangles),
                "normalized_max_dim": args.target_max_dim,
                "bbox": {
                    "min": bbox[0],
                    "max": bbox[1],
                    "extent": [bbox[1][axis] - bbox[0][axis] for axis in range(3)],
                },
                "local_file": out_path.name,
                "dataset": "ABC Dataset",
                "license_note": (
                    "ABC DatasetのCADモデル著作権は作成者に帰属します。"
                    "HuggingFaceミラーのライセンス表示とOnshape Terms of Useに従ってください。"
                ),
            }
        )
        print(f"{args.case_name}: {out_path} を書き出しました")

    if len(metadata) < args.count:
        raise RuntimeError(f"{args.case_name} に必要な{args.count}個を生成できませんでした")

    (case_dir / "attribution.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_json_map(output / "manifest.json", args.case_name, metadata)
    update_json_map(
        output / "case_config.json",
        args.case_name,
        {
            "voxel": args.voxel,
            "footprint_fraction": args.footprint_fraction,
        },
    )


def discover_step_files(max_candidates: int) -> list[str]:
    with urllib.request.urlopen(DATASET_API_URL, timeout=120) as response:
        data = json.load(response)
    paths = [
        item["rfilename"]
        for item in data["siblings"]
        if item["rfilename"].startswith("step_files/simple/")
        and item["rfilename"].endswith(".step")
    ]
    return paths[:max_candidates]


def update_json_map(path: Path, key: str, value: object) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[key] = value
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def download(source_path: str, destination: Path) -> None:
    if destination.exists():
        return
    url = f"{DATASET_BASE_URL}/{source_path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def convert_step_to_stl(gmsh: str, step_path: Path, stl_path: Path) -> None:
    subprocess.run(
        [gmsh, str(step_path), "-3", "-format", "stl", "-o", str(stl_path), "-v", "1"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
