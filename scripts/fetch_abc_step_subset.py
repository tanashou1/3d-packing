#!/usr/bin/env python3
"""HuggingFace上の軽量ABC STEPサブセットを取得し、STLベンチケースを作る。

ABC Dataset本体のtri-mesh/OBJチャンクは数GB単位のため、Pagesで確認する
小さな実データケースとして、整理済みHuggingFaceデータセットのSTEPを
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
    parser.add_argument(
        "--gmsh-timeout-seconds",
        type=float,
        default=45,
        help="1 STEPあたりのgmsh変換タイムアウト秒数",
    )
    parser.add_argument("--case-name", default=CASE_NAME)
    parser.add_argument("--count", type=int, default=CASE_CONFIG["count"])
    parser.add_argument("--target-max-dim", type=float, default=CASE_CONFIG["target_max_dim"])
    parser.add_argument("--voxel", type=float, default=CASE_CONFIG["voxel"])
    parser.add_argument(
        "--footprint-fraction", type=float, default=CASE_CONFIG["footprint_fraction"]
    )
    parser.add_argument("--max-faces", type=int, default=2500)
    parser.add_argument(
        "--min-faces",
        type=int,
        default=0,
        help="変換後STLの面数がこの値未満なら単純すぎるものとしてスキップ",
    )
    parser.add_argument(
        "--min-source-faces",
        type=int,
        default=0,
        help="ABCメタデータ上のCAD face数がこの値未満ならスキップ",
    )
    parser.add_argument(
        "--max-source-faces",
        type=int,
        help="ABCメタデータ上のCAD face数がこの値を超えるならスキップ",
    )
    parser.add_argument(
        "--complexity",
        choices=["simple", "complex", "all"],
        default="simple",
        help="HuggingFace上のSTEP分類。complexは製造業サンプル向けに細部が多い候補を優先する",
    )
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="APIで列挙した候補の先頭からスキップする件数",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="HuggingFace STEPを何件まで試すか",
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
    source_paths = (
        STEP_FILES
        if args.fixed_micro_list
        else discover_step_files(args.complexity, args.candidate_offset, args.max_candidates)
    )
    for source_path in source_paths:
        if len(metadata) >= args.count:
            break
        source_metadata = fetch_source_metadata(source_path)
        source_faces = source_metadata.get("face_count")
        if source_faces is not None and source_faces < args.min_source_faces:
            print(f"{Path(source_path).name}: CAD face数{source_faces}が少ないためスキップしました")
            continue
        if (
            args.max_source_faces is not None
            and source_faces is not None
            and source_faces > args.max_source_faces
        ):
            print(f"{Path(source_path).name}: CAD face数{source_faces}のためスキップしました")
            continue
        step_path = step_dir / Path(source_path).name
        raw_stl_path = raw_stl_dir / f"{step_path.stem}.stl"
        download(source_path, step_path)
        try:
            if not raw_stl_path.exists():
                convert_step_to_stl(
                    args.gmsh,
                    step_path,
                    raw_stl_path,
                    args.gmsh_timeout_seconds,
                )
            triangles = fetch_stl(raw_stl_path.resolve().as_uri())
        except Exception as error:
            print(f"{step_path.name}: 変換をスキップしました: {error}")
            raw_stl_path.unlink(missing_ok=True)
            continue
        if len(triangles) > args.max_faces:
            print(f"{step_path.name}: 面数{len(triangles)}のためスキップしました")
            raw_stl_path.unlink(missing_ok=True)
            continue
        if len(triangles) < args.min_faces:
            print(f"{step_path.name}: 面数{len(triangles)}が少ないためスキップしました")
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
                "step_category": step_category(source_path),
                "source_face_count": source_faces,
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


def discover_step_files(
    complexity: str,
    candidate_offset: int,
    max_candidates: int,
) -> list[str]:
    if candidate_offset < 0:
        raise ValueError("--candidate-offset は0以上にしてください")
    if max_candidates <= 0:
        raise ValueError("--max-candidates は正の整数にしてください")
    with urllib.request.urlopen(DATASET_API_URL, timeout=120) as response:
        data = json.load(response)
    prefixes = {
        "simple": ["step_files/simple/"],
        "complex": ["step_files/complex/"],
        "all": ["step_files/complex/", "step_files/simple/"],
    }[complexity]
    paths = [
        item["rfilename"]
        for item in data["siblings"]
        if any(item["rfilename"].startswith(prefix) for prefix in prefixes)
        and item["rfilename"].endswith(".step")
    ]
    return paths[candidate_offset : candidate_offset + max_candidates]


def step_category(source_path: str) -> str:
    parts = source_path.split("/")
    if len(parts) >= 2 and parts[0] == "step_files":
        return parts[1]
    return "unknown"


def fetch_source_metadata(source_path: str) -> dict:
    stem = Path(source_path).with_suffix(".json").name
    metadata_path = f"metadata/{stem}"
    url = f"{DATASET_BASE_URL}/{metadata_path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


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


def convert_step_to_stl(
    gmsh: str,
    step_path: Path,
    stl_path: Path,
    timeout_seconds: float,
) -> None:
    command = [gmsh, str(step_path), "-3", "-format", "stl", "-o", str(stl_path), "-v", "1"]
    if shutil.which("timeout") is not None:
        command = [
            "timeout",
            "--kill-after=5s",
            f"{timeout_seconds:g}s",
            *command,
        ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds + 10,
    )


if __name__ == "__main__":
    main()
