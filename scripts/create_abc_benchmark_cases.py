#!/usr/bin/env python3
"""ローカルのABC Dataset OBJ/STLからパッキング用ベンチマークケースを作る。

ABC Datasetは巨大なため、このスクリプトはダウンロードを行わない。
ユーザーが取得・展開したOBJ/STL群から、低〜中面数のモデルを決定的に選び、
扱いやすいスケールへ正規化したASCII STLケースを生成する。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from fetch_thingi10k_cases import (
    fetch_stl,
    normalize_triangles,
    safe_name,
    triangle_bbox,
    write_ascii_stl,
)


CASES = {
    "abc_micro": {
        "count": 6,
        "target_max_dim": 10.0,
        "min_faces": 4,
        "max_faces": 600,
        "voxel": 2.5,
        "footprint_fraction": 0.50,
    },
    "abc_small": {
        "count": 12,
        "target_max_dim": 9.0,
        "min_faces": 20,
        "max_faces": 1_500,
        "voxel": 2.5,
        "footprint_fraction": 0.42,
    },
    "abc_mixed": {
        "count": 24,
        "target_max_dim": 8.0,
        "min_faces": 50,
        "max_faces": 3_000,
        "voxel": 2.0,
        "footprint_fraction": 0.34,
    },
}

Vec3 = tuple[float, float, float]
Triangle = tuple[Vec3, Vec3, Vec3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="展開済みABC DatasetのOBJ/STLディレクトリ",
    )
    parser.add_argument("--output", type=Path, default=Path("samples/abc"))
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="生成するケース。複数回指定できる。未指定なら全ケースを生成する。",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"ABC Datasetディレクトリが見つかりません: {args.source}")

    source_files = discover_mesh_files(args.source)
    if not source_files:
        raise FileNotFoundError(f"OBJ/STLファイルが見つかりません: {args.source}")

    selected_cases = args.case or sorted(CASES)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    case_config = {}
    used_sources: set[Path] = set()

    for case_name in selected_cases:
        case = CASES[case_name]
        case_dir = args.output / case_name
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        metadata = build_case(case_name, case, source_files, used_sources, case_dir)
        (case_dir / "attribution.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest[case_name] = metadata
        case_config[case_name] = {
            "voxel": case["voxel"],
            "footprint_fraction": case["footprint_fraction"],
        }
        print(f"{case_name}: {len(metadata)}個のABCモデルを {case_dir} に書き出しました")

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "case_config.json").write_text(
        json.dumps(case_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def discover_mesh_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".obj", ".stl"}
    )


def build_case(
    case_name: str,
    case: dict,
    source_files: list[Path],
    used_sources: set[Path],
    case_dir: Path,
) -> list[dict]:
    metadata = []
    for source_path in source_files:
        if source_path in used_sources:
            continue
        try:
            triangles = read_mesh(source_path)
        except Exception as error:
            print(f"{source_path}: 読み込みをスキップしました: {error}")
            continue

        face_count = len(triangles)
        if not (case["min_faces"] <= face_count <= case["max_faces"]):
            continue

        normalized = normalize_triangles(triangles, case["target_max_dim"])
        bbox = triangle_bbox(normalized)
        out_name = f"{len(metadata):03d}_{safe_name(source_path.stem)}.stl"
        out_path = case_dir / out_name
        write_ascii_stl(out_path, f"{case_name}_{len(metadata):03d}", normalized)
        metadata.append(
            {
                "source": str(source_path),
                "source_format": source_path.suffix.lower().lstrip("."),
                "num_faces_original": face_count,
                "normalized_max_dim": case["target_max_dim"],
                "bbox": {
                    "min": bbox[0],
                    "max": bbox[1],
                    "extent": [bbox[1][axis] - bbox[0][axis] for axis in range(3)],
                },
                "local_file": out_path.name,
                "dataset": "ABC Dataset",
                "license_note": "ABC Datasetの配布条件に従ってください",
            }
        )
        used_sources.add(source_path)

        if len(metadata) >= case["count"]:
            return metadata

    raise RuntimeError(
        f"{case_name} に必要な{case['count']}個のモデルを選べませんでした。"
        f"条件を満たした数={len(metadata)}"
    )


def read_mesh(path: Path) -> list[Triangle]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".stl":
        return fetch_stl(path.resolve().as_uri())
    raise ValueError(f"未対応の拡張子です: {path.suffix}")


def read_obj(path: Path) -> list[Triangle]:
    vertices: list[Vec3] = []
    triangles: list[Triangle] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                face = [parse_obj_index(token, len(vertices)) for token in parts[1:]]
                if any(index < 0 or index >= len(vertices) for index in face):
                    raise ValueError(f"範囲外のface indexがあります: {path}")
                first = vertices[face[0]]
                for i in range(1, len(face) - 1):
                    triangles.append((first, vertices[face[i]], vertices[face[i + 1]]))
    if not triangles:
        raise ValueError("三角形を読み取れませんでした")
    return [tri for tri in triangles if triangle_area(tri) > 1.0e-12]


def parse_obj_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    index = int(raw)
    if index > 0:
        return index - 1
    if index < 0:
        return vertex_count + index
    raise ValueError("OBJ face index 0 は無効です")


def triangle_area(tri: Triangle) -> float:
    a, b, c = tri
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)


if __name__ == "__main__":
    main()
