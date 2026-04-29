#!/usr/bin/env python3
"""Thingi10Kの小さなパッキングケースをダウンロードして正規化する。

公式Thingi10K APIからメタデータを取得し、そのAPIが返す公式リンクからSTLを取得する。
生成したメッシュはパッキング実験向けに正規化し、ASCII STLとして保存する。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import urllib.request
from pathlib import Path


API = "https://ten-thousand-models.appspot.com/api/v2/model/{file_id}"

CASES = {
    "micro": {
        "target_max_dim": 11.0,
        "file_ids": [65585, 65607, 65608, 65609, 65610, 311327],
    },
    "mechanical": {
        "target_max_dim": 13.0,
        "file_ids": [65561, 65564, 98330, 49160, 49163, 73736, 73737, 311322],
    },
    "mixed": {
        "target_max_dim": 12.0,
        "file_ids": [65582, 65583, 65584, 65614, 65615, 49166, 57355, 958471],
    },
    "stacked_small": {
        "target_max_dim": 8.0,
        "file_ids": [
            65585,
            65588,
            311327,
            311329,
            65607,
            65608,
            65609,
            65610,
            98412,
            98413,
            90223,
            90224,
            90225,
            90226,
            57420,
            82058,
            82059,
            123044,
            237735,
            90279,
            90280,
            204952,
            204953,
            60276,
        ],
    },
    "stacked_mixed": {
        "target_max_dim": 9.0,
        "file_ids": [
            49160,
            49163,
            49166,
            49167,
            65561,
            65564,
            65582,
            65583,
            65584,
            65585,
            65586,
            65587,
            237623,
            237626,
            237632,
            237633,
            237634,
            237640,
            65607,
            65608,
            65609,
            65610,
            65614,
            65615,
            188496,
            188497,
            188501,
            65641,
            65642,
            65643,
            65644,
            204954,
            204955,
            204956,
            204957,
            204958,
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/thingi10k"))
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="生成するケース。複数回指定できる。未指定なら全ケースを生成する。",
    )
    args = parser.parse_args()

    selected = args.case or sorted(CASES)
    args.output.mkdir(parents=True, exist_ok=True)
    all_metadata = {}
    for case_name in selected:
        case = CASES[case_name]
        case_dir = args.output / case_name
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        metadata = []
        for file_id in case["file_ids"]:
            model = fetch_json(API.format(file_id=file_id))
            triangles = fetch_stl(model["link"])
            triangles = normalize_triangles(triangles, case["target_max_dim"])
            bbox = triangle_bbox(triangles)
            out_path = case_dir / f"{file_id}_{safe_name(model.get('Name', 'model'))}.stl"
            write_ascii_stl(out_path, f"thingi10k_{file_id}", triangles)
            metadata.append(
                {
                    "file_id": file_id,
                    "thing_id": model.get("thing_id"),
                    "name": model.get("Name"),
                    "author": model.get("Author"),
                    "license": model.get("License"),
                    "source": model.get("link"),
                    "num_faces_original": model.get("num_faces"),
                    "normalized_max_dim": case["target_max_dim"],
                    "bbox": {
                        "min": bbox[0],
                        "max": bbox[1],
                        "extent": [bbox[1][axis] - bbox[0][axis] for axis in range(3)],
                    },
                    "local_file": out_path.name,
                }
            )
            print(f"{case_name}: {out_path} を書き出しました")
        (case_dir / "attribution.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        all_metadata[case_name] = metadata
    (args.output / "manifest.json").write_text(
        json.dumps(all_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_stl(url: str) -> list[tuple[Vec3, Vec3, Vec3]]:
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if is_binary_stl(data):
        return parse_binary_stl(data)
    return parse_ascii_stl(data.decode("utf-8", errors="replace"))


Vec3 = tuple[float, float, float]


def is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    count = struct.unpack_from("<I", data, 80)[0]
    return 84 + count * 50 == len(data)


def parse_binary_stl(data: bytes) -> list[tuple[Vec3, Vec3, Vec3]]:
    count = struct.unpack_from("<I", data, 80)[0]
    triangles = []
    offset = 84
    for _ in range(count):
        offset += 12
        tri = []
        for _ in range(3):
            tri.append(struct.unpack_from("<fff", data, offset))
            offset += 12
        offset += 2
        triangles.append(tuple(tri))
    return triangles


def parse_ascii_stl(text: str) -> list[tuple[Vec3, Vec3, Vec3]]:
    vertices = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(vertices) % 3:
        raise ValueError("ASCII STLの三角形が不完全です")
    return [
        (vertices[i], vertices[i + 1], vertices[i + 2])
        for i in range(0, len(vertices), 3)
    ]


def normalize_triangles(
    triangles: list[tuple[Vec3, Vec3, Vec3]], target_max_dim: float
) -> list[tuple[Vec3, Vec3, Vec3]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for tri in triangles:
        for v in tri:
            for axis in range(3):
                mins[axis] = min(mins[axis], v[axis])
                maxs[axis] = max(maxs[axis], v[axis])
    extent = [maxs[i] - mins[i] for i in range(3)]
    max_dim = max(extent)
    if max_dim <= 0:
        raise ValueError("メッシュのバウンディングボックスがゼロサイズです")
    scale = target_max_dim / max_dim
    normalized = []
    for tri in triangles:
        normalized.append(
            tuple(
                tuple((v[axis] - mins[axis]) * scale for axis in range(3))
                for v in tri
            )
        )
    return normalized


def triangle_bbox(triangles: list[tuple[Vec3, Vec3, Vec3]]) -> tuple[list[float], list[float]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for tri in triangles:
        for v in tri:
            for axis in range(3):
                mins[axis] = min(mins[axis], v[axis])
                maxs[axis] = max(maxs[axis], v[axis])
    return mins, maxs


def write_ascii_stl(path: Path, name: str, triangles: list[tuple[Vec3, Vec3, Vec3]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            normal = triangle_normal(tri)
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def triangle_normal(tri: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    a, b, c = tri
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (n[0] / length, n[1] / length, n[2] / length)


def safe_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:48] or "model"


if __name__ == "__main__":
    main()
