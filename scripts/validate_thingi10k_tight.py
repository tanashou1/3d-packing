#!/usr/bin/env python3
"""Thingi10Kケースをbbox由来のタイトなトレイへパックする。

各ケースについて、物体のバウンディングボックスからトレイを計算する。

* 体積下限 = 各物体bbox体積の合計
* 平置き底面 = 各物体bboxのx/yフットプリント合計
* トレイ底面は、平置き底面より意図的に小さくする
* トレイ高さは体積下限から選び、全物体が入るまで必要最小限だけ増やす

結果として、再現可能な検証JSONとパック済みSTLを出力する。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from fetch_thingi10k_cases import fetch_stl


DEFAULT_CASES = {
    "micro": {"voxel": 2.5, "footprint_fraction": 0.48},
    "mechanical": {"voxel": 3.5, "footprint_fraction": 0.46},
    "mixed": {"voxel": 3.5, "footprint_fraction": 0.46},
    "stacked_small": {"voxel": 2.0, "footprint_fraction": 0.34},
    "stacked_mixed": {"voxel": 2.5, "footprint_fraction": 0.32},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("samples/thingi10k"))
    parser.add_argument("--output", type=Path, default=Path("target/thingi10k/tight_bbox"))
    parser.add_argument("--binary", type=Path, default=Path("target/release/spectral-packing"))
    parser.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES))
    parser.add_argument("--rotations", type=int, default=24)
    parser.add_argument("--height-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not args.binary.exists():
        subprocess.run(["cargo", "build", "--release", "--quiet"], check=True)

    cases = args.case or sorted(DEFAULT_CASES)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case_name in cases:
        config = DEFAULT_CASES[case_name]
        case_dir = args.samples / case_name
        if not case_dir.is_dir():
            raise FileNotFoundError(f"ケースディレクトリが見つかりません: {case_dir}")
        stl_paths = sorted(case_dir.glob("*.stl"))
        dims = [bbox_dimensions(path) for path in stl_paths]
        result = validate_case(
            args.binary,
            case_name,
            case_dir,
            stl_paths,
            dims,
            args.output,
            config["voxel"],
            config["footprint_fraction"],
            args.rotations,
            args.height_weight,
        )
        results[case_name] = result
        print(
            f"{case_name}: {result['packed_objects']}/{result['input_objects']}個を "
            f"{result['tray']} にパックしました。底面 {result['tray_footprint_area']:.2f} "
            f"< 平置き {result['sum_bbox_xy_footprint']:.2f}"
        )

    validation_path = args.samples / "validation.json"
    validation_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    (args.output / "validation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )


def validate_case(
    binary: Path,
    case_name: str,
    case_dir: Path,
    stl_paths: list[Path],
    dims: list[tuple[float, float, float]],
    output_dir: Path,
    voxel: float,
    footprint_fraction: float,
    rotations: int,
    height_weight: float,
) -> dict:
    input_count = len(stl_paths)
    attempts = []
    for footprint_step in [0.0, 0.08, 0.16, 0.26, 0.40]:
        for volume_slack in [1.05, 1.15, 1.30, 1.50, 1.80, 2.20, 2.70, 3.30]:
            tray = compute_tight_tray(
                dims,
                voxel,
                footprint_fraction + footprint_step,
                volume_slack,
            )
            out_path = output_dir / f"{case_name}.stl"
            log_path = output_dir / f"{case_name}.log"
            command = [
                str(binary),
                "pack",
                str(case_dir),
                "--out",
                str(out_path),
                "--width",
                f"{tray[0]:.6g}",
                "--depth",
                f"{tray[1]:.6g}",
                "--height",
                f"{tray[2]:.6g}",
                "--voxel",
                f"{voxel:.6g}",
                "--rotations",
                str(rotations),
                "--height-weight",
                f"{height_weight:.6g}",
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            combined_output = completed.stdout + completed.stderr
            log_path.write_text(combined_output)
            parsed = parse_packer_output(combined_output)
            attempts.append(
                {
                    "tray": tray,
                    "volume_slack": volume_slack,
                    "footprint_fraction": footprint_fraction + footprint_step,
                    "packed_objects": parsed["packed_objects"],
                    "return_code": completed.returncode,
                }
            )
            if completed.returncode == 0 and parsed["packed_objects"] == input_count:
                metrics = bbox_metrics(dims)
                return {
                    "command": " ".join(command),
                    "output": str(out_path),
                    "log": str(log_path),
                    "input_objects": input_count,
                    "packed_objects": parsed["packed_objects"],
                    "voxel_density_percent": parsed["voxel_density_percent"],
                    "mesh_density_percent": parsed["mesh_density_percent"],
                    "ray_disassembly": parsed["ray_disassembly"],
                    "tray": [round(v, 4) for v in tray],
                    "voxel": voxel,
                    "volume_slack": volume_slack,
                    "footprint_fraction": footprint_fraction + footprint_step,
                    "sum_bbox_volume": metrics["sum_bbox_volume"],
                    "sum_bbox_xy_footprint": metrics["sum_bbox_xy_footprint"],
                    "tray_volume": tray[0] * tray[1] * tray[2],
                    "tray_footprint_area": tray[0] * tray[1],
                    "requires_stacking_by_bbox": tray[0] * tray[1]
                    < metrics["sum_bbox_xy_footprint"],
                    "max_bbox_extent": metrics["max_bbox_extent"],
                    "attempts": attempts,
                }

    raise RuntimeError(f"{case_name} の全物体をパックできませんでした。試行={attempts}")


def compute_tight_tray(
    dims: list[tuple[float, float, float]],
    voxel: float,
    footprint_fraction: float,
    volume_slack: float,
) -> tuple[float, float, float]:
    metrics = bbox_metrics(dims)
    max_x, max_y, max_z = metrics["max_bbox_extent"]
    target_footprint = max(
        max_x * max_y * 1.08,
        metrics["sum_bbox_xy_footprint"] * footprint_fraction,
    )
    side = math.sqrt(target_footprint)
    width = max(max_x + voxel, side)
    depth = max(max_y + voxel, target_footprint / width)
    target_volume = max(
        metrics["sum_bbox_volume"] * volume_slack,
        width * depth * (max_z + voxel),
    )
    height = max(max_z + voxel, target_volume / (width * depth))
    return (
        round_up(width, voxel),
        round_up(depth, voxel),
        round_up(height, voxel),
    )


def bbox_metrics(dims: list[tuple[float, float, float]]) -> dict:
    sum_bbox_volume = sum(x * y * z for x, y, z in dims)
    sum_bbox_xy_footprint = sum(x * y for x, y, _ in dims)
    max_bbox_extent = [max(d[axis] for d in dims) for axis in range(3)]
    return {
        "sum_bbox_volume": sum_bbox_volume,
        "sum_bbox_xy_footprint": sum_bbox_xy_footprint,
        "max_bbox_extent": max_bbox_extent,
    }


def round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def bbox_dimensions(path: Path) -> tuple[float, float, float]:
    triangles = fetch_stl(path.resolve().as_uri())
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for tri in triangles:
        for vertex in tri:
            for axis in range(3):
                mins[axis] = min(mins[axis], vertex[axis])
                maxs[axis] = max(maxs[axis], vertex[axis])
    return tuple(maxs[axis] - mins[axis] for axis in range(3))


def parse_packer_output(output: str) -> dict:
    packed = re.search(r"packed (\d+) objects", output)
    if not packed:
        packed = re.search(r"(\d+)個の物体を .+ にパックしました", output)
    density = re.search(r"voxel density: ([0-9.]+)% \| mesh density: ([0-9.]+)%", output)
    if not density:
        density = re.search(r"ボクセル密度: ([0-9.]+)% \| メッシュ密度: ([0-9.]+)%", output)
    ray = re.search(r"ray disassembly: (.*)", output)
    if not ray:
        ray = re.search(r"ray分解判定: (.*)", output)
    return {
        "packed_objects": int(packed.group(1)) if packed else 0,
        "voxel_density_percent": float(density.group(1)) if density else None,
        "mesh_density_percent": float(density.group(2)) if density else None,
        "ray_disassembly": ray.group(1) if ray else None,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"エラー: {error}", file=sys.stderr)
        raise
