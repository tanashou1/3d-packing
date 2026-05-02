#!/usr/bin/env python3
"""検証済みベンチマーク結果をGitHub Pagesビューア用docs/assetsへ反映する。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


DATASET_DEFAULTS = {
    "abc": {
        "samples": Path("samples/abc"),
        "validation": Path("samples/abc/validation.json"),
        "title_prefix": "ABC Dataset",
        "source": "HuggingFace上のABC Dataset STEPサブセットから生成",
        "result_prefix": "abc",
    },
    "thingi10k": {
        "samples": Path("samples/thingi10k"),
        "validation": Path("samples/thingi10k/validation.json"),
        "title_prefix": "Thingi10K",
        "source": "Thingi10K公式APIとHuggingFace raw_meshes",
        "result_prefix": "thingi",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="abc")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--case", action="append", help="公開するケース。未指定なら全ケース")
    parser.add_argument("--title-prefix")
    parser.add_argument("--source")
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    samples = args.samples or defaults["samples"]
    validation_path = args.validation or defaults["validation"]
    title_prefix = args.title_prefix or defaults["title_prefix"]
    source = args.source or defaults["source"]
    result_prefix = defaults["result_prefix"]

    if not validation_path.is_file():
        raise FileNotFoundError(f"検証JSONが見つかりません: {validation_path}")
    results = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_cases = args.case or sorted(results)

    docs_assets = args.docs / "assets"
    packed_dir = docs_assets / "packed"
    dataset_dir = docs_assets / args.dataset
    packed_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    published = []
    for case_name in selected_cases:
        if case_name not in results:
            raise KeyError(f"{validation_path} にケースがありません: {case_name}")
        result = results[case_name]
        output_path = Path(result["output"])
        if not output_path.is_file():
            raise FileNotFoundError(f"{case_name} のSTL出力が見つかりません: {output_path}")

        result_id = f"{result_prefix}-{slug(case_name)}"
        stl_name = f"{result_id}.stl"
        shutil.copy2(output_path, packed_dir / stl_name)

        attribution = copy_attribution(samples, dataset_dir, case_name)
        entry = result_entry(
            result_id,
            case_name,
            result,
            title_prefix,
            source,
            stl_name,
            attribution,
            args.dataset,
        )
        published.append(entry)

    shutil.copy2(validation_path, dataset_dir / "validation.json")
    update_results_json(
        docs_assets / "results.json",
        result_prefix,
        published,
        replace_all=args.case is None,
    )
    print(f"{len(published)}件の{args.dataset}結果を {docs_assets} へ反映しました")


def copy_attribution(samples: Path, dataset_dir: Path, case_name: str) -> str | None:
    source_path = samples / case_name / "attribution.json"
    if not source_path.is_file():
        return None
    out_name = f"{case_name}-attribution.json"
    shutil.copy2(source_path, dataset_dir / out_name)
    return f"assets/{dataset_dir.name}/{out_name}"


def result_entry(
    result_id: str,
    case_name: str,
    result: dict,
    title_prefix: str,
    source: str,
    stl_name: str,
    attribution: str | None,
    dataset: str,
) -> dict:
    tray = result["tray"]
    entry = {
        "id": result_id,
        "title": f"{title_prefix} {human_case_name(case_name)}",
        "stl": f"assets/packed/{stl_name}",
        "source": source,
        "packedObjects": result["packed_objects"],
        "inputObjects": result["input_objects"],
        "tray": " x ".join(format_number(value) for value in tray),
        "voxel": result["voxel"],
        "gridVoxels": grid_voxels(tray, result["voxel"]),
        "voxelDensity": result["voxel_density_percent"],
        "meshDensity": result["mesh_density_percent"],
        "rayDisassembly": result["ray_disassembly"],
        "requiresStackingByBbox": result["requires_stacking_by_bbox"],
        "sumBboxFootprint": round(result["sum_bbox_xy_footprint"], 2),
        "trayFootprint": round(result["tray_footprint_area"], 2),
        "sumBboxVolume": round(result["sum_bbox_volume"], 2),
        "trayVolume": round(result["tray_volume"], 2),
    }
    if attribution is not None:
        entry["attribution"] = attribution
    if result.get("timed_out"):
        entry["note"] = "時間制限で打ち切った部分結果"
    elif not result.get("completed", False):
        entry["note"] = "全物体は未配置の参考結果"
    if dataset == "abc":
        entry["source"] = (
            f"{source}。ABC Datasetの配布条件に従い、元データ本体は同梱しません"
        )
    return entry


def update_results_json(
    path: Path,
    dataset: str,
    published: list[dict],
    *,
    replace_all: bool,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    else:
        existing = []
    prefix = f"{dataset}-"
    published_ids = {entry["id"] for entry in published}
    if replace_all:
        keep = [entry for entry in existing if not entry.get("id", "").startswith(prefix)]
    else:
        keep = [entry for entry in existing if entry.get("id") not in published_ids]
    keep.extend(sorted(published, key=lambda entry: entry["id"]))
    path.write_text(json.dumps(keep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    return value.lower().replace("_", "-")


def human_case_name(value: str) -> str:
    if value.startswith("abc_"):
        value = value.removeprefix("abc_")
    return value.replace("_", " ")


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def grid_voxels(tray: list[float], voxel: float) -> int:
    nx = math.ceil(tray[0] / voxel)
    ny = math.ceil(tray[1] / voxel)
    nz = math.ceil(tray[2] / voxel)
    return nx * ny * nz


if __name__ == "__main__":
    main()
