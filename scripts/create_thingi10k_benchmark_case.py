#!/usr/bin/env python3
"""既存Thingi10Kケースを複製して大規模ベンチマークケースを作る。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("samples/thingi10k"))
    parser.add_argument("--base-case", default="stacked_mixed")
    parser.add_argument("--case", default="stacked_mixed_10x")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat は正の整数にしてください")

    base_dir = args.samples / args.base_case
    out_dir = args.samples / args.case
    attribution_path = base_dir / "attribution.json"
    if not base_dir.is_dir():
        raise FileNotFoundError(f"ベースケースが見つかりません: {base_dir}")
    if not attribution_path.is_file():
        raise FileNotFoundError(f"帰属情報が見つかりません: {attribution_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    base_metadata = json.loads(attribution_path.read_text(encoding="utf-8"))
    expanded_metadata = []
    for replica_index in range(args.repeat):
        for entry in base_metadata:
            source_file = base_dir / entry["local_file"]
            out_name = f"r{replica_index:02d}_{entry['local_file']}"
            out_path = out_dir / out_name
            link_or_copy(source_file, out_path)

            expanded = dict(entry)
            expanded["replica_index"] = replica_index
            expanded["replica_of"] = entry["local_file"]
            expanded["local_file"] = out_name
            expanded_metadata.append(expanded)

    (out_dir / "attribution.json").write_text(
        json.dumps(expanded_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_manifest(args.samples, args.case, expanded_metadata)
    print(
        f"{args.case}: {args.base_case} の{len(base_metadata)}個を"
        f"{args.repeat}回複製し、{len(expanded_metadata)}個のケースを作成しました"
    )


def link_or_copy(source: Path, destination: Path) -> None:
    relative_source = os.path.relpath(source, start=destination.parent)
    try:
        destination.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, destination)


def update_manifest(samples: Path, case_name: str, metadata: list[dict]) -> None:
    manifest_path = samples / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest[case_name] = metadata
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
