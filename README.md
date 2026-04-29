# 3d-packing

Rust CLI for dense 3D STL packing based on the core ideas from **Dense,
Interlocking-Free and Scalable Spectral Packing of Generic 3D Objects**.

Current packed outputs can be inspected in the GitHub Pages viewer:

https://tanashou1.github.io/3d-packing/

This is a CPU reference implementation of the paper's main placement
pipeline:

1. Voxelize each STL mesh.
2. Compute collision counts for every translation with 3D FFT correlation.
3. Compute a proximity score from a Manhattan distance field, also through FFT
   correlation.
4. Add the paper's cubic height penalty term.
5. Greedily pack objects from largest bounding box to smallest.
6. Optionally flood-fill collision-free offsets and only accept placements that
   are translationally reachable from the offset-domain boundary, which avoids
   simple interlocking placements.
7. Refine each accepted placement with sub-voxel binary searches along the
   negative x/y/z axes, using mesh triangle AABB checks to keep a clearance
   margin.
8. Analyze the final tray with ray-casting directional blocking graphs and
   strongly connected components to report objects removable by straight-line
   disassembly.

The implementation intentionally leaves out GPU acceleration and the full
remove-and-reinsert post-disassembly optimizer from the paper.

## Build and test

```bash
cargo test
cargo build --release
```

## Generate sample STL files

The repository includes a small generated sample set under `samples/stl`.
Regenerate it with:

```bash
cargo run -- sample --output samples/stl
```

## Generate Thingi10K sample cases

`scripts/fetch_thingi10k_cases.py` downloads selected models through the
official Thingi10K API, normalizes each STL to a small test scale, and writes
license/source attribution next to each case.

```bash
python3 scripts/fetch_thingi10k_cases.py --output samples/thingi10k
```

Generated cases:

| Case | Models | Purpose |
| --- | ---: | --- |
| `micro` | 6 | Very small low-face-count smoke test |
| `mechanical` | 8 | Mechanical/cup/speaker-like solids |
| `mixed` | 8 | Mixed boxes, aircraft parts, chassis, and flexible mesh |
| `stacked_small` | 24 | Many small low-face-count solids for stacked packing |
| `stacked_mixed` | 36 | Larger mixed set designed to force multi-layer placement |

Each case directory contains `attribution.json`. The top-level
`samples/thingi10k/manifest.json` aggregates the same metadata, including
file IDs, Thingiverse IDs, authors, licenses, original face counts, and source
URLs, and normalized bounding boxes.

## Pack STL files

Pack all STL files in the sample directory into one combined STL:

```bash
cargo run -- pack samples/stl \
  --out target/sample-packed.stl \
  --width 45 --depth 45 --height 35 \
  --voxel 2 \
  --rotations 24
```

Useful options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--width`, `--depth`, `--height` | `80`, `80`, `60` | Rectangular tray dimensions in model units |
| `--voxel` | `2` | Voxel edge length; smaller is more accurate but slower |
| `--rotations` | `24` | Number of right-handed 90-degree orientations to sample |
| `--height-weight` | `10` | Coefficient for `p * q_z^3` height penalization |
| `--refine-margin` | `0.05` | Triangle AABB clearance used by sub-voxel refinement |
| `--no-refine` | off | Disable continuous sub-voxel refinement |
| `--no-interlock` | off | Disable flood-fill reachability filtering |
| `--no-ray-disassembly` | off | Disable ray-casting directional blocking analysis |

Inputs can be individual STL files or directories containing STL files. The
output is one ASCII STL containing all successfully packed objects.

## Thingi10K bbox-tight validation

The generated Thingi10K cases are packed with a tray computed from object
bounding boxes. The validator deliberately chooses a tray footprint smaller
than the sum of all per-object bbox x/y footprints, so a simple flat placement
cannot satisfy the case. A machine readable copy of these results is stored in
`samples/thingi10k/validation.json`.

| Case | Bbox-tight tray | Voxel | Packed | Voxel density | Tray footprint vs sum bbox footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| `micro` | `15 x 15 x 15` | `2.5` | 6/6 | 42.59% | 225.00 / 454.94 |
| `mechanical` | `28 x 28 x 21` | `3.5` | 8/8 | 42.19% | 784.00 / 1206.94 |
| `mixed` | `21 x 21 x 24.5` | `3.5` | 8/8 | 50.00% | 441.00 / 567.40 |
| `stacked_small` | `18 x 18 x 20` | `2.0` | 24/24 | 63.95% | 324.00 / 928.69 |
| `stacked_mixed` | `25 x 25 x 22.5` | `2.5` | 36/36 | 62.56% | 625.00 / 1790.90 |

Regenerate and validate the bbox-tight cases:

```bash
cargo build --release
python3 scripts/fetch_thingi10k_cases.py --output samples/thingi10k
python3 scripts/validate_thingi10k_tight.py \
  --samples samples/thingi10k \
  --output target/thingi10k/tight_bbox
```
