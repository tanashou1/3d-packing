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

Each case directory contains `attribution.json`. The top-level
`samples/thingi10k/manifest.json` aggregates the same metadata, including
file IDs, Thingiverse IDs, authors, licenses, original face counts, and source
URLs.

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

## Thingi10K validation

The generated Thingi10K cases were packed with the release binary. A machine
readable copy of these results is stored in `samples/thingi10k/validation.json`.

| Case | Command-specific tray | Voxel | Packed | Voxel density | Ray disassembly |
| --- | --- | ---: | ---: | ---: | --- |
| `micro` | `24 x 24 x 20` | `2.5` | 6/6 | 11.94% | all removable |
| `mechanical` | `42 x 42 x 32` | `3.5` | 8/8 | 12.53% | all removable |
| `mixed` | `42 x 42 x 32` | `3.5` | 8/8 | 9.57% | all removable |

Example validation command:

```bash
cargo build --release
./target/release/spectral-packing pack samples/thingi10k/micro \
  --out target/thingi10k/tight/micro-24.stl \
  --width 24 --depth 24 --height 20 \
  --voxel 2.5 \
  --rotations 24
```
