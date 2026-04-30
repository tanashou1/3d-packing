use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};
use std::env;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

enum Command {
    Pack {
        inputs: Vec<PathBuf>,
        out: PathBuf,
        width: f32,
        depth: f32,
        height: f32,
        voxel: f32,
        rotations: usize,
        height_weight: f32,
        no_interlock: bool,
        no_refine: bool,
        refine_margin: f32,
        no_ray_disassembly: bool,
        post_opt_passes: usize,
        beam_width: usize,
        strategy: PlacementStrategy,
        order_window: usize,
        bl_candidate_limit: usize,
        repack_passes: usize,
        repack_window: usize,
        repack_unpacked_limit: usize,
        time_limit_seconds: Option<f32>,
    },
    Sample {
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

#[derive(Clone)]
struct Mesh {
    name: String,
    triangles: Vec<[Vec3; 3]>,
}

impl Mesh {
    fn bbox(&self) -> (Vec3, Vec3) {
        bbox_of_triangles(&self.triangles)
    }

    fn bbox_volume(&self) -> f32 {
        let (min, max) = self.bbox();
        let e = max - min;
        e.x.max(0.0) * e.y.max(0.0) * e.z.max(0.0)
    }

    fn signed_volume(&self) -> f32 {
        self.triangles
            .iter()
            .map(|tri| tri[0].dot(tri[1].cross(tri[2])) / 6.0)
            .sum::<f32>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rotation {
    m: [[f32; 3]; 3],
}

impl Rotation {
    fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    fn apply(self, v: Vec3) -> Vec3 {
        let c = [v.x, v.y, v.z];
        Vec3::new(
            self.m[0].iter().enumerate().map(|(i, &s)| s * c[i]).sum(),
            self.m[1].iter().enumerate().map(|(i, &s)| s * c[i]).sum(),
            self.m[2].iter().enumerate().map(|(i, &s)| s * c[i]).sum(),
        )
    }

    fn from_euler(rx: f32, ry: f32, rz: f32) -> Self {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let mx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
        let my = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        let mz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
        Self {
            m: mat_mul(mz, mat_mul(my, mx)),
        }
    }
}

#[derive(Clone)]
struct OrientedMesh {
    name: String,
    triangles: Vec<[Vec3; 3]>,
    #[allow(dead_code)]
    occupied: Vec<bool>,
    cells: Vec<(usize, usize, usize)>,
    nx: usize,
    ny: usize,
    nz: usize,
    voxel_count: usize,
}

#[derive(Clone)]
struct PlacedMesh {
    name: String,
    source_index: usize,
    triangles: Vec<[Vec3; 3]>,
    offset: (usize, usize, usize),
    translation: Vec3,
    refinement: Vec3,
    rotation_index: usize,
    voxel_count: usize,
    mesh_volume: f32,
    bbox: (Vec3, Vec3),
    occupied_cells: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Copy)]
struct Tray {
    width: f32,
    depth: f32,
    height: f32,
    voxel: f32,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl Tray {
    fn new(width: f32, depth: f32, height: f32, voxel: f32) -> Result<Self> {
        if width <= 0.0 || depth <= 0.0 || height <= 0.0 || voxel <= 0.0 {
            bail!("トレイ寸法とボクセルサイズは正の値にしてください");
        }
        let nx = (width / voxel).ceil() as usize;
        let ny = (depth / voxel).ceil() as usize;
        let nz = (height / voxel).ceil() as usize;
        if nx == 0 || ny == 0 || nz == 0 {
            bail!("ボクセルグリッドが空です");
        }
        Ok(Self {
            width,
            depth,
            height,
            voxel,
            nx,
            ny,
            nz,
        })
    }

    fn len(self) -> usize {
        self.nx * self.ny * self.nz
    }
}

#[derive(Clone)]
struct Placement {
    oriented: OrientedMesh,
    rotation_index: usize,
    offset: (usize, usize, usize),
    cost: f32,
}

fn main() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        Command::Pack {
            inputs,
            out,
            width,
            depth,
            height,
            voxel,
            rotations,
            height_weight,
            no_interlock,
            no_refine,
            refine_margin,
            no_ray_disassembly,
            post_opt_passes,
            beam_width,
            strategy,
            order_window,
            bl_candidate_limit,
            repack_passes,
            repack_window,
            repack_unpacked_limit,
            time_limit_seconds,
        } => {
            let deadline = Deadline::from_seconds(time_limit_seconds);
            let tray = Tray::new(width, depth, height, voxel)?;
            let input_paths = collect_stl_inputs(&inputs)?;
            if input_paths.is_empty() {
                bail!("STLファイルが見つかりません");
            }
            let mut meshes = input_paths
                .iter()
                .map(|path| load_stl(path))
                .collect::<Result<Vec<_>>>()?;
            meshes.sort_by(|a, b| {
                b.bbox_volume()
                    .partial_cmp(&a.bbox_volume())
                    .unwrap_or(Ordering::Equal)
            });
            let rotations = rotation_set(rotations.max(1));
            let result = pack_meshes(
                &meshes,
                tray,
                &rotations,
                height_weight,
                !no_interlock,
                !no_refine,
                refine_margin,
                !no_ray_disassembly,
                if no_ray_disassembly {
                    0
                } else {
                    post_opt_passes
                },
                beam_width.max(1),
                strategy,
                order_window.max(1),
                bl_candidate_limit.max(1),
                repack_passes,
                repack_window.max(1),
                repack_unpacked_limit.max(1),
                deadline,
            )?;
            write_combined_stl(&out, &result.placed)
                .with_context(|| format!("{} の書き込みに失敗しました", out.display()))?;
            print_summary(&result, tray, &out);
        }
        Command::Sample { output } => {
            write_sample_set(&output)?;
            println!("サンプルSTLを {} に書き出しました", output.display());
        }
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        bail!("{}", usage());
    };
    match command.as_str() {
        "pack" => parse_pack_args(args.collect()),
        "sample" => parse_sample_args(args.collect()),
        "-h" | "--help" | "help" => bail!("{}", usage()),
        _ => bail!("不明なコマンドです: `{command}`\n\n{}", usage()),
    }
}

fn parse_pack_args(args: Vec<String>) -> Result<Command> {
    let mut inputs = Vec::new();
    let mut out = PathBuf::from("packed.stl");
    let mut width = 80.0;
    let mut depth = 80.0;
    let mut height = 60.0;
    let mut voxel = 2.0;
    let mut rotations = 24;
    let mut height_weight = 10.0;
    let mut no_interlock = false;
    let mut no_refine = false;
    let mut refine_margin = 0.05;
    let mut no_ray_disassembly = false;
    let mut post_opt_passes = 4;
    let mut beam_width = 1;
    let mut strategy = PlacementStrategy::Spectral;
    let mut order_window = 12;
    let mut bl_candidate_limit = 256;
    let mut repack_passes = 2;
    let mut repack_window = 8;
    let mut repack_unpacked_limit = 8;
    let mut time_limit_seconds = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out = PathBuf::from(required_arg(&args, i, "--out")?);
            }
            "--width" => {
                i += 1;
                width = parse_value(&args, i, "--width")?;
            }
            "--depth" => {
                i += 1;
                depth = parse_value(&args, i, "--depth")?;
            }
            "--height" => {
                i += 1;
                height = parse_value(&args, i, "--height")?;
            }
            "--voxel" => {
                i += 1;
                voxel = parse_value(&args, i, "--voxel")?;
            }
            "--rotations" => {
                i += 1;
                rotations = parse_value(&args, i, "--rotations")?;
            }
            "--height-weight" => {
                i += 1;
                height_weight = parse_value(&args, i, "--height-weight")?;
            }
            "--no-interlock" => no_interlock = true,
            "--no-refine" => no_refine = true,
            "--refine-margin" => {
                i += 1;
                refine_margin = parse_value(&args, i, "--refine-margin")?;
            }
            "--no-ray-disassembly" => no_ray_disassembly = true,
            "--no-post-opt" => post_opt_passes = 0,
            "--post-opt-passes" => {
                i += 1;
                post_opt_passes = parse_value(&args, i, "--post-opt-passes")?;
            }
            "--beam-width" => {
                i += 1;
                beam_width = parse_value(&args, i, "--beam-width")?;
            }
            "--strategy" | "--placement-strategy" => {
                i += 1;
                strategy = PlacementStrategy::parse(&required_arg(&args, i, "--strategy")?)?;
            }
            "--order-window" => {
                i += 1;
                order_window = parse_value(&args, i, "--order-window")?;
            }
            "--bl-candidate-limit" => {
                i += 1;
                bl_candidate_limit = parse_value(&args, i, "--bl-candidate-limit")?;
            }
            "--no-repack" => repack_passes = 0,
            "--repack-passes" => {
                i += 1;
                repack_passes = parse_value(&args, i, "--repack-passes")?;
            }
            "--repack-window" => {
                i += 1;
                repack_window = parse_value(&args, i, "--repack-window")?;
            }
            "--repack-unpacked-limit" => {
                i += 1;
                repack_unpacked_limit = parse_value(&args, i, "--repack-unpacked-limit")?;
            }
            "--time-limit-seconds" | "--timeout-seconds" => {
                i += 1;
                time_limit_seconds = Some(parse_value(&args, i, "--time-limit-seconds")?);
            }
            "-h" | "--help" => bail!("{}", usage()),
            arg if arg.starts_with('-') => bail!("不明なpackオプションです: `{arg}`"),
            arg => inputs.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if inputs.is_empty() {
        bail!(
            "packにはSTLファイルまたはディレクトリを1つ以上指定してください\n\n{}",
            usage()
        );
    }
    Ok(Command::Pack {
        inputs,
        out,
        width,
        depth,
        height,
        voxel,
        rotations,
        height_weight,
        no_interlock,
        no_refine,
        refine_margin,
        no_ray_disassembly,
        post_opt_passes,
        beam_width,
        strategy,
        order_window,
        bl_candidate_limit,
        repack_passes,
        repack_window,
        repack_unpacked_limit,
        time_limit_seconds,
    })
}

fn parse_sample_args(args: Vec<String>) -> Result<Command> {
    let mut output = PathBuf::from("samples/stl");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                output = PathBuf::from(required_arg(&args, i, "--output")?);
            }
            "-h" | "--help" => bail!("{}", usage()),
            arg => bail!("不明なsampleオプションです: `{arg}`"),
        }
        i += 1;
    }
    Ok(Command::Sample { output })
}

fn required_arg(args: &[String], index: usize, option: &str) -> Result<String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| anyhow!("{option} には値が必要です"))
}

fn parse_value<T: std::str::FromStr>(args: &[String], index: usize, option: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    required_arg(args, index, option)?
        .parse::<T>()
        .map_err(|err| anyhow!("{option} の値が不正です: {err}"))
}

fn usage() -> &'static str {
    "使い方:\n  spectral-packing sample [--output DIR]\n  spectral-packing pack [OPTIONS] <STL_OR_DIR>...\n\npackオプション:\n  -o, --out FILE              結合した出力STL（既定値: packed.stl）\n      --width N               トレイ幅（既定値: 80）\n      --depth N               トレイ奥行き（既定値: 80）\n      --height N              トレイ高さ（既定値: 60）\n      --voxel N               ボクセルサイズ（既定値: 2）\n      --rotations N           試す姿勢数。24超で追加角度姿勢も試す（既定値: 24）\n      --height-weight N       高さペナルティ係数（既定値: 10）\n      --strategy NAME         spectral または order-bl（既定値: spectral）\n      --beam-width N          残す部分配置候補数（既定値: 1）\n      --order-window N        order-blで次物体候補として見る未配置物体数（既定値: 12）\n      --bl-candidate-limit N  order-blで評価するBL前線候補数（既定値: 256）\n      --refine-margin N       refinement中の三角形AABBクリアランス（既定値: 0.05）\n      --post-opt-passes N     取り外し・再挿入後処理の最大パス数（既定値: 4）\n      --repack-passes N       未配置物体向け局所再パックの最大パス数（既定値: 2）\n      --repack-window N       局所再パックで一度に外す配置済み物体数（既定値: 8）\n      --repack-unpacked-limit N 局所再パックで一度に試す未配置物体数（既定値: 8）\n      --time-limit-seconds N  指定秒数を超えたら部分結果で打ち切る\n      --no-repack             未配置物体向け局所再パックを無効化\n      --no-post-opt           取り外し・再挿入後処理を無効化\n      --no-refine             連続サブボクセルrefinementを無効化\n      --no-interlock          Flood-fill到達可能性フィルタを無効化\n      --no-ray-disassembly    ray-casting分解可能性解析を無効化"
}

struct PackResult {
    placed: Vec<PlacedMesh>,
    unpacked: Vec<String>,
    ray_report: Option<RayDisassemblyReport>,
    post_report: Option<PostOptimizationReport>,
    timed_out: bool,
}

#[derive(Clone)]
struct RayDisassemblyReport {
    removed_by_rays: usize,
    remaining_groups: Vec<Vec<usize>>,
    passes: usize,
}

#[derive(Default)]
struct PostOptimizationReport {
    passes: usize,
    attempted_groups: usize,
    accepted_reinsertions: usize,
    reinserted_objects: usize,
    unresolved_groups: usize,
    repack_passes: usize,
    repack_attempted_groups: usize,
    repack_accepted: usize,
    repack_added_objects: usize,
}

struct RepackAttempt {
    placed: Vec<PlacedMesh>,
    unpacked_indices: Vec<usize>,
    added_objects: usize,
}

struct BeamState {
    placed: Vec<PlacedMesh>,
    occupied: Vec<bool>,
    unpacked_indices: Vec<usize>,
    cost: f32,
    greedy_baseline: bool,
}

#[derive(Clone)]
struct OrderState {
    placed: Vec<PlacedMesh>,
    occupied: Vec<bool>,
    remaining_indices: Vec<usize>,
    unpacked_indices: Vec<usize>,
    cost: f32,
    future_cost: f32,
    rank_score: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlacementStrategy {
    Spectral,
    OrderBl,
}

impl PlacementStrategy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "spectral" => Ok(Self::Spectral),
            "order-bl" | "bl-order" => Ok(Self::OrderBl),
            _ => bail!("--strategy は spectral または order-bl を指定してください: `{value}`"),
        }
    }
}

#[derive(Clone, Copy)]
struct Deadline {
    ends_at: Option<Instant>,
}

impl Deadline {
    fn from_seconds(seconds: Option<f32>) -> Self {
        Self {
            ends_at: seconds
                .map(|seconds| Instant::now() + Duration::from_secs_f32(seconds.max(0.0))),
        }
    }

    fn expired(self) -> bool {
        self.ends_at
            .map(|ends_at| Instant::now() >= ends_at)
            .unwrap_or(false)
    }
}

enum PlaceOutcome {
    Placed(PlacedMesh),
    NotFound,
    TimedOut,
}

enum SearchOutcome {
    Found(Placement),
    NotFound,
    TimedOut,
}

enum SearchManyOutcome {
    Found(Vec<Placement>),
    NotFound,
    TimedOut,
}

fn pack_meshes(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    ray_disassembly: bool,
    post_opt_passes: usize,
    beam_width: usize,
    strategy: PlacementStrategy,
    order_window: usize,
    bl_candidate_limit: usize,
    repack_passes: usize,
    repack_window: usize,
    repack_unpacked_limit: usize,
    deadline: Deadline,
) -> Result<PackResult> {
    let mut occupied = vec![false; tray.len()];
    let mut placed = Vec::new();
    let mut unpacked_indices = Vec::new();
    let mut timed_out = false;

    match strategy {
        PlacementStrategy::Spectral if beam_width <= 1 => {
            for (source_index, mesh) in meshes.iter().enumerate() {
                if deadline.expired() {
                    timed_out = true;
                    break;
                }
                match place_mesh(
                    mesh,
                    source_index,
                    tray,
                    rotations,
                    &mut occupied,
                    &placed,
                    height_weight,
                    interlock_free,
                    refine,
                    refine_margin,
                    None,
                    true,
                    deadline,
                )? {
                    PlaceOutcome::Placed(placed_mesh) => placed.push(placed_mesh),
                    PlaceOutcome::NotFound => {
                        eprintln!("  パックできませんでした");
                        unpacked_indices.push(source_index);
                    }
                    PlaceOutcome::TimedOut => {
                        timed_out = true;
                        break;
                    }
                }
            }
        }
        PlacementStrategy::Spectral => {
            match pack_meshes_beam(
                meshes,
                tray,
                rotations,
                height_weight,
                interlock_free,
                refine,
                refine_margin,
                beam_width,
                deadline,
            )? {
                Some(state) => {
                    placed = state.placed;
                    unpacked_indices = state.unpacked_indices;
                }
                None => timed_out = true,
            }
        }
        PlacementStrategy::OrderBl => {
            match pack_meshes_order_bl(
                meshes,
                tray,
                rotations,
                height_weight,
                interlock_free,
                refine,
                refine_margin,
                beam_width,
                order_window,
                bl_candidate_limit,
                deadline,
            )? {
                (Some(state), order_timed_out) => {
                    placed = state.placed;
                    unpacked_indices = state.unpacked_indices;
                    timed_out = order_timed_out;
                }
                (None, order_timed_out) => timed_out = order_timed_out,
            }
        }
    }

    let post_report = if !timed_out && (ray_disassembly && post_opt_passes > 0 || repack_passes > 0)
    {
        let mut report = PostOptimizationReport::default();
        if ray_disassembly && post_opt_passes > 0 {
            let (disassembly_report, post_timed_out) = optimize_disassembly(
                meshes,
                tray,
                rotations,
                height_weight,
                interlock_free,
                refine,
                refine_margin,
                post_opt_passes,
                &mut placed,
                deadline,
            )?;
            timed_out |= post_timed_out;
            report = disassembly_report;
        }
        if !timed_out && repack_passes > 0 && !unpacked_indices.is_empty() {
            let (repacked, repack_timed_out) = optimize_unpacked_repack(
                meshes,
                tray,
                rotations,
                height_weight,
                interlock_free,
                refine,
                refine_margin,
                repack_passes,
                repack_window,
                repack_unpacked_limit,
                &mut placed,
                &mut unpacked_indices,
                deadline,
            )?;
            timed_out |= repack_timed_out;
            report.repack_passes = repacked.repack_passes;
            report.repack_attempted_groups = repacked.repack_attempted_groups;
            report.repack_accepted = repacked.repack_accepted;
            report.repack_added_objects = repacked.repack_added_objects;
        }
        Some(report)
    } else {
        None
    };

    let ray_report = if ray_disassembly {
        Some(ray_casting_disassembly(tray, &placed))
    } else {
        None
    };

    Ok(PackResult {
        placed,
        unpacked: unpacked_indices
            .into_iter()
            .map(|source_index| meshes[source_index].name.clone())
            .collect(),
        ray_report,
        post_report,
        timed_out,
    })
}

type PlacementKey = (usize, usize, usize, usize);

fn place_mesh(
    mesh: &Mesh,
    source_index: usize,
    tray: Tray,
    rotations: &[Rotation],
    occupied: &mut [bool],
    placed: &[PlacedMesh],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    forbidden: Option<&HashSet<PlacementKey>>,
    log: bool,
    deadline: Deadline,
) -> Result<PlaceOutcome> {
    if log {
        eprintln!("{} を配置中...", mesh.name);
    }
    let best = match search_placement(
        mesh,
        tray,
        rotations,
        occupied,
        height_weight,
        interlock_free,
        forbidden,
        deadline,
    )? {
        SearchOutcome::Found(best) => best,
        SearchOutcome::NotFound => return Ok(PlaceOutcome::NotFound),
        SearchOutcome::TimedOut => return Ok(PlaceOutcome::TimedOut),
    };
    place_with_candidate(
        mesh,
        source_index,
        tray,
        occupied,
        placed,
        refine,
        refine_margin,
        log,
        deadline,
        best,
    )
}

#[allow(clippy::too_many_arguments)]
fn place_with_candidate(
    mesh: &Mesh,
    source_index: usize,
    tray: Tray,
    occupied: &mut [bool],
    placed: &[PlacedMesh],
    refine: bool,
    refine_margin: f32,
    log: bool,
    deadline: Deadline,
    best: Placement,
) -> Result<PlaceOutcome> {
    let discrete_translation = Vec3::new(
        best.offset.0 as f32 * tray.voxel,
        best.offset.1 as f32 * tray.voxel,
        best.offset.2 as f32 * tray.voxel,
    );
    let translation = if refine {
        refine_translation(
            &best.oriented,
            discrete_translation,
            placed,
            tray,
            refine_margin,
            deadline,
        )
    } else {
        discrete_translation
    };
    if deadline.expired() {
        return Ok(PlaceOutcome::TimedOut);
    }
    let refinement = translation - discrete_translation;
    let triangles: Vec<[Vec3; 3]> = best
        .oriented
        .triangles
        .iter()
        .map(|tri| {
            [
                tri[0] + translation,
                tri[1] + translation,
                tri[2] + translation,
            ]
        })
        .collect();
    let occupied_cells = occupied_cells_for_translation(tray, &best.oriented, translation);
    stamp_cells(tray, occupied, &occupied_cells);
    let bbox = bbox_of_triangles(&triangles);
    if log {
        eprintln!(
            "  {:?} にパックしました。refinement ({:.3}, {:.3}, {:.3}), 回転 {}, コスト {:.3}",
            best.offset, refinement.x, refinement.y, refinement.z, best.rotation_index, best.cost
        );
    }
    Ok(PlaceOutcome::Placed(PlacedMesh {
        name: best.oriented.name,
        source_index,
        triangles,
        offset: best.offset,
        translation,
        refinement,
        rotation_index: best.rotation_index,
        voxel_count: best.oriented.voxel_count,
        mesh_volume: mesh.signed_volume().abs(),
        bbox,
        occupied_cells,
    }))
}

#[allow(clippy::too_many_arguments)]
fn pack_meshes_beam(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    beam_width: usize,
    deadline: Deadline,
) -> Result<Option<BeamState>> {
    let mut states = vec![BeamState {
        placed: Vec::new(),
        occupied: vec![false; tray.len()],
        unpacked_indices: Vec::new(),
        cost: 0.0,
        greedy_baseline: true,
    }];
    let baseline_rotation_count = rotations.len().min(24);

    for (source_index, mesh) in meshes.iter().enumerate() {
        if deadline.expired() {
            return Ok(None);
        }
        let Some(oriented_meshes) = build_oriented_meshes(mesh, rotations, tray.voxel, deadline)?
        else {
            return Ok(None);
        };
        let baseline_oriented_meshes = &oriented_meshes[..baseline_rotation_count];
        eprintln!("{} を配置中... beam候補{}個", mesh.name, states.len());
        let mut next_states = Vec::new();
        for state in &states {
            let mut greedy_forbidden = HashSet::new();
            if state.greedy_baseline {
                match search_placements_from_oriented(
                    baseline_oriented_meshes,
                    tray,
                    &state.occupied,
                    height_weight,
                    interlock_free,
                    None,
                    1,
                    deadline,
                )? {
                    SearchManyOutcome::Found(candidates) => {
                        if let Some(placement) = candidates.into_iter().next() {
                            greedy_forbidden.insert((
                                placement.rotation_index,
                                placement.offset.0,
                                placement.offset.1,
                                placement.offset.2,
                            ));
                            let mut occupied = state.occupied.clone();
                            let outcome = place_with_candidate(
                                mesh,
                                source_index,
                                tray,
                                &mut occupied,
                                &state.placed,
                                refine,
                                refine_margin,
                                false,
                                deadline,
                                placement.clone(),
                            )?;
                            match outcome {
                                PlaceOutcome::Placed(placed_mesh) => {
                                    let mut placed = state.placed.clone();
                                    placed.push(placed_mesh);
                                    next_states.push(BeamState {
                                        placed,
                                        occupied,
                                        unpacked_indices: state.unpacked_indices.clone(),
                                        cost: state.cost + placement.cost,
                                        greedy_baseline: true,
                                    });
                                }
                                PlaceOutcome::NotFound => {}
                                PlaceOutcome::TimedOut => return Ok(None),
                            }
                        }
                    }
                    SearchManyOutcome::NotFound => {
                        let mut skipped = state.unpacked_indices.clone();
                        skipped.push(source_index);
                        next_states.push(BeamState {
                            placed: state.placed.clone(),
                            occupied: state.occupied.clone(),
                            unpacked_indices: skipped,
                            cost: state.cost + 1.0e6,
                            greedy_baseline: true,
                        });
                    }
                    SearchManyOutcome::TimedOut => return Ok(None),
                }
            }

            match search_placements_from_oriented(
                &oriented_meshes,
                tray,
                &state.occupied,
                height_weight,
                interlock_free,
                (!greedy_forbidden.is_empty()).then_some(&greedy_forbidden),
                beam_width,
                deadline,
            )? {
                SearchManyOutcome::Found(candidates) => {
                    for placement in candidates {
                        let mut occupied = state.occupied.clone();
                        let outcome = place_with_candidate(
                            mesh,
                            source_index,
                            tray,
                            &mut occupied,
                            &state.placed,
                            refine,
                            refine_margin,
                            false,
                            deadline,
                            placement.clone(),
                        )?;
                        match outcome {
                            PlaceOutcome::Placed(placed_mesh) => {
                                let mut placed = state.placed.clone();
                                placed.push(placed_mesh);
                                next_states.push(BeamState {
                                    placed,
                                    occupied,
                                    unpacked_indices: state.unpacked_indices.clone(),
                                    cost: state.cost + placement.cost,
                                    greedy_baseline: false,
                                });
                            }
                            PlaceOutcome::NotFound => {}
                            PlaceOutcome::TimedOut => return Ok(None),
                        }
                    }
                }
                SearchManyOutcome::NotFound => {}
                SearchManyOutcome::TimedOut => return Ok(None),
            }

            let mut skipped = state.unpacked_indices.clone();
            skipped.push(source_index);
            next_states.push(BeamState {
                placed: state.placed.clone(),
                occupied: state.occupied.clone(),
                unpacked_indices: skipped,
                cost: state.cost + 1.0e6,
                greedy_baseline: false,
            });
        }

        next_states.sort_by(compare_beam_states);
        if next_states.len() > beam_width {
            let greedy_state = next_states
                .iter()
                .position(|state| state.greedy_baseline)
                .map(|index| next_states.remove(index));
            next_states.truncate(beam_width);
            if let Some(greedy_state) = greedy_state {
                if !next_states.iter().any(|state| state.greedy_baseline) {
                    if next_states.len() == beam_width {
                        next_states.pop();
                    }
                    next_states.push(greedy_state);
                    next_states.sort_by(compare_beam_states);
                }
            }
        }
        if let Some(best) = next_states.first() {
            eprintln!(
                "  beam最良: {}/{}個配置、未配置{}個",
                best.placed.len(),
                source_index + 1,
                best.unpacked_indices.len()
            );
        }
        states = next_states;
    }

    states.sort_by(compare_beam_states);
    Ok(states.into_iter().next())
}

fn compare_beam_states(a: &BeamState, b: &BeamState) -> Ordering {
    b.placed
        .len()
        .cmp(&a.placed.len())
        .then_with(|| {
            placed_mesh_volume(&b.placed)
                .partial_cmp(&placed_mesh_volume(&a.placed))
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
}

#[allow(clippy::too_many_arguments)]
fn pack_meshes_order_bl(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    beam_width: usize,
    order_window: usize,
    bl_candidate_limit: usize,
    deadline: Deadline,
) -> Result<(Option<BeamState>, bool)> {
    let empty_occupied = vec![false; tray.len()];
    let mut states = vec![make_order_state(
        tray,
        Vec::new(),
        empty_occupied,
        (0..meshes.len()).collect(),
        Vec::new(),
        0.0,
    )];
    let mut oriented_cache = vec![None; meshes.len()];
    let placement_limit = beam_width.clamp(2, 4);

    while states
        .iter()
        .any(|state| !state.remaining_indices.is_empty())
    {
        if deadline.expired() {
            return Ok((best_order_state(states.clone()), true));
        }
        let mut next_states = Vec::new();
        for state in &states {
            if state.remaining_indices.is_empty() {
                next_states.push(state.clone());
                continue;
            }
            let candidate_indices =
                select_order_candidate_indices(meshes, tray, state, order_window);
            let distance = manhattan_distance_field(tray, &state.occupied);
            for (candidate_rank, source_index) in candidate_indices.iter().copied().enumerate() {
                let mesh = &meshes[source_index];
                if oriented_cache[source_index].is_none() {
                    let Some(oriented_meshes) =
                        build_oriented_meshes(mesh, rotations, tray.voxel, deadline)?
                    else {
                        return Ok((best_order_state(states.clone()), true));
                    };
                    oriented_cache[source_index] = Some(oriented_meshes);
                }
                let oriented_meshes = oriented_cache[source_index].as_ref().expect("cached");
                let object_placement_limit = if candidate_rank == 0 {
                    placement_limit
                } else {
                    1
                };
                let placements = match search_bl_placements(
                    oriented_meshes,
                    tray,
                    &state.occupied,
                    &distance,
                    height_weight,
                    interlock_free,
                    bl_candidate_limit,
                    object_placement_limit,
                    candidate_rank == 0,
                    deadline,
                )? {
                    SearchManyOutcome::Found(placements) => placements.into_iter().rev().collect(),
                    SearchManyOutcome::NotFound => Vec::new(),
                    SearchManyOutcome::TimedOut => {
                        return Ok((best_order_state(states.clone()), true));
                    }
                };
                for placement in placements {
                    let mut occupied = state.occupied.clone();
                    let outcome = place_with_candidate(
                        mesh,
                        source_index,
                        tray,
                        &mut occupied,
                        &state.placed,
                        refine,
                        refine_margin,
                        false,
                        deadline,
                        placement.clone(),
                    )?;
                    match outcome {
                        PlaceOutcome::Placed(placed_mesh) => {
                            let mut placed = state.placed.clone();
                            placed.push(placed_mesh);
                            let mut remaining_indices = state.remaining_indices.clone();
                            if let Some(position) = remaining_indices
                                .iter()
                                .position(|&index| index == source_index)
                            {
                                remaining_indices.remove(position);
                            }
                            let cost = state.cost + placement.cost;
                            next_states.push(make_order_state(
                                tray,
                                placed,
                                occupied,
                                remaining_indices,
                                state.unpacked_indices.clone(),
                                cost,
                            ));
                        }
                        PlaceOutcome::NotFound => {}
                        PlaceOutcome::TimedOut => {
                            return Ok((best_order_state(states.clone()), true));
                        }
                    }
                }
            }

            let mut remaining_indices = state.remaining_indices.clone();
            if let Some(skipped_index) = candidate_indices
                .first()
                .copied()
                .or_else(|| state.remaining_indices.first().copied())
            {
                if let Some(position) = remaining_indices
                    .iter()
                    .position(|&index| index == skipped_index)
                {
                    remaining_indices.remove(position);
                }
                let mut unpacked_indices = state.unpacked_indices.clone();
                unpacked_indices.push(skipped_index);
                next_states.push(make_order_state(
                    tray,
                    state.placed.clone(),
                    state.occupied.clone(),
                    remaining_indices,
                    unpacked_indices,
                    state.cost + 1.0e6,
                ));
            }
        }

        next_states.sort_by(compare_order_states);
        next_states.truncate(beam_width);
        if let Some(best) = next_states.first() {
            eprintln!(
                "order-bl最良: 配置{}個、残り{}個、未配置{}個",
                best.placed.len(),
                best.remaining_indices.len(),
                best.unpacked_indices.len()
            );
        }
        states = next_states;
    }

    states.sort_by(compare_order_states);
    Ok((states.into_iter().next().map(order_state_to_beam), false))
}

fn make_order_state(
    tray: Tray,
    placed: Vec<PlacedMesh>,
    occupied: Vec<bool>,
    remaining_indices: Vec<usize>,
    unpacked_indices: Vec<usize>,
    cost: f32,
) -> OrderState {
    let future_cost = future_space_cost(tray, &occupied);
    let rank_score = order_state_rank_score(
        tray,
        placed.len(),
        placed_mesh_volume(&placed),
        unpacked_indices.len(),
        cost,
        future_cost,
    );
    OrderState {
        placed,
        occupied,
        remaining_indices,
        unpacked_indices,
        cost,
        future_cost,
        rank_score,
    }
}

fn order_state_rank_score(
    tray: Tray,
    placed_count: usize,
    placed_volume: f32,
    unpacked_count: usize,
    cost: f32,
    future_cost: f32,
) -> f32 {
    let voxel_volume = tray.voxel.powi(3).max(1.0e-6);
    placed_count as f32 * 1000.0 + placed_volume / voxel_volume * 0.05
        - unpacked_count as f32 * 180.0
        - future_cost * 18.0
        - cost * 0.02
}

fn compare_order_states(a: &OrderState, b: &OrderState) -> Ordering {
    b.placed
        .len()
        .cmp(&a.placed.len())
        .then_with(|| a.unpacked_indices.len().cmp(&b.unpacked_indices.len()))
        .then_with(|| {
            b.rank_score
                .partial_cmp(&a.rank_score)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            placed_mesh_volume(&b.placed)
                .partial_cmp(&placed_mesh_volume(&a.placed))
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            a.future_cost
                .partial_cmp(&b.future_cost)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
}

fn best_order_state(mut states: Vec<OrderState>) -> Option<BeamState> {
    states.sort_by(compare_order_states);
    states.into_iter().next().map(order_state_to_beam)
}

fn order_state_to_beam(mut state: OrderState) -> BeamState {
    state.unpacked_indices.extend(state.remaining_indices);
    BeamState {
        placed: state.placed,
        occupied: state.occupied,
        unpacked_indices: state.unpacked_indices,
        cost: state.cost,
        greedy_baseline: false,
    }
}

fn select_order_candidate_indices(
    meshes: &[Mesh],
    tray: Tray,
    state: &OrderState,
    order_window: usize,
) -> Vec<usize> {
    let occupied_count = state.occupied.iter().filter(|&&cell| cell).count();
    let fill_ratio = occupied_count as f32 / tray.len() as f32;
    let empty_count = tray.len().saturating_sub(occupied_count);
    let mut scored = state
        .remaining_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(position, index)| {
            let mesh = &meshes[index];
            let (min, max) = mesh.bbox();
            let extent = max - min;
            let bbox_voxels = ((extent.x / tray.voxel).ceil().max(1.0)
                * (extent.y / tray.voxel).ceil().max(1.0)
                * (extent.z / tray.voxel).ceil().max(1.0)) as usize;
            let volume_cells = mesh.bbox_volume() / tray.voxel.powi(3).max(1.0e-6);
            let oversized_penalty = if bbox_voxels > empty_count {
                1.0e6
            } else {
                0.0
            };
            let size_term = if fill_ratio < 0.45 {
                -volume_cells
            } else if fill_ratio < 0.70 {
                -volume_cells.sqrt()
            } else {
                volume_cells
            };
            let footprint_ratio =
                (extent.x * extent.y / (tray.width * tray.depth).max(1.0e-6)).max(0.0);
            let tall_penalty = fill_ratio * extent.z / tray.height.max(1.0e-6) * 40.0;
            let order_penalty = position as f32 * 0.001;
            (
                index,
                oversized_penalty
                    + size_term
                    + footprint_ratio * 6.0
                    + tall_penalty
                    + order_penalty,
            )
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for (index, _) in scored {
        if seen.insert(index) {
            selected.push(index);
            if selected.len() >= order_window {
                break;
            }
        }
    }
    selected
}

fn future_space_cost(tray: Tray, occupied: &[bool]) -> f32 {
    let mut empty_count = 0usize;
    let mut used_top = 0usize;
    let mut low_voids = 0usize;
    let mut column_tops = vec![0usize; tray.nx * tray.ny];
    for z in 0..tray.nz {
        for y in 0..tray.ny {
            for x in 0..tray.nx {
                let i = idx(x, y, z, tray.nx, tray.ny);
                if occupied[i] {
                    used_top = used_top.max(z + 1);
                    column_tops[x + y * tray.nx] = z + 1;
                } else {
                    empty_count += 1;
                }
            }
        }
    }
    if empty_count == 0 {
        return 0.0;
    }

    for y in 0..tray.ny {
        for x in 0..tray.nx {
            let top = column_tops[x + y * tray.nx];
            for z in 0..top {
                if !occupied[idx(x, y, z, tray.nx, tray.ny)] {
                    low_voids += 1;
                }
            }
        }
    }
    let mut roughness = 0usize;
    let mut rough_edges = 0usize;
    for y in 0..tray.ny {
        for x in 0..tray.nx {
            let current = column_tops[x + y * tray.nx];
            if x + 1 < tray.nx {
                roughness += current.abs_diff(column_tops[x + 1 + y * tray.nx]);
                rough_edges += 1;
            }
            if y + 1 < tray.ny {
                roughness += current.abs_diff(column_tops[x + (y + 1) * tray.nx]);
                rough_edges += 1;
            }
        }
    }
    let roughness_norm = if rough_edges > 0 {
        roughness as f32 / rough_edges as f32 / tray.nz as f32
    } else {
        0.0
    };
    let low_void_ratio = low_voids as f32 / empty_count as f32;
    let height_ratio = used_top as f32 / tray.nz as f32;
    low_void_ratio * 60.0 + roughness_norm * 20.0 + height_ratio * 8.0
}

fn optimize_disassembly(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    max_passes: usize,
    placed: &mut Vec<PlacedMesh>,
    deadline: Deadline,
) -> Result<(PostOptimizationReport, bool)> {
    let mut passes = 0;
    let mut attempted_groups = 0;
    let mut accepted_reinsertions = 0;
    let mut reinserted_objects = 0;
    let mut report = ray_casting_disassembly(tray, placed);

    let mut timed_out = false;
    while passes < max_passes && !report.remaining_groups.is_empty() {
        if deadline.expired() {
            timed_out = true;
            break;
        }
        passes += 1;
        let current_score = disassembly_score(&report);
        let groups = report.remaining_groups.clone();
        let mut accepted = false;

        for group in groups {
            attempted_groups += 1;
            let Some((candidate, candidate_report)) = best_reinsertion_candidate(
                meshes,
                tray,
                rotations,
                height_weight,
                interlock_free,
                refine,
                refine_margin,
                placed,
                &group,
                current_score,
                deadline,
            )?
            else {
                continue;
            };

            reinserted_objects += candidate.len().saturating_sub(placed.len() - group.len());
            *placed = candidate;
            report = candidate_report;
            accepted_reinsertions += 1;
            accepted = true;
            break;
        }

        if !accepted {
            break;
        }
    }

    Ok((
        PostOptimizationReport {
            passes,
            attempted_groups,
            accepted_reinsertions,
            reinserted_objects,
            unresolved_groups: report.remaining_groups.len(),
            ..Default::default()
        },
        timed_out,
    ))
}

#[allow(clippy::too_many_arguments)]
fn best_reinsertion_candidate(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    placed: &[PlacedMesh],
    group: &[usize],
    current_score: (usize, usize),
    deadline: Deadline,
) -> Result<Option<(Vec<PlacedMesh>, RayDisassemblyReport)>> {
    let group_set = group.iter().copied().collect::<HashSet<_>>();
    let kept = placed
        .iter()
        .enumerate()
        .filter_map(|(id, object)| (!group_set.contains(&id)).then_some(object.clone()))
        .collect::<Vec<_>>();
    let source_indices = group
        .iter()
        .map(|&object_id| placed[object_id].source_index)
        .collect::<Vec<_>>();
    let original_keys = group
        .iter()
        .map(|&object_id| {
            let object = &placed[object_id];
            (
                object.source_index,
                (
                    object.rotation_index,
                    object.offset.0,
                    object.offset.1,
                    object.offset.2,
                ),
            )
        })
        .collect::<Vec<_>>();

    let mut best: Option<(Vec<PlacedMesh>, RayDisassemblyReport, (usize, usize))> = None;
    for order in reinsertion_orders(&source_indices, meshes) {
        if deadline.expired() {
            break;
        }
        for height_multiplier in [1.0, 1.5, 0.5, 2.5] {
            for avoid_original in [true, false] {
                if deadline.expired() {
                    break;
                }
                let forbidden = if avoid_original {
                    original_keys.clone()
                } else {
                    Vec::new()
                };
                let Some(candidate) = try_reinsert_order(
                    meshes,
                    tray,
                    rotations,
                    &kept,
                    &order,
                    &forbidden,
                    height_weight * height_multiplier,
                    interlock_free,
                    refine,
                    refine_margin,
                    deadline,
                )?
                else {
                    continue;
                };
                let candidate_report = ray_casting_disassembly(tray, &candidate);
                let score = disassembly_score(&candidate_report);
                if score < current_score
                    && best
                        .as_ref()
                        .map(|(_, _, best_score)| score < *best_score)
                        .unwrap_or(true)
                {
                    best = Some((candidate, candidate_report, score));
                }
            }
        }
    }

    Ok(best.map(|(candidate, report, _)| (candidate, report)))
}

#[allow(clippy::too_many_arguments)]
fn try_reinsert_order(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    kept: &[PlacedMesh],
    order: &[usize],
    forbidden_keys: &[(usize, PlacementKey)],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    deadline: Deadline,
) -> Result<Option<Vec<PlacedMesh>>> {
    let mut candidate = kept.to_vec();
    let mut occupied = occupied_from_placed(tray, &candidate);

    for &source_index in order {
        if deadline.expired() {
            return Ok(None);
        }
        let forbidden = forbidden_keys
            .iter()
            .filter_map(|&(key_source, key)| (key_source == source_index).then_some(key))
            .collect::<HashSet<_>>();
        let forbidden = (!forbidden.is_empty()).then_some(forbidden);
        let placed_mesh = match place_mesh(
            &meshes[source_index],
            source_index,
            tray,
            rotations,
            &mut occupied,
            &candidate,
            height_weight,
            interlock_free,
            refine,
            refine_margin,
            forbidden.as_ref(),
            false,
            deadline,
        )? {
            PlaceOutcome::Placed(placed_mesh) => placed_mesh,
            PlaceOutcome::NotFound | PlaceOutcome::TimedOut => return Ok(None),
        };
        candidate.push(placed_mesh);
    }

    Ok(Some(candidate))
}

fn reinsertion_orders(source_indices: &[usize], meshes: &[Mesh]) -> Vec<Vec<usize>> {
    let mut orders = Vec::new();
    push_unique_order(&mut orders, source_indices.to_vec());

    let mut reversed = source_indices.to_vec();
    reversed.reverse();
    push_unique_order(&mut orders, reversed);

    let mut small_first = source_indices.to_vec();
    small_first.sort_by(|&a, &b| {
        meshes[a]
            .bbox_volume()
            .partial_cmp(&meshes[b].bbox_volume())
            .unwrap_or(Ordering::Equal)
    });
    push_unique_order(&mut orders, small_first.clone());

    small_first.reverse();
    push_unique_order(&mut orders, small_first);

    orders
}

fn push_unique_order(orders: &mut Vec<Vec<usize>>, order: Vec<usize>) {
    if !orders.iter().any(|existing| existing == &order) {
        orders.push(order);
    }
}

fn disassembly_score(report: &RayDisassemblyReport) -> (usize, usize) {
    (
        report.remaining_groups.iter().map(Vec::len).sum::<usize>(),
        report.remaining_groups.len(),
    )
}

fn occupied_from_placed(tray: Tray, placed: &[PlacedMesh]) -> Vec<bool> {
    let mut occupied = vec![false; tray.len()];
    for object in placed {
        stamp_cells(tray, &mut occupied, &object.occupied_cells);
    }
    occupied
}

fn placed_mesh_volume(placed: &[PlacedMesh]) -> f32 {
    placed.iter().map(|object| object.mesh_volume).sum()
}

#[allow(clippy::too_many_arguments)]
fn optimize_unpacked_repack(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    max_passes: usize,
    repack_window: usize,
    unpacked_limit: usize,
    placed: &mut Vec<PlacedMesh>,
    unpacked_indices: &mut Vec<usize>,
    deadline: Deadline,
) -> Result<(PostOptimizationReport, bool)> {
    let mut report = PostOptimizationReport::default();
    let mut timed_out = false;

    while report.repack_passes < max_passes && !unpacked_indices.is_empty() {
        if deadline.expired() {
            timed_out = true;
            break;
        }
        report.repack_passes += 1;
        let baseline_count = placed.len();
        let baseline_volume = placed_mesh_volume(placed);
        let groups = repack_removal_groups(placed, repack_window);
        let mut best: Option<RepackAttempt> = None;

        for group in groups {
            report.repack_attempted_groups += 1;
            let group_set = group.iter().copied().collect::<HashSet<_>>();
            let kept = placed
                .iter()
                .enumerate()
                .filter_map(|(index, object)| {
                    (!group_set.contains(&index)).then_some(object.clone())
                })
                .collect::<Vec<_>>();
            let removed_sources = group
                .iter()
                .map(|&index| placed[index].source_index)
                .collect::<Vec<_>>();
            let unpacked_batch = unpacked_indices
                .iter()
                .copied()
                .take(unpacked_limit)
                .collect::<Vec<_>>();

            for order in repack_orders(&unpacked_batch, &removed_sources, meshes) {
                if deadline.expired() {
                    timed_out = true;
                    break;
                }
                let attempt = try_reinsert_partial(
                    meshes,
                    tray,
                    rotations,
                    &kept,
                    &order,
                    height_weight,
                    interlock_free,
                    refine,
                    refine_margin,
                    deadline,
                )?;
                let remaining_unpacked = merge_unpacked_indices(
                    &attempt.unpacked_indices,
                    unpacked_indices.iter().copied().skip(unpacked_batch.len()),
                );
                let added_objects = attempt.placed.len().saturating_sub(baseline_count);
                let improved = attempt.placed.len() > baseline_count
                    || (attempt.placed.len() == baseline_count
                        && placed_mesh_volume(&attempt.placed) > baseline_volume + 1.0e-3);
                if improved
                    && best
                        .as_ref()
                        .map(|current| compare_repack_attempts(&attempt, current) == Ordering::Less)
                        .unwrap_or(true)
                {
                    best = Some(RepackAttempt {
                        placed: attempt.placed,
                        unpacked_indices: remaining_unpacked,
                        added_objects,
                    });
                }
            }
            if timed_out {
                break;
            }
        }

        let Some(best) = best else {
            break;
        };
        report.repack_accepted += 1;
        report.repack_added_objects += best.added_objects;
        *placed = best.placed;
        *unpacked_indices = best.unpacked_indices;
    }

    Ok((report, timed_out))
}

fn repack_removal_groups(placed: &[PlacedMesh], repack_window: usize) -> Vec<Vec<usize>> {
    if placed.is_empty() {
        return Vec::new();
    }
    let window = repack_window.min(placed.len()).max(1);
    let mut groups = Vec::new();
    let tail = (placed.len() - window..placed.len()).collect::<Vec<_>>();
    push_unique_group(&mut groups, tail);

    let mut by_top_z = (0..placed.len()).collect::<Vec<_>>();
    by_top_z.sort_by(|&a, &b| {
        placed[b]
            .bbox
            .1
            .z
            .partial_cmp(&placed[a].bbox.1.z)
            .unwrap_or(Ordering::Equal)
    });
    by_top_z.truncate(window);
    push_unique_group(&mut groups, by_top_z);

    let mut by_small_volume = (0..placed.len()).collect::<Vec<_>>();
    by_small_volume.sort_by(|&a, &b| {
        placed[a]
            .mesh_volume
            .partial_cmp(&placed[b].mesh_volume)
            .unwrap_or(Ordering::Equal)
    });
    by_small_volume.truncate(window);
    push_unique_group(&mut groups, by_small_volume);
    groups
}

fn push_unique_group(groups: &mut Vec<Vec<usize>>, mut group: Vec<usize>) {
    group.sort_unstable();
    if !group.is_empty() && !groups.iter().any(|existing| existing == &group) {
        groups.push(group);
    }
}

fn merge_unpacked_indices(first: &[usize], rest: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut merged = Vec::new();
    for source_index in first.iter().copied().chain(rest) {
        if seen.insert(source_index) {
            merged.push(source_index);
        }
    }
    merged
}

fn repack_orders(unpacked: &[usize], removed: &[usize], meshes: &[Mesh]) -> Vec<Vec<usize>> {
    let mut orders = Vec::new();
    let mut unplaced_first = unpacked.to_vec();
    unplaced_first.extend_from_slice(removed);
    push_unique_order(&mut orders, unplaced_first.clone());

    let mut large_first = unplaced_first.clone();
    large_first.sort_by(|&a, &b| {
        meshes[b]
            .bbox_volume()
            .partial_cmp(&meshes[a].bbox_volume())
            .unwrap_or(Ordering::Equal)
    });
    push_unique_order(&mut orders, large_first.clone());
    large_first.reverse();
    push_unique_order(&mut orders, large_first);

    let mut removed_first = removed.to_vec();
    removed_first.extend_from_slice(unpacked);
    push_unique_order(&mut orders, removed_first);
    orders
}

#[allow(clippy::too_many_arguments)]
fn try_reinsert_partial(
    meshes: &[Mesh],
    tray: Tray,
    rotations: &[Rotation],
    kept: &[PlacedMesh],
    order: &[usize],
    height_weight: f32,
    interlock_free: bool,
    refine: bool,
    refine_margin: f32,
    deadline: Deadline,
) -> Result<RepackAttempt> {
    let mut candidate = kept.to_vec();
    let mut occupied = occupied_from_placed(tray, &candidate);
    let mut unpacked_indices = Vec::new();

    for &source_index in order {
        if deadline.expired() {
            unpacked_indices.push(source_index);
            continue;
        }
        match place_mesh(
            &meshes[source_index],
            source_index,
            tray,
            rotations,
            &mut occupied,
            &candidate,
            height_weight,
            interlock_free,
            refine,
            refine_margin,
            None,
            false,
            deadline,
        )? {
            PlaceOutcome::Placed(placed_mesh) => candidate.push(placed_mesh),
            PlaceOutcome::NotFound | PlaceOutcome::TimedOut => unpacked_indices.push(source_index),
        }
    }

    Ok(RepackAttempt {
        placed: candidate,
        unpacked_indices,
        added_objects: 0,
    })
}

fn compare_repack_attempts(a: &RepackAttempt, b: &RepackAttempt) -> Ordering {
    b.placed.len().cmp(&a.placed.len()).then_with(|| {
        placed_mesh_volume(&b.placed)
            .partial_cmp(&placed_mesh_volume(&a.placed))
            .unwrap_or(Ordering::Equal)
    })
}

fn search_placement(
    mesh: &Mesh,
    tray: Tray,
    rotations: &[Rotation],
    occupied: &[bool],
    height_weight: f32,
    interlock_free: bool,
    forbidden: Option<&HashSet<PlacementKey>>,
    deadline: Deadline,
) -> Result<SearchOutcome> {
    match search_placements(
        mesh,
        tray,
        rotations,
        occupied,
        height_weight,
        interlock_free,
        forbidden,
        1,
        deadline,
    )? {
        SearchManyOutcome::Found(mut placements) => Ok(placements
            .pop()
            .map(SearchOutcome::Found)
            .unwrap_or(SearchOutcome::NotFound)),
        SearchManyOutcome::NotFound => Ok(SearchOutcome::NotFound),
        SearchManyOutcome::TimedOut => Ok(SearchOutcome::TimedOut),
    }
}

#[allow(clippy::too_many_arguments)]
fn search_placements(
    mesh: &Mesh,
    tray: Tray,
    rotations: &[Rotation],
    occupied: &[bool],
    height_weight: f32,
    interlock_free: bool,
    forbidden: Option<&HashSet<PlacementKey>>,
    limit: usize,
    deadline: Deadline,
) -> Result<SearchManyOutcome> {
    let Some(oriented_meshes) = build_oriented_meshes(mesh, rotations, tray.voxel, deadline)?
    else {
        return Ok(SearchManyOutcome::TimedOut);
    };
    search_placements_from_oriented(
        &oriented_meshes,
        tray,
        occupied,
        height_weight,
        interlock_free,
        forbidden,
        limit,
        deadline,
    )
}

fn build_oriented_meshes(
    mesh: &Mesh,
    rotations: &[Rotation],
    voxel: f32,
    deadline: Deadline,
) -> Result<Option<Vec<Option<OrientedMesh>>>> {
    let timed_out = AtomicBool::new(false);
    let built = rotations
        .par_iter()
        .map(|&rotation| -> Result<Option<OrientedMesh>> {
            if deadline.expired() {
                timed_out.store(true, AtomicOrdering::Relaxed);
                return Ok(None);
            }
            let oriented = voxelize_mesh(mesh, rotation, voxel)?;
            Ok((oriented.voxel_count > 0).then_some(oriented))
        })
        .collect::<Result<Vec<_>>>()?;
    if timed_out.load(AtomicOrdering::Relaxed) || deadline.expired() {
        Ok(None)
    } else {
        Ok(Some(built))
    }
}

#[allow(clippy::too_many_arguments)]
fn search_placements_from_oriented(
    oriented_meshes: &[Option<OrientedMesh>],
    tray: Tray,
    occupied: &[bool],
    height_weight: f32,
    interlock_free: bool,
    forbidden: Option<&HashSet<PlacementKey>>,
    limit: usize,
    deadline: Deadline,
) -> Result<SearchManyOutcome> {
    let distance = manhattan_distance_field(tray, occupied);
    if deadline.expired() {
        return Ok(SearchManyOutcome::TimedOut);
    }
    let existing_fft = fft_of_bool_grid(tray, occupied);
    let distance_fft = fft_of_scalar_grid(tray, &distance);

    let timed_out = AtomicBool::new(false);
    let per_rotation = oriented_meshes
        .par_iter()
        .enumerate()
        .map(|(rotation_index, oriented)| -> Result<Vec<Placement>> {
            let Some(oriented) = oriented else {
                return Ok(Vec::new());
            };
            if deadline.expired() {
                timed_out.store(true, AtomicOrdering::Relaxed);
                return Ok(Vec::new());
            }
            if oriented.nx > tray.nx || oriented.ny > tray.ny || oriented.nz > tray.nz {
                return Ok(Vec::new());
            }

            let object_fft = fft_of_object(tray, oriented);
            let collision = inverse_product(&existing_fft, &object_fft, tray, true);
            let proximity = inverse_product(&distance_fft, &object_fft, tray, true);
            let reachable = if interlock_free {
                Some(reachable_offsets(tray, oriented, &collision))
            } else {
                None
            };

            let mut best = Vec::<Placement>::new();
            let max_x = tray.nx - oriented.nx;
            let max_y = tray.ny - oriented.ny;
            let max_z = tray.nz - oriented.nz;
            for z in 0..=max_z {
                if deadline.expired() {
                    timed_out.store(true, AtomicOrdering::Relaxed);
                    return Ok(best);
                }
                let z_norm = if tray.nz > 1 {
                    z as f32 / (tray.nz - 1) as f32
                } else {
                    0.0
                };
                let height_cost = height_weight * z_norm.powi(3);
                for y in 0..=max_y {
                    for x in 0..=max_x {
                        if forbidden
                            .map(|keys| keys.contains(&(rotation_index, x, y, z)))
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let grid_index = idx(x, y, z, tray.nx, tray.ny);
                        if collision[grid_index] > 0.5 {
                            continue;
                        }
                        if let Some(reachable) = &reachable {
                            let offset_index = idx(x, y, z, max_x + 1, max_y + 1);
                            if !reachable[offset_index] {
                                continue;
                            }
                        }
                        let fit_cost = proximity[grid_index] / oriented.voxel_count as f32;
                        let cost = fit_cost + height_cost;
                        push_best_placement(
                            &mut best,
                            limit,
                            Placement {
                                oriented: oriented.clone(),
                                rotation_index,
                                offset: (x, y, z),
                                cost,
                            },
                        );
                    }
                }
            }
            Ok(best)
        })
        .collect::<Result<Vec<_>>>()?;

    if timed_out.load(AtomicOrdering::Relaxed) || deadline.expired() {
        return Ok(SearchManyOutcome::TimedOut);
    }

    let mut best = Vec::<Placement>::new();
    for placements in per_rotation {
        for placement in placements {
            push_best_placement(&mut best, limit, placement);
        }
    }

    if best.is_empty() {
        Ok(SearchManyOutcome::NotFound)
    } else {
        best.sort_by(|a, b| compare_placements_by_cost(b, a));
        Ok(SearchManyOutcome::Found(best))
    }
}

fn compare_placements_by_cost(a: &Placement, b: &Placement) -> Ordering {
    a.cost
        .partial_cmp(&b.cost)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.rotation_index.cmp(&b.rotation_index))
        .then_with(|| a.offset.2.cmp(&b.offset.2))
        .then_with(|| a.offset.1.cmp(&b.offset.1))
        .then_with(|| a.offset.0.cmp(&b.offset.0))
}

#[allow(clippy::too_many_arguments)]
fn search_bl_placements(
    oriented_meshes: &[Option<OrientedMesh>],
    tray: Tray,
    occupied: &[bool],
    distance: &[f32],
    height_weight: f32,
    interlock_free: bool,
    candidate_limit: usize,
    limit: usize,
    contact_fft: bool,
    deadline: Deadline,
) -> Result<SearchManyOutcome> {
    let timed_out = AtomicBool::new(false);
    let big_m = tray.len() as f32 + 1.0;
    let contact_field_fft = contact_fft
        .then(|| fft_of_scalar_grid(tray, &contact_candidate_field(tray, occupied, big_m)));
    let per_rotation = oriented_meshes
        .par_iter()
        .enumerate()
        .map(|(rotation_index, oriented)| -> Vec<Placement> {
            let Some(oriented) = oriented else {
                return Vec::new();
            };
            if deadline.expired() {
                timed_out.store(true, AtomicOrdering::Relaxed);
                return Vec::new();
            }
            if oriented.nx > tray.nx || oriented.ny > tray.ny || oriented.nz > tray.nz {
                return Vec::new();
            }
            let mut best = Vec::new();
            let candidate_offsets = if let Some(contact_field_fft) = &contact_field_fft {
                let object_fft = fft_of_object(tray, oriented);
                let contact_scores = inverse_product(contact_field_fft, &object_fft, tray, true);
                contact_fft_candidate_offsets(
                    tray,
                    occupied,
                    oriented,
                    &contact_scores,
                    big_m,
                    candidate_limit,
                )
            } else {
                bl_candidate_offsets(tray, occupied, oriented, candidate_limit)
            };
            for offset in candidate_offsets {
                if deadline.expired() {
                    timed_out.store(true, AtomicOrdering::Relaxed);
                    break;
                }
                if !can_place_voxels(tray, occupied, oriented, offset) {
                    continue;
                }
                if interlock_free && !linear_axis_reachable(tray, occupied, oriented, offset) {
                    continue;
                }
                let cost =
                    bl_placement_cost(tray, occupied, distance, oriented, offset, height_weight);
                push_best_placement(
                    &mut best,
                    limit,
                    Placement {
                        oriented: oriented.clone(),
                        rotation_index,
                        offset,
                        cost,
                    },
                );
            }
            best
        })
        .collect::<Vec<_>>();

    if timed_out.load(AtomicOrdering::Relaxed) || deadline.expired() {
        return Ok(SearchManyOutcome::TimedOut);
    }

    let mut best = Vec::new();
    for placements in per_rotation {
        for placement in placements {
            push_best_placement(&mut best, limit, placement);
        }
    }
    if best.is_empty() {
        Ok(SearchManyOutcome::NotFound)
    } else {
        best.sort_by(|a, b| compare_placements_by_cost(b, a));
        Ok(SearchManyOutcome::Found(best))
    }
}

fn contact_candidate_field(tray: Tray, occupied: &[bool], big_m: f32) -> Vec<f32> {
    let mut field = vec![0.0; tray.len()];
    for z in 0..tray.nz {
        for y in 0..tray.ny {
            for x in 0..tray.nx {
                let i = idx(x, y, z, tray.nx, tray.ny);
                if occupied[i] {
                    field[i] = big_m;
                    for (nxp, nyp, nzp) in neighbors6(x, y, z, tray.nx, tray.ny, tray.nz) {
                        let ni = idx(nxp, nyp, nzp, tray.nx, tray.ny);
                        if !occupied[ni] && field[ni] < 1.0 {
                            field[ni] = 1.0;
                        }
                    }
                }
            }
        }
    }
    field
}

fn contact_fft_candidate_offsets(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    contact_scores: &[f32],
    big_m: f32,
    candidate_limit: usize,
) -> Vec<(usize, usize, usize)> {
    if !occupied.iter().any(|&cell| cell) {
        return vec![(0, 0, 0)];
    }

    let max_x = tray.nx - object.nx;
    let max_y = tray.ny - object.ny;
    let max_z = tray.nz - object.nz;
    let mut scored = Vec::new();
    for z in 0..=max_z {
        for y in 0..=max_y {
            for x in 0..=max_x {
                let score = contact_scores[idx(x, y, z, tray.nx, tray.ny)];
                if score > 0.5
                    && score < big_m - 0.5
                    && can_place_voxels(tray, occupied, object, (x, y, z))
                {
                    scored.push(((x, y, z), score));
                }
            }
        }
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| compare_offsets_bl(&a.0, &b.0))
    });

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (offset, _) in scored {
        if seen.insert(offset) {
            candidates.push(offset);
            if candidates.len() >= candidate_limit {
                break;
            }
        }
    }

    if candidates.len() < candidate_limit {
        for offset in bl_candidate_offsets(tray, occupied, object, candidate_limit) {
            if seen.insert(offset) {
                candidates.push(offset);
                if candidates.len() >= candidate_limit {
                    break;
                }
            }
        }
    }
    candidates
}

fn bl_candidate_offsets(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    candidate_limit: usize,
) -> Vec<(usize, usize, usize)> {
    let max_x = tray.nx - object.nx;
    let max_y = tray.ny - object.ny;
    let max_z = tray.nz - object.nz;
    let mut raw = HashSet::new();
    raw.insert((0, 0, 0));

    for z in 0..tray.nz {
        for y in 0..tray.ny {
            for x in 0..tray.nx {
                if !occupied[idx(x, y, z, tray.nx, tray.ny)] {
                    continue;
                }
                raw.insert((x.min(max_x), y.min(max_y), z.min(max_z)));
                if x < max_x {
                    raw.insert((x + 1, y.min(max_y), z.min(max_z)));
                }
                if y < max_y {
                    raw.insert((x.min(max_x), y + 1, z.min(max_z)));
                }
                if z < max_z {
                    raw.insert((x.min(max_x), y.min(max_y), z + 1));
                }
            }
        }
    }

    if raw.len() < candidate_limit {
        let column_tops = column_top_heights(tray, occupied);
        let footprint = object_footprint_cells(object);
        let mut extra = HashSet::new();
        for y in 0..=max_y {
            for x in 0..=max_x {
                let z = support_height_for_object(tray, &footprint, &column_tops, x, y).min(max_z);
                extra.insert((x, y, z));
                if z < max_z {
                    extra.insert((x, y, z + 1));
                }
                if x > 0 {
                    extra.insert((x - 1, y, z));
                }
                if y > 0 {
                    extra.insert((x, y - 1, z));
                }
            }
        }

        if extra.len() < candidate_limit {
            'cavities: for z in 0..tray.nz {
                for y in 0..tray.ny {
                    for x in 0..tray.nx {
                        let i = idx(x, y, z, tray.nx, tray.ny);
                        if occupied[i] || cavity_neighbor_count(tray, occupied, x, y, z) < 3 {
                            continue;
                        }
                        add_clamped_anchor_candidates(
                            &mut extra, x, y, z, object, max_x, max_y, max_z,
                        );
                        if extra.len() >= candidate_limit {
                            break 'cavities;
                        }
                    }
                }
            }
        }

        let mut extra_candidates = extra.into_iter().collect::<Vec<_>>();
        extra_candidates.sort_by(compare_offsets_bl);
        extra_candidates.truncate(candidate_limit.saturating_sub(raw.len()));
        raw.extend(extra_candidates);
    }

    let mut settled = HashSet::new();
    for offset in raw {
        if let Some(offset) = settle_bl_offset(tray, occupied, object, offset) {
            settled.insert(offset);
        }
    }
    let mut candidates = settled.into_iter().collect::<Vec<_>>();
    candidates.sort_by(compare_offsets_bl);
    candidates.truncate(candidate_limit);
    candidates
}

fn object_footprint_cells(object: &OrientedMesh) -> Vec<(usize, usize)> {
    let mut seen = HashSet::new();
    for &(x, y, _) in &object.cells {
        seen.insert((x, y));
    }
    seen.into_iter().collect()
}

fn column_top_heights(tray: Tray, occupied: &[bool]) -> Vec<usize> {
    let mut tops = vec![0usize; tray.nx * tray.ny];
    for y in 0..tray.ny {
        for x in 0..tray.nx {
            let mut top = 0usize;
            for z in 0..tray.nz {
                if occupied[idx(x, y, z, tray.nx, tray.ny)] {
                    top = z + 1;
                }
            }
            tops[x + y * tray.nx] = top;
        }
    }
    tops
}

fn support_height_for_object(
    tray: Tray,
    footprint: &[(usize, usize)],
    column_tops: &[usize],
    ox: usize,
    oy: usize,
) -> usize {
    let mut support = 0usize;
    for &(x, y) in footprint {
        let gx = ox + x;
        let gy = oy + y;
        if gx < tray.nx && gy < tray.ny {
            support = support.max(column_tops[gx + gy * tray.nx]);
        }
    }
    support
}

fn cavity_neighbor_count(tray: Tray, occupied: &[bool], x: usize, y: usize, z: usize) -> usize {
    let mut count = 0usize;
    for (nxp, nyp, nzp) in neighbors6(x, y, z, tray.nx, tray.ny, tray.nz) {
        if occupied[idx(nxp, nyp, nzp, tray.nx, tray.ny)] {
            count += 1;
        }
    }
    if x == 0 || x + 1 == tray.nx {
        count += 1;
    }
    if y == 0 || y + 1 == tray.ny {
        count += 1;
    }
    if z == 0 || z + 1 == tray.nz {
        count += 1;
    }
    count
}

fn add_clamped_anchor_candidates(
    raw: &mut HashSet<(usize, usize, usize)>,
    x: usize,
    y: usize,
    z: usize,
    object: &OrientedMesh,
    max_x: usize,
    max_y: usize,
    max_z: usize,
) {
    let x_candidates = [
        x.saturating_sub(object.nx / 2),
        x.saturating_add(1).saturating_sub(object.nx),
        x,
    ];
    let y_candidates = [
        y.saturating_sub(object.ny / 2),
        y.saturating_add(1).saturating_sub(object.ny),
        y,
    ];
    let z_candidates = [
        z.saturating_sub(object.nz / 2),
        z.saturating_add(1).saturating_sub(object.nz),
        z,
    ];
    for ox in x_candidates {
        for oy in y_candidates {
            for oz in z_candidates {
                raw.insert((ox.min(max_x), oy.min(max_y), oz.min(max_z)));
            }
        }
    }
}

fn bl_placement_cost(
    tray: Tray,
    occupied: &[bool],
    distance: &[f32],
    object: &OrientedMesh,
    offset: (usize, usize, usize),
    height_weight: f32,
) -> f32 {
    let z_norm = if tray.nz > 1 {
        offset.2 as f32 / (tray.nz - 1) as f32
    } else {
        0.0
    };
    let y_norm = if tray.ny > 1 {
        offset.1 as f32 / (tray.ny - 1) as f32
    } else {
        0.0
    };
    let x_norm = if tray.nx > 1 {
        offset.0 as f32 / (tray.nx - 1) as f32
    } else {
        0.0
    };
    let mut distance_sum = 0.0;
    let step = object.cells.len().div_ceil(64).max(1);
    let mut sampled = 0usize;
    for &(x, y, z) in object.cells.iter().step_by(step) {
        distance_sum += distance[idx(offset.0 + x, offset.1 + y, offset.2 + z, tray.nx, tray.ny)];
        sampled += 1;
    }
    let fit_cost = distance_sum / sampled.max(1) as f32 / tray.voxel.max(1.0e-6);
    let contact = placement_contact_ratio(tray, occupied, object, offset, step);
    fit_cost * 0.35 + height_weight * z_norm.powi(3) + z_norm * 0.35 + y_norm * 0.04 + x_norm * 0.01
        - contact * 0.6
}

fn placement_contact_ratio(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    offset: (usize, usize, usize),
    step: usize,
) -> f32 {
    let mut contact_faces = 0usize;
    let mut total_faces = 0usize;
    for &(x, y, z) in object.cells.iter().step_by(step) {
        let gx = offset.0 + x;
        let gy = offset.1 + y;
        let gz = offset.2 + z;
        for axis in 0..3 {
            for positive in [false, true] {
                total_faces += 1;
                let neighbor = match (axis, positive) {
                    (0, false) if gx == 0 => None,
                    (0, false) => Some((gx - 1, gy, gz)),
                    (0, true) if gx + 1 == tray.nx => None,
                    (0, true) => Some((gx + 1, gy, gz)),
                    (1, false) if gy == 0 => None,
                    (1, false) => Some((gx, gy - 1, gz)),
                    (1, true) if gy + 1 == tray.ny => None,
                    (1, true) => Some((gx, gy + 1, gz)),
                    (2, false) if gz == 0 => None,
                    (2, false) => Some((gx, gy, gz - 1)),
                    (2, true) if gz + 1 == tray.nz => None,
                    (2, true) => Some((gx, gy, gz + 1)),
                    _ => unreachable!(),
                };
                if neighbor
                    .map(|(nxp, nyp, nzp)| occupied[idx(nxp, nyp, nzp, tray.nx, tray.ny)])
                    .unwrap_or(true)
                {
                    contact_faces += 1;
                }
            }
        }
    }
    contact_faces as f32 / total_faces.max(1) as f32
}

fn compare_offsets_bl(a: &(usize, usize, usize), b: &(usize, usize, usize)) -> Ordering {
    a.2.cmp(&b.2)
        .then_with(|| a.1.cmp(&b.1))
        .then_with(|| a.0.cmp(&b.0))
}

fn settle_bl_offset(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    mut offset: (usize, usize, usize),
) -> Option<(usize, usize, usize)> {
    if !can_place_voxels(tray, occupied, object, offset) {
        return None;
    }
    loop {
        let mut moved = false;
        while offset.2 > 0
            && can_place_voxels(tray, occupied, object, (offset.0, offset.1, offset.2 - 1))
        {
            offset.2 -= 1;
            moved = true;
        }
        while offset.1 > 0
            && can_place_voxels(tray, occupied, object, (offset.0, offset.1 - 1, offset.2))
        {
            offset.1 -= 1;
            moved = true;
        }
        while offset.0 > 0
            && can_place_voxels(tray, occupied, object, (offset.0 - 1, offset.1, offset.2))
        {
            offset.0 -= 1;
            moved = true;
        }
        if !moved {
            return Some(offset);
        }
    }
}

fn can_place_voxels(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    offset: (usize, usize, usize),
) -> bool {
    if offset.0 + object.nx > tray.nx
        || offset.1 + object.ny > tray.ny
        || offset.2 + object.nz > tray.nz
    {
        return false;
    }
    object.cells.iter().all(|&(x, y, z)| {
        !occupied[idx(offset.0 + x, offset.1 + y, offset.2 + z, tray.nx, tray.ny)]
    })
}

fn linear_axis_reachable(
    tray: Tray,
    occupied: &[bool],
    object: &OrientedMesh,
    offset: (usize, usize, usize),
) -> bool {
    let max_x = tray.nx - object.nx;
    let max_y = tray.ny - object.ny;
    let max_z = tray.nz - object.nz;
    (0..=offset.0).all(|x| can_place_voxels(tray, occupied, object, (x, offset.1, offset.2)))
        || (offset.0..=max_x)
            .all(|x| can_place_voxels(tray, occupied, object, (x, offset.1, offset.2)))
        || (0..=offset.1).all(|y| can_place_voxels(tray, occupied, object, (offset.0, y, offset.2)))
        || (offset.1..=max_y)
            .all(|y| can_place_voxels(tray, occupied, object, (offset.0, y, offset.2)))
        || (0..=offset.2).all(|z| can_place_voxels(tray, occupied, object, (offset.0, offset.1, z)))
        || (offset.2..=max_z)
            .all(|z| can_place_voxels(tray, occupied, object, (offset.0, offset.1, z)))
}

fn push_best_placement(best: &mut Vec<Placement>, limit: usize, placement: Placement) {
    if limit == 0 {
        return;
    }
    best.push(placement);
    best.sort_by(compare_placements_by_cost);
    best.truncate(limit);
}

fn stamp_cells(tray: Tray, occupied: &mut [bool], cells: &[(usize, usize, usize)]) {
    for &(x, y, z) in cells {
        occupied[idx(x, y, z, tray.nx, tray.ny)] = true;
    }
}

fn occupied_cells_for_translation(
    tray: Tray,
    object: &OrientedMesh,
    translation: Vec3,
) -> Vec<(usize, usize, usize)> {
    let mut marked = vec![false; tray.len()];
    let mut cells = Vec::new();
    for &(x, y, z) in &object.cells {
        let min = translation
            + Vec3::new(
                x as f32 * tray.voxel,
                y as f32 * tray.voxel,
                z as f32 * tray.voxel,
            );
        let max = min + Vec3::new(tray.voxel, tray.voxel, tray.voxel);
        let x0 = voxel_floor(min.x, tray.voxel, tray.nx);
        let y0 = voxel_floor(min.y, tray.voxel, tray.ny);
        let z0 = voxel_floor(min.z, tray.voxel, tray.nz);
        let x1 = voxel_ceil_exclusive(max.x, tray.voxel, tray.nx);
        let y1 = voxel_ceil_exclusive(max.y, tray.voxel, tray.ny);
        let z1 = voxel_ceil_exclusive(max.z, tray.voxel, tray.nz);
        for gz in z0..z1 {
            for gy in y0..y1 {
                for gx in x0..x1 {
                    let grid_index = idx(gx, gy, gz, tray.nx, tray.ny);
                    if !marked[grid_index] {
                        marked[grid_index] = true;
                        cells.push((gx, gy, gz));
                    }
                }
            }
        }
    }
    cells
}

fn voxel_floor(value: f32, voxel: f32, limit: usize) -> usize {
    ((value / voxel).floor() as isize).clamp(0, limit as isize - 1) as usize
}

fn voxel_ceil_exclusive(value: f32, voxel: f32, limit: usize) -> usize {
    ((value / voxel).ceil() as isize).clamp(1, limit as isize) as usize
}

fn refine_translation(
    object: &OrientedMesh,
    initial: Vec3,
    placed: &[PlacedMesh],
    tray: Tray,
    margin: f32,
    deadline: Deadline,
) -> Vec3 {
    let mut translation = initial;
    let directions = [
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];
    for _ in 0..3 {
        for direction in directions {
            if deadline.expired() {
                return translation;
            }
            let mut lo = 0.0;
            let mut hi = tray.voxel;
            for _ in 0..12 {
                if deadline.expired() {
                    return translation;
                }
                let mid = (lo + hi) * 0.5;
                let candidate = translation + direction * mid;
                if placement_is_valid(object, candidate, placed, tray, margin, deadline) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            if lo > 1.0e-4 {
                translation = translation + direction * lo;
            }
        }
    }
    translation
}

fn placement_is_valid(
    object: &OrientedMesh,
    translation: Vec3,
    placed: &[PlacedMesh],
    tray: Tray,
    margin: f32,
    deadline: Deadline,
) -> bool {
    if deadline.expired() {
        return false;
    }
    let triangles = translated_triangles(&object.triangles, translation);
    let bbox = bbox_of_triangles(&triangles);
    if bbox.0.x < 0.0
        || bbox.0.y < 0.0
        || bbox.0.z < 0.0
        || bbox.1.x > tray.width
        || bbox.1.y > tray.depth
        || bbox.1.z > tray.height
    {
        return false;
    }
    !collides_with_placed(&triangles, bbox, placed, margin, deadline)
}

fn translated_triangles(triangles: &[[Vec3; 3]], translation: Vec3) -> Vec<[Vec3; 3]> {
    triangles
        .iter()
        .map(|tri| {
            [
                tri[0] + translation,
                tri[1] + translation,
                tri[2] + translation,
            ]
        })
        .collect()
}

fn collides_with_placed(
    triangles: &[[Vec3; 3]],
    bbox: (Vec3, Vec3),
    placed: &[PlacedMesh],
    margin: f32,
    deadline: Deadline,
) -> bool {
    for object in placed {
        if deadline.expired() {
            return true;
        }
        if !bbox_overlap(bbox, object.bbox, margin) {
            continue;
        }
        for tri_a in triangles {
            if deadline.expired() {
                return true;
            }
            let bbox_a = bbox_of_triangle(tri_a);
            for tri_b in &object.triangles {
                if bbox_overlap(bbox_a, bbox_of_triangle(tri_b), margin) {
                    return true;
                }
            }
        }
    }
    false
}

fn bbox_of_triangle(tri: &[Vec3; 3]) -> (Vec3, Vec3) {
    let mut min = tri[0];
    let mut max = tri[0];
    for &v in &tri[1..] {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

fn bbox_overlap(a: (Vec3, Vec3), b: (Vec3, Vec3), margin: f32) -> bool {
    a.0.x <= b.1.x + margin
        && a.1.x + margin >= b.0.x
        && a.0.y <= b.1.y + margin
        && a.1.y + margin >= b.0.y
        && a.0.z <= b.1.z + margin
        && a.1.z + margin >= b.0.z
}

fn reachable_offsets(tray: Tray, object: &OrientedMesh, collision: &[f32]) -> Vec<bool> {
    let ox = tray.nx - object.nx + 1;
    let oy = tray.ny - object.ny + 1;
    let oz = tray.nz - object.nz + 1;
    let mut feasible = vec![false; ox * oy * oz];
    for z in 0..oz {
        for y in 0..oy {
            for x in 0..ox {
                feasible[idx(x, y, z, ox, oy)] = collision[idx(x, y, z, tray.nx, tray.ny)] <= 0.5;
            }
        }
    }
    reachable_from_boundary(&feasible, ox, oy, oz)
}

fn reachable_from_boundary(feasible: &[bool], nx: usize, ny: usize, nz: usize) -> Vec<bool> {
    let mut reachable = vec![false; feasible.len()];
    let mut queue = VecDeque::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if x != 0 && y != 0 && z != 0 && x + 1 != nx && y + 1 != ny && z + 1 != nz {
                    continue;
                }
                let i = idx(x, y, z, nx, ny);
                if feasible[i] && !reachable[i] {
                    reachable[i] = true;
                    queue.push_back((x, y, z));
                }
            }
        }
    }
    while let Some((x, y, z)) = queue.pop_front() {
        for (nxp, nyp, nzp) in neighbors6(x, y, z, nx, ny, nz) {
            let i = idx(nxp, nyp, nzp, nx, ny);
            if feasible[i] && !reachable[i] {
                reachable[i] = true;
                queue.push_back((nxp, nyp, nzp));
            }
        }
    }
    reachable
}

fn ray_casting_disassembly(tray: Tray, placed: &[PlacedMesh]) -> RayDisassemblyReport {
    let mut remaining = vec![true; placed.len()];
    let directions = [
        (0, true),
        (0, false),
        (1, true),
        (1, false),
        (2, true),
        (2, false),
    ];
    let mut removed_by_rays = 0;
    let mut passes = 0;

    for _ in 0..2 {
        for direction in directions {
            passes += 1;
            let edges = directional_blocking_graph(tray, placed, &remaining, direction);
            let nodes = remaining_nodes(&remaining);
            let components = strongly_connected_components(&nodes, &edges);
            let mut removed_this_pass = Vec::new();
            for component in components {
                if component.len() == 1 && !edges[component[0]].contains(&component[0]) {
                    removed_this_pass.push(component[0]);
                }
            }
            if removed_this_pass.is_empty() {
                continue;
            }
            for object_id in removed_this_pass {
                if remaining[object_id] {
                    remaining[object_id] = false;
                    removed_by_rays += 1;
                }
            }
            if removed_by_rays == placed.len() {
                return RayDisassemblyReport {
                    removed_by_rays,
                    remaining_groups: Vec::new(),
                    passes,
                };
            }
        }
    }

    let remaining_ids = remaining_nodes(&remaining);
    let remaining_groups = if remaining_ids.is_empty() {
        Vec::new()
    } else {
        let mut union_edges = vec![Vec::new(); placed.len()];
        for direction in directions {
            let edges = directional_blocking_graph(tray, placed, &remaining, direction);
            for (from, to_list) in edges.into_iter().enumerate() {
                for to in to_list {
                    if !union_edges[from].contains(&to) {
                        union_edges[from].push(to);
                    }
                }
            }
        }
        strongly_connected_components(&remaining_ids, &union_edges)
            .into_iter()
            .collect()
    };

    RayDisassemblyReport {
        removed_by_rays,
        remaining_groups,
        passes,
    }
}

fn remaining_nodes(remaining: &[bool]) -> Vec<usize> {
    remaining
        .iter()
        .enumerate()
        .filter_map(|(id, &is_remaining)| is_remaining.then_some(id))
        .collect()
}

fn directional_blocking_graph(
    tray: Tray,
    placed: &[PlacedMesh],
    remaining: &[bool],
    direction: (usize, bool),
) -> Vec<Vec<usize>> {
    let mut owner = vec![None; tray.len()];
    for (object_id, object) in placed.iter().enumerate() {
        if !remaining[object_id] {
            continue;
        }
        for &(x, y, z) in &object.occupied_cells {
            let grid_index = idx(x, y, z, tray.nx, tray.ny);
            if owner[grid_index].is_none() {
                owner[grid_index] = Some(object_id);
            }
        }
    }

    let mut edge_set = HashSet::new();
    match direction {
        (0, positive) => {
            for z in 0..tray.nz {
                for y in 0..tray.ny {
                    scan_line_edges(
                        (0..tray.nx).map(|x| owner[idx(x, y, z, tray.nx, tray.ny)]),
                        positive,
                        &mut edge_set,
                    );
                }
            }
        }
        (1, positive) => {
            for z in 0..tray.nz {
                for x in 0..tray.nx {
                    scan_line_edges(
                        (0..tray.ny).map(|y| owner[idx(x, y, z, tray.nx, tray.ny)]),
                        positive,
                        &mut edge_set,
                    );
                }
            }
        }
        (2, positive) => {
            for y in 0..tray.ny {
                for x in 0..tray.nx {
                    scan_line_edges(
                        (0..tray.nz).map(|z| owner[idx(x, y, z, tray.nx, tray.ny)]),
                        positive,
                        &mut edge_set,
                    );
                }
            }
        }
        _ => unreachable!(),
    }

    let mut edges = vec![Vec::new(); placed.len()];
    for (from, to) in edge_set {
        edges[from].push(to);
    }
    edges
}

fn scan_line_edges<I>(line: I, positive: bool, edge_set: &mut HashSet<(usize, usize)>)
where
    I: IntoIterator<Item = Option<usize>>,
{
    let objects = line.into_iter().collect::<Vec<_>>();
    let iter: Box<dyn Iterator<Item = Option<usize>>> = if positive {
        Box::new(objects.into_iter())
    } else {
        Box::new(objects.into_iter().rev())
    };
    let mut last = None;
    for current in iter.flatten() {
        if Some(current) == last {
            continue;
        }
        if let Some(previous) = last {
            edge_set.insert((previous, current));
        }
        last = Some(current);
    }
}

fn strongly_connected_components(nodes: &[usize], edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Tarjan<'a> {
        edges: &'a [Vec<usize>],
        node_set: Vec<bool>,
        index: usize,
        indices: Vec<Option<usize>>,
        lowlinks: Vec<usize>,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        components: Vec<Vec<usize>>,
    }

    impl Tarjan<'_> {
        fn connect(&mut self, v: usize) {
            self.indices[v] = Some(self.index);
            self.lowlinks[v] = self.index;
            self.index += 1;
            self.stack.push(v);
            self.on_stack[v] = true;

            for &w in &self.edges[v] {
                if !self.node_set[w] {
                    continue;
                }
                if self.indices[w].is_none() {
                    self.connect(w);
                    self.lowlinks[v] = self.lowlinks[v].min(self.lowlinks[w]);
                } else if self.on_stack[w] {
                    self.lowlinks[v] = self.lowlinks[v].min(self.indices[w].unwrap());
                }
            }

            if self.lowlinks[v] == self.indices[v].unwrap() {
                let mut component = Vec::new();
                loop {
                    let w = self.stack.pop().unwrap();
                    self.on_stack[w] = false;
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                self.components.push(component);
            }
        }
    }

    let mut node_set = vec![false; edges.len()];
    for &node in nodes {
        node_set[node] = true;
    }
    let mut tarjan = Tarjan {
        edges,
        node_set,
        index: 0,
        indices: vec![None; edges.len()],
        lowlinks: vec![0; edges.len()],
        stack: Vec::new(),
        on_stack: vec![false; edges.len()],
        components: Vec::new(),
    };
    for &node in nodes {
        if tarjan.indices[node].is_none() {
            tarjan.connect(node);
        }
    }
    tarjan.components
}

fn manhattan_distance_field(tray: Tray, occupied: &[bool]) -> Vec<f32> {
    let mut dist = vec![u32::MAX; tray.len()];
    let mut queue = VecDeque::new();
    for z in 0..tray.nz {
        for y in 0..tray.ny {
            for x in 0..tray.nx {
                let i = idx(x, y, z, tray.nx, tray.ny);
                if occupied[i] || x == 0 || y == 0 || z == 0 {
                    dist[i] = 0;
                    queue.push_back((x, y, z));
                }
            }
        }
    }
    while let Some((x, y, z)) = queue.pop_front() {
        let base = dist[idx(x, y, z, tray.nx, tray.ny)];
        for (nxp, nyp, nzp) in neighbors6(x, y, z, tray.nx, tray.ny, tray.nz) {
            let ni = idx(nxp, nyp, nzp, tray.nx, tray.ny);
            if dist[ni] > base + 1 {
                dist[ni] = base + 1;
                queue.push_back((nxp, nyp, nzp));
            }
        }
    }
    dist.into_iter().map(|d| d as f32 * tray.voxel).collect()
}

fn fft_of_bool_grid(tray: Tray, grid: &[bool]) -> Vec<Complex32> {
    let mut data = grid
        .iter()
        .map(|&v| Complex32::new(if v { 1.0 } else { 0.0 }, 0.0))
        .collect::<Vec<_>>();
    fft3(&mut data, tray.nx, tray.ny, tray.nz, false);
    data
}

fn fft_of_scalar_grid(tray: Tray, grid: &[f32]) -> Vec<Complex32> {
    let mut data = grid
        .iter()
        .map(|&v| Complex32::new(v, 0.0))
        .collect::<Vec<_>>();
    fft3(&mut data, tray.nx, tray.ny, tray.nz, false);
    data
}

fn fft_of_object(tray: Tray, object: &OrientedMesh) -> Vec<Complex32> {
    let mut data = vec![Complex32::new(0.0, 0.0); tray.len()];
    for &(x, y, z) in &object.cells {
        data[idx(x, y, z, tray.nx, tray.ny)] = Complex32::new(1.0, 0.0);
    }
    fft3(&mut data, tray.nx, tray.ny, tray.nz, false);
    data
}

fn inverse_product(
    field_fft: &[Complex32],
    object_fft: &[Complex32],
    tray: Tray,
    conjugate_object: bool,
) -> Vec<f32> {
    let mut data = field_fft
        .iter()
        .zip(object_fft)
        .map(|(&a, &b)| {
            if conjugate_object {
                a * b.conj()
            } else {
                a * b
            }
        })
        .collect::<Vec<_>>();
    fft3(&mut data, tray.nx, tray.ny, tray.nz, true);
    data.into_iter()
        .map(|c| if c.re.abs() < 1.0e-4 { 0.0 } else { c.re })
        .collect()
}

fn fft3(data: &mut [Complex32], nx: usize, ny: usize, nz: usize, inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let fft_x = if inverse {
        planner.plan_fft_inverse(nx)
    } else {
        planner.plan_fft_forward(nx)
    };
    let fft_y = if inverse {
        planner.plan_fft_inverse(ny)
    } else {
        planner.plan_fft_forward(ny)
    };
    let fft_z = if inverse {
        planner.plan_fft_inverse(nz)
    } else {
        planner.plan_fft_forward(nz)
    };

    let mut scratch = vec![Complex32::default(); nx.max(ny).max(nz)];
    for z in 0..nz {
        for y in 0..ny {
            let start = idx(0, y, z, nx, ny);
            fft_x.process(&mut data[start..start + nx]);
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            for y in 0..ny {
                scratch[y] = data[idx(x, y, z, nx, ny)];
            }
            fft_y.process(&mut scratch[..ny]);
            for y in 0..ny {
                data[idx(x, y, z, nx, ny)] = scratch[y];
            }
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            for z in 0..nz {
                scratch[z] = data[idx(x, y, z, nx, ny)];
            }
            fft_z.process(&mut scratch[..nz]);
            for z in 0..nz {
                data[idx(x, y, z, nx, ny)] = scratch[z];
            }
        }
    }
    if inverse {
        let scale = (nx * ny * nz) as f32;
        for value in data {
            *value /= scale;
        }
    }
}

fn voxelize_mesh(mesh: &Mesh, rotation: Rotation, voxel: f32) -> Result<OrientedMesh> {
    let rotated = mesh
        .triangles
        .iter()
        .map(|tri| {
            [
                rotation.apply(tri[0]),
                rotation.apply(tri[1]),
                rotation.apply(tri[2]),
            ]
        })
        .collect::<Vec<_>>();
    let (min, max) = bbox_of_triangles(&rotated);
    let extent = max - min;
    let nx = (extent.x / voxel).ceil() as usize + 1;
    let ny = (extent.y / voxel).ceil() as usize + 1;
    let nz = (extent.z / voxel).ceil() as usize + 1;
    let local = rotated
        .iter()
        .map(|tri| [tri[0] - min, tri[1] - min, tri[2] - min])
        .collect::<Vec<_>>();
    let mut surface = vec![false; nx * ny * nz];
    for tri in &local {
        mark_intersecting_voxels(*tri, voxel, nx, ny, nz, &mut surface);
    }
    let occupied = fill_interior(&surface, nx, ny, nz);
    let cells = occupied_cells_local(&occupied, nx, ny, nz);
    let voxel_count = cells.len();
    Ok(OrientedMesh {
        name: mesh.name.clone(),
        triangles: local,
        occupied,
        cells,
        nx,
        ny,
        nz,
        voxel_count,
    })
}

fn occupied_cells_local(
    occupied: &[bool],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<(usize, usize, usize)> {
    let mut cells = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if occupied[idx(x, y, z, nx, ny)] {
                    cells.push((x, y, z));
                }
            }
        }
    }
    cells
}

fn mark_intersecting_voxels(
    tri: [Vec3; 3],
    voxel: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    grid: &mut [bool],
) {
    let (tri_min, tri_max) = bbox_of_triangle(&tri);
    let x0 = voxel_floor(tri_min.x, voxel, nx);
    let y0 = voxel_floor(tri_min.y, voxel, ny);
    let z0 = voxel_floor(tri_min.z, voxel, nz);
    let x1 = voxel_ceil_exclusive(tri_max.x, voxel, nx);
    let y1 = voxel_ceil_exclusive(tri_max.y, voxel, ny);
    let z1 = voxel_ceil_exclusive(tri_max.z, voxel, nz);

    for z in z0..z1 {
        for y in y0..y1 {
            for x in x0..x1 {
                let min = Vec3::new(x as f32 * voxel, y as f32 * voxel, z as f32 * voxel);
                let max = min + Vec3::new(voxel, voxel, voxel);
                if triangle_intersects_aabb(tri, min, max) {
                    grid[idx(x, y, z, nx, ny)] = true;
                }
            }
        }
    }
}

fn triangle_intersects_aabb(tri: [Vec3; 3], box_min: Vec3, box_max: Vec3) -> bool {
    let (tri_min, tri_max) = bbox_of_triangle(&tri);
    if !bbox_overlap((tri_min, tri_max), (box_min, box_max), 0.0) {
        return false;
    }

    let center = (box_min + box_max) * 0.5;
    let half = (box_max - box_min) * 0.5;
    let v0 = tri[0] - center;
    let v1 = tri[1] - center;
    let v2 = tri[2] - center;
    let vertices = [v0, v1, v2];

    if axis_separates(Vec3::new(1.0, 0.0, 0.0), &vertices, half)
        || axis_separates(Vec3::new(0.0, 1.0, 0.0), &vertices, half)
        || axis_separates(Vec3::new(0.0, 0.0, 1.0), &vertices, half)
    {
        return false;
    }

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let edges = [e0, e1, e2];
    let box_axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for edge in edges {
        for box_axis in box_axes {
            if axis_separates(edge.cross(box_axis), &vertices, half) {
                return false;
            }
        }
    }

    let normal = e0.cross(e1);
    !axis_separates(normal, &vertices, half)
}

fn axis_separates(axis: Vec3, vertices: &[Vec3; 3], half: Vec3) -> bool {
    const EPS: f32 = 1.0e-6;
    if axis.dot(axis) <= EPS {
        return false;
    }
    let p0 = vertices[0].dot(axis);
    let p1 = vertices[1].dot(axis);
    let p2 = vertices[2].dot(axis);
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    let radius = half.x * axis.x.abs() + half.y * axis.y.abs() + half.z * axis.z.abs();
    min > radius + EPS || max < -radius - EPS
}

fn fill_interior(surface: &[bool], nx: usize, ny: usize, nz: usize) -> Vec<bool> {
    let px = nx + 2;
    let py = ny + 2;
    let pz = nz + 2;
    let mut blocked = vec![false; px * py * pz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if surface[idx(x, y, z, nx, ny)] {
                    blocked[idx(x + 1, y + 1, z + 1, px, py)] = true;
                }
            }
        }
    }
    let mut outside = vec![false; blocked.len()];
    let mut queue = VecDeque::new();
    outside[0] = true;
    queue.push_back((0, 0, 0));
    while let Some((x, y, z)) = queue.pop_front() {
        for (nxp, nyp, nzp) in neighbors6(x, y, z, px, py, pz) {
            let i = idx(nxp, nyp, nzp, px, py);
            if !blocked[i] && !outside[i] {
                outside[i] = true;
                queue.push_back((nxp, nyp, nzp));
            }
        }
    }

    let mut occupied = vec![false; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let original = idx(x, y, z, nx, ny);
                let padded = idx(x + 1, y + 1, z + 1, px, py);
                occupied[original] = surface[original] || !outside[padded];
            }
        }
    }
    occupied
}

fn bbox_of_triangles(triangles: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for tri in triangles {
        for &v in tri {
            min = min.min(v);
            max = max.max(v);
        }
    }
    (min, max)
}

fn collect_stl_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for input in inputs {
        if input.is_dir() {
            for entry in fs::read_dir(input).with_context(|| {
                format!("ディレクトリ {} の読み込みに失敗しました", input.display())
            })? {
                let path = entry?.path();
                if path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("stl"))
                    .unwrap_or(false)
                {
                    paths.push(path);
                }
            }
        } else {
            paths.push(input.clone());
        }
    }
    paths.sort();
    Ok(paths)
}

fn load_stl(path: &Path) -> Result<Mesh> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("{} を開けませんでした", path.display()))?
        .read_to_end(&mut bytes)?;
    let triangles = if is_binary_stl(&bytes) {
        parse_binary_stl(&bytes)?
    } else {
        parse_ascii_stl(&String::from_utf8_lossy(&bytes))?
    };
    if triangles.is_empty() {
        bail!("{} に三角形が含まれていません", path.display());
    }
    Ok(Mesh {
        name: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("object")
            .to_string(),
        triangles,
    })
}

fn is_binary_stl(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    84 + count * 50 == bytes.len()
}

fn parse_binary_stl(bytes: &[u8]) -> Result<Vec<[Vec3; 3]>> {
    let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    let mut triangles = Vec::with_capacity(count);
    let mut offset = 84;
    for _ in 0..count {
        offset += 12; // 法線
        let mut tri = [Vec3::default(); 3];
        for vertex in &mut tri {
            let x = read_f32_le(bytes, offset)?;
            let y = read_f32_le(bytes, offset + 4)?;
            let z = read_f32_le(bytes, offset + 8)?;
            *vertex = Vec3::new(x, y, z);
            offset += 12;
        }
        offset += 2; // 属性バイト数
        triangles.push(tri);
    }
    Ok(triangles)
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("binary STLが途中で切れています"))?;
    Ok(f32::from_le_bytes(slice.try_into().unwrap()))
}

fn parse_ascii_stl(text: &str) -> Result<Vec<[Vec3; 3]>> {
    let mut vertices = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("vertex") {
            let values = rest
                .split_whitespace()
                .map(str::parse::<f32>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if values.len() != 3 {
                bail!("ASCII STLの頂点行が不正です: {line}");
            }
            vertices.push(Vec3::new(values[0], values[1], values[2]));
        }
    }
    if vertices.len() % 3 != 0 {
        bail!("ASCII STLの三角形が不完全です");
    }
    Ok(vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn write_combined_stl(path: &Path, placed: &[PlacedMesh]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut file = File::create(path)?;
    for object in placed {
        writeln!(file, "solid {}", object.name)?;
        for tri in &object.triangles {
            write_facet(&mut file, tri)?;
        }
        writeln!(file, "endsolid {}", object.name)?;
    }
    Ok(())
}

fn write_mesh_stl(path: &Path, name: &str, triangles: &[[Vec3; 3]]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "solid {name}")?;
    for tri in triangles {
        write_facet(&mut file, tri)?;
    }
    writeln!(file, "endsolid {name}")?;
    Ok(())
}

fn write_facet(file: &mut File, tri: &[Vec3; 3]) -> Result<()> {
    let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    let len = normal.length();
    let n = if len > 0.0 {
        normal * (1.0 / len)
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    };
    writeln!(file, "  facet normal {} {} {}", n.x, n.y, n.z)?;
    writeln!(file, "    outer loop")?;
    for v in tri {
        writeln!(file, "      vertex {} {} {}", v.x, v.y, v.z)?;
    }
    writeln!(file, "    endloop")?;
    writeln!(file, "  endfacet")?;
    Ok(())
}

fn write_sample_set(output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let samples = vec![
        (
            "box_12x8x6",
            cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(12.0, 8.0, 6.0)),
        ),
        (
            "box_10x10x4",
            cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 4.0)),
        ),
        (
            "l_block",
            joined_boxes(&[
                (Vec3::new(0.0, 0.0, 0.0), Vec3::new(14.0, 5.0, 5.0)),
                (Vec3::new(0.0, 5.0, 0.0), Vec3::new(5.0, 14.0, 5.0)),
                (Vec3::new(0.0, 0.0, 5.0), Vec3::new(5.0, 5.0, 12.0)),
            ]),
        ),
        (
            "u_bridge",
            joined_boxes(&[
                (Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 14.0, 5.0)),
                (Vec3::new(12.0, 0.0, 0.0), Vec3::new(16.0, 14.0, 5.0)),
                (Vec3::new(0.0, 10.0, 0.0), Vec3::new(16.0, 14.0, 5.0)),
            ]),
        ),
        ("tri_prism", triangular_prism(14.0, 9.0, 7.0)),
        ("low_cylinder", cylinder(5.0, 6.0, 24)),
        ("tall_cylinder", cylinder(4.0, 13.0, 24)),
        ("star", star_prism(8.0, 4.2, 5.0, 5)),
    ];
    for (name, triangles) in samples {
        write_mesh_stl(&output.join(format!("{name}.stl")), name, &triangles)?;
    }
    Ok(())
}

fn cuboid(min: Vec3, max: Vec3) -> Vec<[Vec3; 3]> {
    let p = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    vec![
        [p[0], p[2], p[1]],
        [p[0], p[3], p[2]],
        [p[4], p[5], p[6]],
        [p[4], p[6], p[7]],
        [p[0], p[1], p[5]],
        [p[0], p[5], p[4]],
        [p[1], p[2], p[6]],
        [p[1], p[6], p[5]],
        [p[2], p[3], p[7]],
        [p[2], p[7], p[6]],
        [p[3], p[0], p[4]],
        [p[3], p[4], p[7]],
    ]
}

fn joined_boxes(boxes: &[(Vec3, Vec3)]) -> Vec<[Vec3; 3]> {
    boxes
        .iter()
        .flat_map(|&(min, max)| cuboid(min, max))
        .collect()
}

fn triangular_prism(length: f32, width: f32, height: f32) -> Vec<[Vec3; 3]> {
    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(length, 0.0, 0.0);
    let p2 = Vec3::new(0.0, width, 0.0);
    let p3 = Vec3::new(0.0, 0.0, height);
    let p4 = Vec3::new(length, 0.0, height);
    let p5 = Vec3::new(0.0, width, height);
    vec![
        [p0, p1, p2],
        [p3, p5, p4],
        [p0, p3, p4],
        [p0, p4, p1],
        [p1, p4, p5],
        [p1, p5, p2],
        [p2, p5, p3],
        [p2, p3, p0],
    ]
}

fn cylinder(radius: f32, height: f32, segments: usize) -> Vec<[Vec3; 3]> {
    let mut tris = Vec::new();
    let bottom_center = Vec3::new(radius, radius, 0.0);
    let top_center = Vec3::new(radius, radius, height);
    for i in 0..segments {
        let a = 2.0 * PI * i as f32 / segments as f32;
        let b = 2.0 * PI * (i + 1) as f32 / segments as f32;
        let p0 = Vec3::new(radius + radius * a.cos(), radius + radius * a.sin(), 0.0);
        let p1 = Vec3::new(radius + radius * b.cos(), radius + radius * b.sin(), 0.0);
        let p2 = Vec3::new(p0.x, p0.y, height);
        let p3 = Vec3::new(p1.x, p1.y, height);
        tris.push([bottom_center, p1, p0]);
        tris.push([top_center, p2, p3]);
        tris.push([p0, p1, p3]);
        tris.push([p0, p3, p2]);
    }
    tris
}

fn star_prism(outer: f32, inner: f32, height: f32, points: usize) -> Vec<[Vec3; 3]> {
    let count = points * 2;
    let center = Vec3::new(outer, outer, 0.0);
    let top_center = Vec3::new(outer, outer, height);
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for i in 0..count {
        let r = if i % 2 == 0 { outer } else { inner };
        let a = PI * i as f32 / points as f32 - PI / 2.0;
        bottom.push(Vec3::new(outer + r * a.cos(), outer + r * a.sin(), 0.0));
        top.push(Vec3::new(outer + r * a.cos(), outer + r * a.sin(), height));
    }
    let mut tris = Vec::new();
    for i in 0..count {
        let j = (i + 1) % count;
        tris.push([center, bottom[j], bottom[i]]);
        tris.push([top_center, top[i], top[j]]);
        tris.push([bottom[i], bottom[j], top[j]]);
        tris.push([bottom[i], top[j], top[i]]);
    }
    tris
}

fn rotation_set(count: usize) -> Vec<Rotation> {
    let mut rotations = vec![Rotation::identity()];
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for perm in perms {
        for sx in [-1, 1] {
            for sy in [-1, 1] {
                for sz in [-1, 1] {
                    let signs = [sx, sy, sz];
                    let mut m = [[0.0; 3]; 3];
                    for row in 0..3 {
                        m[row][perm[row]] = signs[row] as f32;
                    }
                    let rotation = Rotation { m };
                    if (determinant(m) - 1.0).abs() < 1.0e-5 {
                        push_unique_rotation(&mut rotations, rotation);
                    }
                }
            }
        }
    }
    if rotations.len() < count {
        let angles = [
            PI / 4.0,
            -PI / 4.0,
            PI / 6.0,
            -PI / 6.0,
            PI / 3.0,
            -PI / 3.0,
        ];
        for angle in angles {
            for euler in [
                (angle, 0.0, 0.0),
                (0.0, angle, 0.0),
                (0.0, 0.0, angle),
                (angle, angle, 0.0),
                (angle, 0.0, angle),
                (0.0, angle, angle),
                (angle, angle, angle),
            ] {
                push_unique_rotation(
                    &mut rotations,
                    Rotation::from_euler(euler.0, euler.1, euler.2),
                );
                if rotations.len() >= count {
                    return rotations;
                }
            }
        }
    }
    rotations.truncate(count);
    rotations
}

fn push_unique_rotation(rotations: &mut Vec<Rotation>, rotation: Rotation) {
    if !rotations
        .iter()
        .any(|existing| rotations_approx_eq(*existing, rotation))
    {
        rotations.push(rotation);
    }
}

fn rotations_approx_eq(a: Rotation, b: Rotation) -> bool {
    a.m.iter()
        .flatten()
        .zip(b.m.iter().flatten())
        .all(|(a, b)| (*a - *b).abs() < 1.0e-5)
}

fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = (0..3).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    out
}

fn determinant(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn print_summary(result: &PackResult, tray: Tray, out: &Path) {
    let voxel_volume = tray.voxel.powi(3);
    let occupied_volume = result
        .placed
        .iter()
        .map(|p| p.voxel_count as f32 * voxel_volume)
        .sum::<f32>();
    let mesh_volume = result.placed.iter().map(|p| p.mesh_volume).sum::<f32>();
    let tray_volume = tray.width * tray.depth * tray.height;
    println!(
        "{}個の物体を {} にパックしました",
        result.placed.len(),
        out.display()
    );
    println!(
        "ボクセル密度: {:.2}% | メッシュ密度: {:.2}%",
        occupied_volume / tray_volume * 100.0,
        mesh_volume / tray_volume * 100.0
    );
    if !result.unpacked.is_empty() {
        println!(
            "未パックの物体 {}個: {}",
            result.unpacked.len(),
            result.unpacked.join(", ")
        );
    }
    if result.timed_out {
        println!(
            "時間制限により打ち切りました。部分結果として{}個の物体を出力します",
            result.placed.len()
        );
    }
    if let Some(report) = &result.post_report {
        println!(
            "後処理最適化: {}パス、{}グループ試行、{}回採用、{}個を再挿入、未解決グループ{}個",
            report.passes,
            report.attempted_groups,
            report.accepted_reinsertions,
            report.reinserted_objects,
            report.unresolved_groups
        );
        if report.repack_passes > 0 || report.repack_attempted_groups > 0 {
            println!(
                "局所再パック: {}パス、{}グループ試行、{}回採用、{}個を追加配置",
                report.repack_passes,
                report.repack_attempted_groups,
                report.repack_accepted,
                report.repack_added_objects
            );
        }
    }
    if let Some(report) = &result.ray_report {
        if report.remaining_groups.is_empty() {
            println!(
                "ray分解判定: パック済み{}個すべてがDirectional Blocking解析で取り出し可能（{}パス）",
                report.removed_by_rays, report.passes
            );
        } else {
            println!(
                "ray分解判定: {}個はrayで取り出し可能。噛み込みの可能性があるグループが{}個残っています",
                report.removed_by_rays,
                report.remaining_groups.len()
            );
            for group in &report.remaining_groups {
                let names = group
                    .iter()
                    .map(|&object_id| result.placed[object_id].name.as_str())
                    .collect::<Vec<_>>();
                println!("  残ったグループ: {}", names.join(", "));
            }
        }
    }
    for placed in &result.placed {
        println!(
            "  {} オフセット={:?} 平行移動=({:.3}, {:.3}, {:.3}) refinement=({:.3}, {:.3}, {:.3}) 回転={}",
            placed.name,
            placed.offset,
            placed.translation.x,
            placed.translation.y,
            placed.translation.z,
            placed.refinement.x,
            placed.refinement.y,
            placed.refinement.z,
            placed.rotation_index
        );
    }
}

fn neighbors6(
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut out = Vec::with_capacity(6);
    if x > 0 {
        out.push((x - 1, y, z));
    }
    if x + 1 < nx {
        out.push((x + 1, y, z));
    }
    if y > 0 {
        out.push((x, y - 1, z));
    }
    if y + 1 < ny {
        out.push((x, y + 1, z));
    }
    if z > 0 {
        out.push((x, y, z - 1));
    }
    if z + 1 < nz {
        out.push((x, y, z + 1));
    }
    out.into_iter()
}

fn idx(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    (z * ny + y) * nx + x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_correlation_matches_bruteforce_inside_domain() {
        let tray = Tray::new(4.0, 3.0, 2.0, 1.0).unwrap();
        let mut field = vec![false; tray.len()];
        field[idx(2, 1, 0, tray.nx, tray.ny)] = true;
        field[idx(3, 2, 1, tray.nx, tray.ny)] = true;

        let object = OrientedMesh {
            name: "test".to_string(),
            triangles: Vec::new(),
            occupied: vec![true, true, false, false],
            cells: vec![(0, 0, 0), (1, 0, 0)],
            nx: 2,
            ny: 1,
            nz: 2,
            voxel_count: 2,
        };
        let field_fft = fft_of_bool_grid(tray, &field);
        let object_fft = fft_of_object(tray, &object);
        let corr = inverse_product(&field_fft, &object_fft, tray, true);
        for z in 0..=tray.nz - object.nz {
            for y in 0..=tray.ny - object.ny {
                for x in 0..=tray.nx - object.nx {
                    let mut brute = 0.0;
                    for oz in 0..object.nz {
                        for oy in 0..object.ny {
                            for ox in 0..object.nx {
                                if object.occupied[idx(ox, oy, oz, object.nx, object.ny)]
                                    && field[idx(x + ox, y + oy, z + oz, tray.nx, tray.ny)]
                                {
                                    brute += 1.0;
                                }
                            }
                        }
                    }
                    assert!(
                        (corr[idx(x, y, z, tray.nx, tray.ny)] - brute).abs() < 1.0e-3,
                        "mismatch at {x},{y},{z}"
                    );
                }
            }
        }
    }

    #[test]
    fn flood_fill_keeps_enclosed_offset_unreachable() {
        let nx = 5;
        let ny = 5;
        let nz = 3;
        let mut feasible = vec![true; nx * ny * nz];
        feasible[idx(2, 1, 1, nx, ny)] = false;
        feasible[idx(1, 2, 1, nx, ny)] = false;
        feasible[idx(3, 2, 1, nx, ny)] = false;
        feasible[idx(2, 3, 1, nx, ny)] = false;
        feasible[idx(2, 2, 0, nx, ny)] = false;
        feasible[idx(2, 2, 2, nx, ny)] = false;
        let reachable = reachable_from_boundary(&feasible, nx, ny, nz);
        assert!(!reachable[idx(2, 2, 1, nx, ny)]);
        assert!(reachable[idx(0, 0, 0, nx, ny)]);
    }

    #[test]
    fn generated_rotations_are_right_handed() {
        let rotations = rotation_set(24);
        assert_eq!(rotations.len(), 24);
        assert!(rotations
            .iter()
            .all(|r| (determinant(r.m) - 1.0).abs() < 1.0e-5));
        assert!(rotation_set(32).len() > 24);
    }

    #[test]
    fn triangle_aabb_overlap_marks_plane_slice_without_vertices_inside() {
        let tri = [
            Vec3::new(-1.0, -1.0, 0.5),
            Vec3::new(2.0, -1.0, 0.5),
            Vec3::new(-1.0, 2.0, 0.5),
        ];
        assert!(triangle_intersects_aabb(
            tri,
            Vec3::new(0.25, 0.25, 0.0),
            Vec3::new(0.75, 0.75, 1.0)
        ));
    }

    #[test]
    fn conservative_voxelization_marks_every_intersected_cell() {
        let tri = [
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(3.0, 0.0, 0.5),
            Vec3::new(0.0, 3.0, 0.5),
        ];
        let mut grid = vec![false; 4 * 4 * 2];
        mark_intersecting_voxels(tri, 1.0, 4, 4, 2, &mut grid);
        assert!(grid[idx(0, 0, 0, 4, 4)]);
        assert!(grid[idx(1, 1, 0, 4, 4)]);
        assert!(grid[idx(2, 0, 0, 4, 4)]);
        assert!(!grid[idx(3, 3, 0, 4, 4)]);
    }

    #[test]
    fn ray_disassembly_removes_simple_chain() {
        let tray = Tray::new(4.0, 1.0, 1.0, 1.0).unwrap();
        let placed = vec![
            test_placed("a", vec![(0, 0, 0)]),
            test_placed("b", vec![(2, 0, 0)]),
        ];
        let report = ray_casting_disassembly(tray, &placed);
        assert_eq!(report.removed_by_rays, 2);
        assert!(report.remaining_groups.is_empty());
    }

    #[test]
    fn post_optimizer_reinserts_interlocked_group_when_it_improves_disassembly() {
        let tray = Tray::new(5.0, 5.0, 5.0, 1.0).unwrap();
        let meshes = vec![
            Mesh {
                name: "a".to_string(),
                triangles: cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)),
            },
            Mesh {
                name: "b".to_string(),
                triangles: cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)),
            },
        ];
        let mut placed = vec![
            test_placed_with_source(
                "a",
                0,
                vec![
                    (0, 1, 1),
                    (2, 1, 1),
                    (1, 0, 1),
                    (1, 2, 1),
                    (1, 1, 0),
                    (1, 1, 2),
                ],
            ),
            test_placed_with_source("b", 1, vec![(1, 1, 1)]),
        ];
        let before = ray_casting_disassembly(tray, &placed);
        assert!(!before.remaining_groups.is_empty());

        let post = optimize_disassembly(
            &meshes,
            tray,
            &rotation_set(1),
            1.0,
            true,
            false,
            0.0,
            2,
            &mut placed,
            Deadline { ends_at: None },
        )
        .unwrap();
        let post = post.0;
        let after = ray_casting_disassembly(tray, &placed);
        assert!(after.remaining_groups.is_empty());
        assert_eq!(post.accepted_reinsertions, 1);
    }

    #[test]
    fn refinement_moves_toward_tray_floor() {
        let tray = Tray::new(5.0, 5.0, 5.0, 1.0).unwrap();
        let mesh = Mesh {
            name: "box".to_string(),
            triangles: cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        };
        let object = voxelize_mesh(&mesh, Rotation::identity(), 1.0).unwrap();
        let refined = refine_translation(
            &object,
            Vec3::new(1.0, 1.0, 1.0),
            &[],
            tray,
            0.0,
            Deadline { ends_at: None },
        );
        assert!(refined.z < 0.01);
        assert!(refined.x < 0.01);
        assert!(refined.y < 0.01);
    }

    #[test]
    fn expired_deadline_returns_partial_result() {
        let tray = Tray::new(5.0, 5.0, 5.0, 1.0).unwrap();
        let meshes = vec![Mesh {
            name: "box".to_string(),
            triangles: cuboid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        }];
        let result = pack_meshes(
            &meshes,
            tray,
            &rotation_set(1),
            1.0,
            true,
            true,
            0.0,
            true,
            1,
            1,
            PlacementStrategy::Spectral,
            12,
            256,
            0,
            8,
            8,
            Deadline::from_seconds(Some(0.0)),
        )
        .unwrap();
        assert!(result.timed_out);
        assert!(result.placed.is_empty());
    }

    fn test_placed(name: &str, occupied_cells: Vec<(usize, usize, usize)>) -> PlacedMesh {
        test_placed_with_source(name, 0, occupied_cells)
    }

    fn test_placed_with_source(
        name: &str,
        source_index: usize,
        occupied_cells: Vec<(usize, usize, usize)>,
    ) -> PlacedMesh {
        PlacedMesh {
            name: name.to_string(),
            source_index,
            triangles: Vec::new(),
            offset: (0, 0, 0),
            translation: Vec3::default(),
            refinement: Vec3::default(),
            rotation_index: 0,
            voxel_count: occupied_cells.len(),
            mesh_volume: occupied_cells.len() as f32,
            bbox: (Vec3::default(), Vec3::default()),
            occupied_cells,
        }
    }
}
