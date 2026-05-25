#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saint-Venant dispatch simulation for the 0-456 main canal.

Current dispatch condition:
- head inflow ramps from 0 to 80 m3/s;
- diversion outlet capacities are set to 20, 20, 5, 5, 12, 5, and 5 m3/s
  with their corresponding assumed demands;
- the upstream ghost cell uses Manning normal depth compatible with the
  boundary discharge, while downstream open boundaries use extrapolated states;
- diversion discharge is constrained by specified outlet capacity, estimated
  branch-canal safe capacity, remaining demand, and local available water.

The branch-canal capacity is only a first engineering bound because no branch
water-level or gate geometry is available. It is estimated with Manning's
formula at h_safe = 0.90D of the first branch canal segment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

import muskingum_cunge_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "data" / "configuration.json"

G = 9.81
MAIN_START = 0
MAIN_END = 456
DIVERSION_NODES = [71, 89, 150, 194, 287, 349, 383]
_NORMAL_DEPTH_CACHE: Dict[tuple, float] = {}


def _read_configuration(path: Path = CONFIG_PATH) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_CONFIG_DEFAULTS = _read_configuration()


def _config_float(data: Mapping[str, object], *keys: str, default: float) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    return default


def _config_int(data: Mapping[str, object], *keys: str, default: int) -> int:
    for key in keys:
        if key in data:
            return int(data[key])
    return default


def _config_str(data: Mapping[str, object], *keys: str, default: str) -> str:
    for key in keys:
        if key in data:
            return str(data[key])
    return default


def _duration_hours(data: Mapping[str, object], default: float = 14.0) -> float:
    if "duration hours" in data or "duration_hours" in data:
        return _config_float(data, "duration hours", "duration_hours", default=default)
    if "start" in data and "stop" in data:
        return max(float(data["stop"]) - float(data["start"]), 0.0)
    return default


@dataclass
class DiversionSpec:
    node: int
    max_flow_m3s: float
    demand_m3: float


@dataclass
class BranchLimit:
    branch_node: int | None
    safe_capacity_m3s: float
    depth_m: float | None
    bottom_width_m: float | None
    bed_slope: float | None
    entry_length_m: float | None = None


@dataclass
class Config:
    solver: str = _config_str(_CONFIG_DEFAULTS, "solver", "scheme", default="implicit-network")
    start_hours: float = _config_float(_CONFIG_DEFAULTS, "start", "start_hours", default=0.0)
    dt_seconds: float = _config_float(_CONFIG_DEFAULTS, "time step", "dt_seconds", default=1.0)
    space_step_m: float = _config_float(_CONFIG_DEFAULTS, "space step", "space_step_m", default=0.0)
    duration_hours: float = _duration_hours(_CONFIG_DEFAULTS, default=14.0)
    ramp_hours: float = _config_float(_CONFIG_DEFAULTS, "ramp hours", "ramp_hours", default=0.5)
    target_head_flow_m3s: float = _config_float(
        _CONFIG_DEFAULTS,
        "target head flow m3s",
        "target_head_flow_m3s",
        default=80.0,
    )
    initial_depth_m: float = _config_float(_CONFIG_DEFAULTS, "initial depth m", "initial_depth_m", default=0.08)
    min_depth_m: float = _config_float(_CONFIG_DEFAULTS, "min depth m", "min_depth_m", default=0.03)
    output_interval_seconds: float = _config_float(
        _CONFIG_DEFAULTS,
        "output interval seconds",
        "output_interval_seconds",
        default=60.0,
    )
    safe_depth_ratio: float = _config_float(_CONFIG_DEFAULTS, "safe depth ratio", "safe_depth_ratio", default=0.90)
    min_bed_slope: float = _config_float(_CONFIG_DEFAULTS, "min bed slope", "min_bed_slope", default=5.0e-5)
    implicit_picard_iterations: int = _config_int(
        _CONFIG_DEFAULTS,
        "implicit picard iterations",
        "implicit_picard_iterations",
        default=4,
    )
    implicit_active_set_iterations: int = _config_int(
        _CONFIG_DEFAULTS,
        "implicit active set iterations",
        "implicit_active_set_iterations",
        default=5,
    )


def load_config(path: Path = CONFIG_PATH) -> Config:
    data = _read_configuration(path)
    return Config(
        solver=_config_str(data, "solver", "scheme", default=Config.solver),
        start_hours=_config_float(data, "start", "start_hours", default=Config.start_hours),
        dt_seconds=_config_float(data, "time step", "dt_seconds", default=Config.dt_seconds),
        space_step_m=_config_float(data, "space step", "space_step_m", default=Config.space_step_m),
        duration_hours=_duration_hours(data, default=Config.duration_hours),
        ramp_hours=_config_float(data, "ramp hours", "ramp_hours", default=Config.ramp_hours),
        target_head_flow_m3s=_config_float(
            data,
            "target head flow m3s",
            "target_head_flow_m3s",
            default=Config.target_head_flow_m3s,
        ),
        initial_depth_m=_config_float(data, "initial depth m", "initial_depth_m", default=Config.initial_depth_m),
        min_depth_m=_config_float(data, "min depth m", "min_depth_m", default=Config.min_depth_m),
        output_interval_seconds=_config_float(
            data,
            "output interval seconds",
            "output_interval_seconds",
            default=Config.output_interval_seconds,
        ),
        safe_depth_ratio=_config_float(data, "safe depth ratio", "safe_depth_ratio", default=Config.safe_depth_ratio),
        min_bed_slope=_config_float(data, "min bed slope", "min_bed_slope", default=Config.min_bed_slope),
        implicit_picard_iterations=_config_int(
            data,
            "implicit picard iterations",
            "implicit_picard_iterations",
            default=Config.implicit_picard_iterations,
        ),
        implicit_active_set_iterations=_config_int(
            data,
            "implicit active set iterations",
            "implicit_active_set_iterations",
            default=Config.implicit_active_set_iterations,
        ),
    )


@dataclass
class ChannelGrid:
    name: str
    ids: List[int]
    x: np.ndarray
    dx_cell: np.ndarray
    bed: np.ndarray
    depth: np.ndarray
    bottom_width: np.ndarray
    side_slope: np.ndarray
    manning_n: np.ndarray
    slope: np.ndarray


@dataclass
class BranchNetwork:
    grid: ChannelGrid
    starts: Dict[int, int]
    ends: Dict[int, int]
    internal_left: np.ndarray
    internal_right: np.ndarray


SPECS: Dict[int, DiversionSpec] = {
    71: DiversionSpec(71, 20.0, 250_000.0),
    89: DiversionSpec(89, 20.0, 300_000.0),
    150: DiversionSpec(150, 5.0, 60_000.0),
    194: DiversionSpec(194, 5.0, 80_000.0),
    287: DiversionSpec(287, 12.0, 160_000.0),
    349: DiversionSpec(349, 5.0, 50_000.0),
    383: DiversionSpec(383, 5.0, 40_000.0),
}


def head_inflow(t_s: float, cfg: Config) -> float:
    t_h = t_s / 3600.0
    if t_h <= 0.0:
        return 0.0
    if t_h < cfg.ramp_hours:
        r = 0.5 * (1.0 - math.cos(math.pi * t_h / cfg.ramp_hours))
        return cfg.target_head_flow_m3s * r
    return cfg.target_head_flow_m3s


def area_from_depth(h: float, sec: stage1.SectionParam) -> float:
    y = max(h, 0.0)
    return sec.bottom_width * y + sec.side_slope * y * y


def area_from_depth_array(h, bottom_width, side_slope):
    y = np.maximum(h, 0.0)
    return bottom_width * y + side_slope * y * y


def depth_from_area(area: float, sec: stage1.SectionParam) -> float:
    a = max(area, 1.0e-9)
    b = sec.bottom_width
    z = sec.side_slope
    if abs(z) < 1.0e-12:
        return a / max(b, 1.0e-9)
    return (-b + math.sqrt(b * b + 4.0 * z * a)) / (2.0 * z)


def depth_from_area_array(area, bottom_width, side_slope):
    a = np.maximum(area, 1.0e-9)
    b = bottom_width
    z = side_slope
    trapezoid = (-b + np.sqrt(b * b + 4.0 * z * a)) / np.maximum(2.0 * z, 1.0e-12)
    rectangle = a / np.maximum(b, 1.0e-9)
    return np.where(np.abs(z) < 1.0e-12, rectangle, trapezoid)


def top_width(h: float, sec: stage1.SectionParam) -> float:
    return sec.bottom_width + 2.0 * sec.side_slope * max(h, 0.0)


def top_width_array(h, bottom_width, side_slope):
    return bottom_width + 2.0 * side_slope * np.maximum(h, 0.0)


def wetted_perimeter(h: float, sec: stage1.SectionParam) -> float:
    return sec.bottom_width + 2.0 * max(h, 0.0) * math.sqrt(1.0 + sec.side_slope * sec.side_slope)


def wetted_perimeter_array(h, bottom_width, side_slope):
    return bottom_width + 2.0 * np.maximum(h, 0.0) * np.sqrt(1.0 + side_slope * side_slope)


def pressure_integral(h: float, sec: stage1.SectionParam) -> float:
    y = max(h, 0.0)
    return 0.5 * sec.bottom_width * y * y + (sec.side_slope * y ** 3) / 3.0


def pressure_integral_array(h, bottom_width, side_slope):
    y = np.maximum(h, 0.0)
    return 0.5 * bottom_width * y * y + (side_slope * y ** 3) / 3.0


def normal_depth(q: float, sec: stage1.SectionParam, slope: float, cfg: Config) -> float:
    if q <= 1.0e-9:
        return cfg.min_depth_m
    return stage1.normal_depth(q, sec, max(slope, cfg.min_bed_slope))


def friction_coefficient(area: float, q_reference: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    perimeter = wetted_perimeter(h, sec)
    radius = area / max(perimeter, 1.0e-9)
    return sec.manning_n ** 2 * abs(q_reference) / max(area * area * radius ** (4.0 / 3.0), 1.0e-9)


def friction_coefficient_array(area, q_reference, grid: ChannelGrid):
    h = depth_from_area_array(area, grid.bottom_width, grid.side_slope)
    perimeter = wetted_perimeter_array(h, grid.bottom_width, grid.side_slope)
    radius = area / np.maximum(perimeter, 1.0e-9)
    return grid.manning_n ** 2 * np.abs(q_reference) / np.maximum(area * area * radius ** (4.0 / 3.0), 1.0e-9)


def wave_speed(area: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    hydraulic_depth = area / max(top_width(h, sec), 1.0e-9)
    return math.sqrt(G * max(hydraulic_depth, 1.0e-9))


def wave_speed_array(area, bottom_width, side_slope):
    h = depth_from_area_array(area, bottom_width, side_slope)
    hydraulic_depth = area / np.maximum(top_width_array(h, bottom_width, side_slope), 1.0e-9)
    return np.sqrt(G * np.maximum(hydraulic_depth, 1.0e-9))


def flux(area: float, q: float, sec: stage1.SectionParam) -> Tuple[float, float]:
    h = depth_from_area(area, sec)
    return q, q * q / max(area, 1.0e-9) + G * pressure_integral(h, sec)


def flux_array(area, q, bottom_width, side_slope):
    h = depth_from_area_array(area, bottom_width, side_slope)
    return q, q * q / np.maximum(area, 1.0e-9) + G * pressure_integral_array(h, bottom_width, side_slope)


def hll_flux(a_l: float, q_l: float, sec_l: stage1.SectionParam, a_r: float, q_r: float, sec_r: stage1.SectionParam) -> Tuple[float, float]:
    u_l = q_l / max(a_l, 1.0e-9)
    u_r = q_r / max(a_r, 1.0e-9)
    c_l = wave_speed(a_l, sec_l)
    c_r = wave_speed(a_r, sec_r)
    s_l = min(u_l - c_l, u_r - c_r)
    s_r = max(u_l + c_l, u_r + c_r)
    f_l = flux(a_l, q_l, sec_l)
    f_r = flux(a_r, q_r, sec_r)
    if s_l >= 0.0:
        return f_l
    if s_r <= 0.0:
        return f_r
    return (
        (s_r * f_l[0] - s_l * f_r[0] + s_l * s_r * (a_r - a_l)) / (s_r - s_l),
        (s_r * f_l[1] - s_l * f_r[1] + s_l * s_r * (q_r - q_l)) / (s_r - s_l),
    )


def hll_flux_array(a_l, q_l, bw_l, z_l, a_r, q_r, bw_r, z_r):
    u_l = q_l / np.maximum(a_l, 1.0e-9)
    u_r = q_r / np.maximum(a_r, 1.0e-9)
    c_l = wave_speed_array(a_l, bw_l, z_l)
    c_r = wave_speed_array(a_r, bw_r, z_r)
    s_l = np.minimum(u_l - c_l, u_r - c_r)
    s_r = np.maximum(u_l + c_l, u_r + c_r)
    fa_l, fq_l = flux_array(a_l, q_l, bw_l, z_l)
    fa_r, fq_r = flux_array(a_r, q_r, bw_r, z_r)
    denom = np.maximum(s_r - s_l, 1.0e-9)
    fa_hll = (s_r * fa_l - s_l * fa_r + s_l * s_r * (a_r - a_l)) / denom
    fq_hll = (s_r * fq_l - s_l * fq_r + s_l * s_r * (q_r - q_l)) / denom
    fa = np.where(s_l >= 0.0, fa_l, np.where(s_r <= 0.0, fa_r, fa_hll))
    fq = np.where(s_l >= 0.0, fq_l, np.where(s_r <= 0.0, fq_r, fq_hll))
    return fa, fq


def manning_q_from_arrays(h: float, bottom_width: float, side_slope: float, manning_n: float, slope: float) -> float:
    area = float(area_from_depth_array(h, bottom_width, side_slope))
    perimeter = float(wetted_perimeter_array(h, bottom_width, side_slope))
    radius = area / max(perimeter, 1.0e-9)
    return (1.0 / manning_n) * area * (radius ** (2.0 / 3.0)) * math.sqrt(max(slope, 1.0e-7))


def diversion_factor(h: float, channel_depth: float) -> float:
    h_start = 0.20 * channel_depth
    h_full = 0.60 * channel_depth
    if h <= h_start:
        return 0.0
    if h >= h_full:
        return 1.0
    return math.sqrt((h - h_start) / max(h_full - h_start, 1.0e-9))


def build_grid():
    nodes = stage1.parse_nodes(DATA_DIR / "input.txt")
    params = stage1.parse_line_params(DATA_DIR / "lineParam.txt")
    neighbors = stage1.parse_neighbors(DATA_DIR / "neighborId.txt")
    ids = [n for n in range(MAIN_START, MAIN_END + 1) if n in nodes]
    n = len(ids)
    x = [0.0]
    for i in range(1, n):
        x.append(x[-1] + stage1.distance(nodes[ids[i - 1]], nodes[ids[i]]))
    dx_cell = []
    for i in range(n):
        if i == 0:
            dx_cell.append(x[1] - x[0])
        elif i == n - 1:
            dx_cell.append(x[-1] - x[-2])
        else:
            dx_cell.append(0.5 * (x[i + 1] - x[i - 1]))
    sections = [stage1.section_for_node(node_id, nodes, params) for node_id in ids]
    bed = [nodes[node_id].elev for node_id in ids]
    diversion_indices = {node: ids.index(node) for node in DIVERSION_NODES}
    return nodes, params, neighbors, ids, x, dx_cell, sections, bed, diversion_indices


def _cell_lengths_from_x(x: List[float]) -> np.ndarray:
    n = len(x)
    dx_cell: List[float] = []
    for i in range(n):
        if n == 1:
            dx_cell.append(1.0)
        elif i == 0:
            dx_cell.append(x[1] - x[0])
        elif i == n - 1:
            dx_cell.append(x[-1] - x[-2])
        else:
            dx_cell.append(0.5 * (x[i + 1] - x[i - 1]))
    return np.maximum(np.asarray(dx_cell, dtype=float), 1.0)


def _local_slopes_from_bed(x: np.ndarray, bed: np.ndarray, cfg: Config) -> np.ndarray:
    n = len(bed)
    slopes = np.empty(n, dtype=float)
    if n == 1:
        slopes[0] = cfg.min_bed_slope
        return slopes
    slopes[0] = (bed[0] - bed[1]) / max(x[1] - x[0], 1.0)
    slopes[-1] = (bed[-2] - bed[-1]) / max(x[-1] - x[-2], 1.0)
    if n > 2:
        slopes[1:-1] = (bed[:-2] - bed[2:]) / np.maximum(x[2:] - x[:-2], 1.0)
    return np.maximum(slopes, cfg.min_bed_slope)


def _build_channel_grid(
    name: str,
    ids: List[int],
    nodes: Dict[int, stage1.Node],
    params: Dict[str, stage1.SectionParam],
    cfg: Config,
) -> ChannelGrid:
    x = [0.0]
    for u, v in zip(ids[:-1], ids[1:]):
        x.append(x[-1] + stage1.distance(nodes[u], nodes[v]))
    x_arr = np.asarray(x, dtype=float)
    sections = [stage1.section_for_node(node_id, nodes, params) for node_id in ids]
    bed = np.asarray([nodes[node_id].elev for node_id in ids], dtype=float)
    return ChannelGrid(
        name=name,
        ids=ids,
        x=x_arr,
        dx_cell=_cell_lengths_from_x(x),
        bed=bed,
        depth=np.asarray([sec.depth for sec in sections], dtype=float),
        bottom_width=np.asarray([sec.bottom_width for sec in sections], dtype=float),
        side_slope=np.asarray([sec.side_slope for sec in sections], dtype=float),
        manning_n=np.asarray([sec.manning_n for sec in sections], dtype=float),
        slope=_local_slopes_from_bed(x_arr, bed, cfg),
    )


def _resample_channel_grid(grid: ChannelGrid, step_m: float, synthetic_start: int, cfg: Config) -> ChannelGrid:
    """Interpolate a channel grid to a target maximum spacing while preserving original node ids."""

    if step_m <= 0.0 or len(grid.ids) <= 1 or grid.x[-1] <= step_m:
        return grid

    regular_x = np.arange(0.0, float(grid.x[-1]), step_m, dtype=float)
    tagged_points = [(float(x), None) for x in regular_x]
    tagged_points.append((float(grid.x[-1]), grid.ids[-1]))
    tagged_points.extend((float(x), node_id) for x, node_id in zip(grid.x, grid.ids))
    tagged_points.sort(key=lambda item: (round(item[0], 6), item[1] is None))

    min_gap = min(0.25 * step_m, 0.25)
    selected: List[tuple[float, int | None]] = []
    for x_value, node_id in tagged_points:
        if not selected or x_value - selected[-1][0] >= min_gap:
            selected.append((x_value, node_id))
            continue
        if node_id is not None and selected[-1][1] is None:
            selected[-1] = (x_value, node_id)

    new_x = np.asarray([round(x, 6) for x, _ in selected], dtype=float)
    new_ids: List[int] = []
    synthetic_count = 0
    for _, node_id in selected:
        if node_id is None:
            node_id = synthetic_start - synthetic_count
            synthetic_count += 1
        new_ids.append(node_id)

    bed = np.interp(new_x, grid.x, grid.bed)
    return ChannelGrid(
        name=grid.name,
        ids=new_ids,
        x=new_x,
        dx_cell=_cell_lengths_from_x(new_x.tolist()),
        bed=bed,
        depth=np.interp(new_x, grid.x, grid.depth),
        bottom_width=np.interp(new_x, grid.x, grid.bottom_width),
        side_slope=np.interp(new_x, grid.x, grid.side_slope),
        manning_n=np.interp(new_x, grid.x, grid.manning_n),
        slope=_local_slopes_from_bed(new_x, bed, cfg),
    )


def _first_branch_child(node: int, nodes: Dict[int, stage1.Node], neighbors: Dict[int, List[int]]) -> int | None:
    main_nodes = set(range(MAIN_START, MAIN_END + 1))
    branch_children = [child for child in neighbors.get(node, []) if child not in main_nodes and child in nodes]
    return branch_children[0] if branch_children else None


def _trace_branch_chain(root: int, nodes: Dict[int, stage1.Node], neighbors: Dict[int, List[int]]) -> List[int]:
    main_nodes = set(range(MAIN_START, MAIN_END + 1))
    ids = [root]
    seen = {root}
    current = root
    while True:
        children = [child for child in neighbors.get(current, []) if child not in main_nodes and child in nodes]
        children = [child for child in children if child not in seen]
        if not children:
            break
        current = children[0]
        ids.append(current)
        seen.add(current)
    return ids


def build_network(cfg: Config):
    """Build the main channel plus dynamically routed diversion branch chains."""

    nodes = stage1.parse_nodes(DATA_DIR / "input.txt")
    params = stage1.parse_line_params(DATA_DIR / "lineParam.txt")
    neighbors = stage1.parse_neighbors(DATA_DIR / "neighborId.txt")
    main_ids = [node_id for node_id in range(MAIN_START, MAIN_END + 1) if node_id in nodes]
    main_grid = _build_channel_grid("main", main_ids, nodes, params, cfg)
    main_grid = _resample_channel_grid(main_grid, cfg.space_step_m, -1_000_000, cfg)
    branch_grids: Dict[int, ChannelGrid] = {}
    for node in DIVERSION_NODES:
        root = _first_branch_child(node, nodes, neighbors)
        if root is None:
            continue
        branch_ids = _trace_branch_chain(root, nodes, neighbors)
        branch_grid = _build_channel_grid(f"branch_{node}", branch_ids, nodes, params, cfg)
        branch_grids[node] = _resample_channel_grid(branch_grid, cfg.space_step_m, -2_000_000 - node * 100_000, cfg)
    limits = branch_limits(nodes, params, neighbors, cfg)
    return nodes, params, neighbors, main_grid, branch_grids, limits


def _concatenate_branch_network(branch_grids: Dict[int, ChannelGrid]) -> BranchNetwork | None:
    ordered_nodes = [node for node in DIVERSION_NODES if node in branch_grids]
    if not ordered_nodes:
        return None
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    ids: List[int] = []
    x_parts = []
    dx_parts = []
    bed_parts = []
    depth_parts = []
    bottom_width_parts = []
    side_slope_parts = []
    manning_parts = []
    slope_parts = []
    internal_left: List[int] = []
    offset = 0
    for node in ordered_nodes:
        grid = branch_grids[node]
        count = len(grid.ids)
        starts[node] = offset
        ends[node] = offset + count - 1
        ids.extend(grid.ids)
        x_parts.append(grid.x)
        dx_parts.append(grid.dx_cell)
        bed_parts.append(grid.bed)
        depth_parts.append(grid.depth)
        bottom_width_parts.append(grid.bottom_width)
        side_slope_parts.append(grid.side_slope)
        manning_parts.append(grid.manning_n)
        slope_parts.append(grid.slope)
        internal_left.extend(range(offset, offset + count - 1))
        offset += count
    left = np.asarray(internal_left, dtype=int)
    grid = ChannelGrid(
        name="diversion_branches",
        ids=ids,
        x=np.concatenate(x_parts),
        dx_cell=np.concatenate(dx_parts),
        bed=np.concatenate(bed_parts),
        depth=np.concatenate(depth_parts),
        bottom_width=np.concatenate(bottom_width_parts),
        side_slope=np.concatenate(side_slope_parts),
        manning_n=np.concatenate(manning_parts),
        slope=np.concatenate(slope_parts),
    )
    return BranchNetwork(
        grid=grid,
        starts=starts,
        ends=ends,
        internal_left=left,
        internal_right=left + 1,
    )


def local_slope(i: int, x: List[float], bed: List[float], cfg: Config) -> float:
    if i == 0:
        raw = (bed[i] - bed[i + 1]) / max(x[i + 1] - x[i], 1.0)
    elif i == len(bed) - 1:
        raw = (bed[i - 1] - bed[i]) / max(x[i] - x[i - 1], 1.0)
    else:
        raw = (bed[i - 1] - bed[i + 1]) / max(x[i + 1] - x[i - 1], 1.0)
    return max(raw, cfg.min_bed_slope)


def branch_limits(nodes, params, neighbors, cfg: Config) -> Dict[int, BranchLimit]:
    main_nodes = set(range(MAIN_START, MAIN_END + 1))
    limits: Dict[int, BranchLimit] = {}
    for node in DIVERSION_NODES:
        branch_children = [c for c in neighbors.get(node, []) if c not in main_nodes and c in nodes]
        if not branch_children:
            limits[node] = BranchLimit(None, float("inf"), None, None, None, None)
            continue
        child = branch_children[0]
        sec = stage1.section_for_node(child, nodes, params)
        dist = stage1.distance(nodes[node], nodes[child])
        raw_slope = (nodes[node].elev - nodes[child].elev) / max(dist, 1.0)
        slope = max(raw_slope, cfg.min_bed_slope)
        h_safe = cfg.safe_depth_ratio * sec.depth
        q_safe = stage1.manning_q(h_safe, sec, slope)
        limits[node] = BranchLimit(child, q_safe, sec.depth, sec.bottom_width, slope, dist)
    return limits


def _section_from_grid(grid: ChannelGrid, idx: int) -> stage1.SectionParam:
    return stage1.SectionParam(
        depth=float(grid.depth[idx]),
        bottom_width=float(grid.bottom_width[idx]),
        side_slope=float(grid.side_slope[idx]),
        manning_n=float(grid.manning_n[idx]),
    )


def _normal_depth_for_grid(q_value: float, grid: ChannelGrid, idx: int, cfg: Config) -> float:
    key = (grid.name, idx, round(q_value, 6), round(float(grid.slope[idx]), 9), cfg.min_depth_m)
    cached = _NORMAL_DEPTH_CACHE.get(key)
    if cached is not None:
        return cached
    value = normal_depth(q_value, _section_from_grid(grid, idx), float(grid.slope[idx]), cfg)
    _NORMAL_DEPTH_CACHE[key] = value
    return value


def _transport_update(
    grid: ChannelGrid,
    area: np.ndarray,
    discharge: np.ndarray,
    cfg: Config,
    t_s: float,
    head_boundary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(grid.ids)
    min_area = area_from_depth_array(cfg.min_depth_m, grid.bottom_width, grid.side_slope)
    work_area = np.maximum(area, min_area)
    flux_a = np.zeros(n + 1, dtype=float)
    flux_q = np.zeros(n + 1, dtype=float)

    if head_boundary:
        q_left = head_inflow(t_s, cfg)
        h_left = max(_normal_depth_for_grid(q_left, grid, 0, cfg), cfg.min_depth_m)
        a_left = float(area_from_depth_array(h_left, grid.bottom_width[0], grid.side_slope[0]))
        fa, fq = hll_flux_array(
            np.asarray([a_left]),
            np.asarray([q_left]),
            np.asarray([grid.bottom_width[0]]),
            np.asarray([grid.side_slope[0]]),
            work_area[:1],
            discharge[:1],
            grid.bottom_width[:1],
            grid.side_slope[:1],
        )
        flux_a[0] = fa[0]
        flux_q[0] = fq[0]
    else:
        h_wall = depth_from_area_array(work_area[0], grid.bottom_width[0], grid.side_slope[0])
        flux_a[0] = 0.0
        flux_q[0] = G * pressure_integral_array(h_wall, grid.bottom_width[0], grid.side_slope[0])

    if n > 1:
        fa, fq = hll_flux_array(
            work_area[:-1],
            discharge[:-1],
            grid.bottom_width[:-1],
            grid.side_slope[:-1],
            work_area[1:],
            discharge[1:],
            grid.bottom_width[1:],
            grid.side_slope[1:],
        )
        flux_a[1:n] = fa
        flux_q[1:n] = fq

    a_right = float(work_area[-1])
    fa, fq = hll_flux_array(
        work_area[-1:],
        discharge[-1:],
        grid.bottom_width[-1:],
        grid.side_slope[-1:],
        np.asarray([a_right]),
        np.asarray([float(discharge[-1])]),
        np.asarray([grid.bottom_width[-1]]),
        np.asarray([grid.side_slope[-1]]),
    )
    flux_a[-1] = fa[0]
    flux_q[-1] = fq[0]

    da = -(flux_a[1:] - flux_a[:-1]) / grid.dx_cell
    dq = -(flux_q[1:] - flux_q[:-1]) / grid.dx_cell
    dq += G * work_area * grid.slope
    new_area = area + cfg.dt_seconds * da
    new_discharge = discharge + cfg.dt_seconds * dq
    return new_area, new_discharge


def _transport_branch_network_update(
    network: BranchNetwork,
    area: np.ndarray,
    discharge: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    grid = network.grid
    n = len(grid.ids)
    min_area = area_from_depth_array(cfg.min_depth_m, grid.bottom_width, grid.side_slope)
    work_area = np.maximum(area, min_area)
    left_flux_a = np.zeros(n, dtype=float)
    left_flux_q = np.zeros(n, dtype=float)
    right_flux_a = np.zeros(n, dtype=float)
    right_flux_q = np.zeros(n, dtype=float)

    if len(network.internal_left):
        li = network.internal_left
        ri = network.internal_right
        fa, fq = hll_flux_array(
            work_area[li],
            discharge[li],
            grid.bottom_width[li],
            grid.side_slope[li],
            work_area[ri],
            discharge[ri],
            grid.bottom_width[ri],
            grid.side_slope[ri],
        )
        right_flux_a[li] = fa
        right_flux_q[li] = fq
        left_flux_a[ri] = fa
        left_flux_q[ri] = fq

    start_idx = np.asarray(list(network.starts.values()), dtype=int)
    h_wall = depth_from_area_array(work_area[start_idx], grid.bottom_width[start_idx], grid.side_slope[start_idx])
    left_flux_a[start_idx] = 0.0
    left_flux_q[start_idx] = G * pressure_integral_array(h_wall, grid.bottom_width[start_idx], grid.side_slope[start_idx])

    for end in network.ends.values():
        a_right = float(work_area[end])
        fa, fq = hll_flux_array(
            work_area[end : end + 1],
            discharge[end : end + 1],
            grid.bottom_width[end : end + 1],
            grid.side_slope[end : end + 1],
            np.asarray([a_right]),
            np.asarray([float(discharge[end])]),
            np.asarray([grid.bottom_width[end]]),
            np.asarray([grid.side_slope[end]]),
        )
        right_flux_a[end] = fa[0]
        right_flux_q[end] = fq[0]

    da = -(right_flux_a - left_flux_a) / grid.dx_cell
    dq = -(right_flux_q - left_flux_q) / grid.dx_cell
    dq += G * work_area * grid.slope
    new_area = area + cfg.dt_seconds * da
    new_discharge = discharge + cfg.dt_seconds * dq
    return new_area, new_discharge


def _apply_friction_and_limits(grid: ChannelGrid, area: np.ndarray, discharge: np.ndarray, old_discharge: np.ndarray, cfg: Config):
    min_area = area_from_depth_array(cfg.min_depth_m, grid.bottom_width, grid.side_slope)
    area = np.maximum(area, min_area)
    cf = friction_coefficient_array(area, old_discharge, grid)
    denom = 1.0 + cfg.dt_seconds * G * area * cf
    discharge = discharge / np.maximum(denom, 1.0e-9)
    discharge = np.where(np.abs(discharge) < 1.0e-7, 0.0, discharge)
    c = wave_speed_array(area, grid.bottom_width, grid.side_slope)
    q_limit = 8.0 * area * np.maximum(c, 0.1)
    discharge = np.clip(discharge, -q_limit, q_limit)
    return area, discharge


def _junction_flow_from_energy(
    main_grid: ChannelGrid,
    branch_grid: ChannelGrid,
    main_idx: int,
    main_area: np.ndarray,
    branch_area: np.ndarray,
    supplied_m3: float,
    close_h: float | None,
    limit: BranchLimit,
    spec: DiversionSpec,
    cfg: Config,
) -> float:
    if close_h is not None or limit.branch_node is None:
        return 0.0

    h_main = float(depth_from_area_array(main_area[main_idx], main_grid.bottom_width[main_idx], main_grid.side_slope[main_idx]))
    h_branch = float(depth_from_area_array(branch_area[0], branch_grid.bottom_width[0], branch_grid.side_slope[0]))
    head_main = float(main_grid.bed[main_idx] + h_main)
    head_branch = float(branch_grid.bed[0] + h_branch)
    head_drop = head_main - head_branch
    if head_drop <= 0.0:
        return 0.0

    entry_length = max(limit.entry_length_m or float(branch_grid.dx_cell[0]), 1.0)
    energy_slope = head_drop / entry_length
    h_inlet = max(head_main - float(branch_grid.bed[0]), cfg.min_depth_m)
    q_energy = manning_q_from_arrays(
        h_inlet,
        float(branch_grid.bottom_width[0]),
        float(branch_grid.side_slope[0]),
        float(branch_grid.manning_n[0]),
        energy_slope,
    )
    factor = diversion_factor(h_main, float(main_grid.depth[main_idx]))
    remaining = max(spec.demand_m3 - supplied_m3, 0.0)
    main_min_area = float(area_from_depth_array(cfg.min_depth_m, main_grid.bottom_width[main_idx], main_grid.side_slope[main_idx]))
    main_storage = max(float(main_area[main_idx]) - main_min_area, 0.0) * float(main_grid.dx_cell[main_idx]) / cfg.dt_seconds
    branch_safe_h = cfg.safe_depth_ratio * float(branch_grid.depth[0])
    branch_safe_area = float(area_from_depth_array(branch_safe_h, branch_grid.bottom_width[0], branch_grid.side_slope[0]))
    branch_storage = max(branch_safe_area - float(branch_area[0]), 0.0) * float(branch_grid.dx_cell[0]) / cfg.dt_seconds
    q_capacity = min(spec.max_flow_m3s, limit.safe_capacity_m3s, q_energy) * factor
    return max(0.0, min(q_capacity, remaining / cfg.dt_seconds, main_storage, branch_storage))


def _simulate_explicit(cfg: Config):
    _, _, _, main_grid, branch_grids, limits = build_network(cfg)
    branch_network = _concatenate_branch_network(branch_grids)
    main_index = {node: idx for idx, node in enumerate(main_grid.ids)}
    main_area = area_from_depth_array(cfg.initial_depth_m, main_grid.bottom_width, main_grid.side_slope)
    main_q = np.zeros(len(main_grid.ids), dtype=float)
    if branch_network is not None:
        branch_area = area_from_depth_array(
            cfg.initial_depth_m,
            branch_network.grid.bottom_width,
            branch_network.grid.side_slope,
        )
        branch_q = np.zeros(len(branch_network.grid.ids), dtype=float)
    else:
        branch_area = np.asarray([], dtype=float)
        branch_q = np.asarray([], dtype=float)

    supplied = {node: 0.0 for node in DIVERSION_NODES}
    first_positive = {node: None for node in DIVERSION_NODES}
    close_time = {node: None for node in DIVERSION_NODES}
    qdiv_current = {node: 0.0 for node in DIVERSION_NODES}

    output_every = max(1, int(round(cfg.output_interval_seconds / cfg.dt_seconds)))
    times_h: List[float] = []
    qdiv_series: Dict[int, List[float]] = {node: [] for node in DIVERSION_NODES}
    supplied_series: Dict[int, List[float]] = {node: [] for node in DIVERSION_NODES}
    head_q_series: List[float] = []
    key_nodes = [0, 71, 89, 150, 194, 287, 349, 383, 456]
    key_indices = {node: main_index[node] for node in key_nodes if node in main_index}
    depth_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    water_level_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    depth_limit_series: Dict[int, float] = {node: float(main_grid.depth[idx]) for node, idx in key_indices.items()}
    safe_depth_series: Dict[int, float] = {node: cfg.safe_depth_ratio * float(main_grid.depth[idx]) for node, idx in key_indices.items()}
    main_hmax = depth_from_area_array(main_area, main_grid.bottom_width, main_grid.side_slope)
    main_hmax_time = np.zeros(len(main_grid.ids), dtype=float)

    total_steps = int(round(cfg.duration_hours * 3600.0 / cfg.dt_seconds))
    for step in range(total_steps + 1):
        t_s = step * cfg.dt_seconds
        clock_h = cfg.start_hours + t_s / 3600.0
        main_depth = depth_from_area_array(main_area, main_grid.bottom_width, main_grid.side_slope)
        improved = main_depth > main_hmax
        main_hmax = np.where(improved, main_depth, main_hmax)
        main_hmax_time = np.where(improved, clock_h, main_hmax_time)

        if step % output_every == 0:
            times_h.append(clock_h)
            head_q_series.append(head_inflow(t_s, cfg))
            for node in DIVERSION_NODES:
                qdiv_series[node].append(qdiv_current[node])
                supplied_series[node].append(supplied[node])
            for node, idx in key_indices.items():
                h_now = float(main_depth[idx])
                depth_series[node].append(h_now)
                water_level_series[node].append(float(main_grid.bed[idx] + h_now))

        if step == total_steps:
            break

        old_main_q = main_q.copy()
        old_branch_q = branch_q.copy()
        new_main_area, new_main_q = _transport_update(main_grid, main_area, main_q, cfg, t_s, head_boundary=True)
        if branch_network is not None:
            new_branch_area, new_branch_q = _transport_branch_network_update(branch_network, branch_area, branch_q, cfg)
        else:
            new_branch_area = branch_area.copy()
            new_branch_q = branch_q.copy()

        qdiv_current = {node: 0.0 for node in DIVERSION_NODES}
        for node in DIVERSION_NODES:
            if branch_network is None or node not in branch_grids or node not in main_index or node not in branch_network.starts:
                continue
            main_idx = main_index[node]
            grid = branch_grids[node]
            branch_start = branch_network.starts[node]
            branch_slice = slice(branch_start, branch_network.ends[node] + 1)
            spec = SPECS[node]
            qdiv = _junction_flow_from_energy(
                main_grid,
                grid,
                main_idx,
                new_main_area,
                new_branch_area[branch_slice],
                supplied[node],
                close_time[node],
                limits[node],
                spec,
                cfg,
            )
            if qdiv <= 0.0:
                continue
            if first_positive[node] is None:
                first_positive[node] = clock_h
            supplied[node] += qdiv * cfg.dt_seconds
            if supplied[node] >= spec.demand_m3 - 1.0e-6 and close_time[node] is None:
                close_time[node] = cfg.start_hours + (t_s + cfg.dt_seconds) / 3600.0
            qdiv_current[node] = qdiv

            main_dx = float(main_grid.dx_cell[main_idx])
            branch_dx = float(grid.dx_cell[0])
            main_min_area = float(area_from_depth_array(cfg.min_depth_m, main_grid.bottom_width[main_idx], main_grid.side_slope[main_idx]))
            main_velocity = float(new_main_q[main_idx] / max(new_main_area[main_idx], main_min_area))
            new_main_area[main_idx] -= qdiv * cfg.dt_seconds / main_dx
            new_main_q[main_idx] -= qdiv * main_velocity * cfg.dt_seconds / main_dx

            h_inlet = max(
                float(main_grid.bed[main_idx] + depth_from_area_array(new_main_area[main_idx], main_grid.bottom_width[main_idx], main_grid.side_slope[main_idx]) - grid.bed[0]),
                cfg.min_depth_m,
            )
            inlet_area = float(area_from_depth_array(h_inlet, grid.bottom_width[0], grid.side_slope[0]))
            inlet_velocity = qdiv / max(inlet_area, 1.0e-9)
            new_branch_area[branch_start] += qdiv * cfg.dt_seconds / branch_dx
            new_branch_q[branch_start] += qdiv * inlet_velocity * cfg.dt_seconds / branch_dx

        main_area, main_q = _apply_friction_and_limits(main_grid, new_main_area, new_main_q, old_main_q, cfg)
        if branch_network is not None:
            branch_area, branch_q = _apply_friction_and_limits(
                branch_network.grid,
                new_branch_area,
                new_branch_q,
                old_branch_q,
                cfg,
            )

    return {
        "times_h": times_h,
        "qdiv": qdiv_series,
        "supplied": supplied_series,
        "head_q": head_q_series,
        "depth": depth_series,
        "water_level": water_level_series,
        "depth_limit": depth_limit_series,
        "safe_depth": safe_depth_series,
        "first_positive_h": first_positive,
        "close_h": close_time,
        "branch_limits": limits,
        "config": cfg,
        "main_ids": main_grid.ids,
        "main_dist_km": (main_grid.x / 1000.0).tolist(),
        "main_hmax": main_hmax.tolist(),
        "main_hmax_time": main_hmax_time.tolist(),
        "branch_ids": {node: grid.ids for node, grid in branch_grids.items()},
        "network_coupling": "main and selected diversion branch chains; junction flow from water-level/energy slope with conservative mass transfer",
        "solver": "explicit-hll",
    }


def simulate(cfg: Config):
    """Run the configured Saint-Venant dispatch solver."""

    solver = cfg.solver.strip().lower()
    if solver in {"explicit", "explicit-hll", "hll"}:
        return _simulate_explicit(cfg)
    if solver in {"implicit", "implicit-network", "network-implicit", "preissmann", "preissmann-network"}:
        import implicit_dispatch

        return implicit_dispatch.simulate(cfg)
    raise ValueError(f"Unknown dispatch solver: {cfg.solver!r}")
