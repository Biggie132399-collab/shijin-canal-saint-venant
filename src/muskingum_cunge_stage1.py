#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 model for nodes 0-456:
Muskingum-Cunge / diffusive-wave flow routing with constant discharge
boundary conditions and lateral diversion terms.

This program intentionally uses discharge Q as the routing variable. For this
steady-boundary stage, the head inflow and all diversion flows are constant.
The plotted "routing process" is therefore the numerical convergence from an
initial condition to the final steady state, not a measured unsteady SCADA wave.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "results" / "mc_first_stage_results"

MAIN_START = 0
MAIN_END = 456
KEY_NODES = [0, 71, 89, 150, 194, 287, 349, 383, 456]
HEADFLOW_CONSTANT = 108.0
DIVERSION_CONSTANT = 8.0
DIVERSION_DEFAULTS = {node: DIVERSION_CONSTANT for node in [71, 89, 150, 194, 287, 349, 383]}
NODE_LABELS = {
    0: "0渠首",
    71: "71分水口",
    89: "89分水口",
    150: "150分水口",
    194: "194分水口",
    287: "287分水口",
    349: "349分水口",
    383: "383分水口",
    456: "456渠尾",
}


@dataclass
class Node:
    node_id: int
    x: float
    y: float
    elev: float
    style: str


@dataclass
class SectionParam:
    depth: float
    bottom_width: float
    side_slope: float
    manning_n: float
    water_level: float = 0.0
    infiltration: float = 0.0


@dataclass
class Reach:
    up: int
    down: int
    length_m: float
    bed_slope: float
    section: SectionParam
    celerity: float
    k_seconds: float
    x_weight: float
    c0: float
    c1: float
    c2: float
    c3: float
    courant: float


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def parse_nodes(path: Path) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        node_id = int(parts[0])
        nodes[node_id] = Node(
            node_id=node_id,
            x=float(parts[1]),
            y=float(parts[2]),
            elev=float(parts[3]),
            style=parts[4].strip(),
        )
    return nodes


def angle_to_side_slope(angle_degrees: float) -> float:
    if angle_degrees <= 0:
        return 0.0
    # lineParam angle is treated as the side-wall angle against horizontal.
    # z is horizontal:vertical side slope in the trapezoidal section.
    radians = math.radians(angle_degrees)
    tan_v = math.tan(radians)
    if abs(tan_v) < 1e-9:
        return 0.0
    return max(0.0, 1.0 / tan_v)


def parse_line_params(path: Path) -> Dict[str, SectionParam]:
    params: Dict[str, SectionParam] = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        style = parts[0].strip()
        depth = float(parts[1]) if parts[1] else 3.84
        width = float(parts[2]) if parts[2] else 25.0
        angle = float(parts[3]) if parts[3] else 64.0
        manning = float(parts[4]) if parts[4] else 0.025
        water_level = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
        infiltration = float(parts[6]) if len(parts) > 6 and parts[6] else 0.0
        params[style] = SectionParam(
            depth=depth,
            bottom_width=width,
            side_slope=angle_to_side_slope(angle),
            manning_n=manning,
            water_level=water_level,
            infiltration=infiltration,
        )
    return params


def section_for_node(
    node_id: int,
    nodes: Dict[int, Node],
    params: Dict[str, SectionParam],
) -> SectionParam:
    default = SectionParam(3.84, 25.0, angle_to_side_slope(64.0), 0.025)
    node = nodes.get(node_id)
    if node is None:
        return default
    return params.get(node.style, default)


def parse_neighbors(path: Path) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        node_id = int(parts[0])
        next_ids = []
        for raw in parts[3:]:
            raw = raw.strip()
            if not raw:
                continue
            value = int(raw)
            if value != -1:
                next_ids.append(value)
        neighbors[node_id] = next_ids
    return neighbors


def distance(a: Node, b: Node) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def cumulative_distances(nodes: Dict[int, Node], main_nodes: List[int]) -> Dict[int, float]:
    out = {main_nodes[0]: 0.0}
    acc = 0.0
    for u, v in zip(main_nodes[:-1], main_nodes[1:]):
        acc += distance(nodes[u], nodes[v])
        out[v] = acc
    return out


def representative_bed_slope(nodes: Dict[int, Node], dist_map: Dict[int, float]) -> float:
    """Representative slope used for steady-state Manning depth display.

    Local micro-topography can contain small slope changes. Using each local
    reach slope directly to back-calculate normal depth creates artificial
    depth spikes in a steady-boundary schematic. The representative slope is
    the overall bed drop divided by the 0-456 route length.
    """
    drop = nodes[MAIN_START].elev - nodes[MAIN_END].elev
    length = dist_map[MAIN_END] - dist_map[MAIN_START]
    return max(drop / max(length, 1.0), 5e-5)


def section_area(y: float, sec: SectionParam) -> float:
    return sec.bottom_width * y + sec.side_slope * y * y


def top_width(y: float, sec: SectionParam) -> float:
    return sec.bottom_width + 2.0 * sec.side_slope * y


def wetted_perimeter(y: float, sec: SectionParam) -> float:
    return sec.bottom_width + 2.0 * y * math.sqrt(1.0 + sec.side_slope ** 2)


def hydraulic_radius(y: float, sec: SectionParam) -> float:
    return section_area(y, sec) / max(wetted_perimeter(y, sec), 1e-9)


def flow_velocity(q: float, y: float, sec: SectionParam) -> float:
    return q / max(section_area(y, sec), 1e-9)


def specific_energy(q: float, y: float, sec: SectionParam) -> float:
    return y + flow_velocity(q, y, sec) ** 2 / (2.0 * 9.81)


def friction_slope(q: float, y: float, sec: SectionParam) -> float:
    area = section_area(y, sec)
    radius = hydraulic_radius(y, sec)
    return (q * sec.manning_n / max(area * (radius ** (2.0 / 3.0)), 1e-9)) ** 2


def manning_q(y: float, sec: SectionParam, slope: float) -> float:
    area = section_area(y, sec)
    radius = hydraulic_radius(y, sec)
    return (1.0 / sec.manning_n) * area * (radius ** (2.0 / 3.0)) * math.sqrt(max(slope, 1e-7))


def normal_depth(q: float, sec: SectionParam, slope: float) -> float:
    if q <= 0:
        return 0.01
    lo = 0.01
    hi = max(sec.depth * 2.0, 1.0)
    while manning_q(hi, sec, slope) < q and hi < 30.0:
        hi *= 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if manning_q(mid, sec, slope) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def celerity_from_manning(q_ref: float, sec: SectionParam, slope: float) -> Tuple[float, float, float]:
    y0 = normal_depth(q_ref, sec, slope)
    a0 = section_area(y0, sec)
    dy = max(0.001, y0 * 0.001)
    q_plus = manning_q(y0 + dy, sec, slope)
    q_minus = manning_q(max(0.001, y0 - dy), sec, slope)
    a_plus = section_area(y0 + dy, sec)
    a_minus = section_area(max(0.001, y0 - dy), sec)
    c = (q_plus - q_minus) / max(a_plus - a_minus, 1e-9)
    return max(c, 0.05), y0, a0


def build_reaches(
    nodes: Dict[int, Node],
    params: Dict[str, SectionParam],
    key_nodes: List[int],
    dist_map: Dict[int, float],
    q_ref: float,
    dt_seconds: float,
) -> List[Reach]:
    reaches: List[Reach] = []
    for up, down in zip(key_nodes[:-1], key_nodes[1:]):
        length_m = dist_map[down] - dist_map[up]
        drop = nodes[up].elev - nodes[down].elev
        bed_slope = max(drop / max(length_m, 1.0), 5e-5)
        sec = params.get(nodes[up].style, SectionParam(3.84, 25.0, angle_to_side_slope(64.0), 0.025))
        celerity, y0, _ = celerity_from_manning(max(q_ref, 1.0), sec, bed_slope)
        k_seconds = max(length_m / celerity, 1.0)
        b_top = top_width(y0, sec)
        # Cunge weighting. Bound X to keep the storage weighting physical.
        x_raw = 0.5 * (1.0 - max(q_ref, 1.0) / max(b_top * bed_slope * celerity * length_m, 1e-9))
        x_weight = min(0.49, max(0.0, x_raw))
        denom = k_seconds * (1.0 - x_weight) + 0.5 * dt_seconds
        c0 = (-k_seconds * x_weight + 0.5 * dt_seconds) / denom
        c1 = (k_seconds * x_weight + 0.5 * dt_seconds) / denom
        c2 = (k_seconds * (1.0 - x_weight) - 0.5 * dt_seconds) / denom
        c3 = 0.5 * dt_seconds / denom
        reaches.append(
            Reach(
                up=up,
                down=down,
                length_m=length_m,
                bed_slope=bed_slope,
                section=sec,
                celerity=celerity,
                k_seconds=k_seconds,
                x_weight=x_weight,
                c0=c0,
                c1=c1,
                c2=c2,
                c3=c3,
                courant=celerity * dt_seconds / max(length_m, 1.0),
            )
        )
    return reaches


def constant_headflow_series(
    start: str,
    duration_hours: float,
    dt_seconds: int,
    headflow: float,
) -> Tuple[List[datetime], List[float]]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = start_dt + timedelta(hours=duration_hours)
    times: List[datetime] = []
    flows: List[float] = []
    t = start_dt
    while t <= end_dt:
        times.append(t)
        flows.append(headflow)
        t += timedelta(seconds=dt_seconds)
    return times, flows


def route_flows(
    reaches: List[Reach],
    times: List[datetime],
    headflow: List[float],
    diversions: Dict[int, float],
    dt_seconds: float,
    loss_per_km_m3s: float = 0.0,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[Tuple[int, int], List[float]]]:
    q_by_node: Dict[int, List[float]] = {reaches[0].up: headflow}
    lateral_by_node: Dict[int, List[float]] = {}
    storage_by_reach: Dict[Tuple[int, int], List[float]] = {}

    for reach in reaches:
        inflow = q_by_node[reach.up]
        fixed_diversion = diversions.get(reach.down, 0.0)
        reach_loss = max(loss_per_km_m3s, 0.0) * reach.length_m / 1000.0
        lateral = [-(fixed_diversion + reach_loss) for _ in times]
        # Use zero initial outflow to show convergence under constant boundary
        # conditions. The final time step is the steady-state snapshot.
        outflow = [0.0]
        storage = [reach.k_seconds * (reach.x_weight * inflow[0] + (1.0 - reach.x_weight) * outflow[0])]
        for j in range(1, len(times)):
            q_next = (
                reach.c0 * inflow[j]
                + reach.c1 * inflow[j - 1]
                + reach.c2 * outflow[j - 1]
                + reach.c3 * (lateral[j] + lateral[j - 1])
            )
            outflow.append(max(q_next, 0.0))
            storage.append(reach.k_seconds * (reach.x_weight * inflow[j] + (1.0 - reach.x_weight) * outflow[j]))
        q_by_node[reach.down] = outflow
        lateral_by_node[reach.down] = [-x for x in lateral]
        storage_by_reach[(reach.up, reach.down)] = storage
    return q_by_node, lateral_by_node, storage_by_reach


def integrate(values: List[float], dt_seconds: float) -> List[float]:
    out = [0.0]
    for i in range(1, len(values)):
        out.append(out[-1] + 0.5 * (values[i - 1] + values[i]) * dt_seconds)
    return out


def write_time_series(
    out_path: Path,
    times: List[datetime],
    q_by_node: Dict[int, List[float]],
    lateral_by_node: Dict[int, List[float]],
    key_nodes: List[int],
):
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["time"] + [f"Q_node_{n}_m3s" for n in key_nodes] + [
            f"D_node_{n}_m3s" for n in key_nodes if n in lateral_by_node
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(times):
            row = {"time": t.isoformat(sep=" ")}
            for n in key_nodes:
                row[f"Q_node_{n}_m3s"] = f"{q_by_node[n][i]:.6f}"
            for n in key_nodes:
                if n in lateral_by_node:
                    row[f"D_node_{n}_m3s"] = f"{lateral_by_node[n][i]:.6f}"
            writer.writerow(row)


def steady_node_flows(headflow: float, diversions: Dict[int, float]) -> Dict[int, float]:
    flows: Dict[int, float] = {}
    current = headflow
    for node in KEY_NODES:
        if node == KEY_NODES[0]:
            flows[node] = current
            continue
        if node in diversions:
            current = max(current - diversions[node], 0.0)
        flows[node] = current
    return flows


def steady_node_flows_with_losses(
    headflow: float,
    diversions: Dict[int, float],
    reaches: List[Reach],
    loss_per_km_m3s: float = 0.0,
) -> Dict[int, float]:
    flows: Dict[int, float] = {KEY_NODES[0]: headflow}
    current = headflow
    reach_by_down = {reach.down: reach for reach in reaches}
    for node in KEY_NODES[1:]:
        reach = reach_by_down[node]
        current = max(current - max(loss_per_km_m3s, 0.0) * reach.length_m / 1000.0, 0.0)
        if node in diversions:
            current = max(current - diversions[node], 0.0)
        flows[node] = current
    return flows


def node_depth_slopes(reaches: List[Reach]) -> Dict[int, float]:
    """Slope used for key-node normal-depth back calculation.

    The first node uses the first downstream reach slope. Other key nodes use
    the immediately upstream reach slope ending at that node.
    """
    slopes: Dict[int, float] = {reaches[0].up: reaches[0].bed_slope}
    for reach in reaches:
        slopes[reach.down] = reach.bed_slope
    return slopes


def main_node_flows_from_key_flows(
    main_nodes: List[int],
    key_flows: Dict[int, float],
) -> Dict[int, float]:
    """Assign steady discharge to all discrete nodes from the previous key node.

    Diversions are located at key nodes. The discharge plotted at a diversion
    node is the post-diversion discharge, and the next downstream interval uses
    that reduced discharge.
    """
    out: Dict[int, float] = {}
    key_set = set(key_flows)
    current = key_flows[KEY_NODES[0]]
    for node in main_nodes:
        if node in key_set:
            current = key_flows[node]
        out[node] = current
    return out


def solve_upstream_depth_standard_step(
    up_node: Node,
    down_node: Node,
    up_sec: SectionParam,
    down_sec: SectionParam,
    q_up: float,
    q_down: float,
    y_down: float,
    length_m: float,
    fallback_slope: float,
) -> float:
    """Solve steady gradually-varied depth by the standard-step equation."""
    e_down = specific_energy(q_down, y_down, down_sec)
    sf_down = friction_slope(q_down, y_down, down_sec)

    def residual(y_up: float) -> float:
        sf_up = friction_slope(q_up, y_up, up_sec)
        hf = 0.5 * (sf_up + sf_down) * length_m
        return up_node.elev + specific_energy(q_up, y_up, up_sec) - down_node.elev - e_down - hf

    lo = 0.05
    hi = max(up_sec.depth * 3.0, y_down + 2.0, 6.0)
    f_lo = residual(lo)
    f_hi = residual(hi)
    while f_lo * f_hi > 0.0 and hi < 50.0:
        hi *= 1.5
        f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        return normal_depth(q_up, up_sec, fallback_slope)

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def steady_depths_standard_step(
    nodes: Dict[int, Node],
    params: Dict[str, SectionParam],
    main_nodes: List[int],
    dist_map: Dict[int, float],
    key_flows: Dict[int, float],
    downstream_slope: float,
) -> Dict[int, float]:
    """Backwater-style steady profile for Figure 2 and the steady CSV.

    This keeps the true section geometry and bed elevation, while avoiding the
    artificial spike caused by treating a single locally flat bed segment as a
    complete normal-depth control.
    """
    node_flows = main_node_flows_from_key_flows(main_nodes, key_flows)
    depths: Dict[int, float] = {
        main_nodes[-1]: normal_depth(
            max(node_flows[main_nodes[-1]], 0.01),
            section_for_node(main_nodes[-1], nodes, params),
            downstream_slope,
        )
    }
    for up, down in reversed(list(zip(main_nodes[:-1], main_nodes[1:]))):
        length_m = distance(nodes[up], nodes[down])
        depths[up] = solve_upstream_depth_standard_step(
            nodes[up],
            nodes[down],
            section_for_node(up, nodes, params),
            section_for_node(down, nodes, params),
            max(node_flows[up], 0.01),
            max(node_flows[down], 0.01),
            depths[down],
            length_m,
            downstream_slope,
        )
    return {node: depths[node] for node in KEY_NODES}


def write_steady_profile(
    out_path: Path,
    steady_flows: Dict[int, float],
    depths: Dict[int, float],
    dist_map: Dict[int, float],
    nodes: Dict[int, Node],
    params: Dict[str, SectionParam],
    depth_slopes: Dict[int, float],
):
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node",
                "distance_km",
                "style",
                "design_depth_m",
                "bottom_width_m",
                "side_slope_h_per_v",
                "manning_n",
                "depth_slope_used",
                "steady_Q_m3s",
                "normal_depth_m",
                "water_level_m",
            ]
        )
        for node in KEY_NODES:
            sec = section_for_node(node, nodes, params)
            water_level = nodes[node].elev + depths[node]
            writer.writerow(
                [
                    node,
                    f"{dist_map[node] / 1000.0:.6f}",
                    nodes[node].style,
                    f"{sec.depth:.6f}",
                    f"{sec.bottom_width:.6f}",
                    f"{sec.side_slope:.6f}",
                    f"{sec.manning_n:.6f}",
                    f"{depth_slopes[node]:.8f}",
                    f"{steady_flows[node]:.6f}",
                    f"{depths[node]:.6f}",
                    f"{water_level:.6f}",
                ]
            )


def write_reaches(out_path: Path, reaches: List[Reach]):
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "up_node",
                "down_node",
                "length_m",
                "bed_slope",
                "bottom_width_m",
                "side_slope_h_per_v",
                "manning_n",
                "celerity_m_s",
                "K_s",
                "X",
                "C0",
                "C1",
                "C2",
                "C3",
                "Courant",
            ]
        )
        for r in reaches:
            writer.writerow(
                [
                    r.up,
                    r.down,
                    f"{r.length_m:.3f}",
                    f"{r.bed_slope:.8f}",
                    f"{r.section.bottom_width:.3f}",
                    f"{r.section.side_slope:.4f}",
                    f"{r.section.manning_n:.5f}",
                    f"{r.celerity:.5f}",
                    f"{r.k_seconds:.2f}",
                    f"{r.x_weight:.5f}",
                    f"{r.c0:.6f}",
                    f"{r.c1:.6f}",
                    f"{r.c2:.6f}",
                    f"{r.c3:.6f}",
                    f"{r.courant:.6f}",
                ]
            )


def write_mass_balance(
    out_path: Path,
    times: List[datetime],
    q_by_node: Dict[int, List[float]],
    lateral_by_node: Dict[int, List[float]],
    storage_by_reach: Dict[Tuple[int, int], List[float]],
    dt_seconds: float,
):
    inflow_volume = integrate(q_by_node[KEY_NODES[0]], dt_seconds)
    outflow_volume = integrate(q_by_node[KEY_NODES[-1]], dt_seconds)
    diversion_series = [sum(lateral_by_node.get(n, [0.0] * len(times))[i] for n in lateral_by_node) for i in range(len(times))]
    diversion_volume = integrate(diversion_series, dt_seconds)
    storage_total = [sum(s[i] for s in storage_by_reach.values()) for i in range(len(times))]
    storage_change = [s - storage_total[0] for s in storage_total]
    residual = [
        inflow_volume[i] - outflow_volume[i] - diversion_volume[i] - storage_change[i]
        for i in range(len(times))
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "cum_inflow_m3",
                "cum_outflow_m3",
                "cum_diversion_m3",
                "storage_change_m3",
                "residual_m3",
                "relative_residual",
            ]
        )
        for i, t in enumerate(times):
            denom = max(inflow_volume[i], 1.0)
            writer.writerow(
                [
                    t.isoformat(sep=" "),
                    f"{inflow_volume[i]:.3f}",
                    f"{outflow_volume[i]:.3f}",
                    f"{diversion_volume[i]:.3f}",
                    f"{storage_change[i]:.3f}",
                    f"{residual[i]:.3f}",
                    f"{residual[i] / denom:.8f}",
                ]
            )
    return {
        "final_inflow_volume_m3": inflow_volume[-1],
        "final_outflow_volume_m3": outflow_volume[-1],
        "final_diversion_volume_m3": diversion_volume[-1],
        "final_storage_change_m3": storage_change[-1],
        "final_residual_m3": residual[-1],
        "final_relative_residual": residual[-1] / max(inflow_volume[-1], 1.0),
        "max_abs_relative_residual": max(abs(residual[i]) / max(inflow_volume[i], 1.0) for i in range(1, len(times))),
    }


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_line_chart(
    path: Path,
    title: str,
    times: List[datetime],
    series: List[Tuple[str, List[float], str]],
    y_label: str,
    caption: str,
    size: Tuple[int, int] = (1800, 1050),
):
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    title_font = get_font(40, True)
    text_font = get_font(24)
    small_font = get_font(20)
    left, top, right, bottom = 130, 135, w - 80, h - 170
    d.text((80, 45), title, font=title_font, fill="#162033")
    all_values = [v for _, vals, _ in series for v in vals]
    y_min = min(0.0, min(all_values) * 0.95)
    y_max = max(all_values) * 1.08
    if y_max <= y_min:
        y_max = y_min + 1.0
    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(6):
        y = top + (bottom - top) * i / 5
        value = y_max - (y_max - y_min) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=2)
        d.text((38, y - 14), f"{value:.1f}", font=small_font, fill="#586579")
    n = len(times)
    for i in range(0, n, max(1, n // 6)):
        x = left + (right - left) * i / max(n - 1, 1)
        d.line((x, bottom, x, bottom + 10), fill="#8795A7", width=2)
        elapsed_h = (times[i] - times[0]).total_seconds() / 3600.0
        d.text((x - 32, bottom + 18), f"{elapsed_h:.0f} h", font=small_font, fill="#586579")
    def xy(idx: int, value: float) -> Tuple[float, float]:
        x = left + (right - left) * idx / max(n - 1, 1)
        y = bottom - (bottom - top) * (value - y_min) / (y_max - y_min)
        return x, y
    for label, vals, color in series:
        pts = [xy(i, vals[i]) for i in range(n)]
        d.line(pts, fill=color, width=4)
    legend_x = left
    legend_y = h - 115
    for label, _, color in series:
        d.line((legend_x, legend_y, legend_x + 58, legend_y), fill=color, width=5)
        d.text((legend_x + 70, legend_y - 14), label, font=small_font, fill="#263241")
        legend_x += 260
        if legend_x > w - 300:
            legend_x = left
            legend_y += 36
    d.text((28, top + 260), y_label, font=text_font, fill="#263241")
    d.text((80, h - 48), caption, font=small_font, fill="#586579")
    img.save(path, quality=95)


def draw_mass_balance_chart(
    path: Path,
    times: List[datetime],
    mass_csv: Path,
):
    rows = []
    with mass_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    series = [
        ("累计入流", [float(r["cum_inflow_m3"]) / 1e6 for r in rows], "#0B5CAD"),
        ("累计出流", [float(r["cum_outflow_m3"]) / 1e6 for r in rows], "#2A9D8F"),
        ("累计分水", [float(r["cum_diversion_m3"]) / 1e6 for r in rows], "#E09F3E"),
        ("残差", [float(r["residual_m3"]) / 1e6 for r in rows], "#C75146"),
    ]
    draw_line_chart(
        path,
        "图3  恒定边界条件下的累计水量平衡",
        times,
        series,
        "累计水量 (10^6 m³)",
        "水量平衡残差 = 累计入流 - 累计出流 - 累计分水 - 槽蓄变化；接近0表示质量守恒闭合。",
    )


def draw_spatial_profile(
    path: Path,
    nodes: Dict[int, Node],
    dist_map: Dict[int, float],
    reaches: List[Reach],
    steady_flows: Dict[int, float],
    steady_depths: Dict[int, float],
    depth_slopes: Dict[int, float],
):
    w, h = 1800, 1250
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = get_font(40, True)
    small = get_font(21)
    d.text((80, 45), "图2  稳态沿程流量、Manning反算水深与水位分布", font=title_font, fill="#162033")
    left, right = 190, w - 90
    top_q, bottom_q = 135, 405
    top_h, bottom_h = 480, 750
    top_z, bottom_z = 825, 1095
    xs = [dist_map[n] / 1000.0 for n in KEY_NODES]
    q_vals = [steady_flows[n] for n in KEY_NODES]
    h_vals = [steady_depths[n] for n in KEY_NODES]
    bed_vals = [nodes[n].elev for n in KEY_NODES]
    wl_vals = [nodes[n].elev + steady_depths[n] for n in KEY_NODES]
    x_min, x_max = min(xs), max(xs)
    q_min, q_max = min(q_vals) * 0.92, max(q_vals) * 1.05
    h_min, h_max = min(h_vals) * 0.90, max(h_vals) * 1.08
    z_min = min(bed_vals + wl_vals) - 0.4
    z_max = max(bed_vals + wl_vals) + 0.4
    def x_pos(x):
        return left + (right - left) * (x - x_min) / max(x_max - x_min, 1e-9)
    def y_scale(v, y_min, y_max, top, bottom):
        return bottom - (bottom - top) * (v - y_min) / max(y_max - y_min, 1e-9)
    def draw_panel(top, bottom, y_min, y_max, ylabel):
        d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
        for tick in range(5):
            y = top + (bottom - top) * tick / 4
            value = y_max - (y_max - y_min) * tick / 4
            d.line((left, y, right, y), fill="#EEF2F7", width=2)
            d.text((140, y - 13), f"{value:.2f}" if y_max < 10 else f"{value:.1f}", font=small, fill="#586579")
        d.text((32, top + (bottom - top) / 2 - 14), ylabel, font=small, fill="#263241")
    draw_panel(top_q, bottom_q, q_min, q_max, "Q (m³/s)")
    draw_panel(top_h, bottom_h, h_min, h_max, "h (m)")
    draw_panel(top_z, bottom_z, z_min, z_max, "Z (m)")
    for x in xs:
        xp = x_pos(x)
        d.line((xp, top_q, xp, bottom_z), fill="#F0F3F7", width=1)
    q_pts = [(x_pos(xs[i]), y_scale(q_vals[i], q_min, q_max, top_q, bottom_q)) for i in range(len(xs))]
    h_pts = [(x_pos(xs[i]), y_scale(h_vals[i], h_min, h_max, top_h, bottom_h)) for i in range(len(xs))]
    bed_pts = [(x_pos(xs[i]), y_scale(bed_vals[i], z_min, z_max, top_z, bottom_z)) for i in range(len(xs))]
    wl_pts = [(x_pos(xs[i]), y_scale(wl_vals[i], z_min, z_max, top_z, bottom_z)) for i in range(len(xs))]
    d.line(q_pts, fill="#0B5CAD", width=5)
    d.line(h_pts, fill="#8C5A2B", width=5)
    d.line(bed_pts, fill="#6B7280", width=4)
    d.line(wl_pts, fill="#2A9D8F", width=5)
    for i, n in enumerate(KEY_NODES):
        x = x_pos(xs[i])
        d.ellipse((q_pts[i][0] - 7, q_pts[i][1] - 7, q_pts[i][0] + 7, q_pts[i][1] + 7), fill="#0B5CAD")
        d.ellipse((h_pts[i][0] - 7, h_pts[i][1] - 7, h_pts[i][0] + 7, h_pts[i][1] + 7), fill="#8C5A2B")
        d.ellipse((wl_pts[i][0] - 7, wl_pts[i][1] - 7, wl_pts[i][0] + 7, wl_pts[i][1] + 7), fill="#2A9D8F")
        d.text((x - 28, bottom_z + 18), str(n), font=small, fill="#586579")
        d.text((q_pts[i][0] - 32, q_pts[i][1] - 34), f"{q_vals[i]:.1f}", font=small, fill="#263241")
        d.text((h_pts[i][0] - 32, h_pts[i][1] - 34), f"{h_vals[i]:.2f}", font=small, fill="#263241")
        d.text((wl_pts[i][0] - 38, wl_pts[i][1] - 32), f"{wl_vals[i]:.2f}", font=small, fill="#263241")
    d.text((685, h - 125), "节点编号（横向位置按距渠首里程绘制，km）", font=small, fill="#263241")
    legend_y = h - 88
    d.line((190, legend_y, 260, legend_y), fill="#0B5CAD", width=5)
    d.text((275, legend_y - 15), "稳态流量 Q", font=small, fill="#263241")
    d.line((510, legend_y, 580, legend_y), fill="#8C5A2B", width=5)
    d.text((595, legend_y - 15), "反算水深 h", font=small, fill="#263241")
    d.line((835, legend_y, 905, legend_y), fill="#2A9D8F", width=5)
    d.text((920, legend_y - 15), "反算水位 Zbed+h", font=small, fill="#263241")
    d.line((1245, legend_y, 1315, legend_y), fill="#6B7280", width=4)
    d.text((1330, legend_y - 15), "渠底高程 Zbed", font=small, fill="#263241")
    slope_min = min(depth_slopes[n] for n in KEY_NODES)
    slope_max = max(depth_slopes[n] for n in KEY_NODES)
    if abs(slope_max - slope_min) < 1e-12:
        slope_note = f"代表性能坡 S={slope_min:.6f} m/m"
    else:
        slope_note = f"局部床坡范围={slope_min:.6f}-{slope_max:.6f} m/m"
    d.text(
        (80, h - 45),
        f"水深按节点真实断面、稳态流量与Manning公式反算；{slope_note}。",
        font=small,
        fill="#586579",
    )
    img.save(path, quality=95)


def draw_topology(
    path: Path,
    nodes: Dict[int, Node],
    main_nodes: List[int],
    dist_map: Dict[int, float],
):
    w, h = 1800, 1050
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = get_font(40, True)
    small = get_font(22)
    d.text((80, 45), "图1  0-456主干渠与分水口边界示意", font=title_font, fill="#162033")
    xs = [nodes[n].x for n in main_nodes]
    ys = [nodes[n].y for n in main_nodes]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    left, top, right, bottom = 125, 135, w - 90, h - 170
    def xy(n):
        x = left + (nodes[n].x - minx) / max(maxx - minx, 1e-9) * (right - left)
        y = bottom - (nodes[n].y - miny) / max(maxy - miny, 1e-9) * (bottom - top)
        return x, y
    pts = [xy(n) for n in main_nodes]
    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    d.line(pts, fill="#0B5CAD", width=7)
    offsets = {
        0: (25, -35),
        71: (-120, 25),
        89: (35, 20),
        150: (-150, -40),
        194: (35, -35),
        287: (-165, 20),
        349: (-130, 28),
        383: (35, -42),
        456: (-100, -55),
    }
    for n in KEY_NODES:
        x, y = xy(n)
        color = "#D1495B" if n in DIVERSION_DEFAULTS else "#0B5CAD"
        d.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color, outline="white", width=3)
        ox, oy = offsets.get(n, (25, -25))
        label = NODE_LABELS.get(n, str(n))
        box_w, box_h = 190, 48
        bx = max(60, min(x + ox, w - box_w - 60))
        by = max(110, min(y + oy, h - box_h - 130))
        d.rounded_rectangle((bx, by, bx + box_w, by + box_h), radius=12, fill="#F8FAFD", outline="#B9C5D6", width=2)
        d.line((x, y, bx + box_w / 2, by + box_h / 2), fill="#8795A7", width=2)
        d.text((bx + 12, by + 12), label, font=small, fill="#263241")
    d.line((125, h - 78, 195, h - 78), fill="#0B5CAD", width=6)
    d.text((210, h - 93), "主干渠流量路由方向", font=small, fill="#263241")
    d.ellipse((570, h - 90, 594, h - 66), fill="#D1495B", outline="white", width=2)
    d.text((610, h - 93), "恒定侧向出流边界", font=small, fill="#263241")
    d.text((80, h - 45), "渠首恒定入流 Q0=108 m³/s；每个分水口恒定分水 Dk=8 m³/s。", font=small, fill="#586579")
    img.save(path, quality=95)


def summarize_depths(
    q_by_node: Dict[int, List[float]],
    reaches: List[Reach],
    snapshot_idx: int,
    display_slope: float,
) -> Dict[int, float]:
    depths = {}
    reference_section = reaches[0].section
    for i, n in enumerate(KEY_NODES):
        depths[n] = normal_depth(max(q_by_node[n][snapshot_idx], 0.01), reference_section, display_slope)
    return depths


def main():
    parser = argparse.ArgumentParser(description="Muskingum-Cunge stage-1 canal routing for nodes 0-456.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--start-date", default="2024-03-21")
    parser.add_argument("--duration-hours", type=float, default=18.0)
    parser.add_argument("--dt-minutes", type=int, default=5)
    parser.add_argument("--headflow", type=float, default=HEADFLOW_CONSTANT)
    parser.add_argument("--diversion", type=float, default=DIVERSION_CONSTANT)
    parser.add_argument(
        "--loss-per-km",
        type=float,
        default=0.0,
        help="Optional distributed conveyance loss in m3/s per km. Default is 0 because no calibrated loss data are used.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    nodes = parse_nodes(data_dir / "input.txt")
    params = parse_line_params(data_dir / "lineParam.txt")
    main_nodes = [n for n in range(MAIN_START, MAIN_END + 1) if n in nodes]
    dist_map = cumulative_distances(nodes, main_nodes)
    dt_seconds = args.dt_minutes * 60
    diversions = {node: args.diversion for node in DIVERSION_DEFAULTS}
    times, headflow = constant_headflow_series(args.start_date, args.duration_hours, dt_seconds, args.headflow)
    q_ref = args.headflow - sum(diversions.values()) * 0.4
    reaches = build_reaches(nodes, params, KEY_NODES, dist_map, max(q_ref, 10.0), dt_seconds)
    q_by_node, lateral_by_node, storage_by_reach = route_flows(
        reaches,
        times,
        headflow,
        diversions,
        dt_seconds,
        loss_per_km_m3s=args.loss_per_km,
    )

    time_series_csv = out_dir / "stage1_time_series_at_key_nodes.csv"
    reach_csv = out_dir / "stage1_reach_parameters.csv"
    mass_csv = out_dir / "stage1_mass_balance.csv"
    steady_profile_csv = out_dir / "stage1_steady_profile.csv"
    write_time_series(time_series_csv, times, q_by_node, lateral_by_node, KEY_NODES)
    write_reaches(reach_csv, reaches)
    mass_summary = write_mass_balance(mass_csv, times, q_by_node, lateral_by_node, storage_by_reach, dt_seconds)
    steady_flows = steady_node_flows_with_losses(args.headflow, diversions, reaches, args.loss_per_km)
    display_slope = representative_bed_slope(nodes, dist_map)
    depth_slopes = {node: display_slope for node in KEY_NODES}
    depths = {
        node: normal_depth(max(flow, 0.01), section_for_node(node, nodes, params), display_slope)
        for node, flow in steady_flows.items()
    }
    write_steady_profile(steady_profile_csv, steady_flows, depths, dist_map, nodes, params, depth_slopes)

    selected = [0, 194, 287, 349, 456]
    draw_line_chart(
        fig_dir / "flow_hydrographs.png",
        "图4  恒定边界条件下流量路由收敛过程",
        times,
        [(NODE_LABELS[n], q_by_node[n], c) for n, c in zip(selected, ["#0B5CAD", "#E09F3E", "#2A9D8F", "#9B5DE5", "#D1495B"])],
        "流量 Q (m³/s)",
        "该图表示从初始条件向稳态解的计算收敛过程；边界条件全程保持恒定。",
    )
    draw_topology(fig_dir / "topology_diversion_boundaries.png", nodes, main_nodes, dist_map)
    draw_spatial_profile(fig_dir / "spatial_snapshot_flow_depth.png", nodes, dist_map, reaches, steady_flows, depths, depth_slopes)
    draw_mass_balance_chart(fig_dir / "mass_balance_cumulative.png", times, mass_csv)

    report = {
        "model": "Muskingum-Cunge / diffusive-wave discharge routing",
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "start_date": args.start_date,
        "duration_hours": args.duration_hours,
        "dt_seconds": dt_seconds,
        "main_node_count": len(main_nodes),
        "main_length_km": dist_map[MAIN_END] / 1000.0,
        "key_nodes": KEY_NODES,
        "boundary_condition": "constant head inflow and constant diversion outflows; SCADA is not used",
        "headflow_constant_m3s": args.headflow,
        "diversion_constant_m3s_each": args.diversion,
        "distributed_loss_per_km_m3s": args.loss_per_km,
        "distributed_loss_note": "Default is 0.0 because lineParam infiltration values are 0 or not calibrated for the current stage.",
        "diversions_m3s": diversions,
        "headflow_min_m3s": min(headflow),
        "headflow_max_m3s": max(headflow),
        "headflow_initial_m3s": headflow[0],
        "tailflow_initial_m3s": q_by_node[MAIN_END][0],
        "tailflow_final_m3s": q_by_node[MAIN_END][-1],
        "tailflow_steady_m3s": steady_flows[MAIN_END],
        "steady_discharge_balance_m3s": args.headflow - steady_flows[MAIN_END] - sum(diversions.values()),
        "representative_bed_slope_for_depth": display_slope,
        "depth_slope_mode": "representative_energy_slope_with_true_node_section",
        "depth_slopes_used": {str(k): v for k, v in depth_slopes.items()},
        "steady_snapshot": "analytical steady distribution from constant inflow minus cumulative diversions",
        "steady_flows_m3s": {str(k): v for k, v in steady_flows.items()},
        "snapshot_depths_m": {str(k): v for k, v in depths.items()},
        "mass_balance": mass_summary,
        "outputs": {
            "time_series_csv": str(time_series_csv),
            "reach_parameters_csv": str(reach_csv),
            "mass_balance_csv": str(mass_csv),
            "steady_profile_csv": str(steady_profile_csv),
            "figures": [str(p) for p in sorted(fig_dir.glob("*.png"))],
        },
    }
    (out_dir / "stage1_run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
