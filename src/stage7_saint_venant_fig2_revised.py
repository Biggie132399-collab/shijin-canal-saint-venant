#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised Saint-Venant forward simulation for Figure 2.

Revision relative to the first Saint-Venant trial:
- head inflow ramps from 0 to 80 m3/s;
- diversion outlet capacities are set to 20, 20, 5, 5, 12, 5, and 5 m3/s
  with their corresponding assumed demands;
- upstream and downstream ghost cells use Manning normal depth compatible with
  the boundary discharge;
- diversion discharge is constrained by specified outlet capacity, estimated
  branch-canal safe capacity, remaining demand, and local available water.

The branch-canal capacity is only a first engineering bound because no branch
water-level or gate geometry is available. It is estimated with Manning's
formula at h_safe = 0.90D of the first branch canal segment.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

import muskingum_cunge_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"

G = 9.81
MAIN_START = 0
MAIN_END = 456
DIVERSION_NODES = [71, 89, 150, 194, 287, 349, 383]
LARGE_PANEL_NODES = [71, 89, 287]
SMALL_PANEL_NODES = [150, 194, 349, 383]


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


@dataclass
class Config:
    dt_seconds: float = 1.0
    duration_hours: float = 14.0
    ramp_hours: float = 0.5
    target_head_flow_m3s: float = 80.0
    initial_depth_m: float = 0.08
    min_depth_m: float = 0.03
    output_interval_seconds: float = 60.0
    safe_depth_ratio: float = 0.90
    min_bed_slope: float = 5.0e-5


SPECS: Dict[int, DiversionSpec] = {
    71: DiversionSpec(71, 20.0, 250_000.0),
    89: DiversionSpec(89, 20.0, 300_000.0),
    150: DiversionSpec(150, 5.0, 60_000.0),
    194: DiversionSpec(194, 5.0, 80_000.0),
    287: DiversionSpec(287, 12.0, 160_000.0),
    349: DiversionSpec(349, 5.0, 50_000.0),
    383: DiversionSpec(383, 5.0, 40_000.0),
}

COLORS = {
    71: "#0B5CAD",
    89: "#D1495B",
    150: "#2A9D8F",
    194: "#E09F3E",
    287: "#7B2CBF",
    349: "#6B7280",
    383: "#C75146",
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


def depth_from_area(area: float, sec: stage1.SectionParam) -> float:
    a = max(area, 1.0e-9)
    b = sec.bottom_width
    z = sec.side_slope
    if abs(z) < 1.0e-12:
        return a / max(b, 1.0e-9)
    return (-b + math.sqrt(b * b + 4.0 * z * a)) / (2.0 * z)


def top_width(h: float, sec: stage1.SectionParam) -> float:
    return sec.bottom_width + 2.0 * sec.side_slope * max(h, 0.0)


def wetted_perimeter(h: float, sec: stage1.SectionParam) -> float:
    return sec.bottom_width + 2.0 * max(h, 0.0) * math.sqrt(1.0 + sec.side_slope * sec.side_slope)


def pressure_integral(h: float, sec: stage1.SectionParam) -> float:
    y = max(h, 0.0)
    return 0.5 * sec.bottom_width * y * y + (sec.side_slope * y ** 3) / 3.0


def normal_depth(q: float, sec: stage1.SectionParam, slope: float, cfg: Config) -> float:
    if q <= 1.0e-9:
        return cfg.min_depth_m
    return stage1.normal_depth(q, sec, max(slope, cfg.min_bed_slope))


def friction_coefficient(area: float, q_reference: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    perimeter = wetted_perimeter(h, sec)
    radius = area / max(perimeter, 1.0e-9)
    return sec.manning_n ** 2 * abs(q_reference) / max(area * area * radius ** (4.0 / 3.0), 1.0e-9)


def wave_speed(area: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    hydraulic_depth = area / max(top_width(h, sec), 1.0e-9)
    return math.sqrt(G * max(hydraulic_depth, 1.0e-9))


def flux(area: float, q: float, sec: stage1.SectionParam) -> Tuple[float, float]:
    h = depth_from_area(area, sec)
    return q, q * q / max(area, 1.0e-9) + G * pressure_integral(h, sec)


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
            limits[node] = BranchLimit(None, float("inf"), None, None, None)
            continue
        child = branch_children[0]
        sec = stage1.section_for_node(child, nodes, params)
        dist = stage1.distance(nodes[node], nodes[child])
        raw_slope = (nodes[node].elev - nodes[child].elev) / max(dist, 1.0)
        slope = max(raw_slope, cfg.min_bed_slope)
        h_safe = cfg.safe_depth_ratio * sec.depth
        q_safe = stage1.manning_q(h_safe, sec, slope)
        limits[node] = BranchLimit(child, q_safe, sec.depth, sec.bottom_width, slope)
    return limits


def simulate(cfg: Config):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nodes, params, neighbors, ids, x, dx_cell, sections, bed, diversion_indices = build_grid()
    limits = branch_limits(nodes, params, neighbors, cfg)
    n = len(ids)
    a = [area_from_depth(cfg.initial_depth_m, sec) for sec in sections]
    q = [0.0 for _ in range(n)]
    min_area = [area_from_depth(cfg.min_depth_m, sec) for sec in sections]
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
    key_indices = {node: ids.index(node) for node in key_nodes if node in ids}
    depth_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    water_level_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    depth_limit_series: Dict[int, float] = {node: sections[idx].depth for node, idx in key_indices.items()}
    safe_depth_series: Dict[int, float] = {node: cfg.safe_depth_ratio * sections[idx].depth for node, idx in key_indices.items()}

    total_steps = int(round(cfg.duration_hours * 3600.0 / cfg.dt_seconds))
    for step in range(total_steps + 1):
        t_s = step * cfg.dt_seconds
        if step % output_every == 0:
            times_h.append(t_s / 3600.0)
            head_q_series.append(head_inflow(t_s, cfg))
            for node in DIVERSION_NODES:
                qdiv_series[node].append(qdiv_current[node])
                supplied_series[node].append(supplied[node])
            for node, idx in key_indices.items():
                h_now = depth_from_area(max(a[idx], min_area[idx]), sections[idx])
                depth_series[node].append(h_now)
                water_level_series[node].append(bed[idx] + h_now)

        if step == total_steps:
            break

        q_in = head_inflow(t_s, cfg)
        h_left = normal_depth(q_in, sections[0], local_slope(0, x, bed, cfg), cfg)
        a_left = area_from_depth(max(h_left, cfg.min_depth_m), sections[0])
        q_left = q_in

        q_down = max(q[-1], 0.0)
        h_right = normal_depth(q_down, sections[-1], local_slope(n - 1, x, bed, cfg), cfg)
        a_right = area_from_depth(max(h_right, cfg.min_depth_m), sections[-1])
        q_right = q[-1]

        fluxes: List[Tuple[float, float]] = []
        fluxes.append(hll_flux(a_left, q_left, sections[0], a[0], q[0], sections[0]))
        for i in range(n - 1):
            fluxes.append(hll_flux(a[i], q[i], sections[i], a[i + 1], q[i + 1], sections[i + 1]))
        fluxes.append(hll_flux(a[-1], q[-1], sections[-1], a_right, q_right, sections[-1]))

        new_a = a[:]
        new_q = q[:]
        qdiv_current = {node: 0.0 for node in DIVERSION_NODES}

        for i in range(n):
            dx = max(dx_cell[i], 1.0)
            da = -(fluxes[i + 1][0] - fluxes[i][0]) / dx
            dq = -(fluxes[i + 1][1] - fluxes[i][1]) / dx
            dq += G * max(a[i], min_area[i]) * local_slope(i, x, bed, cfg)
            new_a[i] = a[i] + cfg.dt_seconds * da
            new_q[i] = q[i] + cfg.dt_seconds * dq

        for node, idx in diversion_indices.items():
            if close_time[node] is not None:
                continue
            spec = SPECS[node]
            sec = sections[idx]
            h_local = depth_from_area(max(new_a[idx], min_area[idx]), sec)
            factor = diversion_factor(h_local, sec.depth)
            remaining = max(spec.demand_m3 - supplied[node], 0.0)
            storage_above_min = max(new_a[idx] - min_area[idx], 0.0) * dx_cell[idx] / cfg.dt_seconds
            q_capacity = min(spec.max_flow_m3s, limits[node].safe_capacity_m3s) * factor
            qdiv = min(q_capacity, remaining / cfg.dt_seconds, storage_above_min)
            if qdiv > 1.0e-6 and first_positive[node] is None:
                first_positive[node] = t_s / 3600.0
            supplied[node] += qdiv * cfg.dt_seconds
            if supplied[node] >= spec.demand_m3 - 1.0e-6 and close_time[node] is None:
                close_time[node] = (t_s + cfg.dt_seconds) / 3600.0
            qdiv_current[node] = qdiv
            new_a[idx] -= qdiv * cfg.dt_seconds / max(dx_cell[idx], 1.0)
            velocity = new_q[idx] / max(new_a[idx], min_area[idx])
            new_q[idx] -= qdiv * velocity * cfg.dt_seconds / max(dx_cell[idx], 1.0)

        for i in range(n):
            if new_a[i] < min_area[i]:
                new_a[i] = min_area[i]
            cf = friction_coefficient(max(new_a[i], min_area[i]), q[i], sections[i])
            denom = 1.0 + cfg.dt_seconds * G * max(new_a[i], min_area[i]) * cf
            new_q[i] = new_q[i] / max(denom, 1.0e-9)
            if abs(new_q[i]) < 1.0e-7:
                new_q[i] = 0.0
            c = wave_speed(new_a[i], sections[i])
            q_limit = 8.0 * new_a[i] * max(c, 0.1)
            new_q[i] = max(min(new_q[i], q_limit), -q_limit)

        a, q = new_a, new_q

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
    }


def draw_dashed_vertical(d, x, y0, y1, color, width=2):
    y = y0
    while y < y1:
        d.line((x, y, x, min(y + 9, y1)), fill=color, width=width)
        y += 18


def draw_panel(d, result, nodes, rect, y_max, label, show_x_ticks=True):
    left, top, right, bottom = rect
    small = stage1.get_font(19)
    font = stage1.get_font(22)
    times = result["times_h"]
    max_t = max(times)
    d.rectangle(rect, outline="#CAD3DF", width=2)
    for i in range(6):
        x = left + (right - left) * i / 5
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        if show_x_ticks:
            d.text((x - 28, bottom + 14), f"{max_t*i/5:.1f}", font=small, fill="#586579")
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((58, y - 12), f"{y_max*i/5:.1f}", font=small, fill="#586579")

    def xp(t):
        return left + (right - left) * t / max_t

    def yp(qv):
        return bottom - (bottom - top) * qv / y_max

    for node in nodes:
        vals = result["qdiv"][node]
        close_h = result["close_h"][node]
        pts = [(xp(t), yp(v)) for t, v in zip(times, vals) if close_h is None or t <= close_h]
        if len(pts) >= 2:
            d.line(pts, fill=COLORS[node], width=4)
        if close_h is not None:
            x_close = xp(close_h)
            y_top = yp(max(vals))
            draw_dashed_vertical(d, x_close, y_top, yp(0), COLORS[node])
            d.ellipse((x_close - 5, yp(0) - 5, x_close + 5, yp(0) + 5), fill=COLORS[node])
            d.text((x_close + 8, max(top + 4, y_top - 28)), f"{node}关 {close_h:.2f}h", font=small, fill=COLORS[node])

    d.text((left + 14, top + 10), label, font=font, fill="#263241")


def draw_fig2(result):
    w, h = 1800, 1180
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = stage1.get_font(40, True)
    font = stage1.get_font(23)
    small = stage1.get_font(19)

    d.text((80, 45), "图2  各配水口实际出流过程", font=title, fill="#162033")
    left, right = 170, w - 105
    top1, bottom1 = 140, 555
    top2, bottom2 = 660, 995

    y_large = max(max(result["qdiv"][n]) for n in LARGE_PANEL_NODES) * 1.12
    y_small = max(max(result["qdiv"][n]) for n in SMALL_PANEL_NODES) * 1.25
    draw_panel(d, result, LARGE_PANEL_NODES, (left, top1, right, bottom1), max(y_large, 1.0), "大/中型配水口", show_x_ticks=False)
    draw_panel(d, result, SMALL_PANEL_NODES, (left, top2, right, bottom2), max(y_small, 1.0), "小型配水口", show_x_ticks=True)

    legend_x, legend_y = left, h - 125
    for node in DIVERSION_NODES:
        spec = SPECS[node]
        d.line((legend_x, legend_y, legend_x + 42, legend_y), fill=COLORS[node], width=5)
        d.text((legend_x + 52, legend_y - 13), f"{node}口  Qmax={spec.max_flow_m3s:g} m³/s", font=small, fill="#263241")
        legend_x += 255
        if legend_x > right - 230:
            legend_x = left
            legend_y += 34

    d.text(((left + right) / 2 - 85, h - 70), "时间 t (h)", font=font, fill="#263241")
    d.text((30, (top1 + bottom2) / 2 - 20), "实际出流 Qdiv (m³/s)", font=font, fill="#263241")
    d.text(
        (80, h - 36),
        "说明：渠首流量0→80 m³/s平滑爬升；分水口按给定最大能力放水，实际出流受局部水深、剩余需水量和干渠可供水量共同约束；虚线表示达需后的关闸时刻。",
        font=small,
        fill="#586579",
    )

    out = FIG_DIR / "fig2_diversion_outflow_process.png"
    img.save(out, quality=95)
    return out


def write_outputs(result, fig_path: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "fig2_diversion_outflow_timeseries.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["time_h", "head_inflow_m3s"] + [f"Qdiv_{n}_m3s" for n in DIVERSION_NODES] + [f"supplied_{n}_m3" for n in DIVERSION_NODES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(result["times_h"]):
            row = {"time_h": f"{t:.6f}", "head_inflow_m3s": f"{result['head_q'][i]:.6f}"}
            for n in DIVERSION_NODES:
                row[f"Qdiv_{n}_m3s"] = f"{result['qdiv'][n][i]:.6f}"
                row[f"supplied_{n}_m3"] = f"{result['supplied'][n][i]:.3f}"
            writer.writerow(row)

    table_path = OUT_DIR / "dispatch_feasibility_outlets.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "node",
            "specified_qmax_m3s",
            "branch_node",
            "branch_safe_capacity_m3s",
            "demand_m3",
            "first_positive_time_h",
            "close_time_h",
            "final_supplied_m3",
            "demand_satisfied",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in DIVERSION_NODES:
            spec = SPECS[n]
            limit = result["branch_limits"][n]
            final_supplied = result["supplied"][n][-1]
            writer.writerow(
                {
                    "node": n,
                    "specified_qmax_m3s": f"{spec.max_flow_m3s:.3f}",
                    "branch_node": "" if limit.branch_node is None else limit.branch_node,
                    "branch_safe_capacity_m3s": f"{limit.safe_capacity_m3s:.3f}",
                    "demand_m3": f"{spec.demand_m3:.3f}",
                    "first_positive_time_h": "" if result["first_positive_h"][n] is None else f"{result['first_positive_h'][n]:.6f}",
                    "close_time_h": "" if result["close_h"][n] is None else f"{result['close_h'][n]:.6f}",
                    "final_supplied_m3": f"{final_supplied:.3f}",
                    "demand_satisfied": final_supplied >= spec.demand_m3 - 1.0e-6,
                }
            )

    cfg = result["config"]
    summary = {
        "figure": str(fig_path),
        "timeseries_csv": str(OUT_DIR / "fig2_diversion_outflow_timeseries.csv"),
        "dispatch_table_csv": str(table_path),
        "model": "1-D Saint-Venant finite-volume HLL forward model with semi-implicit Manning friction",
        "head_boundary": {
            "initial_flow_m3s": 0.0,
            "target_flow_m3s": cfg.target_head_flow_m3s,
            "ramp_hours": cfg.ramp_hours,
            "boundary_depth": "Manning normal depth compatible with boundary discharge",
        },
        "diversion_rule": "Qdiv=min(specified Qmax, branch safe capacity, remaining demand/dt, local available water) times depth activation factor",
        "safe_depth_ratio_for_branch_capacity": cfg.safe_depth_ratio,
        "diversions": {
            str(n): {
                "specified_qmax_m3s": SPECS[n].max_flow_m3s,
                "demand_m3": SPECS[n].demand_m3,
                "branch_node": result["branch_limits"][n].branch_node,
                "branch_safe_capacity_m3s": result["branch_limits"][n].safe_capacity_m3s,
                "first_positive_time_h": result["first_positive_h"][n],
                "close_time_h": result["close_h"][n],
                "final_supplied_m3": result["supplied"][n][-1],
            }
            for n in DIVERSION_NODES
        },
    }
    (OUT_DIR / "fig2_diversion_outflow_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    cfg = Config()
    result = simulate(cfg)
    fig_path = draw_fig2(result)
    write_outputs(result, fig_path)


if __name__ == "__main__":
    main()
