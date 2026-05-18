#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Saint-Venant dispatch simulation for the 0-456 main canal.

This script is the first Saint-Venant-based replacement for the earlier
dynamic-routing preview. It solves the 1-D shallow-water/Saint-Venant equations
with an explicit finite-volume HLL flux on the existing 0-456 node grid:

    ∂A/∂t + ∂Q/∂x = -q_div
    ∂Q/∂t + ∂(Q²/A + g I1)/∂x = g A (S0 - Sf) - q_div u

where A is wetted area, Q is discharge, I1 is the hydrostatic pressure
integral for a trapezoidal section, S0 is local bed slope, Sf is Manning
friction slope, and q_div is the lateral diversion sink per unit length.

The model is intentionally a first forward Saint-Venant experiment:
- initial canal is numerically wetted with a small depth;
- head inflow rises smoothly from 0 to 108 m3/s in 0.5 h;
- diversion outflow is controlled by local depth threshold, maximum outlet
  capacity, and remaining demand.
- the advective/hydrostatic flux is explicit, while Manning friction is
  treated with a semi-implicit linearized update for better startup stability.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

import muskingum_cunge_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "saint_venant_legacy_dispatch_results"
FIG_DIR = OUT_DIR / "figures"

G = 9.81
MAIN_START = 0
MAIN_END = 456
DIVERSION_NODES = [71, 89, 150, 194, 287, 349, 383]


@dataclass
class DiversionSpec:
    node: int
    max_flow_m3s: float
    demand_m3: float


@dataclass
class Config:
    dt_seconds: float = 1.0
    duration_hours: float = 14.0
    ramp_hours: float = 0.5
    target_head_flow_m3s: float = 108.0
    initial_depth_m: float = 0.08
    min_depth_m: float = 0.03
    output_interval_seconds: float = 60.0


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


def friction_slope(area: float, q: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    perimeter = wetted_perimeter(h, sec)
    radius = area / max(perimeter, 1.0e-9)
    return q * abs(q) * sec.manning_n ** 2 / max(area * area * radius ** (4.0 / 3.0), 1.0e-9)


def friction_coefficient(area: float, q_reference: float, sec: stage1.SectionParam) -> float:
    """Linearized coefficient Sf(Q^{n+1}) ~= Cf * Q^{n+1}.

    Manning's friction slope is Sf = n^2 Q|Q| / (A^2 R^(4/3)). In the
    semi-implicit step, |Q|, A and R are evaluated from the latest known state
    while Q itself is solved implicitly.
    """
    h = depth_from_area(area, sec)
    perimeter = wetted_perimeter(h, sec)
    radius = area / max(perimeter, 1.0e-9)
    return sec.manning_n ** 2 * abs(q_reference) / max(area * area * radius ** (4.0 / 3.0), 1.0e-9)


def wave_speed(area: float, sec: stage1.SectionParam) -> float:
    h = depth_from_area(area, sec)
    b = top_width(h, sec)
    hydraulic_depth = area / max(b, 1.0e-9)
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
    return nodes, ids, x, dx_cell, sections, bed, diversion_indices


def simulate(cfg: Config):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nodes, ids, x, dx_cell, sections, bed, diversion_indices = build_grid()
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

        # Boundary ghost states.
        q_in = head_inflow(t_s, cfg)
        a_left = max(a[0], min_area[0])
        q_left = q_in
        a_right = max(a[-1], min_area[-1])
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

            # Bed slope and Manning friction source terms.
            if i == 0:
                s0 = (bed[i] - bed[i + 1]) / max(x[i + 1] - x[i], 1.0)
            elif i == n - 1:
                s0 = (bed[i - 1] - bed[i]) / max(x[i] - x[i - 1], 1.0)
            else:
                s0 = (bed[i - 1] - bed[i + 1]) / max(x[i + 1] - x[i - 1], 1.0)
            # Bed-slope source is explicit. Manning friction is applied after
            # the provisional update with a semi-implicit linearized solve.
            dq += G * max(a[i], min_area[i]) * s0

            new_a[i] = a[i] + cfg.dt_seconds * da
            new_q[i] = q[i] + cfg.dt_seconds * dq

        # Apply diversion sinks after the conservative update.
        for node, idx in diversion_indices.items():
            if close_time[node] is not None:
                continue
            spec = SPECS[node]
            h_local = depth_from_area(max(new_a[idx], min_area[idx]), sections[idx])
            factor = diversion_factor(h_local, sections[idx].depth)
            remaining = max(spec.demand_m3 - supplied[node], 0.0)
            possible_by_storage = max(new_a[idx] - min_area[idx], 0.0) * dx_cell[idx] / cfg.dt_seconds
            q_capacity = spec.max_flow_m3s * factor
            qdiv = min(q_capacity, remaining / cfg.dt_seconds, possible_by_storage)
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
            # Mild limiter for numerical startup noise near dry state.
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
        "first_positive_h": first_positive,
        "close_h": close_time,
    }


def draw_fig2(result):
    w, h = 1800, 1050
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = stage1.get_font(40, True)
    font = stage1.get_font(23)
    small = stage1.get_font(19)
    d.text((80, 45), "图2  Saint-Venant半隐式正演下各配水口实际出流过程", font=title, fill="#162033")
    left, top, right, bottom = 170, 140, w - 100, h - 185
    times = result["times_h"]
    max_t = max(times)
    qmax = max(max(result["qdiv"][n]) for n in DIVERSION_NODES) * 1.10
    qmax = max(qmax, 1.0)
    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(6):
        x = left + (right - left) * i / 5
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        d.text((x - 28, bottom + 16), f"{max_t*i/5:.1f}", font=small, fill="#586579")
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((55, y - 13), f"{qmax*i/5:.1f}", font=small, fill="#586579")

    def xp(t):
        return left + (right - left) * t / max_t

    def yp(qv):
        return bottom - (bottom - top) * qv / qmax

    label_offsets = {
        71: (-8, -30),
        89: (8, -18),
        150: (10, -34),
        194: (10, -10),
        287: (10, -32),
        349: (10, 12),
        383: (10, 34),
    }
    for node in DIVERSION_NODES:
        vals = result["qdiv"][node]
        cl = result["close_h"][node]
        if cl is None:
            pts = [(xp(t), yp(v)) for t, v in zip(times, vals)]
        else:
            pts = [(xp(t), yp(v)) for t, v in zip(times, vals) if t <= cl]
        if len(pts) >= 2:
            d.line(pts, fill=COLORS[node], width=4)
        cl = result["close_h"][node]
        if cl is not None:
            ox, oy = label_offsets.get(node, (8, -20))
            y_top = yp(max(vals))
            x_cl = xp(cl)
            dash = 10
            y0 = y_top
            while y0 < yp(0):
                d.line((x_cl, y0, x_cl, min(y0 + dash, yp(0))), fill=COLORS[node], width=2)
                y0 += dash * 2
            d.ellipse((x_cl - 5, yp(0) - 5, x_cl + 5, yp(0) + 5), fill=COLORS[node])
            d.text((x_cl + ox, y_top + oy), f"{node}关 {cl:.2f}h", font=small, fill=COLORS[node])

    legend_x, legend_y = left, h - 102
    for node in DIVERSION_NODES:
        spec = SPECS[node]
        d.line((legend_x, legend_y, legend_x + 42, legend_y), fill=COLORS[node], width=5)
        d.text((legend_x + 52, legend_y - 13), f"{node}口 max={spec.max_flow_m3s:.0f}", font=small, fill="#263241")
        legend_x += 210
        if legend_x > right - 220:
            legend_x = left
            legend_y += 34

    d.text(((left + right) / 2 - 85, h - 58), "时间 t (h)", font=font, fill="#263241")
    d.text((28, (top + bottom) / 2 - 13), "实际出流 Qdiv (m³/s)", font=font, fill="#263241")
    d.text(
        (80, h - 35),
        "渠首 0→108 m³/s 平滑爬升；HLL通量显式推进，Manning摩擦半隐式处理；虚线表示满足需水后的关闸动作。",
        font=small,
        fill="#586579",
    )
    out = FIG_DIR / "fig2_saint_venant_semi_implicit_diversion_outflow.png"
    img.save(out, quality=95)
    return out


def write_outputs(result, fig_path: Path, cfg: Config):
    with (OUT_DIR / "legacy_dispatch_timeseries.csv").open("w", encoding="utf-8-sig", newline="") as f:
        depth_nodes = sorted(result.get("depth", {}).keys())
        fields = (
            ["time_h", "head_inflow_m3s"]
            + [f"Qdiv_{n}_m3s" for n in DIVERSION_NODES]
            + [f"supplied_{n}_m3" for n in DIVERSION_NODES]
            + [f"depth_{n}_m" for n in depth_nodes]
            + [f"water_level_{n}_m" for n in depth_nodes]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(result["times_h"]):
            row = {"time_h": f"{t:.6f}", "head_inflow_m3s": f"{result['head_q'][i]:.6f}"}
            for n in DIVERSION_NODES:
                row[f"Qdiv_{n}_m3s"] = f"{result['qdiv'][n][i]:.6f}"
                row[f"supplied_{n}_m3"] = f"{result['supplied'][n][i]:.3f}"
            for n in depth_nodes:
                row[f"depth_{n}_m"] = f"{result['depth'][n][i]:.6f}"
                row[f"water_level_{n}_m"] = f"{result['water_level'][n][i]:.6f}"
            writer.writerow(row)
    summary = {
        "figure": str(fig_path),
        "model": "1-D Saint-Venant finite-volume HLL forward model with semi-implicit Manning friction",
        "dt_seconds": cfg.dt_seconds,
        "duration_hours": cfg.duration_hours,
        "head_boundary": {
            "initial_flow_m3s": 0.0,
            "target_flow_m3s": cfg.target_head_flow_m3s,
            "ramp_hours": cfg.ramp_hours,
            "initial_wetting_depth_m": cfg.initial_depth_m,
        },
        "diversions": {
            str(n): {
                "max_flow_m3s": SPECS[n].max_flow_m3s,
                "demand_m3": SPECS[n].demand_m3,
                "first_positive_time_h": result["first_positive_h"][n],
                "close_time_h": result["close_h"][n],
                "final_supplied_m3": result["supplied"][n][-1],
            }
            for n in DIVERSION_NODES
        },
    }
    (OUT_DIR / "legacy_dispatch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    cfg = Config()
    result = simulate(cfg)
    fig_path = draw_fig2(result)
    write_outputs(result, fig_path, cfg)


if __name__ == "__main__":
    main()
