#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6: maximum water-depth envelope along the 0-456 main canal.

This script reruns the same Saint-Venant dispatch condition and records the
maximum hydraulic depth at every main-canal node during the simulation. The
envelope is checked against h_safe = 0.90D and the channel depth D.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

import muskingum_cunge_stage1 as stage1
import stage7_saint_venant_fig2_revised as sv


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"


def simulate_envelope(cfg: sv.Config):
    nodes, params, neighbors, ids, x, dx_cell, sections, bed, diversion_indices = sv.build_grid()
    # Reuse the full simulation implementation by adding a compact local copy of
    # the state loop that tracks hmax at every cell.
    n = len(ids)
    a = [sv.area_from_depth(cfg.initial_depth_m, sec) for sec in sections]
    q = [0.0 for _ in range(n)]
    min_area = [sv.area_from_depth(cfg.min_depth_m, sec) for sec in sections]
    supplied = {node: 0.0 for node in sv.DIVERSION_NODES}
    close_time = {node: None for node in sv.DIVERSION_NODES}
    qdiv_current = {node: 0.0 for node in sv.DIVERSION_NODES}
    limits = sv.branch_limits(nodes, params, neighbors, cfg)

    hmax = [cfg.initial_depth_m for _ in range(n)]
    hmax_time = [0.0 for _ in range(n)]
    total_steps = int(round(cfg.duration_hours * 3600.0 / cfg.dt_seconds))

    for step in range(total_steps + 1):
        t_s = step * cfg.dt_seconds
        for i in range(n):
            h_now = sv.depth_from_area(max(a[i], min_area[i]), sections[i])
            if h_now > hmax[i]:
                hmax[i] = h_now
                hmax_time[i] = t_s / 3600.0
        if step == total_steps:
            break

        q_in = sv.head_inflow(t_s, cfg)
        h_left = sv.normal_depth(q_in, sections[0], sv.local_slope(0, x, bed, cfg), cfg)
        a_left = sv.area_from_depth(max(h_left, cfg.min_depth_m), sections[0])
        q_left = q_in

        q_down = max(q[-1], 0.0)
        h_right = sv.normal_depth(q_down, sections[-1], sv.local_slope(n - 1, x, bed, cfg), cfg)
        a_right = sv.area_from_depth(max(h_right, cfg.min_depth_m), sections[-1])
        q_right = q[-1]

        fluxes = [sv.hll_flux(a_left, q_left, sections[0], a[0], q[0], sections[0])]
        for i in range(n - 1):
            fluxes.append(sv.hll_flux(a[i], q[i], sections[i], a[i + 1], q[i + 1], sections[i + 1]))
        fluxes.append(sv.hll_flux(a[-1], q[-1], sections[-1], a_right, q_right, sections[-1]))

        new_a = a[:]
        new_q = q[:]
        qdiv_current = {node: 0.0 for node in sv.DIVERSION_NODES}

        for i in range(n):
            dx = max(dx_cell[i], 1.0)
            da = -(fluxes[i + 1][0] - fluxes[i][0]) / dx
            dq = -(fluxes[i + 1][1] - fluxes[i][1]) / dx
            dq += sv.G * max(a[i], min_area[i]) * sv.local_slope(i, x, bed, cfg)
            new_a[i] = a[i] + cfg.dt_seconds * da
            new_q[i] = q[i] + cfg.dt_seconds * dq

        for node, idx in diversion_indices.items():
            if close_time[node] is not None:
                continue
            spec = sv.SPECS[node]
            sec = sections[idx]
            h_local = sv.depth_from_area(max(new_a[idx], min_area[idx]), sec)
            factor = sv.diversion_factor(h_local, sec.depth)
            remaining = max(spec.demand_m3 - supplied[node], 0.0)
            storage_above_min = max(new_a[idx] - min_area[idx], 0.0) * dx_cell[idx] / cfg.dt_seconds
            q_capacity = min(spec.max_flow_m3s, limits[node].safe_capacity_m3s) * factor
            qdiv = min(q_capacity, remaining / cfg.dt_seconds, storage_above_min)
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
            cf = sv.friction_coefficient(max(new_a[i], min_area[i]), q[i], sections[i])
            denom = 1.0 + cfg.dt_seconds * sv.G * max(new_a[i], min_area[i]) * cf
            new_q[i] = new_q[i] / max(denom, 1.0e-9)
            if abs(new_q[i]) < 1.0e-7:
                new_q[i] = 0.0
            c = sv.wave_speed(new_a[i], sections[i])
            q_limit = 8.0 * new_a[i] * max(c, 0.1)
            new_q[i] = max(min(new_q[i], q_limit), -q_limit)
        a, q = new_a, new_q

    dist_km = [v / 1000.0 for v in x]
    depth_limit = [sec.depth for sec in sections]
    safe_depth = [cfg.safe_depth_ratio * sec.depth for sec in sections]
    return {
        "ids": ids,
        "dist_km": dist_km,
        "hmax": hmax,
        "hmax_time": hmax_time,
        "safe_depth": safe_depth,
        "depth_limit": depth_limit,
        "close_h": close_time,
    }


def draw_dashed_line(d, p0, p1, fill, width=2, dash=14, gap=10):
    x0, y0 = p0
    x1, y1 = p1
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length <= 1.0e-9:
        return
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length
    dist = 0.0
    while dist < length:
        end = min(dist + dash, length)
        d.line((x0 + ux * dist, y0 + uy * dist, x0 + ux * end, y0 + uy * end), fill=fill, width=width)
        dist += dash + gap


def draw_fig6(result):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 1800, 1040
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = stage1.get_font(40, True)
    font = stage1.get_font(23)
    small = stage1.get_font(19)
    d.text((80, 45), "图6  干渠沿程最大水深包络线与安全校核", font=title, fill="#162033")

    left, top, right, bottom = 170, 145, w - 105, h - 190
    max_x = max(result["dist_km"])
    y_max = max(max(result["hmax"]), max(result["depth_limit"])) * 1.12
    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)

    for i in range(7):
        x = left + (right - left) * i / 6
        xv = max_x * i / 6
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        d.text((x - 26, bottom + 14), f"{xv:.1f}", font=small, fill="#586579")
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        yv = y_max * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((55, y - 13), f"{yv:.2f}", font=small, fill="#586579")

    def xp(xv):
        return left + (right - left) * xv / max_x

    def yp(yv):
        return bottom - (bottom - top) * yv / max(y_max, 1.0e-12)

    h_pts = [(xp(x), yp(y)) for x, y in zip(result["dist_km"], result["hmax"])]
    safe_pts = [(xp(x), yp(y)) for x, y in zip(result["dist_km"], result["safe_depth"])]
    depth_pts = [(xp(x), yp(y)) for x, y in zip(result["dist_km"], result["depth_limit"])]
    d.line(h_pts, fill="#0B5CAD", width=4)
    # These are constant for 0-456, but drawn as polylines for extensibility.
    for p0, p1 in zip(safe_pts[:-1], safe_pts[1:]):
        draw_dashed_line(d, p0, p1, "#F59E0B", width=3)
    for p0, p1 in zip(depth_pts[:-1], depth_pts[1:]):
        draw_dashed_line(d, p0, p1, "#DC2626", width=3)

    # Mark diversion locations.
    node_to_idx = {node: i for i, node in enumerate(result["ids"])}
    for node in sv.DIVERSION_NODES:
        if node not in node_to_idx:
            continue
        idx = node_to_idx[node]
        x = xp(result["dist_km"][idx])
        d.line((x, top, x, bottom), fill="#D0D5DD", width=1)
        d.text((x - 22, bottom + 42), str(node), font=small, fill="#586579")

    # Highlight exceedances if any.
    exceed_safe = [
        i for i, (hm, hs) in enumerate(zip(result["hmax"], result["safe_depth"])) if hm > hs
    ]
    exceed_design = [
        i for i, (hm, hd) in enumerate(zip(result["hmax"], result["depth_limit"])) if hm > hd
    ]
    for i in exceed_safe:
        x, y = xp(result["dist_km"][i]), yp(result["hmax"][i])
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#F59E0B")
    for i in exceed_design:
        x, y = xp(result["dist_km"][i]), yp(result["hmax"][i])
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#DC2626")

    max_i = max(range(len(result["hmax"])), key=lambda i: result["hmax"][i])
    x_m = xp(result["dist_km"][max_i])
    y_m = yp(result["hmax"][max_i])
    d.ellipse((x_m - 7, y_m - 7, x_m + 7, y_m + 7), fill="#0B5CAD", outline="white", width=2)
    d.text((x_m + 12, y_m - 32), f"最大 {result['hmax'][max_i]:.2f} m，节点{result['ids'][max_i]}", font=small, fill="#0B5CAD")

    legend_y = h - 125
    d.line((left, legend_y, left + 55, legend_y), fill="#0B5CAD", width=5)
    d.text((left + 70, legend_y - 13), "最大水深包络线", font=small, fill="#263241")
    draw_dashed_line(d, (left + 300, legend_y), (left + 360, legend_y), "#F59E0B", width=3)
    d.text((left + 375, legend_y - 13), "安全水深", font=small, fill="#263241")
    draw_dashed_line(d, (left + 520, legend_y), (left + 580, legend_y), "#DC2626", width=3)
    d.text((left + 595, legend_y - 13), "渠深上限", font=small, fill="#263241")

    d.text(((left + right) / 2 - 110, h - 72), "沿程距离 x (km)", font=font, fill="#263241")
    d.text((35, (top + bottom) / 2 - 20), "水深 h (m)", font=font, fill="#263241")
    status = "未发现超过安全水深或渠深上限的节点。" if not exceed_safe and not exceed_design else "图中橙/红色点分别表示超过安全水深/渠深上限的节点。"
    d.text((80, h - 36), f"说明：包络线表示各干渠节点在整个模拟时段内达到的最大水深；灰色竖线为分水口位置。{status}", font=small, fill="#586579")

    out = FIG_DIR / "fig6_max_depth_envelope.png"
    img.save(out, quality=95)
    return out, exceed_safe, exceed_design


def save_outputs(result, fig_path, exceed_safe, exceed_design):
    csv_path = OUT_DIR / "fig6_max_depth_envelope.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["node", "distance_km", "hmax_m", "hmax_time_h", "safe_depth_m", "design_depth_m", "exceeds_safe", "exceeds_design"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, node in enumerate(result["ids"]):
            writer.writerow(
                {
                    "node": node,
                    "distance_km": f"{result['dist_km'][i]:.6f}",
                    "hmax_m": f"{result['hmax'][i]:.6f}",
                    "hmax_time_h": f"{result['hmax_time'][i]:.6f}",
                    "safe_depth_m": f"{result['safe_depth'][i]:.6f}",
                    "design_depth_m": f"{result['depth_limit'][i]:.6f}",
                    "exceeds_safe": i in exceed_safe,
                    "exceeds_design": i in exceed_design,
                }
            )
    max_i = max(range(len(result["hmax"])), key=lambda i: result["hmax"][i])
    summary = {
        "figure": str(fig_path),
        "csv": str(csv_path),
        "safe_depth_rule": "h_safe=0.90D",
        "max_depth": {
            "node": result["ids"][max_i],
            "distance_km": result["dist_km"][max_i],
            "hmax_m": result["hmax"][max_i],
            "time_h": result["hmax_time"][max_i],
        },
        "num_exceed_safe_nodes": len(exceed_safe),
        "num_exceed_design_nodes": len(exceed_design),
    }
    (OUT_DIR / "fig6_max_depth_envelope_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    result = simulate_envelope(sv.Config())
    fig_path, exceed_safe, exceed_design = draw_fig6(result)
    save_outputs(result, fig_path, exceed_safe, exceed_design)


if __name__ == "__main__":
    main()
