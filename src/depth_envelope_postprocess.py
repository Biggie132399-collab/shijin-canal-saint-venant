#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximum water-depth envelope postprocessing for the 0-456 main canal.

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
import dispatch as sv


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"


def compute_depth_envelope(cfg: sv.Config):
    simulation = sv.simulate(cfg)
    _, _, _, main_grid, _, _ = sv.build_network(cfg)
    ids = simulation["main_ids"]
    dist_km = simulation["main_dist_km"]
    hmax = simulation["main_hmax"]
    hmax_time = simulation["main_hmax_time"]
    depth_limit = main_grid.depth.tolist()
    safe_depth = (cfg.safe_depth_ratio * main_grid.depth).tolist()
    return {
        "ids": ids,
        "dist_km": dist_km,
        "hmax": hmax,
        "hmax_time": hmax_time,
        "safe_depth": safe_depth,
        "depth_limit": depth_limit,
        "close_h": simulation["close_h"],
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
    result = compute_depth_envelope(sv.load_config())
    fig_path, exceed_safe, exceed_design = draw_fig6(result)
    save_outputs(result, fig_path, exceed_safe, exceed_design)


if __name__ == "__main__":
    main()
