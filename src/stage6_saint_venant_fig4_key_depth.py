#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: dynamic water-depth response at key main-canal nodes.

The depth series is recorded from the same Saint-Venant forward simulation used
for Figures 2 and 3. It shows how the head inflow wave, lateral diversions and
closure operations jointly change local hydraulic depth along the 0-456 reach.
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

KEY_NODES = [0, 71, 89, 194, 287, 383, 456]
NODE_LABELS = {
    0: "渠首0",
    71: "71口",
    89: "89口",
    194: "194口",
    287: "287口",
    383: "383口",
    456: "渠尾456",
}
COLORS = {
    0: "#111827",
    71: "#0B5CAD",
    89: "#D1495B",
    194: "#E09F3E",
    287: "#7B2CBF",
    383: "#C75146",
    456: "#2A9D8F",
}


def draw_fig4(result):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 1800, 1080
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    title = stage1.get_font(40, True)
    font = stage1.get_font(23)
    small = stage1.get_font(19)
    d.text((80, 45), "图4  关键节点水深动态变化过程", font=title, fill="#162033")

    left, top, right, bottom = 170, 140, w - 110, h - 195
    times = result["times_h"]
    max_t = max(times)
    depth_values = []
    for node in KEY_NODES:
        depth_values.extend(result["depth"][node])
    y_min = 0.0
    y_max = max(depth_values) * 1.10
    y_max = max(y_max, 0.5)

    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(8):
        x = left + (right - left) * i / 7
        tv = max_t * i / 7
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        d.text((x - 26, bottom + 16), f"{tv:.1f}", font=small, fill="#586579")
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        vv = y_min + (y_max - y_min) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((56, y - 13), f"{vv:.2f}", font=small, fill="#586579")

    def xp(t):
        return left + (right - left) * t / max_t

    def yp(v):
        return bottom - (bottom - top) * (v - y_min) / max(y_max - y_min, 1.0e-12)

    # Mark diversion closure moments as faint vertical guide lines.
    for node in sv.DIVERSION_NODES:
        close_h = result["close_h"][node]
        if close_h is None:
            continue
        x = xp(close_h)
        yy = top
        while yy < bottom:
            d.line((x, yy, x, min(yy + 8, bottom)), fill="#D0D5DD", width=1)
            yy += 18

    for node in KEY_NODES:
        vals = result["depth"][node]
        pts = [(xp(t), yp(v)) for t, v in zip(times, vals)]
        d.line(pts, fill=COLORS[node], width=4)
        last_x, last_y = pts[-1]
        d.ellipse((last_x - 5, last_y - 5, last_x + 5, last_y + 5), fill=COLORS[node], outline="white", width=2)

    legend_x, legend_y = left, h - 132
    for node in KEY_NODES:
        d.line((legend_x, legend_y, legend_x + 42, legend_y), fill=COLORS[node], width=5)
        d.text((legend_x + 52, legend_y - 13), NODE_LABELS[node], font=small, fill="#263241")
        legend_x += 190
        if legend_x > right - 160:
            legend_x = left
            legend_y += 34

    d.text(((left + right) / 2 - 80, h - 80), "时间 t (h)", font=font, fill="#263241")
    d.text((34, (top + bottom) / 2 - 18), "水深 h (m)", font=font, fill="#263241")
    d.text(
        (80, h - 38),
        "说明：水深由 Saint-Venant 正演计算断面面积 A 后按真实梯形断面反算得到；淡灰色虚线为各配水口满足需水后的关闸时刻。",
        font=small,
        fill="#586579",
    )

    out = FIG_DIR / "fig4_key_node_depth_saint_venant.png"
    img.save(out, quality=95)
    return out


def save_csv_and_summary(result, fig_path: Path):
    csv_path = OUT_DIR / "stage6_fig4_key_node_depth.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["time_h"] + [f"depth_{node}_m" for node in KEY_NODES] + [f"water_level_{node}_m" for node in KEY_NODES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(result["times_h"]):
            row = {"time_h": f"{t:.6f}"}
            for node in KEY_NODES:
                row[f"depth_{node}_m"] = f"{result['depth'][node][i]:.6f}"
                row[f"water_level_{node}_m"] = f"{result['water_level'][node][i]:.6f}"
            writer.writerow(row)

    summary = {
        "figure": str(fig_path),
        "csv": str(csv_path),
        "model": "1-D Saint-Venant finite-volume HLL forward model with semi-implicit Manning friction",
        "plotted_variable": "hydraulic depth h at key nodes",
        "unit": "m",
        "key_nodes": KEY_NODES,
        "note": "water level Z can be obtained as bed elevation plus depth; both series are saved in the CSV.",
    }
    (OUT_DIR / "stage6_fig4_key_node_depth_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    result = sv.simulate(sv.Config())
    fig_path = draw_fig4(result)
    save_csv_and_summary(result, fig_path)


if __name__ == "__main__":
    main()
