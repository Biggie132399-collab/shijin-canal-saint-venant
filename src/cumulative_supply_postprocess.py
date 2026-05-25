#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cumulative-supply postprocessing for the Saint-Venant dispatch result.

Figure 3 shows cumulative supplied water volume at each diversion outlet:

    W_k(t) = ∫ Q_div,k(t) dt

The discharge time series comes from dispatch.py, i.e. the configured
Saint-Venant dispatch solver.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

import muskingum_cunge_stage1 as stage1
import dispatch as sv
import dispatch_postprocess as dispatch_plot


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"


def load_or_run_result():
    cfg = sv.load_config()
    result = sv.simulate(cfg)
    return cfg, result


def draw_fig3(result, out_path: Path | None = None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 1800, 1120
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title = stage1.get_font(40, True)
    font = stage1.get_font(23)
    small = stage1.get_font(19)
    d.text((80, 45), "图3  各配水口累计供水量变化过程", font=title, fill="#162033")

    left, top, right, bottom = 170, 150, w - 300, h - 250
    times = result["times_h"]
    max_t = max(times)
    max_w = max(spec.demand_m3 for spec in sv.SPECS.values()) / 1e4 * 1.10

    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(6):
        x = left + (right - left) * i / 5
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        d.text((x - 28, bottom + 16), f"{max_t*i/5:.1f}", font=small, fill="#586579")
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((48, y - 13), f"{max_w*i/5:.1f}", font=small, fill="#586579")

    def xp(t):
        return left + (right - left) * t / max_t

    def yp(volume_1e4):
        return bottom - (bottom - top) * volume_1e4 / max_w

    label_offsets = {
        89: (14, -18),
        71: (14, 8),
        287: (14, -18),
        194: (14, 8),
        150: (14, -24),
        349: (14, 2),
        383: (14, 28),
    }

    for node in sv.DIVERSION_NODES:
        vals = [v / 1e4 for v in result["supplied"][node]]
        pts = [(xp(t), yp(v)) for t, v in zip(times, vals)]
        d.line(pts, fill=dispatch_plot.COLORS[node], width=4)

        close_h = result["close_h"][node]
        demand = sv.SPECS[node].demand_m3 / 1e4
        if close_h is not None:
            x_close = xp(close_h)
            y_close = yp(demand)
            # Inflection point marker: exact completion/closure moment.
            d.ellipse((x_close - 7, y_close - 7, x_close + 7, y_close + 7), fill=dispatch_plot.COLORS[node], outline="white", width=2)
            d.line((x_close, y_close, right, y_close), fill=dispatch_plot.COLORS[node], width=1)
            ox, oy = label_offsets.get(node, (14, 0))
            d.text((right + ox, y_close + oy), f"{node}口  {demand:.1f}×10⁴ m³", font=small, fill=dispatch_plot.COLORS[node])

    draw_local_inset(d, result, (left + 64, top + 44, left + 610, top + 382))

    legend_x, legend_y = left, h - 146
    for node in sv.DIVERSION_NODES:
        d.line((legend_x, legend_y, legend_x + 42, legend_y), fill=dispatch_plot.COLORS[node], width=5)
        d.text((legend_x + 52, legend_y - 13), f"{node}口", font=small, fill="#263241")
        legend_x += 160
        if legend_x > right - 160:
            legend_x = left
            legend_y += 32

    d.text(((left + right) / 2 - 80, h - 98), "时间 t (h)", font=font, fill="#263241")
    d.text((28, (top + bottom) / 2 - 18), "累计供水量 W (10⁴ m³)", font=font, fill="#263241")
    note1 = "注：累计供水量由 Saint-Venant 动力波正演实际出流过程数值积分获得；曲线滞发起点反映水流传播延迟，曲线变平表示达需水目标并关闸。"
    note2 = "由于积分平滑效应会掩盖局部涌波/水锤扰动，后续引入残差 Delta W(t) = W_actual(t) - W_ref(t) 进行剥离与量化分析。"
    d.text((80, h - 63), note1, font=small, fill="#586579")
    d.text((80, h - 35), note2, font=small, fill="#586579")
    out = out_path or FIG_DIR / "fig3_cumulative_supply_saint_venant.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    return out


def linear_fit(points):
    n = len(points)
    sx = sum(t for t, _ in points)
    sy = sum(v for _, v in points)
    sxx = sum(t * t for t, _ in points)
    sxy = sum(t * v for t, v in points)
    slope = (n * sxy - sx * sy) / max(n * sxx - sx * sx, 1.0e-12)
    intercept = (sy - slope * sx) / n
    return slope, intercept


def draw_dashed_line(d, p0, p1, fill, width=3, dash=11, gap=7):
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


def draw_polyline_dashed(d, pts, fill, width=3):
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw_dashed_line(d, p0, p1, fill, width=width)


def draw_local_inset(d, result, rect):
    """Inset for outlet 89 around the outlet-71 closure disturbance."""
    ix0, iy0, ix1, iy1 = rect
    title_font = stage1.get_font(21, True)
    tick_font = stage1.get_font(16)
    note_font = stage1.get_font(17)
    d.rectangle(rect, fill="white", outline="#667085", width=2)

    plot_left, plot_top = ix0 + 68, iy0 + 74
    plot_right, plot_bottom = ix1 - 28, iy1 - 58
    t_min, t_max = 4.0, 4.5
    close_71 = result["close_h"][71]
    times = result["times_h"]
    actual = result["supplied"][89]
    fit_points = [(t, w) for t, w in zip(times, actual) if t_min <= t <= close_71]
    slope, intercept = linear_fit(fit_points)
    window = [(t, w, slope * t + intercept) for t, w in zip(times, actual) if t_min <= t <= t_max]
    residual = [(t, w - r) for t, w, r in window]
    y_min = min(0.0, min(v for _, v in residual))
    y_max = max(v for _, v in residual)
    pad = max((y_max - y_min) * 0.18, 50.0)
    y_min -= pad
    y_max += pad

    def ix(t):
        return plot_left + (plot_right - plot_left) * (t - t_min) / (t_max - t_min)

    def iy(v):
        return plot_bottom - (plot_bottom - plot_top) * (v - y_min) / max(y_max - y_min, 1.0e-12)

    for i in range(4):
        x = plot_left + (plot_right - plot_left) * i / 3
        tv = t_min + (t_max - t_min) * i / 3
        d.line((x, plot_top, x, plot_bottom), fill="#EEF2F7", width=1)
        d.text((x - 20, plot_bottom + 10), f"{tv:.2f}", font=tick_font, fill="#667085")
    for i in range(4):
        y = plot_bottom - (plot_bottom - plot_top) * i / 3
        vv = y_min + (y_max - y_min) * i / 3
        d.line((plot_left, y, plot_right, y), fill="#EEF2F7", width=1)
        d.text((ix0 + 8, y - 11), f"{vv:.0f}", font=tick_font, fill="#667085")
    d.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#AAB4C2", width=1)

    actual_pts = [(ix(t), iy(v)) for t, v in residual]
    ref_pts = [(ix(t), iy(0.0)) for t, _ in residual]
    draw_polyline_dashed(d, ref_pts, "#64748B", width=4)
    d.line(actual_pts, fill=dispatch_plot.COLORS[89], width=5)

    x_close = ix(close_71)
    y = plot_top
    while y < plot_bottom:
        d.line((x_close, y, x_close, min(y + 8, plot_bottom)), fill=dispatch_plot.COLORS[71], width=2)
        y += 16

    # Mark the maximum separation in the inset window.
    max_row = max(residual, key=lambda row: row[1])
    x_m = ix(max_row[0])
    y_m = iy(max_row[1])
    d.ellipse((x_m - 6, y_m - 6, x_m + 6, y_m + 6), fill=dispatch_plot.COLORS[89], outline="white", width=2)
    d.line((x_m, y_m, x_m + 52, y_m - 36), fill="#475467", width=1)
    d.text((x_m + 56, y_m - 50), f"偏离 {max_row[1]:.0f} m³", font=tick_font, fill="#475467")

    d.text((ix0 + 16, iy0 + 14), "局部放大：89口残差响应", font=title_font, fill="#162033")
    d.text((ix0 + 16, iy0 + 42), "Delta W89 = 实际累计 - 关闸前趋势，单位 m³", font=tick_font, fill="#667085")
    d.line((ix0 + 18, iy1 - 30, ix0 + 54, iy1 - 30), fill=dispatch_plot.COLORS[89], width=5)
    d.text((ix0 + 62, iy1 - 42), "实际残差", font=note_font, fill="#263241")
    draw_dashed_line(d, (ix0 + 230, iy1 - 30), (ix0 + 266, iy1 - 30), "#64748B", width=4)
    d.text((ix0 + 274, iy1 - 42), "趋势基准", font=note_font, fill="#263241")


def save_csv_and_summary(result, fig_path: Path):
    csv_path = OUT_DIR / "cumulative_supply_timeseries.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["time_h"] + [f"W_{n}_m3" for n in sv.DIVERSION_NODES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(result["times_h"]):
            row = {"time_h": f"{t:.6f}"}
            for n in sv.DIVERSION_NODES:
                row[f"W_{n}_m3"] = f"{result['supplied'][n][i]:.3f}"
            writer.writerow(row)
    summary = {
        "figure": str(fig_path),
        "csv": str(csv_path),
        "formula": "W_k(t)=integral_0^t Q_div,k(tau) d tau",
        "unit": "m3; plotted as 1e4 m3",
        "source": "dispatch.py finite-volume dispatch simulation",
        "completion": {
            str(n): {
                "demand_m3": sv.SPECS[n].demand_m3,
                "close_time_h": result["close_h"][n],
                "final_supplied_m3": result["supplied"][n][-1],
            }
            for n in sv.DIVERSION_NODES
        },
    }
    (OUT_DIR / "cumulative_supply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    _, result = load_or_run_result()
    fig_path = draw_fig3(result)
    save_csv_and_summary(result, fig_path)


if __name__ == "__main__":
    main()
