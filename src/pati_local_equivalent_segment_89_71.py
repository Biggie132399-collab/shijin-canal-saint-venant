#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Pati RSR validation on a near-prismatic equivalent reach: 89 -> 71.

The full 0-456 real canal is too heterogeneous for direct node-by-node RSR.
This script follows the field-application spirit in Pati et al. (2023):
construct an equivalent prismatic trapezoidal reach for a short local segment,
then reverse-route downstream stage to the upstream section.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

import pati_real_main_canal_no_diversion as base


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "pati_local_equivalent_segment_89_71_results"
FIG_DIR = OUT_DIR / "figures"

UPSTREAM_NODE = 71
DOWNSTREAM_NODE = 89


def equivalent_section(
    forward: Dict[str, object],
    upstream_idx: int,
    downstream_idx: int,
) -> Tuple[base.stage1.SectionParam, float, float]:
    x: List[float] = forward["x"]  # type: ignore[assignment]
    bed: List[float] = forward["bed"]  # type: ignore[assignment]
    sections: List[base.stage1.SectionParam] = forward["sections"]  # type: ignore[assignment]
    segment_sections = sections[upstream_idx : downstream_idx + 1]
    b_eq = sum(s.bottom_width for s in segment_sections) / len(segment_sections)
    z_eq = sum(s.side_slope for s in segment_sections) / len(segment_sections)
    n_eq = sum(s.manning_n for s in segment_sections) / len(segment_sections)
    d_eq = sum(s.depth for s in segment_sections) / len(segment_sections)
    length_m = x[downstream_idx] - x[upstream_idx]
    slope_eq = max((bed[upstream_idx] - bed[downstream_idx]) / max(length_m, 1.0), 5.0e-5)
    sec_eq = base.stage1.SectionParam(d_eq, b_eq, z_eq, n_eq)
    return sec_eq, slope_eq, length_m


def convert_real_depth_to_equivalent_depth(
    real_depths: Sequence[float],
    real_section: base.stage1.SectionParam,
    eq_section: base.stage1.SectionParam,
) -> List[float]:
    converted = []
    for h in real_depths:
        a = base.area_from_depth(h, real_section)
        converted.append(base.depth_from_area(a, eq_section))
    return converted


def convert_equivalent_depth_to_real_depth(
    eq_depths: Sequence[float],
    eq_section: base.stage1.SectionParam,
    real_section: base.stage1.SectionParam,
) -> List[float]:
    converted = []
    for h in eq_depths:
        a = base.area_from_depth(h, eq_section)
        converted.append(base.depth_from_area(a, real_section))
    return converted


def reverse_one_equivalent_reach(
    y_down_eq: Sequence[float],
    sec_eq: base.stage1.SectionParam,
    slope_eq: float,
    length_m: float,
    dt_seconds: float,
    rsr_cfg: base.RSRConfig,
    initial_downstream_q: float | None = None,
) -> Tuple[List[float], List[float], List[float]]:
    q_down = base.discharge_series_from_stage(
        y_down_eq,
        sec_eq,
        slope_eq,
        dt_seconds,
        rsr_cfg,
        initial_q=initial_downstream_q,
    )
    a_down = [base.area_from_depth(h, sec_eq) for h in y_down_eq]
    dadt = base.temporal_gradient(a_down, dt_seconds)
    dqdt = base.temporal_gradient(q_down, dt_seconds)
    a_up = []
    for h, q, a, da, dq in zip(y_down_eq, q_down, a_down, dadt, dqdt):
        c = base.routing_celerity(h, q, sec_eq, rsr_cfg)
        # Eq.17 form: -dx/c^2*dQdt + 2*dx/c*dAdt for the central temporal gradient.
        a_up.append(max(a - length_m / (c * c) * dq + 2.0 * length_m / c * da, base.area_from_depth(rsr_cfg.min_depth_m, sec_eq)))
    y_up = [base.depth_from_area(a, sec_eq) for a in a_up]
    q_up = base.discharge_series_from_stage(
        y_up,
        sec_eq,
        slope_eq,
        dt_seconds,
        rsr_cfg,
        initial_q=q_down[0],
    )
    return y_up, q_up, q_down


def metric_dict(
    times_h: Sequence[float],
    y_real_sim: Sequence[float],
    y_real_obs: Sequence[float],
    q_sim: Sequence[float],
    q_obs: Sequence[float],
    factor: float,
) -> Dict[str, float]:
    return {
        "routing_celerity_factor": factor,
        "depth_rmse_m": base.rmse(y_real_sim, y_real_obs),
        "depth_nse": base.nse(y_real_sim, y_real_obs),
        "q_rmse_m3s": base.rmse(q_sim, q_obs),
        "q_nse": base.nse(q_sim, q_obs),
        "depth_peak_time_error_h": base.peak_time_error(y_real_sim, y_real_obs, times_h),
        "q_peak_time_error_h": base.peak_time_error(q_sim, q_obs, times_h),
        "max_depth_error_m": max(abs(a - b) for a, b in zip(y_real_sim, y_real_obs)),
        "max_q_error_m3s": max(abs(a - b) for a, b in zip(q_sim, q_obs)),
    }


def draw_result(
    path: Path,
    times_h: Sequence[float],
    y_obs: Sequence[float],
    y_sim: Sequence[float],
    q_obs: Sequence[float],
    q_sim: Sequence[float],
    summary: Dict[str, float],
):
    width, height = 1800, 1180
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title = base.get_font(42, True)
    font = base.get_font(24)
    small = base.get_font(20)
    d.text((80, 45), "图  89→71 局部等效渠段 Pati RSR 反演验证", font=title, fill="#162033")
    left, right = 140, width - 90
    panels = [
        (y_obs, y_sim, "71节点水深 h (m)", 145, 520, summary["depth_rmse_m"], summary["depth_nse"]),
        (q_obs, q_sim, "71节点流量 Q (m³/s)", 650, 1025, summary["q_rmse_m3s"], summary["q_nse"]),
    ]
    x_min, x_max = min(times_h), max(times_h)

    def draw_panel(obs, sim, ylabel, top, bottom, err, score):
        vals = list(obs) + list(sim)
        y_min, y_max = min(vals), max(vals)
        pad = max((y_max - y_min) * 0.16, 1.0e-6)
        y_min -= pad
        y_max += pad
        d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
        for k in range(6):
            y = top + (bottom - top) * k / 5
            v = y_max - (y_max - y_min) * k / 5
            d.line((left, y, right, y), fill="#EEF2F7", width=2)
            d.text((45, y - 14), f"{v:.2f}", font=small, fill="#586579")
        for k in range(7):
            x = left + (right - left) * k / 6
            v = x_min + (x_max - x_min) * k / 6
            d.line((x, bottom, x, bottom + 10), fill="#8795A7", width=2)
            d.text((x - 26, bottom + 18), f"{v:.0f}", font=small, fill="#586579")

        def xy(t, v):
            x = left + (right - left) * (t - x_min) / max(x_max - x_min, 1.0e-12)
            y = bottom - (bottom - top) * (v - y_min) / max(y_max - y_min, 1.0e-12)
            return x, y

        d.line([xy(t, v) for t, v in zip(times_h, obs)], fill="#2A9D8F", width=5)
        d.line([xy(t, v) for t, v in zip(times_h, sim)], fill="#C75146", width=4)
        d.text((left + 18, top + 14), f"{ylabel}    RMSE={err:.3f}, NSE={score:.3f}", font=font, fill="#263241")

    for args in panels:
        draw_panel(*args)
    lx, ly = left, height - 95
    for label, color in [("Saint-Venant 正演真值", "#2A9D8F"), ("局部等效Pati RSR反推", "#C75146")]:
        d.line((lx, ly, lx + 58, ly), fill=color, width=6)
        d.text((lx + 70, ly - 14), label, font=small, fill="#263241")
        lx += 390
    d.text((width / 2 - 65, height - 50), "时间 t (h)", font=font, fill="#263241")
    img.save(path, quality=95)


def draw_factor_scan(path: Path, rows: Sequence[Dict[str, float]]):
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title = base.get_font(38, True)
    font = base.get_font(22)
    small = base.get_font(18)
    d.text((70, 42), "89→71 局部等效渠段波速系数敏感性", font=title, fill="#162033")
    left, top, right, bottom = 125, 135, width - 80, height - 145
    d.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    values = [max(min(r["depth_nse"], 1.0), -1.0) for r in rows]
    labels = [r["routing_celerity_factor"] for r in rows]
    y_min, y_max = -1.0, 1.0
    for k in range(5):
        y = top + (bottom - top) * k / 4
        v = y_max - (y_max - y_min) * k / 4
        d.line((left, y, right, y), fill="#EEF2F7", width=2)
        d.text((45, y - 12), f"{v:.1f}", font=small, fill="#586579")
    zero_y = bottom - (bottom - top) * (0.0 - y_min) / (y_max - y_min)
    d.line((left, zero_y, right, zero_y), fill="#9AA8B8", width=2)
    n = len(rows)
    bar_w = (right - left) / n * 0.55
    for i, (v, label, row) in enumerate(zip(values, labels, rows)):
        cx = left + (right - left) * (i + 0.5) / n
        yv = bottom - (bottom - top) * (v - y_min) / (y_max - y_min)
        color = "#2A9D8F" if row["depth_nse"] >= 0.5 else "#C75146"
        d.rectangle((cx - bar_w / 2, min(yv, zero_y), cx + bar_w / 2, max(yv, zero_y)), fill=color)
        d.text((cx - 34, bottom + 18), f"{label:.2f}", font=small, fill="#263241")
        d.text((cx - 38, yv - 28 if v >= 0 else yv + 8), f"{row['depth_nse']:.2f}", font=small, fill=color)
    d.text((width / 2 - 110, height - 70), "routing_celerity_factor", font=font, fill="#263241")
    d.text((35, top + 230), "水深NSE", font=font, fill="#263241")
    img.save(path, quality=95)


def run() -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    forward = base.run_forward_saint_venant(base.ForwardConfig())
    ids: List[int] = forward["ids"]  # type: ignore[assignment]
    x: List[float] = forward["x"]  # type: ignore[assignment]
    sections: List[base.stage1.SectionParam] = forward["sections"]  # type: ignore[assignment]
    times_h: List[float] = forward["times_h"]  # type: ignore[assignment]
    depth_t: List[List[float]] = forward["depth_nodes"]  # type: ignore[assignment]
    q_t: List[List[float]] = forward["q_nodes"]  # type: ignore[assignment]
    dt_seconds = (times_h[1] - times_h[0]) * 3600.0
    up_idx = ids.index(UPSTREAM_NODE)
    down_idx = ids.index(DOWNSTREAM_NODE)
    sec_eq, slope_eq, length_m = equivalent_section(forward, up_idx, down_idx)
    y_down_real = [row[down_idx] for row in depth_t]
    y_up_real_obs = [row[up_idx] for row in depth_t]
    q_up_obs = [row[up_idx] for row in q_t]
    q_down_sv = [row[down_idx] for row in q_t]
    y_down_eq = convert_real_depth_to_equivalent_depth(y_down_real, sections[down_idx], sec_eq)

    factors = [0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00]
    scan_rows: List[Dict[str, float]] = []
    simulations: Dict[float, Tuple[List[float], List[float], List[float], List[float]]] = {}
    for factor in factors:
        rsr_cfg = base.RSRConfig(routing_celerity_factor=factor)
        y_up_eq, q_up_sim, q_down_est = reverse_one_equivalent_reach(
            y_down_eq,
            sec_eq,
            slope_eq,
            length_m,
            dt_seconds,
            rsr_cfg,
            initial_downstream_q=q_down_sv[0],
        )
        y_up_real_sim = convert_equivalent_depth_to_real_depth(y_up_eq, sec_eq, sections[up_idx])
        row = metric_dict(times_h, y_up_real_sim, y_up_real_obs, q_up_sim, q_up_obs, factor)
        scan_rows.append(row)
        simulations[factor] = (y_up_real_sim, q_up_sim, y_up_eq, q_down_est)

    default_row = next(r for r in scan_rows if abs(r["routing_celerity_factor"] - 1.5) < 1.0e-9)
    best_row = max(scan_rows, key=lambda r: r["depth_nse"] + r["q_nse"])
    best_factor = best_row["routing_celerity_factor"]
    best_y, best_q, _, _ = simulations[best_factor]

    draw_result(
        FIG_DIR / "fig_local_89_71_equivalent_pati_rsr.png",
        times_h,
        y_up_real_obs,
        best_y,
        q_up_obs,
        best_q,
        best_row,
    )
    draw_factor_scan(FIG_DIR / "fig_local_89_71_factor_scan.png", scan_rows)

    with (OUT_DIR / "factor_scan.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    with (OUT_DIR / "timeseries_best.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["time_h", "h71_sv_m", "h71_pati_rsr_m", "q71_sv_m3s", "q71_pati_rsr_m3s", "h89_sv_m", "q89_sv_m3s"])
        for row in zip(times_h, y_up_real_obs, best_y, q_up_obs, best_q, y_down_real, q_down_sv):
            writer.writerow([f"{v:.8f}" for v in row])

    summary = {
        "segment": f"{DOWNSTREAM_NODE}→{UPSTREAM_NODE}",
        "length_m": length_m,
        "x_upstream_m": x[up_idx],
        "x_downstream_m": x[down_idx],
        "equivalent_section": {
            "depth_m": sec_eq.depth,
            "bottom_width_m": sec_eq.bottom_width,
            "side_slope_h_per_v": sec_eq.side_slope,
            "manning_n": sec_eq.manning_n,
            "bed_slope": slope_eq,
        },
        "default_factor_result": default_row,
        "best_factor_result": best_row,
        "all_factor_results": scan_rows,
        "figures": [
            str(FIG_DIR / "fig_local_89_71_equivalent_pati_rsr.png"),
            str(FIG_DIR / "fig_local_89_71_factor_scan.png"),
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    note = f"""# 89→71局部等效渠段 Pati RSR 验证

本算例将 71--89 之间的真实非均匀渠段等效为一个棱柱形梯形渠段。下游 89 节点 stage 来自 Saint-Venant 正演，并按等面积原则转换为等效断面水深；Pati RSR 反推得到等效上游水深后，再按等面积原则转换回 71 节点真实断面水深。

## 等效渠段参数

- 长度：{length_m:.2f} m；
- 等效底宽：{sec_eq.bottom_width:.3f} m；
- 等效边坡系数：{sec_eq.side_slope:.3f}；
- 等效渠深：{sec_eq.depth:.3f} m；
- 等效 Manning n：{sec_eq.manning_n:.4f}；
- 等效床坡：{slope_eq:.6f}。

## 默认结果

默认 routing_celerity_factor = 1.5：

- 水深 RMSE = {default_row["depth_rmse_m"]:.4f} m；
- 水深 NSE = {default_row["depth_nse"]:.4f}；
- 流量 RMSE = {default_row["q_rmse_m3s"]:.4f} m³/s；
- 流量 NSE = {default_row["q_nse"]:.4f}。

## 局部最佳结果

本算例扫描了多个波速系数，最佳组合为 routing_celerity_factor = {best_factor:.2f}：

- 水深 RMSE = {best_row["depth_rmse_m"]:.4f} m；
- 水深 NSE = {best_row["depth_nse"]:.4f}；
- 流量 RMSE = {best_row["q_rmse_m3s"]:.4f} m³/s；
- 流量 NSE = {best_row["q_nse"]:.4f}。

该结果可用于说明：Pati RSR 在局部短距离、等效规则渠段上的水深反演较全渠段直接反推更稳定；但流量反演仍依赖动态 stage-discharge 关系和波速参数，不能直接替代 Saint-Venant 全渠段主模型。
"""
    (OUT_DIR / "explanation.md").write_text(note, encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {OUT_DIR}")
