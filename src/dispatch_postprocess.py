#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocessing helpers for Saint-Venant dispatch outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping

from PIL import Image, ImageDraw

import dispatch
import muskingum_cunge_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"

LARGE_PANEL_NODES = [71, 89, 287]
SMALL_PANEL_NODES = [150, 194, 349, 383]

COLORS = {
    71: "#0B5CAD",
    89: "#D1495B",
    150: "#2A9D8F",
    194: "#E09F3E",
    287: "#7B2CBF",
    349: "#6B7280",
    383: "#C75146",
}


def draw_dashed_vertical(d: ImageDraw.ImageDraw, x: float, y0: float, y1: float, color: str, width: int = 2) -> None:
    y = y0
    while y < y1:
        d.line((x, y, x, min(y + 9, y1)), fill=color, width=width)
        y += 18


def draw_panel(
    d: ImageDraw.ImageDraw,
    result: Mapping[str, object],
    nodes: list[int],
    rect: tuple[int, int, int, int],
    y_max: float,
    label: str,
    show_x_ticks: bool = True,
) -> None:
    left, top, right, bottom = rect
    small = stage1.get_font(19)
    font = stage1.get_font(22)
    times = result["times_h"]
    max_t = max(times)  # type: ignore[arg-type]
    d.rectangle(rect, outline="#CAD3DF", width=2)
    for i in range(6):
        x = left + (right - left) * i / 5
        d.line((x, top, x, bottom), fill="#EEF2F7", width=1)
        if show_x_ticks:
            d.text((x - 28, bottom + 14), f"{max_t*i/5:.1f}", font=small, fill="#586579")
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill="#EEF2F7", width=1)
        d.text((58, y - 12), f"{y_max*i/5:.1f}", font=small, fill="#586579")

    def xp(t: float) -> float:
        return left + (right - left) * t / max_t

    def yp(qv: float) -> float:
        return bottom - (bottom - top) * qv / y_max

    qdiv = result["qdiv"]  # type: ignore[assignment]
    close_h = result["close_h"]  # type: ignore[assignment]
    for node in nodes:
        vals = qdiv[node]
        node_close_h = close_h[node]
        pts = [(xp(t), yp(v)) for t, v in zip(times, vals) if node_close_h is None or t <= node_close_h]
        if len(pts) >= 2:
            d.line(pts, fill=COLORS[node], width=4)
        if node_close_h is not None:
            x_close = xp(node_close_h)
            y_top = yp(max(vals))
            draw_dashed_vertical(d, x_close, y_top, yp(0), COLORS[node])
            d.ellipse((x_close - 5, yp(0) - 5, x_close + 5, yp(0) + 5), fill=COLORS[node])
            d.text((x_close + 8, max(top + 4, y_top - 28)), f"{node}关 {node_close_h:.2f}h", font=small, fill=COLORS[node])

    d.text((left + 14, top + 10), label, font=font, fill="#263241")


def draw_diversion_outflow(result: Mapping[str, object], out_path: Path | None = None) -> Path:
    """Draw Figure 2, the actual outflow process at every diversion outlet."""

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

    qdiv = result["qdiv"]  # type: ignore[assignment]
    y_large = max(max(qdiv[n]) for n in LARGE_PANEL_NODES) * 1.12
    y_small = max(max(qdiv[n]) for n in SMALL_PANEL_NODES) * 1.25
    draw_panel(d, result, LARGE_PANEL_NODES, (left, top1, right, bottom1), max(y_large, 1.0), "大/中型配水口", show_x_ticks=False)
    draw_panel(d, result, SMALL_PANEL_NODES, (left, top2, right, bottom2), max(y_small, 1.0), "小型配水口", show_x_ticks=True)

    legend_x, legend_y = left, h - 125
    for node in dispatch.DIVERSION_NODES:
        spec = dispatch.SPECS[node]
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

    out = out_path or FIG_DIR / "fig2_diversion_outflow_process.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    return out


def write_dispatch_outputs(result: Mapping[str, object], fig_path: Path, out_dir: Path | None = None) -> Dict[str, Path]:
    """Write Figure 2 time series, outlet feasibility table, and summary JSON."""

    target_dir = out_dir or OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = target_dir / "fig2_diversion_outflow_timeseries.csv"
    with timeseries_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = (
            ["time_h", "head_inflow_m3s"]
            + [f"Qdiv_{node}_m3s" for node in dispatch.DIVERSION_NODES]
            + [f"supplied_{node}_m3" for node in dispatch.DIVERSION_NODES]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        times_h = result["times_h"]  # type: ignore[assignment]
        head_q = result["head_q"]  # type: ignore[assignment]
        qdiv = result["qdiv"]  # type: ignore[assignment]
        supplied = result["supplied"]  # type: ignore[assignment]
        for i, t_h in enumerate(times_h):
            row = {"time_h": f"{t_h:.6f}", "head_inflow_m3s": f"{head_q[i]:.6f}"}
            for node in dispatch.DIVERSION_NODES:
                row[f"Qdiv_{node}_m3s"] = f"{qdiv[node][i]:.6f}"
                row[f"supplied_{node}_m3"] = f"{supplied[node][i]:.3f}"
            writer.writerow(row)

    table_path = target_dir / "dispatch_feasibility_outlets.csv"
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
        branch_limits = result["branch_limits"]  # type: ignore[assignment]
        supplied = result["supplied"]  # type: ignore[assignment]
        first_positive_h = result["first_positive_h"]  # type: ignore[assignment]
        close_h = result["close_h"]  # type: ignore[assignment]
        for node in dispatch.DIVERSION_NODES:
            spec = dispatch.SPECS[node]
            limit = branch_limits[node]
            final_supplied = supplied[node][-1]
            writer.writerow(
                {
                    "node": node,
                    "specified_qmax_m3s": f"{spec.max_flow_m3s:.3f}",
                    "branch_node": "" if limit.branch_node is None else limit.branch_node,
                    "branch_safe_capacity_m3s": f"{limit.safe_capacity_m3s:.3f}",
                    "demand_m3": f"{spec.demand_m3:.3f}",
                    "first_positive_time_h": "" if first_positive_h[node] is None else f"{first_positive_h[node]:.6f}",
                    "close_time_h": "" if close_h[node] is None else f"{close_h[node]:.6f}",
                    "final_supplied_m3": f"{final_supplied:.3f}",
                    "demand_satisfied": final_supplied >= spec.demand_m3 - 1.0e-6,
                }
            )

    cfg = result["config"]
    branch_limits = result["branch_limits"]  # type: ignore[assignment]
    supplied = result["supplied"]  # type: ignore[assignment]
    first_positive_h = result["first_positive_h"]  # type: ignore[assignment]
    close_h = result["close_h"]  # type: ignore[assignment]
    summary = {
        "figure": str(fig_path),
        "timeseries_csv": str(timeseries_path),
        "dispatch_table_csv": str(table_path),
        "model": "1-D Saint-Venant finite-volume HLL forward model with semi-implicit Manning friction",
        "network_coupling": result.get(
            "network_coupling",
            "main canal with diversion junction source terms",
        ),
        "head_boundary": {
            "initial_flow_m3s": 0.0,
            "target_flow_m3s": cfg.target_head_flow_m3s,
            "ramp_hours": cfg.ramp_hours,
            "boundary_depth": "Manning normal depth compatible with boundary discharge",
        },
        "diversion_rule": "At each diversion junction, Qdiv is solved from main/branch water-level difference as an energy-slope Manning capacity, then limited by specified Qmax, branch safe capacity, remaining demand, local main-canal storage, and branch freeboard.",
        "safe_depth_ratio_for_branch_capacity": cfg.safe_depth_ratio,
        "branch_chain_node_count": {
            str(node): len(result.get("branch_ids", {}).get(node, []))
            for node in dispatch.DIVERSION_NODES
        },
        "diversions": {
            str(node): {
                "specified_qmax_m3s": dispatch.SPECS[node].max_flow_m3s,
                "demand_m3": dispatch.SPECS[node].demand_m3,
                "branch_node": branch_limits[node].branch_node,
                "branch_safe_capacity_m3s": branch_limits[node].safe_capacity_m3s,
                "first_positive_time_h": first_positive_h[node],
                "close_time_h": close_h[node],
                "final_supplied_m3": supplied[node][-1],
            }
            for node in dispatch.DIVERSION_NODES
        },
    }
    summary_path = target_dir / "fig2_diversion_outflow_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "timeseries_csv": timeseries_path,
        "dispatch_table_csv": table_path,
        "summary_json": summary_path,
    }


def main() -> None:
    cfg = dispatch.load_config()
    result = dispatch.simulate(cfg)
    fig_path = draw_diversion_outflow(result)
    paths = write_dispatch_outputs(result, fig_path)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
