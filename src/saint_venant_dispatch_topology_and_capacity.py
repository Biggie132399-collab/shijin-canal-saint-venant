#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute branch-canal capacity from real branch geometry and draw the geographic
topology for the 0-456 main-canal reach.
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw

import muskingum_cunge_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "saint_venant_dispatch_results"
FIG_DIR = OUT_DIR / "figures"
MAIN_START = 0
MAIN_END = 456
DIVERSION_NODES = [71, 89, 150, 194, 287, 349, 383]
KEY_LABELS = {
    0: "渠首0",
    71: "71分水口",
    89: "89分水口",
    150: "150分水口",
    194: "194分水口",
    287: "287分水口",
    349: "349分水口",
    383: "383分水口",
    456: "渠尾456",
}
MAIN_COLOR = "#075AA5"
BRANCH_COLOR = "#2E8B70"
NODE_COLOR = "#B42318"


def load_data():
    nodes = stage1.parse_nodes(DATA_DIR / "input.txt")
    params = stage1.parse_line_params(DATA_DIR / "lineParam.txt")
    neighbors = stage1.parse_neighbors(DATA_DIR / "neighborId.txt")
    return nodes, params, neighbors


def main_nodes(nodes):
    return [node for node in range(MAIN_START, MAIN_END + 1) if node in nodes]


def first_branch_child(node, neighbors, nodes, main_set):
    for child in neighbors.get(node, []):
        if child in nodes and child not in main_set:
            return child
    return None


def follow_branch(start_child, neighbors, nodes, main_set, max_nodes=80):
    if start_child is None:
        return []
    path = [start_child]
    seen = {start_child}
    current = start_child
    while len(path) < max_nodes:
        next_candidates = [n for n in neighbors.get(current, []) if n in nodes and n not in main_set and n not in seen]
        if not next_candidates:
            break
        current = next_candidates[0]
        path.append(current)
        seen.add(current)
    return path


def path_length_and_drop(anchor, path, nodes, use_nodes=None):
    if not path:
        return 0.0, 0.0
    selected = path if use_nodes is None else path[:use_nodes]
    full = [anchor] + selected
    length = 0.0
    for u, v in zip(full[:-1], full[1:]):
        length += stage1.distance(nodes[u], nodes[v])
    drop = nodes[anchor].elev - nodes[selected[-1]].elev
    return length, drop


def branch_capacity_rows():
    nodes, params, neighbors = load_data()
    main_set = set(main_nodes(nodes))
    rows = []
    for node in [71, 89]:
        child = first_branch_child(node, neighbors, nodes, main_set)
        path = follow_branch(child, neighbors, nodes, main_set)
        sec = stage1.section_for_node(child, nodes, params)
        h_safe = 0.90 * sec.depth
        first_len, first_drop = path_length_and_drop(node, path, nodes, use_nodes=1)
        ten_len, ten_drop = path_length_and_drop(node, path, nodes, use_nodes=min(10, len(path)))
        full_len, full_drop = path_length_and_drop(node, path, nodes, use_nodes=len(path))

        def q_for(length, drop):
            raw_slope = drop / max(length, 1.0)
            if raw_slope <= 0.0:
                return raw_slope, 0.0
            return raw_slope, stage1.manning_q(h_safe, sec, raw_slope)

        first_slope, first_q = q_for(first_len, first_drop)
        ten_slope, ten_q = q_for(ten_len, ten_drop)
        full_slope, full_q = q_for(full_len, full_drop)

        rows.append(
            {
                "node": node,
                "branch_start_node": child,
                "branch_nodes_used": len(path),
                "depth_D_m": sec.depth,
                "bottom_width_b_m": sec.bottom_width,
                "side_slope_z": sec.side_slope,
                "manning_n": sec.manning_n,
                "safe_depth_0.9D_m": h_safe,
                "first_segment_length_m": first_len,
                "first_segment_slope": first_slope,
                "capacity_first_segment_m3s": first_q,
                "first_10_nodes_length_m": ten_len,
                "first_10_nodes_slope": ten_slope,
                "capacity_first_10_nodes_m3s": ten_q,
                "full_branch_length_m": full_len,
                "full_branch_slope": full_slope,
                "capacity_full_branch_m3s": full_q,
            }
        )
    return rows


def extract_subgraph_nodes(nodes, neighbors, max_branch_nodes=12):
    main = main_nodes(nodes)
    main_set = set(main)
    selected = set(main)
    for node in main:
        child = first_branch_child(node, neighbors, nodes, main_set)
        if child is None:
            continue
        queue = deque([(child, 1)])
        while queue:
            current, depth = queue.popleft()
            if current in selected:
                continue
            selected.add(current)
            if depth >= max_branch_nodes:
                continue
            for nxt in neighbors.get(current, []):
                if nxt in nodes and nxt not in main_set and nxt not in selected:
                    queue.append((nxt, depth + 1))
    return selected, main


def draw_topology():
    nodes, _, neighbors = load_data()
    selected, main = extract_subgraph_nodes(nodes, neighbors)
    main_set = set(main)
    edges = []
    for u in selected:
        for v in neighbors.get(u, []):
            if v in selected:
                edges.append((u, v))

    xs = [nodes[n].x for n in selected]
    ys = [nodes[n].y for n in selected]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = 2000, 1350
    left_margin, right_margin = 110, 110
    top_margin, bottom_margin = 210, 135
    scale = min(
        (width - left_margin - right_margin) / max(max_x - min_x, 1.0),
        (height - top_margin - bottom_margin) / max(max_y - min_y, 1.0),
    )

    def xy(node):
        x = left_margin + (nodes[node].x - min_x) * scale
        y = height - bottom_margin - (nodes[node].y - min_y) * scale
        return x, y

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title = stage1.get_font(42, True)
    font = stage1.get_font(22)
    small = stage1.get_font(18)
    d.text((80, 45), "图1  0-456主干渠及分水支渠首段地理拓扑图", font=title, fill="#162033")

    for u, v in edges:
        color = MAIN_COLOR if u in main_set and v in main_set else BRANCH_COLOR
        line_width = 5 if color == MAIN_COLOR else 2
        d.line((*xy(u), *xy(v)), fill=color, width=line_width)

    for node in main:
        if node in KEY_LABELS:
            x, y = xy(node)
            r = 8
            d.ellipse((x - r, y - r, x + r, y + r), fill=NODE_COLOR, outline="white", width=2)

    offsets = {
        0: (88, -12),
        71: (-118, -36),
        89: (118, 36),
        150: (-125, 42),
        194: (118, 40),
        287: (-120, 40),
        349: (116, 42),
        383: (-118, 42),
        456: (92, 42),
    }
    for node, label in KEY_LABELS.items():
        if node not in selected:
            continue
        x, y = xy(node)
        ox, oy = offsets.get(node, (55, -45))
        tx, ty = x + ox, y + oy
        d.line((x, y, tx, ty), fill="#667085", width=1)
        box = d.textbbox((tx, ty), label, font=small)
        pad = 7
        d.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill="#FFFFFF", outline="#98A2B3")
        d.text((tx, ty), label, font=small, fill="#263241")

    legend_x, legend_y = 90, height - 92
    d.line((legend_x, legend_y, legend_x + 70, legend_y), fill=MAIN_COLOR, width=6)
    d.text((legend_x + 84, legend_y - 14), "主干渠", font=font, fill="#263241")
    d.line((legend_x + 220, legend_y, legend_x + 290, legend_y), fill=BRANCH_COLOR, width=3)
    d.text((legend_x + 304, legend_y - 14), "分水支渠", font=font, fill="#263241")
    d.ellipse((legend_x + 475, legend_y - 8, legend_x + 491, legend_y + 8), fill=NODE_COLOR, outline="white", width=2)
    d.text((legend_x + 510, legend_y - 14), "关键控制/分水节点", font=font, fill="#263241")
    d.text((90, height - 42), "说明：拓扑由 input.txt、neighborId.txt 与 lineParam.txt 解析得到；主干渠截取节点0-456，绿色线为各分水口支渠首段。", font=small, fill="#586579")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig1_geographic_topology_main_canal_branches.png"
    img.save(out, quality=95)
    return out


def save_capacity_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "branch_capacity_71_89.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path = OUT_DIR / "branch_capacity_71_89_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main():
    rows = branch_capacity_rows()
    csv_path, json_path = save_capacity_outputs(rows)
    fig_path = draw_topology()
    print(json.dumps({"capacity_csv": str(csv_path), "capacity_json": str(json_path), "topology": str(fig_path), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
