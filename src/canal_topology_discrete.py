import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import math

# 解决 Matplotlib 中文显示问题
system = platform.system()
if system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 强制定制的业务节点名称映射 (根据工程实际截断至456)
# ---------------------------------------------------------
CUSTOM_LABELS = {
    0: '起点(黄壁庄)',
    71: '分水口1',
    89: '分水口2',
    150: '分水口3',
    194: '分水口4',
    287: '分水口5',
    349: '分水口6',
    383: '分水口7',
    456: '渠尾'
}


def parse_gates(filepath):
    """解析 Gates.ini：兼容 GBK/UTF-8 中文编码，精准过滤无意义英文站点"""
    gates = {}
    if not os.path.exists(filepath):
        return gates

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

    current_idx = None
    current_name = None
    for line in lines:
        line = line.strip()
        if line.startswith('['):
            if current_idx is not None and current_name is not None:
                if "leakage" not in current_name.lower() and "inflow" not in current_name.lower():
                    gates[current_idx] = current_name
            current_idx = None
            current_name = None
        elif '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                k = parts[0].strip().lower()
                v = parts[1].strip()
                if k == 'nindex':
                    try:
                        current_idx = int(v)
                    except:
                        pass
                elif k == 'name':
                    current_name = v

    if current_idx is not None and current_name is not None:
        if "leakage" not in current_name.lower() and "inflow" not in current_name.lower():
            gates[current_idx] = current_name

    return gates


def parse_line_params(filepath):
    """解析 lineParam.txt"""
    line_params = {}
    if not os.path.exists(filepath): return line_params
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'): continue
            parts = line.split('\t')
            if len(parts) >= 5:
                style_id = parts[0].strip()
                depth = float(parts[1]) if parts[1].replace('.', '', 1).isdigit() else 3.0
                bottom_width = float(parts[2]) if parts[2].replace('.', '', 1).isdigit() else 10.0
                angle = float(parts[3]) if parts[3].replace('.', '', 1).isdigit() else 90.0
                manning = float(parts[4]) if parts[4].replace('.', '', 1).isdigit() else 0.02
                line_params[style_id] = {'depth': depth, 'width': bottom_width, 'angle': angle, 'manning': manning}
    return line_params


def parse_nodes(filepath):
    """解析 input.txt: 加载全部节点的坐标和高程"""
    nodes = {}
    if not os.path.exists(filepath): return nodes
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'): continue
            parts = line.split('\t')
            if len(parts) >= 5:
                node_id = int(parts[0])
                x, y, elev = float(parts[1]), float(parts[2]), float(parts[3])
                style_id = parts[4].strip()
                nodes[node_id] = {'x': x, 'y': y, 'elev': elev, 'style': style_id}
    return nodes


def parse_edges_and_build_graph(filepath, nodes_dict, line_params):
    """解析 neighborId.txt 构建完整图"""
    G = nx.DiGraph()
    if not os.path.exists(filepath): return G
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'): continue
            parts = line.split('\t')
            if len(parts) >= 4:
                node_id = int(parts[0])
                if node_id not in nodes_dict: continue
                for next_id_str in parts[3:]:
                    next_id_str = next_id_str.strip()
                    if not next_id_str: continue
                    next_id = int(next_id_str)
                    if next_id == -1 or next_id not in nodes_dict: continue

                    x1, y1 = nodes_dict[node_id]['x'], nodes_dict[node_id]['y']
                    x2, y2 = nodes_dict[next_id]['x'], nodes_dict[next_id]['y']
                    length_m = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    style = nodes_dict[node_id]['style']
                    param = line_params.get(style, {'width': 10.0, 'depth': 3.0, 'angle': 90.0, 'manning': 0.02})

                    G.add_node(node_id, **nodes_dict[node_id])
                    G.add_node(next_id, **nodes_dict[next_id])
                    G.add_edge(node_id, next_id, length_m=length_m, **param)
    return G


def extract_target_subgraph_by_nodes(G, main_start=0, main_end=456):
    """严格锚定 0-456 节点为主干渠，提取相关支渠。"""
    main_nodes = [n for n in range(main_start, main_end + 1) if n in G]

    subgraph_nodes = set(main_nodes)
    for n in main_nodes:
        for child in G.successors(n):
            if child not in main_nodes:
                if abs(child - n) < 100:
                    continue
                subgraph_nodes.add(child)
                subgraph_nodes.update(nx.descendants(G, child))

    G_sub = G.subgraph(subgraph_nodes).copy()

    total_length_km = 0.0
    for i in range(len(main_nodes) - 1):
        u = main_nodes[i]
        v = main_nodes[i + 1]
        if G_sub.has_edge(u, v):
            total_length_km += G_sub[u][v]['length_m'] / 1000.0

    print(f"📍 成功锁定主干渠区段 (节点 {main_nodes[0]} -> {main_nodes[-1]})，物理真实里程约 {total_length_km:.2f} km")
    return G_sub, main_nodes


def get_node_label(n, gates_dict):
    """获取节点标签：优先自定义字典，其次 Gates.ini，最后默认"""
    if n in CUSTOM_LABELS:
        return CUSTOM_LABELS[n]
    if n in gates_dict:
        return gates_dict[n]
    return f"分水({n})"


def plot_real_geographical_topology(G_sub, main_nodes, gates):
    """绘制高精度地理平面拓扑图 (包含标签避让算法)"""
    print(">>> [1/3] 正在绘制地理二维拓扑图...")
    fig, ax = plt.subplots(figsize=(16, 12))

    pos = {n: (d['x'], d['y']) for n, d in G_sub.nodes(data=True)}

    branch_edges = [(u, v) for u, v in G_sub.edges() if not (u in main_nodes and v in main_nodes)]
    for u, v in branch_edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#458B74',
                linewidth=1.2, alpha=0.8, solid_capstyle='round')

    if branch_edges:
        ax.plot([], [], color='#458B74', linewidth=1.2, label='支渠 (Branch Canals)')

    main_x = [pos[n][0] for n in main_nodes]
    main_y = [pos[n][1] for n in main_nodes]
    ax.plot(main_x, main_y, color='#08519C', linewidth=3.5, alpha=0.9,
            solid_capstyle='round', solid_joinstyle='round', label='主干渠 (Main Canal)')

    gate_nodes = [n for n in G_sub.nodes() if n in gates]
    branch_points = [n for n in main_nodes if G_sub.out_degree(n) > 1]
    custom_nodes = [n for n in CUSTOM_LABELS.keys() if n in G_sub.nodes()]

    highlight_nodes = list(set(gate_nodes + branch_points + custom_nodes + [main_nodes[0], main_nodes[-1]]))

    highlight_x = [pos[n][0] for n in highlight_nodes]
    highlight_y = [pos[n][1] for n in highlight_nodes]
    ax.scatter(highlight_x, highlight_y, s=40, color='#B22222', edgecolor='white', linewidths=1.0, zorder=4)

    highlight_nodes_sorted = sorted(highlight_nodes, key=lambda n: pos[n][0])

    # 4. 标签避让算法：在渠道右上和左下交替分布
    for idx, n in enumerate(highlight_nodes_sorted):
        label_text = get_node_label(n, gates)

        # 灌区走向自西北向东南，法线方向为右上(+x, +y) 和 左下(-x, -y)
        if idx % 2 == 0:
            x_offset, y_offset = 40, 45  # 右上方
        else:
            x_offset, y_offset = -40, -45  # 左下方

        ax.annotate(
            label_text,
            xy=pos[n],
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            fontsize=10, fontweight='bold', color='#333333',
            ha='center', va='center',
            # 设置气泡框样式
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#999999", linewidth=1.0, alpha=0.9),
            # 设置引线样式与节点缩进 (shrinkB=6)
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.0", color='#555555', lw=1.2, shrinkA=0,
                            shrinkB=6),
            zorder=5
        )

    ax.set_title(
        f"石津灌区高精度地理拓扑图 (主干渠 {main_nodes[0]}-{main_nodes[-1]})\nSpatial Topology of the Main Canal and Branches",
        pad=20, fontsize=16, fontweight='bold', color='#222222')
    ax.set_xlabel("X Coordinate (m)", fontsize=11, color='#444444')
    ax.set_ylabel("Y Coordinate (m)", fontsize=11, color='#444444')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='#CCCCCC')

    ax.grid(True, linestyle=':', color='#DDDDDD', alpha=0.8)
    ax.set_aspect('equal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    plt.tight_layout()
    plt.show()


def plot_high_res_longitudinal_profile(G_sub, main_nodes, gates):
    """绘制带中文站名标注的干渠沿程剖面图"""
    print(">>> [2/3] 正在绘制干渠一维沿程剖面图...")

    distances_km = [0.0]
    elevations = [G_sub.nodes[main_nodes[0]]['elev']]

    depth_0 = G_sub[main_nodes[0]][main_nodes[1]]['depth'] if len(main_nodes) > 1 and G_sub.has_edge(main_nodes[0],
                                                                                                     main_nodes[
                                                                                                         1]) else 3.0
    max_water_levels = [elevations[0] + depth_0]

    current_dist = 0.0
    for i in range(len(main_nodes) - 1):
        u = main_nodes[i]
        v = main_nodes[i + 1]

        if G_sub.has_edge(u, v):
            dist_km = G_sub[u][v]['length_m'] / 1000.0
            depth = G_sub[u][v]['depth']
        else:
            x1, y1 = G_sub.nodes[u]['x'], G_sub.nodes[u]['y']
            x2, y2 = G_sub.nodes[v]['x'], G_sub.nodes[v]['y']
            dist_km = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000.0
            depth = 3.0

        current_dist += dist_km
        distances_km.append(current_dist)

        bed_elev = G_sub.nodes[v]['elev']
        elevations.append(bed_elev)
        max_water_levels.append(bed_elev + depth)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(distances_km, elevations, color='#8B5A2B', linewidth=2.0, solid_joinstyle='round',
            label='Canal Bed Elevation')
    ax.plot(distances_km, max_water_levels, color='#08519C', linestyle='-', linewidth=1.5, alpha=0.8,
            label='Max Design Water Level')

    ax.fill_between(distances_km, min(elevations) - 2, elevations, color='#D2B48C', alpha=0.4)
    ax.fill_between(distances_km, elevations, max_water_levels, color='#E6F2FF', alpha=0.6, label='Flow Capacity')

    custom_nodes = [n for n in CUSTOM_LABELS.keys() if n in G_sub.nodes()]
    highlight_nodes = list(set([n for n in main_nodes if n in gates or G_sub.out_degree(n) > 1] + custom_nodes))

    highlight_info = []
    for n in highlight_nodes:
        idx = main_nodes.index(n)
        if idx < len(distances_km):
            highlight_info.append((n, distances_km[idx], max_water_levels[idx]))

    highlight_info.sort(key=lambda x: x[1])

    track_last_x = [-999] * 5

    for n, x_coord, y_coord in highlight_info:
        label_text = get_node_label(n, gates)

        assigned_track = 0
        for track in range(5):
            if x_coord - track_last_x[track] > 1.2:
                assigned_track = track
                break
        else:
            assigned_track = (track_last_x.index(min(track_last_x)) + 1) % 5

        track_last_x[assigned_track] = x_coord

        y_offset = 40 + assigned_track * 45

        ax.annotate(
            label_text,
            xy=(x_coord, y_coord),
            xytext=(0, y_offset),
            textcoords='offset points',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#333333',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAAAAA", linewidth=0.8, alpha=0.95),
            arrowprops=dict(arrowstyle="-|>", color='#888888', lw=1.2, shrinkA=0, shrinkB=3),
            zorder=5
        )

    ax.set_title(
        f"石津灌区主干渠 (Node {main_nodes[0]}-{main_nodes[-1]}) 沿程剖面与闸门分布\nLongitudinal Profile and Gate Distribution",
        pad=30, fontsize=16, fontweight='bold', color='#222222')
    ax.set_xlabel("Distance from Source (km)", fontsize=11, color='#444444')
    ax.set_ylabel("Absolute Elevation (m)", fontsize=11, color='#444444')

    ax.set_ylim(bottom=min(elevations) - 1.0, top=max(max_water_levels) + 1.5)

    ax.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='#CCCCCC')
    ax.grid(True, linestyle=':', color='#EEEEEE')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    plt.subplots_adjust(top=0.80, bottom=0.1, left=0.08, right=0.95)
    plt.show()


def plot_1d_schematic_topology(G_sub, main_nodes, gates):
    """绘制简化版一维逻辑拓扑推演图"""
    print(">>> [3/3] 正在提取关键控制点，绘制一维逻辑拓扑推演图...")

    # 1. 提取骨干控制点
    key_main_nodes = []
    for n in main_nodes:
        if n in CUSTOM_LABELS or G_sub.out_degree(n) > 1 or n == main_nodes[0] or n == main_nodes[-1]:
            if n not in key_main_nodes:
                key_main_nodes.append(n)

    # 2. 构建简化逻辑图 G_logic
    G_logic = nx.DiGraph()
    main_edges = []
    branch_edges = []
    edge_labels = {}

    for i in range(len(key_main_nodes) - 1):
        u = key_main_nodes[i]
        v = key_main_nodes[i + 1]
        G_logic.add_edge(u, v)
        main_edges.append((u, v))

        dist_m = 0.0
        curr = u
        while curr != v:
            next_nodes = [nxt for nxt in G_sub.successors(curr) if nxt in main_nodes]
            if not next_nodes: break
            nxt = next_nodes[0]
            if G_sub.has_edge(curr, nxt):
                dist_m += G_sub[curr][nxt].get('length_m', 50.0)
            curr = nxt
        edge_labels[(u, v)] = f"{dist_m / 1000.0:.2f}km"

    branch_counter = 1
    branch_labels = {}
    for n in key_main_nodes:
        if "分水" in get_node_label(n, gates):
            b_name = f"branch_{branch_counter}"
            G_logic.add_edge(n, b_name)
            branch_edges.append((n, b_name))
            branch_labels[b_name] = f"支渠{branch_counter}"
            branch_counter += 1

    pos_logic = {}
    curr_x = 0.0
    for n in key_main_nodes:
        pos_logic[n] = (curr_x, 0)
        curr_x += 4.0

    for u, v in branch_edges:
        pos_logic[v] = (pos_logic[u][0], -1.5)

    plt.figure(figsize=(20, 6))

    nx.draw_networkx_edges(G_logic, pos_logic, edgelist=main_edges, edge_color='#FF4500', width=3.5, arrows=True,
                           arrowsize=20, node_size=800, node_shape='s')
    nx.draw_networkx_edges(G_logic, pos_logic, edgelist=branch_edges, edge_color='#5F9EA0', width=2.0, arrows=True,
                           arrowsize=15, node_size=400, node_shape='s')

    main_labels = {n: get_node_label(n, gates) for n in key_main_nodes}

    nx.draw_networkx_labels(G_logic, pos_logic, labels=main_labels, font_size=11, font_weight='bold',
                            bbox=dict(boxstyle="square,pad=0.5", ec="#FF4500", fc="#FFA07A", lw=2, alpha=1.0))
    nx.draw_networkx_labels(G_logic, pos_logic, labels=branch_labels, font_size=10, font_weight='bold',
                            bbox=dict(boxstyle="square,pad=0.4", ec="#4682B4", fc="#E0FFFF", lw=1.5, alpha=1.0))

    nx.draw_networkx_edge_labels(G_logic, pos_logic, edge_labels=edge_labels, font_color='#B22222', font_size=10,
                                 font_weight='bold',
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

    plt.title("石津灌区一维拓扑路由逻辑推演图 (模型运算简化版)", pad=25, fontsize=18, fontweight='bold',
              color="#222222")

    plt.margins(0.10)
    plt.axis('off')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.tight_layout()
        except:
            pass
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("💧 正在启动高精度离散渠系解析引擎 (0-456节点截断)...")
    print("=" * 60)

    gates_dict = parse_gates('Gates.ini')
    line_params = parse_line_params('lineParam.txt')
    nodes_dict = parse_nodes('input.txt')

    if not line_params or not nodes_dict:
        print("❌ 核心数据文件缺失，程序中止。请将 txt 和 ini 文件放入同级目录。")
    else:
        G_full = parse_edges_and_build_graph('neighborId.txt', nodes_dict, line_params)

        if G_full.number_of_nodes() > 0:
            # 提取 0-456 区间
            G_sub, main_nodes_target = extract_target_subgraph_by_nodes(G_full, main_start=0, main_end=456)

            # [图1] 平面二维拓扑图
            plot_real_geographical_topology(G_sub, main_nodes_target, gates_dict)

            # [图2] 沿程纵断面高程图
            plot_high_res_longitudinal_profile(G_sub, main_nodes_target, gates_dict)

            # [图3] 一维方框逻辑拓扑推演图
            plot_1d_schematic_topology(G_sub, main_nodes_target, gates_dict)