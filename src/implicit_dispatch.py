#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fully implicit network-coupled Saint-Venant dispatch solver.

The solver uses node water level and link discharge as the coupled unknowns.
For each time step, link momentum equations are linearized implicitly and
substituted into node continuity equations. The resulting tree graph system is
solved by direct leaf-to-root elimination. Diversion connectors are handled as
active-set link constraints, so a junction can be governed either by the
implicit water-level/momentum relation or by an operating limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import dispatch


@dataclass
class ImplicitNetwork:
    bed: np.ndarray
    storage_dx: np.ndarray
    depth_limit: np.ndarray
    bottom_width: np.ndarray
    side_slope: np.ndarray
    manning_n: np.ndarray
    local_slope: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    edge_length: np.ndarray
    edge_bed: np.ndarray
    edge_bottom_width: np.ndarray
    edge_side_slope: np.ndarray
    edge_manning_n: np.ndarray
    connector_owner: np.ndarray
    outlet_nodes: np.ndarray
    outlet_slope: np.ndarray
    main_indices: np.ndarray
    main_ids: List[int]
    main_x: np.ndarray
    main_index_by_id: Dict[int, int]
    branch_ids: Dict[int, List[int]]
    connector_edge_by_node: Dict[int, int]
    limits: Dict[int, dispatch.BranchLimit]
    tree_parent: np.ndarray
    tree_parent_edge: np.ndarray
    tree_preorder: np.ndarray
    tree_postorder: np.ndarray


def _build_tree_order(node_count: int, edge_from: np.ndarray, edge_to: np.ndarray, root: int = 0):
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(node_count)]
    for edge_idx, (u, v) in enumerate(zip(edge_from, edge_to)):
        adjacency[int(u)].append((int(v), edge_idx))
        adjacency[int(v)].append((int(u), edge_idx))

    parent = np.full(node_count, -1, dtype=int)
    parent_edge = np.full(node_count, -1, dtype=int)
    parent[root] = root
    order = [root]
    for node in order:
        for neighbor, edge_idx in adjacency[node]:
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = node
            parent_edge[neighbor] = edge_idx
            order.append(neighbor)
    if len(order) != node_count:
        raise ValueError("Implicit network graph must be connected for tree solve.")
    return (
        parent,
        parent_edge,
        np.asarray(order, dtype=int),
        np.asarray(order[:0:-1], dtype=int),
    )


def _average(a: float, b: float) -> float:
    return 0.5 * (float(a) + float(b))


def _append_grid_nodes(
    grid: dispatch.ChannelGrid,
    bed: List[float],
    storage_dx: List[float],
    depth_limit: List[float],
    bottom_width: List[float],
    side_slope: List[float],
    manning_n: List[float],
    local_slope: List[float],
) -> List[int]:
    start = len(bed)
    bed.extend(float(v) for v in grid.bed)
    storage_dx.extend(float(v) for v in grid.dx_cell)
    depth_limit.extend(float(v) for v in grid.depth)
    bottom_width.extend(float(v) for v in grid.bottom_width)
    side_slope.extend(float(v) for v in grid.side_slope)
    manning_n.extend(float(v) for v in grid.manning_n)
    local_slope.extend(float(v) for v in grid.slope)
    return list(range(start, start + len(grid.ids)))


def _append_edge(
    edge_from: List[int],
    edge_to: List[int],
    edge_length: List[float],
    edge_bed: List[float],
    edge_bottom_width: List[float],
    edge_side_slope: List[float],
    edge_manning_n: List[float],
    connector_owner: List[int],
    u: int,
    v: int,
    length: float,
    bed: float,
    bottom_width: float,
    side_slope: float,
    manning_n: float,
    owner: int = -1,
) -> None:
    edge_from.append(u)
    edge_to.append(v)
    edge_length.append(max(float(length), 1.0e-6))
    edge_bed.append(float(bed))
    edge_bottom_width.append(float(bottom_width))
    edge_side_slope.append(float(side_slope))
    edge_manning_n.append(float(manning_n))
    connector_owner.append(owner)


def build_implicit_network(cfg: dispatch.Config) -> ImplicitNetwork:
    """Build a graph representation of the main canal and routed branch chains."""

    _, _, _, main_grid, branch_grids, limits = dispatch.build_network(cfg)

    bed: List[float] = []
    storage_dx: List[float] = []
    depth_limit: List[float] = []
    bottom_width: List[float] = []
    side_slope: List[float] = []
    manning_n: List[float] = []
    local_slope: List[float] = []

    main_indices = _append_grid_nodes(
        main_grid,
        bed,
        storage_dx,
        depth_limit,
        bottom_width,
        side_slope,
        manning_n,
        local_slope,
    )
    branch_indices: Dict[int, List[int]] = {}
    for node in dispatch.DIVERSION_NODES:
        if node not in branch_grids:
            continue
        branch_indices[node] = _append_grid_nodes(
            branch_grids[node],
            bed,
            storage_dx,
            depth_limit,
            bottom_width,
            side_slope,
            manning_n,
            local_slope,
        )

    edge_from: List[int] = []
    edge_to: List[int] = []
    edge_length: List[float] = []
    edge_bed: List[float] = []
    edge_bottom_width: List[float] = []
    edge_side_slope: List[float] = []
    edge_manning_n: List[float] = []
    connector_owner: List[int] = []
    connector_edge_by_node: Dict[int, int] = {}

    for k in range(len(main_indices) - 1):
        u = main_indices[k]
        v = main_indices[k + 1]
        _append_edge(
            edge_from,
            edge_to,
            edge_length,
            edge_bed,
            edge_bottom_width,
            edge_side_slope,
            edge_manning_n,
            connector_owner,
            u,
            v,
            main_grid.x[k + 1] - main_grid.x[k],
            _average(main_grid.bed[k], main_grid.bed[k + 1]),
            _average(main_grid.bottom_width[k], main_grid.bottom_width[k + 1]),
            _average(main_grid.side_slope[k], main_grid.side_slope[k + 1]),
            _average(main_grid.manning_n[k], main_grid.manning_n[k + 1]),
        )

    for node, grid in branch_grids.items():
        indices = branch_indices[node]
        main_idx = main_grid.ids.index(node)
        branch_start = indices[0]
        limit = limits[node]
        connector_edge_by_node[node] = len(edge_from)
        _append_edge(
            edge_from,
            edge_to,
            edge_length,
            edge_bed,
            edge_bottom_width,
            edge_side_slope,
            edge_manning_n,
            connector_owner,
            main_indices[main_idx],
            branch_start,
            limit.entry_length_m or max(float(grid.dx_cell[0]), 1.0),
            float(grid.bed[0]),
            float(grid.bottom_width[0]),
            float(grid.side_slope[0]),
            float(grid.manning_n[0]),
            owner=node,
        )
        for k in range(len(indices) - 1):
            _append_edge(
                edge_from,
                edge_to,
                edge_length,
                edge_bed,
                edge_bottom_width,
                edge_side_slope,
                edge_manning_n,
                connector_owner,
                indices[k],
                indices[k + 1],
                grid.x[k + 1] - grid.x[k],
                _average(grid.bed[k], grid.bed[k + 1]),
                _average(grid.bottom_width[k], grid.bottom_width[k + 1]),
                _average(grid.side_slope[k], grid.side_slope[k + 1]),
                _average(grid.manning_n[k], grid.manning_n[k + 1]),
            )

    outlet_nodes = [main_indices[-1]]
    outlet_slope = [float(main_grid.slope[-1])]
    for node, indices in branch_indices.items():
        grid = branch_grids[node]
        outlet_nodes.append(indices[-1])
        outlet_slope.append(float(grid.slope[-1]))

    edge_from_arr = np.asarray(edge_from, dtype=int)
    edge_to_arr = np.asarray(edge_to, dtype=int)
    tree_parent, tree_parent_edge, tree_preorder, tree_postorder = _build_tree_order(
        len(bed),
        edge_from_arr,
        edge_to_arr,
    )

    return ImplicitNetwork(
        bed=np.asarray(bed, dtype=float),
        storage_dx=np.asarray(storage_dx, dtype=float),
        depth_limit=np.asarray(depth_limit, dtype=float),
        bottom_width=np.asarray(bottom_width, dtype=float),
        side_slope=np.asarray(side_slope, dtype=float),
        manning_n=np.asarray(manning_n, dtype=float),
        local_slope=np.asarray(local_slope, dtype=float),
        edge_from=edge_from_arr,
        edge_to=edge_to_arr,
        edge_length=np.asarray(edge_length, dtype=float),
        edge_bed=np.asarray(edge_bed, dtype=float),
        edge_bottom_width=np.asarray(edge_bottom_width, dtype=float),
        edge_side_slope=np.asarray(edge_side_slope, dtype=float),
        edge_manning_n=np.asarray(edge_manning_n, dtype=float),
        connector_owner=np.asarray(connector_owner, dtype=int),
        outlet_nodes=np.asarray(outlet_nodes, dtype=int),
        outlet_slope=np.asarray(outlet_slope, dtype=float),
        main_indices=np.asarray(main_indices, dtype=int),
        main_ids=main_grid.ids,
        main_x=main_grid.x,
        main_index_by_id={node_id: main_indices[i] for i, node_id in enumerate(main_grid.ids)},
        branch_ids={node: grid.ids for node, grid in branch_grids.items()},
        connector_edge_by_node=connector_edge_by_node,
        limits=limits,
        tree_parent=tree_parent,
        tree_parent_edge=tree_parent_edge,
        tree_preorder=tree_preorder,
        tree_postorder=tree_postorder,
    )


def _manning_rating_and_derivative(
    h,
    bottom_width,
    side_slope,
    manning_n,
    slope,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.maximum(h, 1.0e-9)
    area = dispatch.area_from_depth_array(y, bottom_width, side_slope)
    top = dispatch.top_width_array(y, bottom_width, side_slope)
    perimeter = dispatch.wetted_perimeter_array(y, bottom_width, side_slope)
    radius = area / np.maximum(perimeter, 1.0e-9)
    dperimeter = 2.0 * np.sqrt(1.0 + side_slope * side_slope)
    dradius = (top * perimeter - area * dperimeter) / np.maximum(perimeter * perimeter, 1.0e-9)
    sqrt_slope = np.sqrt(np.maximum(slope, 1.0e-9))
    q = area * radius ** (2.0 / 3.0) * sqrt_slope / np.maximum(manning_n, 1.0e-9)
    dqdh = (
        top * radius ** (2.0 / 3.0)
        + area * (2.0 / 3.0) * radius ** (-1.0 / 3.0) * dradius
    ) * sqrt_slope / np.maximum(manning_n, 1.0e-9)
    return q, np.maximum(dqdh, 0.0)


def _edge_linearization(
    net: ImplicitNetwork,
    h_linearization: np.ndarray,
    q_old: np.ndarray,
    q_reference: np.ndarray,
    fixed_q: np.ndarray,
    cfg: dispatch.Config,
) -> Tuple[np.ndarray, np.ndarray]:
    h_u = np.maximum(h_linearization[net.edge_from] - net.bed[net.edge_from], cfg.min_depth_m)
    h_v = np.maximum(h_linearization[net.edge_to] - net.bed[net.edge_to], cfg.min_depth_m)
    h_edge = np.maximum(0.5 * (h_u + h_v), cfg.min_depth_m)
    area = dispatch.area_from_depth_array(h_edge, net.edge_bottom_width, net.edge_side_slope)
    perimeter = dispatch.wetted_perimeter_array(h_edge, net.edge_bottom_width, net.edge_side_slope)
    radius = area / np.maximum(perimeter, 1.0e-9)
    normal_slope = np.maximum(
        (net.bed[net.edge_from] - net.bed[net.edge_to]) / np.maximum(net.edge_length, 1.0e-9),
        cfg.min_bed_slope,
    )
    q_normal, _ = _manning_rating_and_derivative(
        h_edge,
        net.edge_bottom_width,
        net.edge_side_slope,
        net.edge_manning_n,
        normal_slope,
    )
    friction_reference = np.maximum.reduce(
        [
            np.abs(q_old),
            np.abs(q_reference),
            0.01 * q_normal,
            np.full_like(q_old, 1.0e-6),
        ]
    )
    friction = (
        cfg.dt_seconds
        * dispatch.G
        * net.edge_manning_n ** 2
        * friction_reference
        / np.maximum(area * radius ** (4.0 / 3.0), 1.0e-9)
    )
    denom = 1.0 + friction
    conductance = cfg.dt_seconds * dispatch.G * area / np.maximum(net.edge_length, 1.0e-9) / denom
    rhs = q_old / denom
    fixed = np.isfinite(fixed_q)
    conductance = np.where(fixed, 0.0, conductance)
    rhs = np.where(fixed, fixed_q, rhs)
    return conductance, rhs


def _matvec(
    x: np.ndarray,
    base_diag: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    conductance: np.ndarray,
) -> np.ndarray:
    y = base_diag * x
    if len(conductance):
        diff = conductance * (x[edge_from] - x[edge_to])
        y += np.bincount(edge_from, weights=diff, minlength=len(x))
        y -= np.bincount(edge_to, weights=diff, minlength=len(x))
    return y


def _tree_solve(
    net: ImplicitNetwork,
    base_diag: np.ndarray,
    conductance: np.ndarray,
    rhs: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    diag = base_diag.copy()
    diag += np.bincount(net.edge_from, weights=conductance, minlength=len(base_diag))
    diag += np.bincount(net.edge_to, weights=conductance, minlength=len(base_diag))
    work_rhs = rhs.copy()

    for node in net.tree_postorder:
        edge_idx = net.tree_parent_edge[node]
        parent = net.tree_parent[node]
        k = conductance[edge_idx]
        pivot = max(float(diag[node]), 1.0e-12)
        diag[parent] -= k * k / pivot
        work_rhs[parent] += k * work_rhs[node] / pivot

    x = np.zeros_like(rhs)
    root = int(net.tree_preorder[0])
    x[root] = work_rhs[root] / max(float(diag[root]), 1.0e-12)
    for node in net.tree_preorder[1:]:
        edge_idx = net.tree_parent_edge[node]
        parent = net.tree_parent[node]
        k = conductance[edge_idx]
        x[node] = (work_rhs[node] + k * x[parent]) / max(float(diag[node]), 1.0e-12)

    residual = float(
        np.linalg.norm(rhs - _matvec(x, base_diag, net.edge_from, net.edge_to, conductance))
    )
    return x, 1, residual


def _linear_solve_for_head(
    net: ImplicitNetwork,
    h_old: np.ndarray,
    h_linearization: np.ndarray,
    q_old: np.ndarray,
    q_reference: np.ndarray,
    fixed_q: np.ndarray,
    inflow_m3s: float,
    cfg: dispatch.Config,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    conductance, edge_rhs = _edge_linearization(net, h_linearization, q_old, q_reference, fixed_q, cfg)
    depth = np.maximum(h_linearization - net.bed, cfg.min_depth_m)
    storage = dispatch.top_width_array(depth, net.bottom_width, net.side_slope) * net.storage_dx
    base_diag = storage / cfg.dt_seconds
    rhs = base_diag * h_old

    edge_const = np.bincount(net.edge_from, weights=edge_rhs, minlength=len(h_old))
    edge_const -= np.bincount(net.edge_to, weights=edge_rhs, minlength=len(h_old))
    rhs -= edge_const

    rhs[net.main_indices[0]] += inflow_m3s

    outlet_h = np.maximum(h_linearization[net.outlet_nodes] - net.bed[net.outlet_nodes], cfg.min_depth_m)
    q_out, dqdh_out = _manning_rating_and_derivative(
        outlet_h,
        net.bottom_width[net.outlet_nodes],
        net.side_slope[net.outlet_nodes],
        net.manning_n[net.outlet_nodes],
        np.maximum(net.outlet_slope, cfg.min_bed_slope),
    )
    outlet_const = q_out - dqdh_out * h_linearization[net.outlet_nodes]
    base_diag[net.outlet_nodes] += dqdh_out
    rhs[net.outlet_nodes] -= outlet_const

    head, iterations, residual = _tree_solve(net, base_diag, conductance, rhs)
    head = np.maximum(head, net.bed + cfg.min_depth_m)
    return head, conductance, iterations, residual


def _compute_edge_flow(
    net: ImplicitNetwork,
    head: np.ndarray,
    h_linearization: np.ndarray,
    q_old: np.ndarray,
    q_reference: np.ndarray,
    fixed_q: np.ndarray,
    cfg: dispatch.Config,
) -> np.ndarray:
    conductance, edge_rhs = _edge_linearization(net, h_linearization, q_old, q_reference, fixed_q, cfg)
    return edge_rhs + conductance * (head[net.edge_from] - head[net.edge_to])


def _same_active_set(a: np.ndarray, b: np.ndarray) -> bool:
    both_free = np.isnan(a) & np.isnan(b)
    both_fixed = np.isfinite(a) & np.isfinite(b) & np.isclose(a, b, rtol=1.0e-8, atol=1.0e-8)
    return bool(np.all(both_free | both_fixed))


def _connector_limits(
    supplied: Dict[int, float],
    head: np.ndarray,
    cfg: dispatch.Config,
    net: ImplicitNetwork,
) -> Dict[int, float]:
    limits: Dict[int, float] = {}
    for node in dispatch.DIVERSION_NODES:
        if node not in net.connector_edge_by_node:
            limits[node] = 0.0
            continue
        spec = dispatch.SPECS[node]
        branch_limit = net.limits[node]
        edge_idx = net.connector_edge_by_node[node]
        main_idx = int(net.edge_from[edge_idx])
        main_depth = max(float(head[main_idx] - net.bed[main_idx]), cfg.min_depth_m)
        depth_factor = dispatch.diversion_factor(main_depth, float(net.depth_limit[main_idx]))
        remaining = max(spec.demand_m3 - supplied[node], 0.0)
        limits[node] = max(
            0.0,
            min(
                spec.max_flow_m3s,
                branch_limit.safe_capacity_m3s,
                remaining / max(cfg.dt_seconds, 1.0e-9),
            )
            * depth_factor,
        )
    return limits


def _solve_time_step(
    net: ImplicitNetwork,
    head_old: np.ndarray,
    q_old: np.ndarray,
    supplied: Dict[int, float],
    t_s: float,
    cfg: dispatch.Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    inflow = dispatch.head_inflow(t_s + cfg.dt_seconds, cfg)
    fixed_q = np.full(len(net.edge_from), np.nan, dtype=float)
    connector_limit = _connector_limits(supplied, head_old, cfg, net)
    for node, q_limit in connector_limit.items():
        if q_limit <= 0.0 and node in net.connector_edge_by_node:
            fixed_q[net.connector_edge_by_node[node]] = 0.0

    head = head_old.copy()
    final_iterations = 0
    final_residual = math.inf
    active_iterations = 0

    for active_iter in range(max(cfg.implicit_active_set_iterations, 1)):
        active_iterations = active_iter + 1
        linearization = head.copy()
        q_reference = np.abs(q_old).copy()
        conductance = np.zeros(len(net.edge_from), dtype=float)
        for _ in range(max(cfg.implicit_picard_iterations, 1)):
            solved, conductance, final_iterations, final_residual = _linear_solve_for_head(
                net,
                head_old,
                linearization,
                q_old,
                q_reference,
                fixed_q,
                inflow,
                cfg,
            )
            linearization = 0.65 * solved + 0.35 * linearization
        head = np.maximum(linearization, net.bed + cfg.min_depth_m)
        q_new = _compute_edge_flow(net, head, head, q_old, q_reference, fixed_q, cfg)

        new_fixed_q = fixed_q.copy()
        for node, edge_idx in net.connector_edge_by_node.items():
            q_limit = connector_limit[node]
            if q_limit <= 0.0 or q_new[edge_idx] <= 0.0:
                new_fixed_q[edge_idx] = 0.0
            elif q_new[edge_idx] >= q_limit:
                new_fixed_q[edge_idx] = q_limit
            else:
                new_fixed_q[edge_idx] = np.nan

        if _same_active_set(fixed_q, new_fixed_q):
            break
        fixed_q = new_fixed_q
    else:
        solved, conductance, final_iterations, final_residual = _linear_solve_for_head(
            net,
            head_old,
            head,
            q_old,
            np.abs(q_old),
            fixed_q,
            inflow,
            cfg,
        )
        head = np.maximum(solved, net.bed + cfg.min_depth_m)

    q_new = _compute_edge_flow(net, head, head, q_old, np.maximum(np.abs(q_old), np.abs(q_new)), fixed_q, cfg)
    for node, edge_idx in net.connector_edge_by_node.items():
        q_new[edge_idx] = min(max(q_new[edge_idx], 0.0), connector_limit[node])

    return head, q_new, {
        "inflow_m3s": inflow,
        "linear_iterations": float(final_iterations),
        "linear_residual": float(final_residual),
        "active_iterations": float(active_iterations),
    }


def simulate(cfg: dispatch.Config):
    net = build_implicit_network(cfg)
    head = net.bed + cfg.initial_depth_m
    q_edge = np.zeros(len(net.edge_from), dtype=float)

    supplied = {node: 0.0 for node in dispatch.DIVERSION_NODES}
    first_positive = {node: None for node in dispatch.DIVERSION_NODES}
    close_time = {node: None for node in dispatch.DIVERSION_NODES}
    qdiv_current = {node: 0.0 for node in dispatch.DIVERSION_NODES}

    output_every = max(1, int(round(cfg.output_interval_seconds / cfg.dt_seconds)))
    times_h: List[float] = []
    qdiv_series: Dict[int, List[float]] = {node: [] for node in dispatch.DIVERSION_NODES}
    supplied_series: Dict[int, List[float]] = {node: [] for node in dispatch.DIVERSION_NODES}
    head_q_series: List[float] = []
    key_nodes = [0, 71, 89, 150, 194, 287, 349, 383, 456]
    key_indices = {node: net.main_index_by_id[node] for node in key_nodes if node in net.main_index_by_id}
    depth_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    water_level_series: Dict[int, List[float]] = {node: [] for node in key_indices}
    depth_limit_series: Dict[int, float] = {node: float(net.depth_limit[idx]) for node, idx in key_indices.items()}
    safe_depth_series: Dict[int, float] = {node: cfg.safe_depth_ratio * float(net.depth_limit[idx]) for node, idx in key_indices.items()}
    main_depth = np.maximum(head[net.main_indices] - net.bed[net.main_indices], cfg.min_depth_m)
    main_hmax = main_depth.copy()
    main_hmax_time = np.zeros(len(net.main_indices), dtype=float)
    linear_iterations: List[float] = []
    linear_residuals: List[float] = []
    active_iterations: List[float] = []

    total_steps = int(round(cfg.duration_hours * 3600.0 / cfg.dt_seconds))
    for step in range(total_steps + 1):
        t_s = step * cfg.dt_seconds
        clock_h = cfg.start_hours + t_s / 3600.0
        main_depth = np.maximum(head[net.main_indices] - net.bed[net.main_indices], cfg.min_depth_m)
        improved = main_depth > main_hmax
        main_hmax = np.where(improved, main_depth, main_hmax)
        main_hmax_time = np.where(improved, clock_h, main_hmax_time)

        if step % output_every == 0:
            times_h.append(clock_h)
            head_q_series.append(dispatch.head_inflow(t_s, cfg))
            for node in dispatch.DIVERSION_NODES:
                qdiv_series[node].append(qdiv_current[node])
                supplied_series[node].append(supplied[node])
            for node, idx in key_indices.items():
                h_now = float(max(head[idx] - net.bed[idx], cfg.min_depth_m))
                depth_series[node].append(h_now)
                water_level_series[node].append(float(head[idx]))

        if step == total_steps:
            break

        head, q_edge, stats = _solve_time_step(net, head, q_edge, supplied, t_s, cfg)
        linear_iterations.append(stats["linear_iterations"])
        linear_residuals.append(stats["linear_residual"])
        active_iterations.append(stats["active_iterations"])

        qdiv_current = {node: 0.0 for node in dispatch.DIVERSION_NODES}
        for node, edge_idx in net.connector_edge_by_node.items():
            qdiv = float(q_edge[edge_idx])
            if qdiv <= 1.0e-9:
                continue
            if first_positive[node] is None:
                first_positive[node] = clock_h
            supplied[node] += qdiv * cfg.dt_seconds
            spec = dispatch.SPECS[node]
            if supplied[node] >= spec.demand_m3 - 1.0e-6 and close_time[node] is None:
                close_time[node] = cfg.start_hours + (t_s + cfg.dt_seconds) / 3600.0
                supplied[node] = min(supplied[node], spec.demand_m3)
            qdiv_current[node] = qdiv

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
        "branch_limits": net.limits,
        "config": cfg,
        "main_ids": net.main_ids,
        "main_dist_km": (net.main_x / 1000.0).tolist(),
        "main_hmax": main_hmax.tolist(),
        "main_hmax_time": main_hmax_time.tolist(),
        "branch_ids": net.branch_ids,
        "network_coupling": (
            "fully implicit main-branch network coupling; node continuity and link "
            "momentum are solved as one graph system with active-set diversion limits"
        ),
        "solver": "implicit-network",
        "solver_stats": {
            "linear_solver": "tree-direct",
            "mean_linear_iterations": float(np.mean(linear_iterations)) if linear_iterations else 0.0,
            "max_linear_iterations": float(np.max(linear_iterations)) if linear_iterations else 0.0,
            "max_linear_residual": float(np.max(linear_residuals)) if linear_residuals else 0.0,
            "mean_active_iterations": float(np.mean(active_iterations)) if active_iterations else 0.0,
        },
    }
