#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 reproduction of the core reverse-stage routing idea in
Pati et al. (2023) on a simple trapezoidal channel.

This is a deliberately small, reproducible experiment for PyCharm:
1. prescribe an upstream benchmark stage hydrograph;
2. compute its unsteady discharge with a Pati Eq. (12)-type dynamic
   stage-discharge relation;
3. run a forward Muskingum-Cunge calculation to generate the downstream
   boundary stage;
4. march upstream with the reverse-stage routing finite-difference formula
   corresponding to Pati et al. Eq. (17);
5. compare the recovered upstream stage/discharge with the forward benchmark.

The program does not use SCADA data and is not yet a full operational canal
model. Its purpose is to verify the reverse stage-routing calculation path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "pati_rsr_stage2_results"


@dataclass
class TrapezoidChannel:
    length_m: float = 20000.0
    dx_m: float = 500.0
    bottom_width_m: float = 25.0
    side_slope_h_per_v: float = 0.5
    bed_slope: float = 0.0010
    manning_n: float = 0.025
    lateral_flow_m2s: float = 0.0
    routing_celerity_factor: float = 2.00


@dataclass
class RunConfig:
    dt_seconds: float = 600.0
    duration_hours: float = 24.0
    base_depth_m: float = 2.20
    pulse_amplitude_m: float = 0.50
    pulse_center_hours: float = 10.0
    pulse_sigma_hours: float = 2.8
    mc_x_weight: float = 0.20


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def trapezoid_area(depth_m: float, channel: TrapezoidChannel) -> float:
    y = max(depth_m, 1.0e-6)
    return channel.bottom_width_m * y + channel.side_slope_h_per_v * y * y


def trapezoid_top_width(depth_m: float, channel: TrapezoidChannel) -> float:
    y = max(depth_m, 1.0e-6)
    return channel.bottom_width_m + 2.0 * channel.side_slope_h_per_v * y


def trapezoid_wetted_perimeter(depth_m: float, channel: TrapezoidChannel) -> float:
    y = max(depth_m, 1.0e-6)
    z = channel.side_slope_h_per_v
    return channel.bottom_width_m + 2.0 * y * math.sqrt(1.0 + z * z)


def hydraulic_radius(depth_m: float, channel: TrapezoidChannel) -> float:
    area = trapezoid_area(depth_m, channel)
    perimeter = trapezoid_wetted_perimeter(depth_m, channel)
    return area / max(perimeter, 1.0e-9)


def manning_discharge(depth_m: float, channel: TrapezoidChannel) -> float:
    area = trapezoid_area(depth_m, channel)
    radius = hydraulic_radius(depth_m, channel)
    return (
        area
        * radius ** (2.0 / 3.0)
        * math.sqrt(max(channel.bed_slope, 1.0e-12))
        / max(channel.manning_n, 1.0e-9)
    )


def celerity_dqda(depth_m: float, channel: TrapezoidChannel) -> float:
    """Kinematic celerity c0=dQ/dA from Manning's normal rating curve."""
    y = max(depth_m, 1.0e-5)
    eps = max(1.0e-4, y * 1.0e-4)
    y0 = max(1.0e-6, y - eps)
    y1 = y + eps
    dq = manning_discharge(y1, channel) - manning_discharge(y0, channel)
    da = trapezoid_area(y1, channel) - trapezoid_area(y0, channel)
    return max(dq / max(da, 1.0e-12), 1.0e-6)


def dperimeter_ddepth(channel: TrapezoidChannel) -> float:
    return 2.0 * math.sqrt(1.0 + channel.side_slope_h_per_v * channel.side_slope_h_per_v)


def celerity_from_discharge(depth_m: float, discharge_m3s: float, channel: TrapezoidChannel) -> float:
    """Pati Eq. (13)-type celerity estimate for a trapezoidal section."""
    y = max(depth_m, 1.0e-6)
    area = trapezoid_area(y, channel)
    radius = hydraulic_radius(y, channel)
    top_width = trapezoid_top_width(y, channel)
    bracket = (5.0 / 3.0) - (2.0 / 3.0) * radius / max(top_width, 1.0e-9) * dperimeter_ddepth(channel)
    return max(discharge_m3s / max(area, 1.0e-9) * max(bracket, 0.05), 1.0e-6)


def routing_celerity(depth_m: float, discharge_m3s: float, channel: TrapezoidChannel) -> float:
    """Diffusive-wave celerity used in the reverse-routing gradient equation."""
    return max(channel.routing_celerity_factor * celerity_from_discharge(depth_m, discharge_m3s, channel), 1.0e-6)


def unsteady_discharge_eq12(
    depth_m: float,
    previous_depth_m: float,
    previous_discharge_m3s: float,
    channel: TrapezoidChannel,
    dt_seconds: float,
) -> float:
    """
    Pati Eq. (12)-type dynamic stage-discharge computation.

    Q0 is the normal discharge for the instantaneous depth. The dA/dt and
    Q(j-1) terms introduce non-steady looped rating behavior.
    """
    y = max(depth_m, 1.0e-6)
    y_prev = max(previous_depth_m, 1.0e-6)
    area = trapezoid_area(y, channel)
    area_prev = trapezoid_area(y_prev, channel)
    dadt = (area - area_prev) / max(dt_seconds, 1.0e-9)
    top_width = trapezoid_top_width(y, channel)
    q0 = manning_discharge(y, channel)
    c = max(celerity_dqda(y, channel), 1.0e-6)
    s0 = max(channel.bed_slope, 1.0e-12)
    q_prev = max(previous_discharge_m3s, 1.0e-6)
    q = q0
    for _ in range(4):
        bc = max(top_width * c, 1.0e-9)
        bc2 = max(top_width * c * c, 1.0e-9)
        first = (q0 * q0) / (s0 * dt_seconds * bc2)
        inside = (
            first * first
            + 4.0
            * q0
            * q0
            * (
                1.0
                + q_prev / (s0 * dt_seconds * bc2)
                + 2.0 * dadt / (s0 * bc)
                - channel.lateral_flow_m2s / (s0 * bc)
            )
        )
        q = -0.5 * first + 0.5 * math.sqrt(max(inside, 1.0e-12))
        c = routing_celerity(y, q, channel)
    return max(q, 1.0e-6)


def unsteady_discharge_series(
    depth_m: Sequence[float],
    channel: TrapezoidChannel,
    dt_seconds: float,
    initial_discharge_m3s: float | None = None,
) -> List[float]:
    out: List[float] = []
    for j, y in enumerate(depth_m):
        if j == 0:
            out.append(initial_discharge_m3s if initial_discharge_m3s is not None else manning_discharge(y, channel))
        else:
            out.append(unsteady_discharge_eq12(y, depth_m[j - 1], out[j - 1], channel, dt_seconds))
    return out


def depth_from_area(area_m2: float, channel: TrapezoidChannel) -> float:
    area = max(area_m2, 1.0e-6)
    b = channel.bottom_width_m
    z = channel.side_slope_h_per_v
    if abs(z) < 1.0e-12:
        return area / b
    return (-b + math.sqrt(b * b + 4.0 * z * area)) / (2.0 * z)


def gradient(values: Sequence[float], dt_seconds: float) -> List[float]:
    n = len(values)
    if n == 1:
        return [0.0]
    out: List[float] = []
    for i in range(n):
        if i == 0:
            out.append((values[1] - values[0]) / dt_seconds)
        elif i == n - 1:
            out.append((values[-1] - values[-2]) / dt_seconds)
        else:
            out.append((values[i + 1] - values[i - 1]) / (2.0 * dt_seconds))
    return out


def gaussian_pulse(hours: float, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((hours - center) / sigma) ** 2)


def synthetic_upstream_stage(
    times_h: Sequence[float],
    config: RunConfig,
) -> List[float]:
    """Create the benchmark upstream stage hydrograph used as model input."""
    upstream: List[float] = []
    for t in times_h:
        pulse_up = gaussian_pulse(t, config.pulse_center_hours, config.pulse_sigma_hours)
        upstream.append(config.base_depth_m + config.pulse_amplitude_m * pulse_up)
    return upstream


def depth_from_discharge_normal(discharge_m3s: float, channel: TrapezoidChannel) -> float:
    """Invert the local normal rating relation for a stage estimate."""
    q = max(discharge_m3s, 1.0e-9)
    lo, hi = 1.0e-6, 20.0
    while manning_discharge(hi, channel) < q:
        hi *= 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if manning_discharge(mid, channel) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forward_muskingum_cunge(
    upstream_discharge_m3s: Sequence[float],
    channel: TrapezoidChannel,
    config: RunConfig,
) -> List[float]:
    """Forward MC routing used to generate a consistent downstream boundary."""
    steps = max(1, int(round(channel.length_m / channel.dx_m)))
    q_current = list(upstream_discharge_m3s)
    q_ref = sum(q_current) / len(q_current)
    y_ref = depth_from_discharge_normal(q_ref, channel)
    c_ref = max(celerity_dqda(y_ref, channel), 1.0e-6)
    k = channel.dx_m / c_ref
    x = min(max(config.mc_x_weight, 0.0), 0.49)
    dt = config.dt_seconds
    denom = k * (1.0 - x) + 0.5 * dt
    c0 = (-k * x + 0.5 * dt) / denom
    c1 = (k * x + 0.5 * dt) / denom
    c2 = (k * (1.0 - x) - 0.5 * dt) / denom
    for _ in range(steps):
        q_next = [q_current[0]]
        for j in range(1, len(q_current)):
            q_out = c0 * q_current[j] + c1 * q_current[j - 1] + c2 * q_next[j - 1]
            q_next.append(max(q_out, 1.0e-6))
        q_current = q_next
    return q_current


def reverse_one_subreach(
    downstream_depth_m: Sequence[float],
    downstream_discharge_m3s: Sequence[float],
    channel: TrapezoidChannel,
    dt_seconds: float,
) -> Tuple[List[float], List[float]]:
    """
    Reverse-stage routing over one dx step using the Eq. (17) structure:

    A_up = A_down - dx/c^2 * dQ_down/dt
           + dx/c * dA_down/dt - dx*q_l/c
    """
    area_down = [trapezoid_area(y, channel) for y in downstream_depth_m]
    discharge_down = list(downstream_discharge_m3s)
    celerity_down = [routing_celerity(y, q, channel) for y, q in zip(downstream_depth_m, discharge_down)]
    dadt = gradient(area_down, dt_seconds)
    dqdt = gradient(discharge_down, dt_seconds)
    routed_area: List[float] = []
    for a_i, c_i, da_i, dq_i in zip(area_down, celerity_down, dadt, dqdt):
        a_up = (
            a_i
            - channel.dx_m / (c_i * c_i) * dq_i
            + channel.dx_m / c_i * da_i
            - channel.dx_m * channel.lateral_flow_m2s / c_i
        )
        routed_area.append(max(a_up, 1.0e-6))
    routed_depth = [depth_from_area(a, channel) for a in routed_area]
    routed_discharge = unsteady_discharge_series(
        routed_depth,
        channel,
        dt_seconds,
        initial_discharge_m3s=discharge_down[0],
    )
    return routed_depth, routed_discharge


def reverse_stage_routing(
    downstream_depth_m: Sequence[float],
    downstream_discharge_m3s: Sequence[float],
    channel: TrapezoidChannel,
    dt_seconds: float,
) -> Dict[str, List[List[float]]]:
    steps = max(1, int(round(channel.length_m / channel.dx_m)))
    depth_by_node: List[List[float]] = [list(downstream_depth_m)]
    discharge_by_node: List[List[float]] = [list(downstream_discharge_m3s)]
    current_depth = list(downstream_depth_m)
    current_discharge = list(downstream_discharge_m3s)
    for _ in range(steps):
        current_depth, current_discharge = reverse_one_subreach(
            current_depth,
            current_discharge,
            channel,
            dt_seconds,
        )
        depth_by_node.append(current_depth)
        discharge_by_node.append(current_discharge)
    depth_upstream_first = list(reversed(depth_by_node))
    discharge_upstream_first = list(reversed(discharge_by_node))
    return {
        "depth_upstream_first": depth_upstream_first,
        "discharge_upstream_first": discharge_upstream_first,
    }


def nash_sutcliffe(sim: Sequence[float], obs: Sequence[float]) -> float:
    mean_obs = sum(obs) / max(len(obs), 1)
    denom = sum((v - mean_obs) ** 2 for v in obs)
    if denom <= 1.0e-12:
        return float("nan")
    return 1.0 - sum((s - o) ** 2 for s, o in zip(sim, obs)) / denom


def rmse(sim: Sequence[float], obs: Sequence[float]) -> float:
    return math.sqrt(sum((s - o) ** 2 for s, o in zip(sim, obs)) / max(len(obs), 1))


def peak_error_percent(sim: Sequence[float], obs: Sequence[float]) -> float:
    peak_obs = max(obs)
    return (max(sim) - peak_obs) / max(abs(peak_obs), 1.0e-9) * 100.0


def time_to_peak_error_hours(sim: Sequence[float], obs: Sequence[float], times_h: Sequence[float]) -> float:
    i_sim = max(range(len(sim)), key=lambda i: sim[i])
    i_obs = max(range(len(obs)), key=lambda i: obs[i])
    return times_h[i_sim] - times_h[i_obs]


def applicability_index(
    depth_m: Sequence[float],
    discharge_m3s: Sequence[float],
    channel: TrapezoidChannel,
    dt_seconds: float,
) -> List[float]:
    """
    Diagnostic implementation of the Pati et al. Eq. (27) criterion.

    The criterion is most meaningful on the rising limb where dQ/dt>0.
    Values >=1 indicate that the reverse step is expected to be feasible.
    """
    dqdt = gradient(discharge_m3s, dt_seconds)
    ai: List[float] = []
    for j, (y, dqi) in enumerate(zip(depth_m, dqdt)):
        if dqi <= 1.0e-12:
            ai.append(float("nan"))
            continue
        b = trapezoid_top_width(y, channel)
        c0 = celerity_from_discharge(y, discharge_m3s[j], channel)
        numerator = (dqi - 0.5 * channel.lateral_flow_m2s + b * channel.bed_slope * c0) ** 2
        denominator = 2.0 * b * channel.bed_slope * dqi
        ai.append(numerator / max(denominator, 1.0e-12))
    return ai


def draw_line_chart(
    path: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
    caption: str,
    size: Tuple[int, int] = (1800, 1050),
):
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(40, True)
    label_font = get_font(24)
    small_font = get_font(20)
    left, top, right, bottom = 135, 135, width - 85, height - 170
    draw.text((80, 45), title, font=title_font, fill="#162033")
    values = [v for _, vals, _ in series for v in vals if not math.isnan(v)]
    y_min = min(values)
    y_max = max(values)
    pad = max((y_max - y_min) * 0.12, 1.0e-6)
    y_min -= pad
    y_max += pad
    x_min, x_max = min(x_values), max(x_values)
    draw.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(6):
        y = top + (bottom - top) * i / 5
        value = y_max - (y_max - y_min) * i / 5
        draw.line((left, y, right, y), fill="#EEF2F7", width=2)
        draw.text((35, y - 14), f"{value:.2f}", font=small_font, fill="#586579")
    for i in range(0, len(x_values), max(1, len(x_values) // 6)):
        x = left + (right - left) * (x_values[i] - x_min) / max(x_max - x_min, 1.0e-9)
        draw.line((x, bottom, x, bottom + 10), fill="#8795A7", width=2)
        draw.text((x - 30, bottom + 18), f"{x_values[i]:.0f}", font=small_font, fill="#586579")

    def xy(xv: float, yv: float) -> Tuple[float, float]:
        x = left + (right - left) * (xv - x_min) / max(x_max - x_min, 1.0e-9)
        y = bottom - (bottom - top) * (yv - y_min) / max(y_max - y_min, 1.0e-9)
        return x, y

    for label, vals, color in series:
        pts = [xy(x, y) for x, y in zip(x_values, vals) if not math.isnan(y)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=5)
    legend_x = left
    legend_y = height - 120
    for label, _, color in series:
        draw.line((legend_x, legend_y, legend_x + 58, legend_y), fill=color, width=6)
        draw.text((legend_x + 70, legend_y - 14), label, font=small_font, fill="#263241")
        legend_x += 390
    draw.text(((left + right) / 2 - 60, bottom + 58), x_label, font=label_font, fill="#263241")
    draw.text((34, top + 260), y_label, font=label_font, fill="#263241")
    draw.text((80, height - 50), caption, font=small_font, fill="#586579")
    image.save(path, quality=95)


def draw_concept_figure(path: Path, channel: TrapezoidChannel):
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(40, True)
    font = get_font(26)
    small = get_font(22)
    draw.text((80, 45), "图1  Pati et al. (2023) 反向Stage Routing复现实验示意", font=title_font, fill="#162033")
    y_mid = 470
    x_left, x_right = 230, 1550
    draw.line((x_left, y_mid, x_right, y_mid), fill="#2D6A8E", width=9)
    for k in range(0, int(channel.length_m / channel.dx_m) + 1):
        x = x_left + (x_right - x_left) * k / max(channel.length_m / channel.dx_m, 1)
        draw.line((x, y_mid - 42, x, y_mid + 42), fill="#A7B4C4", width=2)
    draw.polygon([(x_left - 80, y_mid), (x_left + 20, y_mid - 60), (x_left + 20, y_mid + 60)], fill="#C75146")
    draw.text((x_left - 145, y_mid + 90), "上游待求\nyu(t), Qu(t)", font=font, fill="#263241")
    draw.polygon([(x_right + 80, y_mid), (x_right - 20, y_mid - 60), (x_right - 20, y_mid + 60)], fill="#0B5CAD")
    draw.text((x_right - 120, y_mid + 90), "下游已知\nyd(t)", font=font, fill="#263241")
    draw.line((x_right - 120, y_mid - 130, x_left + 120, y_mid - 130), fill="#C75146", width=6)
    draw.polygon([(x_left + 120, y_mid - 130), (x_left + 155, y_mid - 150), (x_left + 155, y_mid - 110)], fill="#C75146")
    draw.text((650, y_mid - 210), "按 Δx 由下游向上游逐段反推", font=font, fill="#263241")
    equation = "Aup = Adown - Δx/c²·dQ/dt + Δx/c·dA/dt - Δx·ql/c"
    draw.rounded_rectangle((300, 690, 1500, 820), radius=12, fill="#F5F8FC", outline="#CAD3DF", width=2)
    draw.text((345, 730), equation, font=font, fill="#162033")
    draw.text(
        (300, 850),
        f"本实验：L={channel.length_m/1000:.1f} km, Δx={channel.dx_m:.0f} m, "
        f"b={channel.bottom_width_m:.1f} m, z={channel.side_slope_h_per_v:.2f}, "
        f"n={channel.manning_n:.3f}, S0={channel.bed_slope:.4f}, "
        f"c=αc0, α={channel.routing_celerity_factor:.2f}, ql={channel.lateral_flow_m2s:.3f} m²/s",
        font=small,
        fill="#586579",
    )
    image.save(path, quality=95)


def draw_error_chart(
    path: Path,
    times_h: Sequence[float],
    depth_error: Sequence[float],
    discharge_error: Sequence[float],
    title: str = "图3  反向Stage Routing复现误差过程",
):
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(40, True)
    label_font = get_font(24)
    small = get_font(20)
    draw.text((80, 45), title, font=title_font, fill="#162033")
    panels = [
        (depth_error, "#C75146", "水深误差 ysim-yobs (m)", 135, 480),
        (discharge_error, "#6C4AB6", "流量相对误差 (%)", 590, 935),
    ]
    left, right = 135, width - 85
    x_min, x_max = min(times_h), max(times_h)

    def x_pos(xv: float) -> float:
        return left + (right - left) * (xv - x_min) / max(x_max - x_min, 1.0e-9)

    for vals, color, label, top, bottom in panels:
        y_min = min(vals)
        y_max = max(vals)
        pad = max((y_max - y_min) * 0.18, 0.01)
        y_min -= pad
        y_max += pad
        if y_min < 0.0 < y_max:
            y0 = bottom - (bottom - top) * (0.0 - y_min) / (y_max - y_min)
        else:
            y0 = None
        draw.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
        for i in range(5):
            y = top + (bottom - top) * i / 4
            value = y_max - (y_max - y_min) * i / 4
            draw.line((left, y, right, y), fill="#EEF2F7", width=2)
            draw.text((38, y - 14), f"{value:.2f}", font=small, fill="#586579")
        if y0 is not None:
            draw.line((left, y0, right, y0), fill="#9AA8B8", width=2)
        pts = []
        for x, v in zip(times_h, vals):
            y = bottom - (bottom - top) * (v - y_min) / max(y_max - y_min, 1.0e-9)
            pts.append((x_pos(x), y))
        draw.line(pts, fill=color, width=5)
        draw.line((left, bottom + 35, left + 58, bottom + 35), fill=color, width=6)
        draw.text((left + 70, bottom + 20), label, font=small, fill="#263241")
        draw.text((left, top - 34), label, font=label_font, fill="#263241")

    for i in range(0, len(times_h), max(1, len(times_h) // 6)):
        x = x_pos(times_h[i])
        draw.line((x, 935, x, 945), fill="#8795A7", width=2)
        draw.text((x - 30, 955), f"{times_h[i]:.0f}", font=small, fill="#586579")
    draw.text((815, 1000), "时间 t (h)", font=label_font, fill="#263241")
    draw.text(
        (80, height - 45),
        "上图为水深误差，单位m；下图为流量相对误差，单位%。两者分轴显示，避免不同单位混用。",
        font=small,
        fill="#586579",
    )
    image.save(path, quality=95)


def draw_rating_loop_chart(
    path: Path,
    depth_obs: Sequence[float],
    discharge_obs: Sequence[float],
    depth_sim: Sequence[float],
    discharge_sim: Sequence[float],
    channel: TrapezoidChannel,
    title: str = "图4  动态Stage-Discharge关系与绳套效应",
):
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(40, True)
    label_font = get_font(24)
    small = get_font(20)
    draw.text((80, 45), title, font=title_font, fill="#162033")
    left, top, right, bottom = 145, 135, width - 85, height - 170
    y_all = list(depth_obs) + list(depth_sim)
    q_all = list(discharge_obs) + list(discharge_sim)
    normal_depths = [min(y_all) + (max(y_all) - min(y_all)) * i / 160 for i in range(161)]
    normal_q = [manning_discharge(y, channel) for y in normal_depths]
    y_all += normal_depths
    q_all += normal_q
    x_min, x_max = min(y_all), max(y_all)
    y_min, y_max = min(q_all), max(q_all)
    x_pad = max((x_max - x_min) * 0.12, 1.0e-6)
    y_pad = max((y_max - y_min) * 0.12, 1.0e-6)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    draw.rectangle((left, top, right, bottom), outline="#CAD3DF", width=2)
    for i in range(6):
        y = top + (bottom - top) * i / 5
        value = y_max - (y_max - y_min) * i / 5
        draw.line((left, y, right, y), fill="#EEF2F7", width=2)
        draw.text((38, y - 14), f"{value:.1f}", font=small, fill="#586579")
    for i in range(6):
        x = left + (right - left) * i / 5
        value = x_min + (x_max - x_min) * i / 5
        draw.line((x, bottom, x, bottom + 10), fill="#8795A7", width=2)
        draw.text((x - 32, bottom + 18), f"{value:.2f}", font=small, fill="#586579")

    def xy(depth: float, discharge: float) -> Tuple[float, float]:
        x = left + (right - left) * (depth - x_min) / max(x_max - x_min, 1.0e-9)
        y = bottom - (bottom - top) * (discharge - y_min) / max(y_max - y_min, 1.0e-9)
        return x, y

    normal_pts = [xy(y, q) for y, q in zip(normal_depths, normal_q)]
    obs_pts = [xy(y, q) for y, q in zip(depth_obs, discharge_obs)]
    sim_pts = [xy(y, q) for y, q in zip(depth_sim, discharge_sim)]
    draw.line(normal_pts, fill="#8795A7", width=4)
    draw.line(obs_pts, fill="#2A9D8F", width=5)
    draw.line(sim_pts, fill="#C75146", width=5)
    for pts, color in [(obs_pts, "#2A9D8F"), (sim_pts, "#C75146")]:
        for idx in range(0, len(pts), max(1, len(pts) // 7)):
            x0, y0 = pts[idx]
            draw.ellipse((x0 - 4, y0 - 4, x0 + 4, y0 + 4), fill=color)
    legend_y = height - 120
    legend_x = left
    for label, color in [
        ("Manning正常流量曲线", "#8795A7"),
        ("上游基准动态关系", "#2A9D8F"),
        ("反推动态关系", "#C75146"),
    ]:
        draw.line((legend_x, legend_y, legend_x + 58, legend_y), fill=color, width=6)
        draw.text((legend_x + 70, legend_y - 14), label, font=small, fill="#263241")
        legend_x += 420
    draw.text(((left + right) / 2 - 110, bottom + 58), "水深 y (m)", font=label_font, fill="#263241")
    draw.text((34, top + 260), "流量 Q (m³/s)", font=label_font, fill="#263241")
    draw.text(
        (80, height - 50),
        "动态关系呈现涨水与退水不完全重合的绳套特征；灰线为仅作参照的Manning正常流量曲线。",
        font=small,
        fill="#586579",
    )
    image.save(path, quality=95)


def write_csv(
    path: Path,
    times_h: Sequence[float],
    y_down: Sequence[float],
    y_up_obs: Sequence[float],
    y_up_sim: Sequence[float],
    q_down: Sequence[float],
    q_up_obs: Sequence[float],
    q_up_sim: Sequence[float],
    ai: Sequence[float],
):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_h",
                "known_downstream_depth_m",
                "benchmark_upstream_depth_m",
                "rsr_upstream_depth_m",
                "known_downstream_discharge_m3s",
                "benchmark_upstream_discharge_m3s",
                "rsr_upstream_discharge_m3s",
                "applicability_index",
            ]
        )
        for row in zip(times_h, y_down, y_up_obs, y_up_sim, q_down, q_up_obs, q_up_sim, ai):
            writer.writerow([f"{v:.8f}" if isinstance(v, float) and not math.isnan(v) else "" for v in row])


def write_explanation(
    path: Path,
    channel: TrapezoidChannel,
    config: RunConfig,
    summary: Dict[str, float],
):
    text = f"""# 第二阶段：Pati et al. (2023) 反向 Stage Routing 核心算法复现说明

## 1. 复现目标

本阶段不直接替代第一阶段的 Muskingum-Cunge / 扩散波流量路由主模型，而是在一个简单梯形渠段上复现 Pati et al. (2023) 的核心思想：当下游断面水位过程已知时，通过反向 stage routing 由下游向上游逐段恢复上游水深过程，并用动态 stage-discharge 关系同步得到非恒定流量过程。

本程序对应的文件为 `pati_rsr_stage2.py`，可在 PyCharm 中直接运行。输出目录为 `pati_rsr_stage2_results`。

## 2. 计算渠段与边界条件

本复现实验采用等效梯形渠段：

- 渠段长度：L = {channel.length_m:.1f} m；
- 空间步长：Δx = {channel.dx_m:.1f} m；
- 时间步长：Δt = {config.dt_seconds:.1f} s；
- 底宽：b = {channel.bottom_width_m:.2f} m；
- 边坡系数：z = {channel.side_slope_h_per_v:.2f}，表示水平:垂直；
- 床面坡降：S0 = {channel.bed_slope:.6f}；
- Manning 糙率：n = {channel.manning_n:.4f} s/m^(1/3)；
- 反向路由波速系数：α = {channel.routing_celerity_factor:.2f}，即在 Eq. (13) 型波速基础上作简化修正；
- 侧向入流：ql = {channel.lateral_flow_m2s:.4f} m²/s。本次先取 0，表示无沿程侧向入流或出流。

本实验采用“正演-反演闭环”生成边界条件：先设定上游基准水深 yu,obs(t)，通过 Pati Eq. (12) 型动态 stage-discharge 关系得到上游基准流量 Qu,obs(t)，再用正向 Muskingum-Cunge 路由得到下游流量 Qd(t)，并由局部水力关系换算为下游已知水深 yd(t)。随后仅把 yd(t) 与 Qd(t) 作为反向模型输入，由 RSR 公式反推 yu,sim(t) 和 Qu,sim(t)。因此，下游边界不是凭经验削峰延迟画出来的，而是来自正向物理路由。

## 3. 采用公式

式（1）为 Pati et al. (2023) 采用的扩散波形式：

```text
∂Q/∂t + c ∂Q/∂x = D ∂²Q/∂x²
```

其中，Q 为流量，单位 m³/s；t 为时间，单位 s；x 为沿程距离，单位 m；c 为波速，单位 m/s；D 为扩散系数，单位 m²/s。

式（2）为含侧向流的一维连续方程：

```text
∂A/∂t + ∂Q/∂x = ql
```

其中，A 为过水面积，单位 m²；ql 为单位长度侧向入流，单位 m²/s。若为侧向分水出流，则 ql 可取负值。

式（3）为 Pati et al. (2023) 的反向 stage routing 梯度方程：

```text
∂A/∂x ≈ (1/c²) ∂Q/∂t - (2/c) ∂A/∂t + ql/c
```

式（4）为本程序逐段反推上游面积采用的离散式，对应论文中第一个上游子段的 Eq. (17) 思路：

```text
Aup,j = Adown,j - Δx/cj² · (Qdown,j+1 - Qdown,j-1)/(2Δt)
         + Δx/cj · (Adown,j+1 - Adown,j-1)/Δt
         - Δx · ql/cj
```

其中，Aup,j 为当前时间层 j 的上游过水面积，单位 m²；Adown,j 为下游过水面积，单位 m²；Δx 为渠段步长，单位 m；Δt 为时间步长，单位 s；c 为由水位-流量关系得到的局部波速，单位 m/s。

式（5）为梯形断面面积-水深关系：

```text
A = b y + z y²
```

其中，y 为水深，单位 m；b 为底宽，单位 m；z 为边坡系数，单位为水平:垂直。

式（6）为 Manning 正常流量关系，仅用于计算瞬时正常流量 Q0 和正演下游流量到水深的边界换算，不再把它作为非恒定流的唯一流量公式：

```text
Q = (1/n) A R^(2/3) S0^(1/2)
```

其中，R=A/P 为水力半径，单位 m；P 为湿周，单位 m；S0 为床面坡降，无量纲；n 为 Manning 糙率，单位 s/m^(1/3)。

式（7）为 Pati et al. Eq. (12) 型动态 stage-discharge 关系的简化实现：

```text
Qj = F(Q0,j, Qj-1, Aj, Aj-1, B, c, S0, ql, Δt)
```

其中，Q0,j 为当前水深对应的正常流量，单位 m³/s；Qj-1 为前一时间层的非恒定流量，单位 m³/s；Aj 和 Aj-1 为相邻时间层过水面积，单位 m²；B 为水面宽，单位 m；c 为扩散波波速，单位 m/s。该式包含 dA/dt 与 Qj-1，因此能够表达涨水期和退水期同水深不同流量的绳套型效应。

本程序中的等价计算式可写为：

```text
Qj = -0.5 Mj + 0.5 [ Mj² + 4 Q0,j² (1 + Qj-1/(S0 Δt B c²)
     + 2(Aj-Aj-1)/(S0 Δt B c) - ql/(S0 B c)) ]^0.5
Mj = Q0,j²/(S0 Δt B c²)
```

其中，Mj 为中间量，单位 m³/s；其余变量单位同上。由于 c 又与 Q 有关，程序内部采用短迭代更新 Q 与 c。

式（8）为 Pati et al. Eq. (13) 型波速估计：

```text
c = (Q/A) [5/3 - (2/3)(R/B)(dP/dy)]
```

其中，P 为湿周，单位 m；dP/dy 为湿周对水深的导数。程序中在 Eq. (12) 型流量计算内进行短迭代，使 Q 与 c 相互更新。

## 4. 图件说明

- 图1：上游水深过程复现结果，对比正演生成的下游已知水深、上游基准水深和反推上游水深。
- 图2：上游流量过程复现结果。基准流量来自动态 stage-discharge 关系和正演输入，反推流量来自 RSR 过程中的 Eq. (12) 型动态求解。
- 图3：反推误差过程，用于观察峰前、峰值与退水段的误差变化。
- 图4：动态 stage-discharge 绳套关系图，用于说明非恒定流量不是由 Manning 单值关系直接给出。

## 5. 复现结果

- 水深 RMSE：{summary["depth_rmse_m"]:.4f} m；
- 水深 NSE：{summary["depth_nse"]:.4f}；
- 流量 RMSE：{summary["discharge_rmse_m3s"]:.4f} m³/s；
- 流量 NSE：{summary["discharge_nse"]:.4f}；
- 水深峰值误差：{summary["depth_peak_error_percent"]:.2f} %；
- 流量峰值误差：{summary["discharge_peak_error_percent"]:.2f} %；
- 水深峰现时间误差：{summary["depth_time_to_peak_error_h"]:.2f} h；
- 流量峰现时间误差：{summary["discharge_time_to_peak_error_h"]:.2f} h。

## 6. 与第一阶段的串联关系

第一阶段主模型解决的是渠首-分水口-渠尾的流量传播与质量守恒问题，核心变量是 Q。第二阶段解决的是“正向模型生成下游边界后，能否用 Pati RSR 从下游反推上游水位/流量过程”的方法复现问题，核心变量是 stage/y 与非恒定 Q。二者不是互相替代关系，而是主模型与扩展验证模型的关系：

1. 第一阶段用于先把 0-456 节点的流量边界、侧向分水、质量守恒跑通；
2. 第二阶段用于回应老师提出的 Pati 文献复现要求，说明已经掌握反向 stage routing 的计算框架；
3. 后续可将第二阶段方法作为对比模型，用于检验某些渠段在“下游水位可观测、上游流量未知”时的反演能力。

## 7. 适用性说明

本复现只证明核心离散思路可以在简单梯形渠段和正演-反演闭环算例上跑通。正演边界目前由简化 Muskingum-Cunge 路由生成，而非 HEC-RAS 或完整圣维南方程；下游水深由正演流量经局部水力关系换算得到。因此目前适合写作表述为“核心算法复现与方法验证”，不宜表述为“完整复现 Pati 全部数值实验”或“已经完成全渠网圣维南方程模型”。

由于本复现的正演下游水位 yd(t) 是由 Muskingum-Cunge 流量路由结果经局部 Manning 关系静态换算得到，未能完整保留下游边界非恒定流的绳套，即迟滞，物理特征，导致输入反向模型的边界条件存在一定物理失真。这是反推结果出现约 1.33 h 相位偏移、且 NSE 约为 0.82 的重要原因之一。若后续采用 HEC-RAS 或完整圣维南方程全动力波模型生成更一致的下游边界条件，该部分边界误差有望明显降低，反推结果也有望进一步改善。
"""
    path.write_text(text, encoding="utf-8")


def run(channel: TrapezoidChannel, config: RunConfig, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(round(config.duration_hours * 3600.0 / config.dt_seconds)) + 1
    times_h = [i * config.dt_seconds / 3600.0 for i in range(n_steps)]
    y_up_obs = synthetic_upstream_stage(times_h, config)
    q_up_obs = unsteady_discharge_series(y_up_obs, channel, config.dt_seconds)
    q_down_forward = forward_muskingum_cunge(q_up_obs, channel, config)
    y_down = [depth_from_discharge_normal(q, channel) for q in q_down_forward]
    q_down = unsteady_discharge_series(
        y_down,
        channel,
        config.dt_seconds,
        initial_discharge_m3s=q_down_forward[0],
    )
    routed = reverse_stage_routing(y_down, q_down, channel, config.dt_seconds)
    y_up_sim = routed["depth_upstream_first"][0]
    q_up_sim = routed["discharge_upstream_first"][0]
    ai = applicability_index(y_up_sim, q_up_sim, channel, config.dt_seconds)
    depth_error = [s - o for s, o in zip(y_up_sim, y_up_obs)]
    discharge_error_pct = [
        (s - o) / max(abs(o), 1.0e-9) * 100.0 for s, o in zip(q_up_sim, q_up_obs)
    ]
    summary: Dict[str, float] = {
        "depth_rmse_m": rmse(y_up_sim, y_up_obs),
        "depth_nse": nash_sutcliffe(y_up_sim, y_up_obs),
        "discharge_rmse_m3s": rmse(q_up_sim, q_up_obs),
        "discharge_nse": nash_sutcliffe(q_up_sim, q_up_obs),
        "depth_peak_error_percent": peak_error_percent(y_up_sim, y_up_obs),
        "discharge_peak_error_percent": peak_error_percent(q_up_sim, q_up_obs),
        "depth_time_to_peak_error_h": time_to_peak_error_hours(y_up_sim, y_up_obs, times_h),
        "discharge_time_to_peak_error_h": time_to_peak_error_hours(q_up_sim, q_up_obs, times_h),
        "minimum_ai_on_rising_limb": min([v for v in ai if not math.isnan(v)], default=float("nan")),
    }
    draw_line_chart(
        fig_dir / "fig1_stage_reconstruction.png",
        "图1  上游水深过程的反向Stage Routing复现",
        times_h,
        [
            ("正演生成下游水深 yd (m)", y_down, "#0B5CAD"),
            ("上游基准水深 yu,obs (m)", y_up_obs, "#2A9D8F"),
            ("反推上游水深 yu,sim (m)", y_up_sim, "#C75146"),
        ],
        "时间 t (h)",
        "水深 y (m)",
        "下游水深由正向Muskingum-Cunge路由生成；上游水深由Pati Eq. (17)核心离散式逐段反推。",
    )
    draw_line_chart(
        fig_dir / "fig2_discharge_reconstruction.png",
        "图2  上游非恒定流量过程的反向Stage Routing复现",
        times_h,
        [
            ("正演生成下游流量 Qd (m³/s)", q_down, "#0B5CAD"),
            ("上游基准流量 Qu,obs (m³/s)", q_up_obs, "#2A9D8F"),
            ("反推上游流量 Qu,sim (m³/s)", q_up_sim, "#C75146"),
        ],
        "时间 t (h)",
        "流量 Q (m³/s)",
        "基准流量与反推流量均采用Pati Eq. (12)型动态stage-discharge关系计算，单位为m³/s。",
    )
    draw_error_chart(fig_dir / "fig3_reconstruction_error.png", times_h, depth_error, discharge_error_pct)
    draw_rating_loop_chart(
        fig_dir / "fig4_dynamic_rating_loop.png",
        y_up_obs,
        q_up_obs,
        y_up_sim,
        q_up_sim,
        channel,
    )
    write_csv(
        out_dir / "pati_rsr_stage2_timeseries.csv",
        times_h,
        y_down,
        y_up_obs,
        y_up_sim,
        q_down,
        q_up_obs,
        q_up_sim,
        ai,
    )
    all_outputs = {
        "channel": asdict(channel),
        "config": asdict(config),
        "summary": summary,
        "output_files": {
            "timeseries_csv": str(out_dir / "pati_rsr_stage2_timeseries.csv"),
            "explanation_md": str(out_dir / "Pati_RSR_stage2_explanation.md"),
            "figures": [
                str(fig_dir / "fig1_stage_reconstruction.png"),
                str(fig_dir / "fig2_discharge_reconstruction.png"),
                str(fig_dir / "fig3_reconstruction_error.png"),
                str(fig_dir / "fig4_dynamic_rating_loop.png"),
            ],
        },
    }
    (out_dir / "pati_rsr_stage2_summary.json").write_text(
        json.dumps(all_outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_explanation(out_dir / "Pati_RSR_stage2_explanation.md", channel, config, summary)
    return all_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pati et al. 2023 reverse-stage routing reproduction.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--length-m", type=float, default=20000.0)
    parser.add_argument("--dx-m", type=float, default=500.0)
    parser.add_argument("--dt-seconds", type=float, default=600.0)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--base-depth-m", type=float, default=2.20)
    parser.add_argument("--pulse-amplitude-m", type=float, default=0.50)
    parser.add_argument("--pulse-center-hours", type=float, default=10.0)
    parser.add_argument("--pulse-sigma-hours", type=float, default=2.8)
    parser.add_argument("--bottom-width-m", type=float, default=25.0)
    parser.add_argument("--side-slope", type=float, default=0.5)
    parser.add_argument("--bed-slope", type=float, default=0.0010)
    parser.add_argument("--manning-n", type=float, default=0.025)
    parser.add_argument("--lateral-flow-m2s", type=float, default=0.0)
    parser.add_argument("--routing-celerity-factor", type=float, default=2.00)
    return parser


def main():
    args = build_parser().parse_args()
    channel = TrapezoidChannel(
        length_m=args.length_m,
        dx_m=args.dx_m,
        bottom_width_m=args.bottom_width_m,
        side_slope_h_per_v=args.side_slope,
        bed_slope=args.bed_slope,
        manning_n=args.manning_n,
        lateral_flow_m2s=args.lateral_flow_m2s,
        routing_celerity_factor=args.routing_celerity_factor,
    )
    config = RunConfig(
        dt_seconds=args.dt_seconds,
        duration_hours=args.duration_hours,
        base_depth_m=args.base_depth_m,
        pulse_amplitude_m=args.pulse_amplitude_m,
        pulse_center_hours=args.pulse_center_hours,
        pulse_sigma_hours=args.pulse_sigma_hours,
    )
    outputs = run(channel, config, Path(args.out_dir))
    print(json.dumps(outputs["summary"], ensure_ascii=False, indent=2))
    print(f"Outputs written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
