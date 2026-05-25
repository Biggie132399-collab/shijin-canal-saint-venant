#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified postprocessing entry point for report figures and tables.

The hydraulic solvers stay in their original modules. This file coordinates
plot/table generation, reuses a single dispatch simulation where possible, and
syncs canonical outputs into ``results/figures`` and ``results/tables``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import cumulative_supply_postprocess as cumulative_supply
import dispatch
import dispatch_postprocess
import key_node_depth_postprocess as key_depth
import pati_local_reach_applicability as pati
import depth_envelope_postprocess as depth_envelope
import saint_venant_dispatch_topology_and_capacity as topology
import simulator


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FINAL_FIG_DIR = RESULTS_DIR / "figures"
FINAL_TABLE_DIR = RESULTS_DIR / "tables"
SV_OUT_DIR = RESULTS_DIR / "saint_venant_dispatch_results"
SV_FIG_DIR = SV_OUT_DIR / "figures"
PATI_OUT_DIR = RESULTS_DIR / "pati_local_reach_applicability_results"
PATI_FIG_DIR = PATI_OUT_DIR / "figures"


@dataclass
class PostprocessSummary:
    """Paths produced by a postprocessing run."""

    figures: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)
    summaries: Dict[str, str] = field(default_factory=dict)
    synced: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "PostprocessSummary") -> None:
        self.figures.update(other.figures)
        self.tables.update(other.tables)
        self.summaries.update(other.summaries)
        self.synced.update(other.synced)

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {
            "figures": self.figures,
            "tables": self.tables,
            "summaries": self.summaries,
            "synced": self.synced,
        }


def _ensure_output_dirs() -> None:
    for path in (SV_OUT_DIR, SV_FIG_DIR, PATI_OUT_DIR, PATI_FIG_DIR, FINAL_FIG_DIR, FINAL_TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _quiet_call(func, *args, **kwargs):
    """Call legacy helpers that print JSON summaries without polluting CLI output."""

    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _copy(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Cannot sync missing postprocess output: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_cumulative_supply_outputs(result: Mapping[str, object]) -> PostprocessSummary:
    """Write canonical cumulative-supply figure, table, and summary."""

    fig_path = SV_FIG_DIR / "fig3_cumulative_supply_process.png"
    cumulative_supply.draw_fig3(result, out_path=fig_path)

    csv_path = SV_OUT_DIR / "fig3_cumulative_supply_timeseries.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["time_h"] + [f"W_{node}_m3" for node in dispatch.DIVERSION_NODES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        times_h: List[float] = result["times_h"]  # type: ignore[assignment]
        supplied = result["supplied"]  # type: ignore[assignment]
        for i, t_h in enumerate(times_h):
            row = {"time_h": f"{t_h:.6f}"}
            for node in dispatch.DIVERSION_NODES:
                row[f"W_{node}_m3"] = f"{supplied[node][i]:.3f}"
            writer.writerow(row)

    summary_path = SV_OUT_DIR / "fig3_cumulative_supply_summary.json"
    summary = {
        "figure": str(fig_path),
        "csv": str(csv_path),
        "formula": "W_k(t)=integral_0^t Q_div,k(tau) d tau",
        "unit": "m3; plotted as 1e4 m3",
        "source": result.get("solver", "dispatch.py configured dispatch simulation"),
        "network_coupling": result.get("network_coupling"),
        "completion": {
            str(node): {
                "demand_m3": dispatch.SPECS[node].demand_m3,
                "close_time_h": result["close_h"][node],  # type: ignore[index]
                "final_supplied_m3": result["supplied"][node][-1],  # type: ignore[index]
            }
            for node in dispatch.DIVERSION_NODES
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return PostprocessSummary(
        figures={"fig3_cumulative_supply": str(fig_path)},
        tables={"fig3_cumulative_supply": str(csv_path)},
        summaries={"fig3_cumulative_supply": str(summary_path)},
    )


def write_key_depth_outputs(result: Mapping[str, object]) -> PostprocessSummary:
    """Write canonical key-node depth figure, table, and summary."""

    fig_path = SV_FIG_DIR / "fig4_key_node_depth_process.png"
    key_depth.draw_fig4(result, out_path=fig_path)

    csv_path = SV_OUT_DIR / "fig4_key_node_depth_timeseries.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = (
            ["time_h"]
            + [f"depth_{node}_m" for node in key_depth.KEY_NODES]
            + [f"water_level_{node}_m" for node in key_depth.KEY_NODES]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        times_h: List[float] = result["times_h"]  # type: ignore[assignment]
        depth = result["depth"]  # type: ignore[assignment]
        water_level = result["water_level"]  # type: ignore[assignment]
        for i, t_h in enumerate(times_h):
            row = {"time_h": f"{t_h:.6f}"}
            for node in key_depth.KEY_NODES:
                row[f"depth_{node}_m"] = f"{depth[node][i]:.6f}"
                row[f"water_level_{node}_m"] = f"{water_level[node][i]:.6f}"
            writer.writerow(row)

    summary_path = SV_OUT_DIR / "fig4_key_node_depth_summary.json"
    summary = {
        "figure": str(fig_path),
        "csv": str(csv_path),
        "model": result.get("solver", "implicit-network"),
        "network_coupling": result.get("network_coupling"),
        "plotted_variable": "hydraulic depth h at key nodes",
        "unit": "m",
        "key_nodes": key_depth.KEY_NODES,
        "note": "Water level Z is saved as bed elevation plus hydraulic depth.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return PostprocessSummary(
        figures={"fig4_key_node_depth": str(fig_path)},
        tables={"fig4_key_node_depth": str(csv_path)},
        summaries={"fig4_key_node_depth": str(summary_path)},
    )


def run_topology_postprocess(sync: bool = True) -> PostprocessSummary:
    """Generate topology/capacity outputs for Figure 1."""

    _ensure_output_dirs()
    rows = topology.branch_capacity_rows()
    capacity_csv, capacity_json = topology.save_capacity_outputs(rows)
    fig_path = topology.draw_topology()

    summary = PostprocessSummary(
        figures={"fig1_topology": str(fig_path)},
        tables={"branch_capacity_71_89": str(capacity_csv)},
        summaries={"branch_capacity_71_89": str(capacity_json)},
    )
    if sync:
        summary.merge(sync_final_outputs(keys=["fig1_topology", "branch_capacity_table"]))
    return summary


def run_dispatch_postprocess(cfg: dispatch.Config | None = None, sync: bool = True) -> PostprocessSummary:
    """Run the dispatch simulation once and generate Figures 2, 3, and 4."""

    _ensure_output_dirs()
    cfg = cfg or dispatch.load_config()
    result = simulator.run_dispatch(cfg)

    fig2_path = dispatch_postprocess.draw_diversion_outflow(
        result,
        out_path=SV_FIG_DIR / "fig2_diversion_outflow_process.png",
    )
    dispatch_paths = dispatch_postprocess.write_dispatch_outputs(result, fig2_path, out_dir=SV_OUT_DIR)
    summary = PostprocessSummary(
        figures={"fig2_diversion_outflow": str(fig2_path)},
        tables={
            "fig2_diversion_outflow": str(dispatch_paths["timeseries_csv"]),
            "dispatch_feasibility": str(dispatch_paths["dispatch_table_csv"]),
        },
        summaries={"fig2_diversion_outflow": str(dispatch_paths["summary_json"])},
    )
    summary.merge(write_cumulative_supply_outputs(result))
    summary.merge(write_key_depth_outputs(result))
    if sync:
        summary.merge(
            sync_final_outputs(
                keys=[
                    "fig2_diversion_outflow",
                    "fig3_cumulative_supply",
                    "fig4_key_node_depth",
                    "fig2_table",
                    "fig3_table",
                    "fig4_table",
                    "dispatch_table",
                ]
            )
        )
    return summary


def run_envelope_postprocess(cfg: dispatch.Config | None = None, sync: bool = True) -> PostprocessSummary:
    """Generate the maximum-depth envelope outputs for Figure 5/6."""

    _ensure_output_dirs()
    cfg = cfg or dispatch.load_config()
    result = depth_envelope.compute_depth_envelope(cfg)
    fig_path, exceed_safe, exceed_design = depth_envelope.draw_fig6(result)
    _quiet_call(depth_envelope.save_outputs, result, fig_path, exceed_safe, exceed_design)
    summary = PostprocessSummary(
        figures={"fig6_max_depth_envelope": str(fig_path)},
        tables={"fig6_max_depth_envelope": str(SV_OUT_DIR / "fig6_max_depth_envelope.csv")},
        summaries={"fig6_max_depth_envelope": str(SV_OUT_DIR / "fig6_max_depth_envelope_summary.json")},
    )
    if sync:
        summary.merge(sync_final_outputs(keys=["fig6_max_depth_envelope", "fig6_table"]))
    return summary


def run_pati_postprocess(sync: bool = True) -> PostprocessSummary:
    """Generate Pati RSR local-reach applicability outputs."""

    _ensure_output_dirs()
    rows = simulator.run_pati_local_applicability()
    pati.write_outputs(rows)
    summary_path = PATI_OUT_DIR / "summary.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    figures = result.get("figures", [])
    summary = PostprocessSummary(
        figures={"pati_local_reach_applicability": str(PATI_FIG_DIR / "fig_pati_local_reach_applicability.png")},
        tables={"pati_local_reach_applicability": str(PATI_OUT_DIR / "pati_local_reach_applicability.csv")},
        summaries={"pati_local_reach_applicability": str(PATI_OUT_DIR / "summary.json")},
    )
    if len(figures) > 1:
        summary.figures["pati_local_reach_applicability_table"] = str(figures[1])
    if sync:
        summary.merge(sync_final_outputs(keys=["pati_fig", "pati_table_fig", "pati_table"]))
    return summary


def run_all(sync: bool = True, include_pati: bool = True) -> PostprocessSummary:
    """Generate every report figure/table from the current source modules."""

    summary = PostprocessSummary()
    summary.merge(run_topology_postprocess(sync=sync))
    summary.merge(run_dispatch_postprocess(sync=sync))
    summary.merge(run_envelope_postprocess(sync=sync))
    if include_pati:
        summary.merge(run_pati_postprocess(sync=sync))
    return summary


SYNC_TARGETS: Mapping[str, tuple[Path, Path]] = {
    "fig1_topology": (
        SV_FIG_DIR / "fig1_geographic_topology_main_canal_branches.png",
        FINAL_FIG_DIR / "fig1_geographic_topology_main_canal_branches.png",
    ),
    "fig2_diversion_outflow": (
        SV_FIG_DIR / "fig2_diversion_outflow_process.png",
        FINAL_FIG_DIR / "fig2_diversion_outflow_process.png",
    ),
    "fig3_cumulative_supply": (
        SV_FIG_DIR / "fig3_cumulative_supply_process.png",
        FINAL_FIG_DIR / "fig3_cumulative_supply_process.png",
    ),
    "fig4_key_node_depth": (
        SV_FIG_DIR / "fig4_key_node_depth_process.png",
        FINAL_FIG_DIR / "fig4_key_node_depth_process.png",
    ),
    "fig6_max_depth_envelope": (
        SV_FIG_DIR / "fig6_max_depth_envelope.png",
        FINAL_FIG_DIR / "fig6_max_depth_envelope.png",
    ),
    "pati_fig": (
        PATI_FIG_DIR / "fig_pati_local_reach_applicability.png",
        FINAL_FIG_DIR / "fig_pati_local_reach_applicability.png",
    ),
    "pati_table_fig": (
        PATI_FIG_DIR / "fig_pati_local_reach_applicability_table.png",
        FINAL_FIG_DIR / "fig_pati_local_reach_applicability_table.png",
    ),
    "branch_capacity_table": (
        SV_OUT_DIR / "branch_capacity_71_89.csv",
        FINAL_TABLE_DIR / "branch_capacity_71_89.csv",
    ),
    "fig2_table": (
        SV_OUT_DIR / "fig2_diversion_outflow_timeseries.csv",
        FINAL_TABLE_DIR / "fig2_diversion_outflow_timeseries.csv",
    ),
    "fig3_table": (
        SV_OUT_DIR / "fig3_cumulative_supply_timeseries.csv",
        FINAL_TABLE_DIR / "fig3_cumulative_supply_timeseries.csv",
    ),
    "fig4_table": (
        SV_OUT_DIR / "fig4_key_node_depth_timeseries.csv",
        FINAL_TABLE_DIR / "fig4_key_node_depth_timeseries.csv",
    ),
    "fig6_table": (
        SV_OUT_DIR / "fig6_max_depth_envelope.csv",
        FINAL_TABLE_DIR / "fig6_max_depth_envelope.csv",
    ),
    "dispatch_table": (
        SV_OUT_DIR / "dispatch_feasibility_outlets.csv",
        FINAL_TABLE_DIR / "dispatch_feasibility_outlets.csv",
    ),
    "pati_table": (
        PATI_OUT_DIR / "pati_local_reach_applicability.csv",
        FINAL_TABLE_DIR / "pati_local_reach_applicability.csv",
    ),
}


def sync_final_outputs(keys: Iterable[str] | None = None) -> PostprocessSummary:
    """Copy canonical module outputs into the final report folders."""

    _ensure_output_dirs()
    selected = list(keys) if keys is not None else list(SYNC_TARGETS)
    synced: Dict[str, str] = {}
    for key in selected:
        src, dst = SYNC_TARGETS[key]
        synced[key] = str(_copy(src, dst))
    return PostprocessSummary(synced=synced)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report postprocess figures and tables.")
    parser.add_argument(
        "targets",
        nargs="*",
        choices=["all", "topology", "dispatch", "envelope", "pati", "sync"],
        help="Postprocess targets. Default: all.",
    )
    parser.add_argument("--no-sync", action="store_true", help="Do not copy outputs into results/figures and results/tables.")
    parser.add_argument("--skip-pati", action="store_true", help="Skip the Pati RSR applicability run when target is all.")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    targets = args.targets or ["all"]
    if "all" in targets and len(targets) > 1:
        raise SystemExit("Use either 'all' or specific targets, not both.")

    sync = not args.no_sync
    summary = PostprocessSummary()
    if targets == ["all"]:
        summary.merge(run_all(sync=sync, include_pati=not args.skip_pati))
    else:
        for target in targets:
            if target == "topology":
                summary.merge(run_topology_postprocess(sync=sync))
            elif target == "dispatch":
                summary.merge(run_dispatch_postprocess(sync=sync))
            elif target == "envelope":
                summary.merge(run_envelope_postprocess(sync=sync))
            elif target == "pati":
                summary.merge(run_pati_postprocess(sync=sync))
            elif target == "sync":
                summary.merge(sync_final_outputs())

    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
