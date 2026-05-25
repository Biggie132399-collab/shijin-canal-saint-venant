#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified simulation entry points for the canal modelling prototypes.

This module centralizes model execution. The numerical kernels remain in their
original scripts, while callers use these functions to run Saint-Venant,
Muskingum-Cunge/Pati validation, and local applicability simulations.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pati_local_equivalent_segment_89_71 as pati_local_89_71
import pati_local_reach_applicability as pati_applicability
import pati_real_main_canal_no_diversion as pati_real
import pati_rsr_stage2
import pati_single_reach_validation
import dispatch as dispatch_model
import saint_venant_legacy_dispatch_simulation as legacy_dispatch


ROOT = Path(__file__).resolve().parents[1]


def run_dispatch(cfg: dispatch_model.Config | None = None) -> Dict[str, object]:
    """Run the current Saint-Venant dispatch simulation for the 0-456 main canal."""

    return dispatch_model.simulate(cfg or dispatch_model.load_config())


def run_legacy_dispatch(cfg: legacy_dispatch.Config | None = None) -> Dict[str, object]:
    """Run the older stage-6 Saint-Venant dispatch prototype."""

    return legacy_dispatch.simulate(cfg or legacy_dispatch.Config())


def run_no_diversion_forward(cfg: pati_real.ForwardConfig | None = None) -> Dict[str, object]:
    """Run the real 0-456 main-canal Saint-Venant forward model without diversions."""

    return pati_real.run_forward_saint_venant(cfg or pati_real.ForwardConfig())


def run_pati_reverse(
    forward: Dict[str, object] | None = None,
    forward_cfg: pati_real.ForwardConfig | None = None,
    rsr_cfg: pati_real.RSRConfig | None = None,
    use_sv_tail_q: bool = True,
) -> Dict[str, object]:
    """Reverse-route the no-diversion Saint-Venant result with Pati RSR."""

    forward_result = forward or run_no_diversion_forward(forward_cfg)
    return pati_real.reverse_stage_routing(
        forward_result,
        rsr_cfg or pati_real.RSRConfig(),
        use_sv_tail_q=use_sv_tail_q,
    )


def run_pati_local_applicability(
    forward: Dict[str, object] | None = None,
    forward_cfg: pati_real.ForwardConfig | None = None,
    reaches: Sequence[tuple[int, int]] | None = None,
) -> List[Dict[str, object]]:
    """Evaluate local equivalent-reach Pati RSR applicability between control nodes.

    The returned rows are raw simulation/evaluation results. Writing CSV/figures
    is intentionally left to the postprocess module.
    """

    forward_result = forward or run_no_diversion_forward(forward_cfg)
    reach_pairs = list(reaches or pati_applicability.REACHES)
    return [
        pati_applicability.evaluate_reach(forward_result, downstream, upstream)
        for downstream, upstream in reach_pairs
    ]


def run_pati_local_89_71_validation() -> Dict[str, object]:
    """Run the local 89->71 equivalent-reach validation driver.

    This legacy validation function writes its own artefacts and returns the
    generated summary dictionary.
    """

    return pati_local_89_71.run()


def run_pati_real_main_canal_validation() -> Dict[str, float]:
    """Run the full no-diversion main-canal Pati RSR validation driver."""

    return pati_real.run()


def run_pati_stage2_validation(
    channel: pati_rsr_stage2.TrapezoidChannel | None = None,
    config: pati_rsr_stage2.RunConfig | None = None,
    out_dir: Path | None = None,
) -> Dict[str, object]:
    """Run the synthetic Pati RSR stage-2 validation driver."""

    return pati_rsr_stage2.run(
        channel or pati_rsr_stage2.TrapezoidChannel(),
        config or pati_rsr_stage2.RunConfig(),
        out_dir or pati_rsr_stage2.DEFAULT_OUT,
    )


def run_pati_single_reach_validation(
    out_dir: Path | None = None,
) -> Dict[str, float]:
    """Run the synthetic single-reach Pati RSR validation driver."""

    return pati_single_reach_validation.run(out_dir or pati_single_reach_validation.OUT_DIR)


def summarize_dispatch(
    result: Mapping[str, object],
    specs: Mapping[int, object] | None = None,
    cfg: object | None = None,
) -> Dict[str, object]:
    specs = specs or dispatch_model.SPECS
    supplied = result["supplied"]  # type: ignore[assignment]
    close_h = result["close_h"]  # type: ignore[assignment]
    first_positive_h = result["first_positive_h"]  # type: ignore[assignment]
    diversion_nodes = list(specs)
    final_supplied = {node: supplied[node][-1] for node in diversion_nodes}
    total_supplied = sum(final_supplied.values())
    total_demand = sum(specs[node].demand_m3 for node in diversion_nodes)  # type: ignore[attr-defined]
    cfg_obj = result.get("config", cfg) if isinstance(result, dict) else cfg
    times_h = result["times_h"]  # type: ignore[assignment]
    duration_hours = getattr(cfg_obj, "duration_hours", max(times_h))
    if len(times_h) > 1:
        inferred_dt = (times_h[1] - times_h[0]) * 3600.0
    else:
        inferred_dt = None
    dt_seconds = getattr(cfg_obj, "dt_seconds", inferred_dt)
    return {
        "kind": "dispatch_saint_venant",
        "solver": result.get("solver", getattr(cfg_obj, "solver", None)),
        "network_coupling": result.get("network_coupling"),
        "space_step_m": getattr(cfg_obj, "space_step_m", None),
        "branch_chain_node_count": {
            str(node): len(result.get("branch_ids", {}).get(node, []))  # type: ignore[union-attr]
            for node in diversion_nodes
        },
        "duration_hours": duration_hours,
        "dt_seconds": dt_seconds,
        "output_count": len(times_h),
        "total_demand_m3": total_demand,
        "total_supplied_m3": total_supplied,
        "demand_satisfied": all(
            final_supplied[node] >= specs[node].demand_m3 - 1.0e-6  # type: ignore[attr-defined]
            for node in diversion_nodes
        ),
        "first_positive_h": first_positive_h,
        "close_h": close_h,
        "solver_stats": result.get("solver_stats"),
    }


def summarize_forward(result: Mapping[str, object]) -> Dict[str, object]:
    head_q = result["head_q_boundary"]  # type: ignore[assignment]
    return {
        "kind": "no_diversion_forward_saint_venant",
        "node_count": len(result["ids"]),  # type: ignore[arg-type]
        "output_count": len(result["times_h"]),  # type: ignore[arg-type]
        "duration_hours": max(result["times_h"]),  # type: ignore[arg-type]
        "head_flow_initial_m3s": head_q[0],
        "head_flow_final_m3s": head_q[-1],
        "head_flow_max_m3s": max(head_q),
    }


def summarize_pati_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in rows:
        level = str(row["applicability"])
        counts[level] = counts.get(level, 0) + 1
    return {
        "kind": "pati_local_applicability",
        "reach_count": len(rows),
        "applicability_counts": counts,
        "reaches": [row["reach"] for row in rows],
    }


def _quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def run_target(target: str) -> Dict[str, object]:
    """Run one simulator CLI target and return a compact JSON-safe summary."""

    if target == "dispatch":
        return summarize_dispatch(run_dispatch())
    if target == "legacy-dispatch":
        cfg = legacy_dispatch.Config()
        return summarize_dispatch(
            run_legacy_dispatch(cfg),
            specs=legacy_dispatch.SPECS,
            cfg=cfg,
        )
    if target == "no-diversion":
        return summarize_forward(run_no_diversion_forward())
    if target == "pati-reverse":
        forward = run_no_diversion_forward()
        reverse = run_pati_reverse(forward=forward)
        return {
            "kind": "pati_reverse",
            "node_count": len(forward["ids"]),  # type: ignore[arg-type]
            "output_count": len(forward["times_h"]),  # type: ignore[arg-type]
            "use_sv_tail_q": reverse["use_sv_tail_q"],
        }
    if target == "pati-applicability":
        return summarize_pati_rows(run_pati_local_applicability())
    if target == "pati-local-89-71":
        return {"kind": target, "summary": _quiet_call(run_pati_local_89_71_validation)}
    if target == "pati-real-validation":
        return {"kind": target, "summary": _quiet_call(run_pati_real_main_canal_validation)}
    if target == "pati-stage2":
        return {"kind": target, "summary": _quiet_call(run_pati_stage2_validation)}
    if target == "pati-single":
        return {"kind": target, "summary": _quiet_call(run_pati_single_reach_validation)}
    raise ValueError(f"Unknown simulator target: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canal model simulations without report postprocessing.")
    parser.add_argument(
        "target",
        choices=[
            "dispatch",
            "legacy-dispatch",
            "no-diversion",
            "pati-reverse",
            "pati-applicability",
            "pati-local-89-71",
            "pati-real-validation",
            "pati-stage2",
            "pati-single",
        ],
        help="Simulation target to run.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_target(args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
