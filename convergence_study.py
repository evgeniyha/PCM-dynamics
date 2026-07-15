r"""Optional mesh/time/domain convergence study for the GST225 model.

Nothing in this file runs automatically. Examples:

    .\.venv\Scripts\python.exe convergence_study.py --quick
    .\.venv\Scripts\python.exe convergence_study.py --mode all
    .\.venv\Scripts\python.exe convergence_study.py --mode spatial --skip-crystallinity

The full study is intentionally expensive. It writes a CSV table, a JSON
summary, temperature-only convergence plots, and surface temperature profiles
under output/convergence/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gst_model import (
    AxisymmetricThermalModel,
    Grid,
    LaserPulse,
    Substrate,
    ThermalResult,
)


@dataclass(frozen=True)
class StudyCase:
    group: str
    name: str
    dr_um: float
    dz_film_nm: float
    dz_substrate_nm: float
    dt_ns: float
    radial_extent_um: float


@dataclass
class CaseOutput:
    result: ThermalResult
    incident_fluence: np.ndarray
    absorbed_energy_density: np.ndarray
    pulse_end_temperature: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("spatial", "time", "domain", "all"),
                        default="all")
    parser.add_argument("--quick", action="store_true",
                        help="run only two deliberately inexpensive cases per group")
    parser.add_argument("--skip-crystallinity", action="store_true",
                        help="kept for old commands; convergence now uses only temperature")
    parser.add_argument("--energy-nj", type=float, default=650.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--end-ns", type=float, default=250.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--output", type=Path, default=Path("output/convergence"))
    parser.add_argument("--tolerance-percent", type=float, default=2.0)
    parser.add_argument("--no-phase-change", action="store_true",
                        help="disable melting/freezing latent heat; use constant amorphous GST properties")
    return parser.parse_args()


def make_cases(quick: bool) -> dict[str, list[StudyCase]]:
    spatial = [
        StudyCase("spatial", "coarse", 1.0, 10.0, 40.0, 0.5, 60.0),
        StudyCase("spatial", "medium", 0.5, 5.0, 20.0, 0.5, 60.0),
        # StudyCase("spatial", "fine", 0.25, 2.5, 10.0, 0.5, 60.0),
    ]
    temporal = [
        StudyCase("time", "dt_1ns", 0.5, 5.0, 20.0, 1.0, 60.0),
        StudyCase("time", "dt_0.5ns", 0.5, 5.0, 20.0, 0.5, 60.0),
        StudyCase("time", "dt_0.25ns", 0.5, 5.0, 20.0, 0.25, 60.0),
        StudyCase("time", "dt_0.125ns", 0.5, 5.0, 20.0, 0.125, 60.0),
    ]
    domain = [
        StudyCase("domain", "r45um", 0.5, 5.0, 20.0, 0.5, 45.0),
        StudyCase("domain", "r60um", 0.5, 5.0, 20.0, 0.5, 60.0),
        StudyCase("domain", "r90um", 0.5, 5.0, 20.0, 0.5, 90.0),
    ]
    if quick:
        # Quick mode checks the workflow, not production convergence.
        spatial = [
            StudyCase("spatial", "quick_coarse", 2.0, 20.0, 80.0, 2.0, 45.0),
            StudyCase("spatial", "quick_finer", 1.0, 10.0, 40.0, 2.0, 45.0),
        ]
        temporal = [
            StudyCase("time", "quick_dt4", 2.0, 20.0, 80.0, 4.0, 45.0),
            StudyCase("time", "quick_dt2", 2.0, 20.0, 80.0, 1.0, 45.0),
        ]
        domain = [
            StudyCase("domain", "quick_r30", 2.0, 20.0, 80.0, 2.0, 10.0),
            StudyCase("domain", "quick_r45", 2.0, 20.0, 80.0, 2.0, 45.0),
        ]
    return {"spatial": spatial, "time": temporal, "domain": domain}


def furthest_radius(mask: np.ndarray, r: np.ndarray) -> float:
    where = np.flatnonzero(np.any(mask, axis=0))
    return float(r[where[-1]]) if len(where) else 0.0


def deepest_depth(mask: np.ndarray, z: np.ndarray) -> float:
    where = np.flatnonzero(np.any(mask, axis=1))
    return float(z[where[-1]]) if len(where) else 0.0


def run_case(case: StudyCase, args: argparse.Namespace) -> tuple[dict, CaseOutput]:
    substrate = Substrate.silica() if args.substrate == "silica" else Substrate.tungsten()
    grid = Grid(
        film_thickness=args.film_nm * 1e-9,
        radial_extent=case.radial_extent_um * 1e-6,
        dr=case.dr_um * 1e-6,
        dz_film=case.dz_film_nm * 1e-9,
        dz_substrate=case.dz_substrate_nm * 1e-9,
    )
    model = AxisymmetricThermalModel(
        substrate=substrate,
        pulse=LaserPulse(energy=args.energy_nj * 1e-9),
        grid=grid,
        include_phase_change=not args.no_phase_change,
    )
    incident_fluence = (
        2.0 * model.pulse.energy / (np.pi * model.pulse.radius_1e2**2)
        * np.exp(-2.0 * model.r**2 / model.pulse.radius_1e2**2)
    )
    absorbed_energy_density = model.absorbed_energy_density()
    pulse_end_temperature = model.pulse_end_temperature()
    started = perf_counter()
    result = model.simulate_cooling(
        end_time=args.end_ns * 1e-9,
        time_step=case.dt_ns * 1e-9,
        progress=True,
    )
    runtime = perf_counter() - started

    film_temperature = result.temperature[:, result.film_mask, :]
    surface_temperature = result.temperature[:, 0, :]
    peak_field = np.max(film_temperature, axis=0)
    melted = peak_field >= model.gst.melting_temperature
    final_field = film_temperature[-1]
    surface_peak_profile = np.max(surface_temperature, axis=0)
    metrics = {
        **asdict(case),
        "cells": int(model.nr * model.nz),
        "time_steps": int(len(result.time) - 1),
        "runtime_s": runtime,
        "peak_temperature_K": float(np.max(film_temperature)),
        "final_peak_temperature_K": float(np.max(final_field)),
        "surface_peak_temperature_K": float(np.max(surface_peak_profile)),
        "surface_center_peak_temperature_K": float(surface_peak_profile[0]),
        "surface_edge_peak_temperature_K": float(surface_peak_profile[-1]),
        "melt_radius_um": furthest_radius(melted, result.r) * 1e6,
        "melt_depth_nm": deepest_depth(melted, result.z[result.film_mask]) * 1e9,
        "include_phase_change": not args.no_phase_change,
    }

    return metrics, CaseOutput(
        result=result,
        incident_fluence=incident_fluence,
        absorbed_energy_density=absorbed_energy_density,
        pulse_end_temperature=pulse_end_temperature,
    )


METRICS = (
    "peak_temperature_K",
    "final_peak_temperature_K",
    "surface_peak_temperature_K",
    "surface_center_peak_temperature_K",
)


def add_successive_differences(rows: list[dict]) -> None:
    previous_by_group: dict[str, dict] = {}
    for row in rows:
        previous = previous_by_group.get(row["group"])
        for metric in METRICS:
            key = f"change_{metric}_percent"
            value = row[metric]
            if previous is None or not np.isfinite(value) or previous[metric] == 0:
                row[key] = np.nan
            else:
                row[key] = 100.0 * abs(value - previous[metric]) / abs(value)
        previous_by_group[row["group"]] = row


def write_results(
    rows: list[dict],
    results: list[tuple[dict, CaseOutput]],
    output: Path,
    tolerance: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "convergence_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for group in sorted({row["group"] for row in rows}):
        last = [row for row in rows if row["group"] == group][-1]
        changes = {
            metric: last[f"change_{metric}_percent"] for metric in METRICS
            if np.isfinite(last[f"change_{metric}_percent"])
        }
        summary[group] = {
            "last_case": last["name"],
            "changes_percent": changes,
            "passes_tolerance": bool(changes) and all(
                value <= tolerance for value in changes.values()
            ),
        }
    with (output / "convergence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    phase_mode = "with phase change" if rows[0]["include_phase_change"] else "constant amorphous properties"
    fig.suptitle(f"Convergence metrics ({phase_mode})")
    for ax, metric in zip(axes.ravel(), METRICS):
        for group in sorted({row["group"] for row in rows}):
            selected = [row for row in rows if row["group"] == group]
            values = [row[metric] for row in selected]
            if np.any(np.isfinite(values)):
                ax.plot([row["name"] for row in selected], values, "o-", label=group)
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "convergence_metrics.png", dpi=180)
    plt.close(fig)

    groups = sorted({row["group"] for row in rows})
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(4.8 * len(groups), 4),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    axis_by_group = {group: axes[i] for i, group in enumerate(groups)}
    line_styles = {
        "start": "-",
        "middle": "--",
        "end": ":",
    }
    for row, case_output in results:
        result = case_output.result
        ax = axis_by_group[row["group"]]
        time_indices = {
            "start": 0,
            "middle": len(result.time) // 2,
            "end": len(result.time) - 1,
        }
        for moment, time_index in time_indices.items():
            surface_profile = result.temperature[time_index, 0, :]
            time_ns = result.time[time_index] * 1e9
            ax.plot(
                result.r * 1e6,
                surface_profile,
                line_styles[moment],
                label=f"{row['name']}, {time_ns:.3g} ns",
            )
        ax.set_title(row["group"])
        ax.set_xlabel("r, um")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Surface temperature, K")
    fig.suptitle(f"Surface temperature profiles ({phase_mode})")
    fig.tight_layout()
    fig.savefig(output / "surface_temperature_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(4.8 * len(groups), 7),
        squeeze=False,
    )
    for row, case_output in results:
        group_index = groups.index(row["group"])
        result = case_output.result
        r_um = result.r * 1e6
        axes[0, group_index].plot(
            r_um,
            case_output.incident_fluence,
            label=row["name"],
        )
        axes[1, group_index].plot(
            r_um,
            case_output.absorbed_energy_density[0, :],
            label=row["name"],
        )
    for group_index, group in enumerate(groups):
        axes[0, group_index].set_title(group)
        axes[0, group_index].set_ylabel("Incident fluence, J/m²")
        axes[1, group_index].set_ylabel("Absorbed energy density at surface, J/m³")
        axes[1, group_index].set_xlabel("r, um")
        for ax in axes[:, group_index]:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"Laser absorption radial profiles ({phase_mode})")
    fig.tight_layout()
    fig.savefig(output / "laser_absorption_radial_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(4.8 * len(groups), 7),
        sharey=True,
        squeeze=False,
    )
    for row, case_output in results:
        group_index = groups.index(row["group"])
        result = case_output.result
        axes[0, group_index].plot(
            case_output.absorbed_energy_density[:, 0],
            result.z * 1e9,
            label=row["name"],
        )
        axes[1, group_index].plot(
            case_output.pulse_end_temperature[:, 0],
            result.z * 1e9,
            label=row["name"],
        )
    for group_index, group in enumerate(groups):
        axes[0, group_index].set_title(group)
        axes[0, group_index].set_xlabel("Absorbed energy density, J/m³")
        axes[1, group_index].set_xlabel("Temperature after pulse, K")
        for ax in axes[:, group_index]:
            ax.grid(alpha=0.25)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("z, nm")
    axes[1, 0].set_ylabel("z, nm")
    fig.suptitle(f"Vertical center profiles, r~0 ({phase_mode})")
    fig.tight_layout()
    fig.savefig(output / "vertical_center_profiles.png", dpi=180)
    plt.close(fig)

    print(f"\nSaved {csv_path}")
    for group, info in summary.items():
        state = "PASS" if info["passes_tolerance"] else "NOT CONVERGED"
        print(f"{group}: {state} at {tolerance:g}% tolerance")


def main() -> None:
    args = parse_args()
    groups = make_cases(args.quick)
    selected_groups = groups if args.mode == "all" else {args.mode: groups[args.mode]}
    rows: list[dict] = []
    results: list[tuple[dict, CaseOutput]] = []
    total = sum(len(cases) for cases in selected_groups.values())
    number = 0
    for group, cases in selected_groups.items():
        for case in cases:
            number += 1
            print(f"\n=== Case {number}/{total}: {group}/{case.name} ===")
            metrics, result = run_case(case, args)
            rows.append(metrics)
            results.append((metrics, result))
            radius_label = "Rmelt" if metrics["include_phase_change"] else "R(T>=Tm)"
            print(
                f"Tmax={metrics['peak_temperature_K']:.1f} K, "
                f"{radius_label}={metrics['melt_radius_um']:.3g} um, "
                f"Tsurf,max={metrics['surface_peak_temperature_K']:.1f} K, "
                f"runtime={metrics['runtime_s']:.1f} s"
            )
    add_successive_differences(rows)
    write_results(rows, results, args.output, args.tolerance_percent)


if __name__ == "__main__":
    main()
