"""Compare COMSOL traces with Python phase-change runs for several sigmoid widths.

This is a diagnostic for the equivalent heat capacity method:

    c_E(T) = c + Delta H_f * d f_m / dT

where ``f_m`` is a sigmoid with width ``melt_smoothing_width``.  COMSOL traces
from ``test_radial.txt`` are plotted as points; Python curves are plotted for
several smoothing widths.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_comsol_radial import POINTS, read_comsol_table, sample_python_result
from gst_model import AxisymmetricThermalModel, GST225, Grid, LaserPulse, Substrate
from plot_utils import savefig_overwrite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol", type=Path, default=Path("test_radial.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/comsol_comparison/melt_smoothing_width_comparison.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/comsol_comparison/melt_smoothing_width_comparison.csv"),
    )
    parser.add_argument("--widths-k", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--energy-nj", type=float, default=650.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--spot-radius-um", type=float, default=35.0)
    parser.add_argument("--alpha-linear-um", type=float, default=10.0)
    parser.add_argument("--alpha-eff-um", type=float, default=16.2)
    parser.add_argument("--dt-ns", type=float, default=2.0)
    parser.add_argument("--dr-um", type=float, default=0.5)
    parser.add_argument("--dz-film-nm", type=float, default=5.0)
    parser.add_argument("--dz-substrate-nm", type=float, default=20.0)
    parser.add_argument("--radial-extent-um", type=float, default=60.0)
    return parser.parse_args()


def run_python_trace_set(args: argparse.Namespace, width_k: float, end_ns: float):
    gst = GST225(
        alpha_linear=args.alpha_linear_um * 1e6,
        alpha_eff=args.alpha_eff_um * 1e6,
        melt_smoothing_width=width_k,
    )
    substrate = Substrate.silica() if args.substrate == "silica" else Substrate.tungsten()
    model = AxisymmetricThermalModel(
        gst=gst,
        substrate=substrate,
        pulse=LaserPulse(
            energy=args.energy_nj * 1e-9,
            radius_1e2=args.spot_radius_um * 1e-6,
        ),
        grid=Grid(
            film_thickness=args.film_nm * 1e-9,
            radial_extent=args.radial_extent_um * 1e-6,
            dr=args.dr_um * 1e-6,
            dz_film=args.dz_film_nm * 1e-9,
            dz_substrate=args.dz_substrate_nm * 1e-9,
        ),
        include_phase_change=True,
    )
    result = model.simulate_cooling(
        end_time=end_ns * 1e-9,
        time_step=args.dt_ns * 1e-9,
        progress=True,
    )
    traces = {}
    for point_name, _label, r_m, z_m in POINTS:
        traces[point_name] = sample_python_result(result, r_m, z_m)
    return result.time * 1e9, traces


def main() -> None:
    args = parse_args()
    comsol_time_ns, comsol = read_comsol_table(args.comsol)
    end_ns = float(comsol_time_ns[-1])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True)
    axes_flat = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(args.widths_k)))
    summary_rows: list[tuple[float, str, str, float, float]] = []

    python_results = []
    for width_k in args.widths_k:
        print(f"Running Python phase-change model with melt_smoothing_width={width_k:g} K")
        python_results.append((width_k, *run_python_trace_set(args, width_k, end_ns)))

    for ax, (point_name, label, _r_m, _z_m) in zip(axes_flat, POINTS):
        comsol_trace = comsol[point_name]
        ax.plot(comsol_time_ns, comsol_trace, "ko", ms=3.0, label="COMSOL const cp")

        for color, (width_k, py_time_ns, traces) in zip(colors, python_results):
            py_trace = traces[point_name]
            py_at_comsol = np.interp(comsol_time_ns, py_time_ns, py_trace)
            diff = py_at_comsol - comsol_trace
            rmse = float(np.sqrt(np.mean(diff**2)))
            max_abs = float(np.max(np.abs(diff)))
            summary_rows.append((width_k, point_name, label, rmse, max_abs))
            ax.plot(
                py_time_ns,
                py_trace,
                "-",
                lw=1.5,
                color=color,
                label=f"Python ΔT={width_k:g} K",
            )

        ax.set_title(f"{point_name}: {label}")
        ax.set_xlabel("time, ns")
        ax.set_ylabel("T, K")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    axes_flat[-1].axis("off")
    fig.suptitle(
        "Effect of melt sigmoid width in Python equivalent heat capacity model\n"
        f"E={args.energy_nj:g} nJ, w={args.spot_radius_um:g} um, "
        f"alpha={args.alpha_linear_um:g} um^-1, alpha_eff={args.alpha_eff_um:g} um^-1"
    )
    fig.tight_layout()
    savefig_overwrite(fig, args.output, dpi=180)
    plt.close(fig)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8") as f:
        f.write("melt_smoothing_width_K,point,label,rmse_K,max_abs_error_K\n")
        for width_k, point_name, label, rmse, max_abs in summary_rows:
            f.write(f"{width_k:.8g},{point_name},{label},{rmse:.8g},{max_abs:.8g}\n")
    print(f"Saved {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
