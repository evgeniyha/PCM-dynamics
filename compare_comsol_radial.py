"""Compare Python heat simulation with COMSOL point traces.

The default COMSOL file is ``test_radial.txt``.  The expected point mapping is
based on the user's description and on the initial temperatures in the file:

    Point 5: r=0 um,  z=-10 nm
    Point 4: r=0 um,  z=-70 nm
    Point 3: r=0 um,  z=-150 nm
    Point 8: r=5 um,  z=-10 nm
    Point 9: r=10 um, z=-10 nm

COMSOL uses negative z into the film; the Python model uses positive z downward
from the GST surface.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gst_model import AxisymmetricThermalModel, GST225, Grid, LaserPulse, Substrate
from plot_utils import savefig_overwrite


POINTS = [
    ("Point 5", "r=0 um, z=-10 nm", 0.0e-6, 10.0e-9),
    ("Point 4", "r=0 um, z=-70 nm", 0.0e-6, 70.0e-9),
    ("Point 3", "r=0 um, z=-150 nm", 0.0e-6, 150.0e-9),
    ("Point 8", "r=5 um, z=-10 nm", 5.0e-6, 10.0e-9),
    ("Point 9", "r=10 um, z=-10 nm", 10.0e-6, 10.0e-9),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol", type=Path, default=Path("test_radial.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/comsol_comparison/comsol_radial_comparison.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/comsol_comparison/comsol_radial_comparison.csv"),
    )
    parser.add_argument("--energy-nj", type=float, default=650.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--spot-radius-um", type=float, default=30.0)
    parser.add_argument("--alpha-linear-um", type=float, default=10.0)
    parser.add_argument("--alpha-eff-um", type=float, default=16.2)
    parser.add_argument("--dt-ns", type=float, default=1.0)
    parser.add_argument("--dr-um", type=float, default=0.5)
    parser.add_argument("--dz-film-nm", type=float, default=5.0)
    parser.add_argument("--dz-substrate-nm", type=float, default=20.0)
    parser.add_argument("--radial-extent-um", type=float, default=60.0)
    parser.add_argument("--no-phase-change", action="store_false")
    return parser.parse_args()


def read_comsol_table(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a COMSOL text table with comment lines starting with '%'."""
    data = np.loadtxt(path, comments="%")
    time_ns = data[:, 0]
    columns = {
        "Point 3": data[:, 1],
        "Point 4": data[:, 2],
        "Point 5": data[:, 3],
        "Point 8": data[:, 4],
        "Point 9": data[:, 5],
    }
    return time_ns, columns


def sample_python_result(result, r_m: float, z_m: float) -> np.ndarray:
    """Interpolate Python T(t,z,r) at a requested point."""
    values = np.empty(len(result.time), dtype=float)
    # The finite-volume grid stores cell centers.  In particular, the first
    # radial value is dr/2 rather than exactly r=0.  Clamp requested COMSOL
    # points to the represented Python domain instead of returning NaN.
    z_sample = float(np.clip(z_m, result.z[0], result.z[-1]))
    r_sample = float(np.clip(r_m, result.r[0], result.r[-1]))
    point = np.array([[z_sample, r_sample]], dtype=float)
    for i, temperature in enumerate(result.temperature):
        interpolator = RegularGridInterpolator(
            (result.z, result.r),
            temperature,
            bounds_error=False,
            fill_value=np.nan,
        )
        values[i] = float(interpolator(point)[0])
    return values


def main() -> None:
    args = parse_args()
    comsol_time_ns, comsol = read_comsol_table(args.comsol)
    end_ns = float(comsol_time_ns[-1])

    gst = GST225(
        alpha_linear=args.alpha_linear_um * 1e6,
        alpha_eff=args.alpha_eff_um * 1e6,
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
        include_phase_change=not args.no_phase_change,
    )
    result = model.simulate_cooling(
        end_time=end_ns * 1e-9,
        time_step=args.dt_ns * 1e-9,
        progress=True,
    )

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True)
    axes_flat = axes.ravel()

    for ax, (point_name, label, r_m, z_m) in zip(axes_flat, POINTS):
        py_trace = sample_python_result(result, r_m, z_m)
        py_at_comsol = np.interp(comsol_time_ns * 1e-9, result.time, py_trace)
        comsol_trace = comsol[point_name]
        diff = py_at_comsol - comsol_trace

        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        rows.append((point_name, label, rmse, max_abs))

        ax.plot(comsol_time_ns, comsol_trace, "o", ms=3.0, label="COMSOL")
        ax.plot(result.time * 1e9, py_trace, "-", lw=1.8, label="Python")
        ax.set_title(f"{point_name}: {label}\nRMSE={rmse:.1f} K, max={max_abs:.1f} K")
        ax.set_xlabel("time, ns")
        ax.set_ylabel("T, K")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes_flat[-1].axis("off")
    fig.suptitle(
        "Python vs COMSOL radial point traces\n"
        f"E={args.energy_nj:g} nJ, w={args.spot_radius_um:g} um, "
        f"alpha={args.alpha_linear_um:g} um^-1, "
        f"alpha_eff={args.alpha_eff_um:g} um^-1, "
        f"{'no phase change' if args.no_phase_change else 'phase change on'}"
    )
    fig.tight_layout()
    savefig_overwrite(fig, args.output, dpi=180)
    plt.close(fig)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8") as f:
        f.write("point,label,rmse_K,max_abs_error_K\n")
        for point_name, label, rmse, max_abs in rows:
            f.write(f"{point_name},{label},{rmse:.8g},{max_abs:.8g}\n")
    print(f"Saved {args.output_csv.resolve()}")
    for point_name, label, rmse, max_abs in rows:
        print(f"{point_name:7s} {label:20s} RMSE={rmse:8.2f} K  max={max_abs:8.2f} K")


if __name__ == "__main__":
    main()
