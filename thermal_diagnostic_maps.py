"""Build thermal diagnostic maps for the GST film.

The maps are useful for separating heat-transfer effects from the later
TTT-to-crystallinity post-processing:

* peak temperature Tmax(r, z);
* time after pulse when a point cools down through Tm;
* total residence time in a chosen crystallization-temperature band.
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

from gst_model import AxisymmetricThermalModel, Grid, LaserPulse, Substrate
from plot_utils import savefig_overwrite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-nj", type=float, default=600.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--substrate-thickness-um", type=float, default=None)
    parser.add_argument("--end-ns", type=float, default=450.0)
    parser.add_argument("--dt-ns", type=float, default=0.5)
    parser.add_argument("--dr-um", type=float, default=0.5)
    parser.add_argument("--dz-film-nm", type=float, default=10.0)
    parser.add_argument("--dz-substrate-nm", type=float, default=20.0)
    parser.add_argument("--radial-extent-um", type=float, default=60.0)
    parser.add_argument("--no-phase-change", action="store_true")
    parser.add_argument("--band-low-k", type=float, default=600.0)
    parser.add_argument("--band-high-k", type=float, default=730.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/thermal_diagnostics/thermal_maps.png"),
    )
    return parser.parse_args()


def substrate_from_args(args: argparse.Namespace) -> Substrate:
    thickness = None
    if args.substrate_thickness_um is not None:
        thickness = args.substrate_thickness_um * 1e-6
    if args.substrate == "silica":
        return Substrate.silica() if thickness is None else Substrate.silica(thickness)
    return Substrate.tungsten() if thickness is None else Substrate.tungsten(thickness)


def downward_crossing_time(time: np.ndarray, trace: np.ndarray, threshold: float) -> float:
    """Time of downward threshold crossing after the trace maximum."""
    trace = np.asarray(trace, dtype=float)
    if np.nanmax(trace) < threshold:
        return np.nan
    start = int(np.nanargmax(trace))
    for i in range(start, len(trace) - 1):
        t0 = trace[i]
        t1 = trace[i + 1]
        if t0 >= threshold and t1 <= threshold:
            if t0 == t1:
                return float(time[i])
            fraction = (threshold - t0) / (t1 - t0)
            return float(time[i] + fraction * (time[i + 1] - time[i]))
    return np.nan


def interval_overlap_duration(
    t0: float,
    t1: float,
    y0: float,
    y1: float,
    low: float,
    high: float,
) -> float:
    """Duration within [t0, t1] where a linear segment is inside [low, high]."""
    if not np.isfinite(y0) or not np.isfinite(y1):
        return 0.0
    if t1 <= t0:
        return 0.0
    if y0 == y1:
        return (t1 - t0) if low <= y0 <= high else 0.0

    candidates = [0.0, 1.0]
    for threshold in (low, high):
        fraction = (threshold - y0) / (y1 - y0)
        if 0.0 < fraction < 1.0:
            candidates.append(float(fraction))
    candidates = sorted(candidates)

    duration = 0.0
    for a, b in zip(candidates[:-1], candidates[1:]):
        mid = 0.5 * (a + b)
        y_mid = y0 + mid * (y1 - y0)
        if low <= y_mid <= high:
            duration += (b - a) * (t1 - t0)
    return duration


def residence_time_in_band(
    time: np.ndarray,
    trace: np.ndarray,
    low: float,
    high: float,
) -> float:
    total = 0.0
    for i in range(len(time) - 1):
        total += interval_overlap_duration(
            float(time[i]),
            float(time[i + 1]),
            float(trace[i]),
            float(trace[i + 1]),
            low,
            high,
        )
    return total


def symmetric_field(r: np.ndarray, field_zr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_um = np.concatenate((-r[::-1], r)) * 1e6
    field = np.concatenate((field_zr[:, ::-1], field_zr), axis=1)
    return r_um, field


def save_maps(
    path: Path,
    r: np.ndarray,
    z: np.ndarray,
    tmax: np.ndarray,
    tm_cross_time: np.ndarray,
    band_time: np.ndarray,
    args: argparse.Namespace,
    melting_temperature: float,
) -> None:
    r_um, tmax_sym = symmetric_field(r, tmax)
    _, cross_sym = symmetric_field(r, tm_cross_time * 1e9)
    _, band_sym = symmetric_field(r, band_time * 1e9)
    z_nm = z * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)

    cmap_cross = plt.get_cmap("viridis").copy()
    cmap_cross.set_bad("white")

    panels = [
        (tmax_sym, "Tmax (K)", "Peak temperature", "inferno", None, None),
        (
            np.ma.masked_invalid(cross_sym),
            "time (ns)",
            f"Cooling through Tm={melting_temperature:g} K",
            cmap_cross,
            None,
            None,
        ),
        (
            band_sym,
            "time (ns)",
            f"Residence in {args.band_low_k:g}-{args.band_high_k:g} K",
            "magma",
            0.0,
            None,
        ),
    ]
    for ax, (field, cbar_label, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.pcolormesh(
            r_um,
            z_nm,
            field,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()
        ax.set_xlim(-30.0, 30.0)
        ax.set_xlabel("r (um)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=cbar_label)
    axes[0].set_ylabel("z in GST (nm)")

    fig.suptitle(
        f"Thermal diagnostics, {args.energy_nj:g} nJ, "
        f"{args.film_nm:g} nm GST / {args.substrate}, "
        f"{'phase change on' if not args.no_phase_change else 'constant GST properties'}"
    )
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    substrate = substrate_from_args(args)
    model = AxisymmetricThermalModel(
        substrate=substrate,
        pulse=LaserPulse(energy=args.energy_nj * 1e-9),
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
        end_time=args.end_ns * 1e-9,
        time_step=args.dt_ns * 1e-9,
        progress=True,
    )

    film_indices = np.flatnonzero(result.film_mask)
    film_temperature = result.temperature[:, film_indices, :]
    tmax = np.nanmax(film_temperature, axis=0)
    nz, nr = tmax.shape
    tm_cross_time = np.full((nz, nr), np.nan, dtype=float)
    band_time = np.zeros((nz, nr), dtype=float)

    total = nz * nr
    done = 0
    for iz in range(nz):
        for ir in range(nr):
            trace = film_temperature[:, iz, ir]
            tm_cross_time[iz, ir] = downward_crossing_time(
                result.time,
                trace,
                model.gst.melting_temperature,
            )
            band_time[iz, ir] = residence_time_in_band(
                result.time,
                trace,
                args.band_low_k,
                args.band_high_k,
            )
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"Diagnostics: {done}/{total}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_maps(
        args.output,
        result.r,
        result.z[film_indices],
        tmax,
        tm_cross_time,
        band_time,
        args,
        model.gst.melting_temperature,
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        r_m=result.r,
        z_m=result.z[film_indices],
        tmax_K=tmax,
        tm_cross_time_s=tm_cross_time,
        residence_time_s=band_time,
        band_low_K=args.band_low_k,
        band_high_K=args.band_high_k,
        melting_temperature_K=model.gst.melting_temperature,
        energy_nJ=args.energy_nj,
    )
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {args.output.with_suffix('.npz').resolve()}")


if __name__ == "__main__":
    main()
