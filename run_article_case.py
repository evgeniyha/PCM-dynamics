"""Run a compact reproduction of the 650 nJ / 230 nm article case."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from gst_model import (
    AxisymmetricThermalModel,
    CrystallizationKinetics,
    Grid,
    LaserPulse,
    Substrate,
    estimate_linear_cooling_rate,
)
from plot_utils import savefig_overwrite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-nj", type=float, default=650.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--end-ns", type=float, default=850.0)
    parser.add_argument("--dt-ns", type=float, default=0.5)
    parser.add_argument("--no-phase-change", action="store_true",
                        help="disable melting/freezing latent heat; use constant amorphous GST properties")
    parser.add_argument(
        "--crystallinity-method",
        choices=("cooling_rate", "path_intersection"),
        default="path_intersection",
        help="method for converting thermal histories to final crystallinity",
    )
    parser.add_argument("--output", type=Path, default=Path("output/article_case.png"))
    parser.add_argument(
        "--output-4c", type=Path, default=Path("output/figure_4c_melt_edge.png"),
        help="standalone cooling trace plus TTT curves, analogous to Fig. 4c",
    )
    parser.add_argument(
        "--output-profile", type=Path,
        default=Path("output/crystallinity_profile.png"),
        help="2D crystallinity map analogous to Fig. 5",
    )
    return parser.parse_args()


def time_since_downward_crossing(
    time: np.ndarray, temperature: np.ndarray, threshold: float
) -> np.ndarray:
    """Shift time so t=0 is the downward crossing of ``threshold``."""
    below = np.flatnonzero(temperature <= threshold)
    if not len(below):
        return time - time[0]
    i = int(below[0])
    if i == 0:
        crossing = time[0]
    else:
        fraction = ((temperature[i - 1] - threshold)
                    / (temperature[i - 1] - temperature[i]))
        crossing = time[i - 1] + fraction * (time[i] - time[i - 1])
    return time - crossing


def save_figure_4c(
    path: Path,
    result,
    trace: np.ndarray,
    trace_radius: float,
    kinetics: CrystallizationKinetics,
) -> None:
    """Plot a melt-boundary cooling curve over the calculated TTT family."""
    temperatures = np.linspace(430.0, 870.0, 180)
    levels = (1e-8, 0.03, 0.1, 0.25, 0.5, 0.95)
    shifted_time = time_since_downward_crossing(
        result.time, trace, kinetics.gst.melting_temperature
    )
    valid = shifted_time > 0.0

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    grey = np.linspace(0.15, 0.75, len(levels))
    for shade, level in zip(grey, levels):
        ttt = kinetics.ttt_curve(level, temperatures)
        ax.plot(ttt * 1e9, temperatures, color=str(shade), lw=1.4,
                label=f"$X_c={level:g}$")
    ax.plot(shifted_time[valid] * 1e9, trace[valid], color="#0072B2", lw=2.4,
            label=f"melt-edge cooling, r={trace_radius * 1e6:.1f} um")
    ax.axhline(kinetics.gst.melting_temperature, color="0.45", ls="--", lw=1,
               label=f"$T_m={kinetics.gst.melting_temperature:.0f}$ K")
    ax.set_xscale("log")
    ax.set_xlim(0.1, max(1e4, shifted_time[valid][-1] * 1e9 * 1.2))
    ax.set_ylim(420, 900)
    ax.set_xlabel("time after cooling through $T_m$ (ns)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Cooling at the melt boundary and TTT curves")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)


def save_crystallinity_profile(path: Path, profile, energy_nj: float) -> None:
    """Save a symmetric r-z section analogous to Fig. 5 of the paper."""
    r_symmetric = np.concatenate((-profile.r[::-1], profile.r)) * 1e6
    xc_symmetric = np.concatenate(
        (profile.crystallinity[:, ::-1], profile.crystallinity), axis=1
    )
    maximum = float(np.max(profile.crystallinity))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.pcolormesh(
        r_symmetric, profile.z * 1e9, xc_symmetric,
        shading="nearest", cmap="gray", vmin=0.0, vmax=1.0,
    )
    ax.invert_yaxis()
    ax.set_xlim(-30.0, 30.0)
    ax.set_xlabel("r (um)")
    ax.set_ylabel("z in GST (nm)")
    ax.set_title(
        f"Calculated crystalline fraction, {energy_nj:g} nJ "
        f"($X_{{c,max}}={maximum:.3g}$, {profile.method})"
    )
    fig.colorbar(image, ax=ax, label="$X_c$")
    fig.tight_layout()
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)
    np.savez_compressed(
        path.with_suffix(".npz"),
        r_m=profile.r,
        z_m=profile.z,
        crystallinity=profile.crystallinity,
        melted=profile.melted,
        unresolved_melt=profile.unresolved_melt,
        peak_temperature_K=profile.peak_temperature,
        cooling_rate_K_per_s=profile.cooling_rate,
        method=profile.method,
        pulse_energy_nJ=energy_nj,
    )
    savemat(
        path.with_suffix(".mat"),
        {
            "r_m": profile.r,
            "z_m": profile.z,
            "r_um": profile.r * 1e6,
            "z_nm": profile.z * 1e9,
            "r_symmetric_um": r_symmetric,
            "crystallinity": profile.crystallinity,
            "crystallinity_symmetric": xc_symmetric,
            "melted": profile.melted.astype(np.uint8),
            "unresolved_melt": profile.unresolved_melt.astype(np.uint8),
            "peak_temperature_K": profile.peak_temperature,
            "cooling_rate_K_per_s": (
                np.asarray(profile.cooling_rate)
                if profile.cooling_rate is not None
                else np.asarray([])
            ),
            "method": str(profile.method),
            "pulse_energy_nJ": float(energy_nj),
        },
    )


def main() -> None:
    args = parse_args()
    substrate = Substrate.silica() if args.substrate == "silica" else Substrate.tungsten()
    model = AxisymmetricThermalModel(
        substrate=substrate,
        pulse=LaserPulse(energy=args.energy_nj * 1e-9),
        # grid=Grid(film_thickness=args.film_nm * 1e-9),
        grid=Grid(
            film_thickness=args.film_nm * 1e-9,
            radial_extent=60e-6,
            dr=0.5e-6,
            dz_film=5e-9,
            dz_substrate=20e-9,
        ),
        include_phase_change=not args.no_phase_change,
    )
    result = model.simulate_cooling(
        end_time=args.end_ns * 1e-9,
        time_step=args.dt_ns * 1e-9,
        progress=True,
    )
    kinetics = CrystallizationKinetics(model.gst)

    # The paper evaluates cooling traces at spot boundaries, not at the hottest
    # centre. Select the outermost surface cell that reaches Tm.
    surface_peak = result.temperature[0, 0]
    molten = np.flatnonzero(surface_peak >= model.gst.melting_temperature)
    trace_index = int(molten[-1]) if len(molten) else int(np.argmax(surface_peak))
    trace_radius = result.r[trace_index]
    trace = result.temperature[:, 0, trace_index]
    rate = estimate_linear_cooling_rate(result.time, trace)
    xc = kinetics.crystallinity_from_cooling_rate(rate) if np.isfinite(rate) else np.nan

    print("Building TTT curves and figures...")
    temperatures = np.linspace(430.0, 870.0, 120)
    levels = (1e-8, 0.03, 0.1, 0.5, 0.95)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    image = axes[0].pcolormesh(
        result.r * 1e6, result.z * 1e9, result.temperature[0], shading="auto"
    )
    axes[0].invert_yaxis()
    axes[0].set(xlabel="r (um)", ylabel="z (nm)", title="Temperature at pulse end")
    fig.colorbar(image, ax=axes[0], label="T (K)")

    axes[1].plot(result.time * 1e9, trace)
    axes[1].axhline(model.gst.melting_temperature, color="0.5", ls="--")
    axes[1].set(
        xlabel="time (ns)", ylabel="T (K)",
        title=f"Cooling trace at melt edge, r={trace_radius * 1e6:.1f} um",
    )

    for level in levels:
        times = kinetics.ttt_curve(level, temperatures)
        axes[2].plot(times * 1e9, temperatures, label=f"X={level:g}")
    axes[2].set_xscale("log")
    axes[2].set(xlabel="time (ns)", ylabel="T (K)", title="Calculated TTT curves")
    axes[2].legend(fontsize=8)
    axes[2].set_xlim(0.1, 1e4)

    fig.suptitle(
        f"{args.film_nm:g} nm GST / {substrate.name}, {args.energy_nj:g} nJ; "
        f"cooling={rate / 1e9:.2g} K/ns, estimated X={xc:.3g}; "
        f"{'phase change on' if not args.no_phase_change else 'constant amorphous properties'}"
    )
    fig.tight_layout()
    savefig_overwrite(fig, args.output, dpi=180)
    plt.close(fig)
    save_figure_4c(args.output_4c, result, trace, trace_radius, kinetics)
    profile = kinetics.crystallinity_profile(
        result,
        method=args.crystallinity_method,
        progress=True,
    )
    save_crystallinity_profile(args.output_profile, profile, args.energy_nj)
    unresolved = int(np.count_nonzero(profile.unresolved_melt))
    if unresolved:
        print(
            f"WARNING: {unresolved} melted cells have not cooled through Tm; "
            "increase --end-ns for a final profile."
        )
    print(f"Pulse-end centre temperature: {surface_peak[0]:.1f} K")
    print(f"Selected melt-edge radius: {trace_radius * 1e6:.2f} um")
    print(f"Fitted cooling rate: {rate / 1e9:.3g} K/ns")
    print(f"TTT-mapped crystallinity: {xc:.4g}")


if __name__ == "__main__":
    main()
