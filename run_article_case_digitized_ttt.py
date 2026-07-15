"""Run the article thermal case using digitized TTT curves from the figure.

This script is intentionally separate from ``run_article_case.py``.  It uses the
same heat-transfer model, but converts thermal histories to crystallinity using
digitized TTT contours instead of recalculating TTT curves from nucleation/growth
equations.

It uses the WebPlotDigitizer export ``output/wpd_datasets.csv``.
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

from gst_model import (
    AxisymmetricThermalModel,
    Grid,
    LaserPulse,
    Substrate,
    estimate_linear_cooling_rate,
)
from plot_utils import savefig_overwrite
from run_article_case import save_crystallinity_profile, time_since_downward_crossing
from wpd_ttt_curves import WpdTTT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-nj", type=float, default=650.0)
    parser.add_argument("--film-nm", type=float, default=230.0)
    parser.add_argument("--substrate", choices=("silica", "tungsten"), default="silica")
    parser.add_argument("--end-ns", type=float, default=450.0)
    parser.add_argument("--dt-ns", type=float, default=0.5)
    parser.add_argument(
        "--no-phase-change",
        action="store_true",
        help="disable melting/freezing latent heat; use constant amorphous GST properties",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/article_case_digitized_ttt.png"),
    )
    parser.add_argument(
        "--output-4c",
        type=Path,
        default=Path("output/figure_4c_digitized_ttt.png"),
        help="standalone cooling trace plus digitized TTT curves",
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=Path("output/crystallinity_profile_digitized_ttt.png"),
        help="2D crystallinity map using digitized TTT curves",
    )
    parser.add_argument(
        "--output-ttt",
        type=Path,
        default=Path("output/wpd_ttt/wpd_ttt_curves.png"),
        help="control plot of the digitized TTT curves",
    )
    parser.add_argument(
        "--output-traces",
        type=Path,
        default=Path("output/cooling_traces_digitized_ttt.png"),
        help="cooling curves at article-like A-F points over digitized TTT curves",
    )
    parser.add_argument(
        "--wpd-dir",
        type=Path,
        default=Path("output/wpd_datasets.csv"),
        help="WebPlotDigitizer CSV/JSON file or folder used by default for TTT contours",
    )
    parser.add_argument(
        "--wpd-time-unit",
        choices=("s", "ns", "us", "ms"),
        default="s",
        help="time unit used in WPD exported X values",
    )
    parser.add_argument(
        "--wpd-smooth",
        type=float,
        default=0.001,
        help="spline smoothing for WPD contours in log10(time)-T coordinates",
    )
    parser.add_argument(
        "--wpd-samples",
        type=int,
        default=320,
        help="number of samples per smoothed WPD contour",
    )
    parser.add_argument(
        "--interpolate-ttt",
        action="store_true",
        help="interpolate crystallinity between neighboring WPD TTT contours; disabled by default",
    )
    parser.add_argument(
        "--min-delta-t-for-crystallinity-k",
        type=float,
        default=350.0,
        help=(
            "skip crystallinity calculation in GST cells whose peak temperature "
            "rise above ambient is smaller than this value; use 0 to disable"
        ),
    )
    return parser.parse_args()


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def article_like_trace_points(result) -> list[dict[str, float | int | str]]:
    """Return points close to labels A-F in the article's Fig. 4(b,c).

    Coordinates are approximate because the original points are read from the
    figure, not from a table.  z is measured downward from the GST surface.
    """
    requested = [
        ("A", 0.0e-6, 0.0e-9, "#0072B2"),
        ("B", 12.0e-6, 0.0e-9, "#E69F00"),
        ("C", 24.0e-6, 0.0e-9, "#009E73"),
        ("D", 0.0e-6, 160.0e-9, "#FF0000"),
        ("E", 0.0e-6, 100.0e-9, "#CC79A7"),
        ("F", 0.0e-6, 70.0e-9, "#A65E4E"),
    ]
    points: list[dict[str, float | int | str]] = []
    for label, r_target, z_target, color in requested:
        ir = nearest_index(result.r, r_target)
        iz = nearest_index(result.z, z_target)
        points.append({
            "label": label,
            "r_target": r_target,
            "z_target": z_target,
            "r": float(result.r[ir]),
            "z": float(result.z[iz]),
            "ir": ir,
            "iz": iz,
            "color": color,
        })
    return points


def trace_time_for_crystallization(
    time: np.ndarray,
    trace: np.ndarray,
    melting_temperature: float,
) -> np.ndarray:
    """Article-style time axis for one cooling trace.

    For melted cells, t=0 is the downward crossing of Tm.  For cells that never
    melted, t=0 is pulse end.
    """
    if np.nanmax(trace) >= melting_temperature:
        return time_since_downward_crossing(time, trace, melting_temperature)
    return time - time[0]


def save_digitized_figure_4c(
    path: Path,
    result,
    trace: np.ndarray,
    trace_radius: float,
    ttt: WpdTTT,
    melting_temperature: float,
) -> None:
    """Plot a melt-boundary cooling curve over the digitized TTT family."""
    temperatures = np.linspace(460.0, 900.0, 260)
    shifted_time = time_since_downward_crossing(result.time, trace, melting_temperature)
    valid = shifted_time > 0.0

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    grey = np.linspace(0.08, 0.75, len(ttt.levels))
    for shade, level in zip(grey, ttt.levels):
        if hasattr(ttt, "curve_points"):
            time_s, temperature_k = ttt.curve_points(float(level))
            ax.plot(
                time_s * 1e9,
                temperature_k,
                color=str(shade),
                lw=1.6,
                label=f"$X_c={level:g}$",
            )
        else:
            required = ttt.ttt_curve(float(level), temperatures)
            finite = np.isfinite(required)
            ax.plot(
                required[finite] * 1e9,
                temperatures[finite],
                color=str(shade),
                lw=1.6,
                label=f"$X_c={level:g}$",
            )

    ax.plot(
        shifted_time[valid] * 1e9,
        trace[valid],
        color="#0072B2",
        lw=2.4,
        label=f"melt-edge cooling, r={trace_radius * 1e6:.1f} um",
    )
    ax.axhline(
        melting_temperature,
        color="0.45",
        ls="--",
        lw=1,
        label=f"$T_m={melting_temperature:.0f}$ K",
    )
    ax.set_xscale("log")
    ax.set_xlim(5.0, 300.0)
    ax.set_ylim(460.0, 900.0)
    ax.set_xlabel("time after cooling through $T_m$ (ns)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Cooling at the melt boundary and digitized TTT curves")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)


def save_article_like_cooling_traces(
    path: Path,
    result,
    points: list[dict[str, float | int | str]],
    ttt: WpdTTT,
    melting_temperature: float,
) -> None:
    """Plot article-like A-F cooling traces over digitized TTT curves."""
    temperatures = np.linspace(460.0, 900.0, 260)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    grey = np.linspace(0.08, 0.78, len(ttt.levels))
    for shade, level in zip(grey, ttt.levels):
        if hasattr(ttt, "curve_points"):
            time_s, temperature_k = ttt.curve_points(float(level))
            ax.plot(
                time_s,
                temperature_k,
                color=str(shade),
                lw=1.4,
                label=f"{level:g}",
            )
        else:
            required = ttt.ttt_curve(float(level), temperatures)
            finite = np.isfinite(required)
            ax.plot(
                required[finite],
                temperatures[finite],
                color=str(shade),
                lw=1.4,
                label=f"{level:g}",
            )

    for point in points:
        iz = int(point["iz"])
        ir = int(point["ir"])
        trace = result.temperature[:, iz, ir]
        shifted_time = trace_time_for_crystallization(
            result.time,
            trace,
            melting_temperature,
        )
        valid = (shifted_time > 0.0) & np.isfinite(trace)
        color = str(point["color"])
        label = str(point["label"])
        ax.plot(
            shifted_time[valid],
            trace[valid],
            "--",
            lw=1.8,
            color=color,
            label=(
                f"{label}: r={float(point['r']) * 1e6:.1f} um, "
                f"z={float(point['z']) * 1e9:.0f} nm"
            ),
        )
        if np.any(valid):
            label_index = np.flatnonzero(valid)[min(8, np.count_nonzero(valid) - 1)]
            ax.text(
                shifted_time[label_index],
                trace[label_index],
                label,
                color=color,
                fontsize=11,
                weight="bold",
            )

    ax.set_xscale("log")
    ax.set_xlim(5.0e-9, 2.5e-7)
    ax.set_ylim(460.0, 900.0)
    ax.set_xlabel("time, s")
    ax.set_ylabel("Temperature, K")
    ax.set_title("Article-like cooling curves over digitized TTT contours")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(title="TTT / traces", fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    substrate = Substrate.silica() if args.substrate == "silica" else Substrate.tungsten()
    ttt = WpdTTT.from_directory(
        args.wpd_dir,
        time_unit=args.wpd_time_unit,
        smooth=args.wpd_smooth,
        samples=args.wpd_samples,
    )
    ttt_label = "WPD-TTT"

    model = AxisymmetricThermalModel(
        substrate=substrate,
        pulse=LaserPulse(energy=args.energy_nj * 1e-9),
        grid=Grid(
            film_thickness=args.film_nm * 1e-9,
            radial_extent=60e-6,
            dr=0.5e-6,
            dz_film=10e-9,
            dz_substrate=20e-9,
        ),
        include_phase_change=not args.no_phase_change,
    )
    result = model.simulate_cooling(
        end_time=args.end_ns * 1e-9,
        time_step=args.dt_ns * 1e-9,
        progress=True,
    )
    max_temperature_all = float(np.nanmax(result.temperature))
    max_temperature_final = float(np.nanmax(result.temperature[-1]))

    # The paper evaluates cooling traces at spot boundaries, not at the hottest
    # centre. Select the outermost surface cell that reaches Tm.
    surface_peak = result.temperature[0, 0]
    molten = np.flatnonzero(surface_peak >= model.gst.melting_temperature)
    trace_index = int(molten[-1]) if len(molten) else int(np.argmax(surface_peak))
    trace_radius = result.r[trace_index]
    trace = result.temperature[:, 0, trace_index]
    rate = estimate_linear_cooling_rate(result.time, trace)

    shifted_time = time_since_downward_crossing(
        result.time, trace, model.gst.melting_temperature
    )
    valid = shifted_time > 0.0
    trace_xc = ttt.crystallinity_from_cooling_curve(
        shifted_time[valid],
        trace[valid],
        interpolate=args.interpolate_ttt,
    )
    trace_points = article_like_trace_points(result)

    print("Building figures with digitized TTT curves...")
    ttt.plot(args.output_ttt)

    temperatures = np.linspace(460.0, 900.0, 260)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    image = axes[0].pcolormesh(
        result.r * 1e6,
        result.z * 1e9,
        result.temperature[0],
        shading="auto",
    )
    axes[0].invert_yaxis()
    axes[0].set(xlabel="r (um)", ylabel="z (nm)", title="Temperature at pulse end")
    for point in trace_points:
        r_um = float(point["r"]) * 1e6
        z_nm = float(point["z"]) * 1e9
        color = str(point["color"])
        label = str(point["label"])
        axes[0].plot(r_um, z_nm, "o", ms=4.5, color=color, mec="black", mew=0.5)
        axes[0].text(r_um + 1.2, z_nm + 8.0, label, color="black", fontsize=9)
    fig.colorbar(image, ax=axes[0], label="T (K)")

    axes[1].plot(result.time * 1e9, trace)
    axes[1].axhline(model.gst.melting_temperature, color="0.5", ls="--")
    axes[1].set(
        xlabel="time (ns)",
        ylabel="T (K)",
        title=f"Cooling trace at melt edge, r={trace_radius * 1e6:.1f} um",
    )

    for level in ttt.levels:
        if hasattr(ttt, "curve_points"):
            time_s, temperature_k = ttt.curve_points(float(level))
            axes[2].plot(time_s * 1e9, temperature_k, label=f"X={level:g}")
        else:
            required = ttt.ttt_curve(float(level), temperatures)
            finite = np.isfinite(required)
            axes[2].plot(required[finite] * 1e9, temperatures[finite], label=f"X={level:g}")
    axes[2].set_xscale("log")
    axes[2].set(
        xlabel="time (ns)",
        ylabel="T (K)",
        title="Digitized TTT curves",
    )
    axes[2].legend(fontsize=8)
    axes[2].set_xlim(5.0, 300.0)
    axes[2].set_ylim(460.0, 900.0)

    fig.suptitle(
        f"{args.film_nm:g} nm GST / {substrate.name}, {args.energy_nj:g} nJ; "
        f"cooling={rate / 1e9:.2g} K/ns, {ttt_label} X={trace_xc:.3g}; "
        f"{'phase change on' if not args.no_phase_change else 'constant amorphous properties'}"
    )
    fig.tight_layout()
    savefig_overwrite(fig, args.output, dpi=180)
    plt.close(fig)

    save_digitized_figure_4c(
        args.output_4c,
        result,
        trace,
        trace_radius,
        ttt,
        model.gst.melting_temperature,
    )
    save_article_like_cooling_traces(
        args.output_traces,
        result,
        trace_points,
        ttt,
        model.gst.melting_temperature,
    )
    profile_kwargs = {"result": result, "gst": model.gst, "progress": True}
    profile_kwargs["interpolate"] = args.interpolate_ttt
    profile_kwargs["min_temperature_rise"] = args.min_delta_t_for_crystallinity_k
    profile = ttt.crystallinity_profile(**profile_kwargs)
    if args.min_delta_t_for_crystallinity_k > 0.0:
        cold_mask = (profile.peak_temperature - model.ambient) < args.min_delta_t_for_crystallinity_k
        skipped = int(np.count_nonzero(cold_mask))
        print(
            f"Crystallinity threshold: skipped {skipped} GST cells with "
            f"Delta T_max < {args.min_delta_t_for_crystallinity_k:g} K"
        )
    save_crystallinity_profile(args.output_profile, profile, args.energy_nj)

    unresolved = int(np.count_nonzero(profile.unresolved_melt))
    if unresolved:
        print(
            f"WARNING: {unresolved} melted cells have not cooled through Tm; "
            "increase --end-ns for a final profile."
        )
    print(f"Pulse-end centre temperature: {surface_peak[0]:.1f} K")
    print(f"Maximum temperature over saved calculation: {max_temperature_all:.1f} K")
    print(f"Maximum temperature at final time ({result.time[-1] * 1e9:.3g} ns): {max_temperature_final:.1f} K")
    print(f"Selected melt-edge radius: {trace_radius * 1e6:.2f} um")
    print(f"Fitted cooling rate: {rate / 1e9:.3g} K/ns")
    print(f"{ttt_label} crystallinity at selected trace: {trace_xc:.4g}")


if __name__ == "__main__":
    main()
