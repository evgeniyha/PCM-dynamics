r"""Compare formula-based GST225 TTT curves with WPD-digitized article curves.

Open this file in Spyder and press Run, or execute:

    .\.venv\Scripts\python.exe ttt_diagnostics.py

The script does not run the heat solver. It compares the TTT curves calculated
from ``CrystallizationKinetics`` with WebPlotDigitizer contours loaded from
``output/wpd_datasets.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gst_model import CrystallizationKinetics, GST225
from plot_utils import savefig_overwrite
from wpd_ttt_curves import WpdTTT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpd", type=Path, default=Path("output/wpd_datasets.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ttt_diagnostics"))
    parser.add_argument("--wpd-time-unit", choices=("s", "ns", "us", "ms"), default="s")
    parser.add_argument("--wpd-smooth", type=float, default=0.001)
    parser.add_argument("--wpd-samples", type=int, default=320)
    parser.add_argument("--fragility", type=float, default=67.0)
    parser.add_argument("--temperature-min-k", type=float, default=460.0)
    parser.add_argument("--temperature-max-k", type=float, default=900.0)
    parser.add_argument("--n-temperatures", type=int, default=260)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_time_on_wpd_temperatures(
    kinetics: CrystallizationKinetics,
    level: float,
    wpd_time_s: np.ndarray,
    wpd_temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_time_s = kinetics.ttt_curve(float(level), wpd_temperature_k)
    valid = (
        np.isfinite(wpd_time_s)
        & np.isfinite(wpd_temperature_k)
        & np.isfinite(model_time_s)
        & (wpd_time_s > 0.0)
        & (model_time_s > 0.0)
    )
    return model_time_s, valid


def compare_to_wpd(
    kinetics: CrystallizationKinetics,
    wpd: WpdTTT,
) -> list[dict]:
    rows: list[dict] = []
    for level in wpd.levels:
        wpd_time_s, wpd_temperature_k = wpd.curve_points(float(level))
        model_time_s, valid = model_time_on_wpd_temperatures(
            kinetics,
            float(level),
            wpd_time_s,
            wpd_temperature_k,
        )
        if not np.any(valid):
            rows.append({
                "Xc": float(level),
                "n_points": 0,
                "rms_log10_time_error": np.nan,
                "geometric_time_error_factor": np.nan,
                "median_model_over_wpd_time": np.nan,
                "wpd_min_time_ns": np.nan,
                "model_min_time_ns": np.nan,
                "wpd_nose_temperature_K": np.nan,
                "model_time_at_wpd_nose_ns": np.nan,
            })
            continue

        log_ratio = np.log10(model_time_s[valid] / wpd_time_s[valid])
        nose_index = int(np.nanargmin(wpd_time_s))
        rows.append({
            "Xc": float(level),
            "n_points": int(np.count_nonzero(valid)),
            "rms_log10_time_error": float(np.sqrt(np.mean(log_ratio**2))),
            "geometric_time_error_factor": float(10.0 ** np.sqrt(np.mean(log_ratio**2))),
            "median_model_over_wpd_time": float(np.median(model_time_s[valid] / wpd_time_s[valid])),
            "wpd_min_time_ns": float(np.nanmin(wpd_time_s) * 1e9),
            "model_min_time_ns": float(np.nanmin(model_time_s[valid]) * 1e9),
            "wpd_nose_temperature_K": float(wpd_temperature_k[nose_index]),
            "model_time_at_wpd_nose_ns": float(model_time_s[nose_index] * 1e9)
            if np.isfinite(model_time_s[nose_index])
            else np.nan,
        })
    return rows


def plot_ttt_comparison(
    path: Path,
    kinetics: CrystallizationKinetics,
    wpd: WpdTTT,
    temperatures: np.ndarray,
    fragility: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    model_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(wpd.levels)))
    wpd_colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(wpd.levels)))
    for model_color, wpd_color, level in zip(model_colors, wpd_colors, wpd.levels):
        wpd_time_s, wpd_temperature_k = wpd.curve_points(float(level))
        model_time_s = kinetics.ttt_curve(float(level), temperatures)
        finite = np.isfinite(model_time_s)
        ax.plot(
            wpd_time_s * 1e9,
            wpd_temperature_k,
            "s",
            ms=2.4,
            color=wpd_color,
            alpha=0.75,
            label=f"WPD X={level:g}",
        )
        ax.plot(
            model_time_s[finite] * 1e9,
            temperatures[finite],
            "-",
            lw=1.7,
            color=model_color,
            label=f"model X={level:g}",
        )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 1.0e4)
    ax.set_ylim(460.0, 900.0)
    ax.set_xlabel("time, ns")
    ax.set_ylabel("Temperature, K")
    ax.set_title(f"Formula TTT vs WPD digitized contours, m={fragility:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_overwrite(fig, path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wpd = WpdTTT.from_directory(
        args.wpd,
        time_unit=args.wpd_time_unit,
        smooth=args.wpd_smooth,
        samples=args.wpd_samples,
    )
    kinetics = CrystallizationKinetics(replace(GST225(), fragility=args.fragility))
    temperatures = np.linspace(
        args.temperature_min_k,
        args.temperature_max_k,
        args.n_temperatures,
    )

    comparison_rows = compare_to_wpd(kinetics, wpd)
    write_csv(args.output_dir / "ttt_vs_wpd_summary.csv", comparison_rows)
    plot_ttt_comparison(
        args.output_dir / "ttt_vs_wpd.png",
        kinetics,
        wpd,
        temperatures,
        args.fragility,
    )

    print(f"Saved {args.output_dir / 'ttt_vs_wpd_summary.csv'}")
    print(f"Saved {args.output_dir / 'ttt_vs_wpd.png'}")

    print()
    print(f"Formula TTT vs WPD, m={args.fragility:g}:")
    for row in comparison_rows:
        print(
            f"Xc={row['Xc']:g}: "
            f"error factor={row['geometric_time_error_factor']:.3g}, "
            f"median model/WPD={row['median_model_over_wpd_time']:.3g}, "
            f"WPD nose={row['wpd_min_time_ns']:.3g} ns"
        )


if __name__ == "__main__":
    main()
