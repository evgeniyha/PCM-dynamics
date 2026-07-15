r"""Diagnostics for GST225 TTT curves against the article scale.

Open this file in Spyder and press Run, or execute:

    .\.venv\Scripts\python.exe ttt_diagnostics.py

The script does not run the heat solver. It only inspects the crystallization
kinetics: TTT curves, nose positions, and critical cooling rates.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Parameters to edit in Spyder
# ---------------------------------------------------------------------------

OUTPUT = Path("output/ttt_diagnostics")

# Article Fig. 3-like levels visible on the published TTT diagram.
ARTICLE_LEVELS = np.array([0.023, 0.058, 0.125, 0.165, 0.3])

# Extra levels used in our run_article_case.py figure.
COMMON_LEVELS = np.array([1e-8, 0.03, 0.1, 0.5, 0.95])

# Temperature window for plotting/comparison.
TEMPERATURE_MIN_K = 600.0
TEMPERATURE_MAX_K = 850.0
N_TEMPERATURES = 160

# Fragility values to compare on the TTT plots.
# The article reports best match at m = 67.
FRAGILITY_VALUES = [67.0]

# Automatic scan ranges for matching article critical-rate anchors.
SCAN_FRAGILITY = True
SCAN_M_MIN = 30.0
SCAN_M_MAX = 140.0
SCAN_M_POINTS = 16

SCAN_INTERFACE_ENERGY = False
SCAN_SIGMA_MIN = 0.052
SCAN_SIGMA_MAX = 0.061
SCAN_SIGMA_POINTS = 37

# Approximate reference critical cooling rates quoted in the article text.
# These are not exact digitized curve values; they are sanity-check anchors.
ARTICLE_REFERENCE_RATES_K_PER_NS = {
    1e-8: 316.0,
    0.03: 18.0,
    0.1: 6.4,
    0.95: 3.8,
}


def finite_min_time_ns(times_s: np.ndarray) -> float:
    finite = times_s[np.isfinite(times_s)]
    return float(np.min(finite) * 1e9) if finite.size else np.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_article_scale(
    path: Path,
    kinetics_by_label: dict[str, CrystallizationKinetics],
    temperatures: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    shades = np.linspace(0.15, 0.75, len(ARTICLE_LEVELS))
    for label, kinetics in kinetics_by_label.items():
        for shade, level in zip(shades, ARTICLE_LEVELS):
            times_ns = kinetics.ttt_curve(float(level), temperatures) * 1e9
            ax.plot(
                times_ns,
                temperatures,
                color=str(shade),
                lw=1.5,
                label=f"{label}, X={level:g}",
            )
    ax.set_xlim(10, 80)
    ax.set_ylim(600, 850)
    ax.set_xlabel("time, ns")
    ax.set_ylabel("Temperature, K")
    ax.set_title("TTT curves on article-like axes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_log_scale(
    path: Path,
    kinetics_by_label: dict[str, CrystallizationKinetics],
    temperatures: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    for label, kinetics in kinetics_by_label.items():
        for level in COMMON_LEVELS:
            times_ns = kinetics.ttt_curve(float(level), temperatures) * 1e9
            ax.plot(times_ns, temperatures, lw=1.6, label=f"{label}, X={level:g}")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(420, 880)
    ax.set_xlabel("time, ns")
    ax.set_ylabel("Temperature, K")
    ax.set_title("TTT curves, wide log scale")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def collect_summary(
    kinetics_by_label: dict[str, CrystallizationKinetics],
    temperatures: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    levels = np.unique(np.concatenate((ARTICLE_LEVELS, COMMON_LEVELS)))
    for label, kinetics in kinetics_by_label.items():
        for level in levels:
            curve = kinetics.ttt_curve(float(level), temperatures)
            critical_rate, tangent_temperature, tangent_time = kinetics.critical_cooling_rate(
                float(level)
            )
            reference = ARTICLE_REFERENCE_RATES_K_PER_NS.get(float(level), np.nan)
            rows.append({
                "case": label,
                "Xc": float(level),
                "min_time_in_plot_window_ns": finite_min_time_ns(curve),
                "critical_rate_K_per_ns": critical_rate / 1e9,
                "article_reference_K_per_ns": reference,
                "ratio_model_to_article": (
                    critical_rate / 1e9 / reference
                    if np.isfinite(reference) and reference > 0
                    else np.nan
                ),
                "tangent_temperature_K": tangent_temperature,
                "tangent_time_ns": tangent_time * 1e9,
            })
    return rows


def scan_fragility() -> list[dict]:
    """Compare critical cooling-rate anchors across many fragility values."""
    rows: list[dict] = []
    scan_values = np.linspace(SCAN_M_MIN, SCAN_M_MAX, SCAN_M_POINTS)
    anchors = sorted(ARTICLE_REFERENCE_RATES_K_PER_NS)
    for fragility in scan_values:
        kinetics = CrystallizationKinetics(replace(GST225(), fragility=float(fragility)))
        squared_log_errors = []
        rates = {}
        for level in anchors:
            model_rate = kinetics.critical_cooling_rate(float(level))[0] / 1e9
            reference = ARTICLE_REFERENCE_RATES_K_PER_NS[level]
            rates[level] = model_rate
            squared_log_errors.append(np.log(model_rate / reference) ** 2)
        rows.append({
            "fragility": float(fragility),
            "rms_log_error": float(np.sqrt(np.mean(squared_log_errors))),
            "geometric_error_factor": float(np.exp(np.sqrt(np.mean(squared_log_errors)))),
            **{f"rate_Xc_{level:g}_K_per_ns": rates[level] for level in anchors},
        })
    return rows


def scan_interface_energy() -> list[dict]:
    """Compare critical cooling-rate anchors across interfacial energies."""
    rows: list[dict] = []
    scan_values = np.linspace(SCAN_SIGMA_MIN, SCAN_SIGMA_MAX, SCAN_SIGMA_POINTS)
    anchors = sorted(ARTICLE_REFERENCE_RATES_K_PER_NS)
    for sigma in scan_values:
        kinetics = CrystallizationKinetics(replace(GST225(), interface_energy=float(sigma)))
        squared_log_errors = []
        rates = {}
        for level in anchors:
            model_rate = kinetics.critical_cooling_rate(float(level))[0] / 1e9
            reference = ARTICLE_REFERENCE_RATES_K_PER_NS[level]
            rates[level] = model_rate
            squared_log_errors.append(np.log(model_rate / reference) ** 2)
        rows.append({
            "interface_energy_J_per_m2": float(sigma),
            "rms_log_error": float(np.sqrt(np.mean(squared_log_errors))),
            "geometric_error_factor": float(np.exp(np.sqrt(np.mean(squared_log_errors)))),
            **{f"rate_Xc_{level:g}_K_per_ns": rates[level] for level in anchors},
        })
    return rows


def plot_fragility_scan(path: Path, rows: list[dict]) -> None:
    fragility = np.asarray([row["fragility"] for row in rows])
    error_factor = np.asarray([row["geometric_error_factor"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(fragility, error_factor, "o-", ms=3)
    best = rows[int(np.argmin(error_factor))]
    ax.axvline(best["fragility"], color="0.35", ls="--", lw=1)
    ax.set_xlabel("fragility m")
    ax.set_ylabel("geometric error factor vs article anchors")
    ax.set_title(f"Fragility scan; best m≈{best['fragility']:.1f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_interface_energy_scan(path: Path, rows: list[dict]) -> None:
    sigma = np.asarray([row["interface_energy_J_per_m2"] for row in rows])
    error_factor = np.asarray([row["geometric_error_factor"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(sigma, error_factor, "o-", ms=3)
    best = rows[int(np.argmin(error_factor))]
    ax.axvline(best["interface_energy_J_per_m2"], color="0.35", ls="--", lw=1)
    ax.set_xlabel("interface energy sigma, J/m²")
    ax.set_ylabel("geometric error factor vs article anchors")
    ax.set_title(f"Sigma scan; best sigma≈{best['interface_energy_J_per_m2']:.4f} J/m²")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    temperatures = np.linspace(TEMPERATURE_MIN_K, TEMPERATURE_MAX_K, N_TEMPERATURES)
    kinetics_by_label = {
        f"m={fragility:g}": CrystallizationKinetics(
            replace(GST225(), fragility=float(fragility))
        )
        for fragility in FRAGILITY_VALUES
    }

    rows = collect_summary(kinetics_by_label, temperatures)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "ttt_summary.csv", rows)
    plot_article_scale(OUTPUT / "ttt_article_scale.png", kinetics_by_label, temperatures)
    plot_log_scale(OUTPUT / "ttt_log_scale.png", kinetics_by_label, temperatures)
    scan_rows = []
    if SCAN_FRAGILITY:
        scan_rows = scan_fragility()
        write_csv(OUTPUT / "fragility_scan.csv", scan_rows)
        plot_fragility_scan(OUTPUT / "fragility_scan.png", scan_rows)
    sigma_scan_rows = []
    if SCAN_INTERFACE_ENERGY:
        sigma_scan_rows = scan_interface_energy()
        write_csv(OUTPUT / "interface_energy_scan.csv", sigma_scan_rows)
        plot_interface_energy_scan(OUTPUT / "interface_energy_scan.png", sigma_scan_rows)

    print(f"Saved {OUTPUT / 'ttt_summary.csv'}")
    print(f"Saved {OUTPUT / 'ttt_article_scale.png'}")
    print(f"Saved {OUTPUT / 'ttt_log_scale.png'}")
    if SCAN_FRAGILITY:
        print(f"Saved {OUTPUT / 'fragility_scan.csv'}")
        print(f"Saved {OUTPUT / 'fragility_scan.png'}")
    if SCAN_INTERFACE_ENERGY:
        print(f"Saved {OUTPUT / 'interface_energy_scan.csv'}")
        print(f"Saved {OUTPUT / 'interface_energy_scan.png'}")
    print()
    print("Critical cooling-rate anchors:")
    for row in rows:
        if np.isfinite(row["article_reference_K_per_ns"]):
            print(
                f"{row['case']}, Xc={row['Xc']:g}: "
                f"model={row['critical_rate_K_per_ns']:.3g} K/ns, "
                f"article~={row['article_reference_K_per_ns']:.3g} K/ns, "
                f"ratio={row['ratio_model_to_article']:.3g}"
            )
    if scan_rows:
        best = min(scan_rows, key=lambda row: row["geometric_error_factor"])
        print()
        print(
            "Best fragility in scan: "
            f"m={best['fragility']:.2f}, "
            f"geometric error factor={best['geometric_error_factor']:.3g}"
        )
    if sigma_scan_rows:
        best = min(sigma_scan_rows, key=lambda row: row["geometric_error_factor"])
        print()
        print(
            "Best interface energy in scan: "
            f"sigma={best['interface_energy_J_per_m2']:.5f} J/m^2, "
            f"geometric error factor={best['geometric_error_factor']:.3g}"
        )


if __name__ == "__main__":
    main()
