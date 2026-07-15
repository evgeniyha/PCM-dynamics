"""Digitize approximate temperature profiles from the article Fig. 4 screenshot.

This is a diagnostic helper for ``compare_article_temperature_profile.py``.
It reads the screenshot used in the conversation, uses the colorbar in panel
(b) as a color-to-temperature lookup table, and extracts:

* a vertical center-line profile T(z) near r=0;
* a lateral near-surface profile T(r).

The digitization is approximate because the screenshot contains labels,
markers, antialiasing and raster interpolation.  The script therefore samples
small windows and rejects pixels whose color is too far from the colorbar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plot_utils import savefig_overwrite


DEFAULT_IMAGE = Path("Screenshot 2026-07-15 121007.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=Path("output/digitized_article_temperature"))
    parser.add_argument("--distance-threshold", type=float, default=90.0)
    parser.add_argument(
        "--plateau-tm-k",
        type=float,
        default=880.0,
        help="corrected temperature assigned to the red melting plateau",
    )
    parser.add_argument(
        "--plateau-raw-low-k",
        type=float,
        default=760.0,
        help="lower raw digitized temperature of the red plateau correction band",
    )
    parser.add_argument(
        "--plateau-raw-high-k",
        type=float,
        default=850.0,
        help="upper raw digitized temperature of the red plateau correction band",
    )
    return parser.parse_args()


def nearest_colorbar_temperature(
    pixels: np.ndarray,
    colorbar_rgb: np.ndarray,
    colorbar_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = pixels.reshape(-1, 3).astype(float)
    distances2 = np.sum((p[:, None, :] - colorbar_rgb[None, :, :]) ** 2, axis=2)
    idx = np.argmin(distances2, axis=1)
    return colorbar_t[idx], np.sqrt(distances2[np.arange(len(p)), idx])


def window_temperatures(
    image: np.ndarray,
    x: int,
    y: int,
    *,
    half_width: int,
    half_height: int,
    colorbar_rgb: np.ndarray,
    colorbar_t: np.ndarray,
    distance_threshold: float,
) -> tuple[float, float, int]:
    height, width, _ = image.shape
    xs = np.arange(max(0, x - half_width), min(width, x + half_width + 1))
    ys = np.arange(max(0, y - half_height), min(height, y + half_height + 1))
    pixels = image[ys[:, None], xs[None, :]].reshape(-1, 3)
    temperatures, distances = nearest_colorbar_temperature(pixels, colorbar_rgb, colorbar_t)
    valid = distances < distance_threshold
    if not np.any(valid):
        return np.nan, np.nan, 0
    tv = temperatures[valid]
    # Median is robust to labels/markers; p75 is useful where markers darken the center.
    return float(np.median(tv)), float(np.percentile(tv, 75)), int(valid.sum())


def correct_plateau_temperature(
    temperature_k: float,
    *,
    raw_low_k: float,
    raw_high_k: float,
    plateau_tm_k: float,
) -> float:
    """Map the red raster plateau to Tm without shifting the rest of the scale."""
    if not np.isfinite(temperature_k):
        return temperature_k
    if raw_low_k <= temperature_k <= raw_high_k:
        return plateau_tm_k
    return temperature_k


def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=float)

    # Pixel geometry for the supplied screenshot.  These values are taken from
    # axes/tick positions in the raster image.
    panel_left = 218.0
    panel_right = 798.0
    panel_top = 543.0
    panel_bottom = 1007.0
    r_min_um = -50.0
    r_max_um = 50.0
    z_min_nm = 0.0
    z_max_nm = 230.0
    x_center = 0.5 * (panel_left + panel_right)
    px_per_um = (panel_right - panel_left) / (r_max_um - r_min_um)
    px_per_nm = (panel_bottom - panel_top) / (z_max_nm - z_min_nm)

    # Colorbar geometry.  Tick labels show 1500 K at y~350 and 500 K at y~584,
    # which extrapolates to about 300 K at the dark bottom.
    y_colorbar = np.arange(543, 1008)
    x_colorbar = np.arange(166, 189)
    colorbar_rgb = np.median(image[y_colorbar[:, None], x_colorbar[None, :]], axis=1)
    colorbar_t = 1500.0 - (y_colorbar - 550.0) * (1000.0 / (930.0 - 550.0))

    z_samples_nm = np.linspace(0.0, 230.0, 40)
    vertical_rows: list[tuple[float, float, float, float, float, int]] = []
    for z_nm in z_samples_nm:
        x = int(round(x_center))
        y = int(round(panel_top + z_nm * px_per_nm))
        x = int(np.clip(x, panel_left + 5, panel_right - 5))
        y = int(np.clip(y, panel_top + 5, panel_bottom - 5))
        t_med, t_p75, n_valid = window_temperatures(
            image,
            x,
            y,
            half_width=18,
            half_height=2,
            colorbar_rgb=colorbar_rgb,
            colorbar_t=colorbar_t,
            distance_threshold=args.distance_threshold,
        )
        t_med_corrected = correct_plateau_temperature(
            t_med,
            raw_low_k=args.plateau_raw_low_k,
            raw_high_k=args.plateau_raw_high_k,
            plateau_tm_k=args.plateau_tm_k,
        )
        t_p75_corrected = correct_plateau_temperature(
            t_p75,
            raw_low_k=args.plateau_raw_low_k,
            raw_high_k=args.plateau_raw_high_k,
            plateau_tm_k=args.plateau_tm_k,
        )
        vertical_rows.append((z_nm, t_med_corrected, t_med, t_p75_corrected, t_p75, n_valid))

    r_samples_um = np.linspace(-50.0, 50.0, 40)
    lateral_rows: list[tuple[float, float, float, float, float, int]] = []
    # The surface rows contain colored A/B/C markers and the "(b)" label.
    # Sampling below them avoids reading annotation pixels as temperatures.
    lateral_z_nm = 45.0
    y_lateral = int(round(panel_top + lateral_z_nm * px_per_nm))
    for r_um in r_samples_um:
        x = int(round(x_center + r_um * px_per_um))
        x = int(np.clip(x, panel_left + 5, panel_right - 5))
        y_lateral_safe = int(np.clip(y_lateral, panel_top + 5, panel_bottom - 5))
        t_med, t_p75, n_valid = window_temperatures(
            image,
            x,
            y_lateral_safe,
            half_width=2,
            half_height=2,
            colorbar_rgb=colorbar_rgb,
            colorbar_t=colorbar_t,
            distance_threshold=args.distance_threshold,
        )
        t_med_corrected = correct_plateau_temperature(
            t_med,
            raw_low_k=args.plateau_raw_low_k,
            raw_high_k=args.plateau_raw_high_k,
            plateau_tm_k=args.plateau_tm_k,
        )
        t_p75_corrected = correct_plateau_temperature(
            t_p75,
            raw_low_k=args.plateau_raw_low_k,
            raw_high_k=args.plateau_raw_high_k,
            plateau_tm_k=args.plateau_tm_k,
        )
        lateral_rows.append((r_um, t_med_corrected, t_med, t_p75_corrected, t_p75, n_valid))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertical_csv = args.output_dir / "vertical_centerline_from_image.csv"
    lateral_csv = args.output_dir / "lateral_surface_from_image.csv"
    np.savetxt(
        vertical_csv,
        np.asarray(vertical_rows),
        delimiter=",",
        header="z_nm,T_corrected_K,T_raw_median_K,T_p75_corrected_K,T_raw_p75_K,n_valid_pixels",
        comments="",
    )
    np.savetxt(
        lateral_csv,
        np.asarray(lateral_rows),
        delimiter=",",
        header="r_um,T_corrected_K,T_raw_median_K,T_p75_corrected_K,T_raw_p75_K,n_valid_pixels",
        comments="",
    )
    print(f"Saved {vertical_csv.resolve()}")
    print(f"Saved {lateral_csv.resolve()}")

    print("\nVertical center-line, corrected is used by default:")
    for row in vertical_rows:
        print(
            f"z={row[0]:6.1f} nm  "
            f"T_corr={row[1]:7.1f} K  T_raw={row[2]:7.1f} K  "
            f"T_p75_corr={row[3]:7.1f} K  n={row[5]}"
        )

    print(f"\nLateral profile at z~{lateral_z_nm:g} nm:")
    for row in lateral_rows:
        print(
            f"r={row[0]:6.1f} um  "
            f"T_corr={row[1]:7.1f} K  T_raw={row[2]:7.1f} K  "
            f"T_p75_corr={row[3]:7.1f} K  n={row[5]}"
        )

    fig, (ax_z, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    v = np.asarray(vertical_rows)
    l = np.asarray(lateral_rows)
    ax_z.plot(v[:, 1], v[:, 0], "o-", label="corrected median")
    ax_z.plot(v[:, 2], v[:, 0], "o:", label="raw median")
    ax_z.plot(v[:, 3], v[:, 0], "s--", label="corrected p75")
    ax_z.invert_yaxis()
    ax_z.set_xlabel("Temperature, K")
    ax_z.set_ylabel("z, nm")
    ax_z.set_title("Digitized vertical profile")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend()

    ax_r.plot(l[:, 0], l[:, 1], "o-", label="corrected median")
    ax_r.plot(l[:, 0], l[:, 2], "o:", label="raw median")
    ax_r.plot(l[:, 0], l[:, 3], "s--", label="corrected p75")
    ax_r.set_xlabel("r, um")
    ax_r.set_ylabel("Temperature, K")
    ax_r.set_title(f"Digitized lateral profile at z~{lateral_z_nm:g} nm")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()
    fig.tight_layout()
    savefig_overwrite(fig, args.output_dir / "digitized_profiles.png", dpi=180)


if __name__ == "__main__":
    main()
