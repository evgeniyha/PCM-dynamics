"""Digitize and fit the fluence profile from the article screenshot.

The screenshot is assumed to show a Gaussian fluence curve with:
    horizontal grid spacing = 20 um
    vertical grid spacing   = 10 mJ/cm^2

The fitted model is
    F(r) = F0 * exp(-2 * ((r - r0) / w)^2)
where w is the 1/e^2 intensity radius.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit


IMAGE_PATH = Path("Screenshot 2026-07-15 121007.png")
OUTPUT_DIR = Path("output") / "fluence_fit_from_image"

X_GRID_STEP_UM = 20.0
Y_GRID_STEP_MJ_CM2 = 10.0


def _group_positions(mask_1d: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Return centers of contiguous True regions in a 1D mask."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return np.array([], dtype=float)

    breaks = np.flatnonzero(np.diff(idx) > max_gap) + 1
    groups = np.split(idx, breaks)
    return np.array([g.mean() for g in groups if g.size > 0], dtype=float)


def _gaussian_um(r_um: np.ndarray, f0: float, r0_um: float, w_um: float) -> np.ndarray:
    return f0 * np.exp(-2.0 * ((r_um - r0_um) / w_um) ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE_PATH,
        help="article screenshot containing the fluence panel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for digitized points and fit plot",
    )
    parser.add_argument("--panel-left", type=int, default=185)
    parser.add_argument("--panel-right", type=int, default=800)
    parser.add_argument("--panel-top", type=int, default=70)
    parser.add_argument("--panel-bottom", type=int, default=505)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)
    panel_mask = np.zeros(red.shape, dtype=bool)
    panel_mask[
        max(0, args.panel_top):min(red.shape[0], args.panel_bottom),
        max(0, args.panel_left):min(red.shape[1], args.panel_right),
    ] = True

    # Red curve mask.  This intentionally ignores pale grid/text colors.
    red_curve = (
        (red > 145)
        & (green < 130)
        & (blue < 130)
        & ((red - green) > 55)
        & ((red - blue) > 55)
        & panel_mask
    )
    y_red, x_red = np.nonzero(red_curve)
    if x_red.size < 20:
        raise RuntimeError("Too few red pixels found; adjust red curve threshold.")

    # Use gray grid lines for calibration.
    # The screenshot has light gray grid lines and a darker frame/text.
    grayish = (
        (np.abs(red - green) < 8)
        & (np.abs(red - blue) < 8)
        & (red > 155)
        & (red < 235)
    )

    # Restrict grid detection to the data-plot neighborhood, not labels.
    x_left_guess = max(0, args.panel_left)
    x_right_guess = min(rgb.shape[1], args.panel_right)
    y_top_guess = max(0, args.panel_top)
    y_bottom_guess = min(rgb.shape[0], args.panel_bottom)

    col_counts = grayish[y_top_guess:y_bottom_guess, :].sum(axis=0)
    col_threshold = max(20, 0.25 * (y_bottom_guess - y_top_guess))
    x_grid = _group_positions(col_counts > col_threshold)
    x_grid = x_grid[(x_grid > x_left_guess) & (x_grid < x_right_guess)]

    row_counts = grayish[:, x_left_guess:x_right_guess].sum(axis=1)
    row_threshold = max(20, 0.25 * (x_right_guess - x_left_guess))
    y_grid = _group_positions(row_counts > row_threshold)
    y_grid = y_grid[(y_grid > y_top_guess) & (y_grid < y_bottom_guess)]

    if x_grid.size < 2:
        raise RuntimeError(f"Need at least two vertical grid lines, found {x_grid}.")
    if y_grid.size < 2:
        raise RuntimeError(f"Need at least two horizontal grid lines, found {y_grid}.")

    x_spacing_px = float(np.median(np.diff(np.sort(x_grid))))
    y_spacing_px = float(np.median(np.diff(np.sort(y_grid))))

    # In the current article screenshot the lowest detected horizontal line is
    # the 0 mJ/cm^2 axis.  Older cropped screenshots sometimes missed this
    # darker frame line, but using the current full screenshot is more robust.
    y_zero_px = float(np.max(y_grid))

    # For x=0 use the symmetry center of the red curve; this avoids needing
    # absolute x tick labels from the cropped screenshot.
    # Initial estimate: x at maximum fluence.
    x0_px_guess = float(np.median(x_red[y_red <= np.percentile(y_red, 2.0)]))

    # Collapse red pixels into one digitized point per image column.
    points = []
    for x_px in np.unique(x_red):
        ys = y_red[x_red == x_px]
        if ys.size:
            y_px = float(np.median(ys))
            r_um = (float(x_px) - x0_px_guess) * X_GRID_STEP_UM / x_spacing_px
            f_mj_cm2 = (y_zero_px - y_px) * Y_GRID_STEP_MJ_CM2 / y_spacing_px
            if f_mj_cm2 >= 0.0:
                points.append((r_um, f_mj_cm2, float(x_px), y_px))

    data = np.array(points, dtype=float)
    r_um = data[:, 0]
    f_mj_cm2 = data[:, 1]

    # Fit the digitized curve.
    p0 = [float(f_mj_cm2.max()), 0.0, 35.0]
    bounds = ([0.0, -10.0, 5.0], [100.0, 10.0, 100.0])
    popt, pcov = curve_fit(_gaussian_um, r_um, f_mj_cm2, p0=p0, bounds=bounds)
    f0, r0_um, w_um = popt
    perr = np.sqrt(np.diag(pcov))

    fit = _gaussian_um(r_um, *popt)
    rmse = float(np.sqrt(np.mean((fit - f_mj_cm2) ** 2)))

    # Convert fitted radius back to the expected peak fluence at 650 nJ.
    energy_j = 650e-9
    f0_from_energy_j_m2 = 2.0 * energy_j / (np.pi * (w_um * 1e-6) ** 2)
    f0_from_energy_mj_cm2 = f0_from_energy_j_m2 * 0.1

    out_csv = output_dir / "digitized_fluence.csv"
    np.savetxt(
        out_csv,
        np.column_stack([r_um, f_mj_cm2]),
        delimiter=",",
        header="r_um,fluence_mJ_cm2",
        comments="",
    )

    order = np.argsort(r_um)
    r_plot = np.linspace(r_um.min(), r_um.max(), 500)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.plot(r_um[order], f_mj_cm2[order], ".", ms=3, label="digitized red curve")
    ax.plot(r_plot, _gaussian_um(r_plot, *popt), "-", lw=2, label="Gaussian fit")
    ax.set_xlabel("r, um")
    ax.set_ylabel("Fluence, mJ/cm$^2$")
    ax.grid(True, alpha=0.35)
    ax.legend()
    ax.set_title("Fluence profile digitized from article screenshot")
    fit_text = (
        f"$F_0$ = {f0:.2f} mJ/cm$^2$\n"
        f"$w$ = {w_um:.2f} $\\mu$m\n"
        f"$r_0$ = {r0_um:.2f} $\\mu$m\n"
        f"RMSE = {rmse:.3f} mJ/cm$^2$\n"
        f"$F_0$(650 nJ, w) = {f0_from_energy_mj_cm2:.2f}"
    )
    ax.text(
        0.03,
        0.97,
        fit_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.5", "alpha": 0.88},
    )
    fig.tight_layout()
    out_png = output_dir / "fluence_fit.png"
    fig.savefig(out_png)
    plt.close(fig)

    print(f"image: {args.image}")
    print(f"vertical grid lines px: {np.round(x_grid, 1).tolist()}")
    print(f"horizontal grid lines px: {np.round(y_grid, 1).tolist()}")
    print(f"x grid spacing: {x_spacing_px:.3f} px = {X_GRID_STEP_UM:g} um")
    print(f"y grid spacing: {y_spacing_px:.3f} px = {Y_GRID_STEP_MJ_CM2:g} mJ/cm^2")
    print(f"F0 fit: {f0:.3f} ± {perr[0]:.3f} mJ/cm^2")
    print(f"r0 fit: {r0_um:.3f} ± {perr[1]:.3f} um")
    print(f"w fit: {w_um:.3f} ± {perr[2]:.3f} um  (1/e^2 radius)")
    print(f"RMSE: {rmse:.3f} mJ/cm^2")
    print(f"F0 implied by 650 nJ and fitted w: {f0_from_energy_mj_cm2:.3f} mJ/cm^2")
    print(f"csv: {out_csv}")
    print(f"plot: {out_png}")


if __name__ == "__main__":
    main()
