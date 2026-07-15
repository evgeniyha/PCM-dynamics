r"""Plot convergence of the dimensionless bracket in article Eq. (28).

This script does not use material parameters.  It isolates the dimensionless
argument

    x = t / tau_l

and compares the Eq. (28) bracket for different truncation lengths of the
infinite series.  It is useful for checking whether the summation is stable
over the time range used in TTT calculations.

Run:

    .\.venv\Scripts\python.exe eq28_convergence_diagnostic.py
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

from gst_model import CrystallizationKinetics
from plot_utils import savefig_overwrite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/eq28_diagnostic"))
    parser.add_argument("--x-min", type=float, default=1.0e-4)
    parser.add_argument("--x-max", type=float, default=1.0e2)
    parser.add_argument("--n-x", type=int, default=600)
    parser.add_argument("--terms", type=int, nargs="+", default=[5, 10, 20, 50, 120, 300, 800])
    parser.add_argument(
        "--reference-terms",
        type=int,
        default=2000,
        help="series length used as a numerical reference",
    )
    return parser.parse_args()


def eq28_bracket(x: np.ndarray, terms: int) -> np.ndarray:
    return np.array(
        [CrystallizationKinetics._article_eq28_bracket(float(value), terms=terms) for value in x],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    if args.x_min < 0.0 or args.x_max <= 0.0:
        raise SystemExit("Eq. (28) requires non-negative time: use x_min >= 0 and x_max > 0")
    if args.x_min == 0.0:
        # A logarithmic x-axis cannot include exactly zero.  The physical point
        # x=0 has B(0)=0 and Xc=0, but plotting starts from a small positive x.
        args.x_min = np.nextafter(0.0, 1.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = np.logspace(np.log10(args.x_min), np.log10(args.x_max), args.n_x)
    reference = eq28_bracket(x, args.reference_terms)

    fig_b, ax_b = plt.subplots(figsize=(6.6, 4.8))
    fig_err, ax_err = plt.subplots(figsize=(6.6, 4.8))
    fig_xc, ax_xc = plt.subplots(figsize=(6.6, 4.8))

    rows = [x]
    names = ["x_t_over_tau"]
    for terms in args.terms:
        values = eq28_bracket(x, terms)
        rel_error = np.abs(values - reference) / np.maximum(np.abs(reference), 1e-300)
        pseudo_xc = -np.expm1(-np.clip(values, 0.0, 750.0))

        rows.extend([values, rel_error, pseudo_xc])
        names.extend([
            f"B_terms_{terms}",
            f"relative_error_terms_{terms}",
            f"pseudo_Xc_terms_{terms}",
        ])

        ax_b.plot(x, values, label=f"{terms} terms")
        ax_err.plot(x, rel_error, label=f"{terms} terms")
        ax_xc.plot(x, pseudo_xc, label=f"{terms} terms")

    ax_b.plot(x, reference, "k--", lw=1.8, label=f"reference {args.reference_terms} terms")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$x=t/\tau_l$")
    ax_b.set_ylabel(r"Eq. (28) bracket $B(x)$")
    ax_b.set_title("Eq. (28) bracket convergence")
    ax_b.grid(True, which="both", alpha=0.25)
    ax_b.legend(fontsize=8)
    fig_b.tight_layout()
    savefig_overwrite(fig_b, args.output_dir / "eq28_bracket_convergence.png", dpi=180)
    plt.close(fig_b)

    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel(r"$x=t/\tau_l$")
    ax_err.set_ylabel("relative error vs reference")
    ax_err.set_title(f"Eq. (28) truncation error vs {args.reference_terms} terms")
    ax_err.grid(True, which="both", alpha=0.25)
    ax_err.legend(fontsize=8)
    fig_err.tight_layout()
    savefig_overwrite(fig_err, args.output_dir / "eq28_relative_error.png", dpi=180)
    plt.close(fig_err)

    ax_xc.set_xscale("log")
    ax_xc.set_xlabel(r"$x=t/\tau_l$")
    ax_xc.set_ylabel(r"pseudo $X_c=1-\exp[-B(x)]$")
    ax_xc.set_ylim(0.0, 1.02)
    ax_xc.set_title("Dimensionless crystallinity shape from Eq. (28)")
    ax_xc.grid(True, which="both", alpha=0.25)
    ax_xc.legend(fontsize=8)
    fig_xc.tight_layout()
    savefig_overwrite(fig_xc, args.output_dir / "eq28_pseudo_crystallinity.png", dpi=180)
    plt.close(fig_xc)

    csv_path = args.output_dir / "eq28_convergence.csv"
    np.savetxt(
        csv_path,
        np.column_stack(rows),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    print(f"Saved {csv_path.resolve()}")
    print(f"Saved {(args.output_dir / 'eq28_bracket_convergence.png').resolve()}")
    print(f"Saved {(args.output_dir / 'eq28_relative_error.png').resolve()}")
    print(f"Saved {(args.output_dir / 'eq28_pseudo_crystallinity.png').resolve()}")


if __name__ == "__main__":
    main()
