"""Convert an existing crystallinity profile NPZ file to MATLAB MAT format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("output/crystallinity_profile_digitized_ttt.npz"),
        help="profile NPZ file produced by run_article_case*.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output MAT file; default is input path with .mat suffix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".mat")
    data = np.load(args.input, allow_pickle=True)

    r_m = data["r_m"]
    z_m = data["z_m"]
    crystallinity = data["crystallinity"]
    r_symmetric_um = np.concatenate((-r_m[::-1], r_m)) * 1e6
    crystallinity_symmetric = np.concatenate(
        (crystallinity[:, ::-1], crystallinity),
        axis=1,
    )

    mat = {
        "r_m": r_m,
        "z_m": z_m,
        "r_um": r_m * 1e6,
        "z_nm": z_m * 1e9,
        "r_symmetric_um": r_symmetric_um,
        "crystallinity": crystallinity,
        "crystallinity_symmetric": crystallinity_symmetric,
    }
    for key in (
        "melted",
        "unresolved_melt",
        "peak_temperature_K",
        "cooling_rate_K_per_s",
        "method",
        "pulse_energy_nJ",
    ):
        if key in data.files:
            value = data[key]
            if value.dtype == object:
                value = str(value.item())
            mat[key] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, mat)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
