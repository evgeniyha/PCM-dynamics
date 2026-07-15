"""Spyder-friendly launcher for convergence_study.py.

Open this file in Spyder, edit the parameters below, and press Run.
The calculation results and figures will be written to OUTPUT.
"""

from __future__ import annotations

import sys
from pathlib import Path

from convergence_study import main


# ---------------------------------------------------------------------------
# Parameters to edit in Spyder
# ---------------------------------------------------------------------------

# Which convergence group to run:
#   "spatial" - refine dr/dz
#   "time"    - refine dt
#   "domain"  - increase radial domain size
#   "all"     - run all groups
MODE = "spatial"

# True = fast smoke test; False = normal cases from convergence_study.py.
QUICK = False

# Pulse and sample parameters.
ENERGY_NJ = 650.0
FILM_NM = 230.0
SUBSTRATE = "silica"  # "silica" or "tungsten"

# Simulation time interval.
END_NS = 550.0

# True = include melting/freezing latent heat, as in the article model.
# False = pure heat conduction with constant amorphous GST properties.
PHASE_CHANGE = True

# Convergence criterion for temperature metrics.
TOLERANCE_PERCENT = 2.0

# Output folder. Figures include:
#   convergence_metrics.png
#   surface_temperature_profiles.png
#   laser_absorption_radial_profiles.png
#   vertical_center_profiles.png
OUTPUT = Path("output/convergence_spyder")


def build_arguments() -> list[str]:
    """Convert the editable variables above to convergence_study.py arguments."""
    arguments = [
        "convergence_study.py",
        "--mode", MODE,
        "--energy-nj", str(ENERGY_NJ),
        "--film-nm", str(FILM_NM),
        "--substrate", SUBSTRATE,
        "--end-ns", str(END_NS),
        "--tolerance-percent", str(TOLERANCE_PERCENT),
        "--output", str(OUTPUT),
    ]
    if QUICK:
        arguments.append("--quick")
    if not PHASE_CHANGE:
        arguments.append("--no-phase-change")
    return arguments


if __name__ == "__main__":
    old_argv = sys.argv
    try:
        sys.argv = build_arguments()
        main()
    finally:
        sys.argv = old_argv
