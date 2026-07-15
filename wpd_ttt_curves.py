"""TTT contours loaded from WebPlotDigitizer datasets.

This module reads TTT contours exported from WebPlotDigitizer.
The important difference is that each TTT contour is treated as a parametric
polyline in ``log10(time)``--``temperature`` coordinates.  A C-shaped TTT
contour is not a single-valued function ``t(T)``, so crystallinity is evaluated
by geometric intersection of cooling curves with these polylines.

Supported inputs:

* one CSV per contour, with two numeric columns;
* one WPD CSV export with columns like ``Dataset, X, Y``;
* WPD JSON export containing ``datasetColl``.

The crystallinity level is inferred from the dataset name / file name.  Names
such as ``0.03``, ``X=0.03``, ``1e-4`` and ``10^-4`` are accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

from gst_model import CrystallinityProfile, GST225, ProgressBar, ThermalResult
from plot_utils import savefig_overwrite


Array = NDArray[np.float64]

DEFAULT_WPD_DIR = Path("output/wpd_datasets")
KNOWN_LEVELS = np.asarray([1.0e-4, 0.03, 0.18, 0.25, 0.5, 0.95], dtype=float)


def _parse_level(name: str) -> float | None:
    text = name.lower().replace(",", ".")
    text = text.replace("×", "x")
    text = re.sub(r"10\s*\^\s*\{?\s*(-?\d+)\s*\}?", lambda m: f"1e{m.group(1)}", text)
    text = re.sub(r"10\s*\*\*\s*(-?\d+)", lambda m: f"1e{m.group(1)}", text)
    candidates = re.findall(r"(?<![a-z])(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text)
    values: list[float] = []
    for candidate in candidates:
        try:
            values.append(float(candidate))
        except ValueError:
            pass
    if not values:
        return None
    # Prefer known article levels if the parsed number is close.
    for value in values:
        close = KNOWN_LEVELS[np.argmin(np.abs(KNOWN_LEVELS - value))]
        if np.isclose(value, close, rtol=0.08, atol=1e-5):
            return float(close)
    # Dataset names such as "Dataset 2" are WPD indices, not crystallinity.
    # If no known article level was found, leave the level unresolved and let
    # the caller map datasets by order.
    return None


def _read_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        preview = list(csv.reader(handle, dialect=dialect))
        handle.seek(0)
        if (
            len(preview) >= 2
            and len(preview[0]) >= 2
            and any(cell.strip().lower() == "x" for cell in preview[1])
            and any(cell.strip().lower() == "y" for cell in preview[1])
        ):
            grouped: dict[str, list[tuple[float, float]]] = {}
            header = preview[0]
            xy = preview[1]
            for start in range(0, min(len(header), len(xy)) - 1, 2):
                if xy[start].strip().lower() != "x" or xy[start + 1].strip().lower() != "y":
                    continue
                name = header[start].strip() or f"{path.stem}_{start // 2 + 1}"
                points: list[tuple[float, float]] = []
                for row in preview[2:]:
                    if len(row) <= start + 1:
                        continue
                    try:
                        x = float(row[start].replace(",", "."))
                        y = float(row[start + 1].replace(",", "."))
                    except ValueError:
                        continue
                    points.append((x, y))
                if points:
                    grouped[name] = points
            if grouped:
                return grouped

        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        lower = [field.lower().strip() for field in fieldnames]

        if not fieldnames:
            return {}

        dataset_col = next(
            (
                fieldnames[i]
                for i, field in enumerate(lower)
                if field in {"dataset", "dataset name", "name", "curve"}
            ),
            None,
        )
        x_col = next(
            (
                fieldnames[i]
                for i, field in enumerate(lower)
                if field in {"x", "time", "time_s", "time (s)"}
            ),
            fieldnames[-2] if len(fieldnames) >= 2 else fieldnames[0],
        )
        y_col = next(
            (
                fieldnames[i]
                for i, field in enumerate(lower)
                if field in {"y", "temperature", "temperature_k", "temperature (k)", "t"}
            ),
            fieldnames[-1],
        )

        grouped: dict[str, list[tuple[float, float]]] = {}
        default_name = path.stem
        for row in reader:
            try:
                x = float(str(row[x_col]).replace(",", "."))
                y = float(str(row[y_col]).replace(",", "."))
            except (KeyError, TypeError, ValueError):
                continue
            name = str(row.get(dataset_col, default_name)) if dataset_col else default_name
            grouped.setdefault(name, []).append((x, y))
        return grouped


def _read_json(path: Path) -> dict[str, list[tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    datasets = data.get("datasetColl") or data.get("datasets") or []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for index, dataset in enumerate(datasets):
        name = str(dataset.get("name") or dataset.get("label") or f"dataset_{index}")
        raw_points = dataset.get("data") or dataset.get("points") or []
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if isinstance(point, dict):
                x = point.get("x")
                y = point.get("y")
                if x is None and "value" in point:
                    value = point["value"]
                    if isinstance(value, list) and len(value) >= 2:
                        x, y = value[:2]
            elif isinstance(point, list) and len(point) >= 2:
                x, y = point[:2]
            else:
                continue
            try:
                points.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
        if points:
            grouped[name] = points
    return grouped


def _normalise_time(time_values: Array, unit: str) -> Array:
    unit = unit.lower()
    scale = {
        "s": 1.0,
        "sec": 1.0,
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
    }.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return time_values * scale


def _prepare_polyline(points: list[tuple[float, float]], time_unit: str) -> Array:
    raw = np.asarray(points, dtype=float)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)

    time_s = _normalise_time(raw[:, 0], time_unit)
    temperature_k = raw[:, 1]
    valid = (time_s > 0.0) & np.isfinite(temperature_k)
    time_s = time_s[valid]
    temperature_k = temperature_k[valid]
    if len(time_s) < 2:
        return np.empty((0, 2), dtype=float)

    log_time = np.log10(time_s)
    polyline = np.column_stack([log_time, temperature_k])

    # Remove exact/near duplicate neighbouring points.  WPD exports sometimes
    # contain repeated clicks.
    keep = np.ones(len(polyline), dtype=bool)
    step = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    keep[1:] = step > 1e-9
    polyline = polyline[keep]
    return polyline


def _smooth_polyline(polyline: Array, smooth: float, samples: int) -> Array:
    if len(polyline) < 4 or samples <= len(polyline):
        return polyline
    try:
        tck, _ = splprep(
            [polyline[:, 0], polyline[:, 1]],
            s=smooth,
            k=min(3, len(polyline) - 1),
        )
        u = np.linspace(0.0, 1.0, samples)
        x, y = splev(u, tck)
        return np.column_stack([x, y])
    except Exception:
        # If the spline fails because points are badly ordered, fall back to the
        # raw polyline rather than hiding data.
        return polyline


def _segments_intersect(a0: Array, a1: Array, b0: Array, b1: Array) -> bool:
    def orient(p: Array, q: Array, r: Array) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    return (o1 * o2 <= 0.0) and (o3 * o4 <= 0.0)


def _cooling_intersects_contour(cooling: Array, contour: Array) -> bool:
    """Return True if two polylines intersect in log10(time)--T space."""
    if len(cooling) < 2 or len(contour) < 2:
        return False
    for i in range(len(cooling) - 1):
        a0 = cooling[i]
        a1 = cooling[i + 1]
        low = np.minimum(a0, a1)
        high = np.maximum(a0, a1)
        candidates = (
            (np.minimum(contour[:-1, 0], contour[1:, 0]) <= high[0])
            & (np.maximum(contour[:-1, 0], contour[1:, 0]) >= low[0])
            & (np.minimum(contour[:-1, 1], contour[1:, 1]) <= high[1])
            & (np.maximum(contour[:-1, 1], contour[1:, 1]) >= low[1])
        )
        for j in np.flatnonzero(candidates):
            if _segments_intersect(a0, a1, contour[j], contour[j + 1]):
                return True
    return False


def _resample_polyline(polyline: Array, n: int) -> Array:
    """Resample a polyline uniformly in arc length."""
    if len(polyline) == n:
        return polyline
    if len(polyline) < 2:
        return polyline
    distance = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    )
    if distance[-1] <= 0.0:
        return np.repeat(polyline[:1], n, axis=0)
    target = np.linspace(0.0, distance[-1], n)
    x = np.interp(target, distance, polyline[:, 0])
    y = np.interp(target, distance, polyline[:, 1])
    return np.column_stack([x, y])


def _interpolate_contours(lower: Array, upper: Array, fraction: float) -> Array:
    """Linear interpolation between two digitized contours."""
    n = max(len(lower), len(upper), 2)
    lower_resampled = _resample_polyline(lower, n)
    upper_resampled = _resample_polyline(upper, n)
    return lower_resampled + fraction * (upper_resampled - lower_resampled)


@dataclass(frozen=True)
class WpdTTT:
    """TTT lookup based on WPD contour polylines."""

    polylines: dict[float, Array]

    @classmethod
    def from_directory(
        cls,
        directory: str | Path = DEFAULT_WPD_DIR,
        time_unit: str = "s",
        smooth: float = 0.001,
        samples: int = 320,
    ) -> "WpdTTT":
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)

        grouped: dict[str, list[tuple[float, float]]] = {}
        paths = [directory] if directory.is_file() else sorted(directory.glob("*"))
        for path in paths:
            if path.suffix.lower() == ".csv":
                grouped.update(_read_csv(path))
            elif path.suffix.lower() == ".json":
                grouped.update(_read_json(path))

        polylines: dict[float, Array] = {}
        unresolved_index = 0
        for name, points in grouped.items():
            level = _parse_level(name)
            if level is None:
                if unresolved_index >= len(KNOWN_LEVELS):
                    continue
                level = float(KNOWN_LEVELS[unresolved_index])
                unresolved_index += 1
            polyline = _prepare_polyline(points, time_unit=time_unit)
            polyline = _smooth_polyline(polyline, smooth=smooth, samples=samples)
            if len(polyline) >= 2:
                polylines[level] = polyline

        if not polylines:
            raise ValueError(f"No recognizable WPD TTT datasets found in {directory}")
        return cls(polylines)

    @property
    def levels(self) -> Array:
        return np.asarray(sorted(self.polylines), dtype=float)

    def curve_points(self, level: float) -> tuple[Array, Array]:
        polyline = self.polylines[float(level)]
        return np.power(10.0, polyline[:, 0]), polyline[:, 1]

    def ttt_curve(self, level: float, temperatures_k: Iterable[float]) -> Array:
        """Legacy helper: interpolate right branch time from T.

        The intersection method below does not use this.  It exists so plotting
        plotting helpers can treat WPD contours like the older in-memory
        digitized contours.
        """
        polyline = self.polylines[float(level)]
        right = polyline[polyline[:, 0] >= np.nanmedian(polyline[:, 0])]
        if len(right) < 2:
            right = polyline
        order = np.argsort(right[:, 1])
        temperature = np.asarray(list(temperatures_k), dtype=float)
        log_required = np.interp(
            temperature,
            right[order, 1],
            right[order, 0],
            left=np.inf,
            right=np.inf,
        )
        return np.power(10.0, log_required)

    def crystallinity_from_cooling_curve(
        self,
        elapsed_time_s: Array,
        temperature_k: Array,
        levels: Array | None = None,
        interpolate: bool = True,
        interpolation_iterations: int = 18,
    ) -> float:
        elapsed = np.asarray(elapsed_time_s, dtype=float)
        temperature = np.asarray(temperature_k, dtype=float)
        valid = np.isfinite(elapsed) & np.isfinite(temperature) & (elapsed > 0.0)
        if not np.any(valid):
            return 0.0
        cooling = np.column_stack([np.log10(elapsed[valid]), temperature[valid]])
        if len(cooling) < 2:
            return 0.0

        if levels is None:
            levels = self.levels

        levels = np.asarray(levels, dtype=float)
        hit_by_level: dict[float, bool] = {}
        for level in levels:
            contour = self.polylines[float(level)]
            hit_by_level[float(level)] = _cooling_intersects_contour(cooling, contour)

        hit_levels = [float(level) for level in levels if hit_by_level[float(level)]]
        if not hit_levels:
            return 0.0

        reached = max(hit_levels)
        if not interpolate:
            return reached

        reached_index = int(np.flatnonzero(np.isclose(levels, reached))[0])
        if reached_index >= len(levels) - 1:
            return reached

        next_level = float(levels[reached_index + 1])
        if hit_by_level[next_level]:
            return next_level

        lower = self.polylines[reached]
        upper = self.polylines[next_level]

        lo = 0.0
        hi = 1.0
        for _ in range(interpolation_iterations):
            mid = 0.5 * (lo + hi)
            candidate = _interpolate_contours(lower, upper, mid)
            if _cooling_intersects_contour(cooling, candidate):
                lo = mid
            else:
                hi = mid
        return reached + lo * (next_level - reached)
        return reached

    def crystallinity_profile(
        self,
        result: ThermalResult,
        gst: GST225 | None = None,
        levels: Array | None = None,
        max_crystallinity: float = 0.95,
        interpolate: bool = True,
        min_temperature_rise: float = 0.0,
        progress: bool = True,
    ) -> CrystallinityProfile:
        material = gst or GST225()
        if levels is None:
            levels = self.levels
        levels = np.asarray(levels, dtype=float)

        film_indices = np.flatnonzero(result.film_mask)
        film_z = result.z[film_indices]
        nr = len(result.r)

        xc = np.zeros((len(film_indices), nr), dtype=float)
        peak = result.temperature[:, film_indices, :].max(axis=0)
        if result.temperature.size:
            initial_temperature = float(np.nanmin(result.temperature[0]))
        else:
            initial_temperature = 300.0
        skip_cold = (peak - initial_temperature) < float(min_temperature_rise)
        melted = peak >= material.melting_temperature
        unresolved_melt = np.zeros_like(melted, dtype=bool)

        total = len(film_indices) * nr
        bar = ProgressBar(total, "WPD TTT Xc") if progress else None
        update_every = max(1, total // 200)
        done = 0
        time = result.time

        for local_z, iz in enumerate(film_indices):
            for ir in range(nr):
                if skip_cold[local_z, ir]:
                    done += 1
                    if bar is not None and (done % update_every == 0 or done == total):
                        bar.update(done)
                    continue

                trace = result.temperature[:, iz, ir]
                if melted[local_z, ir]:
                    below = np.flatnonzero(trace <= material.melting_temperature)
                    if below.size == 0:
                        unresolved_melt[local_z, ir] = True
                        done += 1
                        if bar is not None and (done % update_every == 0 or done == total):
                            bar.update(done)
                        continue
                    start_index = int(below[0])
                    start_time = time[start_index]
                    valid_indices = np.arange(start_index, len(time))
                else:
                    start_time = time[0]
                    valid_indices = np.arange(len(time))

                elapsed = time[valid_indices] - start_time
                xc[local_z, ir] = min(
                    self.crystallinity_from_cooling_curve(
                        elapsed,
                        trace[valid_indices],
                        levels=levels,
                        interpolate=interpolate,
                    ),
                    max_crystallinity,
                )

                done += 1
                if bar is not None and (done % update_every == 0 or done == total):
                    bar.update(done)

        return CrystallinityProfile(
            crystallinity=xc,
            r=result.r,
            z=film_z,
            melted=melted,
            unresolved_melt=unresolved_melt,
            peak_temperature=peak,
            cooling_rate=None,
            method="wpd_ttt",
        )

    def plot(self, output_path: str | Path = "output/wpd_ttt/wpd_ttt_curves.png") -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5.2, 5.8), dpi=160)
        colors = ["black", "0.25", "0.4", "0.6", "0.72", "0.84", "C0", "C1"]
        for color, level in zip(colors, self.levels):
            time_s, temperature_k = self.curve_points(float(level))
            ax.plot(time_s, temperature_k, "-", lw=1.7, color=color, label=f"{level:g}")
            ax.plot(time_s, temperature_k, ".", ms=1.5, color=color, alpha=0.5)

        ax.set_xscale("log")
        ax.set_xlim(5e-9, 2.5e-7)
        ax.set_ylim(460.0, 900.0)
        ax.set_xlabel("time, s")
        ax.set_ylabel("Temperature, K")
        ax.set_title("WPD-interpolated article TTT contours")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(title="Crystallinity")
        fig.tight_layout()
        savefig_overwrite(fig, output)
        plt.close(fig)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wpd-dir", type=Path, default=DEFAULT_WPD_DIR)
    parser.add_argument("--time-unit", choices=("s", "ns", "us", "ms"), default="s")
    parser.add_argument("--smooth", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("output/wpd_ttt/wpd_ttt_curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ttt = WpdTTT.from_directory(
        args.wpd_dir,
        time_unit=args.time_unit,
        smooth=args.smooth,
        samples=args.samples,
    )
    output = ttt.plot(args.output)
    print(f"Saved WPD TTT plot: {output}")
    print("Available crystallinity contours:", ", ".join(f"{x:g}" for x in ttt.levels))


if __name__ == "__main__":
    main()
