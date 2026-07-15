"""Thermal and crystallization model for amorphous Ge2Sb2Te5 (GST225).

This module implements the model described by Kunkel et al., Materials
Science in Semiconductor Processing 139 (2022) 106350, DOI:
10.1016/j.mssp.2021.106350.

The implementation contains:
  * an axisymmetric finite-volume heat solver for a GST film/substrate stack;
  * Beer-Lambert absorption of a Gaussian femtosecond pulse;
  * optional latent heat represented by an effective heat capacity;
  * classical transient nucleation and crystal-growth kinetics;
  * isothermal TTT curves and the paper's tangent construction for converting
    cooling rate to final crystallinity.

SI units are used throughout. The original paper calibrates alpha_eff and the
fragility index m against experiment; the defaults below are those calibrated
values, not universal GST constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import sys
import time
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Avogadro, Boltzmann, electron_volt, pi
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve


Array = NDArray[np.float64]


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(self, total: int, label: str = "Calculating", width: int = 32) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.started = time.perf_counter()
        self.last_length = 0

    def update(self, current: int) -> None:
        current = min(max(int(current), 0), self.total)
        fraction = current / self.total
        filled = round(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started
        text = (
            f"\r{self.label}: [{bar}] {100.0 * fraction:5.1f}% "
            f"({current}/{self.total}) {elapsed:6.1f} s"
        )
        padding = " " * max(0, self.last_length - len(text))
        sys.stdout.write(text + padding)
        sys.stdout.flush()
        self.last_length = len(text)
        if current == self.total:
            sys.stdout.write("\n")


@dataclass(frozen=True)
class GST225:
    """GST225 parameters from Table 1 of the paper, converted to SI."""

    thermal_conductivity: float = 0.25  # W/(m K)
    specific_heat: float = 212.0  # J/(kg K)
    density: float = 5870.0  # kg/m^3
    reflectivity: float = 0.428
    alpha_linear: float = 10.0e6  # 1/m; tabulated amorphous GST absorption at 1030 nm
    alpha_eff: float = 16.2e6  # 1/m; effective Beer-Lambert attenuation alpha + alpha_f
    melting_temperature: float = 880.0  # K (890 K is used once in the text)
    glass_temperature: float = 420.0  # K
    eta_infinity: float = 1.1e-5  # Pa s
    fragility: float = 67.0  # fitted in the paper
    fusion_enthalpy_volume: float = 670e6  # J/m^3
    interface_energy: float = 0.063  # J/m^2
    viscosity_activation_energy: float = 2.0 * electron_volt  # J
    molar_mass_formula: float = 1.02678  # kg/mol of Ge2Sb2Te5
    atoms_per_formula: int = 9
    melt_smoothing_width: float = 5  # K, numerical regularisation

    @property
    def monomer_volume(self) -> float:
        """Average atomic monomer volume [m^3].

        The article treats a monomer as an arbitrary atom of the GST225
        composition, so ``v_m`` is estimated from the density and the average
        atomic molar mass, not from the full Ge2Sb2Te5 formula-unit volume.
        """
        mean_atomic_molar_mass = self.molar_mass_formula / self.atoms_per_formula
        return mean_atomic_molar_mass / (self.density * Avogadro)

    @property
    def fusion_enthalpy_mass(self) -> float:
        return self.fusion_enthalpy_volume / self.density

    def melt_fraction(self, temperature: Array | float) -> Array:
        x = (np.asarray(temperature, dtype=float) - self.melting_temperature)
        x /= self.melt_smoothing_width
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def effective_specific_heat(self, temperature: Array | float) -> Array:
        f = self.melt_fraction(temperature)
        df_dT = f * (1.0 - f) / self.melt_smoothing_width
        return self.specific_heat + self.fusion_enthalpy_mass * df_dT

    def enthalpy_above(self, temperature: Array | float, reference: float) -> Array:
        """Specific enthalpy relative to ``reference`` [J/kg]."""
        t = np.asarray(temperature, dtype=float)
        return (
            self.specific_heat * (t - reference)
            + self.fusion_enthalpy_mass
            * (self.melt_fraction(t) - self.melt_fraction(reference))
        )

    def temperature_from_enthalpy(self, enthalpy: Array, reference: float) -> Array:
        """Invert the regularised enthalpy relation with vectorised bisection."""
        h = np.asarray(enthalpy, dtype=float)
        lo = np.full_like(h, min(100.0, reference))
        hi = np.maximum(reference + h / self.specific_heat + 200.0, 1500.0)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            below = self.enthalpy_above(mid, reference) < h
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Substrate:
    name: str
    thermal_conductivity: float  # W/(m K)
    density: float  # kg/m^3
    specific_heat: float  # J/(kg K)
    thickness: float  # m

    @property
    def diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @classmethod
    def silica(cls, thickness: float = 1e-6) -> "Substrate":
        # k and diffusivity match Table 1; rho*c follows from k/a.
        k, a, rho = 1.45, 9.2e-7, 2200.0
        return cls("SiO2", k, rho, k / (a * rho), thickness)

    @classmethod
    def tungsten(cls, thickness: float = 0.5e-6) -> "Substrate":
        # Effective homogeneous conductive substrate used by the paper.
        k, a, rho = 163.0, 6.3e-5, 19300.0
        return cls("W effective", k, rho, k / (a * rho), thickness)


@dataclass(frozen=True)
class LaserPulse:
    energy: float = 650e-9  # J
    duration: float = 185e-15  # s
    wavelength: float = 1030e-9  # m
    radius_1e2: float = 35e-6  # m; Gaussian 1/e^2 fluence radius


@dataclass(frozen=True)
class Grid:
    film_thickness: float = 230e-9
    radial_extent: float = 60e-6
    dr: float = 1.0e-6
    dz_film: float = 10e-9
    dz_substrate: float = 40e-9


@dataclass
class ThermalResult:
    time: Array
    temperature: Array  # shape (nt, nz, nr)
    r: Array
    z: Array
    film_mask: Array
    metadata: dict = field(default_factory=dict)

    def trace(self, r: float = 0.0, z: float = 0.0) -> Array:
        ir = int(np.argmin(np.abs(self.r - r)))
        iz = int(np.argmin(np.abs(self.z - z)))
        return self.temperature[:, iz, ir]


@dataclass
class CrystallinityProfile:
    crystallinity: Array  # shape (n_film_z, nr)
    r: Array
    z: Array
    melted: NDArray[np.bool_]
    unresolved_melt: NDArray[np.bool_]
    peak_temperature: Array
    cooling_rate: Array | None = None
    method: str = "cooling_rate"


class AxisymmetricThermalModel:
    """Finite-volume solution of Eqs. (3)-(14) in the paper.

    Heating is applied as an instantaneous enthalpy increment. During 185 fs,
    thermal diffusion in GST is below one nanometre, so this is numerically much
    cheaper than the paper's 1 fs stepping and has the same thermal limit.
    Cooling uses backward Euler and Picard iterations for latent heat.
    """

    def __init__(
        self,
        gst: GST225 | None = None,
        substrate: Substrate | None = None,
        pulse: LaserPulse | None = None,
        grid: Grid | None = None,
        ambient_temperature: float = 300.0,
        top_heat_transfer: float = 0.0,
        interface_resistance: float = 0.0,
        include_phase_change: bool = True,
    ) -> None:
        self.gst = gst or GST225()
        self.substrate = substrate or Substrate.silica()
        self.pulse = pulse or LaserPulse()
        self.grid = grid or Grid()
        self.ambient = ambient_temperature
        self.top_heat_transfer = top_heat_transfer
        self.interface_resistance = interface_resistance
        self.include_phase_change = include_phase_change
        self._build_grid()
        self._build_conduction_operator()

    def _build_grid(self) -> None:
        g = self.grid
        nr = max(2, int(np.ceil(g.radial_extent / g.dr)))
        self.r_edges = np.linspace(0.0, nr * g.dr, nr + 1)
        self.r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

        nf = max(1, int(np.ceil(g.film_thickness / g.dz_film)))
        ns = max(1, int(np.ceil(self.substrate.thickness / g.dz_substrate)))
        film_edges = np.linspace(0.0, g.film_thickness, nf + 1)
        sub_edges = g.film_thickness + np.linspace(
            self.substrate.thickness / ns, self.substrate.thickness, ns
        )
        self.z_edges = np.concatenate((film_edges, sub_edges))
        self.z = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        self.dz = np.diff(self.z_edges)
        self.film_rows = self.z < g.film_thickness
        self.nz, self.nr = len(self.z), len(self.r)

        annulus_area = pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)
        self.volume = self.dz[:, None] * annulus_area[None, :]
        self.k = np.where(
            self.film_rows[:, None],
            self.gst.thermal_conductivity,
            self.substrate.thermal_conductivity,
        ) * np.ones((self.nz, self.nr))
        self.rho = np.where(
            self.film_rows[:, None], self.gst.density, self.substrate.density
        ) * np.ones((self.nz, self.nr))

    @staticmethod
    def _series_resistance(distance_a: float, k_a: float, distance_b: float,
                           k_b: float, contact_resistance: float = 0.0) -> float:
        return distance_a / k_a + contact_resistance + distance_b / k_b

    def _build_conduction_operator(self) -> None:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diagonal = np.zeros(self.nz * self.nr)
        boundary_source = np.zeros(self.nz * self.nr)

        def idx(iz: int, ir: int) -> int:
            return iz * self.nr + ir

        def connect(a: int, b: int, conductance: float) -> None:
            rows.extend((a, b))
            cols.extend((b, a))
            data.extend((conductance, conductance))
            diagonal[a] -= conductance
            diagonal[b] -= conductance

        # Radial internal faces.
        for iz in range(self.nz):
            for ir in range(self.nr - 1):
                face_area = 2.0 * pi * self.r_edges[ir + 1] * self.dz[iz]
                resistance = self._series_resistance(
                    0.5 * (self.r[ir + 1] - self.r[ir]), self.k[iz, ir],
                    0.5 * (self.r[ir + 1] - self.r[ir]), self.k[iz, ir + 1],
                )
                connect(idx(iz, ir), idx(iz, ir + 1), face_area / resistance)

        # Axial internal faces, including optional GST/substrate TBR.
        ring_area = pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)
        for iz in range(self.nz - 1):
            is_interface = self.film_rows[iz] and not self.film_rows[iz + 1]
            rc = self.interface_resistance if is_interface else 0.0
            for ir in range(self.nr):
                resistance = self._series_resistance(
                    0.5 * self.dz[iz], self.k[iz, ir],
                    0.5 * self.dz[iz + 1], self.k[iz + 1, ir], rc,
                )
                connect(idx(iz, ir), idx(iz + 1, ir), ring_area[ir] / resistance)

        # Top convection (usually negligible); bottom held at room temperature.
        for ir in range(self.nr):
            a = idx(0, ir)
            g_top = self.top_heat_transfer * ring_area[ir]
            diagonal[a] -= g_top
            boundary_source[a] += g_top * self.ambient

            b = idx(self.nz - 1, ir)
            g_bottom = self.k[-1, ir] * ring_area[ir] / (0.5 * self.dz[-1])
            diagonal[b] -= g_bottom
            boundary_source[b] += g_bottom * self.ambient

        n = self.nz * self.nr
        rows.extend(range(n))
        cols.extend(range(n))
        data.extend(diagonal.tolist())
        self.operator: csr_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n, n)
        ).tocsr()
        self.boundary_source = boundary_source

    def absorbed_energy_density(self) -> Array:
        """Deposited energy density [J/m^3] in GST at pulse end.

        The incident fluence is normalised as

            F(r) = 2 E / (pi w^2) * exp(-2 r^2 / w^2),

        where ``w = LaserPulse.radius_1e2`` is the 1/e^2 radius.  This
        integrates to the pulse energy ``E``.  The printed Eq. (5) in the
        article omits this radial factor of two, but Fig. 4a is consistent
        with the 1/e^2 convention used here.
        """
        rr = self.r[None, :]
        zz = self.z[:, None]
        p = self.pulse
        g = self.gst
        incident_fluence = (
            2.0 * p.energy / (pi * p.radius_1e2**2)
            * np.exp(-2.0 * rr**2 / p.radius_1e2**2)
        )
        q = ((1.0 - g.reflectivity) * g.alpha_linear * incident_fluence
             * np.exp(-g.alpha_eff * zz))
        return np.where(self.film_rows[:, None], q, 0.0)

    def pulse_end_temperature(self) -> Array:
        deposited_per_mass = self.absorbed_energy_density() / self.rho
        if not self.include_phase_change:
            return self.ambient + deposited_per_mass / self._base_specific_heat_field()
        return self.gst.temperature_from_enthalpy(deposited_per_mass, self.ambient)

    def _base_specific_heat_field(self) -> Array:
        cp = np.full((self.nz, self.nr), self.substrate.specific_heat)
        cp[self.film_rows, :] = self.gst.specific_heat
        return cp

    def _specific_heat_field(self, temperature: Array) -> Array:
        if not self.include_phase_change:
            return self._base_specific_heat_field()
        cp = np.full_like(temperature, self.substrate.specific_heat)
        cp[self.film_rows, :] = self.gst.effective_specific_heat(
            temperature[self.film_rows, :]
        )
        return cp

    def simulate_cooling(
        self,
        end_time: float = 100e-9,
        time_step: float = 0.5e-9,
        save_every: int = 1,
        max_picard_iterations: int = 8,
        tolerance: float = 1e-3,
        progress: bool = False,
    ) -> ThermalResult:
        t = self.pulse_end_temperature()
        snapshots = [t.copy()]
        times = [0.0]
        n_steps = int(np.ceil(end_time / time_step))
        volume_flat = self.volume.ravel()
        progress_bar = ProgressBar(n_steps, "Thermal calculation") if progress else None

        for step in range(1, n_steps + 1):
            old = t.ravel()
            guess = old.copy()
            for _ in range(max_picard_iterations):
                cp = self._specific_heat_field(guess.reshape(self.nz, self.nr)).ravel()
                capacity_over_dt = self.rho.ravel() * cp * volume_flat / time_step
                system = diags(capacity_over_dt) - self.operator
                rhs = capacity_over_dt * old + self.boundary_source
                new = spsolve(system, rhs)
                if np.max(np.abs(new - guess)) < tolerance:
                    guess = new
                    break
                guess = 0.5 * guess + 0.5 * new
            t = guess.reshape(self.nz, self.nr)
            if step % save_every == 0 or step == n_steps:
                snapshots.append(t.copy())
                times.append(min(step * time_step, end_time))
            if progress_bar is not None:
                progress_bar.update(step)

        return ThermalResult(
            time=np.asarray(times),
            temperature=np.asarray(snapshots),
            r=self.r.copy(),
            z=self.z.copy(),
            film_mask=self.film_rows.copy(),
            metadata={
                "pulse_energy_J": self.pulse.energy,
                "film_thickness_m": self.grid.film_thickness,
                "substrate": self.substrate.name,
                "alpha_linear_1_per_m": self.gst.alpha_linear,
                "alpha_eff_1_per_m": self.gst.alpha_eff,
                "fragility": self.gst.fragility,
                "include_phase_change": self.include_phase_change,
            },
        )


class CrystallizationKinetics:
    """Eqs. (15)-(28): transient homogeneous nucleation and growth."""

    def __init__(self, gst: GST225 | None = None) -> None:
        self.gst = gst or GST225()

    def viscosity(self, temperature: float | Array) -> Array:
        g = self.gst
        t = np.asarray(temperature, dtype=float)
        log10_eta = np.empty_like(t)
        low = t < g.glass_temperature
        log10_eta[low] = 12.0 + (
            g.viscosity_activation_energy / (Boltzmann * np.log(10.0))
            * (1.0 / t[low] - 1.0 / g.glass_temperature)
        )
        c = 12.0 - np.log10(g.eta_infinity)
        high_t = t[~low]
        log10_eta[~low] = (
            np.log10(g.eta_infinity)
            + c * g.glass_temperature / high_t
            * np.exp(
                (g.fragility / c - 1.0)
                * (g.glass_temperature / high_t - 1.0)
            )
        )
        return np.power(10.0, np.clip(log10_eta, -20.0, 300.0))

    def chemical_potential(self, temperature: float) -> float:
        g = self.gst
        return (
            g.monomer_volume * g.fusion_enthalpy_volume
            * (g.melting_temperature - temperature) / g.melting_temperature
            * (2.0 * temperature / (temperature + g.melting_temperature))
        )

    def kinetic_parameters(self, temperature: float) -> tuple[float, float, float]:
        """Return nucleation lag Theta [s], J_ss [m^-3 s^-1], U [m/s]."""
        g = self.gst
        if not g.glass_temperature < temperature < g.melting_temperature:
            return np.inf, 0.0, 0.0
        mu = self.chemical_potential(temperature)
        vm = g.monomer_volume
        nc = 32.0 * pi * g.interface_energy**3 * vm**2 / (3.0 * mu**3)
        jump = vm ** (1.0 / 3.0)
        eta = float(self.viscosity(temperature))
        diffusion = Boltzmann * temperature / (3.0 * pi * jump * eta)
        gamma = 6.0 * diffusion / jump**2

        def cluster_energy(n: float) -> float:
            return (36.0 * pi) ** (1.0 / 3.0) * vm ** (2.0 / 3.0) * (
                n ** (2.0 / 3.0)
            ) * g.interface_energy - n * mu

        delta_attach = cluster_energy(nc + 1.0) - cluster_energy(nc)
        dc = 4.0 * nc ** (2.0 / 3.0) * gamma * np.exp(
            np.clip(-delta_attach / (2.0 * Boltzmann * temperature), -700, 700)
        )
        theta = 4.0 * nc * Boltzmann * temperature / (mu * dc)
        delta_gc = 16.0 * pi * g.interface_energy**3 * vm**2 / (3.0 * mu**2)

        log_j = (
            -np.log(vm) + np.log(4.0) + (2.0 / 3.0) * np.log(nc)
            + np.log(gamma)
            + 0.5 * np.log(mu / (6.0 * pi * Boltzmann * temperature * nc))
            - delta_gc / (Boltzmann * temperature)
        )
        jss = float(np.exp(np.clip(log_j, -745.0, 700.0)))
        growth = (
            4.0 * Boltzmann * temperature / (3.0 * pi * jump**2 * eta)
            * (1.0 - np.exp(-mu / (Boltzmann * temperature)))
        )
        return theta, jss, growth

    @staticmethod
    def _article_eq28_bracket(x: float, terms: int = 120) -> float:
        """Bracketed dimensionless term in the article's Eq. (28).

        The paper writes

            ln(1 - Xc) = -pi/3 * Jss * U^3 * tau_l^4 * B(t/tau_l),

        where ``B`` is the polynomial-plus-series bracket implemented here.
        The powers are pi^2, pi^4 and pi^6 in the x^3, x^2 and x terms,
        respectively.
        """
        if x <= 0.0:
            return 0.0
        n = np.arange(1, terms + 1, dtype=float)
        alternating = np.where((n.astype(int) % 2) == 0, 1.0, -1.0)
        series = np.sum(
            alternating * np.expm1(-n**2 * x) / n**8
        )
        bracket = (
            x**4
            - (2.0 * pi**2 / 3.0) * x**3
            + (7.0 * pi**4 / 30.0) * x**2
            - (31.0 * pi**6 / 630.0) * x
            + 48.0 * series
        )
        return max(float(bracket), 0.0)

    def transformed_exponent(self, temperature: float, time: float) -> float:
        theta, jss, growth = self.kinetic_parameters(temperature)
        if jss == 0.0 or growth == 0.0 or not np.isfinite(theta):
            return 0.0
        tau_l = 6.0 * theta / pi**2
        bracket = self._article_eq28_bracket(time / tau_l)
        return max((pi / 3.0) * jss * growth**3 * tau_l**4 * bracket, 0.0)

    def crystallinity(self, temperature: float, time: float) -> float:
        y = self.transformed_exponent(temperature, time)
        return float(-np.expm1(-min(y, 750.0)))

    @lru_cache(maxsize=8192)
    def ttt_time(self, temperature: float, target_crystallinity: float) -> float:
        if not 0.0 < target_crystallinity < 1.0:
            raise ValueError("target_crystallinity must be between 0 and 1")
        target = -np.log1p(-target_crystallinity)

        def residual(log_time: float) -> float:
            return self.transformed_exponent(temperature, np.exp(log_time)) - target

        lo, hi = np.log(1e-15), np.log(1e4)
        if residual(hi) < 0.0:
            return np.inf
        return float(np.exp(brentq(residual, lo, hi, maxiter=150)))

    def ttt_curve(
        self, target_crystallinity: float, temperatures: Iterable[float]
    ) -> Array:
        return np.asarray([
            self.ttt_time(float(t), float(target_crystallinity))
            for t in temperatures
        ])

    def critical_cooling_rate(
        self,
        target_crystallinity: float,
        temperatures: Array | None = None,
    ) -> tuple[float, float, float]:
        """Paper's tangent construction: return rate [K/s], nose T and time.

        For a linear cooling line starting at Tm, tangency corresponds to the
        largest secant slope (Tm-T)/t_X(T). A dense temperature grid provides a
        robust numerical approximation to the construction in Fig. 3.
        """
        g = self.gst
        if temperatures is None:
            temperatures = np.linspace(g.glass_temperature + 1.0,
                                       g.melting_temperature - 1.0, 300)
        times = self.ttt_curve(target_crystallinity, temperatures)
        rates = (g.melting_temperature - temperatures) / times
        i = int(np.nanargmax(rates))
        return float(rates[i]), float(temperatures[i]), float(times[i])

    def crystallinity_from_cooling_rate(
        self,
        cooling_rate: float,
        levels: Array | None = None,
    ) -> float:
        """Invert the article's critical-rate/TTT mapping approximately."""
        if levels is None:
            levels = np.unique(np.concatenate((
                np.logspace(-8, -2, 18), np.linspace(0.02, 0.95, 30)
            )))
        critical = np.asarray([self.critical_cooling_rate(float(x))[0] for x in levels])
        order = np.argsort(critical)
        # Critical rate decreases as the requested crystallinity rises.
        return float(np.interp(cooling_rate, critical[order], levels[order],
                               left=levels[order][0], right=levels[order][-1]))

    def crystallinity_profile(
        self,
        result: ThermalResult,
        levels: Array | None = None,
        temperature_grid: Array | None = None,
        method: str = "path_intersection",
        cooling_high: float = 850.0,
        cooling_low: float = 600.0,
        progress: bool = False,
    ) -> CrystallinityProfile:
        """Estimate Xc(r,z) from every non-isothermal thermal trajectory.

        ``method="path_intersection"`` follows the article's graphical
        construction most directly: the crystallinity formed during a cooling
        regime is determined by the TTT curve intersected by the cooling curve.
        For a melted cell, elapsed time starts when it cools through Tm; for an
        unmelted cell it starts at the end of the pulse. At every point on the
        cooling path, elapsed time is compared with the isothermal TTT time at
        the same temperature. The largest reached TTT level is assigned to the
        cell.

        ``method="cooling_rate"`` is a diagnostic approximation based on the
        critical cooling rates inferred from tangent lines to TTT curves.

        Neither method is a fully coupled non-isothermal population-balance
        calculation.
        """
        if method not in {"cooling_rate", "path_intersection"}:
            raise ValueError("method must be 'cooling_rate' or 'path_intersection'")
        if levels is None:
            levels = np.asarray([
                1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2,
                0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.25, 0.4,
                0.6, 0.8, 0.95,
            ])
        else:
            levels = np.sort(np.asarray(levels, dtype=float))
        if temperature_grid is None:
            temperature_grid = np.linspace(
                self.gst.glass_temperature + 1.0,
                self.gst.melting_temperature - 1.0,
                160,
            )
        else:
            temperature_grid = np.asarray(temperature_grid, dtype=float)

        ttt_table = None
        critical_rates = None
        if method == "path_intersection":
            table_bar = ProgressBar(len(levels), "TTT lookup table") if progress else None
            ttt_table = np.empty((len(levels), len(temperature_grid)))
            for i, level in enumerate(levels):
                ttt_table[i] = self.ttt_curve(float(level), temperature_grid)
                if table_bar is not None:
                    table_bar.update(i + 1)
        else:
            rate_bar = ProgressBar(len(levels), "Critical cooling rates") if progress else None
            critical_rates = np.empty(len(levels))
            for i, level in enumerate(levels):
                critical_rates[i] = self.critical_cooling_rate(float(level))[0]
                if rate_bar is not None:
                    rate_bar.update(i + 1)
            order = np.argsort(critical_rates)
            critical_rates = critical_rates[order]
            levels_for_rates = levels[order]

        film_indices = np.flatnonzero(result.film_mask)
        n_cells = len(film_indices) * len(result.r)
        cell_bar = ProgressBar(n_cells, "Crystallinity profile") if progress else None
        xc = np.zeros((len(film_indices), len(result.r)))
        cooling_rate = np.full_like(xc, np.nan)
        peak = np.max(result.temperature[:, film_indices, :], axis=0)
        melted = peak >= self.gst.melting_temperature
        unresolved_melt = np.zeros_like(melted)
        completed = 0

        for local_z, global_z in enumerate(film_indices):
            for ir in range(len(result.r)):
                trace = result.temperature[:, global_z, ir]
                if melted[local_z, ir]:
                    below = np.flatnonzero(trace <= self.gst.melting_temperature)
                    if not len(below):
                        unresolved_melt[local_z, ir] = True
                        completed += 1
                        if cell_bar is not None:
                            cell_bar.update(completed)
                        continue
                    crossing_index = int(below[0])
                    if crossing_index == 0:
                        origin = result.time[0]
                    else:
                        t1, t2 = result.time[crossing_index - 1:crossing_index + 1]
                        y1, y2 = trace[crossing_index - 1:crossing_index + 1]
                        origin = t1 + (y1 - self.gst.melting_temperature) / (y1 - y2) * (t2 - t1)
                else:
                    origin = result.time[0]

                elapsed = result.time - origin
                if method == "cooling_rate":
                    rate = cooling_rate_from_trace(
                        result.time, trace, origin, cooling_high, cooling_low
                    )
                    cooling_rate[local_z, ir] = rate
                    if np.isfinite(rate):
                        xc[local_z, ir] = float(np.interp(
                            rate,
                            critical_rates,
                            levels_for_rates,
                            left=levels_for_rates[0],
                            right=levels_for_rates[-1],
                        ))
                else:
                    valid = (
                        (elapsed > 0.0)
                        & (trace > temperature_grid[0])
                        & (trace < temperature_grid[-1])
                    )
                    if np.any(valid):
                        tv = trace[valid]
                        elapsed_v = elapsed[valid]
                        reached = np.zeros(len(levels), dtype=bool)
                        for il in range(len(levels)):
                            required = np.interp(tv, temperature_grid, ttt_table[il])
                            reached[il] = np.any(elapsed_v >= required)
                        if np.any(reached):
                            xc[local_z, ir] = levels[np.flatnonzero(reached)[-1]]

                completed += 1
                if cell_bar is not None:
                    cell_bar.update(completed)

        return CrystallinityProfile(
            crystallinity=xc,
            r=result.r.copy(),
            z=result.z[film_indices].copy(),
            melted=melted,
            unresolved_melt=unresolved_melt,
            peak_temperature=peak,
            cooling_rate=cooling_rate,
            method=method,
        )


def cooling_rate_from_trace(
    time: Array,
    temperature: Array,
    origin: float = 0.0,
    high: float = 850.0,
    low: float = 600.0,
) -> float:
    """Estimate local cooling rate from downward crossings of two temperatures."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(temperature, dtype=float)
    valid = t >= origin
    t = t[valid]
    y = y[valid]
    if len(t) < 2:
        return np.nan

    def downward_crossing(level: float) -> float:
        above = y[:-1] >= level
        below = y[1:] <= level
        indices = np.flatnonzero(above & below)
        if not len(indices):
            return np.nan
        i = int(indices[0])
        y1, y2 = y[i], y[i + 1]
        if y1 == y2:
            return float(t[i])
        fraction = (y1 - level) / (y1 - y2)
        return float(t[i] + fraction * (t[i + 1] - t[i]))

    t_high = downward_crossing(high)
    t_low = downward_crossing(low)
    if np.isfinite(t_high) and np.isfinite(t_low) and t_low > t_high:
        return float((high - low) / (t_low - t_high))
    return estimate_linear_cooling_rate(t, y, high=high, low=low)


def estimate_linear_cooling_rate(time: Array, temperature: Array,
                                 high: float = 850.0, low: float = 600.0) -> float:
    """Fit |dT/dt| while a cooling trace lies between ``high`` and ``low``."""
    mask = (temperature <= high) & (temperature >= low)
    if np.count_nonzero(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(time[mask], temperature[mask], 1)
    return float(abs(slope))


def with_energy(model: AxisymmetricThermalModel, energy_joule: float) -> AxisymmetricThermalModel:
    """Return an equivalent model with a different pulse energy."""
    return AxisymmetricThermalModel(
        gst=model.gst,
        substrate=model.substrate,
        pulse=replace(model.pulse, energy=energy_joule),
        grid=model.grid,
        ambient_temperature=model.ambient,
        top_heat_transfer=model.top_heat_transfer,
        interface_resistance=model.interface_resistance,
        include_phase_change=model.include_phase_change,
    )
