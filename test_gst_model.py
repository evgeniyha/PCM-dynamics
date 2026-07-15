import unittest

import numpy as np

from gst_model import (
    AxisymmetricThermalModel,
    CrystallizationKinetics,
    Grid,
    LaserPulse,
    cooling_rate_from_trace,
)


class GSTModelTests(unittest.TestCase):
    def test_zero_energy_stays_near_ambient(self):
        model = AxisymmetricThermalModel(
            pulse=LaserPulse(energy=0.0),
            grid=Grid(radial_extent=4e-6, dr=1e-6,
                      dz_film=20e-9, dz_substrate=100e-9),
        )
        result = model.simulate_cooling(end_time=1e-9, time_step=0.5e-9)
        self.assertLess(np.max(np.abs(result.temperature - 300.0)), 1e-8)

    def test_absorption_decreases_with_depth_and_radius(self):
        model = AxisymmetricThermalModel(
            grid=Grid(radial_extent=4e-6, dr=1e-6,
                      dz_film=20e-9, dz_substrate=100e-9)
        )
        q = model.absorbed_energy_density()
        self.assertGreater(q[0, 0], q[1, 0])
        self.assertGreater(q[0, 0], q[0, 1])
        self.assertTrue(np.all(q[~model.film_rows] == 0.0))

    def test_no_phase_change_uses_constant_heat_capacity(self):
        model = AxisymmetricThermalModel(
            grid=Grid(radial_extent=4e-6, dr=1e-6,
                      dz_film=20e-9, dz_substrate=100e-9),
            include_phase_change=False,
        )
        expected = 300.0 + model.absorbed_energy_density() / (
            model.rho * model.gst.specific_heat
        )
        np.testing.assert_allclose(model.pulse_end_temperature(), expected)

    def test_cooling_rate_from_threshold_crossings(self):
        time = np.array([0.0, 1e-9, 2e-9, 3e-9])
        temperature = np.array([900.0, 800.0, 700.0, 500.0])
        rate = cooling_rate_from_trace(time, temperature, high=850.0, low=600.0)
        self.assertAlmostEqual(rate / 1e9, 125.0)

    def test_article_eq28_bracket_is_positive(self):
        bracket = CrystallizationKinetics._article_eq28_bracket(3.0)
        self.assertGreater(bracket, 0.0)

    def test_ttt_nose_and_monotonic_crystallization_time(self):
        kinetics = CrystallizationKinetics()
        temperatures = np.linspace(450.0, 850.0, 41)
        low = kinetics.ttt_curve(1e-8, temperatures)
        high = kinetics.ttt_curve(0.1, temperatures)
        self.assertTrue(np.all(high >= low))
        self.assertLess(500.0, temperatures[np.argmin(low)])
        self.assertGreater(800.0, temperatures[np.argmin(low)])


if __name__ == "__main__":
    unittest.main()
