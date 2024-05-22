"""."""
import math

import matplotlib.pyplot as plt
import numpy as np


class CloudPrimitives:
    """."""

    def __init__(self, num_particles: int, x_max: float) -> None:
        """."""
        self.weight_radius_factor = num_particles / x_max
        self.min_radius = x_max / 200.0

    def get_radius(self, weight: float) -> float:
        """."""
        rad = self.weight_radius_factor * weight
        min_rad = self.min_radius
        return rad if rad > min_rad else min_rad

    def ls_circles(self, particles: list[tuple[float, np.ndarray]]) -> list[plt.Circle]:
        """."""
        factor = self.weight_radius_factor
        return [plt.Circle((x, y), self.get_radius(w)) for w, (x, y, a) in particles]

    def ls_orientation_segments(self,
                                particles: list[tuple[float, np.ndarray]]
                                ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """."""
        f = self.weight_radius_factor
        return [((x, y), (x + w * f * math.cos(a), y + w * f * math.sin(a))) for w, (x, y, a) in particles]

