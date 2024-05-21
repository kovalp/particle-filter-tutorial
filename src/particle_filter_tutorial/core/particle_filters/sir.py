import numpy as np

from particle_filter_tutorial.core.particle_filters.particle_filter_base import ParticleFilter
from particle_filter_tutorial.core.resampling.resampler import Resampler, ResamplingAlgorithms


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations.
    """

    def __init__(self,
                 number_of_particles: int,
                 limits: tuple[float, float, float, float],
                 process_noise: tuple[float, float],
                 measurement_noise: tuple[float, float],
                 resampling_algorithm: ResamplingAlgorithms):
        """
        Initialize the SIR particle filter.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self) -> bool:
        """
        Method that determines whether not a core step is needed for the current particle filter state estimate.
        The sampling importance core (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether core is needed.
        """
        return True

    def update(self,
               robot_forward_motion: float,
               robot_angular_motion: float,
               measurements: list[np.ndarray],
               landmarks: list[tuple[float, float]]) -> None:
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Loop over all particles
        new_particles = []
        for (weight, sample) in self.particles:

            # Propagate the particle state according to the current particle
            propagated_state = self.propagate_sample(sample, robot_forward_motion, robot_angular_motion)

            # Compute current particle's weight
            new_weight = weight * self.compute_likelihood(propagated_state, measurements, landmarks)

            # Store
            new_particles.append((new_weight, propagated_state))

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
