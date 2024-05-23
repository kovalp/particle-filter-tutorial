"""."""

from particle_filter_tutorial.simulator.world import World
from particle_filter_tutorial.core.particle_filters.sir import ParticleFilterSIR
from particle_filter_tutorial.core.particle_filters.range_only import ParticleFilterRangeOnly
from particle_filter_tutorial.core.resampling.resampler import ResamplingAlgorithms


def get_particle_filter_sir(world: World, num_particles: int = 1000) -> ParticleFilterSIR:
    """."""
    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter: resample every time step
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles=num_particles,
        limits=(0.0, world.x_max, 0.0, world.y_max),
        process_noise=(0.1, 0.2),  # Process model noise (zero mean additive Gaussian noise)
        measurement_noise=(0.4, 0.3),  # Measurement noise (zero mean additive Gaussian noise)
        resampling_algorithm=algorithm)
    return particle_filter_sir


def get_range_only_filter(world: World,
                          num_particles: int = 1000,
                          measurement_noise: tuple[float, float] = (0.4, 0.3)
                          ) -> ParticleFilterRangeOnly:
    """."""
    particle_filter = ParticleFilterRangeOnly(
        num_particles=num_particles,
        limits=(0.0, world.x_max, 0.0, world.y_max),
        process_noise=(0.1, 0.2),  # Process model noise (zero mean additive Gaussian noise)
        measurement_noise=measurement_noise,  # Measurement noise (zero mean additive Gaussian noise)
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL)
    return particle_filter

