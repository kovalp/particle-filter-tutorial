"""."""

from particle_filter_tutorial.core.particle_filters.range_only import ParticleFilterRangeOnly
from particle_filter_tutorial.simulator.world import World
from particle_filter_tutorial.simulator.robot import Robot
from particle_filter_tutorial.core.resampling.resampler import ResamplingAlgorithms

FWD = 0.25
ANG = 0.04


def get_world_robot_filter(num_particles: int, edge_len: float) -> tuple[World, Robot, ParticleFilterRangeOnly]:
    """."""
    len1 = edge_len - 1.0
    len05 = edge_len / 2.0

    world = World(edge_len, edge_len, [(1.0, 1.0), (len1, 1.0), (1.0, len1), (len1, len1)])
    particle_filter = ParticleFilterRangeOnly(
        num_particles=num_particles,
        limits=(0.0, world.x_max, 0.0, world.y_max),
        process_noise=(0.1, 0.2),  # Process model noise (zero mean additive Gaussian noise)
        measurement_noise=(1.0, 1.0),  # Measurement noise (zero mean additive Gaussian noise)
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL)

    particle_filter.initialize_particles_uniform()
    robot = Robot(len05, len05, 0.0, 0.0, 0.0, 0.0, 0.0)
    return world, robot, particle_filter
