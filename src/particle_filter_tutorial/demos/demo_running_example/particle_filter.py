from particle_filter_tutorial.simulator.world import World
from particle_filter_tutorial.core.particle_filters.sir import ParticleFilterSIR
from particle_filter_tutorial.core.resampling.resampler import ResamplingAlgorithms


def get_particle_filter_sir(world: World, num_particles: int = 1000) -> ParticleFilterSIR:
    """."""

    pf_state_limits = [0, world.x_max, 0, world.y_max]

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter: resample every time step
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles=num_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resampling_algorithm=algorithm)
    particle_filter_sir.initialize_particles_uniform()
    return particle_filter_sir
