#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from particle_filter_tutorial.simulator import RobotRange, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from particle_filter_tutorial.core.resampling import ResamplingAlgorithms

# Particle filters
from particle_filter_tutorial.core.particle_filters import ParticleFilterRangeOnly

# For showing plots (plt.show())
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print("Running example particle filter demo with range only measurements.")

    ##
    # Set simulated world and visualization properties
    ##
    # world = World(10.0, 10.0, [[5.0, 5.0]])
    world = World(10.0, 10.0, [(2.0, 8.0)])

    # Number of simulated time steps
    n_time_steps = 200

    # Initialize visualization
    show_particle_pose = False  # only set to true for low #particles (very slow)
    visualizer = Visualizer(show_particle_pose)
    visualizer.update_robot_radius(0.2)
    visualizer.update_landmark_size(7)

    ##
    # True robot properties (simulator settings)
    ##

    # Setpoint (desired) motion robot
    robot_setpoint_motion_forward = 0.25
    robot_setpoint_motion_turn = 0.02

    # True simulated robot motion is set point plus additive zero mean Gaussian noise with these standard deviation
    true_robot_motion_forward_std = 0.005
    true_robot_motion_turn_std = 0.002

    # Robot measurements are corrupted by measurement noise
    true_robot_meas_noise_distance_std = 0.2

    # Initialize simulated robot
    robot = RobotRange(x=world.x_max * 0.8,
                       y=world.y_max / 6.0,
                       theta=3.14 / 2.0,
                       std_forward=true_robot_motion_forward_std,
                       std_turn=true_robot_motion_turn_std,
                       std_meas_distance=true_robot_meas_noise_distance_std)

    ##
    # Particle filter settings
    ##

    number_of_particles = 1000

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = meas_model_distance_std

    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter with range only measurements
    particle_filter = ParticleFilterRangeOnly(
        num_particles=number_of_particles,
        limits=(0.0, world.x_max, 0.0, world.y_max),
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resampling_algorithm=algorithm)
    particle_filter.initialize_particles_uniform()

    ##
    # Start simulation
    ##
    for i in range(n_time_steps):

        # Simulate robot motion (required motion will not exactly be achieved)
        robot.move(desired_distance=robot_setpoint_motion_forward,
                   desired_rotation=robot_setpoint_motion_turn,
                   world=world)

        # Simulate measurement
        measurements = robot.measure(world)

        # Update particle filter
        particle_filter.update(robot_forward_motion=robot_setpoint_motion_forward,
                               robot_angular_motion=robot_setpoint_motion_turn,
                               measurements=measurements,
                               landmarks=world.landmarks)

        # Visualization
        visualizer.draw_world(world, robot, particle_filter.particles, hold_on=False, particle_color='g')
        plt.pause(0.1)
