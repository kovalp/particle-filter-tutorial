import math

import numpy as np

from particle_filter_tutorial.simulator import Robot, World
from particle_filter_tutorial.simulator.robot import get_robot_without_noise
from particle_filter_tutorial.demos.demo_running_example.particle_filter import get_particle_filter_sir
from particle_filter_tutorial.demos.demo_running_example.visualizer import get_visualizer

import matplotlib.pyplot as plt


if __name__ == '__main__':
    """."""
    np.set_printoptions(precision=3)

    world = World(10.0, 10.0, [(2.0, 2.0), (2.0, 8.0), (9.0, 2.0), (8.0, 9.0)])
    robot = Robot(7.5, 2.0, math.pi / 2.0, 0.05, 0.002, 0.2, 0.05)
    filter_sir = get_particle_filter_sir(world, 1000)
    filter_sir.initialize_particles_uniform()

    # Desired robot's motion
    robot_motion_forward = 0.25
    robot_motion_turn = 0.04

    visualizer = get_visualizer(False)
    visualizer.draw_world(world, robot, filter_sir.particles, hold_on=False, particle_color='g')
    plt.pause(1.5)

    for i in range(150):
        robot.move(robot_motion_forward, robot_motion_turn, world)  # Move the robot with noise
        measurements = robot.measure(world)  # Simulate measurement
        filter_sir.update(robot_forward_motion=robot_motion_forward,
                          robot_angular_motion=robot_motion_turn,
                          measurements=measurements,
                          landmarks=world.landmarks)
        print(robot, filter_sir.get_average_state())
        visualizer.draw_world(world, robot, filter_sir.particles, hold_on=False, particle_color='g')
        plt.pause(0.01)
