"""."""

import math
import numpy as np
import matplotlib.pyplot as plt

from particle_filter_tutorial.simulator import World, Robot
from particle_filter_tutorial.demos.particle_filter_factory import get_range_only_filter
from particle_filter_tutorial.mpl.axes import clear_axes, get_ax
from particle_filter_tutorial.viz.cloud_driver import CloudDriver, CloudPrimitives


world = World(10.0, 10.0, [(1.0, 1.0), (9.0, 1.0), (1.0, 9.0), (9.0, 9.0)])
range_filter = get_range_only_filter(world, 10)
range_filter.initialize_particles_uniform()
robot = Robot(5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
measurements = robot.measure(world)

ax = get_ax(world.x_max, world.y_max)
primitives = CloudPrimitives(range_filter.n_particles, range_filter.x_max)
driver = CloudDriver(ax, primitives)

for draw in range(20):
    clear_axes(ax)
    particles = range_filter.normalize_weights(range_filter.particles)
    driver.add_particles(particles)

    new_particles = range_filter.get_new_particles(0.25, 0.04, measurements, world.landmarks)
    norm_particles = range_filter.normalize_weights(new_particles)
    for i, p in enumerate(norm_particles):
        print(i, p[0], p[1][:2])

    circles = driver.add_circles(norm_particles, 'blue', 'black', 22, 0.5)
    plt.pause(0.2)
