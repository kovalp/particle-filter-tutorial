"""."""

import numpy as np
import matplotlib.pyplot as plt

from particle_filter_tutorial.simulator import World, Robot
from particle_filter_tutorial.demos.particle_filter_factory import get_range_only_filter, ResamplingAlgorithms
from particle_filter_tutorial.mpl.axes import clear_axes, get_ax
from particle_filter_tutorial.viz.cloud_driver import CloudDriver, CloudPrimitives

LEN = 50.0
LEN1 = LEN - 1.0
LEN05 = LEN / 2.0
NUM_PRT = 1000
world = World(LEN, LEN, [(1.0, 1.0), (LEN1, 1.0), (1.0, LEN1), (LEN1, LEN1)])
range_filter = get_range_only_filter(world, NUM_PRT, (1.0, 1.0))
range_filter.initialize_particles_uniform()
robot = Robot(LEN05, LEN05, 0.0, 0.0, 0.0, 0.0, 0.0)
measurements = robot.measure(world)

ax = get_ax(world.x_max, world.y_max)
primitives = CloudPrimitives(range_filter.n_particles, range_filter.x_max)
driver = CloudDriver(ax, primitives)

for draw in range(10):
    clear_axes(ax)
    particles = range_filter.normalize_weights(range_filter.particles)
    driver.add_particles(particles)

    new_particles = range_filter.get_new_particles(0.25, 0.04, measurements, world.landmarks)
    norm_particles = range_filter.normalize_weights(new_particles)
    ww = np.fromiter((w for w, s in norm_particles), dtype=float)

    driver.add_circles(norm_particles, 'blue', 'black', 22, 0.15)

    resampled = range_filter.resampler.resample(norm_particles, NUM_PRT, ResamplingAlgorithms.RESIDUAL)
    driver.add_circles(resampled, 'red', 'b', 23, 0.75)

    plt.pause(1.5)
