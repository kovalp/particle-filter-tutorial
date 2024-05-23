"""."""

import numpy as np
import matplotlib.pyplot as plt

from particle_filter_tutorial.demos.plot_particles.factory import get_world_robot_filter, FWD, ANG
from particle_filter_tutorial.mpl.axes import clear_axes, get_ax
from particle_filter_tutorial.viz.cloud_driver import CloudDriver, CloudPrimitives

world, robot, range_filter = get_world_robot_filter(20, 5.0)

measurements = robot.measure(world)

ax = get_ax(world.x_max, world.y_max)
primitives = CloudPrimitives(range_filter.n_particles, range_filter.x_max)
driver = CloudDriver(ax, primitives)

for draw in range(10):
    clear_axes(ax)
    driver.add_particles(range_filter.normalize_weights(range_filter.particles))

    new_particles = range_filter.get_new_particles(FWD, ANG, measurements, world.landmarks)
    norm_particles = range_filter.normalize_weights(new_particles)

    driver.add_circles(norm_particles, 'blue', 'black', 22, 0.5)

    ww = np.fromiter((w for w, s in norm_particles), dtype=float)
    ax.set_title(f'Number of "heavy" particles {np.count_nonzero(ww > 1.0 / len(ww))}')
    plt.pause(1.0)
