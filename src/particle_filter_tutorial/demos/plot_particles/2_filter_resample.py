"""."""

import numpy as np
import matplotlib.pyplot as plt


from particle_filter_tutorial.demos.plot_particles.factory import get_world_robot_filter, ResamplingAlgorithms, FWD, ANG
from particle_filter_tutorial.mpl.axes import clear_axes, get_ax
from particle_filter_tutorial.viz.cloud_driver import CloudDriver, CloudPrimitives

world, robot, range_filter = get_world_robot_filter(20, 5.0)

measurements = robot.measure(world)
ax = get_ax(world.x_max, world.y_max)
primitives = CloudPrimitives(range_filter.n_particles, range_filter.x_max)
driver = CloudDriver(ax, primitives)

for draw in range(20):
    clear_axes(ax)
    new_particles = range_filter.get_new_particles(FWD, ANG, measurements, world.landmarks)
    norm_particles = range_filter.normalize_weights(new_particles)
    pp = np.fromiter((p[0] for p in norm_particles), dtype=float)

    driver.add_circles(norm_particles, 'blue', 'black', 22, 0.5)

    resampled = range_filter.resampler.resample(norm_particles, range_filter.n_particles,
                                                ResamplingAlgorithms.MULTINOMIAL)
    driver.add_circles(resampled, 'red', 'b', 23, 0.75)

    ss = np.fromiter((e for _, xya in resampled for e in xya), dtype=float).reshape((range_filter.n_particles, -1))
    unique, counts = np.unique(ss, axis=0, return_counts=True)
    print(unique, counts)
    ax.set_title(f'n_heavy {np.count_nonzero(pp > 1.0 / pp.size)} n_unique {counts.size}')

    plt.pause(1.0)
