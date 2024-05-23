"""."""
import numpy as np

from particle_filter_tutorial.demos.plot_particles.factory import get_world_robot_filter, FWD, ANG
from particle_filter_tutorial.demos.plot_particles.resample_multinomial import resample_multinomial


np.set_printoptions(precision=4)

world, robot, range_filter = get_world_robot_filter(10, 5.0)
measurements = robot.measure(world)

new_particles = range_filter.get_new_particles(FWD, ANG, measurements, world.landmarks)
norm_particles = range_filter.normalize_weights(new_particles)

resampled = resample_multinomial(norm_particles)
