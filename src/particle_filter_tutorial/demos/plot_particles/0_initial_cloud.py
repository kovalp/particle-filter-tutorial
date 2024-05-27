"""."""

import matplotlib.pyplot as plt

from particle_filter_tutorial.simulator import World
from particle_filter_tutorial.demos.particle_filter_factory import get_particle_filter_sir
from particle_filter_tutorial.mpl.collections import get_lines_collection, get_patch_collection
from particle_filter_tutorial.mpl.axes import get_ax
from particle_filter_tutorial.viz.cloud_primitives import CloudPrimitives


world = World(10.0, 10.0, [])
sir = get_particle_filter_sir(world, 1000)
sir.initialize_particles_uniform()

ax = get_ax(world.x_max, world.y_max)

primitives = CloudPrimitives(sir.n_particles, sir.x_max)

circles = primitives.ls_circles(sir.particles)
ax.add_collection(get_patch_collection(circles))

segments = primitives.ls_orientation_segments(sir.particles)
ax.add_collection(get_lines_collection(segments))

plt.show()

