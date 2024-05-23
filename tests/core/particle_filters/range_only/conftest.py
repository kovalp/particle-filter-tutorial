"""."""
import numpy as np
import pytest

from particle_filter_tutorial.core.particle_filters.range_only import ParticleFilterRangeOnly, ResamplingAlgorithms


@pytest.fixture
def my_filter() -> ParticleFilterRangeOnly:
    """."""
    shape = 4, 4
    size = shape[0] * shape[1]
    x_max, y_max = 4.0, 4.0
    flt = ParticleFilterRangeOnly(size, (0.0, x_max, 0.0, y_max),
                                  (1.0, 1.0),
                                  (1.0, 1.0), ResamplingAlgorithms.MULTINOMIAL)
    flt.initialize_particles_uniform()

    xy = np.indices(shape).reshape((2, -1)).T * x_max / shape[0] + 1.0
    aa = np.linspace(0.0, 2 * np.pi, num=size)
    for i, ((x, y), a, (w, _s)) in enumerate(zip(xy, aa, flt.particles)):
        flt.particles[i] = (w, np.array((x, y, a)))

    return flt

