"""."""

import numpy as np
import pytest
from pytest_mock import MockFixture

from particle_filter_tutorial.core.particle_filters.range_only import ParticleFilterRangeOnly

from tests.core.particle_filters.conftest import RandomMock


def test_init(my_filter: ParticleFilterRangeOnly, mocker: MockFixture, random_mock: RandomMock) -> None:
    """."""
    mocker.patch('numpy.random.normal', random_mock.fake_normal)
    mocker.patch('numpy.random.uniform', random_mock.fake_uniform)

    measurements = [np.zeros(2), 3 * np.ones(2), 2 * np.ones(2)]
    landmarks = [(1.0, 1.0), (4.0, 1.0), (1.0, 4.0)]
    my_filter.update(0.5, 0.1, measurements, landmarks)

    for (w, s) in my_filter.particles:
        assert w == pytest.approx(0.0625)
        assert s == pytest.approx((1.00959439, 0.99594356, 5.88318531))


