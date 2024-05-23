"""."""
import numpy as np
import pytest

import itertools


class RandomMock:
    """."""
    def __init__(self, num: int = 97) -> None:
        """."""
        self.normal_sequence = np.linspace(-0.5, 0.5, num)
        self.uniform_sequence = np.linspace(0.0, 1.0, num)
        self.normal_cycle = itertools.cycle(self.normal_sequence)
        self.uniform_cycle = itertools.cycle(self.uniform_sequence)

    def fake_normal(self, loc: float, noise: float, size: int = 1) -> np.ndarray:
        """."""
        return np.fromiter((loc + u * noise for u in self.normal_cycle), dtype=float, count=size)

    def fake_uniform(self, low: float, high: float, size: int = 1) -> np.ndarray:
        """."""
        diff = high - low
        return np.fromiter((u * diff + low for u in self.uniform_cycle), dtype=float, count=size)


@pytest.fixture
def random_mock() -> RandomMock:
    """."""
    return RandomMock()
