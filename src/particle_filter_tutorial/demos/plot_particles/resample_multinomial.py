import numpy as np

from particle_filter_tutorial.core.particle_filters.base import PST
from particle_filter_tutorial.core.resampling.helpers import cumulative_sum, naive_search


def resample_multinomial(samples: PST) -> PST:
    """."""
    n = len(samples)
    ww = np.fromiter((w for w, c in samples), dtype=float)
    qq = cumulative_sum(ww)
    print(qq)

    i = 0  # As long as the number of new samples is insufficient
    new_samples = []
    while i < n:
        u = float(np.random.uniform(1e-6, 1.0, 1)[0])  # Draw a random sample u
        m = naive_search(qq, u)  # Naive search (alternative: binary search)
        print(f'{i} --> {m}  {u:.9f} --> {samples[m][0]:.9f}')
        new_samples.append((1.0 / n, np.array(samples[m][1].copy())))
        i += 1

    return new_samples
