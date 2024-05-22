"""."""

import matplotlib.pyplot as plt
import matplotlib.collections

figure = plt.gcf()
ax: plt.Axes = figure.gca()
ax.set_ylim((-2.0, 5.0))
ax.set_xlim((-2.0, 5.0))

ls_circles = [plt.Circle((0, 0), 1), plt.Circle((2, 1), 2)]
collection = matplotlib.collections.PatchCollection(ls_circles)
collection.set_facecolor('r')
collection.set_edgecolor('b')
collection.set_alpha(0.6)
collection.set_zorder(10)
ax.add_collection(collection)

plt.show()


