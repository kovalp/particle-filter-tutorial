"""."""
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np

from particle_filter_tutorial.viz.cloud_primitives import CloudPrimitives
from particle_filter_tutorial.mpl.collections import get_patch_collection, get_lines_collection


class CloudDriver:
    """."""
    def __init__(self, ax: plt.Axes, primitives: CloudPrimitives) -> None:
        """."""
        self.ax = ax
        self.primitives: CloudPrimitives = primitives
        self.face_color: str = 'yellow'
        self.edge_color: str = 'green'
        self.line_width: float = 0.2
        self.z_order: int = 20
        self.alpha: float = 1.0

    def add_circles(self, particles: list[tuple[float, np.ndarray]],
                    face_color: str, edge_color: str, z_order: int, alpha: float) -> PatchCollection:
        """."""
        ax, primitives = self.ax, self.primitives
        circles = primitives.ls_circles(particles)
        collection = get_patch_collection(circles, face_colors=(face_color,), edge_colors=(edge_color,))
        collection.set_zorder(z_order)
        collection.set_alpha(alpha)
        ax.add_collection(collection)
        return collection

    def add_particles(self, particles: list[tuple[float, np.ndarray]]) -> tuple[PatchCollection, LineCollection]:
        """."""
        ax, primitives = self.ax, self.primitives
        circles_collection = self.add_circles(particles, self.face_color, self.edge_color, self.z_order, self.alpha)
        segments = primitives.ls_orientation_segments(particles)
        line_collection = get_lines_collection(segments)
        line_collection.set_zorder(21)
        ax.add_collection(line_collection)
        return circles_collection, line_collection
