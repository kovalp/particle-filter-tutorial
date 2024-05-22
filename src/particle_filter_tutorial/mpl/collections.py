"""."""
from typing import Sequence

from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Patch


def get_lines_collection(segments: list[tuple[tuple[float, float], tuple[float, float]]],
                         colors: Sequence[str] = ('blue',),
                         line_widths: Sequence[float] = (0.2,)) -> LineCollection:
    """."""
    lines = LineCollection(segments)
    lines.set(colors=colors)
    lines.set(linewidths=line_widths)
    return lines


def get_patch_collection(ls_patches: list[Patch],
                         line_widths: Sequence[float] = (0.2,),
                         face_colors: Sequence[str] = ('yellow',),
                         edge_colors: Sequence[str] = ('green',)) -> PatchCollection:
    """."""
    collection = PatchCollection(ls_patches)
    collection.set(linewidths=line_widths, facecolors=face_colors, edgecolors=edge_colors)
    return collection
