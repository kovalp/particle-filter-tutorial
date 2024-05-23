from matplotlib import pyplot as plt


def get_ax(x_max: float, y_max: float) -> plt.Axes:
    """."""
    figure = plt.gcf()
    figure.set_dpi(100)
    ax: plt.Axes = figure.gca()
    ax.set_ylim((0.0, x_max))
    ax.set_xlim((0.0, y_max))
    return ax


def clear_axes(ax: plt.Axes) -> None:
    """."""
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.clear()
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
