"""."""

from particle_filter_tutorial.viz.visualizer import Visualizer


def get_visualizer(show_particle_pose: bool = False) -> Visualizer:
    """
    show_particle_pose: only set to true for low #particles (very slow).
    """
    visualizer = Visualizer(show_particle_pose)
    visualizer.update_robot_radius(0.2)
    visualizer.update_landmark_size(7)
    return visualizer
