"""."""


class World:
    """."""
    def __init__(self, size_x: float, size_y: float, landmarks: list[tuple[float, float]]) -> None:
        """
        Initialize world with given dimensions.
        :param size_x: Length world in x-direction (m)
        :param size_y: Length world in y-direction (m)
        :param landmarks: List with 2D-positions of landmarks
        """
        self.x_max = size_x
        self.y_max = size_y

        # Check if each list within landmarks list contains exactly two elements (x- and y-position)
        if any(len(lm) != 2 for lm in landmarks):
            raise ValueError(f"Invalid landmarks provided to World: {landmarks}")
        else:
            self.landmarks = landmarks
