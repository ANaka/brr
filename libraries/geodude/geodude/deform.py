import numpy as np


class Cloth:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.vertices = np.array([[(x, y) for x in range(width)] for y in range(height)], dtype=object)
        for i in range(height):
            for j in range(width):
                self.vertices[i, j] = np.array(self.vertices[i, j])  # Convert tuples to numpy arrays

        self.edges = self._initialize_edges()

    def _initialize_edges(self):
        edges = []
        for i in range(self.height):
            for j in range(self.width):
                if j < self.width - 1:
                    edges.append((self.vertices[i, j], self.vertices[i, j + 1]))
                if i < self.height - 1:
                    edges.append((self.vertices[i, j], self.vertices[i + 1, j]))
        return edges

    def apply_gravity(self, gravity: float):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i, j] = self.vertices[i, j] - np.array([0, gravity])

    def apply_wind(self, wind_vector: np.ndarray):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i, j] = self.vertices[i, j] + wind_vector

    def update(self, time_step: float, gravity: float, wind_vector: np.ndarray):
        """
        Update the cloth's state over a single time step.

        :param time_step: The duration of the time step.
        :param gravity: The strength of gravity applied to the cloth.
        :param wind_vector: The wind vector applied to the cloth.
        """
        self.apply_gravity(gravity * time_step)
        self.apply_wind(wind_vector * time_step)

        self.satisfy_constraints(1)

    def satisfy_constraints(self, iterations: int):
        """
        Satisfy the cloth constraints by adjusting the positions of the vertices.

        :param iterations: The number of times to iterate through the constraints.
        """
        for _ in range(iterations):
            for i in range(self.height):
                for j in range(self.width):
                    if j < self.width - 1:
                        self.vertices[i, j], self.vertices[i, j + 1] = self._satisfy_constraint(
                            self.vertices[i, j], self.vertices[i, j + 1], 1
                        )
                    if i < self.height - 1:
                        self.vertices[i, j], self.vertices[i + 1, j] = self._satisfy_constraint(
                            self.vertices[i, j], self.vertices[i + 1, j], 1
                        )

    def _satisfy_constraint(self, p1: np.ndarray, p2: np.ndarray, rest_length: float):
        delta = p2 - p1
        delta_length = np.linalg.norm(delta)
        correction = delta * (1 - rest_length / delta_length)
        p1 += correction / 2
        p2 -= correction / 2
        return p1, p2  # Return the corrected vertices

    def handle_collision(self, sphere_center: np.ndarray, sphere_radius: float):
        """
        Handle collision with a sphere by adjusting the positions of the vertices.

        :param sphere_center: The center of the sphere.
        :param sphere_radius: The radius of the sphere.
        """
        for i in range(self.height):
            for j in range(self.width):
                vertex_to_center = self.vertices[i, j] - sphere_center
                distance = np.linalg.norm(vertex_to_center)
                if distance < sphere_radius:
                    self.vertices[i, j] = sphere_center + vertex_to_center / distance * sphere_radius
