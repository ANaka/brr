import numpy as np
from itertools import product
from shapely import MultiLineString
class Cloth:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.vertices = np.zeros((height, width, 2), dtype=np.float64)
        for i in range(height):
            for j in range(width):
                self.vertices[i, j] = np.array([j, i], dtype=np.float64)
        self.edges = self._initialize_edges()

    def _initialize_edges(self):
        edges = []
        for ii, jj in product(range(self.height), range(self.width)):
            if jj < self.width - 1:
                edges.append((self.vertices[ii, jj], self.vertices[ii, jj + 1]))
            if ii < self.height - 1:
                edges.append((self.vertices[ii, jj], self.vertices[ii + 1, jj]))
        return edges

    def apply_gravity(self, gravity: float):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i, j] -= np.array([0, gravity])

    def apply_wind(self, wind_vector: np.ndarray):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i, j] += wind_vector

    def update(self, time_step: float, gravity: float, wind_vector: np.ndarray):
        self.apply_gravity(gravity * time_step)
        self.apply_wind(wind_vector * time_step)
        self.satisfy_constraints(1)

    def satisfy_constraints(self, iterations: int):
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
        return p1, p2

    def handle_collision(self, sphere_center: np.ndarray, sphere_radius: float):
        for i in range(self.height):
            for j in range(self.width):
                vertex_to_center = self.vertices[i, j] - sphere_center
                distance = np.linalg.norm(vertex_to_center)
                if distance < sphere_radius:
                    self.vertices[i, j] = sphere_center + vertex_to_center / distance * sphere_radius

    def to_multilinestring(self) -> MultiLineString:
        """Convert the cloth's geometry into a shapely MultiLineString for plotting."""
        lines = []

        # Add horizontal lines
        for i in range(self.height):
            for j in range(self.width - 1):
                lines.append((tuple(self.vertices[i, j]), tuple(self.vertices[i, j + 1])))

        # Add vertical lines
        for i in range(self.height - 1):
            for j in range(self.width):
                lines.append((tuple(self.vertices[i, j]), tuple(self.vertices[i + 1, j])))

        return MultiLineString(lines)