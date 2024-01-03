from typing import List

import numpy as np
from shapely import MultiLineString


class Vertex:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.half_edge = None


class HalfEdge:
    def __init__(self, rest_length: float):
        self.vertex = None
        self.opposite = None
        self.next = None
        self.rest_length = rest_length


class Cloth:
    def __init__(self, width: int, height: int, rest_length: float = 1.0):
        self.width = width
        self.height = height

        # Initialize vertices and half-edges
        self.vertices = [[Vertex(np.array([j, i], dtype=np.float64)) for j in range(width)] for i in range(height)]
        self.half_edges = [[HalfEdge(rest_length) for j in range(width)] for i in range(height)]

        # Connect vertices and half-edges
        for i in range(height):
            for j in range(width):
                self.vertices[i][j].half_edge = self.half_edges[i][j]
                self.half_edges[i][j].vertex = self.vertices[i][j]

                # Connect half-edges horizontally
                if j < width - 1:
                    self.half_edges[i][j].next = self.half_edges[i][j + 1]
                    self.half_edges[i][j + 1].opposite = self.half_edges[i][j]

                # Connect half-edges vertically
                if i < height - 1:
                    self.half_edges[i][j].opposite = self.half_edges[i + 1][j]
                    self.half_edges[i + 1][j].next = self.half_edges[i][j]

    def apply_gravity(self, gravity: float):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i][j].position -= np.array([0, gravity])

    def apply_wind(self, wind_vector: np.ndarray):
        for i in range(self.height):
            for j in range(self.width):
                self.vertices[i][j].position += wind_vector

    def update(self, time_step: float, gravity: float, wind_vector: np.ndarray):
        self.apply_gravity(gravity * time_step)
        self.apply_wind(wind_vector * time_step)
        self.satisfy_constraints(1)

    def satisfy_constraints(self, iterations: int = 1):
        for _ in range(iterations):
            for i in range(self.height):
                for j in range(self.width):
                    half_edge = self.half_edges[i][j]
                    p1 = half_edge.vertex.position
                    if half_edge.next is not None:
                        p2 = half_edge.next.vertex.position
                        delta = p2 - p1
                        distance = np.linalg.norm(delta)
                        difference = (half_edge.rest_length - distance) / distance
                        half_edge.vertex.position += delta * 0.5 * difference
                        half_edge.next.vertex.position -= delta * 0.5 * difference

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
                vertex_to_center = self.vertices[i][j].position - sphere_center
                distance = np.linalg.norm(vertex_to_center)
                if distance < sphere_radius:
                    self.vertices[i][j].position = sphere_center + vertex_to_center / distance * sphere_radius

    def get_vertex_positions(self) -> np.ndarray:
        positions = np.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                positions[i, j] = self.vertices[i][j].position
        return positions

    def get_edge_lengths(self) -> List[float]:
        edge_lengths = []
        for i in range(self.height):
            for j in range(self.width):
                half_edge = self.half_edges[i][j]
                if half_edge.next is not None:
                    p1 = half_edge.vertex.position
                    p2 = half_edge.next.vertex.position
                    length = np.linalg.norm(p2 - p1)
                    edge_lengths.append(length)
        return edge_lengths

    def to_multilinestring(self) -> MultiLineString:
        """Convert the cloth's geometry into a shapely MultiLineString for plotting."""
        lines = []

        # Add horizontal lines
        for i in range(self.height):
            for j in range(self.width - 1):
                lines.append((tuple(self.vertices[i][j].position), tuple(self.vertices[i][j + 1].position)))

        # Add vertical lines
        for i in range(self.height - 1):
            for j in range(self.width):
                lines.append((tuple(self.vertices[i][j].position), tuple(self.vertices[i + 1][j].position)))

        return MultiLineString(lines)

    def jitter_vertices(self, jitter_amount: float):
        for i in range(self.height):
            for j in range(self.width):
                random_offset = np.random.uniform(-jitter_amount, jitter_amount, 2)
                self.vertices[i][j].position += random_offset
