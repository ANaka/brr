import numpy as np
from geodude.deform import Cloth


def test_cloth_initialization():
    width = 10
    height = 10
    cloth = Cloth(width, height)

    assert isinstance(cloth, Cloth), "Cloth object not created correctly"
    assert cloth.width == width, "Cloth width not initialized correctly"
    assert cloth.height == height, "Cloth height not initialized correctly"
    assert len(cloth.vertices) == height, "Cloth vertices not initialized correctly"
    assert len(cloth.vertices[0]) == width, "Cloth vertices not initialized correctly"


def test_apply_gravity():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.get_vertex_positions())
    cloth.apply_gravity(0.5)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert cloth.vertices[i][j].position[1] == initial_vertices[i, j][1] - 0.5, "Gravity not applied correctly"


def test_apply_wind():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.get_vertex_positions())
    wind_vector = np.array([1.0, 0.0])
    cloth.apply_wind(wind_vector)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert cloth.vertices[i][j].position[0] == initial_vertices[i, j][0] + 1.0, "Wind not applied correctly"


def test_update_cloth():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.get_vertex_positions())
    gravity = 0.5
    wind_vector = np.array([1.0, 0.0])
    time_step = 1.0
    cloth.update(time_step, gravity, wind_vector)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert (
                cloth.vertices[i][j].position[0] == initial_vertices[i, j][0] + wind_vector[0]
            ), "Cloth update not working correctly"
            assert (
                cloth.vertices[i][j].position[1] == initial_vertices[i, j][1] - gravity
            ), "Cloth update not working correctly"


def test_cloth_constraints():
    cloth = Cloth(4, 4)
    initial_edges = cloth.get_edge_lengths()
    cloth.apply_gravity(1)
    cloth.satisfy_constraints(10)
    tolerance = 0.01
    updated_edges = cloth.get_edge_lengths()

    for i in range(len(initial_edges)):
        assert np.abs(updated_edges[i] - initial_edges[i]) < tolerance, f"Edge {i} not satisfied"


def test_cloth_collision_response():
    cloth = Cloth(5, 5)
    sphere_center = np.array([2.5, 2.5])
    sphere_radius = 1.0
    cloth.handle_collision(sphere_center, sphere_radius)

    # Check if the cloth vertices are outside the sphere
    for i in range(cloth.height):
        for j in range(cloth.width):
            dist_to_center = np.linalg.norm(cloth.vertices[i][j].position - sphere_center)
            assert dist_to_center >= sphere_radius - 1e-05, "Cloth collision response not working correctly"
