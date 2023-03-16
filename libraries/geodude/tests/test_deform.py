import numpy as np
from geodude.deform import Cloth


def test_cloth_initialization():
    width = 10
    height = 10
    cloth = Cloth(width, height)

    assert isinstance(cloth, Cloth), "Cloth object not created correctly"
    assert cloth.width == width, "Cloth width not initialized correctly"
    assert cloth.height == height, "Cloth height not initialized correctly"
    assert cloth.vertices.shape == (height, width), "Cloth vertices not initialized correctly"
    assert len(cloth.edges) == (width - 1) * height + width * (height - 1), "Cloth edges not initialized correctly"


def test_apply_gravity():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.vertices)
    cloth.apply_gravity(0.5)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert cloth.vertices[i, j][1] == initial_vertices[i, j][1] - 0.5, "Gravity not applied correctly"


def test_apply_wind():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.vertices)
    wind_vector = np.array([1.0, 0.0])
    cloth.apply_wind(wind_vector)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert cloth.vertices[i, j][0] == initial_vertices[i, j][0] + 1.0, "Wind not applied correctly"


def test_update_cloth():
    cloth = Cloth(5, 5)
    initial_vertices = np.copy(cloth.vertices)
    gravity = 0.5
    wind_vector = np.array([1.0, 0.0])
    time_step = 1.0
    cloth.update(time_step, gravity, wind_vector)

    for i in range(cloth.height):
        for j in range(cloth.width):
            assert (
                cloth.vertices[i, j][0] == initial_vertices[i, j][0] + wind_vector[0]
            ), "Cloth update not working correctly"
            assert cloth.vertices[i, j][1] == initial_vertices[i, j][1] - gravity, "Cloth update not working correctly"


def test_cloth_constraints():
    cloth = Cloth(5, 5)
    cloth.vertices[1, 1] += np.array([2, 2])  # Modify a vertex to violate the constraint
    np.copy(cloth.vertices)
    cloth.satisfy_constraints(1)

    # Check if the constraints are satisfied
    for i in range(cloth.height - 1):
        for j in range(cloth.width - 1):
            horizontal_dist = np.linalg.norm(cloth.vertices[i, j] - cloth.vertices[i, j + 1])
            vertical_dist = np.linalg.norm(cloth.vertices[i, j] - cloth.vertices[i + 1, j])
            assert np.isclose(horizontal_dist, 1, rtol=1e-05), "Horizontal cloth constraint not satisfied"
            assert np.isclose(vertical_dist, 1, rtol=1e-05), "Vertical cloth constraint not satisfied"


def test_cloth_collision_response():
    cloth = Cloth(5, 5)
    sphere_center = np.array([2.5, 2.5])
    sphere_radius = 1.0
    cloth.handle_collision(sphere_center, sphere_radius)

    # Check if the cloth vertices are outside the sphere
    for i in range(cloth.height):
        for j in range(cloth.width):
            dist_to_center = np.linalg.norm(cloth.vertices[i, j] - sphere_center)
            assert dist_to_center >= sphere_radius - 1e-05, "Cloth collision response not working correctly"
