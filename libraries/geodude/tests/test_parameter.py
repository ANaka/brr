import random

import numpy as np
import pytest
from geodude.parameter import Parameter
from hypothesis import given
from hypothesis import strategies as st


def test_constant_parameter():
    p = Parameter(10)
    assert p() == 10


def test_function_parameter():
    p = Parameter(lambda x: x * 2, 5)
    assert p() == 10


def test_random_parameter():
    p = Parameter(random.uniform, 0, 10)
    value = p()
    assert 0 <= value <= 10


def test_numpy_random_parameter():
    p = Parameter(np.random.normal, 0, 1)
    np.random.seed(42)  # Set seed for reproducibility
    assert pytest.approx(p(), 0.01) == 0.496


@given(st.integers())
def test_constant_parameter_with_hypothesis(value):
    p = Parameter(value)
    assert p() == value


@given(st.integers(), st.integers())
def test_function_parameter_with_hypothesis(a, b):
    p = Parameter(lambda x, y: x + y, a, b)
    assert p() == a + b


@given(st.floats(0, 10))
def test_random_parameter_with_hypothesis(lower_bound):
    p = Parameter(random.uniform, lower_bound, 10)
    value = p()
    assert lower_bound <= value <= 10


def test_deep_copy():
    original_list = [1, 2, 3]
    p = Parameter(original_list, deep_copy=True)
    original_list.append(4)
    assert p() == [1, 2, 3]
