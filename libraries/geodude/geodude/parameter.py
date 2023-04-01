import collections
import copy
from collections import UserDict
from functools import partial


class Parameter:
    """
    A generic parameter wrapper that can handle constant values, functions, or random variables.

    Usage:
    >>> x = Parameter(10)
    >>> y = Parameter(lambda: random.uniform(0, 10))
    >>> z = Parameter(np.random.normal, 0, 1)  # mean 0, std 1
    """

    def __init__(self, value, *args, deep_copy=False, iterate: bool = False, **kwargs):
        """
        Initialize the Parameter with a constant value, a function, or a random variable.

        :param value: The constant value, function, or random variable.
        :param args: Optional positional arguments for functions or random variables.
        :param deep_copy: Optional flag to create a deep copy of mutable collections.
        :param kwargs: Optional keyword arguments for functions or random variables.
        """
        self.iterate = iterate
        if self.iterate and isinstance(value, collections.abc.Iterable):
            value = iter(value)

        if deep_copy and isinstance(value, collections.abc.MutableSequence):
            self.value = copy.deepcopy(value)
        elif callable(value):
            self.value = partial(value, *args, **kwargs)
        else:
            self.value = value

    def __call__(self, *args, **kwargs):
        """
        Evaluate and return the parameter value.

        :return: The constant value or the result of the function or random variable.
        """
        if callable(self.value):
            return self.value(*args, **kwargs)
        elif isinstance(self.value, collections.abc.Iterable):
            if self.iterate:
                return next(self.value)
            else:
                return self.value
        return self.value

    def __repr__(self) -> str:
        return f"Parameter({self.value})"


class Prm(Parameter):
    pass


class PrmDict(UserDict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        value = value() if isinstance(value, Parameter) else value
        return value

    def instantiate(self):
        return {k: v() if isinstance(v, Parameter) else v for k, v in self.items()}


def prms(**kwargs):
    """Wrapper function that takes kwargs and creates a dictionary of Parameters."""
    return PrmDict(**{k: Parameter(v) for k, v in kwargs.items()})


def unpack_prms(f):
    """Wrapper function that checks if args and kwargs to a function are Parameters; if so, unpacks them."""

    def wrapper(*args, **kwargs):
        args = [a() if isinstance(a, Parameter) else a for a in args]
        kwargs = {k: v() if isinstance(v, Parameter) else v for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return wrapper
