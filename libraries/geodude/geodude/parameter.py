class Parameter:
    """
    A generic parameter wrapper that can handle constant values, functions, or random variables.

    Usage:
    >>> x = Parameter(10)
    >>> y = Parameter(lambda: random.uniform(0, 10))
    >>> z = Parameter(np.random.normal, 0, 1)  # mean 0, std 1
    """

    def __init__(self, value, *args, **kwargs):
        """
        Initialize the Parameter with a constant value, a function, or a random variable.

        :param value: The constant value, function, or random variable.
        :param args: Optional positional arguments for functions or random variables.
        :param kwargs: Optional keyword arguments for functions or random variables.
        """
        if callable(value):
            self.value = lambda: value(*args, **kwargs)
        else:
            self.value = value

    def __call__(self, *args, **kwargs):
        """
        Evaluate and return the parameter value.

        :return: The constant value or the result of the function or random variable.
        """
        if callable(self.value):
            return self.value(*args, **kwargs)
        return self.value
