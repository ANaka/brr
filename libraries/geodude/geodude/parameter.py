from abc import ABC, abstractmethod
from typing import overload, Iterator, Any

class AbstractParameter(ABC):
    """
    Abstract base class for a Parameter.
    """
    
    
    
    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        """
        Yields the next value of the parameter.
        """
        pass
    




class Parameter(AbstractParameter):
    
    def __init__(self, f):
        if not callable(f) and not isinstance(f, Iterator):
            self._f = lambda: f
        
        # If f is a collection, wrap it in a function that iterates over the collection
        if isinstance(f, Iterator):
            def iter_fn():
                for val in f:
                    yield val
            self._f = iter_fn