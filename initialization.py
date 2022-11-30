from abc import ABC, abstractmethod
import numpy as np

class Initialization(ABC):

    def __init__(self, size:int) -> None:
        assert size >= 0
        
        self.size = size

    @abstractmethod
    def get(self):
        pass
    

class GaussianInitialization(Initialization):

    def __init__(self, size: int) -> None:
        super().__init__(size)

    def get(self):
        return np.random.standard_normal(self.size)