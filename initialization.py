from abc import ABC, abstractmethod
import numpy as np

class Initialization(ABC):

    def __init__(self) -> None:
        pass

    def set_layer_size(self, input_size:int, output_size:int) -> None:
        assert input_size >= 0
        assert output_size >= 0
        assert not (input_size == 0 and output_size == 0)
        
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_bias(self) -> np.ndarray:
        pass

class UniformInitialization(Initialization):

    def __init__(self) -> None:
        super().__init__()

    def get_weights(self):
        return np.random.rand(self.input_size, self.output_size) - 0.5

    def get_bias(self):
        return np.random.rand(1, self.output_size) - 0.5


class GaussianInitialization(Initialization):

    def __init__(self) -> None:
        super().__init__()

    def get_weights(self):
        return np.random.standard_normal(size=(self.input_size, self.output_size))

    def get_bias(self):
        return np.random.standard_normal(size=(1, self.output_size))
